//! Tests of the transform module: the derived cost table predicts the emitted programs, and
//! the matcher never accepts a site on the word of a tag; the hash-merge peephole reproduces
//! the merged emission and rejects near misses.

use super::cost::{category, derive_tree_access_costs, Counts};
use super::graph::SsaGraph;
use super::hash_merge::{apply_hash_merge, HashMergeReport, HashMergeRule, HashMergeSite, ProposalSource};
use super::rule::Rule;
use super::tree_access::TreeAccessRule;
use crate::kernel_builder::config::{AccessEncoding, CachedSelect, IndexForm, KernelConfig, ProblemSpec, SelectBitSource};
use crate::kernel_builder::emit_ssa::eliminate_ssa_dead_code;
use crate::kernel_builder::SsaTickTockKernelBuilder;
use crate::types::{RegionKind, SsaOp};

/// Merged hash, mirrored index (the context of the calibrated structure model).
fn context(batch_size: usize, bit_source: SelectBitSource) -> ProblemSpec {
    ProblemSpec {
        rounds: 16,
        batch_size,
        merge_hash_stages: true,
        index_form: IndexForm::Mirrored,
        select_bit_source: bit_source,
        ..Default::default()
    }
}

/// The program the emitter writes for a plan, after dead code elimination.
fn emitted_program(spec: &ProblemSpec, plan: &[Vec<AccessEncoding>]) -> crate::kernel_builder::SsaProgram {
    let config = KernelConfig {
        spec: ProblemSpec { access_plan: Some(plan.to_vec()), ..spec.clone() },
        ..KernelConfig::new(spec.forest_height, spec.n_nodes, spec.batch_size, spec.rounds)
    };
    let mut program = SsaTickTockKernelBuilder::new(config).build();
    let (ops, vreg_def, const_alternatives) = eliminate_ssa_dead_code(
        std::mem::take(&mut program.ops),
        &mut program.vreg_def,
        &mut program.const_alternatives,
        program.zero_vreg,
        Some(&program.vector_group_lanes),
    );
    crate::kernel_builder::SsaProgram { ops, vreg_def, const_alternatives, ..program }
}

/// Counts of the program a plan emits, after dead code elimination.
fn emitted_counts(spec: &ProblemSpec, plan: &[Vec<AccessEncoding>]) -> Counts {
    let ops = emitted_program(spec, plan).ops;
    let mut c = [0; 10];
    for op in &ops {
        if let Some(k) = category(op).unwrap() {
            c[k] += 1;
        }
    }
    c
}

/// Deterministic plans: canonical, each encoding on every subset of the cached levels, and
/// pseudo-random mixes (the shapes of `design_plans` / `random_rows` in the Python model).
fn test_plans(spec: &ProblemSpec, n_random: usize) -> Vec<Vec<Vec<AccessEncoding>>> {
    use AccessEncoding::*;
    let g = spec.n_groups();
    let depth = |r: usize| spec.round_depth(r);
    let uniform = |f: &dyn Fn(usize, usize) -> AccessEncoding| -> Vec<Vec<AccessEncoding>> {
        (0..spec.rounds).map(|r| vec![f(r, depth(r)); g]).collect()
    };
    let canonical = |_: usize, d: usize| if d == 0 { Bits } else { Gather };
    let mut plans = vec![uniform(&canonical)];
    // Every subset of the cached levels 1 to 4 bit-driven (the rest gathering), with each
    // encoding: pins every `table:d` and every `and:table:*` product against real emission.
    for e in [Bits, BitsFlow, Mobius] {
        for subset in 1u32..16 {
            let on = move |d: usize| (1..=4).contains(&d) && subset & (1 << (d - 1)) != 0;
            plans.push(uniform(&|_, d| if d == 0 { Bits } else if on(d) { e } else { Gather }));
        }
    }
    let mut state: u64 = 0x9E3779B97F4A7C15;
    let mut next = |n: usize| -> usize {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 33) as usize) % n
    };
    for _ in 0..n_random {
        let mut plan = uniform(&canonical);
        for r in 0..spec.rounds {
            if !(1..=4).contains(&depth(r)) {
                continue;
            }
            let options = [Gather, Bits, BitsFlow, Mobius];
            match next(4) {
                0 => {}
                1 => plan[r] = vec![options[next(4)]; g],
                _ => plan[r] = (0..g).map(|_| options[next(4)]).collect(),
            }
        }
        plans.push(plan);
    }
    plans
}

fn check_predictions(spec: &ProblemSpec, n_random: usize) {
    let table = derive_tree_access_costs(spec).unwrap();
    assert!(table.problems.is_empty(), "problems: {:#?}", table.problems);
    for (i, plan) in test_plans(spec, n_random).iter().enumerate() {
        assert_eq!(table.predict(plan), emitted_counts(spec, plan), "plan {}: {:?}", i, plan);
    }
}

#[test]
fn test_derived_table_predicts_emitted_counts_4_groups() {
    check_predictions(&context(32, SelectBitSource::Index), 12);
}

#[test]
fn test_derived_table_predicts_emitted_counts_path_bits() {
    check_predictions(&context(32, SelectBitSource::Path), 6);
}

#[test]
fn test_derived_table_predicts_emitted_counts_32_groups() {
    check_predictions(&context(256, SelectBitSource::Index), 4);
}

#[test]
fn test_derived_table_shared_constants_give_the_table_products() {
    // The scalar constants 3 to 15 address the cached nodes of levels 2, 3 and 4 and are
    // derived from each other: their liveness is an OR over those levels, whose expansion
    // gives the product terms.
    let table = derive_tree_access_costs(&context(32, SelectBitSource::Index)).unwrap();
    let alu = |name: &str| table.coefficients[name][2];
    assert_eq!(
        [alu("and:table:2+table:3"), alu("and:table:2+table:4"), alu("and:table:3+table:4"), alu("and:table:2+table:3+table:4")],
        [-3, -3, -5, 3]
    );
    let shared = table
        .resources
        .iter()
        .find(|r| r.any_of == ["table:2", "table:3", "table:4"])
        .expect("ops shared by levels 2, 3 and 4");
    assert_eq!(shared.counts[2], 3);
}

#[test]
fn test_derive_needs_the_context_only() {
    let mut spec = context(32, SelectBitSource::Index);
    spec.cached_select = crate::kernel_builder::config::CachedSelect::Bits;
    assert!(derive_tree_access_costs(&spec).is_err());
}

// ----------------------------------------------------------------------------
// Tags are hints: a wrong tag never makes a site
// ----------------------------------------------------------------------------

fn canonical_program(spec: &ProblemSpec) -> crate::kernel_builder::SsaProgram {
    let config = KernelConfig {
        spec: ProblemSpec { access_plan: Some(super::cost::canonical_plan(spec)), ..spec.clone() },
        ..KernelConfig::new(spec.forest_height, spec.n_nodes, spec.batch_size, spec.rounds)
    };
    SsaTickTockKernelBuilder::new(config).build()
}

fn rule() -> TreeAccessRule {
    TreeAccessRule { index_form: IndexForm::Mirrored }
}

#[test]
fn test_correct_tags_match_every_gather() {
    let spec = context(32, SelectBitSource::Index);
    let program = canonical_program(&spec);
    let graph = SsaGraph::new(&program.ops, &program.vector_group_lanes);
    let proposals = rule().propose(&graph);
    // 4 groups x the 14 gathering rounds (all but the two root rounds).
    assert_eq!(proposals.len(), 4 * 14);
    for p in &proposals {
        let site = rule().match_site(&graph, p).unwrap_or_else(|e| panic!("{:?}: {:?}", p, e));
        assert_eq!(site.depth, spec.round_depth(p.round.unwrap() as usize));
        assert_eq!(site.ops.len(), 10);
        let chain = site.chain.as_ref().expect("mirrored index chain");
        assert_eq!(chain.bits.len(), site.depth);
    }
}

#[test]
fn test_stripped_tags_propose_nothing() {
    let spec = context(32, SelectBitSource::Index);
    let mut program = canonical_program(&spec);
    for op in &mut program.ops {
        op.region = None;
    }
    let graph = SsaGraph::new(&program.ops, &program.vector_group_lanes);
    assert!(rule().propose(&graph).is_empty());
}

/// Retag the ops of every region matching `select` with `change`.
fn retag(ops: &mut [SsaOp], select: impl Fn(&crate::types::RegionTag) -> bool, change: impl Fn(&mut crate::types::RegionTag)) {
    for op in ops {
        if let Some(tag) = op.region.as_mut() {
            if select(tag) {
                change(tag);
            }
        }
    }
}

#[test]
fn test_wrong_depth_tag_is_rejected() {
    let spec = context(32, SelectBitSource::Index);
    let mut program = canonical_program(&spec);
    // Claim depth 3 for the depth-2 gather of round 2, group 1.
    retag(
        &mut program.ops,
        |t| t.kind == RegionKind::TreeAccess(AccessEncoding::Gather) && t.round == Some(2) && t.group == Some(1),
        |t| t.depth = Some(3),
    );
    let graph = SsaGraph::new(&program.ops, &program.vector_group_lanes);
    let p = rule().propose(&graph).into_iter().find(|p| p.round == Some(2) && p.group == Some(1)).unwrap();
    let err = rule().match_site(&graph, &p).unwrap_err();
    assert!(err.0.contains("depth"), "{:?}", err);
}

#[test]
fn test_hash_tagged_as_tree_access_is_rejected() {
    let spec = context(32, SelectBitSource::Index);
    let mut program = canonical_program(&spec);
    // The hash of round 5, group 2 claims to be a gather; so does the node XOR of round 6.
    let is_wrong = |t: &crate::types::RegionTag| {
        (t.kind == RegionKind::Hash && t.round == Some(5) && t.group == Some(2))
            || (t.kind == RegionKind::NodeXor && t.round == Some(6) && t.group == Some(0))
    };
    let wrong_ids: std::collections::BTreeSet<u32> =
        program.ops.iter().filter_map(|o| o.region).filter(|t| is_wrong(t)).map(|t| t.id).collect();
    retag(&mut program.ops, is_wrong, |t| t.kind = RegionKind::TreeAccess(AccessEncoding::Gather));
    let graph = SsaGraph::new(&program.ops, &program.vector_group_lanes);
    let wrong: Vec<_> = rule().propose(&graph).into_iter().filter(|p| p.region_id.map_or(false, |id| wrong_ids.contains(&id))).collect();
    assert_eq!(wrong.len(), 2, "{:?}", wrong);
    for p in wrong {
        assert!(rule().match_site(&graph, &p).is_err(), "{:?}", p);
    }
}

#[test]
fn test_gather_tag_on_partial_region_is_rejected() {
    // The shape is checked from definitions: the matcher binds the address op through the
    // loads, so a tag without the address op still gives the same site. A tag without the
    // XOR (the anchor) is rejected.
    let spec = context(32, SelectBitSource::Index);
    let target = |t: &crate::types::RegionTag| {
        t.kind == RegionKind::TreeAccess(AccessEncoding::Gather) && t.round == Some(3) && t.group == Some(0)
    };
    let full = canonical_program(&spec);
    let full_graph = SsaGraph::new(&full.ops, &full.vector_group_lanes);
    let full_site = rule()
        .match_site(&full_graph, &rule().propose(&full_graph).into_iter().find(|p| p.round == Some(3) && p.group == Some(0)).unwrap())
        .unwrap();
    let mut no_address = canonical_program(&spec);
    for op in &mut no_address.ops {
        if op.region.map_or(false, |t| target(&t)) && op.opcode == crate::types::Opcode::Sub {
            op.region = None;
        }
    }
    let graph = SsaGraph::new(&no_address.ops, &no_address.vector_group_lanes);
    let p = rule().propose(&graph).into_iter().find(|p| p.round == Some(3) && p.group == Some(0)).unwrap();
    assert_eq!(p.ops.len(), 9);
    assert_eq!(rule().match_site(&graph, &p).unwrap(), full_site);

    let mut program = canonical_program(&spec);
    for op in &mut program.ops {
        if op.region.map_or(false, |t| target(&t)) && op.opcode == crate::types::Opcode::Xor {
            op.region = None;
        }
    }
    let graph = SsaGraph::new(&program.ops, &program.vector_group_lanes);
    let p = rule().propose(&graph).into_iter().find(|p| p.round == Some(3) && p.group == Some(0)).unwrap();
    assert!(rule().match_site(&graph, &p).is_err());
}

// ----------------------------------------------------------------------------
// Apply: the rewritten canonical program is the planned program, op for op
// ----------------------------------------------------------------------------

/// Every op printed with VRegs renamed in order of first appearance (lanes of a vector group
/// named right after the group), and region ids dropped (they are dense per emission).
pub(super) fn canonical_ops(program: &crate::kernel_builder::SsaProgram) -> Vec<String> {
    use crate::types::{Operand, VReg};
    use std::collections::HashMap;
    let mut names: HashMap<VReg, usize> = HashMap::new();
    let mut name = |v: VReg, names: &mut HashMap<VReg, usize>| -> String {
        if !names.contains_key(&v) {
            let n = names.len();
            names.insert(v, n);
            if let Some(lanes) = program.vector_group_lanes.get(&v) {
                for &l in lanes {
                    let n = names.len();
                    names.entry(l).or_insert(n);
                }
            }
        }
        format!("{}{}", if v.is_scalar() { "s" } else { "v" }, names[&v])
    };
    program
        .ops
        .iter()
        .map(|op| {
            let srcs: Vec<String> = op
                .srcs
                .iter()
                .map(|s| match s {
                    Operand::VReg(v) => name(*v, &mut names),
                    Operand::Imm(i) => format!("#{}", i),
                })
                .collect();
            let keep: Option<Vec<String>> =
                op.keep_live.as_ref().map(|k| k.iter().map(|v| name(*v, &mut names)).collect());
            let delta = op.delta_vreg.map(|v| name(v, &mut names));
            let sub_temp = op.sub_temp_vreg.map(|v| name(v, &mut names));
            let dest = op.dest.map(|v| name(v, &mut names));
            format!(
                "{:?} {:?} {:?} <- {:?} keep={:?} delta={:?} sub={} sub_temp={:?} vt={:?} src={:?} sym={:?} mem={:?} \
                 alts={:?} k={:?} trace={:?} region={:?}",
                op.engine, op.opcode, dest, srcs, keep, delta, op.needs_subtraction, sub_temp, op.var_type,
                op.source_info, op.symmetry_hint, op.mem_loc, op.const_alternatives, op.produces_constant,
                op.trace_id, op.region.map(|t| (t.kind, t.round, t.group, t.depth)),
            )
        })
        .collect::<Vec<String>>()
        .into_iter()
        .chain({
            // The lane map: every group with its key and lanes named as in the ops; a group
            // no op references is listed as such (it would still reserve scratch).
            let mut groups: Vec<String> = program
                .vector_group_lanes
                .iter()
                .map(|(key, lanes)| match names.get(key) {
                    Some(k) => format!("group v{} = {:?}", k, lanes.iter().map(|l| names.get(l).copied()).collect::<Vec<_>>()),
                    None => "group not referenced by any op".to_string(),
                })
                .collect();
            groups.sort();
            groups
        })
        .collect()
}

fn check_apply(spec: &ProblemSpec, plan: &[Vec<AccessEncoding>], label: &str) {
    let applied = canonical_ops(&super::apply::apply_tree_access_plan(spec, plan).unwrap());
    let emitted = canonical_ops(&emitted_program(spec, plan));
    if let Some(i) = (0..applied.len().min(emitted.len())).find(|&i| applied[i] != emitted[i]) {
        panic!("{}: first difference at op {}:\n applied {}\n emitted {}\nplan {:?}", label, i, applied[i], emitted[i], plan);
    }
    assert_eq!(applied.len(), emitted.len(), "{}: op count", label);
}

#[test]
fn test_apply_reproduces_the_planned_program_4_groups() {
    let spec = context(32, SelectBitSource::Index);
    for (i, plan) in test_plans(&spec, 16).iter().enumerate() {
        check_apply(&spec, plan, &format!("plan {}", i));
    }
}

#[test]
fn test_apply_reproduces_the_planned_program_path_bits() {
    let spec = context(32, SelectBitSource::Path);
    for (i, plan) in test_plans(&spec, 6).iter().enumerate() {
        check_apply(&spec, plan, &format!("plan {}", i));
    }
}

#[test]
fn test_apply_reproduces_the_planned_program_32_groups() {
    let spec = context(256, SelectBitSource::Index);
    // flow90.json's structure: rounds 1-4 and 12-15 mixed or selected, round 15 gathering.
    use AccessEncoding::*;
    let mut plan = super::cost::canonical_plan(&spec);
    for r in [1, 2, 12, 13] {
        plan[r] = vec![Bits; 32];
    }
    for r in [3, 14] {
        plan[r] = vec![BitsFlow; 32];
    }
    plan[4] = (0..32).map(|g| if g < 17 { Gather } else { BitsFlow }).collect();
    check_apply(&spec, &plan, "flow90-like");
    for (i, plan) in test_plans(&spec, 3).iter().enumerate() {
        check_apply(&spec, plan, &format!("plan {}", i));
    }
}

#[test]
fn test_applied_program_schedules_like_the_emitted_one() {
    // Downstream of the apply pass the pipeline runs unchanged: the greedy schedule of the
    // rewritten program has the emitted program's cycles, ops, peak live and budget flags
    // (shaped greedy, as the recorded results use).
    use crate::kernel_builder::config::{GreedyBarrierMode, GreedyPriorityKind};
    use crate::kernel_builder::pipeline::{PipelineConfig, PipelineContext};
    use AccessEncoding::*;
    let schedule = |program| {
        let mut config = PipelineConfig::default();
        config.compiler.greedy_barriers = GreedyBarrierMode::None;
        config.compiler.greedy_priority = GreedyPriorityKind::Wavefront;
        config.compiler.greedy_split_shift = true;
        let mut ctx = PipelineContext::new(program, config);
        ctx.run_greedy_pipeline(true).unwrap();
        let greedy_exceeded = ctx.greedy_metrics.budget_exceeded;
        let r = ctx.into_result().unwrap();
        (r.cycles, r.ops.len(), r.address_result.peak_live, r.address_result.budget_exceeded, greedy_exceeded)
    };
    let small = context(32, SelectBitSource::Index);
    let mut plan = super::cost::canonical_plan(&small);
    plan[2] = vec![Mobius, Bits, Gather, BitsFlow];
    plan[4] = vec![Gather, Gather, BitsFlow, BitsFlow];
    plan[13] = vec![Bits; 4];
    let large = context(256, SelectBitSource::Index);
    let mut flow90 = super::cost::canonical_plan(&large);
    for r in [1, 2, 12, 13] {
        flow90[r] = vec![Bits; 32];
    }
    for r in [3, 14] {
        flow90[r] = vec![BitsFlow; 32];
    }
    flow90[4] = (0..32).map(|g| if g < 17 { Gather } else { BitsFlow }).collect();
    for (label, spec, plan) in [("4 groups", &small, &plan), ("32 groups, flow90-like", &large, &flow90)] {
        let applied = schedule(super::apply::apply_tree_access_plan(spec, plan).unwrap());
        let emitted = schedule(emitted_program(spec, plan));
        assert_eq!(
            applied, emitted,
            "{}: (cycles, ops, peak live, address budget exceeded, greedy budget exceeded)",
            label
        );
    }
}

#[test]
fn test_apply_reproduces_the_planned_program_at_other_shapes() {
    // Rounds 15: the last round is at depth 4, so the final index conversion reads L_4; a
    // plan with no gather at depth 4 creates L_4 there (not in round 4). Also height 8 and
    // 26 rounds (more visits per level).
    for spec in [shape(15, 10, 32), shape(15, 8, 32), shape(26, 10, 32)] {
        for (i, plan) in test_plans(&spec, 4).iter().enumerate() {
            check_apply(&spec, plan, &format!("rounds {} height {} plan {}", spec.rounds, spec.forest_height, i));
        }
    }
}

#[test]
fn test_apply_rejects_a_choice_outside_the_cached_levels() {
    let spec = context(32, SelectBitSource::Index);
    let mut plan = super::cost::canonical_plan(&spec);
    plan[6] = vec![AccessEncoding::Bits; 4];
    assert!(super::apply::apply_tree_access_plan(&spec, &plan).is_err());
}

// ----------------------------------------------------------------------------
// Hash merge: a peephole on the finished program
// ----------------------------------------------------------------------------

pub(super) fn raw_program(spec: &ProblemSpec) -> crate::kernel_builder::SsaProgram {
    let config = KernelConfig { spec: spec.clone(), ..KernelConfig::new(spec.forest_height, spec.n_nodes, spec.batch_size, spec.rounds) };
    SsaTickTockKernelBuilder::new(config).build()
}

fn merge_rule(source: ProposalSource) -> HashMergeRule {
    HashMergeRule { source }
}

pub(super) fn clone_program(p: &crate::kernel_builder::SsaProgram) -> crate::kernel_builder::SsaProgram {
    crate::kernel_builder::SsaProgram {
        ops: p.ops.clone(),
        vreg_def: p.vreg_def.clone(),
        vreg_count: p.vreg_count,
        const_alternatives: p.const_alternatives.clone(),
        zero_vreg: p.zero_vreg,
        vector_group_lanes: p.vector_group_lanes.clone(),
    }
}

/// First difference of two op lists printed by `canonical_ops`, if any, after allowing the
/// emitter's variable-type leak.
///
/// In the default emission the stage-4 multiplier 9 is a CSE hit (the stage-3 shift made it),
/// so the `VarType::HashState` set for it is never consumed and labels the next op the emitter
/// writes (the first op of the depth-3 setup when there is one). The rewrite keeps its
/// input's labels; the merged emission consumes the label on the new constant. Only that
/// label may differ, on at most one op.
///
/// # Returns
/// The first other difference, and the indices of ops differing only by the leaked label.
pub(super) fn first_difference(a: &[String], b: &[String]) -> (Option<String>, Vec<usize>) {
    let mut leaked = Vec::new();
    for i in 0..a.len().min(b.len()) {
        if a[i] == b[i] {
            continue;
        }
        if a[i].replacen("vt=Some(HashState)", "vt=None", 1) == b[i] && leaked.is_empty() {
            leaked.push(i);
            continue;
        }
        return (Some(format!("op {}:\n rewritten {}\n emitted   {}", i, a[i], b[i])), leaked);
    }
    ((a.len() != b.len()).then(|| format!("op count {} vs {}", a.len(), b.len())), leaked)
}

/// Gate 1: rewriting the default emission of `spec` gives the `--merge-hash-stages` emission
/// op for op (VRegs renamed, region ids dropped, both before dead code elimination), except
/// for the emitter's leaked variable-type label (`first_difference`).
fn check_hash_merge_parity(spec: &ProblemSpec, label: &str) -> HashMergeReport {
    let spec = ProblemSpec { merge_hash_stages: false, ..spec.clone() };
    let (rewritten, report) = apply_hash_merge(raw_program(&spec), &merge_rule(ProposalSource::Ops), |_| true).unwrap();
    let emitted = raw_program(&ProblemSpec { merge_hash_stages: true, ..spec.clone() });
    assert_eq!(report.sites, spec.rounds * spec.n_groups(), "{}: sites", label);
    // The shift constant 9 (and its broadcast) moved to the merged form's setup.
    assert_eq!(report.surgery.moved, 2, "{}", label);
    let (diff, leaked) = first_difference(&canonical_ops(&rewritten), &canonical_ops(&emitted));
    if let Some(d) = diff {
        panic!("{}: {}", label, d);
    }
    // The leak lands on a setup op, never on a hash op.
    for &i in &leaked {
        assert!(rewritten.ops[i].region.is_none(), "{}: label leak on {:?}", label, rewritten.ops[i]);
    }
    eprintln!("{}: {} sites, op-for-op parity, {} op(s) with the leaked label", label, report.sites, leaked.len());
    report
}

/// The flow90 plan (docs/emission_plans/flow90.json), read from the file.
pub(super) fn flow90_plan(spec: &ProblemSpec) -> Vec<Vec<AccessEncoding>> {
    use AccessEncoding::*;
    let named = |s: &str| AccessEncoding::from_str(s).unwrap();
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../docs/emission_plans/flow90.json");
    let file: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let access = file["access"].as_array().unwrap();
    assert_eq!(access.len(), spec.rounds);
    access
        .iter()
        .enumerate()
        .map(|(r, entry)| match entry {
            serde_json::Value::String(s) => vec![named(s); spec.n_groups()],
            // serde_json does not keep key order; the file's one mixed round is 17 groups
            // gathering, then 15 bits-flow.
            serde_json::Value::Object(m) => {
                let counts = (r, m.len(), m.get("gather").and_then(|v| v.as_u64()), m.get("bits-flow").and_then(|v| v.as_u64()));
                assert_eq!(counts, (4, 2, Some(17), Some(15)));
                (0..spec.n_groups()).map(|g| if g < 17 { Gather } else { BitsFlow }).collect()
            }
            other => panic!("unexpected plan entry {:?}", other),
        })
        .collect()
}

#[test]
fn test_hash_merge_parity_2_rounds() {
    let spec = ProblemSpec { rounds: 2, ..Default::default() };
    check_hash_merge_parity(&spec, "2r default");
    let spec = ProblemSpec { index_form: IndexForm::Mirrored, cached_select: CachedSelect::Bits, ..spec };
    check_hash_merge_parity(&spec, "2r mirrored bits");
}

#[test]
fn test_hash_merge_parity_16_rounds_and_cost() {
    let report = check_hash_merge_parity(&ProblemSpec::default(), "16r default");
    // One site class: each site removes a multiply_add and 3 shiftable vector ops and adds 2
    // multiply_adds and 1 shiftable op.
    let mut delta = [0i64; 10];
    delta[0] = 1;
    delta[1] = -2;
    assert_eq!(report.site_classes.len(), 1, "{:?}", report.site_classes);
    assert_eq!(report.site_classes.get(&delta), Some(&512));
    // Shared setup, paid once: 3 new constants with broadcasts (C2 + C3, 33 << 9, C2 << 9) less
    // the two that die (C2, C3). The shift constant 9 survives: it is also the stage-4
    // multiplier. No other constant's alternatives change.
    let mut resource = [0i64; 10];
    resource[0] = 1;
    resource[8] = 1;
    assert_eq!(report.resource_delta, resource);
    assert!(report.surgery.alternatives.is_empty(), "{:?}", report.surgery.alternatives);
    use crate::kernel_builder::config::HASH_STAGES;
    use crate::types::Opcode::{ConstChoice, Vbroadcast};
    let dead: Vec<_> = report.surgery.cascaded.iter().map(|&(_, op, k)| (op, k)).collect();
    let (c2, c3) = (Some(HASH_STAGES[2].1), Some(HASH_STAGES[3].1));
    assert_eq!(dead, vec![(ConstChoice, c2), (Vbroadcast, c2), (ConstChoice, c3), (Vbroadcast, c3)]);

    let mirrored = ProblemSpec { index_form: IndexForm::Mirrored, ..ProblemSpec::default() };
    check_hash_merge_parity(&ProblemSpec { cached_select: CachedSelect::Bits, ..mirrored.clone() }, "16r mirrored bits");
    let plan = flow90_plan(&mirrored);
    check_hash_merge_parity(&ProblemSpec { access_plan: Some(plan), ..mirrored }, "16r flow90 plan");
}

#[test]
fn test_hash_merge_finds_every_site_without_tags() {
    let tagged = raw_program(&ProblemSpec::default());
    let mut stripped = clone_program(&tagged);
    for op in &mut stripped.ops {
        op.region = None;
    }
    let sites = |p: &crate::kernel_builder::SsaProgram, src| {
        let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
        merge_rule(src).find_sites(&g).0
    };
    let untagged = sites(&stripped, ProposalSource::Ops);
    assert_eq!(untagged.len(), 512);
    assert!(untagged.iter().all(|s| s.round.is_none() && s.group.is_none()));
    // Tags propose nothing without the ops agreeing, and add only labels: both sources find
    // the same sites in the tagged program.
    assert!(sites(&stripped, ProposalSource::Tags).is_empty());
    let shape = |v: &[HashMergeSite]| v.iter().map(|s| (s.ops(), s.input, s.output, s.values)).collect::<Vec<_>>();
    let by_ops = sites(&tagged, ProposalSource::Ops);
    let by_tags = sites(&tagged, ProposalSource::Tags);
    assert_eq!(shape(&by_ops), shape(&untagged));
    assert_eq!(shape(&by_tags), shape(&untagged));
    let labels: std::collections::BTreeSet<_> = by_tags.iter().map(|s| (s.round.unwrap(), s.group.unwrap())).collect();
    assert_eq!(labels.len(), 512);
    // The untagged rewrite is the tagged one without tags.
    let (a, _) = apply_hash_merge(stripped, &merge_rule(ProposalSource::Ops), |_| true).unwrap();
    let (mut b, _) = apply_hash_merge(tagged, &merge_rule(ProposalSource::Tags), |_| true).unwrap();
    for op in &mut b.ops {
        op.region = None;
    }
    assert_eq!(canonical_ops(&a), canonical_ops(&b));
}

#[test]
fn test_hash_merge_matches_the_proved_instance_only_once() {
    let spec = ProblemSpec { rounds: 2, ..Default::default() };
    let p = raw_program(&spec);
    let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
    let (sites, rejected) = merge_rule(ProposalSource::Ops).find_sites(&g);
    assert_eq!(sites.len(), 64);
    let expected = crate::kernel_builder::config::merged_hash_stage_constants(2).unwrap();
    for s in &sites {
        assert_eq!(s.merged, expected);
        assert!(s.facts.iter().any(|f| f.name == "proof_instance" && f.value.starts_with("R8")));
    }
    // Stage 0 (read by stage 1's xor and shift right), stage 4 (read by stage 5's shift right)
    // and the index update (addend not constant) are rejected.
    assert_eq!(rejected.values().sum::<usize>(), 3 * 64, "{:?}", rejected);
    // The merged emission has nothing left to merge.
    let merged = raw_program(&ProblemSpec { merge_hash_stages: true, ..spec });
    let g = SsaGraph::new(&merged.ops, &merged.vector_group_lanes);
    assert!(merge_rule(ProposalSource::Ops).find_sites(&g).0.is_empty());
}

/// Mutate the first site of the 2-round default program.
///
/// # Returns
/// The first site's rejection (`None` if it still matches) and how many sites remain.
fn mutate_first_site(mutate: impl Fn(&mut crate::kernel_builder::SsaProgram, &HashMergeSite)) -> (Option<String>, usize) {
    let mut p = raw_program(&ProblemSpec { rounds: 2, ..Default::default() });
    let site = {
        let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
        merge_rule(ProposalSource::Ops).find_sites(&g).0.remove(0)
    };
    mutate(&mut p, &site);
    let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
    let n = merge_rule(ProposalSource::Ops).find_sites(&g).0.len();
    (merge_rule(ProposalSource::Ops).match_at(&g, site.fma).err().map(|e| e.0), n)
}

#[test]
fn test_hash_merge_rejects_near_misses() {
    use crate::types::{Opcode, Operand, VReg};
    let expect = |(err, n): (Option<String>, usize), text: &str, sites: usize| {
        let err = err.unwrap_or_else(|| panic!("expected a rejection containing {:?}", text));
        assert!(err.contains(text), "{:?} does not contain {:?}", err, text);
        assert_eq!(n, sites, "{}", text);
    };
    // Right shift: (x + d) ^ (x >> k) does not merge (R8-hash-stage-merge-needs-left-shift).
    expect(mutate_first_site(|p, s| p.ops[s.shl].opcode = Opcode::Shr), "right shift", 63);
    // An extra reader of the intermediate x.
    expect(
        mutate_first_site(|p, s| {
            let x = p.ops[s.fma].dest.unwrap();
            let mut extra = p.ops[s.xor].clone();
            extra.dest = Some(VReg::vector(p.vreg_count));
            p.vreg_count += 1;
            extra.srcs = vec![Operand::VReg(x), Operand::VReg(x)];
            p.ops.insert(s.xor + 1, extra);
        }),
        "3 readers",
        63,
    );
    // x kept live by another op: not a read, but removing x would leave it dangling.
    expect(
        mutate_first_site(|p, s| {
            let x = p.ops[s.fma].dest.unwrap();
            p.ops.last_mut().unwrap().keep_live = Some(vec![x]);
        }),
        "kept live",
        63,
    );
    // Wrong constant relationship: the add's other operand varies.
    expect(
        mutate_first_site(|p, s| {
            let x = p.ops[s.fma].dest.unwrap();
            p.ops[s.add].srcs = vec![Operand::VReg(x), Operand::VReg(s.input)];
        }),
        "add operand is not a broadcast constant",
        63,
    );
    // Wrong constant relationship: the multiplier varies.
    expect(mutate_first_site(|p, s| p.ops[s.fma].srcs[1] = Operand::VReg(s.input)), "multiplier is not a broadcast constant", 63);
    // Shift amount out of range: the shared shift constant becomes 40, immediate and
    // annotation (every site rejects).
    expect(
        mutate_first_site(|p, s| {
            let k_vec = p.vreg_def[&s.operands[3]];
            let k_scalar = p.ops[k_vec].srcs[0].as_vreg().unwrap();
            let def = p.vreg_def[&k_scalar];
            p.ops[def].srcs[0] = Operand::Imm(40);
            p.ops[def].produces_constant = Some(40);
            p.ops[k_vec].produces_constant = Some(40);
        }),
        "not in 0..32",
        0,
    );
    // The add and the shift do not feed one common op.
    expect(
        mutate_first_site(|p, s| {
            let t = p.ops[s.add].dest.unwrap();
            p.ops[s.xor].srcs = vec![Operand::VReg(t), Operand::VReg(s.input)];
        }),
        "not read by exactly one common op",
        63,
    );
    // Combined with + instead of ^: that is FMA fusion (R1), not this rule.
    expect(mutate_first_site(|p, s| p.ops[s.xor].opcode = Opcode::Add), "not t ^ s", 63);
}

#[test]
fn test_hash_merge_finds_sites_in_a_form_the_emitter_never_writes() {
    // Gate 2 (small): each site rewritten into an equivalent shape the emitter never writes:
    // multiply_add factors commuted, add and xor operands swapped, the shift placed before the
    // add, and private (not CSE-shared) broadcasts of the four constants. All sites are found
    // and the rewrite has the merged emission's op counts per category.
    use crate::types::{Engine, Opcode, Operand, VReg};
    let spec = ProblemSpec { rounds: 2, ..Default::default() };
    let p = raw_program(&spec);
    let sites = {
        let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
        merge_rule(ProposalSource::Ops).find_sites(&g).0
    };
    let by_fma: std::collections::BTreeMap<usize, &HashMergeSite> = sites.iter().map(|s| (s.fma, s)).collect();
    let moved: std::collections::BTreeSet<usize> = sites.iter().flat_map(|s| [s.add, s.shl]).collect();
    let xors: std::collections::BTreeSet<usize> = sites.iter().map(|s| s.xor).collect();
    let mut next = p.vreg_count;
    let mut ops: Vec<SsaOp> = Vec::new();
    for (i, op) in p.ops.iter().enumerate() {
        if moved.contains(&i) {
            continue;
        }
        let Some(s) = by_fma.get(&i) else {
            let mut op = op.clone();
            if xors.contains(&i) {
                op.srcs.reverse();
            }
            ops.push(op);
            continue;
        };
        let mut private = Vec::new();
        for &v in &s.operands {
            let b = &p.ops[p.vreg_def[&v]];
            let mut nb = SsaOp::new(0, Engine::Valu, Opcode::Vbroadcast, Some(VReg::vector(next)), b.srcs.clone());
            next += 1;
            nb.produces_constant = b.produces_constant;
            private.push(nb.dest.unwrap());
            ops.push(nb);
        }
        let x = op.dest.unwrap();
        let mut fma = op.clone();
        fma.srcs = vec![Operand::VReg(private[0]), Operand::VReg(s.input), Operand::VReg(private[1])];
        ops.push(fma);
        let mut shl = p.ops[s.shl].clone();
        shl.srcs = vec![Operand::VReg(x), Operand::VReg(private[3])];
        ops.push(shl);
        let mut add = p.ops[s.add].clone();
        add.srcs = vec![Operand::VReg(private[2]), Operand::VReg(x)];
        ops.push(add);
    }
    for (i, op) in ops.iter_mut().enumerate() {
        op.id = i;
        op.region = None;
    }
    let vreg_def = ops.iter().filter_map(|o| o.dest.map(|d| (d, o.id))).fold(std::collections::HashMap::new(), |mut m, (d, i)| {
        m.entry(d).or_insert(i);
        m
    });
    let q = crate::kernel_builder::SsaProgram { ops, vreg_def, vreg_count: next, ..clone_program(&p) };
    let (found, _) = {
        let g = SsaGraph::new(&q.ops, &q.vector_group_lanes);
        merge_rule(ProposalSource::Ops).find_sites(&g)
    };
    assert_eq!(found.len(), 64);
    let (rewritten, report) = apply_hash_merge(q, &merge_rule(ProposalSource::Ops), |_| true).unwrap();
    assert_eq!(report.sites, 64);
    let counts = |mut prog: crate::kernel_builder::SsaProgram| {
        let (ops, _, _) = eliminate_ssa_dead_code(
            std::mem::take(&mut prog.ops),
            &mut prog.vreg_def,
            &mut prog.const_alternatives,
            prog.zero_vreg,
            Some(&prog.vector_group_lanes),
        );
        let mut c = [0i64; 10];
        for op in &ops {
            if let Some(k) = category(op).unwrap() {
                c[k] += 1;
            }
        }
        c
    };
    assert_eq!(counts(rewritten), counts(raw_program(&ProblemSpec { merge_hash_stages: true, ..spec })));
}

#[test]
fn test_hash_merge_on_some_sites_keeps_the_old_constants() {
    // Rewriting only round 0's sites: the other round still reads C2, C3 and the shift, so
    // nothing cascades and the new constants are paid on top (3 constants, 3 broadcasts).
    let spec = ProblemSpec { rounds: 2, ..Default::default() };
    let p = raw_program(&spec);
    let round0 = |s: &HashMergeSite| s.round == Some(0);
    let (rewritten, report) = apply_hash_merge(p, &merge_rule(ProposalSource::Tags), round0).unwrap();
    assert_eq!(report.sites, 32);
    assert!(report.surgery.cascaded.is_empty(), "{:?}", report.surgery.cascaded);
    // C2, C3 and 9 (scalars and broadcasts) are still read, so they join the new setup.
    assert_eq!(report.surgery.moved, 6);
    let mut resource = [0i64; 10];
    resource[0] = 3;
    resource[8] = 3;
    assert_eq!(report.resource_delta, resource);
    let g = SsaGraph::new(&rewritten.ops, &rewritten.vector_group_lanes);
    let left = merge_rule(ProposalSource::Ops).find_sites(&g).0;
    assert_eq!(left.len(), 32);
    assert!(left.iter().all(|s| s.round == Some(1)));
}

#[test]
fn test_hash_merge_pass_schedules_like_the_merged_emission() {
    // The pass in the greedy pipeline gives the merged emission's op count; cycles are
    // reported (the shared constant sits elsewhere, which the greedy order may see).
    use crate::kernel_builder::config::{CompilerConfig, TransformKind};
    use crate::kernel_builder::pipeline::{PipelineConfig, PipelineContext};
    let spec = ProblemSpec { rounds: 2, ..Default::default() };
    let schedule = |program, transforms: Vec<TransformKind>| {
        let config = PipelineConfig { compiler: CompilerConfig { transforms, ..Default::default() }, ..Default::default() };
        let mut ctx = PipelineContext::new(program, config);
        ctx.run_greedy_pipeline(true).unwrap();
        let r = ctx.into_result().unwrap();
        (r.cycles, r.ops.len())
    };
    let rewritten = schedule(raw_program(&spec), vec![TransformKind::HashMerge]);
    let merged = schedule(raw_program(&ProblemSpec { merge_hash_stages: true, ..spec.clone() }), vec![]);
    let default = schedule(raw_program(&spec), vec![]);
    eprintln!("greedy (cycles, ops): rewritten {:?} merged {:?} default {:?}", rewritten, merged, default);
    assert_eq!(rewritten.1, merged.1);
    assert_ne!(default.1, merged.1);
}

// ----------------------------------------------------------------------------
// Hash merge: the rewrite keeps values (interp), and foreign input is refused per site
// ----------------------------------------------------------------------------

/// Deterministic pseudo-random 32-bit words.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    /// A word biased towards edge values.
    fn word(&mut self) -> u32 {
        const EDGES: [u32; 6] = [0, 1, 0x8000_0000, 0xFFFF_FFFF, 0x7FFF_FFFF, 33];
        match self.next() % 3 {
            0 => EDGES[self.next() as usize % EDGES.len()],
            _ => self.next(),
        }
    }
}

/// A program of `n` independent stage pairs `r = (a*m + c + d) ^ ((a*m + c) << k)`, not
/// written by the emitter: per site its own loaded input `a`, its own constants (values
/// repeat across sites, ops do not), and the shape varied by the site index (multiply_add
/// factors commuted, add operands swapped, shift emitted before the add, xor operands
/// swapped). Outputs: `r` and `r ^ a` are stored.
///
/// # Returns
/// The program and each site's `(m, c, d, k)`.
fn synthetic_hash_program(seed: u64, n: usize) -> (crate::kernel_builder::SsaProgram, Vec<[i64; 4]>) {
    use crate::types::{Engine, Opcode, Operand, VReg};
    let mut rng = Lcg(seed);
    let mut next_id = 0u32;
    let mut fresh = |vector: bool| {
        next_id += 1;
        if vector { VReg::vector(next_id - 1) } else { VReg::scalar(next_id - 1) }
    };
    let mut ops: Vec<SsaOp> = Vec::new();
    let mut constants = Vec::new();
    let push = |ops: &mut Vec<SsaOp>, engine, opcode, dest, srcs: Vec<Operand>| {
        let mut op = SsaOp::new(ops.len(), engine, opcode, dest, srcs);
        if let (Opcode::ConstChoice, Some(Operand::Imm(k))) = (opcode, op.srcs.first()) {
            op.produces_constant = Some(*k);
        }
        ops.push(op);
        dest
    };
    for i in 0..n {
        let k = match i % 4 {
            0 => 0,
            1 => 31,
            _ => (rng.next() % 32) as i64,
        };
        let [m, c, d] = [rng.word() as i64, rng.word() as i64, rng.word() as i64];
        constants.push([m, c, d, k]);
        // Addresses are registers, as lowering reads them (`Const`, so the tests that rewrite
        // the sites' `ConstChoice` immediates by position do not see them).
        let mut address = |ops: &mut Vec<SsaOp>, a: i64| {
            let d = push(ops, Engine::Load, Opcode::Const, Some(fresh(false)), vec![Operand::Imm(a)]).unwrap();
            ops.last_mut().unwrap().produces_constant = Some(a);
            d
        };
        let (load_at, out1_at, out2_at) =
            (address(&mut ops, 1000 + 8 * i as i64), address(&mut ops, 5000 + 8 * i as i64), address(&mut ops, 9000 + 8 * i as i64));
        let a = push(&mut ops, Engine::Load, Opcode::Vload, Some(fresh(true)), vec![Operand::VReg(load_at)]);
        let mut vec_const = |ops: &mut Vec<SsaOp>, value: i64| {
            let s = push(ops, Engine::Load, Opcode::ConstChoice, Some(fresh(false)), vec![Operand::Imm(value)]).unwrap();
            let v = push(ops, Engine::Valu, Opcode::Vbroadcast, Some(fresh(true)), vec![Operand::VReg(s)]).unwrap();
            ops.last_mut().unwrap().produces_constant = Some(value);
            v
        };
        let (vm, vc, vd, vk) = (vec_const(&mut ops, m), vec_const(&mut ops, c), vec_const(&mut ops, d), vec_const(&mut ops, k));
        let a = a.unwrap();
        let v = |x: VReg| Operand::VReg(x);
        let fma_srcs = if i & 1 == 0 { vec![v(a), v(vm), v(vc)] } else { vec![v(vm), v(a), v(vc)] };
        let x = push(&mut ops, Engine::Valu, Opcode::MultiplyAdd, Some(fresh(true)), fma_srcs).unwrap();
        let add_srcs = if i & 2 == 0 { vec![v(x), v(vd)] } else { vec![v(vd), v(x)] };
        let (t, s) = if i & 4 == 0 {
            let t = push(&mut ops, Engine::Valu, Opcode::Add, Some(fresh(true)), add_srcs).unwrap();
            (t, push(&mut ops, Engine::Valu, Opcode::Shl, Some(fresh(true)), vec![v(x), v(vk)]).unwrap())
        } else {
            let s = push(&mut ops, Engine::Valu, Opcode::Shl, Some(fresh(true)), vec![v(x), v(vk)]).unwrap();
            (push(&mut ops, Engine::Valu, Opcode::Add, Some(fresh(true)), add_srcs).unwrap(), s)
        };
        let xor_srcs = if i & 8 == 0 { vec![v(t), v(s)] } else { vec![v(s), v(t)] };
        let r = push(&mut ops, Engine::Valu, Opcode::Xor, Some(fresh(true)), xor_srcs).unwrap();
        push(&mut ops, Engine::Store, Opcode::Vstore, None, vec![v(out1_at), v(r)]);
        let out = push(&mut ops, Engine::Valu, Opcode::Xor, Some(fresh(true)), vec![v(r), v(a)]).unwrap();
        push(&mut ops, Engine::Store, Opcode::Vstore, None, vec![v(out2_at), v(out)]);
    }
    let vreg_def = ops.iter().filter_map(|o| o.dest.map(|d| (d, o.id))).collect();
    let program = crate::kernel_builder::SsaProgram {
        ops,
        vreg_def,
        vreg_count: next_id,
        const_alternatives: Vec::new(),
        zero_vreg: None,
        vector_group_lanes: Default::default(),
    };
    (program, constants)
}

/// Run `before` and `after` on the same random inputs and compare every store.
fn assert_same_values(before: &crate::kernel_builder::SsaProgram, after: &crate::kernel_builder::SsaProgram, seed: u64) {
    use super::interp::run;
    // A random memory image (words biased to edge values); both programs run on a copy and
    // the final memories must agree.
    let mut r = Lcg(seed);
    let image: Vec<u32> = (0..10_000).map(|_| r.word()).collect();
    let (mut mb, mut ma) = (image.clone(), image);
    run(before, &mut mb).unwrap();
    run(after, &mut ma).unwrap();
    for (i, (x, y)) in mb.iter().zip(&ma).enumerate() {
        assert_eq!(x, y, "mem[{}] differs (seed {})", i, seed);
    }
}

#[test]
fn test_hash_merge_preserves_values_on_random_constants() {
    // Random m, c, d per site (edge words 0, 1, 2^31, 2^32 - 1 included), k = 0, 31 and
    // random, every shape variant, random input lanes biased to edge words.
    for seed in 1..=8u64 {
        let (program, constants) = synthetic_hash_program(seed, 32);
        let before = clone_program(&program);
        let (after, report) = apply_hash_merge(program, &merge_rule(ProposalSource::Ops), |_| true).unwrap();
        assert_eq!(report.sites, 32, "seed {}: {:?}", seed, report.rejections);
        let g = SsaGraph::new(&after.ops, &after.vector_group_lanes);
        assert!(merge_rule(ProposalSource::Ops).find_sites(&g).0.is_empty());
        assert!(constants.iter().any(|c| c[3] == 0) && constants.iter().any(|c| c[3] == 31));
        for run_seed in 0..4 {
            assert_same_values(&before, &after, seed * 100 + run_seed);
        }
    }
}

#[test]
fn test_hash_merge_preserves_values_with_zero_results() {
    // Constants that make x, t, s or r zero on every lane (m = 0, c = 0, d = -c, k = 31 with
    // even x), and the emitter's constants.
    use crate::kernel_builder::config::{fma_multiplier, HASH_STAGES};
    let cases: [[i64; 4]; 4] = [
        [0, 0, 0, 0],
        [0, 5, 0xFFFF_FFFB, 31],
        [2, 0, 0, 31],
        [fma_multiplier(2).unwrap(), HASH_STAGES[2].1, HASH_STAGES[3].1, HASH_STAGES[3].4],
    ];
    let (mut program, _) = synthetic_hash_program(77, cases.len());
    // Overwrite each site's constants (both the immediate and the annotation).
    let mut consts = program.ops.iter().enumerate().filter(|(_, o)| o.opcode == crate::types::Opcode::ConstChoice).map(|(i, _)| i).collect::<Vec<_>>().into_iter();
    for case in cases {
        for value in case {
            let i = consts.next().unwrap();
            program.ops[i].srcs[0] = crate::types::Operand::Imm(value);
            program.ops[i].produces_constant = Some(value);
            program.ops[i + 1].produces_constant = Some(value);
        }
    }
    let before = clone_program(&program);
    let (after, report) = apply_hash_merge(program, &merge_rule(ProposalSource::Ops), |_| true).unwrap();
    assert_eq!(report.sites, cases.len());
    for s in 0..4 {
        assert_same_values(&before, &after, 500 + s);
    }
}

/// Apply the rule to the 2-round default program after `mutate` changed it around the first
/// site.
fn rewrite_mutated(
    mutate: impl Fn(&mut crate::kernel_builder::SsaProgram, &HashMergeSite),
) -> Result<(crate::kernel_builder::SsaProgram, HashMergeReport), String> {
    let mut p = raw_program(&ProblemSpec { rounds: 2, ..Default::default() });
    let site = {
        let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
        merge_rule(ProposalSource::Ops).find_sites(&g).0.remove(0)
    };
    mutate(&mut p, &site);
    apply_hash_merge(p, &merge_rule(ProposalSource::Ops), |_| true)
}

/// The rewrite of a mutated program rejected the first site with a reason containing `text`
/// and rewrote the other 63.
fn assert_first_site_rejected(result: Result<(crate::kernel_builder::SsaProgram, HashMergeReport), String>, text: &str) {
    let (_, report) = result.unwrap_or_else(|e| panic!("{}: the rewrite failed instead of rejecting one site: {}", text, e));
    assert_eq!(report.sites, 63, "{}: {:?}", text, report.rejections);
    assert!(report.rejections.keys().any(|k| k.contains(text)), "{}: {:?}", text, report.rejections);
}

#[test]
fn test_hash_merge_rejects_a_site_op_carrying_keep_live() {
    // The emitter attaches keep_live to the next op it emits; on a site op the list would be
    // lost with the op.
    for pick in 0..4 {
        let result = rewrite_mutated(|p, s| {
            let other = p.ops[s.fma - 1].dest.unwrap();
            p.ops[s.ops()[pick]].keep_live = Some(vec![other]);
        });
        assert_first_site_rejected(result, "keep_live");
    }
}

#[test]
fn test_hash_merge_refuses_redefined_vregs() {
    use super::surgery::Surgery;
    use crate::types::{Engine, Opcode, Operand};
    // A second definition of the first site's x (after the site): not single assignment.
    let redefine = |p: &mut crate::kernel_builder::SsaProgram, s: &HashMergeSite| {
        let x = p.ops[s.fma].dest.unwrap();
        let mut op = SsaOp::new(0, Engine::Valu, Opcode::Xor, Some(x), vec![Operand::VReg(s.input), Operand::VReg(s.input)]);
        op.id = p.ops.len();
        let at = s.xor + 1;
        p.ops.insert(at, op);
    };
    let mut p = raw_program(&ProblemSpec { rounds: 2, ..Default::default() });
    let site = {
        let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
        merge_rule(ProposalSource::Ops).find_sites(&g).0.remove(0)
    };
    redefine(&mut p, &site);
    {
        let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
        assert_eq!(g.redefined().len(), 1);
        let err = merge_rule(ProposalSource::Ops).match_at(&g, site.fma).unwrap_err();
        assert!(err.0.contains("defined more than once"), "{:?}", err);
    }
    let err = Surgery::new(clone_program(&p)).err().expect("Surgery refuses a program that is not single assignment");
    assert!(err.contains("more than once"), "{}", err);
    // The pass leaves such a program unchanged and says why.
    let before = canonical_ops(&p);
    let (after, report) = apply_hash_merge(p, &merge_rule(ProposalSource::Ops), |_| true).unwrap();
    assert!(report.refused.as_deref().map_or(false, |r| r.contains("more than once")), "{:?}", report.refused);
    assert_eq!(canonical_ops(&after), before);
}

#[test]
fn test_hash_merge_treats_a_select_delta_as_a_read() {
    use crate::types::{Engine, Opcode, Operand, VReg};
    // A SelectChoice whose delta (read by its multiply_add lowering, not in srcs) is x.
    let result = rewrite_mutated(|p, s| {
        let x = p.ops[s.fma].dest.unwrap();
        let mut op = SsaOp::new(0, Engine::Valu, Opcode::SelectChoice, Some(VReg::vector(p.vreg_count)), vec![
            Operand::VReg(s.input),
            Operand::VReg(s.input),
            Operand::VReg(s.input),
        ]);
        p.vreg_count += 1;
        op.delta_vreg = Some(x);
        p.ops.insert(s.xor + 1, op);
    });
    assert_first_site_rejected(result, "3 readers");
}

#[test]
fn test_hash_merge_rejects_vector_group_sites_per_site() {
    use crate::types::VReg;
    // r is the key of a vector group (its lanes read elsewhere): renaming r would orphan them.
    let lanes = |p: &mut crate::kernel_builder::SsaProgram| {
        let l: Vec<VReg> = (0..8).map(|k| VReg::scalar(p.vreg_count + k)).collect();
        p.vreg_count += 8;
        l
    };
    let result = rewrite_mutated(|p, s| {
        let r = p.ops[s.xor].dest.unwrap();
        let l = lanes(p);
        p.vector_group_lanes.insert(r, l);
    });
    assert_first_site_rejected(result, "vector group");
    // A lane of x named in keep_live.
    let result = rewrite_mutated(|p, s| {
        let x = p.ops[s.fma].dest.unwrap();
        let l = lanes(p);
        let first = l[0];
        p.vector_group_lanes.insert(x, l);
        p.ops.last_mut().unwrap().keep_live = Some(vec![first]);
    });
    assert_first_site_rejected(result, "kept live");
}

#[test]
fn test_hash_merge_keeps_foreign_const_alternatives() {
    use crate::types::{ConstEncoding, Opcode};
    // A ConstChoice whose alternatives are not the emitter's function of the constants before
    // it: the program is still rewritten. Its alternatives are kept, less any ALU derivation
    // whose operands no longer exist before it.
    let mut p = raw_program(&ProblemSpec { rounds: 2, ..Default::default() });
    let c2 = crate::kernel_builder::config::HASH_STAGES[2].1;
    let target = p.ops.iter().rposition(|o| o.opcode == Opcode::ConstChoice && o.produces_constant.map_or(false, |v| v > 100)).unwrap();
    let value = p.ops[target].produces_constant.unwrap();
    let derived = ConstEncoding::Add { a_value: c2, b_value: value.wrapping_sub(c2) };
    let foreign = ConstEncoding::Add { a_value: 1, b_value: 2 };
    p.ops[target].const_alternatives = Some(vec![ConstEncoding::Load, foreign.clone(), derived.clone()]);
    let (after, report) = apply_hash_merge(p, &merge_rule(ProposalSource::Ops), |_| true).unwrap();
    assert_eq!(report.sites, 64);
    let op = after.ops.iter().find(|o| o.opcode == Opcode::ConstChoice && o.produces_constant == Some(value)).unwrap();
    // C2 is gone, so the derivation from it is dropped; the rest is kept as given.
    assert_eq!(op.const_alternatives.as_ref().unwrap(), &vec![ConstEncoding::Load, foreign]);
    assert_eq!(report.surgery.foreign_alternatives, 1);
}

#[test]
fn test_hash_merge_cost_cannot_block_the_rewrite() {
    use crate::types::{Engine, Opcode, Operand};
    // An op on an engine the cost categories do not know (a debug op).
    let result = rewrite_mutated(|p, s| {
        let op = SsaOp::new(0, Engine::Debug, Opcode::DebugPrint, None, vec![Operand::VReg(s.input)]);
        p.ops.insert(s.fma, op);
    });
    let (_, report) = result.unwrap();
    assert_eq!(report.sites, 64);
    assert!(report.cost_error.as_deref().map_or(false, |e| e.contains("Debug")), "{:?}", report.cost_error);
}

#[test]
fn test_hash_merge_checks_the_constant_annotation() {
    use crate::types::Operand;
    // The shift constant's immediate says 10 while its annotation still says 9: the matcher
    // must not trust the annotation.
    let (err, n) = mutate_first_site(|p, s| {
        let k_scalar = p.ops[p.vreg_def[&s.operands[3]]].srcs[0].as_vreg().unwrap();
        let def = p.vreg_def[&k_scalar];
        p.ops[def].srcs[0] = Operand::Imm(10);
    });
    assert!(err.unwrap().contains("shift amount is not a broadcast constant"));
    assert_eq!(n, 0);
}

// ----------------------------------------------------------------------------
// The graph view against dead code elimination; other shapes; unsupported contexts
// ----------------------------------------------------------------------------

#[test]
fn test_graph_liveness_equals_dead_code_elimination() {
    use crate::kernel_builder::config::CachedSelect;
    let mut zero_prune_bits = ProblemSpec { rounds: 16, batch_size: 32, prune_dead_index: true, ..Default::default() };
    zero_prune_bits.cached_select = CachedSelect::Bits;
    zero_prune_bits.tree_reduce_max_depth = 4;
    let mut programs = vec![
        ("default 4 rounds", SsaTickTockKernelBuilder::new(KernelConfig::new(10, 2047, 32, 4)).build()),
        ("zero-based prune bits", SsaTickTockKernelBuilder::new(KernelConfig { spec: zero_prune_bits, ..KernelConfig::new(10, 2047, 32, 16) }).build()),
        ("canonical mirrored", canonical_program(&context(32, SelectBitSource::Index))),
        ("instantiated", super::cost::instantiate_tree_access(&context(32, SelectBitSource::Index)).unwrap().0),
        ("instantiated, path bits", super::cost::instantiate_tree_access(&context(32, SelectBitSource::Path)).unwrap().0),
    ];
    let mut values_only = context(32, SelectBitSource::Index);
    values_only.values_only = true;
    programs.push(("values only", canonical_program(&values_only)));
    for (label, mut program) in programs {
        let graph = SsaGraph::new(&program.ops, &program.vector_group_lanes);
        let by_graph: Vec<usize> =
            graph.live().iter().enumerate().filter(|(_, l)| **l).map(|(i, _)| program.ops[i].id).collect();
        let (kept, _, _) = eliminate_ssa_dead_code(
            std::mem::take(&mut program.ops),
            &mut program.vreg_def,
            &mut program.const_alternatives,
            program.zero_vreg,
            Some(&program.vector_group_lanes),
        );
        let by_dce: Vec<usize> = kept.iter().map(|o| o.id).collect();
        assert_eq!(by_graph, by_dce, "{}", label);
    }
}

fn shape(rounds: usize, forest_height: usize, batch_size: usize) -> ProblemSpec {
    ProblemSpec {
        rounds,
        forest_height,
        n_nodes: (1 << (forest_height + 1)) - 1,
        batch_size,
        merge_hash_stages: true,
        index_form: IndexForm::Mirrored,
        ..Default::default()
    }
}

#[test]
fn test_derived_table_predicts_emitted_counts_at_other_shapes() {
    // Rounds 15 (the final index is at depth 4 and reads L_4), height 8 (wrap after level 8,
    // so levels 1 to 4 are visited twice in 15 rounds), 26 rounds (three visits of some levels).
    for spec in [shape(15, 10, 32), shape(15, 8, 32), shape(26, 10, 32), shape(9, 6, 16)] {
        let table = derive_tree_access_costs(&spec).unwrap();
        if !table.problems.is_empty() {
            // Reported, not fitted: the prediction must then not be trusted, so nothing to compare.
            eprintln!("rounds {} height {}: {:?}", spec.rounds, spec.forest_height, table.problems);
            continue;
        }
        for (i, plan) in test_plans(&spec, 6).iter().enumerate() {
            assert_eq!(table.predict(plan), emitted_counts(&spec, plan), "rounds {} height {} plan {}", spec.rounds, spec.forest_height, i);
        }
    }
}

#[test]
fn test_unsupported_contexts_report_problems() {
    let zero_based = ProblemSpec { rounds: 16, batch_size: 32, merge_hash_stages: true, ..Default::default() };
    let table = derive_tree_access_costs(&zero_based).unwrap();
    assert!(table.problems.iter().any(|p| p.contains("zero-based")), "{:?}", table.problems.first());
    let mut values_only = context(32, SelectBitSource::Index);
    values_only.values_only = true;
    let table = derive_tree_access_costs(&values_only).unwrap();
    assert!(table.problems.iter().any(|p| p.contains("per-group")), "{:?}", table.problems.first());
}

#[test]
fn test_plan_dependent_const_category_is_reported() {
    // 14 = 6 + 8 is ALU-derivable only if the lazy constant 6 (a gather base's
    // level-end constant in a choice round) was created before it, which depends on the plan.
    use super::cost::{plan_dependent_const_categories, Liveness, Shape};
    use crate::types::{Engine, Opcode, RegionTag};
    let spec = context(32, SelectBitSource::Index);
    let mut emit = crate::kernel_builder::SsaEmitContext::new();
    emit.reserve_zero_vreg();
    emit.emit_const_choice(8);
    emit.set_round_context(2, 2);
    emit.set_region(RegionKind::TreeAccess(AccessEncoding::Gather), None);
    emit.emit_const_choice(6);
    emit.clear_region();
    emit.clear_round_context();
    emit.emit_const_choice(14);
    let ops = emit.take_ops();
    assert!(ops.iter().all(|o| o.engine == Engine::Load && o.opcode == Opcode::ConstChoice));
    let tag: Option<RegionTag> = ops[1].region;
    assert!(tag.is_some());
    let live = vec![Liveness::Always; ops.len()];
    let users = vec![Vec::new(); ops.len()];
    let problems = plan_dependent_const_categories(&ops, &live, &users, true, &Shape::new(&spec));
    assert_eq!(problems.len(), 1, "{:?}", problems);
    assert!(problems[0].starts_with("constant 14"), "{:?}", problems);
}
