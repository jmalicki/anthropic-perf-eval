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
    // The leak lands on exactly one known op, or none: the first op written after the hash
    // constants, i.e. the first op of the selection setup (an ALU add, no region). Any other
    // label difference fails the gate.
    use crate::types::{EmissionPhase, Engine, Opcode};
    let first_selection_setup =
        rewritten.ops.iter().position(|o| o.source_info.as_ref().map_or(false, |s| s.phase == EmissionPhase::ValuSelectionInit));
    for &i in &leaked {
        let op = &rewritten.ops[i];
        assert_eq!(Some(i), first_selection_setup, "{}: label leak on {:?}", label, op);
        assert!(op.region.is_none() && op.engine == Engine::Alu && op.opcode == Opcode::Add, "{}: label leak on {:?}", label, op);
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
        // The proved instance, not the general identity (which also gives these constants).
        assert!(
            s.facts.iter().any(|f| f.name == "proof_instance" && f.value == "R8-hash-stage-merge (stages 2+3)"),
            "{:?}",
            s.facts
        );
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

/// The value an ALU encoding computes from its operands, mod 2^32 (`None` for Load / AddImm).
fn encoding_value(e: &crate::types::ConstEncoding) -> Option<i64> {
    use crate::types::ConstEncoding;
    const MASK: i64 = 0xFFFF_FFFF;
    Some(match *e {
        ConstEncoding::Load | ConstEncoding::AddImm { .. } => return None,
        ConstEncoding::Add { a_value, b_value } => a_value.wrapping_add(b_value) & MASK,
        ConstEncoding::Sub { a_value, b_value } => a_value.wrapping_sub(b_value) & MASK,
        ConstEncoding::Mul { a_value, b_value } => a_value.wrapping_mul(b_value) & MASK,
        ConstEncoding::Shift { base_value, shift_amount } => (base_value << shift_amount) & MASK,
    })
}

#[test]
fn test_hash_merge_keeps_foreign_const_alternatives() {
    use crate::types::{ConstEncoding, Opcode, Operand};
    // A ConstChoice whose alternatives are not the emitter's function of the constants before
    // it: the program is still rewritten. Its alternatives are kept as given, less any ALU
    // derivation whose operands no longer exist before it.
    let spec = ProblemSpec { rounds: 2, ..Default::default() };
    let consts = |p: &crate::kernel_builder::SsaProgram| -> std::collections::BTreeSet<i64> {
        p.ops.iter().filter(|o| o.opcode == Opcode::ConstChoice).filter_map(|o| o.produces_constant).collect()
    };
    let mut p = raw_program(&spec);
    let (merged, _) = apply_hash_merge(raw_program(&spec), &merge_rule(ProposalSource::Ops), |_| true).unwrap();
    let (before, after_consts) = (consts(&p), consts(&merged));
    // A constant the rewrite removes, and the first one after it that stays and is no site's
    // operand. Its value becomes `g + w` (w: a surviving constant before it), so that
    // `Add { g, w }` is a valid derivation that loses its operand g in the rewrite.
    let site_values: std::collections::BTreeSet<i64> = {
        let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
        merge_rule(ProposalSource::Ops).find_sites(&g).0.iter().flat_map(|s| s.values).collect()
    };
    let is_const = |i: usize| p.ops[i].opcode == Opcode::ConstChoice;
    let value_at = |i: usize| p.ops[i].produces_constant.unwrap();
    let ig = (0..p.ops.len()).find(|&i| is_const(i) && !after_consts.contains(&value_at(i))).expect("a removed constant");
    let g = value_at(ig);
    let target = (ig + 1..p.ops.len())
        .find(|&i| is_const(i) && after_consts.contains(&value_at(i)) && !site_values.contains(&value_at(i)))
        .expect("a surviving constant after it");
    let w = (0..target)
        .filter(|&i| is_const(i) && after_consts.contains(&value_at(i)) && value_at(i) != 0)
        .map(value_at)
        .find(|&w| !before.contains(&((g + w) & 0xFFFF_FFFF)))
        .expect("a surviving operand");
    let value = (g + w) & 0xFFFF_FFFF;
    let derived = ConstEncoding::Add { a_value: g, b_value: w };
    // Not the emitter's order (Load, AddImm, derivations).
    let foreign = vec![derived.clone(), ConstEncoding::AddImm { value }, ConstEncoding::Load];
    for e in &foreign {
        assert!(encoding_value(e).map_or(true, |v| v == value), "{:?} does not produce {}", e, value);
    }
    p.ops[target].srcs[0] = Operand::Imm(value);
    p.ops[target].produces_constant = Some(value);
    p.ops[target].const_alternatives = Some(foreign);
    let (after, report) = apply_hash_merge(p, &merge_rule(ProposalSource::Ops), |_| true).unwrap();
    assert_eq!(report.sites, 64);
    let op = after.ops.iter().find(|o| o.opcode == Opcode::ConstChoice && o.produces_constant == Some(value)).unwrap();
    // g is gone, so the derivation from it is dropped; the rest is kept in the given order.
    assert_eq!(op.const_alternatives.as_ref().unwrap(), &vec![ConstEncoding::AddImm { value }, ConstEncoding::Load]);
    assert_eq!(report.surgery.foreign_alternatives, 1);
}

#[test]
fn test_merged_constants_reduce_addends_before_adding() {
    use super::hash_merge::merged_constants;
    // Addends outside 0..2^32 (the matcher does not range-check them): the sum is taken mod
    // 2^32 without overflowing i64 (which panics in debug builds).
    let (m1, c1, m2, c2) = merged_constants(3, i64::MAX, i64::MAX, 4);
    assert_eq!(c1, (0xFFFF_FFFFi64 * 2) & 0xFFFF_FFFF);
    assert_eq!((m1, m2, c2), (3, 48, (i64::MAX << 4) & 0xFFFF_FFFF));
    assert_eq!(merged_constants(1, -1, 2, 0).1, 1);
    assert_eq!(merged_constants(1, i64::MIN, -1, 0).1, 0xFFFF_FFFF);
}

#[test]
fn test_hash_merge_leaves_kept_live_operands_in_place() {
    // An op before the anchor names a site operand in keep_live (a lifetime extension, not a
    // read). The regroup must leave that operand where it is: moved after the anchor, it
    // would be defined after the op that keeps it live.
    let (_, base) = rewrite_mutated(|_, _| {}).unwrap();
    assert!(base.regroup_skipped.is_none());
    // The operands of the first site that something besides the sites reads (the others are
    // removed with the sites; a keep_live naming one of those is a different refusal).
    let shared: Vec<usize> = {
        let p = raw_program(&ProblemSpec { rounds: 2, ..Default::default() });
        let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
        let sites = merge_rule(ProposalSource::Ops).find_sites(&g).0;
        let removed: std::collections::BTreeSet<usize> = sites.iter().flat_map(|s| s.ops()).collect();
        (0..4).filter(|&k| g.readers(g.def_index(sites[0].operands[k]).unwrap()).iter().any(|r| !removed.contains(r))).collect()
    };
    assert!(!shared.is_empty());
    let mut fewer_moved = false;
    for &pick in &shared {
        let (after, report) = rewrite_mutated(|p, s| {
            let v = s.operands[pick];
            let d = p.ops.iter().position(|o| o.dest == Some(v)).unwrap();
            p.ops[d + 1].keep_live = Some(vec![v]);
        })
        .unwrap_or_else(|e| panic!("operand {}: {}", pick, e));
        assert_eq!(report.sites, 64, "operand {}: {:?}", pick, report.rejections);
        assert!(report.regroup_skipped.is_none(), "operand {}: {:?}", pick, report.regroup_skipped);
        super::surgery::Surgery::check(&after).unwrap();
        fewer_moved |= report.surgery.moved < base.surgery.moved;
    }
    // At least one kept-live operand is one the regroup would otherwise move.
    assert!(fewer_moved && base.surgery.moved > 0);
}

/// One synthetic hash site: 0-2 address constants (load, two stores), 3 vload a, 4-5
/// constant and broadcast M, 6-7 C, 8-9 D, 10-11 K, then the site and its stores.
fn one_site_program() -> crate::kernel_builder::SsaProgram {
    synthetic_hash_program(5, 1).0
}

#[test]
fn test_surgery_moves_an_op_once() {
    use super::surgery::Surgery;
    let p = one_site_program();
    let m = p.ops[4].dest.unwrap();
    let mut surgery = Surgery::new(clone_program(&p)).unwrap();
    surgery.move_after(&[4, 5], 10).unwrap();
    // A second placement of op 4 is refused, and the first one stands.
    let err = surgery.move_after(&[4], 11).unwrap_err();
    assert!(err.contains("already moved"), "{}", err);
    let (after, report) = surgery.finish().unwrap();
    assert_eq!(report.moved, 2);
    assert_eq!(after.ops.iter().filter(|o| o.dest == Some(m)).count(), 1);
    let at = |v| after.ops.iter().position(|o| o.dest == Some(v)).unwrap();
    assert!(at(p.ops[10].dest.unwrap()) < at(m) && at(m) < at(p.ops[11].dest.unwrap()));
    // A removed op has no placement to move.
    let mut surgery = Surgery::new(p).unwrap();
    surgery.remove(6);
    let err = surgery.move_after(&[6], 11).unwrap_err();
    assert!(err.contains("removed"), "{}", err);
}

#[test]
fn test_surgery_reuses_a_moved_constant_only_where_it_now_is() {
    use super::surgery::{Stamp, Surgery};
    let p = one_site_program();
    let (m, bm) = (p.ops[4].dest.unwrap(), p.ops[5].dest.unwrap());
    let value = p.ops[4].produces_constant.unwrap();
    let stamp = Stamp::of(&p.ops[4]);
    let mut surgery = Surgery::new(clone_program(&p)).unwrap();
    surgery.move_after(&[4, 5], 10).unwrap();
    // Before op 10 the moved constant and broadcast are not there yet: new ones.
    let early = surgery.constant(value, 6, &stamp);
    assert_ne!(early, m);
    assert_ne!(surgery.broadcast(m, 6, &stamp), bm);
    // From op 10 on they are.
    assert_eq!(surgery.constant(value, 10, &stamp), m);
    assert_eq!(surgery.broadcast(m, 11, &stamp), bm);
    surgery.finish().unwrap();
}

#[test]
fn test_surgery_does_not_move_a_kept_live_value_past_its_keeper() {
    use super::surgery::Surgery;
    let mut p = one_site_program();
    p.ops[6].keep_live = Some(vec![p.ops[4].dest.unwrap()]);
    let mut surgery = Surgery::new(clone_program(&p)).unwrap();
    let err = surgery.move_after(&[4, 5], 10).unwrap_err();
    assert!(err.contains("kept live by op 6"), "{}", err);
    // Nothing was recorded.
    let (after, report) = surgery.finish().unwrap();
    assert_eq!(report.moved, 0);
    assert_eq!(after.ops[4].dest, p.ops[4].dest);
    // Past the keeper too, it may move.
    let mut surgery = Surgery::new(p).unwrap();
    surgery.move_after(&[4, 5, 6, 7], 10).unwrap();
    surgery.finish().unwrap();
}

#[test]
fn test_surgery_checks_where_moved_readers_and_keepers_are() {
    use super::surgery::Surgery;
    let p = one_site_program();
    // Op 5 (broadcast of M) reads op 4 (M). Moved after op 8, the reader is before op 10:
    // M cannot follow op 10.
    let mut surgery = Surgery::new(clone_program(&p)).unwrap();
    surgery.move_after(&[5], 8).unwrap();
    let err = surgery.move_after(&[4], 10).unwrap_err();
    assert!(err.contains("read by op 5"), "{}", err);
    surgery.finish().unwrap();
    // At the same anchor the reader moved first comes first: its operand cannot follow.
    let mut surgery = Surgery::new(clone_program(&p)).unwrap();
    surgery.move_after(&[5], 10).unwrap();
    let err = surgery.move_after(&[4], 10).unwrap_err();
    assert!(err.contains("read by op 5"), "{}", err);
    surgery.finish().unwrap();
    // The other order is valid: the reader after op 10, its operand after op 8.
    let mut surgery = Surgery::new(clone_program(&p)).unwrap();
    surgery.move_after(&[5], 10).unwrap();
    surgery.move_after(&[4], 8).unwrap();
    surgery.finish().unwrap();
    // The same for an op that names M in keep_live: moved after op 8, it is before op 10.
    let mut q = clone_program(&p);
    q.ops[6].keep_live = Some(vec![p.ops[4].dest.unwrap()]);
    let mut surgery = Surgery::new(q).unwrap();
    surgery.move_after(&[6, 7], 8).unwrap();
    let err = surgery.move_after(&[4, 5], 10).unwrap_err();
    assert!(err.contains("kept live by op 6"), "{}", err);
    surgery.finish().unwrap();
}

#[test]
fn test_surgery_drops_a_moved_op_that_is_then_removed() {
    use super::surgery::{Stamp, Surgery};
    // Move the broadcast of M after op 10, then replace it there by a new op with the same
    // destination: the moved copy must not be placed as well.
    let p = one_site_program();
    let bm = p.ops[5].dest.unwrap();
    let mut surgery = Surgery::new(clone_program(&p)).unwrap();
    surgery.move_after(&[4, 5], 10).unwrap();
    surgery.remove(5);
    surgery.insert_after(10, p.ops[5].clone(), &Stamp::of(&p.ops[5]));
    let (after, _) = surgery.finish().unwrap();
    assert_eq!(after.ops.iter().filter(|o| o.dest == Some(bm)).count(), 1);
}

#[test]
fn test_hash_merge_keeps_a_kept_live_constant() {
    use crate::types::{Opcode, Operand};
    // m = 3, c = 5, d = 7, k = 1, and the surviving last store names the scalar constant 7 in
    // keep_live. The rewrite removes the last reader of 7 (the broadcast of D); dead code
    // elimination must keep 7 for the store. Without the keeper, 7 goes with the site.
    let build = |keep: bool| {
        let (mut p, _) = synthetic_hash_program(11, 1);
        let consts: Vec<usize> = (0..p.ops.len()).filter(|&i| p.ops[i].opcode == Opcode::ConstChoice).collect();
        for (&i, value) in consts.iter().zip([3i64, 5, 7, 1]) {
            p.ops[i].srcs[0] = Operand::Imm(value);
            p.ops[i].produces_constant = Some(value);
            p.ops[i + 1].produces_constant = Some(value);
        }
        let seven = p.ops[consts[2]].dest.unwrap();
        if keep {
            p.ops.last_mut().unwrap().keep_live = Some(vec![seven]);
        }
        (p, seven)
    };
    let defines = |p: &crate::kernel_builder::SsaProgram, v| p.ops.iter().any(|o| o.dest == Some(v));
    let (p, seven) = build(true);
    let before = clone_program(&p);
    let (after, report) = apply_hash_merge(p, &merge_rule(ProposalSource::Ops), |_| true).unwrap();
    assert_eq!(report.sites, 1, "{:?}", report.rejections);
    assert!(defines(&after, seven));
    super::surgery::Surgery::check(&after).unwrap();
    assert_same_values(&before, &after, 3);
    let (p, seven) = build(false);
    let (after, report) = apply_hash_merge(p, &merge_rule(ProposalSource::Ops), |_| true).unwrap();
    assert_eq!(report.sites, 1);
    assert!(!defines(&after, seven));
    assert!(report.surgery.cascaded.iter().any(|&(_, _, c)| c == Some(7)), "{:?}", report.surgery.cascaded);
}

#[test]
fn test_transforms_leave_no_keep_live_only_reference() {
    use crate::kernel_builder::config::TransformKind::{self, *};
    use crate::kernel_builder::pipeline::passes::transform::apply_transform_with;
    use crate::types::VReg;
    // Pipeline dead code elimination (DcePass, after TransformPass) follows `srcs` only
    // (BUGS.md). On the emitter's programs every keep_live value is also read through `srcs`
    // (the round R-2 hash output, read by round R-1), and every transform keeps it that way,
    // alone and combined.
    let combos: [&[TransformKind]; 4] = [&[HashMerge], &[PruneIndex], &[ValuesOnly], &[HashMerge, PruneIndex, ValuesOnly]];
    for rounds in [2, 16] {
        let spec = ProblemSpec { rounds, ..Default::default() };
        for combo in combos {
            let mut p = raw_program(&spec);
            for &t in combo {
                let (next, outcome) = apply_transform_with(p, t, Some(spec.batch_size), Some(spec.n_nodes)).unwrap();
                assert!(outcome.sites > 0 && outcome.refused.is_none(), "rounds {} {:?}: {}", rounds, t, outcome.summary);
                p = next;
            }
            let read: std::collections::HashSet<VReg> = p.ops.iter().flat_map(|o| o.reads()).collect();
            let kept: Vec<VReg> = p.ops.iter().flat_map(|o| o.keep_live.iter().flatten().copied()).collect();
            assert_eq!(kept.len(), 32, "rounds {} {:?}", rounds, combo);
            assert!(kept.iter().all(|v| read.contains(v)), "rounds {} {:?}: a keep_live-only value", rounds, combo);
        }
    }
}

#[test]
fn test_hash_merge_drops_a_dead_keeper_of_a_removed_value() {
    use crate::types::{Engine, Opcode, Operand, VReg};
    // An unused broadcast (already dead) names the scalar D in keep_live. The rewrite removes
    // D's last live reader, so D goes, and the dead keeper with it.
    let mut p = one_site_program();
    let d = p.ops[8].dest.unwrap();
    let mut op = SsaOp::new(p.ops.len(), Engine::Valu, Opcode::Vbroadcast, Some(VReg::vector(p.vreg_count)), vec![Operand::VReg(p.ops[4].dest.unwrap())]);
    p.vreg_count += 1;
    op.keep_live = Some(vec![d]);
    p.ops.push(op);
    let (after, report) = apply_hash_merge(p, &merge_rule(ProposalSource::Ops), |_| true).unwrap();
    assert_eq!(report.sites, 1, "{:?}", report.rejections);
    assert!(!after.ops.iter().any(|o| o.dest == Some(d)));
    assert!(!after.ops.iter().any(|o| o.keep_live.is_some()));
    assert!(report.surgery.dropped_dead >= 1);
}

#[test]
fn test_hash_merge_keeps_a_value_read_by_a_select_delta() {
    use crate::types::{Engine, Opcode, Operand, VReg};
    // A live SelectChoice reads the broadcast of D as its delta (not in srcs). The rewrite
    // removes D's other reader (the site's add); the broadcast must stay for the select.
    let mut p = one_site_program();
    let (a, bd) = (p.ops[3].dest.unwrap(), p.ops[9].dest.unwrap());
    let sel = VReg::vector(p.vreg_count);
    p.vreg_count += 1;
    let mut op = SsaOp::new(p.ops.len(), Engine::Flow, Opcode::SelectChoice, Some(sel), vec![Operand::VReg(a), Operand::VReg(a), Operand::VReg(a)]);
    op.delta_vreg = Some(bd);
    p.ops.push(op);
    p.ops.push(SsaOp::new(p.ops.len(), Engine::Store, Opcode::Vstore, None, vec![Operand::Imm(20000), Operand::VReg(sel)]));
    let (after, report) = apply_hash_merge(p, &merge_rule(ProposalSource::Ops), |_| true).unwrap();
    assert_eq!(report.sites, 1, "{:?}", report.rejections);
    assert!(after.ops.iter().any(|o| o.dest == Some(bd)));
}

#[test]
fn test_surgery_rename_cycle_is_an_error() {
    use super::surgery::Surgery;
    let p = one_site_program();
    let (m, c) = (p.ops[5].dest.unwrap(), p.ops[7].dest.unwrap());
    // An identity rename is nothing.
    let mut surgery = Surgery::new(clone_program(&p)).unwrap();
    surgery.replace_uses(m, m);
    surgery.finish().unwrap();
    // A cycle is an error, not a panic.
    let mut surgery = Surgery::new(clone_program(&p)).unwrap();
    surgery.replace_uses(m, c);
    surgery.replace_uses(c, m);
    let err = surgery.finish().err().expect("a rename cycle is rejected");
    assert!(err.contains("rename cycle"), "{}", err);
}

#[test]
fn test_surgery_checks_inserted_readers_before_a_move() {
    use super::surgery::{Stamp, Surgery};
    use crate::types::{Engine, Opcode, Operand};
    // A new op inserted before op 8 reads M (op 4): M cannot move after op 10.
    let p = one_site_program();
    let m = p.ops[4].dest.unwrap();
    let mut surgery = Surgery::new(clone_program(&p)).unwrap();
    let dest = surgery.fresh(true);
    let op = SsaOp::new(0, Engine::Valu, Opcode::Vbroadcast, Some(dest), vec![Operand::VReg(m)]);
    surgery.insert_before(8, op, &Stamp::of(&p.ops[4]));
    let err = surgery.move_after(&[4, 5], 10).unwrap_err();
    assert!(err.contains("inserted next to op 8"), "{}", err);
    surgery.finish().unwrap();
}

#[test]
fn test_transform_pass_keeps_the_program_on_an_error() {
    use crate::kernel_builder::config::TransformKind;
    use crate::kernel_builder::pipeline::{PipelineConfig, PipelineContext};
    // prune-index without a batch size fails after hash-merge succeeded: the context keeps
    // the hash-merged program (for the error export), not an empty one.
    let spec = ProblemSpec { rounds: 2, ..Default::default() };
    let (merged, _) = apply_hash_merge(raw_program(&spec), &merge_rule(ProposalSource::Ops), |_| true).unwrap();
    let mut config = PipelineConfig::default();
    config.compiler.transforms = vec![TransformKind::HashMerge, TransformKind::PruneIndex];
    config.compiler.transform_batch_size = None;
    let mut ctx = PipelineContext::new(raw_program(&spec), config);
    let err = ctx.run_greedy_pipeline(false).err().expect("prune-index needs the batch size");
    assert!(err.contains("batch size"), "{}", err);
    // Every IR field is the hash-merged program's, and only hash-merge is recorded.
    let kept = crate::kernel_builder::SsaProgram {
        ops: ctx.ops.clone(),
        vreg_def: ctx.vreg_def.clone(),
        vreg_count: ctx.vreg_count,
        const_alternatives: ctx.const_alternatives.clone(),
        zero_vreg: ctx.zero_vreg,
        vector_group_lanes: ctx.vector_group_lanes.clone(),
    };
    assert_eq!(canonical_ops(&kept), canonical_ops(&merged));
    assert_eq!(kept.vreg_def, merged.vreg_def);
    assert_eq!(kept.vreg_count, merged.vreg_count);
    assert_eq!(kept.zero_vreg, merged.zero_vreg);
    assert_eq!(kept.vector_group_lanes, merged.vector_group_lanes);
    assert_eq!(format!("{:?}", kept.const_alternatives), format!("{:?}", merged.const_alternatives));
    assert_eq!(ctx.transform_sites, vec![("hash-merge".to_string(), 64)]);
}

#[test]
fn test_hash_merge_keeps_the_xor_operand_order() {
    use crate::types::Opcode;
    // A traced r = s ^ t (shift first) becomes h ^ l (the merged form of the shift first).
    let mut p = one_site_program();
    let xor = p.ops.iter().position(|o| o.opcode == Opcode::Xor).unwrap();
    let shl = p.ops.iter().position(|o| o.opcode == Opcode::Shl).unwrap();
    if p.ops[xor].srcs[0].as_vreg() != p.ops[shl].dest {
        p.ops[xor].srcs.swap(0, 1);
    }
    p.ops[xor].trace_id = Some("r".into());
    // k = 3 (not 0), so that h and l have different multipliers.
    p.ops[10].srcs[0] = crate::types::Operand::Imm(3);
    p.ops[10].produces_constant = Some(3);
    p.ops[11].produces_constant = Some(3);
    let (m, k) = (p.ops[4].produces_constant.unwrap(), 3);
    assert_ne!((m << k) & 0xFFFF_FFFF, m & 0xFFFF_FFFF);
    let (after, report) = apply_hash_merge(p, &merge_rule(ProposalSource::Ops), |_| true).unwrap();
    assert_eq!(report.sites, 1);
    let g = SsaGraph::new(&after.ops, &after.vector_group_lanes);
    let r = after.ops.iter().find(|o| o.trace_id.as_deref() == Some("r")).unwrap();
    let multiplier = |v| g.broadcast_const(g.def_op(v).unwrap().srcs[1].as_vreg().unwrap());
    assert_eq!(multiplier(r.srcs[0].as_vreg().unwrap()), Some((m << k) & 0xFFFF_FFFF), "first operand is h");
    assert_eq!(multiplier(r.srcs[1].as_vreg().unwrap()), Some(m & 0xFFFF_FFFF), "second operand is l");
}

#[test]
fn test_surgery_drops_a_dead_reader_of_an_alias_whose_lanes_are_gone() {
    use super::surgery::Surgery;
    use crate::types::{Engine, Opcode, Operand, VReg};
    // Eight scalar lanes assembled under an alias K, read only by an already-dead op. Removing
    // a lane leaves K undefined: the dead reader goes with it.
    let mut p = one_site_program();
    let lanes: Vec<VReg> = (0..8).map(|k| VReg::scalar(p.vreg_count + k)).collect();
    let key = VReg::vector(p.vreg_count + 8);
    let dead = VReg::vector(p.vreg_count + 9);
    p.vreg_count += 10;
    let first_lane = p.ops.len();
    for (k, &l) in lanes.iter().enumerate() {
        let mut op = SsaOp::new(p.ops.len(), Engine::Load, Opcode::ConstChoice, Some(l), vec![Operand::Imm(100 + k as i64)]);
        op.produces_constant = Some(100 + k as i64);
        p.ops.push(op);
    }
    p.vector_group_lanes.insert(key, lanes);
    p.ops.push(SsaOp::new(p.ops.len(), Engine::Valu, Opcode::Add, Some(dead), vec![Operand::VReg(key), Operand::VReg(key)]));
    let mut surgery = Surgery::new(p).unwrap();
    surgery.remove(first_lane);
    let (after, report) = surgery.finish().unwrap();
    assert!(!after.ops.iter().any(|o| o.dest == Some(dead)));
    assert!(report.dropped_dead >= 1);
}

#[test]
fn test_surgery_refuses_to_lose_a_trace_id() {
    use super::surgery::{Finished, Stamp, Surgery};
    use crate::types::Opcode;
    // The site's XOR r = t ^ s carries a trace id. Removing it (readers moved to t) loses the
    // id: refused. Replacing it by a new op that carries the id is accepted.
    let mut p = one_site_program();
    let xor = p.ops.iter().position(|o| o.opcode == Opcode::Xor).unwrap();
    p.ops[xor].trace_id = Some("r".into());
    let (r, t) = (p.ops[xor].dest.unwrap(), p.ops[xor].srcs[0].as_vreg().unwrap());
    let mut surgery = Surgery::new(clone_program(&p)).unwrap();
    surgery.replace_uses(r, t);
    surgery.remove(xor);
    match surgery.finish_or_refuse().unwrap() {
        Finished::Refused(_, why) => assert!(why.contains("trace id"), "{}", why),
        Finished::Done(..) => panic!("a lost trace id was accepted"),
    }
    let mut surgery = Surgery::new(clone_program(&p)).unwrap();
    let dest = surgery.fresh(true);
    let mut op = p.ops[xor].clone();
    op.dest = Some(dest);
    surgery.insert_after(xor, op, &Stamp::of(&p.ops[xor]));
    surgery.replace_uses(r, dest);
    surgery.remove(xor);
    assert!(matches!(surgery.finish_or_refuse().unwrap(), Finished::Done(..)));
}

#[test]
fn test_surgery_rejects_a_second_definition_before_dead_code_elimination() {
    use super::surgery::{Stamp, Surgery};
    use crate::types::{Engine, Opcode, Operand};
    // A caller-built op redefining an existing VReg right before it: liveness would follow
    // the first definer and silently drop the original one.
    let p = one_site_program();
    let m = p.ops[4].dest.unwrap();
    let mut op = SsaOp::new(0, Engine::Load, Opcode::ConstChoice, Some(m), vec![Operand::Imm(9)]);
    op.produces_constant = Some(9);
    let mut surgery = Surgery::new(clone_program(&p)).unwrap();
    surgery.insert_before(4, op, &Stamp::of(&p.ops[4]));
    let err = surgery.finish().err().expect("a second definer is rejected");
    assert!(err.contains("defined more than once"), "{}", err);
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
    // Every shape is expected to derive without problems; a shape that reports one is a
    // regression here, not a reason to skip it silently.
    let mut compared = 0;
    for spec in [shape(15, 10, 32), shape(15, 8, 32), shape(26, 10, 32), shape(9, 6, 16)] {
        let table = derive_tree_access_costs(&spec).unwrap();
        assert!(
            table.problems.is_empty(),
            "rounds {} height {}: expected an exact table, got problems {:?}",
            spec.rounds,
            spec.forest_height,
            table.problems
        );
        for (i, plan) in test_plans(&spec, 6).iter().enumerate() {
            assert_eq!(table.predict(plan), emitted_counts(&spec, plan), "rounds {} height {} plan {}", spec.rounds, spec.forest_height, i);
        }
        compared += 1;
    }
    assert_eq!(compared, 4);
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

#[test]
fn test_mirrored_gather_base_must_be_a_level_end() {
    // The mirrored gather base is forest_p + L_d. A constant that is no level end is rejected
    // (the depth search used to run to d = 62 and overflow); 6 = L_2 still reads as depth 2,
    // which the tag (depth 3) then contradicts.
    let spec = context(32, SelectBitSource::Index);
    let base_const = |p: &crate::kernel_builder::SsaProgram| -> usize {
        let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
        let prop = rule().propose(&g).into_iter().find(|p| p.round == Some(3) && p.group == Some(0)).unwrap();
        let sub = *prop.ops.iter().find(|&&i| p.ops[i].opcode == crate::types::Opcode::Sub).unwrap();
        let bcast = g.def_op(p.ops[sub].srcs[0].as_vreg().unwrap()).unwrap();
        let add = g.def_op(bcast.srcs[0].as_vreg().unwrap()).unwrap();
        g.def_index(add.srcs[1].as_vreg().unwrap()).unwrap()
    };
    let check = |value: i64| -> Result<(), String> {
        let mut p = canonical_program(&spec);
        let k = base_const(&p);
        assert_eq!(p.ops[k].produces_constant, Some(14), "depth 3's level end");
        p.ops[k].srcs[0] = crate::types::Operand::Imm(value);
        p.ops[k].produces_constant = Some(value);
        let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
        let prop = rule().propose(&g).into_iter().find(|p| p.round == Some(3) && p.group == Some(0)).unwrap();
        rule().match_site(&g, &prop).map(|_| ()).map_err(|e| e.0)
    };
    let err = check(5).unwrap_err();
    assert!(err.contains("not a level-end index"), "{}", err);
    let err = check((1i64 << 40) + 7).unwrap_err();
    assert!(err.contains("not a level-end index"), "{}", err);
    let err = check(6).unwrap_err();
    assert!(err.contains("the address base is level 2"), "{}", err);
}
