//! Tests of the transform module: the derived cost table predicts the emitted programs, and
//! the matcher never accepts a site on the word of a tag.

use super::cost::{category, derive_tree_access_costs, Counts};
use super::graph::SsaGraph;
use super::rule::Rule;
use super::tree_access::TreeAccessRule;
use crate::kernel_builder::config::{AccessEncoding, IndexForm, KernelConfig, ProblemSpec, SelectBitSource};
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
    let wrong: Vec<_> = rule().propose(&graph).into_iter().filter(|p| wrong_ids.contains(&p.region_id)).collect();
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
fn canonical_ops(program: &crate::kernel_builder::SsaProgram) -> Vec<String> {
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
