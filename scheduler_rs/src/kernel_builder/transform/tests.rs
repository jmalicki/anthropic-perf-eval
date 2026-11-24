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

/// Counts of the program a plan emits, after dead code elimination.
fn emitted_counts(spec: &ProblemSpec, plan: &[Vec<AccessEncoding>]) -> Counts {
    let config = KernelConfig {
        spec: ProblemSpec { access_plan: Some(plan.to_vec()), ..spec.clone() },
        ..KernelConfig::new(spec.forest_height, spec.n_nodes, spec.batch_size, spec.rounds)
    };
    let mut program = SsaTickTockKernelBuilder::new(config).build();
    let (ops, _, _) = eliminate_ssa_dead_code(
        std::mem::take(&mut program.ops),
        &mut program.vreg_def,
        &mut program.const_alternatives,
        program.zero_vreg,
        Some(&program.vector_group_lanes),
    );
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
