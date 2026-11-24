//! Cost by instantiation: the op counts of every tree-access alternative and of the shared
//! setup they need, derived from one program that holds them all.
//!
//! `derive_tree_access_costs` emits the canonical program (depth 0 from the shared root,
//! every other round gathering) with every bit-driven alternative of every choice site built
//! next to its gather (`TreeAccessInstantiation`) and the select tables of every cached level.
//! Then, for every op, it computes which choices keep it alive, exactly as dead code
//! elimination would: an op of a candidate lives iff that candidate is chosen; any other op
//! lives iff some user lives; stores and barriers always live. That liveness condition is an
//! OR of (round, group, encoding) choices, which is turned into the features of the
//! calibrated structure model (`scripts/emission_structure.py`, `plan_features`):
//!
//! * a resource used by every group of the rounds at depth `d` that use some encodings is the
//!   feature of that set of encodings (`table:d`, `pairs:d`, `pairs_high:d`, `mobius:d`), or
//!   `gather_depth:d` / `gather_round:r` for gathers;
//! * an OR over several such features is expanded by inclusion-exclusion, which yields the
//!   product terms `and:table:a+table:b[+...]` for setup shared between levels (the derived
//!   scalar constants that address the cached nodes);
//! * the per-site difference to gathering is `groups:d:e`, checked to be the same at every
//!   site of the depth.
//!
//! Anything that does not fit (a resource of some groups only, a product the model has no
//! feature for) is reported in `problems`, never rounded into a coefficient.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{json, Value};

use crate::types::{ConstEncoding, Engine, Opcode, SsaOp};

use super::super::config::{AccessEncoding, CachedSelect, IndexForm, KernelConfig, ProblemSpec, SelectBitSource, MAX_TREE_REDUCE_DEPTH};
use super::super::tick_tock_ssa::SsaTickTockKernelBuilder;
use super::super::emit_ssa::is_dce_root;
use super::graph::SsaGraph;
use super::tree_access::{encoding_name, TreeAccessInstantiation, BIT_DRIVEN, TREE_ACCESS_ALTERNATIVES, TREE_ACCESS_RULE};

/// The ten op categories of `scripts/emission_structure.py` (`CATEGORIES`), in its order.
pub const CATEGORIES: [&str; 10] = [
    "vec_fixed", "vec_shiftable", "alu", "load", "store", "flow", "select1", "select2", "const", "const_alu",
];

/// Op counts per category (`CATEGORIES` order).
pub type Counts = [i64; 10];

/// Category of an op as `emission_counts` reads it from the scheduler JSON.
///
/// # Arguments
/// * `op` - The op.
///
/// # Returns
/// The index into `CATEGORIES`, `None` for barriers (not real ops), or an error for an engine
/// the model has no category for.
pub fn category(op: &SsaOp) -> Result<Option<usize>, String> {
    Ok(Some(match op.opcode {
        Opcode::SelectChoice => {
            if op.delta_vreg.is_some() || !op.needs_subtraction {
                6
            } else {
                7
            }
        }
        Opcode::ConstChoice => {
            let alu = op.const_alternatives.as_ref().map_or(false, |alts| {
                alts.iter().any(|a| !matches!(a, ConstEncoding::Load | ConstEncoding::AddImm { .. }))
            });
            if alu {
                9
            } else {
                8
            }
        }
        _ => match op.engine {
            Engine::Valu => {
                if matches!(op.opcode, Opcode::MultiplyAdd | Opcode::Vbroadcast) {
                    0
                } else {
                    1
                }
            }
            Engine::Alu => 2,
            Engine::Load => 3,
            Engine::Store => 4,
            Engine::Flow => 5,
            Engine::Barrier => return Ok(None),
            other => return Err(format!("op {} on engine {:?} has no category", op.id, other)),
        },
    }))
}

/// Counts of a set of ops.
fn counts_of<'a>(ops: impl Iterator<Item = &'a SsaOp>) -> Result<Counts, String> {
    let mut c = [0; 10];
    for op in ops {
        if let Some(k) = category(op)? {
            c[k] += 1;
        }
    }
    Ok(c)
}

/// Encodings in plan order: index into this is the encoding id of an atom.
const ENCODINGS: [AccessEncoding; 4] =
    [AccessEncoding::Gather, AccessEncoding::Bits, AccessEncoding::BitsFlow, AccessEncoding::Mobius];

fn encoding_id(e: AccessEncoding) -> usize {
    ENCODINGS.iter().position(|&x| x == e).expect("plannable encoding")
}

/// One choice: site (round, group) uses encoding `enc` (index into `ENCODINGS`).
type Atom = (u32, u32, usize);

/// When an op is alive, as a function of the plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Liveness {
    /// In no plan.
    Dead,
    /// In every plan.
    Always,
    /// Iff some listed choice is made.
    AnyOf(BTreeSet<Atom>),
}

/// A feature of the structure model that an OR of choices reduces to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Term {
    Table(usize),
    Pairs(usize),
    PairsHigh(usize),
    Mobius(usize),
    GatherDepth(usize),
    GatherRound(usize),
}

impl Term {
    fn name(&self) -> String {
        match self {
            Term::Table(d) => format!("table:{}", d),
            Term::Pairs(d) => format!("pairs:{}", d),
            Term::PairsHigh(d) => format!("pairs_high:{}", d),
            Term::Mobius(d) => format!("mobius:{}", d),
            Term::GatherDepth(d) => format!("gather_depth:{}", d),
            Term::GatherRound(r) => format!("gather_round:{}", r),
        }
    }
}

/// The problem shape the features are defined over (`Context` in the Python model).
#[derive(Clone, Debug)]
pub(super) struct Shape {
    rounds: usize,
    n_groups: usize,
    /// Choice rounds (depth 1 to `MAX_TREE_REDUCE_DEPTH`) by depth.
    rounds_at: BTreeMap<usize, Vec<usize>>,
    depth_of: Vec<usize>,
}

impl Shape {
    pub(super) fn new(spec: &ProblemSpec) -> Self {
        let depth_of: Vec<usize> = (0..spec.rounds).map(|r| spec.round_depth(r)).collect();
        let mut rounds_at: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (r, &d) in depth_of.iter().enumerate() {
            if (1..=MAX_TREE_REDUCE_DEPTH as usize).contains(&d) {
                rounds_at.entry(d).or_default().push(r);
            }
        }
        Shape { rounds: spec.rounds, n_groups: spec.n_groups(), rounds_at, depth_of }
    }

    fn choice_rounds(&self) -> Vec<usize> {
        (0..self.rounds).filter(|&r| self.rounds_at.contains_key(&self.depth_of[r])).collect()
    }

    /// Product features: every combination of two or more `table:d`, `d >= 2`
    /// (`interactions` in the Python model).
    fn interactions(&self) -> Vec<Vec<usize>> {
        let tables: Vec<usize> = self.rounds_at.keys().copied().filter(|&d| d >= 2).collect();
        let mut out = Vec::new();
        for k in 2..=tables.len() {
            combinations(&tables, k, &mut Vec::new(), 0, &mut out);
        }
        out
    }

    /// Feature names in the Python model's order (`feature_names`, without `tail:*`).
    fn feature_names(&self) -> Vec<String> {
        let mut names = vec!["base".to_string()];
        for &d in self.rounds_at.keys() {
            for e in BIT_DRIVEN {
                names.push(format!("groups:{}:{}", d, encoding_name(e)));
            }
            for t in [Term::Table(d), Term::Pairs(d), Term::PairsHigh(d), Term::Mobius(d), Term::GatherDepth(d)] {
                names.push(t.name());
            }
        }
        for r in self.choice_rounds() {
            names.push(Term::GatherRound(r).name());
        }
        for combo in self.interactions() {
            names.push(product_name(&combo));
        }
        names
    }
}

fn combinations(items: &[usize], k: usize, cur: &mut Vec<usize>, start: usize, out: &mut Vec<Vec<usize>>) {
    if cur.len() == k {
        out.push(cur.clone());
        return;
    }
    for i in start..items.len() {
        cur.push(items[i]);
        combinations(items, k, cur, i + 1, out);
        cur.pop();
    }
}

fn product_name(depths: &[usize]) -> String {
    let parts: Vec<String> = depths.iter().map(|d| format!("table:{}", d)).collect();
    format!("and:{}", parts.join("+"))
}

/// Reduce "alive iff any of these choices" to features: returns the OR's terms.
fn terms_of(shape: &Shape, atoms: &BTreeSet<Atom>) -> Result<Vec<Term>, String> {
    // (round, encoding) -> groups.
    let mut by_round: BTreeMap<(usize, usize), BTreeSet<u32>> = BTreeMap::new();
    for &(r, g, e) in atoms {
        by_round.entry((r as usize, e)).or_default().insert(g);
    }
    for (&(r, e), groups) in &by_round {
        if groups.len() != shape.n_groups {
            return Err(format!(
                "used only by groups {:?} of round {} with {} (a per-group resource has no feature)",
                groups, r, encoding_name(ENCODINGS[e])
            ));
        }
    }
    let mut terms = Vec::new();
    for (&d, rounds) in &shape.rounds_at {
        let rounds_with = |e: usize| -> Vec<usize> {
            rounds.iter().copied().filter(|&r| by_round.contains_key(&(r, e))).collect()
        };
        // Gathers: the whole depth or single rounds.
        let gathering = rounds_with(0);
        if !gathering.is_empty() {
            if gathering.len() == rounds.len() {
                terms.push(Term::GatherDepth(d));
            } else {
                terms.extend(gathering.iter().map(|&r| Term::GatherRound(r)));
            }
        }
        // Bit-driven: the set of encodings must use the resource in every round of the depth.
        let mut used = [false; 4];
        for e in 1..4 {
            let with = rounds_with(e);
            if with.is_empty() {
                continue;
            }
            if with.len() != rounds.len() {
                return Err(format!(
                    "used by {} in rounds {:?} only of depth {} (a per-round table has no feature)",
                    encoding_name(ENCODINGS[e]), with, d
                ));
            }
            used[e] = true;
        }
        let term = match (used[1], used[2], used[3]) {
            (false, false, false) => None,
            (false, true, false) => Some(Term::PairsHigh(d)),
            (true, true, false) => Some(Term::Pairs(d)),
            (false, false, true) => Some(Term::Mobius(d)),
            (true, true, true) => Some(Term::Table(d)),
            other => {
                return Err(format!(
                    "used by (bits, bits-flow, mobius) = {:?} at depth {}: no feature is that OR",
                    other, d
                ))
            }
        };
        terms.extend(term);
    }
    Ok(terms)
}

/// Inclusion-exclusion of an OR of terms into signed features.
fn expand_or(terms: &[Term]) -> Result<Vec<(String, i64)>, String> {
    if terms.len() > 8 {
        return Err(format!("an OR of {} features is too wide to expand", terms.len()));
    }
    let mut out = Vec::new();
    for mask in 1u32..(1 << terms.len()) {
        let chosen: Vec<Term> = (0..terms.len()).filter(|i| mask & (1 << i) != 0).map(|i| terms[i]).collect();
        let sign = if chosen.len() % 2 == 1 { 1 } else { -1 };
        let name = if chosen.len() == 1 {
            chosen[0].name()
        } else {
            let depths: Option<Vec<usize>> =
                chosen.iter().map(|t| if let Term::Table(d) = t { Some(*d) } else { None }).collect();
            match depths {
                Some(mut ds) if ds.iter().all(|&d| d >= 2) => {
                    ds.sort_unstable();
                    product_name(&ds)
                }
                _ => {
                    let names: Vec<String> = chosen.iter().map(|t| t.name()).collect();
                    return Err(format!("needs the product {} (no such feature)", names.join(" * ")));
                }
            }
        };
        out.push((name, sign));
    }
    Ok(out)
}

/// One alternative of one site, with its op counts.
#[derive(Clone, Debug)]
pub struct SiteCost {
    /// Round, group and depth of the site.
    pub round: usize,
    /// Group label.
    pub group: usize,
    /// Depth.
    pub depth: usize,
    /// Op counts per alternative (plan order: gather, bits, bits-flow, mobius); `None` if the
    /// alternative was not built.
    pub alternatives: [Option<Counts>; 4],
    /// Facts the match relied on: (name, value, source).
    pub facts: Vec<(String, String, String)>,
}

/// One shared op group: the ops whose liveness reduces to the same OR of features.
#[derive(Clone, Debug)]
pub struct ResourceCost {
    /// The OR's terms (feature names), e.g. `["table:2", "table:3", "table:4"]`.
    pub any_of: Vec<String>,
    /// Ops whose liveness is exactly that OR.
    pub ops: usize,
    /// Their counts.
    pub counts: Counts,
}

/// The tree-access cost table, derived by instantiation.
#[derive(Clone, Debug)]
pub struct DerivedCostTable {
    /// The emission options the table is for.
    pub spec: ProblemSpec,
    /// Feature -> counts (every feature of the model, zeros included).
    pub coefficients: BTreeMap<String, Counts>,
    /// Feature names in the Python model's order.
    pub feature_order: Vec<String>,
    /// Per-site candidate costs.
    pub sites: Vec<SiteCost>,
    /// Shared setup, grouped by liveness.
    pub resources: Vec<ResourceCost>,
    /// Everything that did not fit the feature model (empty = exact).
    pub problems: Vec<String>,
}

impl DerivedCostTable {
    /// Feature values of a plan (`plan_features` in the Python model, without `tail:*`).
    ///
    /// # Arguments
    /// * `plan` - `[round][group]` encodings.
    ///
    /// # Returns
    /// Feature name -> value.
    pub fn plan_features(&self, plan: &[Vec<AccessEncoding>]) -> BTreeMap<String, i64> {
        let shape = Shape::new(&self.spec);
        let mut f: BTreeMap<String, i64> = self.feature_order.iter().map(|n| (n.clone(), 0)).collect();
        f.insert("base".into(), 1);
        for r in shape.choice_rounds() {
            let d = shape.depth_of[r];
            let row = &plan[r];
            for e in BIT_DRIVEN {
                *f.get_mut(&format!("groups:{}:{}", d, encoding_name(e))).unwrap() +=
                    row.iter().filter(|&&x| x == e).count() as i64;
            }
            let uses = |e: AccessEncoding| row.contains(&e);
            let mut set = |t: Term| {
                f.insert(t.name(), 1);
            };
            if BIT_DRIVEN.iter().any(|&e| uses(e)) {
                set(Term::Table(d));
            }
            if uses(AccessEncoding::Bits) || uses(AccessEncoding::BitsFlow) {
                set(Term::Pairs(d));
            }
            if uses(AccessEncoding::BitsFlow) {
                set(Term::PairsHigh(d));
            }
            if uses(AccessEncoding::Mobius) {
                set(Term::Mobius(d));
            }
            if uses(AccessEncoding::Gather) {
                set(Term::GatherDepth(d));
                set(Term::GatherRound(r));
            }
        }
        for combo in shape.interactions() {
            let all = combo.iter().all(|d| f[&Term::Table(*d).name()] == 1);
            f.insert(product_name(&combo), all as i64);
        }
        f
    }

    /// Predicted counts of a plan: the sum of features times coefficients.
    pub fn predict(&self, plan: &[Vec<AccessEncoding>]) -> Counts {
        let f = self.plan_features(plan);
        let mut c = [0; 10];
        for (name, coef) in &self.coefficients {
            for k in 0..10 {
                c[k] += f[name] * coef[k];
            }
        }
        c
    }

    /// JSON: context, coefficients (Python `CostTable` layout), sites, resources, problems.
    pub fn to_json(&self) -> Value {
        let counts_json = |c: &Counts| -> Value {
            Value::Object(CATEGORIES.iter().zip(c).map(|(k, v)| (k.to_string(), json!(v))).collect())
        };
        let coefficients: serde_json::Map<String, Value> =
            self.feature_order.iter().map(|n| (n.clone(), counts_json(&self.coefficients[n]))).collect();
        let sites: Vec<Value> = self
            .sites
            .iter()
            .map(|s| {
                let alts: serde_json::Map<String, Value> = ENCODINGS
                    .iter()
                    .zip(&s.alternatives)
                    .filter_map(|(e, c)| c.as_ref().map(|c| (encoding_name(*e).to_string(), counts_json(c))))
                    .collect();
                json!({
                    "round": s.round, "group": s.group, "depth": s.depth,
                    "alternatives": alts,
                    "facts": s.facts.iter().map(|(n, v, src)| json!({"name": n, "value": v, "source": src})).collect::<Vec<_>>(),
                })
            })
            .collect();
        let resources: Vec<Value> = self
            .resources
            .iter()
            .map(|r| json!({"any_of": r.any_of, "ops": r.ops, "counts": counts_json(&r.counts)}))
            .collect();
        let alternatives: Vec<Value> = TREE_ACCESS_ALTERNATIVES
            .iter()
            .map(|a| json!({"name": a.name, "proofs": a.proofs}))
            .collect();
        json!({
            "rule": TREE_ACCESS_RULE,
            "context": {
                "rounds": self.spec.rounds,
                "batch_size": self.spec.batch_size,
                "forest_height": self.spec.forest_height,
                "merge_hash_stages": self.spec.merge_hash_stages,
                "index_form": match self.spec.index_form { IndexForm::Mirrored => "mirrored", IndexForm::ZeroBased => "zero-based" },
                "select_bit_source": match self.spec.select_bit_source { SelectBitSource::Index => "index", SelectBitSource::Path => "path" },
                "values_only": self.spec.values_only,
            },
            "alternatives": alternatives,
            "coefficients": coefficients,
            "sites": sites,
            "resources": resources,
            "problems": self.problems,
        })
    }
}

/// The canonical plan: the root from the shared vector (bits) at depth 0, gather elsewhere.
pub fn canonical_plan(spec: &ProblemSpec) -> Vec<Vec<AccessEncoding>> {
    (0..spec.rounds)
        .map(|r| {
            let e = if spec.round_depth(r) == 0 { AccessEncoding::Bits } else { AccessEncoding::Gather };
            vec![e; spec.n_groups()]
        })
        .collect()
}

/// Emit the canonical program of `spec` with every alternative of every choice site
/// instantiated next to it.
///
/// # Arguments
/// * `spec` - Emission options (the context); its tree-access options must be unset.
///
/// # Returns
/// The program's ops, its vector groups and the instantiation record.
pub fn instantiate_tree_access(
    spec: &ProblemSpec,
) -> Result<(crate::kernel_builder::SsaProgram, TreeAccessInstantiation), String> {
    if spec.access_plan.is_some() || spec.cached_select != CachedSelect::Standard || spec.depth4_select_after_wrap_only {
        return Err("the cost table is derived from the canonical program: give the context only \
                    (no emission plan, cached_select or depth-4 option)"
            .into());
    }
    let config = KernelConfig {
        spec: ProblemSpec { access_plan: Some(canonical_plan(spec)), ..spec.clone() },
        ..KernelConfig::new(spec.forest_height, spec.n_nodes, spec.batch_size, spec.rounds)
    };
    let mut builder = SsaTickTockKernelBuilder::new_instantiating(config);
    let program = builder.build();
    let inst = builder.instantiation.take().expect("instantiation record");
    Ok((program, inst))
}

/// ConstChoice ops whose category (`const` or `const_alu`) can differ between plans.
///
/// A ConstChoice is `const_alu` iff it can be derived by ALU ops from the constants created
/// before it (`emit_const_choice`): a property of emission order, not of liveness. The level
/// end constants `L_d` of gather bases are created lazily, at the first round that gathers at
/// depth `d`, so whether one exists before a later constant, and where `L_d` itself is
/// created, depend on the plan. For every constant live in some plan this recomputes the
/// category with every subset of the lazy constants before it missing, and for a lazy
/// constant also at each later user (where a plan without the earlier gather creates it).
///
/// # Returns
/// One problem per constant whose category changes (empty: categories are plan-independent).
pub(super) fn plan_dependent_const_categories(
    ops: &[SsaOp],
    live: &[Liveness],
    users: &[Vec<usize>],
    has_zero: bool,
    shape: &Shape,
) -> Vec<String> {
    use crate::types::{compute_const_encodings_alu_ready, RegionKind};
    use std::collections::HashSet;
    let consts: Vec<(usize, i64)> = ops
        .iter()
        .enumerate()
        .filter(|(_, o)| o.opcode == Opcode::ConstChoice)
        .map(|(i, o)| (i, o.produces_constant.expect("ConstChoice carries its value")))
        .collect();
    let is_lazy = |i: usize| {
        ops[i].region.map_or(false, |t| {
            t.kind == RegionKind::TreeAccess(AccessEncoding::Gather)
                && t.group.is_none()
                && t.round.map_or(false, |r| shape.rounds_at.contains_key(&shape.depth_of[r as usize]))
        })
    };
    let lazy: Vec<(usize, i64)> = consts.iter().copied().filter(|&(i, _)| is_lazy(i)).collect();
    let available_before = |pos: usize, missing: &HashSet<i64>| -> HashSet<i64> {
        let mut a: HashSet<i64> = consts
            .iter()
            .filter(|&&(j, v)| j < pos && !missing.contains(&v))
            .map(|&(_, v)| v)
            .collect();
        if has_zero {
            a.insert(0);
        }
        a
    };
    let derivable = |v: i64, a: &HashSet<i64>| !compute_const_encodings_alu_ready(v, a, 2).is_empty();
    let mut problems = Vec::new();
    for &(x, value) in &consts {
        if live[x] == Liveness::Dead {
            continue;
        }
        let as_emitted = derivable(value, &available_before(x, &HashSet::new()));
        // Positions where some plan may create it: here, and for a lazy constant at each user.
        let mut positions = vec![x];
        if is_lazy(x) {
            positions.extend(users[x].iter().copied());
        }
        'pos: for pos in positions {
            let before: Vec<i64> = lazy.iter().filter(|&&(j, _)| j < pos && j != x).map(|&(_, v)| v).collect();
            if before.len() > 6 {
                problems.push(format!("constant {} (op {}): too many lazy constants to check", value, x));
                break;
            }
            for mask in 0u32..(1 << before.len()) {
                let mut missing: HashSet<i64> =
                    (0..before.len()).filter(|k| mask & (1 << k) != 0).map(|k| before[k]).collect();
                missing.insert(value);
                if derivable(value, &available_before(pos, &missing)) != as_emitted {
                    problems.push(format!(
                        "constant {} (op {}) is {} as emitted but {} when created at op {} without {:?}: its \
                         category depends on the plan",
                        value,
                        x,
                        if as_emitted { "const_alu" } else { "const" },
                        if as_emitted { "const" } else { "const_alu" },
                        pos,
                        missing.iter().filter(|&&m| m != value).collect::<BTreeSet<_>>()
                    ));
                    break 'pos;
                }
            }
        }
    }
    problems
}

/// Derive the tree-access cost table of `spec` by instantiation (module docs).
///
/// # Arguments
/// * `spec` - Emission options (the context): shape, hash merge, index form, bit source,
///   values only. Tree-access options must be unset.
///
/// # Returns
/// The table; `problems` lists everything that does not fit the feature model.
pub fn derive_tree_access_costs(spec: &ProblemSpec) -> Result<DerivedCostTable, String> {
    let (program, inst) = instantiate_tree_access(spec)?;
    let shape = Shape::new(spec);
    let mut problems: Vec<String> = Vec::new();
    for (p, why) in &inst.rejections {
        problems.push(format!("round {:?} group {:?}: {}", p.round, p.group, why.0));
    }
    let ops = &program.ops;
    let graph = SsaGraph::new(ops, &program.vector_group_lanes);
    let users = graph.users();

    // Ops of each candidate; atoms.
    let mut cand_of: Vec<Option<usize>> = vec![None; ops.len()];
    for (c, cand) in inst.candidates.iter().enumerate() {
        for &i in &cand.ops {
            if let Some(other) = cand_of[i] {
                return Err(format!("op {} belongs to candidates {} and {}", i, other, c));
            }
            cand_of[i] = Some(c);
        }
    }
    let atom_of = |c: usize| -> Atom {
        let cand = &inst.candidates[c];
        let site = &inst.sites[cand.site];
        (site.round, site.group, encoding_id(cand.encoding))
    };

    // Liveness, last op first (users come after their producers).
    let mut live: Vec<Liveness> = vec![Liveness::Dead; ops.len()];
    for i in (0..ops.len()).rev() {
        if let Some(c) = cand_of[i] {
            for &u in &users[i] {
                let inside = cand_of[u] == Some(c);
                let is_output = ops[i].dest == Some(inst.candidates[c].output);
                if !inside && !is_output {
                    problems.push(format!("op {} of a candidate is read outside it (op {})", i, u));
                }
            }
            live[i] = Liveness::AnyOf(BTreeSet::from([atom_of(c)]));
            continue;
        }
        let op = &ops[i];
        if is_dce_root(op) {
            live[i] = Liveness::Always;
            continue;
        }
        let mut acc = Liveness::Dead;
        for &u in &users[i] {
            assert!(u > i, "op {} is read by the earlier op {}", i, u);
            acc = match (acc, &live[u]) {
                (Liveness::Always, _) | (_, Liveness::Always) => Liveness::Always,
                (a, Liveness::Dead) => a,
                (Liveness::Dead, Liveness::AnyOf(s)) => Liveness::AnyOf(s.clone()),
                (Liveness::AnyOf(mut a), Liveness::AnyOf(s)) => {
                    a.extend(s.iter().copied());
                    Liveness::AnyOf(a)
                }
            };
        }
        live[i] = acc;
    }

    problems.extend(plan_dependent_const_categories(ops, &live, &users, program.zero_vreg.is_some(), &shape));

    // Shared ops: base, or features through their OR.
    let feature_order = shape.feature_names();
    let mut coefficients: BTreeMap<String, Counts> = feature_order.iter().map(|n| (n.clone(), [0; 10])).collect();
    let mut resources: BTreeMap<Vec<String>, (usize, Counts)> = BTreeMap::new();
    for (i, op) in ops.iter().enumerate() {
        if cand_of[i].is_some() {
            continue;
        }
        let Some(k) = category(op)? else { continue };
        match &live[i] {
            Liveness::Dead => {}
            Liveness::Always => coefficients.get_mut("base").unwrap()[k] += 1,
            Liveness::AnyOf(atoms) => {
                let terms = match terms_of(&shape, atoms) {
                    Ok(t) => t,
                    Err(e) => {
                        problems.push(format!("op {} ({:?}): {}", i, op.opcode, e));
                        continue;
                    }
                };
                let names: Vec<String> = terms.iter().map(|t| t.name()).collect();
                let entry = resources.entry(names).or_insert((0, [0; 10]));
                entry.0 += 1;
                entry.1[k] += 1;
                match expand_or(&terms) {
                    Ok(signed) => {
                        for (name, sign) in signed {
                            match coefficients.get_mut(&name) {
                                Some(c) => c[k] += sign,
                                None => problems.push(format!("op {}: feature {} is not in the model", i, name)),
                            }
                        }
                    }
                    Err(e) => problems.push(format!("op {} ({:?}): {}", i, op.opcode, e)),
                }
            }
        }
    }

    // Sites: gather counts go to base, the difference of each bit-driven alternative to
    // groups:d:e (the same at every site of the depth).
    let mut sites: Vec<SiteCost> = inst
        .sites
        .iter()
        .map(|s| SiteCost {
            round: s.round as usize,
            group: s.group as usize,
            depth: s.depth,
            alternatives: [None; 4],
            facts: s.facts.iter().map(|f| (f.name.to_string(), f.value.clone(), f.source.as_str().to_string())).collect(),
        })
        .collect();
    for cand in &inst.candidates {
        sites[cand.site].alternatives[encoding_id(cand.encoding)] = Some(counts_of(cand.ops.iter().map(|&i| &ops[i]))?);
    }
    let mut per_group: BTreeMap<String, (Counts, (usize, usize))> = BTreeMap::new();
    let mut choice_sites = 0;
    for s in &sites {
        if !shape.rounds_at.contains_key(&s.depth) {
            continue;
        }
        choice_sites += 1;
        let Some(gather) = s.alternatives[0] else {
            problems.push(format!("round {} group {}: no gather candidate", s.round, s.group));
            continue;
        };
        for k in 0..10 {
            coefficients.get_mut("base").unwrap()[k] += gather[k];
        }
        for e in 1..4 {
            let Some(alt) = s.alternatives[e] else {
                problems.push(format!("round {} group {}: {} was not built", s.round, s.group, encoding_name(ENCODINGS[e])));
                continue;
            };
            let mut delta = [0; 10];
            for k in 0..10 {
                delta[k] = alt[k] - gather[k];
            }
            let name = format!("groups:{}:{}", s.depth, encoding_name(ENCODINGS[e]));
            match per_group.get(&name) {
                None => {
                    per_group.insert(name, (delta, (s.round, s.group)));
                }
                Some((first, at)) if *first != delta => problems.push(format!(
                    "{}: round {} group {} costs {:?}, round {} group {} costs {:?} (not uniform)",
                    name, at.0, at.1, first, s.round, s.group, delta
                )),
                Some(_) => {}
            }
        }
    }
    let expected_sites = shape.rounds_at.values().map(|r| r.len()).sum::<usize>() * shape.n_groups;
    if choice_sites != expected_sites {
        problems.push(format!("{} choice sites matched, the shape has {}", choice_sites, expected_sites));
    }
    for (name, (delta, _)) in per_group {
        *coefficients.get_mut(&name).expect("groups feature") = delta;
    }
    Ok(DerivedCostTable {
        spec: spec.clone(),
        coefficients,
        feature_order,
        sites,
        resources: resources
            .into_iter()
            .map(|(any_of, (ops, counts))| ResourceCost { any_of, ops, counts })
            .collect(),
        problems,
    })
}
