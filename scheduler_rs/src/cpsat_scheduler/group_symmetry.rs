//! Symmetry between structurally identical vector groups (`--cpsat-symmetry-rows checked`).
//!
//! The kernel processes every vector group (8 batch elements) with the same ops. Two groups
//! `A` and `B` are *identical* when swapping them is an automorphism of the CP-SAT model: the
//! map `pi` that sends the k-th op of `A` (in index order) to the k-th op of `B` and back,
//! and every other op to itself, preserves
//!
//! * each op's engine, opcode, operand kinds, destination width, delta / sub_temp /
//!   keep_live operands (so slots, shift and select encodings and scratch sizes match);
//! * dependency edges and the values read, with values defined in `A` mapped to the
//!   corresponding values of `B` (vector-group lanes and aliases included), values defined
//!   outside both unchanged, and no value of one group read by the other;
//! * directed main-memory ordering edges, including their access kind;
//! * every op outside `A` and `B` that reads ops of the groups (it must read corresponding
//!   ops of both, so the swap leaves it unchanged);
//! * ConstChoice derivations (the producers of the values an ALU encoding combines) and the
//!   ops that use an op of the groups as such a producer;
//! * CSE groups of SelectChoice consumers (both in one shared group that the swap maps to
//!   itself, or in two groups that correspond position by position).
//!
//! Groups are found from the dependency graph, not from emission metadata: an op that reaches
//! more sinks than the median op is shared; groups are the connected components of the rest
//! (the same rule as the greedy wavefront priority).
//!
//! Identical groups form orbits. If the model is symmetric under every swap within an orbit
//! (the module also makes the domains of corresponding ops equal, see
//! [`symmetric_domains`]), then for every feasible schedule there is one of the same makespan
//! whose groups are sorted by the cycle of one fixed anchor op, so the chain
//! `anchor(G_1) <= anchor(G_2) <= ...` excludes no makespan. `verification/group_symmetry.py`
//! checks this on small instances, and shows that two anchors per group (what one
//! emission-declared stream per round amounts to) or an order between different ops of the
//! same group can exclude every optimal schedule.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use super::cse_groups::CseGroup;
use crate::types::{MemEdgeKind, Opcode, Operand, SsaOp, VReg};

/// Structural groups and the orbits of identical ones.
#[derive(Debug, Clone, Default)]
pub(crate) struct GroupOrbits {
    /// Op indices of each group, ascending
    pub(crate) groups: Vec<Vec<usize>>,
    /// Group of each op (None: shared)
    pub(crate) group_of: Vec<Option<usize>>,
    /// Orbits of identical groups (indices into `groups`, ascending); only orbits of 2 or more
    pub(crate) orbits: Vec<Vec<usize>>,
    /// Why groups of equal size were not identical (for diagnostics), at most a few
    pub(crate) rejections: Vec<String>,
}

impl GroupOrbits {
    /// Position of op `idx` in its group.
    pub(crate) fn position(&self, idx: usize) -> Option<(usize, usize)> {
        let g = self.group_of[idx]?;
        self.groups[g].binary_search(&idx).ok().map(|p| (g, p))
    }

    /// Orbit containing group `g`.
    pub(crate) fn orbit_of(&self, g: usize) -> Option<usize> {
        self.orbits.iter().position(|o| o.contains(&g))
    }
}

/// Structural groups: connected components of the ops that reach no more sinks than the
/// median op.
pub(crate) fn find_groups(ops: &[SsaOp]) -> (Vec<Vec<usize>>, Vec<Option<usize>>) {
    let n = ops.len();
    let id_to_idx: HashMap<usize, usize> = ops.iter().enumerate().map(|(i, op)| (op.id, i)).collect();
    let mut consumers: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, op) in ops.iter().enumerate() {
        for dep in &op.dep_ids {
            if let Some(&d) = id_to_idx.get(dep) {
                if d != i {
                    consumers[d].push(i);
                }
            }
        }
    }
    // Topological order (Kahn, smallest index first).
    let mut indeg: Vec<usize> = vec![0; n];
    for cs in &consumers {
        for &c in cs {
            indeg[c] += 1;
        }
    }
    let mut ready: BTreeSet<usize> = (0..n).filter(|&i| indeg[i] == 0).collect();
    let mut order = Vec::with_capacity(n);
    while let Some(i) = ready.pop_first() {
        order.push(i);
        for &c in &consumers[i] {
            indeg[c] -= 1;
            if indeg[c] == 0 {
                ready.insert(c);
            }
        }
    }
    if order.len() != n {
        return (Vec::new(), vec![None; n]);
    }
    let sinks: Vec<usize> = (0..n).filter(|&i| consumers[i].is_empty()).collect();
    let words = sinks.len().div_ceil(64).max(1);
    let mut reach = vec![0u64; n * words];
    for (k, &s) in sinks.iter().enumerate() {
        reach[s * words + k / 64] |= 1 << (k % 64);
    }
    for &i in order.iter().rev() {
        for &c in &consumers[i] {
            for w in 0..words {
                let bits = reach[c * words + w];
                reach[i * words + w] |= bits;
            }
        }
    }
    let reached: Vec<usize> =
        (0..n).map(|i| (0..words).map(|w| reach[i * words + w].count_ones() as usize).sum()).collect();
    let mut counts: Vec<usize> = reached.iter().copied().filter(|&r| r > 0).collect();
    counts.sort_unstable();
    let median = counts.get(counts.len().saturating_sub(1) / 2).copied().unwrap_or(0);
    let private: Vec<bool> = reached.iter().map(|&r| r > 0 && r <= median).collect();
    let mut parent: Vec<usize> = (0..n).collect();
    fn root(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    for i in 0..n {
        if !private[i] {
            continue;
        }
        for &c in &consumers[i] {
            if private[c] {
                let (a, b) = (root(&mut parent, i), root(&mut parent, c));
                parent[a.max(b)] = a.min(b);
            }
        }
    }
    let mut by_root: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for i in 0..n {
        if private[i] {
            let r = root(&mut parent, i);
            by_root.entry(r).or_default().push(i);
        }
    }
    let groups: Vec<Vec<usize>> = by_root.into_values().collect();
    let mut group_of = vec![None; n];
    for (g, members) in groups.iter().enumerate() {
        for &i in members {
            group_of[i] = Some(g);
        }
    }
    (groups, group_of)
}

/// Everything [`check_identical`] needs besides the two groups.
struct Context<'a> {
    ops: &'a [SsaOp],
    id_to_idx: HashMap<usize, usize>,
    vreg_def: HashMap<VReg, usize>,
    group_of: &'a [Option<usize>],
    /// Read-group key (defined vector) -> lanes; write-group lanes (sorted) -> alias
    lanes_of: HashMap<VReg, Vec<VReg>>,
    alias_of_lanes: HashMap<Vec<VReg>, VReg>,
    /// Ops reading each op (indices)
    consumers: Vec<Vec<usize>>,
    /// Targets and kinds of each op's outgoing main-memory ordering edges.
    memory_successors: Vec<Vec<(usize, MemEdgeKind)>>,
    /// First producer of each constant value (ConstChoice derivations use it)
    value_producer: HashMap<i64, usize>,
    /// Ops whose ConstChoice derivations use op `p` as a producer
    derivation_users: HashMap<usize, BTreeSet<usize>>,
    /// CSE group of each SelectChoice consumer, and the consumers of each group
    cse_of: HashMap<usize, usize>,
    cse_groups: &'a [CseGroup],
}

impl<'a> Context<'a> {
    fn new(ops: &'a [SsaOp], group_of: &'a [Option<usize>], lanes: Option<&HashMap<VReg, Vec<VReg>>>, cse_groups: &'a [CseGroup]) -> Self {
        let id_to_idx: HashMap<usize, usize> = ops.iter().enumerate().map(|(i, op)| (op.id, i)).collect();
        let vreg_def: HashMap<VReg, usize> = ops.iter().enumerate().filter_map(|(i, op)| op.dest.map(|d| (d, i))).collect();
        let mut lanes_of = HashMap::new();
        let mut alias_of_lanes = HashMap::new();
        if let Some(groups) = lanes {
            for (key, l) in groups {
                if vreg_def.contains_key(key) {
                    lanes_of.insert(*key, l.clone());
                } else {
                    alias_of_lanes.insert(l.clone(), *key);
                }
            }
        }
        let mut consumers: Vec<Vec<usize>> = vec![Vec::new(); ops.len()];
        let mut memory_successors = vec![Vec::new(); ops.len()];
        for (i, op) in ops.iter().enumerate() {
            for dep in &op.dep_ids {
                if let Some(&d) = id_to_idx.get(dep) {
                    consumers[d].push(i);
                }
            }
            for &(source_id, kind) in &op.mem_deps {
                if let Some(&source) = id_to_idx.get(&source_id) {
                    memory_successors[source].push((i, kind));
                }
            }
        }
        let mut value_producer = HashMap::new();
        for (i, op) in ops.iter().enumerate() {
            if let Some(v) = op.produces_constant {
                value_producer.entry(v).or_insert(i);
            }
        }
        let mut derivation_users: HashMap<usize, BTreeSet<usize>> = HashMap::new();
        for (i, op) in ops.iter().enumerate() {
            for enc in op.const_alternatives.iter().flatten() {
                for v in enc.dependencies() {
                    if let Some(&p) = value_producer.get(&v) {
                        derivation_users.entry(p).or_default().insert(i);
                    }
                }
            }
        }
        let mut cse_of = HashMap::new();
        for (g, group) in cse_groups.iter().enumerate() {
            for &c in &group.consumer_indices {
                cse_of.insert(c, g);
            }
        }
        Context { ops, id_to_idx, vreg_def, group_of, lanes_of, alias_of_lanes, consumers, memory_successors, value_producer, derivation_users, cse_of, cse_groups }
    }
}

/// Whether swapping groups `a` and `b` (the k-th op of one with the k-th of the other) is an
/// automorphism of the model (see the module doc).
///
/// # Returns
/// `Ok(())`, or the first property that differs.
fn check_identical(cx: &Context<'_>, groups: &[Vec<usize>], a: usize, b: usize) -> Result<(), String> {
    let (ga, gb) = (&groups[a], &groups[b]);
    if ga.len() != gb.len() {
        return Err(format!("{} vs {} ops", ga.len(), gb.len()));
    }
    let ops = cx.ops;
    let pos_a: HashMap<usize, usize> = ga.iter().enumerate().map(|(k, &i)| (i, k)).collect();
    let pos_b: HashMap<usize, usize> = gb.iter().enumerate().map(|(k, &i)| (i, k)).collect();
    // Value map A -> B: destinations, sub_temps, then lanes and aliases.
    let mut vmap: HashMap<VReg, VReg> = HashMap::new();
    for (&ia, &ib) in ga.iter().zip(gb) {
        let (oa, ob) = (&ops[ia], &ops[ib]);
        match (oa.dest, ob.dest) {
            (Some(x), Some(y)) if x.width == y.width => {
                vmap.insert(x, y);
            }
            (None, None) => {}
            _ => return Err(format!("op {} vs {}: destination", oa.id, ob.id)),
        }
        match (oa.sub_temp_vreg, ob.sub_temp_vreg) {
            (Some(x), Some(y)) => {
                vmap.insert(x, y);
            }
            (None, None) => {}
            _ => return Err(format!("op {} vs {}: sub_temp", oa.id, ob.id)),
        }
    }
    let defined: Vec<VReg> = vmap.keys().copied().collect();
    for key in defined {
        if let Some(la) = cx.lanes_of.get(&key) {
            let lb = cx.lanes_of.get(&vmap[&key]).ok_or_else(|| format!("lanes of {:?}", key))?;
            if la.len() != lb.len() {
                return Err("lane count".into());
            }
            for (x, y) in la.iter().zip(lb) {
                vmap.insert(*x, *y);
            }
        }
    }
    // A swap must preserve write aliases on both sides, including aliases that no op
    // reads directly: scratch pools still account for their lanes as one vector.
    let mapped_values: HashSet<_> = vmap.values().copied().collect();
    let mut alias_images = HashSet::new();
    let mut mapped_aliases = Vec::new();
    for (lanes, &alias) in &cx.alias_of_lanes {
        if lanes.iter().all(|v| vmap.contains_key(v)) {
            let mapped: Vec<VReg> = lanes.iter().map(|v| vmap[v]).collect();
            let &other = cx.alias_of_lanes.get(&mapped)
                .ok_or_else(|| format!("alias {:?} has no image", alias))?;
            alias_images.insert(other);
            mapped_aliases.push((alias, other));
        }
    }
    for (lanes, &alias) in &cx.alias_of_lanes {
        if lanes.iter().all(|v| mapped_values.contains(v)) && !alias_images.contains(&alias) {
            return Err(format!("alias {:?} has no preimage", alias));
        }
    }
    vmap.extend(mapped_aliases);
    let in_group = |v: &VReg, g: usize| cx.vreg_def.get(v).and_then(|&d| cx.group_of[d]) == Some(g);
    let map_value = |v: VReg| -> Result<VReg, String> {
        if let Some(&w) = vmap.get(&v) {
            return Ok(w);
        }
        if in_group(&v, b) {
            return Err(format!("group reads a value of the other group ({:?})", v));
        }
        Ok(v)
    };
    let map_op = |i: usize| -> Result<usize, String> {
        if let Some(&k) = pos_a.get(&i) {
            return Ok(gb[k]);
        }
        if pos_b.contains_key(&i) {
            return Err(format!("group depends on the other group (op {})", ops[i].id));
        }
        Ok(i)
    };
    for (&ia, &ib) in ga.iter().zip(gb) {
        let (oa, ob) = (&ops[ia], &ops[ib]);
        if oa.engine != ob.engine || oa.opcode != ob.opcode || oa.srcs.len() != ob.srcs.len() {
            return Err(format!("op {} vs {}: engine/opcode/operands", oa.id, ob.id));
        }
        for (sa, sb) in oa.srcs.iter().zip(&ob.srcs) {
            match (sa, sb) {
                (Operand::VReg(x), Operand::VReg(y)) => {
                    if map_value(*x)? != *y {
                        return Err(format!("op {} vs {}: operand {:?} vs {:?}", oa.id, ob.id, x, y));
                    }
                }
                (Operand::VReg(_), _) | (_, Operand::VReg(_)) => {
                    return Err(format!("op {} vs {}: operand kinds", oa.id, ob.id));
                }
                _ => {}
            }
        }
        let same_opt = |x: Option<VReg>, y: Option<VReg>| -> Result<bool, String> {
            Ok(match (x, y) {
                (Some(x), Some(y)) => map_value(x)? == y,
                (None, None) => true,
                _ => false,
            })
        };
        if !same_opt(oa.delta_vreg, ob.delta_vreg)? {
            return Err(format!("op {} vs {}: delta", oa.id, ob.id));
        }
        let kl_a: Vec<VReg> = oa.keep_live.iter().flatten().copied().collect();
        let kl_b: Vec<VReg> = ob.keep_live.iter().flatten().copied().collect();
        if kl_a.len() != kl_b.len() {
            return Err(format!("op {} vs {}: keep_live", oa.id, ob.id));
        }
        for (x, y) in kl_a.iter().zip(&kl_b) {
            if map_value(*x)? != *y {
                return Err(format!("op {} vs {}: keep_live", oa.id, ob.id));
            }
        }
        // Dependencies.
        let deps_a: Result<BTreeSet<usize>, String> = oa
            .dep_ids
            .iter()
            .filter_map(|d| cx.id_to_idx.get(d))
            .map(|&d| map_op(d))
            .collect();
        let deps_b: BTreeSet<usize> = ob.dep_ids.iter().filter_map(|d| cx.id_to_idx.get(d).copied()).collect();
        if deps_a? != deps_b {
            return Err(format!("op {} vs {}: dependencies", oa.id, ob.id));
        }
        // Both directions matter: an incoming edge may come from a shared op, and an
        // outgoing edge to a shared op must still be present when its source is swapped.
        let incoming_memory = |op: &SsaOp| {
            op.mem_deps
                .iter()
                .filter_map(|&(id, kind)| cx.id_to_idx.get(&id).map(|&i| (i, kind)))
                .collect::<Vec<_>>()
        };
        for (a, b) in [
            (&incoming_memory(oa)[..], &incoming_memory(ob)[..]),
            (&cx.memory_successors[ia][..], &cx.memory_successors[ib][..]),
        ] {
            let mapped: Result<BTreeSet<_>, _> = a
                .iter()
                .map(|&(i, kind)| map_op(i).map(|j| (j, kind)))
                .collect();
            if mapped? != b.iter().copied().collect() {
                return Err(format!("op {} vs {}: memory ordering", oa.id, ob.id));
            }
        }
        // Readers outside both groups read corresponding ops of both.
        for (x, y, own, other) in [(ia, ib, &pos_a, &pos_b), (ib, ia, &pos_b, &pos_a)] {
            for &c in &cx.consumers[x] {
                if own.contains_key(&c) {
                    continue;
                }
                if other.contains_key(&c) {
                    return Err("one group reads the other".into());
                }
                if !cx.consumers[y].contains(&c) {
                    return Err(format!("shared op {} reads op {} but not op {}", ops[c].id, ops[x].id, ops[y].id));
                }
            }
        }
        if oa.needs_subtraction != ob.needs_subtraction || oa.produces_constant.is_some() != ob.produces_constant.is_some() {
            return Err(format!("op {} vs {}: select/constant flags", oa.id, ob.id));
        }
        // ConstChoice derivations.
        let encs_a: Vec<_> = oa.const_alternatives.iter().flatten().collect();
        let encs_b: Vec<_> = ob.const_alternatives.iter().flatten().collect();
        if encs_a.len() != encs_b.len() {
            return Err(format!("op {} vs {}: const alternatives", oa.id, ob.id));
        }
        for (ea, eb) in encs_a.iter().zip(&encs_b) {
            let pa: Vec<Option<usize>> = ea.dependencies().iter().map(|v| cx.value_producer.get(v).copied()).collect();
            let pb: Vec<Option<usize>> = eb.dependencies().iter().map(|v| cx.value_producer.get(v).copied()).collect();
            if pa.len() != pb.len() || ea.alu_opcode() != eb.alu_opcode() {
                return Err(format!("op {} vs {}: const encoding", oa.id, ob.id));
            }
            for (x, y) in pa.iter().zip(&pb) {
                let mapped = match x {
                    Some(p) => Some(map_op(*p)?),
                    None => None,
                };
                if mapped != *y {
                    return Err(format!("op {} vs {}: const derivation producer", oa.id, ob.id));
                }
            }
        }
        let users_a: Result<BTreeSet<usize>, String> =
            cx.derivation_users.get(&ia).into_iter().flatten().map(|&u| map_op(u)).collect();
        let users_b: BTreeSet<usize> = cx.derivation_users.get(&ib).cloned().unwrap_or_default();
        if users_a? != users_b {
            return Err(format!("op {} vs {}: derivation users", oa.id, ob.id));
        }
        // CSE groups.
        match (cx.cse_of.get(&ia), cx.cse_of.get(&ib)) {
            (None, None) => {}
            (Some(&x), Some(&y)) => {
                let image: Result<BTreeSet<usize>, String> = cx.cse_groups[x].consumer_indices.iter().map(|&c| {
                    // The swap: A -> B, B -> A, others fixed.
                    if let Some(&k) = pos_b.get(&c) { Ok(ga[k]) } else { map_op(c) }
                }).collect();
                let target: BTreeSet<usize> = cx.cse_groups[y].consumer_indices.iter().copied().collect();
                if image? != target {
                    return Err(format!("op {} vs {}: CSE group", oa.id, ob.id));
                }
            }
            _ => return Err(format!("op {} vs {}: CSE membership", oa.id, ob.id)),
        }
    }
    Ok(())
}

/// Find the groups and the orbits of identical ones.
///
/// # Arguments
/// * `ops` - SSA ops
/// * `vector_group_lanes` - Vector group key/alias -> lanes
/// * `cse_groups` - SelectChoice CSE groups of the model
pub(crate) fn group_orbits(
    ops: &[SsaOp],
    vector_group_lanes: Option<&HashMap<VReg, Vec<VReg>>>,
    cse_groups: &[CseGroup],
) -> GroupOrbits {
    let (groups, group_of) = find_groups(ops);
    let cx = Context::new(ops, &group_of, vector_group_lanes, cse_groups);
    let mut orbits: Vec<Vec<usize>> = Vec::new();
    let mut rejections = Vec::new();
    for g in 0..groups.len() {
        let mut placed = false;
        for orbit in orbits.iter_mut() {
            let rep = orbit[0];
            if groups[rep].len() != groups[g].len() {
                continue;
            }
            match check_identical(&cx, &groups, rep, g) {
                Ok(()) => {
                    orbit.push(g);
                    placed = true;
                    break;
                }
                Err(why) => {
                    if rejections.len() < 5 {
                        rejections.push(format!("group {} vs {}: {}", rep, g, why));
                    }
                }
            }
        }
        if !placed {
            orbits.push(vec![g]);
        }
    }
    orbits.retain(|o| o.len() >= 2);
    GroupOrbits { groups, group_of, orbits, rejections }
}

/// Anchor of each orbit: the first position whose op is neither shiftable nor a SelectChoice
/// (its issue cycle is then the cycle of its only slot).
pub(crate) fn anchor_position(ops: &[SsaOp], orbits: &GroupOrbits, orbit: usize, shiftable: &HashSet<usize>) -> Option<usize> {
    let rep = &orbits.groups[orbits.orbits[orbit][0]];
    rep.iter().position(|&i| !shiftable.contains(&i) && ops[i].opcode != Opcode::SelectChoice)
}

/// Make corresponding ops of an orbit share one domain: each gets the union (smallest lower
/// bound, largest upper bound) over the orbit. The swaps are then automorphisms of the
/// model including its domains, which the ordering rows require.
pub(crate) fn symmetric_domains(ops: &[SsaOp], orbits: &GroupOrbits, bounds: &mut [&mut HashMap<usize, i32>; 2]) {
    let [lo, hi] = bounds;
    for orbit in &orbits.orbits {
        let len = orbits.groups[orbit[0]].len();
        for k in 0..len {
            let ids: Vec<usize> = orbit.iter().map(|&g| ops[orbits.groups[g][k]].id).collect();
            if let Some(min) = ids.iter().filter_map(|id| lo.get(id).copied()).min() {
                for id in &ids {
                    lo.insert(*id, min);
                }
            }
            if let Some(max) = ids.iter().filter_map(|id| hi.get(id).copied()).max() {
                for id in &ids {
                    hi.insert(*id, max);
                }
            }
        }
    }
}

/// Classify declared ordering pairs `(i, j)` (`cycle[i] <= cycle[j]`): justified by a swap of
/// identical groups when `i` and `j` are the same position of two groups of one orbit.
///
/// # Returns
/// (justified, same group, other)
pub(crate) fn classify_pairs(orbits: &GroupOrbits, pairs: &[(usize, usize)]) -> (usize, usize, usize) {
    let (mut justified, mut same_group, mut other) = (0, 0, 0);
    for &(i, j) in pairs {
        match (orbits.position(i), orbits.position(j)) {
            (Some((gi, pi)), Some((gj, pj))) if gi == gj => {
                let _ = (pi, pj);
                same_group += 1;
            }
            (Some((gi, pi)), Some((gj, pj))) if pi == pj && orbits.orbit_of(gi).is_some() && orbits.orbit_of(gi) == orbits.orbit_of(gj) => {
                justified += 1;
            }
            _ => other += 1,
        }
    }
    (justified, same_group, other)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Engine;

    /// Shared constant 0, then per group: load(addr) -> add(load, const) -> store.
    fn program(groups: usize, tweak: impl Fn(usize, &mut Vec<SsaOp>)) -> Vec<SsaOp> {
        let mut ops = Vec::new();
        let mut next = 1u32;
        let shared = VReg::scalar(0);
        let mut c = SsaOp::new(0, Engine::Load, Opcode::Load, Some(shared), vec![]);
        c.produces_constant = Some(7);
        ops.push(c);
        for g in 0..groups {
            let base = ops.len();
            let a = VReg::vector(next);
            let b = VReg::vector(next + 1);
            next += 2;
            let mut load = SsaOp::new(base, Engine::Load, Opcode::Vload, Some(a), vec![Operand::Imm(g as i64)]);
            load.dep_ids = Default::default();
            let mut add = SsaOp::new(base + 1, Engine::Valu, Opcode::Add, Some(b), vec![Operand::VReg(a), Operand::VReg(shared)]);
            add.dep_ids = [base, 0].into_iter().collect();
            let mut store = SsaOp::new(base + 2, Engine::Store, Opcode::Vstore, None, vec![Operand::VReg(b)]);
            store.dep_ids = [base + 1].into_iter().collect();
            ops.extend([load, add, store]);
            tweak(g, &mut ops);
        }
        ops
    }

    #[test]
    fn test_identical_groups_form_one_orbit() {
        let ops = program(3, |_, _| {});
        let o = group_orbits(&ops, None, &[]);
        assert_eq!(o.groups.len(), 3);
        assert_eq!(o.orbits, vec![vec![0, 1, 2]], "{:?}", o.rejections);
    }

    #[test]
    fn test_a_different_opcode_breaks_the_symmetry() {
        let ops = program(3, |g, ops| {
            if g == 1 {
                let n = ops.len();
                ops[n - 2].opcode = Opcode::Xor;
            }
        });
        let o = group_orbits(&ops, None, &[]);
        assert_eq!(o.orbits, vec![vec![0, 2]]);
    }

    /// A shared op that reads one group but not the other is not left unchanged by a swap.
    #[test]
    fn test_a_shared_reader_of_one_group_breaks_the_symmetry() {
        let mut ops = program(2, |_, _| {});
        let mut reader = SsaOp::new(100, Engine::Alu, Opcode::Add, Some(VReg::scalar(99)), vec![]);
        reader.dep_ids = [2].into_iter().collect(); // the add of group 0
        ops.push(reader);
        let groups = [vec![1, 2, 3], vec![4, 5, 6]];
        let group_of = [None, Some(0), Some(0), Some(0), Some(1), Some(1), Some(1), None];
        let cx = Context::new(&ops, &group_of, None, &[]);
        let rejection = check_identical(&cx, &groups, 0, 1).unwrap_err();
        assert_eq!(rejection, "shared op 100 reads op 2 but not op 5");
    }

    #[test]
    fn test_a_write_alias_on_only_one_group_breaks_the_symmetry() {
        let ops: Vec<_> = (0..2 * crate::types::VLEN)
            .map(|i| SsaOp::new(
                i,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(i as u32)),
                vec![],
            ))
            .collect();
        let lanes = crate::types::VLEN;
        let groups = [(0..lanes).collect(), (lanes..2 * lanes).collect()];
        let group_of: Vec<_> = (0..2 * lanes).map(|i| Some(i / lanes)).collect();
        let write_lanes = |start| (start..start + lanes).map(|i| VReg::scalar(i as u32)).collect();
        let mut aliases = HashMap::from([(VReg::vector(100), write_lanes(lanes))]);
        let cx = Context::new(&ops, &group_of, Some(&aliases), &[]);
        assert!(check_identical(&cx, &groups, 0, 1).unwrap_err().contains("has no preimage"));
        assert!(check_identical(&cx, &groups, 1, 0).unwrap_err().contains("has no image"));

        aliases.insert(VReg::vector(101), write_lanes(0));
        let cx = Context::new(&ops, &group_of, Some(&aliases), &[]);
        assert_eq!(check_identical(&cx, &groups, 0, 1), Ok(()));
    }

    #[test]
    fn test_matching_memory_edges_keep_the_symmetry() {
        let ops = program(3, |_, ops| {
            let n = ops.len();
            ops[n - 1].mem_deps.push((n - 3, MemEdgeKind::War));
        });
        let o = group_orbits(&ops, None, &[]);
        assert_eq!(o.orbits, vec![vec![0, 1, 2]], "{:?}", o.rejections);
    }

    #[test]
    fn test_a_memory_edge_between_groups_breaks_the_symmetry() {
        let mut ops = program(3, |_, _| {});
        ops[6].mem_deps.push((3, MemEdgeKind::Waw));
        let o = group_orbits(&ops, None, &[]);
        assert!(o.orbits.is_empty(), "orbits {:?}", o.orbits);
    }

    #[test]
    fn test_memory_edge_kind_affects_symmetry() {
        let ops = program(3, |group, ops| {
            let n = ops.len();
            let kind = if group == 1 { MemEdgeKind::Raw } else { MemEdgeKind::War };
            ops[n - 1].mem_deps.push((n - 3, kind));
        });
        let o = group_orbits(&ops, None, &[]);
        assert_eq!(o.orbits, vec![vec![0, 2]], "{:?}", o.rejections);
    }

    #[test]
    fn test_classify_pairs() {
        let ops = program(3, |_, _| {});
        let o = group_orbits(&ops, None, &[]);
        // same position of two groups; two ops of one group; a shared op
        let pairs = [(2, 5), (1, 2), (0, 2)];
        assert_eq!(classify_pairs(&o, &pairs), (1, 1, 1));
    }
}
