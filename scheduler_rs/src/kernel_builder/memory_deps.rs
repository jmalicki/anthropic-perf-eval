//! Main-memory dependencies of an SSA program.
//!
//! `compute_ssa_dependencies` orders ops by VReg data flow only (RAW on scratch values).
//! Main memory is the ISA's only data-dependent addressing, and nothing else orders two
//! accesses to it. This module adds exactly the orderings memory needs:
//!
//! 1. Every memory op carries a [`MemLocation`] set by the emitter (region, offset form,
//!    width). A memory op without one is an error: edges are exact, never conservative.
//! 2. Two accesses in program (emission) order conflict when at least one writes and
//!    [`MemLocation::may_alias`] holds. The pair becomes a candidate edge of kind
//!    [`MemEdgeKind::Raw`] (store, then load), [`MemEdgeKind::War`] (load, then store) or
//!    [`MemEdgeKind::Waw`] (store, then store), with latency [`MemEdgeKind::latency`]
//!    (1, 0 and 1: within a cycle the simulator reads before it writes).
//! 3. A candidate is dropped when the `dep_ids` graph already implies it: some path from the
//!    source to the target whose total latency, with the precedence offsets the schedulers use
//!    ([`PrecedenceOffset::precedence_offset_for_asap`], the minimum over the shift flag), is
//!    at least the edge's latency. The rest are kept in [`SsaOp::mem_deps`].
//!
//! Only VReg data edges count as paths. Barrier edges are excluded: they are dropped before
//! CP-SAT (`elide_barriers`), so an edge implied only through a barrier would be lost there.
//! A store defines no VReg, so no data path starts at a store: RAW and WAW candidates are
//! never implied, and only WAR candidates can be dropped. Every kernel emitted today stores
//! only to indices, values (both after the last read of the same words, through data flow)
//! and, with the Rust-only partition stores, the extra room, so `mem_deps` is empty.

use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};

use crate::types::{MemEdgeKind, MemLocation, Opcode, PrecedenceOffset, SsaOp};

/// Counts from one [`compute_memory_dependencies`] run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MemDepStats {
    /// Memory ops (loads and stores) in the program
    pub memory_ops: usize,
    /// Conflicting pairs found (candidate edges)
    pub computed: usize,
    /// Candidates dropped because a `dep_ids` path already implies them
    pub implied: usize,
    /// Candidates kept in `mem_deps`
    pub kept: usize,
}

impl MemDepStats {
    /// One line for verbose output.
    pub fn summary(&self) -> String {
        format!(
            "memory edges: {} computed, {} implied by data dependencies, {} kept ({} memory ops)",
            self.computed, self.implied, self.kept, self.memory_ops
        )
    }
}

/// Minimum latency the schedulers give a `dep_ids` edge out of `op`: the ASAP precedence
/// offset, taken as the minimum over both values of the shift flag so the answer does not
/// depend on which ops end up shiftable.
///
/// Out of a vbroadcast it is 0: when the vbroadcast is eliminated its shifted readers read
/// the scalar producer directly (the makespan rows use the same rule). Only WAR edges
/// (latency 0) can be implied, so this changes no decision; it keeps the filter sound if
/// that ever changes.
fn min_data_edge_latency(op: &SsaOp) -> i32 {
    if op.opcode == Opcode::Vbroadcast {
        return 0;
    }
    op.precedence_offset_for_asap(false).min(op.precedence_offset_for_asap(true))
}

/// Longest `dep_ids` path latency from each ancestor of `target` to `target`, capped at `cap`.
///
/// Ops are in program order and every data dependency points to an earlier op, so visiting
/// ops from the highest index down finalises each op after all of its consumers.
///
/// # Arguments
/// * `ops` - SSA ops with `dep_ids` set (data edges only)
/// * `id_to_idx` - Op id to index
/// * `target` - Index of the target op
/// * `floor` - Ancestors with a smaller index are not explored (no candidate source there)
/// * `cap` - Largest latency of interest
///
/// # Returns
/// Map from ancestor index to its capped longest-path latency (the target maps to 0), or an
/// error if a dependency does not point to an earlier op.
fn capped_path_latency_to(
    ops: &[SsaOp],
    id_to_idx: &HashMap<usize, usize>,
    target: usize,
    floor: usize,
    cap: i32,
) -> Result<HashMap<usize, i32>, String> {
    let mut best: HashMap<usize, i32> = HashMap::new();
    best.insert(target, 0);
    let mut queued: HashSet<usize> = HashSet::new();
    queued.insert(target);
    let mut heap: BinaryHeap<usize> = BinaryHeap::new();
    heap.push(target);
    while let Some(v) = heap.pop() {
        let lv = best[&v];
        for dep_id in &ops[v].dep_ids {
            let Some(&d) = id_to_idx.get(dep_id) else {
                continue;
            };
            if d >= v {
                return Err(format!(
                    "op id={} depends on op id={}, which is not earlier in program order",
                    ops[v].id, ops[d].id
                ));
            }
            if d < floor {
                continue;
            }
            let through = (lv + min_data_edge_latency(&ops[d])).min(cap);
            let entry = best.entry(d).or_insert(i32::MIN);
            if through > *entry {
                *entry = through;
            }
            if queued.insert(d) {
                heap.push(d);
            }
        }
    }
    Ok(best)
}

/// Compute exact main-memory edges and keep those the data dependencies do not imply.
///
/// Clears and recomputes `mem_deps` on every op. Must run after `compute_ssa_dependencies`
/// and before barrier dependencies are added (only data paths may imply an edge).
///
/// # Arguments
/// * `ops` - SSA ops in program order, with data `dep_ids` set
///
/// # Returns
/// The counts, or an error naming a memory op without a location (or a non-memory op with
/// one, or a dependency that does not point backwards).
pub fn compute_memory_dependencies(ops: &mut [SsaOp]) -> Result<MemDepStats, String> {
    for op in ops.iter_mut() {
        op.mem_deps.clear();
    }

    // Memory ops in program order: (index, location, writes).
    let mut mem_ops: Vec<(usize, MemLocation, bool)> = Vec::new();
    for (idx, op) in ops.iter().enumerate() {
        if op.is_memory_op() {
            let loc = op.mem_loc.ok_or_else(|| {
                format!(
                    "memory op id={} ({}) has no memory location; the emitter must set one on \
                     every load and store (source: {:?})",
                    op.id,
                    op.opcode.as_str(),
                    op.source
                )
            })?;
            mem_ops.push((idx, loc, op.writes_memory()));
        } else if op.mem_loc.is_some() {
            return Err(format!(
                "op id={} ({}) does not touch memory but has memory location {:?}",
                op.id,
                op.opcode.as_str(),
                op.mem_loc
            ));
        }
    }
    let mut stats = MemDepStats { memory_ops: mem_ops.len(), ..Default::default() };

    // Candidate edges by target index; sources in increasing index order.
    let mut candidates: BTreeMap<usize, Vec<(usize, MemEdgeKind)>> = BTreeMap::new();
    for (j, &(tgt, tgt_loc, tgt_writes)) in mem_ops.iter().enumerate() {
        for &(src, src_loc, src_writes) in &mem_ops[..j] {
            let kind = match (src_writes, tgt_writes) {
                (false, false) => continue,
                (true, false) => MemEdgeKind::Raw,
                (false, true) => MemEdgeKind::War,
                (true, true) => MemEdgeKind::Waw,
            };
            if !src_loc.may_alias(&tgt_loc) {
                continue;
            }
            candidates.entry(tgt).or_default().push((src, kind));
        }
    }

    let id_to_idx: HashMap<usize, usize> = ops.iter().enumerate().map(|(i, op)| (op.id, i)).collect();
    let mut kept: Vec<(usize, Vec<(usize, MemEdgeKind)>)> = Vec::new();
    for (&tgt, edges) in &candidates {
        stats.computed += edges.len();
        let cap = edges.iter().map(|(_, k)| k.latency()).max().unwrap_or(0);
        let floor = edges.iter().map(|&(s, _)| s).min().unwrap_or(tgt);
        let reach = capped_path_latency_to(ops, &id_to_idx, tgt, floor, cap)?;
        let mut keep: Vec<(usize, MemEdgeKind)> = Vec::new();
        for &(src, kind) in edges {
            let implied = reach.get(&src).map_or(false, |&l| l >= kind.latency());
            if implied {
                stats.implied += 1;
            } else {
                stats.kept += 1;
                keep.push((ops[src].id, kind));
            }
        }
        if !keep.is_empty() {
            keep.sort_unstable();
            kept.push((tgt, keep));
        }
    }
    for (tgt, edges) in kept {
        ops[tgt].mem_deps = edges;
    }
    Ok(stats)
}

/// Add round-barrier ordering to `dep_ids` (see `compute_ssa_dependencies_with_barriers`).
///
/// Barriers create hard scheduling boundaries between rounds:
/// - Each barrier depends on all ops in its preceding section (since last barrier)
/// - Each barrier also depends on the previous barrier (to chain them)
/// - All ops after a barrier depend on that barrier
pub fn add_barrier_dependencies(ops: &mut [SsaOp]) {
    let barrier_indices: Vec<usize> = ops
        .iter()
        .enumerate()
        .filter(|(_, op)| op.opcode == Opcode::Barrier)
        .map(|(idx, _)| idx)
        .collect();

    if barrier_indices.is_empty() {
        return;
    }

    let mut prev_barrier_idx: Option<usize> = None;

    for &barrier_idx in &barrier_indices {
        let barrier_op_id = ops[barrier_idx].id;

        // Find start of this section (after previous barrier, or start of ops)
        let section_start = prev_barrier_idx.map_or(0, |idx| idx + 1);

        // Barrier depends on all non-barrier ops in its section
        for idx in section_start..barrier_idx {
            if ops[idx].opcode != Opcode::Barrier {
                let id = ops[idx].id;
                ops[barrier_idx].dep_ids.insert(id);
            }
        }

        // Barrier also depends on previous barrier (to chain them)
        if let Some(prev_idx) = prev_barrier_idx {
            let id = ops[prev_idx].id;
            ops[barrier_idx].dep_ids.insert(id);
        }

        // All ops after this barrier (until next barrier or end) depend on it
        let next_barrier_idx = barrier_indices
            .iter()
            .find(|&&i| i > barrier_idx)
            .copied()
            .unwrap_or(ops.len());

        for idx in (barrier_idx + 1)..next_barrier_idx {
            if ops[idx].opcode != Opcode::Barrier {
                ops[idx].dep_ids.insert(barrier_op_id);
            }
        }

        prev_barrier_idx = Some(barrier_idx);
    }
}

/// Memory ordering predecessors of each op, as `(index, latency)` lists indexed like `ops`.
/// Sources not in `ops` are skipped (like `dep_ids` entries without an op).
pub fn memory_predecessors(ops: &[SsaOp], id_to_idx: &HashMap<usize, usize>) -> Vec<Vec<(usize, i32)>> {
    ops.iter()
        .map(|op| {
            op.mem_deps
                .iter()
                .filter_map(|&(src_id, kind)| id_to_idx.get(&src_id).map(|&s| (s, kind.latency())))
                .collect()
        })
        .collect()
}

/// Memory ordering successors of each op, as `(index, latency)` lists indexed like `ops`, in
/// increasing target index.
pub fn memory_successors(ops: &[SsaOp], id_to_idx: &HashMap<usize, usize>) -> Vec<Vec<(usize, i32)>> {
    let mut succ: Vec<Vec<(usize, i32)>> = vec![Vec::new(); ops.len()];
    for (tgt, op) in ops.iter().enumerate() {
        for &(src_id, kind) in &op.mem_deps {
            if let Some(&src) = id_to_idx.get(&src_id) {
                succ[src].push((tgt, kind.latency()));
            }
        }
    }
    succ
}

/// Check that a schedule satisfies every memory edge:
/// `cycle[target] >= cycle[source] + latency`. Ops with a negative cycle (skipped dead ops)
/// and sources not in `ops` impose nothing.
///
/// # Arguments
/// * `ops` - SSA ops with `mem_deps`
/// * `cycle_of` - Cycle of the op at an index
///
/// # Returns
/// The first violated edge as an error message.
pub fn check_memory_edges(ops: &[SsaOp], cycle_of: impl Fn(usize) -> i32) -> Result<(), String> {
    if ops.iter().all(|op| op.mem_deps.is_empty()) {
        return Ok(());
    }
    let id_to_idx: HashMap<usize, usize> = ops.iter().enumerate().map(|(i, op)| (op.id, i)).collect();
    for (tgt, op) in ops.iter().enumerate() {
        let tgt_cycle = cycle_of(tgt);
        for &(src_id, kind) in &op.mem_deps {
            let Some(&src) = id_to_idx.get(&src_id) else {
                continue;
            };
            let src_cycle = cycle_of(src);
            if src_cycle < 0 || tgt_cycle < 0 {
                continue;
            }
            if tgt_cycle < src_cycle + kind.latency() {
                return Err(format!(
                    "memory {} edge violated: op id={} at cycle {} must be at least {} cycle(s) after op id={} at cycle {}",
                    kind.as_str(),
                    op.id,
                    tgt_cycle,
                    kind.latency(),
                    src_id,
                    src_cycle
                ));
            }
        }
    }
    Ok(())
}

// ============================================================================
// Test programs (shared with the greedy and CP-SAT tests)
// ============================================================================

#[cfg(test)]
pub(crate) mod test_programs {
    use crate::kernel_builder::emit_ssa::{compute_ssa_dependencies, SsaEmitContext};
    use crate::types::{MemLocation, MemRegion, Opcode, SsaOp};

    /// Op indices of [`store_load_same_word`].
    #[derive(Debug, Clone, Copy)]
    pub(crate) struct StoreLoadSameWord {
        /// Load of extra-room word 0 (its result feeds `first_store`)
        pub load_before: usize,
        /// Store of `load_before + load_before` to extra-room word 1
        pub first_store: usize,
        /// Store of a constant to extra-room word 0: WAR after `load_before`
        pub store_word0: usize,
        /// Load of extra-room word 0: RAW after `store_word0`
        pub load_after: usize,
        /// Store to extra-room word 1 of a value derived from `load_after`: WAW after
        /// `first_store`
        pub last_store: usize,
    }

    /// A program that stores to a word it has read and then reads the word again, the shape
    /// a scatter store followed by a load produces:
    ///
    /// ```text
    /// p, w, q = const 100, 7, 101                (ops 0, 1, 2)
    /// x  = load  er[0]          load_before      (3)
    /// y  = x + x                                 (4)
    ///      store er[1] = y      first_store      (5)
    ///      store er[0] = w      store_word0      (6)  WAR after load_before, latency 0
    /// z  = load  er[0]          load_after       (7)  RAW after store_word0, latency 1
    /// z2 = (z + z) + (z + z)                     (8, 9)
    ///      store er[1] = z2     last_store       (10) WAW after first_store, latency 1
    /// ```
    ///
    /// No data path implies any of the three edges. The load_after chain is the longest, so
    /// the shortest schedule (6 cycles) puts store_word0 in the cycle of load_before.
    ///
    /// # Returns
    /// Ops with data `dep_ids` (memory edges not computed yet) and the op indices.
    pub(crate) fn store_load_same_word() -> (Vec<SsaOp>, StoreLoadSameWord) {
        let mut emit = SsaEmitContext::new();
        let p = emit.emit_const_choice(100);
        let w = emit.emit_const_choice(7);
        let q = emit.emit_const_choice(101);
        let er = |word| MemLocation::word(MemRegion::ExtraRoom, word);

        emit.set_next_mem_loc(er(0));
        let x = emit.emit_load(p);
        let load_before = emit.ops.len() - 1;
        let y = emit.emit_alu(Opcode::Add, x, x);
        emit.set_next_mem_loc(er(1));
        emit.emit_store(q, y);
        let first_store = emit.ops.len() - 1;
        emit.set_next_mem_loc(er(0));
        emit.emit_store(p, w);
        let store_word0 = emit.ops.len() - 1;
        emit.set_next_mem_loc(er(0));
        let z = emit.emit_load(p);
        let load_after = emit.ops.len() - 1;
        let z1 = emit.emit_alu(Opcode::Add, z, z);
        let z2 = emit.emit_alu(Opcode::Add, z1, z1);
        emit.set_next_mem_loc(er(1));
        emit.emit_store(q, z2);
        let last_store = emit.ops.len() - 1;
        assert!(emit.next_mem_loc.is_none());

        let vreg_def = std::mem::take(&mut emit.vreg_def);
        compute_ssa_dependencies(&mut emit.ops, &vreg_def, emit.zero_vreg, None);
        (emit.ops, StoreLoadSameWord { load_before, first_store, store_word0, load_after, last_store })
    }
}

#[cfg(test)]
mod tests {
    use super::test_programs::store_load_same_word;
    use super::*;
    use crate::kernel_builder::emit_ssa::{compute_ssa_dependencies, SsaEmitContext};
    use crate::types::{MemOffset, MemRegion, VLEN};

    /// Alias decision: same region and overlapping word ranges; a data-dependent offset covers
    /// its whole range; different regions never alias.
    #[test]
    fn test_may_alias() {
        let vec_at = |region, word| MemLocation::block(region, word, VLEN as u32);
        assert!(vec_at(MemRegion::Values, 8).may_alias(&MemLocation::word(MemRegion::Values, 15)));
        assert!(!vec_at(MemRegion::Values, 8).may_alias(&MemLocation::word(MemRegion::Values, 16)));
        assert!(!vec_at(MemRegion::Values, 8).may_alias(&vec_at(MemRegion::Indices, 8)));
        let gather = MemLocation::word_in(MemRegion::Tree, 0, 2047);
        assert_eq!(gather.offset, MemOffset::Range { lo: 0, hi: 2047 });
        assert_eq!(gather.word_range(), (0, 2047));
        assert!(gather.may_alias(&MemLocation::word(MemRegion::Tree, 2046)));
        assert!(!gather.may_alias(&MemLocation::word(MemRegion::Tree, 2047)));
        assert!(!gather.may_alias(&MemLocation::word(MemRegion::Header, 4)));
    }

    /// Scatter stores, then a vload of the words they wrote, then a vstore over them: the
    /// vload gets a RAW edge from every store (latency 1), the vstore a WAW edge from every
    /// store (latency 1) and a WAR edge from the vload (latency 0). A load from another
    /// region gets nothing. All are kept: no data path starts at a store, and the vstore's
    /// value does not come from the vload.
    #[test]
    fn test_scatter_then_vload_edges() {
        let mut emit = SsaEmitContext::new();
        let base = emit.emit_const_choice(500);
        let val = emit.emit_const_choice(3);
        let mut stores = Vec::new();
        for k in 0..VLEN as i64 {
            emit.set_next_mem_loc(MemLocation::word(MemRegion::ExtraRoom, k));
            emit.emit_store(base, val);
            stores.push(emit.ops.len() - 1);
        }
        emit.set_next_mem_loc(MemLocation::block(MemRegion::ExtraRoom, 0, VLEN as u32));
        let _gathered = emit.emit_vload(base);
        let vload = emit.ops.len() - 1;
        emit.set_next_mem_loc(MemLocation::word(MemRegion::Tree, 0));
        let _tree = emit.emit_load(base);
        let tree_load = emit.ops.len() - 1;
        let other = emit.emit_vbroadcast(val);
        emit.set_next_mem_loc(MemLocation::block(MemRegion::ExtraRoom, 0, VLEN as u32));
        emit.emit_vstore(base, other);
        let vstore = emit.ops.len() - 1;
        let vreg_def = std::mem::take(&mut emit.vreg_def);
        compute_ssa_dependencies(&mut emit.ops, &vreg_def, emit.zero_vreg, None);

        let stats = compute_memory_dependencies(&mut emit.ops).unwrap();
        let ops = &emit.ops;
        let store_ids: Vec<usize> = stores.iter().map(|&i| ops[i].id).collect();

        let expected_vload: Vec<(usize, MemEdgeKind)> =
            store_ids.iter().map(|&id| (id, MemEdgeKind::Raw)).collect();
        assert_eq!(ops[vload].mem_deps, expected_vload);
        assert!(ops[tree_load].mem_deps.is_empty(), "the tree is another region");
        let mut expected_vstore: Vec<(usize, MemEdgeKind)> =
            store_ids.iter().map(|&id| (id, MemEdgeKind::Waw)).collect();
        expected_vstore.push((ops[vload].id, MemEdgeKind::War));
        expected_vstore.sort_unstable();
        assert_eq!(ops[vstore].mem_deps, expected_vstore);
        for &s in &stores {
            assert!(ops[s].mem_deps.is_empty(), "the scatter stores write distinct words");
        }
        assert_eq!(stats, MemDepStats { memory_ops: 11, computed: 17, implied: 0, kept: 17 });
        assert_eq!(
            (MemEdgeKind::Raw.latency(), MemEdgeKind::War.latency(), MemEdgeKind::Waw.latency()),
            (1, 0, 1)
        );
    }

    /// The redundancy filter drops a WAR edge whose load feeds the store through data flow
    /// (as for the final value stores of every kernel today) and keeps one whose store does
    /// not depend on the load.
    #[test]
    fn test_redundancy_filter_drops_implied_keeps_others() {
        let mut emit = SsaEmitContext::new();
        let addr = emit.emit_const_choice(40);
        let block = |word| MemLocation::block(MemRegion::Values, word, VLEN as u32);
        // Implied: vload values[0..8] -> xor -> vstore values[0..8]
        emit.set_next_mem_loc(block(0));
        let loaded = emit.emit_vload(addr);
        let changed = emit.emit_valu(Opcode::Xor, loaded, loaded);
        emit.set_next_mem_loc(block(0));
        emit.emit_vstore(addr, changed);
        let implied_store = emit.ops.len() - 1;
        // Not implied: vload values[8..16], then vstore values[8..16] of an unrelated vector
        emit.set_next_mem_loc(block(8));
        let other_loaded = emit.emit_vload(addr);
        let kept_load = emit.ops.len() - 1;
        let unrelated = emit.emit_vbroadcast(addr);
        emit.set_next_mem_loc(block(8));
        emit.emit_vstore(addr, unrelated);
        let kept_store = emit.ops.len() - 1;
        // Use the second load's result elsewhere so it is not dead.
        emit.set_next_mem_loc(block(16));
        emit.emit_vstore(addr, other_loaded);
        let vreg_def = std::mem::take(&mut emit.vreg_def);
        compute_ssa_dependencies(&mut emit.ops, &vreg_def, emit.zero_vreg, None);

        let stats = compute_memory_dependencies(&mut emit.ops).unwrap();
        assert!(emit.ops[implied_store].mem_deps.is_empty(), "the xor path implies the WAR edge");
        assert_eq!(emit.ops[kept_store].mem_deps, vec![(emit.ops[kept_load].id, MemEdgeKind::War)]);
        assert_eq!((stats.computed, stats.implied, stats.kept), (2, 1, 1));
    }

    /// A memory op without a location is an error, never a conservative edge.
    #[test]
    fn test_unknown_location_is_error() {
        let mut emit = SsaEmitContext::new();
        let addr = emit.emit_const_choice(4);
        let _x = emit.emit_load(addr);
        let vreg_def = std::mem::take(&mut emit.vreg_def);
        compute_ssa_dependencies(&mut emit.ops, &vreg_def, emit.zero_vreg, None);
        let err = compute_memory_dependencies(&mut emit.ops).unwrap_err();
        assert!(err.contains("no memory location"), "{}", err);
    }

    /// The edges of the store/load program, with kinds and latencies.
    #[test]
    fn test_store_load_same_word_edges() {
        let (mut ops, ix) = store_load_same_word();
        let stats = compute_memory_dependencies(&mut ops).unwrap();
        assert_eq!(ops[ix.store_word0].mem_deps, vec![(ops[ix.load_before].id, MemEdgeKind::War)]);
        assert_eq!(ops[ix.load_after].mem_deps, vec![(ops[ix.store_word0].id, MemEdgeKind::Raw)]);
        assert_eq!(ops[ix.last_store].mem_deps, vec![(ops[ix.first_store].id, MemEdgeKind::Waw)]);
        assert_eq!((stats.computed, stats.implied, stats.kept), (3, 0, 3));
    }

    /// ASAP, ALAP and the max-latency critical path follow memory edges with their own
    /// latency: 0 for WAR, 1 for RAW and WAW.
    #[test]
    fn test_bounds_follow_memory_edges() {
        use crate::cpsat_scheduler::{compute_max_latency_critical_path, compute_ssa_alap, compute_ssa_asap};
        let (mut ops, ix) = store_load_same_word();
        compute_memory_dependencies(&mut ops).unwrap();
        let none = HashSet::new();
        let id = |i: usize| ops[i].id;
        let asap = compute_ssa_asap(&ops, &none);
        // p at 0, load_before at 1, store_word0 also at 1 (WAR 0), load_after at 2 (RAW 1).
        assert_eq!(asap[&id(ix.load_before)], 1);
        assert_eq!(asap[&id(ix.store_word0)], 1);
        assert_eq!(asap[&id(ix.load_after)], 2);
        assert_eq!(asap[&id(ix.last_store)], 5);
        let alap = compute_ssa_alap(&ops, 6, &none);
        assert_eq!(alap[&id(ix.load_after)], 2);
        assert_eq!(alap[&id(ix.store_word0)], 1);
        assert_eq!(alap[&id(ix.load_before)], 1, "WAR: load_before no later than store_word0");
        assert_eq!(alap[&id(ix.first_store)], 4, "WAW: first_store before last_store");
        assert_eq!(compute_max_latency_critical_path(&ops, &none), 6);
    }

    /// The static heads and tails behind the makespan rows follow memory edges too.
    #[test]
    fn test_static_paths_follow_memory_edges() {
        use crate::cpsat_scheduler::makespan_rows::static_paths;
        let (mut ops, ix) = store_load_same_word();
        compute_memory_dependencies(&mut ops).unwrap();
        let paths = static_paths(&ops, &HashSet::new());
        assert_eq!(paths.head[ix.store_word0], 1, "WAR: same cycle as load_before");
        assert_eq!(paths.head[ix.load_after], 2, "RAW: one after store_word0");
        assert_eq!(paths.head[ix.last_store], 5);
        assert_eq!(paths.tail[ix.store_word0], 4, "store_word0 -> load_after -> z1 -> z2 -> last_store");
        assert_eq!(paths.tail[ix.load_before], 4, "load_before -> store_word0 (0) -> ...");
    }

    /// Memory edges carry no scratch value: live intervals are the same with and without them.
    #[test]
    fn test_memory_edges_do_not_change_liveness() {
        let (mut ops, _) = store_load_same_word();
        // Any cycles will do: use the program order.
        for (i, op) in ops.iter_mut().enumerate() {
            op.cycle = i as i32;
        }
        let without = ops.clone();
        compute_memory_dependencies(&mut ops).unwrap();
        assert!(ops.iter().any(|op| !op.mem_deps.is_empty()));

        let a = crate::kernel_builder::emit_ssa::compute_live_intervals(&ops);
        let b = crate::kernel_builder::emit_ssa::compute_live_intervals(&without);
        assert_eq!(a.len(), b.len());
        for (vreg, iv) in &a {
            assert_eq!((iv.start, iv.end), (b[vreg].start, b[vreg].end), "{:?}", vreg);
        }

        let config = crate::address_assign::AddressAssignConfig::default();
        let a = crate::address_assign::compute_live_intervals(&config, &ops).unwrap();
        let b = crate::address_assign::compute_live_intervals(&config, &without).unwrap();
        assert_eq!(a.len(), b.len());
        for (vreg, iv) in &a {
            assert_eq!((iv.start, iv.end), (b[vreg].start, b[vreg].end), "{:?}", vreg);
        }
    }

    /// The schedule checker and the bulk-move constraints see memory edges, a same-cycle WAR
    /// included.
    #[test]
    fn test_check_memory_edges_and_schedule_constraints() {
        let (mut ops, ix) = store_load_same_word();
        compute_memory_dependencies(&mut ops).unwrap();
        let mut cycles: Vec<i32> = vec![0, 0, 0, 1, 2, 3, 1, 2, 3, 4, 5];
        assert_eq!(cycles.len(), ops.len());
        assert_eq!(cycles[ix.store_word0], cycles[ix.load_before], "same-cycle WAR");
        check_memory_edges(&ops, |i| cycles[i]).unwrap();

        let id_to_idx: HashMap<usize, usize> = ops.iter().enumerate().map(|(i, o)| (o.id, i)).collect();
        let constraints = crate::schedule_constraints::build_ssa_schedule_constraints(
            &ops,
            &id_to_idx,
            &HashMap::new(),
            &HashMap::new(),
            &HashSet::new(),
            &HashSet::new(),
            &HashSet::new(),
            &HashSet::new(),
            100,
        );
        let war = crate::schedule_constraints::ScheduleConstraint::Precedence {
            consumer_idx: ix.store_word0,
            dep_idx: ix.load_before,
            result_offset: 0,
            dep_is_shifted: false,
        };
        assert!(constraints.contains(&war));
        crate::schedule_constraints::check_schedule_constraints(&ops, &cycles, &HashMap::new(), &constraints)
            .unwrap();

        cycles[ix.load_after] = cycles[ix.store_word0];
        let err = check_memory_edges(&ops, |i| cycles[i]).unwrap_err();
        assert!(err.contains("raw"), "{}", err);
        let violation = crate::schedule_constraints::check_schedule_constraints(
            &ops,
            &cycles,
            &HashMap::new(),
            &constraints,
        )
        .unwrap_err();
        assert!(violation.message.contains("Precedence"), "{}", violation.message);
    }

    /// DCE keeps a store that a kept load reads (memory RAW edge).
    #[test]
    fn test_raw_edge_source_is_live_for_dce() {
        let (mut ops, ix) = store_load_same_word();
        compute_memory_dependencies(&mut ops).unwrap();
        let mut vreg_def: HashMap<crate::types::VReg, usize> =
            ops.iter().filter_map(|o| o.dest.map(|d| (d, o.id))).collect();
        let store_id = ops[ix.store_word0].id;
        let n = ops.len();
        let (kept, _, _) = crate::kernel_builder::emit_ssa::eliminate_ssa_dead_code(
            ops,
            &mut vreg_def,
            &mut Vec::new(),
            None,
            None,
        );
        assert!(kept.iter().any(|o| o.id == store_id));
        assert_eq!(kept.len(), n, "every op of the program is live");
    }
}
