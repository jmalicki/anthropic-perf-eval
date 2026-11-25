//! Resource-aware earliest and latest issue cycles (`--asap-alap-tightening energetic`).
//!
//! Precedence alone says an op starts after its longest chain of producers. Slot capacity
//! says more: everything an op depends on has to run first, and only so much fits in a cycle.
//!
//! # The rule
//!
//! Take an op `o` and the set `A` of its ancestors over `dep_ids`. In every schedule of the
//! model (`op_times.rs`) every slot an ancestor occupies is before the issue cycle of `o`
//! (rows `prec_raw_*` and `consumer_after_completion_*` hold whatever is shifted, and an op's
//! completion is at least its issue cycle and every lane). If the ancestors whose slots are
//! all at cycle `r` or later need `W` slots of an engine with `cap` slots per cycle, they fill
//! at least `ceil(W / cap)` cycles of `[r, issue(o))`, so
//!
//! ```text
//! issue(o) >= r + ceil(W / cap)        for every r and every engine
//! ```
//!
//! The mirror image bounds the execution end of `o` (the completion of a shiftable op, the
//! issue cycle of any other, as in `op_times.rs`) from above by the work of its descendants,
//! counted from the horizon. For a select on the multiply-add path that is the issue cycle,
//! not the multiply_add at issue + 1; only issue-cycle domains are tightened.
//!
//! The rule never decides which op of a crowded cycle has to move, so it needs no ordering
//! between ops and holds with or without symmetry rows. (`energetic` also runs the
//! symmetry-aware pass, which does depend on the declared ordering rows and is skipped when
//! they are not in the model; see `model.rs`.) (The rule it replaces counted the ops
//! of an ordering group with the same earliest cycle and pushed the first of them; a schedule
//! that runs those first is then cut off. `verification/resource_bounds.py`, check
//! `RB-old-counting-rule`, has the four-op counterexample.)
//!
//! # What "the slots of an op" are, and where the rule is weaker for it
//!
//! * Work is counted with the cheapest encoding (`makespan_rows::min_work`): a vector op that
//!   cannot be shifted takes a VALU slot; every vector op takes `VLEN` units and every scalar
//!   op one unit of the joint VALU+ALU engine; selects, constants and vbroadcasts count
//!   nothing (their engine, or whether they run at all, is the solver's choice).
//! * A shifted op runs lane by lane, and lane `i` waits only for lane `i` of a shifted
//!   producer. Its lanes can therefore run before its own issue cycle (which the model leaves
//!   free up to the completion cycle) and before a producer has completed. Two consequences:
//!   - the earliest slot of a shiftable ancestor is its *lane* bound, which follows the lane
//!     rows and is at most its issue bound ([`ResourceBounds::lane_lb`]);
//!   - for a shiftable op (or a vbroadcast, whose shifted readers may read its scalar
//!     producer instead) only descendants that cannot be shifted are certain to run after
//!     its completion, so only those are counted in its tail.
//!
//! Soundness of exactly this computation, against an encoding of the model's rows, is checked
//! in `verification/resource_bounds.py` (`./scripts/test.sh proofs`), together with the
//! variants that are not sound (full tails for shiftable ops, issue bounds for shifted
//! ancestors).
//!
//! # Cost
//!
//! One bit set of ancestors and one of descendants per op (`n^2 / 8` bytes each, one at a
//! time) and one pass over each set: about `2 n^2` word operations in total.

use std::collections::{HashMap, HashSet};

use super::makespan_rows::{min_work, OpClasses, RowCaps, RowEngine};
use crate::types::{Opcode, SsaOp};

const ENGINES: usize = RowEngine::ALL.len();

/// Lower bounds that hold in every schedule of the model whose makespan is at most the
/// horizon, indexed like `ops`.
#[derive(Debug, Clone)]
pub(crate) struct ResourceBounds {
    /// The issue cycle is at least this.
    pub(crate) issue_lb: Vec<i32>,
    /// Every slot the op occupies (issue cycle when not shifted, each lane when shifted) is
    /// at least this. Equal to `issue_lb` for an op that cannot be shifted.
    pub(crate) lane_lb: Vec<i32>,
    /// The op's execution end as `op_times.rs` defines it (the completion of a shiftable op,
    /// the issue cycle of any other) is at most `horizon - 1 - finish_tail`. This is not the
    /// last cycle the op occupies in one case: a select on the multiply-add path also
    /// occupies issue + 1, which this bound does not cover. The caller tightens only the
    /// domain of the issue cycle.
    pub(crate) finish_tail: Vec<i32>,
}

/// What [`tighten_asap_alap_by_resources`] changed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct TighteningStats {
    pub(crate) asap_raised: usize,
    pub(crate) asap_cycles: i64,
    pub(crate) alap_lowered: usize,
    pub(crate) alap_cycles: i64,
}

/// Dependency edges over `dep_ids` between ops of the slice, and a topological order.
struct Graph {
    preds: Vec<Vec<usize>>,
    succs: Vec<Vec<usize>>,
    order: Vec<usize>,
}

impl Graph {
    fn new(ops: &[SsaOp]) -> Self {
        let n = ops.len();
        let id_to_idx: HashMap<usize, usize> = ops.iter().enumerate().map(|(i, op)| (op.id, i)).collect();
        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut succs: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, op) in ops.iter().enumerate() {
            for dep in &op.dep_ids {
                if let Some(&d) = id_to_idx.get(dep) {
                    if d != i {
                        preds[i].push(d);
                        succs[d].push(i);
                    }
                }
            }
        }
        let mut indegree: Vec<usize> = preds.iter().map(Vec::len).collect();
        let mut ready: Vec<usize> = (0..n).filter(|&i| indegree[i] == 0).collect();
        let mut order = Vec::with_capacity(n);
        while let Some(i) = ready.pop() {
            order.push(i);
            for &s in &succs[i] {
                indegree[s] -= 1;
                if indegree[s] == 0 {
                    ready.push(s);
                }
            }
        }
        assert_eq!(order.len(), n, "dependency graph has a cycle");
        Graph { preds, succs, order }
    }
}

/// For every op, the set of ops reachable over `edges` (transitively), as rows of a bit
/// matrix. `order` lists an op after everything `edges` leads to from it.
struct Reach {
    words: usize,
    bits: Vec<u64>,
}

impl Reach {
    fn new(order: impl Iterator<Item = usize>, edges: &[Vec<usize>]) -> Self {
        let n = edges.len();
        let words = n.div_ceil(64).max(1);
        let mut bits = vec![0u64; n * words];
        for i in order {
            for &p in &edges[i] {
                let (row_i, row_p) = if p < i {
                    let (lo, hi) = bits.split_at_mut(i * words);
                    (&mut hi[..words], &lo[p * words..(p + 1) * words])
                } else {
                    let (lo, hi) = bits.split_at_mut(p * words);
                    (&mut lo[i * words..(i + 1) * words], &hi[..words])
                };
                for (a, b) in row_i.iter_mut().zip(row_p) {
                    *a |= *b;
                }
                row_i[p / 64] |= 1u64 << (p % 64);
            }
        }
        Reach { words, bits }
    }

    fn for_each(&self, i: usize, mut f: impl FnMut(usize)) {
        for (w, &word) in self.bits[i * self.words..(i + 1) * self.words].iter().enumerate() {
            let mut rest = word;
            while rest != 0 {
                f(w * 64 + rest.trailing_zeros() as usize);
                rest &= rest - 1;
            }
        }
    }
}

/// `max over r and engines of r + ceil(W(r) / cap)`, where `W(r)` is the work of the members
/// whose key is at least `r`; `None` when no member has work.
struct WindowBound {
    caps: [i64; ENGINES],
    /// Work per key and engine; all zero between calls
    hist: Vec<[i64; ENGINES]>,
}

impl WindowBound {
    fn new(caps: &RowCaps) -> Self {
        WindowBound { caps: RowEngine::ALL.map(|e| caps.of(e)), hist: Vec::new() }
    }

    fn add(&mut self, key: i32, work: &[i64; ENGINES], range: &mut Option<(usize, usize)>) {
        let key = key.max(0) as usize;
        if self.hist.len() <= key {
            self.hist.resize(key + 1, [0; ENGINES]);
        }
        for (h, w) in self.hist[key].iter_mut().zip(work) {
            *h += *w;
        }
        *range = Some(range.map_or((key, key), |(lo, hi)| (lo.min(key), hi.max(key))));
    }

    /// The bound over the keys added since the last call, and reset.
    fn take(&mut self, range: Option<(usize, usize)>) -> Option<i32> {
        let (lo, hi) = range?;
        let mut total = [0i64; ENGINES];
        let mut best: Option<i64> = None;
        for r in (lo..=hi).rev() {
            for e in 0..ENGINES {
                total[e] += self.hist[r][e];
                if total[e] > 0 && self.caps[e] > 0 {
                    let bound = r as i64 + (total[e] + self.caps[e] - 1) / self.caps[e];
                    best = Some(best.map_or(bound, |b| b.max(bound)));
                }
            }
            self.hist[r] = [0; ENGINES];
        }
        best.map(|b| b as i32)
    }
}

/// Resource-aware bounds of every op (see the module doc).
///
/// # Arguments
/// * `ops` - SSA ops; dependencies on ops outside the slice are ignored
/// * `asap` - A valid lower bound on each op's issue cycle, by op id (missing: 0)
/// * `shiftable` - Indices of ops that may be shifted to scalar ALU (empty: shift off)
/// * `caps` - Slots per cycle
///
/// # Returns
/// The bounds, indexed like `ops`.
pub(crate) fn resource_bounds(
    ops: &[SsaOp],
    asap: &HashMap<usize, i32>,
    shiftable: &HashSet<usize>,
    caps: &RowCaps,
) -> ResourceBounds {
    let n = ops.len();
    let graph = Graph::new(ops);
    // A vbroadcast may be eliminated when its readers are shifted. Which ones qualify is
    // decided when the model is built; treating every vbroadcast as a candidate only weakens
    // the bounds.
    let optional_vbs: HashSet<usize> = if shiftable.is_empty() {
        HashSet::new()
    } else {
        (0..n).filter(|&i| ops[i].opcode == Opcode::Vbroadcast).collect()
    };
    let classes = OpClasses { shiftable, eliminable_vbs: &optional_vbs };
    let work: Vec<[i64; ENGINES]> =
        (0..n).map(|i| RowEngine::ALL.map(|e| min_work(ops, i, e, &classes))).collect();
    let has_work: Vec<bool> = work.iter().map(|w| w.iter().any(|&x| x > 0)).collect();
    let mut window = WindowBound::new(caps);

    // Heads, producers first.
    let mut issue_lb = vec![0i32; n];
    let mut lane_lb = vec![0i32; n];
    // Earliest cycle a shifted reader's lane can run after this op.
    let mut lane_ready = vec![0i32; n];
    {
        let ancestors = Reach::new(graph.order.iter().copied(), &graph.preds);
        for &i in &graph.order {
            let mut lb = asap.get(&ops[i].id).copied().unwrap_or(0).max(0);
            for &p in &graph.preds[i] {
                lb = lb.max(issue_lb[p] + 1);
            }
            let mut range = None;
            ancestors.for_each(i, |a| {
                if has_work[a] {
                    window.add(lane_lb[a], &work[a], &mut range);
                }
            });
            if let Some(bound) = window.take(range) {
                lb = lb.max(bound);
            }
            issue_lb[i] = lb;
            let from_producers = graph.preds[i].iter().map(|&p| lane_ready[p]).max().unwrap_or(0);
            lane_lb[i] = if shiftable.contains(&i) { lb.min(from_producers) } else { lb };
            lane_ready[i] = if optional_vbs.contains(&i) { from_producers } else { lane_lb[i] + 1 };
        }
    }

    // Tails, readers first.
    let mut finish_tail = vec![0i32; n];
    {
        let descendants = Reach::new(graph.order.iter().rev().copied(), &graph.succs);
        for &i in graph.order.iter().rev() {
            let mut tail = 0i32;
            for &s in &graph.succs[i] {
                tail = tail.max(finish_tail[s] + 1);
            }
            // Lanes of a shifted descendant can run before a shiftable op completes, and
            // before a vbroadcast when they read its scalar producer.
            let only_unshiftable = shiftable.contains(&i) || optional_vbs.contains(&i);
            let mut range = None;
            descendants.for_each(i, |d| {
                if has_work[d] && !(only_unshiftable && shiftable.contains(&d)) {
                    window.add(finish_tail[d], &work[d], &mut range);
                }
            });
            if let Some(bound) = window.take(range) {
                tail = tail.max(bound);
            }
            finish_tail[i] = tail;
        }
    }

    ResourceBounds { issue_lb, lane_lb, finish_tail }
}

/// Tighten issue-cycle bounds with [`resource_bounds`].
///
/// Excludes no schedule of the model whose makespan is at most `horizon`. An op whose bounds
/// would cross (no schedule fits the horizon) keeps the bounds it came with.
///
/// # Arguments
/// * `ops` - SSA ops
/// * `asap` - Lower bounds on the issue cycles, by op id
/// * `alap` - Upper bounds on the issue cycles, by op id
/// * `horizon` - Every cycle of the schedule is below this
/// * `caps` - Slots per cycle
/// * `shiftable` - Indices of ops that may be shifted to scalar ALU
///
/// # Returns
/// The tightened `(asap, alap)` and what changed.
pub(crate) fn tighten_asap_alap_by_resources(
    ops: &[SsaOp],
    asap: &HashMap<usize, i32>,
    alap: &HashMap<usize, i32>,
    horizon: i32,
    caps: &RowCaps,
    shiftable: &HashSet<usize>,
) -> (HashMap<usize, i32>, HashMap<usize, i32>, TighteningStats) {
    let bounds = resource_bounds(ops, asap, shiftable, caps);
    let mut new_asap = asap.clone();
    let mut new_alap = alap.clone();
    let mut stats = TighteningStats::default();
    for (i, op) in ops.iter().enumerate() {
        let old_lb = asap.get(&op.id).copied().unwrap_or(0);
        let old_ub = alap.get(&op.id).copied().unwrap_or(horizon - 1);
        let lb = old_lb.max(bounds.issue_lb[i]);
        let ub = old_ub.min(horizon - 1 - bounds.finish_tail[i]);
        if lb > ub {
            continue;
        }
        if lb > old_lb {
            new_asap.insert(op.id, lb);
            stats.asap_raised += 1;
            stats.asap_cycles += i64::from(lb - old_lb);
        }
        if ub < old_ub {
            new_alap.insert(op.id, ub);
            stats.alap_lowered += 1;
            stats.alap_cycles += i64::from(old_ub - ub);
        }
    }
    (new_asap, new_alap, stats)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Engine, VReg};

    fn op(id: usize, engine: Engine, opcode: Opcode, deps: &[usize]) -> SsaOp {
        let mut op = SsaOp::new(id, engine, opcode, Some(VReg::scalar(id as u32)), vec![]);
        op.dep_ids = deps.iter().copied().collect();
        op
    }

    fn caps(valu: i64, alu: i64) -> RowCaps {
        RowCaps { valu, alu, load: 2, store: 2, flow: 1 }
    }

    fn bounds(ops: &[SsaOp], shiftable: &[usize], caps: &RowCaps) -> ResourceBounds {
        resource_bounds(ops, &HashMap::new(), &shiftable.iter().copied().collect(), caps)
    }

    /// Five loads feed one op; two loads fit in a cycle, so the op cannot start before cycle 3.
    #[test]
    fn test_head_counts_the_slots_of_all_ancestors() {
        let mut ops: Vec<SsaOp> = (0..5).map(|i| op(i, Engine::Load, Opcode::Load, &[])).collect();
        ops.push(op(5, Engine::Alu, Opcode::Add, &[0, 1, 2, 3, 4]));
        let b = bounds(&ops, &[], &caps(6, 12));
        assert_eq!(b.issue_lb, vec![0, 0, 0, 0, 0, 3]);
        assert_eq!(b.finish_tail, vec![1, 1, 1, 1, 1, 0]);
    }

    /// One load feeds seven scalar ops on a machine with two scalar slots: they need four
    /// cycles after it.
    #[test]
    fn test_tail_counts_the_slots_of_all_descendants() {
        let mut ops = vec![op(0, Engine::Load, Opcode::Load, &[])];
        ops.extend((1..8).map(|i| op(i, Engine::Alu, Opcode::Add, &[0])));
        let b = bounds(&ops, &[], &caps(6, 2));
        assert_eq!(b.finish_tail[0], 4);
        assert_eq!(b.issue_lb[1..], [1; 7]);
    }

    /// A chain gets its precedence bounds and nothing more.
    #[test]
    fn test_chain_is_plain_precedence() {
        let ops: Vec<SsaOp> = (0..6)
            .map(|i| {
                let deps: Vec<usize> = if i == 0 { vec![] } else { vec![i - 1] };
                op(i, Engine::Alu, Opcode::Add, &deps)
            })
            .collect();
        let b = bounds(&ops, &[], &caps(6, 12));
        assert_eq!(b.issue_lb, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(b.finish_tail, vec![5, 4, 3, 2, 1, 0]);
    }

    /// Four vector readers on a machine whose joint engine holds one vector op per cycle.
    /// After a multiply_add they need four cycles. After a shiftable op only the readers that
    /// cannot be shifted are counted: shifted lanes may run before the op's completion.
    #[test]
    fn test_tail_of_a_shiftable_op_counts_only_unshiftable_descendants() {
        let build = |producer: Opcode, reader: Opcode| -> Vec<SsaOp> {
            let mut ops = vec![op(0, Engine::Valu, producer, &[])];
            ops.extend((1..5).map(|i| op(i, Engine::Valu, reader, &[0])));
            ops
        };
        let one_vector_per_cycle = caps(1, 0);
        let readers = [1, 2, 3, 4];
        let all = [0, 1, 2, 3, 4];
        // multiply_add -> shiftable readers
        assert_eq!(bounds(&build(Opcode::MultiplyAdd, Opcode::Xor), &readers, &one_vector_per_cycle).finish_tail[0], 4);
        // shiftable -> shiftable readers
        assert_eq!(bounds(&build(Opcode::Add, Opcode::Xor), &all, &one_vector_per_cycle).finish_tail[0], 1);
        // shiftable -> multiply_add readers
        assert_eq!(bounds(&build(Opcode::Add, Opcode::MultiplyAdd), &[0], &one_vector_per_cycle).finish_tail[0], 4);
    }

    /// The lanes of a shiftable op wait for its producers lane by lane, not for the slots of
    /// all its ancestors: its lane bound stays at the precedence bound while its issue bound
    /// is resource-aware, and a reader counts it from the lane bound.
    #[test]
    fn test_lanes_of_a_shiftable_op_keep_the_precedence_bound() {
        let mut ops: Vec<SsaOp> = (0..5).map(|i| op(i, Engine::Load, Opcode::Load, &[])).collect();
        ops.push(op(5, Engine::Valu, Opcode::Add, &[0, 1, 2, 3, 4]));
        ops.push(op(6, Engine::Valu, Opcode::MultiplyAdd, &[5]));
        let b = bounds(&ops, &[5], &caps(6, 12));
        assert_eq!((b.issue_lb[5], b.lane_lb[5]), (3, 1));
        assert_eq!((b.issue_lb[6], b.lane_lb[6]), (4, 4));
    }

    /// With the shift off a vbroadcast always runs, so it is work; with it on it may be
    /// eliminated and counts nothing.
    #[test]
    fn test_vbroadcast_is_work_only_when_nothing_can_be_shifted() {
        let mut ops: Vec<SsaOp> = (0..3).map(|i| op(i, Engine::Valu, Opcode::Vbroadcast, &[])).collect();
        ops.push(op(3, Engine::Valu, Opcode::Add, &[0, 1, 2]));
        assert_eq!(bounds(&ops, &[], &caps(1, 12)).issue_lb[3], 3);
        assert_eq!(bounds(&ops, &[3], &caps(1, 12)).issue_lb[3], 1);
    }

    /// Bounds are intersected with the given ones, and an op whose bounds would cross (the
    /// horizon is too short for any schedule) is left alone.
    #[test]
    fn test_tightening_intersects_and_never_empties_a_domain() {
        let mut ops: Vec<SsaOp> = (0..5).map(|i| op(i, Engine::Load, Opcode::Load, &[])).collect();
        ops.push(op(5, Engine::Alu, Opcode::Add, &[0, 1, 2, 3, 4]));
        let asap: HashMap<usize, i32> = (0..6).map(|i| (i, i32::from(i == 5))).collect();
        let none = HashSet::new();
        let alap = |horizon: i32| -> HashMap<usize, i32> {
            (0..6).map(|i| (i, if i == 5 { horizon - 1 } else { horizon - 2 })).collect()
        };
        let (lo, hi, stats) = tighten_asap_alap_by_resources(&ops, &asap, &alap(10), 10, &caps(6, 12), &none);
        assert_eq!(lo[&5], 3);
        assert_eq!(hi, alap(10));
        assert_eq!(stats, TighteningStats { asap_raised: 1, asap_cycles: 2, alap_lowered: 0, alap_cycles: 0 });
        // Horizon 3: the reader would need cycle 3. Its bounds stay as given.
        let (lo, hi, stats) = tighten_asap_alap_by_resources(&ops, &asap, &alap(3), 3, &caps(6, 12), &none);
        assert_eq!((lo[&5], hi[&5]), (1, 2));
        assert_eq!(stats, TighteningStats::default());
    }
}
