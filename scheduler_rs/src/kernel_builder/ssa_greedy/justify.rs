//! Forward/backward justification of a greedy schedule.
//!
//! The greedy scheduler builds its schedule front to back, so ops that could run late are
//! placed as early as their operands allow and compete for slots with ops on the critical
//! path. Justification (the "forward-backward improvement" of project scheduling) repairs
//! that with two cheap list-scheduling passes:
//!
//! 1. **Backward pass**: schedule the reversed dependency graph, trying the ops that complete
//!    latest in the current schedule first, then mirror the times. Every op is now as late
//!    as the schedule's end allows.
//! 2. **Forward pass**: schedule the original graph, trying ops in the order of the mirrored
//!    schedule's start times. Every op is as early as possible again, but the order now
//!    reflects how urgent each op is for the end of the schedule.
//!
//! The shorter of the two is where the next iteration starts, for `max_iterations`
//! iterations, until a schedule reproduces itself, or until `max_stalled_iterations`
//! consecutive iterations have not produced a shorter accepted schedule; the shortest
//! schedule seen that passes every check is the result.
//!
//! Only cycles and the shift decision (vector slot, or VLEN scalar lanes, possibly split
//! over two consecutive cycles) change. The encodings the greedy chose for ConstChoice and
//! SelectChoice ops are kept, and dead ops the greedy skipped (negative cycle) stay skipped.
//!
//! The pass does not trust its own list scheduler: every candidate is written to a copy of
//! the ops in the pipeline's schedule representation and checked there by
//! [`verify_schedule`] (slot capacities per cycle, every dependency with its precedence
//! offset, memory edges) and by the pipeline's liveness accounting
//! ([`interval_liveness_curve`], the one the greedy's counter is checked against, plus
//! optionally address assignment itself). A candidate that fails a check is dropped; if
//! nothing passes, the input schedule is returned unchanged.

use std::collections::{BTreeSet, HashMap, HashSet};

use crate::address_assign::{assign_addresses, AddressAssignConfig};
use crate::kernel_builder::memory_deps::{check_memory_edges, memory_predecessors};
use crate::types::{
    ConstChoiceEncoding, Engine, Opcode, PrecedenceOffset, SelectChoiceEncoding, SsaOp, VReg, VLEN,
};

use super::super::ssa_lower::can_ssa_valu_shift_to_alu;
use super::{interval_liveness_curve, scratch_pool_peaks};

/// Scalar ALU slots per cycle.
const ALU_SLOTS: usize = 12;
/// Vector ALU slots per cycle.
const VALU_SLOTS: usize = 6;
/// Load slots per cycle.
const LOAD_SLOTS: usize = 2;
/// Store slots per cycle.
const STORE_SLOTS: usize = 2;
/// Flow slots per cycle.
const FLOW_SLOTS: usize = 1;

/// Consecutive iterations without a shorter accepted schedule after which the pipeline's
/// pass stops (`JustifyConfig::max_stalled_iterations`). The iteration lengths are not
/// monotone, so one iteration without a gain is not the end, but an orbit that never
/// reproduces its starting schedule exactly would otherwise run all `max_iterations`.
pub const DEFAULT_MAX_STALLED_ITERATIONS: usize = 4;

// ============================================================================
// Configuration and outcome
// ============================================================================

/// Options of [`justify_schedule`].
#[derive(Debug, Clone, Default)]
pub struct JustifyConfig {
    /// Most backward+forward iterations to run (0 = leave the schedule alone).
    pub max_iterations: usize,
    /// Stop after this many consecutive iterations in which no shorter schedule was accepted
    /// (0 = no such limit: run `max_iterations` unless a schedule reproduces itself). The
    /// pipeline uses [`DEFAULT_MAX_STALLED_ITERATIONS`].
    pub max_stalled_iterations: usize,
    /// Whether the pass may change which shiftable vector ops run as scalar lanes. False
    /// when something other than the greedy owns the shift decision (forced encodings, hint
    /// passes): every op then keeps the encoding it has.
    pub allow_shift_changes: bool,
    /// Whether a shifted op may straddle two consecutive cycles (as `GreedyShape::split_shift`).
    pub split_shift: bool,
    /// Scratch budget in words: a candidate whose peak live words exceed it is rejected.
    pub scratch_budget: usize,
    /// Scratch model `pools`: also require `8 * max vectors + max scalars <= scratch_budget`.
    pub scratch_pools: bool,
    /// When set, a candidate must also be placed by address assignment with this
    /// configuration (its `shifted_alu_cycles` are replaced by the candidate's) without
    /// exceeding the configuration's budget.
    pub allocator: Option<AddressAssignConfig>,
}

/// What [`justify_schedule`] did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct JustifyOutcome {
    /// Schedule length before the pass.
    pub cycles_before: usize,
    /// Schedule length after the pass (equal to `cycles_before` when nothing improved).
    pub cycles_after: usize,
    /// Backward+forward iterations run.
    pub iterations: usize,
    /// Shifted ops in the resulting schedule.
    pub shifted: usize,
    /// Shifted ops of the resulting schedule whose lanes straddle two cycles.
    pub split: usize,
    /// Peak live words of the input schedule (interval liveness).
    pub peak_live_before: usize,
    /// Peak live words of the resulting schedule (interval liveness).
    pub peak_live_after: usize,
    /// Candidate schedules that were shorter than the current one and were checked.
    pub candidates_checked: usize,
    /// Checked candidates rejected because of scratch (peak live words, pools, allocator).
    pub scratch_rejections: usize,
    /// Checked candidates rejected by [`verify_schedule`] (a bug in the list scheduler if
    /// it ever happens; the candidate is dropped).
    pub verify_rejections: usize,
    /// Checked candidates that are valid schedules but place an op outside its
    /// `[ASAP, ALAP]` domain (only when domains are given, i.e. the schedule becomes a
    /// CP-SAT hint). Not a bug: the list scheduler does not know the domains.
    pub domain_rejections: usize,
    /// Why the pass did nothing or stopped early, when that is not "no shorter schedule".
    pub note: Option<String>,
}

impl JustifyOutcome {
    /// One-line summary for logs.
    pub fn summary(&self) -> String {
        format!(
            "{} -> {} cycles in {} iteration(s), {} shifted ({} split), peak live {} -> {} words, \
             {} candidate(s) checked, {} rejected for scratch, {} rejected by verification, \
             {} rejected for ASAP/ALAP domains{}",
            self.cycles_before,
            self.cycles_after,
            self.iterations,
            self.shifted,
            self.split,
            self.peak_live_before,
            self.peak_live_after,
            self.candidates_checked,
            self.scratch_rejections,
            self.verify_rejections,
            self.domain_rejections,
            self.note.as_ref().map(|n| format!(" [{}]", n)).unwrap_or_default()
        )
    }
}

// ============================================================================
// Encodings: which slot an op uses, as the greedy decided
// ============================================================================

/// Engine an op occupies given the greedy's ConstChoice / SelectChoice decisions.
///
/// # Arguments
/// * `op` - The op
/// * `const_choice` - ConstChoice encoding per op id (absent = Load)
/// * `select_choice` - SelectChoice encoding per op id (absent = vselect)
///
/// # Returns
/// The engine, or `Err` for an encoding this pass does not model (ALU-derived constants).
fn decided_engine(
    op: &SsaOp,
    const_choice: &HashMap<usize, ConstChoiceEncoding>,
    select_choice: &HashMap<usize, SelectChoiceEncoding>,
) -> Result<Engine, String> {
    Ok(match op.opcode {
        Opcode::ConstChoice => match const_choice.get(&op.id) {
            Some(ConstChoiceEncoding::AddImm) => Engine::Flow,
            Some(ConstChoiceEncoding::Load) | None => Engine::Load,
            Some(ConstChoiceEncoding::AluDerive(_)) => {
                return Err(format!("op {} is an ALU-derived constant, which justification does not model", op.id))
            }
        },
        Opcode::SelectChoice => match select_choice.get(&op.id) {
            Some(SelectChoiceEncoding::Fma) => Engine::Valu,
            Some(SelectChoiceEncoding::Vselect) | None => Engine::Flow,
        },
        _ => op.engine,
    })
}

/// True for a SelectChoice on the multiply-add path without a precomputed delta: the
/// subtract runs at the op's cycle and the multiply_add one cycle later (one vector slot
/// each), so the result is readable one cycle later than the op's precedence offset says.
fn is_two_cycle_select(op: &SsaOp, select_choice: &HashMap<usize, SelectChoiceEncoding>) -> bool {
    op.opcode == Opcode::SelectChoice
        && op.sub_temp_vreg.is_some()
        && matches!(select_choice.get(&op.id), Some(SelectChoiceEncoding::Fma))
}

// ============================================================================
// Schedule verification (on the pipeline's representation)
// ============================================================================

/// Check a schedule in the pipeline's representation: `op.cycle`, the shifted set and the
/// per-lane cycles of split shifts.
///
/// Checks, for every op with a non-negative cycle:
/// * slot capacities per cycle (a shifted op uses one scalar slot per lane at that lane's
///   cycle; a two-cycle select uses a vector slot at its cycle and the next);
/// * every `dep_ids` edge: the consumer's first cycle is at least the producer's cycle plus
///   its precedence offset (`PrecedenceOffset::precedence_offset_for_asap`, plus one for a
///   two-cycle select), where a shifted op's cycle is its last lane's and its first cycle
///   its first lane's;
/// * every memory edge with its own latency (`check_memory_edges`);
/// * shape of the shift records (only shiftable vector ops, VLEN lanes, last lane = cycle).
///
/// # Arguments
/// * `ops` - Ops with cycles; negative cycles (skipped dead ops) are ignored
/// * `enable_valu_shift` - Whether shifting is enabled (defines the shiftable set)
/// * `shifted_ops` - Ids of ops run as scalar lanes
/// * `shifted_lane_cycles` - Per-lane cycles of shifted ops whose lanes are not all in the op's cycle
/// * `const_choice` - ConstChoice encodings by op id
/// * `select_choice` - SelectChoice encodings by op id
///
/// # Returns
/// The schedule length (last used cycle + 1), or the first violation found.
pub(crate) fn verify_schedule(
    ops: &[SsaOp],
    enable_valu_shift: bool,
    shifted_ops: &HashSet<usize>,
    shifted_lane_cycles: &HashMap<usize, Vec<i32>>,
    const_choice: &HashMap<usize, ConstChoiceEncoding>,
    select_choice: &HashMap<usize, SelectChoiceEncoding>,
) -> Result<usize, String> {
    let id_to_idx: HashMap<usize, usize> = ops.iter().enumerate().map(|(idx, op)| (op.id, idx)).collect();
    let horizon = ops.iter().map(|op| op.cycle).max().unwrap_or(-1) + 2;
    if horizon <= 1 {
        return Ok(0);
    }
    let horizon = horizon as usize;
    // Slots used per cycle: [load, store, alu, valu, flow].
    let mut used: Vec<[usize; 5]> = vec![[0; 5]; horizon];
    // First cycle in which each op reads its operands, and the cycle its result is readable.
    let mut first_cycle: Vec<i32> = vec![-1; ops.len()];
    let mut result_cycle: Vec<i32> = vec![-1; ops.len()];
    let mut makespan: i32 = 0;
    let mut scheduled_ids: HashSet<usize> = HashSet::new();
    for (idx, op) in ops.iter().enumerate() {
        if op.cycle < 0 {
            continue;
        }
        scheduled_ids.insert(op.id);
        let engine = decided_engine(op, const_choice, select_choice)?;
        let two_cycle = is_two_cycle_select(op, select_choice);
        let shiftable = enable_valu_shift && can_ssa_valu_shift_to_alu(op);
        result_cycle[idx] = op.cycle + op.precedence_offset_for_asap(shiftable) + i32::from(two_cycle);
        makespan = makespan.max(op.cycle + 1 + i32::from(two_cycle));
        if shifted_ops.contains(&op.id) {
            if engine != Engine::Valu || !can_ssa_valu_shift_to_alu(op) || two_cycle {
                return Err(format!("op {} ({:?}) is recorded as shifted but cannot be shifted", op.id, op.opcode));
            }
            match shifted_lane_cycles.get(&op.id) {
                Some(lanes) => {
                    if lanes.len() != VLEN || lanes.iter().any(|&c| c < 0) {
                        return Err(format!("shifted op {} has malformed lane cycles {:?}", op.id, lanes));
                    }
                    if lanes.iter().copied().max() != Some(op.cycle) {
                        return Err(format!(
                            "shifted op {} has cycle {} but its last lane runs at {:?}",
                            op.id,
                            op.cycle,
                            lanes.iter().max()
                        ));
                    }
                    for &c in lanes {
                        used[c as usize][2] += 1;
                    }
                    first_cycle[idx] = lanes.iter().copied().min().unwrap_or(op.cycle);
                }
                None => {
                    used[op.cycle as usize][2] += VLEN;
                    first_cycle[idx] = op.cycle;
                }
            }
            continue;
        }
        first_cycle[idx] = op.cycle;
        let column = match engine {
            Engine::Load => 0,
            Engine::Store => 1,
            Engine::Alu => 2,
            Engine::Valu => 3,
            Engine::Flow => 4,
            _ => continue,
        };
        used[op.cycle as usize][column] += 1;
        if two_cycle {
            used[op.cycle as usize + 1][column] += 1;
        }
    }
    // HashMap order is irrelevant here: any stray entry is an error.
    if let Some(stray) = shifted_lane_cycles.keys().find(|id| !shifted_ops.contains(id) || !scheduled_ids.contains(id)) {
        return Err(format!("lane cycles recorded for op {} which is not a scheduled shifted op", stray));
    }
    let capacity = [LOAD_SLOTS, STORE_SLOTS, ALU_SLOTS, VALU_SLOTS, FLOW_SLOTS];
    let names = ["load", "store", "alu", "valu", "flow"];
    for (cycle, row) in used.iter().enumerate() {
        for column in 0..capacity.len() {
            if row[column] > capacity[column] {
                return Err(format!(
                    "cycle {} uses {} {} slots (capacity {})",
                    cycle, row[column], names[column], capacity[column]
                ));
            }
        }
    }
    for (idx, op) in ops.iter().enumerate() {
        if op.cycle < 0 {
            continue;
        }
        for dep_id in &op.dep_ids {
            let Some(&dep_idx) = id_to_idx.get(dep_id) else {
                continue; // external dependency: satisfied, as in the greedy
            };
            if ops[dep_idx].cycle < 0 {
                continue; // skipped dead op: satisfied, as in the greedy
            }
            if first_cycle[idx] < result_cycle[dep_idx] {
                return Err(format!(
                    "op {} starts at cycle {} but its operand from op {} is readable from cycle {}",
                    op.id, first_cycle[idx], dep_id, result_cycle[dep_idx]
                ));
            }
        }
    }
    check_memory_edges(ops, |idx| ops[idx].cycle)?;
    Ok(makespan as usize)
}

// ============================================================================
// The scheduling problem the list scheduler sees
// ============================================================================

/// Slot class of a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Slot {
    Load,
    Store,
    Alu,
    Valu,
    Flow,
    /// Occupies no slot (barriers).
    Free,
}

/// Whether a vector op may or must run as scalar lanes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shift {
    /// Always in its own engine.
    Never,
    /// A vector slot if one is free, scalar lanes otherwise.
    Optional,
    /// Always as scalar lanes (the encoding is fixed from outside).
    Always,
}

/// Scheduled ops as nodes, with every edge at latency one from the producer's completion.
///
/// `dep_ids` edges have latency one from completion by `PrecedenceOffset` (a two-cycle
/// select completes one cycle after it starts). Memory edges have latency zero or one; the
/// list scheduler uses one for both, which is conservative, and [`verify_schedule`] checks
/// the real latencies.
struct Graph {
    /// Index into `ops` per node.
    op_index: Vec<usize>,
    slot: Vec<Slot>,
    shift: Vec<Shift>,
    /// Two-cycle select: one vector slot in each of two consecutive cycles.
    two_cycle: Vec<bool>,
    /// A shifted run of this node may straddle two cycles.
    splittable: Vec<bool>,
    pred: Vec<Vec<usize>>,
    succ: Vec<Vec<usize>>,
}

impl Graph {
    /// Build the graph of the scheduled ops (non-negative cycle).
    ///
    /// # Arguments
    /// * `ops` - Ops with the greedy's cycles
    /// * `enable_valu_shift` - Whether shifting is enabled at all
    /// * `shifted_ops` - The greedy's shifted set (decides `Shift::Always` when shift changes are not allowed)
    /// * `const_choice` / `select_choice` - The greedy's encodings
    /// * `config` - Pass options
    ///
    /// # Returns
    /// The graph, or `Err` when an op uses something the pass does not model.
    fn build(
        ops: &[SsaOp],
        enable_valu_shift: bool,
        shifted_ops: &HashSet<usize>,
        const_choice: &HashMap<usize, ConstChoiceEncoding>,
        select_choice: &HashMap<usize, SelectChoiceEncoding>,
        config: &JustifyConfig,
    ) -> Result<Graph, String> {
        let id_to_idx: HashMap<usize, usize> = ops.iter().enumerate().map(|(idx, op)| (op.id, idx)).collect();
        let op_index: Vec<usize> = (0..ops.len()).filter(|&idx| ops[idx].cycle >= 0).collect();
        let mut node_of: Vec<Option<usize>> = vec![None; ops.len()];
        for (node, &idx) in op_index.iter().enumerate() {
            node_of[idx] = Some(node);
        }
        let n = op_index.len();
        let mem_pred = memory_predecessors(ops, &id_to_idx);
        let mut graph = Graph {
            op_index,
            slot: Vec::with_capacity(n),
            shift: Vec::with_capacity(n),
            two_cycle: Vec::with_capacity(n),
            splittable: vec![false; n],
            pred: vec![Vec::new(); n],
            succ: vec![Vec::new(); n],
        };
        for node in 0..n {
            let idx = graph.op_index[node];
            let op = &ops[idx];
            let engine = decided_engine(op, const_choice, select_choice)?;
            let slot = match engine {
                Engine::Load => Slot::Load,
                Engine::Store => Slot::Store,
                Engine::Alu => Slot::Alu,
                Engine::Valu => Slot::Valu,
                Engine::Flow => Slot::Flow,
                Engine::Barrier => Slot::Free,
                other => return Err(format!("op {} has engine {:?}, which justification does not model", op.id, other)),
            };
            let can_shift = enable_valu_shift && engine == Engine::Valu && can_ssa_valu_shift_to_alu(op);
            let shift = if shifted_ops.contains(&op.id) && !config.allow_shift_changes {
                Shift::Always
            } else if can_shift && config.allow_shift_changes {
                Shift::Optional
            } else {
                Shift::Never
            };
            graph.slot.push(slot);
            graph.shift.push(shift);
            graph.two_cycle.push(is_two_cycle_select(op, select_choice));
            let mut preds: Vec<usize> = op
                .dep_ids
                .iter()
                .filter_map(|dep_id| id_to_idx.get(dep_id).and_then(|&dep_idx| node_of[dep_idx]))
                .chain(mem_pred[idx].iter().filter_map(|&(src, _)| node_of[src]))
                .collect();
            preds.sort_unstable();
            preds.dedup();
            if preds.contains(&node) {
                return Err(format!("op {} depends on itself", op.id));
            }
            graph.pred[node] = preds;
        }
        for node in 0..n {
            for k in 0..graph.pred[node].len() {
                let p = graph.pred[node][k];
                graph.succ[p].push(node);
            }
        }
        // A destination nothing reads is released the cycle after its first write, before a
        // split's last lanes are written: such an op is only shifted whole (as in the greedy).
        for node in 0..n {
            let has_reader = ops[graph.op_index[node]].dest.is_some() && !graph.succ[node].is_empty();
            graph.splittable[node] = config.split_shift && has_reader;
        }
        Ok(graph)
    }

    fn len(&self) -> usize {
        self.op_index.len()
    }
}

// ============================================================================
// Times: a schedule of the graph
// ============================================================================

/// Start and completion cycle of every node.
///
/// `first_lanes` is 0 for a node in its own engine, VLEN for a node shifted whole
/// (`start == done`), and `k` in `1..VLEN` for a split shift with `k` lanes at `start` and
/// the rest at `done == start + 1`. A two-cycle select has `done == start + 1`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Times {
    start: Vec<i32>,
    done: Vec<i32>,
    first_lanes: Vec<u8>,
}

impl Times {
    /// Schedule length: last completion + 1.
    fn makespan(&self) -> i32 {
        self.done.iter().copied().max().map_or(0, |m| m + 1)
    }

    /// The same schedule with time reversed: cycle `t` becomes `makespan - 1 - t`, so start
    /// and completion swap and a split's lane counts swap.
    fn mirrored(&self) -> Times {
        let last = self.makespan() - 1;
        Times {
            start: self.done.iter().map(|&d| last - d).collect(),
            done: self.start.iter().map(|&s| last - s).collect(),
            first_lanes: self
                .first_lanes
                .iter()
                .map(|&k| if k == 0 || k as usize == VLEN { k } else { VLEN as u8 - k })
                .collect(),
        }
    }

    /// Read the schedule the pipeline holds.
    fn from_ops(
        graph: &Graph,
        ops: &[SsaOp],
        shifted_ops: &HashSet<usize>,
        shifted_lane_cycles: &HashMap<usize, Vec<i32>>,
    ) -> Times {
        let n = graph.len();
        let mut times = Times { start: vec![0; n], done: vec![0; n], first_lanes: vec![0; n] };
        for node in 0..n {
            let op = &ops[graph.op_index[node]];
            let (start, done, first_lanes) = match shifted_lane_cycles.get(&op.id) {
                Some(lanes) if !lanes.is_empty() => {
                    let first = lanes.iter().copied().min().unwrap_or(op.cycle);
                    let count = lanes.iter().filter(|&&c| c == first).count();
                    (first, op.cycle, count.min(VLEN) as u8)
                }
                _ if shifted_ops.contains(&op.id) => (op.cycle, op.cycle, VLEN as u8),
                _ if graph.two_cycle[node] => (op.cycle, op.cycle + 1, 0),
                _ => (op.cycle, op.cycle, 0),
            };
            times.start[node] = start;
            times.done[node] = done;
            times.first_lanes[node] = first_lanes;
        }
        times
    }

    /// Write the schedule in the pipeline's representation.
    ///
    /// `op.cycle` is the completion cycle (the last lane of a shifted op), except for a
    /// two-cycle select, whose cycle is its subtract's (the first of the two). Only nodes
    /// whose shift is `Optional` change membership of `shifted_ops`.
    fn write(
        &self,
        graph: &Graph,
        ops: &mut [SsaOp],
        shifted_ops: &mut HashSet<usize>,
        shifted_lane_cycles: &mut HashMap<usize, Vec<i32>>,
    ) {
        for node in 0..graph.len() {
            let op = &mut ops[graph.op_index[node]];
            op.cycle = if graph.two_cycle[node] { self.start[node] } else { self.done[node] };
            let lanes = self.first_lanes[node] as usize;
            if graph.shift[node] == Shift::Optional {
                if lanes > 0 {
                    shifted_ops.insert(op.id);
                } else {
                    shifted_ops.remove(&op.id);
                }
            }
            if lanes > 0 && lanes < VLEN {
                let cycles = (0..VLEN).map(|l| if l < lanes { self.start[node] } else { self.done[node] }).collect();
                shifted_lane_cycles.insert(op.id, cycles);
            } else {
                shifted_lane_cycles.remove(&op.id);
            }
        }
    }
}

// ============================================================================
// List scheduling
// ============================================================================

/// Cycle-by-cycle list scheduling of the graph, forward or on the reversed graph.
///
/// In each cycle the nodes whose predecessors have all completed at least one cycle before
/// are tried in `key` order (ties by node index). A node takes a slot of its engine if one
/// is free. A shiftable vector node that finds the vector slots full takes VLEN scalar
/// slots instead, or, when it is splittable and no other split is in progress, the scalar
/// slots that are left now and the rest in the next cycle (where they are reserved before
/// anything else is placed). A two-cycle select takes a vector slot now and one in the
/// next cycle.
///
/// # Arguments
/// * `graph` - The scheduling problem
/// * `reversed` - Schedule the reversed graph (successors before predecessors)
/// * `key` - Priority per node, smallest first
///
/// # Returns
/// Times in the direction scheduled (mirror a reversed result), or `Err` if the scheduler
/// does not terminate (a cycle in the graph).
fn list_schedule(graph: &Graph, reversed: bool, key: &[i64]) -> Result<Times, String> {
    let n = graph.len();
    let (pred, succ) = if reversed { (&graph.succ, &graph.pred) } else { (&graph.pred, &graph.succ) };
    let mut waiting: Vec<usize> = pred.iter().map(Vec::len).collect();
    let mut earliest: Vec<i32> = vec![0; n];
    let mut ready: BTreeSet<(i64, usize)> = (0..n).filter(|&i| waiting[i] == 0).map(|i| (key[i], i)).collect();
    let mut times = Times { start: vec![-1; n], done: vec![-1; n], first_lanes: vec![0; n] };
    let mut left = n;
    let mut cycle: i32 = 0;
    let mut carry_alu: usize = 0;
    let mut carry_valu: usize = 0;
    let max_cycles = 4 * n as i32 + 16;
    let mut candidates: Vec<(i64, usize)> = Vec::new();
    while left > 0 {
        if cycle > max_cycles {
            return Err(format!("list scheduling did not finish: {} of {} nodes left at cycle {}", left, n, cycle));
        }
        let mut alu = std::mem::take(&mut carry_alu);
        let mut valu = std::mem::take(&mut carry_valu);
        let (mut load, mut store, mut flow) = (0usize, 0usize, 0usize);
        candidates.clear();
        candidates.extend(ready.iter().filter(|&&(_, i)| earliest[i] <= cycle).copied());
        for &(k, i) in &candidates {
            // (completion cycle, first lanes) when the node is placed in this cycle.
            let placed: Option<(i32, u8)> = match graph.slot[i] {
                Slot::Free => Some((cycle, 0)),
                Slot::Load if load < LOAD_SLOTS => {
                    load += 1;
                    Some((cycle, 0))
                }
                Slot::Store if store < STORE_SLOTS => {
                    store += 1;
                    Some((cycle, 0))
                }
                Slot::Alu if alu < ALU_SLOTS => {
                    alu += 1;
                    Some((cycle, 0))
                }
                Slot::Flow if flow < FLOW_SLOTS => {
                    flow += 1;
                    Some((cycle, 0))
                }
                Slot::Valu => {
                    let shift = graph.shift[i];
                    if shift != Shift::Always && valu < VALU_SLOTS {
                        valu += 1;
                        if graph.two_cycle[i] {
                            carry_valu += 1;
                            Some((cycle + 1, 0))
                        } else {
                            Some((cycle, 0))
                        }
                    } else if shift != Shift::Never && alu + VLEN <= ALU_SLOTS {
                        alu += VLEN;
                        Some((cycle, VLEN as u8))
                    } else if shift != Shift::Never && graph.splittable[i] && carry_alu == 0 && alu < ALU_SLOTS {
                        // alu + VLEN > ALU_SLOTS here, so 1..VLEN lanes fit now.
                        let first = ALU_SLOTS - alu;
                        alu = ALU_SLOTS;
                        carry_alu = VLEN - first;
                        Some((cycle + 1, first as u8))
                    } else {
                        None
                    }
                }
                _ => None,
            };
            let Some((done, first_lanes)) = placed else {
                continue;
            };
            ready.remove(&(k, i));
            times.start[i] = cycle;
            times.done[i] = done;
            times.first_lanes[i] = first_lanes;
            left -= 1;
            for &s in &succ[i] {
                earliest[s] = earliest[s].max(done + 1);
                waiting[s] -= 1;
                if waiting[s] == 0 {
                    ready.insert((key[s], s));
                }
            }
        }
        cycle += 1;
    }
    Ok(times)
}

// ============================================================================
// Candidate checks
// ============================================================================

/// A copy of the pipeline's schedule state that candidates are written to and checked in.
struct Workspace {
    ops: Vec<SsaOp>,
    shifted_ops: HashSet<usize>,
    shifted_lane_cycles: HashMap<usize, Vec<i32>>,
}

/// Why a candidate was dropped.
enum Rejection {
    /// [`verify_schedule`] found a violation, or disagrees with the list scheduler's length.
    Verification(String),
    /// A valid schedule with an op outside its `[ASAP, ALAP]` domain.
    Domain(String),
    /// Peak live words, the pool condition or address assignment over the budget.
    Scratch(String),
}

/// Peak live words of the schedule in `work` and whether it fits the scratch budget.
///
/// Uses `interval_liveness_curve` (the liveness the greedy's counter is checked against
/// and address assignment computes), the pool condition when the scratch model is `pools`,
/// and address assignment itself when `config.allocator` is set.
///
/// # Returns
/// The peak, or the reason the schedule does not fit.
fn scratch_check(
    work: &Workspace,
    vector_group_lanes: Option<&HashMap<VReg, Vec<VReg>>>,
    config: &JustifyConfig,
) -> Result<usize, String> {
    let (_, peak, _) = interval_liveness_curve(&work.ops, vector_group_lanes, &work.shifted_lane_cycles)?;
    if peak > config.scratch_budget {
        return Err(format!("peak live {} words > budget {}", peak, config.scratch_budget));
    }
    if config.scratch_pools {
        let pools = scratch_pool_peaks(&work.ops, vector_group_lanes, &work.shifted_lane_cycles)?;
        if pools.split_words() > config.scratch_budget {
            return Err(format!("pools need {} words > budget {}", pools.split_words(), config.scratch_budget));
        }
    }
    if let Some(template) = &config.allocator {
        let allocator = AddressAssignConfig {
            shifted_alu_cycles: work.shifted_lane_cycles.clone(),
            ..template.clone()
        };
        let scheduled: Vec<SsaOp> = work.ops.iter().filter(|op| op.cycle >= 0).cloned().collect();
        let result = assign_addresses(&scheduled, &allocator)?;
        if result.budget_exceeded {
            return Err(format!(
                "address assignment does not fit: peak {} words, highest address {} (budget {})",
                result.peak_live, result.max_addr, result.budget
            ));
        }
    }
    Ok(peak)
}

// ============================================================================
// The pass
// ============================================================================

/// Shorten a greedy schedule by forward/backward justification (see the module docs).
///
/// On return `ops`, `shifted_ops` and `shifted_lane_cycles` hold either the input schedule
/// (nothing shorter passed the checks) or a strictly shorter one that passed
/// [`verify_schedule`] and the scratch checks.
///
/// # Arguments
/// * `ops` - Ops with the greedy's cycles (negative = skipped); cycles are rewritten
/// * `enable_valu_shift` - Whether VALU→ALU shifting is enabled
/// * `shifted_ops` - Shifted op ids; rewritten
/// * `shifted_lane_cycles` - Per-lane cycles of split shifts; rewritten
/// * `const_choice` - The greedy's ConstChoice encodings (kept)
/// * `select_choice` - The greedy's SelectChoice encodings (kept)
/// * `vector_group_lanes` - Vector groups, as passed to the greedy
/// * `domains` - Optional `(asap, alap)` by op id: every op of an accepted schedule must have
///   its cycle inside (used when CP-SAT follows and takes the schedule as a hint)
/// * `config` - Options
///
/// # Returns
/// What happened. `Err` only when liveness of the *input* schedule cannot be computed.
#[allow(clippy::too_many_arguments)]
pub(crate) fn justify_schedule(
    ops: &mut [SsaOp],
    enable_valu_shift: bool,
    shifted_ops: &mut HashSet<usize>,
    shifted_lane_cycles: &mut HashMap<usize, Vec<i32>>,
    const_choice: &HashMap<usize, ConstChoiceEncoding>,
    select_choice: &HashMap<usize, SelectChoiceEncoding>,
    vector_group_lanes: Option<&HashMap<VReg, Vec<VReg>>>,
    domains: Option<(&HashMap<usize, i32>, &HashMap<usize, i32>)>,
    config: &JustifyConfig,
) -> Result<JustifyOutcome, String> {
    let mut outcome = JustifyOutcome::default();
    let (_, input_peak, _) = interval_liveness_curve(ops, vector_group_lanes, shifted_lane_cycles)?;
    outcome.peak_live_before = input_peak;
    outcome.peak_live_after = input_peak;
    let count_shifts = |outcome: &mut JustifyOutcome, shifted: &HashSet<usize>, lanes: &HashMap<usize, Vec<i32>>| {
        outcome.shifted = shifted.len();
        outcome.split = lanes.len();
    };
    count_shifts(&mut outcome, shifted_ops, shifted_lane_cycles);

    // The verifier must accept the greedy's own schedule; if it does not, this pass's model
    // of the machine differs from the greedy's and nothing it produces can be trusted.
    let input_cycles =
        match verify_schedule(ops, enable_valu_shift, shifted_ops, shifted_lane_cycles, const_choice, select_choice) {
            Ok(cycles) => cycles,
            Err(e) => {
                outcome.cycles_before = ops.iter().map(|op| op.cycle + 1).max().unwrap_or(0).max(0) as usize;
                outcome.cycles_after = outcome.cycles_before;
                outcome.note = Some(format!("skipped: input schedule fails verification: {}", e));
                return Ok(outcome);
            }
        };
    outcome.cycles_before = input_cycles;
    outcome.cycles_after = input_cycles;
    if config.max_iterations == 0 {
        return Ok(outcome);
    }
    let graph = match Graph::build(ops, enable_valu_shift, shifted_ops, const_choice, select_choice, config) {
        Ok(graph) => graph,
        Err(e) => {
            outcome.note = Some(format!("skipped: {}", e));
            return Ok(outcome);
        }
    };
    if graph.len() == 0 {
        return Ok(outcome);
    }

    let mut work = Workspace {
        ops: ops.to_vec(),
        shifted_ops: shifted_ops.clone(),
        shifted_lane_cycles: shifted_lane_cycles.clone(),
    };
    // Write a candidate to the workspace and check it there.
    let check = |times: &Times, work: &mut Workspace, outcome: &mut JustifyOutcome| -> Result<usize, Rejection> {
        outcome.candidates_checked += 1;
        times.write(&graph, &mut work.ops, &mut work.shifted_ops, &mut work.shifted_lane_cycles);
        let verified = verify_schedule(
            &work.ops,
            enable_valu_shift,
            &work.shifted_ops,
            &work.shifted_lane_cycles,
            const_choice,
            select_choice,
        )
        .and_then(|cycles| {
            if cycles as i32 != times.makespan() {
                return Err(format!("list scheduler reports {} cycles, verifier {}", times.makespan(), cycles));
            }
            Ok(())
        });
        if let Err(e) = verified {
            outcome.verify_rejections += 1;
            return Err(Rejection::Verification(e));
        }
        if let Some((asap, alap)) = domains {
            for op in work.ops.iter().filter(|op| op.cycle >= 0) {
                let lo = asap.get(&op.id).copied().unwrap_or(i32::MIN);
                let hi = alap.get(&op.id).copied().unwrap_or(i32::MAX);
                if op.cycle < lo || op.cycle > hi {
                    outcome.domain_rejections += 1;
                    return Err(Rejection::Domain(format!(
                        "op {} at cycle {} is outside its domain [{}, {}]",
                        op.id, op.cycle, lo, hi
                    )));
                }
            }
        }
        scratch_check(work, vector_group_lanes, config).map_err(|e| {
            outcome.scratch_rejections += 1;
            Rejection::Scratch(e)
        })
    };

    // SCHEDULER_DEBUG: print the length of every pass.
    let debug = std::env::var("SCHEDULER_DEBUG").is_ok();
    let trace = |line: &str| {
        if debug {
            eprintln!("Greedy justify: {}", line);
        }
    };
    // `basis` is the schedule the next passes take their priorities from. It follows the
    // plain algorithm even when a candidate is rejected for scratch (a rejected schedule is
    // still a good priority order); `best` is the shortest candidate that passed every check.
    let mut basis = Times::from_ops(&graph, ops, shifted_ops, shifted_lane_cycles);
    let mut best: Option<(Times, usize)> = None;
    let mut best_cycles = input_cycles as i32;
    // Check the candidates shorter than the best so far, shortest first, and keep the first
    // that passes. Returns whether one was accepted.
    let consider = |candidates: &[&Times],
                    best: &mut Option<(Times, usize)>,
                    best_cycles: &mut i32,
                    work: &mut Workspace,
                    outcome: &mut JustifyOutcome|
     -> bool {
        for candidate in candidates {
            if candidate.makespan() >= *best_cycles {
                continue;
            }
            match check(candidate, work, outcome) {
                Ok(peak) => {
                    *best_cycles = candidate.makespan();
                    *best = Some(((*candidate).clone(), peak));
                    return true;
                }
                Err(Rejection::Scratch(e)) => outcome.note = Some(format!("last scratch rejection: {}", e)),
                Err(Rejection::Domain(e)) => outcome.note = Some(format!("last domain rejection: {}", e)),
                Err(Rejection::Verification(e)) => {
                    outcome.note = Some(format!("candidate failed verification: {}", e))
                }
            }
        }
        false
    };

    // Step 0: one forward pass in the order of the input schedule: the same order without
    // the greedy's pressure heuristics. The iterations start from it whatever its length.
    let start_key: Vec<i64> = basis.start.iter().map(|&s| s as i64).collect();
    match list_schedule(&graph, false, &start_key) {
        Ok(forward) => {
            trace(&format!("order of the input schedule: {} cycles", forward.makespan()));
            consider(&[&forward], &mut best, &mut best_cycles, &mut work, &mut outcome);
            basis = forward;
        }
        Err(e) => outcome.note = Some(e),
    }
    // Iterations since a shorter schedule was last accepted (or since the start).
    let mut stalled: usize = 0;
    for _ in 0..config.max_iterations {
        // Backward: latest completion first on the reversed graph, then mirror.
        let backward_key: Vec<i64> = basis.done.iter().map(|&d| -(d as i64)).collect();
        let backward = match list_schedule(&graph, true, &backward_key) {
            Ok(times) => times.mirrored(),
            Err(e) => {
                outcome.note = Some(e);
                break;
            }
        };
        // Forward: in the order of the right-justified schedule's start times.
        let forward_key: Vec<i64> = backward.start.iter().map(|&s| s as i64).collect();
        let forward = match list_schedule(&graph, false, &forward_key) {
            Ok(times) => times,
            Err(e) => {
                outcome.note = Some(e);
                break;
            }
        };
        outcome.iterations += 1;
        trace(&format!(
            "iteration {}: backward {} cycles, forward {} cycles",
            outcome.iterations,
            backward.makespan(),
            forward.makespan()
        ));
        // Shorter first; the backward schedule on a tie.
        let (shorter, longer) =
            if forward.makespan() < backward.makespan() { (forward, backward) } else { (backward, forward) };
        let improved = consider(&[&shorter, &longer], &mut best, &mut best_cycles, &mut work, &mut outcome);
        stalled = if improved { 0 } else { stalled + 1 };
        // The next iteration starts from the shorter schedule even when it is no shorter
        // than the last one: lengths are not monotone and later iterations still find
        // shorter schedules. A schedule that reproduces itself ends the search, and so do
        // `max_stalled_iterations` iterations in a row without a shorter accepted schedule
        // (a longer orbit never reproduces `basis` exactly).
        if shorter == basis {
            break;
        }
        if config.max_stalled_iterations > 0 && stalled >= config.max_stalled_iterations {
            trace(&format!("stopped after {} iteration(s) without a shorter accepted schedule", stalled));
            break;
        }
        basis = shorter;
    }
    let Some((best, best_peak)) = best else {
        return Ok(outcome);
    };
    if outcome.scratch_rejections == 0 && outcome.verify_rejections == 0 && outcome.domain_rejections == 0 {
        outcome.note = None;
    }

    // Commit: write the accepted schedule to the pipeline's state and verify once more
    // there; restore the input on any failure.
    let saved_cycles: Vec<i32> = ops.iter().map(|op| op.cycle).collect();
    let saved_shifted = shifted_ops.clone();
    let saved_lanes = shifted_lane_cycles.clone();
    best.write(&graph, ops, shifted_ops, shifted_lane_cycles);
    match verify_schedule(ops, enable_valu_shift, shifted_ops, shifted_lane_cycles, const_choice, select_choice) {
        Ok(cycles) if cycles as i32 == best_cycles => {
            outcome.cycles_after = cycles;
            outcome.peak_live_after = best_peak;
            count_shifts(&mut outcome, shifted_ops, shifted_lane_cycles);
        }
        other => {
            for (op, &cycle) in ops.iter_mut().zip(&saved_cycles) {
                op.cycle = cycle;
            }
            *shifted_ops = saved_shifted;
            *shifted_lane_cycles = saved_lanes;
            outcome.note = Some(format!("fell back to the input schedule: final verification gave {:?}", other));
        }
    }
    Ok(outcome)
}
