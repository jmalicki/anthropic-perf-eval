//! Per-section (per-round) peak liveness and budget check.
//!
//! Used by the two-phase CP-SAT flow: after greedy scheduling with barriers,
//! we compute peak scratch usage in each section (ops between barriers).
//! If a section exceeds the scratch budget, we run CP-SAT feasibility on that
//! section only before the full optimization pass.
//!
//! **Hints for per-round CP-SAT**: The per-round feasibility run must receive
//! cycle hints so CP-SAT has a warm start. Build hints in local cycle space:
//! for each op in the segment, hint = greedy_cycle[global_idx] - section_offset.
//! Use `segment_hints_for_section` for this.

use std::collections::{HashMap, HashSet};

use crate::types::{Opcode, SsaOp, VReg};

use super::barrier_elision::section_boundaries;

/// Maps a scratch operand to the SSA values that own its words. A lane of a
/// decomposed vector uses the vector's allocation. An assembled vector uses
/// the individual lane definitions.
#[derive(Default)]
pub(crate) struct ScratchAliases {
    owners: HashMap<VReg, Vec<VReg>>,
}

impl ScratchAliases {
    pub(crate) fn new(groups: &HashMap<VReg, Vec<VReg>>, defined: impl Fn(&VReg) -> bool) -> Self {
        let mut owners = HashMap::new();
        for (&group, lanes) in groups {
            if defined(&group) {
                for &lane in lanes {
                    owners.insert(lane, vec![group]);
                }
            } else {
                owners.insert(group, lanes.clone());
            }
        }
        Self { owners }
    }

    fn resolve<'a>(&'a self, value: &'a VReg) -> &'a [VReg] {
        self.owners.get(value).map(Vec::as_slice).unwrap_or_else(|| std::slice::from_ref(value))
    }

    /// The values that own `value`'s words (itself, its decomposed vector or its lanes).
    pub(crate) fn resolve_owned(&self, value: VReg) -> Vec<VReg> {
        self.resolve(&value).to_vec()
    }

    /// [`Self::reads`] without the op's `delta_vreg`.
    pub(crate) fn reads_except_delta<'a>(&'a self, op: &'a SsaOp) -> impl Iterator<Item = VReg> + 'a {
        op.srcs.iter().filter_map(|operand| match operand {
            crate::types::Operand::VReg(v) => Some(v),
            _ => None,
        }).chain(op.keep_live.iter().flatten())
            .flat_map(|v| self.resolve(v).iter().copied())
    }

    pub(crate) fn reads<'a>(&'a self, op: &'a SsaOp) -> impl Iterator<Item = VReg> + 'a {
        op.srcs.iter().filter_map(|operand| match operand {
            crate::types::Operand::VReg(v) => Some(v),
            _ => None,
        }).chain(op.delta_vreg.iter()).chain(op.keep_live.iter().flatten())
            .flat_map(|v| self.resolve(v).iter().copied())
    }
}

/// Result of per-section budget check.
#[derive(Debug, Clone)]
pub struct SectionBudgetResult {
    /// Peak live words in each section (same length as boundaries).
    pub section_peaks: Vec<usize>,
    /// True for sections where peak > scratch_budget.
    pub section_exceeds_budget: Vec<bool>,
}

/// Compute peak liveness in each section and which sections exceed the scratch budget.
///
/// Counts on the cycle timeline (a value is live over the closed `[def, last access]` cycles)
/// with the allocator's intervals; [`section_budget`] states the rules. Ops with cycle < 0
/// (e.g. dead stores) are skipped.
///
/// * `ops`: Full SSA op list with barriers; each op must have `cycle` set (from greedy).
/// * `scratch_budget`: Max allowed simultaneous live words.
/// * `select_fma_op_ids`: Default SelectChoice FMA set (e.g. greedy). Used for sections that
///   do not have a per-section override.
/// * `section_select_fma_overrides`: When provided, overrides[k] is the SelectChoice FMA set
///   for section k (e.g. from per-section CP-SAT). Use this when measuring peak for sections
///   that were re-scheduled by per-section CP-SAT so the same FMA/vselect decisions are used.
///   This supports the merged-hint flow: each section's peak is measured with that section's
///   actual solution (greedy or per-section CP-SAT), so post-merge check matches segment model.
/// * ConstChoice ops: each is a single op with one dest VReg. Scratch usage is one word
///   regardless of whether we later lower to Load or AddImm, so no extra parameter or
///   logic is needed (unlike SelectChoice's sub_temp_vreg when FMA is chosen).
pub fn sections_exceeding_budget(
    ops: &[SsaOp],
    scratch_budget: usize,
    select_fma_op_ids: Option<&HashSet<usize>>,
    section_select_fma_overrides: Option<&[Option<HashSet<usize>>]>,
) -> SectionBudgetResult {
    sections_exceeding_budget_with(
        ops,
        scratch_budget,
        select_fma_op_ids,
        section_select_fma_overrides,
        crate::cpsat_scheduler::ScratchTime::Cycle,
    )
}

/// [`sections_exceeding_budget`] on a chosen timeline (see [`section_budget`]), without
/// shifted lanes, ALU-derived constants or vector groups.
pub fn sections_exceeding_budget_with(
    ops: &[SsaOp],
    scratch_budget: usize,
    select_fma_op_ids: Option<&HashSet<usize>>,
    section_select_fma_overrides: Option<&[Option<HashSet<usize>>]>,
    scratch_time: crate::cpsat_scheduler::ScratchTime,
) -> SectionBudgetResult {
    section_budget(ops, scratch_budget, select_fma_op_ids, section_select_fma_overrides, scratch_time, &SectionLiveness::default())
        .expect("schedule reads a value before it is written")
}

/// What the allocator needs, besides the ops, to compute a schedule's live intervals: the
/// per-lane cycles of VALU ops run on the ALU, the ConstChoice ops derived on the ALU (their
/// lowered op reads the producers at its cycle) and the vector groups. With these the
/// section check counts exactly what `assign_addresses` will see.
#[derive(Default, Clone, Copy)]
pub(crate) struct SectionLiveness<'a> {
    /// Per-lane ALU cycles of shifted VALU ops (`SchedulingDecisions::shifted_alu_cycles`).
    pub shifted_alu_cycles: Option<&'a HashMap<usize, Vec<i32>>>,
    /// ConstChoice ops lowered as an ALU derivation, by op id.
    pub const_alu_derive_ops: Option<&'a HashMap<usize, crate::types::AluDeriveInfo>>,
    /// Vector group alias to its 8 lanes.
    pub vector_group_lanes: Option<&'a HashMap<VReg, Vec<VReg>>>,
}

/// Peak live words of each section, counted by the allocator itself: live intervals from
/// `address_assign::compute_live_intervals` (shifted lanes, multiply-add selects, ALU-derived
/// constants, vector groups), spans from `compute_value_spans`, then the most words covering
/// one half-cycle point of the section's range.
///
/// `ScratchTime::Phase` uses the allocator's spans as they are (proof A8 in
/// verification/allocation_conjectures.py: a word read for the last time in a cycle is free
/// for a value written in that cycle), so with it the check agrees with address assignment
/// exactly. `ScratchTime::Cycle` widens every span to whole cycles, `[2 start, 2 end + 1]`:
/// a value counts over every cycle from its first write to its last access, as
/// `--cpsat-scratch-time cycle` says.
///
/// A section's range runs from its first cycle (earliest shifted lane included) through the
/// cycle before the next section's first cycle, so values that cross a gap are counted.
/// Section `k`'s multiply-add selects are taken from `section_select_fma_overrides[k]` when
/// given, otherwise from `select_fma_op_ids`.
///
/// # Returns
/// Per-section peaks, or `Err` if the schedule reads a value before it is written.
pub(crate) fn section_budget(
    ops: &[SsaOp],
    scratch_budget: usize,
    select_fma_op_ids: Option<&HashSet<usize>>,
    section_select_fma_overrides: Option<&[Option<HashSet<usize>>]>,
    scratch_time: crate::cpsat_scheduler::ScratchTime,
    liveness: &SectionLiveness<'_>,
) -> Result<SectionBudgetResult, String> {
    use crate::address_assign::{compute_live_intervals, compute_value_spans, last_write_cycles, AddressAssignConfig};

    let boundaries = section_boundaries(ops);
    let empty_shifts = HashMap::new();
    let shifted = liveness.shifted_alu_cycles.unwrap_or(&empty_shifts);

    // Multiply-add selects: each section's own set (override or default), restricted to its ops.
    let mut select_fma_ops: HashSet<usize> = HashSet::new();
    for (k, &(start, end)) in boundaries.iter().enumerate() {
        let fma_ids = section_select_fma_overrides
            .and_then(|o| o.get(k).and_then(|x| x.as_ref()))
            .or(select_fma_op_ids);
        if let Some(fma_ids) = fma_ids {
            select_fma_ops.extend(ops[start..end].iter().map(|op| op.id).filter(|id| fma_ids.contains(id)));
        }
    }

    let scheduled: Vec<SsaOp> = ops
        .iter()
        .filter(|op| op.cycle >= 0 && op.opcode != Opcode::Barrier)
        .cloned()
        .collect();
    let scheduled_ids: HashSet<usize> = scheduled.iter().map(|op| op.id).collect();
    // A group none of whose values is written by a scheduled op occupies nothing
    // (`compute_live_intervals` would give it a phantom interval).
    let written: HashSet<VReg> = scheduled.iter().filter_map(|op| op.dest).collect();
    let vector_group_lanes: HashMap<VReg, Vec<VReg>> = liveness
        .vector_group_lanes
        .into_iter()
        .flatten()
        .filter(|(key, lanes)| written.contains(*key) || lanes.iter().any(|l| written.contains(l)))
        .map(|(key, lanes)| (*key, lanes.clone()))
        .collect();
    let config = AddressAssignConfig {
        select_fma_ops,
        shifted_alu_cycles: shifted
            .iter()
            .filter(|(id, _)| scheduled_ids.contains(id))
            .map(|(id, lanes)| (*id, lanes.clone()))
            .collect(),
        const_alu_derive_ops: liveness
            .const_alu_derive_ops
            .into_iter()
            .flatten()
            .filter(|(id, _)| scheduled_ids.contains(id))
            .map(|(id, info)| (*id, info.clone()))
            .collect(),
        vector_group_lanes,
        ..Default::default()
    };
    let intervals = compute_live_intervals(&config, &scheduled)?;
    let spans = compute_value_spans(&intervals, &last_write_cycles(&scheduled, &config));

    // Section cycle ranges, earliest shifted lane included.
    let section_ranges: Vec<(i32, i32)> = boundaries
        .iter()
        .map(|&(start, end)| {
            let mut range = (i32::MAX, i32::MIN);
            for op in &ops[start..end] {
                if op.opcode == Opcode::Barrier || op.cycle < 0 {
                    continue;
                }
                let lanes = shifted.get(&op.id).map_or(&[][..], |l| l.as_slice());
                // A multiply-add select writes its result one cycle after its subtract.
                let fma_write = config.select_fma_ops.contains(&op.id).then_some(op.cycle + 1);
                for c in std::iter::once(op.cycle).chain(lanes.iter().copied()).chain(fma_write) {
                    range = (range.0.min(c), range.1.max(c));
                }
            }
            range
        })
        .collect();

    // Live words at every half-cycle point (read phase 2c, write phase 2c + 1).
    let cycle_timeline = scratch_time == crate::cpsat_scheduler::ScratchTime::Cycle;
    let n_points = spans.values().map(|&(_, hi)| (hi | 1) + 2).max().unwrap_or(0).max(0) as usize;
    let mut delta: Vec<i64> = vec![0; n_points + 1];
    for (vreg, &(lo, hi)) in &spans {
        let (lo, hi) = if cycle_timeline { (lo - lo.rem_euclid(2), hi | 1) } else { (lo, hi) };
        if lo < 0 {
            continue;
        }
        delta[lo as usize] += vreg.size() as i64;
        delta[(hi + 1) as usize] -= vreg.size() as i64;
    }
    let mut curve: Vec<usize> = Vec::with_capacity(n_points);
    let mut live = 0i64;
    for d in delta.iter().take(n_points) {
        live += d;
        curve.push(live.max(0) as usize);
    }

    let mut section_peaks = Vec::with_capacity(boundaries.len());
    let mut section_exceeds_budget = Vec::with_capacity(boundaries.len());
    for k in 0..boundaries.len() {
        let (min_c, max_c) = section_ranges[k];
        // Sample through the gap after this section, up to the next section's first cycle.
        let end_c = section_ranges
            .get(k + 1)
            .and_then(|&(next_min, _)| (next_min != i32::MAX && next_min > max_c).then_some(next_min - 1))
            .unwrap_or(max_c);
        let peak = if min_c <= end_c {
            let lo = (2 * min_c as i64).max(0) as usize;
            let hi = (2 * end_c as i64 + 1) as usize;
            curve.get(lo..=hi.min(curve.len().saturating_sub(1))).map_or(0, |s| s.iter().copied().max().unwrap_or(0))
        } else {
            0
        };
        section_peaks.push(peak);
        section_exceeds_budget.push(peak > scratch_budget);
    }

    Ok(SectionBudgetResult { section_peaks, section_exceeds_budget })
}

/// VReg -> section index that defines it (from full ops and section boundaries).
pub fn vreg_def_section(ops: &[SsaOp], boundaries: &[(usize, usize)]) -> HashMap<VReg, usize> {
    let mut out = HashMap::new();
    for (sec, &(start, end)) in boundaries.iter().enumerate() {
        for op in &ops[start..end] {
            if op.opcode == Opcode::Barrier {
                continue;
            }
            if let Some(dest) = op.dest {
                out.insert(dest, sec);
            }
        }
    }
    out
}

/// Live-in VRegs for a section: defined in an earlier section, read in this section.
/// Returns (VReg, segment_last_use_idx): segment index (0-based within non-barrier ops of this section)
/// of the op that last uses this VReg (by greedy cycle).
pub fn live_in_for_section(
    ops: &[SsaOp],
    boundaries: &[(usize, usize)],
    section_k: usize,
    vreg_def_section: &HashMap<VReg, usize>,
) -> Vec<(VReg, usize)> {
    let (start, end) = match boundaries.get(section_k) {
        Some(&b) => b,
        None => return vec![],
    };

    // Segment = non-barrier ops in this section; segment index = order in (start..end) skipping barriers
    let global_to_segment: Vec<usize> = (start..end)
        .filter(|&i| ops[i].opcode != Opcode::Barrier)
        .collect();

    live_in_for_section_filtered(ops, boundaries, section_k, vreg_def_section, &global_to_segment)
}

/// Live-in VRegs for a section when the segment is filtered (e.g. AddImm alternatives removed).
/// Returns (VReg, filtered_segment_last_use_idx): index in `filtered_global_indices` of the op
/// that last uses this VReg (by greedy cycle). Use this so CP-SAT's cycle_vars indices match.
pub fn live_in_for_section_filtered(
    ops: &[SsaOp],
    boundaries: &[(usize, usize)],
    section_k: usize,
    vreg_def_section: &HashMap<VReg, usize>,
    filtered_global_indices: &[usize],
) -> Vec<(VReg, usize)> {
    ScratchAliases::default().live_in(ops, boundaries, section_k, vreg_def_section, filtered_global_indices)
}

impl ScratchAliases {
    pub(crate) fn live_in(
        &self,
        ops: &[SsaOp],
        boundaries: &[(usize, usize)],
        section_k: usize,
        vreg_def_section: &HashMap<VReg, usize>,
        filtered_global_indices: &[usize],
    ) -> Vec<(VReg, usize)> {
        let (_start, _end) = match boundaries.get(section_k) {
            Some(&b) => b,
            None => return vec![],
        };

        // VReg -> (filtered_seg_idx, cycle) for last use in this section (only among filtered ops)
        let mut last_use: HashMap<VReg, (usize, i32)> = HashMap::new();
        for (filtered_idx, &global_idx) in filtered_global_indices.iter().enumerate() {
            if global_idx >= ops.len() {
                continue;
            }
            let op = &ops[global_idx];
            if op.opcode == Opcode::Barrier {
                continue;
            }
            let c = op.cycle;
            if c < 0 {
                continue;
            }
            for vreg in self.reads(op) {
                let def_sec = match vreg_def_section.get(&vreg) {
                    Some(&s) => s,
                    None => continue,
                };
                if def_sec >= section_k {
                    continue;
                }
                let entry = last_use.entry(vreg).or_insert((filtered_idx, c));
                if c > entry.1 {
                    *entry = (filtered_idx, c);
                }
            }
        }

        // Sorted: the per-section CP-SAT model creates one interval per entry in this order, and
        // HashMap order differs between processes (same options, different model).
        let mut live_in: Vec<(VReg, usize)> = last_use
            .into_iter()
            .map(|(vreg, (filtered_idx, _))| (vreg, filtered_idx))
            .collect();
        live_in.sort();
        live_in
    }
}

/// VRegs defined in section `section_k` whose last use is in a later section.
/// For per-section CP-SAT we must treat these as live until the end of the segment so the
/// scratch constraint matches the full-kernel view.
///
/// Uses OP INDEX to determine section membership (not cycle), since ops can be interleaved
/// across sections. An op at index i belongs to section s if boundaries[s].0 <= i < boundaries[s].1.
pub fn live_out_from_section(
    ops: &[SsaOp],
    boundaries: &[(usize, usize)],
    section_k: usize,
    vreg_def_section: &HashMap<VReg, usize>,
) -> HashSet<VReg> {
    ScratchAliases::default().live_after(ops, boundaries, section_k, vreg_def_section).0
}

/// VRegs defined in ANY section before `section_k` (sections 0..section_k-1) whose last use
/// is in a section AFTER `section_k` (sections section_k+1..N).
/// They are live for the entire segment k (carry-through). Per-section CP-SAT for section k
/// must add [0, segment_end] intervals so scratch matches the full-kernel view.
///
/// Uses OP INDEX to determine section membership (not cycle), since ops can be interleaved
/// across sections. An op at index i belongs to section s if boundaries[s].0 <= i < boundaries[s].1.
pub fn live_through_for_section(
    ops: &[SsaOp],
    boundaries: &[(usize, usize)],
    section_k: usize,
    vreg_def_section: &HashMap<VReg, usize>,
) -> HashSet<VReg> {
    ScratchAliases::default().live_after(ops, boundaries, section_k, vreg_def_section).1
}

impl ScratchAliases {
    /// Values needed after the section, defined either in it (live-out) or before it (live-through).
    pub(crate) fn live_after(
        &self,
        ops: &[SsaOp],
        boundaries: &[(usize, usize)],
        section_k: usize,
        defs: &HashMap<VReg, usize>,
    ) -> (HashSet<VReg>, HashSet<VReg>) {
        let mut live_out = HashSet::new();
        let mut live_through = HashSet::new();
        for &(start, end) in boundaries.iter().skip(section_k + 1) {
            for op in &ops[start..end] {
                for value in self.reads(op) {
                    match defs.get(&value) {
                        Some(&sec) if sec == section_k => { live_out.insert(value); }
                        Some(&sec) if sec < section_k => { live_through.insert(value); }
                        _ => {}
                    }
                }
            }
        }
        (live_out, live_through)
    }
}

/// Build cycle hints for a single section in **local** cycle space (0-based).
///
/// Per-round CP-SAT runs on segment ops only and uses cycles 0..makespan locally.
/// Hints must be in that space: for each op in the segment, hint = global_cycle - section_offset.
/// Pass the returned slice as `greedy_hints` to the segment CP-SAT call (one hint per segment op, same order as ops[start..end]).
///
/// * `ops`: Full SSA op list (with barriers) with `cycle` set from greedy or merged schedule.
/// * `boundaries`: From [section_boundaries](super::barrier_elision::section_boundaries).
/// * `section_k`: Section index (0..boundaries.len()).
/// * `section_offset`: Global cycle at which this section starts (sum of previous sections' makespans).
pub fn segment_hints_for_section(
    ops: &[SsaOp],
    boundaries: &[(usize, usize)],
    section_k: usize,
    section_offset: i32,
) -> Vec<i32> {
    let (start, end) = boundaries.get(section_k).copied().unwrap_or((0, 0));
    (start..end)
        .map(|idx| {
            let c = ops[idx].cycle;
            if c >= 0 {
                (c - section_offset).max(0)
            } else {
                c
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Engine, Operand};

    fn make_op_with_cycle(
        id: usize,
        engine: Engine,
        opcode: Opcode,
        dest: Option<VReg>,
        srcs: Vec<Operand>,
        deps: Vec<usize>,
        cycle: i32,
    ) -> SsaOp {
        let mut op = SsaOp::new(id, engine, opcode, dest, srcs);
        op.dep_ids = deps.into_iter().collect();
        op.cycle = cycle;
        op
    }

    fn make_barrier_with_cycle(id: usize, deps: Vec<usize>, cycle: i32) -> SsaOp {
        let mut op = SsaOp::new(id, Engine::Barrier, Opcode::Barrier, None, vec![]);
        op.dep_ids = deps.into_iter().collect();
        op.cycle = cycle;
        op
    }

    /// A chain where each input dies as the next value is written: the cycle count holds two
    /// values at every step, the phase count one (same-cycle reuse), so at a one-vector budget
    /// only `Phase` accepts it.
    #[test]
    fn test_sections_exceeding_budget_phase_reuses_words_read_in_the_cycle() {
        use crate::cpsat_scheduler::ScratchTime;
        let v: Vec<VReg> = (0..3).map(VReg::vector).collect();
        let ops = vec![
            make_op_with_cycle(0, Engine::Valu, Opcode::Vbroadcast, Some(v[0]), vec![], vec![], 0),
            make_op_with_cycle(1, Engine::Valu, Opcode::Add, Some(v[1]), vec![Operand::VReg(v[0]), Operand::VReg(v[0])], vec![0], 1),
            make_op_with_cycle(2, Engine::Valu, Opcode::Add, Some(v[2]), vec![Operand::VReg(v[1]), Operand::VReg(v[1])], vec![1], 2),
            make_op_with_cycle(3, Engine::Store, Opcode::Vstore, None, vec![Operand::Imm(0), Operand::VReg(v[2])], vec![2], 3),
        ];
        let cycle = sections_exceeding_budget_with(&ops, 8, None, None, ScratchTime::Cycle);
        let phase = sections_exceeding_budget_with(&ops, 8, None, None, ScratchTime::Phase);
        assert_eq!(cycle.section_peaks, vec![16]);
        assert_eq!(phase.section_peaks, vec![8]);
        assert_eq!(phase.section_exceeds_budget, vec![false]);
        assert_eq!(sections_exceeding_budget(&ops, 8, None, None).section_peaks, vec![16]);
    }

    /// A SelectChoice on the multiply-add path reads its operands again at cycle + 1: cond,
    /// true and false values (8 words each) are live with the subtract temp in the read phase
    /// of cycle 2, 32 words; the result is written in the write phase of cycle 2, after those
    /// reads, as address assignment counts it. On the vselect path the peak is 24.
    #[test]
    fn test_sections_exceeding_budget_fma_operands_read_at_next_cycle() {
        use crate::cpsat_scheduler::ScratchTime;
        let (c, t, f, d, sub, x) =
            (VReg::vector(1), VReg::vector(2), VReg::vector(3), VReg::vector(4), VReg::vector(5), VReg::vector(6));
        let mut select = make_op_with_cycle(
            3,
            Engine::Valu,
            Opcode::SelectChoice,
            Some(d),
            vec![Operand::VReg(c), Operand::VReg(t), Operand::VReg(f)],
            vec![0, 1, 2],
            1,
        );
        select.sub_temp_vreg = Some(sub);
        let ops = vec![
            make_op_with_cycle(0, Engine::Valu, Opcode::Vbroadcast, Some(c), vec![], vec![], 0),
            make_op_with_cycle(1, Engine::Valu, Opcode::Vbroadcast, Some(t), vec![], vec![], 0),
            make_op_with_cycle(2, Engine::Valu, Opcode::Vbroadcast, Some(f), vec![], vec![], 0),
            select,
            make_op_with_cycle(4, Engine::Valu, Opcode::Add, Some(x), vec![Operand::VReg(d), Operand::VReg(d)], vec![3], 3),
            make_op_with_cycle(5, Engine::Store, Opcode::Vstore, None, vec![Operand::Imm(0), Operand::VReg(x)], vec![4], 4),
        ];
        let fma: HashSet<usize> = [3].into_iter().collect();
        let with_fma = sections_exceeding_budget_with(&ops, 1536, Some(&fma), None, ScratchTime::Phase);
        let vselect = sections_exceeding_budget_with(&ops, 1536, None, None, ScratchTime::Phase);
        assert_eq!(with_fma.section_peaks, vec![32]);
        assert_eq!(vselect.section_peaks, vec![24]);
    }

    #[test]
    fn test_sections_exceeding_budget() {
        // Section 0: op0 (cycle 0) def v0, op1 (cycle 1) reads v0 and def v1 -> at cycle 1 both v0 and v1 live. Peak 2 words.
        // Barrier at cycle 2
        // Section 1: op3 (cycle 3) reads v0,v1 and def v3 -> v0,v1,v3 live at 3. Peak 3 words.
        let v0 = VReg::scalar(0);
        let v1 = VReg::scalar(1);
        let v3 = VReg::scalar(3);
        let ops = vec![
            make_op_with_cycle(0, Engine::Load, Opcode::Const, Some(v0), vec![], vec![], 0),
            make_op_with_cycle(
                1,
                Engine::Alu,
                Opcode::Add,
                Some(v1),
                vec![Operand::VReg(v0), Operand::VReg(v0)],
                vec![0],
                1,
            ),
            make_barrier_with_cycle(2, vec![0, 1], 2),
            make_op_with_cycle(
                3,
                Engine::Alu,
                Opcode::Add,
                Some(v3),
                vec![Operand::VReg(v0), Operand::VReg(v1)],
                vec![0, 1, 2],
                3,
            ),
        ];
        let result = sections_exceeding_budget(&ops, 2, None, None);
        assert_eq!(result.section_peaks.len(), 2);
        assert_eq!(
            result.section_peaks[0], 2,
            "Section 0: at cycle 1 both v0 and v1 live"
        );
        assert_eq!(
            result.section_peaks[1], 3,
            "Section 1: v0, v1, v3 live at cycle 3"
        );
        assert!(
            !result.section_exceeds_budget[0],
            "Section 0 peak 2 <= budget 2"
        );
        assert!(
            result.section_exceeds_budget[1],
            "Section 1 peak 3 > budget 2"
        );
    }

    #[test]
    fn test_sections_exceeding_budget_no_barriers() {
        let v0 = VReg::scalar(0);
        let ops = vec![make_op_with_cycle(
            0,
            Engine::Load,
            Opcode::Const,
            Some(v0),
            vec![],
            vec![],
            0,
        )];
        let result = sections_exceeding_budget(&ops, 1536, None, None);
        assert_eq!(result.section_peaks.len(), 1);
        assert_eq!(result.section_peaks[0], 1);
        assert!(!result.section_exceeds_budget[0]);
    }

    /// SelectChoice with FMA path chosen must count sub_temp_vreg in section peak.
    /// The subtraction temporary is live for one cycle (def at op.cycle, read at op.cycle+1 by the lowered FMA).
    #[test]
    fn test_sections_exceeding_budget_selectchoice_fma_counts_sub_temp() {
        use std::collections::HashSet;

        // One section: one SelectChoice op (id=1) at cycle 10, FMA path chosen.
        // dest = vector (8 words), sub_temp_vreg = vector (8 words), live [10, 11].
        let cond = VReg::vector(0);
        let true_val = VReg::vector(1);
        let false_val = VReg::vector(2);
        let dest = VReg::vector(3);
        let sub_temp = VReg::vector(4);

        let mut op = SsaOp::new(
            1,
            Engine::Flow,
            Opcode::SelectChoice,
            Some(dest),
            vec![
                Operand::VReg(cond),
                Operand::VReg(true_val),
                Operand::VReg(false_val),
            ],
        );
        op.cycle = 10;
        op.sub_temp_vreg = Some(sub_temp); // FMA path without delta: sub_temp is used

        let ops = vec![op];

        // Without passing select_fma_op_ids, sub_temp is not counted -> peak = 8 (dest only).
        let result_no_fma = sections_exceeding_budget(&ops, 1536, None, None);
        assert_eq!(result_no_fma.section_peaks.len(), 1);
        assert_eq!(
            result_no_fma.section_peaks[0], 8,
            "Without FMA selection, only dest (8 words) is counted; sub_temp is ignored"
        );

        // With FMA path chosen for op 1, sub_temp must be counted: live at 10 and 11 (8 words each).
        // At cycle 10: dest defined (8), sub_temp defined (8) -> 16 words.
        let mut select_fma = HashSet::new();
        select_fma.insert(1);

        let result_with_fma = sections_exceeding_budget(&ops, 1536, Some(&select_fma), None);
        assert_eq!(result_with_fma.section_peaks.len(), 1);
        assert_eq!(
            result_with_fma.section_peaks[0], 16,
            "With FMA chosen, dest (8) + sub_temp_vreg (8) must be counted in peak"
        );
    }

    #[test]
    fn test_vreg_def_section_and_live_in() {
        let v0 = VReg::scalar(0);
        let v1 = VReg::scalar(1);
        let v3 = VReg::scalar(3);
        let ops = vec![
            make_op_with_cycle(0, Engine::Load, Opcode::Const, Some(v0), vec![], vec![], 0),
            make_op_with_cycle(
                1,
                Engine::Alu,
                Opcode::Add,
                Some(v1),
                vec![Operand::VReg(v0), Operand::VReg(v0)],
                vec![0],
                1,
            ),
            make_barrier_with_cycle(2, vec![0, 1], 2),
            make_op_with_cycle(
                3,
                Engine::Alu,
                Opcode::Add,
                Some(v3),
                vec![Operand::VReg(v0), Operand::VReg(v1)],
                vec![0, 1, 2],
                3,
            ),
        ];
        let boundaries = section_boundaries(&ops);
        let vds = vreg_def_section(&ops, &boundaries);
        assert_eq!(vds.get(&v0), Some(&0), "v0 defined in section 0");
        assert_eq!(vds.get(&v1), Some(&0), "v1 defined in section 0");
        assert_eq!(vds.get(&v3), Some(&1), "v3 defined in section 1");

        let live_in_1 = live_in_for_section(&ops, &boundaries, 1, &vds);
        // Section 1 reads v0 and v1 (both defined in section 0) -> live-in (v0, seg_idx of last use), (v1, seg_idx of last use)
        // Section 1 has one op (index 3) at segment index 0, and it reads both v0 and v1
        assert_eq!(live_in_1.len(), 2, "v0 and v1 are live-in for section 1");
        let seg_idxs: Vec<usize> = live_in_1.iter().map(|(_, i)| *i).collect();
        assert!(seg_idxs.contains(&0), "last use is segment op 0");
    }

    #[test]
    fn test_section_boundaries_resolve_vector_group_aliases() {
        use crate::kernel_builder::emit_ssa::SsaEmitContext;

        let mut emit = SsaEmitContext::new();
        let vector = emit.emit_vload_imm(0);
        let parts = emit.decompose_vector(vector);
        let addr = emit.emit_const(64);
        let (assembled, lanes) = emit.fresh_vector_group();
        for &lane in &lanes {
            emit.emit_load_into(addr, lane);
        }
        emit.emit_barrier();
        emit.emit_const(7);
        emit.emit_barrier();
        emit.emit_store_imm(0, parts[2]);
        emit.emit_vstore_imm(8, assembled);
        for (i, op) in emit.ops.iter_mut().enumerate() {
            op.cycle = i as i32;
        }
        let ops = &emit.ops;
        let boundaries = section_boundaries(ops);
        let defs = vreg_def_section(ops, &boundaries);
        let aliases = ScratchAliases::new(&emit.vector_group_lanes, |v| defs.contains_key(v));
        let expected: HashSet<VReg> = lanes.iter().copied().chain([vector]).collect();
        assert_eq!(aliases.live_after(ops, &boundaries, 0, &defs).0, expected);
        assert_eq!(aliases.live_after(ops, &boundaries, 1, &defs).1, expected);
        let (first, end) = boundaries[2];
        let filtered: Vec<_> = (first..end).collect();
        let incoming = aliases.live_in(ops, &boundaries, 2, &defs, &filtered);
        let expected_in: HashMap<_,_> = lanes.iter().map(|&v| (v, 1)).chain([(vector, 0)]).collect();
        assert_eq!(incoming.iter().copied().collect::<HashMap<_, _>>(), expected_in);
        let liveness = SectionLiveness { vector_group_lanes: Some(&emit.vector_group_lanes), ..Default::default() };
        let measured = section_budget(ops, 15, None, None, crate::cpsat_scheduler::ScratchTime::Phase, &liveness).unwrap();
        assert!(measured.section_peaks[1] >= 16);
        #[cfg(feature = "cpsat")]
        for budget in [15, 16] {
            use crate::cpsat_scheduler::{schedule_ssa_cpsat, SsaCpsatConfig, SsaCpsatHints};
            let solution = schedule_ssa_cpsat(
                &ops[first..end],
                SsaCpsatConfig { scratch_budget: budget, verbose: false, ..Default::default() },
                SsaCpsatHints {
                    initial_live: Some(&incoming),
                    vector_group_lanes: Some(&emit.vector_group_lanes),
                    ..Default::default()
                },
                Some(false),
            ).expect("solve receiving section");
            assert_eq!(solution.is_some(), budget == 16);
        }
    }

    #[test]
    fn test_segment_hints_for_section() {
        let v0 = VReg::scalar(0);
        let v1 = VReg::scalar(1);
        let ops = vec![
            make_op_with_cycle(0, Engine::Load, Opcode::Const, Some(v0), vec![], vec![], 10),
            make_op_with_cycle(
                1,
                Engine::Alu,
                Opcode::Add,
                Some(v1),
                vec![Operand::VReg(v0), Operand::VReg(v0)],
                vec![0],
                11,
            ),
            make_barrier_with_cycle(2, vec![0, 1], 12),
            make_op_with_cycle(
                3,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(3)),
                vec![Operand::VReg(v0), Operand::VReg(v1)],
                vec![0, 1, 2],
                20,
            ),
        ];
        let boundaries = section_boundaries(&ops);
        assert_eq!(boundaries.len(), 2);
        // Section 0: ops 0, 1 (cycles 10, 11). Offset 0 -> hints [10, 11]
        let h0 = segment_hints_for_section(&ops, &boundaries, 0, 0);
        assert_eq!(h0, vec![10, 11]);
        // Section 1: op 3 (cycle 20). Offset 13 (e.g. section 0 makespan 3) -> hint 20 - 13 = 7
        let h1 = segment_hints_for_section(&ops, &boundaries, 1, 13);
        assert_eq!(h1, vec![7]);
    }

    /// Regression: live_out_from_section identifies VRegs defined in section k but last-used in a later section.
    /// Per-section CP-SAT must extend their scratch interval to segment end so the merged schedule stays within budget.
    #[test]
    fn test_live_out_from_section() {
        let v0 = VReg::scalar(0);
        let v1 = VReg::scalar(1);
        let v3 = VReg::scalar(3);
        // Section 0: op0 def v0 (cycle 0), op1 def v1 (cycle 1), both read in section 1
        // Section 1: op3 reads v0, v1 and def v3 (cycle 3)
        // v0 and v1 are defined in section 0 but last used in section 1 -> live-out from section 0
        // v3 is defined in section 1 and last used in section 1 -> not live-out from section 1
        let ops = vec![
            make_op_with_cycle(0, Engine::Load, Opcode::Const, Some(v0), vec![], vec![], 0),
            make_op_with_cycle(
                1,
                Engine::Alu,
                Opcode::Add,
                Some(v1),
                vec![Operand::VReg(v0), Operand::VReg(v0)],
                vec![0],
                1,
            ),
            make_barrier_with_cycle(2, vec![0, 1], 2),
            make_op_with_cycle(
                3,
                Engine::Alu,
                Opcode::Add,
                Some(v3),
                vec![Operand::VReg(v0), Operand::VReg(v1)],
                vec![0, 1, 2],
                3,
            ),
        ];
        let boundaries = section_boundaries(&ops);
        let vds = vreg_def_section(&ops, &boundaries);
        // Section 0 live-out: v0 and v1 are defined in 0, last use in section 1
        let live_out_0 = live_out_from_section(&ops, &boundaries, 0, &vds);
        assert!(
            live_out_0.contains(&v0),
            "v0 defined in section 0, last use in section 1"
        );
        assert!(
            live_out_0.contains(&v1),
            "v1 defined in section 0, last use in section 1"
        );
        assert_eq!(live_out_0.len(), 2);
        // Section 1 live-out: v3 is defined in 1 and last use in 1 -> empty
        let live_out_1 = live_out_from_section(&ops, &boundaries, 1, &vds);
        assert!(
            live_out_1.is_empty(),
            "v3 last use in section 1, no live-out"
        );
    }

    /// live_through_for_section: VRegs defined in section k-1, last use in section > k (live for whole of section k).
    #[test]
    fn test_live_through_for_section() {
        let v0 = VReg::scalar(0);
        let v1 = VReg::scalar(1);
        let v2 = VReg::scalar(2);
        let v3 = VReg::scalar(3);
        // Section 0: op0 def v0 (cycle 0), op1 def v1 (cycle 1, reads v0)
        // Section 1: op3 def v2 (cycle 3, reads v0)
        // Section 2: op4 def v3 (cycle 5, reads v1, v2)
        // v0: def 0, last use section 1 -> not live-through for section 1
        // v1: def 0, last use section 2 -> live-through for section 1
        // v2: def 1, last use section 2 -> not live-through for section 2 (last use not > 2)
        let ops = vec![
            make_op_with_cycle(0, Engine::Load, Opcode::Const, Some(v0), vec![], vec![], 0),
            make_op_with_cycle(
                1,
                Engine::Alu,
                Opcode::Add,
                Some(v1),
                vec![Operand::VReg(v0), Operand::VReg(v0)],
                vec![0],
                1,
            ),
            make_barrier_with_cycle(2, vec![0, 1], 2),
            make_op_with_cycle(
                3,
                Engine::Alu,
                Opcode::Add,
                Some(v2),
                vec![Operand::VReg(v0), Operand::VReg(v0)],
                vec![0, 1, 2],
                3,
            ),
            make_barrier_with_cycle(4, vec![0, 1, 2, 3], 4),
            make_op_with_cycle(
                5,
                Engine::Alu,
                Opcode::Add,
                Some(v3),
                vec![Operand::VReg(v1), Operand::VReg(v2)],
                vec![1, 2, 3, 4],
                5,
            ),
        ];
        let boundaries = section_boundaries(&ops);
        let vds = vreg_def_section(&ops, &boundaries);
        let live_through_0 = live_through_for_section(&ops, &boundaries, 0, &vds);
        assert!(
            live_through_0.is_empty(),
            "section 0 has no previous section"
        );
        let live_through_1 = live_through_for_section(&ops, &boundaries, 1, &vds);
        assert!(
            live_through_1.contains(&v1),
            "v1 defined in section 0, last use in section 2"
        );
        assert!(
            !live_through_1.contains(&v0),
            "v0 last use in section 1, not > 1"
        );
        assert_eq!(live_through_1.len(), 1);
        let live_through_2 = live_through_for_section(&ops, &boundaries, 2, &vds);
        assert!(
            live_through_2.is_empty(),
            "v2 last use in section 2, so 2 > 2 is false"
        );
    }

    /// Test that live_through_for_section correctly handles VRegs from early sections
    /// that are live-through multiple later sections (the bug fix: def_sec < section_k, not def_sec == prev_sec).
    #[test]
    fn test_live_through_for_section_multi_section_span() {
        let v0 = VReg::scalar(0);
        let v1 = VReg::scalar(1);
        let v2 = VReg::scalar(2);
        let v3 = VReg::scalar(3);
        let v4 = VReg::scalar(4);
        // Section 0: def v0, v1
        // Section 1: def v2 (uses v0)
        // Section 2: def v3 (uses v2)
        // Section 3: def v4 (uses v0, v1) - v0 and v1 are live-through sections 1 AND 2
        let ops = vec![
            make_op_with_cycle(0, Engine::Load, Opcode::Const, Some(v0), vec![], vec![], 0),
            make_op_with_cycle(1, Engine::Load, Opcode::Const, Some(v1), vec![], vec![], 1),
            make_barrier_with_cycle(2, vec![0, 1], 2),
            // Section 1
            make_op_with_cycle(
                3,
                Engine::Alu,
                Opcode::Add,
                Some(v2),
                vec![Operand::VReg(v0), Operand::VReg(v0)],
                vec![0, 2],
                3,
            ),
            make_barrier_with_cycle(4, vec![0, 1, 2, 3], 4),
            // Section 2
            make_op_with_cycle(
                5,
                Engine::Alu,
                Opcode::Add,
                Some(v3),
                vec![Operand::VReg(v2), Operand::VReg(v2)],
                vec![3, 4],
                5,
            ),
            make_barrier_with_cycle(6, vec![0, 1, 3, 4, 5], 6),
            // Section 3: uses v0, v1 from section 0
            make_op_with_cycle(
                7,
                Engine::Alu,
                Opcode::Add,
                Some(v4),
                vec![Operand::VReg(v0), Operand::VReg(v1)],
                vec![0, 1, 5, 6],
                7,
            ),
        ];
        let boundaries = section_boundaries(&ops);
        assert_eq!(boundaries.len(), 4, "Should have 4 sections");
        let vds = vreg_def_section(&ops, &boundaries);

        // Section 1 live-through: v0 and v1 are both defined in section 0, last use in section 3 (> 1)
        // v0 is ALSO used in section 1, but its LAST use is in section 3, so it's still live-through
        let live_through_1 = live_through_for_section(&ops, &boundaries, 1, &vds);
        assert!(
            live_through_1.contains(&v1),
            "v1: def=0, last_use=3 > 1, live-through section 1"
        );
        assert!(
            live_through_1.contains(&v0),
            "v0: def=0, last_use=3 > 1 (also used in 1, but LAST use is 3)"
        );
        assert_eq!(live_through_1.len(), 2);

        // Section 2 live-through: v0 and v1 are both defined in section 0 (< 2), last use in section 3 (> 2)
        // This is the key test case for the bug fix - v0 and v1 should be live-through section 2
        // even though they were defined in section 0 (not section 1 = prev_sec)
        let live_through_2 = live_through_for_section(&ops, &boundaries, 2, &vds);
        assert!(
            live_through_2.contains(&v0),
            "v0: def=0 < 2, last_use=3 > 2, live-through section 2"
        );
        assert!(
            live_through_2.contains(&v1),
            "v1: def=0 < 2, last_use=3 > 2, live-through section 2"
        );
        assert_eq!(live_through_2.len(), 2);

        // Section 3 live-through: nothing, v0 and v1 are last-used in section 3 (not > 3)
        let live_through_3 = live_through_for_section(&ops, &boundaries, 3, &vds);
        assert!(
            live_through_3.is_empty(),
            "Section 3 has no live-through (all last uses in section 3)"
        );
    }

    /// Test that live_out VRegs (defined in section, used after) are counted correctly in peak calculation.
    /// This validates that sections_exceeding_budget correctly accounts for VRegs that extend beyond
    /// the section's cycle range.
    #[test]
    fn test_live_out_vregs_counted_in_peak() {
        // Create 3 sections: section 1 defines VReg A (only used in section 2)
        let vreg_a = VReg::scalar(100);
        let vreg_1 = VReg::scalar(1);
        let vreg_3 = VReg::scalar(3);

        // Section 0: some ops
        let ops = vec![
            make_op_with_cycle(0, Engine::Alu, Opcode::Add, Some(vreg_1), vec![], vec![], 0),
            // Barrier between section 0 and 1
            make_barrier_with_cycle(10, vec![0], 5),
            // Section 1: defines A at cycle 10
            make_op_with_cycle(1, Engine::Alu, Opcode::Add, Some(vreg_a), vec![], vec![10], 10),
            // Barrier between section 1 and 2
            make_barrier_with_cycle(11, vec![1], 15),
            // Section 2: uses A at cycle 20
            make_op_with_cycle(
                2,
                Engine::Alu,
                Opcode::Add,
                Some(vreg_3),
                vec![Operand::VReg(vreg_a)],
                vec![1, 11],
                20,
            ),
        ];

        // Check section 1's peak: A should be counted as live throughout section 1 (cycle 10)
        // even though it's only used in section 2 (cycle 20)
        let result = sections_exceeding_budget(&ops, 1000, None, None);
        assert_eq!(result.section_peaks.len(), 3);
        // Section 1 should have peak >= vreg_a.size() (A is live at cycle 10)
        assert!(
            result.section_peaks[1] >= vreg_a.size(),
            "Section 1 peak should include live_out VReg A"
        );
    }

    /// Test that sections_exceeding_budget correctly handles VRegs defined and used within a section.
    /// This is a sanity check that regular VRegs (not live_out) are counted correctly.
    #[test]
    fn test_regular_vregs_counted_in_peak() {
        // Section 0: defines VReg A at cycle 5, uses it at cycle 10
        let vreg_a = VReg::scalar(100);
        let vreg_200 = VReg::scalar(200);

        let ops = vec![
            make_op_with_cycle(0, Engine::Alu, Opcode::Add, Some(vreg_a), vec![], vec![], 5),
            make_op_with_cycle(
                1,
                Engine::Alu,
                Opcode::Add,
                Some(vreg_200),
                vec![Operand::VReg(vreg_a)],
                vec![0],
                10,
            ),
        ];

        let result = sections_exceeding_budget(&ops, 1000, None, None);
        assert_eq!(result.section_peaks.len(), 1);
        // Peak should be at least vreg_a.size() (A is live from cycle 5 to 10)
        assert!(
            result.section_peaks[0] >= vreg_a.size(),
            "Section 0 peak should include VReg A"
        );
    }
}
