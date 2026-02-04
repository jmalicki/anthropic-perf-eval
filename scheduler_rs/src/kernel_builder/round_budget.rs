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
/// Uses the same liveness model as the greedy scheduler / address assignment:
/// a VReg is live at cycle t if def_cycle <= t <= last_use_cycle (closed interval).
/// Ops with cycle < 0 (e.g. dead stores) are skipped for def/last_use and for
/// section cycle range.
///
/// * `ops`: Full SSA op list with barriers; each op must have `cycle` set (from greedy).
/// * `scratch_budget`: Max allowed simultaneous live words.
/// * `select_fma_op_ids`: When provided (e.g. greedy or CP-SAT SelectChoice decisions), any
///   SelectChoice op whose id is in this set and that has `sub_temp_vreg` is treated as using
///   the FMA path; the sub_temp VReg is then counted in peak (def at op.cycle, last use at
///   op.cycle+1). Pass `None` when no provisional selection is available.
/// * ConstChoice ops: each is a single op with one dest VReg. Scratch usage is one word
///   regardless of whether we later lower to Load or AddImm, so no extra parameter or
///   logic is needed (unlike SelectChoice's sub_temp_vreg when FMA is chosen).
pub fn sections_exceeding_budget(
    ops: &[SsaOp],
    scratch_budget: usize,
    select_fma_op_ids: Option<&HashSet<usize>>,
) -> SectionBudgetResult {
    let boundaries = section_boundaries(ops);

    // VReg -> defining op index (for cycle); only ops with cycle >= 0
    let mut vreg_def_cycle: HashMap<VReg, i32> = HashMap::new();
    for op in ops.iter() {
        if op.cycle >= 0 {
            if let Some(dest) = op.dest {
                vreg_def_cycle.insert(dest, op.cycle);
            }
        }
    }

    // VReg -> last use cycle (max cycle among ops that read it); skip ops with cycle < 0
    let mut vreg_last_use_cycle: HashMap<VReg, i32> = HashMap::new();
    for op in ops.iter() {
        if op.cycle < 0 {
            continue;
        }
        let c = op.cycle;
        for vreg in op.reads() {
            vreg_last_use_cycle
                .entry(vreg)
                .and_modify(|m| *m = (*m).max(c))
                .or_insert(c);
        }
    }

    // When FMA path is provisionally selected for a SelectChoice, count its sub_temp_vreg
    // (def at op.cycle, read by lowered FMA at op.cycle+1).
    if let Some(fma_ids) = select_fma_op_ids {
        for op in ops.iter() {
            if op.cycle >= 0 && fma_ids.contains(&op.id) {
                if let Some(sub_temp) = op.sub_temp_vreg {
                    vreg_def_cycle.insert(sub_temp, op.cycle);
                    vreg_last_use_cycle.insert(sub_temp, op.cycle + 1);
                }
            }
        }
    }

    let mut section_peaks = Vec::with_capacity(boundaries.len());
    let mut section_exceeds_budget = Vec::with_capacity(boundaries.len());

    for &(start, end) in &boundaries {
        // Cycle range for this section: min and max cycle among non-barrier ops with cycle >= 0
        let mut min_c = i32::MAX;
        let mut max_c = i32::MIN;
        for op in &ops[start..end] {
            if op.opcode == Opcode::Barrier {
                continue;
            }
            if op.cycle >= 0 {
                min_c = min_c.min(op.cycle);
                max_c = max_c.max(op.cycle);
            }
        }

        let peak = if min_c <= max_c {
            let mut peak_words = 0usize;
            for t in min_c..=max_c {
                let mut live = 0usize;
                for (vreg, &def_c) in &vreg_def_cycle {
                    let last_c = vreg_last_use_cycle.get(vreg).copied().unwrap_or(def_c);
                    if def_c <= t && t <= last_c {
                        live += vreg.size();
                    }
                }
                peak_words = peak_words.max(live);
            }
            peak_words
        } else {
            0
        };

        section_peaks.push(peak);
        section_exceeds_budget.push(peak > scratch_budget);
    }

    SectionBudgetResult {
        section_peaks,
        section_exceeds_budget,
    }
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

    // VReg -> (segment_idx, cycle) for last use in this section
    let mut last_use: HashMap<VReg, (usize, i32)> = HashMap::new();
    for (seg_idx, &global_idx) in global_to_segment.iter().enumerate() {
        let op = &ops[global_idx];
        let c = op.cycle;
        if c < 0 {
            continue;
        }
        for vreg in op.reads() {
            let def_sec = match vreg_def_section.get(&vreg) {
                Some(&s) => s,
                None => continue,
            };
            if def_sec >= section_k {
                continue; // defined in this or later section, not live-in
            }
            let entry = last_use.entry(vreg).or_insert((seg_idx, c));
            if c > entry.1 {
                *entry = (seg_idx, c);
            }
        }
    }

    last_use
        .into_iter()
        .map(|(vreg, (seg_idx, _))| (vreg, seg_idx))
        .collect()
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
        for vreg in op.reads() {
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

    last_use
        .into_iter()
        .map(|(vreg, (filtered_idx, _))| (vreg, filtered_idx))
        .collect()
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
    // Build a map from vreg -> section of last use (by op index, not cycle)
    let mut vreg_last_use_section: HashMap<VReg, usize> = HashMap::new();
    for (sec_idx, &(start, end)) in boundaries.iter().enumerate() {
        for op in &ops[start..end] {
            if op.opcode == Opcode::Barrier {
                continue;
            }
            for vreg in op.reads() {
                vreg_last_use_section
                    .entry(vreg)
                    .and_modify(|s| *s = (*s).max(sec_idx))
                    .or_insert(sec_idx);
            }
        }
    }

    let mut out = HashSet::new();
    for (vreg, &def_sec) in vreg_def_section {
        if def_sec != section_k {
            continue;
        }
        let last_use_section = match vreg_last_use_section.get(vreg) {
            Some(&s) => s,
            None => continue,
        };
        if last_use_section > section_k {
            out.insert(*vreg);
        }
    }
    out
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
    if section_k == 0 {
        return HashSet::new();
    }

    // Build a map from vreg -> section of last use (by op index, not cycle)
    // This correctly handles interleaved ops where cycles don't correspond to sections
    let mut vreg_last_use_section: HashMap<VReg, usize> = HashMap::new();
    for (sec_idx, &(start, end)) in boundaries.iter().enumerate() {
        for op in &ops[start..end] {
            if op.opcode == Opcode::Barrier {
                continue;
            }
            for vreg in op.reads() {
                // Update last use section to the max section index where vreg is used
                vreg_last_use_section
                    .entry(vreg)
                    .and_modify(|s| *s = (*s).max(sec_idx))
                    .or_insert(sec_idx);
            }
        }
    }

    let mut out = HashSet::new();
    for (vreg, &def_sec) in vreg_def_section {
        // VReg must be defined BEFORE section_k (in any earlier section)
        if def_sec >= section_k {
            continue;
        }
        let last_use_section = match vreg_last_use_section.get(vreg) {
            Some(&s) => s,
            None => continue,
        };
        // VReg is live-through if its last use is in a section AFTER section_k
        if last_use_section > section_k {
            out.insert(*vreg);
        }
    }
    out
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
        let result = sections_exceeding_budget(&ops, 2, None);
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
        let result = sections_exceeding_budget(&ops, 1536, None);
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
        let result_no_fma = sections_exceeding_budget(&ops, 1536, None);
        assert_eq!(result_no_fma.section_peaks.len(), 1);
        assert_eq!(
            result_no_fma.section_peaks[0], 8,
            "Without FMA selection, only dest (8 words) is counted; sub_temp is ignored"
        );

        // With FMA path chosen for op 1, sub_temp must be counted: live at 10 and 11 (8 words each).
        // At cycle 10: dest defined (8), sub_temp defined (8) -> 16 words.
        let mut select_fma = HashSet::new();
        select_fma.insert(1);

        let result_with_fma = sections_exceeding_budget(&ops, 1536, Some(&select_fma));
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
}
