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

use std::collections::HashMap;

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
pub fn sections_exceeding_budget(ops: &[SsaOp], scratch_budget: usize) -> SectionBudgetResult {
    let boundaries = section_boundaries(ops);

    // VReg -> defining op index (for cycle); only ops with cycle >= 0
    let mut vreg_def_cycle: HashMap<VReg, i32> = HashMap::new();
    for (idx, op) in ops.iter().enumerate() {
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

    let mut section_peaks = Vec::with_capacity(boundaries.len());
    let mut section_exceeds_budget = Vec::with_capacity(boundaries.len());

    for &(start, end) in &boundaries {
        // Cycle range for this section: min and max cycle among non-barrier ops with cycle >= 0
        let mut min_c = i32::MAX;
        let mut max_c = i32::MIN;
        for idx in start..end {
            let op = &ops[idx];
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
        for idx in start..end {
            let op = &ops[idx];
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
        let result = sections_exceeding_budget(&ops, 2);
        assert_eq!(result.section_peaks.len(), 2);
        assert_eq!(result.section_peaks[0], 2, "Section 0: at cycle 1 both v0 and v1 live");
        assert_eq!(result.section_peaks[1], 3, "Section 1: v0, v1, v3 live at cycle 3");
        assert!(!result.section_exceeds_budget[0], "Section 0 peak 2 <= budget 2");
        assert!(result.section_exceeds_budget[1], "Section 1 peak 3 > budget 2");
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
        let result = sections_exceeding_budget(&ops, 1536);
        assert_eq!(result.section_peaks.len(), 1);
        assert_eq!(result.section_peaks[0], 1);
        assert!(!result.section_exceeds_budget[0]);
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
}
