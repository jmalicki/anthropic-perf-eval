//! Canonical symmetry ordering: single source of truth for ordering groups.
//!
//! This module consumes symmetry info already set on ops at emission (`symmetry_hint.ordering_group`).
//! It does not infer or add new symmetry. Building the ordering is only valid when the pipeline
//! has been run with emission that attaches those hints. See docs/ASAP_ALAP_SLOT_TIGHTENING.md
//! and the plan "Symmetry as its own construct".

use std::collections::{BTreeMap, HashMap};

use crate::types::SsaOp;

/// Canonical ordering of ops by symmetry group: group key → sorted op indices.
///
/// Built once from (ops, asap, alap, greedy_hints). Sort within each group is by
/// (greedy_cycle, ALAP) so the order is bounds-dependent. Used by symmetry-aware
/// tightening and by the CP-SAT model for `cycle[i] <= cycle[j]` constraints.
#[derive(Debug, Clone)]
pub struct SymmetryOrdering {
    /// Group key (from op.symmetry_hint.ordering_group) → sorted list of op indices.
    /// Only ops with non-empty ordering_group are included; ops without hints are not in any group.
    /// Ordered map: groups are iterated to tighten bounds and to add ordering constraints,
    /// and both must happen in the same order on every run.
    groups: BTreeMap<Vec<i64>, Vec<usize>>,
}

impl SymmetryOrdering {
    /// Build the canonical ordering from current bounds.
    ///
    /// Requires that emission has already set `symmetry_hint` with non-empty `ordering_group`
    /// on ops that should participate. Ops without hints are simply not in any group (no error).
    /// Use [Self::validate_has_groups] when symmetry-aware tightening is enabled if you want
    /// to require at least one group to exist.
    pub fn from_bounds(
        ops: &[SsaOp],
        asap: &HashMap<usize, i32>,
        alap: &HashMap<usize, i32>,
        greedy_hints: Option<&[i32]>,
        precedence_integrated: bool,
    ) -> Self {
        let mut groups: BTreeMap<Vec<i64>, Vec<usize>> = BTreeMap::new();
        for (idx, op) in ops.iter().enumerate() {
            let Some(ref hint) = op.symmetry_hint else {
                continue;
            };
            if hint.ordering_group.is_empty() {
                continue;
            }
            groups
                .entry(hint.ordering_group.clone())
                .or_default()
                .push(idx);
        }
        for indices in groups.values_mut() {
            if let Some(hints) = greedy_hints {
                indices.sort_by_key(|&idx| {
                    let greedy = hints.get(idx).copied().unwrap_or(0);
                    let alap_val = *alap.get(&ops[idx].id).unwrap_or(&0);
                    let asap_val = if precedence_integrated {
                        *asap.get(&ops[idx].id).unwrap_or(&0)
                    } else {
                        0
                    };
                    (greedy, alap_val, asap_val, idx)
                });
            } else {
                indices.sort_by_key(|&idx| {
                    let alap_val = *alap.get(&ops[idx].id).unwrap_or(&0);
                    let asap_val = if precedence_integrated {
                        *asap.get(&ops[idx].id).unwrap_or(&0)
                    } else {
                        0
                    };
                    (alap_val, asap_val, idx)
                });
            }
        }
        Self { groups }
    }

    /// Return ordering pairs (i, j) for symmetry breaking: cycle[i] <= cycle[j].
    /// Consecutive pairs within each group in sorted order.
    pub fn ordering_pairs(&self) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        for indices in self.groups.values() {
            for w in indices.windows(2) {
                pairs.push((w[0], w[1]));
            }
        }
        pairs
    }

    /// Iterate over groups: (group_key, sorted op indices).
    /// Used by symmetry-aware tightening to get per-group order and next-op for capping.
    pub fn iter_groups(&self) -> impl Iterator<Item = (&Vec<i64>, &Vec<usize>)> {
        self.groups.iter()
    }

    /// Validate that for every ordering pair (i, j), asap[i] <= alap[j].
    /// Panics with a clear message if not (constraint cycle[i] <= cycle[j] would be unsatisfiable).
    pub fn validate_bounds(
        &self,
        ops: &[SsaOp],
        asap: &HashMap<usize, i32>,
        alap: &HashMap<usize, i32>,
        greedy_hints: Option<&[i32]>,
    ) {
        for (group_key, indices) in &self.groups {
            for w in indices.windows(2) {
                let i = w[0];
                let j = w[1];
                let asap_i = *asap.get(&ops[i].id).unwrap_or(&0);
                let alap_j = *alap.get(&ops[j].id).unwrap_or(&0);
                if asap_i > alap_j {
                    panic!(
                        "Symmetry ordering inconsistency: ordering group {:?} requires cycle[{}] <= cycle[{}], \
                         but ASAP[{}]={} > ALAP[{}]={}. The constraint cycle[i] <= cycle[j] is unsatisfiable. \
                         Greedy hints: i={}, j={}. Fix: greedy or ASAP/ALAP bounds are inconsistent.",
                        group_key, i, j, i, asap_i, j, alap_j,
                        greedy_hints.and_then(|h| h.get(i).copied()).unwrap_or(-1),
                        greedy_hints.and_then(|h| h.get(j).copied()).unwrap_or(-1),
                    );
                }
            }
        }
    }

    /// Return true if at least one group has more than one op (non-trivial ordering).
    #[allow(dead_code)]
    pub fn has_ordering(&self) -> bool {
        self.groups.values().any(|v| v.len() >= 2)
    }
}
