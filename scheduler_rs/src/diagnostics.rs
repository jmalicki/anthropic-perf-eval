//! Scheduling diagnostics and analysis.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::types::Op;

/// Diagnostic information collected during scheduling
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SchedulingDiagnostics {
    // Summary counters
    pub total_cycles: usize,
    pub total_ops_scheduled: usize,

    // Ready but blocked (slot was full)
    pub cycles_with_blocked_ops: usize,
    pub total_blocked_ops: usize,
    pub blocked_by_engine: HashMap<String, usize>,

    // "Almost ready" ops (waiting on 1-2 dependencies)
    pub cycles_with_almost_ready: usize,
    pub total_almost_ready_ops: usize,
    pub almost_ready_by_engine: HashMap<String, usize>,

    // Priority inversions (high-slack op scheduled while low-slack waited)
    pub priority_inversions: usize,
    pub inversion_slack_delta_total: i32,

    // Consumer urgency violations
    pub consumer_urgency_violations: usize,

    // Slack distribution
    pub slack_zero: usize,   // Critical path (slack = 0)
    pub slack_low: usize,    // Slack 1-3
    pub slack_medium: usize, // Slack 4-10
    pub slack_high: usize,   // Slack > 10

    // VALU→ALU shifting stats
    pub valu_shifted: usize,
    pub valu_slot_available: usize,
    pub valu_not_shiftable: usize,
    pub valu_alu_no_capacity: usize,
    pub cycles_valu_saturated: usize,

    // Broadcast elimination stats
    pub broadcasts_eliminated: usize,
    pub broadcast_forwards: usize,
}

impl SchedulingDiagnostics {
    pub fn new() -> Self {
        let mut diag = Self::default();
        // Initialize engine maps
        for engine in ["alu", "valu", "load", "store", "flow"] {
            diag.blocked_by_engine.insert(engine.to_string(), 0);
            diag.almost_ready_by_engine.insert(engine.to_string(), 0);
        }
        diag
    }

    pub fn add_slack_sample(&mut self, slack: i32) {
        if slack == 0 {
            self.slack_zero += 1;
        } else if slack <= 3 {
            self.slack_low += 1;
        } else if slack <= 10 {
            self.slack_medium += 1;
        } else {
            self.slack_high += 1;
        }
    }

    pub fn record_blocked(&mut self, engine: &str) {
        self.total_blocked_ops += 1;
        *self
            .blocked_by_engine
            .entry(engine.to_string())
            .or_insert(0) += 1;
    }

    pub fn record_almost_ready(&mut self, engine: &str) {
        self.total_almost_ready_ops += 1;
        *self
            .almost_ready_by_engine
            .entry(engine.to_string())
            .or_insert(0) += 1;
    }

    /// Generate a human-readable summary
    pub fn summary(&self) -> String {
        let mut lines = vec![
            "=== SCHEDULING DIAGNOSTICS ===".to_string(),
            format!("Total cycles: {}", self.total_cycles),
            format!("Total ops scheduled: {}", self.total_ops_scheduled),
            String::new(),
            "--- Blocked Ops (slot was full) ---".to_string(),
            format!("Cycles with blocked ops: {}", self.cycles_with_blocked_ops),
            format!("Total ops blocked: {}", self.total_blocked_ops),
        ];

        if self.total_blocked_ops > 0 {
            lines.push("Blocked by engine:".to_string());
            for (engine, count) in &self.blocked_by_engine {
                if *count > 0 {
                    lines.push(format!("  {}: {}", engine, count));
                }
            }
        }

        lines.push(String::new());
        lines.push("--- Almost Ready Ops (1-2 deps remaining) ---".to_string());
        lines.push(format!(
            "Cycles with almost-ready ops: {}",
            self.cycles_with_almost_ready
        ));
        lines.push(format!(
            "Total almost-ready ops: {}",
            self.total_almost_ready_ops
        ));
        if self.total_almost_ready_ops > 0 {
            lines.push("Almost-ready by engine:".to_string());
            for (engine, count) in &self.almost_ready_by_engine {
                if *count > 0 {
                    lines.push(format!("  {}: {}", engine, count));
                }
            }
        }

        lines.push(String::new());
        lines.push("--- Priority Analysis ---".to_string());
        lines.push(format!("Priority inversions: {}", self.priority_inversions));
        if self.priority_inversions > 0 {
            let avg_delta =
                self.inversion_slack_delta_total as f64 / self.priority_inversions as f64;
            lines.push(format!("Avg slack delta in inversions: {:.1}", avg_delta));
        }
        lines.push(format!(
            "Consumer urgency violations: {}",
            self.consumer_urgency_violations
        ));

        lines.push(String::new());
        lines.push("--- Slack Distribution ---".to_string());
        lines.push(format!("  zero (critical): {}", self.slack_zero));
        lines.push(format!("  low (1-3): {}", self.slack_low));
        lines.push(format!("  medium (4-10): {}", self.slack_medium));
        lines.push(format!("  high (>10): {}", self.slack_high));

        if self.valu_shifted > 0 || self.valu_alu_no_capacity > 0 {
            lines.push(String::new());
            lines.push("--- VALU→ALU Shifting ---".to_string());
            lines.push(format!("  Shifted to ALU: {}", self.valu_shifted));
            lines.push(format!("  VALU slot available: {}", self.valu_slot_available));
            lines.push(format!("  Not shiftable: {}", self.valu_not_shiftable));
            lines.push(format!("  ALU no capacity: {}", self.valu_alu_no_capacity));
            lines.push(format!(
                "  Cycles VALU saturated: {}",
                self.cycles_valu_saturated
            ));
        }

        lines.join("\n")
    }

    /// Serialize to JSON
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }
}

/// Analysis of shared temp dependencies
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SharedTempAnalysis {
    pub total_deps: usize,
    pub shared_temp_deps: usize,
    pub other_deps: usize,
    pub shared_temp_pct: f64,
    pub hot_addresses_count: usize,
    pub top_problematic: Vec<(i64, usize)>, // (address, dep_count)
}

impl SharedTempAnalysis {
    pub fn summary(&self) -> String {
        let mut lines = vec![
            "=== SHARED TEMP DEPENDENCY ANALYSIS ===".to_string(),
            format!("Total dependencies: {}", self.total_deps),
            format!(
                "Dependencies through shared temps: {} ({:.1}%)",
                self.shared_temp_deps, self.shared_temp_pct
            ),
            format!("Other dependencies: {}", self.other_deps),
            format!(
                "Hot addresses (written >10 times): {}",
                self.hot_addresses_count
            ),
        ];

        if !self.top_problematic.is_empty() {
            lines.push(String::new());
            lines.push("Top problematic shared temp addresses:".to_string());
            for (addr, count) in &self.top_problematic {
                lines.push(format!("  Address {}: {} dependencies", addr, count));
            }
        }

        lines.push(String::new());
        if self.shared_temp_pct > 10.0 {
            lines.push(
                "WARNING: High percentage of dependencies through shared temps!".to_string(),
            );
            lines.push(
                "         Per-group temps may significantly improve parallelism.".to_string(),
            );
        } else if self.shared_temp_pct > 5.0 {
            lines.push("NOTE: Moderate shared temp dependency percentage.".to_string());
            lines.push("      Per-group temps may provide some benefit.".to_string());
        } else {
            lines.push("INFO: Low shared temp dependency percentage.".to_string());
            lines.push("      Shared temps are not a major bottleneck.".to_string());
        }

        lines.join("\n")
    }
}

/// Analyze shared temp dependencies in ops
pub fn analyze_shared_temp_dependencies(ops: &[Op]) -> SharedTempAnalysis {
    // Count writes per address to identify shared temps
    let mut write_counts: HashMap<i64, usize> = HashMap::new();
    for op in ops {
        for &addr in &op.writes {
            *write_counts.entry(addr).or_insert(0) += 1;
        }
    }

    // Identify "hot" addresses - those written many times (likely shared temps)
    let hot_threshold = std::cmp::max(10, ops.len() / 100);
    let hot_addresses: HashSet<i64> = write_counts
        .iter()
        .filter(|(_, &count)| count >= hot_threshold)
        .map(|(&addr, _)| addr)
        .collect();

    // Build op lookup
    let op_by_id: HashMap<usize, &Op> = ops.iter().map(|op| (op.id, op)).collect();

    // Analyze dependencies
    let mut total_deps = 0;
    let mut shared_temp_deps = 0;
    let mut dep_through_address: HashMap<i64, usize> = HashMap::new();

    for op in ops {
        for &dep_id in &op.dep_ids {
            total_deps += 1;

            if let Some(dep_op) = op_by_id.get(&dep_id) {
                // Check if dependency is through a shared temp
                let shared_addrs: Vec<i64> = dep_op
                    .writes
                    .iter()
                    .filter(|addr| op.reads.contains(addr) && hot_addresses.contains(addr))
                    .copied()
                    .collect();

                if !shared_addrs.is_empty() {
                    shared_temp_deps += 1;
                    for addr in shared_addrs {
                        *dep_through_address.entry(addr).or_insert(0) += 1;
                    }
                }
            }
        }
    }

    // Find most problematic addresses
    let mut sorted_deps: Vec<(i64, usize)> = dep_through_address.into_iter().collect();
    sorted_deps.sort_by(|a, b| b.1.cmp(&a.1));
    let top_problematic: Vec<(i64, usize)> = sorted_deps.into_iter().take(10).collect();

    SharedTempAnalysis {
        total_deps,
        shared_temp_deps,
        other_deps: total_deps - shared_temp_deps,
        shared_temp_pct: if total_deps > 0 {
            (shared_temp_deps as f64 / total_deps as f64) * 100.0
        } else {
            0.0
        },
        hot_addresses_count: hot_addresses.len(),
        top_problematic,
    }
}
