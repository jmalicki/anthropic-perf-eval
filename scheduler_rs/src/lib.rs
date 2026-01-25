use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::cmp::Reverse;

const VLEN: usize = 8;

// =============================================================================
// Scheduling Diagnostics
// =============================================================================

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
    pub slack_zero: usize,    // Critical path (slack = 0)
    pub slack_low: usize,     // Slack 1-3
    pub slack_medium: usize,  // Slack 4-10
    pub slack_high: usize,    // Slack > 10

    // VALU→ALU shifting stats
    pub valu_shifted: usize,
    pub valu_slot_available: usize,
    pub valu_not_shiftable: usize,
    pub valu_alu_no_capacity: usize,
    pub cycles_valu_saturated: usize,
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
        *self.blocked_by_engine.entry(engine.to_string()).or_insert(0) += 1;
    }

    pub fn record_almost_ready(&mut self, engine: &str) {
        self.total_almost_ready_ops += 1;
        *self.almost_ready_by_engine.entry(engine.to_string()).or_insert(0) += 1;
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
        lines.push(format!("Cycles with almost-ready ops: {}", self.cycles_with_almost_ready));
        lines.push(format!("Total almost-ready ops: {}", self.total_almost_ready_ops));
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
            let avg_delta = self.inversion_slack_delta_total as f64 / self.priority_inversions as f64;
            lines.push(format!("Avg slack delta in inversions: {:.1}", avg_delta));
        }
        lines.push(format!("Consumer urgency violations: {}", self.consumer_urgency_violations));

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
            lines.push(format!("  Cycles VALU saturated: {}", self.cycles_valu_saturated));
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
    pub top_problematic: Vec<(i64, usize)>,  // (address, dep_count)
}

impl SharedTempAnalysis {
    pub fn summary(&self) -> String {
        let mut lines = vec![
            "=== SHARED TEMP DEPENDENCY ANALYSIS ===".to_string(),
            format!("Total dependencies: {}", self.total_deps),
            format!("Dependencies through shared temps: {} ({:.1}%)",
                    self.shared_temp_deps, self.shared_temp_pct),
            format!("Other dependencies: {}", self.other_deps),
            format!("Hot addresses (written >10 times): {}", self.hot_addresses_count),
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
            lines.push("WARNING: High percentage of dependencies through shared temps!".to_string());
            lines.push("         Per-group temps may significantly improve parallelism.".to_string());
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
fn analyze_shared_temp_dependencies(ops: &[Op]) -> SharedTempAnalysis {
    // Count writes per address to identify shared temps
    let mut write_counts: HashMap<i64, usize> = HashMap::new();
    for op in ops {
        for &addr in &op.writes {
            *write_counts.entry(addr).or_insert(0) += 1;
        }
    }

    // Identify "hot" addresses - those written many times (likely shared temps)
    let hot_threshold = std::cmp::max(10, ops.len() / 100);
    let hot_addresses: HashSet<i64> = write_counts.iter()
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
                let shared_addrs: Vec<i64> = dep_op.writes.iter()
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
        shared_temp_pct: if total_deps > 0 { (shared_temp_deps as f64 / total_deps as f64) * 100.0 } else { 0.0 },
        hot_addresses_count: hot_addresses.len(),
        top_problematic,
    }
}

// =============================================================================
// Symbolic IR and Constant Analysis
// =============================================================================

/// A symbolic operand that may or may not have a physical register assigned
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SymbolicOperand {
    /// Literal constant value (not yet materialized)
    Const(i64),
    /// Virtual register (will be allocated later)
    Virtual(usize),
    /// Physical register (already allocated or externally defined)
    Physical(i64),
    /// Reference to constant pool entry
    ConstPoolRef(i64),  // value as key
}

/// Tracks usage of each constant value
#[derive(Clone, Debug, Default)]
pub struct ConstantUsage {
    /// How many times this constant is used
    pub use_count: usize,
    /// First cycle where it's needed (ASAP)
    pub first_use_cycle: Option<usize>,
    /// Last cycle where it's needed
    pub last_use_cycle: Option<usize>,
    /// Destinations where this value is stored
    pub destinations: HashSet<i64>,
}

/// Plan for generating a constant value
#[derive(Clone, Debug)]
pub enum ConstantGeneration {
    /// Load directly via const instruction
    DirectLoad,
    /// Generate via XOR with self: dest = x ^ x = 0
    XorSelf { source: i64 },
    /// Generate via addition: dest = a + b
    Add { src1: i64, src2: i64 },
    /// Generate via multiplication: dest = a * b
    Mul { src1: i64, src2: i64 },
    /// Generate via shift: dest = a << n
    Shift { src: i64, amount: u8 },
}

/// A derivation tree node for constants
#[derive(Clone, Debug)]
pub struct ConstantNode {
    pub value: i64,
    pub generation: ConstantGeneration,
    pub dest_register: Option<i64>,  // Assigned after register allocation
    pub depth: usize,                // Depth in derivation tree (0 = bootstrap)
    pub is_bootstrap: bool,          // Must be loaded directly
}

/// Complete constant derivation plan
#[derive(Clone, Debug, Default)]
pub struct ConstantPlan {
    /// Values that must be loaded directly (bootstrap set)
    pub bootstrap_loads: Vec<i64>,
    /// Derivation tree: value -> how to generate it
    pub derivations: HashMap<i64, ConstantNode>,
    /// Order to generate constants (respects dependencies)
    pub generation_order: Vec<i64>,
    /// Values promoted to constant pool (long-lived, frequently used)
    pub constant_pool: HashSet<i64>,
    /// Stats
    pub total_const_loads_saved: usize,
    pub total_alu_ops_added: usize,
}

/// Analyze constant usage across all ops
fn analyze_constant_usage(ops: &[Op]) -> HashMap<i64, ConstantUsage> {
    let mut usage: HashMap<i64, ConstantUsage> = HashMap::new();

    for op in ops {
        // Find const loads
        if op.engine == "load" {
            if let Some(SlotVal::Str(opcode)) = op.slot.get(0) {
                if opcode == "const" {
                    if let (Some(SlotVal::Int(dest)), Some(SlotVal::Int(val))) =
                        (op.slot.get(1), op.slot.get(2))
                    {
                        let entry = usage.entry(*val).or_default();
                        entry.use_count += 1;
                        entry.destinations.insert(*dest);

                        // Track lifecycle (use original cycle from input)
                        let cycle = op.cycle.max(0) as usize;
                        entry.first_use_cycle = Some(
                            entry.first_use_cycle.map_or(cycle, |c| c.min(cycle))
                        );
                        entry.last_use_cycle = Some(
                            entry.last_use_cycle.map_or(cycle, |c| c.max(cycle))
                        );
                    }
                }
            }
        }
    }

    usage
}

/// Check if a value can be derived from available values
fn can_derive(value: i64, available: &HashSet<i64>) -> Option<ConstantGeneration> {
    // Special case: 0 can always be derived via XOR if we have ANY value
    // Prefer using 1 since it's always our primary bootstrap constant
    if value == 0 && !available.is_empty() {
        // Prefer 1, then smallest available value for determinism
        let source = if available.contains(&1) {
            1
        } else {
            *available.iter().min().unwrap()
        };
        return Some(ConstantGeneration::XorSelf { source });
    }

    // Try addition: value = a + b
    for &a in available {
        let b = value - a;
        if available.contains(&b) {
            return Some(ConstantGeneration::Add { src1: a, src2: b });
        }
    }

    // Try multiplication: value = a * b (for values > 1)
    if value > 1 {
        for &a in available {
            if a > 0 && a <= value && value % a == 0 {
                let b = value / a;
                if available.contains(&b) {
                    return Some(ConstantGeneration::Mul { src1: a, src2: b });
                }
            }
        }
    }

    // Try shift: value = a << n (for powers of 2 relationship)
    for &a in available {
        if a > 0 && value > a && value % a == 0 {
            let ratio = value / a;
            // Check if ratio is power of 2
            if ratio > 0 && (ratio & (ratio - 1)) == 0 {
                let shift = (ratio as f64).log2() as u8;
                if shift <= 31 {
                    return Some(ConstantGeneration::Shift { src: a, amount: shift });
                }
            }
        }
    }

    None
}

/// Build optimal constant derivation plan
fn build_constant_plan(usage: &HashMap<i64, ConstantUsage>, threshold_hot: usize) -> ConstantPlan {
    let mut plan = ConstantPlan::default();

    // Collect all needed constants
    let mut needed: Vec<i64> = usage.keys().copied().collect();
    needed.sort();  // Process in order for determinism

    // Identify large constants that MUST be loaded (can't derive)
    let large_threshold = 1_000_000i64;

    // Bootstrap: always need 1 (or smallest positive if no 1)
    let mut bootstrap: HashSet<i64> = HashSet::new();
    let mut available: HashSet<i64> = HashSet::new();

    // Add 1 to bootstrap if needed
    if needed.contains(&1) {
        bootstrap.insert(1);
        available.insert(1);
    } else if let Some(&smallest) = needed.iter().filter(|&&v| v > 0 && v < large_threshold).min() {
        bootstrap.insert(smallest);
        available.insert(smallest);
    }

    // Add all large constants to bootstrap (can't derive)
    for &val in &needed {
        if val.abs() >= large_threshold {
            bootstrap.insert(val);
            available.insert(val);
        }
    }

    // Iteratively build derivation tree
    let mut generation_order: Vec<i64> = Vec::new();
    let mut derivations: HashMap<i64, ConstantNode> = HashMap::new();

    // First, add bootstrap nodes
    for &val in &bootstrap {
        derivations.insert(val, ConstantNode {
            value: val,
            generation: ConstantGeneration::DirectLoad,
            dest_register: None,
            depth: 0,
            is_bootstrap: true,
        });
        generation_order.push(val);
    }

    // Process remaining constants in order of derivability
    let mut remaining: Vec<i64> = needed.iter()
        .filter(|v| !bootstrap.contains(v))
        .copied()
        .collect();

    let mut depth = 1;
    let max_depth = 10;  // Prevent infinite loops

    while !remaining.is_empty() && depth < max_depth {
        let mut derived_this_round: Vec<i64> = Vec::new();

        for &val in &remaining {
            if let Some(gen) = can_derive(val, &available) {
                derivations.insert(val, ConstantNode {
                    value: val,
                    generation: gen,
                    dest_register: None,
                    depth,
                    is_bootstrap: false,
                });
                derived_this_round.push(val);
                available.insert(val);
            }
        }

        if derived_this_round.is_empty() {
            // Can't derive any more - add remaining to bootstrap
            for &val in &remaining {
                bootstrap.insert(val);
                available.insert(val);
                derivations.insert(val, ConstantNode {
                    value: val,
                    generation: ConstantGeneration::DirectLoad,
                    dest_register: None,
                    depth,
                    is_bootstrap: true,
                });
            }
            generation_order.extend(remaining.iter().copied());
            break;
        }

        generation_order.extend(derived_this_round.iter().copied());
        remaining.retain(|v| !derived_this_round.contains(v));
        depth += 1;
    }

    // Identify hot constants for constant pool
    let constant_pool: HashSet<i64> = usage.iter()
        .filter(|(_, u)| u.use_count >= threshold_hot)
        .map(|(&v, _)| v)
        .collect();

    // Calculate savings
    let original_loads = usage.values().map(|u| u.use_count).sum::<usize>();
    let bootstrap_count = bootstrap.len();
    let derived_count = derivations.len() - bootstrap_count;

    plan.bootstrap_loads = bootstrap.into_iter().collect();
    plan.bootstrap_loads.sort();
    plan.derivations = derivations;
    plan.generation_order = generation_order;
    plan.constant_pool = constant_pool;
    plan.total_const_loads_saved = original_loads.saturating_sub(bootstrap_count + derived_count);
    plan.total_alu_ops_added = derived_count;

    plan
}

/// Generate ops for constant derivation
fn emit_constant_generation_ops(
    plan: &ConstantPlan,
    value_to_register: &HashMap<i64, i64>,  // value -> physical register
    start_op_id: usize,
) -> Vec<Op> {
    let mut ops = Vec::new();
    let mut op_id = start_op_id;

    for &value in &plan.generation_order {
        let node = match plan.derivations.get(&value) {
            Some(n) => n,
            None => continue,
        };

        let dest = match value_to_register.get(&value) {
            Some(&r) => r,
            None => continue,
        };

        let (engine, slot) = match &node.generation {
            ConstantGeneration::DirectLoad => {
                ("load".to_string(), vec![
                    SlotVal::Str("const".to_string()),
                    SlotVal::Int(dest),
                    SlotVal::Int(value),
                ])
            }
            ConstantGeneration::XorSelf { source } => {
                let src_reg = value_to_register.get(source).copied().unwrap_or(*source);
                ("alu".to_string(), vec![
                    SlotVal::Str("^".to_string()),
                    SlotVal::Int(dest),
                    SlotVal::Int(src_reg),
                    SlotVal::Int(src_reg),
                ])
            }
            ConstantGeneration::Add { src1, src2 } => {
                let r1 = value_to_register.get(src1).copied().unwrap_or(*src1);
                let r2 = value_to_register.get(src2).copied().unwrap_or(*src2);
                ("alu".to_string(), vec![
                    SlotVal::Str("+".to_string()),
                    SlotVal::Int(dest),
                    SlotVal::Int(r1),
                    SlotVal::Int(r2),
                ])
            }
            ConstantGeneration::Mul { src1, src2 } => {
                let r1 = value_to_register.get(src1).copied().unwrap_or(*src1);
                let r2 = value_to_register.get(src2).copied().unwrap_or(*src2);
                ("alu".to_string(), vec![
                    SlotVal::Str("*".to_string()),
                    SlotVal::Int(dest),
                    SlotVal::Int(r1),
                    SlotVal::Int(r2),
                ])
            }
            ConstantGeneration::Shift { src, amount } => {
                let r1 = value_to_register.get(src).copied().unwrap_or(*src);
                // Need to load the shift amount into a register or use immediate
                // For now, we'll use a simpler approach: a << n = a * (1 << n)
                // This isn't ideal but works - TODO: use shift instruction
                let multiplier = 1i64 << amount;
                if let Some(&mult_reg) = value_to_register.get(&multiplier) {
                    ("alu".to_string(), vec![
                        SlotVal::Str("*".to_string()),
                        SlotVal::Int(dest),
                        SlotVal::Int(r1),
                        SlotVal::Int(mult_reg),
                    ])
                } else {
                    // Fall back to direct load
                    ("load".to_string(), vec![
                        SlotVal::Str("const".to_string()),
                        SlotVal::Int(dest),
                        SlotVal::Int(value),
                    ])
                }
            }
        };

        let mut reads = HashSet::new();
        let mut writes = HashSet::new();
        writes.insert(dest);

        // Set up reads based on generation type
        match &node.generation {
            ConstantGeneration::DirectLoad => {}
            ConstantGeneration::XorSelf { .. } => {
                // XOR self has no dependencies (result is always 0)
            }
            ConstantGeneration::Add { src1, src2 } | ConstantGeneration::Mul { src1, src2 } => {
                if let Some(&r) = value_to_register.get(src1) { reads.insert(r); }
                if let Some(&r) = value_to_register.get(src2) { reads.insert(r); }
            }
            ConstantGeneration::Shift { src, .. } => {
                if let Some(&r) = value_to_register.get(src) { reads.insert(r); }
            }
        }

        ops.push(Op {
            id: op_id,
            cycle: -1,
            engine,
            slot,
            reads,
            writes,
            dep_ids: HashSet::new(),
        });

        op_id += 1;
    }

    ops
}

/// Print constant plan summary
fn print_constant_plan_summary(plan: &ConstantPlan) {
    println!("=== CONSTANT DERIVATION PLAN ===");
    println!("Bootstrap loads needed: {}", plan.bootstrap_loads.len());
    println!("  Values: {:?}", &plan.bootstrap_loads[..plan.bootstrap_loads.len().min(10)]);
    println!("Total derivations: {}", plan.derivations.len());
    println!("ALU ops added: {}", plan.total_alu_ops_added);
    println!("Const loads saved: {}", plan.total_const_loads_saved);
    println!("Constant pool size: {}", plan.constant_pool.len());

    // Show derivation tree sample
    let mut by_depth: HashMap<usize, Vec<i64>> = HashMap::new();
    for (val, node) in &plan.derivations {
        by_depth.entry(node.depth).or_default().push(*val);
    }
    println!("Derivation by depth:");
    for depth in 0..=by_depth.keys().max().copied().unwrap_or(0) {
        if let Some(vals) = by_depth.get(&depth) {
            let display: Vec<_> = vals.iter().take(8).collect();
            let suffix = if vals.len() > 8 { format!("... ({} more)", vals.len() - 8) } else { String::new() };
            println!("  Depth {}: {:?}{}", depth, display, suffix);
        }
    }
}

/// Slot limits per engine per cycle
const SLOT_LIMITS: [(&str, usize); 6] = [
    ("load", 2),
    ("store", 2),
    ("alu", 12),
    ("valu", 6),
    ("flow", 1),
    ("debug", 64),
];

fn get_slot_limit(engine: &str) -> usize {
    for (e, limit) in SLOT_LIMITS.iter() {
        if *e == engine {
            return *limit;
        }
    }
    1
}

/// VALU ops that can be converted to scalar ALU ops (binary operations)
/// NOTE: 'multiply_add' is intentionally EXCLUDED because it's a fused op
const VALU_TO_ALU_OPS: &[&str] = &["+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>", "==", "!=", "<", "<=", ">", ">="];

/// Check if a VALU op can be converted to 8 scalar ALU ops
fn can_convert_valu_to_alu(op: &Op) -> bool {
    if op.engine != "valu" {
        return false;
    }
    if op.slot.is_empty() {
        return false;
    }
    if let SlotVal::Str(op_type) = &op.slot[0] {
        VALU_TO_ALU_OPS.contains(&op_type.as_str())
    } else {
        false
    }
}

/// Expand a VALU op into 8 scalar ALU ops
fn expand_valu_to_alu(op: &Op, next_op_id: usize) -> Vec<Op> {
    // slot format: (op_type, dest, src1, src2)
    let op_type = match &op.slot[0] {
        SlotVal::Str(s) => s.clone(),
        _ => return vec![],
    };
    let dest = match &op.slot[1] {
        SlotVal::Int(n) => *n,
        _ => return vec![],
    };
    let src1 = match &op.slot[2] {
        SlotVal::Int(n) => *n,
        _ => return vec![],
    };
    let src2 = match &op.slot[3] {
        SlotVal::Int(n) => *n,
        _ => return vec![],
    };

    let mut new_ops = Vec::with_capacity(VLEN);
    for i in 0..VLEN {
        let i = i as i64;
        let new_op = Op {
            id: next_op_id + i as usize,
            engine: "alu".to_string(),
            slot: vec![
                SlotVal::Str(op_type.clone()),
                SlotVal::Int(dest + i),
                SlotVal::Int(src1 + i),
                SlotVal::Int(src2 + i),
            ],
            cycle: -1,
            dep_ids: op.dep_ids.clone(),
            reads: [src1 + i, src2 + i].into_iter().collect(),
            writes: [dest + i].into_iter().collect(),
        };
        new_ops.push(new_op);
    }
    new_ops
}

/// Slot value - can be string, int, or other
#[derive(Clone, Debug)]
enum SlotVal {
    Str(String),
    Int(i64),
}

/// Operation with its metadata
#[derive(Clone, Debug)]
struct Op {
    id: usize,
    engine: String,
    slot: Vec<SlotVal>,
    cycle: i32,
    dep_ids: HashSet<usize>,
    reads: HashSet<i64>,
    writes: HashSet<i64>,
}

/// Result of scheduling
struct ScheduleResult {
    cycles: usize,
    #[allow(dead_code)]
    assignments: Vec<(usize, usize)>,
    diagnostics: Option<SchedulingDiagnostics>,
    /// Expanded ALU ops from VALU→ALU shifting (these replace shifted VALU ops)
    expanded_ops: Vec<Op>,
}

/// Compute critical path length for each op (longest path to any sink)
fn compute_critical_path(ops: &[Op]) -> HashMap<usize, usize> {
    let mut critical_path: HashMap<usize, usize> = HashMap::new();

    // Build successor lists
    let mut successors: HashMap<usize, Vec<usize>> = HashMap::new();
    for op in ops {
        successors.insert(op.id, Vec::new());
    }
    for op in ops {
        for &dep_id in &op.dep_ids {
            if let Some(succ) = successors.get_mut(&dep_id) {
                succ.push(op.id);
            }
        }
    }

    // Start from sinks (ops with no successors)
    let mut work_queue: VecDeque<usize> = VecDeque::new();
    for op in ops {
        if successors.get(&op.id).map(|s| s.is_empty()).unwrap_or(true) {
            work_queue.push_back(op.id);
            critical_path.insert(op.id, 1);
        }
    }

    // Build out-degree for reverse topo processing
    let mut out_degree: HashMap<usize, usize> = HashMap::new();
    for op in ops {
        out_degree.insert(op.id, successors.get(&op.id).map(|s| s.len()).unwrap_or(0));
    }

    // Build op lookup
    let op_by_id: HashMap<usize, &Op> = ops.iter().map(|op| (op.id, op)).collect();

    // Process in reverse order
    let mut remaining_out = out_degree.clone();
    while let Some(op_id) = work_queue.pop_front() {
        let my_cp = *critical_path.get(&op_id).unwrap_or(&1);

        if let Some(op) = op_by_id.get(&op_id) {
            for &dep_id in &op.dep_ids {
                let new_cp = my_cp + 1;
                let current = critical_path.entry(dep_id).or_insert(0);
                if new_cp > *current {
                    *current = new_cp;
                }

                if let Some(rem) = remaining_out.get_mut(&dep_id) {
                    *rem -= 1;
                    if *rem == 0 {
                        work_queue.push_back(dep_id);
                    }
                }
            }
        }
    }

    critical_path
}

/// Priority mode for scheduling
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PriorityMode {
    CriticalPath,       // Prioritize ops on critical path
    CriticalPathValu,   // Prioritize VALU first, then critical path
    EngineBalance,      // Balance engine utilization
    Slack,              // Prioritize low-slack ops (standard compiler scheduling)
    LoadEarly,          // Prioritize loads
    LoadEarlySlack,     // Combine load-early with slack
    OriginalOrder,      // Schedule in original emission order
}

const ALL_PRIORITY_MODES: [PriorityMode; 7] = [
    PriorityMode::CriticalPath,
    PriorityMode::CriticalPathValu,
    PriorityMode::EngineBalance,
    PriorityMode::Slack,
    PriorityMode::LoadEarly,
    PriorityMode::LoadEarlySlack,
    PriorityMode::OriginalOrder,
];

/// Priority for scheduling (lower tuple = higher priority in min-heap with Reverse)
fn compute_priority(
    op_id: usize,
    op_idx: usize,
    engine: &str,
    critical_path: &HashMap<usize, usize>,
    slack_map: &HashMap<usize, i32>,
    mode: PriorityMode,
) -> (i32, i32, usize) {
    let cp = *critical_path.get(&op_id).unwrap_or(&1) as i32;
    let slack = *slack_map.get(&op_id).unwrap_or(&0);
    let is_valu = if engine == "valu" { 0 } else { 1 };
    let is_load = if engine == "load" { 0 } else { 1 };

    match mode {
        PriorityMode::CriticalPath => (-cp, is_valu, op_id),
        PriorityMode::CriticalPathValu => (is_valu, -cp, op_id),
        PriorityMode::EngineBalance => {
            // Matches Python: -cp * load_factor where load_factor is 0.6, 0.8, 0.5, 1.0
            // Use float conversion to match Python's rounding behavior
            let load_factor: f64 = match engine {
                "valu" => 0.6,
                "load" | "store" => 0.8,
                "flow" => 0.5,
                _ => 1.0,  // ALU and others
            };
            let adjusted_cp = (-(cp as f64) * load_factor) as i32;
            (adjusted_cp, op_id as i32, op_id)
        }
        PriorityMode::Slack => (slack, -cp, op_id),
        PriorityMode::LoadEarly => (is_load, -cp, op_id),
        PriorityMode::LoadEarlySlack => (is_load, slack, op_id),
        PriorityMode::OriginalOrder => (op_idx as i32, is_valu, op_id),
    }
}

/// Compute ASAP (earliest possible cycle) for each op
fn compute_asap(ops: &[Op]) -> HashMap<usize, i32> {
    let mut asap: HashMap<usize, i32> = HashMap::new();

    // Build in-degree for topological sort
    let mut in_degree: HashMap<usize, usize> = HashMap::new();
    for op in ops {
        in_degree.insert(op.id, op.dep_ids.len());
    }

    let op_by_id: HashMap<usize, &Op> = ops.iter().map(|op| (op.id, op)).collect();

    // Start with ops that have no dependencies
    let mut queue: VecDeque<usize> = VecDeque::new();
    for op in ops {
        if op.dep_ids.is_empty() {
            queue.push_back(op.id);
            asap.insert(op.id, 0);
        }
    }

    // Process in topological order
    while let Some(op_id) = queue.pop_front() {
        let my_asap = *asap.get(&op_id).unwrap_or(&0);

        // Find successors (ops that depend on this one)
        for other in ops {
            if other.dep_ids.contains(&op_id) {
                // Update successor's ASAP
                let new_asap = my_asap + 1;
                let current = asap.entry(other.id).or_insert(0);
                if new_asap > *current {
                    *current = new_asap;
                }

                // Decrement in-degree
                if let Some(deg) = in_degree.get_mut(&other.id) {
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(other.id);
                    }
                }
            }
        }
    }

    asap
}

/// Compute ALAP (latest possible cycle) for each op given total cycles
fn compute_alap(ops: &[Op], total_cycles: i32) -> HashMap<usize, i32> {
    let mut alap: HashMap<usize, i32> = HashMap::new();

    // Build consumer map (ops that depend on each op)
    let mut consumers: HashMap<usize, Vec<usize>> = HashMap::new();
    for op in ops {
        consumers.insert(op.id, Vec::new());
    }
    for op in ops {
        for &dep_id in &op.dep_ids {
            if let Some(c) = consumers.get_mut(&dep_id) {
                c.push(op.id);
            }
        }
    }

    // Build out-degree
    let mut out_degree: HashMap<usize, usize> = HashMap::new();
    for op in ops {
        out_degree.insert(op.id, consumers.get(&op.id).map(|c| c.len()).unwrap_or(0));
    }

    // Start with sinks (ops with no consumers)
    let mut queue: VecDeque<usize> = VecDeque::new();
    for op in ops {
        if consumers.get(&op.id).map(|c| c.is_empty()).unwrap_or(true) {
            queue.push_back(op.id);
            alap.insert(op.id, total_cycles - 1);
        }
    }

    let op_by_id: HashMap<usize, &Op> = ops.iter().map(|op| (op.id, op)).collect();

    // Process in reverse topological order
    while let Some(op_id) = queue.pop_front() {
        let my_alap = *alap.get(&op_id).unwrap_or(&(total_cycles - 1));

        // Update predecessors (ops we depend on)
        if let Some(op) = op_by_id.get(&op_id) {
            for &dep_id in &op.dep_ids {
                let new_alap = my_alap - 1;
                let current = alap.entry(dep_id).or_insert(total_cycles - 1);
                if new_alap < *current {
                    *current = new_alap;
                }

                if let Some(deg) = out_degree.get_mut(&dep_id) {
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(dep_id);
                    }
                }
            }
        }
    }

    alap
}

/// Compute slack = ALAP - ASAP for each op
fn compute_slack(ops: &[Op], critical_path: &HashMap<usize, usize>) -> HashMap<usize, i32> {
    let total_cycles = critical_path.values().max().copied().unwrap_or(1) as i32;
    let asap = compute_asap(ops);
    let alap = compute_alap(ops, total_cycles);

    ops.iter()
        .map(|op| {
            let a = *asap.get(&op.id).unwrap_or(&0);
            let l = *alap.get(&op.id).unwrap_or(&(total_cycles - 1));
            (op.id, l - a)
        })
        .collect()
}

/// Greedy list scheduler with configurable priority mode and VALU→ALU shifting
fn schedule_greedy(
    ops: &mut [Op],
    critical_path: &HashMap<usize, usize>,
    enable_valu_shift: bool,
    priority_mode: PriorityMode,
) -> ScheduleResult {
    schedule_greedy_with_diagnostics(ops, critical_path, enable_valu_shift, priority_mode, false)
}

/// Greedy list scheduler with optional diagnostics collection
fn schedule_greedy_with_diagnostics(
    ops: &mut [Op],
    critical_path: &HashMap<usize, usize>,
    enable_valu_shift: bool,
    priority_mode: PriorityMode,
    collect_diagnostics: bool,
) -> ScheduleResult {
    let n = ops.len();
    if n == 0 {
        return ScheduleResult { cycles: 0, assignments: vec![], diagnostics: None, expanded_ops: vec![] };
    }

    // Initialize diagnostics if collecting
    let mut diagnostics = if collect_diagnostics {
        Some(SchedulingDiagnostics::new())
    } else {
        None
    };

    // Compute slack for slack-based priority modes
    let slack_map: HashMap<usize, i32> = if matches!(priority_mode, PriorityMode::Slack | PriorityMode::LoadEarlySlack) {
        compute_slack(ops, critical_path)
    } else {
        HashMap::new()
    };

    // Build op lookup (by index for mutation)
    let op_id_to_idx: HashMap<usize, usize> = ops.iter().enumerate()
        .map(|(i, op)| (op.id, i)).collect();

    // Build consumer map (reverse of dep_ids) - O(1) lookup of who depends on each op
    let mut consumers: HashMap<usize, Vec<usize>> = HashMap::new();
    for op in ops.iter() {
        consumers.insert(op.id, Vec::new());
    }
    for op in ops.iter() {
        for &dep_id in &op.dep_ids {
            if let Some(c) = consumers.get_mut(&dep_id) {
                c.push(op.id);
            }
        }
    }

    // Compute in-degree
    let mut in_degree: HashMap<usize, usize> = HashMap::new();
    for op in ops.iter() {
        in_degree.insert(op.id, op.dep_ids.len());
    }

    // Ready queue: ops with in_degree == 0, using priority tuples
    // Format: (priority_tuple, op_id) - lower tuple = higher priority
    let mut ready: BinaryHeap<Reverse<((i32, i32, usize), usize)>> = BinaryHeap::new();
    for (idx, op) in ops.iter().enumerate() {
        if op.dep_ids.is_empty() {
            let prio = compute_priority(op.id, idx, &op.engine, critical_path, &slack_map, priority_mode);
            ready.push(Reverse((prio, op.id)));
        }
    }

    // Track slot usage per cycle
    let mut slot_usage: HashMap<(usize, String), usize> = HashMap::new();

    // For VALU→ALU shifting: track expanded ops and their schedules
    let mut expanded_ops: Vec<Op> = Vec::new();
    let mut expanded_op_cycles: HashMap<usize, i32> = HashMap::new();
    let mut next_expanded_id = n + 1000000; // Start IDs high to avoid collision
    let mut shifted_valu_ops: HashSet<usize> = HashSet::new();

    let get_slot_usage = |slot_usage: &HashMap<(usize, String), usize>, cycle: usize, engine: &str| -> usize {
        *slot_usage.get(&(cycle, engine.to_string())).unwrap_or(&0)
    };

    let can_schedule_n = |slot_usage: &HashMap<(usize, String), usize>, cycle: usize, engine: &str, count: usize| -> bool {
        let usage = get_slot_usage(slot_usage, cycle, engine);
        let limit = get_slot_limit(engine);
        usage + count <= limit
    };

    let earliest_cycle_fn = |ops: &[Op], op_id: usize, op_id_to_idx: &HashMap<usize, usize>,
                            expanded_cycles: &HashMap<usize, i32>| -> usize {
        let op = &ops[*op_id_to_idx.get(&op_id).unwrap()];
        if op.dep_ids.is_empty() {
            return 0;
        }
        let mut max_cycle = 0;
        for &dep_id in &op.dep_ids {
            // Check if dependency was shifted (use expanded op cycle)
            if let Some(&exp_cycle) = expanded_cycles.get(&dep_id) {
                if exp_cycle >= 0 {
                    max_cycle = max_cycle.max(exp_cycle as usize + 1);
                }
            } else if let Some(&dep_idx) = op_id_to_idx.get(&dep_id) {
                let dep_cycle = ops[dep_idx].cycle;
                if dep_cycle >= 0 {
                    max_cycle = max_cycle.max(dep_cycle as usize + 1);
                }
            }
        }
        max_cycle
    };

    let mut assignments: Vec<(usize, usize)> = Vec::with_capacity(n);

    while let Some(Reverse((_prio, op_id))) = ready.pop() {
        let op_idx = *op_id_to_idx.get(&op_id).unwrap();
        let engine = ops[op_idx].engine.clone();
        let mut cycle = earliest_cycle_fn(ops, op_id, &op_id_to_idx, &expanded_op_cycles);

        // DIAGNOSTICS: Track slack and blocked ops
        if let Some(ref mut diag) = diagnostics {
            let op_slack = *slack_map.get(&op_id).unwrap_or(&0);
            diag.add_slack_sample(op_slack);

            // Check for blocked ops in ready queue
            let mut blocked_this_cycle = false;
            for other in ready.iter() {
                let other_id = other.0.1;
                let other_idx = *op_id_to_idx.get(&other_id).unwrap();
                let other_engine = &ops[other_idx].engine;
                let other_cycle = earliest_cycle_fn(ops, other_id, &op_id_to_idx, &expanded_op_cycles);

                if get_slot_usage(&slot_usage, other_cycle, other_engine) >= get_slot_limit(other_engine) {
                    diag.record_blocked(other_engine);
                    blocked_this_cycle = true;

                    // Check for priority inversion
                    let other_slack = *slack_map.get(&other_id).unwrap_or(&0);
                    if other_slack < op_slack {
                        diag.priority_inversions += 1;
                        diag.inversion_slack_delta_total += op_slack - other_slack;
                    }
                }
            }
            if blocked_this_cycle {
                diag.cycles_with_blocked_ops += 1;
            }

            // Check for "almost ready" ops (waiting on 1-2 dependencies)
            // Only count unique ops, not per-cycle
            // This is sampled periodically (every 100 cycles) to avoid O(n²) overhead
            if assignments.len() % 100 == 0 {
                let mut almost_ready_count = 0;
                for other in ops.iter() {
                    if other.cycle == -1 && !ready.iter().any(|r| r.0.1 == other.id) {
                        if let Some(&remaining) = in_degree.get(&other.id) {
                            if remaining >= 1 && remaining <= 2 {
                                almost_ready_count += 1;
                            }
                        }
                    }
                }
                if almost_ready_count > 0 {
                    diag.cycles_with_almost_ready += 1;
                    // Record by engine for the first few to get distribution
                    for other in ops.iter().take(1000) {
                        if other.cycle == -1 && !ready.iter().any(|r| r.0.1 == other.id) {
                            if let Some(&remaining) = in_degree.get(&other.id) {
                                if remaining >= 1 && remaining <= 2 {
                                    diag.record_almost_ready(&other.engine);
                                }
                            }
                        }
                    }
                }
            }
        }

        // VALU→ALU shifting: if VALU is full but we can shift to ALU
        if enable_valu_shift && engine == "valu" && can_convert_valu_to_alu(&ops[op_idx]) {
            let valu_usage = get_slot_usage(&slot_usage, cycle, "valu");
            let alu_usage = get_slot_usage(&slot_usage, cycle, "alu");

            // Check if VALU is saturated at this cycle
            if valu_usage >= get_slot_limit("valu") {
                if let Some(ref mut diag) = diagnostics {
                    diag.cycles_valu_saturated += 1;
                }

                if alu_usage + VLEN <= get_slot_limit("alu") {
                    // Expand VALU op to 8 ALU ops
                    let alu_ops = expand_valu_to_alu(&ops[op_idx], next_expanded_id);
                    next_expanded_id += VLEN;

                    // Find cycle where all 8 ALU ops fit
                    while !can_schedule_n(&slot_usage, cycle, "alu", VLEN) {
                        cycle += 1;
                    }

                    // Schedule all 8 ALU ops at this cycle
                    for _alu_op in &alu_ops {
                        *slot_usage.entry((cycle, "alu".to_string())).or_insert(0) += 1;
                    }

                    // Mark original VALU op as shifted - change engine to prevent emission
                    ops[op_idx].engine = "shifted_valu".to_string();
                    ops[op_idx].cycle = cycle as i32;
                    expanded_op_cycles.insert(op_id, cycle as i32);
                    shifted_valu_ops.insert(op_id);
                    expanded_ops.extend(alu_ops);
                    assignments.push((op_id, cycle));

                    if let Some(ref mut diag) = diagnostics {
                        diag.valu_shifted += 1;
                    }

                    // Update consumers
                    if let Some(consumer_ids) = consumers.get(&op_id) {
                        for &consumer_id in consumer_ids {
                            if let Some(deg) = in_degree.get_mut(&consumer_id) {
                                *deg -= 1;
                                if *deg == 0 {
                                    let consumer_idx = *op_id_to_idx.get(&consumer_id).unwrap();
                                    let prio = compute_priority(
                                        consumer_id,
                                        consumer_idx,
                                        &ops[consumer_idx].engine,
                                        critical_path,
                                        &slack_map,
                                        priority_mode,
                                    );
                                    ready.push(Reverse((prio, consumer_id)));
                                }
                            }
                        }
                    }
                    continue;
                } else {
                    if let Some(ref mut diag) = diagnostics {
                        diag.valu_alu_no_capacity += 1;
                    }
                }
            } else {
                if let Some(ref mut diag) = diagnostics {
                    diag.valu_slot_available += 1;
                }
            }
        } else if enable_valu_shift && engine == "valu" && !can_convert_valu_to_alu(&ops[op_idx]) {
            if let Some(ref mut diag) = diagnostics {
                diag.valu_not_shiftable += 1;
            }
        }

        // Normal scheduling (find first cycle with available slot)
        while get_slot_usage(&slot_usage, cycle, &engine) >= get_slot_limit(&engine) {
            cycle += 1;
        }

        // Schedule the op
        ops[op_idx].cycle = cycle as i32;
        *slot_usage.entry((cycle, engine.clone())).or_insert(0) += 1;
        assignments.push((op_id, cycle));

        // Update ready queue for dependent ops
        if let Some(consumer_ids) = consumers.get(&op_id) {
            for &consumer_id in consumer_ids {
                if let Some(deg) = in_degree.get_mut(&consumer_id) {
                    *deg -= 1;
                    if *deg == 0 {
                        let consumer_idx = *op_id_to_idx.get(&consumer_id).unwrap();
                        let prio = compute_priority(
                            consumer_id,
                            consumer_idx,
                            &ops[consumer_idx].engine,
                            critical_path,
                            &slack_map,
                            priority_mode,
                        );
                        ready.push(Reverse((prio, consumer_id)));
                    }
                }
            }
        }
    }

    let max_cycle = ops.iter()
        .filter(|op| op.cycle >= 0)
        .map(|op| op.cycle as usize)
        .max()
        .unwrap_or(0);

    // Finalize diagnostics
    if let Some(ref mut diag) = diagnostics {
        diag.total_cycles = max_cycle + 1;
        diag.total_ops_scheduled = assignments.len();
    }

    ScheduleResult {
        cycles: max_cycle + 1,
        assignments,
        diagnostics,
        expanded_ops,
    }
}

/// Parse ops from Python list of dicts
fn parse_ops(py: Python<'_>, instrs: &Bound<'_, PyList>) -> PyResult<Vec<Op>> {
    let mut ops = Vec::new();
    let mut op_id = 0usize;

    for (cycle, instr) in instrs.iter().enumerate() {
        let instr_dict = instr.downcast::<PyDict>()?;

        for (engine_key, slots) in instr_dict.iter() {
            let engine: String = engine_key.extract()?;
            let slots_list = slots.downcast::<PyList>()?;

            for slot in slots_list.iter() {
                let slot_tuple = slot.downcast::<PyTuple>()?;
                let mut reads: HashSet<i64> = HashSet::new();
                let mut writes: HashSet<i64> = HashSet::new();

                // Parse reads/writes based on op type
                let op_type: String = slot_tuple.get_item(0)?.extract()?;

                match engine.as_str() {
                    "load" => {
                        match op_type.as_str() {
                            "const" => {
                                let dest: i64 = slot_tuple.get_item(1)?.extract()?;
                                writes.insert(dest);
                            }
                            "load" => {
                                let dest: i64 = slot_tuple.get_item(1)?.extract()?;
                                let src: i64 = slot_tuple.get_item(2)?.extract()?;
                                writes.insert(dest);
                                reads.insert(src);
                            }
                            "vload" => {
                                let dest: i64 = slot_tuple.get_item(1)?.extract()?;
                                let src: i64 = slot_tuple.get_item(2)?.extract()?;
                                for i in 0..VLEN as i64 {
                                    writes.insert(dest + i);
                                }
                                reads.insert(src);
                            }
                            _ => {}
                        }
                    }
                    "store" => {
                        match op_type.as_str() {
                            "store" => {
                                let addr: i64 = slot_tuple.get_item(1)?.extract()?;
                                let data: i64 = slot_tuple.get_item(2)?.extract()?;
                                reads.insert(addr);
                                reads.insert(data);
                            }
                            "vstore" => {
                                let addr: i64 = slot_tuple.get_item(1)?.extract()?;
                                let data: i64 = slot_tuple.get_item(2)?.extract()?;
                                reads.insert(addr);
                                for i in 0..VLEN as i64 {
                                    reads.insert(data + i);
                                }
                            }
                            _ => {}
                        }
                    }
                    "alu" => {
                        let dest: i64 = slot_tuple.get_item(1)?.extract()?;
                        let src1: i64 = slot_tuple.get_item(2)?.extract()?;
                        let src2: i64 = slot_tuple.get_item(3)?.extract()?;
                        writes.insert(dest);
                        reads.insert(src1);
                        reads.insert(src2);
                    }
                    "valu" => {
                        match op_type.as_str() {
                            "vbroadcast" => {
                                let dest: i64 = slot_tuple.get_item(1)?.extract()?;
                                let src: i64 = slot_tuple.get_item(2)?.extract()?;
                                for i in 0..VLEN as i64 {
                                    writes.insert(dest + i);
                                }
                                reads.insert(src);
                            }
                            "vmultiply_add" | "multiply_add" => {
                                let dest: i64 = slot_tuple.get_item(1)?.extract()?;
                                let src1: i64 = slot_tuple.get_item(2)?.extract()?;
                                let src2: i64 = slot_tuple.get_item(3)?.extract()?;
                                let src3: i64 = slot_tuple.get_item(4)?.extract()?;
                                for i in 0..VLEN as i64 {
                                    writes.insert(dest + i);
                                    reads.insert(src1 + i);
                                    reads.insert(src2 + i);
                                    reads.insert(src3 + i);
                                }
                            }
                            _ => {
                                // Binary vector ops
                                let dest: i64 = slot_tuple.get_item(1)?.extract()?;
                                let src1: i64 = slot_tuple.get_item(2)?.extract()?;
                                let src2: i64 = slot_tuple.get_item(3)?.extract()?;
                                for i in 0..VLEN as i64 {
                                    writes.insert(dest + i);
                                    reads.insert(src1 + i);
                                    reads.insert(src2 + i);
                                }
                            }
                        }
                    }
                    "flow" => {
                        match op_type.as_str() {
                            "vselect" => {
                                let dest: i64 = slot_tuple.get_item(1)?.extract()?;
                                let src1: i64 = slot_tuple.get_item(2)?.extract()?;
                                let src2: i64 = slot_tuple.get_item(3)?.extract()?;
                                let src3: i64 = slot_tuple.get_item(4)?.extract()?;
                                for i in 0..VLEN as i64 {
                                    writes.insert(dest + i);
                                    reads.insert(src1 + i);
                                    reads.insert(src2 + i);
                                    reads.insert(src3 + i);
                                }
                            }
                            "add_imm" => {
                                let dest: i64 = slot_tuple.get_item(1)?.extract()?;
                                let src: i64 = slot_tuple.get_item(2)?.extract()?;
                                writes.insert(dest);
                                reads.insert(src);
                            }
                            "cond_jump" => {
                                let cond: i64 = slot_tuple.get_item(1)?.extract()?;
                                reads.insert(cond);
                            }
                            _ => {}
                        }
                    }
                    "debug" => {
                        match op_type.as_str() {
                            "vcompare" => {
                                let loc: i64 = slot_tuple.get_item(1)?.extract()?;
                                for i in 0..VLEN as i64 {
                                    reads.insert(loc + i);
                                }
                            }
                            "compare" => {
                                let loc: i64 = slot_tuple.get_item(1)?.extract()?;
                                reads.insert(loc);
                            }
                            _ => {}
                        }
                    }
                    _ => {}
                }

                // Store slot as native Rust values
                let mut slot_vec: Vec<SlotVal> = Vec::new();
                for i in 0..slot_tuple.len() {
                    let item = slot_tuple.get_item(i)?;
                    if let Ok(s) = item.extract::<String>() {
                        slot_vec.push(SlotVal::Str(s));
                    } else if let Ok(n) = item.extract::<i64>() {
                        slot_vec.push(SlotVal::Int(n));
                    } else {
                        // Fallback: convert to string
                        slot_vec.push(SlotVal::Str(item.str()?.to_string()));
                    }
                }

                ops.push(Op {
                    id: op_id,
                    engine: engine.clone(),
                    slot: slot_vec,
                    cycle: cycle as i32,
                    dep_ids: HashSet::new(),
                    reads,
                    writes,
                });
                op_id += 1;
            }
        }
    }

    Ok(ops)
}

/// Compute dependencies between ops
fn compute_dependencies(ops: &mut [Op]) {
    // Map: address -> last op that wrote to it
    let mut last_writer: HashMap<i64, usize> = HashMap::new();
    // Map: address -> ops that read since last write
    let mut readers_since_write: HashMap<i64, Vec<usize>> = HashMap::new();

    for i in 0..ops.len() {
        let reads = ops[i].reads.clone();
        let writes = ops[i].writes.clone();
        let op_id = ops[i].id;

        // RAW: depend on writers of values we read
        for addr in &reads {
            if let Some(&writer_id) = last_writer.get(addr) {
                ops[i].dep_ids.insert(writer_id);
            }
            readers_since_write.entry(*addr).or_default().push(op_id);
        }

        // WAR + WAW: depend on ops for values we write
        for addr in &writes {
            // WAW: depend on previous writer
            if let Some(&writer_id) = last_writer.get(addr) {
                ops[i].dep_ids.insert(writer_id);
            }
            // WAR: depend on all readers since last write
            if let Some(readers) = readers_since_write.get(addr) {
                for &reader_id in readers {
                    if reader_id != op_id {
                        ops[i].dep_ids.insert(reader_id);
                    }
                }
            }
            readers_since_write.remove(addr);
            last_writer.insert(*addr, op_id);
        }
    }
}

/// Emit scheduled ops back to Python instruction list
fn emit_scheduled<'py>(py: Python<'py>, ops: &[Op], max_cycle: usize) -> PyResult<Bound<'py, PyList>> {
    emit_scheduled_with_expanded(py, ops, &[], max_cycle)
}

/// Emit scheduled ops including expanded ALU ops from VALU→ALU shifting
/// Skips ops with engine="shifted_valu" (these are replaced by expanded_ops)
fn emit_scheduled_with_expanded<'py>(
    py: Python<'py>,
    ops: &[Op],
    expanded_ops: &[Op],
    max_cycle: usize,
) -> PyResult<Bound<'py, PyList>> {
    let instrs = PyList::empty(py);

    for cycle in 0..max_cycle {
        let instr = PyDict::new(py);

        // Emit regular ops, SKIP shifted_valu (these are replaced by expanded ALU ops)
        for op in ops {
            if op.cycle == cycle as i32 && op.engine != "shifted_valu" {
                let engine = &op.engine;

                // Get or create the list for this engine
                if !instr.contains(engine.as_str())? {
                    instr.set_item(engine.as_str(), PyList::empty(py))?;
                }

                let slots = instr.get_item(engine.as_str())?.unwrap();
                let slots_list = slots.downcast::<PyList>()?;

                // Create tuple from slot
                let py_items: Vec<PyObject> = op.slot.iter().map(|v| {
                    match v {
                        SlotVal::Str(s) => s.into_pyobject(py).unwrap().into_any().unbind(),
                        SlotVal::Int(n) => n.into_pyobject(py).unwrap().into_any().unbind(),
                    }
                }).collect();
                let slot_tuple = PyTuple::new(py, &py_items)?;
                slots_list.append(slot_tuple)?;
            }
        }

        // Emit expanded ALU ops from VALU→ALU shifting
        for op in expanded_ops {
            if op.cycle == cycle as i32 {
                let engine = &op.engine;

                // Get or create the list for this engine
                if !instr.contains(engine.as_str())? {
                    instr.set_item(engine.as_str(), PyList::empty(py))?;
                }

                let slots = instr.get_item(engine.as_str())?.unwrap();
                let slots_list = slots.downcast::<PyList>()?;

                // Create tuple from slot
                let py_items: Vec<PyObject> = op.slot.iter().map(|v| {
                    match v {
                        SlotVal::Str(s) => s.into_pyobject(py).unwrap().into_any().unbind(),
                        SlotVal::Int(n) => n.into_pyobject(py).unwrap().into_any().unbind(),
                    }
                }).collect();
                let slot_tuple = PyTuple::new(py, &py_items)?;
                slots_list.append(slot_tuple)?;
            }
        }

        if !instr.is_empty() {
            instrs.append(instr)?;
        }
    }

    Ok(instrs)
}

/// Main scheduling function exposed to Python
#[pyfunction]
fn schedule<'py>(py: Python<'py>, instrs: &Bound<'py, PyList>, enable_valu_shift: bool) -> PyResult<Bound<'py, PyList>> {
    // Parse ops
    let mut ops = parse_ops(py, instrs)?;

    if ops.is_empty() {
        return Ok(PyList::empty(py));
    }

    // Reset cycles
    for op in &mut ops {
        op.cycle = -1;
    }

    // Compute dependencies
    compute_dependencies(&mut ops);

    // Compute critical path
    let critical_path = compute_critical_path(&ops);

    // Schedule with default priority mode
    let result = schedule_greedy(&mut ops, &critical_path, enable_valu_shift, PriorityMode::CriticalPath);

    // Emit (include expanded ops from VALU→ALU shifting)
    emit_scheduled_with_expanded(py, &ops, &result.expanded_ops, result.cycles)
}

/// Get just the cycle count without emitting (faster for testing)
#[pyfunction]
fn schedule_cycles(py: Python<'_>, instrs: &Bound<'_, PyList>, enable_valu_shift: bool) -> PyResult<usize> {
    let mut ops = parse_ops(py, instrs)?;

    if ops.is_empty() {
        return Ok(0);
    }

    for op in &mut ops {
        op.cycle = -1;
    }

    compute_dependencies(&mut ops);
    let critical_path = compute_critical_path(&ops);
    let result = schedule_greedy(&mut ops, &critical_path, enable_valu_shift, PriorityMode::CriticalPath);

    Ok(result.cycles)
}

/// Run multiple scheduling passes with different priority modes and return the best
#[pyfunction]
#[pyo3(signature = (instrs, enable_valu_shift, verbose=false))]
fn schedule_multipass<'py>(py: Python<'py>, instrs: &Bound<'py, PyList>, enable_valu_shift: bool, verbose: bool) -> PyResult<Bound<'py, PyList>> {
    let base_ops = parse_ops(py, instrs)?;

    if base_ops.is_empty() {
        return Ok(PyList::empty(py));
    }

    // Compute dependencies once (they don't change between passes)
    let mut ops_for_deps = base_ops.clone();
    compute_dependencies(&mut ops_for_deps);

    // Copy dep_ids to base_ops for reuse
    let dep_ids_map: HashMap<usize, HashSet<usize>> = ops_for_deps.iter()
        .map(|op| (op.id, op.dep_ids.clone()))
        .collect();

    let critical_path = compute_critical_path(&ops_for_deps);

    let mut best_cycles = usize::MAX;
    let mut best_ops: Option<Vec<Op>> = None;
    let mut best_expanded_ops: Vec<Op> = Vec::new();
    let mut best_mode: Option<PriorityMode> = None;

    // Try all priority modes and keep the best result
    for &mode in ALL_PRIORITY_MODES.iter() {
        let mut ops = base_ops.clone();
        for op in &mut ops {
            op.cycle = -1;
            op.dep_ids = dep_ids_map.get(&op.id).cloned().unwrap_or_default();
        }

        let result = schedule_greedy(&mut ops, &critical_path, enable_valu_shift, mode);

        if verbose {
            println!("  {:?}: {} cycles", mode, result.cycles);
        }

        if result.cycles < best_cycles {
            best_cycles = result.cycles;
            best_ops = Some(ops);
            best_expanded_ops = result.expanded_ops;
            best_mode = Some(mode);
        }
    }

    if verbose {
        println!("Best: {:?} with {} cycles", best_mode.unwrap(), best_cycles);
    }

    emit_scheduled_with_expanded(py, &best_ops.unwrap(), &best_expanded_ops, best_cycles)
}

/// Count cycles for a program (simulates cycle counting without full memory simulation)
/// Counts instructions that have any non-debug slots, matching Machine.run() behavior
#[pyfunction]
fn count_cycles(_py: Python<'_>, instrs: &Bound<'_, PyList>) -> PyResult<usize> {
    let mut cycles = 0;
    for instr in instrs.iter() {
        let instr_dict = instr.downcast::<PyDict>()?;
        let has_non_debug = instr_dict.keys().iter().any(|key| {
            let key_str: String = key.extract().unwrap_or_default();
            key_str != "debug"
        });
        if has_non_debug {
            cycles += 1;
        }
    }
    Ok(cycles)
}

/// Get cycle count from multipass (for testing)
#[pyfunction]
#[pyo3(signature = (instrs, enable_valu_shift, verbose=false))]
fn schedule_multipass_cycles(py: Python<'_>, instrs: &Bound<'_, PyList>, enable_valu_shift: bool, verbose: bool) -> PyResult<usize> {
    let base_ops = parse_ops(py, instrs)?;

    if base_ops.is_empty() {
        return Ok(0);
    }

    let mut ops_for_deps = base_ops.clone();
    compute_dependencies(&mut ops_for_deps);

    let dep_ids_map: HashMap<usize, HashSet<usize>> = ops_for_deps.iter()
        .map(|op| (op.id, op.dep_ids.clone()))
        .collect();

    let critical_path = compute_critical_path(&ops_for_deps);

    let mut best_cycles = usize::MAX;
    let mut best_mode: Option<PriorityMode> = None;

    // Try all priority modes
    for &mode in ALL_PRIORITY_MODES.iter() {
        let mut ops = base_ops.clone();
        for op in &mut ops {
            op.cycle = -1;
            op.dep_ids = dep_ids_map.get(&op.id).cloned().unwrap_or_default();
        }

        let result = schedule_greedy(&mut ops, &critical_path, enable_valu_shift, mode);

        if verbose {
            println!("  {:?}: {} cycles", mode, result.cycles);
        }

        if result.cycles < best_cycles {
            best_cycles = result.cycles;
            best_mode = Some(mode);
        }
    }

    if verbose {
        println!("Best: {:?} with {} cycles", best_mode.unwrap(), best_cycles);
    }

    Ok(best_cycles)
}

/// Run multipass scheduling with diagnostics collection, returns JSON report
#[pyfunction]
#[pyo3(signature = (instrs, enable_valu_shift, verbose=false))]
fn run_diagnostics(py: Python<'_>, instrs: &Bound<'_, PyList>, enable_valu_shift: bool, verbose: bool) -> PyResult<String> {
    let base_ops = parse_ops(py, instrs)?;

    if base_ops.is_empty() {
        return Ok("{}".to_string());
    }

    // Compute dependencies once
    let mut ops_for_deps = base_ops.clone();
    compute_dependencies(&mut ops_for_deps);

    let dep_ids_map: HashMap<usize, HashSet<usize>> = ops_for_deps.iter()
        .map(|op| (op.id, op.dep_ids.clone()))
        .collect();

    let critical_path = compute_critical_path(&ops_for_deps);

    // First pass: find best mode
    let mut best_cycles = usize::MAX;
    let mut best_mode: Option<PriorityMode> = None;

    for &mode in ALL_PRIORITY_MODES.iter() {
        let mut ops = base_ops.clone();
        for op in &mut ops {
            op.cycle = -1;
            op.dep_ids = dep_ids_map.get(&op.id).cloned().unwrap_or_default();
        }

        let result = schedule_greedy(&mut ops, &critical_path, enable_valu_shift, mode);

        if verbose {
            println!("  {:?}: {} cycles", mode, result.cycles);
        }

        if result.cycles < best_cycles {
            best_cycles = result.cycles;
            best_mode = Some(mode);
        }
    }

    let best_mode = best_mode.unwrap();
    if verbose {
        println!("Best: {:?} with {} cycles", best_mode, best_cycles);
        println!("Collecting diagnostics for best mode...");
    }

    // Second pass: collect diagnostics for best mode
    let mut ops = base_ops.clone();
    for op in &mut ops {
        op.cycle = -1;
        op.dep_ids = dep_ids_map.get(&op.id).cloned().unwrap_or_default();
    }

    let result = schedule_greedy_with_diagnostics(
        &mut ops, &critical_path, enable_valu_shift, best_mode, true
    );

    // Analyze shared temp dependencies
    let shared_temp_analysis = analyze_shared_temp_dependencies(&ops_for_deps);

    // Build combined report
    #[derive(Serialize)]
    struct DiagnosticReport {
        scheduling: SchedulingDiagnostics,
        shared_temps: SharedTempAnalysis,
        best_mode: String,
        cycles: usize,
    }

    let report = DiagnosticReport {
        scheduling: result.diagnostics.unwrap_or_default(),
        shared_temps: shared_temp_analysis,
        best_mode: format!("{:?}", best_mode),
        cycles: result.cycles,
    };

    if verbose {
        println!("\n{}", report.scheduling.summary());
        println!("\n{}", report.shared_temps.summary());
    }

    Ok(serde_json::to_string_pretty(&report).unwrap_or_else(|_| "{}".to_string()))
}

/// Run shared temp analysis only, returns JSON
#[pyfunction]
fn analyze_shared_temps(py: Python<'_>, instrs: &Bound<'_, PyList>) -> PyResult<String> {
    let mut ops = parse_ops(py, instrs)?;

    if ops.is_empty() {
        return Ok("{}".to_string());
    }

    compute_dependencies(&mut ops);
    let analysis = analyze_shared_temp_dependencies(&ops);

    Ok(analysis.summary() + "\n\n" + &serde_json::to_string_pretty(&analysis).unwrap_or_else(|_| "{}".to_string()))
}

/// Print full diagnostic report (convenience function)
#[pyfunction]
#[pyo3(signature = (instrs, enable_valu_shift=true))]
fn print_diagnostics(py: Python<'_>, instrs: &Bound<'_, PyList>, enable_valu_shift: bool) -> PyResult<()> {
    let json = run_diagnostics(py, instrs, enable_valu_shift, true)?;
    println!("\n=== JSON REPORT ===\n{}", json);
    Ok(())
}

// =============================================================================
// Const-to-ALU Optimization
// =============================================================================

/// Optimize const loads to use ALU operations instead
/// This frees up load slots for actual memory operations
fn optimize_const_to_alu(ops: &mut Vec<Op>) -> (usize, usize) {
    let mut const_loads_removed = 0usize;
    let mut alu_ops_added = 0usize;

    // First pass: find all const loads and their values
    let mut const_loads: Vec<(usize, i64, i64)> = Vec::new(); // (op_idx, dest, value)
    for (idx, op) in ops.iter().enumerate() {
        if op.engine == "load" && op.slot.len() >= 3 {
            if let SlotVal::Str(s) = &op.slot[0] {
                if s == "const" {
                    if let (SlotVal::Int(dest), SlotVal::Int(val)) = (&op.slot[1], &op.slot[2]) {
                        const_loads.push((idx, *dest, *val));
                    }
                }
            }
        }
    }

    if const_loads.is_empty() {
        return (0, 0);
    }

    // Strategy: We only need to load ONE non-zero constant (preferably 1)
    // Everything else can be derived:
    // - 0 = x ^ x (xor anything with itself)
    // - 2 = 1 + 1
    // - 3 = 2 + 1
    // - 4 = 2 + 2
    // - 5 = 4 + 1
    // - 6 = 3 + 3
    // - etc.

    // Find the const 1 load - this is the only one we truly need
    let one_load = const_loads.iter().find(|(_, _, v)| *v == 1);
    let bootstrap_idx: usize;
    let bootstrap_dest: i64;

    if let Some(&(idx, dest, _)) = one_load {
        bootstrap_idx = idx;
        bootstrap_dest = dest;
    } else {
        // No const 1, keep first const as bootstrap
        bootstrap_idx = const_loads[0].0;
        bootstrap_dest = const_loads[0].1;
    }

    // Track destinations for each value we can generate
    let mut value_dests: HashMap<i64, i64> = HashMap::new();

    // Track which ops to convert: (op_idx, new_slot, dest, val)
    let mut conversions: Vec<(usize, Vec<SlotVal>, i64, i64)> = Vec::new();

    // Process const loads - convert all except the bootstrap
    for &(idx, dest, val) in &const_loads {
        if idx == bootstrap_idx {
            // Keep bootstrap load, record its value
            value_dests.insert(val, dest);
            continue;
        }

        // Try to generate this value via ALU
        let conversion = if val == 0 {
            // 0 = bootstrap ^ bootstrap (xor anything with itself)
            Some(vec![
                SlotVal::Str("^".to_string()),
                SlotVal::Int(dest),
                SlotVal::Int(bootstrap_dest),
                SlotVal::Int(bootstrap_dest),
            ])
        } else if val == 1 && value_dests.contains_key(&0) {
            // If we have 0 but not 1, we can't easily generate 1
            // This shouldn't happen if we chose bootstrap correctly
            None
        } else if val == 2 {
            value_dests.get(&1).map(|&one_d| vec![
                SlotVal::Str("+".to_string()),
                SlotVal::Int(dest),
                SlotVal::Int(one_d),
                SlotVal::Int(one_d),
            ])
        } else if val == 3 {
            match (value_dests.get(&2), value_dests.get(&1)) {
                (Some(&two_d), Some(&one_d)) => Some(vec![
                    SlotVal::Str("+".to_string()),
                    SlotVal::Int(dest),
                    SlotVal::Int(two_d),
                    SlotVal::Int(one_d),
                ]),
                _ => None,
            }
        } else if val == 4 {
            value_dests.get(&2).map(|&two_d| vec![
                SlotVal::Str("+".to_string()),
                SlotVal::Int(dest),
                SlotVal::Int(two_d),
                SlotVal::Int(two_d),
            ])
        } else if val == 5 {
            match (value_dests.get(&4), value_dests.get(&1)) {
                (Some(&four_d), Some(&one_d)) => Some(vec![
                    SlotVal::Str("+".to_string()),
                    SlotVal::Int(dest),
                    SlotVal::Int(four_d),
                    SlotVal::Int(one_d),
                ]),
                _ => None,
            }
        } else if val == 6 {
            value_dests.get(&3).map(|&three_d| vec![
                SlotVal::Str("+".to_string()),
                SlotVal::Int(dest),
                SlotVal::Int(three_d),
                SlotVal::Int(three_d),
            ])
        } else if val == 7 {
            match (value_dests.get(&6), value_dests.get(&1)) {
                (Some(&six_d), Some(&one_d)) => Some(vec![
                    SlotVal::Str("+".to_string()),
                    SlotVal::Int(dest),
                    SlotVal::Int(six_d),
                    SlotVal::Int(one_d),
                ]),
                (Some(&four_d), Some(&three_d)) if value_dests.contains_key(&4) && value_dests.contains_key(&3) => Some(vec![
                    SlotVal::Str("+".to_string()),
                    SlotVal::Int(dest),
                    SlotVal::Int(*value_dests.get(&4).unwrap()),
                    SlotVal::Int(*value_dests.get(&3).unwrap()),
                ]),
                _ => None,
            }
        } else if val == 8 {
            value_dests.get(&4).map(|&four_d| vec![
                SlotVal::Str("+".to_string()),
                SlotVal::Int(dest),
                SlotVal::Int(four_d),
                SlotVal::Int(four_d),
            ])
        } else {
            None
        };

        if let Some(new_slot) = conversion {
            conversions.push((idx, new_slot.clone(), dest, val));
            value_dests.insert(val, dest);
            const_loads_removed += 1;
            alu_ops_added += 1;
        }
    }

    // Apply conversions and update reads/writes for proper dependency tracking
    for (idx, new_slot, dest, _val) in conversions {
        ops[idx].engine = "alu".to_string();

        // Update reads: ALU op reads from src1 and src2 (indices 2 and 3)
        // SPECIAL CASE: XOR of x with itself (x ^ x = 0) has NO dependencies
        // because the result is always 0 regardless of x's value
        ops[idx].reads.clear();
        let is_xor_self = if let Some(SlotVal::Str(op)) = new_slot.get(0) {
            if op == "^" {
                if let (Some(SlotVal::Int(src1)), Some(SlotVal::Int(src2))) = (new_slot.get(2), new_slot.get(3)) {
                    src1 == src2
                } else {
                    false
                }
            } else {
                false
            }
        } else {
            false
        };

        if !is_xor_self {
            // Normal case: add read dependencies
            if let Some(SlotVal::Int(src1)) = new_slot.get(2) {
                ops[idx].reads.insert(*src1);
            }
            if let Some(SlotVal::Int(src2)) = new_slot.get(3) {
                ops[idx].reads.insert(*src2);
            }
        }
        // For XOR self: reads stays empty (no dependencies)

        // Update writes: ALU op writes to dest (index 1)
        ops[idx].writes.clear();
        ops[idx].writes.insert(dest);

        ops[idx].slot = new_slot;
    }

    (const_loads_removed, alu_ops_added)
}

/// Format an address with prefix for readability
/// s = scratch (register), i = immediate constant, p = program address
fn format_addr(val: i64, context: &str, slot_idx: usize) -> String {
    match context {
        "const" => {
            // const dest, immediate_value
            if slot_idx == 1 {
                format!("s{}", val)  // destination is scratch
            } else {
                format!("i{}", val)  // value is immediate
            }
        }
        "load" => {
            // load dest, addr_reg
            format!("s{}", val)  // both are scratch addresses
        }
        "vload" => format!("s{}", val),
        "store" | "vstore" => format!("s{}", val),
        "vbroadcast" => {
            // vbroadcast vdest, scalar_src
            if slot_idx == 1 {
                format!("v{}", val)  // vector destination
            } else {
                format!("s{}", val)  // scalar source
            }
        }
        "vselect" => {
            // vselect dest, mask, a, b
            format!("v{}", val)  // all vector addresses
        }
        "multiply_add" | "vmultiply_add" => format!("v{}", val),
        // ALU binary ops: dest, src1, src2
        "+" | "-" | "*" | "/" | "%" | "&" | "|" | "^" | "<<" | ">>" |
        "==" | "!=" | "<" | "<=" | ">" | ">=" => {
            format!("s{}", val)  // all scratch addresses
        }
        "add_imm" => {
            if slot_idx == 3 {
                format!("i{}", val)  // immediate value
            } else {
                format!("s{}", val)  // scratch addresses
            }
        }
        "pause" | "jump" | "branch" => format!("p{}", val),  // program addresses
        _ => val.to_string(),  // unknown, keep as-is
    }
}

/// Cycle-by-cycle instruction output with slot utilization markers
#[derive(Serialize)]
struct CycleOutput {
    cycle: usize,
    load: Vec<String>,  // Compact format: ["op", "dest", "src"] as single string
    load_empty: usize,
    store: Vec<String>,
    store_empty: usize,
    alu: Vec<String>,
    alu_empty: usize,
    valu: Vec<String>,
    valu_empty: usize,
    flow: Vec<String>,
    flow_empty: usize,
}

/// Full schedule output with slot utilization
#[derive(Serialize)]
struct ScheduleOutput {
    total_cycles: usize,
    total_ops: usize,
    best_mode: String,
    slot_limits: HashMap<String, usize>,
    cycles: Vec<CycleOutput>,
    summary: ScheduleSummary,
}

#[derive(Serialize)]
struct ScheduleSummary {
    load_utilization_pct: f64,
    store_utilization_pct: f64,
    alu_utilization_pct: f64,
    valu_utilization_pct: f64,
    total_empty_load: usize,
    total_empty_store: usize,
    total_empty_alu: usize,
    total_empty_valu: usize,
}

/// Schedule instructions and output JSON with slot utilization markers
#[pyfunction]
#[pyo3(signature = (instrs, enable_valu_shift, output_path, optimize_consts=true))]
fn schedule_to_json(py: Python<'_>, instrs: &Bound<'_, PyList>, enable_valu_shift: bool, output_path: String, optimize_consts: bool) -> PyResult<String> {
    let mut base_ops = parse_ops(py, instrs)?;

    if base_ops.is_empty() {
        return Ok("{}".to_string());
    }

    // Apply constant derivation optimization
    if optimize_consts {
        // Analyze constant usage
        let usage = analyze_constant_usage(&base_ops);

        if !usage.is_empty() {
            // Build derivation plan
            let plan = build_constant_plan(&usage, 3);  // hot_threshold = 3

            println!("Const derivation: {} unique values, {} bootstrap loads, {} derivable",
                     usage.len(), plan.bootstrap_loads.len(),
                     plan.derivations.len() - plan.bootstrap_loads.len());

            // Find highest used register to allocate fresh constant registers above it
            let max_reg = base_ops.iter()
                .flat_map(|op| op.reads.iter().chain(op.writes.iter()))
                .max()
                .copied()
                .unwrap_or(0);

            // Allocate FRESH registers for constants to avoid clobbering
            // These won't conflict with any existing register usage
            let mut value_to_register: HashMap<i64, i64> = HashMap::new();
            let mut register_rename: HashMap<i64, i64> = HashMap::new();  // old_dest -> fresh_reg
            let mut next_fresh_reg = max_reg + 1000;  // Start well above existing registers

            // Sort values for deterministic allocation
            let mut sorted_values: Vec<_> = usage.keys().copied().collect();
            sorted_values.sort();

            for value in sorted_values {
                let u = &usage[&value];
                // Allocate a fresh register for this constant
                let fresh_reg = next_fresh_reg;
                next_fresh_reg += 1;
                value_to_register.insert(value, fresh_reg);

                // Map ALL old destinations to the fresh register
                for &dest in &u.destinations {
                    register_rename.insert(dest, fresh_reg);
                }
            }

            let renames_count = register_rename.len();

            // Identify and remove const load ops
            let const_op_ids: HashSet<usize> = base_ops.iter()
                .filter(|op| {
                    op.engine == "load" &&
                    op.slot.get(0).map(|s| matches!(s, SlotVal::Str(ref o) if o == "const")).unwrap_or(false)
                })
                .map(|op| op.id)
                .collect();

            let original_const_count = const_op_ids.len();
            base_ops.retain(|op| !const_op_ids.contains(&op.id));

            // Apply register renaming to remaining ops - READS ONLY
            // We only rename reads because writes define new values that might
            // legitimately reuse a register that previously held a constant
            for op in &mut base_ops {
                // Rename reads in slot values (the instruction encoding)
                // For most ops: slot[0]=opcode, slot[1]=dest, slot[2..]=sources
                // We need to rename sources but NOT destinations
                let opcode: String = op.slot.get(0).and_then(|v| match v {
                    SlotVal::Str(s) => Some(s.clone()),
                    _ => None,
                }).unwrap_or_default();

                for (idx, slot_val) in op.slot.iter_mut().enumerate() {
                    if let SlotVal::Int(reg) = slot_val {
                        // Determine if this slot is a source (read) or destination (write)
                        let is_source = match opcode.as_str() {
                            // ALU/VALU binary ops: dest at 1, sources at 2,3
                            "+" | "-" | "*" | "/" | "%" | "&" | "|" | "^" | "<<" | ">>" |
                            "==" | "!=" | "<" | "<=" | ">" | ">=" => idx >= 2,
                            // multiply_add: dest at 1, sources at 2,3,4
                            "multiply_add" | "vmultiply_add" => idx >= 2,
                            // vbroadcast: dest at 1, source at 2
                            "vbroadcast" => idx == 2,
                            // vselect: dest at 1, sources at 2,3,4
                            "vselect" => idx >= 2,
                            // load: dest at 1, addr at 2
                            "load" | "vload" => idx == 2,
                            // store: value at 1, addr at 2 (both are sources)
                            "store" | "vstore" => idx >= 1,
                            // const: dest at 1, value at 2 (value is immediate, not register)
                            "const" => false,  // No sources to rename
                            _ => idx >= 2,  // Default: assume dest at 1
                        };

                        if is_source {
                            if let Some(&fresh) = register_rename.get(reg) {
                                *reg = fresh;
                            }
                        }
                    }
                }

                // Rename in reads set
                let renamed_reads: HashSet<i64> = op.reads.iter()
                    .map(|r| *register_rename.get(r).unwrap_or(r))
                    .collect();
                op.reads = renamed_reads;
            }

            // Generate constant derivation ops
            let max_id = base_ops.iter().map(|op| op.id).max().unwrap_or(0);
            let const_ops = emit_constant_generation_ops(&plan, &value_to_register, max_id + 1);

            println!("Const optimization: {} loads removed, {} generation ops added, {} registers renamed (saves {})",
                     original_const_count, const_ops.len(), renames_count,
                     original_const_count.saturating_sub(const_ops.len()));

            // Insert constant ops at the beginning
            let mut new_ops = const_ops;
            new_ops.extend(base_ops);
            base_ops = new_ops;
        }
    }

    // Compute dependencies AFTER optimization (so new ALU ops get proper deps)
    let mut ops_for_deps = base_ops.clone();
    compute_dependencies(&mut ops_for_deps);

    // Copy the computed dep_ids back to base_ops
    let dep_ids_map: HashMap<usize, HashSet<usize>> = ops_for_deps.iter()
        .map(|op| (op.id, op.dep_ids.clone()))
        .collect();
    for op in &mut base_ops {
        if let Some(deps) = dep_ids_map.get(&op.id) {
            op.dep_ids = deps.clone();
        }
    }

    let dep_ids_map: HashMap<usize, HashSet<usize>> = ops_for_deps.iter()
        .map(|op| (op.id, op.dep_ids.clone()))
        .collect();

    let critical_path = compute_critical_path(&ops_for_deps);

    // Find best mode
    let mut best_cycles = usize::MAX;
    let mut best_ops: Option<Vec<Op>> = None;
    let mut best_expanded_ops: Vec<Op> = Vec::new();
    let mut best_mode: Option<PriorityMode> = None;

    for &mode in ALL_PRIORITY_MODES.iter() {
        let mut ops = base_ops.clone();
        for op in &mut ops {
            op.cycle = -1;
            op.dep_ids = dep_ids_map.get(&op.id).cloned().unwrap_or_default();
        }

        let result = schedule_greedy(&mut ops, &critical_path, enable_valu_shift, mode);

        if result.cycles < best_cycles {
            best_cycles = result.cycles;
            best_ops = Some(ops);
            best_expanded_ops = result.expanded_ops;
            best_mode = Some(mode);
        }
    }

    let best_mode = best_mode.unwrap();
    let ops = best_ops.unwrap();

    // Build cycle-by-cycle output
    let mut cycles_output: Vec<CycleOutput> = Vec::new();
    let mut total_empty_load = 0usize;
    let mut total_empty_store = 0usize;
    let mut total_empty_alu = 0usize;
    let mut total_empty_valu = 0usize;
    let mut total_load = 0usize;
    let mut total_store = 0usize;
    let mut total_alu = 0usize;
    let mut total_valu = 0usize;

    for cycle in 0..best_cycles {
        let mut load_ops: Vec<String> = Vec::new();
        let mut store_ops: Vec<String> = Vec::new();
        let mut alu_ops: Vec<String> = Vec::new();
        let mut valu_ops: Vec<String> = Vec::new();
        let mut flow_ops: Vec<String> = Vec::new();

        // Process regular ops, SKIP shifted_valu (these are replaced by expanded ALU ops)
        for op in &ops {
            if op.cycle == cycle as i32 && op.engine != "shifted_valu" {
                // Get the opcode for context-aware formatting
                let opcode = op.slot.get(0).map(|v| match v {
                    SlotVal::Str(s) => s.as_str(),
                    SlotVal::Int(_) => "",
                }).unwrap_or("");

                // Format as space-separated string: "op arg1 arg2 ..."
                let parts: Vec<String> = op.slot.iter().enumerate().map(|(idx, v)| match v {
                    SlotVal::Str(s) => s.clone(),
                    SlotVal::Int(n) => format_addr(*n, opcode, idx),
                }).collect();
                let compact = parts.join(" ");

                match op.engine.as_str() {
                    "load" => load_ops.push(compact),
                    "store" => store_ops.push(compact),
                    "alu" => alu_ops.push(compact),
                    "valu" => valu_ops.push(compact),
                    "flow" => flow_ops.push(compact),
                    _ => {}
                }
            }
        }

        // Process expanded ALU ops from VALU→ALU shifting
        for op in &best_expanded_ops {
            if op.cycle == cycle as i32 {
                let opcode = op.slot.get(0).map(|v| match v {
                    SlotVal::Str(s) => s.as_str(),
                    SlotVal::Int(_) => "",
                }).unwrap_or("");

                let parts: Vec<String> = op.slot.iter().enumerate().map(|(idx, v)| match v {
                    SlotVal::Str(s) => s.clone(),
                    SlotVal::Int(n) => format_addr(*n, opcode, idx),
                }).collect();
                let compact = parts.join(" ");

                // Expanded ops are always ALU
                alu_ops.push(compact);
            }
        }

        let load_empty = 2 - load_ops.len().min(2);
        let store_empty = 2 - store_ops.len().min(2);
        let alu_empty = 12 - alu_ops.len().min(12);
        let valu_empty = 6 - valu_ops.len().min(6);
        let flow_empty = 1 - flow_ops.len().min(1);

        total_load += load_ops.len();
        total_store += store_ops.len();
        total_alu += alu_ops.len();
        total_valu += valu_ops.len();
        total_empty_load += load_empty;
        total_empty_store += store_empty;
        total_empty_alu += alu_empty;
        total_empty_valu += valu_empty;

        cycles_output.push(CycleOutput {
            cycle,
            load: load_ops,
            load_empty,
            store: store_ops,
            store_empty,
            alu: alu_ops,
            alu_empty,
            valu: valu_ops,
            valu_empty,
            flow: flow_ops,
            flow_empty,
        });
    }

    let total_load_slots = best_cycles * 2;
    let total_store_slots = best_cycles * 2;
    let total_alu_slots = best_cycles * 12;
    let total_valu_slots = best_cycles * 6;

    let summary = ScheduleSummary {
        load_utilization_pct: (total_load as f64 / total_load_slots as f64) * 100.0,
        store_utilization_pct: (total_store as f64 / total_store_slots as f64) * 100.0,
        alu_utilization_pct: (total_alu as f64 / total_alu_slots as f64) * 100.0,
        valu_utilization_pct: (total_valu as f64 / total_valu_slots as f64) * 100.0,
        total_empty_load,
        total_empty_store,
        total_empty_alu,
        total_empty_valu,
    };

    let mut slot_limits = HashMap::new();
    slot_limits.insert("load".to_string(), 2);
    slot_limits.insert("store".to_string(), 2);
    slot_limits.insert("alu".to_string(), 12);
    slot_limits.insert("valu".to_string(), 6);
    slot_limits.insert("flow".to_string(), 1);

    let output = ScheduleOutput {
        total_cycles: best_cycles,
        total_ops: ops.len(),
        best_mode: format!("{:?}", best_mode),
        slot_limits,
        cycles: cycles_output,
        summary,
    };

    let json = serde_json::to_string_pretty(&output).unwrap_or_else(|_| "{}".to_string());

    // Write to file
    if !output_path.is_empty() {
        std::fs::write(&output_path, &json).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to write {}: {}", output_path, e))
        })?;
        println!("Wrote schedule to {}", output_path);
    }

    Ok(json)
}

/// Analyze constants and build derivation plan
#[pyfunction]
#[pyo3(signature = (instrs, hot_threshold=3))]
fn analyze_constants(py: Python<'_>, instrs: &Bound<'_, PyList>, hot_threshold: usize) -> PyResult<String> {
    let ops = parse_ops(py, instrs)?;

    // Analyze constant usage
    let usage = analyze_constant_usage(&ops);

    if usage.is_empty() {
        return Ok("No constants found".to_string());
    }

    // Build derivation plan
    let plan = build_constant_plan(&usage, hot_threshold);

    // Print detailed summary
    print_constant_plan_summary(&plan);

    // Return JSON summary
    let summary = serde_json::json!({
        "total_unique_constants": usage.len(),
        "bootstrap_loads": plan.bootstrap_loads,
        "total_derivations": plan.derivations.len(),
        "alu_ops_added": plan.total_alu_ops_added,
        "const_loads_saved": plan.total_const_loads_saved,
        "constant_pool": plan.constant_pool.iter().collect::<Vec<_>>(),
        "usage_by_value": usage.iter()
            .map(|(v, u)| (*v, serde_json::json!({
                "count": u.use_count,
                "destinations": u.destinations.len(),
            })))
            .collect::<HashMap<_, _>>(),
    });

    Ok(serde_json::to_string_pretty(&summary).unwrap_or_else(|_| "{}".to_string()))
}

/// Apply constant derivation optimization and return optimized ops
#[pyfunction]
#[pyo3(signature = (instrs, enable_valu_shift, hot_threshold=3, verbose=false))]
fn optimize_constants<'py>(
    py: Python<'py>,
    instrs: &Bound<'py, PyList>,
    enable_valu_shift: bool,
    hot_threshold: usize,
    verbose: bool,
) -> PyResult<Bound<'py, PyList>> {
    let mut ops = parse_ops(py, instrs)?;

    if ops.is_empty() {
        return Ok(PyList::empty(py));
    }

    // Analyze constant usage
    let usage = analyze_constant_usage(&ops);

    if usage.is_empty() {
        // No constants to optimize, return original
        let max_cycle = ops.iter().map(|op| op.cycle.max(0) as usize).max().unwrap_or(0);
        return emit_scheduled(py, &ops, max_cycle + 1);
    }

    // Build derivation plan
    let plan = build_constant_plan(&usage, hot_threshold);

    if verbose {
        print_constant_plan_summary(&plan);
    }

    // Allocate registers for constant values
    // Strategy: Use the first destination from each constant's usage
    let mut value_to_register: HashMap<i64, i64> = HashMap::new();
    for (&value, u) in &usage {
        if let Some(&dest) = u.destinations.iter().next() {
            value_to_register.insert(value, dest);
        }
    }

    // Remove all const load ops
    let const_op_ids: HashSet<usize> = ops.iter()
        .filter(|op| {
            op.engine == "load" &&
            op.slot.get(0).map(|s| matches!(s, SlotVal::Str(ref o) if o == "const")).unwrap_or(false)
        })
        .map(|op| op.id)
        .collect();

    ops.retain(|op| !const_op_ids.contains(&op.id));

    // Generate constant derivation ops
    let max_id = ops.iter().map(|op| op.id).max().unwrap_or(0);
    let const_ops = emit_constant_generation_ops(&plan, &value_to_register, max_id + 1);

    if verbose {
        println!("Generated {} constant ops (was {} loads)",
                 const_ops.len(), const_op_ids.len());
    }

    // Insert constant ops at the beginning
    let mut new_ops = const_ops;
    new_ops.extend(ops);

    // Compute dependencies
    compute_dependencies(&mut new_ops);

    // Schedule
    let critical_path = compute_critical_path(&new_ops);
    let result = schedule_greedy(
        &mut new_ops,
        &critical_path,
        enable_valu_shift,
        PriorityMode::CriticalPath,
    );

    if verbose {
        println!("Scheduled to {} cycles", result.cycles);
    }

    emit_scheduled_with_expanded(py, &new_ops, &result.expanded_ops, result.cycles)
}

#[pymodule]
fn scheduler_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(schedule, m)?)?;
    m.add_function(wrap_pyfunction!(schedule_cycles, m)?)?;
    m.add_function(wrap_pyfunction!(schedule_multipass, m)?)?;
    m.add_function(wrap_pyfunction!(schedule_multipass_cycles, m)?)?;
    m.add_function(wrap_pyfunction!(count_cycles, m)?)?;
    // Diagnostic functions
    m.add_function(wrap_pyfunction!(run_diagnostics, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_shared_temps, m)?)?;
    m.add_function(wrap_pyfunction!(print_diagnostics, m)?)?;
    m.add_function(wrap_pyfunction!(schedule_to_json, m)?)?;
    // Constant optimization functions
    m.add_function(wrap_pyfunction!(analyze_constants, m)?)?;
    m.add_function(wrap_pyfunction!(optimize_constants, m)?)?;
    Ok(())
}
