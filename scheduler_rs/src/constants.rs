//! Constant analysis and derivation planning.
//!
//! This module handles optimization of constant loads by analyzing which constants
//! can be derived from others using ALU operations (add, multiply, xor, shift).

use std::collections::{HashMap, HashSet};

use crate::types::{Op, SlotVal};

/// A symbolic operand that may or may not have a physical register assigned
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[allow(dead_code)]
pub enum SymbolicOperand {
    /// Literal constant value (not yet materialized)
    Const(i64),
    /// Virtual register (will be allocated later)
    Virtual(usize),
    /// Physical register (already allocated or externally defined)
    Physical(i64),
    /// Reference to constant pool entry
    ConstPoolRef(i64), // value as key
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
    pub dest_register: Option<i64>, // Assigned after register allocation
    pub depth: usize,               // Depth in derivation tree (0 = bootstrap)
    pub is_bootstrap: bool,         // Must be loaded directly
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
pub fn analyze_constant_usage(ops: &[Op]) -> HashMap<i64, ConstantUsage> {
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
                        entry.first_use_cycle =
                            Some(entry.first_use_cycle.map_or(cycle, |c| c.min(cycle)));
                        entry.last_use_cycle =
                            Some(entry.last_use_cycle.map_or(cycle, |c| c.max(cycle)));
                    }
                }
            }
        }
    }

    usage
}

/// Check if a value can be derived from available values
pub fn can_derive(value: i64, available: &HashSet<i64>) -> Option<ConstantGeneration> {
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
pub fn build_constant_plan(
    usage: &HashMap<i64, ConstantUsage>,
    threshold_hot: usize,
) -> ConstantPlan {
    let mut plan = ConstantPlan::default();

    // Collect all needed constants
    let mut needed: Vec<i64> = usage.keys().copied().collect();
    needed.sort(); // Process in order for determinism

    // Identify large constants that MUST be loaded (can't derive)
    let large_threshold = 1_000_000i64;

    // Bootstrap: always need 1 (or smallest positive if no 1)
    let mut bootstrap: HashSet<i64> = HashSet::new();
    let mut available: HashSet<i64> = HashSet::new();

    // Add 1 to bootstrap if needed
    if needed.contains(&1) {
        bootstrap.insert(1);
        available.insert(1);
    } else if let Some(&smallest) = needed
        .iter()
        .filter(|&&v| v > 0 && v < large_threshold)
        .min()
    {
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
        derivations.insert(
            val,
            ConstantNode {
                value: val,
                generation: ConstantGeneration::DirectLoad,
                dest_register: None,
                depth: 0,
                is_bootstrap: true,
            },
        );
        generation_order.push(val);
    }

    // Process remaining constants in order of derivability
    let mut remaining: Vec<i64> = needed
        .iter()
        .filter(|v| !bootstrap.contains(v))
        .copied()
        .collect();

    let mut depth = 1;
    let max_depth = 10; // Prevent infinite loops

    while !remaining.is_empty() && depth < max_depth {
        let mut derived_this_round: Vec<i64> = Vec::new();

        for &val in &remaining {
            if let Some(gen) = can_derive(val, &available) {
                derivations.insert(
                    val,
                    ConstantNode {
                        value: val,
                        generation: gen,
                        dest_register: None,
                        depth,
                        is_bootstrap: false,
                    },
                );
                derived_this_round.push(val);
                available.insert(val);
            }
        }

        if derived_this_round.is_empty() {
            // Can't derive any more - add remaining to bootstrap
            for &val in &remaining {
                bootstrap.insert(val);
                available.insert(val);
                derivations.insert(
                    val,
                    ConstantNode {
                        value: val,
                        generation: ConstantGeneration::DirectLoad,
                        dest_register: None,
                        depth,
                        is_bootstrap: true,
                    },
                );
            }
            generation_order.extend(remaining.iter().copied());
            break;
        }

        generation_order.extend(derived_this_round.iter().copied());
        remaining.retain(|v| !derived_this_round.contains(v));
        depth += 1;
    }

    // Identify hot constants for constant pool
    let constant_pool: HashSet<i64> = usage
        .iter()
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
pub fn emit_constant_generation_ops(
    plan: &ConstantPlan,
    value_to_register: &HashMap<i64, i64>, // value -> physical register
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
            ConstantGeneration::DirectLoad => (
                "load".to_string(),
                vec![
                    SlotVal::Str("const".to_string()),
                    SlotVal::Int(dest),
                    SlotVal::Int(value),
                ],
            ),
            ConstantGeneration::XorSelf { source } => {
                let src_reg = value_to_register.get(source).copied().unwrap_or(*source);
                (
                    "alu".to_string(),
                    vec![
                        SlotVal::Str("^".to_string()),
                        SlotVal::Int(dest),
                        SlotVal::Int(src_reg),
                        SlotVal::Int(src_reg),
                    ],
                )
            }
            ConstantGeneration::Add { src1, src2 } => {
                let r1 = value_to_register.get(src1).copied().unwrap_or(*src1);
                let r2 = value_to_register.get(src2).copied().unwrap_or(*src2);
                (
                    "alu".to_string(),
                    vec![
                        SlotVal::Str("+".to_string()),
                        SlotVal::Int(dest),
                        SlotVal::Int(r1),
                        SlotVal::Int(r2),
                    ],
                )
            }
            ConstantGeneration::Mul { src1, src2 } => {
                let r1 = value_to_register.get(src1).copied().unwrap_or(*src1);
                let r2 = value_to_register.get(src2).copied().unwrap_or(*src2);
                (
                    "alu".to_string(),
                    vec![
                        SlotVal::Str("*".to_string()),
                        SlotVal::Int(dest),
                        SlotVal::Int(r1),
                        SlotVal::Int(r2),
                    ],
                )
            }
            ConstantGeneration::Shift { src, amount } => {
                let r1 = value_to_register.get(src).copied().unwrap_or(*src);
                // Need to load the shift amount into a register or use immediate
                // For now, we'll use a simpler approach: a << n = a * (1 << n)
                // This isn't ideal but works - TODO: use shift instruction
                let multiplier = 1i64 << amount;
                if let Some(&mult_reg) = value_to_register.get(&multiplier) {
                    (
                        "alu".to_string(),
                        vec![
                            SlotVal::Str("*".to_string()),
                            SlotVal::Int(dest),
                            SlotVal::Int(r1),
                            SlotVal::Int(mult_reg),
                        ],
                    )
                } else {
                    // Fall back to direct load
                    (
                        "load".to_string(),
                        vec![
                            SlotVal::Str("const".to_string()),
                            SlotVal::Int(dest),
                            SlotVal::Int(value),
                        ],
                    )
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
                if let Some(&r) = value_to_register.get(src1) {
                    reads.insert(r);
                }
                if let Some(&r) = value_to_register.get(src2) {
                    reads.insert(r);
                }
            }
            ConstantGeneration::Shift { src, .. } => {
                if let Some(&r) = value_to_register.get(src) {
                    reads.insert(r);
                }
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
pub fn print_constant_plan_summary(plan: &ConstantPlan) {
    println!("=== CONSTANT DERIVATION PLAN ===");
    println!("Bootstrap loads needed: {}", plan.bootstrap_loads.len());
    println!(
        "  Values: {:?}",
        &plan.bootstrap_loads[..plan.bootstrap_loads.len().min(10)]
    );
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
            let suffix = if vals.len() > 8 {
                format!("... ({} more)", vals.len() - 8)
            } else {
                String::new()
            };
            println!("  Depth {}: {:?}{}", depth, display, suffix);
        }
    }
}
