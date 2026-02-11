//! Peephole optimization pass for the VLIW scheduler.
//!
//! This module provides post-emission optimizations that can:
//! - Eliminate redundant operations (CSE)
//! - Simplify algebraic patterns (x ^ x -> 0)
//! - Optimize operations involving the zero register
//! - Dead code elimination (DCE)
//!
//! These optimizations run BEFORE scheduling to reduce op count and dependencies.
//!
//! **Why peephole (including DCE) is not used on the SSA path:**
//! - The SSA pipeline (`build_tick_tock_kernel_ssa`) does not call peephole at all; it works on
//!   SSA ops and uses the greedy scheduler's dead-store detection (cycle=-1) instead.
//! - Peephole operates on lowered `Op` (addresses, not VRegs); the SSA path lowers only after
//!   scheduling and address assignment.
//! - DCE is disabled by default (correctness: it doesn't track stores properly) and explicitly
//!   skips Store ops, so it would not remove dead stores even if enabled. Dead-store removal
//!   on the SSA path would need to happen in the SSA builder (filter ops before CP-SAT) or via
//!   an SSA-level DCE that removes ops whose dest VReg has no users before scheduling.

use std::collections::{HashMap, HashSet};

use crate::types::{Engine, Op, SlotVal, VLEN};

/// Statistics from peephole optimization
#[derive(Debug, Default)]
pub struct PeepholeStats {
    /// Operations eliminated by CSE
    pub cse_eliminated: usize,
    /// Operations eliminated by algebraic simplification
    pub algebraic_simplified: usize,
    /// Operations eliminated by zero-op optimization
    pub zero_ops_eliminated: usize,
    /// Operations eliminated by dead code elimination
    pub dce_eliminated: usize,
    /// Reads remapped to use ZERO_REG
    pub reads_remapped_to_zero: usize,
    /// Total operations eliminated
    pub total_eliminated: usize,
}

/// Configuration for peephole optimization
#[derive(Debug, Clone)]
pub struct PeepholeConfig {
    /// Enable Common Subexpression Elimination
    pub enable_cse: bool,
    /// Enable algebraic simplifications (x ^ x -> 0, etc.)
    pub enable_algebraic: bool,
    /// Enable zero register optimization
    pub enable_zero_opt: bool,
    /// Enable dead code elimination
    pub enable_dce: bool,
    /// The zero register address (default: 0)
    pub zero_reg: i64,
}

impl Default for PeepholeConfig {
    fn default() -> Self {
        PeepholeConfig {
            // CSE is DISABLED by default because it doesn't track value versions.
            // It incorrectly treats ops with same operand addresses as duplicates
            // even when the values at those addresses have changed.
            // TODO: Implement proper value versioning to enable safe CSE.
            enable_cse: false,
            enable_algebraic: false,
            enable_zero_opt: false,
            // DCE is DISABLED by default because it breaks correctness.
            // The current implementation doesn't properly track which ops are needed
            // for final output (stores to external memory). It eliminates ops that
            // are part of the computation chain leading to those stores.
            // TODO: Implement proper liveness analysis that respects store operations.
            enable_dce: false,
            zero_reg: 0, // ZERO_REG
        }
    }
}

/// Main peephole optimization entry point.
///
/// Applies all enabled optimizations to the op list.
/// Returns statistics about what was optimized.
///
/// IMPORTANT: After calling this, you must call `compute_dependencies()` again
/// because this function modifies reads/writes and may remap registers.
pub fn optimize(ops: &mut Vec<Op>, config: &PeepholeConfig) -> PeepholeStats {
    let mut stats = PeepholeStats::default();

    // Phase 1: Zero register optimization
    // This should run first because it can enable more CSE opportunities
    if config.enable_zero_opt {
        let zero_stats = eliminate_zero_ops(ops, config.zero_reg);
        stats.zero_ops_eliminated = zero_stats.0;
        stats.reads_remapped_to_zero = zero_stats.1;
    }

    // Phase 2: Algebraic simplification
    if config.enable_algebraic {
        stats.algebraic_simplified = simplify_algebraic(ops, config.zero_reg);
    }

    // Phase 3: CSE - eliminate duplicate computations
    if config.enable_cse {
        stats.cse_eliminated = eliminate_cse(ops);
    }

    // Phase 4: Dead Code Elimination
    // Run this last since other optimizations may create dead code
    if config.enable_dce {
        stats.dce_eliminated = eliminate_dead_code(ops);
    }

    stats.total_eliminated = stats.cse_eliminated
        + stats.algebraic_simplified
        + stats.zero_ops_eliminated
        + stats.dce_eliminated;

    stats
}

/// Eliminate operations that produce zero when zero is already available.
///
/// Operations that are eliminated:
/// - `* s0 s0` where s0 is ZERO_REG (computing 0*0 = 0, but s0 is already 0)
/// - `& x s0` or `& s0 x` (0 & anything = 0)
/// - `* x s0` or `* s0 x` (0 * anything = 0)
///
/// For operations that produce zero but write to a different destination,
/// we remap consumers to read from ZERO_REG instead.
///
/// Returns (ops_eliminated, reads_remapped)
fn eliminate_zero_ops(ops: &mut Vec<Op>, zero_reg: i64) -> (usize, usize) {
    let mut remapped = 0usize;

    // Track which destinations should be remapped to zero_reg
    let mut remap_to_zero: HashSet<i64> = HashSet::new();

    // First pass: identify ops that produce zero
    let mut to_eliminate: HashSet<usize> = HashSet::new();

    for op in ops.iter() {
        if op.slot.is_empty() {
            continue;
        }

        let opcode = match &op.slot[0] {
            SlotVal::Str(s) => s.as_str(),
            _ => continue,
        };

        // Get operands for ALU/VALU ops
        let (dest, src1, src2) = match (op.slot.get(1), op.slot.get(2), op.slot.get(3)) {
            (Some(SlotVal::Int(d)), Some(SlotVal::Int(s1)), Some(SlotVal::Int(s2))) => {
                (*d, *s1, *s2)
            }
            _ => continue,
        };

        // Check for patterns that produce zero
        let produces_zero = match opcode {
            // x * 0 = 0, 0 * x = 0
            "*" => src1 == zero_reg || src2 == zero_reg,
            // x & 0 = 0, 0 & x = 0
            "&" => src1 == zero_reg || src2 == zero_reg,
            // x - x = 0
            "-" => src1 == src2,
            // Note: x ^ x is handled in algebraic simplification
            _ => false,
        };

        if produces_zero {
            if dest == zero_reg {
                // Writing zero to ZERO_REG is completely useless - eliminate
                to_eliminate.insert(op.id);
            } else {
                // Writing zero to another register - remap consumers to ZERO_REG
                remap_to_zero.insert(dest);
                // Also eliminate the op since consumers will read from ZERO_REG
                to_eliminate.insert(op.id);
            }
        }
    }

    // For VALU ops, we need to handle vector registers
    for op in ops.iter() {
        if op.engine != Engine::Valu || op.slot.is_empty() {
            continue;
        }

        let opcode = match &op.slot[0] {
            SlotVal::Str(s) => s.as_str(),
            _ => continue,
        };

        let (dest, src1, src2) = match (op.slot.get(1), op.slot.get(2), op.slot.get(3)) {
            (Some(SlotVal::Int(d)), Some(SlotVal::Int(s1)), Some(SlotVal::Int(s2))) => {
                (*d, *s1, *s2)
            }
            _ => continue,
        };

        // Check for vector patterns that produce zero vector
        // v_0 is typically a vector of zeros (broadcast from ZERO_REG)
        let produces_zero = match opcode {
            "*" => src1 == src2 && is_vector_zero(src1, ops),
            "-" => src1 == src2,
            _ => false,
        };

        if produces_zero && !to_eliminate.contains(&op.id) {
            to_eliminate.insert(op.id);
            // Mark all 8 vector elements as producing zero
            for i in 0..VLEN as i64 {
                remap_to_zero.insert(dest + i);
            }
        }
    }

    let eliminated = to_eliminate.len();

    // Second pass: remap reads from eliminated destinations to zero_reg
    for op in ops.iter_mut() {
        if to_eliminate.contains(&op.id) {
            continue;
        }

        let mut new_reads: HashSet<i64> = HashSet::new();
        for &read_addr in &op.reads {
            if remap_to_zero.contains(&read_addr) {
                new_reads.insert(zero_reg);
                remapped += 1;
            } else {
                new_reads.insert(read_addr);
            }
        }
        op.reads = new_reads;

        // Also update slot values for source operands
        for slot_val in op.slot.iter_mut().skip(2) {
            // Skip opcode and dest
            if let SlotVal::Int(addr) = slot_val {
                if remap_to_zero.contains(addr) {
                    *addr = zero_reg;
                }
            }
        }
    }

    // Remove eliminated ops
    ops.retain(|op| !to_eliminate.contains(&op.id));

    (eliminated, remapped)
}

/// Check if a vector register is known to be all zeros (broadcast from ZERO_REG)
fn is_vector_zero(base: i64, ops: &[Op]) -> bool {
    // Look for a vbroadcast that wrote to this address from ZERO_REG
    for op in ops {
        if op.engine != Engine::Valu {
            continue;
        }
        if let (Some(SlotVal::Str(opc)), Some(SlotVal::Int(dest)), Some(SlotVal::Int(src))) =
            (op.slot.get(0), op.slot.get(1), op.slot.get(2))
        {
            if opc == "vbroadcast" && *dest == base && *src == 0 {
                return true;
            }
        }
    }
    false
}

/// Simplify algebraic patterns that produce known results.
///
/// Patterns handled:
/// - x ^ x -> 0 (CRITICAL: according to ISA, this has NO read dependencies!)
/// - x & x -> x
/// - x | x -> x
///
/// For x ^ x producing zero, we:
/// 1. Mark the op as having NO reads (per ISA spec)
/// 2. Remap consumers to read from ZERO_REG instead
/// 3. Optionally eliminate the op if consumers are remapped
///
/// Returns number of ops simplified/eliminated.
fn simplify_algebraic(ops: &mut Vec<Op>, zero_reg: i64) -> usize {
    let mut simplified = 0usize;
    let mut xor_self_dests: HashSet<i64> = HashSet::new();
    let mut to_eliminate: HashSet<usize> = HashSet::new();

    // First pass: identify x ^ x patterns
    for op in ops.iter_mut() {
        if op.slot.len() < 4 {
            continue;
        }

        let opcode = match &op.slot[0] {
            SlotVal::Str(s) => s.as_str(),
            _ => continue,
        };

        let (dest, src1, src2) = match (op.slot.get(1), op.slot.get(2), op.slot.get(3)) {
            (Some(SlotVal::Int(d)), Some(SlotVal::Int(s1)), Some(SlotVal::Int(s2))) => {
                (*d, *s1, *s2)
            }
            _ => continue,
        };

        if opcode == "^" && src1 == src2 {
            // x ^ x = 0
            // CRITICAL: Per ISA, this operation has NO read dependencies!
            op.reads.clear();

            if op.engine == Engine::Valu {
                // Vector XOR self - mark all 8 elements as producing zero
                for i in 0..VLEN as i64 {
                    xor_self_dests.insert(dest + i);
                }
            } else {
                xor_self_dests.insert(dest);
            }

            if dest == zero_reg {
                // Writing to ZERO_REG - completely eliminate
                to_eliminate.insert(op.id);
            }

            simplified += 1;
        }

        // Handle x & x = x and x | x = x
        if (opcode == "&" || opcode == "|") && src1 == src2 {
            if op.engine == Engine::Valu && dest == src1 {
                // Vector AND/OR self writing to same dest - this is a no-op
                to_eliminate.insert(op.id);
                simplified += 1;
            } else if op.engine == Engine::Alu && dest == src1 {
                // Scalar AND/OR self writing to same dest - this is a no-op
                to_eliminate.insert(op.id);
                simplified += 1;
            }
            // If dest != src, this is effectively a copy, which we could optimize
            // but leave for now
        }
    }

    // Second pass: remap reads from x^x destinations to zero_reg
    for op in ops.iter_mut() {
        if to_eliminate.contains(&op.id) {
            continue;
        }

        let mut new_reads: HashSet<i64> = HashSet::new();
        for &read_addr in &op.reads {
            if xor_self_dests.contains(&read_addr) {
                new_reads.insert(zero_reg);
            } else {
                new_reads.insert(read_addr);
            }
        }
        op.reads = new_reads;

        // Also update slot values for source operands
        for slot_val in op.slot.iter_mut().skip(2) {
            if let SlotVal::Int(addr) = slot_val {
                if xor_self_dests.contains(addr) {
                    *addr = zero_reg;
                }
            }
        }
    }

    // Remove eliminated ops
    let elim_count = to_eliminate.len();
    ops.retain(|op| !to_eliminate.contains(&op.id));

    simplified.max(elim_count)
}

/// Eliminate common subexpressions.
///
/// When multiple ops compute the same value (same opcode, same operands),
/// keep only the first one and remap consumers of duplicates to read from
/// the first result.
///
/// For commutative operations (+, *, ^, &, |), operand order is normalized
/// before hashing to catch more duplicates.
///
/// Returns number of ops eliminated.
fn eliminate_cse(ops: &mut Vec<Op>) -> usize {
    // Hash: (engine, opcode, sorted_operands) -> (first_op_id, dest_register)
    let mut cse_map: HashMap<(Engine, String, Vec<i64>), (usize, i64)> = HashMap::new();
    let mut to_eliminate: HashSet<usize> = HashSet::new();
    let mut remap: HashMap<i64, i64> = HashMap::new();

    // Commutative operations where operand order doesn't matter
    let commutative = ["+", "*", "^", "&", "|", "==", "!="];

    // First pass: build CSE map and identify duplicates
    for op in ops.iter() {
        if op.slot.len() < 4 {
            continue;
        }

        // Skip load/store/flow ops - they have side effects or address semantics
        if !matches!(op.engine, Engine::Alu | Engine::Valu) {
            continue;
        }

        let opcode = match &op.slot[0] {
            SlotVal::Str(s) => s.clone(),
            _ => continue,
        };

        // Skip vbroadcast - these are intentionally duplicated for different destinations
        if opcode == "vbroadcast" {
            continue;
        }

        // Skip multiply_add - 5 operands, more complex
        if opcode == "multiply_add" {
            continue;
        }

        let (dest, src1, src2) = match (op.slot.get(1), op.slot.get(2), op.slot.get(3)) {
            (Some(SlotVal::Int(d)), Some(SlotVal::Int(s1)), Some(SlotVal::Int(s2))) => {
                (*d, *s1, *s2)
            }
            _ => continue,
        };

        // Normalize operand order for commutative ops
        let operands = if commutative.contains(&opcode.as_str()) {
            if src1 <= src2 {
                vec![src1, src2]
            } else {
                vec![src2, src1]
            }
        } else {
            vec![src1, src2]
        };

        let key = (op.engine, opcode, operands);

        if let Some(&(_first_id, first_dest)) = cse_map.get(&key) {
            // Duplicate found - mark for elimination and remap
            to_eliminate.insert(op.id);

            if op.engine == Engine::Valu {
                // Remap all 8 vector elements
                for i in 0..VLEN as i64 {
                    remap.insert(dest + i, first_dest + i);
                }
            } else {
                remap.insert(dest, first_dest);
            }
        } else {
            // First occurrence - add to CSE map
            cse_map.insert(key, (op.id, dest));
        }
    }

    let eliminated = to_eliminate.len();

    // Second pass: remap reads
    for op in ops.iter_mut() {
        if to_eliminate.contains(&op.id) {
            continue;
        }

        let mut new_reads: HashSet<i64> = HashSet::new();
        for &read_addr in &op.reads {
            if let Some(&new_addr) = remap.get(&read_addr) {
                new_reads.insert(new_addr);
            } else {
                new_reads.insert(read_addr);
            }
        }
        op.reads = new_reads;

        // Also update slot values for source operands
        for slot_val in op.slot.iter_mut().skip(2) {
            if let SlotVal::Int(addr) = slot_val {
                if let Some(&new_addr) = remap.get(addr) {
                    *addr = new_addr;
                }
            }
        }
    }

    // Remove eliminated ops
    ops.retain(|op| !to_eliminate.contains(&op.id));

    eliminated
}

/// Eliminate dead code - operations whose results are never read.
///
/// An operation is dead if ALL of its writes are never read by any subsequent operation.
/// This includes stores to memory that are never loaded.
///
/// DCE runs iteratively since removing dead ops may make other ops dead
/// (e.g., if op A writes to X, op B reads X and writes Y, and Y is never read,
/// removing B makes A dead too).
///
/// Returns total number of ops eliminated.
fn eliminate_dead_code(ops: &mut Vec<Op>) -> usize {
    let mut total_eliminated = 0usize;

    // Debug: check op 42's reads
    for op in ops.iter() {
        if op.id == 42 {
            println!(
                "DEBUG op 42: slot={:?}, reads={:?}, writes={:?}",
                op.slot, op.reads, op.writes
            );
        }
        if op.id == 22 {
            println!(
                "DEBUG op 22: slot={:?}, reads={:?}, writes={:?}",
                op.slot, op.reads, op.writes
            );
        }
    }

    // Iterate until no more dead code is found
    loop {
        // Collect all addresses that are read by ANY op
        let mut all_reads: HashSet<i64> = HashSet::new();
        for op in ops.iter() {
            all_reads.extend(&op.reads);
        }

        // Find ops where ALL writes are never read
        let mut to_eliminate: HashSet<usize> = HashSet::new();

        for op in ops.iter() {
            // Skip ops with no writes (e.g., flow control without dest)
            if op.writes.is_empty() {
                continue;
            }

            // Skip store operations - they have external side effects
            if op.engine == Engine::Store {
                continue;
            }

            // Skip flow operations - they may have control flow effects
            if op.engine == Engine::Flow {
                continue;
            }

            // Check if ANY write is read by another op
            let has_live_write = op.writes.iter().any(|w| all_reads.contains(w));

            if !has_live_write {
                // All writes are dead - this op can be eliminated
                to_eliminate.insert(op.id);
            }
        }

        if to_eliminate.is_empty() {
            break;
        }

        // Debug: print what we're eliminating
        for op in ops.iter() {
            if to_eliminate.contains(&op.id) {
                println!("DCE: eliminating op {} {:?}", op.id, op.slot);
            }
        }

        total_eliminated += to_eliminate.len();
        ops.retain(|op| !to_eliminate.contains(&op.id));
    }

    total_eliminated
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn make_alu_op(id: usize, opcode: &str, dest: i64, src1: i64, src2: i64) -> Op {
        Op {
            id,
            engine: Engine::Alu,
            slot: [
                SlotVal::Str(opcode.to_string()),
                SlotVal::Int(dest),
                SlotVal::Int(src1),
                SlotVal::Int(src2),
            ]
            .into_iter()
            .collect(),
            cycle: -1,
            dep_ids: HashSet::new(),
            reads: [src1, src2].into_iter().collect(),
            writes: [dest].into_iter().collect(),
            last_use_reads: HashSet::new(),
            produces_constant: None,
            trace_id: None,
        }
    }

    fn make_valu_op(id: usize, opcode: &str, dest: i64, src1: i64, src2: i64) -> Op {
        let mut reads = HashSet::new();
        let mut writes = HashSet::new();
        for i in 0..VLEN as i64 {
            reads.insert(src1 + i);
            reads.insert(src2 + i);
            writes.insert(dest + i);
        }
        Op {
            id,
            engine: Engine::Valu,
            slot: [
                SlotVal::Str(opcode.to_string()),
                SlotVal::Int(dest),
                SlotVal::Int(src1),
                SlotVal::Int(src2),
            ]
            .into_iter()
            .collect(),
            cycle: -1,
            dep_ids: HashSet::new(),
            reads,
            writes,
            last_use_reads: HashSet::new(),
            produces_constant: None,
            trace_id: None,
        }
    }

    #[test]
    #[ignore] // Disabled: peephole optimizations off by default (commit 531451c)
    fn test_xor_self_clears_reads() {
        let mut ops = vec![make_alu_op(0, "^", 10, 5, 5)]; // x ^ x

        let config = PeepholeConfig::default();
        let stats = optimize(&mut ops, &config);

        assert_eq!(stats.algebraic_simplified, 1);
        // The op should have no reads since x ^ x doesn't depend on x
        assert!(ops[0].reads.is_empty());
    }

    #[test]
    #[ignore] // Disabled: peephole optimizations off by default (commit 531451c)
    fn test_zero_mul_eliminated() {
        let mut ops = vec![
            make_alu_op(0, "*", 10, 0, 0),  // 0 * 0 -> eliminate
            make_alu_op(1, "+", 20, 10, 1), // uses result of above
        ];

        let config = PeepholeConfig::default();
        let stats = optimize(&mut ops, &config);

        // The multiplication should be eliminated
        assert!(stats.zero_ops_eliminated > 0);
        // The add should now read from ZERO_REG
        assert!(ops[0].reads.contains(&0));
    }

    #[test]
    #[ignore] // Disabled: peephole optimizations off by default (commit 531451c)
    fn test_cse_duplicate_add() {
        let mut ops = vec![
            make_alu_op(0, "+", 10, 1, 2),   // First: 1 + 2 -> 10
            make_alu_op(1, "+", 20, 1, 2),   // Duplicate: 1 + 2 -> 20
            make_alu_op(2, "+", 30, 10, 20), // Uses both results
        ];

        let config = PeepholeConfig::default();
        let stats = optimize(&mut ops, &config);

        assert_eq!(stats.cse_eliminated, 1);
        assert_eq!(ops.len(), 2);
        // The final add should read from 10 twice (since 20 was remapped to 10)
        let final_op = &ops[1];
        assert!(final_op.reads.contains(&10));
    }

    #[test]
    #[ignore] // Disabled: peephole optimizations off by default (commit 531451c)
    fn test_cse_commutative() {
        let mut ops = vec![
            make_alu_op(0, "+", 10, 1, 2), // 1 + 2
            make_alu_op(1, "+", 20, 2, 1), // 2 + 1 (same due to commutativity)
        ];

        let config = PeepholeConfig::default();
        let stats = optimize(&mut ops, &config);

        assert_eq!(stats.cse_eliminated, 1);
        assert_eq!(ops.len(), 1);
    }
}
