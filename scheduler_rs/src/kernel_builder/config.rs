//! Kernel configuration types.
//!
//! Defines the configuration parameters for kernel generation.

use crate::types::ConstEncodingFlags;

/// Hash stage definition: (op1, constant, op2, op3, shift)
/// Each stage computes: result = (val op1 constant) op2 (val op3 shift)
pub const HASH_STAGES: [(char, i64, char, char, i64); 6] = [
    ('+', 0x7ED55D16, '+', '<', 12), // << is '<' in our shorthand
    ('^', 0xC761C23C, '^', '>', 19), // >> is '>'
    ('+', 0x165667B1, '+', '<', 5),
    ('+', 0xD3A2646C, '^', '<', 9),
    ('+', 0xFD7046C5, '+', '<', 3),
    ('^', 0xB55A4F09, '^', '>', 16),
];

/// Stage 5 XOR constant used for optimization
pub const STAGE5_XOR_CONST: i64 = 0xB55A4F09;

/// Vector length for SIMD operations
pub const VLEN: usize = 8;

/// Maximum scratch space available
pub const SCRATCH_SIZE: usize = 1536;

/// Maximum tree depth for which we have tree-reduce emission (depth 3 = 8-way).
/// tree_reduce_max_depth is clamped to 0..=MAX_TREE_REDUCE_DEPTH.
pub const MAX_TREE_REDUCE_DEPTH: u32 = 3;

/// Kernel configuration parameters
#[derive(Debug, Clone)]
pub struct KernelConfig {
    /// Height of the forest (tree depth - 1)
    pub forest_height: usize,
    /// Number of nodes in the tree
    pub n_nodes: usize,
    /// Batch size (number of items to process)
    pub batch_size: usize,
    /// Number of rounds to execute
    pub rounds: usize,
    /// Enable VALU→ALU shifting during scheduling
    pub enable_valu_shift: bool,
    /// Enable debug operations
    pub enable_debug: bool,
    /// Enable lazy partition for sorted vload tree access (depths 4-8)
    /// When enabled, items are sorted by tree index via stable partition every k rounds,
    /// enabling sequential vloads instead of scattered loads.
    pub enable_lazy_partition: bool,
    /// Partition frequency: partition every k rounds (default: 1)
    /// k=1 means partition every round -> 2-way selection (trivial)
    /// k=2 means partition every 2 rounds -> 4-way selection
    /// k=3 means partition every 3 rounds -> 8-way selection
    /// k=1 is optimal because 2-way select is cheap and partition stores are FREE
    pub partition_frequency: usize,
    /// Enable tick-tock partition stores during rounds.
    /// When disabled (default), only final stores are emitted.
    /// This simplifies debugging but may increase register pressure.
    pub enable_partition_stores: bool,
    /// Enable pseudo-const mode for flexible constant generation.
    /// When enabled, constants are declared via need_const() and the scheduler
    /// dynamically chooses between Load, Add, Mul, etc. based on resource availability.
    /// This can reduce early cycle inefficiency by avoiding long derivation chains.
    pub enable_pseudo_consts: bool,

    // === Pseudo-const encoding flags ===
    /// Enable AddImm encoding via flow slot (add_imm dest, s0, value)
    /// Note: s0 (address 0) is always initialized to 0 at startup, so no precedence constraint needed
    pub enable_add_imm: bool,
    /// Enable Load encoding via load slot (const dest, value)
    pub enable_const_load: bool,
    /// Enable ALU-based derivation (Add, Sub, Mul, Shift)
    pub enable_const_alu: bool,
    /// Maximum derivation depth for ALU-based constant generation.
    /// Values requiring deeper derivation chains will use Load/AddImm instead.
    /// None means unlimited (default). Recommended: 4-6 to prevent long critical paths.
    pub max_alu_depth: Option<usize>,

    /// Maximum tree depth (0..=N) for which tree-reduce selection is used.
    /// Default 3 (8-way at depth 3). Minimum allowed is 2. Cannot exceed MAX_TREE_REDUCE_DEPTH (3).
    pub tree_reduce_max_depth: u32,

    /// Enable SSA dead code elimination before scheduling.
    /// When enabled (default), ops whose result has no users are removed,
    /// reducing cycles, optimizer time, and scratch pressure.
    pub enable_ssa_dce: bool,
}

impl Default for KernelConfig {
    fn default() -> Self {
        KernelConfig {
            forest_height: 10,
            n_nodes: 2047,
            batch_size: 256,
            rounds: 16,
            enable_valu_shift: true,
            enable_debug: false,
            enable_lazy_partition: false,
            partition_frequency: 1,
            enable_partition_stores: false,
            enable_pseudo_consts: false,
            // All encoding methods enabled by default
            enable_add_imm: true,
            enable_const_load: true,
            enable_const_alu: true,
            max_alu_depth: None, // No limit by default
            // Tree-reduce up to depth 3 by default (8-way at depth 3; min allowed is 2)
            tree_reduce_max_depth: 3,
            // SSA DCE on by default (removes dead code before scheduling)
            enable_ssa_dce: true,
        }
    }
}

impl KernelConfig {
    /// Create a new kernel configuration
    pub fn new(forest_height: usize, n_nodes: usize, batch_size: usize, rounds: usize) -> Self {
        KernelConfig {
            forest_height,
            n_nodes,
            batch_size,
            rounds,
            ..Default::default()
        }
    }

    /// Number of groups (batch_size / VLEN)
    pub fn n_groups(&self) -> usize {
        self.batch_size / VLEN
    }

    /// Number of iterations (ceil(n_groups / 8) so that 1..7 groups still get one iter)
    pub fn n_iters(&self) -> usize {
        let n = self.n_groups();
        if n == 0 {
            0
        } else {
            (n + 7) / 8 // UNROLL = 8; ceiling division
        }
    }

    /// Tree height (forest_height + 1)
    pub fn tree_height(&self) -> usize {
        self.forest_height + 1
    }

    /// Get encoding flags for pseudo-consts
    pub fn encoding_flags(&self) -> ConstEncodingFlags {
        ConstEncodingFlags {
            enable_add_imm: self.enable_add_imm,
            enable_load: self.enable_const_load,
            enable_alu: self.enable_const_alu,
            max_alu_depth: self.max_alu_depth,
        }
    }

    /// True if tree-reduce selection should be used at the given tree depth.
    /// Only depths 0..=MAX_TREE_REDUCE_DEPTH have tree-reduce emission; higher depths use fallback.
    /// tree_reduce_max_depth is clamped to 2..=MAX_TREE_REDUCE_DEPTH at the API boundary.
    pub fn use_tree_reduce_at_depth(&self, depth: usize) -> bool {
        depth <= self.tree_reduce_max_depth.min(MAX_TREE_REDUCE_DEPTH) as usize
    }
}

/// Check if a hash stage can use FMA (fused multiply-add) optimization
/// FMA is applicable when: op1='+', op2='+', op3='<<'
pub fn can_use_fma(stage: usize) -> bool {
    let (op1, _, op2, op3, _) = HASH_STAGES[stage];
    op1 == '+' && op2 == '+' && op3 == '<'
}

/// Get the FMA multiplier for a stage (1 + 2^shift)
pub fn fma_multiplier(stage: usize) -> Option<i64> {
    if can_use_fma(stage) {
        let (_, _, _, _, shift) = HASH_STAGES[stage];
        Some(1 + (1 << shift))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// n_iters must be ceiling(n_groups/8) so 1..7 groups get one iter.
    #[test]
    fn test_n_iters_ceiling_small_groups() {
        let cfg = KernelConfig {
            forest_height: 5,
            n_nodes: 31,
            batch_size: 8, // n_groups = 1
            rounds: 1,
            ..Default::default()
        };
        assert_eq!(cfg.n_groups(), 1, "batch 8 -> 1 group");
        assert_eq!(cfg.n_iters(), 1, "1 group must get 1 iter (ceiling)");
    }

    #[test]
    fn test_n_iters_ceiling_two_groups() {
        let cfg = KernelConfig {
            forest_height: 5,
            n_nodes: 31,
            batch_size: 16, // n_groups = 2
            rounds: 2,
            ..Default::default()
        };
        assert_eq!(cfg.n_groups(), 2);
        assert_eq!(cfg.n_iters(), 1, "2 groups -> ceil(2/8)=1");
    }

    #[test]
    fn test_n_iters_eight_groups() {
        let cfg = KernelConfig {
            forest_height: 5,
            n_nodes: 31,
            batch_size: 64, // n_groups = 8
            rounds: 4,
            ..Default::default()
        };
        assert_eq!(cfg.n_groups(), 8);
        assert_eq!(cfg.n_iters(), 1);
    }

    #[test]
    fn test_n_iters_nine_groups() {
        let cfg = KernelConfig {
            forest_height: 5,
            n_nodes: 31,
            batch_size: 72, // n_groups = 9
            rounds: 4,
            ..Default::default()
        };
        assert_eq!(cfg.n_groups(), 9);
        assert_eq!(cfg.n_iters(), 2, "9 groups -> ceil(9/8)=2");
    }
}
