//! Kernel configuration types.
//!
//! Defines the configuration parameters for kernel generation.

/// Hash stage definition: (op1, constant, op2, op3, shift)
/// Each stage computes: result = (val op1 constant) op2 (val op3 shift)
pub const HASH_STAGES: [(char, i64, char, char, i64); 6] = [
    ('+', 0x7ED55D16, '+', '<', 12),  // << is '<' in our shorthand
    ('^', 0xC761C23C, '^', '>', 19),  // >> is '>'
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
    /// Partition frequency: partition every k rounds (default: 3)
    /// Lower k = more accurate sorting but higher overhead
    /// k=2 or k=3 recommended for depths 4-8
    pub partition_frequency: usize,
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
            partition_frequency: 3,
        }
    }
}

impl KernelConfig {
    /// Create a new kernel configuration
    pub fn new(
        forest_height: usize,
        n_nodes: usize,
        batch_size: usize,
        rounds: usize,
    ) -> Self {
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

    /// Number of iterations (n_groups / UNROLL)
    pub fn n_iters(&self) -> usize {
        self.n_groups() / 8  // UNROLL = 8
    }

    /// Tree height (forest_height + 1)
    pub fn tree_height(&self) -> usize {
        self.forest_height + 1
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
