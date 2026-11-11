//! Kernel configuration types.
//!
//! Defines the configuration parameters for kernel generation, split into:
//! - `ProblemSpec` — emission concern: what program to generate
//! - `CompilerConfig` — compilation concern: how to optimize/schedule
//! - `KernelConfig` — combined wrapper for backward compatibility

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

/// Maximum tree depth for which we have tree-reduce emission (depth 4 = 16-way).
/// tree_reduce_max_depth is clamped to 0..=MAX_TREE_REDUCE_DEPTH.
pub const MAX_TREE_REDUCE_DEPTH: u32 = 4;

/// How the kernel represents each lane's tree index between rounds.
///
/// The reference index `idx` at tree depth `d` lies in `[2^d - 1, 2^(d+1) - 2]`
/// (`R7-index-range-by-depth`). `Mirrored` tracks `n = L_d - idx` with
/// `L_d = 2^(d+1) - 2` (`level_last_index`): the node's position in its level counted from
/// the right. Proofs: `R9` to `R11` in `verification/rewrites.py`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IndexForm {
    /// The reference index. Update per round: AND, SUB, multiply_add (3 vector ops).
    #[default]
    ZeroBased,
    /// `n = L_d - idx`. Update per round: AND, multiply_add (2 vector ops); none of the
    /// two at depth 0, where `n` is the branch bit itself, and nothing in the wrap round.
    /// The initial indices are taken to be 0 (as the root broadcast in round 0 already
    /// assumes), so they are not loaded; the final indices are stored as `L_d - n`.
    Mirrored,
}

impl IndexForm {
    /// Parse the CLI / binding spelling.
    ///
    /// # Arguments
    /// * `s` - `"zero-based"` or `"mirrored"`.
    ///
    /// # Returns
    /// The form, or an error message naming the valid spellings.
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "zero-based" => Ok(IndexForm::ZeroBased),
            "mirrored" => Ok(IndexForm::Mirrored),
            other => Err(format!(
                "Invalid index_form '{}': must be 'zero-based' or 'mirrored'",
                other
            )),
        }
    }
}

/// How the node value is selected on the cached tree levels (depth 0 up to
/// `tree_reduce_max_depth`).
///
/// At depth `d` the node is `tree[L_d - n]` with `n = b_0 b_1 .. b_(d-1)` in binary, where
/// `b_r` is the branch bit of the round at depth `r` (`R9`, `R13`). `Bits`, `BitsFlow` and
/// `Mobius` select with those bits, from node constants that already carry the deferred
/// stage-5 constant, so there is no per-group compare, root broadcast or XOR with the
/// constant. Proofs: `R13`, `R14` in `verification/rewrites.py`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CachedSelect {
    /// Compare or extract bits from the index, select, then XOR the deferred constant.
    #[default]
    Standard,
    /// Binary tree of `2^d - 1` two-way selects driven by the branch bits. The `2^(d-1)`
    /// leaf selects are multiply_adds with a precomputed delta; the selects above them are
    /// `SelectChoice`s (flow `vselect`, or SUB + multiply_add). Uses the flow engine exactly
    /// as much as `Standard` does.
    Bits,
    /// As `Bits`, but the leaf selects are `SelectChoice`s too (flow `vselect`, or one
    /// multiply_add): every select can go to the flow engine, which gives the lowest
    /// vector-op count but needs a scheduler that spreads the flow work.
    BitsFlow,
    /// `2^d - 1` multiply_adds evaluating the multilinear (Mobius) form of the node table
    /// in the branch bits. No flow ops and no choice for the scheduler.
    Mobius,
}

impl CachedSelect {
    /// Parse the CLI / binding spelling.
    ///
    /// # Arguments
    /// * `s` - `"standard"`, `"bits"`, `"bits-flow"` or `"mobius"`.
    ///
    /// # Returns
    /// The mode, or an error message naming the valid spellings.
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "standard" => Ok(CachedSelect::Standard),
            "bits" => Ok(CachedSelect::Bits),
            "bits-flow" => Ok(CachedSelect::BitsFlow),
            "mobius" => Ok(CachedSelect::Mobius),
            other => Err(format!(
                "Invalid cached_select '{}': must be 'standard', 'bits', 'bits-flow' or 'mobius'",
                other
            )),
        }
    }
}

/// Where `CachedSelect::Bits` / `BitsFlow` / `Mobius` take the branch bits of earlier rounds from.
///
/// The newest bit (the round just before the select) always comes straight from the branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SelectBitSource {
    /// Extract the older bits from the index as it was before the last update
    /// (`n = b_0 .. b_(d-2)` in binary): `2 (d - 2)` vector ops per group at depth `d >= 2`
    /// (one more to mirror a zero-based index at depth `d >= 3`), and no bit vector stays
    /// live across rounds. Proof: `R15-bits-from-index`.
    #[default]
    Index,
    /// Keep every branch bit live until the select: no extraction ops, but 8 scratch words
    /// per group for each earlier round, which the schedulers may not be able to fit.
    Path,
}

impl SelectBitSource {
    /// Parse the CLI / binding spelling.
    ///
    /// # Arguments
    /// * `s` - `"index"` or `"path"`.
    ///
    /// # Returns
    /// The source, or an error message naming the valid spellings.
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "index" => Ok(SelectBitSource::Index),
            "path" => Ok(SelectBitSource::Path),
            other => Err(format!(
                "Invalid select_bit_source '{}': must be 'index' or 'path'",
                other
            )),
        }
    }
}

/// Largest tree index at depth `depth`: `L_d = 2^(d+1) - 2`.
///
/// # Arguments
/// * `depth` - Tree depth (0 = root).
///
/// # Returns
/// `L_d`; `L_0 = 0` and `L_(d+1) = 2 L_d + 2` (`R11-mirrored-index-range`).
pub fn level_last_index(depth: usize) -> i64 {
    (1i64 << (depth + 1)) - 2
}

/// Emission concern: what program to generate.
///
/// Controls problem dimensions and emission behavior — which ops are
/// emitted, what constant-encoding alternatives are generated, etc.
#[derive(Debug, Clone)]
pub struct ProblemSpec {
    /// Height of the forest (tree depth - 1)
    pub forest_height: usize,
    /// Number of nodes in the tree
    pub n_nodes: usize,
    /// Batch size (number of items to process)
    pub batch_size: usize,
    /// Number of rounds to execute
    pub rounds: usize,
    /// Maximum tree depth (0..=N) for which tree-reduce selection is used.
    /// Default 3 (8-way at depth 3). Minimum allowed is 2. Cannot exceed MAX_TREE_REDUCE_DEPTH (4).
    pub tree_reduce_max_depth: u32,
    /// Enable debug operations
    pub enable_debug: bool,
    /// Enable trace IDs for semantic debugging (trace_input/trace_output)
    pub enable_trace_ids: bool,
    /// Enable lazy partition for sorted vload tree access (depths 4-8)
    pub enable_lazy_partition: bool,
    /// Partition frequency: partition every k rounds (default: 1)
    pub partition_frequency: usize,
    /// Enable tick-tock partition stores during rounds.
    pub enable_partition_stores: bool,
    /// Enable pseudo-const mode for flexible constant generation.
    pub enable_pseudo_consts: bool,
    // === Pseudo-const encoding flags ===
    /// Enable AddImm encoding via flow slot (add_imm dest, s0, value)
    pub enable_add_imm: bool,
    /// Enable Load encoding via load slot (const dest, value)
    pub enable_const_load: bool,
    /// Enable ALU-based derivation (Add, Sub, Mul, Shift)
    pub enable_const_alu: bool,
    /// Maximum derivation depth for ALU-based constant generation.
    pub max_alu_depth: Option<usize>,
    /// Emit hash stages 2 and 3 as two multiply_adds and an XOR (3 vector ops) instead of
    /// multiply_add, add, shift, XOR (4). Proved by `R8-hash-stage-merge` in
    /// `verification/rewrites.py`. Default false.
    pub merge_hash_stages: bool,
    /// Representation of the tree index between rounds. Default `IndexForm::ZeroBased`.
    pub index_form: IndexForm,
    /// Do not emit index work whose result is known (`IndexForm::ZeroBased` only; the
    /// mirrored form has none): at depth 0 every index is 0, so the update is its offset
    /// (no multiply_add) and the initial indices are not loaded; the wrap round shares one
    /// vector of zeros instead of one broadcast per group. Output is unchanged. Proof:
    /// `R12-index-update-from-root`. Default false.
    pub prune_dead_index: bool,
    /// Do not store the final indices. The kernel then leaves the index memory as it found
    /// it, and everything that only fed the index store (the last round's update, and any
    /// index chain after the last gather) is removed by dead code elimination. **Changes
    /// the output**: only the values are correct. Default false.
    pub values_only: bool,
    /// Selection of the node value on the cached levels. Default `CachedSelect::Standard`.
    pub cached_select: CachedSelect,
    /// Source of the older branch bits for `CachedSelect::Bits` / `BitsFlow` / `Mobius`. Default
    /// `SelectBitSource::Index`. Ignored for `CachedSelect::Standard`.
    pub select_bit_source: SelectBitSource,
    /// With `tree_reduce_max_depth = 4`: select among the 16 cached level-4 nodes only on
    /// visits after the first wrap (round 15 of 16), and gather level 4 on the first pass
    /// (round 4). Selecting costs 15 selects per group and saves 8 loads per group; after
    /// the wrap the loads are what binds. Default false (select on every visit).
    pub depth4_select_after_wrap_only: bool,
}

impl Default for ProblemSpec {
    fn default() -> Self {
        ProblemSpec {
            forest_height: 10,
            n_nodes: 2047,
            batch_size: 256,
            rounds: 16,
            tree_reduce_max_depth: 3,
            enable_debug: false,
            enable_trace_ids: false,
            enable_lazy_partition: false,
            partition_frequency: 1,
            enable_partition_stores: false,
            enable_pseudo_consts: false,
            enable_add_imm: true,
            enable_const_load: true,
            enable_const_alu: true,
            max_alu_depth: None,
            merge_hash_stages: false,
            index_form: IndexForm::ZeroBased,
            prune_dead_index: false,
            values_only: false,
            cached_select: CachedSelect::Standard,
            select_bit_source: SelectBitSource::Index,
            depth4_select_after_wrap_only: false,
        }
    }
}

impl ProblemSpec {
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
    pub fn use_tree_reduce_at_depth(&self, depth: usize) -> bool {
        depth <= self.tree_reduce_max_depth.min(MAX_TREE_REDUCE_DEPTH) as usize
    }

    /// True if round `rnd`, which visits `tree_depth`, selects its nodes among cached
    /// values instead of gathering them.
    ///
    /// # Arguments
    /// * `rnd` - Round number (0-based).
    /// * `tree_depth` - Depth the indices are at in that round.
    ///
    /// # Returns
    /// `use_tree_reduce_at_depth(tree_depth)`, except that depth 4 is gathered before the
    /// first wrap when `depth4_select_after_wrap_only` is set.
    pub fn selects_cached_level(&self, rnd: usize, tree_depth: usize) -> bool {
        let first_pass = rnd < self.tree_height();
        self.use_tree_reduce_at_depth(tree_depth)
            && !(tree_depth == 4 && self.depth4_select_after_wrap_only && first_pass)
    }
}

/// Which scheduling pressure policy the greedy scheduler should use.
///
/// Controls how the scheduler reacts to rising scratch (register) pressure.
/// Modeled after LLVM's `GenericScheduler` register-pressure heuristics
/// (`RegExcess`/`RegCritical` in `MachineScheduler.cpp`).
///
/// Five policies available: Greedy, PressureThrottled, PressureGated,
/// Adaptive (default), and Cascade (LLVM-style strict priority).
///
/// See `ssa_greedy::pressure_policy` for implementation details.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PressurePolicyKind {
    /// No pressure awareness: LUC sort, defer at hard budget only, always shift.
    Greedy,
    /// Soft throttling: re-sort by net pressure above 85% budget, suppress
    /// shift when ALU >8/12 used.
    PressureThrottled,
    /// Hard gating: above 95% budget, only schedule ops with non-positive net
    /// pressure. Suppress shift above critical threshold.
    PressureGated,
    /// Adaptive (default): transitions Greedy → Throttled → Gated as pressure rises.
    #[default]
    Adaptive,
    /// Cascade (LLVM-style): strict 5-level priority cascade
    /// (engine saturation → slack → LUC → net pressure → ASAP tiebreak).
    Cascade,
}

/// Compilation concern: how to optimize and schedule.
///
/// Controls scheduling and optimization passes — these don't affect
/// what ops are emitted, only how they're scheduled/optimized.
#[derive(Debug, Clone)]
pub struct CompilerConfig {
    /// Enable VALU→ALU shifting during scheduling
    pub enable_valu_shift: bool,
    /// Enable SSA dead code elimination before scheduling.
    pub enable_ssa_dce: bool,
    /// Scheduling pressure policy for the greedy scheduler.
    ///
    /// Controls how the scheduler reacts to rising scratch (register) pressure.
    /// See `ssa_greedy::pressure_policy` module for details.
    pub scheduling_pressure_policy: PressurePolicyKind,
    /// After lowering, verify no two ops in the same cycle write overlapping scratch (return Err so
    /// export_partial_result_on_error can run). Default true. Disable via CLI --no-verify-scratch-conflict for debugging.
    pub verify_same_cycle_scratch: bool,
    /// When true with greedy-only, use Full bounds (ASAP deferral, clamping, capacity deferral).
    /// Used for comparing PolicyOnly vs Full greedy (feasibility investigation).
    pub greedy_use_full_bounds: bool,
    // === Hint passes (run after GreedySchedule; overwrite greedy_decisions only) ===
    /// When true, run ConstChoice hint pass after greedy: set add_imm where Flow has slack and Load is tight.
    /// Off by default. When enabled, greedy uses fixed default for ConstChoice so the pass owns the decision.
    pub enable_const_choice_hint_pass: bool,
    /// When true, run SelectChoice hint pass after greedy: prefer FMA (Flow) over vselect when VALU saturated.
    /// Off by default. When enabled, greedy uses fixed default for SelectChoice so the pass owns the decision.
    pub enable_select_choice_hint_pass: bool,
    /// When true, run VALU shift hint pass after greedy: add ops to shifted_ops where ALU has slack.
    /// Off by default. When enabled, greedy uses fixed no-shift default so the pass owns the decision.
    pub enable_valu_shift_hint_pass: bool,
    /// When true, run schedule-with-fixed-choices pass after hint passes: re-run greedy with encodings fixed.
    pub enable_schedule_with_fixed_choices: bool,
}

impl Default for CompilerConfig {
    fn default() -> Self {
        CompilerConfig {
            enable_valu_shift: true,
            enable_ssa_dce: true,
            scheduling_pressure_policy: PressurePolicyKind::default(),
            verify_same_cycle_scratch: true,
            greedy_use_full_bounds: false,
            enable_const_choice_hint_pass: false,
            enable_select_choice_hint_pass: false,
            enable_valu_shift_hint_pass: false,
            enable_schedule_with_fixed_choices: false,
        }
    }
}

/// Kernel configuration parameters.
///
/// Thin wrapper combining `ProblemSpec` (what to emit) and `CompilerConfig`
/// (how to optimize). Delegation methods on `KernelConfig` forward to `spec`
/// so that callers using `config.n_groups()` etc. continue to work.
#[derive(Debug, Clone)]
pub struct KernelConfig {
    /// Emission concern: problem dimensions and what to emit.
    pub spec: ProblemSpec,
    /// Compilation concern: scheduling and optimization options.
    pub compiler: CompilerConfig,
}

impl Default for KernelConfig {
    fn default() -> Self {
        KernelConfig {
            spec: ProblemSpec::default(),
            compiler: CompilerConfig::default(),
        }
    }
}

impl KernelConfig {
    /// Create a new kernel configuration
    pub fn new(forest_height: usize, n_nodes: usize, batch_size: usize, rounds: usize) -> Self {
        KernelConfig {
            spec: ProblemSpec {
                forest_height,
                n_nodes,
                batch_size,
                rounds,
                ..Default::default()
            },
            compiler: CompilerConfig::default(),
        }
    }

    // --- Delegation to ProblemSpec ---

    /// Number of groups (batch_size / VLEN)
    pub fn n_groups(&self) -> usize {
        self.spec.n_groups()
    }

    /// Number of iterations (ceil(n_groups / 8) so that 1..7 groups still get one iter)
    pub fn n_iters(&self) -> usize {
        self.spec.n_iters()
    }

    /// Tree height (forest_height + 1)
    pub fn tree_height(&self) -> usize {
        self.spec.tree_height()
    }

    /// Get encoding flags for pseudo-consts
    pub fn encoding_flags(&self) -> ConstEncodingFlags {
        self.spec.encoding_flags()
    }

    /// True if tree-reduce selection should be used at the given tree depth.
    pub fn use_tree_reduce_at_depth(&self, depth: usize) -> bool {
        self.spec.use_tree_reduce_at_depth(depth)
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

/// First of the two adjacent hash stages that `ProblemSpec::merge_hash_stages` merges.
pub const MERGED_HASH_STAGE: usize = 2;

/// Constants of the merged form of hash stages `first` and `first + 1`.
///
/// When stage `first` fuses to `a * m + c` and the next stage is `(x + d) ^ (x << k)`,
/// the two stages together equal `(a * m1 + c1) ^ (a * m2 + c2)` with `m1 = m`,
/// `c1 = c + d`, `m2 = m << k`, `c2 = c << k`, all modulo 2^32 (the left shift distributes
/// over the multiply_add). Proved for stages 2 and 3 by `R8-hash-stage-merge` in
/// `verification/rewrites.py`, which derives the same constants.
///
/// # Arguments
/// * `first` - Index of the first stage of the pair.
///
/// # Returns
/// `Some((m1, c1, m2, c2))`, each in `0..2^32`, or `None` if the pair does not have this form.
pub fn merged_hash_stage_constants(first: usize) -> Option<(i64, i64, i64, i64)> {
    if first + 1 >= HASH_STAGES.len() {
        return None;
    }
    let mult = fma_multiplier(first)?;
    let (_, c, _, _, _) = HASH_STAGES[first];
    let (op1, d, op2, op3, k) = HASH_STAGES[first + 1];
    if !(op1 == '+' && op2 == '^' && op3 == '<') {
        return None;
    }
    const MASK: i64 = 0xFFFF_FFFF;
    Some((mult & MASK, (c + d) & MASK, (mult << k) & MASK, (c << k) & MASK))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Same values as verification/rewrites.py level_last_index (tests/test_rewrite_proofs.py).
    #[test]
    fn test_level_last_index() {
        assert_eq!(
            (0..6).map(level_last_index).collect::<Vec<_>>(),
            vec![0, 2, 6, 14, 30, 62]
        );
        for d in 0..11 {
            assert_eq!(level_last_index(d + 1), 2 * level_last_index(d) + 2);
        }
    }

    #[test]
    fn test_cached_select_from_str() {
        assert_eq!(CachedSelect::from_str("standard"), Ok(CachedSelect::Standard));
        assert_eq!(CachedSelect::from_str("bits"), Ok(CachedSelect::Bits));
        assert_eq!(CachedSelect::from_str("bits-flow"), Ok(CachedSelect::BitsFlow));
        assert_eq!(CachedSelect::from_str("mobius"), Ok(CachedSelect::Mobius));
        assert!(CachedSelect::from_str("flow").is_err());
        assert_eq!(CachedSelect::default(), CachedSelect::Standard);
    }

    #[test]
    fn test_selects_cached_level() {
        let spec = |max_depth, after_wrap_only| ProblemSpec {
            tree_reduce_max_depth: max_depth,
            depth4_select_after_wrap_only: after_wrap_only,
            ..Default::default()
        };
        // Default: levels 0 to 3 are cached on every visit, level 4 never.
        for rnd in [3, 14] {
            assert!(spec(3, false).selects_cached_level(rnd, 3));
        }
        assert!(!spec(3, false).selects_cached_level(4, 4));
        assert!(!spec(3, true).selects_cached_level(15, 4));
        // Depth 4 cached: every visit, or only after the first wrap (round 11 for height 10).
        assert!(spec(4, false).selects_cached_level(4, 4));
        assert!(spec(4, false).selects_cached_level(15, 4));
        assert!(!spec(4, true).selects_cached_level(4, 4));
        assert!(spec(4, true).selects_cached_level(15, 4));
        // The option only concerns depth 4.
        assert!(spec(4, true).selects_cached_level(3, 3));
        assert!(!spec(4, true).selects_cached_level(5, 5));
    }

    #[test]
    fn test_select_bit_source_from_str() {
        assert_eq!(SelectBitSource::from_str("index"), Ok(SelectBitSource::Index));
        assert_eq!(SelectBitSource::from_str("path"), Ok(SelectBitSource::Path));
        assert!(SelectBitSource::from_str("idx").is_err());
        assert_eq!(SelectBitSource::default(), SelectBitSource::Index);
    }

    #[test]
    fn test_index_form_from_str() {
        assert_eq!(IndexForm::from_str("zero-based"), Ok(IndexForm::ZeroBased));
        assert_eq!(IndexForm::from_str("mirrored"), Ok(IndexForm::Mirrored));
        assert!(IndexForm::from_str("one-based").is_err());
        assert_eq!(IndexForm::default(), IndexForm::ZeroBased);
    }

    /// Same numbers as tests/test_rewrite_proofs.py checks for verification/rewrites.py.
    #[test]
    fn test_merged_hash_stage_constants_match_the_proof() {
        assert_eq!(
            merged_hash_stage_constants(MERGED_HASH_STAGE),
            Some((33, 0xE9F8_CC1D, 16896, 0xACCF_6200))
        );
        for first in [0, 1, 3, 4, 5] {
            assert_eq!(merged_hash_stage_constants(first), None, "stages {first}, {}", first + 1);
        }
    }

    /// n_iters must be ceiling(n_groups/8) so 1..7 groups get one iter.
    #[test]
    fn test_n_iters_ceiling_small_groups() {
        let cfg = KernelConfig {
            spec: ProblemSpec {
                forest_height: 5,
                n_nodes: 31,
                batch_size: 8, // n_groups = 1
                rounds: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(cfg.n_groups(), 1, "batch 8 -> 1 group");
        assert_eq!(cfg.n_iters(), 1, "1 group must get 1 iter (ceiling)");
    }

    #[test]
    fn test_n_iters_ceiling_two_groups() {
        let cfg = KernelConfig {
            spec: ProblemSpec {
                forest_height: 5,
                n_nodes: 31,
                batch_size: 16, // n_groups = 2
                rounds: 2,
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(cfg.n_groups(), 2);
        assert_eq!(cfg.n_iters(), 1, "2 groups -> ceil(2/8)=1");
    }

    #[test]
    fn test_n_iters_eight_groups() {
        let cfg = KernelConfig {
            spec: ProblemSpec {
                forest_height: 5,
                n_nodes: 31,
                batch_size: 64, // n_groups = 8
                rounds: 4,
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(cfg.n_groups(), 8);
        assert_eq!(cfg.n_iters(), 1);
    }

    #[test]
    fn test_n_iters_nine_groups() {
        let cfg = KernelConfig {
            spec: ProblemSpec {
                forest_height: 5,
                n_nodes: 31,
                batch_size: 72, // n_groups = 9
                rounds: 4,
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(cfg.n_groups(), 9);
        assert_eq!(cfg.n_iters(), 2, "9 groups -> ceil(9/8)=2");
    }
}
