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

/// How round barriers constrain the greedy scheduler.
///
/// Emission puts a barrier pseudo-op between rounds. With `Round`, every op of round r+1
/// depends on the barrier and the barrier depends on every op of round r, so the greedy
/// schedule runs rounds strictly one after another (and per-section CP-SAT repair can work
/// on one round at a time). With `None`, the barriers are dropped before dependencies are
/// computed and the ready queue may mix rounds; the kernel is then a single section.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GreedyBarrierMode {
    /// Hard barrier between rounds (default).
    #[default]
    Round,
    /// No barriers: only data dependencies order the ops.
    None,
}

impl GreedyBarrierMode {
    /// Parse the CLI/binding spelling.
    ///
    /// # Arguments
    /// * `s` - `"round"` or `"none"`
    ///
    /// # Returns
    /// The mode, or an error message naming the valid values.
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "round" => Ok(GreedyBarrierMode::Round),
            "none" => Ok(GreedyBarrierMode::None),
            other => Err(format!(
                "Invalid greedy_barriers '{}': must be 'round' or 'none'",
                other
            )),
        }
    }

    /// The CLI/binding spelling.
    pub fn as_str(&self) -> &'static str {
        match self {
            GreedyBarrierMode::Round => "round",
            GreedyBarrierMode::None => "none",
        }
    }
}

/// Ready-queue order of the greedy scheduler (`--greedy-priority`).
///
/// See `ssa_greedy::wavefront` for the wavefront key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GreedyPriorityKind {
    /// First-come-first-served by ready cycle, then emission order (default).
    #[default]
    Fifo,
    /// Stagger the independent group chains (`greedy_stagger_cycles`, `greedy_groups_in_flight`).
    Wavefront,
}

impl GreedyPriorityKind {
    /// Parse the CLI/binding spelling.
    ///
    /// # Arguments
    /// * `s` - `"fifo"` or `"wavefront"`
    ///
    /// # Returns
    /// The kind, or an error message naming the valid values.
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "fifo" => Ok(GreedyPriorityKind::Fifo),
            "wavefront" => Ok(GreedyPriorityKind::Wavefront),
            other => Err(format!(
                "Invalid greedy_priority '{}': must be 'fifo' or 'wavefront'",
                other
            )),
        }
    }

    /// The CLI/binding spelling.
    pub fn as_str(&self) -> &'static str {
        match self {
            GreedyPriorityKind::Fifo => "fifo",
            GreedyPriorityKind::Wavefront => "wavefront",
        }
    }
}

/// Default virtual start delay between consecutive group chains, in cycles
/// (`--greedy-stagger-cycles`). About three quarters of one round's critical path.
pub const DEFAULT_GREEDY_STAGGER_CYCLES: u32 = 12;

/// Default number of group chains allowed to run at once (`--greedy-groups-in-flight`).
pub const DEFAULT_GREEDY_GROUPS_IN_FLIGHT: usize = 16;

/// Overrides for the pressure policies' thresholds, as fractions of the scratch budget.
///
/// `None` keeps the policy's built-in value (soft 0.85, critical 0.95, inflation 0.80). The
/// built-in values were tuned before the greedy's live-word counter was exact. A policy
/// ignores the thresholds it does not have (`Greedy` has none; `PressureThrottled` only
/// `soft`; `PressureGated` `soft` and `critical`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PressureThresholds {
    /// Above this fraction the ready queue blends in net pressure and shifts are
    /// suppressed when the scalar ALU is busy.
    pub soft: Option<f64>,
    /// Above this fraction no vector op is shifted to the scalar ALU.
    pub critical: Option<f64>,
    /// Above this fraction producers with slack are deferred (ASAP inflation). A value of
    /// 1.0 or more turns inflation off.
    pub inflation: Option<f64>,
}

impl PressureThresholds {
    /// Check that every override is a positive finite number.
    ///
    /// # Returns
    /// `Err` naming the first invalid threshold.
    pub fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("pressure_soft_threshold", self.soft),
            ("pressure_critical_threshold", self.critical),
            ("pressure_inflation_threshold", self.inflation),
        ] {
            if let Some(v) = value {
                if !(v.is_finite() && v > 0.0) {
                    return Err(format!("{} must be a positive finite number, got {}", name, v));
                }
            }
        }
        Ok(())
    }
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
    /// Whether round barriers constrain the greedy scheduler (default: `Round`).
    pub greedy_barriers: GreedyBarrierMode,
    /// Ready-queue order of the greedy scheduler (default: `Fifo`).
    pub greedy_priority: GreedyPriorityKind,
    /// Wavefront priority: virtual start delay between consecutive group chains, in cycles.
    pub greedy_stagger_cycles: u32,
    /// Wavefront priority: group chains allowed to run at once; 0 = no limit.
    pub greedy_groups_in_flight: usize,
    /// Overrides for the pressure policy's thresholds (default: none).
    pub pressure_thresholds: PressureThresholds,
    /// Words the greedy keeps free below the scratch budget (default: 0). Address
    /// assignment can need a few words more than the greedy's counter reports.
    pub greedy_budget_headroom: usize,
    /// Greedy-only: let a shifted vector op straddle two cycles so all 12 scalar slots can
    /// carry shifted work (default: false). Ignored when CP-SAT follows, because its hints
    /// carry one cycle per op.
    pub greedy_split_shift: bool,
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
            greedy_barriers: GreedyBarrierMode::default(),
            greedy_priority: GreedyPriorityKind::default(),
            greedy_stagger_cycles: DEFAULT_GREEDY_STAGGER_CYCLES,
            greedy_groups_in_flight: DEFAULT_GREEDY_GROUPS_IN_FLIGHT,
            greedy_split_shift: false,
            pressure_thresholds: PressureThresholds::default(),
            greedy_budget_headroom: 0,
        }
    }
}

impl CompilerConfig {
    /// The configured pressure policy with any threshold overrides applied.
    pub fn pressure_policy(&self) -> Box<dyn super::ssa_greedy::pressure_policy::PressurePolicy> {
        use super::ssa_greedy::pressure_policy::{
            AdaptivePolicy, CascadePolicy, GreedyPolicy, PressureGatedPolicy,
            PressureThrottledPolicy,
        };
        let t = &self.pressure_thresholds;
        match self.scheduling_pressure_policy {
            PressurePolicyKind::Greedy => Box::new(GreedyPolicy),
            PressurePolicyKind::PressureThrottled => {
                let mut policy = PressureThrottledPolicy::default();
                policy.soft_threshold_fraction = t.soft.unwrap_or(policy.soft_threshold_fraction);
                Box::new(policy)
            }
            PressurePolicyKind::PressureGated => {
                let mut policy = PressureGatedPolicy::default();
                policy.soft_threshold_fraction = t.soft.unwrap_or(policy.soft_threshold_fraction);
                policy.critical_threshold_fraction =
                    t.critical.unwrap_or(policy.critical_threshold_fraction);
                Box::new(policy)
            }
            PressurePolicyKind::Adaptive => {
                let mut policy = AdaptivePolicy::default();
                policy.soft_threshold_fraction = t.soft.unwrap_or(policy.soft_threshold_fraction);
                policy.critical_threshold_fraction =
                    t.critical.unwrap_or(policy.critical_threshold_fraction);
                policy.inflation_threshold_fraction =
                    t.inflation.unwrap_or(policy.inflation_threshold_fraction);
                Box::new(policy)
            }
            PressurePolicyKind::Cascade => {
                let mut policy = CascadePolicy::default();
                policy.soft_threshold_fraction = t.soft.unwrap_or(policy.soft_threshold_fraction);
                policy.critical_threshold_fraction =
                    t.critical.unwrap_or(policy.critical_threshold_fraction);
                policy.inflation_threshold_fraction =
                    t.inflation.unwrap_or(policy.inflation_threshold_fraction);
                Box::new(policy)
            }
        }
    }

    /// The scratch budget the greedy schedules against: `budget` less the configured headroom.
    pub fn greedy_scratch_budget(&self, budget: usize) -> usize {
        budget.saturating_sub(self.greedy_budget_headroom)
    }

    /// Schedule-shape switches for the greedy scheduler, as configured here.
    pub fn greedy_shape(&self) -> super::ssa_greedy::GreedyShape {
        use super::ssa_greedy::{GreedyPriority, GreedyShape};
        GreedyShape {
            priority: match self.greedy_priority {
                GreedyPriorityKind::Fifo => GreedyPriority::Fifo,
                GreedyPriorityKind::Wavefront => GreedyPriority::Wavefront {
                    stagger_cycles: self.greedy_stagger_cycles,
                    groups_in_flight: (self.greedy_groups_in_flight > 0)
                        .then_some(self.greedy_groups_in_flight),
                },
            },
            split_shift: self.greedy_split_shift,
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

#[cfg(test)]
mod tests {
    use super::*;

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
