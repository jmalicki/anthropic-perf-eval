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

pub use crate::architecture::anthropic_perf_eval::{SCRATCH_SIZE, VLEN};

/// Maximum tree depth for which we have tree-reduce emission (depth 4 = 16-way).
/// tree_reduce_max_depth is clamped to 0..=MAX_TREE_REDUCE_DEPTH.
pub const MAX_TREE_REDUCE_DEPTH: u32 = 4;

/// How the kernel represents each lane's tree index between rounds.
///
/// The reference index `idx` at tree depth `d` lies in `[2^d - 1, 2^(d+1) - 2]`
/// (`R7-index-range-by-depth`). `Mirrored` tracks `n = L_d - idx` with
/// `L_d = 2^(d+1) - 2` (`level_last_index`): the node's position in its level counted from
/// the right. Proofs: `R9` to `R11` in `verification/rewrites.py`.
/// Rounds whose round XOR is pushed below the top select of a cached-level select tree:
/// `val ^ select(bit, hi, lo) = select(bit, val ^ hi, val ^ lo)` (`R18-select-xor-pushdown`).
/// The two XORs run in parallel with the branch bit's AND, so the chain from one round's
/// hash to the next is one cycle shorter; one more vector op per group and round. Only where
/// the top of the tree is a SelectChoice (`bits-flow`; `bits` from depth 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SelectXorPushdown {
    /// Never (default).
    #[default]
    None,
    /// Rounds before the first wrap (the start-up: on the chain to the first gather).
    FirstPass,
    /// Every cached-level select round.
    All,
}

/// Which consumers of the mirrored index `n' = 2n + bit` use the early form
/// `V - n' = madd(n, -2, V) - bit` (`R17-index-early-terms`): the multiply_add depends only on
/// the previous index, so after the newest branch bit only the subtraction remains on the
/// dependency chain (one cycle shorter). Mirrored index form only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IndexEarlyTerms {
    /// Never (default).
    #[default]
    None,
    /// The final index store `L_d - n'`: same op count (`n'` is then dead), one cycle off
    /// the tail.
    Final,
    /// The final index store, and the gather address `(forest_p + L_d) - n'` of a group's
    /// first gather after rounds without one: one more vector op per such gather (`n'` is
    /// still needed), one cycle off the chain to it.
    FirstGather,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IndexForm {
    /// The reference index. Update per round: AND, SUB, multiply_add (3 vector ops).
    #[default]
    ZeroBased,
    /// `n = L_d - idx`. Update per round: AND, multiply_add (2 vector ops); only the AND
    /// at depth 0, where `n` is the branch bit itself, and nothing in the wrap round.
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
    /// `SelectChoice`s (flow `vselect`, or SUB + multiply_add). Per group that is 0, 1, 3 and 7
    /// `SelectChoice`s at depths 1 to 4, against `Standard`'s 1, 3, 3 and 7: fewer choices that
    /// can go to the flow engine at depths 1 and 2, the same at depths 3 and 4.
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
    /// Extract the older bits used at depths 3 and 4 from the index as it was before the
    /// last update (`n = b_0 .. b_(d-2)` in binary): `2 (d - 2)` vector ops per group at depth
    /// `d >= 3` (one more to mirror a zero-based index), so those bits are not kept live. Two
    /// bits still are: the newest bit, from the branch until the next cached select, and at
    /// depth 2 the depth-0 bit, which the select reads directly. With the mirrored index that
    /// bit is the index itself (no extra scratch); with the zero-based index it is a separate
    /// vector, live for one more round. Proof: `R15-bits-from-index`.
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

/// How one group of 8 lanes fetches its tree node in one round (one level visit).
///
/// Without an emission plan every group of a round uses the same encoding, derived from
/// `tree_reduce_max_depth`, `cached_select` and `depth4_select_after_wrap_only`
/// (`ProblemSpec::access_encoding`). With `ProblemSpec::access_plan` it is chosen per round
/// and per group (plan format: `solver/emission_plan.py`; `docs/EMISSION_STRUCTURE_MODEL.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AccessEncoding {
    /// Scalar gather: one vector address op, 8 scalar loads, XOR with the deferred stage-5
    /// constant (`emit_scattered_access`). Any depth from 1.
    Gather,
    /// The cached-level emission of `CachedSelect::Standard` (compares or index bits, select
    /// tree, XOR). Flag path only: an emission plan cannot name it.
    Standard,
    /// `CachedSelect::Bits` on this group (depth 0 to `MAX_TREE_REDUCE_DEPTH`).
    Bits,
    /// `CachedSelect::BitsFlow` on this group.
    BitsFlow,
    /// `CachedSelect::Mobius` on this group.
    Mobius,
}

impl AccessEncoding {
    /// Parse the plan spelling.
    ///
    /// # Arguments
    /// * `s` - `"gather"`, `"bits"`, `"bits-flow"` or `"mobius"`.
    ///
    /// # Returns
    /// The encoding, or an error naming the valid spellings. `Standard` has no spelling.
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "gather" => Ok(AccessEncoding::Gather),
            "bits" => Ok(AccessEncoding::Bits),
            "bits-flow" => Ok(AccessEncoding::BitsFlow),
            "mobius" => Ok(AccessEncoding::Mobius),
            other => Err(format!(
                "Invalid access encoding '{}': must be 'gather', 'bits', 'bits-flow' or 'mobius'",
                other
            )),
        }
    }

    /// The `CachedSelect` mode of a bit-driven encoding, or `None` for `Gather` / `Standard`.
    pub fn cached_select(self) -> Option<CachedSelect> {
        match self {
            AccessEncoding::Bits => Some(CachedSelect::Bits),
            AccessEncoding::BitsFlow => Some(CachedSelect::BitsFlow),
            AccessEncoding::Mobius => Some(CachedSelect::Mobius),
            AccessEncoding::Gather | AccessEncoding::Standard => None,
        }
    }

    /// Check that this encoding can fetch a node at `tree_depth` in a plan.
    ///
    /// # Returns
    /// `Err` for `Standard` (not plannable), for `Gather` at depth 0 (every lane is at the
    /// root; the mirrored index does not even exist there) and for a bit-driven encoding
    /// below `MAX_TREE_REDUCE_DEPTH`.
    pub fn check_plannable(self, tree_depth: usize) -> Result<(), String> {
        match self {
            AccessEncoding::Standard => {
                Err("the standard cached-level encoding cannot be named in an emission plan".to_string())
            }
            AccessEncoding::Gather if tree_depth == 0 => {
                Err("gather is not available at depth 0 (every lane is at the root)".to_string())
            }
            AccessEncoding::Bits | AccessEncoding::BitsFlow | AccessEncoding::Mobius
                if tree_depth > MAX_TREE_REDUCE_DEPTH as usize =>
            {
                Err(format!(
                    "{:?} selects among cached nodes, which exist only down to depth {} (got depth {})",
                    self, MAX_TREE_REDUCE_DEPTH, tree_depth
                ))
            }
            _ => Ok(()),
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
    /// Emission plan: the access encoding of every group in every round, `[round][group]`.
    /// When set it replaces `tree_reduce_max_depth`, `cached_select` and
    /// `depth4_select_after_wrap_only` for the choice of tree access (`access_encoding`);
    /// check it with `check_access_plan`. Default `None`: the emission is unchanged.
    pub access_plan: Option<Vec<Vec<AccessEncoding>>>,
    /// Gather rounds: XOR the deferred hash constant into the value instead of into the
    /// loaded node, `val ^ (node ^ K) = (val ^ K) ^ node` (`R16-gather-xor-reassociation`),
    /// so one XOR instead of two follows the load. Same op count. Default false.
    pub reassociate_gather_xor: bool,
    /// See [`IndexEarlyTerms`]. Default `None`.
    pub index_early_terms: IndexEarlyTerms,
    /// See [`SelectXorPushdown`]. Default `None`.
    pub select_xor_pushdown: SelectXorPushdown,
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
            access_plan: None,
            reassociate_gather_xor: false,
            index_early_terms: IndexEarlyTerms::None,
            select_xor_pushdown: SelectXorPushdown::None,
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

    /// Tree depth visited in round `rnd`: rounds walk down the tree and wrap to the root
    /// after the deepest level.
    pub fn round_depth(&self, rnd: usize) -> usize {
        rnd % self.tree_height()
    }

    /// How group `g` fetches its node in round `rnd`, which visits `tree_depth`.
    ///
    /// # Returns
    /// The plan's entry when `access_plan` is set. Otherwise the flag-driven choice shared by
    /// every group: `Gather` unless `selects_cached_level`, then `Standard` or the bit-driven
    /// encoding of `cached_select`.
    pub fn access_encoding(&self, rnd: usize, tree_depth: usize, g: usize) -> AccessEncoding {
        if let Some(plan) = &self.access_plan {
            return plan[rnd][g];
        }
        if !self.selects_cached_level(rnd, tree_depth) {
            return AccessEncoding::Gather;
        }
        match self.cached_select {
            CachedSelect::Standard => AccessEncoding::Standard,
            CachedSelect::Bits => AccessEncoding::Bits,
            CachedSelect::BitsFlow => AccessEncoding::BitsFlow,
            CachedSelect::Mobius => AccessEncoding::Mobius,
        }
    }

    /// True if some group in some round fetches tree level `tree_depth` with `encoding`.
    pub fn uses_access(&self, tree_depth: usize, encoding: AccessEncoding) -> bool {
        (0..self.rounds).any(|r| {
            self.round_depth(r) == tree_depth
                && (0..self.n_groups()).any(|g| self.access_encoding(r, tree_depth, g) == encoding)
        })
    }

    /// True if round `rnd` is in the first pass down the tree: rounds 0 to `forest_height`,
    /// i.e. up to and including the first wrap round. Relies on round 0 being at depth 0
    /// (`round_depth`), which `check_critical_path_options` asserts.
    pub fn is_first_pass_round(&self, rnd: usize) -> bool {
        rnd <= self.forest_height
    }

    /// True if `R18-select-xor-pushdown` applies to a group fetching its node with `mode` in
    /// round `rnd` at `tree_depth`: the option covers the round (`FirstPass`: first pass only)
    /// and the top of the group's select tree is a SelectChoice (`BitsFlow` from depth 1,
    /// `Bits` from depth 2). The emitter and `check_critical_path_options` both use this.
    pub fn select_xor_pushdown_applies(&self, rnd: Option<usize>, tree_depth: usize, mode: CachedSelect) -> bool {
        let round_ok = match self.select_xor_pushdown {
            SelectXorPushdown::None => false,
            SelectXorPushdown::FirstPass => rnd.map_or(false, |r| self.is_first_pass_round(r)),
            SelectXorPushdown::All => true,
        };
        round_ok
            && tree_depth >= 1
            && (mode == CachedSelect::BitsFlow || (mode == CachedSelect::Bits && tree_depth >= 2))
    }

    /// Reject critical-path rewrite options (docs/CRITICAL_PATH.md) that would be silent
    /// no-ops for this problem, so a direct Rust caller gets the same errors as the CLI.
    ///
    /// # Returns
    /// `Err` naming the option when `index_early_terms` is set without the mirrored index;
    /// when `select_xor_pushdown` is set but no group in any round it covers has a
    /// SelectChoice at the top of its select tree (e.g. `standard` or `mobius` cached select,
    /// or `bits` only at depth 1); when `reassociate_gather_xor` is set but no group ever
    /// gathers.
    pub fn check_critical_path_options(&self) -> Result<(), String> {
        // `is_first_pass_round` counts the first pass from round 0 at depth 0.
        assert_eq!(self.round_depth(0), 0, "round 0 must visit depth 0 (the root)");
        if self.index_early_terms != IndexEarlyTerms::None && self.index_form != IndexForm::Mirrored {
            return Err("index_early_terms (--index-early-terms) needs index_form = Mirrored \
                        (--index-form mirrored); otherwise it is ignored"
                .to_string());
        }
        let any = |pred: &dyn Fn(usize, usize, AccessEncoding) -> bool| {
            (0..self.rounds).any(|r| {
                let d = self.round_depth(r);
                (0..self.n_groups()).any(|g| pred(r, d, self.access_encoding(r, d, g)))
            })
        };
        if self.select_xor_pushdown != SelectXorPushdown::None
            && !any(&|r, d, e| e.cached_select().map_or(false, |m| self.select_xor_pushdown_applies(Some(r), d, m)))
        {
            return Err(format!(
                "select_xor_pushdown = {:?} (--select-xor-pushdown) applies to no round of this kernel: it needs a \
                 select tree whose top is a SelectChoice (bits-flow from depth 1, bits from depth 2){}; otherwise \
                 it is ignored",
                self.select_xor_pushdown,
                if self.select_xor_pushdown == SelectXorPushdown::FirstPass { " in the first pass" } else { "" }
            ));
        }
        if self.reassociate_gather_xor && !any(&|_, _, e| e == AccessEncoding::Gather) {
            return Err("reassociate_gather_xor (--reassociate-gather-xor) applies to gather rounds and this kernel \
                        has none; otherwise it is ignored"
                .to_string());
        }
        Ok(())
    }

    /// Check `access_plan` against the problem shape: one row per round, one entry per group,
    /// and every entry available at its round's depth (`AccessEncoding::check_plannable`).
    ///
    /// # Returns
    /// `Ok(())` when there is no plan or it is valid, else a message naming the first bad entry.
    pub fn check_access_plan(&self) -> Result<(), String> {
        let Some(plan) = &self.access_plan else {
            return Ok(());
        };
        if plan.len() != self.rounds {
            return Err(format!(
                "emission plan has {} rounds, the kernel has {}",
                plan.len(),
                self.rounds
            ));
        }
        for (rnd, row) in plan.iter().enumerate() {
            if row.len() != self.n_groups() {
                return Err(format!(
                    "emission plan round {} has {} groups, the kernel has {}",
                    rnd,
                    row.len(),
                    self.n_groups()
                ));
            }
            let depth = self.round_depth(rnd);
            for (g, encoding) in row.iter().enumerate() {
                encoding
                    .check_plannable(depth)
                    .map_err(|e| format!("emission plan round {} (depth {}) group {}: {}", rnd, depth, g, e))?;
            }
        }
        Ok(())
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

/// What the schedulers' scratch budget bounds.
///
/// `Words` (default): live words at every point of the schedule (greedy counter, CP-SAT
/// cumulative over all values). That does not guarantee addresses: for values of 1 and 8
/// words, a per-point fit does not imply a layout exists (docs/ALLOCATION_CONJECTURES.md).
/// `Pools`: `8 * max_v + max_s <= budget`, `max_v` / `max_s` the most vectors / scalars live
/// at once over the schedule. Address assignment (`address_assign::assign_two_pass`) always
/// fits such a schedule in `8 * max_v + max_s` words.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScratchModel {
    /// Live words at every point (default).
    #[default]
    Words,
    /// Separate vector and scalar maxima, `8 * max_v + max_s <= budget`.
    Pools,
}

impl ScratchModel {
    /// Parse the CLI/binding spelling.
    ///
    /// # Arguments
    /// * `s` - `"words"` or `"pools"`
    ///
    /// # Returns
    /// The model, or an error message naming the valid values.
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "words" => Ok(ScratchModel::Words),
            "pools" => Ok(ScratchModel::Pools),
            other => Err(format!(
                "Invalid scratch_model '{}': must be 'words' or 'pools'",
                other
            )),
        }
    }

    /// True for `Pools`.
    pub fn is_pools(self) -> bool {
        self == ScratchModel::Pools
    }
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
    /// Let a shifted vector op straddle two cycles so all 12 scalar slots can carry shifted
    /// work (default: false). When CP-SAT follows it applies only with
    /// `cpsat_split_shift_hint`.
    pub greedy_split_shift: bool,
    /// Most forward/backward justification iterations applied to the greedy schedule
    /// (default: 0 = off). See `ssa_greedy::justify`.
    pub greedy_justify_iterations: usize,
    // === Relaxed-scratch reference (analysis only) ===
    /// Scratch words the greedy, the per-section repair, CP-SAT and address assignment may use
    /// (default: the machine's `SCRATCH_SIZE`, 1,536). Smaller values are valid kernels; larger
    /// ones are the relaxed-scratch reference and do not run on the real machine (see
    /// `is_relaxed_scratch` and docs/RELAXED_SCRATCH_REFERENCE.md).
    pub scratch_size: usize,
    /// Whether the CP-SAT model contains the scratch cumulative and liveness intervals (default
    /// true). False drops them, so the model has no scratch constraint at all; the greedy and the
    /// allocator still use `scratch_size`, which should then be large
    /// (`UNBOUNDED_SCRATCH_WORDS`).
    pub cpsat_scratch_constraint: bool,
    /// What the scratch budget bounds in both schedulers (default: `Words`).
    pub scratch_model: ScratchModel,
    /// With `greedy_split_shift` and CP-SAT: keep the greedy's split shifts and hint CP-SAT
    /// each lane at its own cycle (default: false, the greedy that feeds CP-SAT makes no
    /// split shifts).
    pub cpsat_split_shift_hint: bool,
    /// When the greedy shifts a vector op to the scalar ALU (default: `WhenFull`).
    pub greedy_shift_policy: GreedyShiftPolicyKind,
    /// Pace target for `greedy_shift_policy` `Pace`/`ReservePace`: vector ops to run as 8
    /// scalar ops over the whole schedule; `None` = the balanced split of the emitted counts.
    pub greedy_shift_target: Option<usize>,
    /// Wavefront: which ready loads get the load slots (default: `Key`, the wavefront key).
    pub greedy_load_priority: super::ssa_greedy::LoadPriority,
    /// Wavefront: which ready flow ops get the flow slot (default: `Key`).
    pub greedy_flow_priority: super::ssa_greedy::FlowPriority,
    /// Where a SelectChoice with a one-op multiply_add form may run (default: `Never`, flow).
    pub greedy_select_overflow: super::ssa_greedy::SelectOverflow,
    /// Scratch model `words`: when a budget-deferred op is forced (default: `SameCycle`).
    pub greedy_budget_force: super::ssa_greedy::BudgetForce,
    /// Wavefront: native scalar ops before vector ops on the scalar ALU (default: `Key`).
    pub greedy_alu_priority: super::ssa_greedy::AluPriority,
    /// Where the greedy materialises ConstChoices (default: `Default`); greedy-only.
    pub greedy_const_policy: super::ssa_greedy::ConstPolicy,
    /// Wavefront: a different stagger for the first groups (default: none).
    pub greedy_early_stagger: Option<super::ssa_greedy::EarlyStagger>,
    /// Encodings forced by `--const-choice`, `--select-choice`, `--valu-shift` in a greedy-only
    /// build (with CP-SAT they come from the CP-SAT config). Default: none forced.
    pub greedy_forced: GreedyForcedChoices,
}

/// Forced encodings for a greedy-only build. `None` = the greedy decides.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GreedyForcedChoices {
    /// Every ConstChoice as this encoding (`Load` or `AddImm`).
    pub const_choice: Option<crate::types::ConstChoiceEncoding>,
    /// Every SelectChoice as this encoding.
    pub select_choice: Option<crate::types::SelectChoiceEncoding>,
    /// `Some(true)`: every shiftable vector op on the scalar ALU; `Some(false)`: none.
    pub valu_shift: Option<bool>,
}

/// Scratch size used for "no scratch constraint" (`--no-scratch-constraint`): large enough that
/// no schedule of the benchmark kernel comes near it, small enough for the simulator to allocate.
pub const UNBOUNDED_SCRATCH_WORDS: usize = 1 << 20;

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
            greedy_justify_iterations: 0,
            cpsat_split_shift_hint: false,
            greedy_shift_policy: GreedyShiftPolicyKind::default(),
            greedy_shift_target: None,
            greedy_load_priority: super::ssa_greedy::LoadPriority::Key,
            greedy_flow_priority: super::ssa_greedy::FlowPriority::Key,
            greedy_select_overflow: super::ssa_greedy::SelectOverflow::Never,
            greedy_budget_force: super::ssa_greedy::BudgetForce::SameCycle,
            greedy_alu_priority: super::ssa_greedy::AluPriority::Key,
            greedy_const_policy: super::ssa_greedy::ConstPolicy::Default,
            greedy_early_stagger: None,
            greedy_forced: GreedyForcedChoices::default(),
            pressure_thresholds: PressureThresholds::default(),
            greedy_budget_headroom: 0,
            scratch_size: SCRATCH_SIZE,
            cpsat_scratch_constraint: true,
            scratch_model: ScratchModel::default(),
        }
    }
}

impl CompilerConfig {
    /// True when the scratch limit is relaxed beyond the machine's: a kernel built this way is
    /// an analysis reference, not a kernel for the real machine.
    pub fn is_relaxed_scratch(&self) -> bool {
        self.scratch_size > SCRATCH_SIZE || !self.cpsat_scratch_constraint
    }

    /// Scratch words the schedulers may count as live: `scratch_size`, less the word reserved
    /// for the zero register when there is one.
    ///
    /// # Arguments
    /// * `zero_vreg_reserved` - Address 0 is reserved for the zero register
    pub fn effective_scratch_budget(&self, zero_vreg_reserved: bool) -> usize {
        super::tick_tock_ssa::effective_scratch_budget(self.scratch_size, zero_vreg_reserved)
    }

    /// Scratch budget handed to a CP-SAT solve: `effective_scratch_budget`, or 0 (no scratch
    /// section in the model) when `cpsat_scratch_constraint` is false.
    ///
    /// # Arguments
    /// * `zero_vreg_reserved` - Address 0 is reserved for the zero register
    pub fn cpsat_scratch_budget(&self, zero_vreg_reserved: bool) -> usize {
        if self.cpsat_scratch_constraint {
            self.effective_scratch_budget(zero_vreg_reserved)
        } else {
            0
        }
    }

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
            scratch_pools: self.scratch_model.is_pools(),
            pool_caps: None,
            load_priority: self.greedy_load_priority,
            flow_priority: self.greedy_flow_priority,
            select_overflow: self.greedy_select_overflow,
            budget_force: self.greedy_budget_force,
            alu_priority: self.greedy_alu_priority,
            const_policy: self.greedy_const_policy,
            early_stagger: self.greedy_early_stagger,
            shift_policy: {
                use super::ssa_greedy::ShiftPolicy;
                let target = self.greedy_shift_target;
                match self.greedy_shift_policy {
                    GreedyShiftPolicyKind::WhenFull => ShiftPolicy::WhenFull,
                    GreedyShiftPolicyKind::Reserve => ShiftPolicy::Reserve,
                    GreedyShiftPolicyKind::Pace => ShiftPolicy::Pace { target },
                    GreedyShiftPolicyKind::ReservePace => ShiftPolicy::ReservePace { target },
                }
            },
        }
    }
}

/// When the greedy runs a shiftable vector op on the scalar ALU (`--greedy-shift-policy`).
///
/// See `ssa_greedy::ShiftPolicy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GreedyShiftPolicyKind {
    /// Only when the vector slots are full (default).
    #[default]
    WhenFull,
    /// Also when the free vector slots are wanted by non-shiftable ops later in the cycle.
    Reserve,
    /// Also whenever behind the pace toward a target number of shifts.
    Pace,
    /// `Reserve` or `Pace`.
    ReservePace,
}

impl GreedyShiftPolicyKind {
    /// Parse the CLI/binding spelling.
    ///
    /// # Arguments
    /// * `s` - `"when-full"`, `"reserve"`, `"pace"` or `"reserve-pace"`
    ///
    /// # Returns
    /// The kind, or an error message naming the valid values.
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "when-full" => Ok(GreedyShiftPolicyKind::WhenFull),
            "reserve" => Ok(GreedyShiftPolicyKind::Reserve),
            "pace" => Ok(GreedyShiftPolicyKind::Pace),
            "reserve-pace" => Ok(GreedyShiftPolicyKind::ReservePace),
            other => Err(format!(
                "Invalid greedy_shift_policy '{}': must be 'when-full', 'reserve', 'pace' or 'reserve-pace'",
                other
            )),
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

/// Reject greedy shape options that would be silently ignored (review of #76).
///
/// * `greedy_select_overflow_limit` only means something with `greedy_select_overflow = "valu"`.
/// * `greedy_flow_priority = "feeds-load-first"` is a wavefront ordering; the FIFO priority
///   never reads it.
/// * `greedy_select_overflow = "valu"` decides each select's form in the greedy; a fixed
///   select encoding (`select_choice` other than `optimize`), the select-choice hint pass or
///   `schedule_with_fixed_choices` owns that decision, and the greedy then never overflows.
///
/// # Returns
/// `Err` with the offending combination, else `Ok(())`.
#[allow(clippy::too_many_arguments)]
pub fn check_greedy_shape_combinations(
    greedy_priority: GreedyPriorityKind,
    greedy_load_priority: &str,
    greedy_flow_priority: &str,
    greedy_alu_priority: &str,
    greedy_select_overflow: &str,
    greedy_select_overflow_limit: Option<usize>,
    select_choice: &str,
    enable_select_choice_hint_pass: bool,
    enable_schedule_with_fixed_choices: bool,
) -> Result<(), String> {
    if greedy_select_overflow_limit.is_some() && greedy_select_overflow != "valu" {
        return Err("greedy_select_overflow_limit (--greedy-select-overflow-limit) needs \
                    greedy_select_overflow = 'valu' (--greedy-select-overflow valu); otherwise it is ignored"
            .to_string());
    }
    if greedy_load_priority != "key" && greedy_priority != GreedyPriorityKind::Wavefront {
        return Err(format!(
            "greedy_load_priority = '{}' (--greedy-load-priority) orders the wavefront priority only; \
             it needs greedy_priority = 'wavefront' (--greedy-priority wavefront)",
            greedy_load_priority
        ));
    }
    if greedy_flow_priority != "key" && greedy_priority != GreedyPriorityKind::Wavefront {
        return Err(format!(
            "greedy_flow_priority = '{}' (--greedy-flow-priority) orders the wavefront priority only; \
             it needs greedy_priority = 'wavefront' (--greedy-priority wavefront)",
            greedy_flow_priority
        ));
    }
    if greedy_alu_priority != "key" && greedy_priority != GreedyPriorityKind::Wavefront {
        return Err(format!(
            "greedy_alu_priority = '{}' (--greedy-alu-priority) orders the wavefront priority only; \
             it needs greedy_priority = 'wavefront' (--greedy-priority wavefront)",
            greedy_alu_priority
        ));
    }
    if greedy_select_overflow == "valu" {
        let owner = if select_choice != "optimize" {
            Some(format!("select_choice = '{}' (--select-choice {})", select_choice, select_choice))
        } else if enable_select_choice_hint_pass {
            Some("the select-choice hint pass (--select-choice-hint-pass / --hint-passes)".to_string())
        } else if enable_schedule_with_fixed_choices {
            Some("schedule_with_fixed_choices (--schedule-with-fixed-choices)".to_string())
        } else {
            None
        };
        if let Some(owner) = owner {
            return Err(format!(
                "greedy_select_overflow = 'valu' (--greedy-select-overflow valu) is ignored when {} owns \
                 the select encoding",
                owner
            ));
        }
    }
    Ok(())
}

/// Reject encoding options another option would silently override or ignore (the same
/// rules as `kernel_build_args.check_encoding_option_combinations`; review of #86).
///
/// # Returns
/// `Err` naming the options for: an early stagger without the wavefront priority; a forced
/// `const_choice` / `select_choice` / `valu_shift` (not "optimize") with the hint pass that
/// would overwrite it; a `greedy_const_policy` other than "default" under CP-SAT, with a
/// forced `const_choice`, or with the const-choice hint pass.
#[allow(clippy::too_many_arguments)]
pub fn check_encoding_option_combinations(
    greedy_priority: GreedyPriorityKind,
    greedy_early_stagger: Option<(usize, u32)>,
    use_cpsat: bool,
    const_choice: &str,
    select_choice: &str,
    valu_shift: &str,
    enable_const_choice_hint_pass: bool,
    enable_select_choice_hint_pass: bool,
    enable_valu_shift_hint_pass: bool,
    greedy_const_policy: &str,
) -> Result<(), String> {
    if greedy_early_stagger.is_some() && greedy_priority != GreedyPriorityKind::Wavefront {
        return Err("greedy_early_stagger (--greedy-early-stagger) staggers the wavefront priority only; \
                    it needs greedy_priority = 'wavefront' (--greedy-priority wavefront)"
            .to_string());
    }
    for (name, forced, hint) in [
        ("const_choice", const_choice, enable_const_choice_hint_pass),
        ("select_choice", select_choice, enable_select_choice_hint_pass),
        ("valu_shift", valu_shift, enable_valu_shift_hint_pass),
    ] {
        if forced != "optimize" && hint {
            return Err(format!(
                "{} = '{}' forces the encoding, and the {} hint pass would overwrite it; use one or the other",
                name, forced, name
            ));
        }
    }
    if greedy_const_policy != "default" {
        let owner = if use_cpsat {
            Some("CP-SAT (the policy is greedy-only: the CP-SAT hint cannot name an ALU derivation)".to_string())
        } else if const_choice != "optimize" {
            Some(format!("const_choice = '{}'", const_choice))
        } else if enable_const_choice_hint_pass {
            Some("the const-choice hint pass".to_string())
        } else {
            None
        };
        if let Some(owner) = owner {
            return Err(format!(
                "greedy_const_policy = '{}' (--greedy-const-policy) is ignored when {} owns the ConstChoice encoding",
                greedy_const_policy, owner
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_check_encoding_option_combinations() {
        use super::{check_encoding_option_combinations as check, GreedyPriorityKind::*};
        let ok = |p, s, cpsat, cc: &str, sc: &str, vs: &str, h: (bool, bool, bool), pol: &str| {
            check(p, s, cpsat, cc, sc, vs, h.0, h.1, h.2, pol)
        };
        let none = (false, false, false);
        assert!(ok(Fifo, None, true, "optimize", "optimize", "optimize", none, "default").is_ok());
        assert!(ok(Wavefront, Some((8, 3)), false, "load", "fma", "alu", none, "default").is_ok());
        assert!(ok(Wavefront, None, false, "optimize", "optimize", "optimize", (true, true, true), "default").is_ok());
        assert!(ok(Wavefront, None, false, "optimize", "optimize", "optimize", (false, true, true), "alu-first").is_ok());
        // Each ignored or overridden combination is an error.
        assert!(ok(Fifo, Some((8, 3)), false, "optimize", "optimize", "optimize", none, "default").is_err());
        assert!(ok(Fifo, None, false, "load", "optimize", "optimize", (true, false, false), "default").is_err());
        assert!(ok(Fifo, None, false, "optimize", "vselect", "optimize", (false, true, false), "default").is_err());
        assert!(ok(Fifo, None, false, "optimize", "optimize", "valu", (false, false, true), "default").is_err());
        assert!(ok(Fifo, None, true, "optimize", "optimize", "optimize", none, "alu-first").is_err());
        assert!(ok(Fifo, None, false, "add_imm", "optimize", "optimize", none, "alu-first").is_err());
        assert!(ok(Fifo, None, false, "optimize", "optimize", "optimize", (true, false, false), "alu-when-load-full").is_err());
    }

    #[test]
    fn test_check_greedy_shape_combinations() {
        use super::{check_greedy_shape_combinations as check, GreedyPriorityKind::*};
        // Defaults and the documented combinations are accepted.
        assert!(check(Fifo, "key", "key", "key", "never", None, "optimize", false, false).is_ok());
        assert!(check(Wavefront, "final-last", "feeds-load-first", "key", "valu", Some(3), "optimize", false, false).is_ok());
        // Each ignored combination is an error.
        assert!(check(Wavefront, "key", "key", "key", "never", Some(3), "optimize", false, false).is_err());
        assert!(check(Fifo, "key", "feeds-load-first", "key", "never", None, "optimize", false, false).is_err());
        assert!(check(Fifo, "final-last", "key", "key", "never", None, "optimize", false, false).is_err());
        assert!(check(Fifo, "tail", "key", "key", "never", None, "optimize", false, false).is_err());
        assert!(check(Wavefront, "key", "key", "key", "valu", None, "fma", false, false).is_err());
        assert!(check(Wavefront, "key", "key", "key", "valu", None, "vselect", false, false).is_err());
        assert!(check(Wavefront, "key", "key", "key", "valu", None, "optimize", true, false).is_err());
        assert!(check(Wavefront, "key", "key", "key", "valu", None, "optimize", false, true).is_err());
        assert!(check(Fifo, "key", "key", "native-first", "never", None, "optimize", false, false).is_err());
        assert!(check(Wavefront, "key", "chain-first", "native-first", "never", None, "optimize", false, false).is_ok());
        // Without overflow, the hint pass and fixed encodings are fine.
        assert!(check(Wavefront, "key", "key", "key", "never", None, "fma", true, true).is_ok());
    }

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
    fn test_access_encoding_from_flags() {
        let spec = ProblemSpec {
            cached_select: CachedSelect::Bits,
            tree_reduce_max_depth: 4,
            depth4_select_after_wrap_only: true,
            ..Default::default()
        };
        assert_eq!(spec.access_encoding(0, 0, 5), AccessEncoding::Bits);
        assert_eq!(spec.access_encoding(4, 4, 5), AccessEncoding::Gather);
        assert_eq!(spec.access_encoding(15, 4, 5), AccessEncoding::Bits);
        assert_eq!(spec.access_encoding(5, 5, 0), AccessEncoding::Gather);
        let standard = ProblemSpec::default();
        assert_eq!(standard.access_encoding(2, 2, 0), AccessEncoding::Standard);
        assert!(standard.uses_access(3, AccessEncoding::Standard));
        assert!(!standard.uses_access(4, AccessEncoding::Standard));
        assert!(standard.uses_access(4, AccessEncoding::Gather));
    }

    #[test]
    fn test_access_plan_overrides_flags_and_is_checked() {
        let n_groups = 4;
        let rounds = 16;
        let mut plan: Vec<Vec<AccessEncoding>> = (0..rounds)
            .map(|r| {
                let depth = r % 11;
                vec![if depth == 0 { AccessEncoding::Bits } else { AccessEncoding::Gather }; n_groups]
            })
            .collect();
        plan[15][1] = AccessEncoding::Mobius;
        let mut spec = ProblemSpec {
            batch_size: n_groups * VLEN,
            rounds,
            cached_select: CachedSelect::Bits,
            access_plan: Some(plan),
            ..Default::default()
        };
        assert_eq!(spec.check_access_plan(), Ok(()));
        assert_eq!(spec.access_encoding(2, 2, 0), AccessEncoding::Gather);
        assert_eq!(spec.access_encoding(15, 4, 1), AccessEncoding::Mobius);
        assert!(spec.uses_access(4, AccessEncoding::Mobius));
        assert!(!spec.uses_access(3, AccessEncoding::Mobius));

        let bad = |spec: &mut ProblemSpec, rnd: usize, g: usize, e: AccessEncoding| {
            let saved = spec.access_plan.as_ref().unwrap()[rnd][g];
            spec.access_plan.as_mut().unwrap()[rnd][g] = e;
            let result = spec.check_access_plan();
            spec.access_plan.as_mut().unwrap()[rnd][g] = saved;
            result
        };
        assert!(bad(&mut spec, 0, 0, AccessEncoding::Gather).unwrap_err().contains("depth 0"));
        assert!(bad(&mut spec, 5, 0, AccessEncoding::Bits).unwrap_err().contains("depth 5"));
        assert!(bad(&mut spec, 2, 3, AccessEncoding::Standard).unwrap_err().contains("standard"));
        spec.access_plan.as_mut().unwrap().pop();
        assert!(spec.check_access_plan().unwrap_err().contains("15 rounds"));
    }

    /// Critical-path options that would be silent no-ops are rejected for any caller of
    /// `ProblemSpec` (review of #91), and `first-pass` is the rounds up to the first wrap.
    #[test]
    fn test_check_critical_path_options() {
        let spec = |f: &dyn Fn(&mut ProblemSpec)| {
            let mut s = ProblemSpec { forest_height: 3, n_nodes: 15, batch_size: 16, rounds: 9, ..Default::default() };
            s.index_form = IndexForm::Mirrored;
            s.tree_reduce_max_depth = 2;
            s.cached_select = CachedSelect::BitsFlow;
            f(&mut s);
            s
        };
        assert_eq!(spec(&|_| {}).check_critical_path_options(), Ok(()));
        // index_early_terms needs the mirrored index.
        let e = spec(&|s| {
            s.index_early_terms = IndexEarlyTerms::Final;
            s.index_form = IndexForm::ZeroBased;
        });
        assert!(e.check_critical_path_options().unwrap_err().contains("index_early_terms"));
        // Pushdown: fine for bits-flow; a no-op for standard, mobius, and bits at depth 1 only.
        for mode in [SelectXorPushdown::FirstPass, SelectXorPushdown::All] {
            assert_eq!(spec(&|s| s.select_xor_pushdown = mode).check_critical_path_options(), Ok(()));
            for (cs, depth) in [(CachedSelect::Standard, 2), (CachedSelect::Mobius, 2), (CachedSelect::Bits, 1)] {
                let s = spec(&|s| {
                    s.select_xor_pushdown = mode;
                    s.cached_select = cs;
                    s.tree_reduce_max_depth = depth;
                });
                assert!(s.check_critical_path_options().unwrap_err().contains("select_xor_pushdown"), "{:?}", cs);
            }
            let bits2 = spec(&|s| {
                s.select_xor_pushdown = mode;
                s.cached_select = CachedSelect::Bits;
            });
            assert_eq!(bits2.check_critical_path_options(), Ok(()));
        }
        // Reassociation needs a gather round: none when every level is cached.
        assert_eq!(spec(&|s| s.reassociate_gather_xor = true).check_critical_path_options(), Ok(()));
        let all_cached = spec(&|s| {
            s.reassociate_gather_xor = true;
            s.tree_reduce_max_depth = 3;
        });
        assert!(all_cached.check_critical_path_options().unwrap_err().contains("reassociate_gather_xor"));
        // First pass: rounds 0 to forest_height (the first wrap round), each at depth = round.
        let s = spec(&|_| {});
        for r in 0..s.rounds {
            assert_eq!(s.is_first_pass_round(r), r <= 3);
            if s.is_first_pass_round(r) {
                assert_eq!(s.round_depth(r), r);
            }
        }
    }

    #[test]
    fn test_access_encoding_from_str() {
        for (s, e) in [
            ("gather", AccessEncoding::Gather),
            ("bits", AccessEncoding::Bits),
            ("bits-flow", AccessEncoding::BitsFlow),
            ("mobius", AccessEncoding::Mobius),
        ] {
            assert_eq!(AccessEncoding::from_str(s), Ok(e));
        }
        assert!(AccessEncoding::from_str("standard").is_err());
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
