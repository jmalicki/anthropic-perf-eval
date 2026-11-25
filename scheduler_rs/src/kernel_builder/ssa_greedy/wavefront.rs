//! Wavefront priority for the greedy scheduler.
//!
//! The kernel is a set of independent chains (one per vector group: its value and index
//! through every round) that share only constants and cached tree values. The default
//! ready-queue order is first-come-first-served, which keeps all chains in the same round:
//! in a gather round the load slots are the bottleneck and the vector ALU idles, in a select
//! round it is the other way round.
//!
//! The wavefront priority gives every op a key
//!
//! ```text
//! key(op) = depth(op) + stagger_cycles * chain_rank(op)
//! ```
//!
//! where `depth` is the op's longest dependency path from the start (its earliest cycle on an
//! unlimited machine) and `chain_rank` is the position of its chain in emission order. Ready
//! ops are tried in key order, so chain `g` runs as if it started `stagger_cycles * g` cycles
//! after chain 0: at any time some chains are gathering while others hash or select.
//! `stagger_cycles = 0` is plain earliest-depth-first (all chains abreast); a very large value
//! finishes one chain before starting the next.
//!
//! `groups_in_flight` bounds scratch use: only the first N unfinished chains may run. A chain
//! that has not started holds no scratch (its initial loads are private to it) and a finished
//! chain has stored its results.
//!
//! Chains are found structurally, not from emission metadata. Every op reaches some number
//! of the program's sinks (ops nothing depends on, i.e. the final stores). The typical op
//! belongs to one chain and reaches that chain's sinks only, so the median of that number is
//! the number of sinks of one chain. An op that reaches more sinks than the median is
//! *shared* (a constant, a cached tree vector, an initial value common to all chains);
//! the others are *private*, and chains are the connected components of private ops. Shared
//! ops take the key of their most urgent consumer.

use std::collections::{HashMap, HashSet};

use crate::types::{PrecedenceOffset, SsaOp};

/// Ready-queue order of the greedy scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GreedyPriority {
    /// First-come-first-served by ready cycle, then emission order (default).
    #[default]
    Fifo,
    /// Wavefront key order (see the module documentation).
    Wavefront {
        /// Virtual start delay between consecutive chains, in cycles.
        stagger_cycles: u32,
        /// Number of unfinished chains allowed to run at once; `None` = no limit.
        groups_in_flight: Option<usize>,
    },
}

/// A different stagger for the first chains: chain `rank` starts
/// `cycles * min(rank, groups) + stagger * max(rank - groups, 0)` cycles after chain 0, so the
/// first `groups` chains can enter together (or closer) while the rest keep the normal stagger.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EarlyStagger {
    /// How many leading chains use `cycles`.
    pub groups: usize,
    /// Stagger between consecutive leading chains.
    pub cycles: u32,
}

/// Virtual start delay of chain `rank` (see [`EarlyStagger`]).
pub(super) fn chain_offset(rank: usize, stagger_cycles: u32, early: Option<EarlyStagger>) -> i64 {
    match early {
        None => stagger_cycles as i64 * rank as i64,
        Some(e) => {
            let lead = rank.min(e.groups);
            e.cycles as i64 * lead as i64 + stagger_cycles as i64 * (rank - lead) as i64
        }
    }
}

/// Per-op wavefront keys and the in-flight window over chains.
#[derive(Debug)]
pub(super) struct Wavefront {
    /// Priority key per op index (lower runs first).
    key: Vec<i64>,
    /// Longest dependency path from the op to the end of the program, per op index.
    height: Vec<i64>,
    /// Whether some load depends on the op, per op index.
    feeds_load: Vec<bool>,
    /// Whether a non-final load (one that some later load depends on) depends on the op.
    feeds_nonfinal_load: Vec<bool>,
    /// Chain rank per op index; `None` for shared ops.
    chain: Vec<Option<usize>>,
    /// Ops still to be scheduled per chain.
    remaining: Vec<usize>,
    /// Chains with a rank below this may run: it is one past the `window`-th unfinished chain.
    admit_below: usize,
    /// Unfinished chains allowed to run at once.
    window: usize,
}

impl Wavefront {
    /// Compute keys and chains for `ops`.
    ///
    /// # Arguments
    /// * `ops` - The ops to schedule, with `dep_ids` set
    /// * `consumers` - Per op index, the indices of the ops that depend on it
    /// * `id_to_idx` - Op id → index into `ops`
    /// * `shiftable_set` - Indices of shiftable vector ops (for precedence offsets)
    /// * `skipped` - Indices the scheduler will not schedule (dead stores)
    /// * `stagger_cycles` - Virtual start delay between consecutive chains
    /// * `groups_in_flight` - Chains allowed to run at once; `None` = no limit
    /// * `early` - A different stagger for the first chains (see [`EarlyStagger`]); `None` = none
    ///
    /// # Returns
    /// The wavefront state, or `Err` if the dependency graph has a cycle.
    pub(super) fn new(
        ops: &[SsaOp],
        consumers: &[Vec<usize>],
        id_to_idx: &HashMap<usize, usize>,
        shiftable_set: &HashSet<usize>,
        skipped: &HashSet<usize>,
        stagger_cycles: u32,
        groups_in_flight: Option<usize>,
        early: Option<EarlyStagger>,
    ) -> Result<Self, String> {
        let n = ops.len();
        let order = topological_order(ops, consumers, id_to_idx)?;

        // Longest dependency path to each op (its earliest cycle on an unlimited machine).
        let mut depth = vec![0i64; n];
        for &idx in &order {
            let offset = ops[idx].precedence_offset_for_asap(shiftable_set.contains(&idx)) as i64;
            for &consumer in &consumers[idx] {
                depth[consumer] = depth[consumer].max(depth[idx] + offset);
            }
        }

        // Longest dependency path from each op to the end of the program (its own latency
        // included): the cycles that must follow its issue on an unlimited machine.
        let mut height = vec![0i64; n];
        for &idx in order.iter().rev() {
            let offset = ops[idx].precedence_offset_for_asap(shiftable_set.contains(&idx)) as i64;
            let below = consumers[idx].iter().map(|&c| height[c]).max().unwrap_or(0);
            height[idx] = offset + below;
        }

        // Whether a load is reachable from the op (through at least one consumer).
        let mut feeds_load = vec![false; n];
        for &idx in order.iter().rev() {
            feeds_load[idx] = consumers[idx]
                .iter()
                .any(|&c| feeds_load[c] || ops[c].engine == crate::types::Engine::Load);
        }
        // Whether a non-final load (a load with `feeds_load`) is reachable from the op.
        let mut feeds_nonfinal_load = vec![false; n];
        for &idx in order.iter().rev() {
            feeds_nonfinal_load[idx] = consumers[idx].iter().any(|&c| {
                feeds_nonfinal_load[c] || (ops[c].engine == crate::types::Engine::Load && feeds_load[c])
            });
        }

        // Which sinks each op reaches, as a bitset over sinks.
        let sinks: Vec<usize> = (0..n)
            .filter(|i| consumers[*i].is_empty() && !skipped.contains(i))
            .collect();
        let words = sinks.len().div_ceil(64).max(1);
        let mut reach = vec![0u64; n * words];
        for (k, &sink) in sinks.iter().enumerate() {
            reach[sink * words + k / 64] |= 1 << (k % 64);
        }
        for &idx in order.iter().rev() {
            for &consumer in &consumers[idx] {
                for w in 0..words {
                    let bits = reach[consumer * words + w];
                    reach[idx * words + w] |= bits;
                }
            }
        }
        // Shared: reaches more sinks than the typical (median) op, which reaches the sinks of
        // its own chain only.
        let reached: Vec<usize> = (0..n)
            .map(|i| {
                if skipped.contains(&i) {
                    0
                } else {
                    (0..words).map(|w| reach[i * words + w].count_ones() as usize).sum()
                }
            })
            .collect();
        let mut counts: Vec<usize> = reached.iter().copied().filter(|&r| r > 0).collect();
        counts.sort_unstable();
        let shared_above = counts.get(counts.len().saturating_sub(1) / 2).copied().unwrap_or(0);
        let private: Vec<bool> = reached.iter().map(|&r| r > 0 && r <= shared_above).collect();

        // Chains: connected components of private ops, ranked by their first op.
        let shared = n;
        let mut parent: Vec<usize> = (0..=n).collect();
        for idx in 0..n {
            if !private[idx] {
                continue;
            }
            for &consumer in &consumers[idx] {
                if private[consumer] {
                    join(&mut parent, idx, consumer);
                }
            }
        }
        // A bounded window can admit these data-flow chains independently only if no
        // chain needs an op from another, unadmitted chain. Join private chains linked
        // by memory ordering. A private chain needed by a shared op must itself be
        // available in every window. A shared source already runs in every window.
        for (target, op) in ops.iter().enumerate() {
            if skipped.contains(&target) {
                continue;
            }
            for (id, _) in &op.mem_deps {
                if let Some(&source) = id_to_idx.get(id) {
                    if private[source] {
                        join(&mut parent, source, if private[target] { target } else { shared });
                    }
                }
            }
        }
        let shared_root = find_root(&mut parent, shared);
        let chain_root: Vec<Option<usize>> = (0..n)
            .map(|idx| private[idx].then(|| find_root(&mut parent, idx)).filter(|&root| root != shared_root))
            .collect();
        let mut roots: Vec<usize> = chain_root.iter().flatten().copied().collect();
        roots.sort_unstable();
        roots.dedup();
        let rank_of_root: HashMap<usize, usize> =
            roots.iter().enumerate().map(|(rank, &root)| (root, rank)).collect();
        let chain: Vec<Option<usize>> = chain_root.iter().map(|root| root.map(|r| rank_of_root[&r])).collect();

        let mut key = vec![i64::MAX; n];
        for idx in 0..n {
            if let Some(rank) = chain[idx] {
                key[idx] = depth[idx] + chain_offset(rank, stagger_cycles, early);
            }
        }
        // A shared op is needed just before its most urgent consumer.
        for &idx in order.iter().rev() {
            if chain[idx].is_none() {
                let urgent = consumers[idx].iter().map(|&c| key[c]).min().unwrap_or(i64::MAX);
                key[idx] = if urgent == i64::MAX { depth[idx] } else { urgent - 1 };
            }
        }

        let mut remaining = vec![0usize; roots.len()];
        for idx in 0..n {
            if let Some(rank) = chain[idx] {
                remaining[rank] += 1;
            }
        }
        let mut wavefront = Wavefront {
            key,
            height,
            feeds_load,
            feeds_nonfinal_load,
            chain,
            remaining,
            admit_below: 0,
            window: groups_in_flight.unwrap_or(usize::MAX).max(1),
        };
        wavefront.refresh_window();
        Ok(wavefront)
    }

    /// Priority key of an op (lower runs first).
    pub(super) fn key(&self, idx: usize) -> i64 {
        self.key[idx]
    }

    /// Longest dependency path from an op to the end of the program (its latency included).
    pub(super) fn height(&self, idx: usize) -> i64 {
        self.height[idx]
    }

    /// Whether some load depends (directly or not) on the op. A load for which this is false
    /// is a *final* load: no later load waits for it.
    pub(super) fn feeds_load(&self, idx: usize) -> bool {
        self.feeds_load[idx]
    }

    /// Whether a non-final load (one some later load depends on) depends on the op.
    pub(super) fn feeds_nonfinal_load(&self, idx: usize) -> bool {
        self.feeds_nonfinal_load[idx]
    }

    /// Chain rank of an op, `None` for shared ops.
    pub(super) fn chain(&self, idx: usize) -> Option<usize> {
        self.chain[idx]
    }

    /// Number of chains found.
    pub(super) fn chain_count(&self) -> usize {
        self.remaining.len()
    }

    /// Whether an op may run now: shared ops always, chain ops when their chain is one of
    /// the first `groups_in_flight` unfinished chains.
    pub(super) fn in_window(&self, idx: usize) -> bool {
        match self.chain[idx] {
            Some(rank) => rank < self.admit_below,
            None => true,
        }
    }

    /// Record that an op has been scheduled.
    pub(super) fn on_scheduled(&mut self, idx: usize) {
        if let Some(rank) = self.chain[idx] {
            self.remaining[rank] -= 1;
            if self.remaining[rank] == 0 {
                self.refresh_window();
            }
        }
    }

    /// Admit the first `window` unfinished chains, whichever chains have finished: a chain
    /// that finishes out of order makes room for the next one.
    fn refresh_window(&mut self) {
        let mut unfinished = 0;
        self.admit_below = self.remaining.len();
        for (rank, &left) in self.remaining.iter().enumerate() {
            if left > 0 {
                unfinished += 1;
                if unfinished == self.window {
                    self.admit_below = rank + 1;
                    break;
                }
            }
        }
    }
}

/// Keep the first emitted op as the representative of the joined chains.
fn join(parent: &mut [usize], a: usize, b: usize) {
    let a = find_root(parent, a);
    let b = find_root(parent, b);
    parent[a.max(b)] = a.min(b);
}

/// Union-find root with path halving.
fn find_root(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

/// Op indices in dependency order (Kahn's algorithm, lowest index first among ready ops).
fn topological_order(
    ops: &[SsaOp],
    consumers: &[Vec<usize>],
    id_to_idx: &HashMap<usize, usize>,
) -> Result<Vec<usize>, String> {
    let n = ops.len();
    let mut pending: Vec<usize> = ops
        .iter()
        .map(|op| op.dep_ids.iter().filter(|dep| id_to_idx.contains_key(dep)).count())
        .collect();
    let mut ready: std::collections::BTreeSet<usize> = (0..n).filter(|&i| pending[i] == 0).collect();
    let mut order = Vec::with_capacity(n);
    while let Some(idx) = ready.pop_first() {
        order.push(idx);
        for &consumer in &consumers[idx] {
            pending[consumer] -= 1;
            if pending[consumer] == 0 {
                ready.insert(consumer);
            }
        }
    }
    if order.len() != n {
        return Err(format!(
            "wavefront priority: dependency cycle ({} of {} ops ordered)",
            order.len(),
            n
        ));
    }
    Ok(order)
}
