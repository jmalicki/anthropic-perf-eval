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
//! Chains are found structurally, not from emission metadata: an op is *shared* when it
//! reaches more than half of the program's sinks (ops nothing depends on, i.e. the final
//! stores), otherwise *private*; chains are the connected components of private ops. Shared
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

/// Per-op wavefront keys and the in-flight window over chains.
#[derive(Debug)]
pub(super) struct Wavefront {
    /// Priority key per op index (lower runs first).
    key: Vec<i64>,
    /// Chain rank per op index; `None` for shared ops.
    chain: Vec<Option<usize>>,
    /// Ops still to be scheduled per chain.
    remaining: Vec<usize>,
    /// Lowest chain rank with ops still to be scheduled.
    front: usize,
    /// Chains allowed to run at once.
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
        // Shared: reaches more than half of the sinks (never with one or two sinks).
        let shared_above = (sinks.len() / 2).max(2);
        let private: Vec<bool> = (0..n)
            .map(|i| {
                let reached: usize =
                    (0..words).map(|w| reach[i * words + w].count_ones() as usize).sum();
                !skipped.contains(&i) && reached > 0 && reached <= shared_above
            })
            .collect();

        // Chains: connected components of private ops, ranked by their first op.
        let mut parent: Vec<usize> = (0..n).collect();
        for idx in 0..n {
            if !private[idx] {
                continue;
            }
            for &consumer in &consumers[idx] {
                if private[consumer] {
                    let (a, b) = (find_root(&mut parent, idx), find_root(&mut parent, consumer));
                    // The root is the smallest op index of the component.
                    parent[a.max(b)] = a.min(b);
                }
            }
        }
        let mut roots: Vec<usize> =
            (0..n).filter(|&i| private[i]).map(|i| find_root(&mut parent, i)).collect();
        roots.sort_unstable();
        roots.dedup();
        let rank_of_root: HashMap<usize, usize> =
            roots.iter().enumerate().map(|(rank, &root)| (root, rank)).collect();
        let chain: Vec<Option<usize>> = (0..n)
            .map(|i| private[i].then(|| rank_of_root[&find_root(&mut parent, i)]))
            .collect();

        let mut key = vec![i64::MAX; n];
        for idx in 0..n {
            if let Some(rank) = chain[idx] {
                key[idx] = depth[idx] + stagger_cycles as i64 * rank as i64;
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
            chain,
            remaining,
            front: 0,
            window: groups_in_flight.unwrap_or(usize::MAX).max(1),
        };
        wavefront.advance_front();
        Ok(wavefront)
    }

    /// Priority key of an op (lower runs first).
    pub(super) fn key(&self, idx: usize) -> i64 {
        self.key[idx]
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
            Some(rank) => rank < self.front.saturating_add(self.window),
            None => true,
        }
    }

    /// Record that an op has been scheduled.
    pub(super) fn on_scheduled(&mut self, idx: usize) {
        if let Some(rank) = self.chain[idx] {
            self.remaining[rank] -= 1;
            self.advance_front();
        }
    }

    fn advance_front(&mut self) {
        while self.front < self.remaining.len() && self.remaining[self.front] == 0 {
            self.front += 1;
        }
    }
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
