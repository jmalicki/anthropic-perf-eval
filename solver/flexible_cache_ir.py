"""
Flexible Cache IR - Different groups can have different caching strategies.

Key insight: Instead of uniform caching (all groups same strategy),
allow heterogeneous strategies:
- Some groups: deeply cached (more scratch, no gather)
- Other groups: shallow/no cache (less scratch, need gather)

The scheduler chooses which groups get deep cache based on:
- Available scratch space
- Cycle cost tradeoff

Example with 32 groups and limited scratch:
- Groups 0-7: Cache rounds 0-5 (deep cache, no gather for 6 rounds)
- Groups 8-31: Cache rounds 0-2 only (shallow cache, gather for rounds 3+)

This maximizes scratch utilization while minimizing total cycles.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class CacheStrategy(Enum):
    NO_CACHE = 0      # Always gather
    SHALLOW = 1       # Cache rounds 0-2
    MEDIUM = 2        # Cache rounds 0-5
    DEEP = 3          # Cache rounds 0-9
    FULL = 4          # Cache all rounds (if scratch allows)


@dataclass
class GroupConfig:
    """Configuration for a single group."""
    group_id: int
    cache_strategy: CacheStrategy
    cache_depth: int  # Number of rounds cached


@dataclass
class FlexibleCacheConfig:
    """Configuration for flexible caching across all groups."""
    n_groups: int
    total_rounds: int
    scratch_budget: int
    group_configs: list[GroupConfig] = field(default_factory=list)

    def calculate_scratch_usage(self) -> int:
        """Calculate total scratch used by this configuration."""
        # Base scratch (constants, pointers, etc.)
        base_scratch = 200

        # Per-group scratch
        VLEN = 8
        per_group_base = VLEN * 2  # idx + val vectors

        total = base_scratch
        for gc in self.group_configs:
            total += per_group_base
            # Cache storage: depth * VLEN per cached round
            total += gc.cache_depth * VLEN

        return total

    def optimize_configuration(self):
        """Find optimal cache depth distribution given scratch budget."""
        VLEN = 8
        base = 200 + self.n_groups * VLEN * 2  # Required scratch
        remaining = self.scratch_budget - base

        if remaining <= 0:
            # No room for caching
            for g in range(self.n_groups):
                self.group_configs.append(GroupConfig(
                    group_id=g,
                    cache_strategy=CacheStrategy.NO_CACHE,
                    cache_depth=0
                ))
            return

        # Calculate how many cache slots we can afford
        cache_words_per_round_per_group = VLEN
        total_cache_slots = remaining // cache_words_per_round_per_group

        # Distribute cache slots to groups
        # Strategy: give deep cache to as many groups as possible
        max_cache_depth = min(self.total_rounds, 10)  # Cap at 10 rounds

        # How many groups can have full depth?
        full_depth_groups = total_cache_slots // max_cache_depth
        remaining_slots = total_cache_slots - (full_depth_groups * max_cache_depth)

        for g in range(self.n_groups):
            if g < full_depth_groups:
                depth = max_cache_depth
                strategy = CacheStrategy.DEEP if depth >= 6 else CacheStrategy.MEDIUM
            elif remaining_slots > 0:
                # Give partial cache
                depth = min(remaining_slots, 3)  # At least shallow
                remaining_slots -= depth
                strategy = CacheStrategy.SHALLOW
            else:
                depth = 0
                strategy = CacheStrategy.NO_CACHE

            self.group_configs.append(GroupConfig(
                group_id=g,
                cache_strategy=strategy,
                cache_depth=depth
            ))

        # Sort by cache depth for reporting
        by_depth = {}
        for gc in self.group_configs:
            by_depth[gc.cache_depth] = by_depth.get(gc.cache_depth, 0) + 1

        print(f"Flexible cache configuration:")
        print(f"  Scratch budget: {self.scratch_budget}")
        print(f"  Estimated usage: {self.calculate_scratch_usage()}")
        print(f"  Cache distribution: {by_depth}")


def estimate_cycles_for_config(config: FlexibleCacheConfig) -> int:
    """Estimate cycle count for a given configuration."""
    VLEN = 8
    VALU_SLOTS = 6
    LOAD_SLOTS = 2
    ALU_SLOTS = 12

    # Per-round costs
    cycles = 0

    for rnd in range(config.total_rounds):
        # Count groups by strategy for this round
        cached_groups = 0
        gather_groups = 0

        for gc in config.group_configs:
            if rnd < gc.cache_depth:
                cached_groups += 1
            else:
                gather_groups += 1

        # Cached groups: just VALU broadcast + XOR + hash + branch
        # ~20 VALU ops per group
        cached_valu_ops = cached_groups * 20

        # Gather groups: ALU + LOAD for gather, then VALU for rest
        # Gather: VLEN loads per group, VLEN ALU ops
        gather_load_ops = gather_groups * VLEN
        gather_alu_ops = gather_groups * VLEN
        gather_valu_ops = gather_groups * 20  # Same processing as cached

        # Estimate cycles for this round
        # VALU is usually bottleneck
        round_valu = cached_valu_ops + gather_valu_ops
        round_load = gather_load_ops
        round_alu = gather_alu_ops

        # Cycles = max(valu/6, load/2, alu/12) approximately
        # But with pipelining, it's more complex
        valu_cycles = (round_valu + VALU_SLOTS - 1) // VALU_SLOTS
        load_cycles = (round_load + LOAD_SLOTS - 1) // LOAD_SLOTS
        alu_cycles = (round_alu + ALU_SLOTS - 1) // ALU_SLOTS

        # For gather groups, load is sequential with ALU
        gather_cycles = max(load_cycles, alu_cycles)

        # Total for round (simplified - ignores pipelining)
        round_cycles = max(valu_cycles, gather_cycles)
        cycles += round_cycles

    return cycles


def find_optimal_config(n_groups: int, total_rounds: int,
                       scratch_budget: int) -> FlexibleCacheConfig:
    """Find the optimal cache configuration."""

    # Try different configurations
    best_config = None
    best_cycles = float('inf')

    # Strategy: try giving all scratch to fewer groups (deep)
    # vs spreading across more groups (shallow)

    for deep_groups in range(n_groups + 1):
        config = FlexibleCacheConfig(
            n_groups=n_groups,
            total_rounds=total_rounds,
            scratch_budget=scratch_budget
        )

        # Custom configuration: deep_groups get max cache, rest get none
        VLEN = 8
        base = 200 + n_groups * VLEN * 2
        remaining = scratch_budget - base

        if deep_groups > 0:
            depth_per_group = remaining // (deep_groups * VLEN)
            depth_per_group = min(depth_per_group, total_rounds)
        else:
            depth_per_group = 0

        config.group_configs = []
        for g in range(n_groups):
            if g < deep_groups:
                depth = depth_per_group
            else:
                depth = 0

            config.group_configs.append(GroupConfig(
                group_id=g,
                cache_strategy=CacheStrategy.DEEP if depth >= 6 else
                              CacheStrategy.MEDIUM if depth >= 3 else
                              CacheStrategy.SHALLOW if depth > 0 else
                              CacheStrategy.NO_CACHE,
                cache_depth=depth
            ))

        if config.calculate_scratch_usage() <= scratch_budget:
            cycles = estimate_cycles_for_config(config)
            if cycles < best_cycles:
                best_cycles = cycles
                best_config = config

    return best_config


if __name__ == "__main__":
    # Test with typical parameters
    N_GROUPS = 32
    TOTAL_ROUNDS = 16
    SCRATCH_BUDGET = 1536

    print("Finding optimal cache configuration...")
    print(f"  Groups: {N_GROUPS}")
    print(f"  Rounds: {TOTAL_ROUNDS}")
    print(f"  Scratch: {SCRATCH_BUDGET}")
    print()

    config = find_optimal_config(N_GROUPS, TOTAL_ROUNDS, SCRATCH_BUDGET)

    print(f"\nOptimal configuration found:")
    print(f"  Estimated cycles: {estimate_cycles_for_config(config)}")

    by_depth = {}
    for gc in config.group_configs:
        by_depth[gc.cache_depth] = by_depth.get(gc.cache_depth, 0) + 1
    print(f"  Groups by cache depth: {by_depth}")

    # Compare to uniform strategies
    print("\nComparison:")

    # All shallow (depth=3)
    uniform_shallow = FlexibleCacheConfig(N_GROUPS, TOTAL_ROUNDS, SCRATCH_BUDGET)
    uniform_shallow.group_configs = [
        GroupConfig(g, CacheStrategy.SHALLOW, 3) for g in range(N_GROUPS)
    ]
    print(f"  Uniform shallow (depth=3): {estimate_cycles_for_config(uniform_shallow)} cycles")

    # All no cache
    no_cache = FlexibleCacheConfig(N_GROUPS, TOTAL_ROUNDS, SCRATCH_BUDGET)
    no_cache.group_configs = [
        GroupConfig(g, CacheStrategy.NO_CACHE, 0) for g in range(N_GROUPS)
    ]
    print(f"  No cache: {estimate_cycles_for_config(no_cache)} cycles")
