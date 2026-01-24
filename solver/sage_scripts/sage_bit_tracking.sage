#!/usr/bin/env sage
"""
SageMath Analysis: Bit Dependency Tracking

This script tracks which input bits affect which output bits through the hash.
The key question: Does the branch decision (output bit 0) depend on all 32 input bits,
or can we compute it from fewer bits?

If bit 0 of output depends on only k < 32 input bits, we could potentially
use a lookup table or simplified computation for branch prediction.
"""

from sage.all import *
import sys

# Constants from the hash function
HASH_CONSTANTS = [
    0x7ED55D16,  # Stage 0
    0xC761C23C,  # Stage 1
    0x165667B1,  # Stage 2
    0xD3A2646C,  # Stage 3
    0xFD7046C5,  # Stage 4
    0xB55A4F09,  # Stage 5
]

MOD = 2^32

def myhash_sage(a):
    """Reference implementation in Sage."""
    a = ZZ(a)
    a = (a + HASH_CONSTANTS[0] + (a << 12)) % MOD
    a = (a ^^ HASH_CONSTANTS[1]) ^^ (a >> 19)
    a = (a + HASH_CONSTANTS[2] + (a << 5)) % MOD
    a = ((a + HASH_CONSTANTS[3]) % MOD) ^^ ((a << 9) % MOD)
    a = (a + HASH_CONSTANTS[4] + (a << 3)) % MOD
    a = (a ^^ HASH_CONSTANTS[5]) ^^ (a >> 16)
    return int(a)

# =============================================================================
# INVESTIGATION 1: Single-Bit Input Sensitivity
# =============================================================================
print("=" * 70)
print("INVESTIGATION 1: Single-Bit Input Sensitivity for Output Bit 0")
print("=" * 70)
print()

# For each input bit i, measure how often flipping it changes output bit 0
print("Testing: If we flip input bit i, how often does output bit 0 change?")
print("(Testing over 10000 random inputs per bit)")
print()

import random
random.seed(int(42))

sensitivity = []
for input_bit in range(32):
    flips = 0
    total = 10000
    for _ in range(total):
        a = random.randint(0, MOD - 1)
        a_flipped = a ^^ (1 << input_bit)

        h_a = myhash_sage(a)
        h_flipped = myhash_sage(a_flipped)

        if (h_a % 2) != (h_flipped % 2):
            flips += 1

    sensitivity.append(float(flips) / float(total))
    print(f"  Bit {input_bit:2d}: {float(100*flips/total):.1f}% of the time flipping changes output bit 0")

print()
print(f"All sensitivities near 50%: {all(0.45 <= s <= 0.55 for s in sensitivity)}")
print("-> Each input bit has roughly 50% chance of affecting output bit 0")
print("-> Cannot ignore any input bits for branch prediction")
print()

# =============================================================================
# INVESTIGATION 2: GF(2) Dependency Matrix Through Stages
# =============================================================================
print("=" * 70)
print("INVESTIGATION 2: GF(2) Bit Dependency Through Stages")
print("=" * 70)
print()

# Track bit dependencies through XOR stages (linear over GF(2))
# Addition stages are nonlinear but we can track "potential" dependencies

def build_shift_matrix(shift, direction='right', size=32):
    """Build a size x size GF(2) matrix for a shift operation."""
    F = GF(2)
    M = matrix(F, size, size)
    for i in range(size):
        if direction == 'right':
            if i + shift < size:
                M[i, i + shift] = 1
        else:  # left
            if i - shift >= 0:
                M[i, i - shift] = 1
    return M

F = GF(2)
I = identity_matrix(F, 32)

print("XOR-only stage transformations (ignoring constants):")
print()

# Stage 1: a' = a ^ (a >> 19)
M_right_19 = build_shift_matrix(19, 'right')
T1 = I + M_right_19
print("Stage 1: a' = a ^ (a >> 19)")
print(f"  Transform rank: {T1.rank()}")

# Stage 5: a' = a ^ (a >> 16)
M_right_16 = build_shift_matrix(16, 'right')
T5 = I + M_right_16
print("Stage 5: a' = a ^ (a >> 16)")
print(f"  Transform rank: {T5.rank()}")
print()

# Compose XOR stages
T_xor = T1 * T5
print(f"Composed XOR stages (T1 * T5) rank: {T_xor.rank()}")
print()

# Which input bits affect output bit 0 through XOR stages only?
# Output bit 0 is row 0 of the composed matrix
row_0 = T_xor[0]
input_bits_for_xor = [i for i in range(32) if row_0[i] == 1]
print(f"For XOR stages only, output bit 0 depends on input bits: {input_bits_for_xor}")
print()

# =============================================================================
# INVESTIGATION 3: Carry Propagation in Addition
# =============================================================================
print("=" * 70)
print("INVESTIGATION 3: Carry Propagation in Addition")
print("=" * 70)
print()

# Addition mod 2^32 causes carries that propagate bit dependencies upward
# Bit i of (a + b) depends on bits 0..i of both a and b (due to carries)

print("In a + b mod 2^32:")
print("  - Bit 0 of result depends only on bits 0 of a and b")
print("  - Bit i of result depends on bits 0..i of a and b")
print()

# For Stage 0: a' = a + (a << 12) + C
# Bit 0 of a' depends on:
#   - Bit 0 of a
#   - Bit 0 of (a << 12) = 0 (since shift left fills with 0)
#   - Bit 0 of C = 0x7ED55D16 & 1 = 0

print("Stage 0: a' = a + (a << 12) + C0 where C0 = 0x7ED55D16")
print(f"  C0 bit 0 = {HASH_CONSTANTS[0] & 1}")
print(f"  (a << 12) bit 0 = 0 (always)")
print("  So bit 0 of Stage 0 output depends only on bit 0 of input + carries")
print()

# But higher stages mix things up
print("After Stage 1 XOR: dependencies spread due to right shift")
print("After Stage 2 addition: carries can propagate from any lower bit")
print()

# =============================================================================
# INVESTIGATION 4: Empirical Bit Dependency
# =============================================================================
print("=" * 70)
print("INVESTIGATION 4: Empirical Bit Dependency Analysis")
print("=" * 70)
print()

# Build an empirical dependency map by observing which input bits
# affect each output bit

print("Computing empirical bit dependency matrix...")
print("(For each output bit, testing which input bits affect it)")
print()

# dep_matrix[out_bit][in_bit] = probability that flipping in_bit changes out_bit
dep_matrix = [[0.0] * 32 for _ in range(32)]

random.seed(int(42))
num_samples = 5000

for in_bit in range(32):
    for _ in range(num_samples):
        a = random.randint(0, MOD - 1)
        a_flipped = a ^^ (1 << in_bit)

        h_a = myhash_sage(a)
        h_flipped = myhash_sage(a_flipped)

        diff = h_a ^^ h_flipped  # which output bits changed

        for out_bit in range(32):
            if (diff >> out_bit) & 1:
                dep_matrix[out_bit][in_bit] += 1.0 / num_samples

# Visualize which input bits most strongly affect output bit 0
print("Input bit influence on output bit 0 (branch decision):")
bit0_deps = dep_matrix[0]
sorted_deps = sorted(enumerate(bit0_deps), key=lambda x: -x[1])
for in_bit, prob in sorted_deps:
    bar = '#' * int(float(prob) * 40)
    print(f"  In bit {in_bit:2d}: {float(prob):.3f} {bar}")
print()

# Check if any input bits have near-0 or near-1 influence
low_influence = [i for i, p in enumerate(bit0_deps) if p < 0.4]
high_influence = [i for i, p in enumerate(bit0_deps) if p > 0.6]

print(f"Input bits with < 40% influence on output bit 0: {low_influence if low_influence else 'None'}")
print(f"Input bits with > 60% influence on output bit 0: {high_influence if high_influence else 'None'}")
print()

# =============================================================================
# INVESTIGATION 5: Partial Hash for Branch Only
# =============================================================================
print("=" * 70)
print("INVESTIGATION 5: Can We Compute Only Bit 0?")
print("=" * 70)
print()

# In principle, if we only need bit 0, we could track only the computations
# that affect bit 0. Let's see if this saves any work.

print("Tracking minimum bits needed through each stage for bit 0:")
print()

# Stage 0: a' = a * 4097 + C0
# Bit 0 of a * 4097:
#   4097 = 2^12 + 1, so a * 4097 = a + (a << 12)
#   Bit 0 of this = bit 0 of a (since a << 12 has 0 in bit 0)
# Bit 0 of a' = bit 0 of (a + C0) = bit 0 of a XOR bit 0 of C0 XOR carry_in
#   where carry_in depends on lower bits (there are none for bit 0)
print("Stage 0: bit 0 of output = f(bit 0 of input)")
print("  Multiplier 4097 = 2^12 + 1, affects bit 0 only via the +1 term")
print()

# But Stage 1 XOR with right shift brings in higher bits!
# a' = a ^ (a >> 19)
# Bit 0 of a' = bit 0 of a ^ bit 19 of a
print("Stage 1: bit 0 of output = bit 0 XOR bit 19 of input")
print("  Now we need bits 0 and 19!")
print()

# Stage 2: a' = a * 33 + C2
# 33 = 2^5 + 1
# Similar to Stage 0, multiplication by 33 affects bit 0 only via +1
print("Stage 2: bit 0 depends on bits 0 (from mult) and 5+ (from carries)")
print()

# Stage 3: a' = (a + C3) ^ (a << 9)
# Bit 0 of a << 9 = 0
# Bit 0 of a' = bit 0 of (a + C3)
print("Stage 3: bit 0 depends on bit 0 of addition (carries complicate)")
print()

# Stage 4: similar to 0 and 2
print("Stage 4: bit 0 depends on multiple bits due to carry chain")
print()

# Stage 5: a' = a ^ (a >> 16)
# Bit 0 of a' = bit 0 of a ^ bit 16 of a
print("Stage 5: bit 0 = bit 0 XOR bit 16 of input")
print()

print("CONCLUSION: Due to carries in addition stages, bit 0 of the final")
print("output depends on ALL 32 input bits. No simplification possible.")
print()

# =============================================================================
# INVESTIGATION 6: Statistical Structure of Branch Decisions
# =============================================================================
print("=" * 70)
print("INVESTIGATION 6: Statistical Structure of Branch Decisions")
print("=" * 70)
print()

# Check if consecutive inputs have correlated branches
print("Correlation between branches for consecutive inputs:")
correlations = []
for offset in [1, 2, 4, 8, 16, 256]:
    same = 0
    for a in range(10000):
        b = (a + offset) % MOD
        ha = myhash_sage(a)
        hb = myhash_sage(b)
        if (ha % 2) == (hb % 2):
            same += 1
    correlations.append((offset, float(same) / 10000))
    print(f"  Offset {offset:3d}: {float(100*same/10000):.1f}% same branch")
print()

print("-> Near 50% for all offsets indicates no exploitable correlation")
print()

# Check if XOR with same value preserves branch
print("Branch correlation when XORing with same constant:")
for c in [1, 0x10, 0x100, 0x10000, 0x12345678]:
    same = 0
    for a in range(10000):
        ha = myhash_sage(a)
        hac = myhash_sage(a ^^ c)
        if (ha % 2) == (hac % 2):
            same += 1
    print(f"  c = 0x{c:08x}: {float(100*same/10000):.1f}% same branch")
print()

# =============================================================================
# SUMMARY
# =============================================================================
print("=" * 70)
print("SUMMARY OF BIT TRACKING FINDINGS")
print("=" * 70)
print()
print("1. BIT SENSITIVITY:")
print("   - Each of 32 input bits has ~50% chance of flipping output bit 0")
print("   - No input bits can be ignored for branch prediction")
print()
print("2. CARRY PROPAGATION:")
print("   - Addition stages cause full carry chains")
print("   - Bit 0 of output ultimately depends on all input bits")
print()
print("3. XOR STAGES:")
print("   - Stage 1: mixes bits 0 and 19")
print("   - Stage 5: mixes bits 0 and 16")
print("   - These spread dependencies across the full word")
print()
print("4. STATISTICAL INDEPENDENCE:")
print("   - Branch decisions show no exploitable correlation")
print("   - ~50/50 distribution for all tested patterns")
print()
print("5. CONCLUSION:")
print("   - Cannot compute branch from subset of input bits")
print("   - Full 32-bit hash computation required for branch")
print("   - No lazy evaluation optimization possible")
print()
