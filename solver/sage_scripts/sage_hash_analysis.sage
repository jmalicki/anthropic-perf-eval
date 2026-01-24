#!/usr/bin/env sage
"""
SageMath Analysis: Hash Function Algebraic Structure

This script analyzes the 6-stage hash function used in the tree traversal kernel.
We explore whether the hash can be simplified or represented in a more computable form.

Hash stages (all mod 2^32):
  Stage 0: a = (a + 0x7ED55D16) + (a << 12)  = a * 4097 + C0
  Stage 1: a = (a ^ 0xC761C23C) ^ (a >> 19)
  Stage 2: a = (a + 0x165667B1) + (a << 5)   = a * 33 + C2
  Stage 3: a = (a + 0xD3A2646C) ^ (a << 9)
  Stage 4: a = (a + 0xFD7046C5) + (a << 3)   = a * 9 + C4
  Stage 5: a = (a ^ 0xB55A4F09) ^ (a >> 16)
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
    # Stage 0: a = (a + C0) + (a << 12) = a * 4097 + C0
    a = (a + HASH_CONSTANTS[0] + (a << 12)) % MOD
    # Stage 1: a = (a ^ C1) ^ (a >> 19)
    a = (a ^^ HASH_CONSTANTS[1]) ^^ (a >> 19)
    # Stage 2: a = (a + C2) + (a << 5) = a * 33 + C2
    a = (a + HASH_CONSTANTS[2] + (a << 5)) % MOD
    # Stage 3: a = (a + C3) ^ (a << 9)
    a = ((a + HASH_CONSTANTS[3]) % MOD) ^^ ((a << 9) % MOD)
    # Stage 4: a = (a + C4) + (a << 3) = a * 9 + C4
    a = (a + HASH_CONSTANTS[4] + (a << 3)) % MOD
    # Stage 5: a = (a ^ C5) ^ (a >> 16)
    a = (a ^^ HASH_CONSTANTS[5]) ^^ (a >> 16)
    return int(a)

# =============================================================================
# INVESTIGATION 1: Linear Stage Composition
# =============================================================================
print("=" * 70)
print("INVESTIGATION 1: Linear Stage Composition")
print("=" * 70)
print()

R = IntegerModRing(MOD)

# Stages 0, 2, 4 are linear: a -> m*a + c
# Stage 0: a * (1 + 2^12) + C0 = a * 4097 + C0
# Stage 2: a * (1 + 2^5) + C2 = a * 33 + C2
# Stage 4: a * (1 + 2^3) + C4 = a * 9 + C4

m0, c0 = R(1 + 2^12), R(HASH_CONSTANTS[0])
m2, c2 = R(1 + 2^5), R(HASH_CONSTANTS[2])
m4, c4 = R(1 + 2^3), R(HASH_CONSTANTS[4])

print(f"Stage 0: a * {m0} + {hex(int(c0))}")
print(f"Stage 2: a * {m2} + {hex(int(c2))}")
print(f"Stage 4: a * {m4} + {hex(int(c4))}")
print()

# Combined multiplier if stages were consecutive
m_combined = R(m0 * m2 * m4)
print(f"Combined multiplier (if stages consecutive): {m_combined}")
print(f"  = {int(m0)} * {int(m2)} * {int(m4)}")
print(f"  = {int(m0) * int(m2) * int(m4)} mod 2^32")
print()

# Check if multipliers are invertible (must be odd for mod 2^32)
print("Multiplier invertibility (must be odd):")
print(f"  m0 = {int(m0)}: {'odd (invertible)' if int(m0) % 2 == 1 else 'EVEN (not invertible)'}")
print(f"  m2 = {int(m2)}: {'odd (invertible)' if int(m2) % 2 == 1 else 'EVEN (not invertible)'}")
print(f"  m4 = {int(m4)}: {'odd (invertible)' if int(m4) % 2 == 1 else 'EVEN (not invertible)'}")
print(f"  m_combined = {int(m_combined)}: {'odd (invertible)' if int(m_combined) % 2 == 1 else 'EVEN (not invertible)'}")
print()

# =============================================================================
# INVESTIGATION 2: XOR Stages as GF(2) Matrices
# =============================================================================
print("=" * 70)
print("INVESTIGATION 2: XOR Stages as GF(2) Linear Transformations")
print("=" * 70)
print()

def build_shift_matrix(shift, direction='right'):
    """Build a 32x32 GF(2) matrix for a shift operation."""
    F = GF(2)
    M = matrix(F, 32, 32)
    for i in range(32):
        if direction == 'right':
            if i + shift < 32:
                M[i, i + shift] = 1  # Bit at position i+shift goes to position i
        else:  # left
            if i - shift >= 0:
                M[i, i - shift] = 1  # Bit at position i-shift goes to position i
    return M

F = GF(2)
I = identity_matrix(F, 32)

# Stage 1: a' = (a ^ C1) ^ (a >> 19)
# Ignoring constant XOR, the linear part is: a' = a ^ (a >> 19) = (I + M_right_19) * a
M_right_19 = build_shift_matrix(19, 'right')
T1_linear = I + M_right_19

# Stage 3: a' = (a + C3) ^ (a << 9)
# The shift part is: a << 9
M_left_9 = build_shift_matrix(9, 'left')

# Stage 5: a' = (a ^ C5) ^ (a >> 16)
M_right_16 = build_shift_matrix(16, 'right')
T5_linear = I + M_right_16

print("Stage 1 XOR transform: a ^ (a >> 19)")
print(f"  Matrix rank: {T1_linear.rank()}")
print(f"  Kernel dimension: {32 - T1_linear.rank()}")
if T1_linear.rank() < 32:
    print(f"  Kernel basis vectors: {T1_linear.right_kernel().basis()}")
print()

print("Stage 5 XOR transform: a ^ (a >> 16)")
print(f"  Matrix rank: {T5_linear.rank()}")
print(f"  Kernel dimension: {32 - T5_linear.rank()}")
if T5_linear.rank() < 32:
    print(f"  Kernel basis vectors: {T5_linear.right_kernel().basis()}")
print()

# Check if these transformations are invertible
print("Invertibility check:")
print(f"  T1 (a ^ (a >> 19)) invertible: {T1_linear.is_invertible()}")
print(f"  T5 (a ^ (a >> 16)) invertible: {T5_linear.is_invertible()}")
print()

# Compute inverse if invertible
if T1_linear.is_invertible():
    T1_inv = T1_linear.inverse()
    print("T1 inverse computed - can reverse Stage 1 XOR operation")
if T5_linear.is_invertible():
    T5_inv = T5_linear.inverse()
    print("T5 inverse computed - can reverse Stage 5 XOR operation")
print()

# =============================================================================
# INVESTIGATION 3: Composed XOR Transformation
# =============================================================================
print("=" * 70)
print("INVESTIGATION 3: Composed XOR Transformations")
print("=" * 70)
print()

# If we could factor out all XOR operations...
# This is hypothetical since XOR doesn't commute with addition mod 2^32
T_xor_composed = T1_linear * T5_linear
print("Composed XOR matrix (T1 * T5):")
print(f"  Rank: {T_xor_composed.rank()}")
print(f"  Invertible: {T_xor_composed.is_invertible()}")

# Check eigenvalues (in algebraic closure of GF(2))
# All eigenvalues of XOR transforms over GF(2) are 1 (nilpotent + identity)
print()
print("Eigenvalue analysis:")
char_poly_1 = T1_linear.characteristic_polynomial()
char_poly_5 = T5_linear.characteristic_polynomial()
print(f"  T1 characteristic polynomial: {char_poly_1}")
print(f"  T5 characteristic polynomial: {char_poly_5}")
print()

# =============================================================================
# INVESTIGATION 4: Fixed Points and Cycles
# =============================================================================
print("=" * 70)
print("INVESTIGATION 4: Fixed Points and Small Cycles")
print("=" * 70)
print()

print("Searching for fixed points (h(a) = a)...")
fixed_points = []
for a in range(min(10000, MOD)):
    if myhash_sage(a) == a:
        fixed_points.append(a)
print(f"Fixed points found in [0, 10000): {fixed_points if fixed_points else 'None'}")
print()

print("Searching for 2-cycles (h(h(a)) = a)...")
two_cycles = []
for a in range(min(10000, MOD)):
    h_a = myhash_sage(a)
    if myhash_sage(h_a) == a and h_a != a:
        two_cycles.append((a, h_a))
if two_cycles:
    print(f"2-cycles found: {two_cycles[:10]}{'...' if len(two_cycles) > 10 else ''}")
else:
    print("No 2-cycles found in [0, 10000)")
print()

# =============================================================================
# INVESTIGATION 5: Output Bit 0 (Branch Decision)
# =============================================================================
print("=" * 70)
print("INVESTIGATION 5: Output Bit 0 Distribution (Branch Decision)")
print("=" * 70)
print()

# Only the LSB of the hash output determines the branch
# Check if there's bias or predictability
count_even = 0
count_odd = 0
sample_size = 100000
for a in range(sample_size):
    h = myhash_sage(a)
    if h % 2 == 0:
        count_even += 1
    else:
        count_odd += 1

print(f"Sample size: {sample_size}")
print(f"Even outputs (bit 0 = 0): {count_even} ({float(100*count_even/sample_size):.2f}%)")
print(f"Odd outputs (bit 0 = 1): {count_odd} ({float(100*count_odd/sample_size):.2f}%)")
print()

# Check if input parity affects output parity
even_to_even = even_to_odd = odd_to_even = odd_to_odd = 0
for a in range(sample_size):
    h = myhash_sage(a)
    if a % 2 == 0:
        if h % 2 == 0:
            even_to_even += 1
        else:
            even_to_odd += 1
    else:
        if h % 2 == 0:
            odd_to_even += 1
        else:
            odd_to_odd += 1

print("Input parity vs output parity:")
print(f"  Even -> Even: {even_to_even}")
print(f"  Even -> Odd:  {even_to_odd}")
print(f"  Odd -> Even:  {odd_to_even}")
print(f"  Odd -> Odd:   {odd_to_odd}")
print()

# =============================================================================
# INVESTIGATION 6: Differential Analysis
# =============================================================================
print("=" * 70)
print("INVESTIGATION 6: Differential Analysis")
print("=" * 70)
print()

# Check how input differences propagate
# For small input differences, what are output differences?
print("Input difference -> Output difference distribution:")
for delta in [1, 2, 4, 8, 16, 256, 65536]:
    output_diffs = {}
    for a in range(10000):
        h_a = myhash_sage(a)
        h_a_delta = myhash_sage((a + delta) % MOD)
        out_diff = (h_a_delta - h_a) % MOD
        # Also track XOR difference
        xor_diff = h_a ^^ h_a_delta
        output_diffs[xor_diff] = output_diffs.get(xor_diff, 0) + 1

    # Find most common XOR differences
    sorted_diffs = sorted(output_diffs.items(), key=lambda x: -x[1])
    print(f"  Input delta = {delta}:")
    print(f"    Unique XOR output diffs: {len(output_diffs)}")
    print(f"    Most common: {sorted_diffs[:3]}")
print()

# =============================================================================
# INVESTIGATION 7: Polynomial Representation Attempt
# =============================================================================
print("=" * 70)
print("INVESTIGATION 7: Polynomial Representation (Stages 0,2,4 only)")
print("=" * 70)
print()

# If we only consider linear stages 0, 2, 4 (ignoring XOR stages):
# f(a) = m4 * (m2 * (m0 * a + c0) + c2) + c4
#      = m4*m2*m0 * a + m4*m2*c0 + m4*c2 + c4
m_total = R(m0 * m2 * m4)
c_total = R(m4 * m2 * c0 + m4 * c2 + c4)

print("Composed linear stages (ignoring XOR):")
print(f"  f(a) = {int(m_total)} * a + {int(c_total)}")
print(f"       = {int(m_total)} * a + {hex(int(c_total))}")
print()

# Verify
def linear_only_hash(a):
    a = R(a)
    a = m0 * a + c0  # Stage 0
    a = m2 * a + c2  # Stage 2
    a = m4 * a + c4  # Stage 4
    return int(a)

def linear_composed(a):
    return int(m_total * R(a) + c_total)

print("Verification (should be equal):")
for test in [0, 1, 100, 12345, 999999]:
    l1 = linear_only_hash(test)
    l2 = linear_composed(test)
    print(f"  a={test}: staged={l1}, composed={l2}, match={l1==l2}")
print()

# =============================================================================
# SUMMARY
# =============================================================================
print("=" * 70)
print("SUMMARY OF FINDINGS")
print("=" * 70)
print()
print("1. LINEAR STAGES (0, 2, 4):")
print(f"   - Can be composed into single affine: a -> {int(m_total)}*a + {hex(int(c_total))}")
print(f"   - All multipliers are odd (invertible mod 2^32)")
print()
print("2. XOR STAGES (1, 5):")
print(f"   - Both are invertible over GF(2)^32")
print(f"   - Cannot be combined with linear stages due to non-commutativity")
print()
print("3. BRANCH PREDICTION:")
print(f"   - Output LSB is essentially uniform (~50/50)")
print(f"   - No simple input->output parity relationship")
print()
print("4. FIXED POINTS:")
print(f"   - {'Some found' if fixed_points else 'None found in sample'}")
print()
print("5. DIFFERENTIAL:")
print("   - Hash shows good diffusion (many unique output differences)")
print("   - No exploitable differential structure found")
print()
