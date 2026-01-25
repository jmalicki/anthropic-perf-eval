# VLIW SIMD Architecture - ISA Reference

## Architecture Overview

This is a **VLIW (Very Large Instruction Word) SIMD** architecture:

- **VLIW**: Multiple independent operations execute in parallel each cycle
- **SIMD**: Vector operations process 8 elements (VLEN=8) simultaneously
- **Cycle timing**: All operations take **1 cycle** to complete
- **Write-after-read**: Results are visible at the end of the cycle

## Slot Limits Per Cycle

Each cycle, you can execute up to these many operations per engine:

| Engine | Slots/Cycle | Description |
|--------|-------------|-------------|
| `alu` | 12 | Scalar ALU operations |
| `valu` | 6 | Vector ALU operations (VLEN=8 elements each) |
| `load` | 2 | Memory loads (scalar or vector) |
| `store` | 2 | Memory stores (scalar or vector) |
| `flow` | 1 | Control flow operations |
| `debug` | 64 | Debug operations (not counted in cycles) |

**Maximum theoretical throughput per cycle:**
- 12 scalar ALU ops + 6 vector ALU ops (6×8=48 scalar equivalents) + 2 loads + 2 stores + 1 flow

## Memory Model

- **Memory**: 32-bit words, accessed by address
- **Scratch Space**: 1536 words, serves as registers/cache
- **All addresses in instructions refer to scratch space** (except load/store which use scratch to hold memory addresses)

## Instruction Timing

**All operations complete in 1 cycle.** There is no latency or pipeline stalls. However:
- Results written in cycle N are readable in cycle N+1
- Within the same cycle, all reads happen before all writes

## ALU Operations (`alu` engine)

Scalar operations on 32-bit integers. All take 1 cycle.

| Operation | Syntax | Description |
|-----------|--------|-------------|
| `+` | `("+", dest, a, b)` | dest = a + b |
| `-` | `("-", dest, a, b)` | dest = a - b |
| `*` | `("*", dest, a, b)` | dest = a * b |
| `//` | `("//", dest, a, b)` | dest = a // b (integer div) |
| `^` | `("^", dest, a, b)` | dest = a XOR b |
| `&` | `("&", dest, a, b)` | dest = a AND b |
| `\|` | `("\|", dest, a, b)` | dest = a OR b |
| `<<` | `("<<", dest, a, b)` | dest = a << b |
| `>>` | `(">>", dest, a, b)` | dest = a >> b |
| `%` | `("%", dest, a, b)` | dest = a % b |
| `<` | `("<", dest, a, b)` | dest = 1 if a < b else 0 |
| `==` | `("==", dest, a, b)` | dest = 1 if a == b else 0 |
| `cdiv` | `("cdiv", dest, a, b)` | dest = ceil(a/b) |

## Vector ALU Operations (`valu` engine)

Process VLEN=8 elements in parallel. All take 1 cycle.

| Operation | Syntax | Description |
|-----------|--------|-------------|
| Same as ALU | `(op, dest, a, b)` | Applies op to all 8 elements |
| `vbroadcast` | `("vbroadcast", dest, src)` | Copy scalar to all 8 vector elements |
| `multiply_add` | `("multiply_add", dest, a, b, c)` | dest = a*b + c (FMA) |

**Vector addressing**: `dest`, `a`, `b` are base addresses; operation applies to `[base:base+8]`

## Load Operations (`load` engine)

| Operation | Syntax | Description | Cycles |
|-----------|--------|-------------|--------|
| `load` | `("load", dest, addr)` | dest = mem[scratch[addr]] | 1 |
| `vload` | `("vload", dest, addr)` | dest[0:8] = mem[scratch[addr]:scratch[addr]+8] | 1 |
| `const` | `("const", dest, val)` | dest = immediate value | 1 |
| `load_offset` | `("load_offset", dest, addr, off)` | dest+off = mem[scratch[addr+off]] | 1 |

**Note**: `vload` requires contiguous memory addresses. For non-contiguous (gather), use scalar loads.

## Store Operations (`store` engine)

| Operation | Syntax | Description | Cycles |
|-----------|--------|-------------|--------|
| `store` | `("store", addr, src)` | mem[scratch[addr]] = scratch[src] | 1 |
| `vstore` | `("vstore", addr, src)` | mem[scratch[addr]:+8] = scratch[src:src+8] | 1 |

## Flow Operations (`flow` engine)

| Operation | Syntax | Description |
|-----------|--------|-------------|
| `select` | `("select", dest, cond, a, b)` | dest = a if cond else b |
| `vselect` | `("vselect", dest, cond, a, b)` | Per-element select on vectors |
| `add_imm` | `("add_imm", dest, a, imm)` | dest = a + immediate |
| `jump` | `("jump", addr)` | PC = addr (absolute) |
| `cond_jump` | `("cond_jump", cond, addr)` | if cond: PC = addr |
| `cond_jump_rel` | `("cond_jump_rel", cond, offset)` | if cond: PC += offset |
| `halt` | `("halt",)` | Stop execution |
| `pause` | `("pause",)` | Pause for sync point |

## Instruction Bundle Format

Each cycle executes one instruction bundle (dict of engine → list of slots):

```python
{
    "alu": [("+", 10, 11, 12), ("-", 13, 14, 15)],  # 2 scalar ops
    "valu": [("^", 100, 108, 116)],                  # 1 vector op (8 elements)
    "load": [("vload", 200, 50)],                    # 1 vector load
    "store": [("vstore", 51, 208)],                  # 1 vector store
}
```

## Performance Guidelines

### Bottlenecks

1. **Gather operations** (loading from non-contiguous addresses):
   - Need 1 scalar load per element
   - 8 elements → 4 cycles minimum (at 2 loads/cycle)

2. **Control flow**:
   - Only 1 flow op per cycle
   - Loops incur overhead

3. **Memory bandwidth**:
   - 2 loads + 2 stores per cycle max
   - Plan data movement carefully

### Optimization Strategies

1. **Maximize VLIW packing**: Fill all available slots each cycle
2. **Use vectors**: 1 valu = 8 scalar alus worth of work
3. **Hide latency**: Start loads early, compute on other data while waiting
4. **Minimize gathers**: Restructure data or precompute addresses
5. **Unroll loops**: Reduce branch overhead, enable more packing
6. **VALU→ALU shifting**: Move vector ops to scalar when ALU has earlier free slots
7. **Constant derivation**: Generate small constants via ALU instead of const loads

### Scratch Space Initialization

**UNSPECIFIED**: The initial value of scratch space registers is not guaranteed.
Do not assume registers are zero-initialized.

However, the **XOR self-identity** (`x ^ x = 0`) can be used to generate zero
without any dependencies:

```python
# Generate zero in cycle 0 - NO dependency on r's actual value!
{"alu": [("^", dest, r, r)]}  # dest = r ^ r = 0 (always)
```

This works because `x ^ x = 0` regardless of x's value - the scheduler
treats this as having **zero read dependencies**.

### Constant Derivation Tree

Instead of using `const` loads (which consume load slots), small constants
can be derived via ALU operations:

```
Bootstrap: const s1 = 1          # Only load we need
Derive:    s0 = s1 ^ s1          # 0 (no deps!)
           s2 = s1 + s1          # 2
           s4 = s2 + s2          # 4
           s8 = s4 + s4          # 8
           s3 = s2 + s1          # 3
           ...
```

**Benefits**: Frees load slots for actual memory operations, can start
deriving constants immediately in cycle 0.

### Current Performance

| Metric | Value |
|--------|-------|
| Baseline | 147,734 cycles |
| Current best | 2,390 cycles |
| Current speedup | 61.8x |
| Target | 1,487 cycles (100x) |

### Theoretical Analysis

Per group (8 elements) per round:
- 8 scalar loads (gather tree nodes): 4 cycles minimum
- 1 XOR + 6 hash stages (18 VALU ops): ~3 cycles
- Branch computation: ~1 cycle

With 32 groups × 16 rounds = 512 iterations:
- **Compute-bound minimum**: ~1,024 cycles (VALU throughput)
- **Memory-bound minimum**: ~256 cycles (load/store throughput)
- **Realistic minimum**: ~1,500-2,000 cycles (with dependencies)

## Example: Hash Stage with multiply_add

```python
# Hash stage using fused multiply_add for branch computation
# idx = idx * 2 + branch_offset

# Cycle 1: Compute branch bit
{"valu": [("&", t, val, v_1)]}    # t = val & 1

# Cycle 2: Add 1 for branch offset
{"valu": [("+", t, t, v_1)]}      # t = t + 1

# Cycle 3: Fused multiply-add (saves 1 VALU op vs separate ops)
{"valu": [("multiply_add", idx, idx, v_2, t)]}  # idx = idx * 2 + t
```
