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

### Theoretical Limits for the Tree Traversal Kernel

Per group (8 elements):
- 2 vloads (indices, values): 1 cycle
- 8 scalar loads (gather tree nodes): 4 cycles
- 1 XOR: 0.17 cycles (can pack with others)
- 18 hash ops (6 stages × 3): 3 cycles (at 6 valu/cycle)
- 6 branch ops: 1 cycle
- 2 vstores: 1 cycle

**Minimum: ~10 cycles per group**

With 32 groups × 16 rounds = 512 iterations:
- **Theoretical minimum: ~5,120 cycles** (just computation, perfect packing)
- **Practical minimum: ~1,500-2,000 cycles** (with overhead)
- **Target: 1,487 cycles** (Claude Opus 4.5 best)

## Example: Optimized Hash Stage

```python
# Hash stage: val = (val + const) + (val << shift)
# Can pack: broadcast + 2 valus, then 1 valu

# Cycle 1: Broadcast constants (2 valu ops)
{"valu": [
    ("vbroadcast", v_hc, hash_const),
    ("vbroadcast", v_hs, shift_amt),
]}

# Cycle 2: First two ops (2 valu ops)
{"valu": [
    ("+", v_t1, v_val, v_hc),    # tmp1 = val + const
    ("<<", v_t2, v_val, v_hs),   # tmp2 = val << shift
]}

# Cycle 3: Final combine (1 valu op)
{"valu": [
    ("+", v_val, v_t1, v_t2),    # val = tmp1 + tmp2
]}

# Total: 3 cycles per hash stage × 6 stages = 18 cycles
# With better packing: ~12 cycles for all hash stages
```
