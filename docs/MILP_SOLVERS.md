# Scheduling Solvers for VLIW Instruction Scheduling

## Current Implementation: CP-SAT

The codebase uses **Google OR-Tools CP-SAT** solver for instruction scheduling, not traditional MILP/ILP solvers. CP-SAT (Constraint Programming with SAT solving) is significantly faster for scheduling problems.

### Why CP-SAT over MILP?

Traditional MILP/ILP formulations for scheduling use binary variables `x[i,t]` for each operation-time pair, leading to:
- ~10,000+ binary variables
- ~50,000+ constraints
- Slow solve times (minutes to hours)

CP-SAT uses a fundamentally different approach:
- **Interval variables**: Each operation is an interval, not binary indicators
- **Cumulative constraints**: Native support for resource limits
- **Constraint propagation**: Sophisticated domain reduction
- **Parallel solving**: Multiple search strategies in parallel

### Performance Comparison

| Approach | Variables | Constraints | Solve Time |
|----------|-----------|-------------|------------|
| MILP (binary x[i,t]) | ~10,000 | ~50,000 | Minutes-hours |
| CP-SAT (intervals) | ~2,000 | ~3,000 | Seconds-minutes |

## Solver Architecture

### Core Scheduler: `cp_scheduler.py`

```python
def schedule_cp(ops: list[Op], op_by_id: dict[int, Op],
                time_limit: float = 60.0,
                greedy_hint: dict[int, int] = None,
                alu_demands: dict[int, int] = None) -> CPScheduleResult
```

**Key features:**
- Creates interval variables for each operation (all take 1 cycle)
- Adds cumulative constraints for VLIW slot limits
- Supports warm-start hints from greedy scheduling
- Multi-slot ALU demand tracking for shifted VALU ops
- Parallel search with interleaved strategies

### Multi-Phase Pipeline: `round_chunked_scheduler.py`

The scheduling uses a three-phase approach:

1. **Phase 1: Chunked Solve**
   - Divides ~10,000 ops into ~1,500-op chunks
   - Solves each chunk with CP-SAT
   - Stitches results respecting cross-chunk dependencies

2. **Phase 2: Full Solve (optional)**
   - Uses chunked solution as warm-start hint
   - Runs global CP-SAT with complete dependencies
   - Often finds 5-15% improvement over chunked

3. **Phase 3: VALU→ALU Shifting**
   - Iteratively moves VALU ops to ALU when ALU has earlier free slots
   - 1 VALU op → 8 ALU ops (uses 8 of 12 slots)
   - Uses CP-SAT to reschedule affected ops after each shift

### VALU→ALU Shifting: `iterative_reschedule.py`

Key insight: VALU and ALU can execute the same operations, but:
- VALU: 1 slot processes 8 elements
- ALU: 8 slots process 8 elements (1 each)

When VALU is bottleneck but ALU has free slots, shifting improves throughput.

**Shiftable operations:**
```python
SHIFTABLE_OPS = {'+', '-', '*', '//', '^', '&', '|', '<', '==', '<<', '>>'}
SPECIAL_SHIFTABLE_OPS = {'multiply_add'}  # Requires 16 ALU slots (2 ops × 8)
```

## Configuration

### Key Parameters

```python
# Kernel generation uses build_tick_tock_kernel_ssa (see generate_ssa_kernel.py)
scheduler_rs.build_tick_tock_kernel_ssa(
    ...,
    cpsat_time_limit=3600.0,    # Time limit for CP-SAT (seconds)
    enable_valu_shift=False,    # Disabled by default to avoid scratch budget issues
)
```

### CP-SAT Solver Settings

**Python (`solver/cp_scheduler.py`):**
```python
solver.parameters.num_search_workers = os.cpu_count()  # Use all cores
solver.parameters.interleave_search = True             # Multiple strategies
solver.parameters.linearization_level = 2              # Stronger LP relaxations
solver.parameters.cp_model_probing_level = 2           # Aggressive probing
solver.parameters.cp_model_presolve = True             # Enable presolve
solver.parameters.randomize_search = True              # Help escape local optima
```

**Rust (`scheduler_rs/src/cpsat_scheduler.rs`):**
```rust
params.num_workers = Some(8);
params.linearization_level = Some(2);              // Stronger LP relaxations
params.cp_model_probing_level = Some(2);           // Aggressive probing
params.cp_model_presolve = Some(true);             // Enable presolve
params.randomize_search = Some(true);              // Help escape local optima
```

Both implementations now use the same LP optimization settings for consistent performance.

## Output Files

The scheduler generates several log files:

| File | Contents |
|------|----------|
| `cpsat_chunks.log` | Per-chunk solve progress |
| `cpsat_fullsolve.log` | Full solve progress |
| `cpsat_reschedule.log` | VALU→ALU shifting progress |

## Alternative Solvers (Historical)

These were evaluated but not used:

### SCIP
- Fastest open-source MILP solver
- Too slow for our problem size with binary formulation
- Could work with cutting planes, but CP-SAT is simpler

### HiGHS
- Excellent LP solver, decent MILP
- Apache 2.0 license
- Not as good as CP-SAT for scheduling

### OR-Tools MILP
- Same OR-Tools library, but MILP backend
- Slower than CP-SAT for this problem class

## Current Results

| Phase | Cycles | Improvement |
|-------|--------|-------------|
| Greedy baseline | ~3,500 | - |
| Chunked CP-SAT | ~2,600 | 26% |
| Full solve | ~2,450 | 6% |
| VALU→ALU shift | ~2,390 | 2% |
| **Final** | **2,390** | **31.7% total** |

## Future Directions

1. **Better chunking**: Use round boundaries more intelligently
2. **Symmetry breaking**: Reduce search space for equivalent schedules
3. **Custom branching**: Domain-specific variable selection
4. **Incremental solving**: Reuse solver state across iterations
