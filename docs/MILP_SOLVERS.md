# MILP Solver Research for VLIW Instruction Scheduling

## Problem Characteristics

Our VLIW instruction scheduling problem has:
- **~10,000+ binary variables** (x[i,t] for each operation-time pair)
- **~50,000+ constraints** (dependencies, resource limits)
- **Sparse constraint matrix** (most entries are 0)
- **Need for fast solutions** (should solve in <60s)

## Open Source MILP Solvers Comparison

### 1. CVXOPT (Current Choice)
**Website:** [cvxopt.org](https://cvxopt.org)

| Pros | Cons |
|------|------|
| Pure Python, easy to install | GLPK backend not the fastest |
| Good for convex optimization | Limited MILP features |
| Clean matrix-based API | No callbacks, cuts |
| Well-documented | Slower on large ILPs |

**Best for:** Small-medium LPs, convex QPs, quick prototyping

### 2. SCIP (Recommended)
**Website:** [scipopt.org](https://scipopt.org/)

| Pros | Cons |
|------|------|
| Fastest open-source MILP solver | Larger binary size |
| Full branch-cut-and-price | More complex API |
| Callbacks, custom branching | Steeper learning curve |
| Active development (v10.0 in 2025) | |

**Best for:** Large-scale MILP, production use, research

```python
# Installation
uv add pyscipopt
```

### 3. HiGHS
**Website:** [highs.dev](https://highs.dev/)

| Pros | Cons |
|------|------|
| Very fast LP solver | Newer, less mature MILP |
| Modern C++ codebase | Fewer advanced features |
| Good Python bindings | |
| Apache 2.0 license | |

**Best for:** LPs, medium MILPs, when license matters

```python
# Installation
uv add highspy
```

### 4. OR-Tools (Google)
**Website:** [developers.google.com/optimization](https://developers.google.com/optimization)

| Pros | Cons |
|------|------|
| Excellent constraint programming | LP/MILP solver just "OK" |
| SCIP integration built-in | Large dependency |
| Great for scheduling problems | Google-style API |
| Active community | |

**Best for:** Constraint satisfaction, routing, scheduling with complex constraints

```python
# Installation
uv add ortools
```

### 5. PuLP (Modeling Layer)
**Website:** [coin-or.github.io/pulp](https://coin-or.github.io/pulp/)

| Pros | Cons |
|------|------|
| Solver-agnostic interface | Wrapper overhead |
| Easy syntax | No advanced features |
| Works with CBC, SCIP, HiGHS | Slow on large problems |
| Great documentation | |

**Best for:** Prototyping, solver comparison, teaching

```python
# Installation
uv add pulp
```

### 6. Python-MIP
**Website:** [python-mip.com](https://www.python-mip.com/)

| Pros | Cons |
|------|------|
| Deep CBC/Gurobi integration | Less solver variety |
| Cut generation, callbacks | |
| MIPstarts, solution pools | |
| PuLP-inspired syntax | |

**Best for:** When you need advanced MIP features with open-source

## Performance Benchmarks

Based on public benchmarks and literature:

| Solver | Relative Speed (LP) | Relative Speed (MILP) | Notes |
|--------|--------------------|-----------------------|-------|
| SCIP | 1.0x (reference) | 1.0x (reference) | Fastest OSS for MILP |
| HiGHS | 1.2x faster | 1.5x slower | Great LP, decent MILP |
| CBC | 2x slower | 2x slower | Solid, well-tested |
| GLPK | 3x slower | 5x slower | Simple, stable |

## Recommendation for Our Problem

### Primary: SCIP via PySCIPOpt

SCIP is the best choice for our VLIW scheduling ILP because:
1. **Fastest** open-source MILP solver
2. **Scheduling-specific** presolve and heuristics
3. **Callbacks** allow adding cuts during solving
4. **Symmetry breaking** reduces search space

### Secondary: OR-Tools with CP-SAT

For scheduling, OR-Tools' CP-SAT solver may actually outperform MILP:
1. **Native scheduling constraints** (NoOverlap, Cumulative)
2. **Better for discrete decisions**
3. **LNS-based search** finds good solutions fast

### Fallback: HiGHS

If licensing or installation is an issue:
1. **Apache 2.0** license (very permissive)
2. **Fast enough** for medium problems
3. **Simple installation** via pip/uv

## Implementation Plan

```python
# Abstract interface for solver switching
class MILPSolver(ABC):
    @abstractmethod
    def add_variable(self, name, lb, ub, is_integer): pass

    @abstractmethod
    def add_constraint(self, coeffs, sense, rhs): pass

    @abstractmethod
    def minimize(self, objective): pass

    @abstractmethod
    def solve(self, time_limit): pass

# Implementations
class SCIPSolver(MILPSolver): ...
class HiGHSSolver(MILPSolver): ...
class ORToolsSolver(MILPSolver): ...
class CVXOPTSolver(MILPSolver): ...
```

## Sources

- [Best Open-Source Mixed Integer Optimization Solver | Baeldung](https://www.baeldung.com/cs/best-open-source-mixed-integer-optimization-solver)
- [SCIP Optimization Suite](https://scipopt.org/)
- [PuLP Documentation](https://coin-or.github.io/pulp/)
- [OR-Tools Documentation](https://developers.google.com/optimization)
- [Python-MIP GitHub](https://github.com/coin-or/python-mip)
- [ResearchGate Discussion on Python Optimization](https://www.researchgate.net/post/What_is_your_recommendation_for_using_Python_in_Optimization_Modelling_PulP_Gurobi_or_Pyomo)
