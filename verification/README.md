# Liveness Specification Verification

Z3-based formal verification of [docs/LIVENESS_SPECIFICATION.md](../docs/LIVENESS_SPECIFICATION.md).

## Setup

```bash
uv pip install -e ".[verify]"
```

## Run

```bash
uv run python -m verification.verify
```

Options:

- `-t T1 -t T2` — Run only specific theorems
- `--timeout 60000` — Timeout in ms (default: 60000 = 1 minute)
- `-q` — Quiet (less output)
- `--links` — Print spec file path with anchors for each theorem (e.g. `path/to/spec.md#t1`)
- `--seed 42` — Random seed for Z3

## Theorems

Each theorem links to its definition in [LIVENESS_SPECIFICATION.md](../docs/LIVENESS_SPECIFICATION.md):

| ID | Description | Spec |
|----|-------------|------|
| T1 | Expiration rule: when `end(v)<=S`, reuse is safe | [#t1](../docs/LIVENESS_SPECIFICATION.md#t1) |
| T2 | Overlap definition: `overlaps([s1,e1],[s2,e2]) iff e1>s2 and e2>s1` | [#t2](../docs/LIVENESS_SPECIFICATION.md#t2) |
| T3 | ASAP bounds: `cycle(succ) >= cycle(pred) + offset` | [#t3](../docs/LIVENESS_SPECIFICATION.md#t3) |
| T4 | ALAP bounds: `cycle(pred) <= cycle(succ) - offset` | [#t4](../docs/LIVENESS_SPECIFICATION.md#t4) |
| T5 | Cycle domain: `ASAP(op) <= cycle(op) <= ALAP(op)` is sound | [#t5](../docs/LIVENESS_SPECIFICATION.md#t5) |
| T6 | Half-open conversion: `end_halfopen = end_closed + 1` | [#t6](../docs/LIVENESS_SPECIFICATION.md#t6) |
| T7 | Slot-aware tightening: one-step lemma | [#t7](../docs/LIVENESS_SPECIFICATION.md#t7) |
| T8 | Scratch-aware tightening: one-step lemma | [#t8](../docs/LIVENESS_SPECIFICATION.md#t8) |
| T9 | Expiration implies no overlap | [#t9](../docs/LIVENESS_SPECIFICATION.md#t9) |
| T10 | Dead shifted VALU dest: `end(v) = max_cycle(v) + 1` | [#t10](../docs/LIVENESS_SPECIFICATION.md#t10) |
| T11 | Dead single-cycle VReg: `end(v) = D + 1` | [#t11](../docs/LIVENESS_SPECIFICATION.md#t11) |
| T12 | Shifted VALU dest with consumer: `end(v) = R` | [#t12](../docs/LIVENESS_SPECIFICATION.md#t12) |
| T13 | Occupied interval semantics: reusable at `end + 1` | [#t13](../docs/LIVENESS_SPECIFICATION.md#t13) |
