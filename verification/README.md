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
- `--seed 42` — Random seed for Z3

## Theorems

- **T1** — Expiration rule: when `end(v)<=S`, reuse is safe (no write-write conflict)
- **T2** — Overlap definition: `overlaps([s1,e1],[s2,e2]) iff e1>s2 and e2>s1`
- **T3** — ASAP bounds: `cycle(succ) >= cycle(pred) + offset` (from A3)
- **T4** — ALAP bounds: `cycle(pred) <= cycle(succ) - offset` (from A3)
- **T5** — Cycle domain: `ASAP(op) <= cycle(op) <= ALAP(op)` is sound (2-node chain model)
- **T6** — Half-open conversion: `end_halfopen = end_closed + 1`
- **T9** — Expiration implies no overlap
- **T10** — Dead shifted VALU dest: `end(v) = max_cycle(v) + 1`
