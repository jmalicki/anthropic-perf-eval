"""Emission plans: which tree-access encoding each group of 8 lanes uses in each round.

A plan is what `--emission-plan PATH` (generate_ssa_kernel.py) and the `emission_plan`
argument of `scheduler_rs.build_tick_tock_kernel_ssa` take. The structure model
(`scripts/emission_structure.py`, docs/EMISSION_STRUCTURE_MODEL.md) writes them.

File format (JSON)::

    {
      "format": "emission-plan/1",
      "rounds": 16, "groups": 32, "forest_height": 10,
      "access": [
        "bits",                          # round 0: every group
        "bits-flow",                     # round 1
        ...
        {"gather": 25, "bits-flow": 7},  # round 15: groups 0-24 gather, 25-31 bits-flow
        ["gather", "bits", ...]          # or one name per group
      ]
    }

Other keys (e.g. "comment", "predicted") are kept for the reader and ignored here. Encodings:

    gather     8 scalar loads per group (any depth from 1)
    bits       select among cached nodes with the branch bits, leaf selects multiply_add
    bits-flow  the same, leaf selects a choice of flow vselect or multiply_add
    mobius     the same table as 2^d - 1 multiply_adds, no flow
    (bits, bits-flow, mobius: depth 0 to 4)
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence, Union

FORMAT = "emission-plan/1"
ENCODINGS = ("gather", "bits", "bits-flow", "mobius")
BIT_DRIVEN = ("bits", "bits-flow", "mobius")
MAX_CACHED_DEPTH = 4

RoundSpec = Union[str, Mapping[str, int], Sequence[str]]


def round_depth(rnd: int, forest_height: int) -> int:
    """Tree depth visited in round `rnd` (rounds wrap to the root after the deepest level)."""
    return rnd % (forest_height + 1)


def allowed_encodings(depth: int) -> tuple[str, ...]:
    """Encodings that can fetch a node at `depth` (as `AccessEncoding::check_plannable`)."""
    if depth == 0:
        return BIT_DRIVEN
    if depth > MAX_CACHED_DEPTH:
        return ("gather",)
    return ENCODINGS


def expand_round(spec: RoundSpec, n_groups: int, rnd: int) -> list[str]:
    """One round of a plan as one encoding name per group.

    Args:
        spec: A name (every group), a mapping name -> group count (contiguous groups in the
            mapping's order; counts must sum to `n_groups`), or a list of names.
        n_groups: Groups in the kernel (batch_size / 8).
        rnd: Round number, for error messages.

    Returns:
        `n_groups` names.
    """
    if isinstance(spec, str):
        return [spec] * n_groups
    if isinstance(spec, Mapping):
        row = []
        for name, count in spec.items():
            if not isinstance(count, int) or count < 0:
                raise ValueError(f"round {rnd}: group count for {name!r} must be a non-negative integer, got {count!r}")
            row += [name] * count
        if len(row) != n_groups:
            raise ValueError(f"round {rnd}: group counts sum to {len(row)}, the kernel has {n_groups} groups")
        return row
    row = list(spec)
    if len(row) != n_groups:
        raise ValueError(f"round {rnd}: {len(row)} entries, the kernel has {n_groups} groups")
    return row


def expand_plan(plan: Any, rounds: int, n_groups: int, forest_height: int) -> list[list[str]]:
    """Validate a plan and return it as `rounds` lists of `n_groups` encoding names.

    Args:
        plan: A plan dict (file format above), its "access" list, or an already expanded list.
        rounds, n_groups, forest_height: Shape of the kernel being built.

    Returns:
        The expanded plan, ready for the `emission_plan` argument of the binding.

    Raises:
        ValueError: on a shape mismatch, an unknown name or an encoding not available at its
            round's depth.
    """
    if isinstance(plan, Mapping):
        if plan.get("format", FORMAT) != FORMAT:
            raise ValueError(f"unknown plan format {plan.get('format')!r}; expected {FORMAT!r}")
        for key, actual in (("rounds", rounds), ("groups", n_groups), ("forest_height", forest_height)):
            if key in plan and plan[key] != actual:
                raise ValueError(f"plan is for {key}={plan[key]}, the kernel has {key}={actual}")
        access = plan.get("access")
        if access is None:
            raise ValueError("plan has no 'access' list")
    else:
        access = plan
    if len(access) != rounds:
        raise ValueError(f"plan has {len(access)} rounds, the kernel has {rounds}")
    rows = [expand_round(spec, n_groups, rnd) for rnd, spec in enumerate(access)]
    for rnd, row in enumerate(rows):
        depth = round_depth(rnd, forest_height)
        allowed = allowed_encodings(depth)
        for g, name in enumerate(row):
            if name not in ENCODINGS:
                raise ValueError(f"round {rnd} group {g}: unknown encoding {name!r}; choose from {ENCODINGS}")
            if name not in allowed:
                raise ValueError(f"round {rnd} (depth {depth}) group {g}: {name!r} not available; "
                                 f"choose from {allowed}")
    return rows


def compact_round(row: Sequence[str]) -> RoundSpec:
    """Shortest file form of one expanded round: a name, a name -> count mapping when the
    groups of each encoding are contiguous, else the list."""
    if len(set(row)) == 1:
        return row[0]
    runs: dict[str, int] = {}
    last = None
    for name in row:
        if name != last and name in runs:
            return list(row)
        runs[name] = runs.get(name, 0) + 1
        last = name
    return runs


def plan_document(rows: Sequence[Sequence[str]], forest_height: int, **extra: Any) -> dict:
    """The file form of an expanded plan (compact rounds), with `extra` keys added."""
    return {"format": FORMAT, "rounds": len(rows), "groups": len(rows[0]) if rows else 0,
            "forest_height": forest_height, "access": [compact_round(r) for r in rows], **extra}


def load_plan(path: str) -> dict:
    """Read a plan file."""
    with open(path) as f:
        return json.load(f)


def describe_plan(rows: Sequence[Sequence[str]]) -> str:
    """One line per kind of round, e.g. 'r0 bits; r1-3 bits-flow; r15 gather x25 + bits-flow x7'."""
    parts = []
    for rnd, row in enumerate(rows):
        spec = compact_round(row)
        text = spec if isinstance(spec, str) else (
            " + ".join(f"{k} x{v}" for k, v in spec.items()) if isinstance(spec, dict) else "per-group list")
        if parts and parts[-1][1] == text:
            parts[-1] = (parts[-1][0][:1] + [rnd], text)
        else:
            parts.append(([rnd], text))
    return "; ".join((f"r{r[0]}" if len(r) == 1 else f"r{r[0]}-{r[-1]}") + f" {t}" for r, t in parts)
