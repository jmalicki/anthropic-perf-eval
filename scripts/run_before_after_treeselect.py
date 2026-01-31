#!/usr/bin/env python3
"""Run kernel build BEFORE and AFTER enabling tree-reduce depth3 (treeselect).

With --use-greedy: uses greedy scheduler + enable_valu_shift (fast, returns cycles + peak_live).
With CP-SAT: 4-round + valu_shift currently hits merged scratch budget, so no stats returned;
  use greedy for before/after numbers, or fewer rounds for CP-SAT.

Usage (from project root):
    uv run python scripts/run_before_after_treeselect.py --use-greedy
    uv run python scripts/run_before_after_treeselect.py --rounds 4 --cpsat-time 180
"""

import argparse
import sys
import time

import scheduler_rs

BASELINE = 147734


def run(
    rounds: int,
    enable_valu_shift: bool,
    enable_tree_reduce_depth3: bool,
    use_cpsat: bool,
    cpsat_time_limit: float,
    verbose: bool,
) -> dict:
    """Build kernel and return stats (cycles, peak_live, budget_exceeded, etc.)."""
    start = time.time()
    try:
        instrs, stats = scheduler_rs.build_tick_tock_kernel_ssa(
            forest_height=10,
            n_nodes=2047,
            batch_size=256,
            rounds=rounds,
            enable_valu_shift=enable_valu_shift,
            verbose=verbose,
            use_cpsat=use_cpsat,
            cpsat_time_limit=cpsat_time_limit,
            output_json=None,
            enable_tree_reduce_depth3=enable_tree_reduce_depth3,
        )
    except Exception as e:
        return {
            "error": str(e),
            "cycles": None,
            "peak_live": None,
            "build_time": time.time() - start,
            "budget_exceeded": None,
        }
    return {
        "cycles": stats["cycles"],
        "peak_live": stats.get("peak_live"),
        "budget_exceeded": stats.get("budget_exceeded", False),
        "build_time": time.time() - start,
        "instr_count": len(instrs),
        "vreg_count": stats.get("vreg_count"),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Before/after treeselect (tree_reduce_depth3) with full options"
    )
    ap.add_argument("--rounds", type=int, default=4, help="Rounds (default 4)")
    ap.add_argument(
        "--cpsat-time",
        type=float,
        default=180.0,
        help="CP-SAT time limit per run when use_cpsat (default 180s)",
    )
    ap.add_argument(
        "--use-greedy",
        action="store_true",
        help="Use greedy scheduler (fast; returns cycles + peak_live even when CP-SAT would fail merge)",
    )
    ap.add_argument("-q", "--quiet", action="store_true", help="Minimal output")
    args = ap.parse_args()
    verbose = not args.quiet
    use_cpsat = not args.use_greedy

    print("=" * 60)
    print("BEFORE vs AFTER tree-reduce depth3 (treeselect)")
    print("  Scheduler: " + ("greedy" if use_cpsat is False else f"CP-SAT ({args.cpsat_time:.0f}s)"))
    print("  enable_valu_shift=True")
    print(f"  rounds={args.rounds}")
    print("=" * 60)

    # BEFORE: tree selection OFF
    print("\n--- BEFORE (enable_tree_reduce_depth3=False) ---")
    sys.stdout.flush()
    before = run(
        rounds=args.rounds,
        enable_valu_shift=True,
        enable_tree_reduce_depth3=False,
        use_cpsat=use_cpsat,
        cpsat_time_limit=args.cpsat_time,
        verbose=verbose,
    )

    # AFTER: tree selection ON
    print("\n--- AFTER (enable_tree_reduce_depth3=True) ---")
    sys.stdout.flush()
    after = run(
        rounds=args.rounds,
        enable_valu_shift=True,
        enable_tree_reduce_depth3=True,
        use_cpsat=use_cpsat,
        cpsat_time_limit=args.cpsat_time,
        verbose=verbose,
    )

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    def row(label: str, b: dict, a: dict) -> None:
        bc = b.get("cycles")
        ac = a.get("cycles")
        bp = b.get("peak_live")
        ap_ = a.get("peak_live")
        be = b.get("error")
        ae = a.get("error")
        print(f"  {label}")
        be_short = (be[:70] + "...") if be and len(be) > 70 else (be or "")
        ae_short = (ae[:70] + "...") if ae and len(ae) > 70 else (ae or "")
        print(f"    BEFORE: cycles={bc}, peak_live={bp}" + (f"  (error: {be_short})" if be else ""))
        print(f"    AFTER:  cycles={ac}, peak_live={ap_}" + (f"  (error: {ae_short})" if ae else ""))
        if bc is not None and ac is not None:
            dcy = ac - bc
            dpl = (ap_ - bp) if (ap_ is not None and bp is not None) else None
            print(f"    Delta:  cycles {dcy:+d}, peak_live {dpl:+d}" if dpl is not None else f"    Delta:  cycles {dcy:+d}")
        print()

    row("Tree-reduce depth3 (8-way selection)", before, after)

    print("Build times:")
    print(f"  BEFORE: {before.get('build_time', 0):.1f}s")
    print(f"  AFTER:  {after.get('build_time', 0):.1f}s")

    if before.get("budget_exceeded"):
        print("\n  WARNING: BEFORE run exceeded scratch budget.")
    if after.get("budget_exceeded"):
        print("\n  WARNING: AFTER run exceeded scratch budget.")

    return 0 if not before.get("error") and not after.get("error") else 1


if __name__ == "__main__":
    sys.exit(main())
