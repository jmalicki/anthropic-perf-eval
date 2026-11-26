"""The per-section scratch check (which sections the per-section CP-SAT repairs before the
full-kernel solve) counts what address assignment counts.

On the two best 16-round greedy kernels (docs/CRITICAL_PATH.md: path bits 1,026 cycles, index
bits 1,038), with split shifts whose lanes run in two cycles, the check's peak on the greedy
schedule equals the allocator's peak of the allocatable words (`scratch_pools.words`). Before,
the check counted on the cycle timeline without lane cycles: 1,597 words on the 1,030-cycle
path kernel whose allocator peak is 1,496, so the CP-SAT pipeline repaired a schedule that fits
(BUGS.md). Greedy builds take about a second each.
"""
import argparse
import json
import os
import shlex
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernel_build_args import add_kernel_build_args, build_kernel_with_args  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMMON = (
    "--use-greedy --rounds 16 --greedy-barriers none --greedy-priority wavefront --greedy-split-shift "
    "--pressure-inflation-threshold 0.95 --greedy-load-priority final-last --greedy-groups-in-flight 32 "
    "--greedy-stagger-cycles 9 --merge-hash-stages --index-form mirrored"
)
KERNELS = {
    "path-1026": COMMON + " --emission-plan docs/emission_plans/path_flow90.json --select-bit-source path"
    " --greedy-alu-priority native-first --greedy-flow-priority chain-first --reassociate-gather-xor"
    " --greedy-const-policy alu-first",
    "index-1038": COMMON + " --emission-plan docs/emission_plans/flow90.json --select-bit-source index"
    " --index-early-terms final",
}


@pytest.mark.parametrize("name", sorted(KERNELS))
def test_section_check_equals_allocator_peak(name, tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    parser = argparse.ArgumentParser()
    add_kernel_build_args(parser)
    args = parser.parse_args(shlex.split(KERNELS[name]))
    out = str(tmp_path / "sched.json")
    _, stats = build_kernel_with_args(args, verbose=False, output_json=out, use_git_sha=False)
    with open(out) as f:
        sched = json.load(f)
    assert sched["shifted_alu_cycles"], "no shifted lanes to test"
    split = [lanes for lanes in sched["shifted_alu_cycles"].values() if len(set(lanes)) > 1]
    assert split, "no split shifts: the lanes-in-two-cycles case is not exercised"
    words = sched["scratch_pools"]["words"]
    assert stats["section_check_peak"] == sched["section_check_peak"] == words
    assert stats["allocator_peak_words"] == words
