"""Z3 proofs of emission rewrites (verification/rewrites.py) and a guard that they are not vacuous."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("z3")

from verification import rewrites  # noqa: E402


@pytest.mark.parametrize("name,desc,check", rewrites.REWRITE_CHECKS, ids=[c[0] for c in rewrites.REWRITE_CHECKS])
def test_rewrite(name, desc, check):
    _, result, _, _ = check()
    assert result == ("sat" if name in rewrites.EXPECT_SAT else "unsat"), desc


def test_fma_multipliers_match_emission():
    """scheduler_rs fuses stages 0, 2, 4 with these multipliers (context_ssa.rs hash_fma_mult)."""
    assert [rewrites.fma_multiplier(i) for i in range(6)] == [4097, None, 33, None, 9, None]


def test_merged_stage_constants_match_emission():
    """scheduler_rs merges stages 2+3 with these constants (config.rs merged_hash_stage_constants)."""
    assert [rewrites.merged_stage_constants(i) for i in range(6)] == [
        None, None, (33, 0xE9F8CC1D, 16896, 0xACCF6200), None, None, None]


def test_stage_merge_proof_rejects_a_wrong_constant(monkeypatch):
    """Off-by-one in any of the four merged constants must make the proof fail."""
    real = rewrites.merged_stage_constants
    for k in range(4):
        monkeypatch.setattr(
            rewrites, "merged_stage_constants",
            lambda first, k=k: tuple(v + (i == k) for i, v in enumerate(real(first))))
        _, result, _, _ = rewrites.check_hash_stage_merge()
        assert result == "sat", f"constant {k}"


def test_fusion_proof_rejects_a_wrong_multiplier(monkeypatch):
    """The proof must fail if the multiplier is off by one, i.e. it really constrains the constants."""
    real = rewrites.fma_multiplier
    monkeypatch.setattr(rewrites, "fma_multiplier", lambda stage: None if real(stage) is None else real(stage) + 1)
    _, result, _, _ = rewrites.check_hash_fma_fusion()
    assert result == "sat"


def test_stage5_proof_rejects_a_wrong_constant(monkeypatch):
    monkeypatch.setattr(rewrites, "STAGE5_XOR_CONST", rewrites.STAGE5_XOR_CONST ^ 4)
    _, result, _, _ = rewrites.check_stage5_xor_deferral()
    assert result == "sat"
