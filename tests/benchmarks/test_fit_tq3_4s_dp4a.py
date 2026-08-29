from __future__ import annotations


def test_fitter_reproduces_checked_in_tq3_4s_dp4a_constants():
    from benchmarks.fit_tq3_4s_dp4a import fit_codebook

    result = fit_codebook()

    assert result["levels"] == [-113, -73, -42, -14, 13, 41, 72, 112]
    assert abs(result["scale"] - 0.017704291602768495) < 1e-15
    assert abs(result["weighted_rmse"] - 0.0023053582035632695) < 1e-15
    assert result["levels_lo_hex"] == "0xF2D6B78F"
    assert result["levels_hi_hex"] == "0x7048290D"
    assert abs(sum(result["gaussian_bin_weights"]) - 1.0) < 1e-15
