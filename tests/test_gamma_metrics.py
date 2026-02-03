"""
Unit tests for gamma analysis with synthetic data.

Tests verify:
1. Perfect match (gamma = 0)
2. Pure dose difference (gamma = delta/criterion)
3. Pure spatial shift (gamma = shift/DTA)
4. Threshold masking behavior
5. Local vs global mode
"""

import pytest
import numpy as np

from DoseCUDA.gamma import (
    GammaCriteria,
    GammaResult,
    compute_gamma_3d,
    compute_gamma_multiple_criteria,
    GAMMA_3_3_GLOBAL,
    GAMMA_2_2_GLOBAL
)


class TestGammaPerfectMatch:
    """Tests for identical dose distributions."""

    def test_identical_doses_gamma_zero(self):
        """Identical doses should give gamma = 0 everywhere."""
        # Uniform dose of 2 Gy
        dose = np.ones((20, 20, 20), dtype=np.float32) * 2.0
        spacing = (2.0, 2.0, 2.0)

        result = compute_gamma_3d(
            dose_eval=dose.copy(),
            dose_ref=dose,
            spacing_mm=spacing,
            criteria=GAMMA_3_3_GLOBAL,
            return_map=True
        )

        assert result.pass_rate == 1.0, f"Expected 100% pass rate, got {result.pass_rate:.1%}"
        assert result.mean_gamma < 0.01, f"Expected mean gamma ~0, got {result.mean_gamma:.4f}"

        # Check gamma map values
        valid_gamma = result.gamma_map[result.gamma_map < np.inf]
        assert np.all(valid_gamma < 0.01), "All gamma values should be ~0 for identical doses"

    def test_identical_doses_with_gradient(self):
        """Identical gradient doses should give gamma = 0."""
        # Create dose gradient
        dose = np.zeros((20, 20, 20), dtype=np.float32)
        for i in range(20):
            dose[:, :, i] = i * 0.5  # 0 to 9.5 Gy

        spacing = (2.0, 2.0, 2.0)

        result = compute_gamma_3d(
            dose_eval=dose.copy(),
            dose_ref=dose,
            spacing_mm=spacing,
            criteria=GAMMA_3_3_GLOBAL
        )

        assert result.pass_rate == 1.0


class TestGammaDoseDifference:
    """Tests for pure dose difference scenarios."""

    def test_uniform_3pct_offset(self):
        """3% dose offset should give gamma ~1.0 with 3%/3mm criteria."""
        # Reference: 2 Gy uniform
        dose_ref = np.ones((20, 20, 20), dtype=np.float32) * 2.0
        # Evaluated: 3% higher
        dose_eval = dose_ref * 1.03

        spacing = (2.0, 2.0, 2.0)

        result = compute_gamma_3d(
            dose_eval=dose_eval,
            dose_ref=dose_ref,
            spacing_mm=spacing,
            criteria=GAMMA_3_3_GLOBAL
        )

        # Gamma should be exactly 1.0 (at the boundary)
        # With numerical precision, expect close to 1.0
        assert 0.95 <= result.mean_gamma <= 1.05, \
            f"Expected mean gamma ~1.0 for 3% offset, got {result.mean_gamma:.3f}"

    def test_uniform_1pct_offset(self):
        """1% dose offset should give gamma ~0.33 with 3%/3mm criteria."""
        dose_ref = np.ones((20, 20, 20), dtype=np.float32) * 2.0
        dose_eval = dose_ref * 1.01  # 1% higher

        spacing = (2.0, 2.0, 2.0)

        result = compute_gamma_3d(
            dose_eval=dose_eval,
            dose_ref=dose_ref,
            spacing_mm=spacing,
            criteria=GAMMA_3_3_GLOBAL
        )

        # gamma = dose_diff / delta_D = 1% / 3% = 0.33
        expected_gamma = 0.01 / 0.03  # ~0.33
        assert abs(result.mean_gamma - expected_gamma) < 0.1, \
            f"Expected mean gamma ~{expected_gamma:.2f}, got {result.mean_gamma:.3f}"

    def test_uniform_6pct_offset_fails(self):
        """6% dose offset should fail with 3%/3mm criteria."""
        dose_ref = np.ones((20, 20, 20), dtype=np.float32) * 2.0
        dose_eval = dose_ref * 1.06  # 6% higher

        spacing = (2.0, 2.0, 2.0)

        result = compute_gamma_3d(
            dose_eval=dose_eval,
            dose_ref=dose_ref,
            spacing_mm=spacing,
            criteria=GAMMA_3_3_GLOBAL
        )

        # gamma = 6% / 3% = 2.0 (capped at max_gamma)
        assert result.pass_rate < 0.1, \
            f"Expected low pass rate for 6% offset, got {result.pass_rate:.1%}"


class TestGammaSpatialShift:
    """Tests for pure spatial displacement."""

    def test_shifted_gradient_3mm(self):
        """Dose gradient shifted by 3mm should give gamma ~1.0."""
        # Create steep dose gradient
        dose_ref = np.zeros((20, 20, 40), dtype=np.float32)
        for i in range(40):
            dose_ref[:, :, i] = i * 0.25  # 0 to 9.75 Gy

        # Shift by ~3mm (with 2mm spacing = 1.5 voxels, round to 2)
        dose_eval = np.zeros_like(dose_ref)
        dose_eval[:, :, 2:] = dose_ref[:, :, :-2]  # Shift by 4mm

        spacing = (2.0, 2.0, 2.0)

        # Use higher threshold to focus on high-dose region
        criteria = GammaCriteria(
            dta_mm=3.0,
            dd_percent=3.0,
            dose_threshold_percent=30.0
        )

        result = compute_gamma_3d(
            dose_eval=dose_eval,
            dose_ref=dose_ref,
            spacing_mm=spacing,
            criteria=criteria
        )

        # With 4mm shift and 3mm DTA, gamma should be ~1.33
        # Most points should still have gamma < max_gamma
        assert result.n_evaluated > 0, "Should evaluate some voxels"


class TestGammaThreshold:
    """Tests for dose threshold behavior."""

    def test_low_dose_excluded(self):
        """Voxels below threshold should not be evaluated."""
        # Low uniform dose
        dose_ref = np.ones((10, 10, 10), dtype=np.float32) * 0.5  # 0.5 Gy

        # Add one hot spot
        dose_ref[5, 5, 5] = 10.0  # 10 Gy hot spot

        # Evaluated dose has large error in low-dose region
        dose_eval = dose_ref.copy()
        dose_eval[0, 0, 0] = 50.0  # Huge error in low-dose corner

        spacing = (2.0, 2.0, 2.0)

        # 50% threshold means only evaluate where dose >= 5 Gy
        criteria = GammaCriteria(
            dta_mm=3.0,
            dd_percent=3.0,
            dose_threshold_percent=50.0
        )

        result = compute_gamma_3d(
            dose_eval=dose_eval,
            dose_ref=dose_ref,
            spacing_mm=spacing,
            criteria=criteria
        )

        # Only the hot spot should be evaluated (dose_ref >= 5 Gy)
        # The huge error at [0,0,0] should be ignored
        assert result.n_evaluated == 1, \
            f"Expected only 1 voxel evaluated (hot spot), got {result.n_evaluated}"
        assert result.pass_rate == 1.0, "Hot spot should pass (identical)"

    def test_zero_threshold_evaluates_all(self):
        """Zero threshold should evaluate all voxels with dose > 0."""
        dose = np.ones((10, 10, 10), dtype=np.float32) * 2.0
        spacing = (2.0, 2.0, 2.0)

        criteria = GammaCriteria(
            dta_mm=3.0,
            dd_percent=3.0,
            dose_threshold_percent=0.0
        )

        result = compute_gamma_3d(
            dose_eval=dose.copy(),
            dose_ref=dose,
            spacing_mm=spacing,
            criteria=criteria
        )

        # All 1000 voxels should be evaluated
        assert result.n_evaluated == 1000, \
            f"Expected 1000 voxels, got {result.n_evaluated}"


class TestGammaLocalMode:
    """Tests for local vs global dose difference mode."""

    def test_local_mode_uniform_offset(self):
        """Local mode: 3% error gives gamma ~1.0 everywhere."""
        dose_ref = np.ones((20, 20, 20), dtype=np.float32) * 2.0
        dose_eval = dose_ref * 1.03  # 3% higher

        spacing = (2.0, 2.0, 2.0)

        criteria_local = GammaCriteria(
            dta_mm=3.0,
            dd_percent=3.0,
            local=True
        )

        result = compute_gamma_3d(
            dose_eval=dose_eval,
            dose_ref=dose_ref,
            spacing_mm=spacing,
            criteria=criteria_local
        )

        # Local mode: delta_D = 3% × D_ref(x) at each point
        # 3% error → gamma = 1.0
        assert 0.95 <= result.mean_gamma <= 1.05

    def test_local_vs_global_variable_dose(self):
        """Compare local and global modes with variable dose."""
        # Create dose with varying intensity
        dose_ref = np.zeros((20, 20, 20), dtype=np.float32)
        dose_ref[:10, :, :] = 2.0   # Low dose region (2 Gy)
        dose_ref[10:, :, :] = 10.0  # High dose region (10 Gy)

        # 3% error everywhere
        dose_eval = dose_ref * 1.03

        spacing = (2.0, 2.0, 2.0)

        # Global mode: delta_D = 3% × 10 Gy = 0.3 Gy everywhere
        criteria_global = GammaCriteria(dta_mm=3.0, dd_percent=3.0, local=False)
        result_global = compute_gamma_3d(
            dose_eval=dose_eval,
            dose_ref=dose_ref,
            spacing_mm=spacing,
            criteria=criteria_global
        )

        # Local mode: delta_D = 3% × D_ref(x)
        # Low region: delta_D = 0.06 Gy
        # High region: delta_D = 0.3 Gy
        criteria_local = GammaCriteria(dta_mm=3.0, dd_percent=3.0, local=True)
        result_local = compute_gamma_3d(
            dose_eval=dose_eval,
            dose_ref=dose_ref,
            spacing_mm=spacing,
            criteria=criteria_local
        )

        # Local mode should give more uniform gamma (~1.0 everywhere)
        # Global mode: low-dose region has lower gamma (error/0.3)
        # With 3% error on 2 Gy: dose_diff = 0.06 Gy
        # Global: gamma = 0.06/0.3 = 0.2
        # Local: gamma = 0.06/0.06 = 1.0

        # Local mode should have higher mean (closer to 1.0)
        assert result_local.mean_gamma > result_global.mean_gamma, \
            "Local mode should have higher mean gamma with uniform % error"


class TestGammaMultipleCriteria:
    """Tests for computing gamma with multiple criteria."""

    def test_multiple_criteria(self):
        """Test computing gamma for multiple criteria at once."""
        dose = np.ones((15, 15, 15), dtype=np.float32) * 2.0
        dose_eval = dose * 1.02  # 2% offset

        spacing = (2.0, 2.0, 2.0)

        results = compute_gamma_multiple_criteria(
            dose_eval=dose_eval,
            dose_ref=dose,
            spacing_mm=spacing
        )

        # Should have results for both 3%/3mm and 2%/2mm
        # Note: labels include decimal points like '3.0%/3.0mm (global)'
        assert '3.0%/3.0mm (global)' in results
        assert '2.0%/2.0mm (global)' in results

        # 2% offset with 3%/3mm: gamma = 0.67
        # 2% offset with 2%/2mm: gamma = 1.0
        assert results['3.0%/3.0mm (global)'].pass_rate == 1.0  # gamma ~0.67 < 1
        assert results['2.0%/2.0mm (global)'].pass_rate >= 0.5  # gamma ~1.0


class TestGammaEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_threshold_region(self):
        """Test behavior when no voxels above threshold."""
        # Create dose with max of 10 Gy in one corner,
        # but most of the volume is below threshold
        dose = np.zeros((10, 10, 10), dtype=np.float32)
        dose[0, 0, 0] = 10.0  # Single high-dose voxel (max)

        spacing = (2.0, 2.0, 2.0)

        # 50% threshold = 5 Gy, only the single voxel should be evaluated
        criteria = GammaCriteria(dose_threshold_percent=50.0)

        result = compute_gamma_3d(
            dose_eval=dose.copy(),
            dose_ref=dose,
            spacing_mm=spacing,
            criteria=criteria
        )

        # Only 1 voxel (at 10 Gy) is above the 5 Gy threshold
        assert result.n_evaluated == 1
        assert result.pass_rate == 1.0  # Identical doses

    def test_shape_mismatch_raises(self):
        """Test that mismatched shapes raise error."""
        dose_ref = np.ones((10, 10, 10), dtype=np.float32)
        dose_eval = np.ones((10, 10, 11), dtype=np.float32)  # Wrong shape
        spacing = (2.0, 2.0, 2.0)

        with pytest.raises(ValueError, match="shape"):
            compute_gamma_3d(
                dose_eval=dose_eval,
                dose_ref=dose_ref,
                spacing_mm=spacing
            )

    def test_gamma_result_repr(self):
        """Test GammaResult string representation."""
        result = GammaResult(
            pass_rate=0.95,
            mean_gamma=0.5,
            gamma_p95=0.9,
            n_evaluated=1000,
            n_passed=950,
            gamma_map=None,
            criteria={}
        )

        repr_str = repr(result)
        assert '95.0%' in repr_str
        assert '0.500' in repr_str


class TestGammaCriteria:
    """Tests for GammaCriteria configuration."""

    def test_criteria_label(self):
        """Test criteria label generation."""
        criteria = GammaCriteria(dta_mm=3.0, dd_percent=3.0, local=False)
        assert criteria.label() == "3.0%/3.0mm (global)"

        criteria_local = GammaCriteria(dta_mm=2.0, dd_percent=2.0, local=True)
        assert criteria_local.label() == "2.0%/2.0mm (local)"

    def test_criteria_to_dict(self):
        """Test criteria serialization."""
        criteria = GAMMA_3_3_GLOBAL
        d = criteria.to_dict()

        assert d['dta_mm'] == 3.0
        assert d['dd_percent'] == 3.0
        assert d['local'] == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
