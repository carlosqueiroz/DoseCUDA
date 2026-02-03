import numpy as np
import pytest

from DoseCUDA.plan_imrt import IMRTPhotonEnergy, IMRTControlPoint


class DummyCP(IMRTControlPoint):
    def __init__(self, xjaws, yjaws, mlc):
        super().__init__(iso=np.zeros(3, dtype=np.float32), mu=1.0,
                         mlc=mlc, ga=0.0, ca=0.0, ta=0.0,
                         xjaws=xjaws, yjaws=yjaws)


def test_output_factor_closed_field_zero():
    model = IMRTPhotonEnergy("6")
    # Minimal required fields for validation
    model.output_factor_equivalent_squares = np.array([1, 2, 3], dtype=np.float32)
    model.output_factor_values = np.array([1, 1, 1], dtype=np.float32)
    model.mu_calibration = 1.0
    model.primary_source_distance = 100.0
    model.scatter_source_distance = 100.0
    model.mlc_distance = 50.0
    model.scatter_source_weight = 0.1
    model.electron_attenuation = 0.01
    model.primary_source_size = 1.0
    model.scatter_source_size = 1.0
    model.profile_radius = np.array([0, 10], dtype=np.float32)
    model.profile_intensities = np.array([1, 1], dtype=np.float32)
    model.profile_softening = np.array([1, 1], dtype=np.float32)
    model.spectrum_attenuation_coefficients = np.array([0.1], dtype=np.float32)
    model.spectrum_primary_weights = np.array([1.0], dtype=np.float32)
    model.spectrum_scatter_weights = np.array([0.0], dtype=np.float32)
    model.electron_source_weight = 0.0
    model.has_xjaws = True
    model.has_yjaws = True
    model.electron_fitted_dmax = 1.0
    model.jaw_transmission = 0.02
    model.mlc_transmission = 0.02
    model.heterogeneity_alpha = 0.0
    model.kernel = np.zeros(36, dtype=np.float32)
    model.kernel_len = 36
    model.validate_parameters()

    # Fully closed MLC and jaws
    xjaws = np.array([0.0, 0.0], dtype=np.float32)
    yjaws = np.array([-50.0, 50.0], dtype=np.float32)
    # mlc: x1=x2=0, offsets spanning y, widths 10
    offsets = np.linspace(-50, 50, 5, dtype=np.float32)
    widths = np.ones_like(offsets) * 10.0
    x1 = np.zeros_like(offsets)
    x2 = np.zeros_like(offsets)
    mlc = np.vstack((x1, x2, offsets, widths)).T.astype(np.float32)

    cp = DummyCP(xjaws, yjaws, mlc)
    of = model.outputFactor(cp)
    assert of == 0.0


def test_output_factor_small_open_positive():
    model = IMRTPhotonEnergy("6")
    # Minimal required fields
    model.output_factor_equivalent_squares = np.array([0.5, 1.0, 2.0], dtype=np.float32)
    model.output_factor_values = np.array([0.7, 0.9, 1.0], dtype=np.float32)
    model.mu_calibration = 1.0
    model.primary_source_distance = 100.0
    model.scatter_source_distance = 100.0
    model.mlc_distance = 50.0
    model.scatter_source_weight = 0.1
    model.electron_attenuation = 0.01
    model.primary_source_size = 1.0
    model.scatter_source_size = 1.0
    model.profile_radius = np.array([0, 10], dtype=np.float32)
    model.profile_intensities = np.array([1, 1], dtype=np.float32)
    model.profile_softening = np.array([1, 1], dtype=np.float32)
    model.spectrum_attenuation_coefficients = np.array([0.1], dtype=np.float32)
    model.spectrum_primary_weights = np.array([1.0], dtype=np.float32)
    model.spectrum_scatter_weights = np.array([0.0], dtype=np.float32)
    model.electron_source_weight = 0.0
    model.has_xjaws = True
    model.has_yjaws = True
    model.electron_fitted_dmax = 1.0
    model.jaw_transmission = 0.02
    model.mlc_transmission = 0.02
    model.heterogeneity_alpha = 0.0
    model.kernel = np.zeros(36, dtype=np.float32)
    model.kernel_len = 36
    model.validate_parameters()

    # 1x1 cm field (10x10 mm) inside jaws
    xjaws = np.array([-5.0, 5.0], dtype=np.float32)
    yjaws = np.array([-5.0, 5.0], dtype=np.float32)
    offsets = np.array([0.0], dtype=np.float32)
    widths = np.array([10.0], dtype=np.float32)
    x1 = np.array([-5.0], dtype=np.float32)
    x2 = np.array([5.0], dtype=np.float32)
    mlc = np.vstack((x1, x2, offsets, widths)).T.astype(np.float32)

    cp = DummyCP(xjaws, yjaws, mlc)
    of = model.outputFactor(cp)
    assert of > 0.0
    assert of <= 1.0
