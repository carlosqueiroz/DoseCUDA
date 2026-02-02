"""
Unit tests for IMRTPlan.readPlanDicom() parser.

These tests use synthetic DICOM RTPLAN datasets created with pydicom
to validate robust parsing of clinical RTPLAN files.
"""
import os
import sys
import tempfile
import pytest
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def create_synthetic_rtplan(
    beams_config,
    n_fractions=30,
):
    """
    Create a synthetic RTPLAN DICOM dataset for testing.

    Parameters
    ----------
    beams_config : list of dict
        Each dict should have:
        - beam_number: int
        - beam_meterset: float
        - control_points: list of dict with:
            - cmw: float (CumulativeMetersetWeight)
            - gantry_angle: float (optional)
            - collimator_angle: float (optional)
            - table_angle: float (optional)
            - isocenter: list of 3 floats (optional)
            - devices: list of dict with:
                - type: str (e.g., 'MLCX', 'ASYMX', 'ASYMY')
                - positions: list of floats
    n_fractions : int
        Number of fractions planned

    Returns
    -------
    pydicom.Dataset
        Synthetic RTPLAN dataset
    """
    # Create file meta
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.481.5'  # RT Plan Storage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

    # Create main dataset
    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b'\x00' * 128)
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.Modality = 'RTPLAN'
    ds.RTPlanLabel = 'TestPlan'
    ds.RTPlanName = 'TestPlan'

    # Create FractionGroupSequence
    fg = Dataset()
    fg.FractionGroupNumber = 1
    fg.NumberOfFractionsPlanned = n_fractions
    fg.NumberOfBeams = len(beams_config)

    # Create ReferencedBeamSequence in FractionGroup
    ref_beam_seq = []
    for beam_cfg in beams_config:
        ref_beam = Dataset()
        ref_beam.ReferencedBeamNumber = beam_cfg['beam_number']
        ref_beam.BeamMeterset = beam_cfg['beam_meterset']
        ref_beam_seq.append(ref_beam)
    fg.ReferencedBeamSequence = Sequence(ref_beam_seq)

    ds.FractionGroupSequence = Sequence([fg])

    # Create BeamSequence
    beam_seq = []
    for beam_cfg in beams_config:
        beam = Dataset()
        beam.BeamNumber = beam_cfg['beam_number']
        beam.BeamName = f"Beam{beam_cfg['beam_number']}"
        beam.TreatmentDeliveryType = 'TREATMENT'
        beam.BeamType = 'STATIC'
        beam.RadiationType = 'PHOTON'
        beam.NumberOfControlPoints = len(beam_cfg['control_points'])

        # Set FinalCumulativeMetersetWeight if provided
        if 'final_cmw' in beam_cfg:
            beam.FinalCumulativeMetersetWeight = beam_cfg['final_cmw']
        else:
            # Default to last CP's CMW
            beam.FinalCumulativeMetersetWeight = beam_cfg['control_points'][-1]['cmw']

        # Create ControlPointSequence
        cp_seq = []
        for idx, cp_cfg in enumerate(beam_cfg['control_points']):
            cp = Dataset()
            cp.ControlPointIndex = idx
            cp.CumulativeMetersetWeight = cp_cfg['cmw']

            # Set nominal beam energy on first CP
            if idx == 0:
                cp.NominalBeamEnergy = cp_cfg.get('energy', 6)

            # Isocenter
            cp.IsocenterPosition = cp_cfg.get('isocenter', [0.0, 0.0, 0.0])

            # Angles
            cp.GantryAngle = cp_cfg.get('gantry_angle', 0.0)
            cp.BeamLimitingDeviceAngle = cp_cfg.get('collimator_angle', 0.0)
            cp.PatientSupportAngle = cp_cfg.get('table_angle', 0.0)
            cp.GantryRotationDirection = 'NONE'

            # Beam limiting devices
            if 'devices' in cp_cfg:
                device_seq = []
                for dev in cp_cfg['devices']:
                    device = Dataset()
                    device.RTBeamLimitingDeviceType = dev['type']
                    device.LeafJawPositions = dev['positions']
                    device_seq.append(device)
                cp.BeamLimitingDevicePositionSequence = Sequence(device_seq)

            cp_seq.append(cp)

        beam.ControlPointSequence = Sequence(cp_seq)
        beam_seq.append(beam)

    ds.BeamSequence = Sequence(beam_seq)

    return ds


def save_temp_rtplan(ds):
    """Save dataset to a temporary file and return the path."""
    fd, path = tempfile.mkstemp(suffix='.dcm')
    os.close(fd)
    ds.save_as(path)
    return path


class MockBeamModel:
    """Mock beam model for testing without actual lookup tables."""
    def __init__(self, n_mlc_pairs=60):
        self.n_mlc_pairs = n_mlc_pairs
        # Create realistic MLC offsets (Varian-like)
        self.mlc_offsets = np.linspace(-195.0, 195.0, n_mlc_pairs).astype(np.single)
        self.mlc_widths = np.full(n_mlc_pairs, 5.0, dtype=np.single)
        # 10mm leaves at center, 5mm elsewhere
        center_start = n_mlc_pairs // 2 - 20
        center_end = n_mlc_pairs // 2 + 20
        self.mlc_widths[center_start:center_end] = 5.0


class TestMUPerBeam:
    """Test that MU is correctly extracted per beam from FractionGroupSequence."""

    def test_two_beams_different_meterset(self):
        """
        Two beams with different BeamMeterset values.
        Sum of segment MU per beam must match BeamMeterset.
        """
        from DoseCUDA.plan_imrt import IMRTPlan

        # Beam 1: 100 MU, 3 CPs -> segments at CMW 0.3 and 1.0
        # Beam 2: 200 MU, 3 CPs -> segments at CMW 0.5 and 1.0
        beams_config = [
            {
                'beam_number': 1,
                'beam_meterset': 100.0,
                'control_points': [
                    {'cmw': 0.0, 'devices': [
                        {'type': 'ASYMX', 'positions': [-50.0, 50.0]},
                        {'type': 'ASYMY', 'positions': [-50.0, 50.0]},
                    ]},
                    {'cmw': 0.3, 'devices': [
                        {'type': 'ASYMX', 'positions': [-50.0, 50.0]},
                        {'type': 'ASYMY', 'positions': [-50.0, 50.0]},
                    ]},
                    {'cmw': 1.0, 'devices': [
                        {'type': 'ASYMX', 'positions': [-50.0, 50.0]},
                        {'type': 'ASYMY', 'positions': [-50.0, 50.0]},
                    ]},
                ]
            },
            {
                'beam_number': 2,
                'beam_meterset': 200.0,
                'control_points': [
                    {'cmw': 0.0, 'devices': [
                        {'type': 'ASYMX', 'positions': [-50.0, 50.0]},
                        {'type': 'ASYMY', 'positions': [-50.0, 50.0]},
                    ]},
                    {'cmw': 0.5, 'devices': [
                        {'type': 'ASYMX', 'positions': [-50.0, 50.0]},
                        {'type': 'ASYMY', 'positions': [-50.0, 50.0]},
                    ]},
                    {'cmw': 1.0, 'devices': [
                        {'type': 'ASYMX', 'positions': [-50.0, 50.0]},
                        {'type': 'ASYMY', 'positions': [-50.0, 50.0]},
                    ]},
                ]
            }
        ]

        ds = create_synthetic_rtplan(beams_config)
        path = save_temp_rtplan(ds)

        try:
            # Create plan with mock beam model
            plan = IMRTPlan.__new__(IMRTPlan)
            plan.beam_models = [MockBeamModel()]
            plan.n_beams = 0
            plan.beam_list = []

            plan.readPlanDicom(path)

            # Verify two beams were loaded
            assert plan.n_beams == 2, f"Expected 2 beams, got {plan.n_beams}"

            # Sum MU for beam 1
            beam1_total_mu = sum(cp.mu for cp in plan.beam_list[0].cp_list)
            assert abs(beam1_total_mu - 100.0) < 0.01, \
                f"Beam 1: expected MU sum=100, got {beam1_total_mu}"

            # Sum MU for beam 2
            beam2_total_mu = sum(cp.mu for cp in plan.beam_list[1].cp_list)
            assert abs(beam2_total_mu - 200.0) < 0.01, \
                f"Beam 2: expected MU sum=200, got {beam2_total_mu}"

        finally:
            os.unlink(path)


class TestSegmentMUFromCMW:
    """Test that segment MU is correctly calculated from CMW delta."""

    def test_cmw_segments_calculation(self):
        """
        CPs with CMW [0.0, 0.3, 1.0], BeamMeterset=200, finalCmw=1.0
        Expected: seg0 MU=60, seg1 MU=140
        """
        from DoseCUDA.plan_imrt import IMRTPlan

        beams_config = [
            {
                'beam_number': 1,
                'beam_meterset': 200.0,
                'final_cmw': 1.0,
                'control_points': [
                    {'cmw': 0.0, 'gantry_angle': 0.0, 'devices': [
                        {'type': 'ASYMX', 'positions': [-50.0, 50.0]},
                        {'type': 'ASYMY', 'positions': [-50.0, 50.0]},
                    ]},
                    {'cmw': 0.3, 'gantry_angle': 90.0, 'devices': [
                        {'type': 'ASYMX', 'positions': [-50.0, 50.0]},
                        {'type': 'ASYMY', 'positions': [-50.0, 50.0]},
                    ]},
                    {'cmw': 1.0, 'gantry_angle': 180.0, 'devices': [
                        {'type': 'ASYMX', 'positions': [-50.0, 50.0]},
                        {'type': 'ASYMY', 'positions': [-50.0, 50.0]},
                    ]},
                ]
            }
        ]

        ds = create_synthetic_rtplan(beams_config)
        path = save_temp_rtplan(ds)

        try:
            plan = IMRTPlan.__new__(IMRTPlan)
            plan.beam_models = [MockBeamModel()]
            plan.n_beams = 0
            plan.beam_list = []

            plan.readPlanDicom(path)

            assert plan.n_beams == 1
            beam = plan.beam_list[0]
            assert beam.n_cps == 2, f"Expected 2 segments, got {beam.n_cps}"

            # Segment 0: (0.3 - 0.0) * 200 = 60 MU
            seg0_mu = beam.cp_list[0].mu
            assert abs(seg0_mu - 60.0) < 0.01, f"Seg0: expected MU=60, got {seg0_mu}"

            # Segment 1: (1.0 - 0.3) * 200 = 140 MU
            seg1_mu = beam.cp_list[1].mu
            assert abs(seg1_mu - 140.0) < 0.01, f"Seg1: expected MU=140, got {seg1_mu}"

            # Verify gantry angles (segment uses geometry from CP[i])
            assert beam.cp_list[0].ga == 0.0, "Seg0 should use gantry angle from CP0"
            assert beam.cp_list[1].ga == 90.0, "Seg1 should use gantry angle from CP1"

        finally:
            os.unlink(path)


class TestDeviceSequenceOrderIndependence:
    """Test that jaws/MLC are correctly parsed regardless of sequence order."""

    def test_shuffled_device_order(self):
        """
        BeamLimitingDevicePositionSequence in non-standard order
        (MLCX first, then ASYMY, then ASYMX) should still parse correctly.
        """
        from DoseCUDA.plan_imrt import IMRTPlan

        # Define expected values
        expected_xjaws = np.array([-40.0, 40.0], dtype=np.single)
        expected_yjaws = np.array([-60.0, 60.0], dtype=np.single)
        n_mlc = 60
        # MLC positions: bank A (left, negative) then bank B (right, positive)
        # All leaf pairs open symmetrically around center
        mlc_bank_a = [-30.0] * n_mlc  # Left bank (all at -30)
        mlc_bank_b = [30.0] * n_mlc   # Right bank (all at +30)
        mlc_positions = mlc_bank_a + mlc_bank_b

        beams_config = [
            {
                'beam_number': 1,
                'beam_meterset': 100.0,
                'control_points': [
                    {'cmw': 0.0, 'devices': [
                        # Intentionally shuffled order
                        {'type': 'MLCX', 'positions': mlc_positions},
                        {'type': 'ASYMY', 'positions': [-60.0, 60.0]},
                        {'type': 'ASYMX', 'positions': [-40.0, 40.0]},
                    ]},
                    {'cmw': 1.0, 'devices': [
                        {'type': 'ASYMY', 'positions': [-60.0, 60.0]},
                        {'type': 'MLCX', 'positions': mlc_positions},
                        {'type': 'ASYMX', 'positions': [-40.0, 40.0]},
                    ]},
                ]
            }
        ]

        ds = create_synthetic_rtplan(beams_config)
        path = save_temp_rtplan(ds)

        try:
            plan = IMRTPlan.__new__(IMRTPlan)
            plan.beam_models = [MockBeamModel(n_mlc_pairs=n_mlc)]
            plan.n_beams = 0
            plan.beam_list = []

            plan.readPlanDicom(path)

            assert plan.n_beams == 1
            cp = plan.beam_list[0].cp_list[0]

            # Check jaws
            np.testing.assert_array_almost_equal(
                cp.xjaws, expected_xjaws,
                err_msg="X jaws not parsed correctly with shuffled order"
            )
            np.testing.assert_array_almost_equal(
                cp.yjaws, expected_yjaws,
                err_msg="Y jaws not parsed correctly with shuffled order"
            )

            # Check MLC shape
            assert cp.mlc.shape == (n_mlc, 4), f"MLC shape should be ({n_mlc}, 4), got {cp.mlc.shape}"

        finally:
            os.unlink(path)

    def test_x_vs_asymx_equivalence(self):
        """Test that 'X' and 'ASYMX' device types are handled equivalently."""
        from DoseCUDA.plan_imrt import IMRTPlan

        # Test with 'X' type
        beams_config_x = [
            {
                'beam_number': 1,
                'beam_meterset': 100.0,
                'control_points': [
                    {'cmw': 0.0, 'devices': [
                        {'type': 'X', 'positions': [-50.0, 50.0]},
                        {'type': 'Y', 'positions': [-50.0, 50.0]},
                    ]},
                    {'cmw': 1.0, 'devices': [
                        {'type': 'X', 'positions': [-50.0, 50.0]},
                        {'type': 'Y', 'positions': [-50.0, 50.0]},
                    ]},
                ]
            }
        ]

        ds_x = create_synthetic_rtplan(beams_config_x)
        path_x = save_temp_rtplan(ds_x)

        # Test with 'ASYMX' type
        beams_config_asym = [
            {
                'beam_number': 1,
                'beam_meterset': 100.0,
                'control_points': [
                    {'cmw': 0.0, 'devices': [
                        {'type': 'ASYMX', 'positions': [-50.0, 50.0]},
                        {'type': 'ASYMY', 'positions': [-50.0, 50.0]},
                    ]},
                    {'cmw': 1.0, 'devices': [
                        {'type': 'ASYMX', 'positions': [-50.0, 50.0]},
                        {'type': 'ASYMY', 'positions': [-50.0, 50.0]},
                    ]},
                ]
            }
        ]

        ds_asym = create_synthetic_rtplan(beams_config_asym)
        path_asym = save_temp_rtplan(ds_asym)

        try:
            # Parse both
            plan_x = IMRTPlan.__new__(IMRTPlan)
            plan_x.beam_models = [MockBeamModel()]
            plan_x.n_beams = 0
            plan_x.beam_list = []
            plan_x.readPlanDicom(path_x)

            plan_asym = IMRTPlan.__new__(IMRTPlan)
            plan_asym.beam_models = [MockBeamModel()]
            plan_asym.n_beams = 0
            plan_asym.beam_list = []
            plan_asym.readPlanDicom(path_asym)

            # Compare jaws
            np.testing.assert_array_equal(
                plan_x.beam_list[0].cp_list[0].xjaws,
                plan_asym.beam_list[0].cp_list[0].xjaws,
                err_msg="X and ASYMX should produce identical xjaws"
            )
            np.testing.assert_array_equal(
                plan_x.beam_list[0].cp_list[0].yjaws,
                plan_asym.beam_list[0].cp_list[0].yjaws,
                err_msg="Y and ASYMY should produce identical yjaws"
            )

        finally:
            os.unlink(path_x)
            os.unlink(path_asym)


class TestLastKnownFallback:
    """Test that CPs with omitted fields use last-known values."""

    def test_cp_omits_devices_uses_last_known(self):
        """
        If a CP omits BeamLimitingDevicePositionSequence,
        the last known jaws/MLC should be used.
        """
        from DoseCUDA.plan_imrt import IMRTPlan

        beams_config = [
            {
                'beam_number': 1,
                'beam_meterset': 100.0,
                'control_points': [
                    {'cmw': 0.0, 'gantry_angle': 0.0, 'devices': [
                        {'type': 'ASYMX', 'positions': [-30.0, 30.0]},
                        {'type': 'ASYMY', 'positions': [-40.0, 40.0]},
                    ]},
                    # CP1 omits devices - should use values from CP0
                    {'cmw': 0.5, 'gantry_angle': 90.0},
                    {'cmw': 1.0, 'gantry_angle': 180.0},
                ]
            }
        ]

        ds = create_synthetic_rtplan(beams_config)
        path = save_temp_rtplan(ds)

        try:
            plan = IMRTPlan.__new__(IMRTPlan)
            plan.beam_models = [MockBeamModel()]
            plan.n_beams = 0
            plan.beam_list = []

            plan.readPlanDicom(path)

            assert plan.n_beams == 1
            beam = plan.beam_list[0]
            assert beam.n_cps == 2

            # Both segments should have the same jaws (from CP0)
            expected_xjaws = np.array([-30.0, 30.0], dtype=np.single)
            expected_yjaws = np.array([-40.0, 40.0], dtype=np.single)

            for i, cp in enumerate(beam.cp_list):
                np.testing.assert_array_almost_equal(
                    cp.xjaws, expected_xjaws,
                    err_msg=f"Segment {i}: xjaws should use last-known value"
                )
                np.testing.assert_array_almost_equal(
                    cp.yjaws, expected_yjaws,
                    err_msg=f"Segment {i}: yjaws should use last-known value"
                )

        finally:
            os.unlink(path)


class TestValidationErrors:
    """Test that appropriate errors/warnings are raised for invalid data."""

    def test_missing_fraction_group_raises_error(self):
        """Missing FractionGroupSequence should raise ValueError."""
        from DoseCUDA.plan_imrt import IMRTPlan

        # Create minimal dataset with file_meta but without FractionGroupSequence
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.481.5'
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

        ds = FileDataset(None, {}, file_meta=file_meta, preamble=b'\x00' * 128)
        ds.Modality = 'RTPLAN'
        ds.BeamSequence = Sequence([])
        # No FractionGroupSequence

        path = save_temp_rtplan(ds)

        try:
            plan = IMRTPlan.__new__(IMRTPlan)
            plan.beam_models = [MockBeamModel()]
            plan.n_beams = 0
            plan.beam_list = []

            with pytest.raises(ValueError, match="FractionGroupSequence"):
                plan.readPlanDicom(path)
        finally:
            os.unlink(path)

    def test_decreasing_cmw_raises_error(self):
        """CMW that decreases between CPs should raise ValueError."""
        from DoseCUDA.plan_imrt import IMRTPlan

        beams_config = [
            {
                'beam_number': 1,
                'beam_meterset': 100.0,
                'control_points': [
                    {'cmw': 0.0, 'devices': [
                        {'type': 'ASYMX', 'positions': [-50.0, 50.0]},
                        {'type': 'ASYMY', 'positions': [-50.0, 50.0]},
                    ]},
                    {'cmw': 0.8, 'devices': [
                        {'type': 'ASYMX', 'positions': [-50.0, 50.0]},
                        {'type': 'ASYMY', 'positions': [-50.0, 50.0]},
                    ]},
                    # CMW decreases - invalid
                    {'cmw': 0.5, 'devices': [
                        {'type': 'ASYMX', 'positions': [-50.0, 50.0]},
                        {'type': 'ASYMY', 'positions': [-50.0, 50.0]},
                    ]},
                ]
            }
        ]

        ds = create_synthetic_rtplan(beams_config)
        path = save_temp_rtplan(ds)

        try:
            plan = IMRTPlan.__new__(IMRTPlan)
            plan.beam_models = [MockBeamModel()]
            plan.n_beams = 0
            plan.beam_list = []

            with pytest.raises(ValueError, match="CMW decresce"):
                plan.readPlanDicom(path)
        finally:
            os.unlink(path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
