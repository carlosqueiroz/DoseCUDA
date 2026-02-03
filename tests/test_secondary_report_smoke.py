"""
Smoke tests for secondary check report generation.

Verifies:
1. JSON report generation and schema validation
2. CSV report generation and format
3. Report with various result combinations (pass/fail)
4. Error handling for edge cases
"""

import pytest
import json
import csv
import tempfile
from pathlib import Path
from datetime import datetime

import numpy as np

from DoseCUDA.secondary_report import (
    SecondaryCheckResult,
    SecondaryCheckCriteria,
    generate_json_report,
    generate_csv_report,
    load_report,
)
from DoseCUDA.gamma import GammaResult


@pytest.fixture
def sample_passing_result():
    """Create a sample passing result for testing."""
    result = SecondaryCheckResult(
        patient_id="TEST001",
        plan_name="TestPlan_VMAT",
        plan_uid="1.2.840.10008.5.1.4.1.1.481.5.12345",
        timestamp=datetime.now().isoformat(),
        dosecuda_version="1.0.0",
        schema_version="1.0"
    )

    result.gamma_results = {
        "3%/3mm": {
            "criteria_label": "3.0%/3.0mm (global)",
            "pass_rate": 0.98,
            "mean_gamma": 0.45,
            "gamma_p95": 0.89,
            "n_evaluated": 50000,
            "n_passed": 49000,
            "status": "PASS"
        },
        "2%/2mm": {
            "criteria_label": "2.0%/2.0mm (global)",
            "pass_rate": 0.92,
            "mean_gamma": 0.65,
            "gamma_p95": 1.05,
            "n_evaluated": 50000,
            "n_passed": 46000,
            "status": "PASS"
        }
    }

    result.dvh_results = {
        "PTV_70": {
            "roi_type": "TARGET",
            "volume_cc": 125.5,
            "metrics": {
                "D95%": {
                    "calculated": 66.8,
                    "reference": 67.0,
                    "diff_abs": -0.2,
                    "diff_rel": -0.003,
                    "status": "PASS"
                },
                "Dmean": {
                    "calculated": 70.1,
                    "reference": 70.0,
                    "diff_abs": 0.1,
                    "diff_rel": 0.0014,
                    "status": "PASS"
                }
            },
            "overall_status": "PASS"
        },
        "Bladder": {
            "roi_type": "OAR",
            "volume_cc": 250.0,
            "metrics": {
                "Dmax": {
                    "calculated": 68.5,
                    "reference": 68.0,
                    "diff_abs": 0.5,
                    "diff_rel": 0.007,
                    "status": "PASS"
                },
                "Dmean": {
                    "calculated": 35.2,
                    "reference": 35.0,
                    "diff_abs": 0.2,
                    "diff_rel": 0.006,
                    "status": "PASS"
                }
            },
            "overall_status": "PASS"
        }
    }

    result.mu_sanity = {
        "isocenter_mm": [0.0, -5.0, 100.0],
        "dose_calc_at_iso": 1.95,
        "dose_ref_at_iso": 2.0,
        "total_mu": 450.0,
        "gy_per_mu_calc": 0.00433,
        "gy_per_mu_ref": 0.00444,
        "mu_equiv_ratio": 0.975,
        "status": "INFO",
        "message": "MU ratio 0.9750 within 5.0% tolerance"
    }

    result.overall_status = "PASS"
    result.failure_reasons = []

    return result


@pytest.fixture
def sample_failing_result():
    """Create a sample failing result for testing."""
    result = SecondaryCheckResult(
        patient_id="TEST002",
        plan_name="TestPlan_Failing",
        plan_uid="1.2.840.10008.5.1.4.1.1.481.5.67890",
        timestamp=datetime.now().isoformat(),
        dosecuda_version="1.0.0"
    )

    result.gamma_results = {
        "3%/3mm": {
            "criteria_label": "3.0%/3.0mm (global)",
            "pass_rate": 0.89,  # Below 95% threshold
            "mean_gamma": 0.78,
            "gamma_p95": 1.15,
            "n_evaluated": 50000,
            "n_passed": 44500,
            "status": "FAIL"
        }
    }

    result.dvh_results = {
        "PTV_56": {
            "roi_type": "TARGET",
            "volume_cc": 200.0,
            "metrics": {
                "D95%": {
                    "calculated": 52.0,
                    "reference": 54.0,
                    "diff_abs": -2.0,
                    "diff_rel": -0.037,  # 3.7% > 3% tolerance
                    "status": "FAIL"
                }
            },
            "overall_status": "FAIL"
        }
    }

    result.overall_status = "FAIL"
    result.failure_reasons = [
        "Gamma 3%/3mm pass rate 89.0% < 95%",
        "DVH comparison failed for target PTV_56"
    ]

    return result


class TestJSONReportGeneration:
    """Tests for JSON report generation."""

    def test_json_report_created(self, sample_passing_result):
        """Verify JSON report file is created with correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            generate_json_report(sample_passing_result, str(output_path))

            assert output_path.exists(), "JSON report file not created"

            with open(output_path) as f:
                data = json.load(f)

            # Check required fields
            assert data["patient_id"] == "TEST001"
            assert data["plan_name"] == "TestPlan_VMAT"
            assert data["overall_status"] == "PASS"
            assert data["schema_version"] == "1.0"

            # Check gamma results
            assert "3%/3mm" in data["gamma_results"]
            assert data["gamma_results"]["3%/3mm"]["pass_rate"] == 0.98

            # Check DVH results
            assert "PTV_70" in data["dvh_results"]
            assert data["dvh_results"]["PTV_70"]["roi_type"] == "TARGET"

            # Check MU sanity
            assert data["mu_sanity"]["mu_equiv_ratio"] == 0.975

    def test_json_report_failing_case(self, sample_failing_result):
        """Verify JSON report correctly records failures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            generate_json_report(sample_failing_result, str(output_path))

            with open(output_path) as f:
                data = json.load(f)

            assert data["overall_status"] == "FAIL"
            assert len(data["failure_reasons"]) == 2
            assert "Gamma" in data["failure_reasons"][0]

    def test_json_report_load_roundtrip(self, sample_passing_result):
        """Verify JSON report can be loaded back correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            generate_json_report(sample_passing_result, str(output_path))

            loaded = load_report(str(output_path))

            assert loaded.patient_id == sample_passing_result.patient_id
            assert loaded.overall_status == sample_passing_result.overall_status
            assert loaded.gamma_results == sample_passing_result.gamma_results


class TestJSONSchemaValidation:
    """Tests for JSON schema validation."""

    def test_valid_report_passes_validation(self, sample_passing_result):
        """Verify valid report passes schema validation."""
        from DoseCUDA.secondary_report import validate_report_schema

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            generate_json_report(sample_passing_result, str(output_path))

            with open(output_path) as f:
                data = json.load(f)

            # Try validation - skip if jsonschema not available
            try:
                assert validate_report_schema(data) == True
            except ImportError:
                pytest.skip("jsonschema not available")

    def test_invalid_report_fails_validation(self):
        """Verify invalid report fails schema validation."""
        try:
            from DoseCUDA.secondary_report import validate_report_schema
            import jsonschema
        except ImportError:
            pytest.skip("jsonschema not available")

        invalid_data = {
            "patient_id": "TEST",
            # Missing required fields
        }

        with pytest.raises(jsonschema.ValidationError):
            validate_report_schema(invalid_data)


class TestCSVReportGeneration:
    """Tests for CSV report generation."""

    def test_csv_report_created(self, sample_passing_result):
        """Verify CSV report file is created with correct columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.csv"
            generate_csv_report(sample_passing_result, str(output_path))

            assert output_path.exists(), "CSV report file not created"

            with open(output_path, newline='') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                fieldnames = reader.fieldnames

            # Should have summary row + ROI rows
            assert len(rows) == 3, f"Expected 3 rows (1 summary + 2 ROIs), got {len(rows)}"

            # Check required columns
            assert "patient_id" in fieldnames
            assert "overall_status" in fieldnames
            assert "row_type" in fieldnames
            assert "roi_name" in fieldnames

            # Check summary row
            summary_row = rows[0]
            assert summary_row["row_type"] == "SUMMARY"
            assert summary_row["overall_status"] == "PASS"

            # Check ROI rows
            roi_rows = [r for r in rows if r["row_type"] == "ROI"]
            assert len(roi_rows) == 2

            roi_names = [r["roi_name"] for r in roi_rows]
            assert "PTV_70" in roi_names
            assert "Bladder" in roi_names

    def test_csv_gamma_columns(self, sample_passing_result):
        """Verify CSV includes gamma result columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.csv"
            generate_csv_report(sample_passing_result, str(output_path))

            with open(output_path, newline='') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                fieldnames = reader.fieldnames

            # Check gamma columns exist
            gamma_cols = [c for c in fieldnames if 'gamma' in c.lower()]
            assert len(gamma_cols) > 0, "No gamma columns found"

            # Check summary row has gamma values
            summary_row = rows[0]
            assert 'gamma_3pct_3mm_pass_rate' in summary_row or \
                   any('gamma' in k and 'pass_rate' in k for k in summary_row.keys())


class TestSecondaryCheckCriteria:
    """Tests for SecondaryCheckCriteria configuration."""

    def test_default_criteria(self):
        """Verify default criteria values."""
        criteria = SecondaryCheckCriteria()

        assert criteria.gamma_3_3_pass_rate == 0.95
        assert criteria.gamma_2_2_pass_rate == 0.90
        assert criteria.gamma_dose_threshold == 10.0
        assert criteria.gamma_global_mode == True
        assert criteria.target_d95_tolerance_rel == 0.03
        assert criteria.oar_dmax_tolerance_abs == 2.0

    def test_custom_criteria(self):
        """Verify custom criteria can be set."""
        criteria = SecondaryCheckCriteria(
            gamma_3_3_pass_rate=0.90,
            oar_dmax_tolerance_abs=3.0
        )

        assert criteria.gamma_3_3_pass_rate == 0.90
        assert criteria.oar_dmax_tolerance_abs == 3.0

    def test_criteria_to_dict(self):
        """Verify criteria serialization."""
        criteria = SecondaryCheckCriteria()
        d = criteria.to_dict()

        assert d['gamma_3_3_pass_rate'] == 0.95
        assert d['target_dmean_tolerance_rel'] == 0.02


class TestSecondaryCheckResult:
    """Tests for SecondaryCheckResult class."""

    def test_result_to_dict(self, sample_passing_result):
        """Verify result serialization."""
        d = sample_passing_result.to_dict()

        assert d['patient_id'] == "TEST001"
        assert d['overall_status'] == "PASS"
        assert 'gamma_results' in d
        assert 'dvh_results' in d
        assert 'mu_sanity' in d

    def test_result_without_mu_sanity(self):
        """Verify result without MU sanity check."""
        result = SecondaryCheckResult(
            patient_id="TEST",
            plan_name="Plan",
            plan_uid="1.2.3",
            timestamp=datetime.now().isoformat(),
            dosecuda_version="1.0"
        )
        result.overall_status = "PASS"
        result.mu_sanity = None

        d = result.to_dict()
        assert d['mu_sanity'] is None


class TestEmptyResults:
    """Tests for handling empty or minimal results."""

    def test_empty_gamma_results(self):
        """Test report with no gamma results."""
        result = SecondaryCheckResult(
            patient_id="TEST",
            plan_name="Plan",
            plan_uid="1.2.3",
            timestamp=datetime.now().isoformat(),
            dosecuda_version="1.0"
        )
        result.gamma_results = {}
        result.dvh_results = {}
        result.overall_status = "PASS"

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "report.json"
            csv_path = Path(tmpdir) / "report.csv"

            generate_json_report(result, str(json_path))
            generate_csv_report(result, str(csv_path))

            assert json_path.exists()
            assert csv_path.exists()

    def test_empty_dvh_results(self, sample_passing_result):
        """Test report with no DVH results."""
        sample_passing_result.dvh_results = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "report.csv"
            generate_csv_report(sample_passing_result, str(csv_path))

            with open(csv_path, newline='') as f:
                rows = list(csv.DictReader(f))

            # Should have only summary row
            assert len(rows) == 1
            assert rows[0]["row_type"] == "SUMMARY"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
