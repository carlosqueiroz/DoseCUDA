#!/usr/bin/env python3
"""
=============================================================================
DoseCUDA Kernel Validator
=============================================================================

Validates kernel.csv files to ensure they meet DoseCUDA requirements and
physics constraints.

Checks performed:
1. File format and required columns
2. Parameter ranges and physics constraints
3. Kernel shape and monotonicity
4. Comparison with reference kernels (optional)

Author: DoseCUDA Team
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class KernelValidator:
    """Validates DoseCUDA kernel files."""

    # Required columns
    REQUIRED_COLUMNS = ['theta', 'Am', 'am', 'Bm', 'bm', 'ray_length']

    # Expected number of angles
    VALID_ANGLE_COUNTS = [6, 12]

    # Physical bounds for parameters
    PARAM_BOUNDS = {
        'Am': (1e-8, 100),
        'am': (0.01, 50),
        'Bm': (1e-10, 10),
        'bm': (0.001, 10),
        'ray_length': (0.1, 50),
    }

    def __init__(self, kernel_path: str):
        """
        Parameters:
            kernel_path: Path to kernel.csv file
        """
        self.kernel_path = Path(kernel_path)
        self.df = None
        self.errors = []
        self.warnings = []
        self.info = []

    def load(self) -> bool:
        """Load kernel file."""
        if not self.kernel_path.exists():
            self.errors.append(f"File not found: {self.kernel_path}")
            return False

        try:
            self.df = pd.read_csv(self.kernel_path)
            self.df.columns = [c.strip().lower() for c in self.df.columns]
            self.info.append(f"Loaded {len(self.df)} rows from {self.kernel_path.name}")
            return True
        except Exception as e:
            self.errors.append(f"Failed to read CSV: {e}")
            return False

    def check_columns(self) -> bool:
        """Check required columns exist."""
        missing = [c for c in self.REQUIRED_COLUMNS if c not in self.df.columns]
        if missing:
            self.errors.append(f"Missing columns: {missing}")
            return False
        return True

    def check_row_count(self) -> bool:
        """Check number of angles."""
        n_rows = len(self.df)
        if n_rows not in self.VALID_ANGLE_COUNTS:
            self.warnings.append(
                f"Unusual row count: {n_rows} (expected {self.VALID_ANGLE_COUNTS})"
            )
            return False
        return True

    def check_parameter_ranges(self) -> bool:
        """Check parameters are within physical bounds."""
        all_ok = True

        for param, (low, high) in self.PARAM_BOUNDS.items():
            values = self.df[param].values

            if (values < low).any():
                self.errors.append(
                    f"{param} has values below minimum ({low}): "
                    f"min={values.min():.4e}"
                )
                all_ok = False

            if (values > high).any():
                self.warnings.append(
                    f"{param} has values above typical maximum ({high}): "
                    f"max={values.max():.4e}"
                )

        return all_ok

    def check_positivity(self) -> bool:
        """Check all parameters are positive."""
        all_ok = True

        for param in ['Am', 'am', 'Bm', 'bm', 'ray_length']:
            if (self.df[param] <= 0).any():
                self.errors.append(f"{param} has non-positive values")
                all_ok = False

        return all_ok

    def check_attenuation_ordering(self) -> bool:
        """Check am > bm (primary attenuates faster than scatter)."""
        am = self.df['am'].values
        bm = self.df['bm'].values

        if not (am > bm).all():
            self.warnings.append(
                "am should be > bm (primary attenuates faster than scatter). "
                "This may indicate fitting issues."
            )
            return False
        return True

    def check_angular_coverage(self) -> bool:
        """Check angular bins cover forward and backward directions."""
        theta = self.df['theta'].values

        # Should have angles near 0 (forward) and near 90+ (lateral/backward)
        has_forward = theta.min() < 15
        has_lateral = theta.max() > 75

        if not has_forward:
            self.warnings.append(
                f"Missing forward direction coverage (min theta = {theta.min():.1f}°)"
            )
        if not has_lateral:
            self.warnings.append(
                f"Missing lateral/backward coverage (max theta = {theta.max():.1f}°)"
            )

        # Check monotonic theta
        if not (np.diff(theta) > 0).all():
            self.errors.append("Theta values should be monotonically increasing")
            return False

        return has_forward and has_lateral

    def check_kernel_shape(self) -> bool:
        """Check kernel has physically reasonable shape."""
        # Am should generally decrease with angle (forward peaked)
        Am = self.df['Am'].values

        # Allow some non-monotonicity but check general trend
        if Am[-1] > Am[0]:
            self.warnings.append(
                "Am increases with angle. Kernels are typically forward-peaked. "
                f"Am(forward)={Am[0]:.4e}, Am(backward)={Am[-1]:.4e}"
            )
            return False

        return True

    def check_data_types(self) -> bool:
        """Check all values are numeric."""
        all_ok = True

        for col in self.REQUIRED_COLUMNS:
            if not np.issubdtype(self.df[col].dtype, np.number):
                self.errors.append(f"Column {col} contains non-numeric values")
                all_ok = False

            if self.df[col].isna().any():
                self.errors.append(f"Column {col} contains NaN values")
                all_ok = False

            if np.isinf(self.df[col]).any():
                self.errors.append(f"Column {col} contains infinite values")
                all_ok = False

        return all_ok

    def validate(self) -> bool:
        """Run all validation checks."""
        if not self.load():
            return False

        checks = [
            ('Columns', self.check_columns),
            ('Row count', self.check_row_count),
            ('Data types', self.check_data_types),
            ('Positivity', self.check_positivity),
            ('Parameter ranges', self.check_parameter_ranges),
            ('Attenuation ordering', self.check_attenuation_ordering),
            ('Angular coverage', self.check_angular_coverage),
            ('Kernel shape', self.check_kernel_shape),
        ]

        results = []
        for name, check_fn in checks:
            try:
                passed = check_fn()
                results.append((name, passed))
            except Exception as e:
                self.errors.append(f"Check '{name}' failed with error: {e}")
                results.append((name, False))

        return len(self.errors) == 0

    def report(self) -> str:
        """Generate validation report."""
        lines = []
        lines.append("=" * 60)
        lines.append(f"Kernel Validation Report: {self.kernel_path.name}")
        lines.append("=" * 60)

        if self.info:
            lines.append("\nInfo:")
            for msg in self.info:
                lines.append(f"  • {msg}")

        if self.errors:
            lines.append("\n❌ ERRORS (must fix):")
            for msg in self.errors:
                lines.append(f"  • {msg}")

        if self.warnings:
            lines.append("\n⚠️  WARNINGS (review recommended):")
            for msg in self.warnings:
                lines.append(f"  • {msg}")

        if not self.errors and not self.warnings:
            lines.append("\n✅ All checks passed!")

        if self.df is not None:
            lines.append("\n" + "-" * 40)
            lines.append("Parameter Summary:")
            lines.append(f"  Angles: {len(self.df)}")
            lines.append(f"  Theta range: {self.df['theta'].min():.1f}° - {self.df['theta'].max():.1f}°")
            lines.append(f"  Am range: {self.df['Am'].min():.4e} - {self.df['Am'].max():.4e}")
            lines.append(f"  am range: {self.df['am'].min():.4f} - {self.df['am'].max():.4f}")
            lines.append(f"  Bm range: {self.df['Bm'].min():.4e} - {self.df['Bm'].max():.4e}")
            lines.append(f"  bm range: {self.df['bm'].min():.4f} - {self.df['bm'].max():.4f}")
            lines.append(f"  ray_length: {self.df['ray_length'].mean():.2f} cm (mean)")

        lines.append("")
        return "\n".join(lines)


def validate_kernel(kernel_path: str, verbose: bool = True) -> bool:
    """
    Validate a kernel file.

    Parameters:
        kernel_path: Path to kernel.csv
        verbose: Print report to stdout

    Returns:
        True if validation passed (no errors)
    """
    validator = KernelValidator(kernel_path)
    passed = validator.validate()

    if verbose:
        print(validator.report())

    return passed


def main():
    parser = argparse.ArgumentParser(
        description="Validate DoseCUDA kernel files"
    )
    parser.add_argument('kernel_files', nargs='+', help='Kernel CSV files to validate')
    parser.add_argument('--quiet', '-q', action='store_true', help='Only show errors')

    args = parser.parse_args()

    all_passed = True
    for kernel_file in args.kernel_files:
        passed = validate_kernel(kernel_file, verbose=not args.quiet)
        all_passed = all_passed and passed

        if not passed and args.quiet:
            print(f"FAILED: {kernel_file}")

    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
