"""
CT Calibration Module - HU to Density Conversion
Inspired by OpenTPS architecture for robust clinical usage.

Supports:
- Multiple calibration curves per scanner/protocol
- Validation (monotonicity, range checks)
- Controlled extrapolation
- Legacy CSV format compatibility
"""

import numpy as np
import pandas as pd
import os
import warnings


class CTCalibration:
    """
    HU to density calibration curve with validation and extrapolation control.
    
    Attributes:
        name: Human-readable name (e.g., "Philips_BigBore_120kV")
        hu_points: Array of HU values (must be monotonically increasing)
        density_points: Array of corresponding density values [g/cc]
        extrapolate_low: How to handle HU < min(hu_points) ('clamp', 'linear', 'air')
        extrapolate_high: How to handle HU > max(hu_points) ('clamp', 'linear')
    """
    
    def __init__(self, name, hu_points, density_points, 
                 extrapolate_low='air', extrapolate_high='clamp'):
        """
        Initialize CT calibration curve.
        
        Args:
            name: Calibration curve name
            hu_points: Array of HU values (must be sorted)
            density_points: Array of density values [g/cc]
            extrapolate_low: 'clamp' (use min density), 'linear' (extrapolate), 'air' (0.0012 g/cc)
            extrapolate_high: 'clamp' (use max density), 'linear' (extrapolate)
        """
        self.name = name
        self.hu_points = np.array(hu_points, dtype=np.float64)
        self.density_points = np.array(density_points, dtype=np.float64)
        self.extrapolate_low = extrapolate_low
        self.extrapolate_high = extrapolate_high
        
        self._validate()
    
    def _validate(self):
        """Validate calibration curve integrity."""
        if len(self.hu_points) != len(self.density_points):
            raise ValueError(f"CTCalibration '{self.name}': HU e densidade devem ter mesmo tamanho "
                           f"({len(self.hu_points)} vs {len(self.density_points)})")
        
        if len(self.hu_points) < 2:
            raise ValueError(f"CTCalibration '{self.name}': mínimo 2 pontos necessários")
        
        # Check monotonicity (HU must be strictly increasing)
        if not np.all(np.diff(self.hu_points) > 0):
            raise ValueError(f"CTCalibration '{self.name}': HU deve ser estritamente crescente")
        
        # Check density range (physically reasonable)
        if np.any(self.density_points < 0):
            raise ValueError(f"CTCalibration '{self.name}': densidade negativa detectada")
        
        if np.any(self.density_points > 10.0):
            warnings.warn(f"CTCalibration '{self.name}': densidade > 10 g/cc detectada. "
                        "Verifique se valores são corretos.")
        
        # Check for duplicate HU values
        if len(np.unique(self.hu_points)) != len(self.hu_points):
            raise ValueError(f"CTCalibration '{self.name}': valores de HU duplicados detectados")
    
    def convert(self, hu_values):
        """
        Convert HU values to density [g/cc].
        
        Args:
            hu_values: Array-like of HU values
            
        Returns:
            Array of density values [g/cc], same shape as input
        """
        hu_values = np.asarray(hu_values)
        original_shape = hu_values.shape
        hu_flat = hu_values.flatten()
        
        density_flat = np.zeros_like(hu_flat, dtype=np.float32)
        
        # Interpolate within calibration range
        mask_in_range = (hu_flat >= self.hu_points[0]) & (hu_flat <= self.hu_points[-1])
        density_flat[mask_in_range] = np.interp(
            hu_flat[mask_in_range], 
            self.hu_points, 
            self.density_points
        )
        
        # Handle extrapolation below range
        mask_below = hu_flat < self.hu_points[0]
        if np.any(mask_below):
            if self.extrapolate_low == 'clamp':
                density_flat[mask_below] = self.density_points[0]
            elif self.extrapolate_low == 'linear':
                # Linear extrapolation using first two points
                slope = (self.density_points[1] - self.density_points[0]) / \
                       (self.hu_points[1] - self.hu_points[0])
                density_flat[mask_below] = self.density_points[0] + \
                    slope * (hu_flat[mask_below] - self.hu_points[0])
                # Clamp to air density minimum
                density_flat[mask_below] = np.maximum(density_flat[mask_below], 0.0012)
            elif self.extrapolate_low == 'air':
                density_flat[mask_below] = 0.0012  # Air density
            
            n_below = np.sum(mask_below)
            if n_below > 0:
                warnings.warn(f"{n_below} voxels com HU < {self.hu_points[0]:.0f} "
                            f"(extrapolação: {self.extrapolate_low})")
        
        # Handle extrapolation above range
        mask_above = hu_flat > self.hu_points[-1]
        if np.any(mask_above):
            if self.extrapolate_high == 'clamp':
                density_flat[mask_above] = self.density_points[-1]
            elif self.extrapolate_high == 'linear':
                # Linear extrapolation using last two points
                slope = (self.density_points[-1] - self.density_points[-2]) / \
                       (self.hu_points[-1] - self.hu_points[-2])
                density_flat[mask_above] = self.density_points[-1] + \
                    slope * (hu_flat[mask_above] - self.hu_points[-1])
                # Clamp to physically reasonable maximum
                density_flat[mask_above] = np.minimum(density_flat[mask_above], 5.0)
            
            n_above = np.sum(mask_above)
            if n_above > 0:
                warnings.warn(f"{n_above} voxels com HU > {self.hu_points[-1]:.0f} "
                            f"(extrapolação: {self.extrapolate_high})")
        
        return density_flat.reshape(original_shape).astype(np.float32)
    
    @classmethod
    def from_csv(cls, csv_path, name=None, **kwargs):
        """
        Load calibration curve from CSV file.
        
        CSV format (legacy):
            HU,Density
            -1024,0.0012
            -950,0.0012
            ...
        
        Args:
            csv_path: Path to CSV file
            name: Optional name override (default: filename)
            **kwargs: Additional arguments passed to CTCalibration()
        
        Returns:
            CTCalibration instance
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CT calibration CSV não encontrado: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Validate columns
        required_cols = ['HU', 'Density']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"CSV deve conter coluna '{col}'. Encontradas: {df.columns.tolist()}")
        
        hu_points = df['HU'].to_numpy()
        density_points = df['Density'].to_numpy()
        
        if name is None:
            name = os.path.splitext(os.path.basename(csv_path))[0]
        
        return cls(name, hu_points, density_points, **kwargs)
    
    def to_csv(self, csv_path):
        """
        Save calibration curve to CSV file.
        
        Args:
            csv_path: Output CSV path
        """
        df = pd.DataFrame({
            'HU': self.hu_points,
            'Density': self.density_points
        })
        df.to_csv(csv_path, index=False)
        print(f"✓ CT calibration '{self.name}' salva em: {csv_path}")
    
    def __repr__(self):
        return (f"CTCalibration('{self.name}', "
                f"n_points={len(self.hu_points)}, "
                f"HU_range=[{self.hu_points[0]:.0f}, {self.hu_points[-1]:.0f}], "
                f"density_range=[{self.density_points[0]:.4f}, {self.density_points[-1]:.4f}])")


class CTCalibrationManager:
    """
    Manage multiple CT calibration curves.
    
    Useful for multi-site deployments with different scanners.
    """
    
    def __init__(self):
        self.calibrations = {}
    
    def add_calibration(self, calibration):
        """Add a calibration curve."""
        if not isinstance(calibration, CTCalibration):
            raise TypeError("calibration deve ser uma instância de CTCalibration")
        self.calibrations[calibration.name] = calibration
    
    def get_calibration(self, name):
        """Get calibration by name."""
        if name not in self.calibrations:
            raise KeyError(f"Calibração '{name}' não encontrada. "
                         f"Disponíveis: {list(self.calibrations.keys())}")
        return self.calibrations[name]
    
    def list_calibrations(self):
        """List all available calibrations."""
        return list(self.calibrations.keys())
    
    def load_from_directory(self, directory_path, pattern='*HU_Density.csv'):
        """
        Load all calibration CSVs from a directory.
        
        Args:
            directory_path: Directory containing CSV files
            pattern: Glob pattern for CSV files
        """
        import glob
        
        csv_files = glob.glob(os.path.join(directory_path, pattern))
        
        for csv_path in csv_files:
            try:
                cal = CTCalibration.from_csv(csv_path)
                self.add_calibration(cal)
                print(f"✓ Calibração carregada: {cal.name}")
            except Exception as e:
                warnings.warn(f"Falha ao carregar {csv_path}: {e}")
        
        print(f"✓ Total de {len(self.calibrations)} calibrações carregadas")
    
    def __repr__(self):
        return f"CTCalibrationManager(n_calibrations={len(self.calibrations)})"


# Default calibration curves (commonly used)
def get_default_calibration(scanner_type='generic'):
    """
    Get a default calibration curve.
    
    Args:
        scanner_type: 'generic', 'philips', 'siemens', 'ge'
    
    Returns:
        CTCalibration instance
    """
    if scanner_type == 'generic':
        # Generic calibration (9 points from typical Gammex phantom)
        hu = [-1024, -950, -120, -90, 60, 240, 930, 1060, 1560]
        density = [0.0012, 0.0012, 0.95, 0.98, 1.05, 1.15, 1.53, 1.69, 2.30]
        return CTCalibration('Generic_9pt', hu, density)
    
    elif scanner_type == 'philips':
        # Philips typical calibration (extended range)
        hu = [-1024, -950, -100, -50, 0, 50, 100, 500, 1000, 1500, 2000]
        density = [0.0012, 0.0012, 0.93, 0.97, 1.00, 1.04, 1.08, 1.35, 1.62, 2.10, 2.50]
        return CTCalibration('Philips_Extended', hu, density, extrapolate_high='linear')
    
    else:
        raise ValueError(f"Scanner type '{scanner_type}' não suportado. "
                       "Opções: 'generic', 'philips'")
