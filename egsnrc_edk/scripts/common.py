import yaml
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any
import numpy as np


@dataclass
class KernelGrid:
    energies: List[float]
    theta_edges: np.ndarray  # degrees
    r_edges: np.ndarray      # cm
    cumulative: bool = True

    @property
    def nE(self) -> int:
        return len(self.energies)

    @property
    def nTheta(self) -> int:
        return len(self.theta_edges) - 1

    @property
    def nR(self) -> int:
        return len(self.r_edges) - 1

    @property
    def theta_centers(self) -> np.ndarray:
        return 0.5 * (self.theta_edges[1:] + self.theta_edges[:-1])

    @property
    def r_centers(self) -> np.ndarray:
        return 0.5 * (self.r_edges[1:] + self.r_edges[:-1])

    @property
    def dr_cm(self) -> float:
        return float(self.r_edges[1] - self.r_edges[0])

    @property
    def r0_cm(self) -> float:
        return float(self.r_edges[0])

    @property
    def rmax_cm(self) -> float:
        return float(self.r_edges[-1])


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(f)
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_theta_edges(ncones: int, theta_max_deg: float) -> np.ndarray:
    return np.linspace(0.0, theta_max_deg, ncones + 1, dtype=float)


def build_r_edges(dr_cm: float, rmax_cm: float) -> np.ndarray:
    n_bins = int(np.ceil(rmax_cm / dr_cm))
    return np.linspace(0.0, n_bins * dr_cm, n_bins + 1, dtype=float)


def default_grid_from_config(cfg: Dict[str, Any]) -> KernelGrid:
    theta_max = float(cfg.get("theta_max_deg", 180.0))
    theta_edges = build_theta_edges(int(cfg["ncones_theta"]), theta_max)
    dr_cm = float(cfg["dr_mm"]) / 10.0
    r_edges = build_r_edges(dr_cm, float(cfg["rmax_cm"]))
    energies = [float(e) for e in cfg["energies_MeV"]]
    cumulative = bool(cfg.get("write_cumulative", True))
    return KernelGrid(energies=energies, theta_edges=theta_edges, r_edges=r_edges, cumulative=cumulative)


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
