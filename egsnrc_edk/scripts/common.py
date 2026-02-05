import yaml
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple
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


def _build_r_edges_uniform(dr_cm: float, rmax_cm: float) -> np.ndarray:
    n_bins_float = rmax_cm / dr_cm
    n_bins = int(round(n_bins_float))
    if abs(n_bins_float - n_bins) > 1e-6:
        raise ValueError(
            f"rmax_cm/dr_cm is not integer (got {n_bins_float}); adjust dr or rmax"
        )
    return np.linspace(0.0, n_bins * dr_cm, n_bins + 1, dtype=float)


def _build_r_edges_segmented(segments: List[Dict[str, Any]]) -> np.ndarray:
    edges: List[float] = [0.0]
    for seg in segments:
        rmax_cm = float(seg["rmax_cm"])
        dr_cm = float(seg["dr_mm"]) / 10.0
        if dr_cm <= 0 or rmax_cm <= edges[-1]:
            raise ValueError(f"Invalid radial segment: {seg}")
        span = rmax_cm - edges[-1]
        n_steps = int(round(span / dr_cm))
        if abs(span - n_steps * dr_cm) > 1e-6:
            raise ValueError(
                f"Segment {seg} not an integer multiple of dr; span={span}, dr={dr_cm}"
            )
        new_edges = list(edges[-1] + np.arange(1, n_steps + 1) * dr_cm)
        edges.extend(new_edges)
    # Deduplicate tiny overlaps
    edges = [edges[0]] + [e for i, e in enumerate(edges[1:]) if e - edges[i] > 1e-9]
    return np.array(edges, dtype=float)


def default_grid_from_config(cfg: Dict[str, Any]) -> KernelGrid:
    theta_max = float(cfg.get("theta_max_deg", 180.0))
    theta_edges = build_theta_edges(int(cfg["ncones_theta"]), theta_max)

    radial_cfg = cfg.get("radial")
    if radial_cfg and radial_cfg.get("mode") == "segmented":
        segments = radial_cfg.get("segments", [])
        if not segments:
            raise ValueError("radial.mode=segmented requires segments")
        r_edges = _build_r_edges_segmented(segments)
    else:
        dr_cm = float(
            (radial_cfg or {}).get("dr_mm", cfg.get("dr_mm", 1.0))
        ) / 10.0
        rmax_cm = float(
            (radial_cfg or {}).get("rmax_cm", cfg.get("rmax_cm", 10.0))
        )
        r_edges = _build_r_edges_uniform(dr_cm, rmax_cm)

    energies = [float(e) for e in cfg["energies_MeV"]]
    cumulative = bool(cfg.get("write_cumulative", True))
    return KernelGrid(energies=energies, theta_edges=theta_edges, r_edges=r_edges, cumulative=cumulative)


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
