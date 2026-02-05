#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import re
import json
import os
from typing import Optional, Dict, Any
from common import load_config, default_grid_from_config, KernelGrid, ensure_dir


float_pat = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _read_numbers(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return np.fromstring(text, sep=" ")


def _parse_egsdat(path: Path, grid: KernelGrid, energy_mev: float, ncases_hint: Optional[int]) -> np.ndarray:
    nums = _read_numbers(path)
    nreg = grid.nTheta * grid.nR + 1
    expected = 2 + (nreg - 1) * 12
    if len(nums) < expected:
        raise ValueError(f"{path} is too short for nreg={nreg}")
    data = nums[2 : 2 + (nreg - 1) * 12].reshape((nreg - 1, 12))
    doses = data[:, ::2]  # dose columns for i=0..5

    if ncases_hint is None or ncases_hint <= 0:
        if len(nums) >= 6:
            ncases_hint = int(round(nums[-6]))
    if not ncases_hint or ncases_hint <= 0:
        raise ValueError("Number of histories not provided and not found in .egsdat")

    kdiff_regions = doses[:, 0] / (ncases_hint * energy_mev)
    kdiff = np.zeros((grid.nTheta, grid.nR), dtype=np.float64)
    for idx, val in enumerate(kdiff_regions):
        ix = idx // grid.nTheta
        ic = idx % grid.nTheta
        kdiff[ic, ix] = val
    return kdiff


def read_3d_table(
    e_dir: Path, grid: KernelGrid, energy_mev: float, cfg: Dict[str, Any]
) -> tuple[np.ndarray, str, bool]:
    egsdat_files = list(e_dir.glob("*.egsdat"))
    if egsdat_files:
        ncases_hint = int(cfg.get("histories_per_energy", cfg.get("ncases", 0)))
        try:
            arr = _parse_egsdat(egsdat_files[0], grid, energy_mev, ncases_hint)
            return arr, egsdat_files[0].name, True
        except Exception as exc:
            print(f"Warning: failed to parse {egsdat_files[0]} ({exc}); trying legacy parsers")

    expected = grid.nTheta * grid.nR
    legacy_candidates = [
        e_dir / "edk.out",
        e_dir / "edk.dat",
        e_dir / "edknrc.edk",
        e_dir / "edknrc.out",
        e_dir / "edknrc.3ddose",
        e_dir / "edknrc.egsdat",
    ]
    for path in legacy_candidates:
        if not path.exists():
            continue
        arr = _read_numbers(path)
        if len(arr) == expected:
            return arr.reshape((grid.nTheta, grid.nR)), path.name, False
    raise RuntimeError(f"No parsable EDK output found in {e_dir}")


def normalization_factor(energy_mev: float, cfg: Dict[str, Any]) -> float:
    """
    Convert raw EDKnrc energy deposition (MeV per history) to a
    dimensionless fraction of incident energy interacting at the origin.
    Default: divide by incident energy (MeV) and by norm_per_particle (histories weight).
    """
    mode = cfg.get(
        "normalization",
        "per_incident_energy_interacting_at_origin",
    )
    norm_per_particle = float(cfg.get("norm_per_particle", 1.0))
    if mode == "per_incident_energy_interacting_at_origin":
        return energy_mev * norm_per_particle
    if mode == "per_history":
        return norm_per_particle
    raise ValueError(f"Unknown normalization mode: {mode}")


def normalize_kernel(kdiff: np.ndarray, energy_mev: float, cfg: Dict[str, Any]) -> np.ndarray:
    factor = normalization_factor(energy_mev, cfg)
    if factor <= 0:
        raise ValueError(f"Normalization factor must be >0, got {factor}")
    return kdiff / factor


def to_cumulative(kdiff: np.ndarray, grid: KernelGrid) -> np.ndarray:
    return np.cumsum(kdiff, axis=1)


def validate_kernel(kdiff: np.ndarray, kcum: np.ndarray, energy: float, grid: KernelGrid) -> Dict[str, float]:
    if not np.all(np.isfinite(kdiff)):
        raise ValueError(f"Found NaN/Inf in differential kernel for {energy:.3f} MeV")
    if not np.all(np.isfinite(kcum)):
        raise ValueError(f"Found NaN/Inf in cumulative kernel for {energy:.3f} MeV")

    # Small negative values may appear due to MC noise; clip at tiny threshold.
    neg_mask = kdiff < -1e-12
    if np.any(neg_mask):
        raise ValueError(f"Negative energy deposition detected (min={kdiff.min():.3e}) for {energy:.3f} MeV")
    kdiff[kdiff < 0] = 0.0

    monotonic_violations = np.diff(kcum, axis=1) < -1e-8
    if np.any(monotonic_violations):
        raise ValueError(f"Non-monotonic cumulative kernel for {energy:.3f} MeV")

    total = float(kdiff.sum())
    per_theta = kdiff.sum(axis=1)
    max_theta_total = float(per_theta.max())
    min_theta_total = float(per_theta.min())

    return {
        "total": total,
        "max_theta_total": max_theta_total,
        "min_theta_total": min_theta_total,
    }


def parse_one(e_dir: Path, energy: float, grid: KernelGrid, cfg):
    kdiff, src_name, already_norm = read_3d_table(e_dir, grid, energy, cfg)
    if not already_norm:
        kdiff = normalize_kernel(kdiff, energy, cfg)
    kcum = to_cumulative(kdiff, grid)
    stats = validate_kernel(kdiff, kcum, energy, grid)
    return kdiff, kcum, src_name, stats


def save_numpy(out_dir: Path, energy: float, kdiff: np.ndarray, kcum: np.ndarray, meta: Dict[str, Any]):
    ensure_dir(out_dir)
    np.save(out_dir / f"kdiff_{energy:.3f}MeV.npy", kdiff)
    np.save(out_dir / f"kcum_{energy:.3f}MeV.npy", kcum)
    np.savez_compressed(out_dir / f"kernel_{energy:.3f}MeV.npz", kdiff=kdiff, kcum=kcum, meta=meta)


def main():
    ap = argparse.ArgumentParser(description="Parse EDKnrc outputs into numpy arrays")
    ap.add_argument("--config", required=True)
    ap.add_argument("--work-dir", default="./work_inputs")
    ap.add_argument("--out-dir", default="./numpy")
    ap.add_argument("--csv-dir", default=None, help="If set, dump per-energy CSV for debug")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    grid = default_grid_from_config(cfg)
    work_dir = Path(args.work_dir)
    out_dir = Path(args.out_dir)

    for Ei in grid.energies:
        e_dir = work_dir / f"E_{Ei:.3f}MeV"
        kdiff, kcum, src_name, stats = parse_one(e_dir, Ei, grid, cfg)
        meta = {
            "energy_MeV": Ei,
            "source_file": src_name,
            "normalization": cfg.get(
                "normalization", "per_incident_energy_interacting_at_origin"
            ),
            "total_integral": stats["total"],
        }
        save_numpy(out_dir, Ei, kdiff, kcum, meta)
        if args.csv_dir:
            csv_dir = Path(args.csv_dir)
            ensure_dir(csv_dir)
            theta_cent = grid.theta_centers
            r_cent = grid.r_centers
            with (csv_dir / f"kernel_{Ei:.3f}MeV.csv").open("w", encoding="utf-8") as f:
                f.write("E_MeV,theta_deg,r_cm,value,kind\n")
                for ti, th in enumerate(theta_cent):
                    for ri, r in enumerate(r_cent):
                        f.write(f"{Ei:.6f},{th:.6f},{r:.6f},{kdiff[ti,ri]:.8e},diff\n")
                        f.write(f"{Ei:.6f},{th:.6f},{r:.6f},{kcum[ti,ri]:.8e},cum\n")

        total = stats["total"]
        tol = 0.05  # allow up to 5% loss (escape / cutoffs)
        if not (1 - tol <= total <= 1 + tol):
            print(f"Warning: integral {total:.4e} outside [0.95,1.05] for {Ei:.3f} MeV")
        print(
            f"[parse] {Ei:.3f} MeV total={total:.4e} src={src_name} "
            f"theta[min,max]=({stats['min_theta_total']:.3e},{stats['max_theta_total']:.3e})"
        )

    meta = {
        "theta_edges": grid.theta_edges.tolist(),
        "r_edges_cm": grid.r_edges.tolist(),
        "energies_MeV": grid.energies,
        "normalization": cfg.get(
            "normalization", "per_incident_energy_interacting_at_origin"
        ),
        "cumulative": bool(cfg.get("write_cumulative", True)),
    }
    (out_dir / "grid_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
