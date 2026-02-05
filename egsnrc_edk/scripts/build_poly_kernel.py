#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from common import load_config, default_grid_from_config, KernelGrid


def main():
    ap = argparse.ArgumentParser(description="Build polyenergetic kernel from spectrum")
    ap.add_argument("--config", required=True)
    ap.add_argument("--numpy-dir", default="./numpy")
    ap.add_argument("--spectrum", required=True, help="CSV with E_MeV, weight")
    ap.add_argument("--out", default="./kernel_poly.bin")
    ap.add_argument("--cumulative", action="store_true")
    ap.add_argument("--layout", default="E-major")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    grid = default_grid_from_config(cfg)
    df = pd.read_csv(args.spectrum)
    # expect columns E_MeV, weight
    energies = np.array(df["E_MeV"], dtype=float)
    weights = np.array(df["weight"], dtype=float)
    weights = weights / weights.sum()

    # interpolate weights to grid energies
    w_interp = np.interp(grid.energies, energies, weights, left=0.0, right=0.0)
    w_interp = w_interp / w_interp.sum()

    name = "kcum" if (args.cumulative or cfg.get("write_cumulative", True)) else "kdiff"
    kernels = []
    for Ei in grid.energies:
        npz_path = Path(args.numpy_dir) / f"kernel_{Ei:.3f}MeV.npz"
        if npz_path.exists():
            with np.load(npz_path) as npz:
                kernels.append(npz[name])
        else:
            kernels.append(np.load(Path(args.numpy_dir) / f"{name}_{Ei:.3f}MeV.npy"))
    k = np.tensordot(w_interp, np.stack(kernels, axis=0), axes=1)

    # write binary using packer
    from pack_kernels import pack_binary

    normalization = cfg.get(
        "normalization", "per_incident_energy_interacting_at_origin"
    )
    # Pack as a single-energy kernel (nE=1). The effective energy is left as -1.0 to
    # signal "poly"; downstream may ignore it and rely only on payload.
    grid_poly = KernelGrid(
        energies=[-1.0],
        theta_edges=grid.theta_edges,
        r_edges=grid.r_edges,
        cumulative=(name == "kcum"),
    )

    pack_binary(
        Path(args.out),
        grid_poly,
        k[None, ...],
        cumulative=(name == "kcum"),
        layout=args.layout,
        normalization=normalization,
    )
    print(f"[poly] wrote {args.out} using spectrum {args.spectrum}")


if __name__ == "__main__":
    main()
