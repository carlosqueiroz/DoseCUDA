#!/usr/bin/env python3
import argparse
import struct
from pathlib import Path
import numpy as np
from common import load_config, default_grid_from_config

MAGIC = b"EDKCCCS\0"
VERSION = 2
LAYOUT_BYTES = 16
NORM_BYTES = 64


def _pack_fixed_string(text: str, size: int) -> bytes:
    data = text.encode("ascii", errors="ignore")
    if len(data) > size:
        data = data[:size]
    return data + b"\0" * (size - len(data))


def pack_binary(
    out_path: Path,
    grid,
    k_array: np.ndarray,
    cumulative: bool = True,
    layout: str = "E-major",
    normalization: str = "per_incident_energy_interacting_at_origin",
):
    """
    Header layout (little-endian):
      magic[8] = "EDKCCCS\\0"
      uint32 version
      uint32 nE, nTheta, nR
      float32 energies[nE]
      float32 theta_edges[nTheta+1] (degrees)
      float32 dr_cm, r0_cm, rmax_cm
      uint32 flags (bit0 = cumulative payload)
      char layout[16] (ASCII, NUL padded)
      char normalization[64] (ASCII, NUL padded)
      float32 r_edges[nR+1] (cm)
      payload float32 k_array[E][theta][r] in C order (E-major)
    """
    # k_array shape: (nE, nTheta, nR)
    with out_path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<III", grid.nE, grid.nTheta, grid.nR))
        f.write(struct.pack(f"<{grid.nE}f", *grid.energies))
        f.write(struct.pack(f"<{grid.nTheta+1}f", *grid.theta_edges))
        f.write(struct.pack("<3f", grid.dr_cm, grid.r0_cm, grid.rmax_cm))
        flags = 1 if cumulative else 0
        f.write(struct.pack("<I", flags))
        f.write(_pack_fixed_string(layout, LAYOUT_BYTES))
        f.write(_pack_fixed_string(normalization, NORM_BYTES))
        f.write(struct.pack(f"<{grid.nR+1}f", *grid.r_edges))
        f.write(k_array.astype(np.float32, copy=False).tobytes(order="C"))


def main():
    ap = argparse.ArgumentParser(description="Pack monoenergetic kernels into binary")
    ap.add_argument("--config", required=True)
    ap.add_argument("--numpy-dir", default="./numpy")
    ap.add_argument("--out", default="./kernels_mono.bin")
    ap.add_argument("--use-cumulative", action="store_true", help="Use cumulative arrays")
    ap.add_argument("--layout", default="E-major", help="Data layout tag")
    ap.add_argument(
        "--normalization",
        default=None,
        help="Override normalization string stored in header",
    )
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    grid = default_grid_from_config(cfg)
    numpy_dir = Path(args.numpy_dir)
    use_cum = args.use_cumulative or bool(cfg.get("write_cumulative", True))
    normalization = args.normalization or cfg.get(
        "normalization", "per_incident_energy_interacting_at_origin"
    )

    data = []
    for Ei in grid.energies:
        name = "kcum" if use_cum else "kdiff"
        npz_path = numpy_dir / f"kernel_{Ei:.3f}MeV.npz"
        if npz_path.exists():
            with np.load(npz_path) as npz:
                arr = npz[name]
        else:
            arr = np.load(numpy_dir / f"{name}_{Ei:.3f}MeV.npy")
        data.append(arr)
    k_array = np.stack(data, axis=0)

    pack_binary(
        Path(args.out),
        grid,
        k_array,
        cumulative=use_cum,
        layout=args.layout,
        normalization=normalization,
    )
    print(
        f"[pack] wrote {args.out} shape={k_array.shape} cumulative={use_cum} "
        f"layout={args.layout} norm={normalization}"
    )


if __name__ == "__main__":
    main()
