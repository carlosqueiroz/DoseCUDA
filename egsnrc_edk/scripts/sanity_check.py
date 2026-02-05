#!/usr/bin/env python3
import argparse
import struct
from pathlib import Path
import numpy as np


MAGIC = b"EDKCCCS\0"


def read_binary(path: Path):
    data = path.read_bytes()
    offset = 0
    if data[:8] != MAGIC:
        raise ValueError("Invalid magic")
    offset += 8
    version = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    nE, nTheta, nR = struct.unpack_from("<III", data, offset)
    offset += 12
    energies = struct.unpack_from(f"<{nE}f", data, offset)
    offset += 4 * nE
    theta_edges = struct.unpack_from(f"<{nTheta+1}f", data, offset)
    offset += 4 * (nTheta + 1)
    dr_cm, r0_cm, rmax_cm = struct.unpack_from("<3f", data, offset)
    offset += 12
    flags = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    layout = bytes(data[offset : offset + 16]).split(b"\0", 1)[0].decode("ascii", "ignore")
    offset += 16
    normalization = (
        bytes(data[offset : offset + 64]).split(b"\0", 1)[0].decode("ascii", "ignore")
    )
    offset += 64
    r_edges = struct.unpack_from(f"<{nR+1}f", data, offset)
    offset += 4 * (nR + 1)
    payload = np.frombuffer(data, dtype=np.float32, offset=offset)
    expected = nE * nTheta * nR
    if payload.size != expected:
        raise ValueError(f"Payload size mismatch: got {payload.size}, expected {expected}")
    payload = payload.reshape((nE, nTheta, nR))
    return {
        "version": version,
        "nE": nE,
        "nTheta": nTheta,
        "nR": nR,
        "energies": np.array(energies, dtype=np.float32),
        "theta_edges": np.array(theta_edges, dtype=np.float32),
        "dr_cm": dr_cm,
        "r0_cm": r0_cm,
        "rmax_cm": rmax_cm,
        "flags": flags,
        "layout": layout,
        "normalization": normalization,
        "r_edges": np.array(r_edges, dtype=np.float32),
        "payload": payload,
    }


def run_checks(info):
    payload = info["payload"]
    cumulative = bool(info["flags"] & 0x1)

    if not np.all(np.isfinite(payload)):
        raise ValueError("NaN/Inf detected in payload")

    if cumulative:
        diffs = np.diff(payload, axis=2)
        kdiff = np.concatenate(
            [payload[:, :, :1], diffs], axis=2
        )  # convert cumulative to differential per shell
        if np.any(diffs < -1e-8):
            raise ValueError("Cumulative kernel not monotonic in r")
    else:
        kdiff = payload

    if np.any(kdiff < -1e-10):
        raise ValueError(f"Negative kdiff detected: min={kdiff.min():.3e}")

    sums = kdiff.sum(axis=(1, 2))
    return kdiff, sums


def main():
    ap = argparse.ArgumentParser(description="Sanity-check kernels_mono.bin")
    ap.add_argument("path", help="Path to kernels_mono.bin")
    ap.add_argument("--tol", type=float, default=0.15, help="Allowed deviation of total integral from 1.0")
    args = ap.parse_args()

    info = read_binary(Path(args.path))
    kdiff, sums = run_checks(info)
    ok = True
    for i, total in enumerate(sums):
        if not (1 - args.tol <= total <= 1 + args.tol):
            ok = False
        print(
            f"E[{i}]={info['energies'][i]:.4g} MeV  total={total:.6f} "
            f"min={kdiff[i].min():.3e} max={kdiff[i].max():.3e}"
        )
    if not ok:
        raise SystemExit("Totals outside tolerance; see above.")
    print(f"OK: nE={info['nE']} nTheta={info['nTheta']} nR={info['nR']} cumulative={bool(info['flags'] & 1)}")


if __name__ == "__main__":
    main()
