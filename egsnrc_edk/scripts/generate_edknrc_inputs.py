#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import random
from typing import List
from common import load_config, default_grid_from_config, ensure_dir


def format_csv(vals: List[float], per_line: int = 8) -> str:
    lines = []
    for i in range(0, len(vals), per_line):
        chunk = vals[i : i + per_line]
        lines.append(", ".join(f"{v:.6g}" for v in chunk))
    return "\n        ".join(lines)


def format_radii_multiline(r_edges_cm: np.ndarray, per_line: int = 10) -> str:
    # r_edges_cm includes 0; RADII expects outer edges (exclude 0)
    radii = [float(r) for r in r_edges_cm[1:]]
    lines = []
    for i in range(0, len(radii), per_line):
        chunk = radii[i : i + per_line]
        line = ", ".join(f"{v:.6g}" for v in chunk)
        if i + per_line < len(radii):
            line += ","
        lines.append(line)
    return "\n        ".join(lines)


def write_edk_input(out_path: Path, energy: float, grid, cfg):
    histories = int(cfg.get("histories_per_energy", cfg.get("ncases", 2_00_000_000)))
    rng_seeds = cfg.get("rng_seeds")
    if not rng_seeds:
        rng_seeds = [random.randrange(1, 30000), random.randrange(1, 30000)]
    else:
        rng_seeds = [int(s) for s in rng_seeds]
    medium = cfg.get("medium_name", "H2O521ICRU")
    pegsfile = cfg.get("pegs4_data_file", "521icru.pegs4dat")
    pegs_dir = cfg.get("pegs4_data_dir", "/opt/EGSnrc/HEN_HOUSE/pegs4/data")
    pegs_full = pegsfile if pegsfile.startswith("/") else str(Path(pegs_dir) / pegsfile)
    ecut = cfg.get("ecut_MeV", 0.521)
    pcut = cfg.get("pcut_MeV", 0.010)

    # cones: use angular widths; if uniform emit single entry (EDKnrc expands)
    theta_widths = np.diff(grid.theta_edges)
    if np.allclose(theta_widths, theta_widths[0], rtol=0, atol=1e-6):
        angles_block = f"{theta_widths[0]:.6g}"
    else:
        angles_block = format_csv(theta_widths.tolist())

    # spheres: explicit outer radii list (with trailing commas to force multiline read)
    radii_block = format_radii_multiline(
        grid.r_edges, per_line=int(cfg.get("radii_per_line", 10))
    )
    total_regions = 1 + grid.nTheta * grid.nR

    content = f"""
TITLE= Energy deposition kernel for {energy:.3f} MeV photons
PEGS FILE= {pegs_full}

:start I/O control:
IRESTART= first
STORE DATA ARRAYS= yes
PRINT OUT EDK FILE= yes
:stop I/O control:

:start Monte Carlo inputs:
NUMBER OF HISTORIES= {histories}
INITIAL RANDOM NO. SEEDS= {rng_seeds[0]}, {rng_seeds[1]}
IFULL= ENERGY DEPOSITION KERNEL
DOPPLER BROADENING= On
:stop Monte Carlo inputs:

:start geometrical inputs:
 NUMBER OF CONES= {grid.nTheta}
 ANGLES= {angles_block}

 RADII= {radii_block}
 CAVITY ZONES= 0

MEDIA= {medium};

MEDNUM= 1
START REGION= 2
STOP REGION= {total_regions}
:stop geometrical inputs:

:start source inputs:
INCIDENT PARTICLE= photon
INCIDENT ENERGY= monoenergetic
INCIDENT KINETIC ENERGY(MEV)= {energy:.6f}
SOURCE NUMBER= 0
:stop source inputs:

:start MC transport parameter:
Global ECUT= {ecut}
Global PCUT= {pcut}
Global SMAX= 0.0
ESTEPE= 0.25
XImax= 0.0
Skin depth for BCA= 3
Boundary crossing algorithm= EXACT
Electron-step algorithm= PRESTA-II
Spin effects= on
Bremsstrahlung angular sampling= simple
Bound Compton scattering= on
Pair angular sampling= on
Electron impact ionization= on
:stop MC transport parameter:
"""
    out_path.write_text(content.strip() + "\n", encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description="Generate EDKnrc input decks per energy.")
    p.add_argument("--config", required=True, help="YAML/JSON config file")
    p.add_argument("--work-dir", default="./work_inputs", help="Where to place per-energy dirs")
    p.add_argument("--energies", help="Comma-separated override for energies (MeV)")
    p.add_argument("--ncones", type=int, help="Override number of polar cones")
    p.add_argument("--dr-mm", type=float, help="Override radial bin size (mm)")
    p.add_argument("--rmax-cm", type=float, help="Override max radius (cm)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(Path(args.config))
    if args.energies:
        cfg["energies_MeV"] = [float(x) for x in args.energies.split(",") if x]
    if args.ncones:
        cfg["ncones_theta"] = args.ncones
    if args.dr_mm:
        cfg.setdefault("radial", {"mode": "uniform"})
        cfg["radial"]["mode"] = "uniform"
        cfg["radial"]["dr_mm"] = args.dr_mm
        if args.rmax_cm:
            cfg["radial"]["rmax_cm"] = args.rmax_cm
    elif args.rmax_cm:
        cfg.setdefault("radial", {"mode": "uniform"})
        cfg["radial"]["rmax_cm"] = args.rmax_cm

    grid = default_grid_from_config(cfg)
    work_dir = Path(args.work_dir)
    ensure_dir(work_dir)

    for Ei in grid.energies:
        e_dir = work_dir / f"E_{Ei:.3f}MeV"
        ensure_dir(e_dir)
        input_path = e_dir / "edknrc.egsinp"
        write_edk_input(input_path, Ei, grid, cfg)
        # metadata for downstream steps
        meta = {
            "energy_MeV": Ei,
            "input_path": str(input_path),
            "theta_max_deg": float(cfg.get("theta_max_deg", 180.0)),
            "ncones": grid.nTheta,
            "r_edges_cm": grid.r_edges.tolist(),
        }
        (e_dir / "metadata.json").write_text(
            __import__("json").dumps(meta, indent=2), encoding="utf-8"
        )
        print(f"[generate] wrote {input_path}")


if __name__ == "__main__":
    main()
