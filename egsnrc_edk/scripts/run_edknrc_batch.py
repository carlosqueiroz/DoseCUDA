#!/usr/bin/env python3
import argparse
import subprocess
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import shutil
from common import load_config, default_grid_from_config, ensure_dir


def _stage_input_for_edknrc(input_path: Path, egs_home: Path, e_dir: Path) -> str:
    """
    EDKnrc expects input decks under $EGS_HOME/edknrc/<name> and the
    run directory name is derived from the input *basename*. Avoid
    slashes in the name by staging a flattened symlink.
    """
    target_dir = egs_home / "edknrc"
    ensure_dir(target_dir)
    staged_name = f"{e_dir.name}.egsinp"
    staged = target_dir / staged_name
    if staged.exists() or staged.is_symlink():
        staged.unlink()
    staged.symlink_to(input_path)
    return staged_name


def run_one(
    e_dir: Path,
    edk_exec: Path,
    log_dir: Path,
    pegs_path: str,
    runner: str | None,
    egs_home: Path,
) -> int:
    input_path = e_dir / "edknrc.egsinp"
    log_path = log_dir / f"{e_dir.name}.log"
    staged_rel = _stage_input_for_edknrc(input_path, egs_home, e_dir)
    if edk_exec:
        cmd = [str(edk_exec), "-i", staged_rel, "-p", pegs_path]
    else:
        cmd = [runner or "run_user_code", "edknrc", staged_rel, pegs_path]
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            cwd=e_dir,
        )
    # Copy produced outputs back to the energy directory for downstream parsing.
    base = Path(staged_rel).stem
    out_dir = egs_home / "edknrc"
    # search both top-level and run subdirectories
    candidates = list(out_dir.glob(f"{base}.*")) + list(out_dir.glob(f"egsrun_*{base}*/*"))
    for src in candidates:
        if not src.is_file():
            continue
        if src.suffix.lower() in {".egsdat", ".errors", ".egslst", ".keV", ".out", ".dat"}:
            shutil.copy(src, e_dir / src.name)
    return proc.returncode


def main():
    ap = argparse.ArgumentParser(description="Run EDKnrc for each energy in parallel")
    ap.add_argument("--config", required=True)
    ap.add_argument("--work-dir", default="./work_inputs")
    ap.add_argument(
        "--edk-exec",
        default=None,
        help="Path to EDKnrc executable (optional when using run_user_code)",
    )
    ap.add_argument("--jobs", type=int, default=max(1, cpu_count() - 1))
    ap.add_argument("--log-dir", default="./logs")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    grid = default_grid_from_config(cfg)
    work_dir = Path(args.work_dir)
    log_dir = Path(args.log_dir)
    ensure_dir(log_dir)

    if args.edk_exec:
        edk_exec = Path(args.edk_exec).expanduser()
    else:
        egs_home = Path(
            cfg.get("egs_home", None)
            or os.environ.get("EGS_HOME", "/opt/EGSnrc/egs_home")
        )
        hen_house = Path(
            cfg.get("hen_house", None)
            or os.environ.get("HEN_HOUSE", "/opt/EGSnrc/HEN_HOUSE")
        )
        config_name = Path(os.environ.get("EGS_CONFIG", "")).stem or "docker"
        candidates = [
            egs_home / "bin" / config_name / "edknrc",
            egs_home / "bin" / config_name / "edknrc.exe",
            egs_home.parent / "egs_homebin" / config_name / "edknrc",
            hen_house / "bin" / config_name / "edknrc",
        ]
        edk_exec = next((p for p in candidates if p.exists()), None)
    if not edk_exec or not Path(edk_exec).exists():
        raise SystemExit(
            "EDKnrc executable not found. Pass --edk-exec or set EGS_HOME/EGS_CONFIG. "
            f"Checked: {candidates if 'candidates' in locals() else edk_exec}"
        )

    energies_dirs = [work_dir / f"E_{Ei:.3f}MeV" for Ei in grid.energies]

    pegs_file = cfg.get("pegs4_data_file", "521icru.pegs4dat")
    pegs_dir = cfg.get("pegs4_data_dir", "/opt/EGSnrc/HEN_HOUSE/pegs4/data")
    pegs_path = pegs_file if pegs_file.startswith("/") else str(Path(pegs_dir) / pegs_file)
    runner = str(Path(os.environ.get("HEN_HOUSE", "/opt/EGSnrc/HEN_HOUSE")) / "scripts" / "run_user_code")
    egs_home = Path(
        cfg.get("egs_home", None)
        or os.environ.get("EGS_HOME", "/opt/EGSnrc/egs_home")
    )

    with Pool(processes=args.jobs) as pool:
        results = pool.starmap(
            run_one,
            [(edir, edk_exec, log_dir, pegs_path, runner, egs_home) for edir in energies_dirs],
        )

    failed = [grid.energies[i] for i, rc in enumerate(results) if rc != 0]
    if failed:
        raise SystemExit(f"EDKnrc failed for energies: {failed}")
    print("All EDKnrc runs completed")


if __name__ == "__main__":
    main()
