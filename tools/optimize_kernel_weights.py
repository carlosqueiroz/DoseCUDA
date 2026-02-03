#!/usr/bin/env python3
"""
Improved coordinate-descent optimizer for beam model kernel weights (+ optional gamma).

What it does
------------
- Loads an RTPLAN + CT (from phase dir) + reference RTDOSE (template / TPS dose).
- Recomputes CCC dose on GPU for candidate parameter changes.
- Resamples calculated dose to the reference RTDOSE grid.
- Optimizes `kernel_weights` (6-direction weights) using coordinate descent.

Improvements vs the original script
-----------------------------------
1) Explicit iso spacing control (--iso-spacing) instead of inferring from CT.
2) Applies kernel_weights to ALL beam models in the plan (unless --model-index is given).
3) Objective can include: RMSE + MAE + (optional) gamma pass-rate penalties.
4) Best-effort reuse of the CT-loaded dose grid to avoid re-loading CT for every candidate.
5) Writes a detailed JSONL log + saves intermediate and final best weights/dose.

Usage example
-------------
./DoseCuda/bin/python tools/optimize_kernel_weights_v2.py \
  --phase-dir tests/test_patient_output/phase_2 \
  --rtplan tests/test_patient_output/RTPLAN.dcm \
  --rtdose tests/test_patient_output/RTDOSE_template.dcm \
  --iso-spacing 2.5 \
  --max-iters 10 \
  --use-gamma

Notes
-----
- Requires a working CUDA build of DoseCUDA and a GPU.
- This script does NOT modify lookuptables on disk. It changes weights in-memory per iteration.
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import sys

from DoseCUDA import IMRTDoseGrid, IMRTPlan
from DoseCUDA.dvh import read_reference_rtdose
from DoseCUDA.grid_utils import GridInfo, resample_dose_linear

# Optional gamma
try:
    from DoseCUDA.gamma import GammaCriteria, compute_gamma_3d
    _GAMMA_AVAILABLE = True
except Exception:
    _GAMMA_AVAILABLE = False


def load_reference(rtdose_path: Path):
    dose_ref, origin_ref, spacing_ref, frame_uid = read_reference_rtdose(str(rtdose_path))
    return dose_ref.astype(np.float32), np.array(origin_ref, dtype=np.float32), np.array(spacing_ref, dtype=np.float32)


def objective_mask(dose_ref: np.ndarray, frac: float) -> np.ndarray:
    thr = float(dose_ref.max()) * float(frac)
    return dose_ref > thr


def rmse_mae(dose_calc: np.ndarray, dose_ref: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    diff = (dose_calc - dose_ref)[mask]
    rmse = float(np.sqrt(np.mean(diff * diff)))
    mae = float(np.mean(np.abs(diff)))
    return rmse, mae


def compute_gamma_metrics(
    dose_calc: np.ndarray,
    dose_ref: np.ndarray,
    spacing_mm: np.ndarray,
    mask: Optional[np.ndarray],
    gpu_id: int,
) -> Dict[str, float]:
    if not _GAMMA_AVAILABLE:
        raise RuntimeError("DoseCUDA.gamma not importable; install/enable gamma module or disable --use-gamma")

    c33 = GammaCriteria(dta_mm=3.0, dd_percent=3.0, local=False, dose_threshold_percent=10.0, max_gamma=2.0)
    c22 = GammaCriteria(dta_mm=2.0, dd_percent=2.0, local=False, dose_threshold_percent=10.0, max_gamma=2.0)

    g33 = compute_gamma_3d(
        dose_calc, dose_ref,
        spacing_mm=tuple(float(x) for x in spacing_mm),
        criteria=c33,
        roi_mask=mask,
        return_map=False,
        gpu_id=gpu_id,
    )
    g22 = compute_gamma_3d(
        dose_calc, dose_ref,
        spacing_mm=tuple(float(x) for x in spacing_mm),
        criteria=c22,
        roi_mask=mask,
        return_map=False,
        gpu_id=gpu_id,
    )
    # GammaResult may be a mapping-like or object with attributes
    def _val(obj, name):
        try:
            return float(getattr(obj, name))
        except Exception:
            try:
                return float(obj.get(name, np.nan))
            except Exception:
                return float(np.nan)

    return {
        "pass_3_3": _val(g33, "pass_rate"),
        "p95_3_3": _val(g33, "gamma_p95"),
        "pass_2_2": _val(g22, "pass_rate"),
        "p95_2_2": _val(g22, "gamma_p95"),
    }


def score_objective(
    rmse: float,
    mae: float,
    gamma: Optional[Dict[str, float]],
    target_pass_3_3: float,
    target_pass_2_2: float,
    w_rmse: float,
    w_mae: float,
    w_g33: float,
    w_g22: float,
) -> float:
    score = w_rmse * rmse + w_mae * mae
    if gamma is not None:
        p33 = float(gamma["pass_3_3"])
        p22 = float(gamma["pass_2_2"])
        pen33 = max(0.0, target_pass_3_3 - p33)
        pen22 = max(0.0, target_pass_2_2 - p22)
        score += w_g33 * pen33 + w_g22 * pen22
    return float(score)


def _set_kernel_weights_on_plan(plan: IMRTPlan, weights: np.ndarray, model_index: Optional[int] = None) -> np.ndarray:
    w = np.clip(weights.astype(np.float32), 0.0, None)
    if float(w.sum()) <= 0.0:
        w = np.ones_like(w, dtype=np.float32) / float(len(w))
    else:
        w = w / float(w.sum())
    w = w.astype(np.single)

    if model_index is None:
        for m in plan.beam_models:
            m.kernel_weights = w.copy()
    else:
        plan.beam_models[model_index].kernel_weights = w.copy()
    return w


def _get_current_kernel_weights(plan: IMRTPlan, model_index: Optional[int] = None) -> np.ndarray:
    m = plan.beam_models[0] if model_index is None else plan.beam_models[model_index]
    w = getattr(m, "kernel_weights", None)
    if w is None:
        w = np.ones(6, dtype=np.single) / 6.0
        _set_kernel_weights_on_plan(plan, w, model_index=model_index)
        return w
    w = np.array(w, dtype=np.float32).reshape(-1)
    if w.size != 6:
        raise ValueError(f"Expected 6 kernel weights, got {w.size}")
    w = np.clip(w, 0.0, None)
    if float(w.sum()) <= 0.0:
        w = np.ones(6, dtype=np.float32) / 6.0
    else:
        w = w / float(w.sum())
    return w.astype(np.single)


def optimize_kernel_weights(
    phase_output: Path,
    rtplan_path: Path,
    rtdose_path: Path,
    gpu_id: int = 0,
    iso_spacing: float = 2.5,
    max_iters: int = 10,
    step: float = 0.05,
    tol: float = 1e-5,
    mask_frac: float = 0.05,
    use_gamma: bool = False,
    target_pass_3_3: float = 0.98,
    target_pass_2_2: float = 0.95,
    w_rmse: float = 1.0,
    w_mae: float = 0.25,
    w_g33: float = 50.0,
    w_g22: float = 80.0,
    model_index: Optional[int] = None,
    opt_mu: bool = False,
    mu_step: float = 0.05,
) -> Tuple[np.ndarray, float]:
    phase_output = Path(phase_output)
    rtplan_path = Path(rtplan_path)
    rtdose_path = Path(rtdose_path)

    plan = IMRTPlan()
    plan.readPlanDicom(str(rtplan_path))

    ct_dir = phase_output / "CT"
    if not ct_dir.exists():
        raise FileNotFoundError(f"CT directory not found: {ct_dir}")

    dose_ref, origin_ref, spacing_ref = load_reference(rtdose_path)
    ref_grid = GridInfo(
        origin=origin_ref,
        spacing=spacing_ref,
        size=np.array(dose_ref.shape[::-1], dtype=np.int32),
        direction=np.eye(3, dtype=np.float32),
    )
    mask = objective_mask(dose_ref, mask_frac)

    # Load CT once and resample once (best-effort reuse)
    dg = IMRTDoseGrid()
    dg.loadCTDCM(str(ct_dir))
    dg.resampleCTfromSpacing(float(iso_spacing))

    def compute_dose_resampled(weights: np.ndarray):
        w_norm = _set_kernel_weights_on_plan(plan, weights, model_index=model_index)

        # best-effort reuse; fallback if compute needs clean state
        dg_local = dg
        try:
            if hasattr(dg_local, "dose") and dg_local.dose is not None:
                dg_local.dose[...] = 0.0
        except Exception:
            pass

        try:
            dg_local.computeIMRTPlan(plan, gpu_id=gpu_id)
        except Exception:
            dg_local = IMRTDoseGrid()
            dg_local.loadCTDCM(str(ct_dir))
            dg_local.resampleCTfromSpacing(float(iso_spacing))
            dg_local.computeIMRTPlan(plan, gpu_id=gpu_id)

        calc_grid = GridInfo(
            origin=np.array(dg_local.origin, dtype=np.float32),
            spacing=np.array(dg_local.spacing, dtype=np.float32),
            size=np.array(dg_local.size, dtype=np.int32),
            direction=np.eye(3, dtype=np.float32),
        )
        dose_calc = np.array(dg_local.dose, dtype=np.float32)
        dose_resampled = resample_dose_linear(dose=dose_calc, source_grid=calc_grid, target_grid=ref_grid).astype(np.float32)

        rmse, mae = rmse_mae(dose_resampled, dose_ref, mask)
        gamma_metrics = None
        if use_gamma:
            gamma_metrics = compute_gamma_metrics(dose_resampled, dose_ref, spacing_mm=spacing_ref, mask=mask, gpu_id=gpu_id)

        score = score_objective(
            rmse=rmse,
            mae=mae,
            gamma=gamma_metrics,
            target_pass_3_3=target_pass_3_3,
            target_pass_2_2=target_pass_2_2,
            w_rmse=w_rmse,
            w_mae=w_mae,
            w_g33=w_g33,
            w_g22=w_g22,
        )
        return w_norm, dose_resampled, rmse, mae, (gamma_metrics or {}), score

    # logging
    phase_output.mkdir(parents=True, exist_ok=True)
    log_jsonl = phase_output / "opt_kernel_weights_v2.jsonl"
    hist_json = phase_output / "opt_kernel_weights_v2_history.json"
    if log_jsonl.exists():
        log_jsonl.unlink()

    def log(entry: dict):
        with open(log_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    # Prepare mu optimization if requested
    resolved_model_index = model_index
    if resolved_model_index is None:
        resolved_model_index = 0

    orig_mu = None
    if opt_mu:
        orig_mu = float(plan.beam_models[resolved_model_index].mu_calibration)

    best_weights = _get_current_kernel_weights(plan, model_index=model_index)
    # pack params: [w0..w5] or [w0..w5, mu_mult]
    if opt_mu:
        best_params = np.concatenate([best_weights.astype(np.float32), np.array([1.0], dtype=np.float32)])
    else:
        best_params = best_weights.astype(np.float32)

    def _params_to_weights_and_mu(params: np.ndarray):
        w = params[:6].astype(np.float32)
        mu_mult = float(params[6]) if params.size > 6 else 1.0
        # normalize weights
        if float(w.sum()) <= 0.0:
            w = np.ones_like(w, dtype=np.float32) / 6.0
        else:
            w = w / float(w.sum())
        return w.astype(np.single), mu_mult

    best_w, best_mu_mult = _params_to_weights_and_mu(best_params)
    best_w_norm, best_dose, best_rmse, best_mae, best_gamma, best_score = compute_dose_resampled(best_params)

    log({
        "iter": 0,
        "action": "init",
        "weights": best_weights.tolist(),
        "rmse": best_rmse,
        "mae": best_mae,
        "score": best_score,
        "gamma": best_gamma,
        "cfg": {
            "iso_spacing": iso_spacing,
            "mask_frac": mask_frac,
            "use_gamma": use_gamma,
            "targets": {"pass_3_3": target_pass_3_3, "pass_2_2": target_pass_2_2},
            "objective_weights": {"w_rmse": w_rmse, "w_mae": w_mae, "w_g33": w_g33, "w_g22": w_g22},
            "gpu_id": gpu_id,
            "model_index": model_index,
        },
    })
    print(f"Init score={best_score:.6f} rmse={best_rmse:.6f} mae={best_mae:.6f} gamma={best_gamma}")

    history = [{"iter": 0, "weights": best_w_norm.tolist(), "mu_mult": (best_mu_mult if opt_mu else None), "rmse": best_rmse, "mae": best_mae, "score": best_score, "gamma": best_gamma}]
    cur_step = float(step)
    # coordinate-descent over params (6 weights + optional mu multiplier)
    n_params = 6 + (1 if opt_mu else 0)
    best_params = np.concatenate([best_w_norm, np.array([best_mu_mult], dtype=np.float32)]) if opt_mu else best_w_norm

    for it in range(1, max_iters + 1):
        improved = False

        for idx in range(n_params):
            for direction in (+1, -1):
                cand = best_params.copy()
                cand[idx] = float(cand[idx]) + float(direction) * cur_step
                # ensure positivity for mu multiplier
                if idx == 6 and opt_mu:
                    cand[idx] = max(1e-6, cand[idx])

                # normalize first 6 weights
                w = cand[:6].astype(np.float32)
                if float(w.sum()) <= 0.0:
                    continue
                w = (w / float(w.sum())).astype(np.single)
                if opt_mu:
                    cand = np.concatenate([w, np.array([float(cand[6])], dtype=np.float32)])
                else:
                    cand = w

                w_norm, dose_res, rmse, mae, gamma_m, score = compute_dose_resampled(cand)

                log_entry = {
                    "iter": it,
                    "action": "try",
                    "idx": int(idx),
                    "dir": int(direction),
                    "step": cur_step,
                    "weights": w_norm.tolist(),
                    "rmse": rmse,
                    "mae": mae,
                    "score": score,
                    "gamma": gamma_m,
                }
                if opt_mu:
                    log_entry["mu_mult"] = float(cand[6])

                log(log_entry)

                label = f"mu" if (opt_mu and idx == 6) else f"w{idx}"
                msg = f"Iter {it} try idx={idx}({label}) dir={direction:+d} step={cur_step:.5f} score={score:.6f} rmse={rmse:.5f} mae={mae:.5f}"
                if use_gamma and gamma_m:
                    msg += f" pass3/3={gamma_m.get('pass_3_3', np.nan):.4f} pass2/2={gamma_m.get('pass_2_2', np.nan):.4f}"
                print(msg)

                if score + tol < best_score:
                    best_score = score
                    best_params = cand.copy()
                    best_w_norm = w_norm
                    best_dose = dose_res
                    best_gamma = gamma_m
                    best_rmse, best_mae = rmse, mae
                    improved = True

                    np.save(phase_output / f"dose_best_iter_{it}_idx_{idx}.npy", best_dose)
                    with open(phase_output / f"kernel_weights_best_iter_{it}.json", "w", encoding="utf-8") as wf:
                        out = {"weights": best_w_norm.tolist(), "score": best_score, "rmse": best_rmse, "mae": best_mae, "gamma": best_gamma}
                        if opt_mu:
                            out["mu_mult"] = float(best_params[6])
                        json.dump(out, wf, indent=2)

                    log_accept = {"iter": it, "action": "accept", "weights": best_w_norm.tolist(), "rmse": best_rmse, "mae": best_mae, "score": best_score, "gamma": best_gamma}
                    if opt_mu:
                        log_accept["mu_mult"] = float(best_params[6])
                    log(log_accept)
                    print("  ✅ accepted")
                    break
            if improved:
                break

        history.append({"iter": it, "weights": best_w_norm.tolist(), "mu_mult": (float(best_params[6]) if opt_mu else None), "rmse": best_rmse, "mae": best_mae, "score": best_score, "gamma": best_gamma})

        if not improved:
            print(f"No improvement in iter {it}; reducing step")
            cur_step *= 0.5
            if cur_step < 1e-4:
                print("Step below threshold; stopping")
                break

    np.save(phase_output / "dose_best_final.npy", best_dose)
    with open(phase_output / "kernel_weights_best_final.json", "w", encoding="utf-8") as wf:
        json.dump({"weights": best_weights.tolist(), "score": best_score, "rmse": best_rmse, "mae": best_mae, "gamma": best_gamma}, wf, indent=2)
    with open(hist_json, "w", encoding="utf-8") as hf:
        json.dump(history, hf, indent=2)

    print(f"Best final score={best_score:.6f} rmse={best_rmse:.6f} mae={best_mae:.6f} weights={best_weights.tolist()}")
    return best_weights, best_score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase-dir", required=True, help="Phase output dir, must contain CT/ subdir")
    p.add_argument("--rtplan", required=True)
    p.add_argument("--rtdose", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--iso-spacing", type=float, default=2.5, help="Isotropic CT resample spacing in mm")
    p.add_argument("--max-iters", type=int, default=10)
    p.add_argument("--step", type=float, default=0.05)
    p.add_argument("--tol", type=float, default=1e-5)
    p.add_argument("--mask-frac", type=float, default=0.05, help="Mask voxels where ref dose > mask_frac*max(ref)")
    p.add_argument("--use-gamma", action="store_true", help="Include gamma pass-rate penalties in objective")
    p.add_argument("--target-pass-3-3", type=float, default=0.98)
    p.add_argument("--target-pass-2-2", type=float, default=0.95)

    p.add_argument("--w-rmse", type=float, default=1.0)
    p.add_argument("--w-mae", type=float, default=0.25)
    p.add_argument("--w-g33", type=float, default=50.0)
    p.add_argument("--w-g22", type=float, default=80.0)

    p.add_argument("--model-index", type=int, default=None, help="If set, only tune this beam model index (default: all)")
    p.add_argument("--target-energy", type=str, default=None, help="Dicom energy label to target (e.g. '10' -> 10MV). Resolves to model-index automatically")
    p.add_argument("--interactive", action="store_true", help="Run interactively: prompt before heavy compute and allow selecting energy/model")
    p.add_argument("--yes", action="store_true", help="Agree to run without interactive prompts (use with --interactive to skip confirm)")
    p.add_argument("--opt-mu", action="store_true", help="Also optimize a mu_calibration multiplier")
    p.add_argument("--mu-step", type=float, default=0.05, help="Step size for mu multiplier changes")
    args = p.parse_args()

    # Load RTPLAN early to allow resolving energy -> model index
    plan = IMRTPlan()
    plan.readPlanDicom(str(args.rtplan))

    resolved_model_index = args.model_index
    if args.target_energy is not None:
        # normalize labels in plan.dicom_energy_label
        available = [str(x) for x in plan.dicom_energy_label]
        if args.target_energy in available:
            resolved_model_index = available.index(args.target_energy)
        else:
            print(f"Target energy '{args.target_energy}' not found in plan model list: {available}")
            sys.exit(1)

    # Interactive selection when requested
    if args.interactive:
        print(f"Available energies -> { [str(x) for x in plan.dicom_energy_label] }")
        if resolved_model_index is None:
            sel = None
            try:
                sel = input("Enter index of model to tune (or press Enter to tune all): ")
            except Exception:
                sel = ''
            if sel.strip() != '':
                try:
                    resolved_model_index = int(sel)
                except ValueError:
                    print("Invalid selection; exiting")
                    sys.exit(1)

        if not args.yes:
            conf = None
            try:
                conf = input(f"Ready to run optimizer (phase_dir={args.phase_dir}) on model_index={resolved_model_index}. Proceed? [y/N]: ")
            except Exception:
                conf = ''
            if conf.lower().strip() not in ('y', 'yes'):
                print("Aborting.")
                sys.exit(0)

    # Call optimizer with resolved model index
    opt_mu_flag = bool(args.opt_mu)
    mu_step_val = float(args.mu_step)

    optimize_kernel_weights(
        phase_output=Path(args.phase_dir),
        rtplan_path=Path(args.rtplan),
        rtdose_path=Path(args.rtdose),
        gpu_id=args.gpu,
        iso_spacing=args.iso_spacing,
        max_iters=args.max_iters,
        step=args.step,
        tol=args.tol,
        mask_frac=args.mask_frac,
        use_gamma=bool(args.use_gamma),
        target_pass_3_3=args.target_pass_3_3,
        target_pass_2_2=args.target_pass_2_2,
        w_rmse=args.w_rmse,
        w_mae=args.w_mae,
        w_g33=args.w_g33,
        w_g22=args.w_g22,
        model_index=resolved_model_index,
        opt_mu=opt_mu_flag,
        mu_step=mu_step_val,
    )


if __name__ == "__main__":
    main()
