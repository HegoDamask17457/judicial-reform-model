"""Solve one regime (N, eta) of the model to its stationary inter-year value function.

Runs a full inter-year value iteration to V*, then a final within-year solve at V*,
and saves the value function and policies on the backlog grid.

Output: results/regime_{N}_{eta}.npz
    B_grid, V_star, W_value, d_policy, em_policy, ej_policy, rho_policy,
    resid_hist, ratio_hist, vi_time, n_iter, final_resid, converged, ...

Usage: python code/run_regime.py N eta      e.g. "100 5" (pre-reform) or "150 8"
Calibrated parameters are read from code/calibration.json.
"""
import os
import sys
import json
import time
import argparse
import numpy as np

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)

from solver.model import Params, w_tau
from solver.interyear_vi import run_vi
from solver.dp_solver_v2 import within_year_solve

OUT_DIR = os.path.join(os.path.dirname(CODE_DIR), "results")
CKPT_ROOT = os.path.join(OUT_DIR, "checkpoints")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("N", type=float)
    ap.add_argument("eta", type=float)
    ap.add_argument("--n_d", type=int, default=51)
    ap.add_argument("--within_beta", type=float, default=0.95)
    ap.add_argument("--max_iter", type=int, default=120)
    ap.add_argument("--n_workers", type=int, default=16)
    args = ap.parse_args()

    with open(os.path.join(CODE_DIR, "calibration.json")) as f:
        cp = json.load(f)
    P = Params(share_reversal=True, s_floor=cp["s_floor"],
               alpha2=cp["alpha2"], r0=cp["r0"], gamma_m=cp["gamma_m"],
               gamma_j=cp["gamma_j"], gamma_mj=cp["gamma_mj"],
               N=args.N, eta=args.eta)
    label = f"{int(args.N)}_{int(args.eta)}"
    ckpt_dir = os.path.join(CKPT_ROOT, label)
    os.makedirs(ckpt_dir, exist_ok=True)
    B_grid = np.linspace(0.0, P.B_max, 51)

    print(f"[run_regime] regime (N={P.N}, eta={P.eta})  "
          f"w_tau={w_tau(P):.6f} (alpha1/(T*N))", flush=True)

    t0 = time.time()
    V_star, resid_hist, ratio_hist, summary = run_vi(
        P, B_grid=B_grid, n_d=args.n_d, within_beta=args.within_beta,
        max_iter_total=args.max_iter, tol=None, checkpoint_every=5,
        checkpoint_dir=ckpt_dir, verbose=True)
    vi_time = time.time() - t0

    res = within_year_solve(P, B_grid=B_grid, n_d=args.n_d, variant="V3",
                            V_next=V_star, bug_mode=False,
                            within_beta=args.within_beta,
                            n_workers=args.n_workers)

    out_path = os.path.join(OUT_DIR, f"regime_{label}.npz")
    np.savez(out_path, B_grid=B_grid, V_star=V_star, W_value=res.W,
             d_policy=res.d_pol, em_policy=res.em_pol, ej_policy=res.ej_pol,
             rho_policy=res.rho_pol, resid_hist=resid_hist,
             ratio_hist=ratio_hist, vi_time=vi_time,
             n_iter=summary["n_iter"], final_resid=summary["final_resid"],
             converged=int(summary["converged"]),
             mean_iter_time=summary["mean_iter_time"],
             s_floor=P.s_floor, alpha2=P.alpha2,
             gamma_m=P.gamma_m, gamma_j=P.gamma_j, gamma_mj=P.gamma_mj,
             N=P.N, eta=P.eta, w_tau=w_tau(P), xcheck=str(res.xcheck))
    print(f"DONE {label}: VI {summary['n_iter']} iters in {vi_time:.0f}s, "
          f"final resid {summary['final_resid']:.2e}, "
          f"converged={summary['converged']} -> {os.path.relpath(out_path, os.path.dirname(CODE_DIR))}", flush=True)
    print(f"ej mean={res.ej_pol.mean():.4f} d mean={res.d_pol.mean():.4f} "
          f"rho mean={res.rho_pol.mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
