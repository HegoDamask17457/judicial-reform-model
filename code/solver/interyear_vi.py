"""Inter-year value iteration (numerical check of the Blackwell-Banach contraction).

V^{(k+1)}(B) = W_1^{(k)}(B), where the within-year DP (V3) uses the
terminal continuation  -p*B'^2 + beta*V^{(k)}(B')  at tau=T and
beta*W_{tau+1}(B') inside the year (within_beta = 0.95).

Stopping: sup-norm Bellman residual < 1e-8*(1-beta) OR max_iter_total.
The inter-year operator has Lipschitz modulus beta^T = 0.95^4 ~ 0.8145
(within-year beta discounts each of the T subperiods), so reaching
5e-10 needs ~100 iterations; we also report the empirical contraction
ratio per iteration to verify the Banach-contraction claim numerically.

Checkpoints: results/checkpoints/vi_{k:04d}.npz every few iters.
"""
import os
import time
import numpy as np

try:
    from .model import Params
    from .dp_solver_v2 import within_year_solve
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from solver.model import Params
    from solver.dp_solver_v2 import within_year_solve

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                              "results", "checkpoints")


def run_vi(P: Params, B_grid=None, n_d=51, within_beta=0.95,
           max_iter_total=120, tol=None, checkpoint_every=5,
           checkpoint_dir=CHECKPOINT_DIR, verbose=True):
    if B_grid is None:
        B_grid = np.linspace(0.0, P.B_max, 51)
    if tol is None:
        tol = 1e-8 * (1.0 - P.beta)
    os.makedirs(checkpoint_dir, exist_ok=True)

    V = np.zeros(len(B_grid))
    resid_hist = []
    ratio_hist = []
    iter_times = []
    warm0 = None
    V_star = V.copy()

    t_start = time.time()
    for k in range(max_iter_total):
        t0 = time.time()
        res = within_year_solve(P, B_grid=B_grid, n_d=n_d, variant="V3",
                                V_next=V, bug_mode=False,
                                within_beta=within_beta, warm0=warm0,
                                n_workers=16)
        Vnew = res.V1()
        resid = float(np.max(np.abs(Vnew - V)))
        resid_hist.append(resid)
        iter_times.append(time.time() - t0)
        if k > 0:
            ratio_hist.append(resid / max(resid_hist[k - 1], 1e-300))
        warm0 = {(t, i): (res.em_pol[t, i], res.ej_pol[t, i])
                 for t in range(P.T) for i in range(len(B_grid))}
        V = Vnew
        V_star = V.copy()
        if verbose:
            ratio_s = f"{ratio_hist[-1]:.4f}" if ratio_hist else "  -  "
            print(f"iter {k+1:3d}: resid={resid:.3e} "
                  f"ratio={ratio_s} "
                  f"W1[0]={V[0]:.6f} W1[-1]={V[-1]:.6f} "
                  f"({time.time()-t0:.1f}s)", flush=True)
        if (k + 1) % checkpoint_every == 0:
            np.savez(os.path.join(checkpoint_dir, f"vi_{k+1:04d}.npz"),
                     V=V, resid=resid, iteration=k + 1,
                     d_policy=res.d_pol, em_policy=res.em_pol,
                     ej_policy=res.ej_pol, W=res.W)
        if resid < tol:
            if verbose:
                print(f"CONVERGED at iter {k+1}: resid {resid:.3e} < {tol:.3e}")
            break

    total = time.time() - t_start
    summary = {
        "n_iter": len(resid_hist),
        "final_resid": resid_hist[-1],
        "tol": tol,
        "max_iter": max_iter_total,
        "converged": resid_hist[-1] < tol,
        "mean_ratio": float(np.mean(ratio_hist[-20:])) if ratio_hist else np.nan,
        "mean_iter_time": float(np.mean(iter_times)),
        "total_time": total,
        "V0": V[0], "Vmid": V[len(B_grid) // 2], "Vend": V[-1],
    }
    np.savez(os.path.join(checkpoint_dir, "vi_final.npz"),
             V=V, resid_hist=np.array(resid_hist),
             ratio_hist=np.array(ratio_hist), summary=summary)
    if verbose:
        print("SUMMARY:", summary)
    return V, np.array(resid_hist), np.array(ratio_hist), summary


if __name__ == "__main__":
    P = Params()
    run_vi(P)
