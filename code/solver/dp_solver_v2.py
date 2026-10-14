"""Within-year backward-induction DP solver (independent reimplementation).

Three outer variants:
  V1 continuous : outer d in [0,1] via scipy minimize_scalar(bounded),
                  inner solve via fast Newton (KKT reference in checks).
  V2 joint      : 3-D SLSQP over (d, e_m, e_j), bounds + em+ej<=1.
  V3 grid       : d-grid search (n_d points) with inner fast solve.
                  (mirrors the old architecture; used by grid-convergence study)

Continuation:
  tau<T : beta * W_{tau+1}(B')  (linear interp on B-grid, clamped; counted)
  tau=T : -p*B'^2 + beta*V_next(B')   (V_next from inter-year VI; 0 if None)

Bug-mode (replicates old solver's known defects, for the reproduction gate):
  tau<T : inner (em,ej) solve DROPS the continuation term; the continuation
          enters only the outer d-objective.
  tau=T : inner includes -p*B'^2 only (no beta*V); outer adds nothing.

The inner solve is the warm-started damped Newton path (static_solver.
solve_inner_fast), which falls back to the full KKT active-set enumerator;
the full KKT+SLSQP cross-check runs on a strided subsample of nodes
(see solve_inner / xcheck counters) and in the static/self-check gates.
"""
import numpy as np
from scipy.optimize import minimize_scalar, minimize
from multiprocessing import Pool

from .model import (Params, U, Bnext, phi, psi, g,
                    WContinuation, TermContinuation)
from .static_solver import solve_inner_fast, solve_inner, xcheck_stats

_MAX_WORKERS = 16


def _make_d_grid(P, n_d):
    return np.linspace(0.0, 1.0, n_d)


class WithinYearResult:
    def __init__(self, P, B_grid, W, d_pol, em_pol, ej_pol, rho_pol,
                 n_clamped, n_evals, variant, bug_mode, xcheck=None):
        self.P = P
        self.B_grid = B_grid
        self.W = W                    # (T, n_B)
        self.d_pol = d_pol            # (T, n_B)
        self.em_pol = em_pol
        self.ej_pol = ej_pol
        self.rho_pol = rho_pol        # phi(em)/(phi(em)+psi(ej))
        self.n_clamped = n_clamped
        self.n_evals = n_evals
        self.variant = variant
        self.bug_mode = bug_mode
        self.xcheck = xcheck

    def V1(self):
        """Year-start value function W_1(B)."""
        return self.W[0].copy()

    def save(self, path):
        np.savez(path, B_grid=self.B_grid, W=self.W, d_policy=self.d_pol,
                 em_policy=self.em_pol, ej_policy=self.ej_pol,
                 rho_policy=self.rho_pol, variant=self.variant,
                 bug_mode=int(self.bug_mode), n_clamped=self.n_clamped,
                 n_evals=self.n_evals)


def _cold_start(d, B, P, cont):
    """Robust cold start for the fast Newton path: 1-D Gauss-Seidel sweep
    (two brentq passes) then refine.  Falls back to (0.3, 0.3)."""
    from scipy.optimize import brentq
    from .static_solver import F_grad

    def fm(em, ej):
        return F_grad(em, ej, d, B, P, cont)[0]

    def fj(em, ej):
        return F_grad(em, ej, d, B, P, cont)[1]

    em, ej = 0.3, 0.3
    try:
        # pass 1: solve Fm(., ej)=0 then Fj(em, .)=0
        em = brentq(lambda x: fm(x, ej), 1e-10, 1.0 - ej)
        ej = brentq(lambda x: fj(em, x), 1e-10, 1.0 - em)
        # pass 2: one more alternating sweep
        em = brentq(lambda x: fm(x, ej), 1e-10, 1.0 - ej)
        ej = brentq(lambda x: fj(em, x), 1e-10, 1.0 - em)
    except ValueError:
        return (0.3, 0.3)
    if not (np.isfinite(em) and np.isfinite(ej)):
        return (0.3, 0.3)
    return (float(em), float(ej))


def _v3_node_worker(args):
    """Worker: solve one (tau-pass, B node) via d-grid + warm-started Newton.
    Returns (best em,ej,d,obj, n_fallback, n_clamped)."""
    (B, d_grid, P, inner_cont, outer_extra, is_last,
     warm0) = args
    warm = warm0 if warm0 is not None else None
    best = (None, None, None, -np.inf)
    n_fallback = 0
    n_clamp = 0
    for d in d_grid:
        if warm is None:
            warm = _cold_start(d, B, P, inner_cont)
        em, ej, fval, conv = solve_inner_fast(d, B, P, inner_cont, warm=warm)
        if not conv:
            n_fallback += 1
        if em is None:
            continue
        obj = -0.5 * P.delta * d * d + fval
        if outer_extra is not None:
            bp = Bnext(em, ej, d, B, P)
            obj += outer_extra(bp)
            n_clamp += outer_extra.n_clamped(bp)
        elif inner_cont is not None:
            n_clamp += inner_cont.n_clamped(Bnext(em, ej, d, B, P))
        if obj > best[3]:
            best = (em, ej, d, obj)
        warm = (em, ej)
    return best[0], best[1], best[2], best[3], n_fallback, n_clamp


def within_year_solve(P: Params, B_grid=None, n_d=51, variant="V3",
                      V_next=None, bug_mode=False, within_beta=1.0,
                      inner_method="fast", verbose=False, warm0=None,
                      n_workers=_MAX_WORKERS, xcheck_nodes=None):
    """Backward induction over tau = T..1.

    V_next : array (n_B,) inter-year value function for the terminal
             continuation (None -> no V term).
    within_beta : discount applied to the tau<T continuation W_{tau+1}
             (spec has no within-year discount; the old solver's cache is
             reproduced with within_beta=0.95, so that is the baseline).
    warm0 : dict {(tau,B_index): (em,ej,d)} warm starts from a previous
             within-year solve (used to accelerate VI iterations).
    xcheck_nodes : list of (tau, B_index, d_index) nodes where the full
             KKT+SLSQP cross-check (solve_inner 'both') is run.
    """
    if B_grid is None:
        B_grid = np.linspace(0.0, P.B_max, 51)
    B_grid = np.asarray(B_grid)
    n_B = len(B_grid)
    d_grid = _make_d_grid(P, n_d)
    T = P.T

    W = np.zeros((T, n_B))
    d_pol = np.zeros((T, n_B))
    em_pol = np.zeros((T, n_B))
    ej_pol = np.zeros((T, n_B))
    rho_pol = np.zeros((T, n_B))
    n_clamped = 0
    n_evals = 0
    n_fallback = 0
    xcheck_miss = 0
    xcheck_maxdf = 0.0
    xcheck_maxde = 0.0

    for tau in range(T - 1, -1, -1):
        is_last = (tau == T - 1)
        if is_last:
            inner_cont = TermContinuation(P, B_grid, V_next, within_beta)
            outer_extra = None
        else:
            if bug_mode:
                inner_cont = None
                outer_extra = WContinuation(B_grid, W[tau + 1], within_beta,
                                            label=f"W_{tau+2}")
            else:
                inner_cont = WContinuation(B_grid, W[tau + 1], within_beta,
                                           label=f"W_{tau+2}")
                outer_extra = None

        # strided full cross-check nodes for this tau
        if xcheck_nodes is not None and tau in xcheck_nodes:
            for bi, d_idx in xcheck_nodes[tau]:
                Bv = B_grid[bi]
                dv = d_grid[d_idx]
                em_fast, ej_fast, f_fast, _ = solve_inner_fast(
                    dv, Bv, P, inner_cont)
                em, ej, f1 = solve_inner(dv, Bv, P, cont=inner_cont,
                                         method="both")
                if em_fast is not None and em is not None:
                    df = abs(f1 - f_fast)
                    de = abs(em_fast - em) + abs(ej_fast - ej)
                    if de > 1e-5:
                        xcheck_miss += 1
                    xcheck_maxdf = max(xcheck_maxdf, df)
                    xcheck_maxde = max(xcheck_maxde, de)

        tasks = [(B_grid[i], d_grid, P, inner_cont, outer_extra, is_last,
                  None if warm0 is None else warm0.get((tau, i)))
                 for i in range(n_B)]
        if n_workers > 1 and n_B >= 4:
            with Pool(n_workers) as pool:
                results = pool.map(_v3_node_worker, tasks)
        else:
            results = [_v3_node_worker(t) for t in tasks]
        for i, (em, ej, d, obj, nfb, ncl) in enumerate(results):
            W[tau, i] = obj
            d_pol[tau, i] = d
            em_pol[tau, i] = em
            ej_pol[tau, i] = ej
            ph, ps = phi(em, P), psi(ej, P)
            rho_pol[tau, i] = ph / (ph + ps) if (ph + ps) > 0 else 0.0
            n_evals += 1
            n_fallback += nfb
            n_clamped += ncl
        if verbose:
            print(f"tau={tau+1} done: W[tau,0]={W[tau,0]:.6f} "
                  f"W[tau,-1]={W[tau,-1]:.6f} fallbacks={n_fallback}")

    xcheck = {"n_miss": xcheck_miss, "max_df": xcheck_maxdf,
              "max_de": xcheck_maxde, "n_fallback": n_fallback}
    return WithinYearResult(P, B_grid, W, d_pol, em_pol, ej_pol, rho_pol,
                            n_clamped, n_evals, variant, bug_mode, xcheck)


def _outer_obj(d, B, P, inner_cont, outer_extra, is_last, warm):
    em, ej, fval, _ = solve_inner_fast(d, B, P, inner_cont, warm=warm)
    if em is None:
        em, ej, fval = solve_inner(d, B, P, cont=inner_cont)
    obj = -0.5 * P.delta * d * d + fval
    if outer_extra is not None:
        obj += outer_extra(Bnext(em, ej, d, B, P))
    return obj, (em, ej)


def _solve_v1(B, P, inner_cont, outer_extra, is_last, warm0):
    d_grid = _make_d_grid(P, 11)
    best = None
    warm = warm0
    for d in d_grid:
        obj, e = _outer_obj(d, B, P, inner_cont, outer_extra, is_last, warm)
        warm = e
        if best is None or obj > best[3]:
            best = (e[0], e[1], d, obj)

    def neg(d):
        return -_outer_obj(d, B, P, inner_cont, outer_extra, is_last,
                           best[0:2])[0]

    res = minimize_scalar(neg, bounds=(0.0, 1.0), method="bounded",
                          options={"xatol": 1e-10, "maxiter": 200})
    d = float(np.clip(res.x, 0.0, 1.0))
    obj, e = _outer_obj(d, B, P, inner_cont, outer_extra, is_last, best[0:2])
    return e[0], e[1], d, obj


def _solve_v2(B, P, inner_cont, outer_extra, is_last, warm0):
    """Joint 3-D SLSQP over (d, em, ej), warm-started, then polished by
    alternating exact-inner solves and a bounded 1-D refine in d
    (the SLSQP result alone carries ~5e-4 tolerance on the ill-conditioned
    kappa=0 objective; the polish brings it to the V1 level)."""
    em0, ej0, d0, obj0 = warm0

    def neg(x):
        d, em, ej = x
        v = U(em, ej, d, B, P) - 0.5 * P.delta * d * d
        if inner_cont is not None:
            v += inner_cont(Bnext(em, ej, d, B, P))
        if outer_extra is not None:
            v += outer_extra(Bnext(em, ej, d, B, P))
        return -v

    starts = [(d0, em0, ej0), (0.1, 0.3, 0.3), (0.3, 0.1, 0.3), (0.5, 0.2, 0.2)]
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    cons = [{"type": "ineq", "fun": lambda x: 1.0 - x[1] - x[2]}]
    best = (em0, ej0, d0, obj0)
    for s0 in starts:
        try:
            res = minimize(neg, s0, method="SLSQP", bounds=bounds,
                           constraints=cons,
                           options={"ftol": 1e-12, "maxiter": 400})
        except Exception:
            continue
        if res.x is None:
            continue
        d, em, ej = res.x
        if not (0 <= d <= 1 and 0 <= em and 0 <= ej and em + ej <= 1 + 1e-10):
            continue
        obj = -neg(res.x)
        if obj > best[3]:
            best = (em, ej, d, obj)
    # polish: alternate exact inner solve at current d, then 1-D refine in d
    from scipy.optimize import minimize_scalar
    d, em, ej = best[2], best[0], best[1]
    for _ in range(3):
        em, ej, fv = solve_inner(d, B, P, cont=inner_cont, method="kkt")
        if em is None:
            break

        def neg_d(dd):
            e1, e2, f2 = solve_inner(dd, B, P, cont=inner_cont, method="kkt")
            if e1 is None:
                return 1e10
            o = -0.5 * P.delta * dd * dd + f2
            if outer_extra is not None:
                o += outer_extra(Bnext(e1, e2, dd, B, P))
            return -o

        r = minimize_scalar(neg_d, bounds=(0.0, 1.0), method="bounded",
                            options={"xatol": 1e-10})
        d = float(np.clip(r.x, 0.0, 1.0))
    em, ej, fv = solve_inner(d, B, P, cont=inner_cont, method="kkt")
    if em is None:
        return best
    obj = -0.5 * P.delta * d * d + fv
    if outer_extra is not None:
        obj += outer_extra(Bnext(em, ej, d, B, P))
    if obj > best[3]:
        return (em, ej, d, obj)
    return best


def within_year_solve_v12(P: Params, B_grid=None, n_d=51, V_next=None,
                          bug_mode=False, within_beta=1.0, variants=("V1", "V2"),
                          verbose=False):
    """V1/V2 variants (serial; each node warm-started from the V3 solution
    at that node, which is solved first).  Returns dict of results."""
    if B_grid is None:
        B_grid = np.linspace(0.0, P.B_max, 51)
    n_B = len(B_grid)
    d_grid = _make_d_grid(P, n_d)
    T = P.T

    v3 = within_year_solve(P, B_grid=B_grid, n_d=n_d, variant="V3",
                           V_next=V_next, bug_mode=bug_mode,
                           within_beta=within_beta, n_workers=_MAX_WORKERS)
    out = {"V3": v3}
    for variant in variants:
        W = np.zeros((T, n_B))
        d_pol = np.zeros((T, n_B))
        em_pol = np.zeros((T, n_B))
        ej_pol = np.zeros((T, n_B))
        rho_pol = np.zeros((T, n_B))
        for tau in range(T - 1, -1, -1):
            is_last = (tau == T - 1)
            if is_last:
                inner_cont = TermContinuation(P, B_grid, V_next, within_beta)
                outer_extra = None
            else:
                if bug_mode:
                    inner_cont = None
                    outer_extra = WContinuation(B_grid, W[tau + 1], within_beta)
                else:
                    inner_cont = WContinuation(B_grid, W[tau + 1], within_beta)
                    outer_extra = None
            for i in range(n_B):
                B = B_grid[i]
                warm0 = (v3.em_pol[tau, i], v3.ej_pol[tau, i],
                         v3.d_pol[tau, i], v3.W[tau, i])
                if variant == "V1":
                    em, ej, d, obj = _solve_v1(B, P, inner_cont, outer_extra,
                                               is_last, warm0[0:2])
                else:
                    em, ej, d, obj = _solve_v2(B, P, inner_cont, outer_extra,
                                               is_last, warm0)
                W[tau, i] = obj
                d_pol[tau, i] = d
                em_pol[tau, i] = em
                ej_pol[tau, i] = ej
                ph, ps = phi(em, P), psi(ej, P)
                rho_pol[tau, i] = ph / (ph + ps) if (ph + ps) > 0 else 0.0
        out[variant] = WithinYearResult(P, B_grid, W, d_pol, em_pol, ej_pol,
                                        rho_pol, 0, 0, variant, bug_mode)
    return out
