"""Inner (e_m, e_j) solver — the Phase-1 static subproblem.

max_{em,ej}  F(em,ej) = U(em,ej; d,B) + cont(B'(em,ej,d,B))
s.t. em,ej >= 0, em+ej <= 1

Two independent methods (both implemented, cross-checked):
  (a) SLSQP from 5 multi-starts + coarse-grid fallback (scipy.optimize)
  (b) KKT active-set enumerator: 5 candidate cases solved as small
      equation systems with scipy brentq/root, global max selected.
"""
import numpy as np
from scipy.optimize import minimize, brentq

from .model import (Params, U, Bnext, Um, Uj, phi, psi, g, r, dg, dphi, dpsi,
                    w_tau, Continuation, share, _share_derivs, ccost)


def F_eval(em, ej, d, B, P, cont):
    """Objective value (NOT negated)."""
    v = U(em, ej, d, B, P)
    if cont is not None:
        v += cont(Bnext(em, ej, d, B, P))
    return v


def F_grad(em, ej, d, B, P, cont):
    """Analytic gradient (dF/dem, dF/dej)."""
    gm, gj = Um(em, ej, d, B, P), Uj(em, ej, d, B, P)
    if cont is not None:
        bp = Bnext(em, ej, d, B, P)
        c1 = cont.deriv(bp)
        # dB'/dem = -(1-g)*phi'*N ; dB'/dej = -(1-g)*psi'*N
        gm += c1 * (-(1 - g(d, P)) * dphi(em, P) * P.N)
        gj += c1 * (-(1 - g(d, P)) * dpsi(ej, P) * P.N)
    return gm, gj


def F_hess(em, ej, d, B, P, cont):
    """Analytic Hessian of F in (em, ej).  Valid for alpha3 = 0 (baseline).

    With share_reversal=True the Term2 share channel contributes
      d2T2/dem2  = -a2*r*(1-g)*s_mm
      d2T2/dmdej = -a2*(1-g)*(r'*s_m + r*s_mj)
      d2T2/dej2  = -a2*(1-g)*(r''*s + 2*r'*s_j + r*s_jj)
    (s and its derivatives from _share_derivs; r'' = r1^2*r > 0).
    In share mode the baseline reversal term -a2*r*(1-g) is REPLACED by
    Term2_new, so the share branch uses U_rest second derivatives (correct
    r'' sign).  NOTE (audit finding, baseline branch): the audited
    baseline term -a2*(-r1**2*r)*onemg evaluates to +a2*r1**2*r*(1-g),
    whereas the true second derivative of -a2*r*(1-g) is negative; kept
    verbatim so share_reversal=False stays bit-identical (KKT/SLSQP
    paths are Hessian-free; the Newton path falls back when unreliable).
    """
    gg = g(d, P)
    onemg = 1 - gg
    # U second derivatives.  Baseline U carries the old reversal term
    # -a2*r*(1-g); the share model REPLACES it with Term2_new =
    # -a2*r*(1-g)*s, so in share mode the U_rest second derivatives are
    # taken and the Term2_new curvature terms added (correct r'' sign).
    if P.gamma_m is None:
        cmm, cmj, cjj = P.gamma + P.kappa, P.gamma, P.gamma
    else:
        cmm, cmj, cjj = P.gamma_m, P.gamma_mj, P.gamma_j
    uemem = w_tau(P) * _phi2(em, P) * onemg * P.N - cmm
    uejeej = w_tau(P) * _psi2(ej, P) * onemg * P.N - cjj
    uemej = -cmj
    if P.share_reversal:
        rv = r(ej, P)
        drv = -P.r1 * rv
        s, sm, sj, smm, sjj, smj = _share_derivs(em, ej, P)
        uemem += -P.alpha2 * rv * onemg * smm
        uemej += -P.alpha2 * onemg * (drv * sm + rv * smj)
        uejeej += -P.alpha2 * onemg * (P.r1 ** 2 * rv * s
                                       + 2 * drv * sj + rv * sjj)
    else:
        # audited baseline term (kept verbatim for bit-identical False path)
        uejeej += -P.alpha2 * (-P.r1 ** 2 * r(ej, P)) * onemg
    if cont is not None:
        bp = Bnext(em, ej, d, B, P)
        c1 = cont.deriv(bp)
        c2 = cont.deriv2(bp)
        be = -(1 - gg) * dphi(em, P) * P.N
        bj = -(1 - gg) * dpsi(ej, P) * P.N
        bee = -(1 - gg) * _phi2(em, P) * P.N
        bjj = -(1 - gg) * _psi2(ej, P) * P.N
        uemem += c2 * be * be + c1 * bee
        uejeej += c2 * bj * bj + c1 * bjj
        uemej += c2 * be * bj
    return np.array([[uemem, uemej], [uemej, uejeej]])


def _phi2(em, P):
    if em <= 0:
        return -np.inf
    return P.phi0 * P.phi1 * (P.phi1 - 1) * em ** (P.phi1 - 2)


def _psi2(ej, P):
    if ej <= 0:
        return -np.inf
    return P.psi0 * P.psi1 * (P.psi1 - 1) * ej ** (P.psi1 - 2)


def _fd_hess(em, ej, d, B, P, cont):
    """Central finite-difference Hessian of F (9-point stencil on F_eval).

    Fallback path for share_reversal=True when the analytic Hessian is
    unreliable (non-finite or Newton stalls)."""
    h = 1e-6
    if em > 0 and ej > 0:
        h = min(h, 0.5 * em, 0.5 * ej)
    f = np.empty((3, 3))
    for i, e1 in enumerate((em - h, em, em + h)):
        for j, e2 in enumerate((ej - h, ej, ej + h)):
            f[i, j] = F_eval(max(e1, 0.0), max(e2, 0.0), d, B, P, cont)
    H = np.array([
        [(f[0, 1] - 2 * f[1, 1] + f[2, 1]) / h ** 2,
         (f[2, 2] - f[2, 0] - f[0, 2] + f[0, 0]) / (4 * h * h)],
        [(f[2, 2] - f[2, 0] - f[0, 2] + f[0, 0]) / (4 * h * h),
         (f[1, 0] - 2 * f[1, 1] + f[1, 2]) / h ** 2],
    ])
    return H


def _sf_prims(em, ej, d, P):
    """Share-mode primitives computed ONCE (fast path): returns
    (gg, onemg, ph, ps, dp, dq, d2p, d2q, rv, drv, wt, s, sm, sj, smm,
    sjj, smj).  Floor applied via P.s_floor (derivatives 0 below it)."""
    gg = g(d, P)
    onemg = 1 - gg
    ph = phi(em, P)
    ps = psi(ej, P)
    dp = dphi(em, P)
    dq = dpsi(ej, P)
    d2p = _phi2(em, P)
    d2q = _psi2(ej, P)
    rv = r(ej, P)
    drv = -P.r1 * rv
    wt = w_tau(P)
    den = ph + ps
    if den > 0:
        s = ps / den
        if P.s_floor > 0 and s < P.s_floor:
            s, sm, sj, smm, sjj, smj = P.s_floor, 0.0, 0.0, 0.0, 0.0, 0.0
        else:
            den2, den3 = den * den, den * den * den
            sm = -ps * dp / den2
            sj = ph * dq / den2
            smm = -ps * d2p / den2 + 2 * ps * dp * dp / den3
            sjj = ph * d2q / den2 - 2 * ph * dq * dq / den3
            smj = dp * dq * (ps - ph) / den3
    else:
        s, sm, sj, smm, sjj, smj = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    return (gg, onemg, ph, ps, dp, dq, d2p, d2q, rv, drv, wt,
            s, sm, sj, smm, sjj, smj)


def _sf_grad(em, ej, d, B, P, cont):
    """Share-mode F_grad with primitives computed once (fast Newton path).
    Identical math to F_grad (verified numerically)."""
    (_, onemg, _, _, dp, dq, _, _, rv, drv, wt,
     s, sm, sj, _, _, _) = _sf_prims(em, ej, d, P)
    if P.gamma_m is None:
        dcm, dcj = P.gamma * (em + ej) + P.kappa * em, P.gamma * (em + ej)
    else:
        dcm = P.gamma_m * em + P.gamma_mj * ej
        dcj = P.gamma_j * ej + P.gamma_mj * em
    gm = (wt * dp * onemg * P.N - dcm
          - P.alpha2 * rv * onemg * sm)
    if np.isfinite(dq):
        gj = (wt * dq * onemg * P.N - dcj
              - P.alpha2 * onemg * (drv * s + rv * sj))
    else:
        gj = np.nan   # dq = +inf at ej=0 exactly; only gm is used there
    if cont is not None:
        bp = Bnext(em, ej, d, B, P)
        c1 = cont.deriv(bp)
        gm += c1 * (-onemg * dp * P.N)
        if np.isfinite(dq):
            gj += c1 * (-onemg * dq * P.N)
    return gm, gj


def _sf_hess(em, ej, d, B, P, cont):
    """Share-mode F_hess with primitives computed once (fast Newton path).
    Identical math to F_hess(share branch) (verified numerically)."""
    (_, onemg, _, _, dp, dq, d2p, d2q, rv, drv, wt,
     s, sm, sj, smm, sjj, smj) = _sf_prims(em, ej, d, P)
    if P.gamma_m is None:
        cmm, cmj, cjj = P.gamma + P.kappa, P.gamma, P.gamma
    else:
        cmm, cmj, cjj = P.gamma_m, P.gamma_mj, P.gamma_j
    uemem = (wt * d2p * onemg * P.N - cmm
             - P.alpha2 * rv * onemg * smm)
    uemej = -cmj - P.alpha2 * onemg * (drv * sm + rv * smj)
    uejeej = (wt * d2q * onemg * P.N - cjj
              - P.alpha2 * onemg * (P.r1 ** 2 * rv * s
                                    + 2 * drv * sj + rv * sjj))
    if cont is not None:
        bp = Bnext(em, ej, d, B, P)
        c1 = cont.deriv(bp)
        c2 = cont.deriv2(bp)
        be = -onemg * dp * P.N
        bj = -onemg * dq * P.N
        bee = -onemg * d2p * P.N
        bjj = -onemg * d2q * P.N
        uemem += c2 * be * be + c1 * bee
        uejeej += c2 * bj * bj + c1 * bjj
        uemej += c2 * be * bj
    return np.array([[uemem, uemej], [uemej, uejeej]])


def _sf_eval(em, ej, d, B, P, cont):
    """Share-mode objective value F(em, ej) (primitives once)."""
    (gg, onemg, ph, ps, _, _, _, _, rv, _, wt,
     s, _, _, _, _, _) = _sf_prims(em, ej, d, P)
    out = (wt * ((ph + ps) * onemg * P.N + P.omega * B)
           - P.alpha2 * rv * onemg * s
           - ccost(em, ej, P))
    if cont is not None:
        out += cont(Bnext(em, ej, d, B, P))
    return out


def _sf_corner(d, B, P, cont):
    """Cheap corner candidates for the share path: the e_j=0 corner root
    (brentq on Fm(.,0)=0) plus the three vertices (0,0),(1,0),(0,1).
    The e_m=0 corner is never optimal (Fm(0+,.) = +inf since
    phi'(0+)=+inf), so it is skipped.
    Returns (best_candidate_or_None, emc_or_None, corner_KKT_valid):
    corner_KKT_valid = (Fj(emc, 0+) <= 0).  If the corner root exists but
    is NOT KKT-valid, the near-corner interior max must be searched
    (_sf_nearcorner)."""
    from scipy.optimize import brentq
    best = None
    for (cem, cej) in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)):
        cv = _sf_eval(cem, cej, d, B, P, cont)
        if best is None or cv > best[2]:
            best = (float(cem), float(cej), float(cv))
    emc = None
    valid = False
    try:
        emc = float(brentq(lambda x: _sf_grad(x, 0.0, d, B, P, cont)[0],
                           1e-10, 1.0 - 1e-10))
        fj = _sf_grad(emc, 1e-10, d, B, P, cont)[1]
        if np.isfinite(fj) and fj <= 1e-10:
            valid = True
            cv = _sf_eval(emc, 0.0, d, B, P, cont)
            if cv > best[2]:
                best = (emc, 0.0, float(cv))
    except ValueError:
        pass
    return best, emc, valid


def _sf_polish(em, ej, d, B, P, cont, iters=15):
    """Damped Newton polish to a nearby FOC root using fast primitives."""
    for _ in range(iters):
        fm, fj = _sf_grad(em, ej, d, B, P, cont)
        if max(abs(fm), abs(fj)) < 1e-10:
            return em, ej
        H = _sf_hess(em, ej, d, B, P, cont)
        if not np.all(np.isfinite(H)):
            H = _fd_hess(em, ej, d, B, P, cont)
        try:
            delta = np.linalg.solve(H, -np.array([fm, fj]))
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(delta)):
            break
        step = delta
        for _h in range(10):
            emn, ejn = em + step[0], ej + step[1]
            if emn < 0 or ejn < 0 or emn + ejn > 1:
                step *= 0.5
                continue
            fmn, fjn = _sf_grad(emn, ejn, d, B, P, cont)
            if max(abs(fmn), abs(fjn)) < max(abs(fm), abs(fj)):
                em, ej = emn, ejn
                break
            step *= 0.5
        else:
            break
    return em, ej


def _sf_nearcorner(emc, d, B, P, cont):
    """Near-corner interior max (share fast path).

    When the e_j=0 corner is NOT KKT-valid (Fj(emc, 0+) > 0), F rises
    from the corner and peaks at small e_j.  Find the Fj crossings on
    [1e-10, 0.05] at em = emc, Newton-polish each to the nearby FOC
    root, and keep the best candidate by F VALUE (exact).  A
    H-definiteness requirement is deliberately NOT imposed: with the
    s_floor the max can sit at the floor kink where the FOC is not
    smooth (Newton cannot certify it, but its F value is exact).
    Returns (em, ej, F) or None."""
    from scipy.optimize import brentq
    xs = [1e-10, 1e-6, 1e-4, 1e-3, 0.01, 0.05]
    vals = [_sf_grad(emc, x, d, B, P, cont)[1] for x in xs]
    best = None
    for i in range(len(xs) - 1):
        if vals[i] * vals[i + 1] < 0:
            try:
                ej0 = brentq(lambda x: _sf_grad(emc, x, d, B, P, cont)[1],
                             xs[i], xs[i + 1])
            except ValueError:
                continue
            for (er, jr) in ((emc, ej0), _sf_polish(emc, ej0, d, B, P,
                                                    cont)):
                if not (np.isfinite(er) and np.isfinite(jr) and er > 0
                        and jr > 0 and er + jr < 1 - 1e-10):
                    continue
                fv = _sf_eval(er, jr, d, B, P, cont)
                if best is None or fv > best[2]:
                    best = (float(er), float(jr), float(fv))
    return best


def _sf_binding(d, B, P, cont):
    """KKT binding case em+ej=1 (share fast path): root of
    Fm(em,1-em) - Fj(em,1-em) = 0 with multiplier lambda = Fm >= 0.
    Returns (em, ej, F) or None.  The time constraint binds when the
    interior FOC has no feasible root (damped Newton pushes to the
    boundary), so the fast path must check it explicitly."""
    from scipy.optimize import brentq

    def h(em):
        gm, gj = _sf_grad(em, 1.0 - em, d, B, P, cont)
        return gm - gj

    try:
        emc = brentq(h, 1e-10, 1.0 - 1e-10)
    except ValueError:
        return None
    gm, _ = _sf_grad(emc, 1.0 - emc, d, B, P, cont)
    if not np.isfinite(gm) or gm < -1e-10:
        return None
    return (float(emc), float(1.0 - emc),
            float(_sf_eval(emc, 1.0 - emc, d, B, P, cont)))


def _newton_share(d, B, P, cont, em, ej, max_iters=12):
    """Share-mode damped Newton on the FOC (primitives-once gradient).
    Returns (em, ej, converged, H)."""
    grad_fn = _sf_grad
    hess_fn = _sf_hess
    for _ in range(max_iters):
        fm, fj = grad_fn(em, ej, d, B, P, cont)
        if max(abs(fm), abs(fj)) < 1e-6:
            H = hess_fn(em, ej, d, B, P, cont)
            return em, ej, True, H
        H = hess_fn(em, ej, d, B, P, cont)
        if not np.all(np.isfinite(H)):
            H = _fd_hess(em, ej, d, B, P, cont)
        try:
            delta = np.linalg.solve(H, -np.array([fm, fj]))
        except np.linalg.LinAlgError:
            return em, ej, False, H
        if not np.all(np.isfinite(delta)):
            return em, ej, False, H
        step = delta
        for _h in range(14):  # damping
            emn = em + step[0]
            ejn = ej + step[1]
            if emn < 0 or ejn < 0 or emn + ejn > 1:
                step *= 0.5
                continue
            fmn, fjn = grad_fn(emn, ejn, d, B, P, cont)
            if max(abs(fmn), abs(fjn)) < max(abs(fm), abs(fj)):
                em, ej = emn, ejn
                break
            step *= 0.5
        else:
            return em, ej, False, H
    fm, fj = grad_fn(em, ej, d, B, P, cont)
    H = hess_fn(em, ej, d, B, P, cont)
    return em, ej, max(abs(fm), abs(fj)) < 1e-6, H


def solve_inner_fast(d, B, P, cont=None, warm=None):
    """Fast inner solve: damped Newton with analytic Jacobian, warm-started.
    Falls back to the full KKT active-set enumerator on failure.
    Returns (em, ej, Fval, converged_flag).

    share_reversal=True uses a primitives-once fast path:
      - Newton (max 12 iters) with the local-max check (H negative
        definite) so it cannot rest on a saddle/min root of the
        multi-root share FOC;
      - a cheap corner check (e_j=0 KKT corner + vertices) at every node,
        since with the share term corners can beat interior maxima;
      - the full KKT multi-root enumerator only when Newton fails AND no
        corner exists (rare: interior maxima dominate in most nodes).
    The numerical-Hessian retry covers non-finite analytic Hessians."""
    if warm is None:
        warm = (0.3, 0.3)
    em = float(np.clip(warm[0], 1e-12, 1.0))
    ej = float(np.clip(warm[1], 1e-12, 1.0))
    if em + ej > 1 - 1e-12:
        s = em + ej
        em, ej = em * (1 - 1e-12) / s, ej * (1 - 1e-12) / s
    # Note: with kappa=0 the Hessian is nearly singular along (1,-1)
    # (cost curvature zero in that direction), so Newton stalls at
    # |FOC| ~ 1e-6.  That accuracy implies argmax error ~1e-5 and W error
    # ~1e-11 (envelope), which is ample for VI/grid runs; the canonical
    # runs (bug-mode gate, baseline cache) use the exact KKT+SLSQP path.
    if P.share_reversal:
        em, ej, conv, H = _newton_share(d, B, P, cont, em, ej)
        fv_newton = _sf_eval(em, ej, d, B, P, cont)
        corner, emc, corner_valid = _sf_corner(d, B, P, cont)
        cands = [c for c in (corner,) if c is not None]
        if emc is not None and not corner_valid:
            # near-corner interior max (the corner is nearly optimal but
            # the exact max sits at tiny e_j > 0)
            near = _sf_nearcorner(emc, d, B, P, cont)
            if near is not None:
                cands.append(near)
        binding = _sf_binding(d, B, P, cont)
        if binding is not None:
            cands.append(binding)
        best_f = fv_newton
        best_e = (em, ej)
        for c in cands:
            if c[2] > best_f:
                best_f, best_e = c[2], (c[0], c[1])
        if conv and np.all(np.isfinite(H)) \
                and np.linalg.eigvalsh(H).max() <= 1e-8:
            # Newton found a certified interior local max.  If the
            # near-corner region exists, Newton's warm start may sit on
            # the near-corner basin while the interior basin is better:
            # re-check from the canonical interior start (0.3, 0.3).
            if emc is not None and not corner_valid:
                em2, ej2, conv2, H2 = _newton_share(d, B, P, cont, 0.3, 0.3)
                if conv2 and np.all(np.isfinite(H2)) \
                        and np.linalg.eigvalsh(H2).max() <= 1e-8:
                    f2 = _sf_eval(em2, ej2, d, B, P, cont)
                    if f2 > best_f:
                        best_f, best_e = f2, (em2, ej2)
            return best_e[0], best_e[1], best_f, True
        if conv:
            # Newton at a stationary point that is NOT a local max (saddle
            # or min root of the multi-root FOC): an interior max may be
            # missed, so run the full multi-root enumerator.
            e_ref, f_ref = solve_inner_kkt(d, B, P, cont)
            if e_ref is not None and f_ref > best_f:
                return e_ref[0], e_ref[1], f_ref, False
            return best_e[0], best_e[1], best_f, True
        # Newton oscillated/damped out (interior FOC root infeasible):
        # a boundary/near-corner candidate is the answer if it beats the
        # Newton point; otherwise the full multi-root enumerator.
        if best_f > fv_newton + 1e-9:
            return best_e[0], best_e[1], best_f, True
        e_ref, f_ref = solve_inner_kkt(d, B, P, cont)
        if e_ref is not None and f_ref > best_f:
            return e_ref[0], e_ref[1], f_ref, False
        return best_e[0], best_e[1], best_f, True

    # ---------------- baseline (share_reversal=False) path: unchanged ----
    converged = False
    for _ in range(40):
        fm, fj = F_grad(em, ej, d, B, P, cont)
        if max(abs(fm), abs(fj)) < 1e-6:
            converged = True
            break
        H = F_hess(em, ej, d, B, P, cont)
        try:
            delta = np.linalg.solve(H, -np.array([fm, fj]))
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(delta)):
            break
        step = delta
        for _h in range(14):  # damping
            emn = em + step[0]
            ejn = ej + step[1]
            if emn < 0 or ejn < 0 or emn + ejn > 1:
                step *= 0.5
                continue
            fmn, fjn = F_grad(emn, ejn, d, B, P, cont)
            if max(abs(fmn), abs(fjn)) < max(abs(fm), abs(fj)):
                em, ej, fm, fj = emn, ejn, fmn, fjn
                break
            step *= 0.5
        else:
            break
    if not converged:
        e_ref, f_ref = solve_inner_kkt(d, B, P, cont)
        if e_ref is None:
            return None, None, -np.inf, False
        return e_ref[0], e_ref[1], f_ref, False
    fv = F_eval(em, ej, d, B, P, cont)
    return em, ej, fv, True


# ---------------------------------------------------------------------------
# Method (a): SLSQP multi-start + coarse-grid fallback
# ---------------------------------------------------------------------------

_SLSQP_STARTS = [(0.5, 0.3), (0.3, 0.5), (0.2, 0.2), (0.6, 0.2), (0.1, 0.6)]


def _slsqp_solve(d, B, P, cont, starts):
    cons = [{"type": "ineq", "fun": lambda e: 1.0 - e[0] - e[1]}]
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    best_e, best_f = None, -np.inf
    for s0 in starts:
        try:
            res = minimize(lambda e: -F_eval(e[0], e[1], d, B, P, cont),
                           s0, method="SLSQP", bounds=bounds, constraints=cons,
                           options={"ftol": 1e-12, "maxiter": 300})
        except Exception:
            continue
        if res.x is None:
            continue
        em, ej = float(np.clip(res.x[0], 0, 1)), float(np.clip(res.x[1], 0, 1))
        if em + ej > 1 + 1e-12:
            continue
        fv = F_eval(em, ej, d, B, P, cont)
        if fv > best_f:
            best_f, best_e = fv, (em, ej)
    return best_e, best_f


def _coarse_grid_fallback(d, B, P, cont):
    """11x11 coarse grid of the triangle, best point as an extra start."""
    grid = np.linspace(0.0, 1.0, 11)
    best_e, best_f = None, -np.inf
    for em in grid:
        for ej in grid:
            if em + ej > 1.0 + 1e-12:
                continue
            fv = F_eval(em, ej, d, B, P, cont)
            if fv > best_f:
                best_f, best_e = fv, (float(em), float(ej))
    return best_e, best_f


def solve_inner_slsqp(d, B, P, cont=None):
    starts = list(_SLSQP_STARTS)
    cg, cgf = _coarse_grid_fallback(d, B, P, cont)
    if cg is not None:
        starts.append(cg)
    e, f = _slsqp_solve(d, B, P, cont, starts)
    if e is None:
        return cg, cgf
    return e, f


# ---------------------------------------------------------------------------
# Method (b): KKT active-set enumerator
# ---------------------------------------------------------------------------

def _case_interior_old(d, B, P, cont):
    """Case A (baseline model): em>0, ej>0, em+ej<1.  FOC: Fm=0, Fj=0.
    Nested brentq: for fixed em, solve Fj(em,ej)=0 for ej (exists while the
    unconstrained-in-ej root satisfies ej <= 1-em); then root g(em)=Fm(em,ej*(em)).
    F strictly concave => at most one interior FOC root."""
    def ej_star(em):
        try:
            return brentq(lambda ej: F_grad(em, ej, d, B, P, cont)[1],
                          1e-10, 1.0 - em)
        except ValueError:
            return None  # no interior-in-ej root under the time constraint

    def sum_le(em):
        """em + ej_star(em) - 1; infeasible => positive (constraint violated)."""
        ej = ej_star(em)
        if ej is None:
            return 1e-6
        return em + ej - 1.0

    # find em_cap = largest em with a feasible ej root (em + ej* <= 1)
    if sum_le(1e-10) > 0:
        return None  # constraint binds even at tiny em: no interior case
    lo, hi = 1e-10, 1.0 - 1e-10
    if sum_le(hi) <= 0:
        em_cap = hi
    else:
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if sum_le(mid) <= 0:
                lo = mid
            else:
                hi = mid
        em_cap = lo

    def g_of(em):
        ej = ej_star(em)
        return F_grad(em, ej, d, B, P, cont)[0]

    try:
        em = brentq(g_of, 1e-10, em_cap)
        ej = ej_star(em)
    except ValueError:
        return None
    if not (em > 0 and ej > 0 and em + ej < 1 - 1e-10):
        return None
    return (em, ej)


def _polish_foc(em, ej, d, B, P, cont, iters=20):
    """Damped Newton polish to a nearby FOC root (share fallback path)."""
    for _ in range(iters):
        fm, fj = F_grad(em, ej, d, B, P, cont)
        if max(abs(fm), abs(fj)) < 1e-10:
            return em, ej
        H = F_hess(em, ej, d, B, P, cont)
        if not np.all(np.isfinite(H)):
            H = _fd_hess(em, ej, d, B, P, cont)
        try:
            delta = np.linalg.solve(H, -np.array([fm, fj]))
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(delta)):
            break
        step = delta
        for _h in range(10):
            emn, ejn = em + step[0], ej + step[1]
            if emn < 0 or ejn < 0 or emn + ejn > 1:
                step *= 0.5
                continue
            fmn, fjn = F_grad(emn, ejn, d, B, P, cont)
            if max(abs(fmn), abs(fjn)) < max(abs(fm), abs(fj)):
                em, ej = emn, ejn
                break
            step *= 0.5
        else:
            break
    return em, ej


def _case_interior(d, B, P, cont):
    """Case A: em>0, ej>0, em+ej<1.  FOC: Fm=0, Fj=0.

    Baseline (share_reversal=False): single root (F strictly concave),
    handled by _case_interior_old.

    share_reversal=True: the share penalty on e_j (-a2*(1-g)*r*s_j, s_j>0)
    makes Fj cross zero up to twice in e_j (a min-root near 0 and the
    interior max-root), so the nested brentq could land on the wrong
    branch.  Scan sign changes of Fj in e_j on a coarse em-grid, polish
    each candidate to its FOC root, keep the best by objective."""
    if not P.share_reversal:
        return _case_interior_old(d, B, P, cont)
    best_e, best_f = None, -np.inf
    for em0 in np.linspace(1e-10, 1.0 - 1e-10, 12):
        xs = np.linspace(1e-10, 1.0 - em0, 32)
        fprev = _sf_grad(em0, xs[0], d, B, P, cont)[1]
        for i in range(1, len(xs)):
            fcur = _sf_grad(em0, xs[i], d, B, P, cont)[1]
            if fprev * fcur < 0:
                try:
                    ej0 = brentq(lambda ej: _sf_grad(em0, ej, d, B, P,
                                                     cont)[1],
                                 xs[i - 1], xs[i])
                except ValueError:
                    ej0 = None
                if ej0 is not None:
                    emr, ejr = _sf_polish(em0, ej0, d, B, P, cont)
                    if (np.isfinite(emr) and np.isfinite(ejr) and emr > 0
                            and ejr > 0 and emr + ejr < 1 - 1e-10):
                        fv = _sf_eval(emr, ejr, d, B, P, cont)
                        if fv > best_f:
                            best_f, best_e = fv, (float(emr), float(ejr))
            fprev = fcur
    return best_e


def _case_binding(d, B, P, cont):
    """Case B: em+ej=1, em,ej>0.  KKT: Fm = Fj = lambda >= 0."""
    def h(em):
        ej = 1.0 - em
        fm, fj = F_grad(em, ej, d, B, P, cont)
        return fm - fj
    try:
        em = brentq(h, 1e-10, 1.0 - 1e-10)
        ej = 1.0 - em
    except ValueError:
        return None
    fm, _ = F_grad(em, ej, d, B, P, cont)
    if fm < -1e-10:  # lambda = fm must be >= 0
        return None
    return (em, ej)


def _case_ej_zero(d, B, P, cont):
    """Case C: ej=0, em in (0,1).  Fm(em,0)=0; need Fj(em,0) <= 0.

    share_reversal=True: Fj(em,0) = +inf - inf -> nan exactly (the psi'
    divergence of the output benefit and of the share penalty cancel in
    the limit; for our parameter range the limit is -inf, i.e. the corner
    is KKT-valid).  Evaluate at ej=1e-10 instead to get a finite sign."""
    ej0 = 1e-10 if P.share_reversal else 0.0
    def h(em):
        return F_grad(em, ej0, d, B, P, cont)[0]
    try:
        em = brentq(h, 1e-10, 1.0 - 1e-10)
    except ValueError:
        return None
    fj = F_grad(em, ej0, d, B, P, cont)[1]
    if not np.isfinite(fj) or fj > 1e-10:
        return None
    return (em, 0.0)


def _case_em_zero(d, B, P, cont):
    """Case D: em=0, ej in (0,1).  Fj(0,ej)=0; need Fm(0,ej) <= 0.
    (With phi1<1, Fm(0,.) = +inf, so this is almost always infeasible;
    kept for completeness of the active-set enumeration.)"""
    em0 = 1e-10 if P.share_reversal else 0.0
    def h(ej):
        return F_grad(em0, ej, d, B, P, cont)[1]
    try:
        ej = brentq(h, 1e-10, 1.0 - 1e-10)
    except ValueError:
        return None
    fm = F_grad(em0, ej, d, B, P, cont)[0]
    if not np.isfinite(fm) or fm > 1e-10:
        return None
    return (0.0, ej)


def _vertices(d, B, P, cont):
    """Corner candidates: (0,0), (1,0), (0,1) (Case E and boundary corners)."""
    return [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]


def solve_inner_kkt(d, B, P, cont=None):
    """Enumerate all KKT cases, evaluate, return global maximizer."""
    candidates = []
    for maker in (_case_interior, _case_binding, _case_ej_zero, _case_em_zero):
        cand = maker(d, B, P, cont)
        if cand is not None:
            candidates.append(cand)
    candidates.extend(_vertices(d, B, P, cont))
    best_e, best_f = None, -np.inf
    for (em, ej) in candidates:
        em = float(np.clip(em, 0.0, 1.0))
        ej = float(np.clip(ej, 0.0, 1.0))
        if em + ej > 1.0 + 1e-12:
            continue
        fv = F_eval(em, ej, d, B, P, cont)
        if fv > best_f:
            best_f, best_e = fv, (em, ej)
    return best_e, best_f


# ---------------------------------------------------------------------------
# Static JOINT solver over (d, em, ej): the Phase-1 / T=1 KKT active-set
# solution with d handled by its own active-set (interior-d FOC root via
# brentq on -delta*d + dF/dd at the inner optimum, plus d=0, d=1 corners).
# ---------------------------------------------------------------------------

def _dF_dd(em, ej, d, B, P, cont):
    """dF/dd at fixed (em,ej): dU/dd + cont'(B') * dB'/dd.

    share_reversal=True: dTerm2_new/dd = +a2*r*s*g' (rejection reduces
    reversal exposure through (1-g(d)), scaled by the adjudication share
    s), so the bracket becomes (w_tau*(ph+ps)*N - a2*r*s)."""
    ph, ps = phi(em, P), psi(ej, P)
    gg = g(d, P)
    gd = dg(d, P)
    if P.share_reversal:
        s = share(em, ej, P)
        rev_coef = P.alpha2 * r(ej, P) * s
    else:
        rev_coef = P.alpha2 * r(ej, P)
    out = -(w_tau(P) * (ph + ps) * P.N - rev_coef) * gd
    if cont is not None:
        bp = Bnext(em, ej, d, B, P)
        out += cont.deriv(bp) * (-gd * (1 - ph - ps) * P.N)
    return out


def solve_static_joint(B, P, cont=None, n_d_grid=101, method="kkt"):
    """max_{d,em,ej} -delta/2 d^2 + F(em,ej;d,B) via KKT active-set in d.

    (a) evaluate the objective on a fine d-grid (inner via KKT);
    (b) fit a cubic spline in d to -delta/2 d^2 + Pi(d), find interior
        roots of the d-FOC via brentq on the spline derivative;
    (c) evaluate candidates d=0, d=1 and the interior roots; global max.
    Returns (d*, em*, ej*, obj)."""
    from scipy.interpolate import CubicSpline
    d_grid = np.linspace(0.0, 1.0, n_d_grid)
    vals = np.empty(n_d_grid)
    sols = {}
    for i, d in enumerate(d_grid):
        e, f = solve_inner_kkt(d, B, P, cont)
        if e is None:
            vals[i] = -np.inf
            continue
        obj = -0.5 * P.delta * d * d + f
        vals[i] = obj
        sols[d] = (e[0], e[1], obj)
    # spline of the outer objective
    good = np.isfinite(vals)
    cs = CubicSpline(d_grid[good], vals[good])
    csd = cs.derivative()

    def h(d):
        return csd(d)

    candidates = [0.0, 1.0]
    for d0 in np.linspace(0.02, 0.98, 49):
        try:
            root = brentq(h, max(d0 - 0.02, 0.0), min(d0 + 0.02, 1.0))
        except ValueError:
            continue
        if any(abs(root - c) < 1e-8 for c in candidates):
            continue
        candidates.append(root)
    best = None
    for d in candidates:
        d = float(np.clip(d, 0.0, 1.0))
        if d in sols:
            em, ej, obj = sols[d]
        else:
            e, f = solve_inner_kkt(d, B, P, cont)
            if e is None:
                continue
            em, ej = e
            obj = -0.5 * P.delta * d * d + f
        if best is None or obj > best[3]:
            best = (em, ej, d, obj)
    if best is None:
        # fall back to the grid best
        i = int(np.argmax(vals))
        d = d_grid[i]
        e, f = solve_inner_kkt(d, B, P, cont)
        return e[0], e[1], d, vals[i]
    return best


# ---------------------------------------------------------------------------
# Public wrapper with cross-check
# ---------------------------------------------------------------------------

def solve_inner(d, B, P, cont=None, method="both", check_tol=1e-7):
    """Solve inner (em,ej) problem. Returns (em, ej, Fval).

    method: 'both' (KKT + SLSQP cross-checked), 'kkt', 'slsqp'.
    """
    if method == "kkt":
        e, f = solve_inner_kkt(d, B, P, cont)
        return e[0], e[1], f
    if method == "slsqp":
        e, f = solve_inner_slsqp(d, B, P, cont)
        return e[0], e[1], f

    e1, f1 = solve_inner_kkt(d, B, P, cont)
    e2, f2 = solve_inner_slsqp(d, B, P, cont)
    if e1 is None:
        return e2[0], e2[1], f2
    if e2 is None:
        return e1[0], e1[1], f1
    # cross-check: same max (objective agreement) and argmax agreement
    df = abs(f1 - f2)
    de = abs(e1[0] - e2[0]) + abs(e1[1] - e2[1])
    agreed = (df <= check_tol * max(1.0, abs(f1))) and (de <= 1e-5)
    if not agreed:
        # keep the better objective; log via module-level counter
        _XCHECK_MISSES[0] += 1
        _XCHECK_MAXDF[0] = max(_XCHECK_MAXDF[0], df)
        _XCHECK_MAXDE[0] = max(_XCHECK_MAXDE[0], de)
    e = e1 if f1 >= f2 else e2
    return e[0], e[1], max(f1, f2)


_XCHECK_MISSES = [0]
_XCHECK_MAXDF = [0.0]
_XCHECK_MAXDE = [0.0]


def xcheck_stats():
    return {"n_miss": _XCHECK_MISSES[0],
            "max_df": _XCHECK_MAXDF[0],
            "max_de": _XCHECK_MAXDE[0]}
