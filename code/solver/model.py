"""Model primitives: functional forms, flow utility, backlog transition,
and analytic derivatives (research note, Section 2)."""
import numpy as np
from dataclasses import dataclass


@dataclass
class Params:
    alpha1: float = 0.3
    alpha2: float = 0.5
    alpha3: float = 0.0
    gamma: float = 0.5
    kappa: float = 0.0
    phi0: float = 0.73
    phi1: float = 0.8
    psi0: float = 0.912
    psi1: float = 0.8
    r0: float = 0.3
    r1: float = 5.0
    omega: float = 0.2
    S: float = 400.0
    delta: float = 1.0
    p: float = 1e-5
    beta: float = 0.95
    eta: float = 5.0
    N: float = 100.0
    T: int = 4
    B_max: float = 400.0
    # Share extension: reversal risk attaches only to ADJUDICATED output.
    # Term 2_new = -alpha2 * r(e_j) * (1-g(d)) * s(e_m,e_j),
    # s = psi/(phi+psi) (adjudication share; mediation escapes reversal).
    # share_reversal=False reproduces the old model bit-for-bit.
    share_reversal: bool = False
    # Hybrid fallback: s <- max(s, s_floor) (keeps the intensive margin
    # alive at low e_j; flat floor => zero share derivatives below it).
    s_floor: float = 0.0
    # Asymmetric cost (baseline specification): c = gamma_m e_m^2/2 + gamma_j e_j^2/2
    # + gamma_mj e_m e_j, with gamma_m*gamma_j > gamma_mj^2 (positive
    # definite).  When gamma_m is None the old gamma/kappa path is used
    # verbatim (bit-identical).  The old form maps onto the asymmetric
    # one via gamma_m = gamma+kappa, gamma_j = gamma, gamma_mj = gamma.
    gamma_m: float = None
    gamma_j: float = None
    gamma_mj: float = None


# ---------------------------------------------------------------------------
# Functional forms (vectorized where possible)
# ---------------------------------------------------------------------------

def phi(em, P: Params):
    # builtin max is bit-identical to np.clip(em, 0.0, None) and ~200x faster
    return P.phi0 * np.power(em if em > 0.0 else 0.0, P.phi1)


def psi(ej, P: Params):
    return P.psi0 * np.power(ej if ej > 0.0 else 0.0, P.psi1)


def r(ej, P: Params):
    return P.r0 * np.exp(-P.r1 * ej)


def g(d, P: Params):
    return 1.0 - np.exp(-P.eta * d)


def cost_coefs(P: Params):
    """(gamma_m, gamma_j, gamma_mj) — asymmetric cost coefficients.
    gamma_m is None -> old gamma/kappa path (bit-identical to the
    historical form)."""
    if P.gamma_m is None:
        return P.gamma + P.kappa, P.gamma, P.gamma
    return P.gamma_m, P.gamma_j, P.gamma_mj


def ccost(em, ej, P: Params):
    if P.gamma_m is None:
        return 0.5 * P.gamma * (em + ej) ** 2 + 0.5 * P.kappa * em ** 2
    gm, gj, gmj = P.gamma_m, P.gamma_j, P.gamma_mj
    return 0.5 * gm * em * em + 0.5 * gj * ej * ej + gmj * em * ej


def w_tau(P: Params):
    """w_tau = alpha1 * E[1/S | F_tau]; with N constant, S = T*N."""
    return P.alpha1 / (P.T * P.N)


def U(em, ej, d, B, P: Params):
    """Subperiod flow utility U_tau(e_m, e_j; d, B)."""
    ph, ps = phi(em, P), psi(ej, P)
    gg = g(d, P)
    out = (w_tau(P) * ((ph + ps) * (1 - gg) * P.N + P.omega * B)
           - P.alpha2 * r(ej, P) * (1 - gg)
           + P.alpha3 * _mr(ph, ps)
           - ccost(em, ej, P))
    if P.share_reversal:
        # Share extension: reversal risk attaches only to adjudicated output:
        # Term 2_new = -alpha2*r*(1-g)*s.  Out = out_old - alpha2*r*(1-g)*s
        #            = out_old + alpha2*r*(1-g)*(1-s).
        out += P.alpha2 * r(ej, P) * (1 - gg) * (1.0 - share(em, ej, P))
    return out


def _mr(ph, ps):
    """alpha3 mediation-ratio term phi/(phi+psi)."""
    s = ph + ps
    return ph / s if s > 0 else 0.0


def share(em, ej, P: Params):
    """s(e_m,e_j) = psi(e_j)/(phi(e_m)+psi(e_j)) — adjudication share.

    Share extension: reversal risk is borne only by the adjudicated share of
    output (mediation escapes reversal entirely — extensive margin).
    With P.s_floor > 0 the share is floored: s <- max(s, s_floor)
    (hybrid form that keeps the intensive reversal margin alive at low
    e_j; the floor is flat, so all share derivatives are 0 below it)."""
    ph, ps = phi(em, P), psi(ej, P)
    den = ph + ps
    s = ps / den if den > 0 else 0.0
    if P.s_floor > 0 and s < P.s_floor:
        return P.s_floor
    return s


def _share_derivs(em, ej, P: Params):
    """(s, s_m, s_j, s_mm, s_jj, s_mj) for s = psi/(phi+psi).

    Functional form is the complement of the alpha3 ratio rho = phi/(phi+psi):
    s_m = -rho_m, s_j = -rho_j, etc.  All derivatives vanish below s_floor.
    """
    ph, ps = phi(em, P), psi(ej, P)
    den = ph + ps
    if den <= 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    s = ps / den
    if P.s_floor > 0 and s < P.s_floor:
        return (P.s_floor, 0.0, 0.0, 0.0, 0.0, 0.0)
    dp, dq = dphi(em, P), dpsi(ej, P)
    d2p = _phi2(em, P)
    d2q = _psi2(ej, P)
    den2, den3 = den * den, den * den * den
    sm = -ps * dp / den2
    sj = ph * dq / den2
    smm = -ps * d2p / den2 + 2 * ps * dp * dp / den3
    sjj = ph * d2q / den2 - 2 * ph * dq * dq / den3
    smj = dp * dq * (ps - ph) / den3
    return (s, sm, sj, smm, sjj, smj)


def _phi2(em, P: Params):
    """phi''(e_m)."""
    if em <= 0:
        return -np.inf
    return P.phi0 * P.phi1 * (P.phi1 - 1) * em ** (P.phi1 - 2)


def _psi2(ej, P: Params):
    """psi''(e_j)."""
    if ej <= 0:
        return -np.inf
    return P.psi0 * P.psi1 * (P.psi1 - 1) * ej ** (P.psi1 - 2)


def Bnext(em, ej, d, B, P: Params):
    """Backlog transition B_{tau+1} = (1-omega)B + (1-g(d))(1-phi-psi)N."""
    return (1 - P.omega) * B + (1 - g(d, P)) * (1 - phi(em, P) - psi(ej, P)) * P.N


# ---------------------------------------------------------------------------
# Analytic first derivatives (scalar args)
# ---------------------------------------------------------------------------

def dphi(em, P: Params):
    if em <= 0:
        return np.inf if P.phi1 < 1 else P.phi0 * P.phi1
    return P.phi0 * P.phi1 * em ** (P.phi1 - 1)


def dpsi(ej, P: Params):
    if ej <= 0:
        return np.inf if P.psi1 < 1 else P.psi0 * P.psi1
    return P.psi0 * P.psi1 * ej ** (P.psi1 - 1)


def dr(ej, P: Params):
    return -P.r1 * r(ej, P)


def dg(d, P: Params):
    return P.eta * np.exp(-P.eta * d)


def Um(em, ej, d, B, P: Params):
    """dU/dem."""
    if P.gamma_m is None:
        dc = P.gamma * (em + ej) + P.kappa * em
    else:
        dc = P.gamma_m * em + P.gamma_mj * ej
    out = (w_tau(P) * dphi(em, P) * (1 - g(d, P)) * P.N
           - dc)
    if P.alpha3 != 0:
        ph, ps = phi(em, P), psi(ej, P)
        out += P.alpha3 * dphi(em, P) * ps / (ph + ps) ** 2
    if P.share_reversal:
        # dTerm2_new/dem = -alpha2*r*(1-g)*s_m, s_m < 0 => subsidy on e_m
        _, sm, *_ = _share_derivs(em, ej, P)
        out -= P.alpha2 * r(ej, P) * (1 - g(d, P)) * sm
    return out


def Uj(em, ej, d, B, P: Params):
    """dU/dej."""
    ph, ps = phi(em, P), psi(ej, P)
    if P.gamma_m is None:
        dc = P.gamma * (em + ej)
    else:
        dc = P.gamma_j * ej + P.gamma_mj * em
    out = (w_tau(P) * dpsi(ej, P) * (1 - g(d, P)) * P.N
           - P.alpha2 * dr(ej, P) * (1 - g(d, P))
           - dc)
    if P.alpha3 != 0:
        out -= P.alpha3 * ph * dpsi(ej, P) / (ph + ps) ** 2
    if P.share_reversal:
        # U = U_rest + Term2_new, Term2_new = -a2*r*(1-g)*s.
        # dTerm2_new/dej = -a2*(1-g)*(r'*s + r*s_j).
        # The old reversal term -a2*r*(1-g) is REPLACED, so add back the
        # old intensive margin scaled by the mediation share (1-s) and the
        # share penalty: out_old + a2*(1-g)*(r'*(1-s) - r*s_j).
        s, _, sj, *_ = _share_derivs(em, ej, P)
        onemg = 1 - g(d, P)
        out += P.alpha2 * onemg * (dr(ej, P) * (1.0 - s) - r(ej, P) * sj)
    return out


# ---------------------------------------------------------------------------
# Continuation objects: value + derivative of the continuation function
# ---------------------------------------------------------------------------

class Continuation:
    """cont(x): value of continuation at backlog x; cont.deriv(x): dcont/dx;
    cont.deriv2(x): d^2cont/dx^2.  Picklable (methods, no closures)."""

    def __call__(self, x):
        return self.fn(x)

    def deriv(self, x):
        return self.dfn(x)

    def deriv2(self, x):
        return self.d2fn(x)

    def n_clamped(self, x):
        if self.B_grid is None:
            return 0.0
        return float(x < self.B_grid[0]) + float(x > self.B_grid[-1])


class _SplineCoef:
    """Clamped CubicSpline stored as coefficient arrays only (picklable).
    Evaluation is pure numpy: p(t) = c0*t^3 + c1*t^2 + c2*t + c3, t=x-x_i.
    Outside the grid the boundary finite differences are used."""

    def __init__(self, B_grid, W):
        from scipy.interpolate import CubicSpline
        cs = CubicSpline(np.asarray(B_grid, float), np.asarray(W, float),
                         bc_type="clamped")
        self.x = cs.x
        self.c = cs.c  # (4, n-1): [t^3, t^2, t, const] per interval
        h = B_grid[1] - B_grid[0]
        self.fd0 = (W[1] - W[0]) / h
        self.fdn = (W[-1] - W[-2]) / h
        self.fd0_2 = (W[2] - 2 * W[1] + W[0]) / h ** 2
        self.fdn_2 = (W[-1] - 2 * W[-2] + W[-3]) / h ** 2
        self.lo = float(B_grid[0])
        self.hi = float(B_grid[-1])

    def _seg(self, x):
        if x <= self.lo:
            return 0, 0.0
        if x >= self.hi:
            return len(self.x) - 2, 0.0
        i = int(np.searchsorted(self.x, x) - 1)
        i = min(max(i, 0), len(self.x) - 2)
        return i, x - self.x[i]

    def val(self, x):
        if x < self.lo:
            return self.fd0 * (x - self.lo) + _eval_p(self.c[:, 0], 0.0)
        if x > self.hi:
            return self.fdn * (x - self.hi) + _eval_p(self.c[:, -1], 0.0)
        i, t = self._seg(x)
        return _eval_p(self.c[:, i], t)

    def deriv(self, x):
        if x < self.lo:
            return self.fd0
        if x > self.hi:
            return self.fdn
        i, t = self._seg(x)
        c = self.c[:, i]
        return 3 * c[0] * t ** 2 + 2 * c[1] * t + c[2]

    def deriv2(self, x):
        if x < self.lo:
            return self.fd0_2
        if x > self.hi:
            return self.fdn_2
        i, t = self._seg(x)
        c = self.c[:, i]
        return 6 * c[0] * t + 2 * c[1]


def _eval_p(c, t):
    return c[0] * t ** 3 + c[1] * t ** 2 + c[2] * t + c[3]


def make_spline_deriv(B_grid, W):
    """Smooth clamped derivative of a grid function (CubicSpline coefs)."""
    sp = _SplineCoef(np.asarray(B_grid, float), np.asarray(W, float))
    return sp.deriv


def make_spline_deriv2(B_grid, W):
    """Second derivative of a grid function (CubicSpline coefs)."""
    sp = _SplineCoef(np.asarray(B_grid, float), np.asarray(W, float))
    return sp.deriv2


class WContinuation(Continuation):
    """tau<T continuation: beta * W_{tau+1}(B') (task spec: within-year beta)."""

    def __init__(self, B_grid, W_next, within_beta=1.0, label="W_next"):
        self.B_grid = np.asarray(B_grid)
        self.W_next = np.asarray(W_next)
        self.within_beta = within_beta
        self.label = label
        self._sp = _SplineCoef(self.B_grid, self.W_next)

    def fn(self, x):
        return (self.within_beta * float(
            np.interp(x, self.B_grid, self.W_next)))

    def dfn(self, x):
        return self.within_beta * float(self._sp.deriv(x))

    def d2fn(self, x):
        return self.within_beta * float(self._sp.deriv2(x))


class TermContinuation(Continuation):
    """tau=T continuation: -p*B'^2 + beta*V_next(B') (V_next=None -> 0)."""

    def __init__(self, P: Params, B_grid=None, V_next=None, within_beta=1.0):
        self.P = P
        self.B_grid = None if B_grid is None else np.asarray(B_grid)
        self.within_beta = within_beta
        self.has_V = V_next is not None
        if self.has_V:
            self.V_next = np.asarray(V_next)
            self._sp = _SplineCoef(self.B_grid, self.V_next)

    def fn(self, x):
        v = -self.P.p * x ** 2
        if self.has_V:
            v += self.within_beta * float(np.interp(x, self.B_grid, self.V_next))
        return v

    def dfn(self, x):
        v = -2 * self.P.p * x
        if self.has_V:
            v += self.within_beta * float(self._sp.deriv(x))
        return v

    def d2fn(self, x):
        v = -2 * self.P.p
        if self.has_V:
            v += self.within_beta * float(self._sp.deriv2(x))
        return v
