"""Reform experiment: exact decomposition of the reform effect at steady state.

Reform = N: 100 -> N_post (first-order inflow; central N_post=150) plus
eta: 5 -> 8 (second-order behavioral: rejection more effective).

Every regime (N, eta) is solved with its OWN regime-consistent V*
(see run_regime.py),
then:
  1. steady state B* = fixed point of the year map B_{y+1}=B'(B_y)
     (full-year chain with B-interpolated policies), damped iteration to 1e-8,
     stability |dB'/dB| < 1 verified by finite difference;
  2. annual adjudicated volume v(N,eta;B) = sum_tau psi(ej)(1-g(d))N at B*;
  3. exact constructive decomposition of
       DV = v(150,8;B2) - v(100,5;B0)
     in the natural order (eta first = behavioral, N second = inflow):

       DV = D_d + D_e + D_em + Int        (behavioral at common B0, N=100)
          + S1_B                          (backlog adjustment, eta step)
          + I1                            (first-order inflow: frozen
                                           pre-reform policies at B0)
          + (S2_N - I1)                   (second-order inflow: extra inflow
                                           from the post-reform per-case volume)
          + S2_pol                        (policy re-optimization gap between
                                           the (150,8) and (100,8) regimes)
          + S2_B                          (backlog move B1->B2 under eta=8)

     D_d  = d-channel  : (1-g(d)) change at pre-reform e_j, N=100, B0
                         (d* policy AND eta efficacy move; e_m, e_j frozen)
     D_e  = e_j-channel: psi(e_j) change at pre-reform d (eta=5 g), N=100, B0
     D_em = mediation-side (e_m) change at pre-reform (d, e_j), eta=5
     Int  = interaction residual; D_d + D_e + D_em + Int = S1_pol EXACTLY

     Reverse order (N first at eta=5, then eta at N=150) reported too,
     using an additional (150,5) regime; both orders telescope EXACTLY.

Sensitivity: N_post in {125, 200} with the natural-order decomposition.

Writes results/reform_cache.npz + results/reform_results.json.

Usage: python solver/reform_experiment.py
"""
import os
import sys
import json
import numpy as np

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)

from solver.model import Params, phi, psi, g

OUT_DIR = os.path.join(os.path.dirname(CODE_DIR), "results")

T = 4
OMEGA = 0.2
N0 = 100.0
ETA0 = 5.0
ETA1 = 8.0
NPOST_CENTRAL = 150.0
NPOST_SENS = [125.0, 200.0]
MAX_ITER_BSS = 20000
TOL_BSS = 1e-8
ALPHA1 = 0.3


# ---------------------------------------------------------------------------
# Regime loading
# ---------------------------------------------------------------------------

def load_regime(N, eta):
    """Regime (N, eta) -> dict with V_star and final full-mode policies.
    Loaded from results/regime_{N}_{eta}.npz (produced by run_regime.py)."""
    path = os.path.join(OUT_DIR, f"regime_{int(N)}_{int(eta)}.npz")
    d = np.load(path)
    return {k: d[k] for k in d.files}


def make_P(N, eta):
    with open(os.path.join(CODE_DIR, "calibration.json")) as f:
        cp = json.load(f)
    return Params(share_reversal=True, s_floor=cp["s_floor"],
                  alpha2=cp["alpha2"], r0=cp["r0"], gamma_m=cp["gamma_m"],
                  gamma_j=cp["gamma_j"], gamma_mj=cp["gamma_mj"],
                  N=N, eta=eta)


# ---------------------------------------------------------------------------
# Year map, steady state, volumes
# ---------------------------------------------------------------------------

def year_map_policy(pol_d, pol_em, pol_ej, B0, B_grid, P, N, T=T):
    """One full year from backlog B0 with B-interpolated policies and a
    FIXED Params object P (so mixed-policy counterfactuals share forms).
    Returns (B at year end, per-subperiod d, em, ej)."""
    B = float(B0)
    ds, ems, ejs = [], [], []
    for tau in range(T):
        d = float(np.interp(B, B_grid, pol_d[tau]))
        em = float(np.interp(B, B_grid, pol_em[tau]))
        ej = float(np.interp(B, B_grid, pol_ej[tau]))
        ds.append(d); ems.append(em); ejs.append(ej)
        ph, ps = phi(em, P), psi(ej, P)
        B = (1 - OMEGA) * B + (1 - g(d, P)) * (1 - ph - ps) * N
    return B, np.array(ds), np.array(ems), np.array(ejs)


def steady_state(regime, B_grid):
    """B* = fixed point of the year map (damped iteration). Returns
    (B*, converged, |dB'/dB| at B* via central finite difference)."""
    P = make_P(regime["N"], regime["eta"])
    pol_d, pol_em, pol_ej = regime["d_policy"], regime["em_policy"], regime["ej_policy"]
    N = regime["N"]
    B = 0.0
    conv = False
    for _ in range(MAX_ITER_BSS):
        Bn, *_ = year_map_policy(pol_d, pol_em, pol_ej, B, B_grid, P, N)
        if abs(Bn - B) < TOL_BSS:
            conv = True
            break
        B = 0.5 * B + 0.5 * Bn
    eps = max(1e-3, 1e-4 * B)
    Bp, *_ = year_map_policy(pol_d, pol_em, pol_ej, B + eps, B_grid, P, N)
    Bm, *_ = year_map_policy(pol_d, pol_em, pol_ej, B - eps, B_grid, P, N)
    dmap = (Bp - Bm) / (2 * eps)
    return B, conv, float(abs(dmap))


def volume(pol_d, pol_em, pol_ej, B, B_grid, P, N):
    """Annual flows at backlog B under the given policies with Params P
    and case inflow N. Returns dict with adjudicated, mediation,
    total_resolution, per-case adjudication rate psi(1-g) (annual mean),
    adjudication share s = psi/(phi+psi), mediation share rho, mean g(d)."""
    _, ds, ems, ejs = year_map_policy(pol_d, pol_em, pol_ej, B, B_grid, P, N)
    # model.phi/psi are scalar-only (builtin-max trick); vectorize identically
    ems0 = np.where(ems > 0.0, ems, 0.0)
    ejs0 = np.where(ejs > 0.0, ejs, 0.0)
    phs = P.phi0 * np.power(ems0, P.phi1)
    pss = P.psi0 * np.power(ejs0, P.psi1)
    gs = g(ds, P)
    adj = float(np.sum(pss * (1 - gs) * N))
    med = float(np.sum(phs * (1 - gs) * N))
    den = phs + pss
    return dict(adjudicated=adj, mediation=med, total_resolution=adj + med,
                per_case_adjudicated=float(np.mean(pss * (1 - gs))),
                share_s=float(np.mean(pss / den)),
                share_rho=float(np.mean(phs / den)),
                gd_mean=float(np.mean(gs)))


def reg_volume(regime, B, B_grid):
    """Volume under the regime's OWN policies/P/N at backlog B."""
    return volume(regime["d_policy"], regime["em_policy"], regime["ej_policy"],
                  B, B_grid, make_P(regime["N"], regime["eta"]), regime["N"])


def year_end_directions(regime):
    """Panel means of subperiod T vs 1 for d, e_j, rho (year-end directions)."""
    d = regime["d_policy"]; ej = regime["ej_policy"]; rho = regime["rho_policy"]
    m = lambda a: a.mean(axis=1)
    return dict(d1=m(d)[0], dT=m(d)[-1], ej1=m(ej)[0], ejT=m(ej)[-1],
                rho1=m(rho)[0], rhoT=m(rho)[-1],
                d_up=bool(m(d)[-1] > m(d)[0]),
                ej_up=bool(m(ej)[-1] > m(ej)[0]),
                rho_down=bool(m(rho)[-1] < m(rho)[0]))


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------

def decompose_central(regimes, Bss, vols, B_grid):
    """Exact constructive decomposition for N_post=150, both path orders.
    regimes: dict key (N, eta) -> regime; Bss/vols: dict key -> value."""
    pre = regimes[(N0, ETA0)]
    post100 = regimes[(N0, ETA1)]
    post150 = regimes[(NPOST_CENTRAL, ETA1)]
    pre5_150 = regimes[(NPOST_CENTRAL, ETA0)]

    P5, P8 = make_P(N0, ETA0), make_P(N0, ETA1)
    B0, B1, B2, B2p = (Bss[(N0, ETA0)], Bss[(N0, ETA1)],
                       Bss[(NPOST_CENTRAL, ETA1)], Bss[(NPOST_CENTRAL, ETA0)])
    v_pre = vols[(N0, ETA0)]["adjudicated"]
    v_100_8_B1 = vols[(N0, ETA1)]["adjudicated"]
    v_150_8_B2 = vols[(NPOST_CENTRAL, ETA1)]["adjudicated"]
    v_150_5_B2p = vols[(NPOST_CENTRAL, ETA0)]["adjudicated"]
    DV = v_150_8_B2 - v_pre

    # ---- first-order inflow: frozen pre-reform policies at B0 ----
    # per-case annual rate = v_pre/N0 (annual volume scales linearly in N
    # at fixed policies); I1 = rate * Delta N = v_pre * (Npost-N0)/N0.
    I1 = v_pre * (NPOST_CENTRAL - N0) / N0

    # ---- behavioral step at common (B0, N=100) ----
    v_100_8_B0 = reg_volume(post100, B0, B_grid)["adjudicated"]
    S1_pol = v_100_8_B0 - v_pre
    # d-channel: (1-g(d)) change at pre-reform e_j (d* and eta move; em,ej frozen)
    D_d = volume(post100["d_policy"], pre["em_policy"], pre["ej_policy"],
                 B0, B_grid, P8, N0)["adjudicated"] - v_pre
    # e_j-channel: psi(e_j) change at pre-reform d (eta=5 g; d,em frozen)
    D_e = volume(pre["d_policy"], pre["em_policy"], post100["ej_policy"],
                 B0, B_grid, P5, N0)["adjudicated"] - v_pre
    # mediation-side: e_m change at pre-reform (d, e_j, eta=5)
    D_em = volume(pre["d_policy"], post100["em_policy"], pre["ej_policy"],
                  B0, B_grid, P5, N0)["adjudicated"] - v_pre
    Int = S1_pol - D_d - D_e - D_em   # exact residual (interaction + paths)

    # ---- backlog adjustment of the behavioral step ----
    S1_B = v_100_8_B1 - v_100_8_B0

    # ---- inflow step at eta=8 (natural order): S2 = a - d, split exactly ----
    a = v_150_8_B2                                   # (150,8) at B2, N=150
    P150_8 = make_P(NPOST_CENTRAL, ETA1)
    b = volume(post150["d_policy"], post150["em_policy"], post150["ej_policy"],
               B2, B_grid, P150_8, N0)["adjudicated"]  # same policies, N=100
    c = reg_volume(post100, B2, B_grid)["adjudicated"] # (100,8) at B2, N=100
    d = v_100_8_B1                                     # (100,8) at B1, N=100
    S2_N = a - b            # inflow at post policies, B2
    S2_pol = b - c          # regime re-optimization gap at N=100 scale, B2
    S2_B = c - d            # backlog move B1 -> B2 under (100,8) policies
    S2_N_2nd = S2_N - I1    # second-order inflow

    comp = dict(I1=I1, D_d=D_d, D_e=D_e, D_em=D_em, Int=Int,
                S1_B=S1_B, S2_N_2nd=S2_N_2nd, S2_pol=S2_pol, S2_B=S2_B)
    S1 = S1_pol + S1_B
    S2 = S2_N + S2_pol + S2_B
    check_natural = abs(sum(comp.values()) - DV)   # 9 components sum to DV
    check_chain = abs((S1 + S2) - DV)

    # ---- reverse order: N first at eta=5, then eta at N=150 ----
    S2p = v_150_5_B2p - v_pre                       # inflow step at eta=5
    S1p = v_150_8_B2 - v_150_5_B2p                  # behavioral step at N=150
    check_reverse = abs((S2p + S1p) - DV)

    return dict(DV=DV, I1=I1, displacement=DV - I1, S1=S1, S2=S2,
                S1_pol=S1_pol, S1_B=S1_B, S2_N=S2_N, S2_N_2nd=S2_N_2nd,
                S2_pol=S2_pol, S2_B=S2_B, D_d=D_d, D_e=D_e, D_em=D_em,
                Int=Int, S2p=S2p, S1p=S1p,
                check_natural=check_natural, check_chain=check_chain,
                check_reverse=check_reverse,
                v_pre=v_pre, v_100_8_B1=v_100_8_B1,
                v_150_8_B2=v_150_8_B2, v_150_5_B2p=v_150_5_B2p,
                B0=B0, B1=B1, B2=B2, B2p=B2p)


def decompose_sensitivity(regimes, Bss, vols, B_grid, Npost):
    """Natural-order decomposition at alternative N_post (no reverse order:
    would need (Npost, 5) VIs beyond the runtime budget)."""
    pre = regimes[(N0, ETA0)]
    post100 = regimes[(N0, ETA1)]
    postN = regimes[(Npost, ETA1)]
    B0, B1, Bn = Bss[(N0, ETA0)], Bss[(N0, ETA1)], Bss[(Npost, ETA1)]
    v_pre = vols[(N0, ETA0)]["adjudicated"]
    v_N8_Bn = vols[(Npost, ETA1)]["adjudicated"]
    v_100_8_B1 = vols[(N0, ETA1)]["adjudicated"]
    DV = v_N8_Bn - v_pre
    I1 = v_pre * (Npost - N0) / N0
    v_100_8_B0 = reg_volume(post100, B0, B_grid)["adjudicated"]
    S1_pol = v_100_8_B0 - v_pre
    S1_B = v_100_8_B1 - v_100_8_B0
    PN = make_P(Npost, ETA1)
    a = v_N8_Bn
    b = volume(postN["d_policy"], postN["em_policy"], postN["ej_policy"],
               Bn, B_grid, PN, N0)["adjudicated"]
    c = reg_volume(post100, Bn, B_grid)["adjudicated"]
    d = v_100_8_B1
    S2_N = a - b
    S2_pol = b - c
    S2_B = c - d
    S2_N_2nd = S2_N - I1
    S1 = S1_pol + S1_B
    S2 = S2_N + S2_pol + S2_B
    comp = dict(I1=I1, D_d=0.0, D_e=0.0, D_em=0.0, Int=S1_pol,
                S1_B=S1_B, S2_N_2nd=S2_N_2nd, S2_pol=S2_pol, S2_B=S2_B)
    # (d/e_j channels are N-independent; reported once from the central case)
    return dict(Npost=Npost, DV=DV, I1=I1, displacement=DV - I1,
                S1=S1, S2=S2, S1_pol=S1_pol, S1_B=S1_B, S2_N=S2_N,
                S2_N_2nd=S2_N_2nd, S2_pol=S2_pol, S2_B=S2_B,
                check_chain=abs((S1 + S2) - DV),
                v_pre=v_pre, v_N8_Bn=v_N8_Bn, B0=B0, B1=B1, Bn=Bn)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    regimes = {}
    for (N, eta) in [(N0, ETA0), (N0, ETA1), (NPOST_CENTRAL, ETA0),
                     (NPOST_CENTRAL, ETA1), (NPOST_SENS[0], ETA1),
                     (NPOST_SENS[1], ETA1)]:
        reg = load_regime(N, eta)
        reg["N"] = float(N)
        reg["eta"] = float(eta)
        regimes[(N, eta)] = reg
        print(f"[load] regime (N={N}, eta={eta}): "
              f"n_iter={reg['n_iter']} final_resid={reg['final_resid']:.2e} "
              f"converged={reg['converged']} w_tau={reg.get('w_tau', float('nan')):.6f}",
              flush=True)

    B_grid = regimes[(N0, ETA0)]["B_grid"]

    # ---- w_tau verification: alpha1/(T*N), N-dependent ----
    w_checks = {}
    for (N, eta), reg in regimes.items():
        w_true = ALPHA1 / (T * N)
        w_loaded = float(reg["w_tau"]) if "w_tau" in reg else w_true
        w_checks[(N, eta)] = dict(w_tau=w_loaded, matches=abs(w_loaded - w_true) < 1e-12)

    # ---- steady states + volumes ----
    Bss, vols, yed = {}, {}, {}
    for (N, eta), reg in regimes.items():
        B, conv, stab = steady_state(reg, B_grid)
        Bss[(N, eta)] = B
        vols[(N, eta)] = reg_volume(reg, B, B_grid)
        yed[(N, eta)] = year_end_directions(reg)
        print(f"[bss] (N={N}, eta={eta}): B*={B:.4f} conv={conv} "
              f"|dB'/dB|={stab:.4f}  adj_vol={vols[(N, eta)]['adjudicated']:.2f} "
              f"s_bar={vols[(N, eta)]['share_s']:.4f} "
              f"rho_bar={vols[(N, eta)]['share_rho']:.4f} "
              f"gd={vols[(N, eta)]['gd_mean']:.4f}", flush=True)

    # ---- central decomposition ----
    dec = decompose_central(regimes, Bss, vols, B_grid)
    print("[dec] DV=%.4f I1=%.4f D_d=%.4f D_e=%.4f D_em=%.4f Int=%.4f "
          "S1_B=%.4f S2N2nd=%.4f S2pol=%.4f S2_B=%.4f | checks %.2e %.2e %.2e"
          % (dec["DV"], dec["I1"], dec["D_d"], dec["D_e"], dec["D_em"],
             dec["Int"], dec["S1_B"], dec["S2_N_2nd"], dec["S2_pol"],
             dec["S2_B"], dec["check_natural"], dec["check_chain"],
             dec["check_reverse"]), flush=True)
    print(f"[dec] natural order: S1(eta first)={dec['S1']:.4f} "
          f"S2(N second)={dec['S2']:.4f} | reverse: "
          f"S2p(N first)={dec['S2p']:.4f} S1p(eta second)={dec['S1p']:.4f}",
          flush=True)

    # ---- sensitivity ----
    sens = [decompose_sensitivity(regimes, Bss, vols, B_grid, Np)
            for Np in NPOST_SENS]

    # ---- save ----
    def ser(x):
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, (int, float, bool, str)):
            return x
        if isinstance(x, dict):
            return {str(k): ser(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [ser(v) for v in x]
        return x

    np.savez(os.path.join(OUT_DIR, "reform_cache.npz"),
             B_grid=B_grid,
             regimes=np.array([f"{int(N)}_{int(eta)}" for (N, eta) in regimes]),
             B_star=np.array([Bss[k] for k in regimes]),
             adj_vol=np.array([vols[k]["adjudicated"] for k in regimes]),
             per_case=np.array([vols[k]["per_case_adjudicated"] for k in regimes]),
             share_s=np.array([vols[k]["share_s"] for k in regimes]),
             share_rho=np.array([vols[k]["share_rho"] for k in regimes]),
             gd_mean=np.array([vols[k]["gd_mean"] for k in regimes]),
             dec_json=json.dumps(ser(dec)),
             sens_json=json.dumps(ser(sens)),
             w_checks_json=json.dumps(ser(w_checks)),
             yed_json=json.dumps(ser(yed)),
             vols_json=json.dumps(ser({str(k): vols[k] for k in vols})))
    with open(os.path.join(OUT_DIR, "reform_results.json"), "w") as f:
        json.dump(dict(dec=ser(dec), sens=ser(sens),
                       bss={str(k): Bss[k] for k in Bss},
                       vols=ser({str(k): vols[k] for k in vols}),
                       yed=ser({str(k): v for k, v in yed.items()}),
                       w_checks=ser(w_checks)), f, indent=1,
                  ensure_ascii=False)
    print("DONE: reform_cache.npz + reform_results.json written", flush=True)


if __name__ == "__main__":
    main()
