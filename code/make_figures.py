"""Figures for the research note (reads results/ produced by run_regime.py and
reform_experiment.py; writes figures/*.pdf and figures/*.png).

Fig 1  Within-year profiles along the steady state, pre vs post reform.
Fig 2  Steady-state adjudication share and dismissal rate vs post-reform inflow.
Fig 3  Exact decomposition of the change in annual adjudicated volume.
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
from reform_experiment import (load_regime, make_P, steady_state,  # noqa: E402
                               year_map_policy)
from solver.model import phi, psi, g  # noqa: E402

ROOT = os.path.dirname(CODE_DIR)
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

# Palette: categorical slots 1-2 (pre = blue, post = orange); diverging poles
# blue/red for positive/negative contributions; neutral ink for totals.
PRE, POST = "#2a78d6", "#eb6834"
POS, NEG, TOT = "#2a78d6", "#e34948", "#52514e"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9.5,
    "axes.labelsize": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
    "grid.linewidth": 0.6, "legend.frameon": False, "figure.dpi": 150,
})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight",
                    dpi=300 if ext == "png" else None)
    plt.close(fig)


def steady_path(N, eta):
    """Within-year path (d, e_j, mediation share) starting from the steady-state backlog."""
    reg = load_regime(N, eta)
    reg["N"], reg["eta"] = float(N), float(eta)
    B_grid = reg["B_grid"]
    Bss, _, _ = steady_state(reg, B_grid)
    P = make_P(N, eta)
    _, ds, ems, ejs = year_map_policy(reg["d_policy"], reg["em_policy"],
                                      reg["ej_policy"], Bss, B_grid, P, N)
    ph = np.array([phi(e, P) for e in ems])
    ps = np.array([psi(e, P) for e in ejs])
    rho = ph / (ph + ps)
    return dict(B=Bss, d=ds, ej=ejs, rho=rho, gd=np.array([g(d, P) for d in ds]))


def fig1():
    pre, post = steady_path(100, 5), steady_path(150, 8)
    tau = np.arange(1, 5)
    panels = [("d", "Dismissal effort $d_\\tau$"),
              ("ej", "Adjudication effort $e_{j,\\tau}$"),
              ("rho", "Mediation share $\\rho_\\tau$")]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    for ax in axes:
        ax.set_axisbelow(True)
    for ax, (k, title) in zip(axes, panels):
        for path, col, lab in ((pre, PRE, "Pre-reform"), (post, POST, "Post-reform")):
            ax.plot(tau, path[k], color=col, lw=2, marker="o", ms=4.5,
                    markeredgecolor="white", markeredgewidth=1.2, label=lab)
        ax.set_title(title, loc="left", color=INK)
        ax.set_xticks(tau, [f"Q{t}" for t in tau])
        ax.set_ylim(0, 1.0 if k == "rho" else None)
        ax.set_ylim(bottom=0)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 1.08))
    fig.text(0.5, -0.04, "Subperiod within the year", ha="center", color=INK2)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig1_within_year")
    return pre, post


def fig2():
    r = json.load(open(os.path.join(RES, "reform_results.json")))
    vols = r["vols"]
    Ns = [100.0, 125.0, 150.0, 200.0]
    s_post = [vols[f"({n}, 8.0)"]["share_s"] for n in Ns]
    g_post = [vols[f"({n}, 8.0)"]["gd_mean"] for n in Ns]
    s_pre, g_pre = vols["(100.0, 5.0)"]["share_s"], vols["(100.0, 5.0)"]["gd_mean"]
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.4))
    for ax in axes:
        ax.set_axisbelow(True)
    for ax, ys, y0, title in ((axes[0], s_post, s_pre, "Adjudication share of resolved cases"),
                              (axes[1], g_post, g_pre, "Dismissal rate $g(d)$")):
        ax.axhline(y0, color=PRE, lw=1.2, ls=(0, (4, 3)))
        ax.text(Ns[0], y0 + 0.03, "pre-reform", color=INK2, va="bottom", ha="left", fontsize=8)
        ax.plot(Ns, ys, color=POST, lw=2, marker="o", ms=4.5,
                markeredgecolor="white", markeredgewidth=1.2)
        ax.text(Ns[-1], ys[-1], " post-reform", color=INK2, va="center", ha="left", fontsize=8)
        ax.set_title(title, loc="left", color=INK)
        ax.set_xticks(Ns, [f"{int(n)}" for n in Ns])
        ax.set_xlim(95, 235)
        ax.set_ylim(0, 1)
    fig.text(0.5, -0.04, "Post-reform case inflow per subperiod (pre-reform = 100)",
             ha="center", color=INK2)
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig2_steady_state")


def fig3():
    r = json.load(open(os.path.join(RES, "reform_results.json")))
    dec = r["dec"]
    other = dec["DV"] - (dec["I1"] + dec["D_d"] + dec["D_e"] + dec["S2_N_2nd"])
    items = [("Inflow\n(1st order)", dec["I1"]),
             ("Dismissal\nchannel", dec["D_d"]),
             ("Adjudication-\neffort channel", dec["D_e"]),
             ("Inflow at post-\nreform rates", dec["S2_N_2nd"]),
             ("Interactions\n& other", other)]
    fig, ax = plt.subplots(figsize=(6.2, 2.7))
    ax.set_axisbelow(True)
    level = 0.0
    for i, (lab, v) in enumerate(items):
        lo, hi = sorted((level, level + v))
        ax.bar(i, hi - lo, bottom=lo, width=0.62, color=POS if v > 0 else NEG,
               edgecolor="white", linewidth=1.0)
        ax.text(i, hi + 1.2, f"{v:+.1f}", ha="center", va="bottom", color=INK2, fontsize=8)
        if i < len(items) - 1:
            ax.plot([i + 0.31, i + 0.69], [level + v] * 2, color=MUTED, lw=0.8)
        level += v
    n = len(items)
    lo, hi = sorted((0.0, level))
    ax.bar(n, hi - lo, bottom=lo, width=0.62, color=TOT, edgecolor="white", linewidth=1.0)
    ax.text(n, hi + 1.2, f"{level:+.1f}", ha="center", va="bottom", color=INK, fontsize=8,
            fontweight="bold")
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_xticks(range(n + 1), [lab for lab, _ in items] + ["Total change\n$\\Delta V$"],
                  fontsize=7.5)
    ax.set_ylabel("Change in annual adjudicated cases")
    ax.set_ylim(min(-60, level - 8), 30)
    fig.tight_layout()
    save(fig, "fig3_decomposition")
    return items, level


if __name__ == "__main__":
    pre, post = fig1()
    fig2()
    items, total = fig3()
    summary = {
        "steady_path_pre": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in pre.items()},
        "steady_path_post": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in post.items()},
        "decomposition_items": [(lab.replace("\n", " "), v) for lab, v in items],
        "decomposition_total": total,
    }
    with open(os.path.join(RES, "figure_data.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("figures written to", FIG)
