"""Matplotlib charts for PhishDrift (PNG bytes). No web framework dependency."""
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

matplotlib.rcParams["font.family"] = "sans-serif"

MODEL_ORDER = ["NB", "SVM", "RF"]
MODEL_LABEL = {"NB": "Naive Bayes", "SVM": "SVM", "RF": "Random Forest"}
POLICY_LABEL = {"frozen": "Frozen", "triggered": "Triggered retrain", "every2": "Every 2 years", "every1": "Every year"}
INK, MUTED, LINE, ACCENT = "#17211C", "#5F6B65", "#E3E1D8", "#14342B"
BAD, GOOD, WARN, GREY = "#B42318", "#1E7B4C", "#B7791F", "#A8A69E"
STATUS_COLOR = {"normal": GOOD, "abnormal drift": WARN, "retrain recommended": BAD}
MODEL_COLOR = {"NB": INK, "SVM": "#2B7A6B", "RF": WARN}


def png(fig):
    """Figure -> PNG bytes."""
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    return buf.getvalue()


def _ax(figsize=(6.4, 3.3)):
    fig, ax = plt.subplots(figsize=figsize, dpi=140)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    ax.grid(axis="y", color="#EFEDE6", linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def fig_drift(w, year, thr, warn, years):
    fig, ax = _ax()
    ymax = max(float(w.abn_pct.max()), thr) * 100 * 1.4
    ax.axhspan(thr * 100, ymax, color=BAD, alpha=0.07, lw=0)
    ax.axhline(thr * 100, color=BAD, ls="--", lw=1)
    ax.text(years[-1] + 0.45, thr * 100, "retrain threshold ", ha="right", va="bottom", fontsize=7, color=BAD)
    shown = w[w.year <= year]
    ax.plot(w.year, w.abn_pct * 100, color=LINE, lw=1.2, ls=":", zorder=1)
    ax.plot(shown.year, shown.abn_pct * 100, color=INK, lw=1.8, zorder=2)
    ax.scatter(shown.year, shown.abn_pct * 100, c=[STATUS_COLOR[s] for s in shown.status], s=46, zorder=3,
               edgecolor="white", linewidth=1.2)
    cur = shown[shown.year == year].iloc[0]
    ax.annotate(f"{cur.abn_pct * 100:.0f}%", (year, cur.abn_pct * 100), textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=8, fontweight="bold", color=INK)
    ax.set_xticks(years); ax.set_xlim(years[0] - 0.5, years[-1] + 0.5); ax.set_ylim(0, ymax)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(100, decimals=0))
    ax.set_ylabel("abnormal samples", fontsize=8, color=MUTED)
    return fig


def fig_recall(frozen, trig, control, years, year, lo):
    fig, ax = _ax()
    fz = frozen[frozen.year <= year]
    ax.axhline(control, color=GREY, ls="--", lw=1)
    ax.text(years[0] - 0.4, control, "train-era control ", ha="left", va="bottom", fontsize=7, color=MUTED)
    ax.fill_between(fz.year, fz.phish_recall, control, where=(fz.phish_recall < control).values, color=BAD, alpha=0.10)
    ax.plot(fz.year, fz.phish_recall, "o-", color=INK, lw=1.8, ms=5, label="frozen model")
    if trig is not None:
        tg = trig[trig.year <= year]
        ax.plot(tg.year, tg.phish_recall, "s-", color=ACCENT, lw=1.8, ms=5, label="triggered retraining")
        nr = tg.n_retrains_so_far.values
        for i in range(1, len(nr)):
            if nr[i] > nr[i - 1]:
                ax.axvline(tg.year.values[i] - 0.5, color=ACCENT, ls=":", lw=1.2)
    ax.set_xticks(years); ax.set_xlim(years[0] - 0.5, years[-1] + 0.5); ax.set_ylim(lo, 1.012)
    ax.set_ylabel("phishing recall", fontsize=8, color=MUTED)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    return fig


def fig_hist(ps):
    fig, ax = _ax((6.4, 2.9))
    bins = np.linspace(-3, 5, 48)
    ax.axvspan(1, 5, color=BAD, alpha=0.05, lw=0)
    ax.hist(ps[ps.y == 0].Df.clip(-3, 5), bins=bins, alpha=0.7, color=GOOD, label="legit")
    ax.hist(ps[ps.y == 1].Df.clip(-3, 5), bins=bins, alpha=0.7, color=BAD, label="phishing")
    ax.axvline(1, color=INK, ls="--", lw=1)
    ax.text(1.08, ax.get_ylim()[1] * 0.92, "abnormal (Df > 1)", fontsize=7, color=INK)
    ax.set_xlabel("per-sample drift score Df (clipped to [-3, 5])", fontsize=8, color=MUTED)
    ax.set_ylabel("emails", fontsize=8, color=MUTED)
    ax.legend(fontsize=7, frameon=False)
    return fig


def fig_terms(t):
    fig, ax = _ax((5.0, 2.8))
    ax.grid(axis="x", color="#EFEDE6", linewidth=0.8); ax.grid(axis="y", visible=False)
    if t.get("directional"):
        items = [(a, b, BAD) for a, b in t["phish"]] + [(a, b, GOOD) for a, b in t["legit"]]
        items = sorted(items, key=lambda r: r[1])
        ax.set_xlabel("toward legit   <-   0   ->   toward phishing", fontsize=7.5, color=MUTED)
    else:
        items = sorted([(a, b, INK) for a, b in t.get("influential", [])], key=lambda r: r[1])
        ax.set_xlabel("importance (Random Forest gives no direction)", fontsize=7.5, color=MUTED)
    if not items:
        ax.text(0.5, 0.5, "no informative terms", ha="center", va="center", transform=ax.transAxes, color=GREY)
        return fig
    ax.barh([i[0] for i in items], [i[1] for i in items], color=[i[2] for i in items], height=0.62)
    ax.axvline(0, color=GREY, lw=0.8)
    ax.tick_params(axis="y", labelsize=8)
    return fig


def fig_evidence_recall(piv, ctrl):
    fig, ax = _ax((7.4, 3.5))
    for m in piv.columns:
        c = MODEL_COLOR.get(m, INK)
        ax.plot(piv.index, piv[m], "o-", color=c, lw=1.8, ms=5, label=MODEL_LABEL.get(m, m))
        ax.axhline(ctrl[m], color=c, ls=":", lw=1, alpha=0.7)
    ax.set_xticks(list(piv.index)); ax.set_ylabel("phishing recall", fontsize=8, color=MUTED)
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    return fig


def fig_policy(sim_sum):
    fig, ax = _ax((7.4, 3.2))
    ax.grid(axis="x", color="#EFEDE6", linewidth=0.8); ax.grid(axis="y", visible=False)
    policies = [p for p in ["frozen", "triggered", "every2", "every1"] if p in set(sim_sum.policy)]
    models = [m for m in MODEL_ORDER if m in set(sim_sum.model)]
    for i, m in enumerate(models):
        s = sim_sum[sim_sum.model == m].set_index("policy")
        ys = [policies.index(p) + (i - (len(models) - 1) / 2) * 0.22 for p in s.index if p in policies]
        xs = [s.loc[p, "mean_recall"] for p in s.index if p in policies]
        ax.scatter(xs, ys, s=64, color=MODEL_COLOR.get(m, INK), label=MODEL_LABEL.get(m, m), zorder=3)
    ax.set_yticks(range(len(policies))); ax.set_yticklabels([POLICY_LABEL[p] for p in policies], fontsize=8)
    ax.invert_yaxis(); ax.set_xlabel("mean phishing recall, 2019-25", fontsize=8, color=MUTED)
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    return fig


