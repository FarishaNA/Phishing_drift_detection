#!/usr/bin/env python3
"""PhishDrift web app. Plain HTML/CSS/JS front end served by the Python standard library.

Run:   python server.py            (opens http://localhost:8000)
       python server.py --port 8080 --no-browser

Needs ./artifacts/ = unzipped fesad_outputs from notebook 02. Nothing is trained here; saved models and tables are loaded.
No extra packages: only what notebook 02 and pipeline.py already need (pandas, numpy, scikit-learn, matplotlib, joblib).
"""
import argparse
import ast
import html as _html
import json
import mimetypes
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import joblib
import numpy as np
import pandas as pd

import charts
from pipeline import parse_eml, predict_email, split_raw

BASE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(BASE, "web")
ART = os.environ.get("ARTIFACT_DIR", os.path.join(BASE, "artifacts"))
LOCK = threading.Lock()          # matplotlib and model state are not thread safe; one request at a time is plenty

MODEL_ORDER = charts.MODEL_ORDER
MODEL_LABEL = charts.MODEL_LABEL
POLICY_LABEL = charts.POLICY_LABEL

PROJECT = dict(
    brand="PhishDrift",
    user="Farisha",
    version="v1.0.0",
    course="MCA Final Project, Mar Athanasius College of Engineering, Kothamangalam",
    guide="Prof. Manu John",
    team=["Farisha N A", "Sulfath P M", "Adithya P Ganesh"],   # edit as needed
)

SAMPLES = {
    "Phishing: mailbox quota": dict(truth="phishing",
        subject="Action required: your mailbox is full",
        sender="IT Support <support@mail-quota-update.net>",
        body=("Dear user, your mailbox has exceeded its storage quota and will be suspended within 24 hours. "
              "Click here to verify your account immediately: http://mail-quota-update.net/verify?id=8821 "
              "Failure to confirm will result in permanent loss of access. Regards, IT Administrator")),
    "Phishing: shared document": dict(truth="phishing",
        subject="Shared document: payroll adjustments",
        sender="HR Portal <noreply@docs-share-secure-login.com>",
        body=("A document was shared with you. Review and sign in to access: https://docs-share-secure-login.com/open/x7 "
              "This link will expire in 24 hours. Please confirm your credentials to view the file.")),
    "Legit: internal mail": dict(truth="legit",
        subject="Re: gas deal meeting",
        sender="John Smith <john.smith@enron.com>",
        body=("Mike, let me know if we can move the meeting to Thursday. I forwarded the contract numbers to Jeff. "
              "Talk to you tomorrow. Thanks, John")),
    "Legit: order mail": dict(truth="legit",
        subject="Your order has shipped",
        sender="orders@shop.example.com",
        body=("Hi, your order #48213 has shipped and should arrive on Friday. Track your package: "
              "https://shop.example.com/track/48213 . Thanks for shopping with us.")),
    "Legit: college notice": dict(truth="legit",
        subject="Internal exam timetable published",
        sender="office@college.example.edu",
        body=("Dear students, the internal exam timetable for the S3 MCA batch has been published on the college "
              "website. Please check the notice board for details. Regards, Office")),
}

# ============================================================================ data
_STATES = {}
_TABLES = {}


def models_available():
    return [m for m in MODEL_ORDER if os.path.exists(os.path.join(ART, f"state_{m}_baseline.joblib"))]


def get_state(name):
    if name not in _STATES:
        _STATES[name] = joblib.load(os.path.join(ART, f"state_{name}_baseline.joblib"))
    return _STATES[name]


def get_tables():
    """Load result tables once. Raises FileNotFoundError if the notebook outputs are missing."""
    if not _TABLES:
        p = lambda f: os.path.join(ART, f)
        t = {
            "walk": pd.read_csv(p("walk_frozen_results.csv")),
            "per_sample": pd.read_csv(p("per_sample_Df_frozen.csv")),
            "sim": pd.read_csv(p("policy_simulation.csv")),
            "sim_sum": pd.read_csv(p("policy_summary.csv")),
            "det": pd.read_csv(p("detector_eval.csv")),
            "corr": pd.read_csv(p("drift_vs_recall_spearman.csv")),
        }
        with open(p("config.json")) as f:
            t["cfg"] = json.load(f)
        t["modern"] = pd.read_csv(p("modern_legit_fp.csv")) if os.path.exists(p("modern_legit_fp.csv")) else None
        _TABLES.update(t)
    return _TABLES


# ============================================================================ html building blocks
esc = lambda s: _html.escape(str(s), quote=True)

LOGO = open(os.path.join(WEB, "logo.svg"), encoding="utf-8").read().strip()

_NAV_ICONS = {
    "home": '<path d="M3 11l9-7.5 9 7.5"/><path d="M5 10v10h5v-6h4v6h5V10"/>',
    "drift": '<path d="M3 12h4l3-7 4 14 3-7h4"/>',
    "evidence": '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/><path d="M9 14.5l2 2 4-4"/>',
    "history": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    "about": '<circle cx="12" cy="12" r="9"/><path d="M12 11v5"/><path d="M12 7.6v.1"/>',
}
NAV = [("home", "/", "Home"), ("drift", "/drift", "Drift Monitor"), ("evidence", "/evidence", "Evidence"),
       ("history", "/history", "History"), ("about", "/about", "About")]

ICON_SHIELD = '<svg viewBox="0 0 24 24"><path d="M12 3l8 3v6c0 4.5-3.2 7.8-8 9-4.8-1.2-8-4.5-8-9V6l8-3z"/><path d="M8.8 12l2.2 2.2 4.2-4.4"/></svg>'
ICON_MAIL = '<svg viewBox="0 0 24 24"><rect x="3" y="5" width="18" height="14" rx="2.5"/><path d="M3.5 7.5l8.5 6 8.5-6"/></svg>'
ICON_UPLOAD = '<svg viewBox="0 0 24 24"><path d="M12 15V4"/><path d="M7.5 8.5L12 4l4.5 4.5"/><path d="M5 15v3a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-3"/></svg>'
ICON_REFRESH = '<svg viewBox="0 0 24 24"><path d="M20 11a8 8 0 1 0-2.3 5.7"/><path d="M20 4v7h-7"/></svg>'
ICON_SEARCH = '<svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="6.5"/><path d="M20 20l-4.2-4.2"/></svg>'
ICON_MODELS = '<svg viewBox="0 0 24 24"><path d="M12 3l9 5-9 5-9-5 9-5z"/><path d="M3 12l9 5 9-5"/><path d="M3 16l9 5 9-5"/></svg>'
ICON_DRIFT = '<svg viewBox="0 0 24 24"><path d="M3 12h4l3-7 4 14 3-7h4"/></svg>'


def layout(active, body, title="PhishDrift", scripts=""):
    nav = "".join(
        f'<a href="{href}" class="{"active" if key == active else ""}"><svg viewBox="0 0 24 24">{_NAV_ICONS[key]}</svg>{label}</a>'
        for key, href, label in NAV)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)} | {PROJECT['brand']}</title>
<link rel="icon" type="image/svg+xml" href="/static/logo.svg">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<link rel="stylesheet" href="/static/style.css"></head>
<body data-page="{active}">
<aside class="sidebar">
  <a class="brand" href="/">{LOGO}<div><div class="nm">Phish<b>Drift</b></div><div class="sub">Email Classifier</div></div></a>
  <nav class="nav">{nav}</nav>
  <div class="tip">{ICON_SHIELD}<div><div class="t1">Think before you click.</div><div class="t2">Stay safe. Stay aware.</div></div></div>
</aside>
<div class="main">
  <div class="topbar"><div class="user"><span class="av">{esc(PROJECT['user'][0])}</span>{esc(PROJECT['user'])}</div></div>
  <div class="content">{body}</div>
  <div class="footer"><span>{PROJECT['brand']} &nbsp;{PROJECT['version']}</span><span>Research prototype, not a security product</span></div>
</div>
{scripts}<script src="/static/main.js"></script>
</body></html>"""


def pd_script(extra=None):
    data = dict(modelLabel=MODEL_LABEL)
    data.update(extra or {})
    return "<script>window.PD = " + json.dumps(data).replace("</", "<\\/") + ";</script>"


def header(title, subtitle):
    return f'<h1 class="wtitle">{esc(title)}</h1><p class="wsub">{subtitle}</p>'


def callout(text, tone="info"):
    return f'<div class="callout {tone}">{text}</div>'


def missing_page(active, title):
    return layout(active, header(title, "") + callout(
        f"<b>Notebook outputs not found</b> in <b>{esc(ART)}</b>. Unzip <b>fesad_outputs.zip</b> into the "
        "<b>artifacts</b> folder next to server.py, then refresh.", "bad"), title)


def df_html(df):
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return f"{v:.3f}"
        return esc(v)
    head = "".join(f"<th>{esc(c)}</th>" for c in df.columns)
    rows = "".join("<tr>" + "".join(f"<td>{fmt(v)}</td>" for v in r) + "</tr>" for r in df.itertuples(index=False))
    return f'<table class="dt"><tr>{head}</tr>{rows}</table>'


# ============================================================================ pages
def page_home():
    opts = '<option value="">Choose an example email</option>' + "".join(
        f'<option value="{esc(k)}">{esc(k)}</option>' for k in SAMPLES)
    body = f"""
<h1 class="wtitle">Welcome to PhishDrift</h1>
<p class="wsub">Detect phishing attempts using machine learning.<br><span class="s2">Paste an email below or upload a file to check if it is suspicious.</span></p>
<div class="grid-2">
  <div>
    <section class="card" id="input-card">
      <div class="tabs">
        <button class="tab active" data-tab="paste">{ICON_MAIL}Paste Email</button>
        <button class="tab" data-tab="upload">{ICON_UPLOAD}Upload File</button>
      </div>
      <div id="pane-paste">
        <label class="lbl" for="example">Try an example</label>
        <select id="example">{opts}</select>
        <div id="truth" class="truth hidden"></div>
        <label class="lbl" for="subject">Subject</label>
        <input type="text" id="subject" placeholder="Subject line" autocomplete="off">
        <label class="lbl" for="sender">From</label>
        <input type="text" id="sender" placeholder="Name &lt;user@domain.com&gt;" autocomplete="off">
        <label class="lbl" for="raw">Message</label>
        <div class="ta-wrap"><textarea id="raw" maxlength="5000" placeholder="Paste the email content here..."></textarea><span id="count">0/5000</span></div>
        <div class="hint" style="margin-top:6px">If your paste starts with Subject: and From: lines and the boxes above are empty, they are read automatically.</div>
      </div>
      <div id="pane-upload" class="hidden">
        <label class="drop" id="drop"><input type="file" id="file" accept=".eml,.txt" hidden>
          <div class="ic">{ICON_UPLOAD}</div><div><b>Choose a file</b> or drag it here</div><div class="hint">.eml or .txt, up to 200 KB</div></label>
        <div id="file-info" class="hint"></div>
      </div>
      <div class="actions">
        <button class="btn" id="clear" type="button">{ICON_REFRESH}Clear</button>
        <button class="btn primary" id="classify" type="button">{ICON_SEARCH}Classify Email</button>
      </div>
    </section>
    <section class="feats">
      <div><div class="ic">{ICON_MODELS}</div><div class="ft">Three classifiers</div><div class="fd">Naive Bayes, SVM and Random Forest judge every email side by side.</div></div>
      <div><div class="ic">{ICON_DRIFT}</div><div class="ft">Drift aware</div><div class="fd">Each email gets a drift score that flags unfamiliar input.</div></div>
      <div><div class="ic">{ICON_REFRESH}</div><div class="ft">Retrains when needed</div><div class="fd">The monitor shows when drift calls for retraining.</div></div>
    </section>
  </div>
  <aside>
    <section class="card">
      <div class="rp-title">{ICON_SHIELD}Classification Result</div>
      <div id="result"><div class="empty"><div class="ic">{ICON_MAIL}</div><div class="e1">No email analyzed yet</div>
        <div class="e2">Paste an email or upload a file to get classification results.</div></div></div>
    </section>
  </aside>
</div>
<p class="note"><b>Limitation.</b> The legitimate class is Enron-era corporate mail, so modern automated mail (receipts, certificates,
notifications) can be flagged incorrectly. See Evidence for the held-out test.</p>"""
    return layout("home", body, "Home", pd_script(dict(samples=SAMPLES)))


def page_drift():
    try:
        T = get_tables()
    except FileNotFoundError:
        return missing_page("drift", "Drift Monitor")
    walk = T["walk"]
    models = [m for m in MODEL_ORDER if m in set(walk.model)]
    years = sorted(int(y) for y in walk.year.unique())
    seg_model = "".join(f'<button data-v="{m}" class="{"on" if i == 0 else ""}">{esc(MODEL_LABEL[m])}</button>' for i, m in enumerate(models))
    body = header("Drift Monitor", "Replay of the 2019 to 2025 batches. Models were trained on 2015-17 and calibrated on 2018; "
                                    "each year arrives as a new batch to score.") + f"""
<div class="controls">
  <div class="ctl"><div class="cl">Model</div><div class="seg" id="seg-model">{seg_model}</div></div>
  <div class="ctl"><div class="cl">Model state</div><div class="seg" id="seg-state"><button data-v="frozen" class="on">Frozen</button><button data-v="retrained">With triggered retraining</button></div></div>
</div>
<div class="controls">
  <button class="btn small" id="prev" type="button">Previous</button><button class="btn small" id="next" type="button">Next</button><button class="btn small" id="reset" type="button">Reset</button>
  <div class="stepper"><span class="hint">Batch year</span><input type="range" id="yr-slider" min="0" max="{len(years) - 1}" value="0"><span class="yr" id="yr-label">{years[0]}</span></div>
</div>
<div id="status" class="statusline"></div>
<div class="grid-4" id="stats"></div>
<div class="grid-2c" style="margin-top:18px">
  <div class="imgcard"><h3>Weighted drift by year</h3><img id="img-drift" alt="Weighted drift by year"></div>
  <div class="imgcard"><h3>Phishing recall by year</h3><img id="img-recall" alt="Phishing recall by year"></div>
</div>
<div class="tabsec" style="margin-top:22px">
  <div class="tabs" data-tabset><button class="tab active" data-pane="a">Per-sample drift</button><button class="tab" data-pane="b">Retraining policies</button><button class="tab" data-pane="c">How to read this</button></div>
  <div class="pane" data-pane="a"><div class="imgcard" style="max-width:640px"><img id="img-hist" alt="Per-sample drift histogram"></div><p class="hint" id="hist-cap"></p></div>
  <div class="pane hidden" data-pane="b"><div id="retrain-note" class="callout info"></div>
    <table class="dt"><tr><th>Policy</th><th>Mean recall</th><th>Min recall</th><th>Retrains</th></tr><tbody id="policy-rows"></tbody></table></div>
  <div class="pane hidden prose" data-pane="c"><ul>
    <li><b>Abnormal sample:</b> an email whose drift score <code>Df</code> exceeds 1, meaning it sits further toward the opposite class than the calibration year predicted.</li>
    <li><b>Status:</b> normal below the warning level, abnormal drift between warning and threshold, retrain recommended at or above the threshold. The threshold is the 2018 calibration level plus a margin.</li>
    <li><b>Recall gap:</b> independent evidence of drift. How far recall fell below the cross-validated 2015-17 level.</li>
    <li><b>This is a replay</b> of stored yearly batches, not a live email feed.</li></ul></div>
</div>"""
    return layout("drift", body, "Drift Monitor", pd_script(dict(models=models, years=years)))


def page_evidence():
    try:
        T = get_tables()
    except FileNotFoundError:
        return missing_page("evidence", "Evidence")
    walk, sim_sum, det, corr, cfg, modern = T["walk"], T["sim_sum"], T["det"], T["corr"], T["cfg"], T["modern"]

    piv = walk.pivot(index="year", columns="model", values="phish_recall")
    ctrl = {m: cfg["models"][m]["control_recall"] for m in piv.columns}
    rows = []
    for m in piv.columns:
        wm = walk[walk.model == m]
        rows.append({"Model": MODEL_LABEL.get(m, m), "Control recall": ctrl[m], "Mean frozen recall": piv[m].mean(),
                     "Mean drop (pts)": round(wm.recall_drop.mean() * 100, 1), "Worst drop (pts)": round(wm.recall_drop.max() * 100, 1),
                     "Years over margin": f"{int(wm.true_drift.sum())} of {len(wm)}"})
    t1 = (f'<p>Frozen models against their cross-validated train-era recall. Ground truth for drift: recall falls more than '
          f'<b>{cfg["drop_margin"] * 100:.0f} points</b> below that level, independent of the detector.</p>'
          '<div class="imgcard" style="max-width:760px"><img src="/chart/evidence_recall.png" alt="Frozen recall by year">'
          '<p class="hint">Dotted lines mark each model\'s train-era (cross-validated) recall.</p></div>' + df_html(pd.DataFrame(rows)))

    p = sim_sum.pivot(index="model", columns="policy", values="mean_recall")
    p = p[[c for c in ["frozen", "triggered", "every2", "every1"] if c in p.columns]]
    nr = sim_sum.pivot(index="model", columns="policy", values="n_retrains")
    pv = p.copy(); pv.columns = [POLICY_LABEL[c] for c in pv.columns]; pv.insert(0, "Model", [MODEL_LABEL.get(i, i) for i in pv.index])
    nv = nr[list(p.columns)].copy(); nv.columns = [POLICY_LABEL[c] for c in nv.columns]; nv.insert(0, "Model", [MODEL_LABEL.get(i, i) for i in nv.index])
    bullets = ""
    for m in p.index:
        r = p.loc[m]
        bullets += (f"<li><b>{esc(MODEL_LABEL.get(m, m))}</b>: yearly retraining {(r.get('every1', np.nan) - r.get('frozen', np.nan)) * 100:+.1f} pts mean recall "
                    f"vs frozen; triggered {(r.get('triggered', np.nan) - r.get('frozen', np.nan)) * 100:+.1f} pts with "
                    f"{int(nr.loc[m].get('triggered', 0))} retrain(s).</li>")
    t2 = ('<div class="imgcard" style="max-width:760px"><img src="/chart/policy.png" alt="Retraining policies"></div>'
          "<h4 style='margin:16px 0 4px'>Mean phishing recall</h4>" + df_html(pv) + "<h4 style='margin:4px 0 4px'>Retrains used</h4>" + df_html(nv) +
          f'<ul class="prose">{bullets}</ul>')

    dd = det.copy(); dd["model"] = dd.model.map(lambda m: MODEL_LABEL.get(m, m))
    b3 = ""
    for m in det.model.unique():
        x = det[det.model == m].set_index("detector")
        wr, cr = x[x.index.str.startswith("FeSAD")].iloc[0], x[x.index.str.startswith("Cosine")].iloc[0]
        b3 += (f"<li><b>{esc(MODEL_LABEL.get(m, m))}</b>: weighted flagged {int(wr.flagged_years)} years, caught {int(wr.TP)} of "
               f"{int(wr.true_drift_years)} real drift years, {int(wr.FP)} false alarms. Cosine baseline flagged {int(cr.flagged_years)} years.</li>")
    t3 = ("<p>Weighted FeSAD detector against the unweighted cosine-centroid baseline, scored against the real recall-drop "
          "ground truth. TP means flagged and real drift.</p>" + df_html(dd) + f'<ul class="prose">{b3}</ul>'
          "<p><b>Spearman correlation: drift signal vs recall drop</b> (7 yearly points per model, indicative only)</p>" + df_html(corr))

    if modern is None:
        t4 = callout("<b>Not run.</b> The legitimate class is Enron-era only, so modern legitimate mail is likely misflagged. "
                     "Export modern legit emails, add them in the notebook, and re-run.", "warn")
    else:
        mm = modern.copy()
        cols = {c.lower().strip(): c for c in mm.columns}
        model_col = cols.get("model")
        fp_col = next((cols[k] for k in ("false_positive_rate", "fp_rate", "false_positive", "fpr") if k in cols), None)
        if model_col is not None:
            mm[model_col] = mm[model_col].map(lambda x: MODEL_LABEL.get(x, x))
        if fp_col is not None:
            mm[fp_col] = pd.to_numeric(mm[fp_col], errors="coerce").map(lambda v: "" if pd.isna(v) else f"{v * 100:.1f}%")
            mm = mm.rename(columns={fp_col: "False-positive rate"})
        t4 = df_html(mm) + '<p class="hint">Share of held-out modern legitimate emails (real inbox, never trained on) flagged as phishing.</p>'

    t5 = """<ul class="prose">
<li>The legitimate class is Enron-era corporate mail. A classifier can learn "old vs new" or "casual vs formal" instead of legit vs phishing, so modern legitimate mail can be misflagged.</li>
<li>Only 7 yearly batches per model: correlations and detector hit rates are indicative, not significant.</li>
<li>Drift is mostly on the phishing side; the legitimate pool is static.</li>
<li>The retraining simulation assumes labels for each batch arrive after the batch is scored.</li>
<li>Adapted FeSAD: no genetic-algorithm feature selection; class-set distances use normalised class means; wrong-prediction statistics come from cross-validated training predictions.</li></ul>"""

    body = header("Evidence", "Computed from the notebook outputs. Updates when the notebook is re-run.") + f"""
<div class="tabsec">
  <div class="tabs" data-tabset><button class="tab active" data-pane="a">Is there drift?</button><button class="tab" data-pane="b">Does retraining help?</button>
    <button class="tab" data-pane="c">Does the detector detect?</button><button class="tab" data-pane="d">Real-inbox test</button><button class="tab" data-pane="e">Limitations</button></div>
  <div class="pane prose" data-pane="a">{t1}</div><div class="pane prose hidden" data-pane="b">{t2}</div><div class="pane prose hidden" data-pane="c">{t3}</div>
  <div class="pane prose hidden" data-pane="d">{t4}</div><div class="pane prose hidden" data-pane="e">{t5}</div>
</div>"""
    return layout("evidence", body, "Evidence", pd_script())


def page_history():
    body = header("History", "Emails analyzed in this browser. Stored locally on this computer, not on the server.") + """
<div id="hist"></div>
<div style="margin-top:12px"><button class="btn small hidden" id="clear-hist" type="button">Clear history</button></div>"""
    return layout("history", body, "History", pd_script())


def page_about():
    cfg = None
    try:
        cfg = get_tables()["cfg"]
    except FileNotFoundError:
        pass
    team = ", ".join(PROJECT["team"])
    if cfg:
        split = (f"<b>Train</b> {', '.join(str(y) for y in cfg['train_years'])} &nbsp;|&nbsp; <b>Calibrate</b> {cfg['calib_year']} &nbsp;|&nbsp; "
                 f"<b>Monitor</b> {cfg['walk_years'][0]} to {cfg['walk_years'][-1]}. Legitimate emails are cut into disjoint slices per batch "
                 f"({cfg['legit_per_phish']:.2f} legitimate per phishing email). Phishing: Nazario corpus. Legitimate: Enron corpus.")
        calib = cfg["calib_year"]
    else:
        split, calib = "Train 2015-17, calibrate 2018, monitor 2019-25.", 2018
    t1 = f"""<h3>PhishDrift</h3><p>Feature-drift monitoring and triggered retraining for phishing email classification. Three classifiers judge each email,
a FeSAD-style per-sample drift score flags unfamiliar input, and a monitor replays 2019 to 2025 to show when drift calls for retraining.</p>
<p><b>{esc(PROJECT['course'])}</b><br>Guide: {esc(PROJECT['guide'])}<br>Team: {esc(team)}</p>
<h3>References</h3><ul><li>Fernando &amp; Komninos. FeSAD framework for ransomware detection. <i>Computers &amp; Security</i>.</li>
<li>Fernando &amp; Komninos (2026). Evaluating and Combatting the Impact of Concept Drift on the Performance of Machine Learning-Based Phishing Detection Systems. arXiv:2606.11471.</li></ul>"""
    t2 = f"""<h3>Architecture</h3>
<table class="dt"><tr><th>Layer</th><th>This project</th><th>FeSAD paper</th></tr>
<tr><td>Feature layer</td><td>TF-IDF (5,000 terms, 1-2 grams) + 9 engineered features, fit on training data only</td><td>Genetic-algorithm feature selection</td></tr>
<tr><td>Drift calibration layer</td><td>Held-out calibration year ({calib}): expected drift per class, retrain threshold</td><td>Separate calibration dataset</td></tr>
<tr><td>Drift decision layer</td><td>Per-sample weighted distance, Df = D / De, % abnormal per batch, retrain trigger</td><td>Retrain after z abnormal samples</td></tr></table>
<h3>Drift score</h3>
<div class="formula">r<sub>i</sub> = d<sub>m</sub>(x<sub>i</sub>) / d<sub>b</sub>(x<sub>i</sub>), &nbsp; where d<sub>c</sub>(x) = 1 &minus; x&#770; &middot; &mu;&#770;<sub>c</sub> (cosine distance to the class mean)</div>
<div class="formula">w<sub>i</sub> = p<sub>i</sub> / p&#772;<sub>wrong,c</sub> &nbsp;&nbsp; dw<sub>i</sub> = log(r<sub>i</sub> / w<sub>i</sub>) if predicted phishing, &nbsp; log(r<sub>i</sub> &middot; w<sub>i</sub>) if predicted legit</div>
<div class="formula">D<sub>i</sub> = shift of dw<sub>i</sub> toward the opposite class, &nbsp; De<sub>c</sub> = mean|D|<sub>calib,c</sub> &middot; f<sub>c</sub>, &nbsp; Df<sub>i</sub> = D<sub>i</sub> / De<sub>c</sub></div>
<p>An email is <b>abnormal</b> when <code>Df &gt; 1</code>. A batch is flagged for retraining when its abnormal share reaches the calibration level plus a margin.</p>
<h3>Data and split</h3><p>{split}</p>
<h3>Deviations from the paper</h3><ol>
<li>Mean cosine distance to a class set uses normalised class means rather than pairwise distances.</li>
<li>Wrong-prediction probabilities and training drift come from cross-validated predictions, because a Random Forest fits its training set perfectly.</li>
<li>Drift is signed toward the opposite class; the average drift uses absolute values because the signed mean is zero by construction; the growth factor <code>f</code> is floored at 1.</li>
<li>The retrain threshold is user-defined, as in the paper. Default: calibration abnormal rate plus a margin.</li>
<li>No genetic algorithm. The feature set is justified by ablation.</li></ol>"""
    t3 = """<ul>
<li>The legitimate class is Enron-era corporate mail, so the classifier can learn "old vs new" or "casual vs formal" instead of legit vs phishing. Modern automated mail can be misflagged.</li>
<li>Only 7 yearly batches per model: correlations and detector hit rates are indicative, not significant.</li>
<li>Drift is mostly on the phishing side; the legitimate pool is static.</li>
<li>The retraining simulation assumes labels for each batch arrive after the batch is scored.</li>
<li>This is a research prototype, not a security product. Do not rely on it to decide whether a real email is safe.</li></ul>"""
    body = header("About", "The project, the method behind the drift score, and its limits.") + f"""
<div class="tabsec"><div class="tabs" data-tabset><button class="tab active" data-pane="a">Project</button><button class="tab" data-pane="b">Method</button><button class="tab" data-pane="c">Limitations</button></div>
<div class="pane prose" data-pane="a">{t1}</div><div class="pane prose hidden" data-pane="b">{t2}</div><div class="pane prose hidden" data-pane="c">{t3}</div></div>"""
    return layout("about", body, "About", pd_script())


# ============================================================================ api
def api_classify(payload):
    avail = models_available()
    if not avail:
        raise ValueError(f"No model files found in {ART}. Unzip the notebook outputs there.")
    text = str(payload.get("text", ""))
    if len(text) > 400000:
        raise ValueError("Input is too large.")
    if payload.get("kind") == "eml":
        subject, sender, body = parse_eml(text.encode("utf-8", errors="ignore"))
    else:
        subject, sender, body = split_raw(text)
        subject = str(payload.get("subject", "")).strip() or subject
        sender = str(payload.get("sender", "")).strip() or sender
    if not (body.strip() or subject.strip()):
        raise ValueError("Paste an email or upload a file first.")
    results = {}
    with LOCK:
        for m in avail:
            r = predict_email(subject, sender, body, get_state(m))
            r["features"] = {k: float(v) for k, v in r["features"].items()}
            results[m] = r
    return dict(order=avail, subject=subject, sender=sender, results=results)


def _ensure(model, year, T):
    walk = T["walk"]
    if model not in set(walk.model):
        raise ValueError("unknown model")
    years = sorted(int(y) for y in walk.year.unique())
    if year not in years:
        raise ValueError("unknown year")
    return years


def api_drift(q):
    T = get_tables()
    model = q.get("model", [""])[0]
    year = int(q.get("year", ["0"])[0])
    retrained = q.get("retrained", ["0"])[0] == "1"
    _ensure(model, year, T)
    walk, sim, sim_sum, cfg, ps = T["walk"], T["sim"], T["sim_sum"], T["cfg"], T["per_sample"]
    w = walk[walk.model == model].sort_values("year")
    cur = w[w.year == year].iloc[0]
    active = sim[(sim.model == model) & (sim.policy == ("triggered" if retrained else "frozen"))]
    arow = active[active.year == year].iloc[0]
    control = float(cur.control_recall)
    row = sim_sum[(sim_sum.model == model) & (sim_sum.policy == "triggered")]
    ev = ast.literal_eval(row.iloc[0].retrain_after_years) if len(row) else []
    p = ps[(ps.model == model) & (ps.year == year)]
    pol = sim_sum[sim_sum.model == model]
    return dict(
        status=str(cur.status), abn_pct=float(cur.abn_pct), thr=float(cur.retrain_thr), recall=float(arow.phish_recall),
        control=control, gap_pts=float((control - arow.phish_recall) * 100), drop_margin_pts=float(cfg["drop_margin"] * 100),
        mean_Df=float(cur.mean_Df), retrain_events=[int(e) for e in ev], n_emails=int(len(p)), pct_abnormal_emails=float((p.Df > 1).mean()),
        policies=[dict(policy=POLICY_LABEL.get(r.policy, r.policy), mean_recall=float(r.mean_recall), min_recall=float(r.min_recall),
                       n_retrains=int(r.n_retrains)) for r in pol.itertuples()])


def chart_png(name, q):
    T = get_tables()
    if name in ("drift", "recall", "hist"):
        model = q.get("model", [""])[0]
        year = int(q.get("year", ["0"])[0])
        years = _ensure(model, year, T)
        walk, sim, cfg, ps = T["walk"], T["sim"], T["cfg"], T["per_sample"]
        w = walk[walk.model == model].sort_values("year")
        cur = w[w.year == year].iloc[0]
        if name == "drift":
            return charts.png(charts.fig_drift(w, year, float(cur.retrain_thr), float(cfg["models"][model]["warn_thr"]), years))
        if name == "recall":
            control = float(cur.control_recall)
            frozen = sim[(sim.model == model) & (sim.policy == "frozen")].sort_values("year")
            trig = sim[(sim.model == model) & (sim.policy == "triggered")].sort_values("year")
            retrained = q.get("retrained", ["0"])[0] == "1"
            lo = min(float(sim[sim.model == model].phish_recall.min()), control) - 0.03
            return charts.png(charts.fig_recall(frozen, trig if retrained else None, control, years, year, lo))
        return charts.png(charts.fig_hist(ps[(ps.model == model) & (ps.year == year)]))
    if name == "evidence_recall":
        piv = T["walk"].pivot(index="year", columns="model", values="phish_recall")
        return charts.png(charts.fig_evidence_recall(piv, {m: T["cfg"]["models"][m]["control_recall"] for m in piv.columns}))
    if name == "policy":
        return charts.png(charts.fig_policy(T["sim_sum"]))
    raise ValueError("unknown chart")


# ============================================================================ http
PAGES = {"/": page_home, "/drift": page_drift, "/evidence": page_evidence, "/history": page_history, "/about": page_about}


def _json_default(o):
    if hasattr(o, "item"):
        return o.item()
    return str(o)


class Handler(BaseHTTPRequestHandler):
    server_version = "PhishDrift/1.0"

    def log_message(self, fmt, *args):      # quiet console
        pass

    def _send(self, code, body, ctype, cache=False):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=300" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj, default=_json_default), "application/json; charset=utf-8")

    def do_GET(self):
        u = urlparse(self.path)
        path, q = u.path, parse_qs(u.query)
        try:
            if path in PAGES:
                with LOCK:
                    html = PAGES[path]()
                return self._send(200, html, "text/html; charset=utf-8")
            if path.startswith("/static/"):
                fp = os.path.realpath(os.path.join(WEB, path[len("/static/"):]))
                if not fp.startswith(os.path.realpath(WEB) + os.sep) or not os.path.isfile(fp):
                    return self._send(404, "Not found", "text/plain")
                ext = os.path.splitext(fp)[1].lower()
                ctype = {".css": "text/css; charset=utf-8", ".js": "application/javascript; charset=utf-8",
                         ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon"}.get(
                             ext, mimetypes.guess_type(fp)[0] or "application/octet-stream")
                with open(fp, "rb") as f:
                    return self._send(200, f.read(), ctype)
            if path == "/favicon.ico":
                self.send_response(302); self.send_header("Location", "/static/logo.svg"); self.end_headers(); return
            if path == "/api/drift":
                with LOCK:
                    return self._json(200, api_drift(q))
            if path.startswith("/chart/") and path.endswith(".png"):
                with LOCK:
                    png = chart_png(path[len("/chart/"):-4], q)
                return self._send(200, png, "image/png")
            return self._send(404, "Not found", "text/plain")
        except FileNotFoundError:
            return self._json(404, dict(error="Notebook outputs not found. Unzip fesad_outputs into the artifacts folder."))
        except (ValueError, KeyError, IndexError) as e:
            return self._json(400, dict(error=str(e)))
        except Exception as e:                      # keep the server alive
            import traceback; traceback.print_exc()
            return self._json(500, dict(error=f"{type(e).__name__}: {e}"))

    def do_POST(self):
        if urlparse(self.path).path != "/api/classify":
            return self._send(404, "Not found", "text/plain")
        try:
            n = int(self.headers.get("Content-Length", "0"))
            if n > 600000:
                return self._json(413, dict(error="Request too large."))
            payload = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
            return self._json(200, api_classify(payload))
        except ValueError as e:
            return self._json(400, dict(error=str(e)))
        except Exception as e:
            import traceback; traceback.print_exc()
            return self._json(500, dict(error=f"{type(e).__name__}: {e}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    httpd = None
    for port in range(a.port, a.port + 10):
        try:
            httpd = ThreadingHTTPServer((a.host, port), Handler)
            break
        except OSError:
            continue
    if httpd is None:
        sys.exit("No free port found.")
    url = f"http://{a.host}:{httpd.server_address[1]}"
    avail = models_available()
    print(f"\n  {PROJECT['brand']} running at {url}")
    print(f"  artifacts: {ART}  |  models loaded on demand: {', '.join(avail) if avail else 'NONE FOUND'}")
    print("  Press Ctrl+C to stop.\n")
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.")


if __name__ == "__main__":
    main()
