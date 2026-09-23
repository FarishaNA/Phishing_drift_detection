"""Inference-only code copied from notebook 02. No training here.

Saved states (joblib) hold only sklearn objects + numpy arrays, so they load without the notebook.
Keep this file in sync with the notebook's cleaning/featurizer cells: UI and training must match.
"""
import re
from email import policy
from email.parser import BytesParser

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import normalize

ENGINEERED_COLS = ["body_len", "word_count", "url_count", "exclaim_count", "caps_ratio",
                   "urgency_score", "subject_len", "subject_has_urgency", "sender_domain_suspicious"]

URGENCY_WORDS = ["verify", "urgent", "account locked", "click here", "suspend",
                 "immediately", "act now", "confirm your", "limited time", "expire"]


# ---------------------------------------------------------------- cleaning
def _strip_html(t):
    t = str(t)
    if "<" in t and ">" in t:
        try:
            from bs4 import BeautifulSoup
            t = BeautifulSoup(t, "html.parser").get_text(" ")
        except Exception:
            t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def urgency_score(text):
    text = str(text).lower()
    return sum(text.count(w) for w in URGENCY_WORDS)


def caps_ratio(text):
    letters = [c for c in str(text) if c.isalpha()]
    return (sum(c.isupper() for c in letters) / len(letters)) if letters else 0.0


def sender_domain_suspicious(sender):
    m = re.search(r"@([\w\.-]+)", str(sender))
    if not m:
        return 0
    domain = m.group(1).lower()
    s = 0
    if len(domain) > 25: s += 1
    if domain.count("-") >= 2: s += 1
    if sum(c.isdigit() for c in domain) >= 3: s += 1
    if not re.search(r"\.(com|org|net|edu|gov)$", domain): s += 1
    return s


def clean_and_engineer(raw):
    """raw: DataFrame(subject, sender, body) -> cleaned body + 9 engineered features."""
    d = raw[["subject", "sender", "body"]].copy()
    for c in d.columns:
        d[c] = d[c].fillna("").astype(str)
    html_free = d["body"].apply(_strip_html)
    d["url_count"] = html_free.str.count(r"http[s]?://|www\.")          # BEFORE stripping URLs
    d["body"] = html_free.apply(lambda t: re.sub(r"http\S+|www\.\S+", " ", t))
    d["body"] = d["body"].str.replace(r"\s+", " ", regex=True).str.strip()
    d["body_len"] = np.log1p(d["body"].str.len())
    d["word_count"] = np.log1p(d["body"].str.split().apply(len))
    d["exclaim_count"] = d["body"].str.count("!")
    d["caps_ratio"] = d["body"].apply(caps_ratio)
    d["urgency_score"] = d["body"].apply(urgency_score)
    d["subject_len"] = d["subject"].str.len()
    d["subject_has_urgency"] = d["subject"].apply(lambda s: int(urgency_score(s) > 0))
    d["sender_domain_suspicious"] = d["sender"].apply(sender_domain_suspicious)
    return d


def parse_eml(raw_bytes):
    """.eml bytes -> (subject, sender, body)."""
    msg = BytesParser(policy=policy.default).parsebytes(raw_bytes)
    subject = str(msg["subject"] or "")
    sender = str(msg["from"] or "")
    body = ""
    part = msg.get_body(preferencelist=("plain", "html"))
    if part is not None:
        try:
            body = part.get_content()
        except Exception:
            body = ""
    return subject, sender, body


# ---------------------------------------------------------------- featurizer + FeSAD terms
def featurize(feat, d):
    return hstack([feat["tfidf"].transform(d["body"]),
                   csr_matrix(feat["scaler"].transform(d[feat.get("cols", ENGINEERED_COLS)]))]).tocsr()


def dist_ratio(X, mu_m, mu_b):
    Xn = normalize(X)
    dm = 1 - np.asarray(Xn @ mu_m).ravel()
    db = 1 - np.asarray(Xn @ mu_b).ravel()
    return (dm + 1e-9) / (db + 1e-9)


def weighted_dist(ratio, p1, pred, pbar_m, pbar_b):
    p = np.where(pred == 1, p1, 1 - p1)
    wm, wb = p / pbar_m, p / pbar_b
    return np.where(pred == 1, np.log(ratio / wm), np.log(ratio * wb))


def directional_drift(dw, pred, dwbar_mc, dwbar_bc):
    return np.where(pred == 1, dw - dwbar_mc, dwbar_bc - dw)


def _drift_arrays(st, X, p1):
    pred = (p1 >= 0.5).astype(int)
    ratio = dist_ratio(X, st["mu_m"], st["mu_b"])
    dw = weighted_dist(ratio, p1, pred, st["pbar_m"], st["pbar_b"])
    D = directional_drift(dw, pred, st["dwbar_mc"], st["dwbar_bc"])
    return pred, ratio, D


# ---------------------------------------------------------------- explanation
def _feature_names(st):
    return list(st["feat"]["tfidf"].get_feature_names_out()) + list(st["feat"].get("cols", ENGINEERED_COLS))


def top_terms(st, X, k=8):
    """Which terms in this email drove the verdict. NB/SVM: signed. RF: importance only."""
    names = _feature_names(st)
    x = X.tocsr()
    idx, vals = x.indices, x.data
    model, name = st["model"], st["name"]
    w = None
    try:
        if name == "NB":
            w = model.feature_log_prob_[1] - model.feature_log_prob_[0]
        elif name == "SVM":
            coefs = []
            for cc in model.calibrated_classifiers_:
                est = getattr(cc, "estimator", None) or getattr(cc, "base_estimator")
                coefs.append(est.coef_.ravel())
            w = np.mean(coefs, axis=0)
    except Exception:
        w = None
    if len(idx) == 0:
        return {"directional": False, "influential": []}
    if w is not None:
        contrib = vals * w[idx]
        order = np.argsort(contrib)
        legit = [(names[idx[i]], float(contrib[i])) for i in order[:k] if contrib[i] < 0]
        phish = [(names[idx[i]], float(contrib[i])) for i in order[::-1][:k] if contrib[i] > 0]
        return {"directional": True, "phish": phish, "legit": legit}
    try:
        imp = model.feature_importances_
        contrib = vals * imp[idx]
        order = np.argsort(contrib)[::-1][:k]
        return {"directional": False, "influential": [(names[idx[i]], float(contrib[i])) for i in order]}
    except Exception:
        return {"directional": False, "influential": []}


# ---------------------------------------------------------------- one email
def predict_email(subject, sender, body, st):
    raw = pd.DataFrame([{"subject": subject, "sender": sender, "body": body}])
    d = clean_and_engineer(raw)
    X = featurize(st["feat"], d)
    p1 = st["model"].predict_proba(X)[:, 1]
    pred, ratio, D = _drift_arrays(st, X, p1)
    Df = float(D[0] / (st["De_m"] if pred[0] == 1 else st["De_b"]))
    return dict(label="phishing" if pred[0] == 1 else "legit",
                p_phish=float(p1[0]), dist_ratio=float(ratio[0]), Df=Df, unfamiliar=bool(Df > 1),
                features=d[ENGINEERED_COLS].iloc[0].to_dict(),
                terms=top_terms(st, X))


# ---------------------------------------------------------------- pasted text -> (subject, sender, body)
_HDR = re.compile(r"^(subject|from|to|cc|bcc|date|reply-to|message-id|mime-version|content-type|x-[\w-]+)\s*:\s*(.*)$",
                  re.I)


def split_raw(text):
    """One pasted box -> (subject, sender, body). If the paste starts with header lines
    (Subject:, From:, ...), they are read and removed from the body. Otherwise everything is body."""
    lines = (text or "").replace("\r\n", "\n").split("\n")
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    subject = sender = ""
    found = False
    while i < len(lines) and lines[i].strip():
        m = _HDR.match(lines[i].strip())
        if m is None:
            break
        found = True
        key = m.group(1).lower()
        if key == "subject":
            subject = m.group(2).strip()
        elif key == "from":
            sender = m.group(2).strip()
        i += 1
        while i < len(lines) and lines[i].startswith((" ", "\t")) and lines[i].strip():   # folded header lines
            i += 1
    if found:
        return subject, sender, "\n".join(lines[i:]).strip()
    return "", "", (text or "").strip()
