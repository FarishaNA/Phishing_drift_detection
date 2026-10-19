"""Turn real inbox emails into modern_legit.csv for notebook 02 (legit class is Enron-only otherwise).

Input : a Gmail Takeout .mbox file, OR a folder of .eml files (Gmail: open mail -> 3 dots -> Download message).
Output: modern_legit.csv with columns subject, sender, body, year.

Example:
  python export_inbox.py "All mail.mbox" --since 2020 --redact farisha --redact "Farisha N A" --me you@gmail.com
  python export_inbox.py eml_folder/ --redact farisha

Run it on emails that are LEGIT only (use labels: Updates / Promotions / Forums / Primary after removing spam).
Then OPEN THE CSV and review it. Delete anything sensitive or any spam/phishing row. Never share it.
"""
import argparse
import csv
import email
import glob
import mailbox
import os
import re
import sys
from collections import Counter, defaultdict
from email import policy, utils
from email.parser import BytesParser

SENSITIVE = ["otp", "one-time", "one time", "password", "passcode", "bank", "statement", "salary", "aadhaar",
             "aadhar", "pan card", "credit card", "debit card", "upi", "loan", "medical", "prescription",
             "invoice", "payment", "transaction", "account number", "ifsc"]

EMAIL_RE = re.compile(r"[\w.+-]+@([\w-]+(?:\.[\w-]+)+)")
URL_RE = re.compile(r"(https?://[^/\s<>\"')]+)[^\s<>\"')]*", re.I)
PHONE_RE = re.compile(r"\+?\d[\d\s().-]{8,}\d")
NUM_RE = re.compile(r"\d{5,}")


def iter_messages(path):
    if os.path.isdir(path):
        for fp in sorted(glob.glob(os.path.join(path, "**", "*.eml"), recursive=True)):
            with open(fp, "rb") as f:
                yield BytesParser(policy=policy.default).parse(f)
    else:
        mb = mailbox.mbox(path, factory=lambda f: email.message_from_binary_file(f, policy=policy.default))
        for msg in mb:
            yield msg


def body_of(msg):
    """Prefer text/plain, but fall back to HTML when the plain part is a stub ('view in browser')."""
    texts = {}
    for kind in ("plain", "html"):
        part = msg.get_body(preferencelist=(kind,))
        if part is not None:
            try:
                texts[kind] = part.get_content()
            except Exception:
                texts[kind] = ""
    plain, html = texts.get("plain", ""), texts.get("html", "")
    return plain if len(plain.strip()) >= 80 or not html else html


def sender_domain(sender):
    m = EMAIL_RE.search(sender or "")
    return m.group(1).lower() if m else ""


def scrub(text, redact):
    t = str(text or "")
    t = URL_RE.sub(lambda m: m.group(1) + "/", t)        # keep host only (url_count still works)
    t = EMAIL_RE.sub(lambda m: "user@" + m.group(1), t)
    t = PHONE_RE.sub("<phone>", t)
    t = NUM_RE.sub("<num>", t)
    for name in redact:
        t = re.sub(re.escape(name), "<name>", t, flags=re.I)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--out", default="modern_legit.csv")
    ap.add_argument("--since", type=int, default=2019, help="skip mail older than this year")
    ap.add_argument("--max-per-sender", type=int, default=12, help="cap per sender domain (stops one sender dominating)")
    ap.add_argument("--max-total", type=int, default=400)
    ap.add_argument("--redact", action="append", default=[], help="name/word to replace with <name> (repeatable)")
    ap.add_argument("--me", action="append", default=[], help="your own address(es): mail FROM you is skipped")
    ap.add_argument("--skip", action="append", default=[], help="extra words that make a mail be skipped")
    ap.add_argument("--min-body", type=int, default=40)
    a = ap.parse_args()

    skip_words = SENSITIVE + [w.lower() for w in a.skip]
    per_dom, rows, skipped = defaultdict(int), [], Counter()
    for msg in iter_messages(a.path):
        subj, sender = str(msg["subject"] or ""), str(msg["from"] or "")
        if any(me.lower() in sender.lower() for me in a.me):
            skipped["from me"] += 1; continue
        try:
            year = utils.parsedate_to_datetime(str(msg["date"])).year
        except Exception:
            year = None
        if year is None or year < a.since:
            skipped["too old / no date"] += 1; continue
        body = body_of(msg)
        if len(body.strip()) < a.min_body:
            skipped["too short"] += 1; continue
        low = (subj + " " + body).lower()
        if any(w in low for w in skip_words):
            skipped["sensitive word"] += 1; continue
        dom = sender_domain(sender)
        if per_dom[dom] >= a.max_per_sender:
            skipped["sender cap"] += 1; continue
        per_dom[dom] += 1
        clean_sender = ("user@" + dom) if dom else ""
        rows.append(dict(subject=scrub(subj, a.redact), sender=clean_sender,
                         body=scrub(body, a.redact)[:5000], year=year))
        if len(rows) >= a.max_total:
            break

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["subject", "sender", "body", "year"])
        w.writeheader(); w.writerows(rows)

    print(f"\nwrote {len(rows)} emails -> {a.out}")
    print("by year  :", dict(sorted(Counter(r["year"] for r in rows).items())))
    print("top sender domains:", Counter(sender_domain(r["sender"]) for r in rows).most_common(10))
    print("skipped  :", dict(skipped))
    print("\nNEXT: open the CSV, delete sensitive/spam rows, and remove any email you already tested in the UI "
          "(keep those as true test cases). Do not share this file.")


if __name__ == "__main__":
    sys.exit(main())
