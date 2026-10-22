"""Download conference submissions for a prevalence scan (how common is hidden text?).

Two sources:

  OpenReview (all public submissions of a venue, incl. rejected/withdrawn where public):
      pip install openreview-py
      python eval/download_venue_papers.py openreview --venue ICLR.cc/2025/Conference --n 300

  arXiv preprints whose comment field mentions a venue (e.g. "Accepted at ICLR 2025"):
      python eval/download_venue_papers.py arxiv --comment "ICLR 2025" --n 300

PDFs go to data/prevalence/<source>_<venue>/ together with index.csv (id, title, url).
Then scan them:
      python -m pdfguard scan data/prevalence --quiet --no-llm --report results/prevalence.json

Local, aggregate use only: report counts, not individual papers or authors.
Be polite to the servers: the script waits between downloads (--delay).
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import re
import sys
import time
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UA = {"User-Agent": "pdfguard-prevalence/0.1 (research; local aggregate statistics)"}


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")[:60]


def _save(out: Path, name: str, data: bytes) -> bool:
    if not data or not data[:5].startswith(b"%PDF"):
        return False
    (out / f"{name}.pdf").write_bytes(data)
    return True


# ----------------------------------------------------------------- OpenReview

def openreview(args) -> None:
    try:
        import openreview  # type: ignore
    except ImportError:
        sys.exit("Install the client first:  pip install openreview-py")

    kw = {"baseurl": "https://api2.openreview.net"}
    if os.environ.get("OPENREVIEW_USERNAME"):          # optional; public venues usually work without
        kw.update(username=os.environ["OPENREVIEW_USERNAME"], password=os.environ.get("OPENREVIEW_PASSWORD"))
    client = openreview.api.OpenReviewClient(**kw)

    venue = args.venue
    try:
        submission_name = client.get_group(venue).content["submission_name"]["value"]
    except Exception:
        submission_name = "Submission"
    invitation = f"{venue}/-/{submission_name}"
    print(f"fetching submission list for {invitation} ...")
    notes = client.get_all_notes(invitation=invitation)
    notes = [n for n in notes if n.content.get("pdf", {}).get("value")]
    print(f"{len(notes)} submissions with a PDF")
    if not notes:
        sys.exit("Nothing found. Check the venue id (e.g. ICLR.cc/2025/Conference) or log in "
                 "(set OPENREVIEW_USERNAME / OPENREVIEW_PASSWORD).")

    random.Random(args.seed).shuffle(notes)
    out = Path(args.out) / f"openreview_{_slug(venue)}"
    out.mkdir(parents=True, exist_ok=True)
    rows, got = [], 0
    for note in notes:
        if got >= args.n:
            break
        name = f"or_{note.id}"
        if (out / f"{name}.pdf").exists():
            got += 1
            continue
        try:
            data = client.get_attachment(field_name="pdf", id=note.id)
        except Exception as e:
            print(f"  skip {note.id}: {e}")
            continue
        if _save(out, name, data):
            got += 1
            title = note.content.get("title", {}).get("value", "")
            status = note.content.get("venueid", {}).get("value", "")
            rows.append({"id": note.id, "title": title, "status": status,
                         "url": f"https://openreview.net/forum?id={note.id}"})
            print(f"  [{got}/{args.n}] {note.id}")
        time.sleep(args.delay)
    _write_index(out, rows)


# ---------------------------------------------------------------------- arXiv

def arxiv(args) -> None:
    import requests

    query = f'co:"{args.comment}"'
    if args.category:
        query += f" AND cat:{args.category}"
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": query, "start": 0, "max_results": args.n,
         "sortBy": "submittedDate", "sortOrder": "descending"})
    print(f"searching arXiv: {query}")
    for attempt in range(4):
        try:
            feed = requests.get(url, headers=UA, timeout=90).text
            break
        except Exception as e:
            print(f"  arXiv API slow ({e}); retrying ...")
            time.sleep(10 * (attempt + 1))
    else:
        sys.exit("arXiv API did not answer. Try again later or use eval/download_papers.py --ids ...")

    entries = re.findall(r"<entry>(.*?)</entry>", feed, re.S)
    print(f"{len(entries)} matching preprints")
    out = Path(args.out) / f"arxiv_{_slug(args.comment)}"
    out.mkdir(parents=True, exist_ok=True)
    rows, got = [], 0
    for e in entries:
        m = re.search(r"<id>https?://arxiv\.org/abs/(.*?)</id>", e)
        if not m:
            continue
        aid = m.group(1)                       # includes the version, e.g. 2501.01234v2
        title = re.sub(r"\s+", " ", (re.search(r"<title>(.*?)</title>", e, re.S) or [None, ""])[1]).strip()
        name = "ax_" + aid.replace("/", "_")
        if not (out / f"{name}.pdf").exists():
            try:
                r = requests.get(f"https://arxiv.org/pdf/{aid}", headers=UA, timeout=90)
                if not _save(out, name, r.content):
                    print(f"  skip {aid}: not a PDF")
                    continue
            except Exception as ex:
                print(f"  skip {aid}: {ex}")
                continue
            time.sleep(args.delay)
        got += 1
        rows.append({"id": aid, "title": title, "status": "", "url": f"https://arxiv.org/abs/{aid}"})
        print(f"  [{got}] {aid}")
    _write_index(out, rows)


def _write_index(out: Path, rows: list[dict]) -> None:
    index = out / "index.csv"
    old = []
    if index.exists():
        with open(index, newline="", encoding="utf-8") as f:
            old = list(csv.DictReader(f))
    seen = {r["id"] for r in old}
    with open(index, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "title", "status", "url"])
        w.writeheader()
        w.writerows(old + [r for r in rows if r["id"] not in seen])
    print(f"\n{len(list(out.glob('*.pdf')))} PDFs in {out}/  (index: {index})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="source", required=True)
    o = sub.add_parser("openreview", help="submissions of an OpenReview venue")
    o.add_argument("--venue", default="ICLR.cc/2025/Conference", help="e.g. ICLR.cc/2024/Conference, ICLR.cc/2026/Conference")
    a = sub.add_parser("arxiv", help="arXiv preprints whose comment field mentions a venue")
    a.add_argument("--comment", default="ICLR 2025", help='text in the arXiv comment field, e.g. "ICLR 2025"')
    a.add_argument("--category", default="", help="optional arXiv category filter, e.g. cs.LG")
    for p in (o, a):
        p.add_argument("--n", type=int, default=200, help="number of papers")
        p.add_argument("--out", default=str(ROOT / "data" / "prevalence"))
        p.add_argument("--delay", type=float, default=3.0, help="seconds between downloads")
        p.add_argument("--seed", type=int, default=0, help="random sample seed (OpenReview)")
    args = ap.parse_args()
    {"openreview": openreview, "arxiv": arxiv}[args.source](args)


if __name__ == "__main__":
    main()
