"""Download real, open-access papers from arXiv to use as clean test inputs.

The attack generator (make_attacks.py) inserts clearly-marked [PDFGUARD-TEST]
payloads into copies of these PDFs. The copies stay local and are only used to
measure a detector. Nothing is re-uploaded or redistributed, which is what the
Responsible use section of the top-level README requires for open-access papers.

By default this fetches the maintainer's own paper (safest real-world case) plus
a few open-access materials-science papers for layout diversity.

Usage:
    python eval/download_papers.py                       # author paper + mtrl-sci
    python eval/download_papers.py --author "Pandit_Prakul"
    python eval/download_papers.py --category cond-mat.mtrl-sci --n 6
    python eval/download_papers.py --ids 2605.03205 2401.00001
"""
from __future__ import annotations

import argparse
import re
import time
import urllib.parse
from pathlib import Path

import pymupdf
import requests

API = "http://export.arxiv.org/api/query?"
PDF = "https://arxiv.org/pdf/{id}.pdf"
UA = {"User-Agent": "pdfguard-eval/0.1 (local research; contact maintainer)"}
DEFAULT_DIR = Path(__file__).resolve().parent.parent / "data" / "clean"

# Minimal Atom parsing without extra dependencies.
_ENTRY = re.compile(r"<entry>(.*?)</entry>", re.S)
_TITLE = re.compile(r"<title>(.*?)</title>", re.S)
_ID = re.compile(r"<id>http://arxiv\.org/abs/(.*?)</id>")


def _slug(text: str, fallback: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return (s[:48] or fallback).strip("_")


def query(search: str, n: int) -> list[dict]:
    url = API + urllib.parse.urlencode(
        {"search_query": search, "start": 0, "max_results": n,
         "sortBy": "submittedDate", "sortOrder": "descending"})
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    out = []
    for block in _ENTRY.findall(r.text):
        tid = _ID.search(block)
        title = _TITLE.search(block)
        if tid and title:
            out.append({"id": tid.group(1),
                        "title": re.sub(r"\s+", " ", title.group(1)).strip()})
    return out


def _save_trimmed(data: bytes, dest: Path, max_pages: int) -> int:
    """Save a PDF, trimmed to its first max_pages (0 = keep all), compressed.

    Hidden-content attacks target page 1, so keeping the first few pages
    preserves the real page-1 layout and fonts while keeping image-heavy papers
    small enough for a fast eval loop. Returns the written size in KB.
    """
    doc = pymupdf.open(stream=data, filetype="pdf")
    if max_pages and doc.page_count > max_pages:
        doc.select(range(max_pages))
    doc.save(dest, garbage=4, deflate=True)
    size = dest.stat().st_size
    doc.close()
    return size // 1024


def download(entry: dict, out_dir: Path, prefix: str,
             max_pages: int = 8) -> Path | None:
    arxiv_id = entry["id"]
    bare = arxiv_id.split("v")[0]
    name = f"{prefix}{_slug(entry['title'], bare)}.pdf"
    dest = out_dir / name
    if dest.exists():
        print(f"  skip (exists): {name}")
        return dest
    r = requests.get(PDF.format(id=arxiv_id), headers=UA, timeout=60)
    if r.status_code != 200 or not r.content[:4] == b"%PDF":
        print(f"  FAILED {arxiv_id}: status {r.status_code}, not a PDF")
        return None
    kb = _save_trimmed(r.content, dest, max_pages)
    print(f"  wrote {name}  ({kb} KB)  [{arxiv_id}]")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEFAULT_DIR), help="output directory")
    ap.add_argument("--author", default="Pandit_Prakul",
                    help="arXiv author to fetch (surname_firstname)")
    ap.add_argument("--category", default="cond-mat.mtrl-sci",
                    help="arXiv category for diversity papers")
    ap.add_argument("--n", type=int, default=4,
                    help="number of category papers to fetch")
    ap.add_argument("--ids", nargs="*", default=None,
                    help="explicit arXiv ids, overrides author/category")
    ap.add_argument("--no-author", action="store_true",
                    help="skip the author paper, fetch only category papers")
    ap.add_argument("--max-pages", type=int, default=8,
                    help="trim each paper to its first N pages (0 = keep all)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    got = 0

    if args.ids:
        for i, aid in enumerate(args.ids):
            if download({"id": aid, "title": aid}, out, "arxiv_", args.max_pages):
                got += 1
            time.sleep(3)
        print(f"\n{got} papers in {out}/")
        return

    if not args.no_author:
        print(f"author: {args.author}")
        for e in query(f'au:"{args.author}"', 3):
            if download(e, out, "own_", args.max_pages):
                got += 1
            time.sleep(3)

    if args.n > 0:
        print(f"category: {args.category} (newest {args.n})")
        for e in query(f"cat:{args.category}", args.n):
            if download(e, out, "arxiv_", args.max_pages):
                got += 1
            time.sleep(3)

    print(f"\n{got} real papers downloaded to {out}/")
    print("Local use only. Clearly-marked test payloads, never redistributed.")


if __name__ == "__main__":
    main()
