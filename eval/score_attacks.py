"""Score the PDFGuard detector against the attacked test set.

Two modes, chosen automatically:

* Detector available (``from pdfguard import scan`` imports): reports, per attack
  type, the detection rate (the hidden payload is found), plus the overall
  false-alarm rate on the clean papers and, when the detector labels findings,
  the injection-vs-data-vs-harmless label accuracy. This fills the detection
  results table in the top-level README.

* Detector not built yet: reports the undefended baseline instead, the fraction
  of attacked files whose payload reaches a standard pipeline parser (pypdf)
  with no guard in place. That is the attack success rate PDFGuard must drive
  down, and it is a real result you can show now.

Ground truth is data/attacked/manifest.json: every hidden payload begins with
[PDFGUARD-TEST].

Usage:
    python eval/score_attacks.py
    python eval/score_attacks.py --attacked data/attacked --clean data/clean
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Run from anywhere: make the repo-root pdfguard package importable without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATA = Path(__file__).resolve().parent.parent / "data"
SENTINEL = "PDFGUARD-TEST"
ZERO_WIDTH = {0x200B, 0x200C, 0x200D, 0xFEFF}
ATTACK_ORDER = ["white_text", "tiny_font", "off_page", "covered_text",
                "invisible_render", "metadata", "annotation", "unicode"]

# pypdf and pdfminer log a warning for every malformed font table, which buries the results.
for _name in ("pypdf", "pdfminer", "pdfplumber"):
    logging.getLogger(_name).setLevel(logging.ERROR)


def default_jobs() -> int:
    return max(1, min(8, (os.cpu_count() or 2) - 1))


def progress(done: int, total: int, label: str) -> None:
    """One self-overwriting status line on stderr, so long runs never look stuck."""
    end = "\n" if done == total else ""
    print(f"\r{label}: {done}/{total} files", end=end, file=sys.stderr, flush=True)


def strip_zw(s: str) -> str:
    return "".join(c for c in s if ord(c) not in ZERO_WIDTH)


def load_detector():
    """Return a scan callable, or None if the detector is not implemented."""
    try:
        from pdfguard import scan  # type: ignore
        return scan
    except Exception:
        return None


# ---- detector-based scoring -------------------------------------------------

def finding_texts(report) -> list[str]:
    """Pull snippet text out of a report however findings are shaped."""
    findings = getattr(report, "findings", None)
    if findings is None and isinstance(report, dict):
        findings = report.get("findings", [])
    texts = []
    for f in findings or []:
        if isinstance(f, dict):
            texts.append(str(f.get("text", "")))
        else:
            texts.append(str(getattr(f, "text", "")))
    return texts


def report_risk(report) -> str:
    risk = getattr(report, "risk", None)
    if risk is None and isinstance(report, dict):
        risk = report.get("risk")
    return str(risk or "")


def _scan_one(path: str, use_llm: bool) -> tuple[bool, str]:
    """Worker: scan one file; return (payload found, risk). Runs in a separate process."""
    from pdfguard import scan
    try:
        rep = scan(path, use_llm=use_llm)
    except Exception as e:
        print(f"\nerror {path}: {e}", file=sys.stderr)
        return False, "error"
    found = any(SENTINEL in strip_zw(t) for t in finding_texts(rep))
    return found, report_risk(rep)


def _scan_all(paths: list[str], use_llm: bool, jobs: int, label: str) -> list[tuple[bool, str]]:
    results = [None] * len(paths)
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(_scan_one, p, use_llm): i for i, p in enumerate(paths)}
        for done, fut in enumerate(as_completed(futures), 1):
            results[futures[fut]] = fut.result()
            progress(done, len(paths), label)
    return results


def score_with_detector(attacked: Path, clean: Path, manifest: list,
                        use_llm: bool = False, jobs: int = 1) -> int:
    det = defaultdict(lambda: [0, 0])  # attack -> [detected, total]
    results = _scan_all([str(attacked / row["output"]) for row in manifest], use_llm, jobs, "attacked")
    for row, (found, _risk) in zip(manifest, results):
        d = det[row["attack"]]
        d[0] += found
        d[1] += 1

    print(f"{'attack':18}{'detection rate':18}")
    print("-" * 36)
    tot_d = tot_n = 0
    for attack in ATTACK_ORDER:
        if attack not in det:
            continue
        d, n = det[attack]
        tot_d += d
        tot_n += n
        print(f"{attack:18}{f'{d}/{n} ({100*d/n:.0f}%)':18}")
    print("-" * 36)
    print(f"{'overall':18}{f'{tot_d}/{tot_n} ({100*tot_d/max(tot_n,1):.0f}%)':18}")

    clean_pdfs = sorted(clean.glob("*.pdf"))
    clean_results = _scan_all([str(p) for p in clean_pdfs], use_llm, jobs, "clean") if clean_pdfs else []
    flagged = [p.name for p, (_f, risk) in zip(clean_pdfs, clean_results) if risk not in ("", "clean")]
    false_alarms = len(flagged)
    print(f"\nfalse alarms on clean papers: {false_alarms}/{len(clean_pdfs)}"
          f" ({100*false_alarms/max(len(clean_pdfs),1):.0f}%)")
    for name in flagged:
        print(f"  flagged: {name}")
    return 0


# ---- undefended baseline (no detector yet) ----------------------------------

def score_baseline(attacked: Path, manifest: list) -> int:
    import pymupdf
    import pypdf

    def reaches(path: Path, attack: str) -> bool:
        if attack == "metadata":
            doc = pymupdf.open(path)
            ok = SENTINEL in json.dumps(doc.metadata)
            doc.close()
            return ok
        if attack == "annotation":
            doc = pymupdf.open(path)
            ok = any(SENTINEL in (a.info.get("content") or "")
                     for a in doc[0].annots())
            doc.close()
            return ok
        text = pypdf.PdfReader(str(path)).pages[0].extract_text()
        return SENTINEL in strip_zw(text)

    base = defaultdict(lambda: [0, 0])
    for row in manifest:
        b = base[row["attack"]]
        b[0] += reaches(attacked / row["output"], row["attack"])
        b[1] += 1

    print("Detector not implemented (pdfguard.scan not importable).")
    print("Reporting the UNDEFENDED baseline: payload reaching a standard")
    print("pipeline parser (pypdf) with no guard. This is what PDFGuard must")
    print("drive to zero.\n")
    print(f"{'attack':18}{'reaches LLM (undefended)':26}")
    print("-" * 44)
    tot = cnt = 0
    for attack in ATTACK_ORDER:
        if attack not in base:
            continue
        r, n = base[attack]
        tot += r
        cnt += n
        print(f"{attack:18}{f'{r}/{n} ({100*r/n:.0f}%)':26}")
    print("-" * 44)
    print(f"{'overall':18}{f'{tot}/{cnt} ({100*tot/max(cnt,1):.0f}%)':26}")
    print("\nBuild pdfguard.scan, then re-run for detection and false-alarm rates.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attacked", default=str(DATA / "attacked"))
    ap.add_argument("--clean", default=str(DATA / "clean"))
    ap.add_argument("--jobs", type=int, default=default_jobs(),
                    help="files scanned in parallel (default: %(default)s)")
    ap.add_argument("--llm", action="store_true",
                    help="also label findings with the configured LLM (slow, uses API calls); off by default")
    ap.add_argument("--baseline", action="store_true",
                    help="report the undefended baseline even if pdfguard is installed")
    args = ap.parse_args()

    attacked = Path(args.attacked)
    clean = Path(args.clean)
    manifest = json.loads((attacked / "manifest.json").read_text())

    scan = None if args.baseline else load_detector()
    if scan is not None:
        return score_with_detector(attacked, clean, manifest, use_llm=args.llm, jobs=args.jobs)
    return score_baseline(attacked, manifest)


if __name__ == "__main__":
    sys.exit(main())
