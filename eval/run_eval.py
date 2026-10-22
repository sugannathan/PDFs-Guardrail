"""Detection rate and false alarms over a folder of attacked and a folder of clean PDFs.

    python eval/run_eval.py --attacked data/attacked --clean data/clean --out results/

A file counts as detected if PDFGuard reports it as suspicious or dangerous.
If the attacked file names contain the attack type (e.g. paper12__white_text.pdf,
the part after "__"), results are also broken down per attack type.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pdfguard import scan  # noqa: E402


def attack_type(path: Path) -> str:
    return path.stem.split("__", 1)[1] if "__" in path.stem else "all"


def run(folder: str | None, **kw) -> list[dict]:
    rows = []
    if not folder:
        return rows
    for pdf in sorted(Path(folder).rglob("*.pdf")):
        try:
            r = scan(pdf, **kw)
        except Exception as e:
            print(f"error {pdf}: {e}", file=sys.stderr)
            continue
        rows.append({"file": pdf.name, "attack": attack_type(pdf), "risk": r.risk,
                     "flagged": r.risk != "clean", **{k: v for k, v in r.counts.items()},
                     "labels": ";".join(sorted({f.label for f in r.findings if f.label != "harmless"}))})
        print(f"{r.risk:<10} {pdf.name}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attacked")
    ap.add_argument("--clean")
    ap.add_argument("--out", default="results")
    ap.add_argument("--llm", action="store_true", help="also label with gpt-oss on Cerebras (slower)")
    ap.add_argument("--ocr", action="store_true")
    a = ap.parse_args()

    attacked = run(a.attacked, use_llm=a.llm, use_ocr=a.ocr)
    clean = run(a.clean, use_llm=a.llm, use_ocr=a.ocr)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [dict(r, set="attacked") for r in attacked] + [dict(r, set="clean") for r in clean]
    if rows:
        with open(out / "per_file.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    per_attack = defaultdict(lambda: [0, 0])
    for r in attacked:
        per_attack[r["attack"]][0] += r["flagged"]
        per_attack[r["attack"]][1] += 1
    summary = {
        "detection_rate": round(sum(r["flagged"] for r in attacked) / len(attacked), 3) if attacked else None,
        "false_alarm_rate": round(sum(r["flagged"] for r in clean) / len(clean), 3) if clean else None,
        "n_attacked": len(attacked), "n_clean": len(clean),
        "per_attack": {k: {"detected": d, "total": n, "rate": round(d / n, 3)} for k, (d, n) in sorted(per_attack.items())},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n| Attack | Detected | Rate |\n|---|---|---|")
    for k, v in summary["per_attack"].items():
        print(f"| {k} | {v['detected']}/{v['total']} | {v['rate']:.0%} |")
    if clean:
        print(f"| **False alarms on clean** | {sum(r['flagged'] for r in clean)}/{len(clean)} | {summary['false_alarm_rate']:.0%} |")
    print(f"\nwritten to {out}/")


if __name__ == "__main__":
    main()
