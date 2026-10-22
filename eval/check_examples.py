"""Compare a PDFGuard report with what was actually hidden in each example.

    python -m pdfguard scan examples/ --ocr --report results/all_examples.json
    python eval/check_examples.py results/all_examples.json

Checks per file: risk level, that the hidden payload was found, its source,
and its label. Writes results/check.json and prints a table.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    report_path = Path(sys.argv[1] if len(sys.argv) > 1 else "results/all_examples.json")
    expected_path = Path(sys.argv[2] if len(sys.argv) > 2 else ROOT / "examples" / "expected.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    by_name = {Path(r["file"]).name: r for r in report["reports"]}

    rows, ok_all = [], 0
    for name, exp in expected.items():
        r = by_name.get(name)
        if r is None:
            rows.append((name, exp.get("hidden_by") or "-", "missing", "", "", "", False))
            continue
        payload = exp.get("payload")
        hit = None
        if payload:
            for f in r["findings"]:
                if payload.lower() in ((f.get("decoded") or "") + " " + f["text"]).lower():
                    hit = f
                    break
        risk_ok = r["risk"] == exp["risk"]
        found_ok = (hit is not None) if payload else not any(f["label"] != "harmless" for f in r["findings"])
        source_ok = (hit or {}).get("source") == exp.get("source") if payload else True
        label_ok = (hit or {}).get("label") == exp.get("label") if payload else True
        ok = risk_ok and found_ok and source_ok and label_ok
        ok_all += ok
        rows.append((name, exp.get("hidden_by") or "nothing hidden",
                     f"{r['risk']}" + ("" if risk_ok else f" (expected {exp['risk']})"),
                     ("found" if hit else "NOT FOUND") if payload else ("-" if found_ok else "false alarm"),
                     ((hit or {}).get("label", "") + ("" if label_ok else f" (expected {exp.get('label')})")) if payload else "-",
                     (hit or {}).get("method", "") if payload else "", ok))

    print(f"\n{'file':<36} {'hidden by':<34} {'risk':<11} {'payload':<10} {'label':<11} {'labelled by':<12} ok")
    print("-" * 122)
    for name, how, risk, found, label, method, ok in rows:
        print(f"{name:<36} {how:<34} {risk:<11} {found:<10} {label:<11} {method:<12} {'✓' if ok else '✗'}")
    print(f"\n{ok_all}/{len(rows)} examples match what was hidden in them.")

    out = report_path.parent / "check.json"
    out.write_text(json.dumps([dict(zip(("file", "hidden_by", "risk", "payload", "label", "labelled_by", "ok"), r))
                               for r in rows], indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"written to {out}")
    return 0 if ok_all == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
