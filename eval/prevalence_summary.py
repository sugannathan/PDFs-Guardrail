"""Aggregate prevalence numbers from a PDFGuard report (no paper is named).

    python -m pdfguard scan data/prevalence --quiet --no-llm --report results/prevalence.json
    python eval/prevalence_summary.py results/prevalence.json
"""
import json
import sys
from collections import Counter

report = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/prevalence.json"))
docs = report["reports"]
n = len(docs)
any_hidden = sum(1 for r in docs if r["findings"])
by_label = Counter()
docs_with = Counter()
how = Counter()
for r in docs:
    labels = {f["label"] for f in r["findings"]}
    for lab in labels:
        docs_with[lab] += 1
    for f in r["findings"]:
        by_label[f["label"]] += 1
        if f["label"] != "harmless":
            how.update(f["reasons"])

pct = lambda k: f"{k} ({100 * k / n:.1f}%)" if n else "0"
print(f"papers scanned                         : {n}")
print(f"with any hidden content                : {pct(any_hidden)}")
print(f"with hidden INSTRUCTION-like text      : {pct(docs_with['injection'])}")
print(f"with hidden value claims               : {pct(docs_with['data'])}")
print(f"with unexplained hidden text (unknown) : {pct(docs_with['unknown'])}")
print(f"only harmless hidden content           : {pct(sum(1 for r in docs if r['findings'] and all(f['label'] == 'harmless' for f in r['findings'])))}")
print(f"risk levels                            : {report['totals']}")
if how:
    print("how the non-harmless content was hidden:", ", ".join(f"{k}={v}" for k, v in how.most_common(8)))
print("\nReport these numbers in aggregate. Check every 'injection' finding by hand before claiming it is real.")
