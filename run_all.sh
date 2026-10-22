#!/usr/bin/env bash
# PDFGuard evaluation: answers four questions and writes everything to results/.
# Run from the repo root (the folder with README.md and eval/):
#     bash run_all.sh          rules only (fast, no API calls)
#     bash run_all.sh --llm    also label every finding with the LLM from secrets.env (slower)
#
# Needs: papers in data/clean/ (eval/download_papers.py).
# Optional: data/real_cases/ and data/real_cases_latest/ (the 18 documented arXiv cases).

cd "$(dirname "$0")"
export PYTHONPATH=.
if [ "$1" = "--llm" ]; then
  LLM_FLAG="--llm"
  LLM_FLAG_PARA="--llm"
  export PDFGUARD_PROVIDER="${PDFGUARD_PROVIDER:-local}"     # default: free local Hugging Face model
  echo "== LLM verification ON: $(python3 -c 'from pdfguard import config as c; print(c.provider(), c.model())')"
  echo "   (local model: no API limits; Cerebras/HF API: paced to ~20 requests/min)"
else
  export PDFGUARD_PROVIDER=none        # rules only: fast, no API calls
  LLM_FLAG="--no-llm"
  LLM_FLAG_PARA=""
fi
mkdir -p results reports
rm -f results/q1_* results/q2_* results/q3_* results/q4_* results/q5_* results/verify.txt   # no stale numbers from earlier runs

n_clean=$(ls data/clean/*.pdf 2>/dev/null | wc -l)
if [ "$n_clean" -eq 0 ]; then
  echo "No papers in data/clean/. Run first:  python3 eval/download_papers.py --no-author --n 20"
  exit 1
fi
echo "== $n_clean clean papers found in data/clean/"

echo; echo "== Step A: hiding attacks in copies of the papers (Prakul's generator)"
rm -rf data/attacked
python3 eval/make_attacks.py | tail -2
python3 eval/verify_attacks.py > results/verify.txt 2>&1
echo "   (check: results/verify.txt)"

echo; echo "== Question 1: without PDFGuard, what reaches the LLM?"
python3 eval/score_attacks.py --baseline 2>/dev/null | tee results/q1_no_guard.txt | tail -14
python3 eval/parser_leak.py --csv reports/parsers.csv > results/q1_parsers.txt 2>&1
echo "   (per-parser details: reports/parsers.csv)"

echo; echo "== Question 2 + 3: does PDFGuard catch them? any false alarms on the originals?"
python3 eval/score_attacks.py 2>/dev/null | tee results/q2_q3_pdfguard.txt | tail -16

echo; echo "== Saving PDFGuard's result for every attacked file (all findings, labels, positions)"
python3 -m pdfguard scan data/attacked --quiet $LLM_FLAG --report results/q2_attacked_all.json \
  > results/q2_attacked_all.txt 2>results/q2_attacked_all.log
echo "   one line per file: results/q2_attacked_all.txt   full details: results/q2_attacked_all.json"

echo; echo "== Question 3 in detail: original papers through PDFGuard"
python3 -m pdfguard scan data/clean --quiet $LLM_FLAG --report results/q3_clean_papers.json 2>/dev/null \
  | tee results/q3_clean_papers.txt | grep -v "^\[OK" || echo "   all original papers came back clean"

if ls data/real_cases/*.pdf >/dev/null 2>&1; then
  echo; echo "== Question 4: the real documented arXiv cases"
  python3 -m pdfguard scan data/real_cases --quiet $LLM_FLAG --report results/q4_real_cases.json 2>/dev/null \
    | tee results/q4_real_cases.txt
  if ls data/real_cases_latest/*.pdf >/dev/null 2>&1; then
    echo; echo "   ...and their latest versions (prompt often removed by the authors):"
    python3 -m pdfguard scan data/real_cases_latest --quiet $LLM_FLAG --report results/q4_real_cases_latest.json 2>/dev/null \
      | tee results/q4_real_cases_latest.txt
  fi
else
  echo; echo "== Question 4 skipped (no PDFs in data/real_cases/)"
fi

echo; echo "== Question 5: hidden instructions that match NONE of our phrase patterns (paraphrased, 7 languages)"
python3 eval/paraphrase_attacks.py $LLM_FLAG_PARA --out results/q5_paraphrase.json 2>/dev/null | tail -5 | head -3

echo; echo "================ SUMMARY ================"
python3 - <<'EOF'
import json, re
from pathlib import Path

def read(p):
    p = Path(p)
    return p.read_text() if p.exists() else ""

def overall(text):
    m = re.search(r"overall\s+(\d+)/(\d+)", text)
    return f"{m.group(1)}/{m.group(2)}" if m else "n/a"

def totals(p):
    p = Path(p)
    if not p.exists():
        return None
    t = json.loads(p.read_text())["totals"]
    return f"{t['files']} files: {t['dangerous']} dangerous, {t['suspicious']} suspicious, {t['clean']} clean"

print("Q1 attacks reaching the LLM without a guard :", overall(read("results/q1_no_guard.txt")))
print("Q2 attacks caught by PDFGuard              :", overall(read("results/q2_q3_pdfguard.txt")))
fa = re.search(r"false alarms on clean papers:\s*(\S+)", read("results/q2_q3_pdfguard.txt"))
print("Q3 false alarms on original papers         :", fa.group(1) if fa else "n/a")
t = totals("results/q2_attacked_all.json")
if t:
    print(f"{'Q2 risk levels of the attacked files':<43}:", t)
for name, path in [("Q4 real documented cases", "results/q4_real_cases.json"),
                   ("Q4 their latest versions", "results/q4_real_cases_latest.json")]:
    t = totals(path)
    if t:
        print(f"{name:<43}:", t)
def counts(p):
    p = Path(p)
    return json.loads(p.read_text())["totals"] if p.exists() else None

syn, real = counts("results/q2_attacked_all.json"), counts("results/q4_real_cases.json")
if syn and real:
    files = syn["files"] + real["files"]
    flagged = syn["dangerous"] + syn["suspicious"] + real["dangerous"] + real["suspicious"]
    danger = syn["dangerous"] + real["dangerous"]
    print(f"\nALL injected documents (synthetic + real)  : {files} files: {flagged} flagged "
          f"({danger} dangerous), {files - flagged} missed")
q5 = Path("results/q5_paraphrase.json")
if q5.exists():
    d = json.loads(q5.read_text()); sm = d["summary"]
    tag = "with LLM" if d["llm"] else "rules only"
    print(f"Q5 unseen-wording injections ({tag:<10})   : flagged {sm['injection']['flagged (suspicious or dangerous)']}/"
          f"{sm['injection']['n']}, dangerous {sm['injection']['dangerous']}/{sm['injection']['n']}; "
          f"harmless controls dangerous {sm['control']['dangerous']}/{sm['control']['n']}, "
          f"flagged {sm['control']['flagged (suspicious or dangerous)']}/{sm['control']['n']}")
print("\nAll outputs are in results/  - paste this summary to check the numbers.")
EOF
