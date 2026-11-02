#!/bin/bash
# Per-unit lick-artifact check (unit_lick_check.py) for every sorted session of a batch, before its cleaned copy is deleted
# (2026-10-30, Emily: artifact units excluded everywhere, contaminated units excluded from lick-aligned analyses -- no spike
# masking around licks). One kempner_interactive job per session, all at once; waits for them. Results go into the sort's output
# folder, <base>/aind_output_fordownload/<session>_output/unit_lick_check/, so the session's tar carries them.
# Skips sessions without a finished sort or a cleaned copy, and sessions already checked. Run from the SpikeSortingEF folder:
#   pipeline/lick_clean/unit_check_batch.sh <base> <session> [<session> ...]
set -uo pipefail
BASE=${1:?usage: unit_check_batch.sh <base> <session> ...}; shift
OUT=$BASE/aind_output_fordownload; DONE=$BASE/aind_input; TODO=$BASE/aind_input/todo
PY=/n/holylabs/LABS/bsabatini_lab/Users/eferenczi/envs/raw_extract/bin/python
jobs=()
for s in "$@"; do
  sd="$OUT/${s}_output/AIND_$s"; uc="$OUT/${s}_output/unit_lick_check"
  [ -f "$uc/unit_lick_check.csv" ] && { echo "  $s: unit check exists"; continue; }
  [ -d "$sd" ] || { echo "  $s: no finished sort -- no unit check"; continue; }
  cl="$DONE/$s"; [ -d "$cl/lick_clean" ] || cl="$TODO/$s"
  [ -d "$cl/lick_clean" ] || { echo "  $s: cleaned copy not found -- no unit check"; continue; }
  j=$(sbatch --parsable --job-name=unit_check --partition=kempner_interactive --account=kempner_bsabatini_lab --time=04:00:00 --mem=32G \
      -o "$BASE/unit_check_${s}_%j.out" --wrap="$PY pipeline/lick_clean/unit_lick_check.py $cl $sd $uc")
  echo "  $s: unit check job $j"; jobs+=("$j")
done
if [ ${#jobs[@]} -gt 0 ]; then
  # each id on its own, active states only (2026-11-02: a combined -j list fails outright once one id is purged, and a finished
  # job stays listed for MinJobAge) -- the wait could end before every check had finished
  active() { local j; for j in "${jobs[@]}"; do squeue -h -j "$j" -t PD,CF,R,CG,S 2>/dev/null | grep -q . && return 0; done; return 1; }
  while active; do sleep 120; done
  for s in "$@"; do [ -f "$OUT/${s}_output/unit_lick_check/unit_lick_check.json" ] && echo "  $s: $(tr -d '\n ' < "$OUT/${s}_output/unit_lick_check/unit_lick_check.json" | grep -o "\"classes_old\":{[^}]*}")"; done
fi
