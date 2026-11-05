#!/bin/bash
# Hybrid ground-truth test, one session (EF, 2026-11-05; plan: gpe-neuropixels neuropixels/spikesorting/HYBRID_TEST_PLAN.md).
# Run from the SpikeSortingEF folder inside tmux:
#   pipeline/hybrid/run_hybrid.sh build <session>    hybrid copy (make_hybrid.slrm; waits): <H>/hybrid_raw/<session>
#   pipeline/hybrid/run_hybrid.sh sort <session> [...]  both conditions on those copies, at the same time:
#       clean    production v2: resort_batch.sh emily_v2hybrid (lick cleaning with the production CLEAN_OPTS -> sort -> per-unit
#                lick check -> tar <H>/clean/transfer/<session>_v2.tar)
#       noclean  the same sort on the uncleaned copy (profile emily_v2hybrid_noclean; a hard-linked copy of the hybrid folder,
#                no extra space), then the per-unit lick check against the clean condition's contacts (a view folder: the
#                uncleaned recording + the cleaning's lick_clean/) -> tar <H>/noclean/transfer/<session>_noclean.tar
# <H> = /n/netscratch/bsabatini_lab/Lab/eferenczi/spikesorting_v2_hybrid; design files in <H>/design/<session>_gt_design.npz;
# raw input (read only): the v2 raw folders, <netscratch>/eferenczi/spikesorting/aind_input/<session>; lick onsets from
# spikesorting_v2/lick_inputs. Nothing outside <H> is written.
set -uo pipefail
STAGE=${1:?usage: run_hybrid.sh build|sort <session> [...]}; s=${2:?session}
H=/n/netscratch/bsabatini_lab/Lab/eferenczi/spikesorting_v2_hybrid
RAW=/n/netscratch/bsabatini_lab/Lab/eferenczi/spikesorting/aind_input
LICKS=/n/netscratch/bsabatini_lab/Lab/eferenczi/spikesorting_v2/lick_inputs
PY=/n/holylabs/LABS/bsabatini_lab/Users/eferenczi/envs/raw_extract/bin/python
mkdir -p "$H/hybrid_raw" "$H/logs"
echo "[$(date)] $STAGE $s"

if [ "$STAGE" = build ]; then
  [ -d "$H/hybrid_raw/$s" ] && { echo "  hybrid copy exists: $H/hybrid_raw/$s"; exit 0; }
  [ -f "$H/design/${s}_gt_design.npz" ] || { echo "  design missing: $H/design/${s}_gt_design.npz"; exit 1; }
  jid=$(sbatch --parsable -o "$H/logs/make_hybrid_${s}_%j.out" -e "$H/logs/make_hybrid_${s}_%j.err" pipeline/hybrid/make_hybrid.slrm \
        "$RAW/$s" "$H/hybrid_raw" "$H/design/${s}_gt_design.npz") || exit 1
  echo "  make_hybrid job $jid"
  while squeue -h -j "$jid" -t PD,CF,R,CG,S 2>/dev/null | grep -q .; do sleep 120; done
  [ -d "$H/hybrid_raw/$s" ] && echo "[$(date)] built: $H/hybrid_raw/$s" || { echo "[$(date)] BUILD FAILED, see $H/logs/make_hybrid_${s}_$jid.*"; exit 1; }
  exit 0
fi

[ "$STAGE" = sort ] || { echo "unknown stage $STAGE"; exit 1; }
# sort: every session given (2026-11-05: several at once -- one cleaning / sort batch per condition, since a batch sorts
# everything in its profile's todo folder and refuses while any cleaning there is unfinished)
SESS=("${@:2}")
mkdir -p "$H/clean/lick_inputs" "$H/noclean/aind_input/todo" "$H/noclean/transfer"
for s in "${SESS[@]}"; do
  [ -d "$H/hybrid_raw/$s" ] || { echo "  no hybrid copy: $H/hybrid_raw/$s"; exit 1; }
  cp -n "$LICKS/${s}_lick_onsets_imec.txt" "$H/clean/lick_inputs/"
  if [ ! -d "$H/noclean/aind_input/todo/$s" ] && [ ! -d "$H/noclean/aind_input/$s" ]; then
    cp -al "$H/hybrid_raw/$s" "$H/noclean/aind_input/todo/$s" && echo "  $s noclean input: hard links in $H/noclean/aind_input/todo/$s"
  fi
done
TAG=$(IFS=_; echo "${SESS[*]}")
( RESORT_SRC="$H/hybrid_raw" pipeline/lick_clean/resort_batch.sh emily_v2hybrid "${SESS[@]}" > "$H/logs/clean_${TAG}.log" 2>&1 ) &
pc=$!
( pipeline/shun_spikesort_pipeline.sh pipeline/spike_sort.slrm emily_v2hybrid_noclean > "$H/logs/noclean_${TAG}.log" 2>&1 ) &
pn=$!
echo "  clean (pid $pc, log $H/logs/clean_${TAG}.log) and noclean (pid $pn, log $H/logs/noclean_${TAG}.log) running"
wait $pn; echo "[$(date)] noclean sort(s) finished (exit $?)"

# noclean: per-unit lick check against the clean condition's contacts, as soon as that session's cleaning is done
jobs_nc=()
for s in "${SESS[@]}"; do
  OUT=$H/noclean/aind_output_fordownload/${s}_output; uc=$OUT/unit_lick_check
  [ -d "$OUT/AIND_$s" ] || { echo "  $s: no noclean sort -- no unit check"; continue; }
  [ -f "$uc/unit_lick_check.csv" ] && continue
  until [ -f "$H/clean/aind_input/todo/$s/lick_clean/lick_clean_fit.npz" ] || [ -f "$H/clean/aind_input/$s/lick_clean/lick_clean_fit.npz" ] \
        || ! kill -0 $pc 2>/dev/null; do sleep 120; done
  CL=$H/clean/aind_input/$s; [ -d "$CL/lick_clean" ] || CL=$H/clean/aind_input/todo/$s
  [ -d "$CL/lick_clean" ] || { echo "  $s: cleaned copy not found -- no noclean unit check"; continue; }
  V=$H/noclean_view/$s; mkdir -p "$V"
  ln -sfn "$H/hybrid_raw/$s/${s}_imec0" "$V/${s}_imec0"; ln -sfn "$CL/lick_clean" "$V/lick_clean"
  j=$(sbatch --parsable --job-name=unit_check --partition=kempner_interactive --account=kempner_bsabatini_lab --time=04:00:00 --mem=32G \
      -o "$H/logs/unit_check_noclean_${s}_%j.out" --wrap="$PY pipeline/lick_clean/unit_lick_check.py $V $OUT/AIND_$s $uc")
  echo "  $s: noclean unit check job $j"; jobs_nc+=("$j")
done
wait $pc; echo "[$(date)] clean condition finished (exit $?)"
for j in "${jobs_nc[@]}"; do while squeue -h -j "$j" -t PD,CF,R,CG,S 2>/dev/null | grep -q .; do sleep 120; done; done
for s in "${SESS[@]}"; do
  OUT=$H/noclean/aind_output_fordownload/${s}_output; t=$H/noclean/transfer/${s}_noclean.tar
  if [ -d "$OUT/spikesorted" ] && [ ! -f "$t" ]; then
    tar -cf "$t.tmp" -C "$OUT" . && mv "$t.tmp" "$t" && echo "  $t"
  fi
  echo "[$(date)] HYBRID $s DONE"
done
ls -la "$H/clean/transfer" "$H/noclean/transfer" 2>/dev/null
