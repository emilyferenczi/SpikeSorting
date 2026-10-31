#!/bin/bash
# Injection test for the lick cleaning (lick_clean_inject_test.py), as dependent Slurm jobs on kempner_interactive:
#   1 prepare (injected copies A and B of a window)  ->  2 cleaning of A and B with each option set  ->  3 evaluate
# Run from the SpikeSortingEF folder:
#   pipeline/lick_clean/lick_clean_inject_test.sh <session> <T0> <T1> [<fit npz>]
# Work folder: <netscratch>/eferenczi/spikesorting_v2_pilotnew/inject/<session>/ (results: inject_results.json); the cleaned
# copies (~7 GB each) are deleted after the evaluation.
set -euo pipefail
S=${1:?session}; T0=${2:?T0}; T1=${3:?T1}
# LARGE=<file of sample indices>: inject this session's own big, differently shaped transients instead (work folder <session>_large)
# NEARLICK=1: inject the template transients between licks (40-120 ms after a marker; work folder <session>_nearlick)
L=/n/netscratch/bsabatini_lab/Lab/eferenczi; RAW=$L/spikesorting/aind_input/$S; WK=$L/spikesorting_v2_pilotnew/inject/$S${LARGE:+_large}${NEARLICK:+_nearlick}
FIT=${4:-$L/spikesorting_v2/aind_input/todo/$S/lick_clean/lick_clean_fit.npz}; LICKS=$L/spikesorting_v2/lick_inputs/${S}_lick_onsets_imec.txt
PY=/n/holylabs/LABS/bsabatini_lab/Users/eferenczi/envs/raw_extract/bin/python; D=pipeline/lick_clean
SB="sbatch --parsable --partition=kempner_interactive --account=kempner_bsabatini_lab --cpus-per-task=1"
mkdir -p "$WK"
declare -A CFG=(
  [cur]="--find detect --patch-ms 0.13"
  [new]="--find detect --patch-ms 0.13 --second-contact-ms 0.5 3 --marker-pass"
  [low7]="--find detect --patch-ms 0.13 --second-contact-ms 0.5 3 --marker-pass --low-rule 7 0.8 3"
  [low5]="--find detect --patch-ms 0.13 --second-contact-ms 0.5 3 --marker-pass --low-rule 5 0.85 4"
  [low7res]="--find detect --patch-ms 0.13 --second-contact-ms 0.5 3 --marker-pass --low-rule 7 0.8 3 --residual-pass"
  [final]="--find detect --patch-ms 0.13 --second-contact-ms 0.5 3 --marker-pass --low-rule 5 0.6 3"
  [finalL]="--find detect --patch-ms 0.13 --second-contact-ms 0.5 3 --marker-pass --low-rule 5 0.6 3 --large-rule 30 8 3"
  [finalC]="--find detect --patch-ms 0.13 --second-contact-ms 0.5 3 --marker-pass --low-rule 5 0.6 3"
  [finalCL]="--find detect --patch-ms 0.13 --second-contact-ms 0.5 3 --marker-pass --low-rule 5 0.6 3 --large-rule 30 8 3"
  [finalCLk5]="--find detect --patch-ms 0.13 --second-contact-ms 0.5 3 --marker-pass --low-rule 5 0.6 3 --large-rule 30 8 3 --k 5"
  [finalCLm]="--find detect --patch-ms 0.13 --second-contact-ms 0.5 3 --marker-pass --low-rule 5 0.6 3 --large-rule 30 8 3 --large-model lowrank"
  [finalCLmP]="--find detect --patch-ms 0.13 --second-contact-ms 0.5 3 --marker-pass --low-rule 5 0.6 3 --large-rule 30 8 3 --large-model lowrank --large-patch-ms 0.27"
  [prod4]="--find detect --patch-ms 0.13 --second-contact-ms 0.5 3 --marker-pass --low-rule 5 0.6 3 --large-rule 30 8 3 --large-model lowrank --burst-pass 12 4 4 --burst-pad-ms 4"
  [prod4T2]="--find detect --patch-ms 0.13 --patch-post-ms 0.07 --second-contact-ms 0.5 3 --marker-pass --low-rule 5 0.6 3 --large-rule 30 8 3 --large-model lowrank --burst-pass 12 4 4 --burst-pad-ms 4"
  [prod4T4]="--find detect --patch-ms 0.13 --patch-post-ms 0.13 --second-contact-ms 0.5 3 --marker-pass --low-rule 5 0.6 3 --large-rule 30 8 3 --large-model lowrank --burst-pass 12 4 4 --burst-pad-ms 4"
  [finalCLk8]="--find detect --patch-ms 0.13 --second-contact-ms 0.5 3 --marker-pass --low-rule 5 0.6 3 --large-rule 30 8 3 --k 8"
)
# (finalC / finalCL: the same options as final / finalL with the per-session calibrated, lick-only lower threshold, fork 2026-10-29 17:00)
[ -n "${ONLY:-}" ] && { for c in "${!CFG[@]}"; do [[ " $ONLY " == *" $c "* ]] || unset "CFG[$c]"; done; }   # ONLY="final" runs only those
if [ -f "$WK/inject.npz" ] && [ -d "$WK/A/$S" ] && [ -d "$WK/B/$S" ]; then
  j1=$(sbatch --parsable --partition=kempner_interactive --account=kempner_bsabatini_lab --time=00:01:00 --mem=1G -o /dev/null --wrap="true")   # copies exist
else
j1=$($SB --job-name=inj_prep --time=01:00:00 --mem=16G -o "$WK/prepare_%j.out" \
     --wrap="$PY $D/lick_clean_inject_test.py prepare $RAW $WK --fit $FIT --licks $LICKS --window $T0 $T1 ${LARGE:+--large $LARGE} ${NEARLICK:+--near-licks}")
fi
echo "prepare: job $j1"
deps=""; runs=""
for c in "${!CFG[@]}"; do
  for v in A B; do
    j=$($SB --job-name=inj_$c$v --time=02:00:00 --mem=24G --dependency=afterok:$j1 -o "$WK/clean_${c}_${v}_%j.out" \
        --wrap="mkdir -p $WK/out_$c/$v && $PY $D/lick_clean_raw.py $WK/$v/$S $WK/out_$c/$v --licks $WK/${S}_lick_onsets_window.txt --no-scan ${CFG[$c]}")
    deps="$deps:$j"
  done
  runs="$runs $c=$WK/out_$c/A,$WK/out_$c/B"
done
j3=$($SB --job-name=inj_eval --time=01:00:00 --mem=16G --dependency=afterok$deps -o "$WK/evaluate_%j.out" \
     --wrap="$PY $D/lick_clean_inject_test.py evaluate $WK $runs")
echo "cleaning jobs:${deps//:/ }; evaluate: job $j3 (results: $WK/inject_results.json)"
