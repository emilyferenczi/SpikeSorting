#!/bin/bash
# Re-sort v2, every remaining batch unattended (2026-10-29, Emily: no downtime between batches while she is away).
# Runs the batches of a batch file one after another, like resort_batch.sh, but
#   - cleans the NEXT batch while the current one is being sorted: cleaned copies are written to a staging folder
#     (<base>/aind_input_staged, never the launcher's todo/, so a running sort cannot pick up or move half-made copies) and
#     moved into todo/ only when the previous sort has finished;
#   - waits for any resort_batch.sh / sort launcher already running on this login node before it puts anything into todo/
#     (so it can be started while batch 1 is still running);
#   - after each batch, writes one tar per sorted session (AIND output + its lick_clean report) and deletes the cleaned copies
#     of every session that has a tar (KEEP_CLEANED=1 keeps them);
#   - never stops for one session: a session without its raw folder or lick-onset file, or whose cleaning failed, is skipped
#     and listed at the end. Safe to re-run with the same batch file: sessions with a tar are skipped, so a re-run picks up
#     whatever was skipped (e.g. after copying missing lick files).
# Run from the SpikeSortingEF folder inside /usr/bin/tmux:
#   pipeline/lick_clean/resort_all.sh emily_v2 pipeline/lick_clean/batches_v2.txt
# Settings (environment): CLEAN_PARTITION (kempner_interactive), CLEAN_TIME (06:00:00), MAX_CLEAN (7), KEEP_CLEANED (0),
# RESORT_SRC (raw sessions, default <netscratch>/eferenczi/spikesorting/aind_input). Log: <base>/resort_all.log
set -uo pipefail
PROFILE=${1:?usage: resort_all.sh <profile> <batch file>}; BATCHFILE=${2:?usage: resort_all.sh <profile> <batch file>}
BASE=$(awk -F= -v p="$PROFILE" '$1==p {print $2}' pipeline/user_profiles.conf | tail -1)
[ -n "$BASE" ] || { echo "profile $PROFILE not in pipeline/user_profiles.conf"; exit 1; }
SRC=${RESORT_SRC:-/n/netscratch/bsabatini_lab/Lab/eferenczi/spikesorting/aind_input}
LICKS=$BASE/lick_inputs; TODO=$BASE/aind_input/todo; DONE=$BASE/aind_input; STAGE=$BASE/aind_input_staged
OUT=$BASE/aind_output_fordownload; TR=$BASE/transfer
CLEAN_PARTITION=${CLEAN_PARTITION:-kempner_interactive}; CLEAN_TIME=${CLEAN_TIME:-06:00:00}; MAX_CLEAN=${MAX_CLEAN:-7}
KEEP_CLEANED=${KEEP_CLEANED:-0}
mkdir -p "$TODO" "$TR" "$STAGE"
exec > >(tee -a "$BASE/resort_all.log") 2>&1
mapfile -t BATCHES < <(grep -v '^[[:space:]]*#' "$BATCHFILE" | grep -v '^[[:space:]]*$')
echo "[$(date)] resort_all: ${#BATCHES[@]} batch(es) from $BATCHFILE, profile $PROFILE, base $BASE"
SKIPPED=(); FAILED=()

has_tar() { [ -f "$TR/${1}_v2.tar" ]; }

submit_clean() {   # cleaning jobs for one batch, into the staging folder
  for s in $1; do
    has_tar "$s" && { echo "  $s: already sorted (tar exists)"; continue; }
    if [ -d "$STAGE/$s" ] || [ -d "$TODO/$s" ] || [ -d "$DONE/$s" ]; then echo "  $s: cleaned copy exists"; continue; fi
    [ -d "$SRC/$s" ] || { echo "  $s: raw folder missing in $SRC -- SKIPPED"; SKIPPED+=("$s (raw folder missing)"); continue; }
    [ -f "$LICKS/${s}_lick_onsets_imec.txt" ] || { echo "  $s: lick onsets missing -- SKIPPED"; SKIPPED+=("$s (lick onsets missing)"); continue; }
    while [ "$(squeue -u "$USER" -h -p "$CLEAN_PARTITION" -n lick_clean | wc -l)" -ge "$MAX_CLEAN" ]; do sleep 120; done
    nl=$(wc -l < "$LICKS/${s}_lick_onsets_imec.txt"); mem=$(( 4 + (nl * 3 + 999) / 1000 )); [ $mem -lt 6 ] && mem=6; [ $mem -gt 24 ] && mem=24
    sbatch --partition="$CLEAN_PARTITION" --time="$CLEAN_TIME" --mem=${mem}G pipeline/lick_clean/lick_clean.slrm "$SRC/$s" "$STAGE" \
      --licks "$LICKS/${s}_lick_onsets_imec.txt" --find detect --patch-ms 0.13
  done
}

wait_clean() { while squeue -u "$USER" -h -n lick_clean | grep -q .; do sleep 300; done; }

wait_other_sorts() {   # another resort_batch.sh or launcher on this login node owns todo/ until it finishes
  while pgrep -f "lick_clean/resort_batch.sh" >/dev/null || pgrep -f "shun_spikesort_pipeline.sh" >/dev/null; do
    echo "[$(date)] waiting for a running resort_batch.sh / sort launcher to finish"; sleep 600
  done
}

tar_and_tidy() {
  for s in $1; do
    t="$TR/${s}_v2.tar"
    if [ ! -f "$t" ]; then
      if [ -d "$OUT/${s}_output/spikesorted" ]; then
        cl="$DONE/$s"; [ -d "$cl/lick_clean" ] || cl="$TODO/$s"
        tar -cf "$t.tmp" -C "$OUT/${s}_output" . -C "$cl" lick_clean && mv "$t.tmp" "$t" && echo "  $s: $t"
      else
        echo "  $s: no finished sort -- not tarred"; FAILED+=("$s (no finished sort)")
      fi
    fi
  done
  if [ "$KEEP_CLEANED" != "1" ]; then   # free space: the cleaned copy of every session that has a tar (this batch and earlier ones)
    for d in "$DONE"/*_g[0-9]*; do
      [ -d "$d" ] || continue; s=$(basename "$d"); has_tar "$s" && rm -rf "$d" && echo "  $s: cleaned copy deleted (tar exists)"
    done
  fi
}

[ ${#BATCHES[@]} -gt 0 ] || { echo "no batches"; exit 1; }
echo "[$(date)] cleaning batch 1 of the file into $STAGE"; submit_clean "${BATCHES[0]}"
for i in "${!BATCHES[@]}"; do
  B=${BATCHES[$i]}
  echo "[$(date)] === batch $((i + 1))/${#BATCHES[@]}: $B"
  wait_clean
  for s in $B; do
    if [ -d "$STAGE/$s.partial" ]; then echo "  $s: cleaning did not finish (.partial left) -- SKIPPED"; FAILED+=("$s (cleaning did not finish; see lick_clean_*.out)"); continue; fi
    [ -d "$STAGE/$s" ] && { wait_other_sorts; mv "$STAGE/$s" "$TODO/" && echo "  $s: cleaned copy -> todo"; }
  done
  if [ $((i + 1)) -lt ${#BATCHES[@]} ]; then
    echo "[$(date)] cleaning the next batch while this one sorts"; submit_clean "${BATCHES[$((i + 1))]}"
  fi
  if [ -n "$(ls -A "$TODO")" ]; then
    wait_other_sorts
    echo "[$(date)] sorting: $(ls "$TODO" | tr '\n' ' ')"
    pipeline/shun_spikesort_pipeline.sh pipeline/spike_sort.slrm "$PROFILE" || echo "[$(date)] the launcher reported problems -- see above"
  fi
  tar_and_tidy "$B"
  echo "[$(date)] batch $((i + 1)) done; tars: $(ls "$TR" | grep -c '_v2.tar$')"
done
echo "[$(date)] ALL BATCHES DONE"
[ ${#SKIPPED[@]} -gt 0 ] && { echo "skipped (fix, then run the same command again):"; printf '  %s\n' "${SKIPPED[@]}"; }
[ ${#FAILED[@]} -gt 0 ] && { echo "failed:"; printf '  %s\n' "${FAILED[@]}"; }
ls -la "$TR"
