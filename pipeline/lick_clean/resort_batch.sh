#!/bin/bash
# Re-sort v2, one batch (2026-10-26; cleaning on kempner_interactive since 2026-10-29, Kempner's advice for short CPU-only jobs;
# CLEAN_PARTITION=test switches back to the FASRC test partition): clean each session's raw data (lick_clean.slrm: --find detect --patch-ms 0.13, i.e.
# lick-contact transients found by shape, 3 components per electrode block subtracted, +-0.13 ms patch at each peak), sort
# the cleaned copies with the launcher (EF fixes via the profile), then tar each output with its cleaning report for transfer.
# Safe to re-run (e.g. after a login-node restart): existing cleaned copies are not redone, sessions already sorted are
# skipped by the launcher, existing tars are kept. Refuses to sort while a cleaning job is unfinished or failed.
# Run from the SpikeSortingEF folder inside /usr/bin/tmux:
#   pipeline/lick_clean/resort_batch.sh emily_v2 <session> [<session> ...]
# Inputs: raw session folders in $RESORT_SRC (default <netscratch>/eferenczi/spikesorting/aind_input), lick onsets
# <base>/lick_inputs/<session>_lick_onsets_imec.txt (from the server's sync files; extractLickContacts.py in gpe-neuropixels).
# Outputs: <base>/aind_output_fordownload/<session>_output, tars in <base>/transfer/<session>_v2.tar
set -uo pipefail
PROFILE=${1:?usage: resort_batch.sh <profile> <session> [...]}; shift
[ $# -gt 0 ] || { echo "no sessions given"; exit 1; }
BASE=$(awk -F= -v p="$PROFILE" '$1==p {print $2}' pipeline/user_profiles.conf | tail -1)
[ -n "$BASE" ] || { echo "profile $PROFILE not in pipeline/user_profiles.conf"; exit 1; }
SRC=${RESORT_SRC:-/n/netscratch/bsabatini_lab/Lab/eferenczi/spikesorting/aind_input}
LICKS=$BASE/lick_inputs; TODO=$BASE/aind_input/todo; DONE=$BASE/aind_input; OUT=$BASE/aind_output_fordownload; TR=$BASE/transfer
mkdir -p "$TODO" "$TR"
echo "[$(date)] batch of $# session(s), profile $PROFILE, base $BASE"

# 1. cleaning jobs (CPU only, no GPU, on CLEAN_PARTITION: at most MAX_CLEAN of this user's jobs there at a time)
CLEAN_PARTITION=${CLEAN_PARTITION:-kempner_interactive}; MAX_CLEAN=${MAX_CLEAN:-5}
for s in "$@"; do
  if [ -d "$TODO/$s" ] || [ -d "$DONE/$s" ]; then echo "  $s: cleaned copy exists"; continue; fi
  [ -d "$SRC/$s" ] || { echo "  $s: raw folder missing in $SRC -- SKIPPED"; continue; }
  [ -f "$LICKS/${s}_lick_onsets_imec.txt" ] || { echo "  $s: $LICKS/${s}_lick_onsets_imec.txt missing -- SKIPPED"; continue; }
  while [ "$(squeue -u "$USER" -h -p "$CLEAN_PARTITION" | wc -l)" -ge "$MAX_CLEAN" ]; do sleep 120; done
  # memory from the session's lick count (~2.6 MB per transient for the snippets + ~4 GB): 6 GB for a sated day, ~23 GB
  # for the busiest session, so the request stays close to what the job uses
  nl=$(wc -l < "$LICKS/${s}_lick_onsets_imec.txt"); mem=$(( 4 + (nl * 3 + 999) / 1000 )); [ $mem -lt 6 ] && mem=6; [ $mem -gt 24 ] && mem=24
  sbatch --partition="$CLEAN_PARTITION" --mem=${mem}G pipeline/lick_clean/lick_clean.slrm "$SRC/$s" "$TODO" --licks "$LICKS/${s}_lick_onsets_imec.txt" --find detect --patch-ms 0.13
done

# 2. wait for every cleaning job of this user; refuse to sort if a copy is incomplete
while squeue -u "$USER" -h -n lick_clean | grep -q .; do sleep 300; done
if ls -d "$TODO"/*.partial >/dev/null 2>&1; then
  echo "[$(date)] a cleaning job did not finish (.partial folder left) -- NOT sorting. Check lick_clean_*.out / .err:"; ls "$TODO"; exit 1
fi
echo "[$(date)] cleaning done: $(ls "$TODO" | wc -l) session(s) to sort"

# 3. sort (the launcher skips sessions already sorted; moves todo -> aind_input only when every check passed)
if [ -n "$(ls -A "$TODO")" ]; then
  pipeline/shun_spikesort_pipeline.sh pipeline/spike_sort.slrm "$PROFILE" || echo "[$(date)] the launcher reported problems -- see above"
fi

# 4. one tar per sorted session: the AIND output + its cleaning report
for s in "$@"; do
  t="$TR/${s}_v2.tar"
  [ -f "$t" ] && { echo "  $s: $t exists"; continue; }
  [ -d "$OUT/${s}_output/spikesorted" ] || { echo "  $s: no finished sort in $OUT -- no tar"; continue; }
  cl="$DONE/$s"; [ -d "$cl/lick_clean" ] || cl="$TODO/$s"
  tar -cf "$t.tmp" -C "$OUT/${s}_output" . -C "$cl" lick_clean && mv "$t.tmp" "$t" && echo "  $s: $t"
done
echo "[$(date)] BATCH DONE"; ls -la "$TR"
