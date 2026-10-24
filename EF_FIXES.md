# EF fixes (branch `ef-fixes`, Emily Ferenczi, 2026-10-24)

Three switches in `pipeline/run_capsule_custom.py`, **all off by default**: a run without them is unchanged.
Pass them inside `--preprocessing_args`, e.g.

    --preprocessing_args '--denoising destripe --t-start 0 --t-stop <end> --no-remove-bad-channels --destripe-float32 --destripe-pad-to-group --destripe-strict'

| switch | what it fixes |
|---|---|
| `--destripe-float32` | SpikeInterface < 0.104 (the containers run 0.101.2) divides by destripe's AGC gain **in place in the input dtype**. Our high-pass output is int16, so every sample smaller than the local mean \|x\| became 0 and the rest were truncated to whole multiples of it. With this switch, interpolation / CMR / destripe run on float32 and the result is rounded back to int16 before saving (same file format for the sorter). Same result as SpikeInterface >= 0.104, which casts to float32 itself. |
| `--destripe-pad-to-group` | destripe pads each shank with `n_channel_pad` = 60 channels; SpikeInterface refuses when a shank has fewer channels, and the pipeline then silently fell back to CMR (EF407: shanks with 52-54 channels). Caps the padding / taper at the shank's channel count. |
| `--destripe-strict` | stop with an error when destripe fails, instead of falling back to CMR. |

Unconditional (metadata only): when destripe does fall back to CMR, `processing.json` now records `denoising_strategy: cmr` and
`denoising_requested: destripe` (before, it still said `destripe`; only the notes mentioned the fallback). The switches used
are recorded under `preprocessing.ef_fixes`.

Tested 2026-10-24 with SpikeInterface 0.101.2 on 5 min of 20260805_EF405_g0: as before (int16) 27% exact zeros, noise 9.0 uV;
`--destripe-float32` 7.2% zeros, noise 18.0 uV (= the bug-free result); a 50-channel shank fails without
`--destripe-pad-to-group` and runs with it.

**Status: pending the test re-sort of 20260805_EF405_g0.** Until that is checked, `main` (= Shun's pipeline) is what produced
all existing sorts. Which setup to use for the GPe project is tracked in gpe-neuropixels `neuropixels/spikesorting/README.md`.

## Running the test re-sort on Kempner (20260805_EF405_g0)

The launcher has an `EF_FIXES` knob (default `false`), and `pipeline/user_profiles.conf` has a profile `emily_eftest`:
`emily_noband` (the profile of the existing GPe sorts: high-pass 300 Hz only, destripe, bad channels interpolated, not removed)
+ `EF_FIXES=true`, under its own scratch folder `/n/netscratch/bsabatini_lab/Lab/eferenczi/spikesorting_eftest`. It produces

    --denoising destripe --t-start 0 --t-stop <end> --no-remove-bad-channels --destripe-float32 --destripe-pad-to-group --destripe-strict

(the existing 0805 sort: the same without the last three). Always give a profile: without one the launcher uses Shun's folders.

    cd /n/holylabs/bsabatini_lab/Users/eferenczi
    git clone -b ef-fixes https://github.com/emilyferenczi/SpikeSorting.git SpikeSortingEF
    cd SpikeSortingEF
    mkdir -p /n/netscratch/bsabatini_lab/Lab/eferenczi/spikesorting_eftest/aind_input/todo
    # put (copy or link) the session folder 20260805_EF405_g0 into .../spikesorting_eftest/aind_input/todo/
    tmux new -s eftest
    pipeline/shun_spikesort_pipeline.sh pipeline/spike_sort.slrm emily_eftest

Output to download: `.../spikesorting_eftest/aind_output_fordownload/20260805_EF405_g0_output` (`aind_output_scratch` is the pipeline's working copy). In its `processing.json`, preprocessing
should show `ef_fixes` all true and `denoising_strategy: destripe`; the job log should show
`[custom] EF fixes: interpolation / CMR / destripe on float32`.

After the test, if accepted: `pipeline/shun_spikesort_pipeline.sh pipeline/spike_sort.slrm emily_noband EF_FIXES=true` (or add
`emily_noband.EF_FIXES=true` to `user_profiles.conf`).
