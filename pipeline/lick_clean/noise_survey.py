"""Noise survey of raw AP recordings (EF, 2026-10-29; Emily: 'how noisy is EF407, can it be fixed, how does it affect the
sorting?'). Per session, 6 windows of 5 s spread over the recording, high-passed 300 Hz (as the sorter's input):
  - per channel robust SD (uV): raw; after subtracting each electrode block's median (block common mode); after subtracting the
    whole probe's median (global CMR, roughly what the sorter's spatial filter removes of a shared signal)
  - each block's common mode: robust SD (uV) and its power spectrum (Welch, 300-7500 Hz)
  - one channel per block, after block CMR: power spectrum (what stays after the common part is removed)
  - burstiness: robust SD of the block common mode in 100-ms bins, 5th / 50th / 95th percentile across all bins
Writes <out>/<session>_noise.npz (+ <out>/<session>_noise.json summary). numpy / scipy only.
Run: python noise_survey.py <out dir> <raw session dir> [<raw session dir> ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

import lick_clean_raw as LC
import lick_clean_scan as LS


def survey(src, out):
    src = Path(src); ses = src.name; apf = next((src / f"{ses}_imec0").glob("*.ap.bin")); M = LC.read_meta(apf.with_suffix(".meta"))
    fs = float(M.get("imSampRate", LC.FS_DEFAULT)); nsv = int(M["nSavedChans"]); nch = nsv - 1
    uv = float(M["imAiRangeMax"]) / float(M["imMaxInt"]) / float(M.get("imChan0apGain", 100)) * 1e6
    sh, y = LC.geometry(M); blocks = sh * 2 + (y > LC.SPLIT_UM); ub = np.unique(blocks)
    X = np.memmap(apf, dtype=np.int16, mode="r").reshape(-1, nsv); n = X.shape[0]; good = LS.working_channels(X, nch, fs)
    sos = butter(5, [300, 7500], "bandpass", fs=fs, output="sos"); mg = int(0.05 * fs); L = int(5 * fs)
    starts = np.linspace(int(0.08 * n), int(0.92 * n) - L, 6).astype(int)
    sd_raw, sd_blk, sd_glob, cm_sd, bins = [], [], [], [], []; P_cm, P_ch = 0, 0
    for a in starts:
        x = sosfiltfilt(sos, np.asarray(X[a - mg:a + L + mg, :nch], np.float32), axis=0)[mg:mg + L] * uv
        x = x[:, good]; bg = blocks[good]
        cm = np.stack([np.median(x[:, bg == b], 1) for b in ub], 1)
        xb = x - cm[:, np.searchsorted(ub, bg)]; xg = x - np.median(x, 1)[:, None]
        r = lambda v: 1.4826 * np.median(np.abs(v - np.median(v, 0)), 0)
        sd_raw.append(r(x)); sd_blk.append(r(xb)); sd_glob.append(r(xg)); cm_sd.append(r(cm))
        k = int(0.1 * fs); bins.append(1.4826 * np.median(np.abs(cm[:L // k * k].reshape(-1, k, len(ub))), 1))
        f, p = welch(cm, fs=fs, nperseg=4096, axis=0); P_cm = P_cm + p
        one = np.array([np.flatnonzero(bg == b)[len(np.flatnonzero(bg == b)) // 2] for b in ub])
        f, p2 = welch(xb[:, one], fs=fs, nperseg=4096, axis=0); P_ch = P_ch + p2
    sd_raw, sd_blk, sd_glob = (np.median(np.stack(v), 0) for v in (sd_raw, sd_blk, sd_glob)); cm_sd = np.median(np.stack(cm_sd), 0)
    bins = np.concatenate(bins); b5, b50, b95 = np.percentile(bins, [5, 50, 95], axis=0)
    S = dict(session=ses, working_channels=int(good.sum()), ch_sd_raw_uv=float(np.median(sd_raw)), ch_sd_after_block_cmr_uv=float(np.median(sd_blk)),
             ch_sd_after_global_cmr_uv=float(np.median(sd_glob)), cm_sd_uv_by_block=cm_sd.round(2).tolist(),
             cm_bin_sd_p5_p50_p95_median_over_blocks=[float(np.median(b5)), float(np.median(b50)), float(np.median(b95))],
             blocks=[int(b) for b in ub])
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{ses}_noise.npz", f=f, P_cm=P_cm / len(starts), P_ch=P_ch / len(starts), sd_raw=sd_raw, sd_blk=sd_blk, sd_glob=sd_glob,
                        cm_sd=cm_sd, bins=bins, blocks=ub, depth=y[good], shank=sh[good])
    (out / f"{ses}_noise.json").write_text(json.dumps(S, indent=1)); print(json.dumps(S), flush=True)


def by_licking(src, licks, out, n_win=40, seed=0):
    """2026-10-29: noise DURING licking vs lick-free time (EF407: probe-wide bursts 13x denser while licking -- muscle?). Windows of
    200 ms inside lick bouts (a lick marker within 150 ms on both sides) vs lick-free (> 1 s from every marker); samples within
    -50..+10 ms of a marker (the contact transient) left out. Per window: channel robust SD raw / block CM removed / probe CM
    removed, block common-mode SD (uV). -> <out>/<session>_noise_licking.json"""
    src = Path(src); ses = src.name; apf = next((src / f"{ses}_imec0").glob("*.ap.bin")); M = LC.read_meta(apf.with_suffix(".meta"))
    fs = float(M.get("imSampRate", LC.FS_DEFAULT)); nsv = int(M["nSavedChans"]); nch = nsv - 1
    uv = float(M["imAiRangeMax"]) / float(M["imMaxInt"]) / float(M.get("imChan0apGain", 100)) * 1e6
    sh, y = LC.geometry(M); blocks = sh * 2 + (y > LC.SPLIT_UM); ub = np.unique(blocks)
    X = np.memmap(apf, dtype=np.int16, mode="r").reshape(-1, nsv); n = X.shape[0]; good = LS.working_channels(X, nch, fs); bg = blocks[good]
    mk = np.sort(np.loadtxt(licks, dtype=np.int64).ravel()); rng = np.random.default_rng(seed); w = int(0.2 * fs); mg = int(0.05 * fs)
    sos = butter(5, [300, 7500], "bandpass", fs=fs, output="sos")
    j = np.searchsorted(mk, mk); gaps = np.diff(mk); inb = np.flatnonzero((gaps[:-1] < 0.15 * fs) & (gaps[1:] < 0.15 * fs)) + 1   # markers inside bouts
    bout_t = rng.choice(mk[inb], min(n_win, len(inb)), replace=False) if len(inb) else np.array([], np.int64)
    free_t = []
    while len(free_t) < n_win:
        t = int(rng.integers(mg + w, n - mg - w)); k = np.searchsorted(mk, t)
        d = min(abs(t - mk[k - 1]) if k > 0 else 10 ** 12, abs(mk[k] - t) if k < len(mk) else 10 ** 12)
        if d > fs:
            free_t.append(t)
    res = {}
    for nm, T in (("licking", bout_t), ("lick_free", np.array(free_t))):
        acc = {k: [] for k in ("raw", "block", "global", "cm")}
        for t in T:
            a = int(t - w // 2); x = sosfiltfilt(sos, np.asarray(X[a - mg:a + w + mg, :nch], np.float32), axis=0)[mg:mg + w][:, good] * uv
            keep = np.ones(w, bool)
            for m_ in mk[(mk > a - int(0.02 * fs)) & (mk < a + w + int(0.06 * fs))]:
                keep[max(0, m_ - a - int(0.05 * fs)):max(0, m_ - a + int(0.01 * fs))] = False
            if keep.sum() < 0.3 * w:
                continue
            x = x[keep]; cm = np.stack([np.median(x[:, bg == b], 1) for b in ub], 1)
            r = lambda v: 1.4826 * np.median(np.abs(v - np.median(v, 0)), 0)
            acc["raw"].append(np.median(r(x))); acc["block"].append(np.median(r(x - cm[:, np.searchsorted(ub, bg)])))
            acc["global"].append(np.median(r(x - np.median(x, 1)[:, None]))); acc["cm"].append(np.median(r(cm)))
        res[nm] = {k: [float(np.median(v)), float(np.percentile(v, 90))] if v else None for k, v in acc.items()}; res[nm]["windows"] = len(acc["raw"])
    res["session"] = ses; out = Path(out); out.mkdir(parents=True, exist_ok=True)
    (out / f"{ses}_noise_licking.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--licking":     # noise_survey.py --licking <out> <lick dir> <raw session dir> ...
        for s in sys.argv[4:]:
            try:
                by_licking(s, Path(sys.argv[3]) / f"{Path(s).name}_lick_onsets_imec.txt", sys.argv[2])
            except Exception as e:
                print(Path(s).name, "FAILED:", e, flush=True)
    else:
        for s in sys.argv[2:]:
            try:
                survey(s, sys.argv[1])
            except Exception as e:
                print(Path(s).name, "FAILED:", e, flush=True)
