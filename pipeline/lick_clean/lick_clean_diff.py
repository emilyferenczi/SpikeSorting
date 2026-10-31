"""Cleaned - raw difference of a lick-cleaned recording, kept small so the cleaning can be looked at later without the cleaned
copy (EF, 2026-10-30; Emily: 'is there a way to keep this so that we can check the cleaning vs raw comparison later on the iMac?').
The cleaning changes the recording only in short windows around the cleaned transients, so the cleaned file = the raw file +
this difference, exactly. One sequential pass over both files (4-s chunks):
  - every sample where any channel differs -> runs of changed samples (gaps < 1 ms merged) and the int16 difference there
    (all channels) -> <cleaned>/lick_clean/lick_clean_diff.npz (compressed; travels with the sort's tar to the server)
  - check B6 (RESORT_V2_CHECKS.md): same size, sync channel identical, every changed sample inside a cleaning window
    ([start, start + window) of an event in lick_clean_fit.npz, +- the alignment jitter), no sample pushed to the int16 limits,
    difference within int16 -> lick_clean_diff.json (ok = all passed)
Rebuild anywhere (e.g. the iMac, raw from the server): cleaned_window(raw_memmap, D, s0, s1) below.
Run: python lick_clean_diff.py <raw session dir> <cleaned session dir>      (numpy only)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

import lick_clean_raw as LC


def cleaned_window(X, D, s0, s1):
    """cleaned samples s0:s1 (all saved channels) from the raw memmap X (n x nSavedChans) and a loaded lick_clean_diff.npz D"""
    s0, s1 = max(int(s0), 0), min(int(s1), X.shape[0])                  # 2026-10-31: windows reaching past either end of the file
    out = np.asarray(X[s0:s1], np.int32); st, en, off = D["run_start"], D["run_end"], D["run_offset"]; nch = int(D["n_channels"])
    for i in range(np.searchsorted(en, s0, "right"), np.searchsorted(st, s1, "left")):
        a, b = max(st[i], s0), min(en[i], s1)
        out[a - s0:b - s0, :nch] += D["diff"][off[i] + a - st[i]:off[i] + b - st[i]]
    return out.astype(np.int16)


def main():
    raw, cl = Path(sys.argv[1]), Path(sys.argv[2]); ses = cl.name; t0 = time.time()
    apr = next((raw / f"{ses}_imec0").glob("*.ap.bin")); apc = next((cl / f"{ses}_imec0").glob("*.ap.bin"))
    M = LC.read_meta(apc.with_suffix(".meta")); nsv = int(M["nSavedChans"]); nch = nsv - 1; fs = float(M.get("imSampRate", LC.FS_DEFAULT))
    ok = {"same_size": apr.stat().st_size == apc.stat().st_size}
    X = np.memmap(apr, dtype=np.int16, mode="r").reshape(-1, nsv); Y = np.memmap(apc, dtype=np.int16, mode="r").reshape(-1, nsv); n = min(len(X), len(Y))
    F = np.load(cl / "lick_clean" / "lick_clean_fit.npz", allow_pickle=True)
    W = int((LC.PRE_MS + LC.POST_MS) * fs / 1000); J = int(LC.JIT_MS * fs / 1000)
    s = np.unique(np.asarray(F["starts"], np.int64)); win = (s - J, s + W + J)            # expected windows
    if "burst_start" in F and len(F["burst_start"]):                                      # + the burst pass's stretches
        win = (np.r_[win[0], np.asarray(F["burst_start"], np.int64)], np.r_[win[1], np.asarray(F["burst_end"], np.int64)])
    C = int(4 * fs); runs, diffs, n_sync, n_clip, dmax = [], [], 0, 0, 0
    for c0 in range(0, n, C):
        x = np.asarray(X[c0:c0 + C], np.int32); y = np.asarray(Y[c0:c0 + C], np.int32); d = y - x
        n_sync += int(np.count_nonzero(d[:, nch]))
        ch = np.flatnonzero(np.any(d[:, :nch] != 0, 1))
        if not len(ch):
            continue
        n_clip += int(np.count_nonzero(((y[ch, :nch] >= 32767) | (y[ch, :nch] <= -32768)) & (d[ch, :nch] != 0)))
        dmax = max(dmax, int(np.abs(d[ch, :nch]).max()))
        br = np.flatnonzero(np.diff(ch) > int(0.001 * fs)) + 1                              # runs, gaps < 1 ms merged
        for a, b in zip(np.r_[0, br], np.r_[br, len(ch)]):
            r0, r1 = ch[a], ch[b - 1] + 1; runs.append((c0 + r0, c0 + r1)); diffs.append(d[r0:r1, :nch].astype(np.int16))
    # merge runs split at chunk borders
    mr, md = [], []
    for (a, b), dd in zip(runs, diffs):
        if mr and a <= mr[-1][1] + int(0.001 * fs):
            gap = a - mr[-1][1]; md[-1] = np.concatenate([md[-1], np.zeros((gap, nch), np.int16), dd]); mr[-1] = (mr[-1][0], b)
        else:
            mr.append((a, b)); md.append(dd)
    rs = np.array([a for a, _ in mr], np.int64); re_ = np.array([b for _, b in mr], np.int64)
    # every changed sample inside an expected window
    ws, we = win; o = np.argsort(ws); ws, we = ws[o], np.maximum.accumulate(we[o])
    outside = 0
    for a, b, dd in zip(rs, re_, md):
        t = a + np.flatnonzero(np.any(dd != 0, 1)); i = np.searchsorted(ws, t, "right") - 1
        outside += int(np.count_nonzero((i < 0) | (t >= we[np.maximum(i, 0)])))
    nchg = int(sum(int(np.count_nonzero(np.any(dd != 0, 1))) for dd in md))
    ok.update(sync_identical=n_sync == 0, all_changes_in_windows=outside == 0, no_clipping=n_clip == 0, diff_fits_int16=dmax < 32767)
    out = cl / "lick_clean" / "lick_clean_diff.npz"; off = np.r_[0, np.cumsum(re_ - rs)][:-1] if len(rs) else np.zeros(0, np.int64)
    np.savez_compressed(out, session=ses, n_channels=nch, n_samples=n, fs=fs, run_start=rs, run_end=re_, run_offset=off,
                        diff=np.concatenate(md) if md else np.zeros((0, nch), np.int16))
    # rebuild test on a few runs
    D = np.load(out); rb = True
    for i in np.linspace(0, len(rs) - 1, min(len(rs), 20)).astype(int) if len(rs) else []:
        a, b = max(int(rs[i]) - 50, 0), min(int(re_[i]) + 50, n); rb &= bool(np.array_equal(cleaned_window(X, D, a, b), np.asarray(Y[a:b])))
    ok["rebuild_exact"] = rb
    rep = dict(session=ses, ok=all(ok.values()), checks=ok, n_samples=int(n), changed_samples=nchg, changed_pct=round(100 * nchg / max(n, 1), 3),
               runs=int(len(rs)), samples_stored=int((re_ - rs).sum()), changed_outside_windows=outside, sync_changed=n_sync, clipped=n_clip,
               max_abs_diff_bits=dmax, size_mb=round(out.stat().st_size / 1e6, 1), raw_gb=round(apr.stat().st_size / 1e9, 1), seconds=round(time.time() - t0))
    (cl / "lick_clean" / "lick_clean_diff.json").write_text(json.dumps(rep, indent=1)); print(json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
