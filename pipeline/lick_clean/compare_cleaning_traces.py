"""The same moments in the raw file and in several cleaned copies of a session (EF, 2026-10-30: 'show me plots / traces from the
large model'). Every channel, high-passed 300 Hz, +-20 ms around each event, in uV, for up to N events spread over the session.
Run: python compare_cleaning_traces.py <events .npy (Imec samples)> <out .npz> <N> <label>=<session dir> [<label>=<session dir> ...]
(numpy / scipy only)
"""
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

import lick_clean_raw as LC


def main():
    ef, out, N = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]); srcs = [a.split("=", 1) for a in sys.argv[4:]]
    ses = Path(srcs[0][1]).name; ap = {}
    for lab, d in srcs:
        f = next((Path(d) / f"{ses}_imec0").glob("*.ap.bin")); M = LC.read_meta(f.with_suffix(".meta"))
        nsv = int(M["nSavedChans"]); ap[lab] = np.memmap(f, dtype=np.int16, mode="r").reshape(-1, nsv)
    fs = float(M.get("imSampRate", LC.FS_DEFAULT)); nch = nsv - 1
    uv = float(M["imAiRangeMax"]) / float(M["imMaxInt"]) / float(M.get("imChan0apGain", 100)) * 1e6
    sh, y = LC.geometry(M); blocks = sh * 2 + (y > LC.SPLIT_UM)
    sos = butter(5, 300, "highpass", fs=fs, output="sos"); h, mg = int(0.020 * fs), int(0.03 * fs)
    n = min(len(X) for X in ap.values()); ev = np.sort(np.load(ef).astype(np.int64)); ev = ev[(ev > h + mg) & (ev < n - h - mg)]   # whole window inside the file
    ev = ev[np.linspace(0, len(ev) - 1, min(N, len(ev))).astype(int)]
    res = {}
    for lab, X in ap.items():
        res[f"tr_{lab}"] = np.stack([sosfiltfilt(sos, np.asarray(X[t - h - mg:t + h + mg, :nch], np.float32), axis=0)[mg:mg + 2 * h] * uv
                                     for t in ev]).astype(np.float16)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, session=ses, fs=fs, events=ev, labels=np.array([l for l, _ in srcs]), shank=sh, depth_um=y, blocks=blocks, **res)
    print(out, len(ev), "events", list(res))


if __name__ == "__main__":
    main()
