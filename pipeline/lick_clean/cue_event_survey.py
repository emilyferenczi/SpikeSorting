"""Probe-wide events around the cues, any session (EF, 2026-10-30; Emily: 'is that cue signal present only on those three days?').
The whole-recording scans of the pilot sessions showed probe-wide events ~19 ms after GO cues in EF384 (not transient-shaped, in the
raw file, sorted as spikes). This reads only the windows around each cue from the RAW file, with the scan's own measure
(lick_clean_scan.py): per electrode block the median over its working channels (mean removed), high-passed 300 Hz, divided by its
noise (robust SD over 200 random 20-ms windows of the session); energy = sqrt(sum over blocks of the squared whitened common modes);
events = energy peaks >= 12, >= 0.5 ms apart.
Per cue, -200..+300 ms: every event's latency, energy and size (largest block common mode, uV); the mean energy per cue type.
Input: <raw session dir> <cues .npy: (n, 2) int64 = Imec sample, 1 = Go / 0 = NoGo> <out dir>
Writes <out>/<session>_cue_events.npz + .json (per cue type: trials, trials with an event 10-30 ms after the cue and -40..-20 ms
before it, latency and size of those events).   (numpy / scipy only)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

import lick_clean_raw as LC
import lick_clean_scan as LS

PRE_MS, POST_MS, THR = 200.0, 300.0, 12.0


def cm_window(X, a, b, nch, blocks, ub, good, sos, mg):
    x = np.asarray(X[a - mg:b + mg, :nch], np.float32); x -= x.mean(0)
    cm = np.stack([np.median(x[:, (blocks == u) & good], 1) for u in ub], 1)
    return sosfiltfilt(sos, cm, axis=0)[mg:mg + (b - a)]


def main():
    src, cf, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); ses = src.name; t0 = time.time()
    apf = next((src / f"{ses}_imec0").glob("*.ap.bin")); M = LC.read_meta(apf.with_suffix(".meta"))
    fs = float(M.get("imSampRate", LC.FS_DEFAULT)); nsv = int(M["nSavedChans"]); nch = nsv - 1
    uv = float(M["imAiRangeMax"]) / float(M["imMaxInt"]) / float(M.get("imChan0apGain", 100)) * 1e6
    sh, y = LC.geometry(M); blocks = sh * 2 + (y > LC.SPLIT_UM); ub = np.unique(blocks)
    X = np.memmap(apf, dtype=np.int16, mode="r").reshape(-1, nsv); n = X.shape[0]; good = LS.working_channels(X, nch, fs)
    sos = butter(5, 300, "highpass", fs=fs, output="sos"); mg = int(0.03 * fs); rng = np.random.default_rng(0); w20 = int(0.02 * fs)
    noise = np.concatenate([cm_window(X, a, a + w20, nch, blocks, ub, good, sos, mg) for a in rng.integers(mg + fs, n - w20 - mg - fs, 200)])
    sig = 1.4826 * np.median(np.abs(noise), 0) + 1e-6
    C = np.load(cf); cue, go = C[:, 0].astype(np.int64), C[:, 1].astype(int); p0, p1 = int(PRE_MS * fs / 1000), int(POST_MS * fs / 1000)
    keep = (cue - p0 - mg > 0) & (cue + p1 + mg < n); cue, go = cue[keep], go[keep]
    lat, en, amp, tri = [], [], [], []; E = np.zeros((2, p0 + p1), np.float64); nE = np.zeros(2, int)
    for i, c in enumerate(cue):
        cm = cm_window(X, c - p0, c + p1, nch, blocks, ub, good, sos, mg); e = np.sqrt(((cm / sig) ** 2).sum(1))
        E[go[i]] += e; nE[go[i]] += 1
        pk, _ = find_peaks(e, height=THR, distance=int(0.0005 * fs))
        lat += list((pk - p0) / fs * 1000); en += list(e[pk]); amp += list(np.abs(cm[pk]).max(1) * uv); tri += [i] * len(pk)
    lat, en, amp, tri = (np.asarray(v) for v in (lat, en, amp, tri))
    rep = dict(session=ses, cues=int(len(cue)), working_channels=int(good.sum()), blocks=int(len(ub)), threshold=THR, seconds=0)
    for lab, g in (("go", 1), ("nogo", 0)):
        m = go[tri] == g if len(tri) else np.zeros(0, bool); after = m & (lat >= 10) & (lat <= 30); before = m & (lat >= -40) & (lat <= -20)
        rep[lab] = dict(trials=int((go == g).sum()), trials_event_10_30ms=int(len(np.unique(tri[after]))), trials_event_minus40_20ms=int(len(np.unique(tri[before]))),
                        latency_ms_median=float(np.median(lat[after])) if after.any() else None, latency_ms_iqr=[float(v) for v in np.percentile(lat[after], [25, 75])] if after.any() else None,
                        amp_uv_median=float(np.median(amp[after])) if after.any() else None)
    rep["seconds"] = round(time.time() - t0); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{ses}_cue_events.npz", session=ses, fs=fs, cue=cue, go=go, latency_ms=lat, energy=en, amp_uv=amp, trial=tri,
                        mean_energy_nogo=(E[0] / max(nE[0], 1)).astype(np.float32), mean_energy_go=(E[1] / max(nE[1], 1)).astype(np.float32), pre_ms=PRE_MS)
    (out / f"{ses}_cue_events.json").write_text(json.dumps(rep, indent=1)); print(json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
