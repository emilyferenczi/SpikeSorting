"""Raw traces on every channel around given events (EF, 2026-10-30): what does a probe-wide event look like across the probe --
the same deflection on every channel (electrical: lick contact) or a spatial pattern of spikes (neural)? High-passed 300 Hz,
+-10 ms, all channels, in uV, for up to N events (strongest first by block common-mode energy is not needed: evenly spread).
Run: python event_traces.py <raw session dir> <events .npy (Imec samples)> <out .npz> [N]      (numpy / scipy only)
"""
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

import lick_clean_raw as LC


def main():
    src, ef, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); N = int(sys.argv[4]) if len(sys.argv) > 4 else 12; ses = src.name
    apf = next((src / f"{ses}_imec0").glob("*.ap.bin")); M = LC.read_meta(apf.with_suffix(".meta"))
    fs = float(M.get("imSampRate", LC.FS_DEFAULT)); nsv = int(M["nSavedChans"]); nch = nsv - 1
    uv = float(M["imAiRangeMax"]) / float(M["imMaxInt"]) / float(M.get("imChan0apGain", 100)) * 1e6
    sh, y = LC.geometry(M); X = np.memmap(apf, dtype=np.int16, mode="r").reshape(-1, nsv)
    ev = np.sort(np.load(ef).astype(np.int64)); ev = ev[np.linspace(0, len(ev) - 1, min(N, len(ev))).astype(int)] if len(ev) else ev
    sos = butter(5, 300, "highpass", fs=fs, output="sos"); h, mg = int(0.010 * fs), int(0.03 * fs)
    tr = np.stack([sosfiltfilt(sos, np.asarray(X[t - h - mg:t + h + mg, :nch], np.float32), axis=0)[mg:mg + 2 * h] * uv for t in ev]).astype(np.float16)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, session=ses, fs=fs, events=ev, traces=tr, shank=sh, depth_um=y); print(ses, tr.shape, "->", out)


if __name__ == "__main__":
    main()
