"""Residual-artifact examples for single units (EF, 2026-10-30; Emily: 'can you show me examples of the issue you are worried about
for the residual artifact?'). For each unit of a sort: its spikes just after a cleaned lick contact (-0.5..+2 ms) vs far from
every contact (> 50 ms): mean high-passed waveform on the 8 channels nearest its peak channel, in the CLEANED and the RAW file;
and example contacts (+-3 ms) on the peak channel, raw and cleaned, with the unit's spikes. numpy / scipy only.
Run: python residual_unit_examples.py <raw session dir> <cleaned session dir> <sort AIND_<session> dir> <out npz> <unit> [<unit> ...]
"""
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

import lick_clean_raw as LC


def main():
    raw, cl, sd, out = (Path(a) for a in sys.argv[1:5]); units = [int(u) for u in sys.argv[5:]]; ses = raw.name
    apf = next((raw / f"{ses}_imec0").glob("*.ap.bin")); M = LC.read_meta(apf.with_suffix(".meta")); nsv = int(M["nSavedChans"]); nch = nsv - 1
    fs = float(M.get("imSampRate", LC.FS_DEFAULT)); uv = float(M["imAiRangeMax"]) / float(M["imMaxInt"]) / float(M.get("imChan0apGain", 100)) * 1e6
    sh, y = LC.geometry(M); import re
    xs = np.array([float(e.split(":")[1]) for e in re.findall(r"\(([^()]*)\)", M["snsGeomMap"])[1:]])
    X = np.memmap(apf, dtype=np.int16, mode="r").reshape(-1, nsv); Z = np.memmap(next((cl / f"{ses}_imec0").glob("*.ap.bin")), dtype=np.int16, mode="r").reshape(-1, nsv)
    F = np.load(cl / "lick_clean" / "lick_clean_fit.npz", allow_pickle=True); ev = np.sort(np.asarray(F["starts"], np.int64) + int(LC.PRE_MS * fs / 1000))
    large = np.sort(np.asarray(F["large"], np.int64)) if "large" in F else np.array([], np.int64)
    st = np.load(sd / "spike_times.npy").astype(np.int64).ravel(); cl_ = np.load(sd / "spike_clusters.npy").ravel()
    sos = butter(5, 300, "highpass", fs=fs, output="sos"); mg = int(0.02 * fs); h0, h1 = int(0.001 * fs), int(0.002 * fs); rng = np.random.default_rng(0)
    res = dict(session=ses, fs=fs, units=np.array(units))

    def snip(A, t, chans, a, b):
        x = sosfiltfilt(sos, np.asarray(A[t - a - mg:t + b + mg, :nch], np.float64)[:, chans], axis=0); return x[mg:mg + a + b] * uv
    for u in units:
        t = np.sort(st[cl_ == u])
        far0 = t[np.minimum(np.abs(t[:, None] - ev[np.clip(np.searchsorted(ev, t)[:, None] + np.array([-1, 0]), 0, len(ev) - 1)]).min(1), 10 ** 12) > int(0.05 * fs)] if len(ev) else t
        pick = rng.choice(far0[(far0 > fs) & (far0 < X.shape[0] - fs)], min(100, len(far0)), replace=False)
        wav = np.mean([snip(Z, int(x), np.arange(nch), h0, h1) for x in pick], 0); pc = int(np.argmax(np.ptp(wav, 0)))   # peak channel from the data
        same = np.flatnonzero(sh == sh[pc]); chans = same[np.argsort(np.hypot(xs[same] - xs[pc], y[same] - y[pc]))[:8]]; chans = chans[np.argsort(y[chans])]
        j = np.searchsorted(ev, t); prv = np.where(j > 0, t - ev[np.maximum(j - 1, 0)], 10 ** 12); nxt = np.where(j < len(ev), ev[np.minimum(j, len(ev) - 1)] - t, 10 ** 12)
        near = (prv <= int(0.002 * fs)) | (nxt <= int(0.0005 * fs)); far = np.minimum(prv, nxt) > int(0.05 * fs)
        tn = t[near]; tf = t[far]; tn = tn[(tn > fs) & (tn < X.shape[0] - fs)]; tf = tf[(tf > fs) & (tf < X.shape[0] - fs)]
        sn = rng.choice(tn, min(200, len(tn)), replace=False) if len(tn) else tn; sf = rng.choice(tf, min(200, len(tf)), replace=False) if len(tf) else tf
        k = f"u{u}"
        for nm, A in (("clean", Z), ("raw", X)):
            res[f"{k}_near_{nm}"] = np.mean([snip(A, int(x), chans, h0, h1) for x in sn], 0) if len(sn) else np.zeros((h0 + h1, len(chans)))
            res[f"{k}_far_{nm}"] = np.mean([snip(A, int(x), chans, h0, h1) for x in sf], 0) if len(sf) else np.zeros((h0 + h1, len(chans)))
        # example contacts with a spike of this unit -0.5..+2 ms after: +-3 ms on the peak channel, raw and cleaned
        ce = ev[np.isin(ev, (tn[:, None] - np.arange(-int(0.0005 * fs), int(0.002 * fs) + 1)[None, :]).ravel())] if len(tn) else ev[:0]
        ce = rng.choice(ce, min(8, len(ce)), replace=False) if len(ce) else ce; w = int(0.003 * fs)
        res[f"{k}_ex_t"] = ce; res[f"{k}_ex_large"] = np.isin(ce, large)
        res[f"{k}_ex_raw"] = np.array([snip(X, int(c), [pc], w, w)[:, 0] for c in ce]) if len(ce) else np.zeros((0, 2 * w))
        res[f"{k}_ex_clean"] = np.array([snip(Z, int(c), [pc], w, w)[:, 0] for c in ce]) if len(ce) else np.zeros((0, 2 * w))
        res[f"{k}_ex_spk"] = np.array([np.concatenate([t[(t >= c - w) & (t < c + w)] - c, np.full(10, 10 ** 6)])[:10] for c in ce]) if len(ce) else np.zeros((0, 10))
        res[f"{k}_info"] = np.array([len(t), len(tn), len(tf), pc, int(np.isin(t[near], large[:, None] + np.arange(-15, 61)).sum()) if len(large) else 0])
        res[f"{k}_chans"] = chans; res[f"{k}_depth"] = y[chans]
        print(f"{ses} unit {u}: {len(t)} spikes, {len(tn)} within -0.5..+2 ms of a contact, peak channel {pc}", flush=True)
    np.savez_compressed(out, **res); print("->", out)


if __name__ == "__main__":
    main()
