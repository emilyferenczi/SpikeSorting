"""Hybrid ground-truth copy of one session's raw SpikeGLX AP recording (EF, 2026-11-05; plan: gpe-neuropixels
neuropixels/spikesorting/HYBRID_TEST_PLAN.md). Known spikes are ADDED to a COPY of the raw .ap.bin; the copy then goes through
the same lick cleaning + Kilosort 4 (AIND) pipeline as the real data, and we count what comes back.

Input: a design file (<session>_gt_design.npz, made on the analysis computer by hybridCompareQC.py design): per ground-truth
unit its template in uV (the v2 sort's average template, true time = after the pipeline's phase_shift, rescaled to the
target amplitude, tapered), the channels it is written to (moved channel for channel from the source unit's channels),
and its spike times (AP samples; template sample `nbefore` lands on the spike time).

What is written (int16, same meta, sync channel untouched): copy = raw + round(injection), where per channel c the template is
converted to bits (/ gain_to_uV) and advanced by the channel's ADC sampling offset (inter_sample_shift, Neuropixels
multiplexing): the raw file holds channel c at true time n + shift_c (checked against SpikeInterface's phase_shift), so after
the pipeline's phase_shift the injected spike lines up as the design says. Whole recording in 4-s chunks, one pass.

Outputs, in <out_parent>/<session>/ (a complete session folder: every other file copied unchanged):
  <session>_imec0/<...>.ap.bin / .meta   the hybrid recording
  hybrid/ground_truth.npz                spike samples per unit, raw-space templates (bits), channels, unit table
  hybrid/hybrid_params.json              source, design, counts, code version, verification (copy - raw == round(injection)
                                         exactly on sampled chunks, sync channel identical, samples clipped at the int16 limits)
Usage:
  python make_hybrid.py <raw session dir> <out parent dir> <design.npz> [--verify 40]
The raw session folder is only read. numpy / scipy only (cluster env raw_extract).
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

CHUNK_S = 4.0


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def read_meta(p):
    d = {}
    for line in open(p):
        k, _, v = line.strip().partition("="); d[k.lstrip("~")] = v
    return d


def fshift(x, s):
    """delay each column of x (n x c) by s[c] samples (fractional, FFT; positive = later), zero-padded so nothing wraps"""
    n = x.shape[0]; pad = 64; N = int(2 ** np.ceil(np.log2(n + 2 * pad)))
    y = np.zeros((N, x.shape[1])); y[pad:pad + n] = x
    f = np.fft.rfftfreq(N)
    Y = np.fft.rfft(y, axis=0) * np.exp(-2j * np.pi * f[:, None] * np.asarray(s)[None, :])
    return np.fft.irfft(Y, n=N, axis=0)[pad:pad + n]


def load_design(path):
    """the design as a list of units: dict(chans, tmpl_uv (nt x nc), spikes, ...) + session-level fields"""
    z = np.load(path, allow_pickle=True)
    G = dict(session=str(z["session"]), fs=float(z["fs"]), nbefore=int(z["nbefore"]), n_samples=int(z["n_samples"]),
             gain_uv=np.asarray(z["gain_uv"], float), shift=np.asarray(z["inter_sample_shift"], float),
             table=json.loads(str(z["unit_table"])))
    units = []
    for k, row in enumerate(G["table"]):
        nc = int(row["n_chans"])
        units.append(dict(row, chans=np.asarray(z["chans"][k][:nc], np.int64), tmpl_uv=np.asarray(z["templates_uv"][k][:, :nc], float),
                          spikes=np.asarray(z[f"spikes_{k}"], np.int64)))
    return G, units


def raw_templates(G, units):
    """per unit: the template as written into the raw file, in bits (float), advanced by each channel's ADC offset"""
    for u in units:
        c = u["chans"]
        u["tmpl_bits"] = fshift(u["tmpl_uv"] / G["gain_uv"][c][None, :], -G["shift"][c]).astype(np.float32)
    return units


def injection(units, nbefore, a, b, nch):
    """float32 (b - a) x nch: every ground-truth spike whose template overlaps samples [a, b)"""
    out = np.zeros((b - a, nch), np.float32)
    for u in units:
        T = u["tmpl_bits"]; nt = T.shape[0]; s = u["spikes"]
        i0, i1 = np.searchsorted(s, a - (nt - nbefore) + 1), np.searchsorted(s, b + nbefore)
        for t in s[i0:i1]:
            lo = t - nbefore; x0, x1 = max(lo, a), min(lo + nt, b)
            if x1 > x0:
                out[x0 - a:x1 - a][:, u["chans"]] += T[x0 - lo:x1 - lo]
    return out


def code_version():
    d = str(Path(__file__).resolve().parent)
    try:
        h = subprocess.run(["git", "-C", d, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=20).stdout.strip()
        dirty = subprocess.run(["git", "-C", d, "status", "--porcelain", "--", "."], capture_output=True, text=True, timeout=20).stdout.strip()
        return (h + ("+dirty" if dirty else "")) or None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_dir"); ap.add_argument("out_parent"); ap.add_argument("design")
    ap.add_argument("--verify", type=int, default=40, help="chunks (0.5 s) re-read after writing: copy - raw == round(injection)")
    a = ap.parse_args()
    src = Path(a.session_dir).resolve(); ses = src.name; final = Path(a.out_parent).resolve() / ses; dst = final.with_name(ses + ".partial")
    G, units = load_design(a.design)
    if G["session"] != ses:
        sys.exit(f"design is for {G['session']}, not {ses}")
    apf = next((src / f"{ses}_imec0").glob("*.ap.bin")); M = read_meta(apf.with_suffix(".meta"))
    fs = float(M["imSampRate"]); nsv = int(M["nSavedChans"]); nch = nsv - 1
    if abs(fs - G["fs"]) > 1e-3 or len(G["gain_uv"]) != nch:
        sys.exit(f"design does not match the recording: fs {fs} vs {G['fs']}, channels {nch} vs {len(G['gain_uv'])}")
    X = np.memmap(apf, dtype=np.int16, mode="r").reshape(-1, nsv); n = X.shape[0]
    if final.exists():
        sys.exit(f"output folder exists, not overwriting: {final}")
    if dst.exists():
        log(f"removing an incomplete earlier attempt: {dst}"); shutil.rmtree(dst)
    units = raw_templates(G, units); nb = G["nbefore"]
    nspk = sum(len(u["spikes"]) for u in units)
    log(f"{ses}: {n / fs:.0f} s, {nch} channels; {len(units)} ground-truth units, {nspk} spikes")
    (dst / f"{ses}_imec0").mkdir(parents=True)
    for p in src.rglob("*"):                                   # everything except the AP binary is copied unchanged
        if p.is_file() and p != apf:
            q = dst / p.relative_to(src); q.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, q)
    shutil.copy2(apf.with_suffix(".meta"), dst / f"{ses}_imec0" / apf.with_suffix(".meta").name)
    out_bin = dst / f"{ses}_imec0" / apf.name
    Y = np.memmap(out_bin, dtype=np.int16, mode="w+", shape=X.shape)
    C = int(CHUNK_S * fs); clipped = 0; t0 = time.time()
    for s0 in range(0, n, C):
        s1 = min(s0 + C, n); x = np.asarray(X[s0:s1])
        inj = injection(units, nb, s0, s1, nch)
        y = x.copy(); v = np.rint(x[:, :nch].astype(np.float32) + inj)
        clipped += int(((v > 32767) | (v < -32768)).sum()); y[:, :nch] = np.clip(v, -32768, 32767).astype(np.int16)
        Y[s0:s1] = y
        if (s0 // C) % 150 == 0:
            log(f"  {s1 / fs:.0f} / {n / fs:.0f} s ({(time.time() - t0) / 60:.1f} min)")
    Y.flush(); del Y
    # verification: re-read sampled chunks of both files
    Z = np.memmap(out_bin, dtype=np.int16, mode="r").reshape(-1, nsv); rng = np.random.default_rng(1)
    allspk = np.sort(np.concatenate([u["spikes"] for u in units])); L = int(0.5 * fs)
    starts = np.r_[rng.integers(0, n - L, a.verify // 2), allspk[rng.integers(0, len(allspk), a.verify - a.verify // 2)] - L // 2]
    bad, sync_bad, n_inj_samples = 0, 0, 0
    for s0 in np.clip(starts, 0, n - L):
        s0 = int(s0); x = np.asarray(X[s0:s0 + L]); z = np.asarray(Z[s0:s0 + L])
        want = np.clip(np.rint(x[:, :nch].astype(np.float32) + injection(units, nb, s0, s0 + L, nch)), -32768, 32767).astype(np.int16)
        bad += int((z[:, :nch] != want).sum()); sync_bad += int((z[:, nch] != x[:, nch]).sum()); n_inj_samples += int((z[:, :nch] != x[:, :nch]).sum())
    ver = dict(chunks=int(len(starts)), chunk_s=0.5, samples_not_matching=bad, sync_samples_changed=sync_bad, samples_changed=n_inj_samples,
               clipped_samples=clipped, ok=bool(bad == 0 and sync_bad == 0 and Z.shape == X.shape))
    log("verification:", ver)
    hd = dst / "hybrid"; hd.mkdir(exist_ok=True)
    K = len(units); mx = max(len(u["chans"]) for u in units); nt = units[0]["tmpl_bits"].shape[0]
    TB = np.zeros((K, nt, mx), np.float32); CH = -np.ones((K, mx), np.int64)
    for k, u in enumerate(units):
        TB[k, :, :len(u["chans"])] = u["tmpl_bits"]; CH[k, :len(u["chans"])] = u["chans"]
    np.savez_compressed(hd / "ground_truth.npz", session=ses, fs=fs, nbefore=nb, templates_bits=TB, chans=CH,
                        unit_table=json.dumps([{k: v for k, v in u.items() if k not in ("chans", "tmpl_uv", "tmpl_bits", "spikes")} for u in units]),
                        **{f"spikes_{k}": u["spikes"] for k, u in enumerate(units)})
    shutil.copy2(a.design, hd / Path(a.design).name)
    (hd / "hybrid_params.json").write_text(json.dumps(dict(
        session=ses, source=str(src), design=str(Path(a.design).resolve()), n_units=K, n_spikes=int(nspk), n_samples=int(n), fs=fs,
        rule="copy = raw + round(injection) on the AP channels; sync channel copied; template advanced by inter_sample_shift",
        verification=ver, code_version=code_version(), minutes=(time.time() - t0) / 60, date=time.strftime("%Y-%m-%d %H:%M")), indent=1))
    if not ver["ok"]:
        sys.exit(f"VERIFICATION FAILED -- left as {dst}")
    dst.rename(final); log("done:", final)


if __name__ == "__main__":
    main()
