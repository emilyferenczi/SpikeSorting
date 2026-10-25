"""Lick-contact artifact removal on the RAW SpikeGLX AP file, before spike sorting (EF, 2026-10-24).

Every lick contact puts a ~1 ms electrical transient on all channels (largest on superficial channels). This script writes a
cleaned COPY of a session folder in which that transient is subtracted at every lick; the original files are never modified.

  1. licks       every lick onset from sync_<session>.mat (NIDAQ lick line -> Imec AP sample), or from --licks <file>
  2. template    +-10 ms of every channel around every lick, high-passed (Butterworth order 5, 300 Hz, zero-phase -- as the
                 sorter's input); the session's offset between lick marker and transient is estimated from the data, each
                 lick is aligned within +-3 ms, and the median over all licks is the per-channel template (real spikes
                 do not line up across licks, so the median ignores them); edges tapered to 0
  3. fit         per lick and per electrode block (shank x deep / superficial): --model lowrank (default) fits the k = 3 leading
                 components of all licks' aligned snippets in that block (larger transients have a different shape, not only a
                 larger size); --model template fits the least-squares size of the median template. A lick without a
                 transient gets ~0 either way
  NOT cleaned: licks missing from the sync file's lick line. The LFP band is left untouched on purpose (the slow lick-locked
  wave may be partly neural); the original raw file stays the source for LFP extraction.
  4. subtract    amplitude x template at each aligned lick, from a copy of the .ap.bin (int16, rounded); only > 300 Hz content
                 is subtracted, so the LFP band is unchanged; only +-10 ms around each lick is rewritten
  5. report      lick_clean/lick_clean_report.png + lick_clean_params.json + lick_clean_fit.npz in the output session folder: lick-triggered average before / after
                 per channel and per block, per-lick fitted sizes, alignment shifts
Usage:
  python lick_clean_raw.py <session_dir> <out_parent_dir>            # full session -> <out_parent_dir>/<session>/
  python lick_clean_raw.py <session_dir> <out_parent_dir> --window 740 1040    # test: only that time range (s), written as
                                                                               # a short recording of its own
Runs in the AIND pipeline container (numpy, scipy, matplotlib), see lick_clean.slrm.
"""
import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

FS_DEFAULT = 30000.0
PRE_MS, POST_MS, JIT_MS, CORE_MS, TAPER_MS, MARGIN_MS = 10.0, 10.0, 3.0, 1.5, 3.0, 30.0
SPLIT_UM = 2000.0                                       # electrode blocks: shank x (depth above / below)


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def read_meta(p):
    d = {}
    for line in open(p):
        k, _, v = line.strip().partition("="); d[k.lstrip("~")] = v
    return d


def geometry(M):
    ents = re.findall(r"\(([^()]*)\)", M["snsGeomMap"])[1:]
    g = np.array([[float(v) for v in e.split(":")[:3]] for e in ents])
    return g[:, 0].astype(int), g[:, 2]


def lick_samples(sync_file, n_ap):
    """lick onsets (NIDAQ lick line rising edges) as Imec AP sample indices"""
    try:
        import scipy.io as sio
        m = sio.loadmat(sync_file, variable_names=["lick", "timeNI", "timeImec"], squeeze_me=True)
        lick, tNI, tIm = m["lick"], m["timeNI"], m["timeImec"]
    except NotImplementedError:                                  # MATLAB v7.3
        import h5py
        f = h5py.File(sync_file, "r"); lick, tNI, tIm = (np.asarray(f[k][()]).ravel() for k in ("lick", "timeNI", "timeImec"))
    lick = np.asarray(lick).astype(np.int8).ravel(); tNI = np.asarray(tNI).ravel(); tIm = np.asarray(tIm).ravel()
    on = np.searchsorted(tIm, tNI[np.flatnonzero(np.diff(lick) == 1) + 1])
    return on[(on > 0) & (on < min(len(tIm), n_ap) - 1)]


def snippets(X, centers, pre, post, margin, nch, sos):
    """high-passed float32 snippets (n x (pre+post) x nch) around centers; None where the window leaves the file"""
    out = np.zeros((len(centers), pre + post, nch), np.float32); ok = np.ones(len(centers), bool)
    for i, c in enumerate(centers):
        a, b = c - pre - margin, c + post + margin
        if a < 0 or b > X.shape[0]:
            ok[i] = False; continue
        x = np.asarray(X[a:b, :nch], np.float32)
        out[i] = sosfiltfilt(sos, x, axis=0)[margin:margin + pre + post]
    return out, ok


def estimate(X, trig, fs, nch, blocks):
    pre, post = int(PRE_MS * fs / 1000), int(POST_MS * fs / 1000); J = int(JIT_MS * fs / 1000); W = pre + post
    sos = butter(5, 300, "highpass", fs=fs, output="sos"); margin = int(MARGIN_MS * fs / 1000)
    # session offset between lick marker and transient: peak RMS of the median snippet within -15..+5 ms of the marker
    S0, ok0 = snippets(X, trig[::max(1, len(trig) // 400)], int(20e-3 * fs), int(10e-3 * fs), margin, nch, sos)
    med = np.median(S0[ok0], 0); rms = np.sqrt((med ** 2).mean(1)); t0 = np.arange(len(rms)) - int(20e-3 * fs)
    w = (t0 >= -int(15e-3 * fs)) & (t0 <= int(5e-3 * fs)); offset = int(t0[w][np.argmax(rms[w])])
    log(f"transient at {offset / fs * 1000:+.2f} ms re the lick marker (from {ok0.sum()} licks)")
    cen = trig + offset
    Xs, ok = snippets(X, cen, pre + J, post + J, margin, nch, sos); Xs = Xs[ok]; cen = cen[ok]
    log(f"{len(cen)} licks; snippets {Xs.nbytes / 1e9:.1f} GB")
    shift = np.zeros(len(Xs), int); T = np.median(Xs[:, J:J + W], 0)
    cw = slice(pre - int(CORE_MS * fs / 1000), pre + int(CORE_MS * fs / 1000))
    for _ in range(2):
        Tc = T[cw]; best = np.full(len(Xs), -np.inf)
        for j in range(-J, J + 1):
            sc = np.einsum("ntc,tc->n", Xs[:, J + j + cw.start:J + j + cw.stop], Tc)
            b = sc > best; best[b] = sc[b]; shift[b] = j
        A = np.stack([Xs[i, J + shift[i]:J + shift[i] + W] for i in range(len(Xs))])
        T = np.median(A, 0)
    nt = int(TAPER_MS * fs / 1000); tap = np.ones(W); r = 0.5 - 0.5 * np.cos(np.pi * np.arange(nt) / nt); tap[:nt] = r; tap[-nt:] = r[::-1]
    T = (T * tap[:, None]).astype(np.float32)
    ub = np.unique(blocks); amp = np.zeros((len(A), len(ub)), np.float32)
    for k, b in enumerate(ub):
        Tb = T[cw][:, blocks == b]; den = float((Tb ** 2).sum())
        if den > 0:
            amp[:, k] = np.einsum("ntc,tc->n", A[:, cw][:, :, blocks == b], Tb) / den
    return dict(template=T, starts=cen - pre + shift, amp=amp, blocks=ub, shift=shift, offset=offset, pre=pre, W=W, tap=tap), A


def lowrank(A, blocks, tap, k):
    """per electrode block, the k leading spatio-temporal components of the aligned, tapered lick snippets and each lick's
    coefficients: removal for lick i in block b = coef[b][i] @ basis[b] (W x n_ch_b). Captures that larger transients are not
    just scaled copies of the average (2026-10-25: the 12 largest 0805 licks left 44 uV RMS with one template, 21 uV with k = 3
    per block; injected spikes at the transient kept 97-100% of their amplitude)"""
    basis, coef = {}, {}
    for b in np.unique(blocks):
        m = blocks == b; Xb = (A[:, :, m] * tap[None, :, None]).reshape(len(A), -1).astype(np.float64)
        ev, U = np.linalg.eigh(Xb @ Xb.T); o = np.argsort(ev)[::-1][:k]
        B = (U[:, o] / np.sqrt(np.maximum(ev[o], 1e-12))).T @ Xb                 # k x (W * n_ch_b), orthonormal rows
        basis[int(b)] = B.astype(np.float32); coef[int(b)] = (Xb @ B.T).astype(np.float32)
    return basis, coef


def lick_average(X, starts, W, nch, fs, n=300):
    """high-passed average over (up to) n licks, aligned like the subtraction: W x nch"""
    sos = butter(5, 300, "highpass", fs=fs, output="sos"); sel = starts[np.linspace(0, len(starts) - 1, min(n, len(starts))).astype(int)]
    S, ok = snippets(X, sel + W // 2, W // 2, W - W // 2, int(MARGIN_MS * fs / 1000), nch, sos)
    return S[ok].mean(0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_dir"); ap.add_argument("out_parent")
    ap.add_argument("--window", nargs=2, type=float, default=None, help="test: only this time range (s)")
    ap.add_argument("--model", choices=["template", "lowrank"], default="lowrank",
                    help="template: one median template, scaled per lick and block; lowrank: --k components per block (default)")
    ap.add_argument("--k", type=int, default=3, help="components per electrode block for --model lowrank")
    ap.add_argument("--licks", default=None, help="text file, one lick onset per line as an Imec AP sample index (instead of the "
                    "session folder's sync file -- use when that copy may be outdated, e.g. before the 2026-09 lick-polarity fix)")
    a = ap.parse_args()
    src = Path(a.session_dir).resolve(); ses = src.name; final = Path(a.out_parent).resolve() / ses
    dst = final.with_name(ses + ".partial")                 # written here, renamed when complete (safe if the job is requeued)
    apf = next((src / f"{ses}_imec0").glob("*.ap.bin")); M = read_meta(apf.with_suffix(".meta"))
    fs = float(M.get("imSampRate", FS_DEFAULT)); nsv = int(M["nSavedChans"]); nch = nsv - 1
    uv = float(M["imAiRangeMax"]) / float(M["imMaxInt"]) / float(M.get("imChan0apGain", 100)) * 1e6
    sh, y = geometry(M); blocks = sh * 2 + (y > SPLIT_UM)
    X = np.memmap(apf, dtype=np.int16, mode="r").reshape(-1, nsv); n_ap = X.shape[0]
    if a.licks:
        trig = np.loadtxt(a.licks, dtype=np.int64).ravel(); trig = trig[(trig > 0) & (trig < n_ap - 1)]; lick_src = str(Path(a.licks).resolve())
    else:
        sync = src / f"sync_{ses}.mat"
        if not sync.exists():
            sys.exit(f"sync file not found: {sync} (copy it from the server's 01_recordings/<mouse>/<session>/ folder, or pass --licks)")
        trig = lick_samples(sync, n_ap); lick_src = str(sync)
    log(f"{ses}: {n_ap / fs:.0f} s, {nch} channels, {len(np.unique(blocks))} electrode blocks, {len(trig)} lick onsets")
    if final.exists():
        sys.exit(f"output folder exists, not overwriting: {final}")
    if dst.exists():
        log(f"removing an incomplete earlier attempt: {dst}"); shutil.rmtree(dst)
    (dst / f"{ses}_imec0").mkdir(parents=True)
    # everything except the AP binary is copied unchanged
    for p in src.rglob("*"):
        if p.is_file() and p != apf:
            q = dst / p.relative_to(src); q.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, q)
    out_bin = dst / f"{ses}_imec0" / apf.name
    if a.window:
        s0, s1 = int(a.window[0] * fs), int(a.window[1] * fs)
        log(f"TEST: writing only {a.window[0]:.0f}-{a.window[1]:.0f} s as its own recording")
        Y = np.memmap(out_bin, dtype=np.int16, mode="w+", shape=(s1 - s0, nsv)); Y[:] = X[s0:s1]; Y.flush()
        M2 = open(apf.with_suffix(".meta")).read()
        M2 = re.sub(r"fileSizeBytes=\d+", f"fileSizeBytes={(s1 - s0) * nsv * 2}", M2); M2 = re.sub(r"fileTimeSecs=[\d.]+", f"fileTimeSecs={(s1 - s0) / fs}", M2)
        out_bin.with_suffix(".meta").write_text(M2)
        trig = trig[(trig > s0 + fs) & (trig < s1 - fs)] - s0; Xin = np.memmap(out_bin, dtype=np.int16, mode="r").reshape(-1, nsv)
    else:
        log(f"copying the AP binary ({apf.stat().st_size / 1e9:.1f} GB)"); shutil.copyfile(apf, out_bin)
        shutil.copy2(apf.with_suffix(".meta"), out_bin.with_suffix(".meta")); Xin = X
    est, A = estimate(Xin, trig, fs, nch, blocks)
    bi = np.searchsorted(est["blocks"], blocks); T = est["template"]; W = est["W"]
    if a.model == "lowrank":
        basis, coef = lowrank(A, blocks, est["tap"], a.k); log(f"model: {a.k} components per electrode block")
    else:
        log("model: one template, scaled per lick and block")
    del A
    before = lick_average(Xin, est["starts"], W, nch, fs)
    Y = np.memmap(out_bin, dtype=np.int16, mode="r+").reshape(-1, nsv)
    log("subtracting")
    for i, s in enumerate(est["starts"]):
        if a.model == "lowrank":
            rem = np.zeros((W, nch), np.float32)
            for b, B in basis.items():
                m = blocks == b; rem[:, m] = (coef[b][i] @ B).reshape(W, m.sum())
        else:
            rem = est["amp"][i, bi][None, :] * T
        seg = Y[s:s + W, :nch].astype(np.float32) - rem
        Y[s:s + W, :nch] = np.clip(np.round(seg), -32768, 32767).astype(np.int16)
    Y.flush(); del Y
    Z = np.memmap(out_bin, dtype=np.int16, mode="r").reshape(-1, nsv)
    after = lick_average(Z, est["starts"], W, nch, fs)
    core = slice(est["pre"] - int(CORE_MS * fs / 1000), est["pre"] + int(CORE_MS * fs / 1000))
    summ = dict(session=ses, window_s=a.window, model=a.model, k=a.k if a.model == "lowrank" else None, n_licks=int(len(est["starts"])), offset_ms=est["offset"] / fs * 1000,
                shift_sd_ms=float(est["shift"].std() / fs * 1000), uv_per_bit=uv,
                blocks={int(b): dict(n_ch=int((blocks == b).sum()), amp_median=float(np.median(est["amp"][:, k])),
                                     amp_p10=float(np.percentile(est["amp"][:, k], 10)), amp_p90=float(np.percentile(est["amp"][:, k], 90)),
                                     peak_uV_before=float(np.abs(before[core][:, blocks == b]).max(0).mean() * uv),
                                     peak_uV_after=float(np.abs(after[core][:, blocks == b]).max(0).mean() * uv))
                        for k, b in enumerate(est["blocks"])},
                params=dict(pre_ms=PRE_MS, post_ms=POST_MS, jitter_ms=JIT_MS, core_ms=CORE_MS, taper_ms=TAPER_MS, highpass_hz=300,
                            split_um=SPLIT_UM, source=str(src), licks_from=lick_src), date=time.strftime("%Y-%m-%d %H:%M"))
    rep = dst / "lick_clean"; rep.mkdir(exist_ok=True)                      # report files in their own subfolder
    (rep / "lick_clean_params.json").write_text(json.dumps(summ, indent=1))
    np.savez_compressed(rep / "lick_clean_fit.npz", template=T, starts=est["starts"], amp=est["amp"], blocks=est["blocks"], shift=est["shift"],
                        model=a.model, **({f"basis_{b}": B for b, B in basis.items()} | {f"coef_{b}": c for b, c in coef.items()} if a.model == "lowrank" else {}))
    for b, v in summ["blocks"].items():
        log(f"block {b} ({v['n_ch']} ch): fitted size median {v['amp_median']:.2f} (p10 {v['amp_p10']:.2f}, p90 {v['amp_p90']:.2f}); "
            f"lick average |peak| {v['peak_uV_before']:.1f} -> {v['peak_uV_after']:.1f} uV")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        o = np.lexsort((y, sh)); tms = (np.arange(W) - est["pre"]) / fs * 1000; vm = np.percentile(np.abs(before), 99.5) * uv
        fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
        for j, (nm, L) in enumerate((("before", before), ("after", after))):
            ax[0, j].imshow(L[:, o].T * uv, aspect="auto", cmap="RdBu_r", vmin=-vm, vmax=vm, extent=[tms[0], tms[-1], nch, 0])
            ax[0, j].set_title(f"lick-triggered average, high-passed, {nm} (uV)", fontsize=9); ax[0, j].set_xlabel("ms")
        ax[0, 0].set_ylabel("channel (shank, depth)")
        for k, b in enumerate(est["blocks"]):
            ax[0, 2].plot(tms, before[:, blocks == b].mean(1) * uv, lw=1, color=f"C{k}", label=f"block {b}")
            ax[0, 2].plot(tms, after[:, blocks == b].mean(1) * uv, lw=1, color=f"C{k}", ls=":")
            ax[1, 0].plot(est["amp"][:, k], ".", ms=1.5, color=f"C{k}")
        ax[0, 2].legend(fontsize=7, frameon=False); ax[0, 2].set_title("block means: before (solid) / after (dotted)", fontsize=9)
        ax[1, 0].set_title("fitted size per lick (x template), by block", fontsize=9); ax[1, 0].set_xlabel("lick #"); ax[1, 0].axhline(1, color="k", lw=0.5)
        ax[1, 1].hist(est["shift"] / fs * 1000, 31, color="0.4"); ax[1, 1].set_title("alignment shift per lick (ms)", fontsize=9)
        ax[1, 2].plot(tms, (T * uv)[:, :].mean(1), "k", lw=1); ax[1, 2].set_title("template, mean over channels (uV)", fontsize=9)
        fig.suptitle(f"{ses}: lick-contact transient subtraction ({len(est['starts'])} licks; offset {summ['offset_ms']:+.2f} ms)", fontsize=11)
        fig.tight_layout(); fig.savefig(rep / "lick_clean_report.png", dpi=100)
    except Exception as e:
        log("report figure skipped:", e)
    dst.rename(final); log("done:", final)


if __name__ == "__main__":
    main()
