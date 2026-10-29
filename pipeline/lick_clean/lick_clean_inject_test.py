"""Injection test for the lick cleaning (EF, 2026-10-29; Emily: 'how do we know if we miss them? ... make sure we don't start
removing real spikes'). On a window of one session's raw AP file:
  prepare   two injected copies of the window, each a session folder lick_clean_raw.py can clean:
              A = raw + synthetic lick transients at lick-free times (> 200 ms from every lick marker and cleaned transient): the
                  session's template (lick_clean_fit.npz of an earlier cleaning) times a real transient's per-block size times a
                  scale 0.05-1.0 (1 = a typical transient)
              B = A + synthetic spikes (80 uV trough, Gaussian footprint sigma 25 um on one shank, 1/3 at random lick-free times,
                  1/3 at -2..+3 ms of a real transient, 1/3 at -2..+3 ms of an injected one)
            + inject.npz (all injected times, sizes, spikes, their high-passed troughs) and the window's lick markers
  (clean)   lick_clean_raw.py on A and on B with each option set (lick_clean_inject_test.slrm / any sbatch), --no-scan
  evaluate  per option set: injected transients found (+-1 ms) by scale; detections far from licks and injections (> 200 ms);
            detections the spikes cause (in B, not in A); per spike the % of its trough left after cleaning (B - A, high-passed,
            peak channel) by place and offset; common-mode energy left at found / missed injections
            -> <work>/inject_results.json + inject_results.npz
Run:  python lick_clean_inject_test.py prepare  <raw session dir> <work dir> --fit <lick_clean_fit.npz> --licks <onsets> --window T0 T1
      python lick_clean_inject_test.py evaluate <work dir> <name>=<cleaned parent of A>,<cleaned parent of B> [...]
numpy / scipy only.
"""
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

import lick_clean_raw as LC
import lick_clean_scan as LS

SCALES = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
N_PER_SCALE, N_SPIKES, SPIKE_UV, SIGMA_UM = 25, 300, 80.0, 25.0


def spike_wave(fs):
    t = np.arange(-30, 31) / fs * 1000; w = -np.exp(-(t / 0.12) ** 2) + 0.35 * np.exp(-((t - 0.45) / 0.3) ** 2); return w / -w.min()


def far_times(n, avoid, k, gap, rng, margin, min_sep):
    out = []; avoid = np.sort(avoid); tries = 0
    while len(out) < k and tries < 200000:
        t = int(rng.integers(margin, n - margin)); tries += 1; j = np.searchsorted(avoid, t)
        d = min(abs(t - avoid[j - 1]) if j > 0 else 10 ** 12, abs(avoid[j] - t) if j < len(avoid) else 10 ** 12)
        if d > gap and all(abs(t - o) > min_sep for o in out):
            out.append(t)
    return np.sort(np.array(out, np.int64))


def prepare(src, work, fit, licks, win, seed=1):
    rng = np.random.default_rng(seed); src = Path(src); ses = src.name; work = Path(work); work.mkdir(parents=True, exist_ok=True)
    apf = next((src / f"{ses}_imec0").glob("*.ap.bin")); M = LC.read_meta(apf.with_suffix(".meta")); M_text = apf.with_suffix(".meta").read_text()
    fs = float(M.get("imSampRate", LC.FS_DEFAULT)); nsv = int(M["nSavedChans"]); nch = nsv - 1
    uv = float(M["imAiRangeMax"]) / float(M["imMaxInt"]) / float(M.get("imChan0apGain", 100)) * 1e6
    sh, y = LC.geometry(M); blocks = sh * 2 + (y > LC.SPLIT_UM); ub = np.unique(blocks)
    xs = np.array([float(e.split(":")[1]) for e in re.findall(r"\(([^()]*)\)", M["snsGeomMap"])[1:]])
    X = np.memmap(apf, dtype=np.int16, mode="r").reshape(-1, nsv); s0, s1 = int(win[0] * fs), int(win[1] * fs); n = s1 - s0
    F = np.load(fit, allow_pickle=True); T = np.asarray(F["template"], np.float32); amp = np.asarray(F["amp"], np.float32); pre = int(LC.PRE_MS * fs / 1000); W = T.shape[0]
    bi = np.searchsorted(np.asarray(F["blocks"]), blocks)
    cen_real = np.asarray(F["starts"], np.int64) + pre; cen_real = cen_real[(cen_real > s0) & (cen_real < s1)] - s0
    mk = np.loadtxt(licks, dtype=np.int64).ravel(); mk = mk[(mk > s0) & (mk < s1)] - s0
    good = LS.working_channels(X[s0:s1], nch, fs)
    # transients at lick-free times
    k = N_PER_SCALE * len(SCALES); it = far_times(n, np.concatenate([mk, cen_real]), k, int(0.2 * fs), rng, int(0.05 * fs), int(0.2 * fs))
    if len(it) < k:
        print(f"only {len(it)} lick-free sites for {k} injections (window too busy) -- using them"); k = len(it)
    sc = np.repeat(SCALES, N_PER_SCALE)[:k]; rng.shuffle(sc); src_i = rng.integers(0, len(amp), k)
    # copies written to disk and changed in place (a 5-min window is ~7 GB per copy)
    def make(nm):
        d = work / nm / ses / f"{ses}_imec0"; d.mkdir(parents=True, exist_ok=True); f = d / apf.name
        Y = np.memmap(f, dtype=np.int16, mode="w+", shape=(n, nsv)); C = int(10 * fs)
        for a in range(0, n, C):
            Y[a:min(a + C, n)] = X[s0 + a:s0 + min(a + C, n)]
        M2 = re.sub(r"fileSizeBytes=\d+", f"fileSizeBytes={n * nsv * 2}", M_text); M2 = re.sub(r"fileTimeSecs=[\d.]+", f"fileTimeSecs={n / fs}", M2)
        f.with_suffix(".meta").write_text(M2); return Y

    def add(Y, a0, a1, cols, v):
        seg = Y[a0:a1, :nch].astype(np.float32); seg[:, cols] += v; Y[a0:a1, :nch] = np.clip(np.round(seg), -32768, 32767).astype(np.int16)
    YA = make("A"); allc = np.arange(nch)
    for t, a, i in zip(it, sc, src_i):
        add(YA, t - pre, t - pre + W, allc, a * amp[i, bi][None, :] * T)
    YA.flush()
    # spikes: random lick-free / near real transients / near injected ones (scale >= 0.2)
    wv = spike_wave(fs); third = N_SPIKES // 3
    t_rand = far_times(n, np.concatenate([mk, cen_real, it]), third, int(0.05 * fs), rng, int(0.05 * fs), int(0.01 * fs))
    off_ms = rng.uniform(-2, 3, 2 * third); off = (off_ms * fs / 1000).astype(int)
    big = it[sc >= 0.2]; near_real = rng.choice(cen_real, third) + off[:third] if len(cen_real) else np.array([], np.int64)
    near_inj = rng.choice(big, third) + off[third:] if len(big) else np.array([], np.int64)
    st = np.concatenate([t_rand, near_real, near_inj]).astype(np.int64)
    kind = np.array(["random"] * len(t_rand) + ["near_real_transient"] * len(near_real) + ["near_injected_transient"] * len(near_inj))
    soff = np.concatenate([np.full(len(t_rand), np.nan), off_ms[:len(near_real)], off_ms[third:third + len(near_inj)]])
    gch = np.flatnonzero(good); ch = rng.choice(gch, len(st))
    del YA; fa = next((work / "A" / ses / f"{ses}_imec0").glob("*.ap.bin")); YB = make("B"); del YB
    fb = next((work / "B" / ses / f"{ses}_imec0").glob("*.ap.bin")); shutil.copyfile(fa, fb)
    YB = np.memmap(fb, dtype=np.int16, mode="r+").reshape(-1, nsv)
    for t, c in zip(st, ch):
        dist = np.hypot(xs - xs[c], y - y[c]); g = np.exp(-(dist / SIGMA_UM) ** 2) * (sh == sh[c]) * good; g[g < 0.05] = 0
        cols = np.flatnonzero(g > 0); add(YB, t - 30, t + 31, cols, (SPIKE_UV / uv) * wv[:, None] * g[None, cols])
    YB.flush(); del YB
    sos = butter(5, 300, "highpass", fs=fs, output="sos"); sp_alone = np.zeros(121); sp_alone[30:91] = wv * SPIKE_UV
    trough = float(sosfiltfilt(sos, sp_alone)[50:71].min())                                      # high-passed trough of the spike alone (uV)
    np.savetxt(work / f"{ses}_lick_onsets_window.txt", mk, fmt="%d")
    np.savez_compressed(work / "inject.npz", session=ses, fs=fs, window=np.array(win), s0=s0, inj_t=it, inj_scale=sc, inj_src=src_i,
                        markers=mk, real_transients=cen_real, spike_t=st, spike_ch=ch, spike_kind=kind, spike_off_ms=soff, spike_trough_uv=trough,
                        uv=uv, good=good)
    print(f"{ses} {win[0]:.0f}-{win[1]:.0f} s: {k} injected transients ({N_PER_SCALE} per scale), {len(st)} spikes; A and B in {work}; "
          f"window lick file {work / (ses + '_lick_onsets_window.txt')} ({len(mk)} markers)")


def evaluate(work, runs):
    work = Path(work); I = dict(np.load(work / "inject.npz", allow_pickle=True)); ses = str(I["session"]); fs = float(I["fs"]); uv = float(I["uv"])
    it, sc, mk, real = I["inj_t"], I["inj_scale"], I["markers"], I["real_transients"]
    A = np.memmap(next((work / "A" / ses / f"{ses}_imec0").glob("*.ap.bin")), dtype=np.int16, mode="r")
    nsv = int(LC.read_meta(next((work / "A" / ses / f"{ses}_imec0").glob("*.ap.meta")))["nSavedChans"]); nch = nsv - 1
    A = A.reshape(-1, nsv); sos = butter(5, 300, "highpass", fs=fs, output="sos"); res, arrays = {}, {}
    ms = lambda v: int(round(v * fs / 1000)); pre = int(LC.PRE_MS * fs / 1000)
    for spec in runs:
        name, dirs = spec.split("=", 1); dA, dB = (Path(d) / ses for d in dirs.split(","))
        FA = np.load(dA / "lick_clean" / "lick_clean_fit.npz", allow_pickle=True); FB = np.load(dB / "lick_clean" / "lick_clean_fit.npz", allow_pickle=True)
        cA = np.sort(np.asarray(FA["starts"], np.int64) + pre) if "starts" in FA else np.array([], np.int64)
        cB = np.sort(np.asarray(FB["starts"], np.int64) + pre) if "starts" in FB else np.array([], np.int64)
        def near(x, ref, tol):
            if not len(ref):
                return np.zeros(len(x), bool)
            j = np.searchsorted(ref, x); return np.minimum(np.abs(x - ref[np.clip(j - 1, 0, len(ref) - 1)]), np.abs(ref[np.clip(j, 0, len(ref) - 1)] - x)) <= tol
        found = near(it, cA, ms(1))
        by_scale = {f"{s:g}": float(found[sc == s].mean()) for s in np.unique(sc)}
        others = cA[~near(cA, it, ms(1))]                                       # cleaned events that are not injections
        lick_ok = near(others, np.sort(np.concatenate([mk, real])), ms(200))
        far_det = others[~lick_ok]
        new_in_B = cB[~near(cB, cA, ms(1))]                                     # detections the spikes caused
        # spikes: B - A after cleaning, high-passed, at the spike's channel
        ZA = np.memmap(next((dA / f"{ses}_imec0").glob("*.ap.bin")), dtype=np.int16, mode="r").reshape(-1, nsv)
        ZB = np.memmap(next((dB / f"{ses}_imec0").glob("*.ap.bin")), dtype=np.int16, mode="r").reshape(-1, nsv)
        kept = np.full(len(I["spike_t"]), np.nan)
        for q, (t, c) in enumerate(zip(I["spike_t"], I["spike_ch"])):
            a0, a1 = t - 900, t + 900
            if a0 < 0 or a1 > ZA.shape[0]:
                continue
            d = sosfiltfilt(sos, (np.asarray(ZB[a0:a1, c], np.float64) - np.asarray(ZA[a0:a1, c], np.float64)) * uv)
            kept[q] = 100 * d[900 - ms(0.2):900 + ms(0.2) + 1].min() / float(I["spike_trough_uv"])
        kinds = I["spike_kind"]; off = I["spike_off_ms"]
        spike = {k: dict(n=int((kinds == k).sum()), median_kept_pct=float(np.nanmedian(kept[kinds == k])),
                         pct_below_80=float(100 * np.nanmean(kept[kinds == k] < 80))) for k in np.unique(kinds)}
        near_any = kinds != "random"; bins = [-2, -0.5, -0.2, -0.1, 0.1, 0.2, 0.5, 1, 3]
        by_off = {f"{lo:+g}..{hi:+g} ms": float(np.nanmedian(kept[near_any & (off >= lo) & (off < hi)])) if (near_any & (off >= lo) & (off < hi)).any() else None
                  for lo, hi in zip(bins[:-1], bins[1:])}
        res[name] = dict(injected_found_by_scale=by_scale, injected_found_all=float(found.mean()), cleaned_events=int(len(cA)),
                         cleaned_events_far_from_licks_and_injections=int(len(far_det)), far_per_min=float(len(far_det) / (A.shape[0] / fs / 60)),
                         detections_caused_by_spikes=int(len(new_in_B)), spikes=spike, spikes_near_transients_kept_by_offset=by_off,
                         patched_ms=float(len(cA) * 0.27))
        arrays[f"{name}_found"] = found; arrays[f"{name}_kept"] = kept; arrays[f"{name}_far_det"] = far_det
        print(name, json.dumps(res[name], indent=1))
    (work / "inject_results.json").write_text(json.dumps(dict(session=ses, window=I["window"].tolist(), scales=SCALES, runs=res), indent=1))
    np.savez_compressed(work / "inject_results.npz", **arrays)


if __name__ == "__main__":
    cmd = sys.argv[1]; args = sys.argv[2:]
    if cmd == "prepare":
        g = lambda k, n=1: args[args.index(k) + 1:args.index(k) + 1 + n]
        prepare(args[0], args[1], g("--fit")[0], g("--licks")[0], [float(v) for v in g("--window", 2)])
    elif cmd == "evaluate":
        evaluate(args[0], args[1:])
