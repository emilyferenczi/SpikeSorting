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


def prepare(src, work, fit, licks, win, seed=1, large_txt=None, near_licks=False):
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
    k = N_PER_SCALE * len(SCALES)
    if near_licks:   # 2026-10-29: between licks (40-120 ms after a lick marker, >= 30 ms from every real transient and marker), where the
        # lower threshold applies -- it is restricted to <= 150 ms from licks, so lick-free injections cannot test it
        cand = mk + (rng.uniform(0.040, 0.120, len(mk)) * fs).astype(np.int64); busy = np.sort(np.concatenate([mk, cen_real]))
        j = np.searchsorted(busy, cand); d = np.minimum(np.abs(cand - busy[np.clip(j - 1, 0, len(busy) - 1)]), np.abs(busy[np.clip(j, 0, len(busy) - 1)] - cand))
        cand = cand[(d >= int(0.03 * fs)) & (cand > int(0.05 * fs)) & (cand < n - int(0.05 * fs))]; rng.shuffle(cand); it = []
        for t in cand:
            if all(abs(t - o) > int(0.2 * fs) for o in it):
                it.append(int(t))
            if len(it) >= k:
                break
        it = np.sort(np.array(it, np.int64))
    else:
        it = far_times(n, np.concatenate([mk, cen_real]), k, int(0.2 * fs), rng, int(0.05 * fs), int(0.2 * fs))
    if len(it) < k:
        print(f"only {len(it)} lick-free sites for {k} injections (window too busy) -- using them"); k = len(it)
    if large_txt:   # 2026-10-29: real big, differently shaped contact transients of this session (scan: energy >= 100, shape r < 0.6),
        # high-passed +-10 ms snippets on every channel (cosine edges), injected at 0.25 / 0.5 / 1 x their size
        LT = np.loadtxt(large_txt, dtype=np.int64).ravel(); LT = LT[(LT > s1 + int(0.05 * fs)) | (LT < s0 - int(0.05 * fs))]
        sosL = butter(5, 300, "highpass", fs=fs, output="sos"); mgL = int(0.03 * fs); nt = int(0.003 * fs)
        tapL = np.ones(W, np.float32); rL = 0.5 - 0.5 * np.cos(np.pi * np.arange(nt) / nt); tapL[:nt] = rL; tapL[-nt:] = rL[::-1]
        pick = rng.choice(LT, min(40, len(LT)), replace=False)
        SN = np.stack([sosfiltfilt(sosL, np.asarray(X[t - pre - mgL:t - pre + W + mgL, :nch], np.float32), axis=0)[mgL:mgL + W] * tapL[:, None] for t in pick]).astype(np.float32)
        scales = [0.25, 0.5, 1.0]; k = min(k, N_PER_SCALE * len(scales)); it = it[:k]
        sc = np.repeat(scales, N_PER_SCALE)[:k]; rng.shuffle(sc); src_i = rng.integers(0, len(SN), k)
    else:
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
        add(YA, t - pre, t - pre + W, allc, a * (SN[i] if large_txt else amp[i, bi][None, :] * T))
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
    np.savez_compressed(work / "inject.npz", session=ses, fs=fs, window=np.array(win), s0=s0, inj_t=it, inj_scale=sc, inj_src=src_i, inj_kind="large" if large_txt else "template",
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
        # 2026-10-30: per-channel residual after the REAL transients (cleaned copy A; the residual that contaminated real units in
        # 0604 EF401: -0.5..+2.5 ms): per transient, the channels whose high-passed |value| exceeds 5 x their noise in that window
        # (noise = robust SD at the random spike-free control sites); mean over transients, and the 95th-percentile channel's
        # mean |residual| in uV
        rng_r = np.random.default_rng(5); real_t = I["real_transients"][(I["real_transients"] > 2000) & (I["real_transients"] < ZA.shape[0] - 2000)]
        real_t = np.sort(rng_r.choice(real_t, min(300, len(real_t)), replace=False)) if len(real_t) else real_t
        ctrl = I["spike_t"][I["spike_kind"] == "random"] + int(0.02 * fs)
        good = np.asarray(I["good"]); seg = lambda Z_, t: sosfiltfilt(sos, np.asarray(Z_[t - 900:t + 900, :nch], np.float64)[:, good], axis=0)[900 - ms(0.5):900 + ms(2.5)] * uv
        nz = 1.4826 * np.median(np.abs(np.concatenate([seg(ZA, int(t)) for t in ctrl[:60]])), 0) + 1e-6
        over, mabs, edge = [], [], []
        for t in real_t:
            x = np.abs(seg(ZA, int(t))); over.append(((x / nz).max(0) > 5).sum()); mabs.append(x.mean(0))
            edge.append(((x[ms(0.65):ms(1.0)] / nz).max(0) > 5).sum())   # 2026-10-31: +0.15..+0.5 ms, just after the patch's trailing edge
        over_i, mabs_i = [], []                                               # the same at the INJECTED transients that were found
        for t in it[found][:300]:
            if 2000 < t < ZA.shape[0] - 2000:
                x = np.abs(seg(ZA, int(t))); over_i.append(((x / nz).max(0) > 5).sum()); mabs_i.append(x.mean(0))
        res_ch_inj = dict(n=len(over_i), channels_over_5sd_per_transient=float(np.mean(over_i)) if over_i else None,
                          residual_uv_p95_channel=float(np.percentile(np.mean(mabs_i, 0), 95)) if mabs_i else None)
        res_ch = dict(transients=int(len(real_t)), channels_over_5sd_per_transient=float(np.mean(over)) if over else None, injected=res_ch_inj,
                      channels_over_5sd_post_patch=float(np.mean(edge)) if edge else None,
                      residual_uv_p95_channel=float(np.percentile(np.mean(mabs, 0), 95)) if mabs else None,
                      residual_uv_median_channel=float(np.median(np.mean(mabs, 0))) if mabs else None, noise_uv_median=float(np.median(nz)))
        res[name] = dict(injected_found_by_scale=by_scale, injected_found_all=float(found.mean()), cleaned_events=int(len(cA)), residual_after_real=res_ch,
                         cleaned_events_far_from_licks_and_injections=int(len(far_det)), far_per_min=float(len(far_det) / (A.shape[0] / fs / 60)),
                         detections_caused_by_spikes=int(len(new_in_B)), spikes=spike, spikes_near_transients_kept_by_offset=by_off,
                         patched_ms=float(len(cA) * 0.27))
        arrays[f"{name}_found"] = found; arrays[f"{name}_kept"] = kept; arrays[f"{name}_far_det"] = far_det
        print(name, json.dumps(res[name], indent=1))
    jf, nf = work / "inject_results.json", work / "inject_results.npz"      # merged with earlier runs (later runs replace same names)
    prev = json.loads(jf.read_text())["runs"] if jf.exists() else {}; prev.update(res)
    jf.write_text(json.dumps(dict(session=ses, window=I["window"].tolist(), scales=SCALES, runs=prev), indent=1))
    old = dict(np.load(nf, allow_pickle=True)) if nf.exists() else {}; old.update(arrays); np.savez_compressed(nf, **old)


def stats(work, fitdir):
    """the detector's statistics (as detect(): block common modes whitened by the detection noise, matched filter with the
    detection template) over the whole injected window A: at each injected transient (the strongest score within +-0.5 ms) and at
    2000 random lick-free sites away from injections -- score z, shape r, blocks with the shape, energy -> inject_stats.npz"""
    work = Path(work); I = dict(np.load(work / "inject.npz", allow_pickle=True)); ses = str(I["session"]); fs = float(I["fs"])
    fa = next((work / "A" / ses / f"{ses}_imec0").glob("*.ap.bin")); M = LC.read_meta(fa.with_suffix(".meta")); nsv = int(M["nSavedChans"]); nch = nsv - 1
    sh, y = LC.geometry(M); blocks = sh * 2 + (y > LC.SPLIT_UM); ub = np.unique(blocks); A = np.memmap(fa, dtype=np.int16, mode="r").reshape(-1, nsv)
    F = np.load(Path(fitdir) / ses / "lick_clean" / "lick_clean_fit.npz", allow_pickle=True)
    P = json.loads((Path(fitdir) / ses / "lick_clean" / "lick_clean_params.json").read_text()); sig = np.asarray(P["detection"]["noise_uV_bits"])
    U = np.asarray(F["detect_template"], np.float64) / sig; u = U / np.linalg.norm(U); un = np.linalg.norm(u, axis=0) + 1e-9
    sos = butter(5, 300, "highpass", fs=fs, output="sos"); mg = int(LC.MARGIN_MS * fs / 1000); h = int(LC.DET_HALF_MS * fs / 1000); n = A.shape[0]
    Zs, Rs, NB, E = (np.zeros(n, np.float32) for _ in range(4)); C = int(4 * fs)
    for a in range(mg + h, n - mg - h, C):
        b = min(a + C, n - mg - h); c = LC.common_mode(A, a - h, b + h, nch, blocks, ub, sos, mg) / sig
        mb = np.stack([np.correlate(c[:, j], u[:, j], "valid") for j in range(len(ub))], 1); m = mb.sum(1)
        e = np.sqrt(np.convolve((c ** 2).sum(1), np.ones(2 * h + 1), "valid")) + 1e-6
        eb = np.sqrt(np.stack([np.convolve(c[:, j] ** 2, np.ones(2 * h + 1), "valid") for j in range(len(ub))], 1)) + 1e-6
        Zs[a:b] = m / (np.median(np.abs(m)) / 0.6745 + 1e-6); Rs[a:b] = m / e; NB[a:b] = ((mb / (eb * un)) >= LC.LOW_BLOCK_R).sum(1)
        E[a:b] = np.sqrt((c[h:h + (b - a)] ** 2).sum(1))
    rng = np.random.default_rng(3); it = I["inj_t"]
    null = far_times(n, np.concatenate([I["markers"], I["real_transients"], it]), 2000, int(0.1 * fs), rng, int(0.05 * fs), int(0.002 * fs))
    def at(sites):
        out = []
        for t in sites:
            w = slice(max(0, t - 15), min(n, t + 16)); k = w.start + int(np.argmax(Zs[w])); out.append((Zs[k], Rs[k], NB[k], E[w].max()))
        return np.array(out, np.float32)
    # candidate rules over the WHOLE window: events (strongest per 15 ms) in lick-free time (> 100 ms from every marker, real
    # transient and injection) = false detections per minute; at markers without a cleaned transient (their lag window) = recovered
    ref = int(LC.DET_REFRACT_MS * fs / 1000); busy = np.sort(np.concatenate([I["markers"], I["real_transients"], it]))
    cA = np.sort(np.asarray(F["starts"], np.int64) + int(LC.PRE_MS * fs / 1000)); mk = np.sort(I["markers"])
    j = np.searchsorted(cA, mk - int(0.08 * fs)); has = (j < len(cA)) & (cA[np.minimum(j, len(cA) - 1)] <= mk + int(0.02 * fs))
    lag = (cA[np.minimum(j, len(cA) - 1)] - mk)[has]; w0, w1 = int(np.percentile(lag, 5)) - int(0.002 * fs), int(np.percentile(lag, 95)) + int(0.002 * fs)
    miss = mk[~has]; free = np.ones(n, bool)
    for t in busy:
        free[max(0, t - int(0.1 * fs)):t + int(0.1 * fs)] = False
    free_min = free.sum() / fs / 60; rules = [(15, .6, 0), (10, .8, 0), (7, .8, 3), (5, .6, 3), (4, .6, 3), (4, .5, 3), (3.5, .55, 3), (3, .5, 3), (5, .5, 4), (3, .6, 4)]
    rr = []
    for zt, rt, nb in rules:
        ok = (Zs >= zt) & (Rs >= rt) & (NB >= nb); idx = np.flatnonzero(ok); evs = []
        for q in idx[np.argsort(-Zs[idx])]:
            if not evs or np.min(np.abs(np.array(evs) - q)) > ref:
                evs.append(q)
            if len(evs) > 20000:
                break
        evs = np.sort(np.array(evs, np.int64)); fp = int(free[evs].sum()) if len(evs) else 0
        k = np.searchsorted(evs, miss + w0); rec = int(((k < len(evs)) & (evs[np.minimum(k, len(evs) - 1)] <= miss + w1)).sum()) if len(evs) else 0
        rr.append((zt, rt, nb, len(evs), fp, fp / free_min, rec, len(miss)))
        print(f"rule z>={zt} r>={rt} nb>={nb}: {len(evs)} events; in lick-free time {fp} ({fp / free_min:.2f}/min of {free_min:.1f} min); "
              f"at the {len(miss)} markers without a cleaned transient: {rec}", flush=True)
    np.savez_compressed(work / "inject_stats.npz", inj=at(it), inj_scale=I["inj_scale"], null=at(null), sig=sig, rules=np.array(rr, float))
    print("stats written:", work / "inject_stats.npz")


if __name__ == "__main__":
    cmd = sys.argv[1]; args = sys.argv[2:]
    if cmd == "prepare":
        g = lambda k, n=1: args[args.index(k) + 1:args.index(k) + 1 + n]
        prepare(args[0], args[1], g("--fit")[0], g("--licks")[0], [float(v) for v in g("--window", 2)], large_txt=g("--large")[0] if "--large" in args else None,
                near_licks="--near-licks" in args)
    elif cmd == "stats":
        stats(args[0], args[1])
    elif cmd == "evaluate":
        evaluate(args[0], args[1:])
