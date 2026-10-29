"""Whole-recording scan for lick-contact transients, raw vs cleaned (EF, 2026-10-29; Emily: 'check the entire recordings to
know how many transients are missed / remain'). Every sample of both files, not a sample of licks:
  1. each electrode block's common mode: per 10-s chunk the median over the block's working channels (de-meaned), then
     high-passed 300 Hz (median first, filter after: same events as filtering every channel first at energy >= 12, 4x faster);
     whitened by the raw file's robust SD per block (the same scale for both files)
  2. events: energy = sqrt(sum over blocks of whitened common mode^2) >= 10, the largest within +-0.5 ms; shape r = correlation
     with the session's detection template (best of +-5 samples); reported at energy >= 12 and >= 20, all / with r >= 0.6
  3. each event by its place: at_transient (within +-0.2 ms of a cleaned transient), after_transient (0.2-5 ms after: second
     contacts, tails), before_transient (0.2-5 ms before), missed_at_lick (in the lag window of a lick marker without a cleaned
     transient), near_licks_other (within 100 ms of a marker or transient), away_from_licks (the baseline: rate per minute of
     lick-free time)
Writes lick_clean/lick_clean_scan.npz (events of both files, 10-s timelines per class, the energy envelope per 100 ms,
example snippets per class: block common modes and one channel per block, raw and cleaned) + lick_clean_scan.json (counts).
+ lick_clean_scan_examples.npz: per class the 12 strongest and 12 random events, both files (examples()).
Run alone: python lick_clean_scan.py <raw session dir> <cleaned session dir> [--examples: only the examples, from an existing scan]
(numpy / scipy only)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

import lick_clean_raw as LC

CAND, THR, SHAPE_R = 10.0, (12.0, 20.0), 0.6
CHUNK_S, NEAR_MS, CORE_MS, AFTER_MS = 10.0, 100.0, 0.2, 5.0
CLASSES = ["at_transient", "after_transient", "before_transient", "missed_at_lick", "near_licks_other", "away_from_licks"]
N_EX, EX_MS = 16, 5.0


def working_channels(X, nch, fs, n_win=60, seed=0):
    """channels that carry signal: robust SD of the high-passed trace in random 20-ms windows >= 0.5 bits (reference / switched-off
    channels are constant)"""
    rng = np.random.default_rng(seed); sos = butter(5, 300, "highpass", fs=fs, output="sos"); w = int(0.02 * fs); mg = int(0.03 * fs)
    t = rng.integers(mg, X.shape[0] - w - mg, n_win)
    h = np.concatenate([sosfiltfilt(sos, np.asarray(X[a - mg:a + w + mg, :nch], np.float32), axis=0)[mg:mg + w] for a in t])
    return 1.4826 * np.median(np.abs(h), 0) >= 0.5


def block_cm(X, nch, blocks, ub, good, fs):
    sos = butter(5, 300, "highpass", fs=fs, output="sos"); n = X.shape[0]; C = int(CHUNK_S * fs); mg = int(0.03 * fs)
    out = np.zeros((n, len(ub)), np.float32)
    for a in range(0, n, C):
        b = min(a + C, n); a0, b0 = max(0, a - mg), min(n, b + mg)
        x = np.asarray(X[a0:b0, :nch], np.float32); x -= x.mean(0)
        cm = np.stack([np.median(x[:, (blocks == u) & good], 1) for u in ub], 1)
        out[a:b] = sosfiltfilt(sos, cm, axis=0)[a - a0:a - a0 + (b - a)]
    return out


def shape_r(cm, sig, pk, U):
    if U is None:
        return np.full(len(pk), np.nan, np.float32)
    h = U.shape[0] // 2; u = U / sig; u = u / np.linalg.norm(u); r = np.full(len(pk), np.nan, np.float32)
    for i, k in enumerate(pk):
        if k - h - 5 < 0 or k + h + 6 > len(cm):
            continue
        S = np.stack([cm[k + j - h:k + j + h + 1] / sig for j in range(-5, 6)])            # 11 x (2h+1) x nb
        r[i] = np.max((S * u).sum((1, 2)) / (np.sqrt((S ** 2).sum((1, 2))) + 1e-9))
    return r


def classify(pk, cen, mk_missed, lagwin, near_ev, fs):
    ms = lambda v: int(round(v * fs / 1000)); cls = np.full(len(pk), 5, np.int8)
    if len(near_ev):
        j = np.searchsorted(near_ev, pk); d = np.minimum(np.abs(pk - near_ev[np.clip(j - 1, 0, len(near_ev) - 1)]), np.abs(near_ev[np.clip(j, 0, len(near_ev) - 1)] - pk))
        cls[d <= ms(NEAR_MS)] = 4
    if len(mk_missed) and lagwin is not None:              # event at (marker + lag window) of a marker without a cleaned transient
        j = np.searchsorted(mk_missed, pk - lagwin[1]); ok = (j < len(mk_missed)) & (mk_missed[np.minimum(j, len(mk_missed) - 1)] <= pk - lagwin[0])
        cls[ok] = 3
    if len(cen):
        j = np.searchsorted(cen, pk); prv = cen[np.clip(j - 1, 0, len(cen) - 1)]; nxt = cen[np.clip(j, 0, len(cen) - 1)]
        dp = np.where(j > 0, pk - prv, 10 ** 9); dn = np.where(j < len(cen), nxt - pk, 10 ** 9)
        cls[(dn > ms(CORE_MS)) & (dn <= ms(AFTER_MS)) & (cls > 2)] = 2
        cls[(dp > ms(CORE_MS)) & (dp <= ms(AFTER_MS))] = 1
        cls[(np.minimum(dp, dn) <= ms(CORE_MS))] = 0
    return cls


def lick_free_minutes(ev_all, n, fs):
    w = int(NEAR_MS * fs / 1000); iv = np.stack([np.clip(ev_all - w, 0, n), np.clip(ev_all + w, 0, n)], 1); covered, end = 0, -1
    for a, b in iv:
        if b <= end:
            continue
        covered += b - max(a, end); end = b
    return (n - covered) / fs / 60


def scan(X, Z, F, fs, nch, blocks, uv, rep, ses, log=print):
    t0 = time.time(); ub = np.unique(blocks); n = X.shape[0]; good = working_channels(X, nch, fs)
    cleaned = Z is not None and bool(F.get("cleaned", True)) and "starts" in F
    pre = int(LC.PRE_MS * fs / 1000)
    cen = np.sort(np.asarray(F["starts"], np.int64) + pre) if cleaned else np.sort(np.asarray(F.get("detected", []), np.int64))
    mk = np.sort(np.asarray(F.get("markers", []), np.int64)); U = np.asarray(F["detect_template"], np.float32) if "detect_template" in F else None
    p0, p1 = (int(v * fs / 1000) for v in LC.DET_SEED_MS)
    lagwin, mk_missed = None, mk
    if len(mk) and len(cen):
        j = np.searchsorted(cen, mk + p0); has = (j < len(cen)) & (cen[np.minimum(j, len(cen) - 1)] <= mk + p1)
        mk_missed = mk[~has]
        if has.sum() >= 20:
            lag = (cen[np.minimum(j, len(cen) - 1)] - mk)[has]; lagwin = (int(np.percentile(lag, 5)) - int(0.002 * fs), int(np.percentile(lag, 95)) + int(0.002 * fs))
    near_ev = np.sort(np.concatenate([cen, mk])); free_min = lick_free_minutes(near_ev, n, fs)
    log(f"scan {ses}: {n / fs:.0f} s, {int(good.sum())}/{nch} working channels, {len(cen)} cleaned transients, {len(mk)} lick markers "
        f"({len(mk_missed)} without a transient), lick-free {free_min:.1f} min")
    out, summ, CM = {}, dict(session=ses, duration_min=n / fs / 60, lick_free_min=free_min, n_transients=int(len(cen)), n_markers=int(len(mk)),
                             n_markers_without_transient=int(len(mk_missed)), lag_window_ms=[v / fs * 1000 for v in lagwin] if lagwin else None,
                             working_channels=int(good.sum()), thresholds=list(THR), shape_r=SHAPE_R, classes=CLASSES, files={}), {}
    sig = None
    for nm, A in (("raw", X), ("cleaned", Z if cleaned else None)):
        if A is None:
            continue
        cm = block_cm(A, nch, blocks, ub, good, fs)
        if sig is None:
            sig = 1.4826 * np.median(np.abs(cm), 0) + 1e-6
        e = np.sqrt(((cm / sig.astype(np.float32)) ** 2).sum(1)).astype(np.float32); pk, _ = find_peaks(e, height=CAND, distance=int(0.0005 * fs)); en = e[pk]
        r = shape_r(cm, sig, pk, U); cls = classify(pk, cen, mk_missed, lagwin, near_ev, fs)
        amp = np.abs(cm[pk]).max(1) * uv
        nb10 = int(np.ceil(n / (10 * fs))); tl = np.zeros((len(CLASSES), nb10), np.int32)
        sel = en >= THR[0]; np.add.at(tl, (cls[sel], (pk[sel] // int(10 * fs))), 1)
        k100 = int(0.1 * fs); env = e[:n // k100 * k100].reshape(-1, k100).max(1)
        out.update({f"{nm}_sample": pk.astype(np.int64), f"{nm}_energy": en.astype(np.float32), f"{nm}_r": r, f"{nm}_class": cls,
                    f"{nm}_amp_uv": amp.astype(np.float32), f"{nm}_timeline10s": tl, f"{nm}_env100ms": env})
        c = {}
        for thr in THR:
            for shp in (False, True):
                ok = (en >= thr) & ((r >= SHAPE_R) if shp else True)
                key = f"e{thr:g}" + ("_shape" if shp else "")
                c[key] = {cl: int((ok & (cls == i)).sum()) for i, cl in enumerate(CLASSES)}
                c[key]["away_per_min"] = c[key]["away_from_licks"] / max(free_min, 1e-9)
        if len(cen):     # per cleaned transient: is there an event at it / just after it
            ms_ = lambda v: int(round(v * fs / 1000)); hit = en >= THR[0]; ph = pk[hit]
            j = np.searchsorted(ph, cen - ms_(CORE_MS)); at = (j < len(ph)) & (ph[np.minimum(j, len(ph) - 1)] <= cen + ms_(CORE_MS))
            j2 = np.searchsorted(ph, cen + ms_(CORE_MS) + 1); af = (j2 < len(ph)) & (ph[np.minimum(j2, len(ph) - 1)] <= cen + ms_(AFTER_MS))
            c["transients_with_event_at"] = float(at.mean()); c["transients_with_event_after"] = float(af.mean())
        if len(mk_missed) and lagwin is not None:
            ph = pk[(en >= THR[0]) & (r >= SHAPE_R)] if U is not None else pk[en >= THR[0]]
            j = np.searchsorted(ph, mk_missed + lagwin[0]); hm = (j < len(ph)) & (ph[np.minimum(j, len(ph) - 1)] <= mk_missed + lagwin[1])
            c["markers_without_transient_with_event"] = int(hm.sum())
        summ["files"][nm] = c; CM[nm] = cm
        log(f"  {nm}: {len(pk)} candidate events; at energy >= {THR[0]:g}: " + ", ".join(f"{k} {v}" for k, v in c[f'e{THR[0]:g}'].items()))
    # examples: per class, up to N_EX events (cleaned file's events; missed and away also from the raw file), +-5 ms, both files
    h = int(EX_MS * fs / 1000); T = np.asarray(F["template"]) if "template" in F else None
    chans = np.array([int(np.flatnonzero(blocks == b)[np.argmax(np.ptp(T[:, blocks == b], axis=0))]) for b in ub]) if T is not None else \
        np.array([int(np.flatnonzero((blocks == b) & good)[0]) for b in ub])
    sos = butter(5, 300, "highpass", fs=fs, output="sos"); mg = int(0.03 * fs); rng = np.random.default_rng(0)
    for ci, cl in enumerate(CLASSES):
        for src in ("cleaned", "raw"):
            if f"{src}_sample" not in out:
                continue
            ok = np.flatnonzero((out[f"{src}_class"] == ci) & (out[f"{src}_energy"] >= THR[0]) & (out[f"{src}_sample"] > h + mg) & (out[f"{src}_sample"] < n - h - mg))
            if not len(ok):
                continue
            sel = np.sort(rng.choice(ok, min(N_EX, len(ok)), replace=False)); t = out[f"{src}_sample"][sel]
            out[f"ex_{cl}_{src}_t"] = t
            for nm, A in (("raw", X), ("cleaned", Z if cleaned else None)):
                if A is None:
                    continue
                out[f"ex_{cl}_{src}_cm_{nm}"] = np.stack([CM[nm][k - h:k + h + 1] * uv for k in t]).astype(np.float32)
                out[f"ex_{cl}_{src}_ch_{nm}"] = np.stack([sosfiltfilt(sos, np.asarray(A[k - h - mg:k + h + 1 + mg, :nch][:, chans], np.float32), axis=0)[mg:mg + 2 * h + 1] * uv
                                                          for k in t]).astype(np.float32)
    out.update(session=ses, fs=fs, sig=sig, uv=uv, chans=chans, blocks=ub, transients=cen, markers=mk, markers_missed=mk_missed,
               lag_window=np.array(lagwin if lagwin else [0, 0]), classes=np.array(CLASSES), thresholds=np.array(THR))
    np.savez_compressed(Path(rep) / "lick_clean_scan.npz", **out)
    summ["seconds"] = round(time.time() - t0); (Path(rep) / "lick_clean_scan.json").write_text(json.dumps(summ, indent=1, default=float))
    log(f"scan written: {Path(rep) / 'lick_clean_scan.npz'} ({summ['seconds']} s)")
    return summ


N_STRONG, N_RAND = 12, 12


def event_snippets(A, t, nch, blocks, ub, good, fs, chans, uv, sos):
    """+-EX_MS around each sample in t: each block's common mode (as in the scan) and one channel per block, high-passed, uV"""
    h = int(EX_MS * fs / 1000); mg = int(0.03 * fs); cm, ch = [], []
    for k in t:
        x = np.asarray(A[k - h - mg:k + h + 1 + mg, :nch], np.float32); x -= x.mean(0)
        c = np.stack([np.median(x[:, (blocks == u) & good], 1) for u in ub], 1)
        cm.append(sosfiltfilt(sos, c, axis=0)[mg:mg + 2 * h + 1] * uv); ch.append(sosfiltfilt(sos, x[:, chans], axis=0)[mg:mg + 2 * h + 1] * uv)
    return np.array(cm, np.float32), np.array(ch, np.float32)


def examples(X, Z, S, fs, nch, blocks, uv, rep, log=print, seed=0):
    """2026-10-29 (Emily: 'are the ones shown the only remaining artifacts, or just examples?'): per class the N_STRONG strongest
    events and N_RAND random others (energy >= 12) of the cleaned file (and of the raw file for at_transient / missed_at_lick),
    with both files' snippets, so the worst cases can be seen. Writes lick_clean/lick_clean_scan_examples.npz."""
    ub = np.unique(blocks); good = working_channels(X, nch, fs); sos = butter(5, 300, "highpass", fs=fs, output="sos")
    n = X.shape[0]; h = int(EX_MS * fs / 1000); mg = int(0.03 * fs); rng = np.random.default_rng(seed); chans = np.asarray(S["chans"])
    out = dict(session=S["session"], fs=fs, classes=S["classes"], transients=S["transients"], chans=chans, blocks=ub)
    for src in ("cleaned", "raw"):
        if f"{src}_sample" not in S:
            continue
        pk, en, cls, r = (np.asarray(S[f"{src}_{k}"]) for k in ("sample", "energy", "class", "r"))
        for ci, cl in enumerate(CLASSES):
            if src == "raw" and cl not in ("at_transient", "missed_at_lick"):
                continue
            ok = np.flatnonzero((cls == ci) & (en >= THR[0]) & (pk > h + mg) & (pk < n - h - mg))
            if not len(ok):
                continue
            strong = ok[np.argsort(-en[ok])[:N_STRONG]]; rest = np.setdiff1d(ok, strong)
            rnd = np.sort(rng.choice(rest, min(N_RAND, len(rest)), replace=False)) if len(rest) else np.array([], int)
            sel = np.concatenate([strong, rnd]).astype(int); key = f"x_{cl}_{src}"
            out.update({f"{key}_t": pk[sel], f"{key}_energy": en[sel], f"{key}_r": r[sel], f"{key}_nstrong": len(strong), f"{key}_n": len(ok)})
            for nm, A in (("raw", X), ("cleaned", Z)):
                if A is not None:
                    out[f"{key}_cm_{nm}"], out[f"{key}_ch_{nm}"] = event_snippets(A, pk[sel], nch, blocks, ub, good, fs, chans, uv, sos)
    np.savez_compressed(Path(rep) / "lick_clean_scan_examples.npz", **out); log("examples written:", Path(rep) / "lick_clean_scan_examples.npz")


def main():
    src, cl = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(); ses = src.name
    apf = next((src / f"{ses}_imec0").glob("*.ap.bin")); M = LC.read_meta(apf.with_suffix(".meta"))
    fs = float(M.get("imSampRate", LC.FS_DEFAULT)); nsv = int(M["nSavedChans"]); nch = nsv - 1
    uv = float(M["imAiRangeMax"]) / float(M["imMaxInt"]) / float(M.get("imChan0apGain", 100)) * 1e6
    sh, y = LC.geometry(M); blocks = sh * 2 + (y > LC.SPLIT_UM)
    X = np.memmap(apf, dtype=np.int16, mode="r").reshape(-1, nsv)
    zf = list((cl / f"{ses}_imec0").glob("*.ap.bin")); Z = np.memmap(zf[0], dtype=np.int16, mode="r").reshape(-1, nsv) if zf else None
    if Z is not None and Z.shape != X.shape:   # a --window test copy: the raw side is the same window of the source
        P = json.loads((cl / "lick_clean" / "lick_clean_params.json").read_text()); w = P.get("window_s")
        if not w:
            sys.exit(f"{ses}: cleaned file {Z.shape} differs from raw {X.shape}")
        s0 = int(w[0] * fs); X = X[s0:s0 + Z.shape[0]]
    if "--examples" in sys.argv:   # from an existing scan: strongest + random events per class, both files
        S = dict(np.load(cl / "lick_clean" / "lick_clean_scan.npz", allow_pickle=True))
        F = dict(np.load(cl / "lick_clean" / "lick_clean_fit.npz", allow_pickle=True))
        examples(X, Z if ("starts" in F) else None, S, fs, nch, blocks, uv, cl / "lick_clean", log=LC.log); return
    F = dict(np.load(cl / "lick_clean" / "lick_clean_fit.npz", allow_pickle=True))
    if "cleaned" in F:
        F["cleaned"] = bool(F["cleaned"])
    scan(X, Z, F, fs, nch, blocks, uv, cl / "lick_clean", ses, log=LC.log)
    examples(X, Z if F.get("cleaned", True) and "starts" in F else None, dict(np.load(cl / "lick_clean" / "lick_clean_scan.npz", allow_pickle=True)),
             fs, nch, blocks, uv, cl / "lick_clean", log=LC.log)


if __name__ == "__main__":
    main()
