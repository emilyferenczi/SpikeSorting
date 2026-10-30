"""Per-unit lick-artifact check on a sort of lick-cleaned data (EF, 2026-10-30). Emily: avoid both fake increases and fake dips
around the lick contacts; decide on proportions, not counts; no spike masking around contacts (the cleaning exists to keep them).
Two questions per unit, for all cleaned contacts and for normal / large transients separately:
1. STEP: is the unit's firing in the contact window (-0.5..+2 ms) different from its own firing just around it (3-10 ms before and
   after, both sides, outside every contact window)? Real firing cannot jump up or down within a millisecond of a contact (lick-
   locked firing changes over tens of ms and the symmetric flanks cancel its slope), so a step either way is artifact: residue
   sorted as spikes (fake increase) or spikes lost / distorted at the contact (fake dip). Ratio = contact-window rate / flank rate,
   95% CI and p (counts conditional on the total, binomial on the window lengths), excess spikes per contact.
2. WAVEFORM PER SPIKE: do the unit's spikes in the contact window have its own waveform? Every sampled spike is fitted to the
   unit's far waveform (mean of spikes > 50 ms from every contact, 8 channels with the largest far waveform, -1..+2 ms): amplitude
   a = <x, w> / <w, w> and residual |x - a w|^2 / |w|^2. A spike misfits when a or the residual lies outside the range of the unit's
   own far spikes (0.5-99.5th percentile of a, 99th of the residual; ~2% of far spikes by construction). Fake share = (misfit near
   - misfit far) / (1 - misfit far): the share of the contact-window spikes that are not the unit's spikes (or are its spikes
   distorted by residue). fake_vs_local = fake spikes / the spikes the unit's flank rate predicts in the window.
Also: firing around every contact in 0.1-ms bins, +-15 ms (all / large contacts), for the population and example plots.
The old summary classes are kept for comparison (artifact: >= 50% of spikes in contact windows; few_spikes: < 100 far spikes;
contaminated: near-mean waveform correlation < 0.85 x the expected one with >= 30 near spikes); the rule on the new measures is
chosen from the pilot (Emily).
Reads the CLEANED recording (high-passed 300 Hz, 4-s chunks) twice: pass 1 picks each unit's channels from its far mean (skipped
with --chans-from <earlier unit_lick_check.npz of the same sort>), pass 2 reads the sampled spikes on those channels.
Writes <out>/unit_lick_check.csv / .json / .npz.
--events <npy of sample times> replaces the contacts by other events (2026-10-30: probe-wide events 10-30 ms after the Go cue).
Run: python unit_lick_check.py <cleaned session dir> <sort AIND_<session> dir> <out dir> [--chans-from <npz>] [--events <npy>]   (numpy / scipy)
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.stats import binomtest

import lick_clean_raw as LC
import lick_clean_scan as LS

NEAR_MS, FLANK_MS, FAR_MS = (-0.5, 2.0), (3.0, 10.0), 50.0
N_NEAR, N_FAR, N_CHAN, NCH_FIT = 300, 600, 200, 8
PRE_W, POST_W, PSTH_MS, PSTH_BIN = 1.0, 2.0, 15.0, 3                  # snippet -1..+2 ms; firing around contacts +-15 ms, 3-sample bins
MISFIT_A, MISFIT_R = (0.5, 99.5), 99.0
ARTIFACT_FRAC, MIN_FAR, MIN_NEAR, CORR_DROP = 0.5, 100, 30, 0.85


def union(s, e):
    """merge half-open intervals [s, e) into sorted disjoint ones"""
    if not len(s):
        return s, e
    o = np.argsort(s, kind="stable"); s, e = s[o], e[o]; ce = np.maximum.accumulate(e)
    k = np.flatnonzero(np.r_[True, s[1:] > ce[:-1]])
    return s[k], ce[np.r_[k[1:] - 1, len(ce) - 1]]


def inside(t, iv):
    s, e = iv; i = np.searchsorted(s, t, "right") - 1
    return (i >= 0) & (t < e[np.maximum(i, 0)]) if len(s) else np.zeros(len(t), bool)


def length(iv):
    return int((iv[1] - iv[0]).sum())


def passes(Z, n, nch, fs, jobs, store):
    """one sequential pass: jobs = (time, unit index, slot) sorted by time; store(k, slot, filtered chunk, offset)"""
    sos = butter(5, 300, "highpass", fs=fs, output="sos"); C = int(4 * fs); mg = int(0.01 * fs)
    a0, a1 = int(PRE_W * fs / 1000), int(POST_W * fs / 1000)
    for c0 in range(0, n, C):
        e = jobs[np.searchsorted(jobs[:, 0], c0):np.searchsorted(jobs[:, 0], c0 + C)]
        if not len(e):
            continue
        b0, b1 = max(0, c0 - mg - a0), min(n, c0 + C + mg + a1)
        x = sosfiltfilt(sos, np.asarray(Z[b0:b1, :nch], np.float32), axis=0).astype(np.float32)
        for t, k, sl in e:
            store(k, sl, x[t - a0 - b0:t + a1 - b0])


def main():
    cl, sd, out = (Path(a) for a in sys.argv[1:4]); ses = cl.name; t0 = time.time()
    chans_from = Path(sys.argv[sys.argv.index("--chans-from") + 1]) if "--chans-from" in sys.argv else None
    apf = next((cl / f"{ses}_imec0").glob("*.ap.bin")); M = LC.read_meta(apf.with_suffix(".meta")); nsv = int(M["nSavedChans"]); nch = nsv - 1
    fs = float(M.get("imSampRate", LC.FS_DEFAULT)); uv = float(M["imAiRangeMax"]) / float(M["imMaxInt"]) / float(M.get("imChan0apGain", 100)) * 1e6
    Z = np.memmap(apf, dtype=np.int16, mode="r").reshape(-1, nsv); n = Z.shape[0]; good = LS.working_channels(Z, nch, fs)
    F = np.load(cl / "lick_clean" / "lick_clean_fit.npz", allow_pickle=True); pre = int(LC.PRE_MS * fs / 1000)
    ev = np.unique(np.asarray(F["starts"], np.int64) + pre) if "starts" in F else np.array([], np.int64)
    large = np.unique(np.asarray(F["large"], np.int64)) if "large" in F else np.array([], np.int64)
    if "--events" in sys.argv:          # other event times instead of the contacts (e.g. probe-wide events after the Go cue), as samples
        ev = np.unique(np.load(sys.argv[sys.argv.index("--events") + 1]).astype(np.int64)); large = np.array([], np.int64)
    normal = np.setdiff1d(ev, large)
    st = np.load(sd / "spike_times.npy").astype(np.int64).ravel(); clu = np.load(sd / "spike_clusters.npy").ravel(); units = np.unique(clu)
    o = np.argsort(st, kind="stable"); st, clu = st[o], clu[o]
    ms = lambda v: int(round(v * fs / 1000)); a0, a1 = ms(PRE_W), ms(POST_W); L = a0 + a1; rng = np.random.default_rng(0)
    nlo, nhi, f0, f1, far_s = ms(-NEAR_MS[0]), ms(NEAR_MS[1]), ms(FLANK_MS[0]), ms(FLANK_MS[1]), ms(FAR_MS)
    uidx = np.searchsorted(units, clu); nu = len(units)

    # 1. step: contact windows vs flanks (outside every contact window), per contact group
    win = lambda E, a, b: union(E + a, E + b)
    N_all = win(ev, -nlo, nhi + 1); in_all = inside(st, N_all); tN_all = length(N_all)
    groups = {"all": ev, "normal": normal, "large": large}; step = {}; member = {}
    for g, E in groups.items():
        NG = win(E, -nlo, nhi + 1); FG = union(np.r_[E - f1, E + f0], np.r_[E - f0 + 1, E + f1 + 1])
        inN = inside(st, NG); inF = inside(st, FG) & ~in_all
        tF = length(union(np.r_[FG[0], N_all[0]], np.r_[FG[1], N_all[1]])) - tN_all
        step[g] = dict(n_ev=len(E), tN=length(NG), tF=tF, cN=np.bincount(uidx, inN.astype(float), nu).astype(int), cF=np.bincount(uidx, inF.astype(float), nu).astype(int))
        member[g] = inN
    j = np.searchsorted(ev, st); prv = np.where(j > 0, st - ev[np.maximum(j - 1, 0)], 10 ** 12); nxt = np.where(j < len(ev), ev[np.minimum(j, len(ev) - 1)] - st, 10 ** 12)
    far = np.minimum(prv, nxt) > far_s; ok = (st > a0 + fs) & (st < n - a1 - fs)
    sel_m = {"near_normal": member["all"] & ~member["large"], "near_large": member["large"], "far": far}

    # firing around every contact, 0.1-ms bins
    hw = ms(PSTH_MS); nb = 2 * hw // PSTH_BIN; psth = {g: np.zeros((nu, nb), np.int32) for g in ("all", "large")}
    order = np.argsort(uidx, kind="stable"); bounds = np.r_[0, np.cumsum(np.bincount(uidx, minlength=nu))]   # per unit, in time order
    for k in range(nu):
        t = st[order[bounds[k]:bounds[k + 1]]]
        for g in psth:
            E = groups[g]
            if not len(E) or not len(t):
                continue
            lo, hi = np.searchsorted(t, E - hw), np.searchsorted(t, E + hw); c = hi - lo
            if c.sum():
                ie = np.repeat(np.arange(len(E)), c); pos = lo[ie] + np.arange(c.sum()) - np.repeat(np.cumsum(c) - c, c)
                psth[g][k] = np.bincount((t[pos] - E[ie] + hw) // PSTH_BIN, minlength=nb)[:nb]

    # spikes to read: per unit up to N_NEAR per near group and N_FAR far
    sel = {}
    for k in range(nu):
        iu = order[bounds[k]:bounds[k + 1]]
        for g, m in sel_m.items():
            idx = iu[m[iu] & ok[iu]]; cap = N_FAR if g == "far" else N_NEAR
            sel[k, g] = np.sort(rng.choice(idx, cap, replace=False)) if len(idx) > cap else idx

    # pass 1: channels from the far mean (all channels), unless given
    chans = {}
    if chans_from:
        Wp = np.load(chans_from, allow_pickle=True)
        chans = {k: np.asarray(Wp[f"u{u}_chans"]) for k, u in enumerate(units) if f"u{u}_chans" in Wp}
    todo = [k for k in range(nu) if k not in chans]
    if todo:
        S = np.zeros((nu, L, nch), np.float32); Nn = np.zeros(nu, int)
        sub = {k: sel[k, "far"][::max(1, len(sel[k, "far"]) // N_CHAN)][:N_CHAN] for k in todo}          # spread over the recording
        jb = [np.stack([st[i], np.full(len(i), k), np.zeros(len(i), np.int64)], 1) for k, i in sub.items() if len(i)]
        jb = np.concatenate(jb) if jb else np.zeros((0, 3), np.int64); jb = jb[np.argsort(jb[:, 0], kind="stable")]

        def acc(k, sl, w):
            S[k] += w; Nn[k] += 1
        passes(Z, n, nch, fs, jb, acc)
        for k in todo:
            ptp = np.ptp(S[k] / max(Nn[k], 1), 0) * good; chans[k] = np.argsort(-ptp)[:NCH_FIT]
    print(f"{ses}: channels ready ({round(time.time() - t0)} s, {len(todo)} units from pass 1)", flush=True)

    # pass 2: the sampled spikes on each unit's channels
    gl = list(sel_m); snip = {(k, g): np.zeros((len(sel[k, g]), L, NCH_FIT), np.float32) for k in range(nu) for g in gl}
    jb = [np.stack([st[sel[k, g]], np.full(len(sel[k, g]), k * 3 + gi), np.arange(len(sel[k, g]))], 1) for k in range(nu) for gi, g in enumerate(gl) if len(sel[k, g])]
    jb = np.concatenate(jb); jb = jb[np.argsort(jb[:, 0], kind="stable")]

    def put(kg, sl, w):
        k, gi = divmod(int(kg), 3); snip[k, gl[gi]][sl] = w[:, chans[k]]
    passes(Z, n, nch, fs, jb, put)

    rows, wav = [], {}
    for k, u in enumerate(units):
        iu = order[bounds[k]:bounds[k + 1]]; ns = len(iu); nf = int(far[iu].sum()); nn = int(in_all[iu].sum())
        r = dict(unit=int(u), n_spikes=ns, n_far=nf, n_near=nn, n_near_large=int(member["large"][iu].sum()), frac_near=nn / max(ns, 1),
                 rate_far_hz=nf / max((n - len(ev) * 2 * far_s) / fs, 1), peak_channel=int(chans[k][0]))
        for g, sg in step.items():
            cN, cF = int(sg["cN"][k]), int(sg["cF"][k]); r[f"n_win_{g}"], r[f"n_flank_{g}"] = cN, cF
            if sg["n_ev"] and sg["tF"] > 0 and cF > 0:
                p0 = sg["tN"] / (sg["tN"] + sg["tF"]); bt = binomtest(cN, cN + cF, p0); ci = bt.proportion_ci(0.95)
                odds = lambda p: p / max(1 - p, 1e-12) * sg["tF"] / sg["tN"]
                r[f"step_{g}"] = odds(cN / (cN + cF)); r[f"step_{g}_lo"], r[f"step_{g}_hi"] = odds(ci.low), odds(ci.high); r[f"step_{g}_p"] = bt.pvalue
                r[f"excess_per_contact_{g}"] = (cN - cF * sg["tN"] / sg["tF"]) / sg["n_ev"]
                r[f"flank_rate_hz_{g}"] = cF / (sg["tF"] / fs)
        Fs = snip[k, "far"]
        if len(Fs) >= 20:
            w = Fs.mean(0).ravel(); ww = float(w @ w) + 1e-9
            af = (Fs.reshape(len(Fs), -1) @ w) / ww; rf = ((Fs.reshape(len(Fs), -1) - af[:, None] * w) ** 2).sum(1) / ww
            alo, ahi = np.percentile(af, MISFIT_A); rhi = np.percentile(rf, MISFIT_R)
            mis = lambda a, rr: (a < alo) | (a > ahi) | (rr > rhi); mfar = float(mis(af, rf).mean()); r["misfit_far"] = mfar
            var = Fs.var(0); mf = w - w.mean()
            for g in ("near_normal", "near_large"):
                X = snip[k, g]; nk = len(X); r[f"n_read_{g}"] = nk; gg = g.replace("near_", "")
                if nk >= 5:
                    a = (X.reshape(nk, -1) @ w) / ww; rr = ((X.reshape(nk, -1) - a[:, None] * w) ** 2).sum(1) / ww; m = mis(a, rr)
                    ci = binomtest(int(m.sum()), nk).proportion_ci(0.95); sh = lambda p: max(0.0, (p - mfar) / (1 - mfar))
                    r[f"misfit_{gg}"] = float(m.mean()); r[f"fake_share_{gg}"] = sh(m.mean()); r[f"fake_share_{gg}_lo"], r[f"fake_share_{gg}_hi"] = sh(ci.low), sh(ci.high)
                    if f"step_{gg}" in r:
                        r[f"fake_vs_local_{gg}"] = r[f"fake_share_{gg}"] * r[f"step_{gg}"]   # fake spikes / spikes expected from the flank rate
                    mn = X.mean(0).ravel(); mn = mn - mn.mean()
                    r[f"corr_{g}"] = float(mn @ mf / (np.linalg.norm(mn) * np.linalg.norm(mf) + 1e-9))
                    r[f"corr_{g}_expected"] = float(np.linalg.norm(mf) / np.sqrt(np.linalg.norm(mf) ** 2 + var.sum() / nk + 1e-9))
                    r[f"amp_median_{gg}"] = float(np.median(a))
                    if m.any():
                        wav[f"u{u}_misfit_{gg}"] = X[np.flatnonzero(m)[:10]] * uv
        bad = [g for g in ("near_normal", "near_large") if r.get(f"n_read_{g}", 0) >= MIN_NEAR and f"corr_{g}" in r and r[f"corr_{g}"] < CORR_DROP * r[f"corr_{g}_expected"]]
        r["class_old"] = ("artifact" if r["frac_near"] >= ARTIFACT_FRAC else "few_spikes" if nf < MIN_FAR else "contaminated" if bad else "real")
        r["contaminated_after_old"] = "+".join(b.replace("near_", "") for b in bad)
        rows.append(r); wav[f"u{u}"] = np.stack([snip[k, g].mean(0) if len(snip[k, g]) else np.zeros((L, NCH_FIT), np.float32) for g in gl]) * uv
        wav[f"u{u}_chans"] = chans[k]
    out.mkdir(parents=True, exist_ok=True); keys = []
    for r_ in rows:
        keys += [k_ for k_ in r_ if k_ not in keys]
    with open(out / "unit_lick_check.csv", "w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=keys); w_.writeheader(); w_.writerows(rows)
    np.savez_compressed(out / "unit_lick_check.npz", session=ses, fs=fs, groups=np.array(gl), units=units, psth_all=psth["all"], psth_large=psth["large"],
                        psth_bin=PSTH_BIN, psth_half=hw, **wav)
    cnt = {c: sum(r_["class_old"] == c for r_ in rows) for c in ("real", "contaminated", "artifact", "few_spikes")}
    (out / "unit_lick_check.json").write_text(json.dumps(dict(session=ses, units=len(rows), classes_old=cnt, contacts=int(len(ev)), large=int(len(large)),
                                                               t_window_s=round(tN_all / fs, 1), params=dict(near_ms=NEAR_MS, flank_ms=FLANK_MS, far_ms=FAR_MS,
                                                               n_near=N_NEAR, n_far=N_FAR, misfit_a_pct=MISFIT_A, misfit_r_pct=MISFIT_R, artifact_frac=ARTIFACT_FRAC,
                                                               min_far=MIN_FAR, min_near=MIN_NEAR, corr_drop=CORR_DROP), seconds=round(time.time() - t0)), indent=1))
    print(f"{ses}: {len(rows)} units -> old classes {cnt} ({round(time.time() - t0)} s)", flush=True)


if __name__ == "__main__":
    main()
