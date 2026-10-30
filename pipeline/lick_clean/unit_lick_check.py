"""Per-unit lick-artifact check on a sort of lick-cleaned data (EF, 2026-10-30; Emily: 'yes go ahead and validate it on the pilot
sorts'). Are a unit's spikes close to the lick contacts real spikes or residue of the contact transient? Real spikes keep the unit's
waveform; residue does not. Rate alone cannot tell: strongly lick-locked neurons fire many real spikes at the contact
(0604 EF401 #51 / #258: same waveform near and far).
One sequential pass over the CLEANED recording (high-passed 300 Hz, 4-s chunks): for every unit, the mean waveform on every channel
(-1..+2 ms) of up to N spikes in each group
  near_normal  spikes -0.5..+2 ms after a cleaned normal transient
  near_large   the same after a cleaned large transient (fit 'large')
  far          spikes > 50 ms from every cleaned transient (+ the per-sample variance, for the noise of a mean of n spikes)
Per unit, on its 8 channels with the largest far waveform: correlation of the near means with the far mean, and the correlation
expected if the near spikes had the far waveform (far mean vs far mean + noise of n_near spikes); amplitude ratio near / far.
Classes (rules validated on the 2026-10-29/30 pilot sorts):
  artifact      most spikes at contacts (frac_near >= ARTIFACT_FRAC)
  few_spikes    < MIN_FAR spikes away from contacts: too few to compare (validation 2026-10-30: rare-spiking units, not artifact)
  contaminated  >= MIN_NEAR near spikes whose waveform differs (corr < CORR_DROP x expected) -- keep the unit, drop its spikes
                -0.5..+2 ms after contacts (after normal or large transients, whichever differs)
  real          otherwise (incl. lick-locked units)
Writes <out>/unit_lick_check.csv + unit_lick_check.npz (8-channel mean waveforms per unit and group).
Run: python unit_lick_check.py <cleaned session dir> <sort AIND_<session> dir> <out dir>      (numpy / scipy only)
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

import lick_clean_raw as LC
import lick_clean_scan as LS

NEAR_MS, FAR_MS, N_PER_GROUP, PRE_W, POST_W = (-0.5, 2.0), 50.0, 300, 1.0, 2.0
ARTIFACT_FRAC, MIN_FAR, MIN_NEAR, CORR_DROP = 0.5, 100, 30, 0.85


def main():
    cl, sd, out = (Path(a) for a in sys.argv[1:4]); ses = cl.name; t0 = time.time()
    apf = next((cl / f"{ses}_imec0").glob("*.ap.bin")); M = LC.read_meta(apf.with_suffix(".meta")); nsv = int(M["nSavedChans"]); nch = nsv - 1
    fs = float(M.get("imSampRate", LC.FS_DEFAULT)); uv = float(M["imAiRangeMax"]) / float(M["imMaxInt"]) / float(M.get("imChan0apGain", 100)) * 1e6
    Z = np.memmap(apf, dtype=np.int16, mode="r").reshape(-1, nsv); n = Z.shape[0]; good = LS.working_channels(Z, nch, fs)
    F = np.load(cl / "lick_clean" / "lick_clean_fit.npz", allow_pickle=True); pre = int(LC.PRE_MS * fs / 1000)
    ev = np.sort(np.asarray(F["starts"], np.int64) + pre) if "starts" in F else np.array([], np.int64)
    large = np.sort(np.asarray(F["large"], np.int64)) if "large" in F else np.array([], np.int64)
    st = np.load(sd / "spike_times.npy").astype(np.int64).ravel(); clu = np.load(sd / "spike_clusters.npy").ravel(); units = np.unique(clu)
    a0, a1 = int(PRE_W * fs / 1000), int(POST_W * fs / 1000); L = a0 + a1; rng = np.random.default_rng(0)
    nlo, nhi = int(-NEAR_MS[0] * fs / 1000), int(NEAR_MS[1] * fs / 1000); far_s = int(FAR_MS * fs / 1000)
    # per spike: time since the previous contact / to the next, and whether that contact is a large one
    j = np.searchsorted(ev, st); prv = np.where(j > 0, st - ev[np.maximum(j - 1, 0)], 10 ** 12); nxt = np.where(j < len(ev), ev[np.minimum(j, len(ev) - 1)] - st, 10 ** 12)
    near = (prv <= nhi) | (nxt <= nlo); far = np.minimum(prv, nxt) > far_s
    cprev = np.where(prv <= nhi, ev[np.maximum(j - 1, 0)], ev[np.minimum(j, len(ev) - 1)]) if len(ev) else np.zeros(len(st), np.int64)
    is_large = np.isin(cprev, large) & near
    groups = {"near_normal": near & ~is_large, "near_large": near & is_large, "far": far}
    sel = []                                                          # (time, unit index, group index) to read
    uidx = np.searchsorted(units, clu); order = np.argsort(uidx, kind="stable"); bounds = np.searchsorted(uidx[order], np.arange(len(units) + 1))
    inside = (st > a0 + fs) & (st < n - a1 - fs); gm = list(groups.values())
    for k in range(len(units)):
        iu = order[bounds[k]:bounds[k + 1]]
        for gi, m in enumerate(gm):
            idx = iu[m[iu] & inside[iu]]
            if len(idx) > N_PER_GROUP:
                idx = rng.choice(idx, N_PER_GROUP, replace=False)
            sel.append(np.stack([st[idx], np.full(len(idx), k), np.full(len(idx), gi)], 1))
    sel = np.concatenate(sel); sel = sel[np.argsort(sel[:, 0])]
    S = np.zeros((len(units), 3, L, nch), np.float32); SS = np.zeros((len(units), L, nch), np.float32); N = np.zeros((len(units), 3), int)
    sos = butter(5, 300, "highpass", fs=fs, output="sos"); C = int(4 * fs); mg = int(0.01 * fs)
    for c0 in range(0, n, C):
        e = sel[np.searchsorted(sel[:, 0], c0):np.searchsorted(sel[:, 0], c0 + C)]
        if not len(e):
            continue
        b0, b1 = max(0, c0 - mg - a0), min(n, c0 + C + mg + a1)
        x = sosfiltfilt(sos, np.asarray(Z[b0:b1, :nch], np.float32), axis=0).astype(np.float32)
        for t, k, gi in e:
            w = x[t - a0 - b0:t + a1 - b0]
            S[k, gi] += w; N[k, gi] += 1
            if gi == 2:
                SS[k] += w * w
    rows, wav = [], {}
    for k, u in enumerate(units):
        iu = order[bounds[k]:bounds[k + 1]]; ns = len(iu); nn = int(near[iu].sum()); nf = int(far[iu].sum())
        Mf = S[k, 2] / max(N[k, 2], 1); var = np.maximum(SS[k] / max(N[k, 2], 1) - Mf ** 2, 0)
        ptp = np.ptp(Mf, 0) * good; top = np.argsort(-ptp)[:8]; pc = int(top[0])
        r = dict(unit=int(u), n_spikes=ns, n_near=nn, n_near_large=int((near[iu] & is_large[iu]).sum()), n_far=nf, frac_near=nn / max(ns, 1),
                 peak_channel=pc, ptp_far_uv=float(ptp[pc] * uv))
        # near-window rate vs the unit's own far rate (lick-locked units are above 1 -- information, not a criterion)
        far_time = max(n - len(ev) * 2 * far_s, 1); near_time = max(len(ev) * (nlo + nhi), 1)
        r["near_rate_ratio"] = (nn / near_time) / max(nf / far_time, 1e-12)
        mf = Mf[:, top].ravel(); mf = mf - mf.mean()
        for gi, g in ((0, "near_normal"), (1, "near_large")):
            nk = N[k, gi]
            if nk >= 5 and N[k, 2] >= 20:
                mn = (S[k, gi] / nk)[:, top].ravel(); mn = mn - mn.mean()
                r[f"corr_{g}"] = float(mn @ mf / (np.linalg.norm(mn) * np.linalg.norm(mf) + 1e-9))
                noise = float(var[:, top].sum() / nk)                              # expected: far waveform + the noise of nk spikes
                r[f"corr_{g}_expected"] = float(np.linalg.norm(mf) / np.sqrt(np.linalg.norm(mf) ** 2 + noise + 1e-9))
                r[f"amp_ratio_{g}"] = float(np.ptp((S[k, gi] / nk)[:, pc]) / max(np.ptp(Mf[:, pc]), 1e-9))
            r[f"n_read_{g}"] = int(nk)
        bad = [g for g in ("near_normal", "near_large") if r.get(f"n_read_{g}", 0) >= MIN_NEAR and f"corr_{g}" in r
               and r[f"corr_{g}"] < CORR_DROP * r[f"corr_{g}_expected"]]
        r["class"] = ("artifact" if r["frac_near"] >= ARTIFACT_FRAC else "few_spikes" if nf < MIN_FAR else "contaminated" if bad else "real")
        r["contaminated_after"] = "+".join(b.replace("near_", "") for b in bad)
        rows.append(r); wav[f"u{u}"] = np.stack([S[k, gi][:, top] / max(N[k, gi], 1) for gi in range(3)]) * uv; wav[f"u{u}_chans"] = top
    out.mkdir(parents=True, exist_ok=True); keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open(out / "unit_lick_check.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    np.savez_compressed(out / "unit_lick_check.npz", session=ses, fs=fs, groups=np.array(list(groups)), **wav)
    cnt = {c: sum(r["class"] == c for r in rows) for c in ("real", "contaminated", "artifact", "few_spikes")}
    (out / "unit_lick_check.json").write_text(json.dumps(dict(session=ses, units=len(rows), classes=cnt, contacts=int(len(ev)), large=int(len(large)),
                                                               params=dict(near_ms=NEAR_MS, far_ms=FAR_MS, n_per_group=N_PER_GROUP, artifact_frac=ARTIFACT_FRAC,
                                                                           min_far=MIN_FAR, min_near=MIN_NEAR, corr_drop=CORR_DROP), seconds=round(time.time() - t0)), indent=1))
    print(f"{ses}: {len(rows)} units -> {cnt} ({round(time.time() - t0)} s)", flush=True)


if __name__ == "__main__":
    main()
