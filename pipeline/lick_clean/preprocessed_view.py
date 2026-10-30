"""The data as Kilosort receives it, around given events (re-sort v2 check B9, EF 2026-10-30): the sort's own preprocessed
recording (preprocessed/<rec>.json: phase shift, 300 Hz high-pass, float32, bad channels interpolated, destripe per shank) read
with SpikeInterface, then Kilosort 4's own reference (io.py: each channel's mean removed, then the MEDIAN across channels per sample).
Per event +-10 ms (read with +-100 ms margins for the filters / AGC), all channels; plus the same for random times (noise).
Run inside the pipeline's base container (SpikeInterface 0.101.2):
  singularity exec <containers>/aind-ephys-pipeline-base_si-0.101.2.sif python preprocessed_view.py <preprocessed json> \
      <cleaned session dir> <events .npy> <out .npz> [N]
"""
import json
import sys
from pathlib import Path

import numpy as np
import spikeinterface as si


def fix_paths(o, new):
    if isinstance(o, dict):
        return {k: (new if k == "folder_path" else fix_paths(v, new)) for k, v in o.items()}
    if isinstance(o, list):
        return [fix_paths(v, new) for v in o]
    return o


def main():
    jf, cl, ef, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]); N = int(sys.argv[5]) if len(sys.argv) > 5 else 12
    d = fix_paths(json.loads(jf.read_text()), str(cl.resolve())); d["relative_paths"] = False
    rec = si.load_extractor(d); fs = rec.get_sampling_frequency(); n = rec.get_num_samples()
    h, m = int(0.010 * fs), int(0.100 * fs)
    ev = np.sort(np.load(ef).astype(np.int64)); ev = ev[(ev > m + h) & (ev < n - m - h)]
    ev = ev[np.linspace(0, len(ev) - 1, min(N, len(ev))).astype(int)] if len(ev) else ev
    rnd = np.random.default_rng(0).integers(m + h + int(fs), n - m - h - int(fs), N)

    def view(t):
        x = rec.get_traces(start_frame=int(t - h - m), end_frame=int(t + h + m)).astype(np.float32)
        x = x - x.mean(0, keepdims=True); car = x - np.median(x, 1, keepdims=True)
        return x[m:m + 2 * h], car[m:m + 2 * h]
    P = [view(t) for t in ev]; R = [view(t) for t in rnd]
    loc = rec.get_channel_locations(); grp = rec.get_property("group"); grp = np.zeros(rec.get_num_channels(), int) if grp is None else grp
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, fs=fs, events=ev, pre=np.stack([p[0] for p in P]).astype(np.float16), car=np.stack([p[1] for p in P]).astype(np.float16),
                        rnd_pre=np.stack([p[0] for p in R]).astype(np.float16), rnd_car=np.stack([p[1] for p in R]).astype(np.float16),
                        channel_ids=np.asarray(rec.get_channel_ids()).astype(str), x_um=loc[:, 0], y_um=loc[:, 1], group=np.asarray(grp))
    print(out, len(ev), "events,", rec.get_num_channels(), "channels")


if __name__ == "__main__":
    main()
