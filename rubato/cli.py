"""Command line: transcribe audio files.

    python -m rubato.cli WEIGHTS AUDIO [AUDIO ...] [--mode score|midi|beats] [--device cpu|cuda]
                              [--out-dir DIR] [--json OUT.json] [--threads N]

Greedy decoding with 40 s windows every 20 s (each keeps its first 20 s), for every mode.
Writes <stem>.<mode>.txt per input (stitched transcript) into --out-dir (default: next to stdout listing), and
optionally one JSON with every transcript and its per-window texts.
"""

import argparse
import json
import os
import sys
import time

import torch

from .api import Rubato, default_threads
from .prompts import MODES


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights")
    ap.add_argument("audio", nargs="+")
    ap.add_argument("--mode", choices=MODES, default="score")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads (default: $SLURM_CPUS_PER_TASK or 1)")
    ap.add_argument("--slots", type=int, default=None, help="windows decoded together (default 128 CUDA / 16 CPU)")
    ap.add_argument("--enc-batch", type=int, default=None, help="windows per encoder batch (default 8 CUDA / 2 CPU)")
    ap.add_argument("--no-tf32", action="store_true", help="CUDA: full fp32 matmuls")
    ap.add_argument("--no-cuda-graph", action="store_true", help="CUDA: eager decoder steps")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--json", default=None, help="also write all results (with per-window texts) to this file")
    ap.add_argument("--recover", choices=("off", "track", "on"), default="off",
                    help="track: report each window's attention-failure flag (--json); on: also recovery decoding "
                         "(roll a failing window back and re-decode it with the temperature fallback)")
    ap.add_argument("--rec-tau", type=float, default=None, help="recovery R3: stamp vs attention tolerance (s)")
    ap.add_argument("--rec-k", type=int, default=None, help="recovery R3: consecutive disagreeing stamps")
    ap.add_argument("--rec-back", "--rec-n", dest="rec_back", type=int, default=0,
                    help="recovery: roll back at least this many tokens / Moments (--rec-unit) before the onset")
    ap.add_argument("--rec-unit", choices=("tokens", "moments"), default="tokens")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads or default_threads())
    t0 = time.time()
    model = Rubato.load(a.weights, device=a.device, tf32=not a.no_tf32, cuda_graph=not a.no_cuda_graph)
    print(f"loaded {a.weights} on {a.device} in {time.time() - t0:.1f}s", file=sys.stderr)
    os.makedirs(a.out_dir, exist_ok=True)
    res = {}
    t1 = time.time()
    for r in model.transcribe(a.audio, mode=a.mode, slots=a.slots, enc_batch=a.enc_batch, recover=a.recover,
                              rec_tau=a.rec_tau, rec_k=a.rec_k, rec_back=a.rec_back, rec_unit=a.rec_unit):
        stem = os.path.splitext(os.path.basename(str(r["key"])))[0]
        out = os.path.join(a.out_dir, f"{stem}.{a.mode}.txt")
        with open(out, "w") as f:
            f.write(r["text"] + "\n")
        res[str(r["key"])] = dict(text=r["text"], duration=r["duration"], chunks=[c["text"] for c in r["chunks"]])
        if a.recover != "off":
            res[str(r["key"])]["align"] = [c.get("align") for c in r["chunks"]]
        print(f"{r['key']}: {r['duration']:.1f}s audio -> {out}", file=sys.stderr)
    dt = time.time() - t1
    audio = sum(v["duration"] for v in res.values())
    print(f"{len(res)} files, {audio:.1f}s audio in {dt:.1f}s ({audio / max(dt, 1e-9):.1f}x real time)",
          file=sys.stderr)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(dict(mode=a.mode, weights=os.path.abspath(a.weights), results=res), f,
                      indent=1)


if __name__ == "__main__":
    main()
