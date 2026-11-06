"""Loading a weights file and transcribing pieces of audio."""

import collections
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from . import protocol as P
from .audio import load_audio
from .decode import SlotDecoder
from .model import RubatoModel
from .prompts import MODES
from .tokenizer import Tokenizer

FORMAT = "rubato_lite/1"
MODE_DEFAULTS = {"score": dict(max_generation_delta=1000), "midi": dict(max_generation_delta=2000),
                 "beats": dict(max_generation_delta=1000)}


def default_threads():
    for k in ("SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS"):
        v = os.environ.get(k)
        if v and v.isdigit() and int(v) > 0:
            return int(v)
    return 1


class Rubato:
    """A loaded model. Rubato.load(path, device) then .transcribe(...)."""

    def __init__(self, obj, device="cpu", tf32=True, kv_dtype=None, attn=None, cuda_graph=True):
        if obj.get("format") != FORMAT:
            raise ValueError(f"not a rubato weights file (format {obj.get('format')!r})")
        self.device = torch.device(device)
        self.cfg = obj["config"]
        self.model = RubatoModel(self.cfg)
        sd = {k: v.float() if v.is_floating_point() else v for k, v in obj["state_dict"].items()}
        self.model.load_state_dict(sd, strict=True)
        self.model.eval().to(self.device)
        t = obj["tokenizer"]
        self.tokenizer = Tokenizer(t["pieces"], t["types"], t["unk_surface"], t["add_dummy_prefix"],
                                   t["remove_extra_whitespaces"])
        self.prompts = obj["prompts"]
        self.eos, self.pad = obj["special"]["eos"], obj["special"]["pad"]
        self.max_seq_length = int(obj["max_seq_length"])
        self.ts_values = P.timestamp_values(self.tokenizer.vocab)
        self.source = obj.get("source", {})
        cuda = self.device.type == "cuda"
        if cuda:
            # Enable TF32 for CUDA matrix multiplications and convolutions
            torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
            torch.backends.cudnn.allow_tf32 = bool(tf32)
        self.kv_dtype = kv_dtype or (torch.float16 if cuda else torch.float32)
        self.attn = attn or ("triton" if cuda else "sdpa")
        self.cuda_graph = bool(cuda_graph) and cuda

    @classmethod
    def load(cls, path, device="cpu", **kw):
        obj = torch.load(path, map_location="cpu", weights_only=True)
        return cls(obj, device=device, **kw)

    # ------------------------------------------------------------------------------------------- frames
    def encoder_frames(self, n_samples):
        """Valid encoder frames of a window of n_samples."""
        f = self.cfg["features"]
        t = n_samples // f["hop"]
        for _ in range(int(round(np.log2(self.cfg["encoder"]["sub_factor"])))):
            t = (t + 2 - 3) // 2 + 1
        return t

    # ------------------------------------------------------------------------------------------- main loop
    def transcribe(self, inputs, mode="score", slots=None, enc_batch=None, sync_every=None,
                   max_generation_delta=None, load_workers=2, progress=None, recover="off", rec_tau=None, rec_k=None,
                   rec_seed=0, rec_back=0, rec_unit="tokens"):
        """Transcribe pieces. inputs: iterable of audio file paths, of float32 16 kHz numpy arrays, or of
        (key, path-or-array) pairs. Yields dicts (key, text, chunks, plan, duration) in completion order.

        mode: score | midi | beats. Greedy decoding, 40 s windows every 20 s (each keeps its first 20 s).
        recover: off | track (each chunk dict gains `align`: the attention-failure detector's result on the window)
        | on (also recovery decoding: a window whose detector rule fires is rolled back to its last reliable stamp and
        re-decoded with the temperature fallback; R3 = rec_k consecutive stamps more than rec_tau s from the
        attention, off without rec_tau / rec_k)."""
        if recover not in ("off", "track", "on"):
            raise ValueError("recover must be off, track or on")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}")
        proto = dict(P.W40H20)
        delta = int(max_generation_delta or proto.get("max_generation_delta") or
                    MODE_DEFAULTS[mode]["max_generation_delta"])
        cs, hs = float(proto["chunk_size"]), float(proto["hop_size"])
        mso = float(proto.get("merge_start_offset", 0.0))
        mgt = float(proto["max_generation_timestamp"])
        cuda = self.device.type == "cuda"
        slots = int(slots or (128 if cuda else 16))
        enc_batch = int(enc_batch or (8 if cuda else 2))
        sync_every = int(sync_every or (16 if cuda else 4))
        dev = self.device
        frame_len = int(cs * P.SR)
        Te = self.encoder_frames(frame_len)
        prompt = self.prompts[mode]
        max_new = min(self.max_seq_length, Te + delta) - len(prompt)
        dec = SlotDecoder(self.model, slots, prompt, max_new, Te, self.ts_values, self.eos, self.pad, dev,
                          kv_dtype=self.kv_dtype, attn=self.attn, cuda_graph=self.cuda_graph)
        if recover != "off":
            dec.enable_alignment(recover=recover == "on", det_kw=dict(tau=rec_tau, k=rec_k),
                                 rec_kw=dict(seed=int(rec_seed), back=int(rec_back), unit=rec_unit))

        def norm(i, x):
            if isinstance(x, tuple):
                return x
            return (x if isinstance(x, (str, os.PathLike)) else i, x)

        items = [norm(i, x) for i, x in enumerate(inputs)]

        def prepare(item):
            key, src = item
            audio = load_audio(src) if isinstance(src, (str, os.PathLike)) else np.asarray(src, dtype=np.float32)
            plan = P.plan_piece(len(audio), cs, hs, mso)
            chunks, lens = P.chunk_audio(audio, plan, cs, hs)
            return key, plan, chunks, lens

        pool = ThreadPoolExecutor(max_workers=max(1, load_workers))
        futs = collections.deque(pool.submit(prepare, it) for it in items[:max(2, 2 * load_workers)])
        nxt_item = len(futs)
        pieces = {}
        audio_q = collections.deque()
        enc_q = collections.deque()
        exhausted = False
        stats = collections.Counter()

        def pull_audio(need):
            nonlocal exhausted, nxt_item
            while len(audio_q) < need and not exhausted:
                if not futs:
                    exhausted = True
                    break
                key, plan, chunks, lens = futs.popleft().result()
                if nxt_item < len(items):
                    futs.append(pool.submit(prepare, items[nxt_item]))
                    nxt_item += 1
                pi = stats["pieces"]
                stats["pieces"] += 1
                n = plan["n_chunks"]
                pieces[pi] = dict(key=key, plan=plan, texts=[None] * n, ids=[None] * n, remaining=n, align=[None] * n,
                                  thr=P.eos_thresholds(plan, mgt))
                stats["audio_seconds"] += plan["duration"]
                if n == 0:
                    ready.append(pi)
                    continue
                a = torch.from_numpy(chunks)
                for c in range(n):
                    audio_q.append(((pi, c), a[c], int(lens[c])))

        def run_encoder():
            te = time.time()
            m = min(enc_batch, len(audio_q))
            batch = [audio_q.popleft() for _ in range(m)]
            audio = torch.stack([b[1] for b in batch]).to(dev)
            lens = torch.tensor([b[2] for b in batch], dtype=torch.long, device=dev)
            enc, enc_len = self.model.encode(audio, lens)
            enc = enc[:, :Te]
            for i, b in enumerate(batch):
                enc_q.append((b[0], enc[i], enc_len[i]))
            stats["enc_chunks"] += m
            stats["t_encoder_host"] += time.time() - te   # wall on CPU; launch time only on CUDA

        def admit():
            free = dec.free_slots()
            k = min(len(free), len(enc_q))
            for g in range(0, k, 64):
                its = [enc_q.popleft() for _ in range(min(64, k - g))]
                keys = [it[0] for it in its]
                dec.admit(free[g:g + len(its)], keys, torch.stack([it[1] for it in its]),
                          torch.stack([it[2] for it in its]),
                          [pieces[kk[0]]["thr"][kk[1]] for kk in keys], [max_new] * len(its))

        ready = []

        def handle(done):
            for (pi, c), ids, ln in done:
                p = pieces[pi]
                if ln >= max_new:
                    stats["chunks_at_cap"] += 1
                p["ids"][c] = ids
                p["texts"][c] = self.tokenizer.ids_to_text(ids)
                if dec.det is not None:
                    p["align"][c] = dec.win_info.pop((pi, c), None)
                p["remaining"] -= 1
                stats["chunks"] += 1
                if p["remaining"] == 0:
                    ready.append(pi)

        def finish(pi):
            p = pieces.pop(pi)
            text = P.merge_piece(p["texts"], p["plan"], mso)
            chunks = [dict(ids=i, text=t) for i, t in zip(p["ids"], p["texts"])]
            if dec.det is not None:
                for ch, al in zip(chunks, p["align"]):
                    ch["align"] = al
            return dict(key=p["key"], text=text, chunks=chunks,
                        plan=p["plan"], duration=p["plan"]["duration"])

        t0 = time.time()
        enc_low = max(enc_batch, slots // 4)
        try:
            pull_audio(enc_batch)
            while True:
                target = len(dec.free_slots()) + enc_low
                while len(enc_q) < target and (audio_q or not exhausted):
                    pull_audio(enc_batch)
                    if audio_q:
                        run_encoder()
                admit()
                while ready:
                    yield finish(ready.pop(0))
                drain = exhausted and not audio_q and not enc_q
                if dec.n_busy() == 0 and drain:
                    break
                for _ in range(sync_every):
                    dec.step()
                handle(dec.collect())
                if drain and dec.n_busy() <= dec.n_live // 2:
                    dec.compact()
                if progress:
                    progress(stats, dec)
            while ready:
                yield finish(ready.pop(0))
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
            self.last_stats = dict(stats, wall=time.time() - t0, tokens=dec.tokens, steps=dec.steps,
                                   token_cap=max_new, encoder_frames=Te, slots=slots,
                                   kv_cache_gb=dec.cache_bytes() / 1e9)

    def transcribe_file(self, path, mode="score", **kw):
        """Stitched transcript text of one audio file."""
        return list(self.transcribe([path], mode=mode, **kw))[0]["text"]
