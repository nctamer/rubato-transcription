"""Greedy decoding with continuous batching over fixed KV-cache slots.

The task prompt is processed in one pass. Each subsequent step consumes the
previous token and selects the argmax of the token log-probabilities. Attention
scales Q and K by head_dim ** -0.25 and uses additive -10000 masks.

A window stops at EOS, its timestamp limit, or its token limit. Output excludes
trailing padding and EOS tokens. Finished slots are refilled with new windows,
so a batch does not wait for its longest window.

Optional attention tracking and recovery are disabled by default. When enabled,
the decoder records cross-attention positions using the helpers in align.py.
Recovery rolls a flagged window back to its last reliable timestamp and retries
with temperatures from 0.2 to 1.0.
"""

import bisect
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import align as AL

NEG = -10000.0

try:  # optional fused single-query attention over each slot's filled cache prefix (CUDA only)
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:  # pragma: no cover
    HAVE_TRITON = False

if HAVE_TRITON:
    @triton.jit
    def _decode_attn_kernel(Q, K, V, O, LENS, ARG, sqb, sqh, skb, skh, skt, svb, svh, svt, sob, soh,
                            H, D: tl.constexpr, BLOCK_N: tl.constexpr, WITH_ARG: tl.constexpr):
        pid = tl.program_id(0)
        b = pid // H
        h = pid % H
        n = tl.load(LENS + b)
        offs_d = tl.arange(0, D)
        q = tl.load(Q + b * sqb + h * sqh + offs_d).to(tl.float32)
        m = tl.max(tl.full([BLOCK_N], float("-inf"), tl.float32), 0)
        l = tl.sum(tl.zeros([BLOCK_N], tl.float32), 0)
        acc = tl.zeros([D], tl.float32)
        bidx = tl.sum(tl.zeros([BLOCK_N], tl.int32), 0)
        kb = K + b * skb + h * skh
        vb = V + b * svb + h * svh
        for start in range(0, n, BLOCK_N):
            offs_n = start + tl.arange(0, BLOCK_N)
            valid = offs_n < n
            k = tl.load(kb + offs_n[:, None] * skt + offs_d[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
            s = tl.sum(k * q[None, :], axis=1)
            s = tl.where(valid, s, float("-inf"))
            bm = tl.max(s, 0)
            if WITH_ARG:
                bi = tl.argmax(s, 0) + start
                bidx = tl.where(bm > m, bi, bidx)
            m_new = tl.maximum(m, bm)
            alpha = tl.exp(m - m_new)
            p = tl.exp(s - m_new)
            l = l * alpha + tl.sum(p, 0)
            v = tl.load(vb + offs_n[:, None] * svt + offs_d[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            m = m_new
        l = tl.where(l > 0, l, 1.0)
        tl.store(O + b * sob + h * soh + offs_d, (acc / l).to(O.dtype.element_ty))
        if WITH_ARG:
            tl.store(ARG + b * H + h, bidx)


def decode_attention(q, k, v, lens, block_n=32, arg=None):
    """q [B, H, D] (pre-scaled), k/v [B, H, T, D], lens [B] -> [B, H, D] over keys [0, lens[b]). arg: optional int32
    [B, H] output of the argmax key index (the alignment signal; the attention output is unchanged by it)."""
    B, H, D = q.shape
    o = torch.empty_like(q)
    lens = lens.to(torch.int32)
    _decode_attn_kernel[(B * H,)](q, k, v, o, lens, arg if arg is not None else lens,
                                  q.stride(0), q.stride(1), k.stride(0), k.stride(1), k.stride(2),
                                  v.stride(0), v.stride(1), v.stride(2), o.stride(0), o.stride(1),
                                  H, D=D, BLOCK_N=block_n, WITH_ARG=arg is not None, num_warps=4)
    return o


def score_argmax(q, k, lens):
    """Argmax key index [B, H] of q . k over keys [0, lens[b]) (fp32 products)."""
    s = torch.einsum("bhd,bhtd->bht", q.float(), k.float())
    T = k.shape[2]
    s = s.masked_fill(torch.arange(T, device=q.device)[None, None, :] >= lens[:, None, None], float("-inf"))
    return s.argmax(-1)


def _heads(linear, x, nh):
    B, S, _ = x.shape
    y = linear(x)
    return y.view(B, S, nh, y.shape[-1] // nh).transpose(1, 2)


def _merge(ctx, out_proj):
    B, h, S, hd = ctx.shape
    return out_proj(ctx.transpose(1, 2).contiguous().view(B, S, h * hd))


class SlotDecoder:
    """admit() prefills rows into free slots; step() advances every live slot by one token; collect() returns
    (key, ids) of finished slots and frees them."""

    def _limit_rank(self, limit):
        """Rank of the largest timestamp value <= limit (-1 if none), in double precision on the host."""
        return bisect.bisect_right(self._ts_sorted, float(limit)) - 1

    def __init__(self, model, n_slots, prompt, max_new, enc_frames, ts_values, eos, pad, device,
                 kv_dtype=torch.float32, attn="sdpa", cuda_graph=False):
        dec = model.transf_decoder
        self.emb = dec._embedding
        self.layers = list(dec._decoder.layers)
        self.final_ln = dec._decoder.final_layer_norm
        self.head = model.log_softmax
        self.nh = dec.n_heads
        sa = self.layers[0].first_sub_layer
        self.hd = sa.query_net.out_features // self.nh
        self.sc = math.sqrt(math.sqrt(self.hd))
        self.embed_vocab = self.emb.token_embedding.num_embeddings
        self.eos, self.pad = int(eos), int(pad)
        self.dev = torch.device(device)
        self.kv_dtype = kv_dtype
        self.S = int(n_slots)
        self.prompt = torch.as_tensor(prompt, dtype=torch.long, device=self.dev)
        self.P = int(self.prompt.numel())
        self.max_new = int(max_new)
        self.T = self.P + self.max_new
        self.Te = int(enc_frames)
        V = max(self.embed_vocab, 1 + max(ts_values, default=0))
        # The stop test "written time > window limit" runs on integer RANKS of the timestamp values, so no device
        # needs float64 (Apple MPS has none): rank = position of the token's value in the sorted value list, and a
        # limit becomes the rank of the largest value <= limit, found on the host in double precision
        # (_limit_rank). rank > limit_rank  <=>  value > limit, the float64 decision exactly. Non-timestamp
        # tokens get -2, never above a limit rank >= -1.
        self._ts_sorted = sorted(set(float(v) for v in ts_values.values()))
        rank = {v: i for i, v in enumerate(self._ts_sorted)}
        tv = torch.full((V,), -2, dtype=torch.long)
        for tid, val in ts_values.items():
            tv[tid] = rank[float(val)]
        self.ts_val = tv.to(self.dev)
        tf = torch.full((V,), float("nan"), dtype=torch.float32)      # stamp values (alignment signal only)
        for tid, val in ts_values.items():
            tf[tid] = float(val)
        self.ts_float = tf.to(self.dev)
        self._ts_host = np.where(np.isnan(tf.numpy()), -1.0, tf.numpy()).astype(np.float32)
        self.det = None
        self.rec_on = False
        if attn == "triton" and not (HAVE_TRITON and self.dev.type == "cuda"):
            attn = "sdpa"
        self.attn = attn
        self.cuda_graph = bool(cuda_graph) and attn == "triton"
        self.graphs, self._warm = {}, set()
        self._pool = torch.cuda.graph_pool_handle() if self.cuda_graph else None
        self.buckets = sorted({int(n_slots)} | {b for b in (8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512)
                                                 if b < int(n_slots)})
        S, nh, hd, T, Te = self.S, self.nh, self.hd, self.T, self.Te
        kw = dict(device=self.dev, dtype=kv_dtype)
        self.sk = [torch.zeros(S, nh, T, hd, **kw) for _ in self.layers]
        self.sv = [torch.zeros(S, nh, T, hd, **kw) for _ in self.layers]
        self.ck = [torch.zeros(S, nh, Te, hd, **kw) for _ in self.layers]
        self.cv = [torch.zeros(S, nh, Te, hd, **kw) for _ in self.layers]
        self.cmask = torch.full((S, 1, 1, Te), NEG, device=self.dev)
        self.enc_len = torch.zeros(S, dtype=torch.int32, device=self.dev)
        z = dict(device=self.dev, dtype=torch.long)
        self.cur = torch.full((S,), self.pad, **z)
        self.pos = torch.full((S,), self.P, **z)
        self.gen = torch.zeros(S, **z)
        self.cap = torch.zeros(S, **z)
        self.thr = torch.zeros(S, dtype=torch.long, device=self.dev)          # limit ranks (_limit_rank)
        self.done = torch.ones(S, dtype=torch.bool, device=self.dev)
        self.out = torch.full((S, self.max_new + 1), self.pad, **z)
        self.ar = torch.arange(S, device=self.dev)
        self.arT = torch.arange(max(T, Te), device=self.dev)
        self.owner = [None] * S
        self.age = [0] * S
        self.n_live = S
        self.steps = 0
        self.tokens = 0

    # ------------------------------------------------------------------ alignment signal / recovery
    def enable_alignment(self, recover=False, det_kw=None, rec_kw=None):
        """Track the alignment signal and the detector for every window (win_info[key] after collect(): the
        detector's result on the window's own head-argmax buffer = its attention-failure flag). recover=True: a
        window whose rule fires (R1 rewind, R2 collapse, R3 k stamps off the attention by > tau) is rolled back to its
        last reliable stamp and re-decoded with the temperature fallback rec_kw['temps'] until it is margin_s /
        margin_tok past the failure; at most max_retries per window, then the attempt with the furthest monotone
        progress is kept and finished greedily. Rows that never fire decode exactly as without it."""
        det_kw = dict(AL.DET, **(det_kw or {}))
        self.det_kw = det_kw
        self.det = AL.OnlineDetector(self.S, self.max_new + 1, self.Te, self.dev, **det_kw)
        self._arg = torch.zeros(len(AL.LAYERS), self.S, self.nh, dtype=torch.int32, device=self.dev)
        self.rec_on = bool(recover)
        self.rec_kw = dict(temps=(0.2, 0.4, 0.6, 0.8, 1.0), max_retries=5, margin_s=2.0, margin_tok=32, seed=0)
        self.rec_kw.update(rec_kw or {})
        self.rec = [None] * self.S
        self.win_info = {}
        self.fire_log = []
        self.rec_stats = dict(fires=0, rollbacks=0, exhausted=0, prefill_restores=0, t_host=0.0)

    def cache_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.sk + self.sv + self.ck + self.cv)

    def free_slots(self):
        return [i for i in range(self.S) if self.owner[i] is None]

    def n_busy(self):
        return self.S - self.owner.count(None)

    def _logp(self, h):
        h = self.final_ln(h)
        logp = self.head(h)
        if logp.shape[-1] > self.embed_vocab:
            logp[..., self.embed_vocab:] = -float("inf")
        return logp

    @torch.inference_mode()
    def admit(self, slots, keys, enc, enc_len, eos_thr, cap):
        """Prefill rows `slots` with decoder memory enc [m, Te, H] (float32), enc_len [m], timestamp limits
        eos_thr (seconds) and token caps."""
        m = len(slots)
        if m == 0:
            return
        st = torch.tensor(slots, dtype=torch.long, device=self.dev)
        enc = enc.float()
        valid = self.arT[:enc.shape[1]][None, :] < enc_len[:, None]
        emask = ((1.0 - valid.float()) * NEG)[:, None, None, :]
        P = self.P
        ids = self.prompt[None, :].expand(m, P)
        pos = torch.arange(P, device=self.dev)[None, :].expand(m, P)
        h = self.emb(ids, pos)
        causal = torch.tril(torch.ones(P, P, dtype=torch.bool, device=self.dev))
        cmask_p = ((1 - causal.float()) * NEG)[None, None].expand(m, 1, P, P)
        arg0 = []
        for li, L in enumerate(self.layers):
            sa, ca, ff = L.first_sub_layer, L.second_sub_layer, L.third_sub_layer
            x = L.layer_norm_1(h)
            q, k, v = (_heads(n, x, self.nh) for n in (sa.query_net, sa.key_net, sa.value_net))
            att = F.scaled_dot_product_attention(q / self.sc, k / self.sc, v, attn_mask=cmask_p, scale=1.0)
            self.sk[li][st, :, :P] = (k / self.sc).to(self.kv_dtype)
            self.sv[li][st, :, :P] = v.to(self.kv_dtype)
            h = h + _merge(att, sa.out_projection)
            x = L.layer_norm_2(h)
            qc = _heads(ca.query_net, x, self.nh)
            ck = _heads(ca.key_net, enc, self.nh)
            cv = _heads(ca.value_net, enc, self.nh)
            att = F.scaled_dot_product_attention(qc / self.sc, ck / self.sc, cv, attn_mask=emask, scale=1.0)
            if self.det is not None and li in AL.LAYERS:
                arg0.append(score_argmax((qc[:, :, -1] / self.sc), (ck / self.sc).to(self.kv_dtype), enc_len))
            self.ck[li][st] = (ck / self.sc).to(self.kv_dtype)
            self.cv[li][st] = cv.to(self.kv_dtype)
            h = h + _merge(att, ca.out_projection)
            x = L.layer_norm_3(h)
            h = h + ff(x)
        logp = self._logp(h[:, -1:, :]).squeeze(1)
        t0 = logp.argmax(dim=-1)
        er = torch.tensor([self._limit_rank(e) for e in eos_thr], dtype=torch.long, device=self.dev)
        cp = torch.tensor(list(cap), dtype=torch.long, device=self.dev)
        self.cmask[st] = emask
        self.enc_len[st] = enc_len.to(torch.int32)
        self.cur[st] = t0
        self.pos[st] = P
        self.gen[st] = 1
        self.out[st, 0] = t0
        self.thr[st] = er
        self.cap[st] = cp
        self.done[st] = (t0 == self.eos) | (self.ts_val[t0] > er) | (cp <= 1)
        if self.det is not None:
            self.det.reset(st, track=True, act=self.rec_on)
            self.det.first_token(st, torch.stack(arg0, 1).reshape(m, -1), self.ts_float[t0], enc_len)
        for s, key in zip(slots, keys):
            self.owner[s] = key
            self.age[s] = 0
            if self.det is not None:
                self.rec[s] = dict(key=key, Fn=0, retries=0, tidx=0, attempts=[], fail_clock=None, fires=[],
                                   exhausted=False, enc_len=None) if self.rec_on else None
            if s >= self.n_live:
                self.n_live = s + 1

    def _body(self, n, L=None):
        """Device work of one step for slots [0, n). L: self-attention key length of the SDPA path (None with the
        fused kernel, whose shapes do not depend on positions, so the step can be captured in a CUDA graph)."""
        cur, pos, done = self.cur[:n], self.pos[:n], self.done[:n]
        h = self.emb(cur[:, None], pos[:, None])
        if L is not None:
            smask = torch.where(self.arT[None, :L] <= pos[:, None], 0.0, NEG)[:, None, None, :]
            cm = self.cmask[:n]
        slen = pos + 1
        ar = self.ar[:n]
        for li, Ly in enumerate(self.layers):
            sa, ca, ff = Ly.first_sub_layer, Ly.second_sub_layer, Ly.third_sub_layer
            x = Ly.layer_norm_1(h)
            q = _heads(sa.query_net, x, self.nh)
            k = _heads(sa.key_net, x, self.nh)
            v = _heads(sa.value_net, x, self.nh)
            sk, sv = self.sk[li], self.sv[li]
            sk[ar, :, pos] = (k / self.sc)[:, :, 0].to(self.kv_dtype)
            sv[ar, :, pos] = v[:, :, 0].to(self.kv_dtype)
            if L is not None:
                att = F.scaled_dot_product_attention(q / self.sc, sk[:n, :, :L].float(), sv[:n, :, :L].float(),
                                                     attn_mask=smask, scale=1.0)
            else:
                att = decode_attention((q / self.sc)[:, :, 0], sk[:n], sv[:n], slen)[:, :, None]
            h = h + _merge(att, sa.out_projection)
            x = Ly.layer_norm_2(h)
            qc = _heads(ca.query_net, x, self.nh)
            tr = self.det is not None and li in AL.LAYERS
            if L is not None:
                att = F.scaled_dot_product_attention(qc / self.sc, self.ck[li][:n].float(), self.cv[li][:n].float(),
                                                     attn_mask=cm, scale=1.0)
                if tr:
                    self._arg[AL.LAYERS.index(li), :n] = score_argmax((qc / self.sc)[:, :, 0], self.ck[li][:n],
                                                                      self.enc_len[:n]).to(torch.int32)
            else:
                att = decode_attention((qc / self.sc)[:, :, 0], self.ck[li][:n], self.cv[li][:n],
                                       self.enc_len[:n], arg=self._arg[AL.LAYERS.index(li), :n] if tr else None
                                       )[:, :, None]
            h = h + _merge(att, ca.out_projection)
            x = Ly.layer_norm_3(h)
            h = h + ff(x)
        logp = self._logp(h).squeeze(1)
        nxt = logp.argmax(dim=-1)
        live = ~done
        g = self.gen[:n]
        if self.rec_on:     # temperature rows of a recovery attempt: Gumbel-max, counter-hash noise per (window, attempt)
            tq = self.det.temp[:n]
            z = logp.float() / tq.clamp(min=1e-6)[:, None] + AL.gumbel(self.det.seed[:n], g, logp.shape[-1], self.dev)
            nxt = torch.where(tq > 0, z.argmax(dim=-1), nxt)
        nxt = torch.where(live, nxt, torch.full_like(nxt, self.pad))
        gi = torch.where(live, g, self.max_new)
        self.out[:n].scatter_(1, gi[:, None], nxt[:, None])
        g += live.long()
        hit = (nxt == self.eos) | (self.ts_val[nxt] > self.thr[:n])
        end = hit | (g >= self.cap[:n])
        if self.det is not None:
            a = self._arg[:, :n].permute(1, 0, 2).reshape(n, AL.NHD)
            fired = self.det.update(n, a, self.ts_float[nxt], gi, live, end, self.enc_len)
            if self.rec_on:
                end = end | fired
        done |= live & end
        pos += live.long()
        self.cur[:n] = torch.where(live, nxt, cur)

    @torch.inference_mode()
    def step(self):
        n = self.n_live
        if self.n_busy() == 0:
            return
        if self.attn == "sdpa":
            live_age = [self.age[i] for i in range(n) if self.owner[i] is not None]
            L = min(self.T, self.P + max(live_age) + 1)
            self._body(n, min(self.T, ((L + 31) // 32) * 32))
        elif self.cuda_graph:
            # one graph per slot-count bucket (rows past n_live are free and done: computing them changes
            # nothing); a bucket's first step runs eagerly (warm-up), the second captures, later ones replay
            nb = next(b for b in self.buckets if b >= n)
            g = self.graphs.get(nb)
            if g is None and nb in self._warm:
                g = self.graphs[nb] = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=self._pool, capture_error_mode="thread_local"):
                    self._body(nb)
            if g is not None:
                g.replay()
            else:
                self._body(nb)
                self._warm.add(nb)
        else:
            self._body(n)
        for i in range(n):
            if self.owner[i] is not None:
                self.age[i] += 1
        self.steps += 1

    @torch.inference_mode()
    def collect(self):
        """[(key, ids)] of finished slots (trailing <pad>/<eos> removed); frees them."""
        n = self.n_live
        done = self.done[:n].tolist()
        if self.rec_on:
            fire = self.det.fire[:n].tolist()
            for s in range(n):
                if done[s] and fire[s] and self.owner[s] is not None and self.rec[s] is not None:
                    self._recover(s, fire[s])
                    done[s] = False
        fin = [s for s in range(n) if done[s] and self.owner[s] is not None]
        res = []
        if fin:
            gen = self.gen[fin].tolist()
            outs = self.out[fin].tolist()
            for s, ln, row in zip(fin, gen, outs):
                ids = row[:ln]
                self.tokens += ln
                k = len(ids)
                while k > 0 and ids[k - 1] in (self.pad, self.eos):
                    k -= 1
                res.append((self.owner[s], ids[:k], ln))
                if self.det is not None:
                    self._finish(s, ids[:k])
                self.owner[s] = None
            occ = [i for i in range(self.S) if self.owner[i] is not None]
            self.n_live = (occ[-1] + 1) if occ else 1
        return res

    @torch.inference_mode()
    def compact(self):
        """Move occupied slots to the front so step() only touches live rows."""
        occ = [i for i in range(self.n_live) if self.owner[i] is not None]
        m = len(occ)
        if occ == list(range(m)):
            self.n_live = max(m, 1)
            return
        idx = torch.tensor(occ, device=self.dev, dtype=torch.long)
        for bufs in (self.sk, self.sv, self.ck, self.cv):
            for b in bufs:
                b[:m] = b[idx]
        for name in ("cmask", "enc_len", "cur", "pos", "gen", "cap", "thr", "done", "out"):
            t = getattr(self, name)
            t[:m] = t[idx]
        self.done[m:] = True
        if self.det is not None:
            self.det.compact(idx, m)
            self.rec = [self.rec[i] for i in occ] + [None] * (self.S - m)
        self.owner = [self.owner[i] for i in occ] + [None] * (self.S - m)
        self.age = [self.age[i] for i in occ] + [0] * (self.S - m)
        self.n_live = max(m, 1)

    # ------------------------------------------------------------------ recovery (host side)
    def _finish(self, s, ids):
        """Finished window of slot s (output ids): the attention-failure detector on its own head-argmax buffer, the online
        first fires and the recovery record -> self.win_info[key]."""
        ln = len(ids)
        a = self.det.att[s, :ln].cpu().numpy()
        ts = self._ts_host[np.asarray(ids, np.int64)] if ln else np.zeros(0, np.float32)
        info = dict(n_tok=ln, af=AL.window_result(a, ts, int(self.enc_len[s])), online_ff=self.det.ff[s].tolist())
        r = self.rec[s] if self.rec_on else None
        if r is not None:
            info.update(fires=r["fires"], retries=r["retries"], exhausted=r["exhausted"])
            if r["fires"]:
                r["fires"][-1]["outcome"] = "exhausted_kept" if r["exhausted"] else "finished"
        self.win_info[self.owner[s]] = info
        self.rec[s] = None

    def _recover(self, s, code):
        """Slot s stopped on a fire (rule code 1 R1 / 2 R2 / 3 R3): roll back to its reliable point and retry with the
        next temperature, or, past max_retries, keep the best attempt and finish it greedily without rollbacks."""
        t0 = time.time()
        d, r = self.det, self.rec[s]
        kw = self.rec_kw
        gen = int(self.gen[s])
        ids = self.out[s, :gen].cpu().numpy()
        h = d.fetch(s, gen)
        el = int(self.enc_len[s])
        if r["enc_len"] is None:
            r["enc_len"] = el
        fire_tok = gen - 1
        fire_clock = float(h["clk"][fire_tok])
        # failure onset: R1 the rewind run's first token, R2 the parking start of the heads_needed-th head, R3 the
        # first disagreeing stamp of the run; the reliable point is the last agreeing + monotone stamp before it
        if code == 1:
            onset = int(d.rj0[s])
        elif code == 2:
            onset = AL.collapse_onset(h["att"], h["clk"], el, self.det_kw["heads_needed"], self.det_kw["park_n"],
                                      self.det_kw["park_span"])
            onset = fire_tok if onset is None else onset
        else:
            st = np.flatnonzero(h["agr"] >= 0)
            good = st[h["agr"][st] == 1]
            later = st[st > (good[-1] if len(good) else -1)]
            onset = int(later[0]) if len(later) else fire_tok
        # rollback target: the last reliable stamp (agreeing + monotone) at least `back` tokens before the onset
        # (back = rec_kw['back'] tokens, measured: the audio failure starts before the detector onset); a fire during a
        # retry (temperature still on) escalates: back x (1 + consecutive re-fires of the region)
        in_retry = float(d.temp[s]) > 0
        r["esc"] = r.get("esc", 0) + 1 if in_retry else 0
        nb = int(kw.get("back", 0)) * (1 + r["esc"])
        if kw.get("unit", "tokens") == "moments":     # n Moments: drop the last nb complete Moments before the onset
            st_on = np.flatnonzero(h["agr"][:max(0, onset)] >= 0)
            lim = int(st_on[len(st_on) - nb]) if 0 < nb <= len(st_on) else (0 if nb else onset)
        else:
            lim = onset - nb
        st_all = np.flatnonzero(h["agr"][:max(0, lim)] >= 0)
        good = st_all[h["agr"][st_all] == 1]
        keep = int(good[-1] + 1) if len(good) else (int(st_all[-1] + 1) if len(st_all) else r["Fn"])
        keep = max(keep, r["Fn"])
        keep_clock = float(h["clk"][keep - 1]) if keep > 0 else 0.0
        start = r["attempts"][-1]["restart"] if r["attempts"] else r["Fn"]
        st_a = np.flatnonzero(h["agr"][start:] >= 0)
        ndis = int((h["agr"][start:][st_a] == 0).sum())
        att_rec = dict(retry=r["retries"], temp=float(d.temp[s]), start=start, fire_tok=fire_tok,
                       fire_clock=fire_clock, rule=int(code), onset=int(onset), reliable=keep,
                       reliable_clock=keep_clock, ndis=ndis, ids=ids, h=h)
        r["attempts"].append(att_rec)
        self.rec_stats["fires"] += 1
        fl = dict(key=list(r["key"]) if isinstance(r["key"], tuple) else r["key"], token=fire_tok,
                  clock=round(fire_clock, 3), rule={1: "R1", 2: "R2", 3: "R3"}[int(code)], onset=int(onset),
                  retry=r["retries"], temp=round(float(d.temp[s]), 2), rollback_to=keep,
                  rollback_clock=round(keep_clock, 3), escalation=r["esc"], outcome="refired")
        if r["fires"]:
            r["fires"][-1]["outcome"] = "refired"
        r["fires"].append(fl)
        self.fire_log.append(fl)
        if r["retries"] >= int(kw["max_retries"]):
            best = max(range(len(r["attempts"])), key=lambda i: (r["attempts"][i]["reliable_clock"],
                                                                 -r["attempts"][i]["ndis"], -i))
            b = r["attempts"][best]
            fl.update(action="exhausted", kept_attempt=best)
            r["exhausted"] = True
            self.rec_stats["exhausted"] += 1
            n_keep = b["fire_tok"] + 1
            if best != len(r["attempts"]) - 1:
                self._restore(s, b["ids"][:n_keep], b["h"], n_keep, el)
                self.rec_stats["prefill_restores"] += 1
            d.act[s] = False
            d.temp[s] = 0.0
            d.fire[s] = 0
            self.done[s] = bool(n_keep >= int(self.cap[s]))
            for a_ in r["attempts"]:
                a_.pop("h", None)
            self.rec_stats["t_host"] += time.time() - t0
            return
        new_region = not in_retry
        r["tidx"] = 0 if new_region else min(r["tidx"] + 1, len(kw["temps"]) - 1)
        r["retries"] += 1
        r["fail_clock"] = fire_clock
        T = float(kw["temps"][r["tidx"]])
        att_rec["restart"] = keep
        fl.update(action="rollback", next_temp=T)
        self._rollback(s, keep, h, el, r["Fn"])
        d.temp[s] = T
        d.seed[s] = AL.row_seed(r["key"], r["retries"], kw["seed"])
        d.back_clk[s] = fire_clock + float(kw["margin_s"])
        d.back_tok[s] = keep + int(kw["margin_tok"])
        self.rec_stats["rollbacks"] += 1
        self.rec_stats["t_host"] += time.time() - t0

    def _rollback(self, s, keep, h, el, Fn):
        """Truncate slot s to its first `keep` tokens: the self-attention cache is positional, so the row continues
        from position P + keep - 1 by feeding its last kept token again (the prompt's last token when keep == 0);
        the cross-attention cache is unchanged."""
        self.gen[s] = keep
        self.out[s, keep:] = self.pad
        self.pos[s] = self.P + keep - 1
        self.cur[s] = self.out[s, keep - 1] if keep > 0 else self.prompt[-1]
        self.done[s] = False
        self.det.rebuild(s, keep, h, el, rel0=Fn)
        self.det.act[s] = True

    @torch.inference_mode()
    def _restore(self, s, ids, h, n_keep, el):
        """Rewrite slot s with the token prefix `ids` of an earlier attempt (one teacher-forced pass of prompt +
        ids[:-1] fills the self-attention cache; cross-attention K/V from the slot's cache)."""
        ids = [int(x) for x in ids]
        inp = torch.cat([self.prompt, torch.tensor(ids[:-1], dtype=torch.long, device=self.dev)])[None]
        Lp = inp.shape[1]
        h_ = self.emb(inp, torch.arange(Lp, device=self.dev)[None, :])
        causal = torch.tril(torch.ones(Lp, Lp, dtype=torch.bool, device=self.dev))
        cm = ((1 - causal.float()) * NEG)[None, None]
        for li, L in enumerate(self.layers):
            sa, ca, ff = L.first_sub_layer, L.second_sub_layer, L.third_sub_layer
            x = L.layer_norm_1(h_)
            q, k, v = (_heads(n, x, self.nh) for n in (sa.query_net, sa.key_net, sa.value_net))
            att = F.scaled_dot_product_attention(q / self.sc, k / self.sc, v, attn_mask=cm, scale=1.0)
            self.sk[li][s, :, :Lp] = (k / self.sc)[0].to(self.kv_dtype)
            self.sv[li][s, :, :Lp] = v[0].to(self.kv_dtype)
            h_ = h_ + _merge(att, sa.out_projection)
            x = L.layer_norm_2(h_)
            qc = _heads(ca.query_net, x, self.nh)
            att = F.scaled_dot_product_attention(qc / self.sc, self.ck[li][s:s + 1].float(), self.cv[li][s:s + 1].float(),
                                                 attn_mask=self.cmask[s:s + 1], scale=1.0)
            h_ = h_ + _merge(att, ca.out_projection)
            x = L.layer_norm_3(h_)
            h_ = h_ + ff(x)
        n = len(ids)
        self.out[s] = self.pad
        self.out[s, :n] = torch.tensor(ids, dtype=torch.long, device=self.dev)
        self.gen[s] = n
        self.pos[s] = self.P + n - 1
        self.cur[s] = ids[-1]
        self.det.rebuild(s, n, h, el)
