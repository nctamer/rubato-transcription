"""Attention tracking and failure detection for optional recovery decoding.

For each generated token, the decoder records the encoder-frame argmax of
24 cross-attention heads from decoder layers 1-3. Signals are stored per slot
as int16 arrays with shape [tokens, 24], ordered by layer then head.

window_result evaluates a completed window and returns onset, rewind, and
collapse token indices, or None when no failure is detected. OnlineDetector
applies the following rules during generation, vectorized over decoder slots:

* Rewind: the robust reading position falls behind its running maximum while
  the generated timestamp clock advances. A centered nine-token median filter
  delays decisions by four tokens.
* Collapse: enough heads repeatedly attend to the same two-frame bin while
  the generated timestamp clock advances.
* Timestamp disagreement: generated timestamps disagree with the robust
  attention position for a configured number of consecutive timestamps.

The robust position is the median of non-sink heads. The clock is the running
maximum of generated timestamps. A reliable recovery point is a monotone
timestamp whose value agrees with the attention position.
"""

import zlib

import numpy as np
import torch

FR = 0.04                 # encoder frame (s)
LAYERS = (1, 2, 3)        # decoder layers of the detector
NHD = 24                  # 3 layers x 8 heads
DET = dict(D=1.0, L=32, heads_needed=14, park_n=30, park_span=2.0)   # Default rewind and attention-collapse thresholds


def window_result(att, ts, el, **kw):
    """Evaluate attention failures on one finished window: att [n, 24] argmax frames (layers 1-3), ts [n] stamp value (s) or
    < 0, el valid encoder frames -> dict(onset, rewind, collapse) (token indices or None)."""
    d = dict(DET, **kw)
    return detect(np.asarray(att), np.asarray(ts, np.float32), int(el), **d)


# ------------------------------------------------------------------ detector (numpy)
def median_filter(x, size):
    h = size // 2
    xp = np.concatenate([np.full(h, x[0]), x, np.full(h, x[-1])])
    return np.median(np.lib.stride_tricks.sliding_window_view(xp, size), axis=1).astype(x.dtype)


def clock_of(ts):
    """ts [n] stamp value (s) or < 0 -> written clock per token (running max of 5-stamp-median-filtered stamps)."""
    pos = np.where(ts >= 0)[0]
    c = np.zeros(len(ts), np.float32)
    if len(pos):
        c[pos] = median_filter(ts[pos].astype(np.float32), 5)
    return np.maximum.accumulate(c)


def stray_mask(ts_vals, tol):
    """ts_vals: the stamp values of one sample in order -> bool keep mask (False = stray single stamp)."""
    if len(ts_vals) < 3:
        return np.ones(len(ts_vals), bool)
    return np.abs(ts_vals - median_filter(ts_vals.astype(np.float32), 5)) <= tol


def robust_pos(A, el):
    n = A.shape[0]
    a = A.reshape(n, -1).astype(np.float32)
    a[a >= el - 3] = np.nan
    ok = ~np.all(np.isnan(a), axis=1)
    if ok.sum() < 3:
        return None
    p = np.full(n, np.nan, np.float32)
    p[ok] = np.nanmedian(a[ok], axis=1) * FR
    idx = np.where(ok, np.arange(n), 0)
    idx = np.maximum.accumulate(idx)
    idx[:int(np.argmax(ok))] = int(np.argmax(ok))
    return median_filter(p[idx], 9)


def online_rewind(ps, clock, D, L, adv=0.3):
    front = np.maximum.accumulate(ps)
    rr = ps < front - D
    run, c0, j0 = 0, 0.0, 0
    for j in range(len(ps)):
        if rr[j]:
            if run == 0:
                c0, j0 = clock[j], j
            run += 1
            if run >= L and clock[j] - c0 >= adv:
                return j0
        else:
            run = 0
    return None


def park_times(a, clock, el, n_need=30, span=2.0):
    """tools/eval/attn_failure.park_times: per head the first token at which it is parked; inf if never."""
    n, H = a.shape
    out = np.full(H, np.inf)
    for h in range(H):
        f = a[:, h]
        valid = (f > 2) & (f < el - 3)
        if valid.sum() < n_need:
            continue
        b = (f // 2).astype(np.int64)
        idx = np.where(valid)[0]
        bb = b[idx]
        order = np.argsort(bb, kind="stable")
        bs, ix = bb[order], idx[order]
        cuts = np.flatnonzero(np.diff(bs)) + 1
        for grp in np.split(ix, cuts):
            if len(grp) < n_need:
                continue
            c = clock[grp]
            ok = np.where((np.arange(len(grp)) >= n_need - 1) & (c - c[0] >= span))[0]
            if len(ok):
                out[h] = min(out[h], grp[ok[0]])
    return out


def detect(A, ts, el, D=1.0, L=32, heads_needed=14, park_n=30, park_span=2.0):
    """A [n, 24] argmax frames (layers 1-3) of one sample's generated tokens; ts [n] stamp value or -1; el = valid
    encoder frames. Returns dict(onset, rewind, collapse) (token indices or None)."""
    n = A.shape[0]
    res = dict(onset=None, rewind=None, collapse=None)
    if n < 20:
        return res
    ps = robust_pos(A, el)
    if ps is None:
        return res
    clock = clock_of(ts)
    rw = online_rewind(ps, clock, D, L)
    pt = np.sort(park_times(A.astype(np.int64), clock, el, park_n, park_span))
    co = int(pt[heads_needed - 1]) if np.isfinite(pt[heads_needed - 1]) else None
    ons = [t for t in (rw, co) if t is not None]
    res.update(rewind=rw, collapse=co, onset=min(ons) if ons else None)
    return res



def row_seed(key, attempt, seed=0):
    """Deterministic 31-bit sampling seed of a window attempt (independent of batch composition)."""
    return zlib.crc32(f"{seed}|{key}|{attempt}".encode()) & 0x7FFFFFFF


def _mix32(x):
    """lowbias32 integer hash on int64 tensors holding 32-bit values."""
    M = 0xFFFFFFFF
    x = x & M
    x = x ^ (x >> 16)
    x = (x * 0x7FEB352D) & M
    x = x ^ (x >> 15)
    x = (x * 0x846CA68B) & M
    x = x ^ (x >> 16)
    return x


def gumbel(seed, pos, V, device):
    """Gumbel noise [n, V] from a counter-based hash of (row seed, position, vocab id): reproducible per row whatever
    the rows sharing the step, and capturable in a CUDA graph (no generator state)."""
    v = torch.arange(V, device=device, dtype=torch.int64)
    x = _mix32(seed[:, None].long() * 0x9E3779B1 + pos[:, None].long() * 0x85EBCA77)
    x = _mix32(x ^ (v[None, :] * 0xC2B2AE3D))
    u = (x.float() + 0.5) * (1.0 / 4294967296.0)
    return -torch.log(-torch.log(u))


class OnlineDetector:
    """Per-slot buffers and the causal rule state of S decoder rows with up to T token positions (T - 1 = scratch
    column for finished rows, as SlotDecoder.out). All updates are masked tensor ops (graph-capturable)."""

    BUF = ("att", "rp", "fp", "clk", "agr", "lastp", "hasv", "cclk", "front", "run", "rc0", "rj0", "pcnt", "pfc",
           "parked", "dis", "rel", "ndis", "ff", "r1on", "fire", "trk", "act", "temp", "seed", "back_clk", "back_tok")

    def __init__(self, S, T, Te, device, D=1.0, L=32, adv=0.3, heads_needed=14, park_n=30, park_span=2.0,
                 tau=None, k=None):
        self.S, self.T, self.Te, self.dev = S, T, Te, torch.device(device)
        self.NB = Te // 2 + 1
        self.D, self.L, self.adv = float(D), int(L), float(adv)
        self.hn, self.pn, self.ps = int(heads_needed), int(park_n), float(park_span)
        self.tau = None if tau is None else float(tau)
        self.k3 = None if k is None else int(k)
        dv = self.dev
        f = lambda *s: torch.zeros(*s, device=dv, dtype=torch.float32)        # noqa: E731
        lg = lambda *s: torch.zeros(*s, device=dv, dtype=torch.long)          # noqa: E731
        b = lambda *s: torch.zeros(*s, device=dv, dtype=torch.bool)           # noqa: E731
        self.att = torch.zeros(S, T, NHD, device=dv, dtype=torch.int16)
        self.rp = torch.full((S, T), float("nan"), device=dv)
        self.fp, self.clk = f(S, T), f(S, T)
        self.agr = torch.full((S, T), -1, device=dv, dtype=torch.int8)
        self.lastp = torch.full((S,), float("nan"), device=dv)
        self.hasv, self.cclk = b(S), f(S)
        self.front = torch.full((S,), float("-inf"), device=dv)
        self.run, self.rc0, self.rj0 = lg(S), f(S), lg(S)
        self.pcnt = torch.zeros(S, NHD, self.NB, device=dv, dtype=torch.int16)
        self.pfc = f(S, NHD, self.NB)
        self.parked = b(S, NHD)
        self.dis, self.rel, self.ndis = lg(S), lg(S), lg(S)
        self.ff = torch.full((S, 3), -1, device=dv, dtype=torch.long)       # first token of each rule's fire
        self.r1on = torch.full((S,), -1, device=dv, dtype=torch.long)      # R1 onset (run start) of the first R1
        self.fire = lg(S)                                                   # action code 1/2/3 (act rows), 0 none
        self.trk, self.act = b(S), b(S)
        self.temp = f(S)
        self.seed = lg(S)
        self.back_clk = torch.full((S,), float("inf"), device=dv)
        self.back_tok = torch.full((S,), 1 << 30, device=dv, dtype=torch.long)
        self.arT = torch.arange(T, device=dv)
        self.ar24 = torch.arange(NHD, device=dv)
        self.off9 = torch.arange(-4, 5, device=dv)
        self.arS = torch.arange(S, device=dv)

    def nbytes(self):
        return sum(getattr(self, n).numel() * getattr(self, n).element_size() for n in self.BUF)

    # ------------------------------------------------------------------ admission
    def reset(self, rows, track=True, act=False, rel0=0):
        """New windows in `rows` (LongTensor)."""
        self.rp[rows] = float("nan")
        self.fp[rows] = 0.0
        self.clk[rows] = 0.0
        self.agr[rows] = -1
        self.lastp[rows] = float("nan")
        self.hasv[rows] = False
        self.cclk[rows] = 0.0
        self.front[rows] = float("-inf")
        self.run[rows] = 0
        self.rc0[rows] = 0.0
        self.rj0[rows] = 0
        self.pcnt[rows] = 0
        self.pfc[rows] = 0.0
        self.parked[rows] = False
        self.dis[rows] = 0
        self.rel[rows] = int(rel0)
        self.ndis[rows] = 0
        self.ff[rows] = -1
        self.r1on[rows] = -1
        self.fire[rows] = 0
        self.trk[rows] = bool(track)
        self.act[rows] = bool(act) and bool(track)
        self.temp[rows] = 0.0
        self.seed[rows] = 0
        self.back_clk[rows] = float("inf")
        self.back_tok[rows] = 1 << 30

    def first_token(self, rows, a, v, enc_len):
        """Token 0 of freshly admitted rows (index tensor `rows`; reset() first): a [m, 24] argmax frames of the
        prompt's last query, v [m] stamp value or NaN, enc_len [m]. No rule can fire on one token."""
        m = len(rows)
        live = self.trk[rows]
        a = a.long()
        valid = a < enc_len.long()[:, None] - 3
        kk = valid.sum(1)
        srt = torch.sort(torch.where(valid, a.float(), float("inf")), dim=1).values
        lo = srt.gather(1, ((kk - 1).clamp(min=0) // 2)[:, None]).squeeze(1)
        hi = srt.gather(1, (kk // 2).clamp(max=NHD - 1)[:, None]).squeeze(1)
        med = ((lo + hi) / 2) * FR
        hasraw = (kk > 0) & live
        self.att[rows, 0] = a.to(torch.int16)
        self.rp[rows, 0] = torch.where(hasraw, med, torch.full_like(med, float("nan")))
        self.hasv[rows] = hasraw
        self.lastp[rows] = torch.where(hasraw, med, torch.full_like(med, float("nan")))
        is_ts = live & ~torch.isnan(v)
        c = torch.where(is_ts, torch.nan_to_num(v, nan=0.0).clamp(min=0.0), torch.zeros_like(v))
        self.cclk[rows] = c
        self.clk[rows, 0] = c
        pv = live[:, None] & (a > 2) & valid
        lin = ((rows[:, None] * NHD + self.ar24[None, :]) * self.NB + (a // 2).clamp(max=self.NB - 1)).reshape(-1)
        self.pcnt.view(-1)[lin] = pv.to(torch.int16).reshape(-1)
        self.pfc.view(-1)[lin] = torch.where(pv, c[:, None], torch.zeros_like(c)[:, None]).reshape(-1)
        agree = hasraw & (((v - med).abs() <= self.tau) if self.tau is not None else True)
        ok = is_ts & agree
        bad = is_ts & ~ok
        self.agr[rows, 0] = torch.where(is_ts, ok.to(torch.int8), torch.full_like(ok, -1, dtype=torch.int8))
        self.dis[rows] = bad.long()
        self.ndis[rows] = bad.long()
        self.rel[rows] = torch.where(ok, torch.ones_like(self.rel[rows]), self.rel[rows])

    # ------------------------------------------------------------------ one token per row
    def update(self, n, a, v, gi, live, end, enc_len):
        """Rows [0, n) produced one token each. a [n, 24] argmax frames, v [n] stamp value (float32, NaN for
        non-stamp tokens), gi [n] write index (token index; T - 1 for finished rows), live [n] rows that produced a
        token, end [n] rows whose token ends the window (EOS / stamp EOS / cap), enc_len [>= n] valid encoder
        frames. Returns fire [n] bool: act rows that fired now (the caller stops them for the host's rollback)."""
        T, D = self.T, self.D
        live = live & self.trk[:n]
        a = a.long()
        el = enc_len[:n].long()[:, None]
        valid = a < el - 3                                     # end sink excluded (robust_pos)
        kk = valid.sum(1)
        srt = torch.sort(torch.where(valid, a.float(), float("inf")), dim=1).values
        lo = srt.gather(1, ((kk - 1).clamp(min=0) // 2)[:, None]).squeeze(1)
        hi = srt.gather(1, (kk // 2).clamp(max=NHD - 1)[:, None]).squeeze(1)
        med = ((lo + hi) / 2) * FR                             # numpy nanmedian * FR (float32)
        hasraw = kk > 0
        hasv = self.hasv[:n]
        first = live & hasraw & ~hasv
        p = torch.where(hasraw, med, self.lastp[:n])
        self.att[:n].scatter_(1, gi[:, None, None].expand(n, 1, NHD), a.to(torch.int16)[:, None, :])
        rp, fp, clk = self.rp[:n], self.fp[:n], self.clk[:n]
        rp.scatter_(1, gi[:, None], p[:, None])
        # leading all-sink tokens take the first valid position (robust_pos back-fill); their filtered values too
        arT = self.arT[None, :]
        rp.copy_(torch.where(first[:, None] & (arT < gi[:, None]), med[:, None], rp))
        fp.copy_(torch.where(first[:, None] & (arT <= (gi - 4)[:, None]), med[:, None], fp))
        front = torch.where(first, med, self.front[:n])
        run = torch.where(first, torch.zeros_like(self.run[:n]), self.run[:n])
        self.hasv[:n] = hasv | (live & hasraw)
        self.lastp[:n] = torch.where(live & hasraw, med, self.lastp[:n])
        # written clock
        is_ts = live & ~torch.isnan(v)
        cclk = torch.where(is_ts, torch.maximum(self.cclk[:n], torch.nan_to_num(v, nan=0.0)), self.cclk[:n])
        self.cclk[:n] = cclk
        clk.scatter_(1, gi[:, None], cclk[:, None])
        # R1 rewind on the filtered position at i = g - 4
        i = gi - 4
        fin = live & hasv & ~first & (i >= 0)
        ic = i.clamp(min=0)
        idx = (ic[:, None] + self.off9[None, :]).clamp(min=0)
        f9 = rp.gather(1, idx).sort(1).values[:, 4]
        fp.scatter_(1, torch.where(fin, ic, T - 1)[:, None], f9[:, None])
        front = torch.where(fin, torch.maximum(front, f9), front)
        rr = fin & (f9 < front - D)
        ci = clk.gather(1, ic[:, None]).squeeze(1)
        start = rr & (run == 0)
        self.rc0[:n] = torch.where(start, ci, self.rc0[:n])
        self.rj0[:n] = torch.where(start, ic, self.rj0[:n])
        run = torch.where(fin, torch.where(rr, run + 1, torch.zeros_like(run)), run)
        r1 = rr & (run >= self.L) & (ci - self.rc0[:n] >= self.adv)
        self.front[:n] = front
        self.run[:n] = run
        # R2 collapse (parked heads)
        pv = live[:, None] & (a > 2) & valid
        lin = ((self.arS[:n, None] * NHD + self.ar24[None, :]) * self.NB + (a // 2).clamp(max=self.NB - 1)).reshape(-1)
        pc, pf = self.pcnt.view(-1), self.pfc.view(-1)
        old = pc[lin].view(n, NHD).int()
        cnt = old + pv.int()
        fcl = torch.where(pv & (cnt == 1), cclk[:, None], pf[lin].view(n, NHD))
        pc[lin] = cnt.to(torch.int16).reshape(-1)
        pf[lin] = fcl.reshape(-1)
        pk = pv & (cnt >= self.pn) & (cclk[:, None] - fcl >= self.ps)
        parked = self.parked[:n] | pk
        self.parked[:n] = parked
        r2 = live & (parked.sum(1) >= self.hn)
        # R3 stamp / attention agreement; reliable point
        mono = hasraw & (med >= front - D)
        if self.tau is not None:
            agree = hasraw & ((v - med).abs() <= self.tau)
        else:
            agree = hasraw
        ok = is_ts & agree & mono                   # reliable stamp: agrees and monotone
        bad = is_ts & ~agree                        # R3 counts disagreeing stamps only
        self.agr[:n].scatter_(1, gi[:, None], torch.where(is_ts, ok.to(torch.int8), torch.full_like(ok, -1, dtype=torch.int8))[:, None])
        dis = torch.where(is_ts & agree, torch.zeros_like(self.dis[:n]), torch.where(bad, self.dis[:n] + 1, self.dis[:n]))
        self.dis[:n] = dis
        self.ndis[:n] += bad.long()
        self.rel[:n] = torch.where(ok, gi + 1, self.rel[:n])
        r3 = bad & (dis >= self.k3) if (self.tau is not None and self.k3 is not None) else torch.zeros_like(bad)
        ff = self.ff[:n]
        for j, c in enumerate((r1, r2, r3)):
            ff[:, j] = torch.where(c & (ff[:, j] < 0), gi, ff[:, j])
        self.r1on[:n] = torch.where(r1 & (self.r1on[:n] < 0), self.rj0[:n], self.r1on[:n])
        # action: stop act rows on their first new fire (the host rolls them back)
        anyf = (r1 | r2 | r3) & self.act[:n] & ~end & (self.fire[:n] == 0)
        code = torch.where(r1, 1, torch.where(r2, 2, 3))
        self.fire[:n] = torch.where(anyf, code, self.fire[:n])
        # back to greedy once the clock passed the failure point by the margin
        back = live & (self.temp[:n] > 0) & (cclk >= self.back_clk[:n]) & (gi + 1 >= self.back_tok[:n])
        self.temp[:n] = torch.where(back, torch.zeros_like(self.temp[:n]), self.temp[:n])
        return anyf

    # ------------------------------------------------------------------ host side (rollback / restore)
    def fetch(self, s, n):
        """Host copies of slot s's per-token arrays [0, n)."""
        return dict(att=self.att[s, :n].cpu().numpy(), rp=self.rp[s, :n].cpu().numpy(),
                    fp=self.fp[s, :n].cpu().numpy(), clk=self.clk[s, :n].cpu().numpy(),
                    agr=self.agr[s, :n].cpu().numpy())

    def rebuild(self, s, r, h, el, rel0=0):
        """Restore slot s's rule state to the prefix [0, r) from host arrays h (fetch() of a longer attempt whose
        first r tokens are that prefix). The device per-token arrays beyond r are cleared."""
        T, D = self.T, self.D
        att = h["att"][:r].astype(np.int64)
        rp = h["rp"][:r].copy()
        fp = h["fp"][:r].copy()
        clk = h["clk"][:r].copy()
        agr = h["agr"][:r].copy()
        hasv = bool(np.isfinite(rp).any())
        if not hasv:
            rp[:] = np.nan
        fin = max(0, r - 4) if hasv else 0
        f_ = fp[:fin]
        front = float(f_.max()) if fin else float("-inf")
        run, rj0 = 0, 0
        if fin:
            rr = f_ < np.maximum.accumulate(f_) - D
            nz = np.flatnonzero(~rr)
            run = fin - (nz[-1] + 1 if len(nz) else 0)
            rj0 = fin - run
        rc0 = float(clk[rj0]) if run else 0.0
        lastp = float(rp[np.isfinite(rp)][-1]) if hasv else float("nan")
        cclk = float(clk[-1]) if r else 0.0
        pcnt = np.zeros((NHD, self.NB), np.int64)
        first_c = np.zeros((NHD, self.NB), np.float32)
        last_c = np.zeros((NHD, self.NB), np.float32)
        for hh in range(NHD):
            f = att[:, hh]
            ok = (f > 2) & (f < el - 3)
            bi = f[ok] // 2
            cl = clk[ok]
            np.add.at(pcnt[hh], bi, 1)
            seen = np.zeros(self.NB, bool)
            for b_, c_ in zip(bi, cl):
                if not seen[b_]:
                    first_c[hh, b_] = c_
                    seen[b_] = True
                last_c[hh, b_] = c_
        parked = ((pcnt >= self.pn) & (last_c - first_c >= self.ps)).any(1)
        st = np.flatnonzero(agr >= 0)
        dis = 0
        for j in st[::-1]:
            if agr[j] == 1:
                break
            dis += 1
        good = st[agr[st] == 1]
        rel = int(good[-1] + 1) if len(good) else int(rel0)
        dv = self.dev
        t = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt).to(dv)   # noqa: E731
        self.att[s] = 0
        self.att[s, :r] = t(att, torch.int16)
        self.rp[s] = float("nan")
        self.rp[s, :r] = t(rp)
        self.fp[s] = 0.0
        self.fp[s, :r] = t(fp)
        self.clk[s] = 0.0
        self.clk[s, :r] = t(clk)
        self.agr[s] = -1
        self.agr[s, :r] = t(agr, torch.int8)
        self.lastp[s] = lastp
        self.hasv[s] = hasv
        self.cclk[s] = cclk
        self.front[s] = front
        self.run[s] = run
        self.rc0[s] = rc0
        self.rj0[s] = rj0
        self.pcnt[s] = t(np.minimum(pcnt, 32767), torch.int16)
        self.pfc[s] = t(first_c)
        self.parked[s] = t(parked, torch.bool)
        self.dis[s] = dis
        self.rel[s] = rel
        self.ndis[s] = 0
        ffh = self.ff[s].cpu().numpy()
        ffh = np.where(ffh < r, ffh, -1)
        self.ff[s] = t(ffh, torch.long)
        if ffh[0] < 0:
            self.r1on[s] = -1
        self.fire[s] = 0

    def compact(self, idx, m):
        for nme in self.BUF:
            t = getattr(self, nme)
            t[:m] = t[idx]
        self.trk[m:] = False
        self.act[m:] = False


def collapse_onset(att, clk, el, heads_needed=14, park_n=30, park_span=2.0):
    """Token by which `heads_needed` heads had begun the parking run that parks them (first visit of the parked
    bin), or None."""
    att = np.asarray(att, np.int64)
    starts = []
    for hh in range(att.shape[1]):
        f = att[:, hh]
        ok = np.flatnonzero((f > 2) & (f < el - 3))
        bins = {}
        for j in ok:                      # the bin that parks the head first; its first visit
            lst = bins.setdefault(f[j] // 2, [])
            lst.append(j)
            if len(lst) >= park_n and clk[j] - clk[lst[0]] >= park_span:
                starts.append(lst[0])
                break
    if len(starts) < heads_needed:
        return None
    return int(sorted(starts)[heads_needed - 1])
