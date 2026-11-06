"""Polyphase resampling equal to scipy.signal.resample_poly(x, up, down) with its defaults
(window ('kaiser', 5.0), padtype 'constant'); scipy is used when importable, else a vectorised numpy port.

FIR: h = firwin(2 * half_len + 1, 1 / max(up, down), window kaiser 5.0) * up with half_len = 10 * max(up, down),
applied as upfirdn(h, x, up, down) with scipy's output alignment, in float64 (scipy's result type for float32
input), returned as float32.

Other resamplers, such as librosa's soxr_hq, use different anti-alias filters.
Those differences can change transcripts for non-16 kHz input. Feed 16 kHz audio
to bypass resampling entirely.
"""

from math import gcd

import numpy as np


def _firwin_kaiser(numtaps, cutoff, beta=5.0):
    # scipy.signal.firwin(numtaps, cutoff, window=('kaiser', beta)), cutoff relative to Nyquist, DC gain 1
    m = np.arange(numtaps) - (numtaps - 1) / 2.0
    h = cutoff * np.sinc(cutoff * m) * np.kaiser(numtaps, beta)
    return h / h.sum()


def resample_poly_np(x, up, down, block=1 << 16, work_dtype=np.float64):
    """numpy-only scipy.signal.resample_poly(x, up, down) for 1-D x (computed in work_dtype)."""
    x = np.asarray(x, dtype=np.float64)
    g = gcd(int(up), int(down))
    up, down = int(up) // g, int(down) // g
    if up == down == 1:
        return x.copy()
    max_rate = max(up, down)
    half_len = 10 * max_rate
    h = _firwin_kaiser(2 * half_len + 1, 1.0 / max_rate) * up
    n_in = len(x)
    n_out = n_in * up // down + bool(n_in * up % down)
    n_pre_pad = down - half_len % down
    n_post_pad = 0
    n_pre_remove = (half_len + n_pre_pad) // down
    while (len(h) + n_pre_pad + n_post_pad + (n_in - 1) * up) // down + 1 < n_out + n_pre_remove:
        n_post_pad += 1
    h = np.concatenate((np.zeros(n_pre_pad), h, np.zeros(n_post_pad)))
    # polyphase table: phase p holds taps h[p + k*up], k = 0..K-1 (zero-filled)
    K = -(-len(h) // up)
    hp = np.zeros((up, K))
    for p in range(up):
        taps = h[p::up]
        hp[p, :len(taps)] = taps
    # output j (of the full upfirdn output) = sum_k hp[p_j, k] * x[b_j - k], p_j = j*down % up, b_j = (j*down - p_j)/up
    # x is zero-padded on the left by K-1 so that b_j - k >= -(K-1) indexes a zero
    hp = hp.astype(work_dtype)
    xp = np.concatenate((np.zeros(K - 1), x, np.zeros(1))).astype(work_dtype)
    y = np.empty(n_out, dtype=work_dtype)
    kk = np.arange(K)
    for s in range(0, n_out, block):
        j = np.arange(n_pre_remove + s, n_pre_remove + min(s + block, n_out), dtype=np.int64)
        jd = j * down
        p = jd % up
        b = (jd - p) // up
        idx = (b[:, None] - kk[None, :]) + (K - 1)
        np.clip(idx, 0, len(xp) - 1, out=idx)
        # indices past the signal end read zeros
        idx[idx >= K - 1 + n_in] = len(xp) - 1
        y[s:s + len(j)] = np.einsum('ij,ij->i', hp[p], xp[idx])
    return y


def resample_poly(x, up, down):
    try:
        from scipy.signal import resample_poly as sp_resample_poly
    except Exception:
        return resample_poly_np(x, up, down)
    return sp_resample_poly(x, up, down)


def resample(x, orig_sr, sr):
    """float32 mono x from orig_sr to sr (polyphase)."""
    orig_sr, sr = int(orig_sr), int(sr)
    if orig_sr == sr:
        return np.asarray(x, dtype=np.float32)
    g = gcd(sr, orig_sr)
    return np.asarray(resample_poly(x, sr // g, orig_sr // g), dtype=np.float32)
