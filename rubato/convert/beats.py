"""Beats and downbeats as CSV.

* beats mode: ``|<|1.09|> *<|1.69|>`` -> one row per stamp (``|`` downbeat, ``*`` beat).
* score mode: beats DERIVED from the bar lines and stamps:
  - a downbeat at every stamped bar line;
  - inside a bar whose start and the next bar line are stamped, the grid
    ``bar start + k * beat`` (k >= 1; beat = 3/den when num > 3 and num % 3 == 0,
    else 1/den; no point closer than half a beat to the next bar line).
    A grid point with an event (onset or release) exactly on it takes that
    event's stamp (``source = stamp``); any other grid point lies inside a held note
    or rest and gets the time interpolated linearly between the nearest stamped
    positions of the bar (``source = interpolated``).
  The stamped points are the beats the score itself places in time; the
  interpolated ones are an estimate.

CSV columns: ``time,is_downbeat,source`` (time in seconds, is_downbeat 0/1,
source ``stamp`` / ``interpolated``; beats mode rows are all ``stamp``).
"""
from bisect import bisect_left, bisect_right
from fractions import Fraction
from math import gcd

from ._perf import parse_beats


def beat_length(num, den):
    return Fraction(3, den) if (num > 3 and num % 3 == 0) else Fraction(1, den)


def text_beats(text):
    """beats-mode text -> [(seconds, is_downbeat, 'stamp')]."""
    return [(t, d, 'stamp') for t, d in parse_beats(text)]


def score_beats(score, interpolated=True):
    """read score -> [(seconds, is_downbeat, source)] sorted by time."""
    # integer ticks (1/unit whole note) for the stamp times and the beat grid
    unit = 1
    try:
        dens = {k.denominator for k in score.stamps}
        for b in score.bars:
            dens.add(b.start.denominator)
            dens.add(beat_length(b.ts_top, b.ts_bot).denominator)
    except AttributeError:            # times without numerator/denominator
        return _score_beats_generic(score, interpolated)
    for d in dens:
        if unit % d:
            unit = unit * d // gcd(unit, d)
    st = {k.numerator * (unit // k.denominator): v for k, v in score.stamps.items()}
    keys = sorted(st)
    bars = []
    seen = set()
    for b in score.bars:
        s0 = b.start.numerator * (unit // b.start.denominator)
        if s0 in st and s0 not in seen:
            seen.add(s0)
            bars.append((s0, b))
    out = [(st[s0], True, 'stamp') for s0, b in bars]
    for (a0, a), (c0, c) in zip(bars, bars[1:]):
        bl = beat_length(a.ts_top, a.ts_bot)
        bl = bl.numerator * (unit // bl.denominator)
        inner = keys[bisect_right(keys, a0):bisect_left(keys, c0)]
        k = 1
        while True:
            pos = a0 + bl * k
            if (c0 - pos) * 2 < bl:
                break
            k += 1
            if pos in st:
                out.append((st[pos], False, 'stamp'))
            elif interpolated:
                lo_t, lo_s, hi_t, hi_s = a0, st[a0], c0, st[c0]
                j = bisect_left(inner, pos)          # inner[j-1] < pos < inner[j] (pos is unstamped)
                if j > 0:
                    lo_t, lo_s = inner[j - 1], st[inner[j - 1]]
                if j < len(inner):
                    hi_t, hi_s = inner[j], st[inner[j]]
                frac = (pos - lo_t) / (hi_t - lo_t) if hi_t != lo_t else 0.0
                out.append((lo_s + frac * (hi_s - lo_s), False, 'interpolated'))
    out.sort(key=lambda r: (r[0], not r[1]))
    return out


def _score_beats_generic(score, interpolated=True):
    """score_beats for times that are not rationals (e.g. floats)."""
    st = score.stamps
    keys = sorted(st)
    bars = []
    seen = set()
    for b in score.bars:
        if b.start in st and b.start not in seen:
            seen.add(b.start)
            bars.append(b)
    out = [(st[b.start], True, 'stamp') for b in bars]
    for a, c in zip(bars, bars[1:]):
        bl = beat_length(a.ts_top, a.ts_bot)
        inner = keys[bisect_right(keys, a.start):bisect_left(keys, c.start)]
        k = 1
        while True:
            pos = a.start + bl * k
            if (c.start - pos) * 2 < bl:
                break
            k += 1
            if pos in st:
                out.append((st[pos], False, 'stamp'))
            elif interpolated:
                lo_t, lo_s, hi_t, hi_s = a.start, st[a.start], c.start, st[c.start]
                j = bisect_left(inner, pos)          # inner[j-1] < pos < inner[j] (pos is unstamped)
                if j > 0:
                    lo_t, lo_s = inner[j - 1], st[inner[j - 1]]
                if j < len(inner):
                    hi_t, hi_s = inner[j], st[inner[j]]
                frac = float((pos - lo_t) / (hi_t - lo_t)) if hi_t != lo_t else 0.0
                out.append((lo_s + frac * (hi_s - lo_s), False, 'interpolated'))
    out.sort(key=lambda r: (r[0], not r[1]))
    return out


def to_csv(rows):
    lines = ['time,is_downbeat,source']
    for t, d, src in rows:
        lines.append(f'{t:.3f},{int(bool(d))},{src}')
    return '\n'.join(lines) + '\n'
