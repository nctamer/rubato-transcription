"""Notation choices for writing a ``Score`` as sheet music.

Engraving rules for Rubato transcriptions: every event is split at beats into tied, spellable
values; beams group beamable values inside one beat; tuplet brackets follow the
duration families; accidentals follow the per-bar, per-octave rule with the
tie, grace-note, and unison handling; a grace note is written at twice the
value of the note it precedes (at most a 16th); a tie chain running over four or
more bars keeps only its first note; the rests of secondary voices are hidden.
"""
from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from typing import List

from ._model import Event, Layer, Note

F = Fraction

_TYPES = [(F(4), 'longa'), (F(2), 'breve'), (F(1), 'whole'), (F(1, 2), 'half'), (F(1, 4), 'quarter'),
          (F(1, 8), 'eighth'), (F(1, 16), '16th'), (F(1, 32), '32nd'), (F(1, 64), '64th'),
          (F(1, 128), '128th'), (F(1, 256), '256th')]
_TYPE_VALUE = {name: base for base, name in _TYPES}


def _pow2(r):
    if r <= 0:
        return False
    a, b = r.numerator, r.denominator
    return a & (a - 1) == 0 and b & (b - 1) == 0


def _plain(d):
    for base, name in _TYPES:
        total, add = base, base
        for k in range(4):
            if d == total:
                return name, k
            add = add / 2
            total = total + add
    return None


def tuplet_family(d):
    x = d.denominator
    while x > 0 and x % 2 == 0:
        x //= 2
    return x


@dataclass
class Spelling:
    type: str = 'quarter'
    dots: int = 0
    actual: int = 1
    normal: int = 1
    exact: bool = True


def spell(d, hint_actual=0, hint_normal=0):
    """Written value of a duration: plain type+dots, else the conventional tuplet."""
    s = Spelling()
    if d <= 0:
        return s
    if hint_actual > hint_normal > 0:
        h = _plain(d * F(hint_actual, hint_normal))
        if h and h[1] <= 2:
            return Spelling(h[0], h[1], hint_actual, hint_normal)
    p = _plain(d)
    if p:
        return Spelling(p[0], p[1])
    odd = tuplet_family(d)
    for k in range(3):
        if odd <= 1:
            break
        actual = odd << k
        normal = 1
        while normal * 2 < actual:
            normal *= 2
        t = _plain(d * F(actual, normal))
        if t and t[1] <= 2:
            return Spelling(t[0], t[1], actual, normal)
    best = None
    for base, name in _TYPES:
        diff = abs(float(base) - float(d))
        if best is None or diff < best[0]:
            best = (diff, name)
    return Spelling(best[1], 0, 1, 1, False)


def flags(type_name):
    v = _TYPE_VALUE.get(type_name, F(1, 4))
    n = 0
    while v < F(1, 4):
        v *= 2
        n += 1
    return n


def _recip(d):
    if d.numerator == 1:
        return d.denominator
    if d.numerator == 3 and d.denominator % 2 == 0:
        return d.denominator // 2
    if d.numerator == 7 and d.denominator % 4 == 0:
        return d.denominator // 4
    if d >= 2:
        return 0
    return d.denominator // max(1, d.numerator)


# ── beat splitting ─────────────────────────────────────────────────────────

def _simple(d):
    a, b = d.numerator, d.denominator
    if a <= 0:
        return True
    if d in (2, 4, 8, 3, 6, 12):
        return True
    if a == 1:
        return True
    if a == 3 and b % 2 == 0:
        return True
    if a == 7 and b % 4 == 0:
        return _pow2(F(b // 4))
    if a == 15 and b % 8 == 0:
        return _pow2(F(b // 8))
    t = d * F(8, 15)
    if t.numerator == 1:
        return _pow2(F(t.denominator))
    t = d * F(4, 7)
    if t.numerator == 1:
        return _pow2(F(t.denominator))
    t = d * F(2, 3)
    if t.numerator == 1:
        return True
    return False


_GREEDY = [F(2), F(3, 2), F(1), F(7, 8), F(3, 4), F(1, 2), F(7, 16), F(3, 8), F(1, 4), F(7, 32), F(3, 16),
           F(1, 8), F(7, 64), F(3, 32), F(1, 16), F(7, 128), F(3, 64), F(1, 32), F(3, 128), F(1, 64),
           F(1, 128), F(1, 256)]


def _trunc(r):
    return r.numerator // r.denominator if r >= 0 else -((-r.numerator) // r.denominator)


def beat_segments(offset, dur, beat):
    seg = []
    start_q = offset / beat
    first_boundary = beat * (_trunc(start_q) + 1)
    if (start_q.denominator == 1 and _simple(dur)) or first_boundary >= offset + dur:
        seg.append(dur)
    else:
        rem, pos = dur, offset
        for _ in range(64):
            if rem <= 0:
                break
            best = F(0)
            for d in _GREEDY:
                if d > rem:
                    continue
                q = (pos + d) / beat
                if q.denominator == 1 or d == rem:
                    best = d
                    break
            if best <= 0:
                seg.append(rem)
                break
            seg.append(best)
            pos += best
            rem -= best
    out = []
    for sg in seg:
        if _simple(sg):
            out.append(sg)
            continue
        rem = sg
        for _ in range(64):
            if rem <= 0:
                break
            unit = F(1, rem.denominator)
            if rem.numerator > 10000:
                out.append(rem)
                break
            best = F(0)
            for k in range(rem.numerator, 0, -1):
                if _simple(unit * k):
                    best = unit * k
                    break
            if best <= 0:
                out.append(rem)
                break
            out.append(best)
            rem -= best
    return out


# MuseScore's rest splitting (port), used for rests before the beat split.
class _TS:
    def __init__(self, num, den):
        self.num, self.den = num, den
        self.compound = num > 3 and num % 3 == 0
        bpm = num // (3 if self.compound else 1)
        self.triple = bpm % 3 == 0
        self.duple = bpm % 2 == 0
        self.dunit = F(1, den)
        self.beat = self.dunit * (3 if self.compound else 1)
        self.measure = F(num, den)


DOWNBEAT, C_STRESSED, S_STRESSED, C_UNSTRESSED, S_UNSTRESSED, C_SUBBEAT, SUBBEAT = range(7)


def _mult(pos, unit):
    return unit != 0 and (pos / unit).denominator == 1


def _mod(pos, unit):
    if unit == 0:
        return F(0)
    return pos - _trunc(pos / unit) * unit


def _beat_type(pos, ts):
    if pos == 0:
        return DOWNBEAT
    if not _mult(pos, ts.dunit):
        return SUBBEAT
    if ts.compound and not _mult(pos, ts.beat):
        return C_SUBBEAT
    beat_num = _trunc(pos / ts.beat)
    if ts.triple:
        stress = 3
    elif ts.duple:
        stress = 2
    else:
        stress = (ts.num // (3 if ts.compound else 1) + 1) // 2
    if stress > 0 and beat_num % stress == 0:
        return C_STRESSED if ts.compound else S_STRESSED
    return C_UNSTRESSED if ts.compound else S_UNSTRESSED


def _subbeat_level(pos, ts):
    level, sub = 0, ts.dunit
    while True:
        if _mult(pos, sub):
            return level
        level += 1
        if level > 16:
            return -level
        sub = sub / 2


def _strongest_beat(start, end, ts, save_last):
    strongest, best_pos = SUBBEAT, None
    rem = _mod(start, ts.dunit)
    pos = start + (ts.dunit if rem == 0 else ts.dunit - rem)
    count = 0
    while pos < end:
        count += 1
        bt = _beat_type(pos, ts)
        if (bt <= strongest) if save_last else (bt < strongest):
            strongest, best_pos = bt, pos
        pos += ts.dunit
    return strongest, count, best_pos


def _strongest_level(start, end, ts):
    sub, level = ts.dunit, 0
    while True:
        qs = start / sub
        n = qs.numerator // qs.denominator
        qe = end / sub
        m = qe.numerator - 1 if qe.denominator == 1 else qe.numerator // qe.denominator
        if m > n:
            return level, sub * m
        level += 1
        if level > 16:
            return -level, None
        sub = sub / 2


def _force_simple(is_rest, sb, eb, crossed, strongest):
    if strongest == S_STRESSED:
        if is_rest:
            return True
        if sb <= S_STRESSED or eb <= S_STRESSED:
            return False
        if sb <= S_UNSTRESSED and eb <= S_UNSTRESSED:
            return False
        return True
    if strongest == S_UNSTRESSED:
        if sb <= S_STRESSED and eb <= S_STRESSED:
            return False
        if sb == SUBBEAT or eb == SUBBEAT:
            return is_rest or crossed > 1
        return False
    return False


def _force(is_rest, sb, eb, crossed, strongest, ts):
    if strongest <= S_STRESSED and not ts.triple and not ts.duple:
        return True
    if is_rest:
        if strongest <= S_UNSTRESSED and ts.num == 2:
            return True
        if strongest <= S_UNSTRESSED and ts.triple:
            return True
    if ts.compound:
        if strongest == C_STRESSED:
            return _force_simple(is_rest, sb, eb, crossed // 3, S_STRESSED)
        if strongest == C_UNSTRESSED:
            return False
        if strongest == C_SUBBEAT:
            if sb <= C_UNSTRESSED and eb <= C_UNSTRESSED:
                return False
            if is_rest and sb > C_UNSTRESSED:
                return True
            return _force_simple(is_rest, sb, eb, crossed, S_UNSTRESSED)
        return _force_simple(is_rest, sb, eb, crossed, strongest)
    return _force_simple(is_rest, sb, eb, crossed, strongest)


def _fits(d, max_dots=4):
    if d <= 0:
        return False
    for base in (F(2), F(1), F(1, 2), F(1, 4), F(1, 8), F(1, 16), F(1, 32), F(1, 64), F(1, 128), F(1, 256)):
        if base == d:
            return True
        dotted, add = base, base / 2
        for _ in range(max_dots):
            dotted += add
            if dotted == d:
                return True
            add = add / 2
    return False


def _populate(out, l, is_rest, start, ts):
    end = start + l
    split = F(0)
    sl = _subbeat_level(start, ts)
    el = _subbeat_level(end, ts)
    strongest, pos = _strongest_level(start, end, ts)
    if pos is not None:
        split = pos
    if sl < 0 or el < 0 or strongest < 0:
        out.append(l)
        return
    need = strongest < sl and strongest < el
    if sl == el and strongest == sl - 1:
        need = False
    if sl == el and strongest == sl - 2:
        s = ts.dunit / (2 ** (sl - 1))
        rem = _mod(start, s)
        to_next = s if rem == 0 else s - rem
        need = to_next != rem
    if not need and strongest == 0:
        sb, eb = _beat_type(start, ts), _beat_type(end, ts)
        st, crossed, bpos = _strongest_beat(start, end, ts, sb <= S_UNSTRESSED)
        if bpos is not None:
            split = bpos
        need = _force(is_rest, sb, eb, crossed, st, ts)
    if not need and _fits(l):
        out.append(l)
        return
    if not (start < split < end):
        out.append(l)
        return
    left = split - start
    _populate(out, left, is_rest, start, ts)
    _populate(out, l - left, is_rest, split, ts)


def _compound_split(out, l, is_rest, start, ts):
    end = start + l
    sb, eb = _beat_type(start, ts), _beat_type(end, ts)
    if sb > C_UNSTRESSED:
        rem = _mod(start, ts.beat)
        split_len = ts.beat if rem == 0 else ts.beat - rem
        if end - start > split_len:
            _populate(out, split_len, is_rest, start, ts)
            _compound_split(out, l - split_len, is_rest, start + split_len, ts)
            return
    if eb > C_UNSTRESSED:
        split_len = _mod(end, ts.beat)
        if end - start > split_len:
            _populate(out, l - split_len, is_rest, start, ts)
            _populate(out, split_len, is_rest, end - split_len, ts)
            return
    _populate(out, l, is_rest, start, ts)


def rhythmic_split(dur, is_rest, offset, num, den):
    if dur <= 0:
        return []
    ts = _TS(num, den)
    if is_rest and offset == 0 and dur == ts.measure:
        return [dur]
    out = []
    if ts.compound:
        _compound_split(out, dur, is_rest, offset, ts)
    else:
        _populate(out, dur, is_rest, offset, ts)
    return out or [dur]


# ── per-duration facts (cached) ───────────────────────────────────────────────

_CACHE_MAX = 200000
_DUR = {}             # (num, den) -> (Spelling, recip, family, pow2 without dots, beam count)
_GRACE_SPELL = {}     # value denominator -> Spelling
_SEG = {}             # split key -> tuple of segment durations


def _dur_info(d):
    key = (d.numerator, d.denominator)
    r = _DUR.get(key)
    if r is None:
        sp = spell(d)
        r = (sp, _recip(d), _family(d), _pow2(_no_dots(d)), flags(sp.type))
        if len(_DUR) > _CACHE_MAX:
            _DUR.clear()
        _DUR[key] = r
    return r


def _grace_spelling(nb):
    den = min(nb * 2, 16)
    sp = _GRACE_SPELL.get(den)
    if sp is None:
        sp = _GRACE_SPELL[den] = spell(F(1, den))
    return sp


def _segments(is_rest, t_num, t_den, d, top, bot):
    """Beat split of a duration d at bar offset t_num/t_den (meter top/bot)."""
    key = (is_rest, t_num, t_den, d.numerator, d.denominator, top, bot)
    r = _SEG.get(key)
    if r is None:
        t = F(t_num, t_den)
        beat = F(3, bot) if (top > 3 and top % 3 == 0) else F(1, bot)
        if is_rest:
            seg = []
            a = t
            for part in rhythmic_split(d, True, t, top, bot):
                seg.extend(beat_segments(a, part, beat))
                a += part
        else:
            seg = beat_segments(t, d, beat)
        r = tuple(seg)
        if len(_SEG) > _CACHE_MAX:
            _SEG.clear()
        _SEG[key] = r
    return r


# ── engraved items ──────────────────────────────────────────────────────────

class Item:
    """One engraved event (a note, chord or rest) of a staff voice in a bar."""
    __slots__ = ('ev', 'voice', 'onset', 'spelling', 'visible', 'order', 'accidental', 'trill_acc',
                 'beams', 'tuplet', 'tuplet_actual', 'grace_seq', 'nbeams', 'info')

    def __init__(self, ev, voice, onset):
        self.ev = ev
        self.voice = voice            # compact voice number within the staff (0 = first)
        self.onset = onset            # bar-relative, in ticks of the staff bar
        self.spelling = None
        self.visible = True
        self.order = []               # note indices, low to high
        self.accidental = []
        self.trill_acc = []           # per note: None / '' / accidental name
        self.beams = []               # per level: start/continue/stop/partial
        self.tuplet = None            # start / stop / startStop / '' (inside) / None (not a tuplet)
        self.tuplet_actual = 0
        self.grace_seq = 0
        self.nbeams = 0
        self.info = None              # _dur_info of a non-grace event


_ACC = {1: 'sharp', -1: 'flat', 0: 'natural', 2: 'double-sharp', -2: 'flat-flat'}


def _beam_count(it):
    return it.nbeams


def _assign_beams(span):
    if len(span) < 2:
        return
    nb = [x.nbeams for x in span]
    span[0].beams.extend(['start'] * nb[0])
    for i in range(1, len(span)):
        typ = 'stop' if i + 1 == len(span) else 'continue'
        cur, prev = span[i].beams, span[i - 1].beams
        n, pn = nb[i], nb[i - 1]
        if 0 < n < pn:
            for lv in range(n, min(pn, len(prev))):
                prev[lv] = 'partial' if prev[lv] == 'start' else 'stop'
            cur.extend([typ] * n)
        elif 0 < pn < n:
            for k in range(n):
                cur.append(('partial' if typ == 'stop' else 'start') if k >= pn else typ)
        else:
            cur.extend([typ] * n)


def _beat_of(bar):
    return F(3, bar.ts_bot) if (bar.ts_top > 3 and bar.ts_top % 3 == 0) else F(1, bar.ts_bot)


def _family(d):
    den = d.denominator
    if d.numerator == 3:
        den = d.denominator // 2
    elif d.numerator == 7:
        den = d.denominator // 4
    if den <= 0 or den & (den - 1) == 0:
        return 2
    while den % 2 == 0:
        den //= 2
    return den


def _beams(voices, bar, D):
    """voices: items per voice; onsets in ticks of 1/D whole note."""
    beat = _beat_of(bar)
    bnum, bden = beat.numerator, beat.denominator
    div = D * bnum
    for evs in voices:
        group, grace = [], []
        gbeat, gfam = -1, 0
        for it in evs:
            ev = it.ev
            if ev.grace:
                if group:
                    _assign_beams(group)
                    group = []
                grace.append(it)
                continue
            if grace:
                if len(grace) >= 2:
                    for i, g in enumerate(grace):
                        t = 'start' if i == 0 else ('stop' if i + 1 == len(grace) else 'continue')
                        g.beams.extend([t] * g.nbeams)
                grace = []
            info = it.info
            if not ev.notes or info[1] <= 4:
                if group:
                    _assign_beams(group)
                    group = []
                continue
            b = (it.onset * bden) // div
            fam = info[2]
            if group and (b != gbeat or fam != gfam):
                _assign_beams(group)
                group = []
            if not group:
                gbeat, gfam = b, fam
            group.append(it)
        _assign_beams(group)
        if len(grace) >= 2:
            for i, g in enumerate(grace):
                t = 'start' if i == 0 else ('stop' if i + 1 == len(grace) else 'continue')
                g.beams.extend([t] * g.nbeams)


_RATIO = {}


def _ratio(nd0):
    """(actual, normal) of a tuplet member of duration nd0."""
    key = (nd0.numerator, nd0.denominator)
    r = _RATIO.get(key)
    if r is None:
        nd = nd0
        if nd.numerator == 3 and nd.denominator % 2 == 0:
            nd = F(1, nd.denominator // 2)
        up = F(1)
        while up / 2 >= nd:
            up = up / 2
        mult = nd / up
        r = _RATIO[key] = (mult.denominator, mult.numerator)
    return r


def _no_dots(d):
    if d.numerator == 3 and d.denominator % 2 == 0:
        return F(1, d.denominator // 2)
    if d.numerator == 7 and d.denominator % 4 == 0:
        return F(1, d.denominator // 4)
    return d


def _tuplets(voices):
    for v in voices:
        d = [it for it in v if not it.ev.grace and it.ev.duration.numerator > 0]
        if not d:
            continue
        p2 = [x.info[3] for x in d]
        if all(p2):
            continue
        n = len(d)
        beams = []
        i = 0
        while i < n:
            if d[i].beams and d[i].beams[0] == 'start':
                j = i
                while j < n and not (d[j].beams and d[j].beams[0] == 'stop'):
                    j += 1
                if j < n:
                    beams.append((i, j))
                    i = j + 1
                    continue
            i += 1
        group = [0] * n
        g = 0
        i = 0
        while i < n:
            if p2[i]:
                i += 1
                continue
            end = n - 1
            by_beam = False
            for bs, be in beams:
                if bs != i:
                    continue
                bd = sum((d[j].ev.duration for j in range(bs, be + 1)), F(0))
                dotted = _pow2(bd) or _pow2(bd * F(2, 3)) or _pow2(bd * F(4, 7))
                if dotted and all(not p2[j] for j in range(bs, be + 1)):
                    end, by_beam = be, True
                break
            if not by_beam:
                total = d[i].ev.duration
                end = n - 1
                for j in range(i + 1, n):
                    if p2[j]:
                        end = j - 1
                        break
                    total += d[j].ev.duration
                    if _pow2(total):
                        end = j
                        break
            g += 1
            for j in range(i, end + 1):
                group[j] = g
            i = end + 1
        actual, normal = [0] * n, [0] * n
        for i in range(n):
            if not group[i]:
                continue
            actual[i], normal[i] = _ratio(d[i].ev.duration)

        def in_beam(i, j):
            return any(bs <= i and j <= be for bs, be in beams)

        for i in range(n - 1):
            if not group[i] or not group[i + 1] or group[i] == group[i + 1]:
                continue
            if not in_beam(i, i + 1) or actual[i] != actual[i + 1] or normal[i] != normal[i + 1]:
                continue
            a, b = group[i], group[i + 1]
            na, nn = actual[i] + actual[i + 1], normal[i] + normal[i + 1]
            for j in range(n):
                if group[j] in (a, b):
                    group[j], actual[j], normal[j] = a, na, nn
        i = 0
        while i < n:
            if not group[i]:
                i += 1
                continue
            j = i
            while j < n and group[j] == group[i]:
                j += 1
            counts = {}
            total, maxd = F(0), F(0)
            for k in range(i, j):
                nd = _no_dots(d[k].ev.duration)
                counts[nd] = counts.get(nd, 0) + 1
                total += d[k].ev.duration
                maxd = max(maxd, nd)
            a0 = actual[i]
            scaled = a0
            vals = [counts[k] for k in sorted(counts)]
            if len(vals) == 1 or (len(vals) == 2 and vals[0] == vals[-1]):
                c = vals[0]
                if a0 > 0 and c % a0 == 0 and c // a0 > 1:
                    scaled = c
            elif maxd > 0:
                A = total / maxd
                N = A * F(normal[i], actual[i])
                if A.denominator == 1 and A.numerator > 1 and N.denominator == 1 and N.numerator > 1:
                    scaled = A.numerator
            for k in range(i, j):
                actual[k] = scaled
            i = j
        for i in range(n):
            if not group[i]:
                continue
            first = i == 0 or group[i - 1] != group[i]
            last = i + 1 == n or group[i + 1] != group[i]
            d[i].tuplet = 'startStop' if first and last else ('start' if first else ('stop' if last else ''))
            d[i].tuplet_actual = actual[i]


_SHARP_ORDER = (3, 0, 4, 1, 5, 2, 6)
_FLAT_ORDER = (6, 2, 5, 1, 4, 0, 3)
_LET = 'CDEFGAB'
_LETIDX = {c: i for i, c in enumerate(_LET)}
_KEYSIG = {}


def _keysig(key):
    ks = _KEYSIG.get(key)
    if ks is None:
        ks = [0] * 7
        for i in range(min(7, abs(key))):
            ks[_SHARP_ORDER[i] if key > 0 else _FLAT_ORDER[i]] = 1 if key > 0 else -1
        ks = _KEYSIG[key] = tuple(ks)
    return ks


def _acc_order(it):
    return (it.onset, it.grace_seq if it.ev.grace else 1 << 20, it.voice)


def _accidentals(items, key):
    keysig = _keysig(key)
    dstate, gstate = {}, {}
    seq = items if len(items) < 2 else sorted(items, key=_acc_order)
    first_line = True
    line_t, line_g = None, -1
    concurrent = {}
    for it in seq:
        ev = it.ev
        notes = ev.notes
        grace = ev.grace
        g = it.grace_seq if grace else 1 << 20
        if it.onset != line_t or g != line_g:
            if line_g != -1:
                first_line = False
            line_t, line_g = it.onset, g
            concurrent = {}
        nn = len(notes)
        accl = it.accidental = [None] * nn
        it.trill_acc = [None] * nn
        has_trill = False
        for ni in range(nn):
            n = notes[ni]
            p = n.pitch
            if n.trill:
                has_trill = True
            step = p.step
            li = _LETIDX.get(step)
            if li is None:
                li = _LET.index(step)
            dia = p.octave * 7 + li
            acc = p.alter
            if n.tie_stop:
                if acc != keysig[li] and first_line:
                    dstate[dia] = -1000 + acc
                    gstate[dia] = -1000 + acc
                continue
            if grace:
                if acc != (gstate[dia] if dia in gstate else keysig[dia % 7]):
                    accl[ni] = _ACC.get(acc, 'natural')
                    gstate[dia] = acc
                    dstate[dia] = -1000 + acc
                continue
            conc = concurrent.get(dia, 0)
            if (conc != 0 and conc == acc) or acc != (dstate[dia] if dia in dstate else keysig[dia % 7]):
                accl[ni] = _ACC.get(acc, 'natural')
                concurrent[dia] = acc
                dstate[dia] = acc
                gstate[dia] = acc
        if has_trill:
            for ni, n in enumerate(notes):
                if not n.trill:
                    continue
                k = _LET.index(n.pitch.step) + 1
                up_oct = n.pitch.octave + (k // 7)
                up_midi = n.pitch.midi + 2
                up_acc = up_midi - ((up_oct + 1) * 12 + (0, 2, 4, 5, 7, 9, 11)[k % 7])
                up_dia = up_oct * 7 + k % 7
                it.trill_acc[ni] = ''
                if (dstate[up_dia] if up_dia in dstate else keysig[up_dia % 7]) != up_acc:
                    it.trill_acc[ni] = _ACC.get(up_acc, 'natural')
                    dstate[up_dia] = -1000 + up_acc


# ── whole score ─────────────────────────────────────────────────────────────

def _long_ties(score, max_bars=4):
    """A tie chain over >= max_bars bars keeps its head; later parts become rests.

    The score is not changed: -> {id(event): the event to engrave instead}."""
    over = {}
    voices = {}
    for mi, bar in enumerate(score.bars):
        for st in bar.staves:
            for li, ly in enumerate(st.layers):
                key = (st.label, li)
                lst = voices.get(key)
                if lst is None:
                    lst = voices[key] = []
                for ev in ly.events:
                    if not ev.grace:
                        lst.append((mi, ev))
    for refs in voices.values():
        i = 0
        nr = len(refs)
        while i < nr:
            head = refs[i][1]
            hn = head.notes
            if not hn or not any(n.tie_start for n in hn) or any(n.tie_stop for n in hn):
                i += 1
                continue
            end = i
            for j in range(i + 1, nr):
                if any(n.tie_stop for n in refs[j][1].notes):
                    end = j
                else:
                    break
            if refs[end][0] - refs[i][0] < max_bars:
                i += 1
                continue
            over[id(head)] = Event(head.duration, [Note(n.pitch, False, n.tie_stop, n.trill) for n in hn],
                                   head.visible, head.grace, list(head.articulations))
            for j in range(i + 1, end + 1):
                e = refs[j][1]
                over[id(e)] = Event(e.duration, [], e.visible, e.grace, [])
            i = end + 1
    return over


def _split_staff(layers, bar, over):
    top, bot = bar.ts_top, bar.ts_bot
    out = []
    for ly in layers:
        src = [over.get(id(e), e) for e in ly.events] if over else ly.events
        # bar offsets in ticks of 1/D0 whole note
        D0 = 1
        for ev in src:
            dd = ev.duration.denominator
            if D0 % dd:
                D0 = D0 * dd // gcd(D0, dd)
        evs = []
        t = 0
        for ev in src:
            d = ev.duration
            if ev.grace or d.numerator <= 0:
                evs.append(ev)
                continue
            g = gcd(t, D0)
            seg = _segments(not ev.notes, t // g, D0 // g, d, top, bot)
            t += d.numerator * (D0 // d.denominator)
            if len(seg) <= 1:
                evs.append(ev)
                continue
            last = len(seg) - 1
            for k, sd in enumerate(seg):
                notes = [Note(n.pitch, n.tie_start if k == last else True,
                              n.tie_stop if k == 0 else True, n.trill)
                         for n in ev.notes]
                evs.append(Event(sd, notes, ev.visible, False, list(ev.articulations) if k == 0 else []))
        out.append(evs)
    return out


_PC = (0, 2, 4, 5, 7, 9, 11)


def _pitch_order(notes):
    """Note indices from low to high (diatonic position, then MIDI number)."""
    keys = []
    for n in notes:
        p = n.pitch
        li = _LETIDX.get(p.step)
        if li is None:
            keys.append((p.diatonic, p.midi))      # (raises as the property does)
            continue
        keys.append((p.octave * 7 + li, (p.octave + 1) * 12 + _PC[li] + p.alter))
    return sorted(range(len(notes)), key=keys.__getitem__)


@dataclass
class StaffBar:
    label: str
    clef: str
    clef_changed: bool
    voices: List[List[Item]]


def engrave(score):
    """-> (score, list (per bar) of list (per staff in score order) of StaffBar).

    The score itself is not modified."""
    over = _long_ties(score)
    bars = score.bars
    nbar = len(bars)
    # first clef written at or after each bar, per staff (for staves silent so far)
    next_clef = {}
    for label in score.staff_order:
        nc = [''] * (nbar + 1)
        for mi in range(nbar - 1, -1, -1):
            s2 = bars[mi].staff(label)
            nc[mi] = s2.clef if (s2 is not None and s2.clef) else nc[mi + 1]
        next_clef[label] = nc
    result = []
    clef_state = {}
    for mi, bar in enumerate(bars):
        row = []
        for label in score.staff_order:
            st = bar.staff(label)
            clef = st.clef if st is not None and st.clef else clef_state.get(label, '')
            if not clef:
                clef = next_clef[label][mi]
            changed = clef_state.get(label) != clef
            clef_state[label] = clef
            layers = [ly for ly in (st.layers if st is not None else []) if ly.events]
            if not layers:
                layers = [Layer(0, [Event(bar.length, [], True)])]
            split = _split_staff(layers, bar, over)
            # onsets in ticks of 1/D whole note
            D = 1
            for evs in split:
                for ev in evs:
                    dd = ev.duration.denominator
                    if D % dd:
                        D = D * dd // gcd(D, dd)
            voices = []
            items = []
            for cv, evs in enumerate(split):
                vitems = []
                t = 0
                gseq = 0
                nev = len(evs)
                for k in range(nev):
                    ev = evs[k]
                    it = Item(ev, cv, t)
                    if ev.grace:
                        it.grace_seq = gseq
                        gseq += 1
                        nxt = None
                        for k2 in range(k + 1, nev):
                            if not evs[k2].grace:
                                nxt = evs[k2].duration
                                break
                        nb = max(1, _recip(nxt) if nxt is not None else 4)
                        sp = it.spelling = _grace_spelling(nb)
                        it.nbeams = flags(sp.type)
                    else:
                        gseq = 0
                        info = it.info = _dur_info(ev.duration)
                        it.spelling = info[0]
                        it.nbeams = info[4]
                    notes = ev.notes
                    if not notes:
                        it.visible = cv == 0
                    elif len(notes) == 1:
                        it.order = [0]
                    else:
                        it.order = _pitch_order(notes)
                    vitems.append(it)
                    d = ev.duration
                    t += d.numerator * (D // d.denominator)
                voices.append(vitems)
                items.extend(vitems)
            _beams(voices, bar, D)
            _tuplets(voices)
            _accidentals(items, bar.key)
            row.append(StaffBar(label, clef, changed, voices))
        result.append(row)
    return score, result
