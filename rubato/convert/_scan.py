"""Stage 1 of the score reader: clean Rubato's score text into a regular event stream.

Rubato's score output is a whitespace-separated stream of
  bars      ``|N/DkF`` (meter N/D, key F in fifths; ``|kF`` / ``|`` inherit)
  staves    ``PR:`` / ``PL:`` (the following tokens belong to that staff)
  intervals ``n/d`` (advance the shared clock by n/d whole notes)
  onsets    ``E-4`` (uppercase pitch, optional ``tr`` ``>`` ``.`` suffixes)
  releases  ``e-4`` (lowercase pitch: the matching sounding note ends here)
  stamps    ``<|1.09|>`` (seconds), fused to a token or standalone.

The model's output can contain malformed events, so this stage applies fixed
repair rules (a leading interval is dropped, an interval longer than 16
whole notes or finer than 1/128 counts as zero, a note left sounding for more than 8
of its staff's events is retracted to one event, a duplicate onset of a sounding
pitch is dropped, an unmatched release is dropped, glued tokens are split, a
same-instant onset/release pair is a grace note, ...).  Every repair is counted.

Times are integer ticks: ``unit`` ticks per whole note, a multiple of every
denominator written in the text (so every interval is a whole number of ticks).

The output is a list of typed tokens:
  ('B', raw)        bar line          ('M', label)   staff change
  ('D', ticks)      interval > 0      ('G', kind)    grace marker ('lead' / 'zero')
  ('N', text)       onset             ('F', text)    release
and a map score-time (ticks) -> seconds of the stamps.
"""
from fractions import Fraction
from math import gcd
import re

from ._model import pitch_midi as _pitch_midi

TOMB = ('T', None)
STALE_LIMIT = 8          # staff events after which an unreleased note is retracted
_FENCE = re.compile(r'<\|(.*?)\|>')
_LEAD_DIGITS = re.compile(r'\d+')
_STOL = re.compile(r'\s*[+-]?\d+')
_DENOMS = re.compile(r'/[+-]?(\d+)')
_METER = re.compile(r'(\d+)/(\d+)')
_KEY = re.compile(r'k(-?\d+)')
_ONSET_OK = frozenset('0123456789#-tr>._')
_RELEASE_OK = frozenset('0123456789#-')


class ReadError(ValueError):
    pass


def _stol(s):
    """C ``std::stol`` semantics: optional sign + leading digits, else None."""
    m = _STOL.match(s)
    return int(m.group(0)) if m else None


def is_staff_token(t):
    if len(t) < 2 or t[-1] != ':' or not t[0].isascii() or not t[0].isalpha():
        return False
    return all(c.isascii() and c.isalnum() for c in t[:-1])


def _bar_key_ok(b, at):
    if at == len(b):
        return True
    if b[at] != 'k':
        return False
    at += 1
    if at < len(b) and b[at] == '-':
        at += 1
    if at == len(b):
        return False
    return b[at:].isdigit()


def _bar_body_ok(b):
    if not b:
        return True
    if b.startswith('N/A'):
        return _bar_key_ok(b, 3)
    if b[0] == 'k':
        return _bar_key_ok(b, 0)
    if b[0] == '0':
        return False
    at = 0
    while at < len(b) and b[at].isdigit():
        at += 1
    if at == 0 or at == len(b) or b[at] != '/':
        return False
    at += 1
    if at == len(b) or b[at] == '0' or not b[at].isdigit():
        return False
    while at < len(b) and b[at].isdigit():
        at += 1
    return _bar_key_ok(b, at)


def repair_bar(t):
    body = t[1:]
    while body and not _bar_body_ok(body):
        body = body[:-1]
    return '|' + body


def parse_bar(t, top, bot, key):
    """-> (top, bot, key, has_meter) with inheritance of what the bar omits."""
    body = t[1:]
    has_ts = False
    m = _METER.match(body)
    if m:
        top, bot, has_ts = int(m.group(1)), int(m.group(2)), True
        body = body[m.end():]
    elif body.startswith('N/A'):
        body = body[3:]
    k = _KEY.match(body)
    if k:
        key = int(k.group(1))
    return top, bot, key, has_ts


def bar_meter(t):
    sl = t.find('/')
    if sl < 0:
        return Fraction(0)
    num = _stol(t[1:sl]) or 0
    m = _LEAD_DIGITS.match(t[sl + 1:])
    den = int(m.group(0)) if m else 0
    return Fraction(num, den) if num > 0 and den > 0 else Fraction(0)


def pitch_midi(tok):
    m = _pitch_midi(tok)
    if m is None:
        raise ValueError(tok)
    return m


def tick_unit(text):
    """Ticks per whole note: the LCM of 4 and of every number written after a '/'
    (interval and meter denominators; extra factors are harmless)."""
    T = 4
    for d in set(_DENOMS.findall(text)):
        d = int(d)
        if d > 0 and T % d:
            T = T * d // gcd(T, d)
    return T


def _onset_info(t):
    """Onset token -> (token, fused interval text, repairs, junk, midi, bare pitch text)."""
    reps = []
    bt = t.find('`')
    if bt >= 0:
        t = t[:bt]
        reps.append('lyric_stripped')
    fused = ''
    sl = t.find('/')
    if sl >= 0:
        ns = sl
        while ns > 1 and t[ns - 1].isdigit():
            ns -= 1
        if ns > 1 and ns < sl and sl + 1 < len(t) and t[sl + 1:].isdigit():
            fused = t[ns:]
            t = t[:ns]
            reps.append('glued_interval')
    if not _ONSET_OK.issuperset(t[1:]) and any(not (ch.isdigit() or ch in '#-tr>._') for ch in t[1:]):
        return t, fused, tuple(reps), True, None, None
    midi = _pitch_midi(t)
    bare = t
    while True:
        if len(bare) >= 2 and bare.endswith('tr'):
            bare = bare[:-2]
            continue
        if bare and bare[-1] in '>.':
            bare = bare[:-1]
            continue
        break
    return t, fused, tuple(reps), False, midi, bare


def _release_info(t):
    """Release token -> (token, number of suffixes stripped, junk, midi)."""
    nsuf = 0
    while True:
        if len(t) >= 2 and t.endswith('tr'):
            t = t[:-2]
            nsuf += 1
            continue
        if t and t[-1] in '>.':
            t = t[:-1]
            nsuf += 1
            continue
        break
    if not _RELEASE_OK.issuperset(t[1:]) and any(not (ch.isdigit() or ch in '#-') for ch in t[1:]):
        return t, nsuf, True, None
    return t, nsuf, False, _pitch_midi(t)


class _Open:
    __slots__ = ('at', 'pitch', 'close_after', 'instances', 'run', 'grace',
                 'staff_events', 'last_counted', 'forced', 'out_pos', 'lst')

    def __init__(self, at, pitch, run, out_pos):
        self.at, self.pitch, self.run, self.out_pos = at, pitch, run, out_pos
        self.close_after = -1
        self.instances = 0
        self.grace = False
        self.staff_events = 0
        self.last_counted = -1
        self.forced = False
        self.lst = None


def scan(text, T=None):
    """Rubato score text -> (tokens, stamps, repairs, unit).

    unit: ticks per whole note (``tick_unit(text)`` unless given)."""
    if T is None:
        T = tick_unit(text)
    T16 = 16 * T
    rep = {}

    def repair(cls):
        rep[cls] = rep.get(cls, 0) + 1

    raw = text.split()
    # Notes before the first staff token belong to the SECOND distinct staff seen
    # (the model emits the lower staff first); one staff -> that one; none -> 'P'.
    distinct = []
    for t in raw:
        if t[-1] == ':' and is_staff_token(t):
            lb = t[:-1]
            if lb not in distinct:
                distinct.append(lb)
            if len(distinct) >= 2:
                break
    deferred = distinct[1] if len(distinct) >= 2 else (distinct[0] if distinct else 'P')

    staves = {}                  # label -> {midi: [_Open]}
    out = []
    emit = out.append
    forced = []                  # [pos, staff, release text]
    open_grace_group = {}
    grace_marker_pos = {}
    run_zero_closed = []
    run_zero_staff = ['']
    stamps = {}
    fresh = []                   # opens not yet followed by an interval (close_after < 0)
    run_opens = []               # opens created in the current run
    # scanner state
    clock = 0
    seen_note_or_bar = False
    cur = ''
    pending = False
    have = False
    last_emitted = None
    run = 0
    event = 0

    def stale_tick(label, opens, at_bar):
        if not opens:
            return
        due = False
        if at_bar:
            for lst in opens.values():
                for on in lst:
                    on.staff_events += 1
                    if (on.staff_events > STALE_LIMIT and not on.forced and not on.grace
                            and on.close_after >= 0 and on.instances == 1):
                        due = True
        else:
            for lst in opens.values():
                for on in lst:
                    if on.last_counted != event:
                        on.staff_events += 1
                        on.last_counted = event
                    if (on.staff_events > STALE_LIMIT and not on.forced and not on.grace
                            and on.close_after >= 0 and on.instances == 1):
                        due = True
        if not due:
            return
        for midi in sorted(opens):
            for on in opens[midi]:
                if on.forced or on.grace or on.staff_events <= STALE_LIMIT:
                    continue
                if on.close_after < 0 or on.instances != 1:
                    continue
                forced.append([on.close_after + 1, label, on.pitch[0].lower() + on.pitch[1:]])
                on.forced = True
                repair('stale_retracted')

    def end_run(grace):
        nonlocal run, run_opens
        if grace:
            for on in run_opens:
                on.grace = True
        run += 1
        run_opens = []

    def ensure_staff():
        nonlocal pending, last_emitted
        pending = False
        s = cur if have else deferred
        if s == last_emitted:
            return
        emit(('M', s))
        last_emitted = s
        end_run(False)

    def run_begin():
        r0 = len(out)
        while r0 > 0 and out[r0 - 1][0] in ('N', 'T'):
            r0 -= 1
        return r0

    def lift_group(lbl, pos):
        nonlocal last_emitted
        repair('lifted_grace')
        emit(('M', lbl))
        last_emitted = lbl
        for p in pos:
            out.append(out[p])
            out[p] = TOMB
        emit(('G', 'lead'))
        open_grace_group[lbl] = 1
        grace_marker_pos[lbl] = len(out) - 1
        end_run(False)

    def resolve_run_graces():
        if not run_zero_closed:
            return
        r0 = run_begin()
        allz = all(out[i] is TOMB or i in run_zero_closed for i in range(r0, len(out)))
        lbl = run_zero_staff[0]
        pos = list(run_zero_closed)
        run_zero_closed.clear()
        if allz:
            emit(('G', 'lead'))
            open_grace_group[lbl] = 1
            grace_marker_pos[lbl] = len(out) - 1
            end_run(False)
        else:
            lift_group(lbl, pos)

    def duration_emitted(v):
        nonlocal event, fresh
        if v != 0:
            event += 1
            if fresh:
                pos = len(out) - 1
                counts = {}
                for on in fresh:
                    lst = on.lst
                    if on.close_after >= 0 or on not in lst:      # (identity: no __eq__)
                        continue          # already set, or no longer sounding
                    k = id(lst)
                    s = counts.get(k)
                    if s is None:
                        s = counts[k] = sum(1 for o in lst if not o.grace and not o.forced)
                    on.close_after = pos
                    on.instances = s
                fresh = []
        end_run(v == 0)

    def peel(t):
        num = None
        while True:
            m = _FENCE.search(t)
            if not m:
                break
            inner = m.group(1)
            t = t[:m.start()] + t[m.end():]
            if inner and (inner[0].isdigit() or inner[0] == '-'):
                if inner.endswith('s'):
                    inner = inner[:-1]
                try:
                    num = float(inner)
                    continue
                except ValueError:
                    pass
            repair('unknown_bracket_token')
        return t, num

    def interval(num, den):
        """num/den whole notes -> ticks (den > 0, den | T)."""
        return num * (T // den)

    icache, ocache, rcache, staff_ok = {}, {}, {}, {}
    n_raw = len(raw)
    for in_i in range(n_raw):
        t = raw[in_i]
        if '<|' in t:
            t, fts = peel(t)
        else:
            fts = None
        if not t:
            if fts is not None:
                stamps[clock] = fts
            continue
        c0 = t[0]
        if not ('a' <= c0 <= 'g'):
            if run_zero_closed:
                resolve_run_graces()

        if t[-1] == ':' and (staff_ok[t] if t in staff_ok else staff_ok.setdefault(t, is_staff_token(t))):
            if have and pending:
                repair('orphan_staff')
            pending = True
            cur = t[:-1]
            have = True
            end_run(False)
            if fts is not None:
                stamps[clock] = fts
            continue
        colon = t.find(':')
        if (colon > 0 and colon + 1 < len(t) and ';' not in t and t[0].isascii() and t[0].isalpha()
                and all(c.isascii() and c.isalnum() for c in t[1:colon])):
            repair('glued_staff')
            cur = t[:colon]
            have = True
            end_run(False)
            t = t[colon + 1:]
            c0 = t[0]

        if c0 == '|':
            for lb in sorted(staves):
                stale_tick(lb, staves[lb], True)
            fixed = repair_bar(t)
            if fixed != t:
                repair('malformed_bar')
                t = fixed
            emit(('B', t))
            seen_note_or_bar = True
            end_run(False)
            open_grace_group.clear()
            if fts is not None:
                stamps[clock] = fts
            continue
        if ';' in t and c0.isalpha():          # chord symbol (not part of this output)
            repair('chord_symbol_stripped')
            pending = False
            if fts is not None:
                stamps[clock] = fts
            continue
        if 'A' <= c0 <= 'G':
            info = ocache.get(t)
            if info is None:
                info = ocache[t] = _onset_info(t)
            t, fused, reps, junk, midi, bare = info
            for r in reps:
                repair(r)
            if junk:
                repair('junk_token')
                if fts is not None:
                    stamps[clock] = fts
                continue
            if fts is not None:
                stamps[clock] = fts
            if midi is None:
                repair('unknown_bracket_token')
                continue
            if not have and distinct:
                repair('deferred_staff')
            lbl = cur if have else deferred
            if lbl == last_emitted:
                pending = False
            else:
                ensure_staff()
            seen_note_or_bar = True
            opens = staves.get(lbl)
            if opens is None:
                opens = staves[lbl] = {}
            elif opens:
                stale_tick(lbl, opens, False)
            lst = opens.get(midi)
            drop = lst is not None and any(not o.grace and not o.forced for o in lst)
            if drop:
                repair('duplicate_onset')
                emit(TOMB)
            else:
                on = _Open(clock, bare, run, len(out))
                if lst is None:
                    lst = opens[midi] = []
                lst.append(on)
                on.lst = lst
                fresh.append(on)
                run_opens.append(on)
                open_grace_group.pop(lbl, None)
                emit(('N', t))
            if fused:
                a, b = fused.split('/')
                num, den = int(a), int(b)
                v = 0
                if 0 < den <= 128:
                    v = interval(num, den)
                elif den > 128:
                    repair('tiny_interval')
                if v > T16:
                    repair('huge_interval')
                    v = 0
                clock = clock + v
                emit(('D', v) if v != 0 else ('G', 'zero'))
                duration_emitted(v)
                open_grace_group.clear()
            continue
        if 'a' <= c0 <= 'g':
            info = rcache.get(t)
            if info is None:
                info = rcache[t] = _release_info(t)
            t, nsuf, junk, midi = info
            for _ in range(nsuf):
                repair('release_suffix')
            if junk:
                repair('junk_token')
                if fts is not None:
                    stamps[clock] = fts
                continue
            if fts is not None:
                stamps[clock] = fts
            if midi is None:
                repair('unmatched_release')
                continue
            lbl = cur if have else deferred
            opens = staves.get(lbl)
            if opens is None:
                opens = staves[lbl] = {}
            elif opens:
                stale_tick(lbl, opens, False)
            stack = opens.get(midi)
            if not stack:
                repair('unmatched_release')
                continue
            pick = len(stack)
            if stack[-1].at == clock and not stack[-1].forced:
                pick = len(stack) - 1
            else:
                for k in range(len(stack) - 1, -1, -1):
                    if not stack[k].grace and not stack[k].forced:
                        pick = k
                        break
            if pick == len(stack):
                fi = next((k for k, o in enumerate(stack) if o.forced), None)
                if fi is not None:
                    del stack[fi]
                    if not stack:
                        del opens[midi]
                    nxt = raw[in_i + 1] if in_i + 1 < n_raw else ''
                    if (out and out[-1][0] == 'N' and nxt and 'A' <= nxt[0] <= 'G'):
                        emit(('M', lbl))
                        last_emitted = lbl
                        end_run(False)
                    continue
                repair('unmatched_release')
                continue
            on = stack.pop(pick)
            if not stack:
                del opens[midi]
            if on.at == clock:                  # zero elapsed -> grace
                fold = (open_grace_group.get(lbl, 0) > 0 and out and lbl in grace_marker_pos
                        and grace_marker_pos[lbl] == len(out) - 1)
                op = on.out_pos
                if on.grace or op < 0 or op >= len(out) or out[op] is TOMB:
                    pass
                elif fold:
                    tok = out[op]
                    out[op] = TOMB
                    out.insert(len(out) - 1, tok)
                    grace_marker_pos[lbl] = len(out) - 1
                elif op >= run_begin() and (not run_zero_closed or run_zero_staff[0] == lbl):
                    run_zero_closed.append(op)
                    run_zero_staff[0] = lbl
                else:
                    resolve_run_graces()
                    lift_group(lbl, [op])
            else:
                resolve_run_graces()
                if (cur if have else deferred) == last_emitted:
                    pending = False
                else:
                    ensure_staff()
                emit(('F', t))
                end_run(False)
            continue
        if c0 == '/' or c0.isdigit():
            if not seen_note_or_bar:
                repair('leading_interval')
                continue
            hit = icache.get(t)
            if hit is not None:
                v = hit[0]
                for r in hit[1]:
                    repair(r)
                clock = clock + v
                if fts is not None:
                    stamps[clock] = fts
                emit(('D', v) if v != 0 else ('G', 'zero'))
                duration_emitted(v)
                open_grace_group.clear()
                continue
            t_key = t
            r0 = dict(rep)
            v = 0
            sl = t.find('/')
            if c0 == '/':
                repair('elided_numerator')
                den = _stol(t[1:]) or 0
                if den <= 0:
                    v = 0
                elif den > 128:
                    repair('tiny_interval')
                else:
                    v = interval(1, den)
            elif sl < 0:
                num = _stol(t) or 0
                if num > 16:
                    repair('huge_interval')
                else:
                    v = num * T
            else:
                num = _stol(t[:sl])
                den = _stol(t[sl + 1:])
                if num is None or den is None:
                    num = num if num is not None else 0
                    den = den if den is not None else 0
                if den <= 0:
                    v = 0
                elif den > 128:
                    repair('tiny_interval')
                else:
                    v = interval(num, den)
                    if v > T16:
                        repair('huge_interval')
                        v = 0
            icache[t_key] = (v, tuple(r for r in rep for _ in range(rep[r] - r0.get(r, 0))))
            clock = clock + v
            if fts is not None:
                stamps[clock] = fts
            emit(('D', v) if v != 0 else ('G', 'zero'))
            duration_emitted(v)
            open_grace_group.clear()
            continue
        repair('junk_token')
        if fts is not None:
            stamps[clock] = fts
    resolve_run_graces()

    for lb in sorted(staves):
        for midi in sorted(staves[lb]):
            for on in staves[lb][midi]:
                repair('unreleased_at_end')
                if on.grace or on.forced or on.close_after < 0 or on.instances != 1:
                    continue
                forced.append([on.close_after + 1, lb, on.pitch[0].lower() + on.pitch[1:]])

    if forced:
        ctx_at = []
        cur = ''
        for tok in out:
            ctx_at.append(cur)
            if tok[0] == 'M':
                cur = tok[1]
        ctx_at.append(cur)
        for fc in forced:
            p = min(max(fc[0], 0), len(out))
            while True:
                q = p
                while q < len(out) and out[q][0] == 'N':
                    q += 1
                if q > p and q < len(out) and out[q] == ('G', 'lead'):
                    p = q + 1
                    continue
                break
            fc[0] = p
        order = sorted(range(len(forced)), key=lambda k: -forced[k][0])   # stable, highest first
        forced = [forced[k] for k in order]
        i = 0
        while i < len(forced):
            g = i
            while g < len(forced) and forced[g][0] == forced[i][0]:
                g += 1
            p = min(max(forced[i][0], 0), len(out))
            ctx = ctx_at[forced[i][0]]
            ins = []
            cur = ctx
            for k in range(g - 1, i - 1, -1):
                if forced[k][1] != cur:
                    ins.append(('M', forced[k][1]))
                    cur = forced[k][1]
                ins.append(('F', forced[k][2]))
            if cur != ctx and ctx:
                k = p
                while k < len(out) and out[k][0] in ('B', 'D', 'G'):
                    k += 1
                if k < len(out) and out[k][0] != 'M':
                    ins.append(('M', ctx))
            out[p:p] = ins
            i = g

    kept = []
    next_kept = None
    lifted_between = False
    for tok in reversed(out):
        if tok is TOMB:
            lifted_between = True
            continue
        if tok[0] == 'M' and lifted_between and (next_kept is None or next_kept[0] in ('M', 'B', 'D', 'G')):
            continue
        kept.append(tok)
        next_kept = tok
        lifted_between = False
    kept.reverse()
    return kept, stamps, rep, T
