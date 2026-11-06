"""Readers for Rubato's three output modes.

* ``read_score(text)``       score text  -> ``Score`` (bars, two piano staves, voices,
                                             ties, graces, clefs, stamps)
* ``read_performance(text)`` midi text   -> list of ``PerfNote`` + pedal events
* ``read_beats(text)``       beats text  -> list of (seconds, is_downbeat)

Only Rubato's own bare output is accepted.  Fenced tokens other than numeric
stamps (``<|1.23|>``) and velocities (``<|vel:N|>``) and pedal marks are stripped and
counted, never interpreted.

The score reader repairs malformed events, estimates voices, and selects clefs.
The right-hand staff ``PR`` is the upper staff and ``PL`` the lower one, regardless
of the order in which the model writes them inside an instant.
"""
from fractions import Fraction
from math import gcd

from ._model import Bar, Event, Layer, Note, Pitch, Score, Staff, parse_pitch, pitch_midi
from ._scan import ReadError, parse_bar, scan
from ._stdsort import cxx_sort
from ._voices import assign_voices
from ._perf import PerfNote, parse_beats, parse_performance

__all__ = ['read_score', 'read_performance', 'read_beats', 'ReadError', 'PerfNote']

# Upper staff first.  A known two-staff keyboard part: when only one of its staves
# sounds, the other is still part of the score (an empty staff).
_KEYBOARD = ('PR', 'PL')


# ─────────────────────────────── score text ────────────────────────────────
#
# Internally every time is an integer number of ticks (``unit`` ticks per whole
# note, from the scanner); the Score handed out carries Fractions as documented.

_ONSETS = {}          # onset token -> (step, alter, octave, trill, articulations)


def _parse_onset(text):
    """'C#5tr>.' -> (Pitch, trill, [articulations in written order, deduplicated])"""
    r = _ONSETS.get(text)
    if r is None:
        pp = parse_pitch(text)
        if pp is None:
            raise ReadError(f'bad pitch token {text!r}')
        pitch, rest = pp
        arts, trill = [], False
        while rest:
            if rest.endswith('tr'):
                trill = True
                rest = rest[:-2]
            elif rest[-1] in '.>':
                arts.append(rest[-1])
                rest = rest[:-1]
            else:
                break
        out = []
        for a in reversed(arts):
            if a not in out:
                out.append(a)
        r = (pitch.step, pitch.alter, pitch.octave, trill, tuple(out), pitch.midi)
        if len(_ONSETS) > 100000:
            _ONSETS.clear()
        _ONSETS[text] = r
    return Pitch(r[0], r[1], r[2]), r[3], list(r[4])


def _onset_midi(text):
    r = _ONSETS.get(text)
    if r is None:
        _parse_onset(text)
        r = _ONSETS[text]
    return r[5]


class _Flat:
    __slots__ = ('staff', 'onset', 'offset', 'closed', 'pitch', 'trill', 'arts', 'grace_on',
                 'grace_end', 'tie_stop', 'tie_start', 'pid', 'midi')

    def __init__(self, staff, onset, pitch, midi, trill=False, arts=(), tie_stop=False, pid=-1):
        self.staff, self.onset, self.offset = staff, onset, onset
        self.closed = False
        self.pitch, self.trill, self.arts = pitch, trill, list(arts)
        self.grace_on, self.grace_end = [], []
        self.tie_stop, self.tie_start = tie_stop, False
        self.pid = pid
        self.midi = midi


class _Phys:
    __slots__ = ('staff', 'on', 'off', 'on_ev', 'off_ev', 'midi', 'voice', 'closed')

    def __init__(self, staff, on, on_ev, midi):
        self.staff, self.on, self.off, self.on_ev, self.off_ev = staff, on, on, on_ev, on_ev
        self.midi, self.voice, self.closed = midi, 0, False


class _Plan:
    def __init__(self):
        self.notes = []
        self.cursor = 0          # ordinal of the current clock position (every interval token)
        self.replay = 0
        self.bar_start = 0
        self.collect = True


class _Ctx:
    """State that persists across bars while parsing."""

    def __init__(self):
        self.sustain = {}        # (staff, midi) -> [(pitch, pid)]
        self.interned = []       # staff labels in first-appearance order
        self.onset_order = []    # staves in order of their first onset

    def intern(self, s):
        if s not in self.interned:
            self.interned.append(s)


class _SC:
    """Per-staff state inside one bar."""
    __slots__ = ('open', 'pending', 'free', 'last_closed', 'closed_t')

    def __init__(self):
        self.open = {}
        self.pending = []
        self.free = []
        self.last_closed = -1
        self.closed_t = None


def _grace_event(pitches):
    return Event(0, [Note(p) for p in pitches], True, True, [])


def _parse_bar(toks, start, end, init_staff, ctx, plan, nre=None):
    """Read one bar's tokens: collect its notes (and their clock ordinals into
    plan.notes).  -> (bar state for _build_bar, extent, staff at the end)."""
    notes = []
    sctx = {}
    interned = ctx.interned

    def SC(s):
        c = sctx.get(s)
        if c is None:
            c = sctx[s] = _SC()
            if s not in interned:
                interned.append(s)
        return c

    cur_time = 0
    max_time = 0
    cur = init_staff
    if cur not in interned:
        interned.append(cur)
    seen = []
    cursor = plan.cursor
    bar_start = plan.bar_start
    pnotes = plan.notes

    if ctx.sustain:
        for key in sorted(ctx.sustain):
            s, midi = key
            if s not in interned:
                interned.append(s)
            for pitch, pid in ctx.sustain[key]:
                notes.append(_Flat(s, 0, pitch, midi, tie_stop=True, pid=pid))
                SC(s).open.setdefault(midi, []).append(len(notes) - 1)
        ctx.sustain.clear()

    i = start
    while i < end:
        kind, val = toks[i]
        if kind == 'D':
            cur_time += val
            cursor += 1
            if cur_time > max_time:
                max_time = cur_time
            i += 1
            continue
        if kind == 'N':
            # onset
            if cur not in seen:
                seen.append(cur)
            if cur not in ctx.onset_order:
                ctx.onset_order.append(cur)
            r = _ONSETS.get(val)
            if r is None:
                _parse_onset(val)
                r = _ONSETS[val]
            pitch, trill, arts, midi = Pitch(r[0], r[1], r[2]), r[3], r[4], r[5]
            if nre is not None:
                run_end = nre[i]
                if run_end > end:
                    run_end = end
            else:
                run_end = i
                while run_end < end and toks[run_end][0] == 'N':
                    run_end += 1
            if run_end < end and toks[run_end][0] == 'G':
                stack = [pitch] + [_parse_onset(toks[k][1])[0] for k in range(i + 1, run_end)]
                nxt = run_end
                n = len(stack)
                ok = run_end + n < end
                k = 0
                while ok and k < n:
                    t2 = toks[run_end + 1 + k]
                    if t2[0] != 'F':
                        ok = False
                        break
                    p2 = parse_pitch(t2[1])
                    if p2 is None or p2[0].midi != stack[k].midi:
                        ok = False
                    k += 1
                if ok:
                    nxt = run_end + n
                gc = SC(cur)
                ev = _grace_event(stack)
                if gc.last_closed >= 0 and notes[gc.last_closed].offset == cur_time:
                    notes[gc.last_closed].grace_end.append(ev)
                else:
                    gc.pending.append((cur_time, ev))
                i = nxt + 1
                continue
            oc = sctx.get(cur) or SC(cur)
            fn = _Flat(cur, cur_time, pitch, midi, trill, arts)
            if oc.pending:
                keep = []
                for t, ev in oc.pending:
                    if t == cur_time:
                        fn.grace_on.append(ev)
                    else:
                        keep.append((t, ev))
                oc.pending = keep
            fn.pid = len(pnotes)
            pnotes.append(_Phys(cur, bar_start + cur_time, cursor, midi))
            notes.append(fn)
            stk = oc.open.get(midi)
            if stk is None:
                oc.open[midi] = [len(notes) - 1]
            else:
                stk.append(len(notes) - 1)
            i += 1
            continue
        if kind == 'F':
            if cur not in seen:
                seen.append(cur)
            midi = pitch_midi(val)
            if midi is None:
                raise ReadError(f'bad release token {val!r}')
            close_staff = cur
            stack = (sctx.get(cur) or SC(cur)).open.get(midi)
            if not stack:
                close_staff = None
                for s in interned:
                    if s == cur or s not in sctx:
                        continue
                    if sctx[s].open.get(midi):
                        close_staff = s
                        break
                if close_staff is None:
                    raise ReadError(f'release {val!r} of a pitch not sounding')
                stack = sctx[close_staff].open[midi]
            ni = stack.pop()
            fl = notes[ni]
            fl.offset = cur_time
            fl.closed = True
            if fl.pid >= 0:
                ph = pnotes[fl.pid]
                ph.off = bar_start + cur_time
                ph.off_ev = cursor
                ph.closed = True
            # the latest-starting (then highest) note closed at this instant
            c = sctx.get(close_staff) or SC(close_staff)
            if c.closed_t != cur_time:
                c.closed_t = cur_time
                c.last_closed = ni
            else:
                b = notes[c.last_closed]
                if fl.onset > b.onset or (fl.onset == b.onset and fl.midi > b.midi):
                    c.last_closed = ni
            i += 1
            continue
        if kind == 'B':
            i += 1
            continue
        if kind == 'G':
            cursor += 1
            if cur_time > max_time:
                max_time = cur_time
            i += 1
            continue
        if kind == 'M':
            cur = val
            if cur not in interned:
                interned.append(cur)
            if cur not in seen:
                seen.append(cur)
            i += 1
            continue
        raise ReadError(f'unexpected token kind {kind!r}')
    plan.cursor = cursor

    end_staff = cur
    for s in interned:
        c = sctx.get(s)
        if c is None or not c.pending:
            continue
        for coord, ev in c.pending:
            at_end = at_on = -1
            for ni, fnn in enumerate(notes):
                if fnn.staff != s:
                    continue
                if fnn.offset == coord and (at_end < 0 or fnn.onset > notes[at_end].onset):
                    at_end = ni
                if fnn.onset == coord and (at_on < 0 or fnn.onset > notes[at_on].onset):
                    at_on = ni
            if at_end >= 0:
                notes[at_end].grace_end.append(ev)
            elif at_on >= 0:
                notes[at_on].grace_on.append(ev)
            else:
                c.free.append((coord, ev))
        c.pending = []

    sustain = ctx.sustain
    for fnn in notes:
        if not fnn.closed:
            fnn.offset = max(max_time, fnn.onset)
            fnn.tie_start = True
            key = (fnn.staff, fnn.midi)
            lst = sustain.get(key)
            if lst is None:
                sustain[key] = [(fnn.pitch, fnn.pid)]
            else:
                lst.append((fnn.pitch, fnn.pid))

    return (notes, seen, sctx, len(interned), max_time), max_time, end_staff


def _build_bar(state, ctx, plan):
    """The layers of a bar read by _parse_bar, with the voices now in plan.notes:
    -> [[staff, voice, [events]]]. Reuse the parsed bar state rather than parsing it again."""
    notes, seen, sctx, n_interned, max_time = state
    interned = ctx.interned[:n_interned]
    pnotes = plan.notes
    by_staff = {}
    for k, fnn in enumerate(notes):
        lst = by_staff.get(fnn.staff)
        if lst is None:
            by_staff[fnn.staff] = [k]
        else:
            lst.append(k)
    for s in interned:
        if s in by_staff and s not in seen:
            seen.append(s)

    out = []          # (staff, voice, [events])
    for s in seen:
        idxs = by_staff.get(s)
        if not idxs:
            if s and max_time > 0:
                out.append([s, 0, [Event(max_time, [], True)]])
            continue
        voice = [pnotes[notes[k].pid].voice for k in idxs]
        for L in range(max(voice) + 1):
            ln = [idxs[j] for j in range(len(idxs)) if voice[j] == L]
            if not ln:
                continue
            ons = [notes[k].onset for k in ln]
            if not (all(ons[x] < ons[x + 1] for x in range(len(ons) - 1)) or
                    (len(ons) <= 16 and all(ons[x] <= ons[x + 1] for x in range(len(ons) - 1)))):
                # (a strictly increasing range, or a sorted one of <= 16 elements, is
                # left untouched by the introsort)
                cxx_sort(ln, lambda a, b: notes[a].onset < notes[b].onset)
            evs = []
            cursor_t = 0
            j = 0
            nl = len(ln)
            while j < nl:
                fj = notes[ln[j]]
                onset, offset = fj.onset, fj.offset
                e = j + 1
                while e < nl and notes[ln[e]].onset == onset and notes[ln[e]].offset == offset:
                    e += 1
                if onset > cursor_t:
                    evs.append(Event(onset - cursor_t, [], True))
                members = [notes[ln[x]] for x in range(j, e)]
                ev = Event(offset - onset,
                           [Note(m.pitch, m.tie_start, m.tie_stop, m.trill) for m in members],
                           True, False, list(members[-1].arts))
                for m in members:
                    if m.grace_on:
                        evs.extend(m.grace_on)
                evs.append(ev)
                for m in members:
                    if m.grace_end:
                        evs.extend(m.grace_end)
                cursor_t = offset
                j = e
            if max_time > cursor_t:
                evs.append(Event(max_time - cursor_t, [], True))
            out.append([s, L, evs])

    for s in interned:
        c = sctx.get(s)
        if c is None or not c.free:
            continue
        free = sorted(c.free, key=lambda x: x[0])
        home = next((ld for ld in out if ld[0] == s), None)
        if home is None:
            home = [s, 0, []]
            out.append(home)
        for coord, ev in free:
            at = 0
            pos = len(home[2])
            for k, e2 in enumerate(home[2]):
                if at >= coord:
                    pos = k
                    break
                at += e2.duration
            if at < coord:
                home[2].append(Event(coord - at, [], False))
                pos = len(home[2])
            home[2].insert(pos, ev)
    return out


def _segment(toks):
    """-> list of (bar_index or -1, start, end, top, bot, key, has_meter)."""
    bars = [k for k, t in enumerate(toks) if t[0] == 'B']
    chunks = []
    top, bot, key = 4, 4, 0
    first = bars[0] if bars else len(toks)
    if first > 0:
        t0, b0, k0, h0 = (top, bot, key, False)
        if bars:
            t0, b0, k0, h0 = parse_bar(toks[bars[0]][1], top, bot, key)
        chunks.append((-1, 0, first, t0, b0, k0, h0))
    for bi, pos in enumerate(bars):
        end = bars[bi + 1] if bi + 1 < len(bars) else len(toks)
        top, bot, key, has = parse_bar(toks[pos][1], top, bot, key)
        chunks.append((pos, pos, end, top, bot, key, has))
    return chunks


def _llen(ly):
    s = 0
    for e in ly.events:
        s += e.duration
    return s


def _drop_zero_length(score):
    open_ties, orphaned = {}, {}
    for bar in score.bars:
        next_open = {}
        for st in bar.staves:
            for ly in st.layers:
                evs = ly.events
                i = 0
                while i < len(evs):
                    ev = evs[i]
                    if ev.duration != 0 or ev.grace:
                        for n in ev.notes:
                            key = (st.label, n.pitch.midi)
                            if n.tie_stop:
                                if orphaned.get(key, 0) > 0:
                                    n.tie_stop = False
                                    orphaned[key] -= 1
                                    if orphaned[key] == 0:
                                        del orphaned[key]
                                elif open_ties.get(key):
                                    open_ties[key].pop()
                            if n.tie_start:
                                next_open.setdefault(key, []).append(n)
                        i += 1
                        continue
                    for n in ev.notes:
                        key = (st.label, n.pitch.midi)
                        if n.tie_stop and not n.tie_start:
                            if open_ties.get(key):
                                open_ties[key][-1].tie_start = False
                                open_ties[key].pop()
                        elif n.tie_start and not n.tie_stop:
                            orphaned[key] = orphaned.get(key, 0) + 1
                    del evs[i]
        for key in sorted(next_open):
            open_ties.setdefault(key, []).extend(next_open[key])


def _tidy_rests(score):
    if any(ev.duration == 0 and not ev.grace for b in score.bars for st in b.staves
           for ly in st.layers for ev in ly.events):
        _drop_zero_length(score)
    for bar in score.bars:
        for st in bar.staves:
            if len(st.layers) > 1:
                st.layers.sort(key=lambda ly: ly.index)
            for ly in st.layers:
                merged = []
                for ev in ly.events:
                    if (merged and not merged[-1].notes and not merged[-1].visible and not ev.notes
                            and not ev.visible and not ev.articulations):
                        merged[-1].duration += ev.duration
                    else:
                        merged.append(ev)
                ly.events = merged
            if bar.ts_top > 0 and bar.ts_bot > 0 and bar.length > 0:
                lens = [_llen(ly) for ly in st.layers]
                for ln in lens:
                    if ln > bar.length:
                        bar.length = ln
                target = bar.length
                for ly, ln in zip(st.layers, lens):
                    if not ly.events:
                        continue
                    last = ly.events[-1]
                    if not last.notes and not last.visible:
                        want = target - (ln - last.duration)
                        if want > 0:
                            last.duration = want
                        elif not last.articulations and len(ly.events) > 1:
                            ly.events.pop()
                    else:
                        if ln < target:
                            ly.events.append(Event(target - ln, [], False))
            kept = [ly for ly in st.layers
                    if not all(not ev.notes and not ev.visible and not ev.articulations for ev in ly.events)]
            if not kept and st.layers:
                kept = [st.layers[0]]
            st.layers = kept


# Clef choice: a dynamic programme over every onset of a staff (port of the
# MuseScore-derived clef chooser).  Treble = 'G2', bass = 'F4'.
_CLEF_SPAN = {'G2': (30, 38), 'F4': (18, 26)}     # diatonic index range of the five lines


def _dia(midi):
    return (midi // 12 - 1) * 7 + (0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6)[midi % 12]


def _pitch_pen(midi, span):
    d = _dia(midi)
    outside = span[0] - d if d < span[0] else (d - span[1] if d > span[1] else 0)
    if outside <= 2:
        return 0
    if outside <= 5:
        return 1
    if outside <= 8:
        return 10000
    return 20000


_PEN = {}


def _pen_table(span):
    t = _PEN.get(span)
    if t is None:
        t = _PEN[span] = [_pitch_pen(m, span) for m in range(128)]
    return t


# position fields
_TIME, _MAXDUR, _LO, _HI, _TIEIN, _BARSTART = range(6)


def _compute_clefs(pos, clefs, beat):
    """pos: list of [time, max_dur, lo, hi, tie_in, bar_start] -> clef index per position.
    beat: a quarter note in the unit of the times."""
    n = len(pos)
    if n == 0:
        return []
    spans = [_CLEF_SPAN[c] for c in clefs]
    S = len(spans)
    latest = []
    running = 0
    for p in pos:
        e = p[_TIME] + p[_MAXDUR]
        if e > running:
            running = e
        latest.append(running)
    # rest before position j (j >= 1): the gap from the latest end so far
    gap = [False] * n
    for j in range(1, n):
        gap[j] = pos[j][_TIME] - latest[j - 1] >= beat
    path = [[0] * n for _ in range(S)]

    def change_pen(at, prev):
        if at == 0:
            return 0
        pen = 0
        j = at - 1
        pp = path[prev]
        while j >= 0:
            if gap[j + 1]:
                if j + 1 != at:
                    pen += 2
                break
            if j == at - 5:
                break
            if j == 0 or pp[j] != prev:
                pen += 10001 if j == at - 1 else 1000
                break
            j -= 1
        cnt, has_gap = 0, False
        for j in range(at, n):
            if j > at and gap[j]:
                pen += 2
                has_gap = True
                break
            cnt += 1
            if cnt == 5:
                break
        if not has_gap and cnt < 5:
            pen += 1000
        return pen

    tables = [_pen_table(sp) for sp in spans]
    pens = [[0, 0] for _ in range(S)]
    for at in range(n):
        dp = pos[at]
        lo, hi = dp[_LO], dp[_HI]
        locked = dp[_TIEIN] or not dp[_BARSTART]
        par = (at - 1) % 2
        for c in range(S):
            tb = tables[c]
            a, b = tb[lo] if 0 <= lo < 128 else _pitch_pen(lo, spans[c]), \
                tb[hi] if 0 <= hi < 128 else _pitch_pen(hi, spans[c])
            p_pen = a if a > b else b
            if c != 0:
                p_pen += 1
            if at > 0 and locked:
                # only the same clef can precede (best is then that one)
                pens[c][at % 2] = p_pen + pens[c][par]
                path[c][at] = c
                continue
            best_prev, best = 0, None
            for prev in range(S):
                pen = p_pen
                if at > 0 and prev != c:
                    pen += 40000 + change_pen(at, prev)
                if at > 0:
                    pen += pens[prev][par]
                if best is None or (prev != c and pen < best) or (prev == c and pen <= best):
                    best, best_prev = pen, prev
            pens[c][at % 2] = best
            path[c][at] = best_prev
    last = n - 1
    cur = 0
    for c in range(1, S):
        if pens[c][last % 2] < pens[cur][last % 2]:
            cur = c
    res = [0] * n
    res[last] = cur
    for at in range(last, 0, -1):
        res[at - 1] = path[res[at]][at]
    return res


def _clef_vocabulary(label):
    if label == 'PL':
        return ['F4', 'G2']
    return ['G2', 'F4']


def _estimate_clefs(score, T):
    for label in score.staff_order:
        positions, starts = [], []
        t0 = 0
        for bar in score.bars:
            starts.append(t0)
            pm = {}
            st = bar.staff(label)
            if st is not None:
                for ly in st.layers:
                    t = 0
                    for ev in ly.events:
                        if ev.notes:
                            at = t0 + t
                            p = pm.get(at)
                            if p is None:
                                p = pm[at] = [at, 0, 127, 0, False, at == t0]
                            for nt in ev.notes:
                                m = nt.pitch.midi
                                if m < p[_LO]:
                                    p[_LO] = m
                                if m > p[_HI]:
                                    p[_HI] = m
                                if nt.tie_stop:
                                    p[_TIEIN] = True
                            if ev.duration > p[_MAXDUR]:
                                p[_MAXDUR] = ev.duration
                        t += ev.duration
            if pm:
                positions.extend(pm[k] for k in sorted(pm))
            t0 += bar.length if bar.length > 0 else bar.ts_top * (T // bar.ts_bot)
        if not positions:
            continue
        vocab = _clef_vocabulary(label)
        res = _compute_clefs(positions, vocab, T // 4)
        k = 0
        c = res[0]
        for mi, bar in enumerate(score.bars):
            while k < len(positions) and positions[k][_TIME] <= starts[mi]:
                c = res[k]
                k += 1
            st = bar.staff(label)
            if st is not None:
                st.clef = vocab[c]


def _staff_order(onset_order):
    """Staves in first-onset order, except that the keyboard staves stand in their
    fixed top-to-bottom order (PR above PL) and an unheard PR is restored above PL.
    (The model writes the lower staff first inside an instant; that order carries no
    staff-order information.)"""
    nodes = list(onset_order)
    prio = list(range(len(nodes)))
    if 'PL' in nodes and 'PR' not in nodes:
        nodes.append('PR')
        prio.append(prio[nodes.index('PL')])
    edges = set()
    if 'PR' in nodes and 'PL' in nodes:
        edges.add((nodes.index('PR'), nodes.index('PL')))
    indeg = [0] * len(nodes)
    for a, b in edges:
        indeg[b] += 1
    done = [False] * len(nodes)
    out = []
    for _ in nodes:
        pick = -1
        for i in range(len(nodes)):
            if not done[i] and indeg[i] == 0 and (pick < 0 or prio[i] < prio[pick]):
                pick = i
        done[pick] = True
        out.append(nodes[pick])
        for a, b in edges:
            if a == pick:
                indeg[b] -= 1
    return out


def _rescale(toks, stamps, k):
    """Multiply every tick value by k (a finer unit)."""
    toks = [('D', v * k) if kind == 'D' else (kind, v) for kind, v in toks]
    stamps = {t * k: s for t, s in stamps.items()}
    return toks, stamps


def _pair_ties(score):
    """Tie flags must pair with the neighbouring event of the same staff voice."""
    seqs = {}
    for bar in score.bars:
        for st in bar.staves:
            for ly in st.layers:
                key = (st.label, ly.index)
                lst = seqs.get(key)
                if lst is None:
                    seqs[key] = list(ly.events)
                else:
                    lst.extend(ly.events)
    for key in sorted(seqs):
        evs = seqs[key]
        n = len(evs)
        prev_matched = ()            # note indices of evs[k] tied from evs[k - 1]
        for k in range(n):
            notes = evs[k].notes
            if not notes:
                prev_matched = ()
                continue
            nxt = evs[k + 1].notes if k + 1 < n else ()
            matched = set()
            for an in notes:
                if not an.tie_start:
                    continue
                found = False
                if nxt:
                    m = an.pitch.midi
                    for bi, bn in enumerate(nxt):
                        if bi in matched or not bn.tie_stop or bn.pitch.midi != m:
                            continue
                        matched.add(bi)
                        found = True
                        break
                if not found:
                    an.tie_start = False
            for ni, nt in enumerate(notes):
                if nt.tie_stop and ni not in prev_matched:
                    nt.tie_stop = False
            prev_matched = matched


def read_score(text):
    """Rubato score text -> Score.  Raises ReadError on input it cannot repair."""
    toks, raw_stamps, repairs, T = scan(text)
    chunks = _segment(toks)
    k = 1
    for ch in chunks:
        bot = ch[4]
        if bot > 0 and (T * k) % bot:
            k = k * bot // gcd(T * k, bot)
    if k != 1:
        toks, raw_stamps = _rescale(toks, raw_stamps, k)
        T *= k

    ctx = _Ctx()
    plan = _Plan()
    carry = 'P' if not any(t[0] == 'M' for t in toks) else ''
    # pass 1: collect every note of the piece with its clock ordinals
    # index of the first non-onset token at or after each onset
    nt = len(toks)
    nre = [0] * (nt + 1)
    nre[nt] = nt
    for i in range(nt - 1, -1, -1):
        nre[i] = nre[i + 1] if toks[i][0] == 'N' else i
    staff = carry
    t = 0
    states = []
    for (bidx, a, b, top, bot, key, has) in chunks:
        plan.bar_start = t
        state, mlen, end_staff = _parse_bar(toks, a, b, staff, ctx, plan, nre)
        states.append(state)
        if end_staff:
            staff = end_staff
        if mlen <= 0 and bidx >= 0:
            mlen = _nominal(top, bot, T)
        t += mlen
    for (s, midi), lst in ctx.sustain.items():
        for _, pid in lst:
            ph = plan.notes[pid]
            if not ph.closed:
                ph.off, ph.off_ev, ph.closed = t, plan.cursor, True
    by_staff = {}
    for k, ph in enumerate(plan.notes):
        by_staff.setdefault(ph.staff, []).append(k)
    for s, ids in by_staff.items():
        voices = assign_voices([(plan.notes[k].on, plan.notes[k].off, plan.notes[k].midi,
                                 plan.notes[k].on_ev, plan.notes[k].off_ev) for k in ids], unit=T)
        for k, v in zip(ids, voices):
            plan.notes[k].voice = v
    # pass 2: build the bars with those voices
    score = Score(repairs=repairs)
    has_pickup = bool(chunks) and chunks[0][0] == -1
    if not has_pickup and chunks and chunks[0][0] >= 0 and not chunks[0][6] and \
            not toks[chunks[0][0]][1].startswith('|N/A'):
        has_pickup = True
    number = 0 if has_pickup else 1
    nominal_set = False
    cum = 0
    for (bidx, a, b, top, bot, key, has), state in zip(chunks, states):
        free = bidx >= 0 and toks[bidx][1].startswith('|N/A')
        pre_nominal_bare = bidx >= 0 and not free and not has and not nominal_set
        if has:
            nominal_set = True
        layers = _build_bar(state, ctx, plan)
        mlen = state[4]
        order = []
        for ld in layers:
            if ld[2] and ld[0] not in order:
                order.append(ld[0])
        if not order:
            if bidx < 0 and mlen <= 0:
                continue
            bar = Bar(number, top, bot, key, mlen if mlen > 0 else _nominal(top, bot, T))
            bar.pickup = number == 0 or pre_nominal_bare
            number += 1
            bar.start = cum
            cum += bar.length
            score.bars.append(bar)
            continue
        bar = Bar(number, top, bot, key, 0)
        bar.pickup = number == 0 or pre_nominal_bare
        number += 1
        longest = 0
        for s in order:
            stf = Staff(s)
            for ld in layers:
                if ld[0] != s or not ld[2]:
                    continue
                ly = Layer(ld[1], ld[2])
                longest = max(longest, _llen(ly))
                stf.layers.append(ly)
            bar.staves.append(stf)
        bar.length = max(longest, mlen)
        if bar.length <= 0:
            continue
        bar.start = cum
        cum += bar.length
        score.bars.append(bar)

    score.staff_order = _staff_order(ctx.onset_order)
    rank = {s: k for k, s in enumerate(score.staff_order)}
    for bar in score.bars:
        bar.staves.sort(key=lambda st: rank.get(st.label, 0))
    _tidy_rests(score)
    _tidy_rests(score)
    _restart_times(score)
    _estimate_clefs(score, T)
    _pair_ties(score)

    # rests of secondary voices fill gaps; they are not engraved
    for bar in score.bars:
        for st in bar.staves:
            for ly in st.layers[1:]:
                for ev in ly.events:
                    if not ev.notes:
                        ev.visible = False
    # a voice that is nothing but rests in a bar carries nothing
    for bar in score.bars:
        for st in bar.staves:
            kept = [ly for ly in st.layers if not all(not ev.notes for ev in ly.events)]
            if not kept:
                if len(st.layers) > 1:
                    st.layers = [min(st.layers, key=lambda ly: ly.index)]
                continue
            st.layers = kept
    # a short first bar is a pickup
    if score.bars:
        b0 = score.bars[0]
        if not b0.pickup and b0.number != 0 and 0 < b0.length < _nominal(b0.ts_top, b0.ts_bot, T):
            b0.pickup = True
            for bar in score.bars:
                bar.number -= 1
            repairs['pickup_inferred'] = 1
    F = _to_fractions(score, T)
    score.stamps = {F(t): s for t, s in raw_stamps.items()}
    return score


def _nominal(top, bot, T):
    """Meter length top/bot whole notes in ticks (bot divides T)."""
    return top * (T // bot)


def _to_fractions(score, T):
    """Ticks -> Fractions (whole notes) in the Score handed out."""
    cache = {}

    def F(t):
        f = cache.get(t)
        if f is None:
            f = cache[t] = Fraction(t, T)
        return f
    done = set()
    for bar in score.bars:
        bar.length = F(bar.length)
        bar.start = F(bar.start)
        for st in bar.staves:
            for ly in st.layers:
                for ev in ly.events:
                    if id(ev) in done:
                        continue
                    done.add(id(ev))
                    ev.duration = F(ev.duration)
    return F


def _restart_times(score):
    t = 0
    for bar in score.bars:
        bar.start = t
        t += bar.length


# ─────────────────────────────── midi text ─────────────────────────────────

def read_performance(text):
    """Rubato midi text -> (notes, pedal) with pedal = [(seconds, down: bool)].

    Tokens: ``C4`` onset (uppercase), ``c4`` release, ``<|vel:N|>`` velocity of the
    preceding onset, ``<|t|>`` time stamp of the preceding token(s),
    ``<|CC64:on|>`` / ``<|CC64:off|>`` sustain pedal.  Implementation follows the
    performance parser (see ``_perf``)."""
    return parse_performance(text)


def read_beats(text):
    """Rubato beats text -> [(seconds, is_downbeat)]: ``|`` downbeat, ``*`` beat,
    each followed by its stamp."""
    return parse_beats(text)
