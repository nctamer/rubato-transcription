"""Parsers for Rubato's midi-mode and beats-mode text.

Events are interpreted as follows:

midi mode: ``C4<|vel:60|><|0.91|> c4<|1.01|> <|CC64:on|><|3.20|>`` -- pitch tokens
  (``[A-G]#?\\d+`` onset, lowercase release) and pedal tokens are queued until the
  next stamp ``<|seconds|>``, which gives them all that time; ``<|vel:N|>`` sets the
  velocity of the latest queued onset (default 80).  Events are ordered by time,
  then release < pedal up < onset < pedal down.  A release only ends a sounding
  pitch; a note never released is dropped.  Times are kept at 1 ms resolution
  (truncated).
beats mode: ``|<|1.09|> *<|1.69|>`` -- each stamp ``<|s.ss|>`` (with a fractional
  part) is a beat if the last token before it starts with ``*``, a downbeat (and
  beat) if it starts with ``|``.
"""
import re
from dataclasses import dataclass

DEFAULT_VELOCITY = 80
_SEMI = {'A': 9, 'B': 11, 'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7}


@dataclass
class PerfNote:
    pitch: int
    onset: float
    offset: float
    velocity: int


def _pitch(tok):
    letter = tok[0].upper()
    i, acc = 1, 0
    if tok[1] == '#':
        acc, i = 1, 2
    elif tok[1] in 'bB':
        return 60                         # MIDI-mode pitch names do not include flats
    return 12 * (int(tok[i:]) + 1) + _SEMI[letter] + acc


_RANK = {'off': 0, 'ped_off': 1, 'on': 2, 'ped_on': 3}


def perf_events(text):
    """-> [(ms, kind, pitch, velocity)] sorted; kind in on/off/ped_on/ped_off."""
    events, pending = [], []
    s = text
    i, n = 0, len(s)
    while i < n:
        if s.startswith('<|CC64:on|>', i):
            pending.append(['ped_on', 64, -1])
            i += 11
            continue
        if s.startswith('<|CC64:off|>', i):
            pending.append(['ped_off', 64, -1])
            i += 12
            continue
        if s.startswith('<|vel:', i):
            m = re.match(r'(\d+)\|>', s[i + 6:i + 40])
            if m:
                v = min(int(m.group(1)[:4]), 1000)
                vel = max(0, min(127, v))
                for p in reversed(pending):
                    if p[0] == 'on' and p[2] < 0:
                        p[2] = vel
                        break
                i += 6 + len(m.group(1)) + 2
                continue
        if s.startswith('<|', i):
            m = re.match(r'(\d+(?:\.\d+)?)\|>', s[i + 2:i + 64])
            if m:
                t = float(m.group(1))
                for kind, pitch, vel in pending:
                    v = (DEFAULT_VELOCITY if vel < 0 else vel) if kind == 'on' else 0
                    events.append((t, kind, pitch, v))
                pending = []
                i += 2 + len(m.group(1)) + 2
                continue
        c = s[i]
        if 'A' <= c <= 'G' or 'a' <= c <= 'g':
            m = re.match(r'[A-Ga-g](?:[#b]\d+|\d+)', s[i:i + 12])
            if m and m.group(0)[1:2] in ('#', 'b') and not m.group(0)[2:].isdigit():
                m = None
            if m:
                tok = m.group(0)
                pending.append(['on' if c.isupper() else 'off', _pitch(tok), -1])
                i += len(tok)
                continue
        i += 1
    events.sort(key=lambda e: (e[0], _RANK[e[1]]))          # stable
    out = []
    active = set()
    last = 0
    for t, kind, pitch, vel in events:
        ticks = int(t * 1000.0)
        tick = last + max(0, ticks - last)
        if kind == 'on':
            if not 0 <= pitch <= 127:
                raise ValueError(f'note number {pitch} out of MIDI range at {t} s')
            active.add(pitch)
        elif kind == 'off':
            if pitch not in active:
                continue
            active.discard(pitch)
        out.append((tick, kind, pitch, vel))
        last = ticks
    return out


def parse_performance(text):
    """-> (notes [PerfNote], pedal [(seconds, down)])  -- pairing as a MIDI reader would."""
    ev = perf_events(text)
    open_, notes, pedal = {}, [], []
    for tick, kind, pitch, vel in ev:
        if kind == 'on' and vel > 0:
            open_.setdefault(pitch, []).append((tick, vel))
        elif kind in ('off', 'on'):
            lst = open_.get(pitch)
            if not lst:
                continue
            keep, closed = [], False
            for st, v in lst:
                if st == tick:
                    keep.append((st, v))
                    continue
                notes.append(PerfNote(pitch, st / 1000.0, tick / 1000.0, v))
                closed = True
            if closed and keep:
                open_[pitch] = keep
            else:
                del open_[pitch]
        elif kind == 'ped_on':
            pedal.append((tick / 1000.0, True))
        elif kind == 'ped_off':
            pedal.append((tick / 1000.0, False))
    notes.sort(key=lambda x: (x.onset, x.pitch))
    return notes, pedal


_STAMP = re.compile(r'<\|(\d+)\.(\d+)\|>')


def parse_beats(text):
    """-> [(seconds, is_downbeat)] sorted by time."""
    out = []
    prev_end = 0
    for m in _STAMP.finditer(text):
        prefix = text[prev_end:m.start()].rstrip()
        prev_end = m.end()
        if not prefix:
            continue
        cut = max(prefix.rfind(' '), prefix.rfind('\t'), prefix.rfind('\n'))
        first = prefix[cut + 1]
        if first not in '*|':
            continue
        out.append((float(m.group(1) + '.' + m.group(2)), first == '|'))
    out.sort(key=lambda x: (x[0], not x[1]))
    return out
