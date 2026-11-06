"""Standard MIDI files (type 1, written with ``struct``).

* ``performance_to_midi(text)`` -- midi-mode text -> notes with velocities and the
  sustain pedal (CC64 127 / 0).  Events retain their parsed ordering and 1 ms timing resolution.
* ``score_to_midi(score)`` -- a read score -> notes at their stamped times: a note
  sounds from the stamp of its onset to the stamp of its end (a tied chain is one
  note).  A position without its own stamp takes the time interpolated linearly
  between the nearest stamped positions.  Grace notes sound 30 ms each just
  before their position, velocity 80 (the score carries no dynamics).  Track 0
  carries the tempo (60 qpm, 1 tick = 1 ms) and marker/text events "downbeat" /
  "beat" for the bar lines and the beat-grid points (see ``beats``).
"""
import struct
from bisect import bisect_left
from math import gcd

from ._perf import perf_events

PPQ = 1000          # with 60 qpm: one tick = one millisecond
GRACE_MS = 30
VELOCITY = 80


def _vlq(n):
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append(0x80 | (n & 0x7F))
        n >>= 7
    return bytes(reversed(out))


def _track(events):
    """events: [(tick, order, bytes)] -> MTrk chunk."""
    events = sorted(events, key=lambda e: (e[0], e[1]))
    body = bytearray()
    prev = 0
    for tick, _, data in events:
        body += _vlq(max(0, tick - prev))
        body += data
        prev = max(prev, tick)
    body += b'\x00\xff\x2f\x00'
    return b'MTrk' + struct.pack('>I', len(body)) + bytes(body)


def _meta(kind, payload):
    return bytes([0xFF, kind]) + _vlq(len(payload)) + payload


def write_smf(tracks):
    """tracks: list of event lists (see _track); track 0 gets the tempo."""
    head = b'MThd' + struct.pack('>IHHH', 6, 1, len(tracks), PPQ)
    return head + b''.join(_track(t) for t in tracks)


def _tempo_track(markers=()):
    ev = [(0, 0, _meta(0x51, struct.pack('>I', 1000000)[1:]))]
    for t, label in markers:
        tick = max(0, int(round(t * 1000)))
        lab = label.encode()
        ev.append((tick, 1, _meta(0x01, lab)))
        ev.append((tick, 2, _meta(0x06, lab)))
    return ev


def performance_to_midi(text):
    """midi-mode text -> SMF bytes (one piano track, channel 0)."""
    ev = []
    for k, (tick, kind, pitch, vel) in enumerate(perf_events(text)):
        if kind == 'on':
            data = bytes([0x90, pitch, vel])
        elif kind == 'off':
            data = bytes([0x80, pitch, 0])
        elif kind == 'ped_on':
            data = bytes([0xB0, 64, 127])
        else:
            data = bytes([0xB0, 64, 0])
        ev.append((tick, k, data))
    return write_smf([_tempo_track(), ev])


class TimeMap:
    """score time (Fraction, whole notes) -> seconds, from the score's stamps."""

    def __init__(self, score):
        self.keys = sorted(score.stamps)
        self.vals = [score.stamps[k] for k in self.keys]
        # stamps must not run backwards in score time: keep the running maximum
        for i in range(1, len(self.vals)):
            if self.vals[i] < self.vals[i - 1]:
                self.vals[i] = self.vals[i - 1]
        self.exact = dict(zip(self.keys, self.vals))

    def __call__(self, t):
        if not self.keys:
            return float(t) * 2.0            # no stamps: 120 qpm
        if t in self.exact:
            return self.exact[t]
        lo, hi = 0, len(self.keys)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.keys[mid] < t:
                lo = mid + 1
            else:
                hi = mid
        if lo == 0:
            return self.vals[0]
        if lo == len(self.keys):
            return self.vals[-1]
        k0, k1 = self.keys[lo - 1], self.keys[lo]
        v0, v1 = self.vals[lo - 1], self.vals[lo]
        return v0 + float((t - k0) / (k1 - k0)) * (v1 - v0)


def _unit(score):
    """A common denominator of every stamp time, bar start and event duration."""
    unit = 1
    dens = {k.denominator for k in score.stamps}
    for bar in score.bars:
        dens.add(bar.start.denominator)
        for st in bar.staves:
            for ly in st.layers:
                for ev in ly.events:
                    dens.add(ev.duration.denominator)
    for d in dens:
        if unit % d:
            unit = unit * d // gcd(unit, d)
    return unit


class _TickMap:
    """TimeMap on integer ticks (1/unit whole note); the same floats as TimeMap."""

    def __init__(self, score, unit):
        st = score.stamps
        ik = {k.numerator * (unit // k.denominator): v for k, v in st.items()}
        self.keys = sorted(ik)
        self.vals = [ik[k] for k in self.keys]
        for i in range(1, len(self.vals)):
            if self.vals[i] < self.vals[i - 1]:
                self.vals[i] = self.vals[i - 1]
        self.exact = dict(zip(self.keys, self.vals))
        self.unit = unit

    def __call__(self, t):
        keys = self.keys
        if not keys:
            return t / self.unit * 2.0
        v = self.exact.get(t)
        if v is not None:
            return v
        lo = bisect_left(keys, t)
        if lo == 0:
            return self.vals[0]
        if lo == len(keys):
            return self.vals[-1]
        k0, k1 = keys[lo - 1], keys[lo]
        v0, v1 = self.vals[lo - 1], self.vals[lo]
        return v0 + (t - k0) / (k1 - k0) * (v1 - v0)



def _score_notes_generic(score):
    """score_notes for times that are not rationals (e.g. floats)."""
    tm = TimeMap(score)
    out = []
    chains = {}        # (staff, layer index, midi) -> index in out
    for bar in score.bars:
        for st in bar.staves:
            for ly in st.layers:
                t = bar.start
                evs = ly.events
                k = 0
                while k < len(evs):
                    ev = evs[k]
                    if ev.grace:
                        j = k
                        while j < len(evs) and evs[j].grace:
                            j += 1
                        run = evs[k:j]
                        host = tm(t)
                        for gi, g in enumerate(run):
                            on = max(0.0, host - (len(run) - gi) * GRACE_MS / 1000.0)
                            for n in g.notes:
                                out.append([on, on + GRACE_MS / 1000.0, n.pitch.midi, st.label])
                        k = j
                        continue
                    end = t + ev.duration
                    for n in ev.notes:
                        key = (st.label, ly.index, n.pitch.midi)
                        if n.tie_stop and key in chains:
                            out[chains[key]][1] = tm(end)
                        else:
                            out.append([tm(t), tm(end), n.pitch.midi, st.label])
                            chains[key] = len(out) - 1
                        if not n.tie_start:
                            chains.pop(key, None)
                    t = end
                    k += 1
    for o in out:
        if o[1] <= o[0]:
            o[1] = o[0] + 0.001
    out.sort(key=lambda o: (o[0], o[2]))
    return [tuple(o) for o in out]


def score_notes(score):
    """-> [(onset s, offset s, midi, staff label)] with tied chains merged."""
    try:
        unit = _unit(score)
    except AttributeError:            # times without numerator/denominator
        return _score_notes_generic(score)
    tm = _TickMap(score, unit)
    out = []
    chains = {}        # (staff, layer index, midi) -> index in out
    for bar in score.bars:
        b0 = bar.start.numerator * (unit // bar.start.denominator)
        for st in bar.staves:
            for ly in st.layers:
                t = b0
                evs = ly.events
                k = 0
                while k < len(evs):
                    ev = evs[k]
                    if ev.grace:
                        j = k
                        while j < len(evs) and evs[j].grace:
                            j += 1
                        run = evs[k:j]
                        host = tm(t)
                        for gi, g in enumerate(run):
                            on = max(0.0, host - (len(run) - gi) * GRACE_MS / 1000.0)
                            for n in g.notes:
                                out.append([on, on + GRACE_MS / 1000.0, n.pitch.midi, st.label])
                        k = j
                        continue
                    d = ev.duration
                    end = t + d.numerator * (unit // d.denominator)
                    for n in ev.notes:
                        key = (st.label, ly.index, n.pitch.midi)
                        if n.tie_stop and key in chains:
                            out[chains[key]][1] = tm(end)
                        else:
                            out.append([tm(t), tm(end), n.pitch.midi, st.label])
                            chains[key] = len(out) - 1
                        if not n.tie_start:
                            chains.pop(key, None)
                    t = end
                    k += 1
    for o in out:
        if o[1] <= o[0]:
            o[1] = o[0] + 0.001
    out.sort(key=lambda o: (o[0], o[2]))
    return [tuple(o) for o in out]


def score_to_midi(score, beats=True):
    """Score -> SMF bytes: tempo/marker track + one track per staff (channel 0)."""
    from .beats import score_beats
    markers = []
    if beats:
        for t, down, src in score_beats(score, interpolated=False):
            markers.append((t, 'downbeat' if down else 'beat'))
    tracks = [_tempo_track(markers)]
    notes = score_notes(score)
    for label in score.staff_order:
        ev = []
        k = 0
        for on, off, midi, lab in notes:
            if lab != label:
                continue
            a, b = int(round(on * 1000)), int(round(off * 1000))
            b = max(b, a + 1)
            ev.append((a, 2 * k + 1, bytes([0x90, midi, VELOCITY])))
            ev.append((b, 2 * k, bytes([0x80, midi, 0])))
            k += 1
        tracks.append(ev)
    return write_smf(tracks)
