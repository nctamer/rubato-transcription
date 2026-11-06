"""Score -> MusicXML 4.0 (partwise), with tempo from the time stamps.

Layout: the two keyboard staves (``PR`` upper, ``PL`` lower) form ONE part
"Piano" with two staves and a brace; voices are numbered 1-4 on staff 1 and 5-8 on
staff 2.  Engraving (beat splits, beams, tuplets, accidentals, grace values) comes
from ``_engrave``.

Tempo (default tempo="events")
--------------------------------
Hidden sound-tempo changes integrate every timestamp interval, across all
voices, at exact metric offsets. A positive initial clock value is encoded
as a hidden playback-only pre-roll measure, so ordinary MusicXML playback
starts the music at its absolute timestamp. No external alignment is needed.
Tempo instructions are unique at each metric position; tied fragments do not
move the clock or require tempo changes to coincide with primary-voice notes.

Visible bar marks (tempo="bars")
---------------------------------------------------------------
Rubato stamps the start of (nearly) every event with its time in seconds, so each
bar has a measured duration.  For bar i with metric length L_i (whole notes, the
bar's actual content length) and stamps s_i (bar start) and s_{i+1} (next bar
start, or the end of the piece):

    qpm_i = 4 * L_i / (s_{i+1} - s_i) * 60          (quarter notes per minute)

A ``<direction><metronome>`` (rounded to an integer) with ``<sound tempo>``
(two decimals) is written on the first bar and on every bar whose qpm differs
from the last written mark by more than ``tempo_threshold`` (relative; default
0.10 = 10 %; 0 = every bar).  Bars without two usable stamps (span <= 1 ms or
missing) keep the previous tempo. The bar's actual length is used so a short
or overfull bar is not misread as a tempo change.
"""
from bisect import bisect_left, bisect_right
from fractions import Fraction
from math import gcd
from xml.sax.saxutils import escape

from ._engrave import engrave

_KEYBOARD = ('PR', 'PL')
_CLEF = {'G2': ('G', 2), 'F4': ('F', 4)}
_ARTIC = {'.': 'staccato', '>': 'accent'}


def _lcm(a, b):
    return a * b // gcd(a, b)


def stamp_lookup(score):
    """score time (Fraction) -> seconds, from the stamps: exact stamp if present,
    else the latest stamp at or before that time (else the first stamp)."""
    st = score.stamps
    unit = 1                      # common denominator of the stamp times
    try:
        for k in st:
            d = k.denominator
            if unit % d:
                unit = unit * d // gcd(unit, d)
    except AttributeError:        # times without numerator/denominator
        return _stamp_lookup_generic(score)
    ik = {k.numerator * (unit // k.denominator): v for k, v in st.items()}
    keys = sorted(ik)
    vals = [ik[k] for k in keys]

    def at(t):
        if not keys:
            return None
        num, den = getattr(t, 'numerator', None), getattr(t, 'denominator', None)
        if den is None:
            x = Fraction(t) * unit    # a non-rational query time (exact)
        elif unit % den == 0:
            x = num * (unit // den)
            v = ik.get(x)
            if v is not None or x in ik:
                return v
        else:
            x = t * unit              # not a stamp time
        lo = bisect_right(keys, x)
        return vals[lo - 1] if lo > 0 else vals[0]
    return at


def _stamp_lookup_generic(score):
    """stamp_lookup for times that are not rationals (e.g. floats)."""
    keys = sorted(score.stamps)
    vals = [score.stamps[k] for k in keys]

    def at(t):
        if not keys:
            return None
        if t in score.stamps:
            return score.stamps[t]
        lo, hi = 0, len(keys)
        while lo < hi:
            mid = (lo + hi) // 2
            if keys[mid] <= t:
                lo = mid + 1
            else:
                hi = mid
        return vals[lo - 1] if lo > 0 else vals[0]
    return at


def bar_tempi(score):
    """qpm per bar (None where it cannot be measured)."""
    at = stamp_lookup(score)
    out = []
    nb = len(score.bars)
    for i, bar in enumerate(score.bars):
        s0 = at(bar.start)
        s1 = at(score.bars[i + 1].start) if i + 1 < nb else at(score.end())
        if s0 is None or s1 is None or s1 - s0 <= 0.001 or bar.length <= 0:
            out.append(None)
            continue
        out.append(4.0 * float(bar.length) / (s1 - s0) * 60.0)
    return out


def tempo_marks(score, threshold=0.10):
    """-> {bar index: qpm} of the marks written (see module doc)."""
    marks, last = {}, None
    for i, q in enumerate(bar_tempi(score)):
        if q is None:
            continue
        if last is None or abs(q - last) > threshold * last:
            marks[i] = q
            last = q
    return marks


def _parts(order):
    """Group staff labels into parts: consecutive keyboard staves share one part."""
    parts = []
    i = 0
    while i < len(order):
        if order[i] in _KEYBOARD:
            j = i
            while j < len(order) and order[j] in _KEYBOARD:
                j += 1
            parts.append(order[i:j])
            i = j
        else:
            parts.append([order[i]])
            i += 1
    return parts


def _timestamp_clock(score):
    """Absolute clock -> initial silence plus one conductor tempo per interval.

    A sound tempo is departure state (InterMo G7/G9). Use ALL metric
    anchors, not just primary-voice attacks; a sustained note can cross them.
    """
    from math import isfinite
    lo, hi = score.bars[0].start, score.end()
    anchors = sorted((x, float(t)) for x, t in score.stamps.items() if lo <= x <= hi)
    if not anchors:
        return 0.0, [], {'adjusted': 0, 'max_error': 0.0}
    if any(not isfinite(t) or t < 0 for _, t in anchors):
        raise ValueError('MusicXML timestamp clock must be finite and nonnegative')
    if anchors[0][0] > lo:
        anchors.insert(0, (lo, 0.0))
    # A single positive-tempo timeline cannot run backwards. Project only
    # contradictory source anchors onto the nearest nondecreasing clock (PAVA).
    # The first/last clock values stay fixed; report every changed anchor below.
    first, last = anchors[0][1], anchors[-1][1]
    if last < first:
        raise ValueError('MusicXML timestamp clock ends before it starts')
    blocks = []
    for i, (_, t) in enumerate(anchors[1:-1], 1):
        blocks.append([i, i + 1, min(last, max(first, t)), 1])
        while len(blocks) > 1 and blocks[-2][2] > blocks[-1][2]:
            b = blocks.pop(); a = blocks.pop(); count = a[3] + b[3]
            blocks.append([a[0], b[1], (a[2] * a[3] + b[2] * b[3]) / count, count])
    fitted = list(anchors)
    for lo_i, hi_i, value, _ in blocks:
        for i in range(lo_i, hi_i): fitted[i] = (anchors[i][0], value)
    errors = [abs(t - u) for (_, t), (_, u) in zip(anchors, fitted)]
    audit = {'adjusted': sum(e > 1e-9 for e in errors), 'max_error': max(errors, default=0.0)}
    delay = fitted[0][1]
    anchors = fitted
    marks = []
    for (x, t0), (y, t1) in zip(anchors, anchors[1:]):
        if t1 < t0 - 1e-9:
            raise ValueError(f'MusicXML timestamp clock is not monotonic: {x}:{t0} -> {y}:{t1}')
        # Distinct metric positions can have the same rounded timestamp.
        # A 1 ns interval approximates that zero wall interval without infinity.
        qpm = 240.0 * float(y - x) / max(t1 - t0, 1e-9)
        marks.append((x, qpm))
    if not marks:
        marks.append((lo, 120.0))
    return delay, marks, audit


def to_musicxml(score, title=None, tempo="events", tempo_threshold=0.10):
    """Score -> MusicXML text.  tempo: "events" (default: hidden per-event <sound tempo>, timestamp-based playback timing),
    "bars" (visible metronome marks per bar on a > tempo_threshold change) or False / None (no tempo)."""
    if tempo is True:
        tempo = "events"
    if not score.bars or not score.staff_order:
        import copy
        from ._model import Bar
        score = copy.copy(score)
        score.staff_order = list(score.staff_order) or list(_KEYBOARD)
        score.bars = list(score.bars) or [Bar(1, 4, 4, 0, Fraction(1))]
    sc, rows = engrave(score)
    order = sc.staff_order
    marks = tempo_marks(sc, tempo_threshold) if tempo == "bars" else {}
    delay, clock, clock_audit = _timestamp_clock(sc) if tempo == "events" else (0.0, [], {})

    clock_positions = [x for x, _ in clock]

    # divisions: LCM of the quarter-note denominators of every bar length and duration
    dens = set()
    for bar, row in zip(sc.bars, rows):
        dens.add((bar.length.numerator, bar.length.denominator))
        for sb in row:
            for v in sb.voices:
                for it in v:
                    ev = it.ev
                    if not ev.grace:
                        d = ev.duration
                        dens.add((d.numerator, d.denominator))
    for x, _ in clock:
        dens.add((x.numerator, x.denominator))
    div = 1
    for a, b in sorted(dens):
        q = b // gcd(a * 4, b)               # (Fraction(a, b) * 4).denominator
        div = _lcm(div, q)
    dcache = {}

    def dur(d):
        key = (d.numerator, d.denominator)
        r = dcache.get(key)
        if r is None:
            q = d * 4 * div
            assert q.denominator == 1
            r = dcache[key] = q.numerator
        return r

    out = ['<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
           '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
           '"http://www.musicxml.org/dtds/partwise.dtd">',
           '<score-partwise version="4.0">']
    if title:
        out.append(f'  <work><work-title>{escape(title)}</work-title></work>')
    out.append('  <identification><encoding><software>rubato</software></encoding>')
    if clock:
        out.append('    <miscellaneous>')
        for name, value in (('initial-seconds', delay), ('adjusted-anchors', clock_audit['adjusted']), ('max-anchor-adjustment-seconds', clock_audit['max_error'])):
            out.append(f'      <miscellaneous-field name="rubato-playback-{name}">{value}</miscellaneous-field>')
        out.append('    </miscellaneous>')
    out.append('  </identification>')
    parts = _parts(order)
    out.append('  <part-list>')
    for pi, labels in enumerate(parts):
        name = 'Piano' if labels[0] in _KEYBOARD else labels[0]
        out.append(f'    <score-part id="P{pi + 1}"><part-name>{escape(name)}</part-name>'
                   f'<score-instrument id="P{pi + 1}-I1"><instrument-name>{escape(name)}</instrument-name>'
                   f'</score-instrument><midi-instrument id="P{pi + 1}-I1"><midi-channel>{pi + 1}</midi-channel>'
                   f'<midi-program>1</midi-program></midi-instrument></score-part>')
    out.append('  </part-list>')

    for pi, labels in enumerate(parts):
        out.append(f'  <part id="P{pi + 1}">')
        prev_key = prev_ts = None
        if delay > 0:
            # Explicit silence is part of the playback carrier, not a fitted
            # reader offset. Hide the rest and barline; retain the musical bars.
            out.append('    <measure number="-1" implicit="yes" non-controlling="yes" width="0" id="rubato-playback-preroll-%d">' % (pi + 1))
            attrs = f'<divisions>{div}</divisions><time print-object="no"><beats>1</beats><beat-type>4</beat-type></time>'
            if len(labels) > 1:
                attrs += f'<staves>{len(labels)}</staves>'
            out.append('      <attributes>' + attrs + '</attributes>')
            if pi == 0:
                out.append(f'      <direction print-object="no"><direction-type><words/></direction-type><sound tempo="{60.0 / delay:.12f}"/></direction>')
            staff = '<staff>1</staff>' if len(labels) > 1 else ''
            out.append(f'      <note print-object="no" print-spacing="no"><rest/><duration>{div}</duration><voice>1</voice><type>quarter</type>{staff}</note>')
            out.append('      <barline location="right"><bar-style>none</bar-style></barline>')
            out.append('    </measure>')
        for mi, (bar, row) in enumerate(zip(sc.bars, rows)):
            attrs = {'number': str(bar.number)}
            if bar.pickup:
                attrs['implicit'] = 'yes'
            out.append('    <measure ' + ' '.join(f'{k}="{v}"' for k, v in attrs.items()) + '>')
            sbs = [sb for sb in row if sb.label in labels]
            a = []
            if mi == 0:
                a.append(f'<divisions>{div}</divisions>')
            if bar.key != prev_key:
                a.append(f'<key><fifths>{bar.key}</fifths></key>')
            if (bar.ts_top, bar.ts_bot) != prev_ts:
                a.append(f'<time><beats>{bar.ts_top}</beats><beat-type>{bar.ts_bot}</beat-type></time>')
            if mi == 0 and len(labels) > 1:
                a.append(f'<staves>{len(labels)}</staves>')
            for si, sb in enumerate(sbs):
                clef = sb.clef
                if mi == 0 and clef not in _CLEF:          # a staff that never sounds
                    clef = 'F4' if sb.label == 'PL' else 'G2'
                if (sb.clef_changed or mi == 0) and clef in _CLEF:
                    sign, line = _CLEF[clef]
                    num = f' number="{si + 1}"' if len(labels) > 1 else ''
                    a.append(f'<clef{num}><sign>{sign}</sign><line>{line}</line></clef>')
            prev_key, prev_ts = bar.key, (bar.ts_top, bar.ts_bot)
            if a:
                out.append('      <attributes>' + ''.join(a) + '</attributes>')
            if pi == 0 and mi in marks:
                q = marks[mi]
                out.append('      <direction placement="above"><direction-type><metronome parentheses="no">'
                           f'<beat-unit>quarter</beat-unit><per-minute>{int(round(q))}</per-minute></metronome>'
                           f'</direction-type><staff>1</staff><sound tempo="{q:.2f}"/></direction>')
            if pi == 0 and clock:
                # Walk the conductor cursor to each anchor, then restore it.
                # MuseScore ignores the offset on sound-only directions; using
                # the real cursor also agrees with InterMo's source adapter.
                cursor = 0
                lo = bisect_left(clock_positions, bar.start)
                hi = bisect_left(clock_positions, bar.start + bar.length)
                for x, q in clock[lo:hi]:
                    target = dur(x - bar.start)
                    if target > cursor:
                        out.append(f'      <forward><duration>{target - cursor}</duration></forward>')
                    out.append(f'      <direction print-object="no"><direction-type><words/></direction-type><sound tempo="{q:.12f}"/></direction>')
                    cursor = target
                if cursor:
                    out.append(f'      <backup><duration>{cursor}</duration></backup>')
            first_voice = True
            for si, sb in enumerate(sbs):
                for cv, items in enumerate(sb.voices):
                    if not first_voice:
                        out.append(f'      <backup><duration>{dur(bar.length)}</duration></backup>')
                    first_voice = False
                    voice = si * 4 + cv + 1
                    for k, it in enumerate(items):
                        out.extend(_note_xml(it, voice, si + 1 if len(labels) > 1 else None, dur))
            if mi + 1 == len(sc.bars):
                out.append('      <barline location="right"><bar-style>light-heavy</bar-style></barline>')
            out.append('    </measure>')
        out.append('  </part>')
    out.append('</score-partwise>')
    return '\n'.join(out) + '\n'


_BEAM = {'start': 'begin', 'continue': 'continue', 'stop': 'end'}


_PITCH_XML = {}


def _pitch_xml(p):
    key = (p.step, p.alter, p.octave)
    r = _PITCH_XML.get(key)
    if r is None:
        alter = f'<alter>{p.alter}</alter>' if p.alter else ''
        r = _PITCH_XML[key] = f'<pitch><step>{p.step}</step>{alter}<octave>{p.octave}</octave></pitch>'
    return r


def _note_xml(it, voice, staff, dur):
    ev = it.ev
    sp = it.spelling
    notes = ev.notes
    grace = ev.grace
    lines = []
    members = it.order if notes else (None,)
    # parts shared by every member of the chord
    dur_x = '' if grace else f'<duration>{dur(ev.duration)}</duration>'
    type_x = f'<voice>{voice}</voice><type>{sp.type}</type>' + '<dot/>' * sp.dots
    tm_x = (f'<time-modification><actual-notes>{sp.actual}</actual-notes>'
            f'<normal-notes>{sp.normal}</normal-notes></time-modification>'
            if sp.actual != sp.normal and not grace else '')
    staff_x = f'<staff>{staff}</staff>' if staff is not None else ''
    head = '      <note>' if (it.visible or notes) else '      <note print-object="no">'
    if grace:
        head += '<grace slash="yes"/>'
    for k, ni in enumerate(members):
        x = [head]
        if k > 0:
            x.append('<chord/>')
        n = notes[ni] if ni is not None else None
        if n is None:
            x.append('<rest/>')
        else:
            x.append(_pitch_xml(n.pitch))
        x.append(dur_x)
        if n is not None:
            if n.tie_stop:
                x.append('<tie type="stop"/>')
            if n.tie_start:
                x.append('<tie type="start"/>')
        x.append(type_x)
        if n is not None and it.accidental[ni]:
            x.append(f'<accidental>{it.accidental[ni]}</accidental>')
        x.append(tm_x)
        x.append(staff_x)
        if k == 0:
            for lv, b in enumerate(it.beams):
                if b == 'partial':
                    b = 'backward hook' if _is_last_in_beam(it, lv) else 'forward hook'
                else:
                    b = _BEAM[b]
                x.append(f'<beam number="{lv + 1}">{b}</beam>')
        nots = []
        if n is not None:
            if n.tie_stop:
                nots.append('<tied type="stop"/>')
            if n.tie_start:
                nots.append('<tied type="start"/>')
        if k == 0 and it.tuplet is not None and not grace:
            num = f'<tuplet-actual><tuplet-number>{it.tuplet_actual}</tuplet-number></tuplet-actual>'
            if it.tuplet in ('start', 'startStop'):
                nots.append(f'<tuplet type="start" number="1" bracket="yes">{num}</tuplet>')
            if it.tuplet in ('stop', 'startStop'):
                nots.append('<tuplet type="stop" number="1"/>')
        if k == 0 and ev.articulations:
            arts = ''.join(f'<{_ARTIC[a]}/>' for a in ev.articulations if a in _ARTIC)
            if arts:
                nots.append(f'<articulations>{arts}</articulations>')
        if n is not None and n.trill:
            acc = it.trill_acc[ni] if it.trill_acc else ''
            am = f'<accidental-mark>{acc}</accidental-mark>' if acc else ''
            nots.append(f'<ornaments><trill-mark/>{am}</ornaments>')
        if nots:
            x.append('<notations>' + ''.join(nots) + '</notations>')
        x.append('</note>')
        lines.append(''.join(x))
    return lines


def _is_last_in_beam(it, level):
    return it.beams[0] == 'stop' if it.beams else False
