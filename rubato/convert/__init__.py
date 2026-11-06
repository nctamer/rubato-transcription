"""Convert Rubato's text output to MusicXML, MIDI and beat CSV (standard library only).

Rubato writes one of three texts, depending on the task it was asked for:

  score   ``|3/4k-4 PL: A-3 C4 PR: E-4<|1.09|> 1/8 e-4 A-4<|1.44|> ...``
  midi    ``C4<|vel:63|><|0.91|> c4<|1.01|> ...``
  beats   ``|<|1.09|> *<|1.69|> *<|2.28|> ...``

Each reader accepts exactly that text (nothing else is interpreted; unknown
``<|...|>`` tokens are stripped and counted in ``Score.repairs``).

    from rubato import convert
    xml  = convert.score_to_musicxml(text)          # str
    mid  = convert.score_to_midi(text)              # bytes (SMF type 1)
    csv  = convert.score_to_beats_csv(text)         # str: time,is_downbeat,source
    mid  = convert.midi_to_midi(text)               # bytes, velocities + sustain pedal
    csv  = convert.beats_to_csv(text)               # str
    convert.convert_file('in.txt', 'out.musicxml', mode='score')

Command line: ``python -m rubato.convert MODE INPUT OUTPUT`` (output format
from the extension: .musicxml/.xml, .mid/.midi, .csv).
"""
from .reader import ReadError, read_beats, read_performance, read_score
from . import beats as _beats
from . import midi as _midi
from . import musicxml as _musicxml

__all__ = ['read_score', 'read_performance', 'read_beats', 'ReadError',
           'score_to_musicxml', 'score_to_midi', 'score_to_beats_csv',
           'midi_to_midi', 'beats_to_csv', 'convert_text', 'convert_file']


def _score(text_or_score):
    return read_score(text_or_score) if isinstance(text_or_score, str) else text_or_score


def score_to_musicxml(text, title=None, tempo="events", tempo_threshold=0.10):
    """Score text (or a Score from ``read_score``) -> MusicXML string.

    tempo: "events" (default) encodes initial silence in the file and writes a hidden
    conductor tempo clock from all timestamp anchors, with no external alignment; "bars" writes
    visible metronome marks on bars whose tempo moves by more than ``tempo_threshold``
    (relative) from the last mark; False / None writes no tempo."""
    return _musicxml.to_musicxml(_score(text), title=title, tempo=tempo, tempo_threshold=tempo_threshold)


def score_to_midi(text, beat_markers=True):
    """Score text -> MIDI bytes; notes at their stamped times, beat/downbeat markers."""
    return _midi.score_to_midi(_score(text), beats=beat_markers)


def score_to_beats_csv(text, interpolated=True):
    """Score text -> CSV of downbeats (bar lines) and beats derived from the meter."""
    return _beats.to_csv(_beats.score_beats(_score(text), interpolated=interpolated))


def midi_to_midi(text):
    """Midi-mode text -> MIDI bytes (velocities, sustain pedal)."""
    return _midi.performance_to_midi(text)


def beats_to_csv(text):
    """Beats-mode text -> CSV."""
    return _beats.to_csv(_beats.text_beats(text))


_EXT = {'.musicxml': 'musicxml', '.xml': 'musicxml', '.mid': 'midi', '.midi': 'midi', '.csv': 'csv'}


def convert_text(text, mode, fmt, **kw):
    """mode: 'score' | 'midi' | 'beats';  fmt: 'musicxml' | 'midi' | 'csv'.  -> str or bytes."""
    table = {('score', 'musicxml'): score_to_musicxml, ('score', 'midi'): score_to_midi,
             ('score', 'csv'): score_to_beats_csv, ('midi', 'midi'): midi_to_midi,
             ('beats', 'csv'): beats_to_csv}
    fn = table.get((mode, fmt))
    if fn is None:
        raise ValueError(f'cannot convert {mode} text to {fmt} '
                         f'(supported: {", ".join(f"{m}->{f}" for m, f in table)})')
    return fn(text, **kw)


def convert_file(src, dst, mode, **kw):
    import os
    fmt = _EXT.get(os.path.splitext(dst)[1].lower())
    if fmt is None:
        raise ValueError(f'unknown output extension for {dst!r} (.musicxml .xml .mid .midi .csv)')
    with open(src, encoding='utf-8') as fh:
        text = fh.read()
    out = convert_text(text, mode, fmt, **kw)
    with open(dst, 'wb' if isinstance(out, bytes) else 'w') as fh:
        fh.write(out)
    return dst
