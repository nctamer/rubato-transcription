"""Plain data model shared by the converters (standard library only).

All musical times are ``fractions.Fraction`` in WHOLE notes (a quarter note is 1/4).
"""
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional

_PC = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
_LETTERS = 'CDEFGAB'


@dataclass
class Pitch:
    step: str          # 'A'..'G'
    alter: int         # -2..2 (spelled accidental)
    octave: int        # scientific octave (C4 = middle C)

    @property
    def midi(self) -> int:
        return (self.octave + 1) * 12 + _PC[self.step] + self.alter

    @property
    def diatonic(self) -> int:
        return self.octave * 7 + _LETTERS.index(self.step)

    def text(self) -> str:
        acc = '#' * self.alter if self.alter > 0 else '-' * (-self.alter)
        return f'{self.step}{acc}{self.octave}'


_PARSED = {}          # token -> (step, alter, octave, rest) or None


def _parse(tok):
    r = _PARSED.get(tok)
    if r is not None or tok in _PARSED:
        return r
    r = None
    if tok and tok[0].upper() in _PC:
        i, alter = 1, 0
        while i < len(tok) and tok[i] in '#-':
            alter += 1 if tok[i] == '#' else -1
            i += 1
        if i < len(tok) and tok[i].isdigit():
            r = (tok[0].upper(), alter, int(tok[i]), tok[i + 1:])
    if len(_PARSED) > 100000:
        _PARSED.clear()
    _PARSED[tok] = r
    return r


def parse_pitch(tok: str):
    """'E-4' / 'C#5' / 'b--3' -> (Pitch, rest-of-token) or None.

    Letter, then any run of '#'/'-' accidentals, then ONE octave digit; whatever
    follows is returned unparsed (suffixes)."""
    r = _parse(tok)
    if r is None:
        return None
    return Pitch(r[0], r[1], r[2]), r[3]


_MIDI = {}


def pitch_midi(tok):
    """MIDI number of a pitch token, None if it is not one (see parse_pitch)."""
    m = _MIDI.get(tok)
    if m is None and tok not in _MIDI:
        try:
            r = _parse(tok)
        except ValueError:
            r = None
        m = None if r is None else (r[2] + 1) * 12 + _PC[r[0]] + r[1]
        if len(_MIDI) > 100000:
            _MIDI.clear()
        _MIDI[tok] = m
    return m


@dataclass
class Note:
    pitch: Pitch
    tie_start: bool = False      # sustains into the next event of its layer
    tie_stop: bool = False       # continues the previous event of its layer
    trill: bool = False


@dataclass
class Event:
    duration: Fraction
    notes: List[Note] = field(default_factory=list)   # empty = rest
    visible: bool = True
    grace: bool = False
    articulations: List[str] = field(default_factory=list)   # '.' staccato, '>' accent

    @property
    def is_rest(self) -> bool:
        return not self.notes


@dataclass
class Layer:
    index: int
    events: List[Event] = field(default_factory=list)

    def length(self) -> Fraction:
        return sum((e.duration for e in self.events), Fraction(0))


@dataclass
class Staff:
    label: str
    layers: List[Layer] = field(default_factory=list)
    clef: str = ''               # 'G2' (treble) or 'F4' (bass)


@dataclass
class Bar:
    number: int
    ts_top: int
    ts_bot: int
    key: int
    length: Fraction                     # actual extent (may differ from the meter)
    staves: List[Staff] = field(default_factory=list)
    pickup: bool = False
    start: Fraction = Fraction(0)        # absolute start, whole notes

    @property
    def nominal(self) -> Fraction:
        return Fraction(self.ts_top, self.ts_bot)

    def staff(self, label: str) -> Optional[Staff]:
        for st in self.staves:
            if st.label == label:
                return st
        return None


@dataclass
class Score:
    bars: List[Bar] = field(default_factory=list)
    staff_order: List[str] = field(default_factory=list)
    stamps: Dict[Fraction, float] = field(default_factory=dict)   # score time -> seconds (raw, as written)
    repairs: Dict[str, int] = field(default_factory=dict)          # tolerance repairs applied while reading

    def end(self) -> Fraction:
        return self.bars[-1].start + self.bars[-1].length if self.bars else Fraction(0)
