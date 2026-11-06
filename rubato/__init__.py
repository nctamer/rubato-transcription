"""rubato: piano transcription (score, midi, beats) with the Rubato model, in plain torch + numpy.

The model (`Rubato`) needs torch + numpy; the converters (`rubato.convert`: MusicXML, MIDI, beats CSV) need only
the Python standard library, so torch and numpy are imported lazily, on first use of `Rubato`."""

__all__ = ["Rubato", "MODES"]
__version__ = "0.3.0"


def __getattr__(name):
    if name == "Rubato":
        from .api import Rubato
        return Rubato
    if name == "MODES":
        from .prompts import MODES
        return MODES
    raise AttributeError(f"module 'rubato' has no attribute {name!r}")
