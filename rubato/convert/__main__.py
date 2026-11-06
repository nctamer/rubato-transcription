"""python -m rubato.convert MODE INPUT OUTPUT  (MODE: score | midi | beats)"""
import argparse
import sys

from . import convert_file


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m rubato.convert',
                                 description='Convert Rubato text output to MusicXML / MIDI / CSV.')
    ap.add_argument('mode', choices=['score', 'midi', 'beats'], help='which Rubato output the text is')
    ap.add_argument('input', help='text file with Rubato output')
    ap.add_argument('output', help='.musicxml/.xml, .mid/.midi or .csv')
    ap.add_argument('--no-tempo', action='store_true', help='MusicXML: no metronome marks')
    ap.add_argument('--tempo-threshold', type=float, default=0.10,
                    help='MusicXML: relative tempo change that triggers a new mark (default 0.10)')
    a = ap.parse_args(argv)
    kw = {}
    if a.mode == 'score' and a.output.lower().endswith(('.musicxml', '.xml')):
        kw = dict(tempo=not a.no_tempo, tempo_threshold=a.tempo_threshold)
    convert_file(a.input, a.output, a.mode, **kw)
    return 0


if __name__ == '__main__':
    sys.exit(main())
