# Rubato

**Transcribing Piano Music with Timestamps** · **NeurIPS 2026 Spotlight**

[Project page](https://nctamer.github.io/rubato-transcription/) · [Paper](https://arxiv.org/abs/2605.24291) · [Checkpoint](https://github.com/nctamer/rubato-transcription/releases/download/v1/rubato.ckpt) · [Releases](https://github.com/nctamer/rubato-transcription/releases) · [Citation](#citation)

Rubato transcribes piano recordings into score notation, performed notes, and beats with timestamps. The included **InterMo-lite** converter exports the transcripts as MusicXML, MIDI, or CSV.

**Nazif Can Tamer** · Victoria Ebert · Guang Yang · **Noah A. Smith**

## Quick start

This branch of the repository provides the lightweight inference runtime. You need **Python 3.9+, PyTorch, and NumPy**. PCM WAV input and the bundled converters work without NeMo, Docker, C++ compilation, or external notation software.

```bash
# Install the inference runtime and InterMo-lite converter.
pip install "git+https://github.com/nctamer/rubato-transcription.git@v1"

# Download the self-contained FP32 inference checkpoint (~816 MB).
curl -L --fail -o rubato.ckpt \
  https://github.com/nctamer/rubato-transcription/releases/download/v1/rubato.ckpt

# Transcribe a recording and export the score.
rubato rubato.ckpt piece.wav --mode score --out-dir out
intermo-lite score out/piece.score.txt out/piece.musicxml
```

CUDA is selected automatically when available. To run on CPU explicitly:

```bash
rubato rubato.ckpt piece.wav --mode score --device cpu --threads 8 --out-dir out
```

For MP3, FLAC, and other formats supported by your SoundFile installation, install the optional audio dependency:

```bash
pip install soundfile
```

SciPy is optional for faster resampling: `pip install scipy`.

## Python API

```python
from rubato import Rubato

model = Rubato.load("rubato.ckpt", device="cpu")  # or device="cuda"
text = model.transcribe_file("piece.wav", mode="score")
print(text)
```

## Output modes

| Mode | Transcript | InterMo-lite export |
| --- | --- | --- |
| `score` | Score notation with timestamps | MusicXML |
| `midi` | Performed notes and velocities with timestamps | MIDI |
| `beats` | Beats and downbeats with timestamps | CSV |

Choose the same mode for transcription and conversion:

```bash
rubato rubato.ckpt piece.wav --mode midi --out-dir out
intermo-lite midi out/piece.midi.txt out/piece.mid

rubato rubato.ckpt piece.wav --mode beats --out-dir out
intermo-lite beats out/piece.beats.txt out/piece.csv
```

InterMo-lite is included in the installation. It uses Python's standard library for conversion; the native InterMo C++ library is not required for this inference branch.

## Decoding and checkpoint

The runtime uses KV caching and greedy decoding, processing recordings in 40-second windows with a 20-second hop. Attention uses PyTorch SDPA or the available CUDA/Triton path, depending on the device and configuration.

`rubato.ckpt` contains the post-trained model’s FP32 weights in a self-contained inference bundle. Release `v1` supports inference without NeMo or Docker. This bundle cannot resume training. For training or fine-tuning, use the training branch with its NeMo Docker environment.

Recovery decoding is optional and disabled by default. Enable it with:

```bash
rubato rubato.ckpt piece.wav --mode score --recover on --out-dir out
```

It monitors attention for possible decoding failures and retries flagged windows with temperature fallback. Use `--recover track` to record detector flags without changing the transcription; add `--json out/results.json` to save them. Recovery can change the output and is not guaranteed to improve every recording.

## Related resources

- [Interactive demo](https://rubatoscore.com/demo)
- [GitHub releases](https://github.com/nctamer/rubato-transcription/releases): checkpoint and release files.
- Hugging Face model page: coming soon.
- Native InterMo library and development/training instructions: see the training branch.

## Citation

If you use Rubato, please cite:

```bibtex
@inproceedings{tamer2026rubato,
  title={Rubato: Transcribing Piano Music with Timestamps},
  author={Tamer, Nazif Can and Ebert, Victoria and Yang, Guang and Smith, Noah A.},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```

Citation metadata is also available in [CITATION.cff](CITATION.cff).

## License

Code is provided under [AGPL-3.0](LICENSE). See the release notes for checkpoint provenance and terms.
