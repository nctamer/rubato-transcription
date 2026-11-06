# Rubato v1 — public inference release

Rubato transcribes piano audio into timestamped score, MIDI, and beat representations. This release rebrands rubato-lite 0.3.0 with `from rubato import Rubato`, `rubato`, and `intermo-lite` (`rubato-convert` is an alias).

## Assets

- `rubato.ckpt`: FP32 inference bundle from rubato-lite 0.3.0; 816,004,751 bytes. Private training-path and source-hash metadata removed; model tensors unchanged.
- `rubato_transcription-0.3.0-py3-none-any.whl`: Python runtime and converters.
- `SHA256SUMS`: checksums for the assets.
- GitHub generates source archives automatically from the release tag.

Checkpoint SHA-256: `131649ff95b80713bd327b63d37b2771e1175b3c1c9aa4242a783c67bf27b1fc`.

Only PyTorch and NumPy are required at runtime. PCM WAV input and MusicXML/MIDI/CSV conversion do not require C++ tools. The checkpoint uses the original rubato-lite/1 internal format for compatibility. The public API and command names are rebranded; model computation is unchanged.

The public release tag is `v1`; the inference package retains version `0.3.0` to identify its rubato-lite runtime provenance. Please cite the NeurIPS 2026 paper using the repository citation metadata.

Code license: [AGPL-3.0](https://github.com/nctamer/rubato-transcription/blob/main/LICENSE).
