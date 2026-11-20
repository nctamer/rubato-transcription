# Rubato v1.0.1

Code update of v1. The model is unchanged: use the checkpoint released with v1,
https://github.com/nctamer/rubato-transcription/releases/download/v1/rubato.ckpt
(SHA-256 `131649ff95b80713bd327b63d37b2771e1175b3c1c9aa4242a783c67bf27b1fc`).

## Fixes

- Notes still sounding at a window cut are now closed, in midi and score modes (#1).
- Score to MIDI/MusicXML: an empty bar no longer shifts the times of the notes after it.

## Assets

- `rubato_transcription-0.3.1-py3-none-any.whl`: Python runtime and converters.
- `SHA256SUMS`: checksum of the wheel.
