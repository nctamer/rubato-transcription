"""Audio input: PCM WAV through the standard library (`wave`), any format soundfile can read when it is
installed; channels are averaged to mono and the signal is resampled to 16 kHz (rubato.resample)."""

import wave

import numpy as np

from .resample import resample

SR = 16000


def _read_wav_stdlib(path):
    with wave.open(str(path), "rb") as w:
        n_ch, width, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if width == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v >= 1 << 23, v - (1 << 24), v)
        x = v.astype(np.float32) / float(1 << 23)
    elif width == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / float(1 << 31)
    else:
        raise ValueError(f"{path}: unsupported sample width {width}")
    return x.reshape(-1, n_ch), sr


def read_audio(path):
    """(samples [n, channels] float32, sample rate) as stored in the file."""
    try:
        import soundfile as sf
    except Exception:
        sf = None
    if sf is not None:
        x, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return x, sr
    try:
        return _read_wav_stdlib(path)
    except wave.Error as e:
        raise ValueError(f"{path}: not a PCM WAV file ({e}); install soundfile to read other formats, "
                         f"or convert to 16 kHz WAV first") from e


def load_audio(path, sr=SR):
    """float32 mono signal at `sr` (mean over channels, polyphase resampling)."""
    x, file_sr = read_audio(path)
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    x = np.ascontiguousarray(x, dtype=np.float32)
    if file_sr != sr:
        x = resample(x, file_sr, sr)
    return x
