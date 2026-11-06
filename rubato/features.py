"""Log-mel front end (16 kHz; 1024-sample Hann window and FFT, hop 80, 229 mel bands 30-8000 Hz, per-band
normalisation over the valid frames). Same operations, order and constants as the training front end; the mel
filterbank and window are the trained model's buffers (stored in the weights file)."""

import torch
import torch.nn as nn

LOG_GUARD = 2.0 ** -24
STD_EPS = 1e-5


class MelFrontend(nn.Module):
    def __init__(self, n_fft=1024, hop=80, win=1024, n_mels=229, preemph=0.97):
        super().__init__()
        self.n_fft, self.hop, self.win, self.preemph = n_fft, hop, win, preemph
        self.register_buffer("window", torch.zeros(win))
        self.register_buffer("fb", torch.zeros(1, n_mels, n_fft // 2 + 1))

    def seq_len(self, n):
        # centred STFT: frames = n // hop + 1; the last one is excluded from the valid length
        return torch.div(n + (self.n_fft // 2) * 2 - self.n_fft, self.hop, rounding_mode="floor").long()

    @torch.no_grad()
    def forward(self, x, length):
        """x [B, N] float32, length [B] samples -> features [B, n_mels, T], valid frames [B]."""
        x = x.float()
        seq_len = self.seq_len(length)
        seq_len = torch.where(length == 0, torch.zeros_like(seq_len), seq_len)
        timemask = torch.arange(x.shape[1], device=x.device).unsqueeze(0) < length.unsqueeze(1)
        x = torch.cat((x[:, 0].unsqueeze(1), x[:, 1:] - self.preemph * x[:, :-1]), dim=1)
        x = x.masked_fill(~timemask, 0.0)
        with torch.autocast(x.device.type, enabled=False):
            spec = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop, win_length=self.win, center=True,
                              window=self.window.float(), return_complex=True, pad_mode="constant")
        spec = torch.view_as_real(spec)
        spec = torch.sqrt(spec.pow(2).sum(-1))
        spec = spec.pow(2.0)
        with torch.autocast(x.device.type, enabled=False):
            mel = torch.matmul(self.fb.to(spec.dtype), spec)
        mel = torch.log(mel + LOG_GUARD)
        # per-band normalisation over valid frames (unbiased std)
        B, _, T = mel.shape
        valid = torch.arange(T, device=mel.device).unsqueeze(0).expand(B, T) < seq_len.unsqueeze(1)
        num = torch.where(valid.unsqueeze(1), mel, 0.0).sum(axis=2)
        den = valid.sum(axis=1)
        mean = num / den.unsqueeze(1)
        std = torch.sqrt(torch.sum(torch.where(valid.unsqueeze(1), mel - mean.unsqueeze(2), 0.0) ** 2, axis=2)
                         / (den.unsqueeze(1) - 1.0))
        std = std.masked_fill(std.isnan(), 0.0) + STD_EPS
        mel = (mel - mean.unsqueeze(2)) / std.unsqueeze(2)
        mel = mel.masked_fill(~valid.unsqueeze(1), 0.0)
        return mel, seq_len
