"""The Rubato network in plain torch: log-mel front end, Conformer encoder (8x depthwise-striding
subsampling, relative-position self-attention), encoder->decoder projection, pre-LN Transformer decoder and
log-softmax token classifier.

Parameter names follow the training checkpoint's state_dict, so a converted weights file loads with
load_state_dict(strict=True). Attention uses additive -10000 masks and the checkpoint's scaling and reduction conventions.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import MelFrontend

NEG = -10000.0


# ------------------------------------------------------------------------------------------------ encoder

class ConvSubsampling(nn.Module):
    """3 stride-2 3x3 convolutions over (time, mel) (the 2nd and 3rd depthwise + pointwise), ReLU after each
    block, the input masked by the current valid length before every layer; then a linear layer."""

    def __init__(self, feat_in, d_model, channels, factor=8):
        super().__init__()
        n = int(math.log2(factor))
        layers = [nn.Conv2d(1, channels, 3, 2, 1), nn.ReLU()]
        for _ in range(n - 1):
            layers += [nn.Conv2d(channels, channels, 3, 2, 1, groups=channels), nn.Conv2d(channels, channels, 1),
                       nn.ReLU()]
        self.conv = nn.Sequential(*layers)
        f = feat_in
        for _ in range(n):
            f = (f + 2 - 3) // 2 + 1
        self.out = nn.Linear(channels * f, d_model)

    @staticmethod
    def _mask(x, lengths):
        B, _, T, Fd = x.shape
        m = torch.arange(T, device=x.device).expand(B, T) < lengths.unsqueeze(1)
        return m.unsqueeze(-1).expand(B, T, Fd).to(x.dtype)

    def forward(self, x, lengths):
        """x [B, T, F], lengths [B] -> [B, T', d_model], lengths [B]."""
        x = x.unsqueeze(1)
        cur = lengths.clone().float()
        mask = self._mask(x, cur.long())
        for layer in self.conv:
            x = x * mask.unsqueeze(1)
            x = layer(x)
            if isinstance(layer, nn.Conv2d) and layer.stride != (1, 1):
                cur = (cur + layer.padding[0] + layer.padding[1] - layer.kernel_size[0]) // layer.stride[0] + 1
                mask = self._mask(x, cur.long())
        x = x * mask.unsqueeze(1)
        b, c, t, f = x.shape
        x = self.out(x.transpose(1, 2).reshape(b, t, -1))
        return x, cur.long()


class FeedForward(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d, d_ff)
        self.linear2 = nn.Linear(d_ff, d)

    def forward(self, x):
        return self.linear2(F.silu(self.linear1(x)))


class ConvModule(nn.Module):
    def __init__(self, d, kernel):
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(d, 2 * d, 1)
        self.depthwise_conv = nn.Conv1d(d, d, kernel, padding=(kernel - 1) // 2, groups=d)
        self.batch_norm = nn.BatchNorm1d(d)
        self.pointwise_conv2 = nn.Conv1d(d, d, 1)

    def forward(self, x, pad_mask):
        x = self.pointwise_conv1(x.transpose(1, 2))
        x = F.glu(x, dim=1)
        x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = F.silu(x)
        return self.pointwise_conv2(x).transpose(1, 2)


class RelPosSelfAttention(nn.Module):
    """Transformer-XL relative-position multi-head self-attention (explicit softmax path)."""

    def __init__(self, d, h):
        super().__init__()
        self.h, self.d_k = h, d // h
        self.s_d_k = math.sqrt(self.d_k)
        self.linear_q = nn.Linear(d, d)
        self.linear_k = nn.Linear(d, d)
        self.linear_v = nn.Linear(d, d)
        self.linear_out = nn.Linear(d, d)
        self.linear_pos = nn.Linear(d, d, bias=False)
        self.pos_bias_u = nn.Parameter(torch.zeros(h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.zeros(h, self.d_k))

    @staticmethod
    def rel_shift(x):
        b, h, qlen, pos_len = x.size()
        x = F.pad(x, pad=(1, 0))
        x = x.view(b, h, -1, qlen)
        return x[:, :, 1:].view(b, h, qlen, pos_len)

    def forward(self, x, mask, pos_emb):
        B = x.size(0)
        q = self.linear_q(x).view(B, -1, self.h, self.d_k)            # (B, T, h, dk)
        k = self.linear_k(x).view(B, -1, self.h, self.d_k).transpose(1, 2)
        v = self.linear_v(x).view(B, -1, self.h, self.d_k).transpose(1, 2)
        p = self.linear_pos(pos_emb).view(pos_emb.size(0), -1, self.h, self.d_k).transpose(1, 2)
        q_u = (q + self.pos_bias_u).transpose(1, 2)
        q_v = (q + self.pos_bias_v).transpose(1, 2)
        bd = self.rel_shift(torch.matmul(q_v, p.transpose(-2, -1)))
        ac = torch.matmul(q_u, k.transpose(-2, -1))
        scores = (ac + bd[:, :, :, :ac.size(-1)]) / self.s_d_k
        m = mask.unsqueeze(1)
        scores = scores.masked_fill(m, NEG)
        attn = torch.softmax(scores, dim=-1).masked_fill(m, 0.0)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, -1, self.h * self.d_k)
        return self.linear_out(out)


class ConformerLayer(nn.Module):
    def __init__(self, d, d_ff, h, kernel):
        super().__init__()
        self.norm_feed_forward1 = nn.LayerNorm(d)
        self.feed_forward1 = FeedForward(d, d_ff)
        self.norm_conv = nn.LayerNorm(d)
        self.conv = ConvModule(d, kernel)
        self.norm_self_att = nn.LayerNorm(d)
        self.self_attn = RelPosSelfAttention(d, h)
        self.norm_feed_forward2 = nn.LayerNorm(d)
        self.feed_forward2 = FeedForward(d, d_ff)
        self.norm_out = nn.LayerNorm(d)

    def forward(self, x, att_mask, pos_emb, pad_mask):
        r = x + self.feed_forward1(self.norm_feed_forward1(x)) * 0.5
        r = r + self.self_attn(self.norm_self_att(r), att_mask, pos_emb)
        r = r + self.conv(self.norm_conv(r), pad_mask)
        r = r + self.feed_forward2(self.norm_feed_forward2(r)) * 0.5
        return self.norm_out(r)


class ConformerEncoder(nn.Module):
    def __init__(self, feat_in, d_model, n_layers, n_heads, ff_expansion, conv_kernel, sub_channels, sub_factor):
        super().__init__()
        self.d_model = d_model
        self.pre_encode = ConvSubsampling(feat_in, d_model, sub_channels, sub_factor)
        self.layers = nn.ModuleList([ConformerLayer(d_model, d_model * ff_expansion, n_heads, conv_kernel)
                                     for _ in range(n_layers)])
        self._pe = {}

    def rel_pos_emb(self, T, device, dtype):
        """Sinusoidal embeddings of relative positions T-1 .. -(T-1), computed on `device` (as in training)."""
        key = (T, str(device), dtype)
        pe = self._pe.get(key)
        if pe is None:
            positions = torch.arange(T - 1, -T, -1, dtype=torch.float32, device=device).unsqueeze(1)
            pe = torch.zeros(positions.size(0), self.d_model, device=device)
            div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32, device=device)
                                 * -(math.log(10000.0) / self.d_model))
            pe[:, 0::2] = torch.sin(positions * div_term)
            pe[:, 1::2] = torch.cos(positions * div_term)
            pe = self._pe[key] = pe.unsqueeze(0).to(dtype)
        return pe

    def forward(self, feats, length):
        """feats [B, F, T], length [B] -> [B, T', d_model], length [B]."""
        x, length = self.pre_encode(feats.transpose(1, 2), length)
        T = x.size(1)
        pos_emb = self.rel_pos_emb(T, x.device, x.dtype)
        valid = torch.arange(T, device=x.device).expand(length.size(0), -1) < length.unsqueeze(-1)
        att_mask = ~(valid.unsqueeze(1) & valid.unsqueeze(2))
        pad_mask = ~valid
        for layer in self.layers:
            x = layer(x, att_mask, pos_emb, pad_mask)
        return x, length


# ------------------------------------------------------------------------------------------------ decoder

class Embedding(nn.Module):
    def __init__(self, vocab, d, max_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab, d, padding_idx=0)
        self.position_embedding = nn.Module()
        self.position_embedding.register_buffer("pos_enc", torch.zeros(max_len, d))
        self.layer_norm = nn.LayerNorm(d, eps=1e-5)

    def forward(self, ids, pos):
        """ids [B, S], pos [B, S] -> [B, S, d]."""
        e = self.token_embedding(ids) + torch.embedding(self.position_embedding.pos_enc, pos)
        return self.layer_norm(e)


class Attention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.query_net = nn.Linear(d, d)
        self.key_net = nn.Linear(d, d)
        self.value_net = nn.Linear(d, d)
        self.out_projection = nn.Linear(d, d)


class FF(nn.Module):
    def __init__(self, d, inner):
        super().__init__()
        self.dense_in = nn.Linear(d, inner)
        self.dense_out = nn.Linear(inner, d)

    def forward(self, x):
        return self.dense_out(torch.relu(self.dense_in(x)))


class DecoderLayer(nn.Module):
    def __init__(self, d, inner):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(d, eps=1e-5)
        self.first_sub_layer = Attention(d)
        self.layer_norm_2 = nn.LayerNorm(d, eps=1e-5)
        self.second_sub_layer = Attention(d)
        self.layer_norm_3 = nn.LayerNorm(d, eps=1e-5)
        self.third_sub_layer = FF(d, inner)


class DecoderStack(nn.Module):
    def __init__(self, d, inner, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d, inner) for _ in range(n_layers)])
        self.final_layer_norm = nn.LayerNorm(d, eps=1e-5)


class TransformerDecoder(nn.Module):
    def __init__(self, vocab, d, inner, n_layers, n_heads, max_len):
        super().__init__()
        self.n_heads = n_heads
        self._embedding = Embedding(vocab, d, max_len)
        self._decoder = DecoderStack(d, inner, n_layers)


class Classifier(nn.Module):
    def __init__(self, d, n_classes):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.layer0 = nn.Linear(d, n_classes)

    def forward(self, h):
        return torch.log_softmax(self.mlp.layer0(h), dim=-1)


# ------------------------------------------------------------------------------------------------ model

class RubatoModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        f, e, d = cfg["features"], cfg["encoder"], cfg["decoder"]
        self.preprocessor = nn.Module()
        self.preprocessor.featurizer = MelFrontend(f["n_fft"], f["hop"], f["win"], f["n_mels"], f["preemph"])
        self.encoder = ConformerEncoder(f["n_mels"], e["d_model"], e["n_layers"], e["n_heads"], e["ff_expansion"],
                                        e["conv_kernel"], e["sub_channels"], e["sub_factor"])
        self.encoder_decoder_proj = nn.Linear(e["d_model"], d["hidden"]) if e["d_model"] != d["hidden"] \
            else nn.Identity()
        self.transf_decoder = TransformerDecoder(d["vocab"], d["hidden"], d["inner"], d["n_layers"], d["n_heads"],
                                                 d["max_len"])
        self.log_softmax = Classifier(d["hidden"], d["n_classes"])

    @torch.inference_mode()
    def encode(self, audio, lens):
        """audio [B, N] float32 (zero-padded), lens [B] samples -> decoder memory [B, T, H] (zeros past each
        row's length, float32) and lengths [B]."""
        feats, feat_len = self.preprocessor.featurizer(audio, lens)
        x, enc_len = self.encoder(feats, feat_len)
        enc = self.encoder_decoder_proj(x)
        mask = (torch.arange(enc.shape[1], device=enc.device)[None, :] < enc_len[:, None]).to(enc.dtype)
        return enc.float() * mask.unsqueeze(-1).float(), enc_len
