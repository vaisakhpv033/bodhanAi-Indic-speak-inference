"""Vocos-style decoder: ConvNeXt-1D backbone + iSTFT head.

Self-contained (torch only; the pip `vocos` package is not installed on this
cluster). Structure follows Vocos (arXiv:2306.00814): channels-last LayerNorm
ConvNeXt blocks at frame rate, a single exp-magnitude + phase iSTFT head.

Length contract: input z_q [B, 768, L] (SNAC latent, 46.875 Hz) -> output
waveform [B, 1, 512*L]. The head pads ONE feature frame (replicate) before the
centered iSTFT so hop*(T_feat+1-1) == 512*L exactly — no off-by-hop trims.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int, layer_scale_init: float):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, C, T]
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)                 # [B, T, C]
        x = self.norm(x)
        x = self.pwconv2(self.act(self.pwconv1(x)))
        x = self.gamma * x
        return residual + x.transpose(1, 2)


class NoiseInject(nn.Module):
    """x + Linear(x) * randn — a learned, per-channel stochastic source (as in SNAC's
    NoiseBlock). Zero-initialized so a checkpoint trained without it loads and decodes
    bit-identically; training grows the gain where stochastic fine structure (breath,
    fricatives, room tone) is rewarded by the discriminator."""
    def __init__(self, dim: int):
        super().__init__()
        self.gain = nn.Linear(dim, dim)
        nn.init.zeros_(self.gain.weight); nn.init.zeros_(self.gain.bias)
        self.noise_scale = 1.0   # set 0.0 for a deterministic decode

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, C, T]
        if self.noise_scale == 0.0:
            return x
        g = self.gain(x.transpose(1, 2)).transpose(1, 2)
        return x + g * torch.randn_like(x) * self.noise_scale


class ISTFTHead(nn.Module):
    def __init__(self, dim: int, n_fft: int, hop: int):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.out = nn.Linear(dim, n_fft + 2)  # n_fft//2+1 mags + n_fft//2+1 phases
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, C, T] -> [B, 1, hop*T]
        # pad one frame so centered istft yields exactly hop*T samples
        x = F.pad(x, (0, 1), mode="replicate")
        h = self.out(x.transpose(1, 2)).transpose(1, 2)   # [B, n_fft+2, T+1]
        mag, phase = h.chunk(2, dim=1)
        # exp-magnitude; clamp exponent (e^6~400 > max hann-window mag ~256)
        mag = torch.exp(mag.clamp(max=6.0))
        with torch.autocast(device_type=x.device.type, enabled=False):
            spec = mag.float() * (torch.cos(phase.float()) + 1j * torch.sin(phase.float()))
            audio = torch.istft(
                spec, self.n_fft, hop_length=self.hop, win_length=self.n_fft,
                window=self.window.float(), center=True)
        self.last_spec = spec   # predicted complex STFT, for the consistency loss
        return audio.unsqueeze(1)


class TimeDomainHead(nn.Module):
    """Time-domain output head: backbone features [B, dim, T] at 187.5 Hz -> waveform
    [B, 1, 128*T] via SNAC/DAC DecoderBlocks (weight-normed transposed conv + Snake +
    residual units + NoiseBlock). Phase is built in the time domain — no per-bin phase
    prediction, so no inter-frame phase drift (the 'waa'/machine texture of the iSTFT head)."""
    def __init__(self, dim: int, rates=(8, 4, 4), channels: int = 768, noise: bool = True):
        super().__init__()
        from snac.layers import DecoderBlock, Snake1d, WNConv1d
        assert int(torch.tensor(rates).prod()) == 128
        layers = [WNConv1d(dim, channels, kernel_size=7, padding=3)]
        for i, r in enumerate(rates):
            layers.append(DecoderBlock(channels // 2 ** i, channels // 2 ** (i + 1), r, noise))
        out_dim = channels // 2 ** len(rates)
        layers += [Snake1d(out_dim), WNConv1d(out_dim, 1, kernel_size=7, padding=3), nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class VocosDecoder(nn.Module):
    """SNAC z_q [B, latent_dim, L] -> waveform [B, 1, 512*L]."""

    def __init__(
        self,
        latent_dim: int = 768,
        dim: int = 512,
        intermediate_dim: int = 1536,
        num_blocks_pre: int = 2,     # at 46.875 Hz
        num_blocks_post: int = 8,    # at 93.75 Hz
        upsample: int = 2,
        n_fft: int = 1024,
        hop: int = 256,
        noise_inject: bool = False,
        head_type: str = "istft",
    ):
        super().__init__()
        assert upsample * hop == 512, "must preserve SNAC's 512 samples per latent step"
        self.upsample = upsample
        n_total = num_blocks_pre + num_blocks_post
        ls = 1.0 / n_total
        self.stem = nn.Conv1d(latent_dim, dim, kernel_size=7, padding=3)
        self.pre = nn.ModuleList(
            [ConvNeXtBlock(dim, intermediate_dim, ls) for _ in range(num_blocks_pre)])
        self.up_conv = nn.Conv1d(dim, dim, kernel_size=7, padding=3)
        self.noise_post = NoiseInject(dim) if noise_inject else None
        self.noise_head = NoiseInject(dim) if noise_inject else None
        self.post = nn.ModuleList(
            [ConvNeXtBlock(dim, intermediate_dim, ls) for _ in range(num_blocks_post)])
        self.final_norm = nn.LayerNorm(dim, eps=1e-6)
        if head_type == "timedomain":
            assert upsample == 4, "time-domain head expects 187.5 Hz features (upsample 4, x128)"
            self.head = TimeDomainHead(dim)
        else:
            self.head = ISTFTHead(dim, n_fft=n_fft, hop=hop)

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        x = self.stem(z_q)
        for blk in self.pre:
            x = blk(x)
        x = x.repeat_interleave(self.upsample, dim=-1)   # nearest upsample, no imaging
        x = self.up_conv(x)
        if self.noise_post is not None:
            x = self.noise_post(x)
        for blk in self.post:
            x = blk(x)
        x = self.final_norm(x.transpose(1, 2)).transpose(1, 2)
        if self.noise_head is not None:
            x = self.noise_head(x)
        return self.head(x)
