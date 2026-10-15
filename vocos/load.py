"""Load the fine-tuned Vocos decoder: SNAC z_q [B,768,L] -> waveform [B,1,512*L] @ 24 kHz."""
import torch
from .model import VocosDecoder

LATENT_DIM = 768


def load_vocos(ckpt_path="best.pt", device="cpu"):
    """Build the decoder from the checkpoint's own config, then load its weights."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    dec = VocosDecoder(latent_dim=LATENT_DIM, **ck["config"]["model"])
    # EMA weights are the inference weights; fall back for older checkpoints
    dec.load_state_dict(ck.get("ema") or ck["vocos"])
    return dec.eval().to(device)


if __name__ == "__main__":
    import sys
    dec = load_vocos(sys.argv[1] if len(sys.argv) > 1 else "best.pt")
    with torch.no_grad():
        wav = dec(torch.randn(1, LATENT_DIM, 40))
    assert wav.shape == (1, 1, 512 * 40), wav.shape
    print("OK", tuple(wav.shape))
