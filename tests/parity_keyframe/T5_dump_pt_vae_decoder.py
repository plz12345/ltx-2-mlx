"""T5 (PT side): VAE decoder forward parity, using shared MLX-format weights.

Loads the bf16 weights from our MLX repo's vae_decoder.safetensors,
strips the 'vae_decoder.' prefix, transposes Conv3d weights from MLX
layout (O, K_t, K_h, K_w, I) back to PT layout (O, I, K_t, K_h, K_w),
runs PT VideoDecoder forward on a seeded random latent, dumps output.

Stochasticity neutralized: decode_noise_scale=0.

Run from upstream venv:
    cd /Users/dgrauet/sandbox/ltx-reference
    uv run python /Users/dgrauet/Work/mlx/ports/ltx-2-mlx/tests/parity_keyframe/T5_dump_pt_vae_decoder.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import safetensors
import torch
from ltx_core.model.video_vae.enums import NormLayerType, PaddingModeType
from ltx_core.model.video_vae.video_vae import VideoDecoder

WEIGHTS = os.path.expanduser(
    "~/.cache/huggingface/hub/models--dgrauet--ltx-2.3-mlx-q8/"
    "snapshots/03da129baa459c9a70fc5858dee52fa417b3a93d/vae_decoder.safetensors"
)
EMBEDDED_CFG = os.path.expanduser(
    "~/.cache/huggingface/hub/models--dgrauet--ltx-2.3-mlx-q8/"
    "snapshots/03da129baa459c9a70fc5858dee52fa417b3a93d/embedded_config.json"
)


def load_pt_state_dict() -> dict:
    pt_state = {}
    with safetensors.safe_open(WEIGHTS, framework="pt") as f:
        for k in f.keys():  # noqa: SIM118 (safe_open is not a dict)
            t = f.get_tensor(k).to(torch.float32)
            new_k = k.replace("vae_decoder.", "", 1)
            # Per-channel statistics: PT uses dashes, MLX-decoder uses .mean/.std
            # Decoder per_channel_statistics has different naming than encoder.
            new_k = new_k.replace("per_channel_statistics.mean", "per_channel_statistics.mean-of-means")
            new_k = new_k.replace("per_channel_statistics.std", "per_channel_statistics.std-of-means")
            if new_k.endswith(".conv.weight") and t.ndim == 5:
                t = t.permute(0, 4, 1, 2, 3).contiguous()
            pt_state[new_k] = t
    return pt_state


def main() -> None:
    with open(EMBEDDED_CFG) as f:
        cfg = json.load(f)["vae"]

    decoder = VideoDecoder(
        convolution_dimensions=cfg.get("dims", 3),
        in_channels=cfg.get("latent_channels", 128),
        out_channels=cfg.get("out_channels", 3),
        decoder_blocks=cfg.get("decoder_blocks", []),
        patch_size=cfg.get("patch_size", 4),
        norm_layer=NormLayerType(cfg.get("norm_layer", "pixel_norm")),
        causal=cfg.get("causal_decoder", False),
        timestep_conditioning=cfg.get("timestep_conditioning", True),
        decoder_spatial_padding_mode=PaddingModeType(cfg.get("spatial_padding_mode", "reflect")),
        base_channels=cfg.get("decoder_base_channels", 128),
    )

    state = load_pt_state_dict()
    missing, unexpected = decoder.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"WARN missing={len(missing)} unexpected={len(unexpected)}")
        if missing[:3]:
            print(f"  missing[:3]: {missing[:3]}")
        if unexpected[:3]:
            print(f"  unexpected[:3]: {unexpected[:3]}")

    decoder.train(False)
    decoder = decoder.to(torch.float32)
    # Neutralize stochasticity
    decoder.decode_noise_scale = 0.0

    # Latent shape (B, C, F, H, W) — F=2 covers latent boundary, the suspect zone.
    rng = np.random.default_rng(42)
    latent_np = rng.standard_normal((1, 128, 2, 4, 4)).astype(np.float32)
    timestep = torch.zeros((1,), dtype=torch.float32)

    with torch.no_grad():
        latent = torch.from_numpy(latent_np)
        out = decoder(latent, timestep=timestep)

    print(f"latent shape: {latent.shape}")
    print(f"output shape: {out.shape}")
    print(f"output stats: mean={out.mean().item():.6f} std={out.std().item():.6f}")

    np.savez(
        "/tmp/T5_vae_decoder_pt.npz",
        latent=latent_np,
        timestep=timestep.numpy(),
        output=out.detach().cpu().numpy().astype(np.float32),
    )
    print("wrote /tmp/T5_vae_decoder_pt.npz")


if __name__ == "__main__":
    main()
