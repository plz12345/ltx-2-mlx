"""T1 (PT side): VAE encoder forward parity, using shared MLX-format weights.

Loads the bf16 weights from our MLX repo's vae_encoder.safetensors,
strips the 'vae_encoder.' prefix, transposes Conv3d weights from MLX
layout (O, K_t, K_h, K_w, I) back to PT layout (O, I, K_t, K_h, K_w),
runs PT VideoEncoder forward on a seeded random image, dumps output.

Run from the upstream venv:
    cd /Users/dgrauet/sandbox/ltx-reference
    uv run python /Users/dgrauet/Work/mlx/ports/ltx-2-mlx/tests/parity_keyframe/T1_dump_pt_vae_encoder.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import safetensors
import torch
from ltx_core.model.video_vae.enums import LogVarianceType, NormLayerType, PaddingModeType
from ltx_core.model.video_vae.video_vae import VideoEncoder

WEIGHTS = os.path.expanduser(
    "~/.cache/huggingface/hub/models--dgrauet--ltx-2.3-mlx-q8/"
    "snapshots/03da129baa459c9a70fc5858dee52fa417b3a93d/vae_encoder.safetensors"
)
EMBEDDED_CFG = os.path.expanduser(
    "~/.cache/huggingface/hub/models--dgrauet--ltx-2.3-mlx-q8/"
    "snapshots/03da129baa459c9a70fc5858dee52fa417b3a93d/embedded_config.json"
)


def load_pt_state_dict() -> dict:
    """Load MLX safetensors and convert keys/layouts for PT VideoEncoder."""
    pt_state = {}
    with safetensors.safe_open(WEIGHTS, framework="pt") as f:
        for k in f.keys():  # noqa: SIM118 (safe_open is not a dict)
            t = f.get_tensor(k).to(torch.float32)
            new_k = k.replace("vae_encoder.", "", 1)
            # Per-channel statistics: PT uses dashes, MLX uses underscores+leading underscore
            new_k = new_k.replace("_mean_of_means", "mean-of-means")
            new_k = new_k.replace("_std_of_means", "std-of-means")
            # Conv3d weights: MLX (O, K_t, K_h, K_w, I) -> PT (O, I, K_t, K_h, K_w)
            if new_k.endswith(".conv.weight") and t.ndim == 5:
                t = t.permute(0, 4, 1, 2, 3).contiguous()
            pt_state[new_k] = t
    return pt_state


def main() -> None:
    with open(EMBEDDED_CFG) as f:
        cfg = json.load(f)["vae"]

    encoder = VideoEncoder(
        convolution_dimensions=cfg.get("dims", 3),
        in_channels=cfg.get("in_channels", 3),
        out_channels=cfg.get("latent_channels", 128),
        encoder_blocks=cfg.get("encoder_blocks", []),
        patch_size=cfg.get("patch_size", 4),
        norm_layer=NormLayerType(cfg.get("norm_layer", "pixel_norm")),
        latent_log_var=LogVarianceType(cfg.get("latent_log_var", "uniform")),
        encoder_spatial_padding_mode=PaddingModeType(cfg.get("spatial_padding_mode", "zeros")),
    )

    state = load_pt_state_dict()
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"WARN missing={len(missing)} unexpected={len(unexpected)}")
        if missing[:3]:
            print(f"  missing[:3]: {missing[:3]}")
        if unexpected[:3]:
            print(f"  unexpected[:3]: {unexpected[:3]}")

    encoder.train(False)
    encoder = encoder.to(torch.float32)

    # Need at least 32x32 spatial after patchify (4x) + 32x downscale across blocks.
    # With 64x64 input -> 16x16 after patchify -> 1x1 latent after compress_space x4.
    # Use 128x128 to get 4x4 latent (more meaningful for parity).
    rng = np.random.default_rng(42)
    img_np = rng.standard_normal((1, 3, 1, 128, 128)).astype(np.float32)

    with torch.no_grad():
        img = torch.from_numpy(img_np)
        latent = encoder(img)

    print(f"image shape: {img.shape}")
    print(f"latent shape: {latent.shape}")
    print(f"latent stats: mean={latent.mean().item():.6f} std={latent.std().item():.6f}")

    np.savez(
        "/tmp/T1_vae_encoder_pt.npz",
        image=img_np,
        latent=latent.detach().cpu().numpy().astype(np.float32),
    )
    print("wrote /tmp/T1_vae_encoder_pt.npz")


if __name__ == "__main__":
    main()
