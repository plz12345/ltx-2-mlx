"""T7 (PT side): Spatial 2x upsampler forward parity.

Loads bf16 weights from our MLX repo's spatial_upscaler_x2_v1_1.safetensors,
strips the 'spatial_upscaler_x2_v1_1.' prefix, transposes Conv weights,
runs PT LatentUpsampler forward on a seeded random latent.

Run from upstream venv:
    cd /Users/dgrauet/sandbox/ltx-reference
    uv run python /Users/dgrauet/Work/mlx/ports/ltx-2-mlx/tests/parity_keyframe/T7_dump_pt_upsampler.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import safetensors
import torch
from ltx_core.model.upsampler.model import LatentUpsampler

WEIGHTS = os.path.expanduser(
    "~/.cache/huggingface/hub/models--dgrauet--ltx-2.3-mlx-q8/"
    "snapshots/03da129baa459c9a70fc5858dee52fa417b3a93d/spatial_upscaler_x2_v1_1.safetensors"
)
CFG = os.path.expanduser(
    "~/.cache/huggingface/hub/models--dgrauet--ltx-2.3-mlx-q8/"
    "snapshots/03da129baa459c9a70fc5858dee52fa417b3a93d/spatial_upscaler_x2_v1_1_config.json"
)


def load_pt_state_dict() -> dict:
    pt_state = {}
    with safetensors.safe_open(WEIGHTS, framework="pt") as f:
        for k in f.keys():  # noqa: SIM118 (safe_open not a dict)
            t = f.get_tensor(k).to(torch.float32)
            new_k = k.replace("spatial_upscaler_x2_v1_1.", "", 1)
            # Conv3d weights: MLX (O, K_t, K_h, K_w, I) -> PT (O, I, K_t, K_h, K_w)
            if t.ndim == 5:
                t = t.permute(0, 4, 1, 2, 3).contiguous()
            # Conv2d weights: MLX (O, K_h, K_w, I) -> PT (O, I, K_h, K_w)
            elif t.ndim == 4:
                t = t.permute(0, 3, 1, 2).contiguous()
            pt_state[new_k] = t
    return pt_state


def main() -> None:
    with open(CFG) as f:
        cfg = json.load(f)["config"]

    upsampler = LatentUpsampler(
        in_channels=cfg.get("in_channels", 128),
        mid_channels=cfg.get("mid_channels", 1024),
        num_blocks_per_stage=cfg.get("num_blocks_per_stage", 4),
        dims=cfg.get("dims", 3),
        spatial_upsample=cfg.get("spatial_upsample", True),
        temporal_upsample=cfg.get("temporal_upsample", False),
        spatial_scale=cfg.get("spatial_scale", 2.0),
        rational_resampler=cfg.get("rational_resampler", False),
    )

    # Adjust state dict keys for PT Sequential layout: PT uses upsampler.Sequential which
    # produces upsampler.0.weight/bias (matching MLX list-style keys). Should map directly.
    state = load_pt_state_dict()
    missing, unexpected = upsampler.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"WARN missing={len(missing)} unexpected={len(unexpected)}")
        if missing[:5]:
            print(f"  missing[:5]: {missing[:5]}")
        if unexpected[:5]:
            print(f"  unexpected[:5]: {unexpected[:5]}")

    upsampler.train(False)
    upsampler = upsampler.to(torch.float32)

    # Half-res hedgehog dims: (1, 128, 5, 14, 22) — what stage 1 produces
    rng = np.random.default_rng(42)
    latent_np = rng.standard_normal((1, 128, 5, 14, 22)).astype(np.float32)

    with torch.no_grad():
        latent = torch.from_numpy(latent_np)
        out = upsampler(latent)

    print(f"input shape:  {latent.shape}")
    print(f"output shape: {out.shape}")
    print(f"output stats: mean={out.mean().item():.6f} std={out.std().item():.6f}")

    np.savez(
        "/tmp/T7_upsampler_pt.npz",
        latent=latent_np,
        output=out.detach().cpu().numpy().astype(np.float32),
    )
    print("wrote /tmp/T7_upsampler_pt.npz")


if __name__ == "__main__":
    main()
