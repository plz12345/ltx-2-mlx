"""T7 (MLX side): Spatial 2x upsampler forward — load same latent, compare to PT."""

from __future__ import annotations

import os
import sys

import mlx.core as mx
import numpy as np

from ltx_core_mlx.model.upsampler.model import LatentUpsampler
from ltx_core_mlx.utils.weights import load_split_safetensors

WEIGHTS = os.path.expanduser(
    "~/.cache/huggingface/hub/models--dgrauet--ltx-2.3-mlx-q8/"
    "snapshots/03da129baa459c9a70fc5858dee52fa417b3a93d/spatial_upscaler_x2_v1_1.safetensors"
)
PT = "/tmp/T7_upsampler_pt.npz"


def main() -> None:
    upsampler = LatentUpsampler(
        in_channels=128,
        mid_channels=1024,
        num_blocks_per_stage=4,
        spatial_upsample=True,
        temporal_upsample=False,
        spatial_scale=2.0,
        rational_resampler=False,
    )
    weights = load_split_safetensors(WEIGHTS, prefix="spatial_upscaler_x2_v1_1.")
    # Cast bf16 -> fp32 to match PT precision for diagnostic
    weights_fp32 = {k: v.astype(mx.float32) for k, v in weights.items()}
    upsampler.load_weights(list(weights_fp32.items()))

    pt = dict(np.load(PT))
    latent_np = pt["latent"]
    latent_mlx = mx.array(latent_np)
    out_mlx = upsampler(latent_mlx)
    out_np = np.asarray(out_mlx).astype(np.float32)

    print(f"input shape:  {latent_np.shape}")
    print(f"MLX output shape: {out_np.shape}")
    print(f"PT  output shape: {pt['output'].shape}")

    if out_np.shape != pt["output"].shape:
        print("FAIL: shape mismatch")
        sys.exit(1)

    delta = float(np.max(np.abs(out_np - pt["output"])))
    rel = delta / max(float(np.max(np.abs(pt["output"]))), 1e-9)
    print(f"max_abs diff: {delta:.4e}")
    print(f"relative diff: {rel:.4%}")
    print(f"PT  mean={pt['output'].mean():.6f} std={pt['output'].std():.6f}")
    print(f"MLX mean={out_np.mean():.6f} std={out_np.std():.6f}")

    print()
    print("Per-frame mean (PT vs MLX):")
    for f in range(out_np.shape[2]):
        pt_mean = pt["output"][:, :, f].mean()
        mlx_mean = out_np[:, :, f].mean()
        diff = abs(pt_mean - mlx_mean)
        print(f"  frame {f}: PT={pt_mean:+.4f}  MLX={mlx_mean:+.4f}  diff={diff:.4e}")

    fail = delta > 5e-3
    print()
    print("FAIL" if fail else "OK")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
