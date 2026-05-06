"""T5 (MLX side): VAE decoder forward — same latent, compare to PT dump."""

from __future__ import annotations

import os
import sys

import mlx.core as mx
import numpy as np

from ltx_core_mlx.model.video_vae.video_vae import VideoDecoder
from ltx_core_mlx.utils.weights import load_split_safetensors

WEIGHTS = os.path.expanduser(
    "~/.cache/huggingface/hub/models--dgrauet--ltx-2.3-mlx-q8/"
    "snapshots/03da129baa459c9a70fc5858dee52fa417b3a93d/vae_decoder.safetensors"
)
PT = "/tmp/T5_vae_decoder_pt.npz"


def main() -> None:
    decoder = VideoDecoder()
    weights = load_split_safetensors(WEIGHTS, prefix="vae_decoder.")
    decoder.load_weights(list(weights.items()))
    # Neutralize stochasticity to match PT side (which we set decode_noise_scale=0)
    if hasattr(decoder, "decode_noise_scale"):
        decoder.decode_noise_scale = 0.0

    pt = dict(np.load(PT))
    latent_np = pt["latent"]
    latent_mlx = mx.array(latent_np)
    out_mlx = decoder.decode(latent_mlx)
    out_np = np.asarray(out_mlx).astype(np.float32)

    print(f"latent shape: {latent_np.shape}")
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

    # Per-frame analysis to detect boundary discontinuities
    print()
    print("Per-frame mean (PT vs MLX):")
    for f in range(out_np.shape[2]):
        pt_mean = pt["output"][:, :, f].mean()
        mlx_mean = out_np[:, :, f].mean()
        pt_std = pt["output"][:, :, f].std()
        mlx_std = out_np[:, :, f].std()
        print(
            f"  frame {f}: PT mean={pt_mean:+.4f} std={pt_std:.4f}  |  "
            f"MLX mean={mlx_mean:+.4f} std={mlx_std:.4f}  |  "
            f"diff={abs(pt_mean - mlx_mean):.4e}"
        )

    fail = delta > 5e-3
    print()
    print("FAIL" if fail else "OK")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
