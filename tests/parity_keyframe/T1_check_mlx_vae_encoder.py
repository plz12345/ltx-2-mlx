"""T1 (MLX side): VAE encoder forward — load same image, compare to PT dump."""

from __future__ import annotations

import os
import sys

import mlx.core as mx
import numpy as np

from ltx_core_mlx.model.video_vae.video_vae import VideoEncoder
from ltx_core_mlx.utils.weights import load_split_safetensors

WEIGHTS = os.path.expanduser(
    "~/.cache/huggingface/hub/models--dgrauet--ltx-2.3-mlx-q8/"
    "snapshots/03da129baa459c9a70fc5858dee52fa417b3a93d/vae_encoder.safetensors"
)
PT = "/tmp/T1_vae_encoder_pt.npz"


def main() -> None:
    encoder = VideoEncoder()
    weights = load_split_safetensors(WEIGHTS, prefix="vae_encoder.")
    weights = {
        k.replace("._mean_of_means", ".mean_of_means").replace("._std_of_means", ".std_of_means"): v
        for k, v in weights.items()
    }
    encoder.load_weights(list(weights.items()))

    pt = dict(np.load(PT))
    img_np = pt["image"]
    img_mlx = mx.array(img_np)
    latent_mlx = encoder.encode(img_mlx)
    mx.async_eval(latent_mlx)
    latent_np = np.asarray(latent_mlx).astype(np.float32)

    print(f"image shape: {img_np.shape}")
    print(f"MLX latent shape: {latent_np.shape}")
    print(f"PT  latent shape: {pt['latent'].shape}")

    if latent_np.shape != pt["latent"].shape and latent_np.shape[-1] == pt["latent"].shape[1]:
        latent_np = latent_np.transpose(0, 4, 1, 2, 3)
        print(f"transposed MLX latent shape: {latent_np.shape}")

    delta = float(np.max(np.abs(latent_np - pt["latent"])))
    rel = delta / max(float(np.max(np.abs(pt["latent"]))), 1e-9)
    print(f"max_abs diff: {delta:.4e}")
    print(f"relative diff: {rel:.4%}")
    print(f"PT mean={pt['latent'].mean():.6f} std={pt['latent'].std():.6f}")
    print(f"MLX mean={latent_np.mean():.6f} std={latent_np.std():.6f}")

    fail = delta > 5e-3
    print()
    print("FAIL" if fail else "OK")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
