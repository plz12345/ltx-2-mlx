"""T2-mini (MLX side): AdaLayerNormSingle parity check.

Loads transformer.adaln_single.* weights from our bf16 safetensors,
applies our path (sinusoidal embed -> AdaLayerNormSingle), compares
against PT dump in /tmp/T2_adaln_pt.npz.
"""

from __future__ import annotations

import os
import sys

import mlx.core as mx
import numpy as np
from mlx_arsenal.diffusion import get_timestep_embedding

from ltx_core_mlx.model.transformer.adaln import AdaLayerNormSingle

MLX_WEIGHTS = os.path.expanduser(
    "~/.cache/huggingface/hub/models--dgrauet--ltx-2.3-mlx/snapshots/"
    "baa5f235ea04fd9c95899d751295c4fd825ee4e2/transformer-dev.safetensors"
)
PT = "/tmp/T2_adaln_pt.npz"
TIMESTEP_SCALE = 1000.0


def load_keys(path: str, prefix: str) -> dict:
    """Load keys with given prefix via mx.load (handles bf16), strip prefix, cast to fp32."""
    raw = mx.load(path)
    out = {}
    for k, t in raw.items():
        if k.startswith(prefix):
            out[k.removeprefix(prefix)] = t.astype(mx.float32)
    return out


def build_and_load(prefix: str, dim: int, num_params: int) -> AdaLayerNormSingle:
    model = AdaLayerNormSingle(dim=dim, num_params=num_params)
    state = load_keys(MLX_WEIGHTS, prefix)
    model.load_weights(list(state.items()), strict=False)
    return model


def run_case(model: AdaLayerNormSingle, timestep_np: np.ndarray) -> dict:
    timestep = mx.array(timestep_np, dtype=mx.float32) * TIMESTEP_SCALE
    timestep = timestep.flatten()
    t_emb = get_timestep_embedding(timestep, embedding_dim=256)
    params, embedded = model(t_emb)
    return {
        "params": np.asarray(params).astype(np.float32),
        "embedded": np.asarray(embedded).astype(np.float32),
    }


def diff(name: str, mlx_arr: np.ndarray, pt_arr: np.ndarray) -> tuple[str, float]:
    if mlx_arr.shape != pt_arr.shape:
        return (f"FAIL: shape mlx={mlx_arr.shape} pt={pt_arr.shape}", float("nan"))
    delta = float(np.max(np.abs(mlx_arr - pt_arr)))
    rel = delta / max(float(np.max(np.abs(pt_arr))), 1e-9)
    return (f"max_abs={delta:.4e} rel={rel:.2%}", delta)


def main() -> None:
    pt = dict(np.load(PT))
    video_adaln = build_and_load("transformer.adaln_single.", dim=4096, num_params=9)
    audio_adaln = build_and_load("transformer.audio_adaln_single.", dim=2048, num_params=9)
    cases = [
        ("video_uniform", video_adaln, pt["video_uniform_timestep"]),
        ("audio_uniform", audio_adaln, pt["audio_uniform_timestep"]),
        ("video_per_token", video_adaln, pt["video_per_token_timestep"]),
        ("audio_per_token", audio_adaln, pt["audio_per_token_timestep"]),
    ]
    fail = False
    for label, model, timestep_np in cases:
        print(f"\n=== {label} ===")
        out = run_case(model, timestep_np)
        for sub in ("params", "embedded"):
            verdict, delta = diff(sub, out[sub], pt[f"{label}_{sub}"])
            print(f"  {sub:10s}: {verdict}")
            if delta != delta or delta > 5e-3:
                fail = True
    print()
    print("FAIL" if fail else "ALL OK (max_abs <= 5e-3)")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
