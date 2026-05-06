"""T2-mini (PT side): AdaLayerNormSingle parity, focus on per-token path.

Loads transformer.adaln_single.* weights from upstream bf16 safetensors,
builds PT AdaLayerNormSingle, runs forward on:
  - case A: uniform timestep (B,)
  - case B: per-token timestep (B, N) flattened to (B*N,) — the keyframe
    code path (non-uniform AdaLN)

Tests both video AdaLN (dim=4096, coef=9) and audio AdaLN (dim=2048, coef=9).

Run from upstream venv:
    cd /Users/dgrauet/sandbox/ltx-reference
    uv run python /Users/dgrauet/Work/mlx/ports/ltx-2-mlx/tests/parity_keyframe/T2_dump_pt_adaln.py
"""

from __future__ import annotations

import os

import numpy as np
import safetensors
import torch
from ltx_core.model.transformer.adaln import AdaLayerNormSingle

PT_WEIGHTS = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/"
    "76730e634e70a28f4e8d51f5e29c08e40e2d8e74/ltx-2.3-22b-dev.safetensors"
)
TIMESTEP_SCALE = 1000.0


def load_keys(path: str, prefix: str) -> dict:
    out = {}
    with safetensors.safe_open(path, framework="pt") as f:
        for k in f.keys():  # noqa: SIM118 (safe_open is not a dict)
            if k.startswith(prefix):
                out[k.removeprefix(prefix)] = f.get_tensor(k).to(torch.float32)
    return out


def build_and_load(prefix: str, dim: int, coef: int) -> AdaLayerNormSingle:
    model = AdaLayerNormSingle(embedding_dim=dim, embedding_coefficient=coef)
    state = load_keys(PT_WEIGHTS, prefix)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"WARN {prefix} missing={len(missing)} unexpected={len(unexpected)}")
        if missing[:3]:
            print(f"  missing[:3]: {missing[:3]}")
        if unexpected[:3]:
            print(f"  unexpected[:3]: {unexpected[:3]}")
    model.train(False)
    return model


def run_case(label: str, model: AdaLayerNormSingle, timestep_np: np.ndarray) -> dict:
    """Run model.forward on timestep, return params + embedded_timestep."""
    timestep_pt = torch.from_numpy(timestep_np).to(torch.float32) * TIMESTEP_SCALE
    with torch.no_grad():
        params, embedded = model(timestep_pt.flatten(), hidden_dtype=torch.float32)
    params_np = params.to(torch.float32).numpy()
    embedded_np = embedded.to(torch.float32).numpy()
    print(f"{label}: timestep shape={timestep_np.shape} -> params {params_np.shape} embedded {embedded_np.shape}")
    return {f"{label}_params": params_np, f"{label}_embedded": embedded_np, f"{label}_timestep": timestep_np}


def main() -> None:
    # PT keys are prefixed with model.diffusion_model.
    video_adaln = build_and_load("model.diffusion_model.adaln_single.", dim=4096, coef=9)
    audio_adaln = build_and_load("model.diffusion_model.audio_adaln_single.", dim=2048, coef=9)

    out: dict[str, np.ndarray] = {}

    # Case A: uniform sigma=0.7 over batch=1 (typical sampler input)
    timestep_uniform = np.array([0.7], dtype=np.float32)
    out.update(run_case("video_uniform", video_adaln, timestep_uniform))
    out.update(run_case("audio_uniform", audio_adaln, timestep_uniform))

    # Case B: per-token non-uniform (keyframe path).
    # 16 tokens, mask=[1]*8 + [0]*8 -> sigma_per_token = 0.7 * mask
    n_tokens = 16
    mask = np.array([1.0] * 8 + [0.0] * 8, dtype=np.float32)
    timestep_per_token = (0.7 * mask).reshape(1, n_tokens)
    out.update(run_case("video_per_token", video_adaln, timestep_per_token))
    out.update(run_case("audio_per_token", audio_adaln, timestep_per_token))

    np.savez("/tmp/T2_adaln_pt.npz", **out)
    print("\nwrote /tmp/T2_adaln_pt.npz")
    for k in out:
        print(f"  {k}: {out[k].shape}")


if __name__ == "__main__":
    main()
