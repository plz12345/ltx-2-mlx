# Keyframe parity tests (MLX vs upstream PT)

Targeted PT-vs-MLX parity tests covering the keyframe pipeline code path,
used to localize the keyframe hold-cut-decay regression to the **VAE
decoder forward** (2026-05-06 investigation).

## Verdict

| Test  | Component                              | Result         |
|-------|----------------------------------------|----------------|
| A     | keyframe conditioning code             | bit-exact      |
| B     | sampler step (mask blend, Euler)       | bit-exact      |
| T1    | VAE encoder forward                    | 1 ULP bf16     |
| T2    | AdaLayerNormSingle (uniform + per-token) | a few ULPs fp32 |
| T5    | **VAE decoder forward**                | **DIVERGES (max 0.97, 57% rel)** |

## How to run

PT side (in upstream `Lightricks/LTX-2` checkout):

```bash
cd /Users/dgrauet/sandbox/ltx-reference
uv run python /Users/dgrauet/Work/mlx/ports/ltx-2-mlx/tests/parity_keyframe/dump_pt.py
uv run python /Users/dgrauet/Work/mlx/ports/ltx-2-mlx/tests/parity_keyframe/dump_pt_sampler.py
uv run python /Users/dgrauet/Work/mlx/ports/ltx-2-mlx/tests/parity_keyframe/T1_dump_pt_vae_encoder.py
uv run python /Users/dgrauet/Work/mlx/ports/ltx-2-mlx/tests/parity_keyframe/T2_dump_pt_adaln.py
uv run python /Users/dgrauet/Work/mlx/ports/ltx-2-mlx/tests/parity_keyframe/T5_dump_pt_vae_decoder.py
```

MLX side (in this repo):

```bash
uv run python tests/parity_keyframe/check_mlx.py
uv run python tests/parity_keyframe/check_mlx_sampler.py
uv run python tests/parity_keyframe/T1_check_mlx_vae_encoder.py
uv run python tests/parity_keyframe/T2_check_mlx_adaln.py
uv run python tests/parity_keyframe/T5_check_mlx_vae_decoder.py
```

A/B/T1/T2 print `ALL OK` (max_abs <= 5e-3). T5 prints `FAIL` and a
per-frame divergence table — kept as the diagnostic reference until
the VAE decoder bug is fixed.

## Weight setup

PT and MLX share the same bf16 weights from
`dgrauet/ltx-2.3-mlx-q8/vae_{encoder,decoder}.safetensors` and
`dgrauet/ltx-2.3-mlx/transformer-dev.safetensors`. PT side strips the
MLX prefix and transposes Conv3d weights `(O, K_t, K_h, K_w, I)` ->
`(O, I, K_t, K_h, K_w)`. The full upstream
`Lightricks/LTX-2/ltx-2.3-22b-dev.safetensors` is also cached but
unused by these tests (would be needed for a full LTXModel forward
parity, see `T3` placeholder TBD).

## Per-frame divergence in T5 (current MLX decoder)

Cumulative temporal error pattern — each frame slightly worse than
the previous, suggesting something in the temporal processing chain
(causal padding, frame-trim after pixel_shuffle_3d, channel
ordering in DepthToSpaceUpsample, or PixelNorm timing).

```
frame 0: diff=0.014
frame 1: diff=0.001
frame 2: diff=0.016
frame 3: diff=0.025
frame 4: diff=0.035
frame 5: diff=0.035
frame 6: diff=0.048
frame 7: diff=0.049
frame 8: diff=0.048
```

This explains why T2V works (smooth latents -> per-frame diff
invisible) but keyframe breaks (sharp latent boundary at frame 3->4
amplifies into a visible cut at pixel 24-25).
