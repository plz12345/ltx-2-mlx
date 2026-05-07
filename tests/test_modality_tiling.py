"""Unit tests for VideoModalityTiler."""

from __future__ import annotations

import mlx.core as mx

from ltx_core_mlx.components.modality_tiling import VideoModalityTiler
from ltx_core_mlx.model.video_vae.tiling import DimensionTilingConfig, TileCountConfig


def _make_positions(F: int, H: int, W: int) -> mx.array:  # noqa: N803
    """(1, F*H*W, 3) point coords matching the (f, h, w) grid."""
    f_idx = mx.arange(F)[:, None, None].astype(mx.float32) * mx.ones((F, H, W), dtype=mx.float32)
    h_idx = mx.arange(H)[None, :, None].astype(mx.float32) * mx.ones((F, H, W), dtype=mx.float32)
    w_idx = mx.arange(W)[None, None, :].astype(mx.float32) * mx.ones((F, H, W), dtype=mx.float32)
    pos = mx.stack([f_idx, h_idx, w_idx], axis=-1).reshape(1, F * H * W, 3)
    return pos


class TestVideoModalityTiler:
    def test_tile_count_with_overlap(self):
        tiling = TileCountConfig(
            frames=DimensionTilingConfig(num_tiles=2, overlap=1),
            height=DimensionTilingConfig(num_tiles=2, overlap=2),
            width=DimensionTilingConfig(num_tiles=1),
        )
        tiler = VideoModalityTiler(tiling, latent_shape=(4, 8, 8))
        # 2 (frames) * 2 (height) * 1 (width) = 4 tiles
        assert len(tiler.tiles) == 4

    def test_single_tile_round_trip(self):
        """1x1x1 tile: blend(tile(latent)) must equal latent itself."""
        F, H, W, D = 4, 6, 8, 16
        T = F * H * W
        tiling = TileCountConfig()  # 1 tile per dim
        tiler = VideoModalityTiler(tiling, latent_shape=(F, H, W))
        assert len(tiler.tiles) == 1

        latent = mx.random.normal((1, T, D))
        positions = _make_positions(F, H, W)
        tile = tiler.tiles[0]

        tiled, tiled_pos, _, ctx = tiler.tile(latent, positions, None, tile, normalize_positions=False)
        assert tiled.shape == latent.shape

        out = tiler.blend(tiled, tile, ctx)
        mx.eval(latent, out)
        assert mx.allclose(latent, out, atol=1e-6).item()

    def test_multi_tile_no_overlap_round_trip(self):
        """Tiling with no overlap and trapezoidal mask = 1.0 everywhere
        should reconstruct identity when blend masks are uniform."""
        F, H, W, D = 4, 8, 8, 8
        T = F * H * W
        tiling = TileCountConfig(
            frames=DimensionTilingConfig(num_tiles=2, overlap=0),
            height=DimensionTilingConfig(num_tiles=2, overlap=0),
        )
        tiler = VideoModalityTiler(tiling, latent_shape=(F, H, W))
        assert len(tiler.tiles) == 4

        latent = mx.random.normal((1, T, D))
        positions = _make_positions(F, H, W)

        output = mx.zeros_like(latent)
        for t in tiler.tiles:
            tiled, _, _, ctx = tiler.tile(latent, positions, None, t, normalize_positions=False)
            output = tiler.blend(tiled, t, ctx, output=output)
        mx.eval(latent, output)
        # No overlap → blend masks are all 1.0, sum gives back identity.
        assert mx.allclose(latent, output, atol=1e-6).item()

    def test_multi_tile_with_overlap_blend_weights_sum_to_one(self):
        """With overlap, blended output must equal the original where
        per-pixel blend weights across all tiles sum to 1."""
        F, H, W, D = 4, 8, 8, 4
        T = F * H * W
        tiling = TileCountConfig(
            frames=DimensionTilingConfig(num_tiles=1),
            height=DimensionTilingConfig(num_tiles=2, overlap=2),
        )
        tiler = VideoModalityTiler(tiling, latent_shape=(F, H, W))

        latent = mx.random.normal((1, T, D))
        positions = _make_positions(F, H, W)

        output = mx.zeros_like(latent)
        for t in tiler.tiles:
            tiled, _, _, ctx = tiler.tile(latent, positions, None, t, normalize_positions=False)
            output = tiler.blend(tiled, t, ctx, output=output)
        mx.eval(latent, output)
        # Trapezoidal masks sum to 1 across the overlap → identity.
        assert mx.allclose(latent, output, atol=1e-5).item()

    def test_position_normalization(self):
        """When normalize_positions=True, the tile's generated tokens
        should start at coord zero on every axis."""
        F, H, W = 4, 8, 8
        tiling = TileCountConfig(
            frames=DimensionTilingConfig(num_tiles=2, overlap=0),
            height=DimensionTilingConfig(num_tiles=2, overlap=0),
        )
        tiler = VideoModalityTiler(tiling, latent_shape=(F, H, W))

        latent = mx.zeros((1, F * H * W, 4))
        positions = _make_positions(F, H, W)

        for t in tiler.tiles:
            _, tiled_pos, _, _ = tiler.tile(latent, positions, None, t, normalize_positions=True)
            num_tile_gen = (
                (t.in_coords[0].stop - t.in_coords[0].start)
                * (t.in_coords[1].stop - t.in_coords[1].start)
                * (t.in_coords[2].stop - t.in_coords[2].start)
            )
            mx.eval(tiled_pos)
            gen_pos = tiled_pos[:, :num_tile_gen, :]
            assert float(gen_pos.min().item()) == 0.0

    def test_attention_mask_subset(self):
        """When attention_mask is given, the tiled mask is the kept
        rows x kept cols submatrix."""
        F, H, W = 2, 4, 4
        T = F * H * W
        tiling = TileCountConfig(height=DimensionTilingConfig(num_tiles=2, overlap=0))
        tiler = VideoModalityTiler(tiling, latent_shape=(F, H, W))

        latent = mx.zeros((1, T, 4))
        positions = _make_positions(F, H, W)
        full_mask = mx.random.normal((1, T, T))

        tile = tiler.tiles[0]
        tiled, _, tiled_mask, _ = tiler.tile(latent, positions, full_mask, tile)
        assert tiled_mask is not None
        assert tiled_mask.shape == (1, tiled.shape[1], tiled.shape[1])

    def test_cond_tokens_kept_by_overlap(self):
        """Conditioning tokens whose point coords fall in the tile range
        are kept; others are dropped."""
        F, H, W, D = 2, 4, 4, 4
        T_gen = F * H * W
        # Two cond tokens: one in each height-half of the grid.
        cond_pos = mx.array([[[0.0, 0.0, 0.0], [0.0, 3.0, 0.0]]])  # (1, 2, 3)
        positions = mx.concatenate([_make_positions(F, H, W), cond_pos], axis=1)
        latent = mx.random.normal((1, T_gen + 2, D))

        tiling = TileCountConfig(height=DimensionTilingConfig(num_tiles=2, overlap=0))
        tiler = VideoModalityTiler(tiling, latent_shape=(F, H, W))

        # First tile covers h=[0,2): only the first cond (h=0) is kept.
        # Second tile covers h=[2,4): only the second cond (h=3) is kept.
        for t in tiler.tiles:
            _, _, _, ctx = tiler.tile(latent, positions, None, t)
            cond_keep = ctx.keep_mask[T_gen:]
            mx.eval(cond_keep)
            assert int(cond_keep.sum().item()) == 1
