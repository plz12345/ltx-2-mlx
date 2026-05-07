"""Video modality tiling for the LTX-2 DiT.

MLX-native port of upstream ``ltx_core.modality_tiling``. Splits the
flat patchified video token sequence into spatial/temporal tiles so
each tile can be denoised independently, then blends the tile outputs
back into the full token space with trapezoidal weights at overlaps.

Combined with ``--low-ram`` (block streaming), this lets long / high-
resolution video generations fit into memory by trading wall-clock for
peak working set.

API differences vs upstream
---------------------------

Upstream operates on a ``Modality`` dataclass that bundles latent +
sigma + timesteps + positions + context + masks. Our pipeline passes
those as separate args to :meth:`LTXModel.__call__`, so this helper
takes the relevant tensors directly.

Upstream positions are stored per-token as ``(start, end)`` intervals
on each spatial/temporal axis (shape ``(B, num_axes, T, 2)``). Our
positions are point coordinates (shape ``(B, T, num_axes)``). The
overlap test for kept conditioning tokens is therefore adapted:
a conditioning token is kept iff its point coordinate falls inside
``[tile_start, tile_end)`` on every spatial/temporal dimension that
the tile splits.

Conditioning token bookkeeping
------------------------------

When a pipeline appends conditioning tokens to the end of the latent
(keyframe / reference video), the helper keeps each conditioning
token in every tile whose generated-token window covers the token's
spatial/temporal coordinate. Cond-token contributions from multiple
tiles are weighted by ``1 / num_tiles_that_kept_this_token`` so they
sum to one in the final output.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from ltx_core_mlx.model.video_vae.tiling import (
    Tile,
    TileCountConfig,
    create_tiles,
    identity_mapping_operation,
    split_by_count,
)


def _bool_to_indices(mask: mx.array) -> mx.array:
    """Convert a 1-D bool mask to an int32 array of True positions.

    MLX doesn't support boolean indexing yet, so we round-trip through
    numpy. Cost is O(num_total) per call, dominated by the host
    materialization.
    """
    return mx.array(np.flatnonzero(np.asarray(mask)))


__all__ = ["TileContext", "VideoModalityTiler"]


@dataclass(frozen=True)
class TileContext:
    """Opaque context produced by :meth:`VideoModalityTiler.tile`.

    Carries the token-level keep mask and per-conditioning-token blend
    weights needed by :meth:`VideoModalityTiler.blend`.

    Attributes:
        keep_mask: ``(num_total,)`` bool mask — True for tokens
            included in the tile.
        cond_blend_weights: ``(num_kept_cond,)`` weight per kept
            conditioning token, equal to ``1 / num_tiles_that_keep_it``.
            ``None`` when no conditioning tokens are appended.
    """

    keep_mask: mx.array
    cond_blend_weights: mx.array | None


class VideoModalityTiler:
    """Tile / blend video DiT tokens by spatial+temporal region.

    Stateless helper. Construct once with a :class:`TileCountConfig`
    and the latent ``(F, H, W)`` shape; iterate over :attr:`tiles`,
    call :meth:`tile` to slice out each tile's sub-modality, run the
    DiT on it, then accumulate the result via :meth:`blend`.

    Args:
        tiling: ``TileCountConfig`` describing tile counts + overlap
            per dimension.
        latent_shape: ``(F, H, W)`` of the patchified token grid.

    Notes:
        ``F``/``H``/``W`` are token-grid units, not pixel units —
        they are the values returned by
        :func:`ltx_core_mlx.components.patchifiers.compute_video_latent_shape`.
    """

    def __init__(self, tiling: TileCountConfig, latent_shape: tuple[int, int, int]) -> None:
        self._latent_shape = latent_shape
        F, H, W = latent_shape
        self._num_generated_tokens = F * H * W
        self._tiles: list[Tile] = create_tiles(
            (F, H, W),
            splitters=[
                split_by_count(tiling.frames.num_tiles, tiling.frames.overlap),
                split_by_count(tiling.height.num_tiles, tiling.height.overlap),
                split_by_count(tiling.width.num_tiles, tiling.width.overlap),
            ],
            mappers=[identity_mapping_operation] * 3,
        )

    @property
    def tiles(self) -> list[Tile]:
        """All tiles for the configured layout (call :meth:`tile` per tile)."""
        return self._tiles

    @property
    def num_generated_tokens(self) -> int:
        """Number of generated (non-conditioning) tokens in the full sequence."""
        return self._num_generated_tokens

    def _tile_generated_token_count(self, tile: Tile) -> int:
        f, h, w = tile.in_coords
        return (f.stop - f.start) * (h.stop - h.start) * (w.stop - w.start)

    def _generated_token_indices(self, tile: Tile) -> mx.array:
        """Flat indices of the tile's generated tokens in the full sequence."""
        _, H, W = self._latent_shape
        f, h, w = tile.in_coords
        f_idx = mx.arange(f.start, f.stop)
        h_idx = mx.arange(h.start, h.stop)
        w_idx = mx.arange(w.start, w.stop)
        return (f_idx[:, None, None] * H * W + h_idx[None, :, None] * W + w_idx[None, None, :]).reshape(-1)

    def _keep_mask(self, num_total: int, positions: mx.array, tile: Tile) -> mx.array:
        """Boolean ``(num_total,)`` mask — True for tokens this tile processes.

        Generated tokens are selected by grid position. Conditioning
        tokens (the trailing ``num_total - num_generated`` slots) are
        kept when their point position falls inside the tile range on
        every dimension, or when they have a negative time coord
        (reference tokens with ``t < 0``).
        """
        mask = mx.zeros(num_total, dtype=mx.bool_)
        gen_idx = self._generated_token_indices(tile)
        mask[gen_idx] = mx.array(True)

        if num_total <= self._num_generated_tokens:
            return mask

        # Compute tile spatial/temporal range from kept generated positions.
        # positions shape: (B, T, 3) for video. Use B=0 (assume positions
        # are batch-shared, which holds in our pipelines).
        gen_pos = positions[0, gen_idx, :]  # (num_tile_gen, 3)
        tile_start = gen_pos.min(axis=0)  # (3,)
        tile_end = gen_pos.max(axis=0)  # (3,)
        cond_pos = positions[0, self._num_generated_tokens :, :]  # (num_cond, 3)

        # Keep cond tokens whose point coords fall in [start, end] on
        # every axis. Inclusive end matches upstream's interval-overlap
        # logic for degenerate point intervals.
        in_range = (cond_pos >= tile_start[None, :]) & (cond_pos <= tile_end[None, :])
        keep_cond = in_range.all(axis=-1)  # (num_cond,)

        # Reference / negative-time tokens (e.g. IC-LoRA refs) are kept
        # in every tile.
        has_negative_time = cond_pos[:, 0] < 0
        keep_cond = keep_cond | has_negative_time

        mask[self._num_generated_tokens :] = keep_cond
        return mask

    def tile(
        self,
        latent: mx.array,
        positions: mx.array,
        attention_mask: mx.array | None,
        tile: Tile,
        normalize_positions: bool = True,
    ) -> tuple[mx.array, mx.array, mx.array | None, TileContext]:
        """Slice ``latent`` / ``positions`` / ``attention_mask`` to ``tile``.

        Args:
            latent: ``(B, T, D)`` flat token sequence (generated tokens
                followed by appended conditioning tokens).
            positions: ``(B, T, num_axes)`` per-token point positions.
            attention_mask: optional ``(B, T, T)`` self-attention mask
                in ``[0, 1]``.
            tile: which tile to extract (one of :attr:`tiles`).
            normalize_positions: when True, shift positions so the
                tile's generated tokens start at zero on every axis.

        Returns:
            ``(tiled_latent, tiled_positions, tiled_attention_mask, ctx)``.
            ``ctx`` is opaque; pass it to :meth:`blend` together with
            the model's output.
        """
        num_total = latent.shape[1]
        keep_mask = self._keep_mask(num_total, positions, tile)
        keep_idx = _bool_to_indices(keep_mask)

        tiled_latent = latent[:, keep_idx, :]
        tiled_positions = positions[:, keep_idx, :]
        if normalize_positions:
            num_tile_gen = self._tile_generated_token_count(tile)
            offset = tiled_positions[:, :num_tile_gen, :].min(axis=1, keepdims=True)
            tiled_positions = tiled_positions - offset

        tiled_attention_mask: mx.array | None = None
        if attention_mask is not None:
            tiled_attention_mask = attention_mask[:, keep_idx, :][:, :, keep_idx]

        cond_blend_weights: mx.array | None = None
        if num_total > self._num_generated_tokens:
            cond_keep = keep_mask[self._num_generated_tokens :]
            n_kept = int(cond_keep.sum().item())
            if n_kept > 0:
                # Count how many tiles keep each cond token, restrict to
                # tokens kept by THIS tile.
                cond_counts = mx.zeros(n_kept, dtype=mx.float32)
                cond_keep_idx_in_full = _bool_to_indices(cond_keep)
                for other in self._tiles:
                    other_mask = self._keep_mask(num_total, positions, other)
                    other_cond = other_mask[self._num_generated_tokens :]
                    cond_counts = cond_counts + other_cond[cond_keep_idx_in_full].astype(mx.float32)
                cond_blend_weights = 1.0 / cond_counts

        return (
            tiled_latent,
            tiled_positions,
            tiled_attention_mask,
            TileContext(keep_mask=keep_mask, cond_blend_weights=cond_blend_weights),
        )

    def blend(
        self,
        tile_output: mx.array,
        tile: Tile,
        ctx: TileContext,
        output: mx.array | None = None,
    ) -> mx.array:
        """Blend-weight the tile result and accumulate into the full token buffer.

        The tile's generated-token output is multiplied by the tile's
        per-token trapezoidal blend mask before being added to the
        output buffer at the matching positions. Conditioning-token
        output is multiplied by ``ctx.cond_blend_weights`` (so summed
        contributions from all tiles equal 1).

        Args:
            tile_output: ``(B, num_tile_tokens, D)`` from the model.
                The first ``num_tile_gen`` rows are generated tokens
                (in the tile's order), the remainder are kept cond
                tokens (in their original order in the full sequence).
            tile: the :class:`Tile` used in :meth:`tile`.
            ctx: the :class:`TileContext` from :meth:`tile`.
            output: optional pre-allocated ``(B, num_total, D)`` buffer
                to accumulate into. ``None`` allocates a fresh
                zero-filled buffer.

        Returns:
            The output buffer with the tile's contribution added.
        """
        B, _, D = tile_output.shape
        num_total = ctx.keep_mask.shape[0]
        if output is None:
            output = mx.zeros((B, num_total, D), dtype=tile_output.dtype)
        elif output.shape != (B, num_total, D):
            raise ValueError(f"output shape mismatch: expected {(B, num_total, D)}, got {output.shape}")

        num_tile_gen = self._tile_generated_token_count(tile)
        gen_idx = self._generated_token_indices(tile)
        blend_mask = tile.blend_mask.reshape(-1).astype(tile_output.dtype)

        gen_part = tile_output[:, :num_tile_gen, :] * blend_mask[None, :, None]
        output[:, gen_idx, :] = output[:, gen_idx, :] + gen_part

        if num_total > self._num_generated_tokens and ctx.cond_blend_weights is not None:
            cond_keep = ctx.keep_mask[self._num_generated_tokens :]
            cond_idx_local = _bool_to_indices(cond_keep)
            cond_idx_full = self._num_generated_tokens + cond_idx_local
            weights = ctx.cond_blend_weights.astype(tile_output.dtype)
            cond_part = tile_output[:, num_tile_gen:, :] * weights[None, :, None]
            output[:, cond_idx_full, :] = output[:, cond_idx_full, :] + cond_part

        return output
