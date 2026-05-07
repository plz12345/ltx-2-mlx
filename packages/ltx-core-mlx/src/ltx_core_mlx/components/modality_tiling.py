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


class TiledLTXModel:
    """Drop-in LTXModel wrapper that tiles the video forward across spatial/temporal regions.

    Iterates over :attr:`VideoModalityTiler.tiles`; for each tile it
    slices the video-relevant args (latent, positions, attention_mask,
    optional per-token timesteps) and calls the wrapped model with the
    tiled video + the full audio. Outputs are accumulated:

    - Video velocity / x0: blended via :meth:`VideoModalityTiler.blend`
      with trapezoidal weights at overlaps.
    - Audio velocity / x0: averaged across tiles (the audio path is
      replicated in each tile call, so per-tile outputs differ only in
      the joint audio↔video cross-attention contribution).

    Composes with :class:`~ltx_core_mlx.loader.block_streaming.StreamingLTXModel`:
    wrap the dev/distilled LTXModel in TiledLTXModel, then optionally
    in StreamingLTXModel (or vice versa — order doesn't matter, both
    intercept ``__call__`` and forward to ``self.inner``).

    Args:
        inner: An ``LTXModel`` (or another wrapper around it) — anything
            whose ``__call__`` signature matches LTXModel's.
        tiler: A pre-built :class:`VideoModalityTiler`.
    """

    def __init__(self, inner, tiler: VideoModalityTiler) -> None:
        self._inner = inner
        self._tiler = tiler

    def __call__(self, *args, **kwargs):
        if args:
            raise TypeError("TiledLTXModel expects keyword arguments only")

        video_latent = kwargs["video_latent"]
        audio_latent = kwargs["audio_latent"]
        video_positions = kwargs.get("video_positions")
        video_attention_mask = kwargs.get("video_attention_mask")
        video_timesteps = kwargs.get("video_timesteps")

        if video_positions is None:
            raise ValueError("TiledLTXModel requires video_positions to be provided.")

        video_out: mx.array | None = None
        audio_outs: list[mx.array] = []

        for tile in self._tiler.tiles:
            (
                tiled_video,
                tiled_video_pos,
                tiled_video_mask,
                ctx,
            ) = self._tiler.tile(
                video_latent,
                video_positions,
                video_attention_mask,
                tile,
                normalize_positions=False,
            )

            tiled_video_timesteps = None
            if video_timesteps is not None:
                # video_timesteps is (B, T) — slice with the same keep_mask
                # as the latent so per-token timesteps stay aligned.
                keep_idx = _bool_to_indices(ctx.keep_mask)
                tiled_video_timesteps = video_timesteps[:, keep_idx]

            tile_kwargs = dict(kwargs)
            tile_kwargs["video_latent"] = tiled_video
            tile_kwargs["audio_latent"] = audio_latent
            tile_kwargs["video_positions"] = tiled_video_pos
            tile_kwargs["video_attention_mask"] = tiled_video_mask
            if video_timesteps is not None:
                tile_kwargs["video_timesteps"] = tiled_video_timesteps

            tile_video_out, tile_audio_out = self._inner(**tile_kwargs)

            video_out = self._tiler.blend(tile_video_out, tile, ctx, output=video_out)
            audio_outs.append(tile_audio_out)

        # Average audio across tiles. Each tile saw the full audio so its
        # audio output is a complete prediction; averaging absorbs the
        # variation introduced by the differing video context.
        if len(audio_outs) == 1:
            audio_out = audio_outs[0]
        else:
            audio_out = mx.mean(mx.stack(audio_outs, axis=0), axis=0)

        return video_out, audio_out

    def __getattr__(self, name: str):
        # Proxy other attribute reads (e.g. ``self.config``) to the inner model.
        if name in {"_inner", "_tiler"}:
            raise AttributeError(name)
        try:
            inner = object.__getattribute__(self, "_inner")
        except AttributeError as e:
            raise AttributeError(name) from e
        return getattr(inner, name)
