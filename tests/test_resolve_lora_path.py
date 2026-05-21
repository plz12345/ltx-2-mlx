"""Unit tests for ``resolve_lora_path`` in utils/_orchestration.py.

Covers all four branches:
- local-exists: file on disk → returned as-is
- HF-single: snapshot_download returns a dir with one .safetensors → returned
- HF-multi-warn: multiple .safetensors → first returned + warning logged
- HF-zero-raises: no .safetensors → FileNotFoundError
"""

from __future__ import annotations

import pytest

from ltx_pipelines_mlx.utils._orchestration import resolve_lora_path


def test_local_exists(tmp_path):
    lora = tmp_path / "my_lora.safetensors"
    lora.write_bytes(b"")
    assert resolve_lora_path(str(lora)) == str(lora)


def test_hf_single(tmp_path, monkeypatch):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    sft = repo_dir / "adapter.safetensors"
    sft.write_bytes(b"")

    monkeypatch.setattr(
        "ltx_pipelines_mlx.utils._orchestration.snapshot_download",
        lambda path: str(repo_dir),
    )

    result = resolve_lora_path("some-user/my-lora")
    assert result == str(sft)


def test_hf_multi_warn(tmp_path, monkeypatch, caplog):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    files = [repo_dir / "a.safetensors", repo_dir / "b.safetensors"]
    for f in files:
        f.write_bytes(b"")

    monkeypatch.setattr(
        "ltx_pipelines_mlx.utils._orchestration.snapshot_download",
        lambda path: str(repo_dir),
    )

    import logging

    with caplog.at_level(logging.WARNING):
        result = resolve_lora_path("some-user/multi-lora")

    assert result in {str(f) for f in files}
    assert "Multiple .safetensors" in caplog.text


def test_hf_zero_raises(tmp_path, monkeypatch):
    repo_dir = tmp_path / "empty_repo"
    repo_dir.mkdir()

    monkeypatch.setattr(
        "ltx_pipelines_mlx.utils._orchestration.snapshot_download",
        lambda path: str(repo_dir),
    )

    with pytest.raises(FileNotFoundError, match=r"No \.safetensors"):
        resolve_lora_path("some-user/empty-lora")
