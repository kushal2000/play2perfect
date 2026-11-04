"""Reject corrupt public inputs without replacing an existing scene or weight."""

import hashlib
import io
import json

import pytest

from play2perfect_mjlab import bundles, foundation


def record(content):
    return {
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
        "url": "https://example.invalid/pinned-asset",
    }


def test_download_corruption_never_enters_cache(tmp_path, monkeypatch):
    expected = record(b"mesh")
    monkeypatch.setattr(
        bundles.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"oops")
    )
    with pytest.raises(ValueError, match="checksum"):
        bundles.cached_asset(expected, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_existing_corrupt_cache_is_preserved(tmp_path, monkeypatch):
    expected = record(b"mesh")
    path = tmp_path / expected["sha256"]
    path.write_bytes(b"oops")
    monkeypatch.setattr(
        bundles.urllib.request, "urlopen", lambda *a, **k: pytest.fail("Network used")
    )
    with pytest.raises(ValueError, match="checksum"):
        bundles.cached_asset(expected, tmp_path)
    assert path.read_bytes() == b"oops"


def test_scene_preparation_is_atomic_and_repeatable(tmp_path, monkeypatch):
    resource = tmp_path / "resources"
    (resource / "scene").mkdir(parents=True)
    expected = record(b"mesh")
    manifest = json.dumps({"files": {"part.obj": expected}}).encode()
    (resource / "scene/manifest.json").write_bytes(manifest)
    definition = {
        "destination": "scene",
        "metadata": {"manifest.json": hashlib.sha256(manifest).hexdigest()},
        "asset_hashes": {"part.obj": expected["sha256"]},
    }
    monkeypatch.setattr(bundles, "resources", lambda: resource)
    monkeypatch.setattr(
        bundles,
        "catalog",
        lambda: {
            "bundles": {"scene": definition},
            "assets": {expected["sha256"]: expected},
        },
    )
    monkeypatch.setattr(
        bundles.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"oops")
    )
    destination = tmp_path / "data"
    with pytest.raises(ValueError, match="checksum"):
        bundles.prepare_bundle("scene", destination)
    assert not (destination / "scene").exists()
    assert not list(destination.glob(".prepare-*"))
    monkeypatch.setattr(
        bundles.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"mesh")
    )
    scene = bundles.prepare_bundle("scene", destination)
    monkeypatch.setattr(
        bundles.urllib.request, "urlopen", lambda *a, **k: pytest.fail("Network used")
    )
    assert bundles.prepare_bundle("scene", destination) == scene
    (scene / "manifest.json").write_text("changed by user")
    with pytest.raises(ValueError, match="checksum"):
        bundles.prepare_bundle("scene", destination)
    assert (scene / "manifest.json").read_text() == "changed by user"


def test_foundation_rejects_changed_existing_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(foundation, "FOUNDATION", record(b"weights"))
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"changed")
    monkeypatch.setattr(
        foundation, "cached_asset", lambda *a, **k: pytest.fail("Download attempted")
    )
    with pytest.raises(ValueError, match="checksum"):
        foundation.prepare_foundation(tmp_path)
    assert checkpoint.read_bytes() == b"changed"
