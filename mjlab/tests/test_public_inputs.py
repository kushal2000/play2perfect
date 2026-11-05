"""Reject corrupt public inputs without replacing an existing scene or weight."""

import hashlib
import io
import json
import urllib.error

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


def test_transient_partial_download_restarts_without_appending(tmp_path, monkeypatch):
    attempts = []

    class Interrupted(io.BytesIO):
        def read(self, size):
            if self.tell():
                raise TimeoutError("connection interrupted")
            return super().read(2)

    def open_stream(*args, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise urllib.error.HTTPError("url", 503, "busy", {}, None)
        return Interrupted(b"me") if len(attempts) == 2 else io.BytesIO(b"mesh")

    monkeypatch.setattr(bundles.urllib.request, "urlopen", open_stream)
    monkeypatch.setattr(bundles.time, "sleep", lambda _: None)
    target = bundles.cached_asset(record(b"mesh"), tmp_path)
    assert len(attempts) == 3
    assert target.read_bytes() == b"mesh"
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("status,attempts", [(404, 1), (503, 3)])
def test_download_failure_is_bounded_and_leaves_no_cache(
    tmp_path, monkeypatch, status, attempts
):
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise urllib.error.HTTPError("url", status, "failure", {}, None)

    monkeypatch.setattr(bundles.urllib.request, "urlopen", fail)
    monkeypatch.setattr(bundles.time, "sleep", lambda _: None)
    with pytest.raises(urllib.error.HTTPError):
        bundles.cached_asset(record(b"mesh"), tmp_path)
    assert len(calls) == attempts
    assert list(tmp_path.iterdir()) == []
