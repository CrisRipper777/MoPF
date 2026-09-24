from __future__ import annotations

import json

import pytest

from scripts.run_ssi_mag_v31_nc import _ensure_provenance_lock


def test_provenance_lock_dry_run_does_not_write(tmp_path) -> None:
    provenance = {"schema": "test", "git_commit": "abc", "model": "ssi_mag_v31", "task": "nc", "variant": "full"}
    path, status = _ensure_provenance_lock(tmp_path, [], provenance, dry_run=True)
    assert status == "not_written_dry_run"
    assert not path.exists()


def test_provenance_lock_same_payload_allows_resume_and_different_rejects(tmp_path) -> None:
    provenance = {"schema": "test", "git_commit": "abc", "model": "ssi_mag_v31", "task": "nc", "variant": "full"}
    path, status = _ensure_provenance_lock(tmp_path, [], provenance, dry_run=False)
    assert status == "created"
    assert json.loads(path.read_text()) == provenance
    _, status = _ensure_provenance_lock(tmp_path, [], provenance, dry_run=False)
    assert status == "validated_existing"
    changed = dict(provenance, git_commit="different")
    with pytest.raises(RuntimeError, match="provenance lock mismatch"):
        _ensure_provenance_lock(tmp_path, [], changed, dry_run=False)
