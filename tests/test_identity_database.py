import pytest

pytest.importorskip("torch")

import torch

from technical.pipeline.pipeline import IdentityDatabase


def test_identity_database_returns_none_when_empty():
    db = IdentityDatabase(threshold=0.5)
    query = torch.randn(512)
    assert db.lookup(query) is None


def test_identity_database_returns_best_match():
    torch.manual_seed(0)
    db = IdentityDatabase(threshold=0.4)
    anchor = torch.randn(512)
    db.add("alice", anchor)
    db.add("bob", torch.randn(512))

    match = db.lookup(anchor)
    assert match is not None
    label, score = match
    assert label == "alice"
    assert score >= 0.4
