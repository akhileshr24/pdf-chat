
from types import SimpleNamespace

import pytest

from app.retrieval.hybrid_retriever import fuse_results


def make_point(point_id):
    return SimpleNamespace(
        id=point_id,
        payload={"text": f"Chunk {point_id}"},
    )


def test_fuse_results():
    dense = [
        make_point("A"),
        make_point("B"),
    ]

    bm25 = [
        make_point("A"),
        make_point("C"),
    ]

    results = fuse_results(dense, bm25, limit=3)

    assert len(results) == 3
    assert results[0]["id"] == "A"
    assert results[0]["score"] > results[1]["score"]


def test_fuse_results_removes_duplicates():
    dense = [make_point("A")]
    bm25 = [make_point("A")]

    results = fuse_results(dense, bm25)

    assert len(results) == 1


def test_invalid_limit():
    with pytest.raises(ValueError):
        fuse_results([], [], limit=0)


def test_invalid_k():
    with pytest.raises(ValueError):
        fuse_results([], [], k=0)
