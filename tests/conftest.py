from __future__ import annotations

import pytest

from hrx.data.loader import load
from hrx.rag.index import build_index
from hrx.rag.retriever import HybridRetriever


@pytest.fixture(scope="session")
def data():
    return load()


@pytest.fixture(scope="session")
def index():
    return build_index()


@pytest.fixture(scope="session")
def retriever(index):
    return HybridRetriever(index)
