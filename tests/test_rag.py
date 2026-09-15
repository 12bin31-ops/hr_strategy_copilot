from __future__ import annotations

from pathlib import Path

import pytest

from hrx.rag.chunking import chunk_corpus, chunk_document
from hrx.rag.citations import audit, extract_citations
from hrx.rag.index import build_index, load_index
from hrx.rag.retriever import route
from hrx.rag.tokenize import tokenize


def test_chunking_keeps_article_boundaries():
    path = Path("data/corpus/internal/01_인사규정.md")
    chunks = chunk_document(path)
    articles = [c.article for c in chunks if c.article]
    assert "제5조" in articles and "제15조" in articles
    # 한 청크가 두 개의 조를 삼키지 않아야 한다
    for c in chunks:
        assert c.text.count("### 제") == 0
    assert all(c.doc_id == "INT-HR-001" for c in chunks)


def test_frontmatter_is_parsed():
    chunks = chunk_document(Path("data/corpus/external/01_근로기준법_요지.md"))
    assert chunks[0].corpus == "external"
    assert chunks[0].category == "법령"


def test_chunk_ids_unique():
    chunks = chunk_corpus()
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_tokenizer_strips_particles():
    toks = tokenize("취업규칙을 고용노동부에 신고하여야 한다")
    assert "취업규칙" in toks and "고용노동부" in toks


def test_index_row_order_matches_chunks(tmp_path):
    """BM25/행렬의 행 순서와 SQLite 로드 순서가 어긋나면 점수가 엉뚱한 청크에 붙는다."""
    built = build_index(index_dir=tmp_path)
    loaded = load_index(index_dir=tmp_path)
    assert [c.chunk_id for c in built.chunks] == [c.chunk_id for c in loaded.chunks]
    assert loaded.matrix.shape[0] == len(loaded.chunks)
    assert len(loaded.bm25.doc_freqs) == len(loaded.chunks)


@pytest.mark.parametrize(
    "query,expected",
    [
        ("취업규칙 불이익 변경 동의 절차", "external"),
        ("우리 회사 인사규정 제5조", "internal"),
    ],
)
def test_router(query, expected):
    assert expected in route(query).corpora


def test_search_returns_scored_results(retriever):
    res = retriever.search("평가등급 배분율", top_k=5)
    assert 1 <= len(res) <= 5
    assert [r.rank for r in res] == list(range(1, len(res) + 1))
    assert all(r.score > 0 for r in res)


def test_category_filter_is_hard(retriever):
    res = retriever.search("정년", top_k=5, corpora=("external",), filter_categories=("법령",))
    assert res and all(r.chunk.category == "법령" for r in res)


def test_context_budget_is_respected(retriever):
    ctx, kept = retriever.build_context("승진 적체", top_k=8, char_budget=1500)
    assert len(ctx) <= 1700  # 블록 구분자 여유
    assert kept


def test_citation_audit_detects_unknown():
    text = "차장 적체가 심각하다 [INT-HR-005§제10조]. 근거 없는 주장 [FAKE-DOC-999§제1조]."
    report = audit(text, known_keys={"INT-HR-005§제10조"})
    assert report.unknown_citations == ["FAKE-DOC-999§제1조"]
    assert not report.is_grounded


def test_citation_supports_backtick_form():
    assert extract_citations("표 안의 근거 `INT-HR-003§제4조` 입니다")


def test_citation_coverage_counts_table_rows():
    md = "| 항목 | 근거 |\n|---|---|\n| 침투율 격차 11%p | `INT-HR-003§제4조` |"
    report = audit(md)
    assert report.total_claims == 1 and report.cited_claims == 1
