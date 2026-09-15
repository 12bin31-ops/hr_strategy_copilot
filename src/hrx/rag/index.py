"""인덱스 구축·저장.

두 개의 희소 표현을 함께 만든다.

  * BM25          : 어휘 정확 매칭 — "제5조", "침투율" 같은 고유 용어에 강함
  * TF-IDF (char) : 2~4 음절 char n-gram — 어미 변형·복합명사 부분일치에 강함

둘을 RRF 로 결합하면(retriever.py) 한국어 규정 문서에서 단일 방식보다 안정적이다.
임베딩 모델은 선택적으로 붙일 수 있게 훅만 열어 두고, 기본값은 오프라인 동작이다.
"""

from __future__ import annotations

import json
import math
import pickle
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from hrx.config import SETTINGS
from hrx.rag.chunking import Chunk, chunk_corpus
from hrx.rag.tokenize import tokenize

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL,
    title       TEXT NOT NULL,
    corpus      TEXT NOT NULL,
    category    TEXT,
    section     TEXT,
    article     TEXT,
    text        TEXT NOT NULL,
    source_path TEXT,
    ord         INTEGER,
    meta        TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_corpus ON chunks(corpus);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
"""


@dataclass
class BM25:
    """표준 Okapi BM25. 외부 의존성 없이 직접 구현해 동작을 투명하게 둔다."""

    doc_freqs: list[Counter]
    doc_len: np.ndarray
    avgdl: float
    idf: dict[str, float]
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def fit(cls, docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> "BM25":
        doc_freqs = [Counter(d) for d in docs]
        doc_len = np.array([len(d) for d in docs], dtype=float)
        avgdl = float(doc_len.mean()) if len(doc_len) else 0.0
        df: Counter = Counter()
        for c in doc_freqs:
            df.update(c.keys())
        n = len(docs)
        idf = {t: math.log(1 + (n - v + 0.5) / (v + 0.5)) for t, v in df.items()}
        return cls(doc_freqs, doc_len, avgdl, idf, k1, b)

    def score(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(len(self.doc_freqs), dtype=float)
        if self.avgdl == 0:
            return scores
        for t in query_tokens:
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, freqs in enumerate(self.doc_freqs):
                f = freqs.get(t)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * (f * (self.k1 + 1) / denom)
        return scores


@dataclass
class IndexBundle:
    chunks: list[Chunk]
    bm25: BM25
    vectorizer: TfidfVectorizer
    matrix: object  # scipy sparse

    @property
    def size(self) -> int:
        return len(self.chunks)


def _write_sqlite(chunks: list[Chunk], db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    try:
        con.executescript(SCHEMA)
        con.execute("DELETE FROM chunks")
        con.executemany(
            "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    c.chunk_id, c.doc_id, c.title, c.corpus, c.category, c.section,
                    c.article, c.text, c.source_path, c.order, json.dumps(c.meta, ensure_ascii=False),
                )
                for c in chunks
            ],
        )
        con.commit()
    finally:
        con.close()


def read_sqlite(db_path: Path) -> list[Chunk]:
    con = sqlite3.connect(db_path)
    try:
        # ORDER BY rowid = 삽입 순서. BM25/TF-IDF 행렬의 행 순서와 반드시 일치해야 하므로
        # doc_id 같은 논리 키로 정렬하면 안 된다(정렬 기준이 어긋나면 점수가 엉뚱한 청크에 붙는다).
        rows = con.execute(
            "SELECT chunk_id,doc_id,title,corpus,category,section,article,text,source_path,ord,meta"
            " FROM chunks ORDER BY rowid"
        ).fetchall()
    finally:
        con.close()
    return [
        Chunk(
            chunk_id=r[0], doc_id=r[1], title=r[2], corpus=r[3], category=r[4],
            section=r[5], article=r[6], text=r[7], source_path=r[8], order=r[9],
            meta=json.loads(r[10] or "{}"),
        )
        for r in rows
    ]


def build_index(corpus_dir: Path | None = None, index_dir: Path | None = None) -> IndexBundle:
    cfg = SETTINGS.retrieval
    chunks = chunk_corpus(corpus_dir)
    if not chunks:
        raise RuntimeError(f"코퍼스가 비어 있습니다: {corpus_dir or SETTINGS.paths.corpus}")

    tokenized = [tokenize(f"{c.title} {c.section} {c.text}") for c in chunks]
    bm25 = BM25.fit(tokenized, k1=cfg.bm25_k1, b=cfg.bm25_b)

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=cfg.char_ngram,
        min_df=1,
        sublinear_tf=True,
        max_features=200_000,
    )
    matrix = vectorizer.fit_transform([f"{c.section} {c.text}" for c in chunks])

    out_dir = index_dir or SETTINGS.paths.index
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_sqlite(chunks, out_dir / "chunks.sqlite")
    with (out_dir / "sparse.pkl").open("wb") as fh:
        pickle.dump(
            {
                "bm25": bm25,
                "vectorizer": vectorizer,
                "matrix": matrix,
                # 행 순서 검증용 지문 — 로드 시 청크 순서가 어긋나면 즉시 실패시킨다.
                "chunk_ids": [c.chunk_id for c in chunks],
            },
            fh,
        )
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "chunks": len(chunks),
                "docs": len({c.doc_id for c in chunks}),
                "corpora": sorted({c.corpus for c in chunks}),
                "vocab": len(bm25.idf),
                "char_ngram": list(cfg.char_ngram),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return IndexBundle(chunks, bm25, vectorizer, matrix)


def load_index(index_dir: Path | None = None, auto_build: bool = True) -> IndexBundle:
    d = index_dir or SETTINGS.paths.index
    sparse = d / "sparse.pkl"
    db = d / "chunks.sqlite"
    if not sparse.exists() or not db.exists():
        if not auto_build:
            raise FileNotFoundError(f"인덱스가 없습니다: {d}. `hrx index` 를 먼저 실행하세요.")
        return build_index(index_dir=d)
    with sparse.open("rb") as fh:
        blob = pickle.load(fh)
    chunks = read_sqlite(db)
    expected = blob.get("chunk_ids")
    if expected is not None and [c.chunk_id for c in chunks] != expected:
        # 순서가 어긋나면 점수가 다른 청크에 붙는다. 조용히 틀리느니 다시 만든다.
        return build_index(index_dir=d)
    return IndexBundle(chunks, blob["bm25"], blob["vectorizer"], blob["matrix"])
