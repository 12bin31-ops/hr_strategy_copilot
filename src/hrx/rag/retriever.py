"""하이브리드 검색기.

파이프라인:
    질의 → 라우팅(코퍼스 선택) → BM25 / TF-IDF 병렬 검색 → RRF 융합
         → MMR 다양화 → 컨텍스트 예산 패킹

RRF(Reciprocal Rank Fusion)를 쓰는 이유: BM25 점수와 코사인 유사도는 스케일이 달라
가중합으로 섞으면 하이퍼파라미터에 민감해진다. 순위만 쓰는 RRF 는 튜닝 없이 안정적이다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from hrx.config import SETTINGS
from hrx.rag.chunking import Chunk
from hrx.rag.index import IndexBundle, load_index
from hrx.rag.tokenize import tokenize

# 라우팅 규칙: (정규식, 대상 코퍼스, 가중 카테고리)
ROUTING_RULES: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
    (r"근로기준법|법령|법적|위법|불이익\s?변경|취업규칙|노사협의|동의\s?절차|차별|정년|연장근로",
     ("external",), ("법령",)),
    (r"트렌드|타사|업계|벤치마크|사례|다른\s?회사|일반적으로",
     ("external",), ("트렌드", "사례")),
    (r"우리\s?회사|사내|규정|지침|세칙|제\s?\d+조|한빛전자",
     ("internal",), ()),
    (r"승계|후계자|파이프라인|Ready\s?Now|핵심\s?보직",
     ("internal", "external"), ("승계관리",)),
    (r"평가|등급|배분|캘리브레이션|관대화",
     ("internal", "external"), ("성과관리",)),
    (r"보상|임금|연봉|성과급|직무급|침투율|밴드",
     ("internal", "external"), ("보상",)),
    (r"인원|인건비|정원|조직\s?설계|스팬|적체|이직률",
     ("internal", "external"), ("인력계획",)),
]

ALL_CORPORA = ("internal", "external")


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    rank: int
    bm25_rank: int | None = None
    dense_rank: int | None = None
    bm25_score: float = 0.0
    dense_score: float = 0.0

    def explain(self) -> str:
        return (
            f"#{self.rank} {self.chunk.citation_key} "
            f"(RRF={self.score:.4f}, bm25 r{self.bm25_rank} s{self.bm25_score:.2f}, "
            f"tfidf r{self.dense_rank} s{self.dense_score:.3f})"
        )


@dataclass
class RouteDecision:
    corpora: tuple[str, ...]
    boosted_categories: tuple[str, ...]
    matched_rules: list[str] = field(default_factory=list)


def route(query: str) -> RouteDecision:
    corpora: set[str] = set()
    cats: set[str] = set()
    matched: list[str] = []
    for pattern, target, categories in ROUTING_RULES:
        if re.search(pattern, query, flags=re.IGNORECASE):
            corpora.update(target)
            cats.update(categories)
            matched.append(pattern.split("|")[0])
    if not corpora:
        corpora = set(ALL_CORPORA)
    return RouteDecision(tuple(sorted(corpora)), tuple(sorted(cats)), matched)


def _rrf(rank: int, k: int) -> float:
    return 1.0 / (k + rank)


def _mmr(
    matrix, candidate_idx: list[int], scores: dict[int, float], k: int, lambda_: float = 0.85
) -> list[int]:
    """최대 한계 연관성 — 같은 조문만 여러 개 뽑히는 것을 막는다."""
    if len(candidate_idx) <= k:
        return candidate_idx
    # RRF 점수(≈0.01~0.03)와 코사인 유사도(0~1)는 스케일이 다르다.
    # 정규화하지 않으면 중복 페널티가 연관도를 압도해 정답이 밀려난다.
    vals = np.array([scores[i] for i in candidate_idx], dtype=float)
    lo, hi = float(vals.min()), float(vals.max())
    norm = {i: (scores[i] - lo) / (hi - lo) if hi > lo else 1.0 for i in candidate_idx}
    scores = norm
    sub = matrix[candidate_idx]
    sim = cosine_similarity(sub)
    selected: list[int] = []
    remaining = list(range(len(candidate_idx)))
    while remaining and len(selected) < k:
        best, best_val = None, -1e9
        for pos in remaining:
            relevance = scores[candidate_idx[pos]]
            redundancy = max((sim[pos][s] for s in selected), default=0.0)
            val = lambda_ * relevance - (1 - lambda_) * redundancy
            if val > best_val:
                best, best_val = pos, val
        selected.append(best)
        remaining.remove(best)
    return [candidate_idx[p] for p in selected]


class HybridRetriever:
    def __init__(self, index: IndexBundle | None = None):
        self.index = index or load_index()
        self.cfg = SETTINGS.retrieval

    # ------------------------------------------------------------------ search
    def search(
        self,
        query: str,
        top_k: int | None = None,
        corpora: tuple[str, ...] | None = None,
        categories: tuple[str, ...] | None = None,
        filter_categories: tuple[str, ...] | None = None,
        use_router: bool = True,
    ) -> list[SearchResult]:
        cfg = self.cfg
        top_k = top_k or cfg.top_k
        decision = route(query) if use_router else RouteDecision(ALL_CORPORA, ())
        target_corpora = corpora or decision.corpora
        boost_cats = categories if categories is not None else decision.boosted_categories

        chunks = self.index.chunks
        allowed = np.array(
            [
                i for i, c in enumerate(chunks)
                if c.corpus in target_corpora
                and (filter_categories is None or c.category in filter_categories)
            ],
            dtype=int,
        )
        if allowed.size == 0:
            allowed = np.arange(len(chunks))

        bm25_all = self.index.bm25.score(tokenize(query))
        qv = self.index.vectorizer.transform([query])
        dense_all = cosine_similarity(qv, self.index.matrix).ravel()

        # 카테고리 부스트는 점수가 아니라 순위 산출 전 소폭 가산으로만 적용
        if boost_cats:
            boost = np.array(
                [1.12 if chunks[i].category in boost_cats else 1.0 for i in range(len(chunks))]
            )
            bm25_all = bm25_all * boost
            dense_all = dense_all * boost

        cand_k = min(cfg.candidate_k, allowed.size)
        bm25_order = allowed[np.argsort(-bm25_all[allowed])][:cand_k]
        dense_order = allowed[np.argsort(-dense_all[allowed])][:cand_k]

        # 관련도 하한선: 두 신호 모두 사실상 0 인 후보는 후보군에서 제외한다.
        # RRF 는 "한 리스트에만 들어와도 점수를 준다"는 성질이 있어, 필터가 없으면
        # 문자 n-gram 이 우연히 걸린 무관한 조문이 상위로 올라온다.
        bm25_floor = max(1e-6, float(bm25_all[bm25_order[0]]) * 0.08) if bm25_order.size else 0.0
        dense_floor = max(1e-6, float(dense_all[dense_order[0]]) * 0.30) if dense_order.size else 0.0
        bm25_order = [i for i in bm25_order if bm25_all[i] >= bm25_floor]
        dense_order = [i for i in dense_order if dense_all[i] >= dense_floor]

        bm25_rank = {int(idx): r + 1 for r, idx in enumerate(bm25_order)}
        dense_rank = {int(idx): r + 1 for r, idx in enumerate(dense_order)}

        # 규정·법령 질의는 고유 용어 일치가 결정적이므로 BM25 리스트에 더 큰 가중치를 준다.
        fused: dict[int, float] = {}
        for idx, r in bm25_rank.items():
            fused[idx] = fused.get(idx, 0.0) + 1.0 * _rrf(r, cfg.rrf_k)
        for idx, r in dense_rank.items():
            fused[idx] = fused.get(idx, 0.0) + 0.7 * _rrf(r, cfg.rrf_k)
        if not fused:
            return []

        ordered = sorted(fused, key=lambda i: -fused[i])[: max(top_k * 3, top_k)]
        diversified = _mmr(self.index.matrix, ordered, fused, top_k)
        diversified.sort(key=lambda i: -fused[i])

        return [
            SearchResult(
                chunk=chunks[idx],
                score=fused[idx],
                rank=rank,
                bm25_rank=bm25_rank.get(idx),
                dense_rank=dense_rank.get(idx),
                bm25_score=float(bm25_all[idx]),
                dense_score=float(dense_all[idx]),
            )
            for rank, idx in enumerate(diversified, start=1)
        ]

    # ----------------------------------------------------------------- packing
    def build_context(
        self,
        query: str,
        top_k: int | None = None,
        char_budget: int | None = None,
        **kwargs,
    ) -> tuple[str, list[SearchResult]]:
        """검색 결과를 컨텍스트 예산 안에서 문자열로 패킹한다.

        문서 편중을 막기 위해 문서당 사용량 상한(per_source_char_budget)을 함께 건다.
        """
        results = self.search(query, top_k=top_k, **kwargs)
        budget = char_budget or SETTINGS.harness.context_char_budget
        per_doc = SETTINGS.harness.per_source_char_budget
        used_by_doc: dict[str, int] = {}
        blocks: list[str] = []
        total = 0
        kept: list[SearchResult] = []
        for r in results:
            doc_used = used_by_doc.get(r.chunk.doc_id, 0)
            room = min(budget - total, per_doc - doc_used)
            if room <= 200:
                continue
            block = r.chunk.as_context(max_chars=room)
            blocks.append(block)
            total += len(block)
            used_by_doc[r.chunk.doc_id] = doc_used + len(block)
            kept.append(r)
            if total >= budget:
                break
        return "\n\n---\n\n".join(blocks), kept
