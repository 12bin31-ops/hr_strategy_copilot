"""에이전트 도구.

설계 원칙 두 가지.

1. **도구는 순수 파이썬 함수다.** CrewAI 에 의존하지 않으므로 단위 테스트가 가능하고,
   LLM 이 없는 폴백 경로에서도 같은 함수를 그대로 쓴다.
2. **모든 도구는 인용 키를 함께 돌려준다.** 정량 결과에는 `DATA§지표명`,
   문서 검색 결과에는 `문서ID§조항` 을 붙인다. 이렇게 하면 "숫자는 맞는데 출처가 없는"
   문장을 검증 단계에서 기계적으로 걸러낼 수 있다.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from hrx.analytics import org_health, succession, workforce
from hrx.data.loader import load
from hrx.rag.retriever import HybridRetriever

DATA_KEYS = {
    "조직진단": "DATA§조직진단",
    "인건비시뮬레이션": "DATA§인건비시뮬레이션",
    "승계커버리지": "DATA§승계커버리지",
    "VOC": "DATA§VOC분석",
    "보상형평성": "DATA§보상형평성",
    "직급구성": "DATA§직급구성",
}


@lru_cache(maxsize=1)
def _retriever() -> HybridRetriever:
    return HybridRetriever()


def _table(df: pd.DataFrame, limit: int = 12) -> str:
    return df.head(limit).to_markdown(index=False)


# ---------------------------------------------------------------- 검색 도구
def hr_search(query: str, corpus: str | None = None, top_k: int = 6) -> str:
    """사내 규정·법령·트렌드 문서에서 근거를 검색한다.

    corpus: 'internal'(사내 규정) | 'external'(법령·트렌드·사례) | None(자동 라우팅)
    """
    kwargs = {}
    if corpus in ("internal", "external"):
        kwargs["corpora"] = (corpus,)
    results = _retriever().search(query, top_k=top_k, **kwargs)
    if not results:
        return "검색 결과 없음."
    blocks = [f"[{r.chunk.citation_key}] {r.chunk.title} / {r.chunk.section}\n{r.chunk.text}"
              for r in results]
    return "\n\n---\n\n".join(blocks)


def available_citation_keys(query: str, top_k: int = 8) -> set[str]:
    return {r.chunk.citation_key for r in _retriever().search(query, top_k=top_k)}


# ---------------------------------------------------------------- 진단 도구
def org_diagnostics() -> str:
    """전사 조직진단 지표(스팬·적체·평가편차·침투율·VOC)를 산출한다."""
    return f"{org_health.brief()}\n\n(인용 키: {DATA_KEYS['조직진단']})"


def pay_equity() -> str:
    """입사경로·직급별 임금밴드 침투율 격차를 산출한다."""
    by_unit, gap = org_health.compa_ratio()
    return (
        "### 입사경로별 침투율 격차\n"
        + _table(gap)
        + "\n\n### 본부×직급 평균 침투율 (상위 10)\n"
        + _table(by_unit.sort_values("avg_compa", ascending=False), 10)
        + f"\n\n(인용 키: {DATA_KEYS['보상형평성']})"
    )


def grade_structure() -> str:
    """직급 구성비와 적정 피라미드 대비 편차를 산출한다."""
    return (
        "### 직급 구성비\n" + _table(workforce.grade_pyramid())
        + f"\n\n(인용 키: {DATA_KEYS['직급구성']})"
    )


def voc_analysis(topic: str | None = None, n_quotes: int = 4) -> str:
    """구성원 VOC 를 주제별로 집계하고 대표 응답을 발췌한다."""
    by_topic, by_div = org_health.voc_profile()
    parts = ["### 주제별 VOC", _table(by_topic), "\n### 본부별 부정 응답률", _table(by_div)]
    quotes = org_health.voc_quotes(topic=topic, n=n_quotes)
    if quotes:
        parts.append(f"\n### 대표 응답{f' ({topic})' if topic else ''}")
        for q in quotes:
            parts.append(f"- ({q['division']}/{q['grade']}/{q['topic']}) \"{q['text']}\"")
    parts.append(f"\n(인용 키: {DATA_KEYS['VOC']})")
    return "\n".join(parts)


def succession_coverage() -> str:
    """핵심 보직 승계 커버리지와 위험 보직을 산출한다."""
    diag = succession.diagnose()
    cols = [
        "title", "criticality", "years_to_vacancy", "candidate_count",
        "ready_now_count", "effective_coverage", "status", "shortfall",
    ]
    out = [
        "### 승계 커버리지 요약", diag.headline(),
        "\n### 보직별 현황", _table(diag.coverage[cols], 15),
    ]
    if len(diag.duplicate_candidates):
        out += ["\n### 중복 지명 후보", _table(diag.duplicate_candidates, 8)]
    out.append(f"\n(인용 키: {DATA_KEYS['승계커버리지']})")
    return "\n".join(out)


def workforce_simulation(
    promotion_delta: float = 0.0,
    attrition_delta: float = 0.0,
    base_raise: float | None = None,
    years: int = 3,
) -> str:
    """인원·인건비를 시나리오별로 시뮬레이션한다. delta 는 기준 대비 절대 증감(예: 0.03)."""
    d = load()
    lines = ["### 시나리오별 3개년 인건비"]
    rows = []
    for res in workforce.run_all_scenarios(d, years=years).values():
        rows.append(res.summary())
    lines.append(_table(pd.DataFrame(rows)))

    if promotion_delta or attrition_delta or base_raise is not None:
        sc = workforce.Scenario(
            name="사용자정의",
            promotion_rate={
                g: max(0.0, v + promotion_delta) for g, v in workforce.BASE_PROMOTION.items()
            },
            attrition_rate={
                g: max(0.0, v + attrition_delta) for g, v in workforce.BASE_ATTRITION.items()
            },
            base_raise=0.033 if base_raise is None else base_raise,
        )
        res = workforce.simulate(d, sc, years=years)
        lines += [
            f"\n### 사용자 정의 시나리오 — {sc.describe()}",
            _table(res.cost),
            f"요약: {res.summary()}",
        ]

    lines += ["\n### 승진율 민감도", _table(workforce.sensitivity(d, years=years))]
    lines.append(f"\n(인용 키: {DATA_KEYS['인건비시뮬레이션']})")
    return "\n".join(lines)


TOOL_REGISTRY = {
    "hr_search": hr_search,
    "org_diagnostics": org_diagnostics,
    "pay_equity": pay_equity,
    "grade_structure": grade_structure,
    "voc_analysis": voc_analysis,
    "succession_coverage": succession_coverage,
    "workforce_simulation": workforce_simulation,
}


def crewai_tools() -> list:
    """CrewAI 가 설치된 환경에서만 @tool 래핑을 적용한다."""
    try:
        from crewai.tools import tool as crew_tool  # type: ignore
    except Exception:  # pragma: no cover - crewai 미설치 환경
        return []
    return [crew_tool(name)(fn) for name, fn in TOOL_REGISTRY.items()]
