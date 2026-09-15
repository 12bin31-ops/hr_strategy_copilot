"""평가 하네스.

세 축을 측정한다.

  검색   : Recall@k, MRR, 라우팅 정확도
  근거   : 생성된 보고서의 인용률, 미확인 인용(환각) 건수
  라우팅 : 질의 → 테마 분류 정확도

숫자가 없으면 "검색이 좋아졌다"는 말은 검증할 수 없다. 이 모듈이 그 기준선을 만든다.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from hrx.agents import fallback, themes
from hrx.config import SETTINGS
from hrx.evaluation.golden import GOLDEN, GoldenCase
from hrx.rag.citations import audit
from hrx.rag.retriever import HybridRetriever, route


@dataclass
class CaseResult:
    question: str
    hit: bool
    rank: int | None
    retrieved: list[str]
    expected: list[str]
    route_ok: bool
    theme_ok: bool
    predicted_theme: str


@dataclass
class EvalReport:
    k: int
    cases: list[CaseResult]
    citation_coverage: float
    hallucinated_citations: int
    briefs_checked: int

    @property
    def recall(self) -> float:
        return sum(c.hit for c in self.cases) / len(self.cases)

    @property
    def mrr(self) -> float:
        return sum(1 / c.rank for c in self.cases if c.rank) / len(self.cases)

    @property
    def route_accuracy(self) -> float:
        return sum(c.route_ok for c in self.cases) / len(self.cases)

    @property
    def theme_accuracy(self) -> float:
        return sum(c.theme_ok for c in self.cases) / len(self.cases)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(c) for c in self.cases])

    def summary(self) -> dict:
        return {
            "k": self.k,
            "cases": len(self.cases),
            f"recall@{self.k}": round(self.recall, 3),
            "mrr": round(self.mrr, 3),
            "route_accuracy": round(self.route_accuracy, 3),
            "theme_accuracy": round(self.theme_accuracy, 3),
            "citation_coverage": round(self.citation_coverage, 3),
            "hallucinated_citations": self.hallucinated_citations,
            "briefs_checked": self.briefs_checked,
        }

    def render(self) -> str:
        s = self.summary()
        lines = ["# 평가 결과", ""]
        lines += [f"- **{k}**: {v}" for k, v in s.items()]
        misses = [c for c in self.cases if not c.hit]
        if misses:
            lines += ["", f"## 미검색 케이스 {len(misses)}건", ""]
            for c in misses:
                lines.append(
                    f"- {c.question}\n  - 기대: {', '.join(c.expected)}\n"
                    f"  - 실제: {', '.join(c.retrieved[:4])}"
                )
        return "\n".join(lines)


def _eval_case(case: GoldenCase, retriever: HybridRetriever, k: int) -> CaseResult:
    results = retriever.search(case.question, top_k=k)
    keys = [r.chunk.citation_key for r in results]
    rank = next((i + 1 for i, key in enumerate(keys) if key in case.must_retrieve), None)
    decision = route(case.question)
    predicted = themes.classify(case.question)
    return CaseResult(
        question=case.question,
        hit=rank is not None,
        rank=rank,
        retrieved=keys,
        expected=list(case.must_retrieve),
        route_ok=all(c in decision.corpora for c in case.expected_corpora),
        theme_ok=predicted == case.expected_theme,
        predicted_theme=predicted,
    )


def evaluate(k: int = 6, brief_samples: int = 4) -> EvalReport:
    retriever = HybridRetriever()
    cases = [_eval_case(c, retriever, k) for c in GOLDEN]

    # 생성 품질: 폴백 경로로 보고서를 만들어 인용 규율을 검사한다.
    coverages, halluc = [], 0
    sample_qs = [c.question for c in GOLDEN[:brief_samples]]
    for q in sample_qs:
        brief = fallback.build_brief(q, retriever)
        # 환각 판정 기준은 "인덱스에 실제로 존재하는 근거인가"다.
        # 특정 질의의 top-k 로 한정하면, 대안 카탈로그가 인용한 정당한 조문까지 환각으로 센다.
        known = {c.citation_key for c in retriever.index.chunks}
        known |= set(__import__("hrx.agents.tools", fromlist=["DATA_KEYS"]).DATA_KEYS.values())
        # 실제 산출물(마크다운 보고서) 기준으로 인용률을 잰다.
        # JSON 덤프로 재면 키·구분자까지 문장으로 세어 지표가 왜곡된다.
        report = audit(brief.to_markdown(), known)
        coverages.append(report.coverage)
        halluc += len(report.unknown_citations)

    return EvalReport(
        k=k,
        cases=cases,
        citation_coverage=sum(coverages) / len(coverages) if coverages else 0.0,
        hallucinated_citations=halluc,
        briefs_checked=len(sample_qs),
    )


def save(report: EvalReport, out_dir: Path | None = None) -> Path:
    d = out_dir or SETTINGS.paths.reports
    d.mkdir(parents=True, exist_ok=True)
    (d / "eval.json").write_text(
        json.dumps(report.summary(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report.to_frame().to_csv(d / "eval_cases.csv", index=False, encoding="utf-8-sig")
    path = d / "eval.md"
    path.write_text(report.render(), encoding="utf-8")
    return path
