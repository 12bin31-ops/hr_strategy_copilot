"""CrewAI 멀티에이전트 파이프라인.

5개 에이전트가 순차(sequential)로 붙는다.

    1. 현황분석가   — 사내 지표·규정으로 문제를 정의
    2. 법령벤치마커 — 법령 제약과 외부 사례를 조사
    3. VOC분석가    — 정성 신호를 정량 지표에 연결
    4. 제도설계자   — 대안 2개 이상을 비교하고 추천안을 낸다
    5. 검증관       — 절차 요건과 근거 인용을 검증하고 통과/보류를 판정

각 단계 출력은 harness/schemas.py 의 Pydantic 모델로 받고, harness/guard.py 의
검증·수리 루프를 통과해야 다음 단계로 넘어간다. 한 단계라도 한도 내에 통과하지
못하면 전체를 fallback 경로로 내려앉힌다(조용한 실패 금지).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from hrx.agents import fallback, themes
from hrx.agents.tools import (
    hr_search,
    org_diagnostics,
    pay_equity,
    succession_coverage,
    voc_analysis,
    workforce_simulation,
)
from hrx.config import SETTINGS
from hrx.harness.budget import ContextBudget, Section, compact
from hrx.harness.guard import run_guarded
from hrx.harness.schemas import (
    Benchmark,
    ComplianceReview,
    Diagnosis,
    PolicyBrief,
    PolicyProposal,
    VocInsight,
)
from hrx.rag.retriever import HybridRetriever

SYSTEM_RULES = """너는 국내 대기업 인사기획 담당자다. 다음 규칙을 반드시 지켜라.

1. 출력은 지정된 JSON 스키마만. 설명 문장, 코드펜스 밖 텍스트 금지.
2. 모든 사실 주장에는 근거 인용을 붙인다. 형식은 [문서ID§조항] 또는 [DATA§지표명].
3. 제공된 컨텍스트에 실제로 등장한 인용 키만 쓴다. 없는 근거를 만들지 않는다.
4. 수치는 제공된 산출 결과를 그대로 쓴다. 임의로 추정하거나 반올림하지 않는다.
5. 법적 판단은 단정하지 않는다. "~에 해당할 수 있다 / 검토가 필요하다"로 쓴다.
6. 한국어로 쓴다."""


@dataclass
class StageLog:
    name: str
    ok: bool
    attempts: int
    errors: list[str] = field(default_factory=list)
    coverage: float | None = None

    def line(self) -> str:
        mark = "OK " if self.ok else "FAIL"
        cov = f" cov={self.coverage:.0%}" if self.coverage is not None else ""
        return f"[{mark}] {self.name} (시도 {self.attempts}회{cov})" + (
            f" — {self.errors[-1][:120]}" if self.errors else ""
        )


@dataclass
class CrewRun:
    brief: PolicyBrief
    logs: list[StageLog]
    context_report: str = ""

    def trace(self) -> str:
        return "\n".join(log.line() for log in self.logs)


def _llm_callable():
    """LiteLLM 경유 단일 호출 함수. crewai 미설치/키 없음이면 None."""
    resolved = SETTINGS.resolve_llm()
    if resolved is None:
        return None
    provider, model = resolved
    try:
        import litellm  # type: ignore
    except Exception:
        return None

    def call(prompt: str, system: str = SYSTEM_RULES) -> str:
        resp = litellm.completion(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=4000,
        )
        return resp["choices"][0]["message"]["content"]

    call.provider = provider  # type: ignore[attr-defined]
    call.model = model  # type: ignore[attr-defined]
    return call


def _prompt(role: str, task: str, context: str, schema: type) -> str:
    return (
        f"### 역할\n{role}\n\n### 과업\n{task}\n\n"
        f"### 출력 스키마 (JSON)\n```json\n"
        f"{json.dumps(schema.model_json_schema(), ensure_ascii=False)[:2500]}\n```\n\n"
        f"### 컨텍스트\n{context}\n"
    )


def _gather_context(question: str, theme: str, retriever: HybridRetriever) -> tuple[str, ContextBudget, set[str]]:
    """도구를 호출해 컨텍스트를 모으고 예산 안에서 패킹한다."""
    doc_ctx, results = retriever.build_context(question, top_k=8)
    known = {r.chunk.citation_key for r in results}
    known |= set(__import__("hrx.agents.tools", fromlist=["DATA_KEYS"]).DATA_KEYS.values())

    sections = [
        Section("질의", question, priority=1, min_chars=len(question)),
        Section("조직진단", org_diagnostics(), priority=2, min_chars=800),
        Section("관련 규정·법령·사례", doc_ctx, priority=2, min_chars=2500),
        Section("VOC", voc_analysis(themes.THEME_TO_VOC_TOPIC.get(theme)), priority=3, min_chars=700),
    ]
    if theme in ("보상", "승진", "인력계획"):
        sections.append(Section("보상 형평성", pay_equity(), priority=3, min_chars=500))
        sections.append(
            Section("인건비 시뮬레이션", workforce_simulation(), priority=3, min_chars=600)
        )
    if theme == "승계":
        sections.append(Section("승계 커버리지", succession_coverage(), priority=2, min_chars=900))

    budget = ContextBudget()
    return budget.pack(sections), budget, known


def run_crew(question: str, retriever: HybridRetriever | None = None) -> CrewRun:
    r = retriever or HybridRetriever()
    theme = themes.classify(question)
    call = _llm_callable()
    if call is None:
        brief = fallback.build_brief(question, r)
        return CrewRun(
            brief,
            [StageLog("LLM 미설정 — 결정론적 폴백 경로 실행", True, 0)],
            "",
        )

    context, budget, known = _gather_context(question, theme, r)
    logs: list[StageLog] = []
    fb = fallback.build_brief(question, r)  # 단계 실패 시 부분 대체용

    stages = [
        ("현황분석가", Diagnosis,
         "사내 지표와 규정을 근거로 이 질의와 관련된 문제를 정의하라. 수치 없는 주장 금지."),
        ("법령벤치마커", Benchmark,
         "관련 법령 제약과 외부 사례를 정리하고, 각각이 우리 상황에 적용 가능한지 판단하라."),
        ("VOC분석가", VocInsight,
         "구성원 VOC 를 주제별로 정리하고 각 주제를 정량 지표와 연결하라."),
    ]
    outputs: dict[str, object] = {}
    for role, schema, task in stages:
        res = run_guarded(
            lambda hint, role=role, schema=schema, task=task: call(
                _prompt(role, task, context, schema) + (f"\n\n### 수정 요청\n{hint}" if hint else "")
            ),
            schema,
            known,
        )
        logs.append(
            StageLog(role, res.ok, res.attempts, res.errors,
                     res.citation_report.coverage if res.citation_report else None)
        )
        outputs[role] = res.value

    diagnosis: Diagnosis = outputs["현황분석가"] or fb.diagnosis  # type: ignore[assignment]
    benchmark: Benchmark = outputs["법령벤치마커"] or fb.benchmark  # type: ignore[assignment]
    voc: VocInsight = outputs["VOC분석가"] or fb.voc  # type: ignore[assignment]

    design_ctx = (
        f"{context}\n\n## 앞 단계 산출물\n"
        f"### 진단\n{compact(diagnosis)}\n\n### 벤치마크\n{compact(benchmark)}\n\n"
        f"### VOC\n{compact(voc)}\n\n"
        f"### 참고 대안 카탈로그\n"
        + "\n".join(f"- {t.name}: {t.description}" for t in themes.options_for(theme))
    )
    res = run_guarded(
        lambda hint: call(
            _prompt(
                "제도설계자",
                "대안을 2개 이상 비교하고 추천안과 근거, 4단계 로드맵을 제시하라. "
                "각 대안의 인건비 영향은 제공된 시뮬레이션 수치를 인용하라.",
                design_ctx,
                PolicyProposal,
            )
            + (f"\n\n### 수정 요청\n{hint}" if hint else "")
        ),
        PolicyProposal,
        known,
    )
    logs.append(StageLog("제도설계자", res.ok, res.attempts, res.errors,
                         res.citation_report.coverage if res.citation_report else None))
    proposal: PolicyProposal = res.value or fb.proposal

    review_ctx = (
        f"{context}\n\n## 검증 대상 개편안\n{compact(proposal)}\n\n"
        "## 사내 절차 규정\n" + hr_search("제도 개정 절차 인사위원회 노사협의회", "internal", 4)
    )
    res = run_guarded(
        lambda hint: call(
            _prompt(
                "검증관",
                "개편안의 각 조치에 대해 법적 위험과 필요 절차를 판정하라. "
                "취업규칙 불이익 변경 여부를 반드시 검토하고, 근거 없는 주장을 지적하라.",
                review_ctx,
                ComplianceReview,
            )
            + (f"\n\n### 수정 요청\n{hint}" if hint else "")
        ),
        ComplianceReview,
        known,
    )
    logs.append(StageLog("검증관", res.ok, res.attempts, res.errors,
                         res.citation_report.coverage if res.citation_report else None))
    compliance: ComplianceReview = res.value or fb.compliance

    covs = [log.coverage for log in logs if log.coverage is not None]
    brief = PolicyBrief(
        question=question,
        diagnosis=diagnosis,
        benchmark=benchmark,
        voc=voc,
        proposal=proposal,
        compliance=compliance,
        citation_coverage=sum(covs) / len(covs) if covs else 0.0,
        generated_by=f"crewai/{getattr(call, 'provider', '?')}:{getattr(call, 'model', '?')}",
    )
    return CrewRun(brief, logs, budget.report())
