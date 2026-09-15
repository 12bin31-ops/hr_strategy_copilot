"""에이전트 산출물의 구조화 스키마.

LLM 을 자유 서술로 두지 않고 Pydantic 스키마로 묶는 이유는 세 가지다.
  1) 검증 가능성 — 필수 필드 누락/형식 오류를 파싱 단계에서 잡는다
  2) 수리 가능성 — 실패 사유를 그대로 재시도 프롬프트에 넣을 수 있다(guard.py)
  3) 합성 가능성 — 앞 에이전트의 출력이 뒤 에이전트의 입력 타입이 된다
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Severity = Literal["위험", "주의", "정상"]
LegalRisk = Literal["높음", "중간", "낮음"]


class Evidence(BaseModel):
    """모든 주장에 붙어야 하는 근거."""

    citation: str = Field(description="예: INT-HR-002§제6조 또는 DATA§평가등급분포")
    quote: str = Field(default="", description="근거 원문 또는 산출 수치")

    @field_validator("citation")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or "§" not in v:
            raise ValueError("citation 은 'DOCID§조항' 형식이어야 합니다")
        return v


class DiagnosisItem(BaseModel):
    issue: str = Field(description="발견한 문제 한 문장")
    metric: str = Field(description="근거가 되는 수치")
    severity: Severity
    affected_scope: str
    evidence: list[Evidence] = Field(min_length=1)


class Diagnosis(BaseModel):
    """1번 에이전트: 사내 현황 진단."""

    summary: str
    items: list[DiagnosisItem] = Field(min_length=1)


class BenchmarkItem(BaseModel):
    practice: str
    applicability: str = Field(description="우리 상황에 적용 가능한지와 그 이유")
    evidence: list[Evidence] = Field(min_length=1)


class Benchmark(BaseModel):
    """2번 에이전트: 법령·트렌드·사례 조사."""

    legal_constraints: list[str] = Field(default_factory=list)
    items: list[BenchmarkItem] = Field(min_length=1)
    evidence: list[Evidence] = Field(default_factory=list)


class VocTheme(BaseModel):
    theme: str
    volume: int
    representative_quote: str
    linked_metric: str = Field(default="", description="정량 지표와의 연결")


class VocInsight(BaseModel):
    """3번 에이전트: VOC 분석."""

    themes: list[VocTheme] = Field(min_length=1)
    blind_spots: list[str] = Field(default_factory=list)


class PolicyOption(BaseModel):
    name: str
    description: str
    expected_effect: str
    cost_impact: str = Field(description="인건비 영향 — 정량 추정 포함")
    risks: list[str] = Field(default_factory=list)
    legal_risk: LegalRisk = "중간"
    evidence: list[Evidence] = Field(default_factory=list)


class ImplementationStep(BaseModel):
    phase: str
    action: str
    owner: str
    timeline: str


class PolicyProposal(BaseModel):
    """4번 에이전트: 제도 개편안."""

    title: str
    problem_statement: str
    options: list[PolicyOption] = Field(min_length=2, description="최소 2개 대안 비교")
    recommended: str
    rationale: str
    roadmap: list[ImplementationStep] = Field(min_length=1)
    evidence: list[Evidence] = Field(min_length=1)


class ComplianceIssue(BaseModel):
    target: str = Field(description="검토 대상 조치")
    risk: LegalRisk
    basis: str = Field(description="관련 법령 근거")
    required_procedure: str
    note: str = ""


class ComplianceReview(BaseModel):
    """5번 에이전트: 법적·절차적 검증."""

    verdict: Literal["통과", "조건부 통과", "보류"]
    issues: list[ComplianceIssue] = Field(default_factory=list)
    citation_check: str = Field(default="", description="근거 인용 검증 결과")


class PolicyBrief(BaseModel):
    """최종 산출물 — 인사위원회 상정용 1장 요약."""

    question: str
    diagnosis: Diagnosis
    benchmark: Benchmark
    voc: VocInsight
    proposal: PolicyProposal
    compliance: ComplianceReview
    citation_coverage: float = 0.0
    generated_by: str = "fallback"

    def to_markdown(self) -> str:
        from hrx.agents.render import render_brief

        return render_brief(self)
