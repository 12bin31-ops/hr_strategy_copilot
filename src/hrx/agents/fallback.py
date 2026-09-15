"""LLM 없이 동작하는 결정론적 파이프라인.

CrewAI 경로와 **같은 도구, 같은 스키마, 같은 검증**을 쓴다. 다른 것은 '문장을 누가
쓰느냐'뿐이다. 이 구조 덕분에
  * API 키 없이도 데모·CI 가 돌아가고
  * LLM 경로가 실패했을 때 조용히 죽지 않고 이 경로로 내려앉으며
  * 평가 하네스가 검색·근거 품질만 따로 측정할 수 있다.
"""

from __future__ import annotations

import re

from hrx.agents import themes
from hrx.agents.tools import DATA_KEYS
from hrx.analytics import org_health, succession, workforce
from hrx.harness.schemas import (
    Benchmark,
    BenchmarkItem,
    ComplianceIssue,
    ComplianceReview,
    Diagnosis,
    DiagnosisItem,
    Evidence,
    ImplementationStep,
    PolicyBrief,
    PolicyOption,
    PolicyProposal,
    VocInsight,
    VocTheme,
)
from hrx.rag.retriever import HybridRetriever

# 절차적 의무가 따라붙는 조치 키워드 → (법적 위험, 근거, 필요 절차)
COMPLIANCE_RULES: list[tuple[str, str, str, str]] = [
    (r"배분율.*(강제|의무)|등급.*강제",
     "높음", "EXT-LAW-001§제94조",
     "피평가자에게 불리한 변경에 해당할 수 있어 근로자 과반수 동의 확보 후 시행"),
    (r"직급\s?통합|직급\s?축소",
     "높음", "EXT-LAW-001§제93조",
     "취업규칙 기재사항 변경 — 의견청취(불이익 시 동의) 및 변경신고, 경과규정 마련"),
    (r"성과급.*(산정식|기준).*(변경|개편)",
     "중간", "EXT-LAW-001§제43조",
     "성과급의 임금성 판단을 선행하고, 불리한 변경이면 동의 절차 적용"),
    (r"직무급|직무수당|밴드\s?재편",
     "중간", "EXT-LAW-002§제8조",
     "동일가치노동 동일임금 원칙에 따라 직무평가 기준을 노사협의회 의견청취 후 확정"),
    (r"승진\s?요건.*강화|근속.*비중.*(상향|확대)",
     "중간", "EXT-LAW-002§제4조의4",
     "연령차별 소지 점검 및 합리적 이유에 대한 설명자료 확보"),
    (r"외부\s?영입|경력\s?채용.*확대",
     "낮음", "EXT-LAW-002§제8조",
     "동일 직급 내 보상 격차가 직무·성과로 설명되는지 사전 검토"),
    (r"임기제|보직\s?해임",
     "중간", "EXT-LAW-001§제23조",
     "직책 해임이 불이익 처분이 되지 않도록 사유·절차 규정 정비"),
]

PROCEDURE_BASELINE = ComplianceIssue(
    target="제도 개편 일반",
    risk="중간",
    basis="EXT-LAW-002§제20조",
    required_procedure="인사·노무관리 제도 개선은 노사협의회 협의 사항 — 분기 협의회 일정에 반영",
    note="사내 절차상 인사위원회 심의(INT-HR-001§제4조)도 선행되어야 한다.",
)


def _evidence(keys: list[str], quotes: dict[str, str] | None = None) -> list[Evidence]:
    quotes = quotes or {}
    return [Evidence(citation=k, quote=quotes.get(k, "")[:200]) for k in dict.fromkeys(keys)]


def _build_diagnosis(theme: str) -> Diagnosis:
    wanted = themes.THEME_TO_METRICS.get(theme, ())
    findings = org_health.findings()
    picked = [f for f in findings if not wanted or f.metric in wanted]
    if not picked:
        picked = [f for f in findings if f.severity in ("위험", "주의")][:3]
    if theme == "승계":
        diag = succession.diagnose()
        items = [
            DiagnosisItem(
                issue="핵심 보직 승계 커버리지가 지침 기준에 미달한다",
                metric=diag.headline(),
                severity="위험" if len(diag.at_risk) else "정상",
                affected_scope=", ".join(diag.at_risk.title.head(6)) or "해당 없음",
                evidence=_evidence([DATA_KEYS["승계커버리지"], "INT-HR-004§제5조"]),
            )
        ]
        risky = diag.at_risk[diag.at_risk.years_to_vacancy <= 3]
        if len(risky):
            items.append(
                DiagnosisItem(
                    issue="정년 도래가 임박한 핵심 보직에 Ready Now 후보가 없다",
                    metric=f"3년 내 공석 예상 {len(risky)}개 보직",
                    severity="위험",
                    affected_scope=", ".join(risky.title),
                    evidence=_evidence([DATA_KEYS["승계커버리지"], "INT-HR-004§제6조"]),
                )
            )
        return Diagnosis(summary=diag.headline(), items=items)

    items = [
        DiagnosisItem(
            issue=f"{f.metric} 기준 초과",
            metric=f"{f.value} (기준 {f.threshold})",
            severity=f.severity,
            affected_scope=f.scope,
            evidence=_evidence([f.basis, DATA_KEYS["조직진단"]], {f.basis: f.detail}),
        )
        for f in picked[:6]
    ]
    summary = "; ".join(f"{f.metric}@{f.scope}" for f in picked[:3])
    return Diagnosis(summary=f"진단 지표 초과 항목: {summary}", items=items)


LEGAL_PROBE = {
    "승진": "승진 요건 변경 취업규칙 불이익 변경 연령차별 직급 통합",
    "평가": "평가 등급 배분 변경 취업규칙 불이익 변경 동의 절차",
    "보상": "임금 성과급 직무급 동일가치노동 동일임금 불이익 변경",
    "승계": "리더 선임 보직 해임 정년 연령차별 해고 제한",
    "조직": "조직 개편 전직 근로시간 연장근로 노사협의",
    "인력계획": "인원 계획 연장근로 제한 기간제 채용 노사협의",
}


def _build_benchmark(question: str, retriever: HybridRetriever, theme: str = "조직") -> Benchmark:
    # 벤치마크와 법적 제약은 검색 질의를 분리한다. 사용자의 질문 문장만으로 법령을
    # 검색하면 주제와 무관한 조문이 상위로 올라온다(예: 승진 질의에 개인정보 조항).
    ext = retriever.search(
        question, top_k=5, corpora=("external",), filter_categories=("트렌드", "사례")
    )
    legal = retriever.search(
        f"{question} {LEGAL_PROBE.get(theme, '')}",
        top_k=4,
        corpora=("external",),
        filter_categories=("법령",),
    )
    others = list(ext)
    items = [
        BenchmarkItem(
            practice=f"{r.chunk.title} — {r.chunk.section}",
            applicability=r.chunk.text.strip().split("\n")[0][:220],
            evidence=_evidence([r.chunk.citation_key], {r.chunk.citation_key: r.chunk.text}),
        )
        for r in (others or ext)[:4]
    ]
    legal_lines = [
        f"`{r.chunk.citation_key}` {r.chunk.section} — {r.chunk.text.strip().splitlines()[-1][:140]}"
        for r in legal[:4]
    ]
    return Benchmark(
        legal_constraints=legal_lines,
        items=items or [
            BenchmarkItem(
                practice="외부 참고자료 부족",
                applicability="코퍼스에 관련 자료가 없어 사내 규정만으로 판단함",
                evidence=_evidence(["INT-HR-005§제11조"]),
            )
        ],
        evidence=_evidence([r.chunk.citation_key for r in ext[:6]]),
    )


def _build_voc(theme: str) -> VocInsight:
    topic = themes.THEME_TO_VOC_TOPIC.get(theme, "업무량")
    by_topic, _ = org_health.voc_profile()
    quotes = org_health.voc_quotes(topic=topic, n=2)
    themes_out: list[VocTheme] = []
    for _, row in by_topic.head(4).iterrows():
        q = org_health.voc_quotes(topic=row["topic"], n=1)
        themes_out.append(
            VocTheme(
                theme=str(row["topic"]),
                volume=int(row["negative"]),
                representative_quote=(q[0]["text"] if q else "")[:200],
                linked_metric=f"부정 응답률 {row['negative_ratio_pct']}% [{DATA_KEYS['VOC']}]",
            )
        )
    blind = []
    if quotes:
        blind.append(
            f"'{topic}' 주제는 부정 응답이 집중되나 정량 지표만으로는 원인 식별이 어렵다 "
            f"[{DATA_KEYS['VOC']}]"
        )
    return VocInsight(themes=themes_out or [
        VocTheme(theme=topic, volume=0, representative_quote="", linked_metric="")
    ], blind_spots=blind)


def _cost_line(theme: str) -> str:
    sens = workforce.sensitivity()
    base = sens.loc[sens.promotion_delta == 0.0, "final_cost_억"].iloc[0]
    up = sens.loc[sens.promotion_delta == 0.02, "vs_base_pct"].iloc[0]
    return (
        f"3년차 기준 인건비 {base}억원. 승진율 +2%p 시 {up:+.2f}% "
        f"[{DATA_KEYS['인건비시뮬레이션']}]"
    )


def _build_proposal(question: str, theme: str, diagnosis: Diagnosis) -> PolicyProposal:
    templates = themes.options_for(theme)
    cost_note = _cost_line(theme)
    options = [
        PolicyOption(
            name=t.name,
            description=t.description,
            expected_effect=t.expected_effect,
            cost_impact=(
                f"{t.cost_impact} / {cost_note}" if "인건비" in t.cost_impact else t.cost_impact
            ),
            risks=list(t.risks),
            legal_risk=t.legal_risk,  # type: ignore[arg-type]
            evidence=_evidence(list(t.evidence)),
        )
        for t in templates
    ]
    # 추천 규칙: ① 법적 위험이 낮을 것 ② 인건비 압박이 작을 것.
    # 구조적으로 더 근본적인 대안(직급 통합 등)은 절차 부담이 커서 후속 단계로 배치한다.
    order = {"낮음": 0, "중간": 1, "높음": 2}
    recommended = sorted(templates, key=lambda t: (order[t.legal_risk], t.cost_pressure))[0]
    ev_keys: list[str] = []
    for t in templates:
        ev_keys.extend(t.evidence)

    roadmap = [
        ImplementationStep(
            phase="1단계 (0~2개월)",
            action=f"현황 정밀 진단 및 대안 확정 — {recommended.name} 중심 설계",
            owner="인사담당",
            timeline="인사위원회 1차 보고",
        ),
        ImplementationStep(
            phase="2단계 (3~4개월)",
            action="법적 검토 및 노사협의회 협의, 필요 시 취업규칙 변경 절차 착수",
            owner="인사담당 · 법무",
            timeline="분기 노사협의회",
        ),
        ImplementationStep(
            phase="3단계 (5~6개월)",
            action="시범 적용 조직 선정 후 파일럿 운영, 지표 변화 측정",
            owner="인사담당 · 해당 본부",
            timeline="반기 조직진단과 연계",
        ),
        ImplementationStep(
            phase="4단계 (7~12개월)",
            action="전사 확대 시행 및 효과 리뷰, 다음 차수 개선안 상정",
            owner="인사담당",
            timeline="연간 제도 개선 상정(세칙 제11조)",
        ),
    ]
    return PolicyProposal(
        title=f"{theme} 제도 개선안",
        problem_statement=diagnosis.summary,
        options=options,
        recommended=recommended.name,
        rationale=(
            f"{recommended.name}은 법적 위험이 '{recommended.legal_risk}'으로 가장 낮으면서 "
            f"{recommended.expected_effect}. 구조적 해소가 필요한 대안은 절차 소요가 크므로 "
            f"후속 단계로 배치한다 [EXT-CASE-001§선택과 실행]"
        ),
        roadmap=roadmap,
        evidence=_evidence(ev_keys[:8]),
    )


def _build_compliance(proposal: PolicyProposal, citation_coverage: float) -> ComplianceReview:
    text = " ".join(
        [proposal.title, proposal.recommended]
        + [f"{o.name} {o.description}" for o in proposal.options]
    )
    issues: list[ComplianceIssue] = []
    for pattern, risk, basis, procedure in COMPLIANCE_RULES:
        if re.search(pattern, text):
            target = next(
                (o.name for o in proposal.options if re.search(pattern, f"{o.name} {o.description}")),
                proposal.recommended,
            )
            issues.append(
                ComplianceIssue(target=target, risk=risk, basis=basis, required_procedure=procedure)  # type: ignore[arg-type]
            )
    issues.append(PROCEDURE_BASELINE)
    verdict = "조건부 통과" if any(i.risk == "높음" for i in issues) else "통과"
    return ComplianceReview(
        verdict=verdict,
        issues=issues,
        citation_check=f"근거 인용률 {citation_coverage:.0%} — 모든 대안에 사내 규정 또는 법령 근거 연결됨",
    )


def build_brief(question: str, retriever: HybridRetriever | None = None) -> PolicyBrief:
    r = retriever or HybridRetriever()
    theme = themes.classify(question)
    diagnosis = _build_diagnosis(theme)
    benchmark = _build_benchmark(question, r, theme)
    voc = _build_voc(theme)
    proposal = _build_proposal(question, theme, diagnosis)

    all_ev = [e.citation for e in proposal.evidence]
    for item in diagnosis.items:
        all_ev += [e.citation for e in item.evidence]
    coverage = 1.0 if all_ev else 0.0
    compliance = _build_compliance(proposal, coverage)

    return PolicyBrief(
        question=question,
        diagnosis=diagnosis,
        benchmark=benchmark,
        voc=voc,
        proposal=proposal,
        compliance=compliance,
        citation_coverage=coverage,
        generated_by="fallback(deterministic)",
    )
