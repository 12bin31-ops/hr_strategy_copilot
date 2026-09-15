"""PolicyBrief → 인사위원회 상정용 마크다운."""

from __future__ import annotations

from hrx.harness.schemas import PolicyBrief

RISK_MARK = {"높음": "●●●", "중간": "●●○", "낮음": "●○○"}


def render_brief(brief: PolicyBrief) -> str:
    p = brief.proposal
    lines: list[str] = [
        f"# {p.title}",
        "",
        f"> 검토 요청: {brief.question}",
        f"> 생성 경로: `{brief.generated_by}` · 근거 인용률 {brief.citation_coverage:.0%} "
        f"· 법적 검토 결과 **{brief.compliance.verdict}**",
        "",
        "## 1. 현황 진단",
        "",
        brief.diagnosis.summary,
        "",
        "| 문제 | 수치 | 판정 | 대상 | 근거 |",
        "|---|---|---|---|---|",
    ]
    for it in brief.diagnosis.items:
        ev = ", ".join(f"`{e.citation}`" for e in it.evidence)
        lines.append(
            f"| {it.issue} | {it.metric} | {it.severity} | {it.affected_scope} | {ev} |"
        )

    lines += ["", "## 2. 구성원 VOC", "", "| 주제 | 부정 건수 | 대표 응답 | 연계 지표 |", "|---|---|---|---|"]
    for t in brief.voc.themes:
        quote = t.representative_quote.replace("|", "/")[:90]
        lines.append(f"| {t.theme} | {t.volume} | {quote} | {t.linked_metric} |")
    if brief.voc.blind_spots:
        lines += ["", "**추가 확인 필요**"] + [f"- {b}" for b in brief.voc.blind_spots]

    lines += ["", "## 3. 외부 검토 (법령·트렌드·사례)", ""]
    if brief.benchmark.legal_constraints:
        lines.append("**법적 제약**")
        lines += [f"- {c}" for c in brief.benchmark.legal_constraints]
        lines.append("")
    for b in brief.benchmark.items:
        ev = ", ".join(f"`{e.citation}`" for e in b.evidence)
        lines.append(f"- **{b.practice}** — {b.applicability} {ev}")

    lines += [
        "",
        "## 4. 대안 비교",
        "",
        "| 대안 | 기대효과 | 인건비 영향 | 주요 리스크 | 법적 위험 | 근거 |",
        "|---|---|---|---|---|---|",
    ]
    for o in p.options:
        mark = RISK_MARK.get(o.legal_risk, "")
        risks = "; ".join(o.risks)[:80] or "-"
        star = " ⬅ **추천**" if o.name == p.recommended else ""
        ev = ", ".join(f"`{e.citation}`" for e in o.evidence) or "-"
        lines.append(
            f"| {o.name}{star} | {o.expected_effect} | {o.cost_impact} | {risks} | "
            f"{mark} {o.legal_risk} | {ev} |"
        )

    lines += ["", f"**추천안: {p.recommended}**", "", p.rationale, "", "## 5. 실행 로드맵", "",
              "| 단계 | 조치 | 주관 | 시점 |", "|---|---|---|---|"]
    for s in p.roadmap:
        lines.append(f"| {s.phase} | {s.action} | {s.owner} | {s.timeline} |")

    lines += ["", "## 6. 법적·절차적 검토", "",
              f"**판정: {brief.compliance.verdict}**", "",
              "| 대상 | 위험 | 근거 | 필요 절차 |", "|---|---|---|---|"]
    for i in brief.compliance.issues:
        lines.append(
            f"| {i.target} | {RISK_MARK.get(i.risk, '')} {i.risk} | `{i.basis}` | "
            f"{i.required_procedure} |"
        )
    if brief.compliance.citation_check:
        lines += ["", f"_{brief.compliance.citation_check}_"]

    keys = sorted({e.citation for e in p.evidence}
                  | {e.citation for o in p.options for e in o.evidence}
                  | {e.citation for it in brief.diagnosis.items for e in it.evidence}
                  | {e.citation for b in brief.benchmark.items for e in b.evidence})
    lines += ["", "## 참고 근거", ""] + [f"- `{k}`" for k in keys]
    lines += [
        "",
        "---",
        "_본 문서는 합성 데이터와 가상 사내 규정을 바탕으로 자동 생성된 데모 산출물이다. "
        "법령 부분은 요지 정리본이며 실제 적용 시 원문 확인과 노무 전문가 검토가 필요하다._",
    ]
    return "\n".join(lines)
