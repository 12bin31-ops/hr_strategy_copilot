from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from hrx.agents import fallback, themes
from hrx.agents.crew import run_crew
from hrx.agents.tools import TOOL_REGISTRY, org_diagnostics, succession_coverage
from hrx.harness.budget import ContextBudget, Section, compact
from hrx.harness.guard import extract_json, run_guarded
from hrx.harness.schemas import Diagnosis, Evidence, PolicyProposal


# ------------------------------------------------------------------- 테마 분류
@pytest.mark.parametrize(
    "question,theme",
    [
        ("차장 승진 적체를 어떻게 풀까", "승진"),
        ("핵심 보직 Ready Now 후보가 없다", "승계"),
        ("평가 관대화가 심하다", "평가"),
        ("경력입사자 연봉 역전 문제", "보상"),
        ("팀 규모가 너무 큰데 개편해야 하나", "조직"),
    ],
)
def test_theme_classification(question, theme):
    assert themes.classify(question) == theme


def test_catalog_options_have_evidence():
    for theme, options in themes.CATALOG.items():
        assert len(options) >= 2, theme
        for o in options:
            assert o.evidence and all("§" in e for e in o.evidence)
            assert o.legal_risk in {"낮음", "중간", "높음"}


# ---------------------------------------------------------------------- 도구
def test_tools_emit_citation_keys():
    assert "DATA§조직진단" in org_diagnostics()
    assert "DATA§승계커버리지" in succession_coverage()


def test_tool_registry_callable():
    for name, fn in TOOL_REGISTRY.items():
        assert callable(fn), name


# ------------------------------------------------------------------- 하네스
def test_extract_json_from_fenced_block():
    raw = '설명입니다.\n```json\n{"a": 1,}\n```\n끝'
    assert extract_json(raw) == {"a": 1}


def test_schema_rejects_bad_citation():
    with pytest.raises(ValidationError):
        Evidence(citation="근거없음")


def test_proposal_requires_two_options():
    with pytest.raises(ValidationError):
        PolicyProposal(
            title="t", problem_statement="p", options=[], recommended="r",
            rationale="x", roadmap=[], evidence=[],
        )


def test_guard_repairs_after_failure():
    """1회차는 깨진 JSON, 2회차는 정상 → 가드가 수리 힌트를 되먹여 통과시켜야 한다."""
    calls: list[str | None] = []
    good = {
        "summary": "요약",
        "items": [{
            "issue": "적체", "metric": "1.7배", "severity": "위험",
            "affected_scope": "반도체/G4",
            "evidence": [{"citation": "INT-HR-005§제10조", "quote": ""}],
        }],
    }

    def generate(hint):
        calls.append(hint)
        if len(calls) == 1:
            return "이건 JSON이 아닙니다"
        return json.dumps(good, ensure_ascii=False)

    res = run_guarded(generate, Diagnosis, {"INT-HR-005§제10조"}, require_citations=False)
    assert res.ok and res.attempts == 2
    assert calls[1] and "실패" in calls[1]


def test_guard_blocks_hallucinated_citation():
    bad = {
        "summary": "요약",
        "items": [{
            "issue": "x", "metric": "y", "severity": "위험", "affected_scope": "z",
            "evidence": [{"citation": "FAKE-DOC-999§제1조", "quote": ""}],
        }],
    }
    res = run_guarded(lambda hint: json.dumps(bad), Diagnosis, {"INT-HR-005§제10조"})
    assert not res.ok
    assert any("제공되지 않은 근거" in e for e in res.errors)


def test_context_budget_truncates_low_priority():
    budget = ContextBudget(total=600)
    packed = budget.pack([
        Section("핵심", "A" * 500, priority=1, min_chars=400),
        Section("부가", "B" * 500, priority=9, min_chars=100),
    ])
    assert len(packed) <= 800
    assert "예산 초과로 이하 생략" in packed
    assert any(t.truncated for t in budget.traces)


def test_compact_drops_quotes_and_caps_lists():
    """다음 단계로 넘길 때 근거 원문(quote)은 버리고 인용 키만 남겨야 한다."""
    brief = fallback.build_brief("승진 적체")
    packed = compact(brief.diagnosis)
    full = json.dumps(brief.diagnosis.model_dump(), ensure_ascii=False)
    assert len(packed) < len(full)
    assert "quote" not in packed
    assert "INT-HR-005§제10조" in packed


# -------------------------------------------------------------- 폴백 파이프라인
def test_fallback_brief_is_complete():
    brief = fallback.build_brief("반도체 차장 승진 적체 해소 방안")
    assert len(brief.proposal.options) >= 2
    assert brief.proposal.recommended in {o.name for o in brief.proposal.options}
    assert brief.compliance.verdict in {"통과", "조건부 통과", "보류"}
    assert brief.diagnosis.items


def test_fallback_flags_unfair_change_risk():
    brief = fallback.build_brief("평가 등급 배분율을 강제 적용하려고 한다")
    risky = [i for i in brief.compliance.issues if i.risk == "높음"]
    assert risky, "불이익 변경 소지가 있는 조치를 잡아내지 못했다"
    assert any("제94조" in i.basis for i in risky)
    assert brief.compliance.verdict == "조건부 통과"


def test_brief_markdown_has_all_sections():
    md = fallback.build_brief("승계 공백을 어떻게 메울까").to_markdown()
    for heading in ["현황 진단", "구성원 VOC", "외부 검토", "대안 비교", "실행 로드맵",
                    "법적·절차적 검토", "참고 근거"]:
        assert heading in md, heading


def test_run_crew_falls_back_without_llm(monkeypatch, retriever):
    monkeypatch.setenv("HRX_LLM_PROVIDER", "none")
    import importlib

    import hrx.config as cfg
    importlib.reload(cfg)
    run = run_crew("승진 적체 해소", retriever)
    assert "fallback" in run.brief.generated_by
    assert run.logs and run.logs[0].ok
