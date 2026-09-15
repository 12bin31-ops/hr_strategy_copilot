"""검증·수리 루프 (grounding gate).

LLM 호출을 그대로 신뢰하지 않는다. 모든 단계는

    생성 → 파싱(Pydantic) → 근거 검증(citations) → 실패 시 '수리 프롬프트'로 재시도

를 거치고, 재시도 한도를 넘으면 예외 대신 **명시적 실패 객체**를 반환해 상위에서
폴백으로 전환할 수 있게 한다. 파이프라인이 조용히 잘못된 결과를 내보내는 것보다
시끄럽게 폴백하는 편이 낫다.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from hrx.config import SETTINGS
from hrx.rag.citations import CitationReport, audit

T = TypeVar("T", bound=BaseModel)

JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
BRACE_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class GuardResult(Generic[T]):
    ok: bool
    value: T | None
    attempts: int
    errors: list[str] = field(default_factory=list)
    citation_report: CitationReport | None = None

    def unwrap_or(self, default: T) -> T:
        return self.value if (self.ok and self.value is not None) else default


def extract_json(text: str) -> dict:
    """LLM 출력에서 JSON 을 뽑아낸다. 코드펜스/설명문이 섞여도 동작한다."""
    if not text:
        raise ValueError("빈 응답")
    m = JSON_BLOCK_RE.search(text)
    candidate = m.group(1) if m else None
    if candidate is None:
        m = BRACE_RE.search(text)
        candidate = m.group(0) if m else text
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # 흔한 오류 자동 수리: 트레일링 콤마, 스마트 따옴표
        fixed = re.sub(r",(\s*[}\]])", r"\1", candidate)
        fixed = fixed.replace("“", '"').replace("”", '"').replace("’", "'")
        return json.loads(fixed)


def validate_grounding(
    payload: dict | BaseModel, known_keys: set[str]
) -> CitationReport:
    text = json.dumps(
        payload.model_dump() if isinstance(payload, BaseModel) else payload, ensure_ascii=False
    )
    return audit(text, known_keys)


def run_guarded(
    generate: Callable[[str | None], str],
    schema: type[T],
    known_keys: set[str],
    require_citations: bool = True,
    max_attempts: int | None = None,
) -> GuardResult[T]:
    """generate(repair_hint) → 문자열 응답. 실패 사유를 다음 호출에 되먹인다."""
    limit = max_attempts or SETTINGS.harness.max_repair_attempts + 1
    errors: list[str] = []
    hint: str | None = None
    report: CitationReport | None = None

    for attempt in range(1, limit + 1):
        try:
            raw = generate(hint)
            data = extract_json(raw)
            obj = schema.model_validate(data)
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            msg = str(exc)[:600]
            errors.append(f"[{attempt}] 파싱/검증 실패: {msg}")
            hint = (
                "이전 응답이 스키마 검증에 실패했다. 아래 오류를 고쳐 **JSON 만** 다시 출력하라.\n"
                f"{msg}"
            )
            continue

        report = validate_grounding(obj, known_keys)
        if require_citations:
            problems = []
            if report.unknown_citations:
                problems.append(
                    "제공되지 않은 근거를 인용했다: " + ", ".join(report.unknown_citations)
                )
            if report.coverage < SETTINGS.harness.min_citation_coverage:
                problems.append(
                    f"근거 인용률 {report.coverage:.0%} < 기준 "
                    f"{SETTINGS.harness.min_citation_coverage:.0%}"
                )
            if problems:
                errors.append(f"[{attempt}] 근거 검증 실패: " + " / ".join(problems))
                hint = (
                    "근거 검증에 실패했다. 아래 문제를 고쳐 JSON 만 다시 출력하라. "
                    "제공된 컨텍스트에 실제로 등장한 인용 키만 사용하고, 모든 주장 문장에 "
                    "[문서ID§조항] 형식의 근거를 붙여라.\n" + " / ".join(problems)
                )
                continue
        return GuardResult(True, obj, attempt, errors, report)

    return GuardResult(False, None, limit, errors, report)
