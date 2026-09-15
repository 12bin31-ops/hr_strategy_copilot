"""컨텍스트 예산 관리(= harness engineering 의 실무 핵심).

에이전트 체인이 길어지면 앞 단계의 출력이 그대로 뒤로 흘러 컨텍스트가 폭증한다.
여기서는 세 가지 장치를 둔다.

  * Budget      : 섹션별 문자 예산을 선언하고 초과분을 잘라낸다
  * compact()   : 구조화 객체를 '다음 단계가 실제로 쓰는 필드만' 남겨 축약한다
  * trace       : 어떤 섹션이 얼마나 잘렸는지 기록 — 디버깅과 평가에 쓴다
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel

from hrx.config import SETTINGS


@dataclass
class Section:
    name: str
    text: str
    priority: int = 5  # 낮을수록 먼저 보존
    min_chars: int = 200


@dataclass
class BudgetTrace:
    section: str
    original: int
    kept: int

    @property
    def truncated(self) -> bool:
        return self.kept < self.original


@dataclass
class ContextBudget:
    total: int = field(default_factory=lambda: SETTINGS.harness.context_char_budget)
    traces: list[BudgetTrace] = field(default_factory=list)

    def pack(self, sections: list[Section]) -> str:
        """우선순위가 높은 섹션부터 예산을 배정하고, 남은 예산 안에서 뒤 섹션을 자른다."""
        self.traces = []
        ordered = sorted(sections, key=lambda s: (s.priority, -len(s.text)))
        remaining = self.total
        allocated: dict[str, str] = {}

        # 1차: 최소 보장량 배정
        for s in ordered:
            keep = min(len(s.text), s.min_chars, max(0, remaining))
            allocated[s.name] = s.text[:keep]
            remaining -= keep

        # 2차: 남은 예산을 우선순위 순으로 추가 배분
        for s in ordered:
            if remaining <= 0:
                break
            already = len(allocated[s.name])
            extra = min(len(s.text) - already, remaining)
            if extra > 0:
                allocated[s.name] = s.text[: already + extra]
                remaining -= extra

        blocks = []
        for s in sections:  # 출력 순서는 입력 순서를 유지
            kept = allocated.get(s.name, "")
            self.traces.append(BudgetTrace(s.name, len(s.text), len(kept)))
            if not kept:
                continue
            suffix = "\n…(예산 초과로 이하 생략)" if len(kept) < len(s.text) else ""
            blocks.append(f"## {s.name}\n{kept}{suffix}")
        return "\n\n".join(blocks)

    def report(self) -> str:
        rows = [f"- {t.section}: {t.kept}/{t.original}자" + (" (truncated)" if t.truncated else "")
                for t in self.traces]
        used = sum(t.kept for t in self.traces)
        return f"컨텍스트 {used}/{self.total}자 사용\n" + "\n".join(rows)


def compact(model: BaseModel, keep: dict[str, int] | None = None) -> str:
    """구조화 결과를 다음 단계용으로 축약한다.

    전체 JSON 을 그대로 넘기면 근거 원문(quote)까지 따라가 컨텍스트가 2~3배로 불어난다.
    다음 단계가 실제로 참조하는 필드만 남기고, 인용 키는 유지한다(근거 추적성 보존).
    """
    keep = keep or {}
    value_limit = keep.get("value", 140)
    quote_limit = keep.get("quote", 0)  # 근거 원문은 기본적으로 버리고 인용 키만 남긴다
    list_limit = keep.get("list", 5)
    lines: list[str] = []

    def walk(obj, path: str) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "quote":
                    if quote_limit and v:
                        lines.append(f"{path}.quote: {str(v)[:quote_limit]}")
                    continue
                walk(v, f"{path}.{k}" if path else k)
        elif isinstance(obj, list):
            for i, v in enumerate(obj[:list_limit]):
                walk(v, f"{path}[{i}]")
            if len(obj) > list_limit:
                lines.append(f"{path}: …외 {len(obj) - list_limit}건")
        elif obj not in (None, "", []):
            lines.append(f"{path}: {str(obj)[:value_limit]}")

    walk(model.model_dump(), "")
    return "\n".join(f"- {line}" for line in lines)
