"""인용(citation) 파싱·검증.

에이전트가 생성한 문장에 `[INT-HR-002§제5조]` 형태의 인용 태그를 강제하고,
  * 존재하지 않는 근거를 지어냈는지(= hallucinated citation)
  * 사실 주장 문장 중 몇 %가 근거를 달고 있는지(= citation coverage)
를 계산한다. 이 두 지표가 harness/guard.py 의 통과 조건이 된다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 인용 표기는 세 가지를 허용한다: 본문 [DOC§조항], 표/코드 `DOC§조항`, JSON "DOC§조항".
# (guard 는 구조화 출력을 JSON 으로 직렬화해 검사하므로 따옴표 형태가 필요하다)
# DATA§ 는 정량 산출 결과(analytics)를 가리키는 내부 근거 네임스페이스.
CITATION_RE = re.compile(r"[\[`\"]([A-Z][A-Z0-9_-]{1,24})§([^\]`\"\n]+)[\]`\"]")
# 문장 분리: 마침표 뒤에 공백/끝이 올 때만 자른다("1.1", "1.08" 같은 수치를 보호).
SENTENCE_RE = re.compile(r"[^!?\n]+?(?:[.!?](?=\s|$)|$)")

# 근거가 필요 없는 문장(제목/목차/연결어)
SKIP_PREFIXES = ("##", "#", ">", "1.", "2.", "3.", "4.", "5.")
TABLE_SEPARATOR_RE = re.compile(r"^\|[\s:|-]+\|?$")
SKIP_PATTERNS = re.compile(r"^(표|그림|이상|참고|요약|목차|결론적으로)\b")


@dataclass
class CitationReport:
    total_claims: int
    cited_claims: int
    citations: list[tuple[str, str]]
    unknown_citations: list[str]
    uncited_sentences: list[str]

    @property
    def coverage(self) -> float:
        return self.cited_claims / self.total_claims if self.total_claims else 1.0

    @property
    def is_grounded(self) -> bool:
        return not self.unknown_citations

    def summary(self) -> str:
        return (
            f"근거 인용률 {self.coverage:.0%} ({self.cited_claims}/{self.total_claims} 문장), "
            f"미확인 인용 {len(self.unknown_citations)}건"
        )


def extract_citations(text: str) -> list[tuple[str, str]]:
    return [(m.group(1), m.group(2).strip()) for m in CITATION_RE.finditer(text)]


def _claim_sentences(text: str) -> list[str]:
    out: list[str] = []
    lines = text.splitlines()
    for idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        # 표 헤더(다음 줄이 구분선)는 주장 문장이 아니다.
        nxt = lines[idx + 1].strip() if idx + 1 < len(lines) else ""
        if line.startswith("|") and TABLE_SEPARATOR_RE.match(nxt):
            continue
        if not line or line.startswith(SKIP_PREFIXES) or SKIP_PATTERNS.match(line):
            continue
        # 마크다운 표의 데이터 행은 그 자체가 하나의 사실 주장이다(헤더·구분선은 제외).
        if line.startswith("|"):
            if TABLE_SEPARATOR_RE.match(line) or "---" in line:
                continue
            if len(line) >= 18:
                out.append(line)
            continue
        if line.startswith(("-", "*")):
            line = line.lstrip("-* ").strip()
        for s in SENTENCE_RE.findall(line):
            s = s.strip()
            if len(s) >= 18:
                out.append(s)
    return out


def audit(text: str, known_keys: set[str] | None = None) -> CitationReport:
    """known_keys: 검색 결과로 실제 제공된 citation_key 집합."""
    sentences = _claim_sentences(text)
    cited = [s for s in sentences if CITATION_RE.search(s)]
    citations = extract_citations(text)
    unknown: list[str] = []
    if known_keys is not None:
        for doc_id, article in citations:
            key = f"{doc_id}§{article}"
            if key not in known_keys and doc_id not in {k.split("§")[0] for k in known_keys}:
                unknown.append(key)
    return CitationReport(
        total_claims=len(sentences),
        cited_claims=len(cited),
        citations=citations,
        unknown_citations=sorted(set(unknown)),
        uncited_sentences=[s for s in sentences if not CITATION_RE.search(s)][:10],
    )


def render_source_list(known_keys: list[str], titles: dict[str, str]) -> str:
    lines = ["## 참고 근거"]
    for key in dict.fromkeys(known_keys):
        doc_id = key.split("§")[0]
        lines.append(f"- `{key}` — {titles.get(doc_id, doc_id)}")
    return "\n".join(lines)
