"""조항 인식 청커.

일반적인 고정길이 슬라이딩 윈도우 대신 **조항/절 경계를 우선 존중**한다.
인사규정·법령은 "제N조" 단위가 인용의 최소 단위이므로, 청크가 조 경계를 넘으면
근거 인용이 무의미해진다.

전략:
  1) 마크다운 헤딩(`##`, `###`)으로 1차 분할
  2) 헤딩 제목에서 `제N조 (제목)` 패턴을 추출해 section_label 로 삼음
  3) 조각이 max_chunk_chars 를 넘으면 항(①②③) 또는 문단 경계에서 2차 분할
  4) min_chunk_chars 미만인 조각은 다음 조각과 병합(단, 조 경계는 넘지 않음)
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from hrx.config import SETTINGS

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
ARTICLE_RE = re.compile(r"(제\s?\d+조(?:의\d+)?)")
CLAUSE_SPLIT_RE = re.compile(r"(?=^\s*[①-⑳])", re.MULTILINE)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    corpus: str
    category: str
    section: str
    article: str
    text: str
    source_path: str
    order: int
    meta: dict = field(default_factory=dict)

    @property
    def citation_key(self) -> str:
        """본문에서 쓰는 인용 키. 예) INT-HR-002§제5조"""
        return f"{self.doc_id}§{self.article or self.section or '본문'}"

    def as_context(self, max_chars: int | None = None) -> str:
        body = self.text if max_chars is None else self.text[:max_chars]
        return f"[{self.citation_key}] {self.title} / {self.section}\n{body}"

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_frontmatter(raw: str) -> tuple[dict, str]:
    m = FRONTMATTER_RE.match(raw)
    if not m:
        return {}, raw
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        meta[k.strip()] = v.strip().strip("'\"")
    return meta, raw[m.end():]


def _split_by_heading(body: str) -> list[tuple[str, str]]:
    """(heading, section_body) 리스트. 첫 헤딩 이전 텍스트는 '서문'으로."""
    matches = list(HEADING_RE.finditer(body))
    if not matches:
        return [("본문", body.strip())]
    sections: list[tuple[str, str]] = []
    preamble = body[: matches[0].start()].strip()
    if preamble:
        sections.append(("서문", preamble))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        heading = m.group(2).strip()
        text = body[start:end].strip()
        if text:
            sections.append((heading, text))
        elif heading:
            # 내용 없는 상위 헤딩(장 제목 등)은 버리지 않고 다음 섹션 접두사로 쓰지 않는다.
            continue
    return sections


def _soft_split(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts = [p.strip() for p in CLAUSE_SPLIT_RE.split(text) if p.strip()]
    if len(parts) == 1:
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    out: list[str] = []
    buf = ""
    for p in parts:
        if len(buf) + len(p) + 1 <= max_chars or not buf:
            buf = f"{buf}\n{p}".strip()
        else:
            out.append(buf)
            buf = p
    if buf:
        out.append(buf)
    # 여전히 너무 길면 하드 컷
    final: list[str] = []
    for p in out:
        while len(p) > max_chars * 1.6:
            final.append(p[:max_chars])
            p = p[max_chars:]
        final.append(p)
    return final


def chunk_document(path: Path) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    meta, body = _parse_frontmatter(raw)
    doc_id = meta.get("doc_id") or path.stem
    title = meta.get("title") or path.stem
    corpus = meta.get("corpus") or path.parent.name
    category = meta.get("category", "")
    cfg = SETTINGS.retrieval

    chunks: list[Chunk] = []
    order = 0
    pending: str | None = None
    pending_heading: str | None = None

    def flush(heading: str, text: str) -> None:
        nonlocal order
        art_m = ARTICLE_RE.search(heading)
        article = art_m.group(1).replace(" ", "") if art_m else ""
        for piece in _soft_split(text, cfg.max_chunk_chars):
            order += 1
            digest = hashlib.sha1(f"{doc_id}|{heading}|{piece[:80]}|{order}".encode()).hexdigest()[:10]
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}-{order:03d}-{digest}",
                    doc_id=doc_id,
                    title=title,
                    corpus=corpus,
                    category=category,
                    section=heading,
                    article=article,
                    text=piece.strip(),
                    source_path=str(path),
                    order=order,
                    meta={k: v for k, v in meta.items() if k not in ("doc_id", "title", "corpus")},
                )
            )

    for heading, text in _split_by_heading(body):
        text = text.strip()
        if not text:
            continue
        if pending is not None:
            # 직전 조각이 너무 짧았다 → 같은 조 안에서만 병합
            same_article = ARTICLE_RE.search(pending_heading or "") is None
            if same_article and len(pending) + len(text) <= cfg.max_chunk_chars:
                heading = f"{pending_heading} / {heading}" if pending_heading else heading
                text = f"{pending}\n\n{text}"
            else:
                flush(pending_heading or "본문", pending)
            pending, pending_heading = None, None
        if len(text) < cfg.min_chunk_chars:
            pending, pending_heading = text, heading
            continue
        flush(heading, text)

    if pending:
        flush(pending_heading or "본문", pending)
    return chunks


def chunk_corpus(corpus_dir: Path | None = None) -> list[Chunk]:
    root = corpus_dir or SETTINGS.paths.corpus
    out: list[Chunk] = []
    for path in sorted(root.rglob("*.md")):
        out.extend(chunk_document(path))
    return out
