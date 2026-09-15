"""전역 설정. 경로/시드/LLM 프로바이더/하네스 예산을 한 곳에서 관리한다."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv 는 선택 의존성
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Paths:
    root: Path = PROJECT_ROOT
    data: Path = PROJECT_ROOT / "data"
    corpus: Path = PROJECT_ROOT / "data" / "corpus"
    synthetic: Path = PROJECT_ROOT / "data" / "synthetic"
    index: Path = PROJECT_ROOT / "data" / "index"
    reports: Path = PROJECT_ROOT / "reports"

    def ensure(self) -> "Paths":
        for p in (self.data, self.synthetic, self.index, self.reports):
            p.mkdir(parents=True, exist_ok=True)
        return self


@dataclass(frozen=True)
class RetrievalConfig:
    """하이브리드 검색 파라미터."""

    top_k: int = _env_int("HRX_TOP_K", 6)
    candidate_k: int = _env_int("HRX_CANDIDATE_K", 24)
    rrf_k: int = _env_int("HRX_RRF_K", 60)
    bm25_k1: float = 1.5
    # b=0.75(기본값)는 짧은 조문을 과도하게 우대한다. 규정·법령 코퍼스는 조문 길이 편차가
    # 커서(짧은 정의 조항 ↔ 긴 해설 조항) 길이 정규화를 완화해야 핵심 조문이 살아난다.
    bm25_b: float = 0.4
    min_chunk_chars: int = 120
    max_chunk_chars: int = 900
    char_ngram: tuple[int, int] = (2, 4)


@dataclass(frozen=True)
class HarnessConfig:
    """에이전트 하네스(컨텍스트 예산 / 근거 게이트 / 재시도) 설정."""

    context_char_budget: int = _env_int("HRX_CONTEXT_BUDGET", 12000)
    per_source_char_budget: int = _env_int("HRX_SOURCE_BUDGET", 3500)
    min_citation_coverage: float = _env_float("HRX_MIN_CITATION_COVERAGE", 0.6)
    max_repair_attempts: int = _env_int("HRX_MAX_REPAIRS", 2)


@dataclass(frozen=True)
class Settings:
    seed: int = _env_int("HRX_SEED", 20260915)
    company: str = os.getenv("HRX_COMPANY", "한빛전자")
    headcount: int = _env_int("HRX_HEADCOUNT", 820)
    llm_provider: str = os.getenv("HRX_LLM_PROVIDER", "auto")  # auto|anthropic|openai|none
    llm_model: str = os.getenv("HRX_LLM_MODEL", "")
    paths: Paths = field(default_factory=Paths)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    harness: HarnessConfig = field(default_factory=HarnessConfig)

    def resolve_llm(self) -> tuple[str, str] | None:
        """(provider, model) 또는 None(=오프라인 폴백)을 돌려준다."""
        provider = self.llm_provider.lower()
        if provider == "none":
            return None
        if provider in ("auto", "anthropic") and os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic", self.llm_model or "anthropic/claude-sonnet-4-20250514"
        if provider in ("auto", "openai") and os.getenv("OPENAI_API_KEY"):
            return "openai", self.llm_model or "gpt-4o-mini"
        return None


SETTINGS = Settings()
PATHS = SETTINGS.paths
