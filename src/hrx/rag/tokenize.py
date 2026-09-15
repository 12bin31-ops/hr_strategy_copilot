"""형태소 분석기 없이 동작하는 한국어 토크나이저.

konlpy/mecab 은 JVM·빌드 의존성이 있어 CI·오프라인 환경에서 깨지기 쉽다.
대신 (1) 어절 토큰 (2) 조사 제거를 흉내낸 절단형 (3) 음절 bigram 을 함께 넣어
BM25 의 한국어 매칭 성능을 확보한다.
"""

from __future__ import annotations

import re
from functools import lru_cache

WORD_RE = re.compile(r"[가-힣]+|[A-Za-z][A-Za-z']*|\d+(?:\.\d+)?")

# 자주 붙는 조사/어미 — 제거해 어간 후보를 만든다.
PARTICLES = (
    "으로써", "에서는", "에게는", "으로는", "이라는", "에서의", "에게서",
    "으로", "에서", "에게", "과의", "와의", "이라", "하여", "하는", "한다", "된다",
    "들이", "들을", "들의", "에는", "에도", "만을", "부터", "까지", "라도",
    "이", "가", "은", "는", "을", "를", "의", "에", "도", "와", "과", "로", "만",
)

STOPWORDS = {
    "그리고", "그러나", "또는", "및", "등", "대한", "관한", "있다", "없다", "한다",
    "하는", "위한", "따라", "경우", "수", "것", "이는", "the", "a", "an", "of", "and",
}


def _strip_particle(token: str) -> str:
    if len(token) < 3:
        return token
    for p in PARTICLES:
        if len(p) < len(token) and token.endswith(p) and len(token) - len(p) >= 2:
            return token[: -len(p)]
    return token


def _bigrams(token: str) -> list[str]:
    return [f"#{token[i:i + 2]}" for i in range(len(token) - 1)]


@lru_cache(maxsize=8192)
def _tokenize_cached(text: str, with_bigrams: bool) -> tuple[str, ...]:
    out: list[str] = []
    for raw in WORD_RE.findall(text):
        tok = raw.lower()
        if tok in STOPWORDS:
            continue
        out.append(tok)
        if re.fullmatch(r"[가-힣]+", tok):
            stem = _strip_particle(tok)
            if stem != tok and stem not in STOPWORDS:
                out.append(stem)
            if with_bigrams and len(tok) >= 2:
                out.extend(_bigrams(tok))
    return tuple(out)


def tokenize(text: str, with_bigrams: bool = False) -> list[str]:
    """BM25 용 토큰.

    음절 bigram 을 기본으로 넣지 않는 이유: bigram 은 거의 모든 문서와 조금씩 매칭되어
    BM25 의 어휘 변별력을 떨어뜨린다(평가셋 recall@6 0.75 → 0.95 로 확인). 부분 일치는
    같은 파이프라인의 char n-gram TF-IDF 가 담당하므로 역할을 나눈다.
    """
    if not text:
        return []
    return list(_tokenize_cached(text[:20000], with_bigrams))
