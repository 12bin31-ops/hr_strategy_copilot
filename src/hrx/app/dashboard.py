"""Streamlit 대시보드.

실행: streamlit run src/hrx/app/dashboard.py

탭 구성은 HR 담당자의 업무 흐름을 그대로 따른다.
    진단 → 인력·인건비 계획 → 승계 파이프라인 → 근거 검색 → 제도 개편안
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:  # 패키지 미설치 상태에서도 실행되게
    sys.path.insert(0, str(ROOT / "src"))

from hrx.analytics import org_health, succession, workforce  # noqa: E402
from hrx.config import SETTINGS  # noqa: E402
from hrx.data.loader import load  # noqa: E402
from hrx.rag.retriever import HybridRetriever, route  # noqa: E402

st.set_page_config(page_title="HR Strategy Copilot", page_icon="📊", layout="wide")


@st.cache_resource
def get_retriever() -> HybridRetriever:
    return HybridRetriever()


@st.cache_data
def get_data():
    return load()


@st.cache_data
def get_findings() -> pd.DataFrame:
    return org_health.findings_frame()


d = get_data()

st.title("HR Strategy Copilot")
st.caption(
    f"{SETTINGS.company} · 재직 {len(d.employees)}명 · 조직 {len(d.org_units)}개 "
    "— 합성 데이터 기반 데모"
)

tabs = st.tabs(["조직진단", "인력·인건비", "승계 파이프라인", "근거 검색", "제도 개편안"])

# ------------------------------------------------------------------ 조직진단
with tabs[0]:
    findings = get_findings()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("재직 인원", f"{len(d.employees):,}명")
    c2.metric("위험 판정", int((findings["판정"] == "위험").sum()))
    c3.metric("주의 판정", int((findings["판정"] == "주의").sum()))
    c4.metric("VOC 응답", f"{len(d.voc):,}건")

    st.subheader("진단 결과")
    st.dataframe(findings, use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        st.subheader("스팬 오브 컨트롤")
        span = org_health.span_of_control(d)
        st.bar_chart(span.set_index("unit_name").head(18)["headcount"])
        st.caption("사내 기준: 적정 7~12명, 15명 초과 시 분할 검토 (INT-HR-005§제3조)")
    with right:
        st.subheader("승진 적체 (체류연수 ÷ 표준연수)")
        stag = org_health.promotion_stagnation(d)
        pivot = stag.pivot(index="grade", columns="division", values="ratio")
        st.dataframe(pivot.style.background_gradient(cmap="Reds", axis=None),
                     use_container_width=True)
        st.caption("1.5배 이상이면 적체로 판정 (INT-HR-005§제10조)")

    left, right = st.columns(2)
    with left:
        st.subheader("평가등급 분포 편차")
        st.dataframe(org_health.rating_distribution(d), use_container_width=True, hide_index=True)
    with right:
        st.subheader("입사경로별 임금밴드 침투율")
        _, gap = org_health.compa_ratio(d)
        st.dataframe(gap, use_container_width=True, hide_index=True)

    st.subheader("VOC")
    by_topic, by_div = org_health.voc_profile(d)
    c1, c2 = st.columns([2, 3])
    c1.dataframe(by_topic, use_container_width=True, hide_index=True)
    topic = c2.selectbox("주제 선택", by_topic.topic.tolist())
    for q in org_health.voc_quotes(d, topic=topic, n=5):
        c2.info(f"({q['division']} / {q['grade']}) {q['text']}")

# ------------------------------------------------------------- 인력·인건비
with tabs[1]:
    st.subheader("시나리오별 3개년 인원·인건비")
    years = st.slider("시뮬레이션 기간(년)", 1, 5, 3)
    results = workforce.run_all_scenarios(d, years=years)
    st.dataframe(pd.DataFrame([r.summary() for r in results.values()]),
                 use_container_width=True, hide_index=True)

    cost = pd.DataFrame(
        {name: res.total_cost_by_year / 10_000 for name, res in results.items()}
    )
    st.line_chart(cost)
    st.caption("단위: 억원. 세칙 제7조의 산출 구조(기본인상/승진소요/채용소요/감소분) 적용")

    pick = st.selectbox("상세 시나리오", list(results))
    res = results[pick]
    st.caption(res.scenario.describe())
    c1, c2 = st.columns(2)
    c1.dataframe(res.cost, use_container_width=True, hide_index=True)
    c2.dataframe(res.headcount, use_container_width=True, hide_index=True)

    st.subheader("승진율 민감도")
    st.dataframe(workforce.sensitivity(d, years=years), use_container_width=True, hide_index=True)

    st.subheader("직급 구성비 vs 적정 피라미드")
    st.dataframe(workforce.grade_pyramid(d), use_container_width=True, hide_index=True)

# ------------------------------------------------------- 승계 파이프라인
with tabs[2]:
    diag = succession.diagnose(d)
    st.info(diag.headline())
    c1, c2, c3 = st.columns(3)
    c1.metric("핵심 보직", len(diag.coverage))
    c2.metric("승계 위험 보직", len(diag.at_risk))
    c3.metric("Ready Now 부재", int((diag.coverage.ready_now_count == 0).sum()))

    st.subheader("보직별 커버리지")
    st.dataframe(diag.coverage, use_container_width=True, hide_index=True)
    st.caption("커버리지 기준: 핵심도 5 → Ready Now 1명·후보 3명, 핵심도 4 → 1명·2명 "
               "(INT-HR-004§제5조)")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("9-Box 분포")
        nb = diag.nine_box.pivot(index="perf_band", columns="pot_band", values="headcount")
        st.dataframe(nb.style.background_gradient(cmap="Blues", axis=None),
                     use_container_width=True)
    with c2:
        st.subheader("고성과·고잠재 인재 유지 리스크")
        st.dataframe(succession.talent_at_risk(d), use_container_width=True, hide_index=True)
        st.caption("조직 운영·면담 대상 선정용 지표. 개인 불이익 판단 근거로 쓰지 않는다.")

# --------------------------------------------------------------- 근거 검색
with tabs[3]:
    st.subheader("사내 규정 · 법령 · 트렌드 하이브리드 검색")
    q = st.text_input("질의", "평가 등급 배분율을 강화하면 취업규칙 불이익 변경인가?")
    k = st.slider("top-k", 3, 12, 6)
    if q:
        decision = route(q)
        st.caption(f"라우팅 → 코퍼스 {decision.corpora} / 가중 카테고리 "
                   f"{decision.boosted_categories or '없음'}")
        for res in get_retriever().search(q, top_k=k):
            with st.expander(f"#{res.rank} {res.chunk.citation_key} — {res.chunk.section}"):
                st.code(res.explain(), language="text")
                st.markdown(res.chunk.text)

# ------------------------------------------------------------ 제도 개편안
with tabs[4]:
    st.subheader("멀티에이전트 제도 개편안 생성")
    st.caption(
        "LLM 키가 설정되어 있으면 CrewAI 5-에이전트 경로로, 없으면 결정론적 폴백 경로로 "
        "동일한 스키마의 보고서를 만든다."
    )
    question = st.text_area(
        "검토 요청", "반도체사업본부 차장 승진 적체를 어떻게 해소해야 하나?", height=80
    )
    if st.button("보고서 생성", type="primary"):
        from hrx.agents.crew import run_crew

        with st.spinner("에이전트 실행 중…"):
            run = run_crew(question, get_retriever())
        st.code(run.trace(), language="text")
        if run.context_report:
            with st.expander("컨텍스트 예산 사용 내역"):
                st.code(run.context_report, language="text")
        md = run.brief.to_markdown()
        st.markdown(md)
        st.download_button("보고서 내려받기", md, file_name="policy_brief.md")
