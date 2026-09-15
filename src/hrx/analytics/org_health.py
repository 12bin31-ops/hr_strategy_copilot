"""조직 진단.

「조직운영 및 인원·인건비 계획 세칙」 제9조가 정한 5개 진단 지표를 그대로 구현한다.
    1) 스팬 오브 컨트롤 (제3조)
    2) 직급 구성비 및 승진 적체율 (제10조)
    3) 평가등급 분포 편차 (성과관리 지침 제6조)
    4) 임금밴드 침투율 (보상 지침 제4조)
    5) VOC 부정 응답 비율

각 지표는 판정 근거가 되는 조문을 문자열로 함께 반환한다. 에이전트가 이 값을
그대로 인용할 수 있게 하기 위한 설계다(= 숫자와 근거를 한 객체에 묶는다).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hrx.data.generate import STANDARD_TENURE
from hrx.data.loader import HRData, load

SPAN_MIN, SPAN_MAX, SPAN_ALERT = 7, 12, 15
RATING_DEVIATION_PP = 10.0
COMPA_GAP_PP = 8.0
STAGNATION_MULTIPLE = 1.5


@dataclass
class Finding:
    metric: str
    scope: str
    value: float
    threshold: float
    severity: str  # 정상 | 주의 | 위험
    basis: str  # 판정 근거 조문
    detail: str

    def to_dict(self) -> dict:
        return {
            "지표": self.metric, "대상": self.scope, "값": self.value,
            "기준": self.threshold, "판정": self.severity, "근거": self.basis,
            "설명": self.detail,
        }


# --------------------------------------------------------------------- 지표 1
def span_of_control(data: HRData | None = None) -> pd.DataFrame:
    d = data or load()
    teams = d.org_units[d.org_units.level == "팀"]
    size = d.employees.groupby("unit_id").size()
    df = teams.assign(headcount=teams.unit_id.map(size).fillna(0).astype(int))
    df["status"] = np.select(
        [df.headcount > SPAN_ALERT, df.headcount > SPAN_MAX, df.headcount < 5],
        ["분할검토", "상한초과", "통합검토"],
        default="적정",
    )
    return df[["unit_id", "unit_name", "division", "headcount", "status"]].sort_values(
        "headcount", ascending=False
    )


# --------------------------------------------------------------------- 지표 2
def promotion_stagnation(data: HRData | None = None) -> pd.DataFrame:
    d = data or load()
    e = d.employees[d.employees.grade != "EX"]
    g = (
        e.groupby(["division", "grade"])
        .agg(headcount=("emp_id", "count"), avg_tenure_in_grade=("tenure_in_grade", "mean"))
        .reset_index()
    )
    g["standard_tenure"] = g.grade.map(STANDARD_TENURE)
    g["ratio"] = (g.avg_tenure_in_grade / g.standard_tenure).round(2)
    g["avg_tenure_in_grade"] = g.avg_tenure_in_grade.round(2)
    g["status"] = np.where(g.ratio >= STAGNATION_MULTIPLE, "적체", "정상")
    return g.sort_values("ratio", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------- 지표 3
def rating_distribution(data: HRData | None = None, year: int = 2025) -> pd.DataFrame:
    d = data or load()
    col = f"rating_{year}"
    e = d.employees
    company_sa = float(e[col].isin(["S", "A"]).mean() * 100)
    g = (
        e.groupby("division")[col]
        .apply(lambda s: pd.Series({
            "headcount": len(s),
            "sa_ratio_pct": round(float(s.isin(["S", "A"]).mean() * 100), 2),
            "cd_ratio_pct": round(float(s.isin(["C", "D"]).mean() * 100), 2),
        }))
        .unstack()
        .reset_index()
    )
    g["company_sa_pct"] = round(company_sa, 2)
    g["deviation_pp"] = (g.sa_ratio_pct - company_sa).round(2)
    g["status"] = np.select(
        [g.deviation_pp >= RATING_DEVIATION_PP, g.deviation_pp <= -RATING_DEVIATION_PP],
        ["관대화", "엄격화"],
        default="정상",
    )
    return g.sort_values("deviation_pp", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------- 지표 4
def compa_ratio(data: HRData | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(직급×본부 침투율, 입사경로별 격차)."""
    d = data or load()
    e = d.employees[d.employees.grade != "EX"].copy()
    band_mid = e.groupby("grade").base_pay.median()
    e["compa"] = e.base_pay / e.grade.map(band_mid)
    by_unit = (
        e.groupby(["division", "grade"])
        .agg(headcount=("emp_id", "count"), avg_compa=("compa", "mean"))
        .reset_index()
    )
    by_unit["avg_compa"] = by_unit.avg_compa.round(3)

    gap = (
        e.groupby(["grade", "entry_path"])
        .agg(headcount=("emp_id", "count"), avg_compa=("compa", "mean"))
        .reset_index()
        .pivot(index="grade", columns="entry_path", values="avg_compa")
    )
    gap = gap.reindex(columns=["경력입사", "신입공채"]).dropna(how="all")
    gap["gap_pp"] = ((gap["경력입사"] - gap["신입공채"]) * 100).round(2)
    gap["status"] = np.where(gap.gap_pp.abs() > COMPA_GAP_PP, "형평성 점검 필요", "정상")
    return by_unit, gap.reset_index().round(3)


# --------------------------------------------------------------------- 지표 5
def voc_profile(data: HRData | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = data or load()
    v = d.voc
    by_topic = (
        v.groupby("topic")
        .agg(
            responses=("response_id", "count"),
            negative=("sentiment", lambda s: int((s == "neg").sum())),
        )
        .reset_index()
    )
    by_topic["negative_ratio_pct"] = (by_topic.negative / by_topic.responses * 100).round(1)
    by_topic = by_topic.sort_values("negative", ascending=False).reset_index(drop=True)

    by_div = (
        v.groupby("division")
        .agg(
            responses=("response_id", "count"),
            negative=("sentiment", lambda s: int((s == "neg").sum())),
        )
        .reset_index()
    )
    by_div["negative_ratio_pct"] = (by_div.negative / by_div.responses * 100).round(1)
    return by_topic, by_div.sort_values("negative_ratio_pct", ascending=False).reset_index(drop=True)


def voc_quotes(data: HRData | None = None, topic: str | None = None, n: int = 5) -> list[dict]:
    d = data or load()
    v = d.voc
    if topic:
        v = v[v.topic == topic]
    v = v[v.sentiment == "neg"]
    if v.empty:
        return []
    return v.head(n)[["response_id", "division", "grade", "topic", "text"]].to_dict("records")


# ----------------------------------------------------------------- 종합 진단
def findings(data: HRData | None = None) -> list[Finding]:
    d = data or load()
    out: list[Finding] = []

    span = span_of_control(d)
    over = span[span.headcount > SPAN_ALERT]
    # 같은 유형의 발견이 수십 건이면 컨텍스트만 잡아먹는다. 상위 5건 + 집계 1건으로 압축.
    if len(over) > 5:
        out.append(
            Finding(
                "스팬 오브 컨트롤", "전사 집계", float(len(over)), float(SPAN_ALERT), "위험",
                "INT-HR-005§제3조",
                f"15명 초과 팀 {len(over)}개 / 전체 {len(span)}개 팀 "
                f"(최대 {int(over.headcount.max())}명, 해당 인원 {int(over.headcount.sum())}명)",
            )
        )
    for r in over.head(5).itertuples():
        out.append(
            Finding(
                "스팬 오브 컨트롤", f"{r.division}/{r.unit_name}", float(r.headcount),
                float(SPAN_ALERT), "위험",
                "INT-HR-005§제3조",
                f"팀 인원 {r.headcount}명으로 상한 15명 초과 — 조직 분할 또는 파트 제도 검토 대상",
            )
        )

    stag = promotion_stagnation(d)
    for r in stag[stag.status == "적체"].itertuples():
        out.append(
            Finding(
                "승진 적체", f"{r.division}/{r.grade}", float(r.ratio), STAGNATION_MULTIPLE,
                "위험", "INT-HR-005§제10조",
                f"평균 체류 {r.avg_tenure_in_grade}년 / 표준 {r.standard_tenure}년 "
                f"({r.ratio}배, 대상 {r.headcount}명)",
            )
        )

    rat = rating_distribution(d)
    for r in rat[rat.status != "정상"].itertuples():
        out.append(
            Finding(
                "평가등급 분포 편차", r.division, float(r.deviation_pp), RATING_DEVIATION_PP,
                "주의", "INT-HR-002§제6조",
                f"S·A 비율 {r.sa_ratio_pct}% vs 전사 {r.company_sa_pct}% "
                f"({r.deviation_pp:+.1f}%p, {r.status})",
            )
        )

    _, gap = compa_ratio(d)
    for _, r in gap[gap.status != "정상"].iterrows():
        out.append(
            Finding(
                "임금밴드 침투율 격차", f"{r['grade']} 입사경로", float(r["gap_pp"]), COMPA_GAP_PP,
                "주의", "INT-HR-003§제4조",
                f"경력입사 침투율 {r['경력입사']:.3f} vs 신입공채 {r['신입공채']:.3f} "
                f"({r['gap_pp']:+.1f}%p)",
            )
        )

    by_topic, by_div = voc_profile(d)
    top = by_topic.iloc[0]
    out.append(
        Finding(
            "VOC 부정 응답", f"최다 주제: {top.topic}", float(top.negative_ratio_pct), 50.0,
            "주의" if top.negative_ratio_pct >= 50 else "정상",
            "INT-HR-005§제9조",
            f"'{top.topic}' 주제 {top.responses}건 중 부정 {top.negative}건 "
            f"({top.negative_ratio_pct}%)",
        )
    )
    return out


def findings_frame(data: HRData | None = None) -> pd.DataFrame:
    return pd.DataFrame([f.to_dict() for f in findings(data)])


def brief(data: HRData | None = None) -> str:
    """에이전트 컨텍스트에 바로 넣을 수 있는 진단 요약 텍스트."""
    fs = findings(data)
    lines = ["# 조직진단 요약 (자동 산출)"]
    for sev in ("위험", "주의"):
        sub = [f for f in fs if f.severity == sev]
        if not sub:
            continue
        lines.append(f"\n## {sev} 판정 {len(sub)}건")
        for f in sub[:12]:
            lines.append(f"- [{f.metric}] {f.scope}: {f.detail} (근거 {f.basis})")
    return "\n".join(lines)
