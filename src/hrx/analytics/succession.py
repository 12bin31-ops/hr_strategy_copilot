"""승계·리더십 파이프라인 진단.

「리더십 파이프라인 운영지침」 제5조(커버리지 기준)·제6조(정년 도래 대응)·
제9조(9-Box)를 그대로 지표화한다.

핵심 아이디어는 **표면 커버리지와 실질 커버리지를 분리**하는 것이다.
한 후보가 여러 보직에 중복 지명되어 있으면 장부상 후보 수는 채워져 있어도
실제로는 한 자리밖에 못 채운다(사례집 D 참조).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from hrx.data.loader import HRData, load

CURRENT_YEAR = 2026
READY_NOW = "Ready Now"

# 지침 제5조: 핵심도별 목표
COVERAGE_TARGET = {5: {"ready_now": 1, "total": 3}, 4: {"ready_now": 1, "total": 2}}
DEFAULT_TARGET = {"ready_now": 0, "total": 1}


@dataclass
class SuccessionDiagnosis:
    coverage: pd.DataFrame
    at_risk: pd.DataFrame
    duplicate_candidates: pd.DataFrame
    nine_box: pd.DataFrame

    def headline(self) -> str:
        n = len(self.coverage)
        risk = len(self.at_risk)
        no_ready = int((self.coverage.ready_now_count == 0).sum())
        return (
            f"핵심 보직 {n}개 중 승계 위험 {risk}개, Ready Now 후보가 전무한 보직 {no_ready}개. "
            f"평균 실질 커버리지 {self.coverage.effective_coverage.mean():.2f}명/보직."
        )


def _target(criticality: int) -> dict:
    return COVERAGE_TARGET.get(int(criticality), DEFAULT_TARGET)


def diagnose(data: HRData | None = None) -> SuccessionDiagnosis:
    d = data or load()
    kp = d.key_positions.copy()
    su = d.succession.copy()

    # --- 중복 지명: 한 후보가 몇 개 보직에 올라 있는가 ---------------------
    if len(su):
        dup = (
            su.groupby(["candidate_id", "candidate_name"])
            .agg(nominated_positions=("position_id", "nunique"))
            .reset_index()
            .query("nominated_positions > 1")
            .sort_values("nominated_positions", ascending=False)
        )
        su["nomination_weight"] = su.candidate_id.map(
            su.groupby("candidate_id").position_id.nunique()
        ).rdiv(1.0)
    else:  # pragma: no cover
        dup = pd.DataFrame(columns=["candidate_id", "candidate_name", "nominated_positions"])
        su["nomination_weight"] = []

    rows = []
    for pos in kp.itertuples():
        cands = su[su.position_id == pos.position_id]
        ready = cands[cands.readiness == READY_NOW]
        tgt = _target(pos.criticality)
        years_to_vacancy = int(pos.expected_vacancy_year) - CURRENT_YEAR
        shortfalls = []
        if len(ready) < tgt["ready_now"]:
            shortfalls.append(f"Ready Now {len(ready)}/{tgt['ready_now']}")
        if len(cands) < tgt["total"]:
            shortfalls.append(f"후보 총수 {len(cands)}/{tgt['total']}")
        if years_to_vacancy <= 3 and len(ready) == 0:
            shortfalls.append(f"정년 {years_to_vacancy}년 내 + Ready Now 없음")
        rows.append(
            {
                "position_id": pos.position_id,
                "title": pos.title,
                "division": pos.division,
                "criticality": int(pos.criticality),
                "incumbent_name": pos.incumbent_name,
                "years_to_vacancy": years_to_vacancy,
                "candidate_count": len(cands),
                "ready_now_count": len(ready),
                "effective_coverage": round(float(cands.nomination_weight.sum()), 2),
                "avg_assessment": round(float(cands.assessment_score.mean()), 1) if len(cands) else 0.0,
                "target_ready_now": tgt["ready_now"],
                "target_total": tgt["total"],
                "shortfall": "; ".join(shortfalls),
                "status": "위험" if shortfalls else "충족",
            }
        )
    coverage = pd.DataFrame(rows).sort_values(
        ["status", "criticality", "years_to_vacancy"], ascending=[True, False, True]
    )
    at_risk = coverage[coverage.status == "위험"].copy()
    return SuccessionDiagnosis(coverage, at_risk, dup, nine_box(d))


def nine_box(data: HRData | None = None) -> pd.DataFrame:
    """성과(직전 2개년 평가) × 잠재력(H/M/L) 9-Box. 지침 제9조."""
    d = data or load()
    e = d.employees.copy()
    score = {"S": 5, "A": 4, "B": 3, "C": 2, "D": 1}
    e["perf_score"] = (e.rating_2025.map(score) + e.rating_2024.map(score)) / 2
    e["perf_band"] = pd.cut(
        e.perf_score, bins=[0, 2.9, 3.9, 5.1], labels=["저성과", "기대수준", "고성과"]
    )
    e["pot_band"] = e.potential.map({"L": "저잠재", "M": "중잠재", "H": "고잠재"})
    grid = (
        e.groupby(["perf_band", "pot_band"], observed=False)
        .agg(headcount=("emp_id", "count"), avg_pay=("base_pay", "mean"))
        .reset_index()
    )
    grid["ratio"] = (grid.headcount / grid.headcount.sum()).round(4)
    grid["avg_pay"] = grid.avg_pay.round(0)
    grid["box"] = grid.perf_band.astype(str) + "×" + grid.pot_band.astype(str)
    return grid


def talent_at_risk(data: HRData | None = None, top_n: int = 15) -> pd.DataFrame:
    """고성과·고잠재 인재 중 유지 리스크가 높은 인원 (지침 제9조 제2항).

    개인을 불이익 처분에 쓰기 위한 예측이 아니라, 리더 면담 대상 선정을 위한 조직 운영
    지표로 설계했다 (트렌드 노트 5.2 참조).
    """
    d = data or load()
    e = d.employees.copy()
    e = e[(e.potential == "H") & (e.rating_2025.isin(["S", "A"]))].copy()
    if e.empty:  # pragma: no cover
        return e
    from hrx.data.generate import STANDARD_TENURE

    band_mid = e.groupby("grade").base_pay.median()
    e["compa_ratio"] = (e.base_pay / e.grade.map(band_mid)).round(3)
    e["over_tenure"] = (
        e.tenure_in_grade / e.grade.map(STANDARD_TENURE).replace(99, pd.NA).astype(float)
    ).round(2)
    # 리스크: 밴드 하위 + 승진 지연 + 대형 팀 소속
    team_size = e.unit_id.map(d.employees.groupby("unit_id").size())
    e["risk_score"] = (
        (1 - e.compa_ratio).clip(lower=0) * 100
        + (e.over_tenure.fillna(0) - 1).clip(lower=0) * 30
        + (team_size > 15).astype(int) * 12
    ).round(1)
    cols = [
        "emp_id", "name", "division", "unit_name", "grade", "rating_2025",
        "compa_ratio", "over_tenure", "risk_score",
    ]
    return e.sort_values("risk_score", ascending=False)[cols].head(top_n).reset_index(drop=True)
