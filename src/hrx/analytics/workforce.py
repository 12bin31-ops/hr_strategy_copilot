"""인원·인건비 시뮬레이션.

직급을 상태로 보는 마르코프 전이 모델이다. 매년 각 직급의 인원은
    승진(상위 직급으로), 이직(외부로), 정년(외부로), 잔류
로 분해되고, 부족분은 채용으로 메운다.

「조직운영 및 인원·인건비 계획 세칙」 제7조의 인건비 산출 구조
(기본 인상분 / 승진 소요분 / 채용 소요분 / 감소분)를 그대로 코드로 옮겼다.
경영진 보고용으로 기준·상방·하방 3개 시나리오를 함께 낸다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from hrx.data.generate import BASE_PAY, GRADES
from hrx.data.loader import HRData, load

LADDER = ["G1", "G2", "G3", "G4", "G5", "EX"]
RETIREMENT_AGE = 60


@dataclass
class Scenario:
    """시뮬레이션 가정. 모든 값은 '연간' 기준."""

    name: str
    promotion_rate: dict[str, float]
    attrition_rate: dict[str, float]
    base_raise: float = 0.033
    hiring_grade_mix: dict[str, float] = field(
        default_factory=lambda: {"G1": 0.55, "G2": 0.25, "G3": 0.15, "G4": 0.05}
    )
    fill_ratio: float = 1.0  # 결원 대비 충원 비율
    label: str = ""

    def describe(self) -> str:
        pr = ", ".join(f"{g} {v:.0%}" for g, v in self.promotion_rate.items() if g != "EX")
        ar = ", ".join(f"{g} {v:.1%}" for g, v in self.attrition_rate.items() if g != "EX")
        return f"[{self.name}] 승진율({pr}) / 이직률({ar}) / 기본인상 {self.base_raise:.1%}"


BASE_PROMOTION = {"G1": 0.22, "G2": 0.20, "G3": 0.16, "G4": 0.12, "G5": 0.06, "EX": 0.0}
BASE_ATTRITION = {"G1": 0.085, "G2": 0.078, "G3": 0.055, "G4": 0.041, "G5": 0.030, "EX": 0.040}


def default_scenarios() -> list[Scenario]:
    up = {g: v * 1.3 for g, v in BASE_PROMOTION.items()}
    down = {g: v * 0.75 for g, v in BASE_PROMOTION.items()}
    return [
        Scenario("기준", dict(BASE_PROMOTION), dict(BASE_ATTRITION), 0.033,
                 label="사업계획 기준 가정"),
        Scenario("상방", up, {g: v * 0.8 for g, v in BASE_ATTRITION.items()}, 0.045,
                 label="호황 — 승진 확대·이직 감소·인상률 상향"),
        Scenario("하방", down, {g: v * 1.35 for g, v in BASE_ATTRITION.items()}, 0.020,
                 label="불황 — 승진 억제·이직 증가·인상률 하향"),
    ]


def _retirement_counts(emp: pd.DataFrame, years: list[int]) -> dict[int, dict[str, int]]:
    out: dict[int, dict[str, int]] = {}
    for y in years:
        mask = (emp.birth_year + RETIREMENT_AGE) == y
        out[y] = emp[mask].grade.value_counts().to_dict()
    return out


@dataclass
class SimulationResult:
    scenario: Scenario
    headcount: pd.DataFrame  # year × grade
    cost: pd.DataFrame  # year, 항목별 인건비
    flows: pd.DataFrame  # year, grade, promoted_in/out, attrition, retire, hired

    @property
    def total_cost_by_year(self) -> pd.Series:
        return self.cost.set_index("year")["total_cost"]

    def summary(self) -> dict:
        c = self.cost.set_index("year")["total_cost"]
        h = self.headcount.set_index("year").sum(axis=1)
        first, last = c.index[0], c.index[-1]
        return {
            "scenario": self.scenario.name,
            "start_headcount": int(h.loc[first]),
            "end_headcount": int(h.loc[last]),
            "start_cost_억": round(c.loc[first] / 10_000, 1),
            "end_cost_억": round(c.loc[last] / 10_000, 1),
            "cagr_cost": round((c.loc[last] / c.loc[first]) ** (1 / (len(c) - 1)) - 1, 4)
            if len(c) > 1 else 0.0,
            "cumulative_hires": int(self.flows.hired.sum()),
            "cumulative_attrition": int(self.flows.attrition.sum()),
        }


def simulate(
    data: HRData | None = None,
    scenario: Scenario | None = None,
    years: int = 3,
    start_year: int = 2027,
    target_headcount: dict[int, int] | None = None,
) -> SimulationResult:
    d = data or load()
    sc = scenario or default_scenarios()[0]
    emp = d.employees

    state = emp.grade.value_counts().reindex(LADDER).fillna(0).astype(float)
    year_list = [start_year + i for i in range(years)]
    retire = _retirement_counts(emp, year_list)

    if target_headcount is None:
        plan = d.business_plan.groupby("year")["planned_headcount"].sum().to_dict()
        target_headcount = {y: int(plan.get(y, state.sum())) for y in year_list}

    # 직급별 평균 인건비 단가 (실제 데이터의 평균 기본연봉 사용)
    unit_cost = emp.groupby("grade").base_pay.mean().reindex(LADDER).fillna(
        pd.Series(BASE_PAY).reindex(LADDER)
    )

    hc_rows, cost_rows, flow_rows = [], [], []
    hc_rows.append({"year": start_year - 1, **state.to_dict()})
    prev_cost = float((state * unit_cost).sum())
    cost_rows.append(
        {
            "year": start_year - 1, "base_cost": prev_cost, "raise_cost": 0.0,
            "promotion_cost": 0.0, "hiring_cost": 0.0, "reduction_cost": 0.0,
            "total_cost": prev_cost,
        }
    )

    for year in year_list:
        promoted_out = {g: state[g] * sc.promotion_rate.get(g, 0.0) for g in LADDER}
        promoted_out["EX"] = 0.0
        attrition = {g: state[g] * sc.attrition_rate.get(g, 0.0) for g in LADDER}
        retirees = {g: float(retire.get(year, {}).get(g, 0)) for g in LADDER}

        new_state = state.copy()
        for i, g in enumerate(LADDER):
            new_state[g] -= promoted_out[g] + attrition[g] + retirees[g]
            if i + 1 < len(LADDER):
                new_state[LADDER[i + 1]] += promoted_out[g]
        new_state = new_state.clip(lower=0)

        gap = max(0.0, target_headcount.get(year, float(state.sum())) - float(new_state.sum()))
        hires = {g: gap * sc.fill_ratio * w for g, w in sc.hiring_grade_mix.items()}
        for g, n in hires.items():
            new_state[g] += n

        # ---- 인건비 (세칙 제7조 구조) -------------------------------------
        base_cost = float((state * unit_cost).sum())
        raise_cost = base_cost * sc.base_raise
        promotion_cost = sum(
            promoted_out[g] * max(0.0, float(unit_cost[LADDER[i + 1]] - unit_cost[g]))
            for i, g in enumerate(LADDER[:-1])
        )
        hiring_cost = sum(n * float(unit_cost[g]) * 0.5 for g, n in hires.items())  # 반년 재직 가정
        reduction_cost = sum(
            (attrition[g] + retirees[g]) * float(unit_cost[g]) * 0.5 for g in LADDER
        )
        total = base_cost + raise_cost + promotion_cost + hiring_cost - reduction_cost

        hc_rows.append({"year": year, **{g: round(float(new_state[g]), 1) for g in LADDER}})
        cost_rows.append(
            {
                "year": year,
                "base_cost": round(base_cost, 1),
                "raise_cost": round(raise_cost, 1),
                "promotion_cost": round(promotion_cost, 1),
                "hiring_cost": round(hiring_cost, 1),
                "reduction_cost": round(-reduction_cost, 1),
                "total_cost": round(total, 1),
            }
        )
        for g in LADDER:
            flow_rows.append(
                {
                    "year": year, "grade": g,
                    "promoted_out": round(promoted_out[g], 1),
                    "attrition": round(attrition[g], 1),
                    "retire": retirees[g],
                    "hired": round(hires.get(g, 0.0), 1),
                }
            )
        state = new_state

    return SimulationResult(
        scenario=sc,
        headcount=pd.DataFrame(hc_rows),
        cost=pd.DataFrame(cost_rows),
        flows=pd.DataFrame(flow_rows),
    )


def run_all_scenarios(data: HRData | None = None, years: int = 3) -> dict[str, SimulationResult]:
    d = data or load()
    return {s.name: simulate(d, s, years=years) for s in default_scenarios()}


def sensitivity(
    data: HRData | None = None,
    promotion_deltas: tuple[float, ...] = (-0.04, -0.02, 0.0, 0.02, 0.04),
    years: int = 3,
) -> pd.DataFrame:
    """승진율을 흔들었을 때 3년차 인건비가 얼마나 움직이는지 — 경영진 보고용 민감도."""
    d = data or load()
    rows = []
    for delta in promotion_deltas:
        sc = Scenario(
            f"승진율{delta:+.0%}",
            {g: max(0.0, v + delta) for g, v in BASE_PROMOTION.items()},
            dict(BASE_ATTRITION),
        )
        res = simulate(d, sc, years=years)
        c = res.cost.set_index("year")["total_cost"]
        rows.append(
            {
                "promotion_delta": delta,
                "final_cost_억": round(c.iloc[-1] / 10_000, 1),
                "vs_base_pct": None,
                "end_headcount": int(res.headcount.set_index("year").iloc[-1].sum()),
            }
        )
    df = pd.DataFrame(rows)
    base = df.loc[df.promotion_delta == 0.0, "final_cost_억"].iloc[0]
    df["vs_base_pct"] = ((df.final_cost_억 / base - 1) * 100).round(2)
    return df


def grade_pyramid(data: HRData | None = None) -> pd.DataFrame:
    """현재 직급 구성비와 적정 피라미드 대비 편차."""
    d = data or load()
    actual = d.employees.grade.value_counts(normalize=True).reindex(GRADES).fillna(0)
    ideal = pd.Series(
        {"G1": 0.30, "G2": 0.25, "G3": 0.21, "G4": 0.13, "G5": 0.09, "EX": 0.02}
    ).reindex(GRADES)
    return pd.DataFrame(
        {
            "grade": GRADES,
            "actual_ratio": actual.round(4).to_numpy(),
            "target_ratio": ideal.to_numpy(),
            "gap_pp": ((actual - ideal) * 100).round(2).to_numpy(),
            "headcount": d.employees.grade.value_counts().reindex(GRADES).fillna(0).astype(int).to_numpy(),
        }
    )
