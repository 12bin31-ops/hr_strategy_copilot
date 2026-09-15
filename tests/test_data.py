"""합성 데이터가 '의도한 문제'를 실제로 담고 있는지 검증한다.

데이터 생성기가 조용히 바뀌면 아래 테스트가 먼저 깨지고, 그 다음에야 분석 결과가 바뀐다.
"""

from __future__ import annotations

import pandas as pd

from hrx.data.generate import STANDARD_TENURE, build_dataset


def test_dataset_shapes(data):
    assert len(data.employees) > 500
    assert data.employees.emp_id.is_unique
    assert set(data.org_units.level) == {"본부", "담당", "팀"}
    assert data.employees.manager_id.ne("").mean() > 0.9


def test_seed_is_reproducible():
    a = build_dataset(seed=42).employees
    b = build_dataset(seed=42).employees
    pd.testing.assert_frame_equal(a, b)


def test_planted_promotion_stagnation(data):
    g4 = data.employees[data.employees.grade == "G4"]
    by_div = g4.groupby("division").tenure_in_grade.mean()
    worst = by_div.idxmax()
    assert by_div[worst] / STANDARD_TENURE["G4"] >= 1.5, "승진 적체 시나리오가 사라졌다"


def test_planted_rating_leniency(data):
    sa = data.employees.groupby("division").rating_2025.apply(
        lambda s: s.isin(["S", "A"]).mean()
    )
    assert (sa.max() - sa.mean()) * 100 >= 10, "평가 관대화 시나리오가 사라졌다"


def test_planted_pay_inversion(data):
    e = data.employees[data.employees.grade.isin(["G2", "G3", "G4"])]
    by_path = e.groupby(["grade", "entry_path"]).base_pay.mean().unstack()
    assert (by_path["경력입사"] > by_path["신입공채"]).all(), "보상 역전 시나리오가 사라졌다"


def test_planted_oversized_teams(data):
    size = data.employees.groupby("unit_id").size()
    assert (size > 15).sum() >= 3


def test_succession_gap_exists(data):
    covered = data.succession.position_id.nunique()
    assert covered < len(data.key_positions), "후보가 없는 핵심 보직이 하나도 없다"
