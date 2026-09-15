from __future__ import annotations

import pytest

from hrx.analytics import org_health, succession, workforce


def test_findings_detect_all_planted_issues(data):
    metrics = {f.metric for f in org_health.findings(data)}
    assert "스팬 오브 컨트롤" in metrics
    assert "승진 적체" in metrics
    assert "평가등급 분포 편차" in metrics
    assert "임금밴드 침투율 격차" in metrics


def test_findings_carry_citable_basis(data):
    for f in org_health.findings(data):
        assert "§" in f.basis, f"판정 근거가 인용 가능한 형식이 아니다: {f}"


def test_span_of_control_classification(data):
    span = org_health.span_of_control(data)
    assert set(span.status) <= {"적정", "상한초과", "분할검토", "통합검토"}
    assert (span[span.headcount > 15].status == "분할검토").all()


def test_simulation_conserves_headcount_direction(data):
    res = workforce.simulate(data, years=3)
    hc = res.headcount.set_index("year").sum(axis=1)
    assert (hc > 0).all()
    assert len(res.cost) == 4  # 기준연도 + 3개년
    assert res.cost.total_cost.iloc[-1] > 0


def test_scenarios_are_ordered(data):
    r = workforce.run_all_scenarios(data, years=3)
    end = {k: v.cost.total_cost.iloc[-1] for k, v in r.items()}
    assert end["하방"] < end["기준"] < end["상방"]


def test_sensitivity_monotonic(data):
    df = workforce.sensitivity(data, years=3)
    assert df.final_cost_억.is_monotonic_increasing
    assert df.loc[df.promotion_delta == 0.0, "vs_base_pct"].iloc[0] == 0.0


def test_promotion_cost_zero_when_no_promotion(data):
    sc = workforce.Scenario("무승진", {g: 0.0 for g in workforce.LADDER},
                            dict(workforce.BASE_ATTRITION))
    res = workforce.simulate(data, sc, years=2)
    assert res.cost.promotion_cost.iloc[1:].sum() == pytest.approx(0.0)


def test_succession_coverage_rules(data):
    diag = succession.diagnose(data)
    for row in diag.coverage.itertuples():
        if row.criticality == 5:
            assert row.target_total == 3 and row.target_ready_now == 1
        if row.status == "충족":
            assert row.ready_now_count >= row.target_ready_now
            assert row.candidate_count >= row.target_total


def test_effective_coverage_discounts_duplicates(data):
    diag = succession.diagnose(data)
    assert (diag.coverage.effective_coverage <= diag.coverage.candidate_count + 1e-9).all()


def test_nine_box_totals(data):
    grid = succession.nine_box(data)
    assert grid.headcount.sum() == len(data.employees)
    assert grid.ratio.sum() == pytest.approx(1.0, abs=1e-6)
