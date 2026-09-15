from __future__ import annotations

from hrx.cli import main
from hrx.evaluation.run import evaluate, save


def test_retrieval_quality_baseline():
    """검색 회귀 방지선. 이 값이 깨지면 파라미터 변경을 되돌려야 한다."""
    report = evaluate(k=6, brief_samples=2)
    assert report.recall >= 0.90, report.render()
    assert report.mrr >= 0.70
    assert report.route_accuracy == 1.0
    assert report.theme_accuracy >= 0.90


def test_generated_brief_has_no_hallucinated_citations():
    report = evaluate(k=6, brief_samples=3)
    assert report.hallucinated_citations == 0
    assert report.citation_coverage >= 0.5


def test_eval_report_is_saved(tmp_path):
    report = evaluate(k=6, brief_samples=1)
    path = save(report, tmp_path)
    assert path.exists() and (tmp_path / "eval.json").exists()


def test_cli_search_runs(capsys):
    assert main(["search", "평가등급 배분율", "-k", "3"]) == 0
    assert "INT-HR-002" in capsys.readouterr().out


def test_cli_diagnose_runs(capsys):
    assert main(["diagnose"]) == 0
    assert "조직진단 요약" in capsys.readouterr().out


def test_cli_brief_writes_file(tmp_path, capsys):
    out = tmp_path / "brief.md"
    assert main(["brief", "승진 적체 해소 방안", "-o", str(out)]) == 0
    assert out.exists() and "대안 비교" in out.read_text(encoding="utf-8")


def test_cli_eval_enforces_threshold(capsys):
    assert main(["eval", "--min-recall", "0.9"]) == 0
