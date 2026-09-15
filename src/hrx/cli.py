"""hrx 명령줄 인터페이스.

    hrx data                        합성 인사 데이터 생성
    hrx index                       RAG 인덱스 구축
    hrx search "질의"                하이브리드 검색 결과와 점수 내역 확인
    hrx diagnose                    조직진단 지표 출력
    hrx simulate --years 3          인원·인건비 시나리오 시뮬레이션
    hrx succession                  승계 커버리지 진단
    hrx brief "질의" -o out.md       멀티에이전트 제도 개편안 생성
    hrx eval                        검색·근거 품질 평가
    hrx app                         Streamlit 대시보드 실행 안내
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hrx.config import SETTINGS


def _p(*args) -> None:
    print(*args, flush=True)


def cmd_data(args) -> int:
    from hrx.data.generate import generate

    written = generate(Path(args.out) if args.out else None, seed=args.seed)
    for name, path in written.items():
        _p(f"  {name:16s} → {path}")
    return 0


def cmd_index(args) -> int:
    from hrx.rag.index import build_index

    bundle = build_index()
    _p(f"청크 {bundle.size}개 / 문서 {len({c.doc_id for c in bundle.chunks})}개 인덱싱 완료")
    _p(f"저장 위치: {SETTINGS.paths.index}")
    return 0


def cmd_search(args) -> int:
    from hrx.rag.retriever import HybridRetriever, route

    r = HybridRetriever()
    decision = route(args.query)
    _p(f"라우팅: corpora={decision.corpora} boost={decision.boosted_categories}")
    for res in r.search(args.query, top_k=args.k):
        _p("")
        _p(res.explain())
        _p(f"  {res.chunk.title} / {res.chunk.section}")
        _p(f"  {res.chunk.text[:220].replace(chr(10), ' ')}")
    return 0


def cmd_diagnose(args) -> int:
    from hrx.analytics import org_health

    _p(org_health.brief())
    if args.detail:
        _p("")
        _p(org_health.findings_frame().to_string(index=False))
    return 0


def cmd_simulate(args) -> int:
    from hrx.analytics import workforce

    for name, res in workforce.run_all_scenarios(years=args.years).items():
        _p(f"\n=== {name} — {res.scenario.describe()}")
        _p(res.cost.to_string(index=False))
        _p(f"요약: {res.summary()}")
    _p("\n=== 승진율 민감도")
    _p(workforce.sensitivity(years=args.years).to_string(index=False))
    return 0


def cmd_succession(args) -> int:
    from hrx.analytics import succession

    d = succession.diagnose()
    _p(d.headline())
    _p("")
    _p(d.coverage.to_string(index=False))
    if args.detail:
        _p("\n=== 9-Box")
        _p(d.nine_box.to_string(index=False))
        _p("\n=== 유지 리스크 상위")
        _p(succession.talent_at_risk().to_string(index=False))
    return 0


def cmd_brief(args) -> int:
    from hrx.agents.crew import run_crew

    run = run_crew(args.query)
    _p("--- 실행 로그 ---")
    _p(run.trace())
    if run.context_report:
        _p(run.context_report)
    _p("")
    md = run.brief.to_markdown()
    out = Path(args.out) if args.out else SETTINGS.paths.reports / "brief.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    _p(md if args.stdout else f"보고서 저장: {out}")
    return 0


def cmd_eval(args) -> int:
    from hrx.evaluation.run import evaluate, save

    report = evaluate(k=args.k)
    _p(report.render())
    path = save(report)
    _p(f"\n저장: {path}")
    threshold = args.min_recall
    if report.recall < threshold:
        _p(f"FAIL: recall@{args.k} {report.recall:.2f} < {threshold}")
        return 1
    return 0


def cmd_app(args) -> int:
    _p("Streamlit 대시보드를 실행하려면:")
    _p("  pip install -e '.[app]'")
    _p("  streamlit run src/hrx/app/dashboard.py")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="hrx", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("data", help="합성 인사 데이터 생성")
    p.add_argument("--out")
    p.add_argument("--seed", type=int)
    p.set_defaults(func=cmd_data)

    p = sub.add_parser("index", help="RAG 인덱스 구축")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("search", help="하이브리드 검색")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=6)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("diagnose", help="조직진단")
    p.add_argument("--detail", action="store_true")
    p.set_defaults(func=cmd_diagnose)

    p = sub.add_parser("simulate", help="인원·인건비 시뮬레이션")
    p.add_argument("--years", type=int, default=3)
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("succession", help="승계 커버리지 진단")
    p.add_argument("--detail", action="store_true")
    p.set_defaults(func=cmd_succession)

    p = sub.add_parser("brief", help="제도 개편안 생성")
    p.add_argument("query")
    p.add_argument("-o", "--out")
    p.add_argument("--stdout", action="store_true")
    p.set_defaults(func=cmd_brief)

    p = sub.add_parser("eval", help="검색·근거 품질 평가")
    p.add_argument("-k", type=int, default=6)
    p.add_argument("--min-recall", type=float, default=0.85)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("app", help="대시보드 실행 안내")
    p.set_defaults(func=cmd_app)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
