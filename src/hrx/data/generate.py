"""가상 제조기업 '한빛전자'의 인사 데이터를 합성한다.

실제 기업 데이터를 쓸 수 없으므로 시드 고정된 합성 데이터를 만든다.
다만 무작위 잡음이 아니라 **분석이 잡아내야 할 문제를 의도적으로 심어 둔다.**

  1) 차장(G4) 승진 적체  : 특정 본부 차장 평균 체류연수가 비정상적으로 김
  2) 평가 관대화         : 생산 조직의 S/A 비율이 전사 대비 과도하게 높음
  3) 승계 공백           : 핵심 보직 일부에 Ready-Now 후보가 없음
  4) 보상 역전           : 동일 직급 내 신규 입사자가 장기 재직자보다 높은 밴드
  5) 고이직 조직         : 스팬 오브 컨트롤이 15명을 넘는 팀의 이직 의향 VOC 집중
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from hrx.config import SETTINGS

GRADES = ["G1", "G2", "G3", "G4", "G5", "EX"]
GRADE_NAMES = {
    "G1": "사원",
    "G2": "대리",
    "G3": "과장",
    "G4": "차장",
    "G5": "부장",
    "EX": "임원",
}
GRADE_MIX = [0.26, 0.24, 0.22, 0.16, 0.10, 0.02]
BASE_PAY = {"G1": 4300, "G2": 5500, "G3": 6900, "G4": 8300, "G5": 9900, "EX": 16500}
STD_PAY = {"G1": 320, "G2": 420, "G3": 520, "G4": 640, "G5": 780, "EX": 2200}
# 표준 체류연수(이 값을 크게 넘어서면 '승진 적체')
STANDARD_TENURE = {"G1": 4, "G2": 4, "G3": 5, "G4": 5, "G5": 6, "EX": 99}

JOB_FAMILIES = ["연구개발", "생산기술", "품질보증", "영업마케팅", "구매SCM", "경영지원"]

DIVISIONS = {
    "반도체사업본부": {"families": ["생산기술", "품질보증", "연구개발"], "weight": 0.32},
    "디스플레이사업본부": {"families": ["생산기술", "품질보증"], "weight": 0.20},
    "연구개발본부": {"families": ["연구개발"], "weight": 0.22},
    "글로벌영업본부": {"families": ["영업마케팅", "구매SCM"], "weight": 0.16},
    "경영지원본부": {"families": ["경영지원"], "weight": 0.10},
}

RATINGS = ["S", "A", "B", "C", "D"]
RATING_CURVE = [0.05, 0.20, 0.55, 0.15, 0.05]
LENIENT_CURVE = [0.14, 0.34, 0.46, 0.05, 0.01]  # 관대화 조직
POTENTIALS = ["H", "M", "L"]

VOC_TOPICS = ["평가", "보상", "승진", "육성", "조직문화", "업무량", "리더십", "복리후생"]

VOC_TEMPLATES: dict[str, dict[str, list[str]]] = {
    "평가": {
        "neg": [
            "평가 등급이 어떤 기준으로 나뉘는지 팀원 입장에서 설명을 들은 적이 없습니다. 상대평가 배분이 팀 규모에 따라 달라지는 점도 납득하기 어렵습니다.",
            "연초에 세운 목표와 연말 평가 항목이 달라져 있었습니다. 목표 변경 시 합의 절차가 형식적으로만 운영됩니다.",
            "평가 피드백 면담이 10분 만에 끝났고 등급 근거보다 다음 해 과제 이야기만 했습니다.",
        ],
        "neu": [
            "평가 시스템 자체보다 평가자 간 눈높이 차이가 더 큰 문제라고 생각합니다. 평가자 교육이 있으면 좋겠습니다.",
            "절대평가와 상대평가를 직무 특성에 따라 나눠 적용하는 방안을 검토해 주셨으면 합니다.",
        ],
        "pos": [
            "올해부터 도입된 중간 점검 면담은 도움이 되었습니다. 기대 수준을 미리 맞출 수 있었습니다.",
        ],
    },
    "보상": {
        "neg": [
            "경력 입사자 연봉이 같은 직급 기존 인원보다 높다는 이야기가 공공연히 돕니다. 내부 형평성 점검이 필요합니다.",
            "성과급 산정식이 공개되지 않아 결과를 받아도 왜 그 금액인지 알 수 없습니다.",
            "직무 난이도 차이가 보상에 전혀 반영되지 않습니다. 교대 근무 조직의 보상 체계 재설계가 필요합니다.",
        ],
        "neu": [
            "직무급 요소를 일부 도입하되 기존 인원 보호 장치를 두는 단계적 전환이 현실적이라고 봅니다.",
            "동종업계 대비 기본급 수준을 주기적으로 공유해 주시면 좋겠습니다.",
        ],
        "pos": [
            "작년 임금 인상률 결정 과정을 설명회로 공유한 것은 긍정적이었습니다.",
        ],
    },
    "승진": {
        "neg": [
            "차장 승진 대상자가 계속 쌓이는데 정원 때문에 매년 몇 명만 올라갑니다. 예측 가능성이 전혀 없습니다.",
            "승진 심사 기준에 근속연수 비중이 지나치게 높아 성과가 좋아도 순서를 기다려야 합니다.",
            "직책과 직급이 분리되지 않아 팀장이 될 자리가 없으면 승진도 막힙니다.",
        ],
        "neu": [
            "승진 연한을 줄이는 대신 심사를 엄격히 하는 방향이 맞다고 생각합니다.",
            "전문가 트랙이 생기면 관리자로 가지 않아도 성장 경로가 보일 것 같습니다.",
        ],
        "pos": [
            "발탁 승진 제도가 신설된 점은 긍정적으로 봅니다.",
        ],
    },
    "육성": {
        "neg": [
            "교육 과정이 집합 교육 위주라 교대 근무자는 사실상 참여가 불가능합니다.",
            "직무별 역량 모델이 문서로만 있고 실제 육성 계획과 연결되지 않습니다.",
        ],
        "neu": [
            "리더 후보자 대상 사전 교육이 승진 이후가 아니라 이전에 있어야 합니다.",
            "사내 직무 이동 공고가 더 자주 열렸으면 합니다.",
        ],
        "pos": [
            "올해 신설된 데이터 분석 교육은 실무에 바로 적용할 수 있었습니다.",
        ],
    },
    "조직문화": {
        "neg": [
            "회의가 보고 중심으로 흘러 의사결정이 늦어집니다.",
            "조직 개편이 잦아 담당 업무가 1년에 두 번 바뀌었습니다.",
        ],
        "neu": [
            "부서 간 협업 규칙을 문서화하면 갈등이 줄 것 같습니다.",
        ],
        "pos": [
            "유연근무 확대 이후 업무 몰입도가 올라갔다고 느낍니다.",
        ],
    },
    "업무량": {
        "neg": [
            "팀 인원이 늘지 않은 상태에서 과제만 추가되어 상시 초과근무가 발생합니다. 이직을 고민하고 있습니다.",
            "결원이 6개월째 충원되지 않아 남은 인원이 업무를 나눠 맡고 있습니다.",
            "팀장이 관리해야 할 인원이 너무 많아 개별 면담이 거의 없습니다.",
        ],
        "neu": [
            "업무량 산정 기준이 있어야 증원 요청이 받아들여질 것 같습니다.",
        ],
        "pos": [
            "자동화 도구 도입 후 정기 보고 업무 시간이 줄었습니다.",
        ],
    },
    "리더십": {
        "neg": [
            "팀장이 성과 피드백을 주지 않아 무엇을 개선해야 할지 알기 어렵습니다.",
            "리더 선임이 성과보다 연공 순으로 이뤄진다는 인식이 있습니다.",
        ],
        "neu": [
            "리더 진단 결과를 본인에게만 주지 말고 육성 계획까지 연결해 주세요.",
        ],
        "pos": [
            "새 담당 임원 부임 후 의사결정 속도가 빨라졌습니다.",
        ],
    },
    "복리후생": {
        "neg": [
            "복지 포인트 사용처가 제한적이라 실제로 다 쓰지 못합니다.",
            "사업장별 복지 수준 차이가 큽니다.",
        ],
        "neu": [
            "선택적 복지 항목을 생애주기별로 재설계했으면 합니다.",
        ],
        "pos": [
            "건강검진 항목 확대는 만족도가 높습니다.",
        ],
    },
}

KEY_POSITION_TITLES = [
    ("반도체사업본부장", "반도체사업본부", 5),
    ("디스플레이사업본부장", "디스플레이사업본부", 5),
    ("연구개발본부장", "연구개발본부", 5),
    ("글로벌영업본부장", "글로벌영업본부", 4),
    ("경영지원본부장", "경영지원본부", 4),
    ("공정기술담당", "반도체사업본부", 5),
    ("품질보증담당", "반도체사업본부", 4),
    ("선행연구담당", "연구개발본부", 5),
    ("해외영업담당", "글로벌영업본부", 4),
    ("구매SCM담당", "글로벌영업본부", 3),
    ("인사담당", "경영지원본부", 4),
    ("재무담당", "경영지원본부", 4),
]

SURNAMES = list("김이박최정강조윤장임한오서신권황안송류전홍고문양손배백허유남심노")
GIVEN = [
    "민준", "서연", "도윤", "지우", "예준", "하윤", "주원", "지호", "지민", "수빈",
    "현우", "다은", "건우", "유진", "시우", "가은", "준서", "소율", "동현", "채원",
]


@dataclass
class Dataset:
    org_units: pd.DataFrame
    employees: pd.DataFrame
    key_positions: pd.DataFrame
    succession: pd.DataFrame
    voc: pd.DataFrame
    business_plan: pd.DataFrame

    def write(self, out_dir: Path) -> dict[str, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        for name in ("org_units", "employees", "key_positions", "succession", "voc", "business_plan"):
            path = out_dir / f"{name}.csv"
            getattr(self, name).to_csv(path, index=False, encoding="utf-8-sig")
            written[name] = path
        return written


def _name(rng: np.random.Generator) -> str:
    return rng.choice(SURNAMES) + rng.choice(GIVEN)


def _build_org(rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict] = []
    uid = 0
    for division, meta in DIVISIONS.items():
        uid += 1
        div_id = f"U{uid:03d}"
        rows.append(
            dict(unit_id=div_id, unit_name=division, parent_id="", level="본부", division=division)
        )
        n_offices = int(rng.integers(3, 6))
        for o in range(n_offices):
            uid += 1
            off_id = f"U{uid:03d}"
            family = rng.choice(meta["families"])
            rows.append(
                dict(
                    unit_id=off_id,
                    unit_name=f"{family}{o + 1}담당",
                    parent_id=div_id,
                    level="담당",
                    division=division,
                )
            )
            for t in range(int(rng.integers(2, 6))):
                uid += 1
                rows.append(
                    dict(
                        unit_id=f"U{uid:03d}",
                        unit_name=f"{family}{o + 1}-{t + 1}팀",
                        parent_id=off_id,
                        level="팀",
                        division=division,
                    )
                )
    return pd.DataFrame(rows)


def _assign_ratings(rng: np.random.Generator, division: str, n: int) -> np.ndarray:
    curve = LENIENT_CURVE if division == "디스플레이사업본부" else RATING_CURVE
    return rng.choice(RATINGS, size=n, p=curve)


def _build_employees(rng: np.random.Generator, org: pd.DataFrame, target: int) -> pd.DataFrame:
    teams = org[org.level == "팀"].reset_index(drop=True)
    # 본부 가중치에 맞춰 팀별 정원 배분 (일부 팀은 의도적으로 과대 편성 → 스팬 오브 컨트롤 이슈)
    weights = teams.division.map(lambda d: DIVISIONS[d]["weight"]).to_numpy(dtype=float)
    weights = weights / weights.sum()
    sizes = rng.multinomial(target - len(org[org.level != "팀"]), weights)
    oversized = rng.choice(len(teams), size=6, replace=False)
    sizes[oversized] += 7

    today = date(2026, 9, 1)
    rows: list[dict] = []
    eid = 0

    # 본부장/담당 임원 및 팀장 배치는 아래에서 manager_id 로 연결
    for idx, team in teams.iterrows():
        n = int(sizes[idx])
        if n <= 0:
            continue
        division = team.division
        family = rng.choice(DIVISIONS[division]["families"])
        grades = rng.choice(GRADES[:-1], size=n, p=np.array(GRADE_MIX[:-1]) / sum(GRADE_MIX[:-1]))
        # 팀장은 최상위 직급 1명
        order = np.argsort([GRADES.index(g) for g in grades])[::-1]
        leader_local = int(order[0])
        ratings = {
            year: _assign_ratings(rng, division, n) for year in (2023, 2024, 2025)
        }
        for i in range(n):
            eid += 1
            grade = grades[i]
            std = STANDARD_TENURE[grade]
            # 승진 적체: 반도체사업본부 차장은 체류연수를 크게 늘린다
            if grade == "G4" and division == "반도체사업본부":
                tenure_in_grade = float(rng.normal(std + 3.4, 1.1))
            else:
                tenure_in_grade = float(rng.normal(std * 0.8, 1.6))
            tenure_in_grade = max(0.4, tenure_in_grade)
            # 입사 경로: 경력 입사자는 해당 직급으로 바로 들어와 총 근속이 짧다.
            entry_path = "경력입사" if rng.random() < 0.24 and grade != "G1" else "신입공채"
            if entry_path == "경력입사":
                total_tenure = tenure_in_grade * float(rng.uniform(0.9, 1.25))
            else:
                total_tenure = tenure_in_grade + GRADES.index(grade) * float(rng.normal(3.6, 0.9))
            total_tenure = max(tenure_in_grade, total_tenure)
            hire = today - timedelta(days=int(total_tenure * 365.25))
            grade_start = today - timedelta(days=int(tenure_in_grade * 365.25))
            pay = float(rng.normal(BASE_PAY[grade], STD_PAY[grade]))
            # 보상 역전: 경력 입사자에게 밴드 상단 프리미엄이 누적된다 (사례집 C)
            if entry_path == "경력입사" and grade in ("G2", "G3", "G4"):
                pay *= float(rng.uniform(1.07, 1.17))
            rows.append(
                dict(
                    emp_id=f"E{eid:05d}",
                    name=_name(rng),
                    unit_id=team.unit_id,
                    unit_name=team.unit_name,
                    division=division,
                    job_family=family,
                    grade=grade,
                    grade_name=GRADE_NAMES[grade],
                    entry_path=entry_path,
                    is_team_leader=int(i == leader_local),
                    hire_date=hire.isoformat(),
                    grade_start_date=grade_start.isoformat(),
                    tenure_years=round(total_tenure, 2),
                    tenure_in_grade=round(tenure_in_grade, 2),
                    birth_year=int(1966 + min(38, GRADES.index(grade) * 6) + rng.integers(-4, 8)),
                    base_pay=round(pay, 1),
                    rating_2023=ratings[2023][i],
                    rating_2024=ratings[2024][i],
                    rating_2025=ratings[2025][i],
                    potential=rng.choice(POTENTIALS, p=[0.18, 0.62, 0.20]),
                    manager_id="",
                )
            )

    emp = pd.DataFrame(rows)

    # 임원 배치: 본부별 1~2명
    exec_rows: list[dict] = []
    for division in DIVISIONS:
        for k in range(int(rng.integers(1, 3))):
            eid += 1
            unit = org[(org.division == division) & (org.level == "본부")].iloc[0]
            exec_rows.append(
                dict(
                    emp_id=f"E{eid:05d}",
                    name=_name(rng),
                    unit_id=unit.unit_id,
                    unit_name=unit.unit_name,
                    division=division,
                    job_family="경영진",
                    grade="EX",
                    grade_name="임원",
                    entry_path="신입공채",
                    is_team_leader=0,
                    hire_date=(today - timedelta(days=int(rng.uniform(18, 28) * 365))).isoformat(),
                    grade_start_date=(today - timedelta(days=int(rng.uniform(1, 7) * 365))).isoformat(),
                    tenure_years=round(float(rng.uniform(18, 28)), 2),
                    tenure_in_grade=round(float(rng.uniform(1, 7)), 2),
                    birth_year=int(1970 + rng.integers(-4, 6)),
                    base_pay=round(float(rng.normal(BASE_PAY["EX"], STD_PAY["EX"])), 1),
                    rating_2023=rng.choice(["S", "A", "B"], p=[0.3, 0.5, 0.2]),
                    rating_2024=rng.choice(["S", "A", "B"], p=[0.3, 0.5, 0.2]),
                    rating_2025=rng.choice(["S", "A", "B"], p=[0.3, 0.5, 0.2]),
                    potential=rng.choice(["H", "M"], p=[0.6, 0.4]),
                    manager_id="",
                )
            )
    emp = pd.concat([emp, pd.DataFrame(exec_rows)], ignore_index=True)

    # manager 연결: 팀원 → 팀장, 팀장 → 해당 본부 임원
    leaders = emp[emp.is_team_leader == 1].set_index("unit_id")["emp_id"].to_dict()
    execs = emp[emp.grade == "EX"].groupby("division")["emp_id"].first().to_dict()
    emp["manager_id"] = [
        execs.get(row.division, "")
        if row.is_team_leader == 1 or row.grade == "EX"
        else leaders.get(row.unit_id, "")
        for row in emp.itertuples()
    ]
    return emp


def _build_key_positions(rng: np.random.Generator, emp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pool = emp[emp.grade.isin(["EX", "G5"])]
    for i, (title, division, criticality) in enumerate(KEY_POSITION_TITLES, start=1):
        cand = pool[pool.division == division]
        if cand.empty:
            cand = pool
        incumbent = cand.sample(1, random_state=int(rng.integers(0, 10_000))).iloc[0]
        rows.append(
            dict(
                position_id=f"KP{i:02d}",
                title=title,
                division=division,
                criticality=criticality,
                incumbent_id=incumbent.emp_id,
                incumbent_name=incumbent.name,
                incumbent_birth_year=int(incumbent.birth_year),
                # 정년 60세 기준 예상 공석 연도
                expected_vacancy_year=int(incumbent.birth_year) + 60,
            )
        )
    return pd.DataFrame(rows)


def _build_succession(rng: np.random.Generator, emp: pd.DataFrame, kp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pool = emp[emp.grade.isin(["G4", "G5", "EX"])]
    for pos in kp.itertuples():
        # 승계 공백을 의도적으로 심는다: 일부 핵심 보직은 후보가 0~1명
        if pos.position_id in ("KP06", "KP08", "KP10"):
            n_cand = int(rng.integers(0, 2))
        else:
            n_cand = int(rng.integers(2, 5))
        cand_pool = pool[(pool.division == pos.division) & (pool.emp_id != pos.incumbent_id)]
        if len(cand_pool) < n_cand:
            cand_pool = pool[pool.emp_id != pos.incumbent_id]
        if n_cand == 0 or cand_pool.empty:
            continue
        picks = cand_pool.sample(min(n_cand, len(cand_pool)), random_state=int(rng.integers(0, 10_000)))
        for p in picks.itertuples():
            readiness = rng.choice(
                ["Ready Now", "1-2년", "3년+"],
                p=[0.34, 0.38, 0.28] if pos.criticality < 5 else [0.20, 0.42, 0.38],
            )
            rows.append(
                dict(
                    position_id=pos.position_id,
                    candidate_id=p.emp_id,
                    candidate_name=p.name,
                    candidate_grade=p.grade,
                    readiness=readiness,
                    assessment_score=round(float(np.clip(rng.normal(72, 9), 40, 98)), 1),
                    last_assessed="2026-03",
                )
            )
    return pd.DataFrame(rows)


def _build_voc(rng: np.random.Generator, emp: pd.DataFrame, n: int = 420) -> pd.DataFrame:
    team_size = emp.groupby("unit_id").size().to_dict()
    rows = []
    sample = emp.sample(min(n, len(emp)), random_state=SETTINGS.seed)
    for i, e in enumerate(sample.itertuples(), start=1):
        big_team = team_size.get(e.unit_id, 0) >= 15
        if big_team:
            topic = rng.choice(VOC_TOPICS, p=[0.10, 0.12, 0.12, 0.08, 0.10, 0.28, 0.12, 0.08])
        elif e.grade == "G4":
            topic = rng.choice(VOC_TOPICS, p=[0.12, 0.14, 0.30, 0.08, 0.10, 0.08, 0.10, 0.08])
        else:
            topic = rng.choice(VOC_TOPICS)
        sentiment = rng.choice(["neg", "neu", "pos"], p=[0.55, 0.31, 0.14])
        bucket = VOC_TEMPLATES[topic][sentiment]
        rows.append(
            dict(
                response_id=f"V{i:04d}",
                unit_id=e.unit_id,
                division=e.division,
                grade=e.grade,
                job_family=e.job_family,
                topic=topic,
                sentiment=sentiment,
                survey_wave="2026-상반기",
                text=str(rng.choice(bucket)),
            )
        )
    return pd.DataFrame(rows)


def _build_business_plan(rng: np.random.Generator, emp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    current = emp.groupby("division").size().to_dict()
    growth = {
        "반도체사업본부": 0.06,
        "디스플레이사업본부": -0.04,
        "연구개발본부": 0.09,
        "글로벌영업본부": 0.02,
        "경영지원본부": -0.01,
    }
    for division, base in current.items():
        n = base
        for year in (2027, 2028, 2029):
            n = n * (1 + growth[division] + float(rng.normal(0, 0.01)))
            rows.append(
                dict(
                    year=year,
                    division=division,
                    planned_headcount=int(round(n)),
                    rationale="사업계획 기준 증감률 적용",
                )
            )
    return pd.DataFrame(rows)


def build_dataset(seed: int | None = None, headcount: int | None = None) -> Dataset:
    rng = np.random.default_rng(seed if seed is not None else SETTINGS.seed)
    org = _build_org(rng)
    emp = _build_employees(rng, org, headcount or SETTINGS.headcount)
    kp = _build_key_positions(rng, emp)
    succession = _build_succession(rng, emp, kp)
    voc = _build_voc(rng, emp)
    plan = _build_business_plan(rng, emp)
    return Dataset(org, emp, kp, succession, voc, plan)


def generate(out_dir: Path | None = None, seed: int | None = None) -> dict[str, Path]:
    out = out_dir or SETTINGS.paths.synthetic
    return build_dataset(seed=seed).write(out)
