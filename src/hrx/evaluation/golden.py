"""검색 평가용 골든셋.

실무 HR 담당자가 실제로 던질 법한 질의 20개에 대해, "이 답을 하려면 반드시 봐야 하는
근거"를 사람이 미리 지정했다. 검색기 파라미터를 바꿀 때마다 이 세트로 회귀 검증한다.
(검색 품질을 눈대중이 아니라 숫자로 관리하기 위한 최소 장치)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GoldenCase:
    question: str
    must_retrieve: tuple[str, ...]  # 하나라도 top-k 안에 있으면 hit
    expected_corpora: tuple[str, ...]
    expected_theme: str


GOLDEN: tuple[GoldenCase, ...] = (
    GoldenCase("평가 등급 배분율을 의무화하면 취업규칙 불이익 변경에 해당하나?",
               ("EXT-LAW-001§제94조",), ("external",), "평가"),
    GoldenCase("취업규칙을 고용노동부에 신고해야 하는 기준은?",
               ("EXT-LAW-001§제93조",), ("external",), "조직"),
    GoldenCase("차장 승진 적체를 판정하는 사내 기준이 뭐야?",
               ("INT-HR-005§제10조", "INT-HR-001§제5조"), ("internal",), "승진"),
    GoldenCase("승진율이 낮으면 인사담당이 뭘 해야 하지?",
               ("INT-HR-001§제15조",), ("internal",), "승진"),
    GoldenCase("발탁승진 자격 요건을 알려줘",
               ("INT-HR-001§제14조",), ("internal",), "승진"),
    GoldenCase("전문가 트랙 대상자 보상은 어떻게 정해져 있나?",
               ("INT-HR-003§제8조", "INT-HR-001§제16조"), ("internal",), "승진"),
    GoldenCase("평가등급 권장 배분율이 어떻게 되나?",
               ("INT-HR-002§제5조",), ("internal",), "평가"),
    GoldenCase("특정 본부의 S·A 비율이 지나치게 높으면 어떻게 처리하나?",
               ("INT-HR-002§제6조",), ("internal",), "평가"),
    GoldenCase("평가 결과에 이의를 제기할 수 있는 기간은?",
               ("INT-HR-002§제9조",), ("internal",), "평가"),
    GoldenCase("임금밴드 침투율 격차 관리 기준은?",
               ("INT-HR-003§제4조",), ("internal",), "보상"),
    GoldenCase("직무급을 도입할 때 법적으로 뭘 조심해야 하나?",
               ("EXT-LAW-002§제8조",), ("external",), "보상"),
    GoldenCase("직무급 전환을 단계적으로 하는 방법이 있나?",
               ("EXT-TRD-001§3.1 직무급 전환의 단계적 접근",), ("external",), "보상"),
    GoldenCase("핵심 보직의 승계 후보를 몇 명이나 확보해야 하나?",
               ("INT-HR-004§제5조",), ("internal",), "승계"),
    GoldenCase("정년이 임박한 핵심 보직에 후보가 없으면 어떻게 하나?",
               ("INT-HR-004§제6조",), ("internal",), "승계"),
    GoldenCase("후보자 준비도는 어떻게 구분하나?",
               ("INT-HR-004§제4조",), ("internal",), "승계"),
    GoldenCase("팀 인원이 몇 명을 넘으면 조직을 분할 검토해야 하나?",
               ("INT-HR-005§제3조",), ("internal",), "조직"),
    GoldenCase("인원계획을 세울 때 반영해야 하는 요소는?",
               ("INT-HR-005§제6조",), ("internal",), "인력계획"),
    GoldenCase("인건비 계획은 어떤 항목으로 구성되나?",
               ("INT-HR-005§제7조",), ("internal",), "인력계획"),
    GoldenCase("연장근로는 주 몇 시간까지 가능한가?",
               ("EXT-LAW-001§제53조",), ("external",), "조직"),
    GoldenCase("정년은 몇 세 이상으로 정해야 하나?",
               ("EXT-LAW-002§제19조",), ("external",), "승계"),
)
