# -*- coding: utf-8 -*-
"""
20단계 (강건성 5단계): 투자 대상이나 분위 경계를 바꾼 표본에서 최종 모델을 다시 평가한다. 보고용이며 검증 결론은 바꾸지 않는다.

무엇을 하는가 (연구 기록 19.2.5, 팀 문서 8.6)
- 변형: 기본(재현 확인용), 코스피 단독, 금융주 제외(FnGuide 섹터 "금융"), 분위 경계를 투자 대상 전체로,
  삼성전자(A005930)와 SK하이닉스(A000660) 제외.
- 변형마다 다시 계산: 월별 2x3 WML(05번 함수, 상폐 수익률 포함), 일별 2x3 WML과 일별 시장(08번 함수, 35% 규칙),
  최종 모델의 두 지표(시장 변동성, 모멘텀 변동성, 126거래일), 회전율(12번 함수), 최종 모델 신호(13번 함수)와
  선택·검증·2025-12까지 평가(15번 함수, 비용 0.5%, 붕괴 사건은 그 변형의 원래 WML).
- 모델은 고정한다(지표 10개를 다시 만들어 모델을 다시 고르지 않음).
- 기본 변형이 05번, 08번, 15번 결과와 같은지 먼저 확인하고, 다르면 멈춘다.

업종 자료: 10_업종_Fnguide.xlsx(FnGuide Sector, 월별)를 읽기만 한다. 값이 2000-01부터 있어 1999-12에는 2000-01 값을 쓴다.
섹터 값이 없는 투자 대상 종목은 금융주가 아닌 것으로 본다.

실행: python scripts/20_강건성_표본.py   (먼저 15번까지 실행해 두어야 함)
결과: 결과/강건성/표본.csv
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from python_calamine import CalamineWorkbook

프로젝트 = Path(__file__).resolve().parent.parent
처리폴더 = 프로젝트 / "처리데이터"
결과폴더 = 프로젝트 / "결과" / "강건성"


def 불러오기(이름, 파일):
    명세 = importlib.util.spec_from_file_location(이름, Path(__file__).resolve().parent / 파일)
    모듈 = importlib.util.module_from_spec(명세)
    명세.loader.exec_module(모듈)
    return 모듈


검증코드 = 불러오기("검증", "15_검증.py")
선택코드, 신호코드 = 검증코드.선택코드, 검증코드.신호코드
회전코드 = 불러오기("회전", "12_회전율계산.py")
일별코드 = 불러오기("일별", "08_일별WML계산.py")
월별코드 = 일별코드.월별코드
일별코드.규칙들, 일별코드.방식들 = ["35%"], ["2x3"]   # 이번에는 기본 규칙과 2x3만 계산한다.
시가총액 = 월별코드.시가총액
일창 = 126
두종목 = ["A005930", "A000660"]
쪽 = {"SW": 0.5, "BW": 0.5, "SL": -0.5, "BL": -0.5}


class 표본문제(Exception):
    pass


def 확인(조건, 설명):
    if not 조건:
        raise 표본문제(설명)


def 업종읽기():
    """FnGuide 섹터를 (연월, 종목코드) -> 섹터로 읽는다. 1999-12에는 2000-01 값을 쓴다."""
    줄들 = CalamineWorkbook.from_path(str(프로젝트 / "한국데이터" / "10_업종_Fnguide.xlsx")).get_sheet_by_name("Sheet1") \
        .to_python(skip_empty_area=False)
    확인({str(v) for v in 줄들[11][1:]} == {"CP10000600"}, "업종 파일의 항목 코드가 FnGuide Sector가 아님")
    코드 = [str(v) for v in 줄들[8][1:]]
    날짜 = pd.to_datetime([r[0] for r in 줄들[14:]])
    표 = pd.DataFrame([r[1:] for r in 줄들[14:]], index=날짜.strftime("%Y-%m"), columns=코드).replace("", np.nan)
    표.loc["1999-12"] = 표.loc["2000-01"]
    return 표.stack().rename("섹터").rename_axis(["구성연월", "종목코드"])


def 대상만들기(패널, 변형, 섹터):
    대상 = 월별코드.투자대상표(패널)
    if 변형 == "코스피 단독":
        대상 = 대상[대상["시장구분"] == "KS"]
    elif 변형 == "금융주 제외":
        s = 섹터.reindex(pd.MultiIndex.from_arrays([대상["구성연월"], 대상["종목코드"]])).values
        print(f"    섹터 값이 없는 투자 대상 종목·월 {pd.isna(s).mean() * 100:.2f}%, 금융 {np.mean(s == '금융') * 100:.2f}% (시가총액 비중 "
              f"{대상.loc[s == '금융', 시가총액].sum() / 대상[시가총액].sum() * 100:.1f}%)")
        대상 = 대상[s != "금융"]
    elif 변형 == "삼성전자·SK하이닉스 제외":
        대상 = 대상[~대상["종목코드"].isin(두종목)]
    return 대상


def 월말표준편차(일):
    s = 일.rolling(일창, min_periods=일창).std()
    return s.groupby(s.index.to_period("M").astype(str)).last()


def 변형계산(변형, 패널, 일별, 시장구분, 섹터, 지표표, 보유달):
    대상 = 대상만들기(패널, 변형, 섹터)
    경계 = "전체" if 변형 == "전체 종목 경계" else "코스피"
    표 = 월별코드.바구니나누기(대상.merge(월별코드.모멘텀점수(패널), on=["구성연월", "종목코드"], how="inner"), 경계=경계)

    # 월별 2x3 WML (상폐 반영)
    보유 = 월별코드.다음달수익률붙이기(표, 패널)
    월 = 월별코드.승패계산(월별코드.바구니수익률(보유, "2x3"), "2x3").set_index("연월")["롱숏_%"]

    # 일별 2x3 WML과 일별 시장
    바구니, 시장수익률, _, _ = 일별코드.일별계산(대상, 표, 일별, 시장구분)
    일WML = 일별코드.승패(바구니["35%"], "날짜").set_index("날짜")["롱숏_%"].sort_index()
    일시장 = 시장수익률["35%"].sort_index()

    # 회전율
    보 = 보유[보유["2x3"].isin(쪽) & 보유["수익률"].notna()].copy()
    상폐 = 패널.set_index(["연월", "종목코드"])["상폐월"]
    보["상폐"] = 상폐.reindex(pd.MultiIndex.from_arrays([보["연월"], 보["종목코드"]])).fillna(False).astype(bool).values
    합 = 보.groupby(["구성연월", "2x3"])[시가총액].transform("sum")
    보["비중"] = 보["2x3"].map(쪽) * 보[시가총액] / 합
    재 = (보["비중"] * 보["수익률"]).groupby(보["연월"]).sum()
    확인((재 - 월.reindex(재.index)).abs().max() < 1e-9, f"{변형}: 비중으로 계산한 수익률이 월별 WML과 다름")
    회전 = 회전코드.계산(보).set_index("연월").loc[보유달[0]:보유달[-1]]

    # 최종 모델의 두 지표와 신호
    새표 = 지표표.copy()
    새표["시장변동성"] = 월말표준편차(일시장).reindex(지표표.index)
    새표["모멘텀변동성"] = 월말표준편차(일WML).reindex(지표표.index)
    신호 = 신호코드.위쪽판단(신호코드.위험점수(새표, ["시장변동성", "모멘텀변동성"]), 신호코드.점수창최소)
    return 월, 일WML, 일시장, 회전, 신호, len(대상), 대상.groupby("구성연월").size().mean()


def 실행():
    패널 = pd.read_parquet(처리폴더 / "월별패널.parquet")
    일별 = pd.read_parquet(처리폴더 / "일별수익률.parquet")
    시장구분 = 패널.set_index(["연월", "종목코드"])["시장구분"]
    섹터 = 업종읽기()
    _, _, 지표표, 단독, 보유달 = 검증코드.자료읽기()
    기존월 = pd.read_parquet(처리폴더 / "WML월별수익률.parquet")
    기존월 = 기존월[기존월["방식"] == "2x3"].set_index("연월")["롱숏_%"]
    기존일 = pd.read_parquet(처리폴더 / "일별WML수익률.parquet")
    기존일 = 기존일[기존일["방식"] == "2x3"].set_index("날짜").sort_index()

    줄들 = []
    for 변형 in ["기본", "코스피 단독", "금융주 제외", "전체 종목 경계", "삼성전자·SK하이닉스 제외"]:
        print(f"[{변형}] 계산 중")
        월, 일WML, 일시장, 회전, 신호, _, 평균종목 = 변형계산(변형, 패널, 일별, 시장구분, 섹터, 지표표, 보유달)
        if 변형 == "기본":
            확인((월 - 기존월.reindex(월.index)).abs().max() < 1e-9, "기본: 월별 WML이 05번과 다름")
            확인(np.allclose(일WML.values, 기존일["롱숏_%"].reindex(일WML.index).values), "기본: 일별 WML이 08번과 다름")
            확인(np.allclose(일시장.values, 기존일["시장_%"].reindex(일시장.index).values), "기본: 일별 시장이 08번과 다름")
            기본신호 = 검증코드.신호만들기(지표표, 단독, 검증코드.기대최종)
            확인(np.allclose(신호.values, 기본신호.values, equal_nan=True), "기본: 최종 모델 신호가 다름")
            print("    기본 변형: 월별 WML, 일별 WML, 일별 시장, 신호가 기존 결과와 같음")
        전 = 월.loc[보유달[0]:보유달[-1]]
        w, 채움 = 검증코드.비중만들기(신호, 보유달)
        원래 = 선택코드.비용후(pd.Series(1.0, index=보유달), 전, 회전)
        r = 선택코드.비용후(w, 전, 회전)
        for 구간이름, 구간 in (("선택", 검증코드.선택구간), ("검증", 검증코드.검증구간), ("2025-12까지", 검증코드.끊은구간)):
            e = 검증코드.평가(r, 원래, 전, w, 구간)
            if 변형 == "기본" and 구간이름 == "검증":
                확인(abs(e["t"] - (-1.5311829934)) < 1e-8, "기본: 검증 결과가 15번과 다름")
            줄들.append({"변형": 변형, "구간": 구간이름, "판정": e["결론"], "쉰달": e["켜진달"],
                       "붕괴포착": f"{e['붕괴포착']}/{len(e['_붕괴달'])}", "원래_비용전월평균": 전.loc[구간[0]:구간[1]].mean(),
                       "원래_월평균": e["원래_월평균"], "최종_월평균": e["월평균"], "원래_샤프": e["원래_샤프"], "최종_샤프": e["샤프"],
                       "원래_최대낙폭": e["원래_최대낙폭"], "최종_최대낙폭": e["최대낙폭"], "원래_사건합": e["원래_사건합"],
                       "최종_사건합": e["사건합"], "알파": e["알파"], "t": e["t"], "평균종목수": 평균종목,
                       "월평균회전율": 회전.loc[구간[0]:구간[1], "리밸런싱"].mean(), "선택_채운달": 채움["선택"]})
        print(f"    완료 (구성 달 평균 투자 대상 {평균종목:.0f}종목)")
    표결과 = pd.DataFrame(줄들)
    print("\n[표본 변형] 최종 모델(시장 변동성 + 모멘텀 변동성, 위험 점수), 비용 0.5%")
    with pd.option_context("display.width", 270):
        print(표결과.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    결과폴더.mkdir(parents=True, exist_ok=True)
    표결과.to_csv(결과폴더 / "표본.csv", index=False, encoding="utf-8-sig")
    print(f"\n[저장] {결과폴더 / '표본.csv'}")


if __name__ == "__main__":
    실행()
