# -*- coding: utf-8 -*-
"""
25단계 (미국 시험, 연구 기록 21절): 한국 최종 모델(겹침 조정 위험 점수: 시장 변동성 + 모멘텀 변동성)을
바꾸지 않고 미국 2x3 WML에 적용해 평가한다. 9.5의 결론은 바꾸지 않고 따로 보고한다.

무엇을 하는가
- 지표: 일별 시장(CRSP vwretd), 일별 2x3 WML의 직전 126거래일 표준편차, t월 마지막 거래일 값(11번과 같음).
- 신호: 13번 조합신호(표, -, (시장변동성, 모멘텀변동성), "점수") 그대로. 보유 달 m 비중 = 1 - (m-1월 신호).
- 시험 기간: 신호가 처음 나오는 보유 달부터 2024-12까지. 판정은 편도 0.5%, 15번 평가 함수 그대로.
- 보고: 비용 0%, 0.25%, 0.5%, 붕괴 사건별 결과, 2000년 전과 후, 금융위기 구간(2007-07부터 2009-06까지)과 그 밖, 그림 2개.

실행: python scripts/25_미국_최종모델.py   (먼저 24번을 실행해 두어야 함)
결과: 결과/미국/(판정.csv, 사건표.csv, 그림 png)
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

프로젝트 = Path(__file__).resolve().parent.parent
자료폴더 = 프로젝트 / "처리데이터" / "미국"
결과폴더 = 프로젝트 / "결과" / "미국"
끝달 = "2024-12"
위기 = ("2007-07", "2009-06")


def 불러오기(이름, 파일):
    명세 = importlib.util.spec_from_file_location(이름, Path(__file__).resolve().parent / 파일)
    모듈 = importlib.util.module_from_spec(명세)
    명세.loader.exec_module(모듈)
    return 모듈


검증코드 = 불러오기("검증", "15_검증.py")
선택코드 = 검증코드.선택코드
신호코드 = 검증코드.신호코드
plt = 검증코드.plt
최종지표 = ("시장변동성", "모멘텀변동성")
창 = 126


def 확인(조건, 설명):
    if not 조건:
        raise SystemExit(f"[멈춤] {설명}")


def 자료읽기():
    월 = pd.read_parquet(자료폴더 / "월별WML.parquet").set_index("연월")
    일 = pd.read_parquet(자료폴더 / "일별WML.parquet").set_index("날짜")
    회전 = pd.read_parquet(자료폴더 / "회전율.parquet").set_index("연월")
    return 월, 일, 회전


def 지표만들기(일):
    달 = 일.index.to_period("M").astype(str)
    굴림 = lambda s: s.rolling(창, min_periods=창).std()
    표 = pd.DataFrame({"시장변동성": 굴림(일["시장_%"]).groupby(달).last(),
                      "모멘텀변동성": 굴림(일["롱숏_%"]).groupby(달).last()})
    달력 = pd.period_range(표.index[0], 표.index[-1], freq="M").astype(str)
    확인(list(표.index) == list(달력), "지표 달에 빠진 달이 있음")
    return 표


def 미래정보점검(표, 신호):
    for 달 in ["1974-09", "1987-10", "2008-10", "2020-03"]:
        다시 = 신호코드.조합신호(표.loc[:달], None, 최종지표, "점수")
        확인(다시.loc[달] == 신호.loc[달] or (np.isnan(다시.loc[달]) and np.isnan(신호.loc[달])),
             f"{달}: 그달 이후 자료를 지우면 신호가 달라짐")
    print("[미래 정보 점검] 4개 달에서 그달 이후 자료를 지워도 신호가 같음")


def 실행():
    월, 일, 회전 = 자료읽기()
    표 = 지표만들기(일)
    신호 = 신호코드.조합신호(표, None, 최종지표, "점수")
    미래정보점검(표, 신호)

    첫 = str(pd.Period(신호.first_valid_index()) + 1)
    보유달 = list(pd.period_range(첫, 끝달, freq="M").astype(str))
    s = 신호.reindex([str(pd.Period(m) - 1) for m in 보유달])
    확인(s.notna().all(), "시험 기간 중 신호가 빈 달이 있음")
    w = pd.Series(1 - s.values, index=보유달)
    원래비용전 = 월["롱숏_%"].reindex(보유달)
    회전 = 회전.reindex(보유달)
    확인(원래비용전.notna().all() and 회전.notna().all().all(), "시험 기간 중 WML이나 회전율이 빈 달이 있음")
    print(f"\n[시험 기간] 보유 달 {첫}부터 {끝달}까지 {len(보유달)}개월, 쉰 달 {int((w == 0).sum())}개 "
          f"({(w == 0).mean() * 100:.1f}%)")

    줄들 = []
    구간들 = {"전체(판정)": (첫, 끝달), "2000년 전": (첫, "1999-12"), "2000년부터": ("2000-01", 끝달)}
    for 비용 in (0.0, 0.25, 0.5):
        선택코드.비용률 = 비용
        원래 = 선택코드.비용후(pd.Series(1.0, index=보유달), 원래비용전, 회전)
        r = 선택코드.비용후(w, 원래비용전, 회전)
        for 구간이름, 구간 in 구간들.items():
            e = 검증코드.평가(r, 원래, 원래비용전, w, 구간)
            줄들.append({"비용": 비용, "구간": 구간이름, "개월": e["개월"], "쉰달": e["켜진달"],
                       "월평균": e["월평균"], "원래_월평균": e["원래_월평균"], "샤프": e["샤프"], "원래_샤프": e["원래_샤프"],
                       "최대낙폭": e["최대낙폭"], "원래_최대낙폭": e["원래_최대낙폭"], "사건합": e["사건합"],
                       "원래_사건합": e["원래_사건합"], "알파": e["알파"], "b": e["b"], "t": e["t"], "결론": e["결론"]})
        if 비용 == 0.5:
            원래5, r5 = 원래, r
    판정 = pd.DataFrame(줄들)
    with pd.option_context("display.width", 250):
        print(판정.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # 금융위기 구간과 그 밖(비용 0.5%)
    안 = r5.index.to_series().between(*위기)
    for 이름, m in (("위기", 안), ("위기 밖", ~안)):
        a, b, t = 선택코드.알파t(r5[m].values, 원래5[m].values)
        print(f"[{이름}] {int(m.sum())}개월, 쉰 달 {int((w[m] == 0).sum())}, 알파 월 {a[0]:.2f}% (t {t[0]:.2f})")

    # 붕괴 사건
    선택코드.비용률 = 0.5
    사건, 붕괴달 = 선택코드.사건정의(원래비용전)
    사건표 = pd.DataFrame([{"사건": f"{a}부터 {b}까지" if a != b else a,
                         "원래_비용후": 선택코드.사건수익(원래5, [(a, b)])[0],
                         "모델_비용후": 선택코드.사건수익(r5, [(a, b)])[0],
                         "쉰달": int((w.loc[a:b] == 0).sum()), "개월": len(w.loc[a:b]),
                         "승자": 월.loc[a:b, "롱온리_%"].mean(), "패자": 월.loc[a:b, "패자쪽_%"].mean()}
                        for a, b in 사건])
    print(f"\n[붕괴 사건] 하위 5% {len(붕괴달)}개월, 사건 {len(사건)}개, 붕괴 달 중 쉰 달 "
          f"{int((w.loc[붕괴달] == 0).sum())}개")
    print(사건표.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    결과폴더.mkdir(parents=True, exist_ok=True)
    판정.to_csv(결과폴더 / "판정.csv", index=False, encoding="utf-8-sig")
    사건표.to_csv(결과폴더 / "사건표.csv", index=False, encoding="utf-8-sig")
    그림(w, 원래5, r5, 사건)
    print(f"\n[저장] {결과폴더}")


def 그림(w, 원래, r, 사건):
    from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator
    날 = 선택코드.월말(w.index)
    fig, ax = plt.subplots(figsize=(11, 5.8))
    선택코드.쉼음영(ax, w)
    for 값, 색, 표지 in ((원래, 선택코드.파랑, "원래 WML"), (r, 선택코드.주황, "한국 최종 모델")):
        누 = (1 + 값 / 100).cumprod()
        ax.plot(날, 누.values, color=색, linewidth=1.8, label=f"{표지}  {누.iloc[-1]:.1f}배")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}배"))
    선택코드.꾸미기(ax, "미국 2×3 WML에 한국 최종 모델을 그대로 적용",
                f"점수:시장변동성+모멘텀변동성. {w.index[0]}부터 {w.index[-1]}까지, 시작값 1, 로그 눈금, "
                "거래비용(편도 0.5%) 차감. 회색 = 쉰 달.")
    ax.legend(loc="upper left", frameon=False)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.84, bottom=0.08)
    fig.savefig(결과폴더 / "1_누적수익.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.0))
    선택코드.쉼음영(ax, w)
    for 값, 색, 표지 in ((원래, 선택코드.파랑, "원래 WML"), (r, 선택코드.주황, "한국 최종 모델")):
        낙 = 선택코드.낙폭(값)
        ax.plot(날, 낙.values, color=색, linewidth=1.6, label=f"{표지}  최대 {낙.min():.1f}%")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    선택코드.꾸미기(ax, "미국: 고점 대비 낙폭", "거래비용(편도 0.5%) 차감. 회색 = 쉰 달.")
    ax.legend(loc="lower left", frameon=False)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.82, bottom=0.08)
    fig.savefig(결과폴더 / "2_낙폭.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    실행()
