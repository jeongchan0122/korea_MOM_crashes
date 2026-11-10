# -*- coding: utf-8 -*-
"""
22단계 (탐색, 검증 결과를 본 뒤): Barroso & Santa-Clara (2015)의 변동성 조절 WML을 우리 최종 모델과 비교한다.
9.5의 검증 결론은 바꾸지 않는다.

무엇을 하는가 (연구 기록 20.1)
- 보유 달 m의 비중 = 목표 변동성 / (m-1월 말까지 126거래일 일별 2x3 WML 실현 변동성).
  실현 변동성 = sqrt(252 x 일별 수익률 제곱의 평균)(평균을 빼지 않음, 논문 방식).
- 목표 변동성 = 2000-07부터 2005-06까지 전체 일별 2x3 WML로 같은 방식으로 한 번 계산(선택 기간 전, 결과 전에 고정).
- (a) 비중 상한 없음(논문 그대로), (b) 비중 1 이하로 자름.
- 비용: 편도 0.5% x 종목별 매매량. 매매량 = sum |w_m x 새 비중 - w_(m-1) x 변한 비중| (12번의 보유표, 변한비중).
  w가 1->1, 1->0, 0->1이면 기존 회전율(리밸런싱, 청산, 구축)과 같은지 확인한다.
- 선택 기간, 검증 기간, 2025-12까지를 15번 평가 함수로 판정하고 최종 모델과 나란히 적는다.

실행: python scripts/22_탐색_변동성조절.py   (먼저 15번까지 실행해 두어야 함)
결과: 결과/탐색/변동성조절.csv, 결과/탐색/1_변동성조절_누적수익.png
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

프로젝트 = Path(__file__).resolve().parent.parent
처리폴더 = 프로젝트 / "처리데이터"
결과폴더 = 프로젝트 / "결과" / "탐색"


def 불러오기(이름, 파일):
    명세 = importlib.util.spec_from_file_location(이름, Path(__file__).resolve().parent / 파일)
    모듈 = importlib.util.module_from_spec(명세)
    명세.loader.exec_module(모듈)
    return 모듈


검증코드 = 불러오기("검증", "15_검증.py")
선택코드 = 검증코드.선택코드
회전코드 = 불러오기("회전", "12_회전율계산.py")
plt = 검증코드.plt

창, 연율 = 126, 252
목표구간 = ("2000-07-01", "2005-06-30")
구간들 = {"선택": 검증코드.선택구간, "검증": 검증코드.검증구간, "2025-12까지": 검증코드.끊은구간}


class 탐색문제(Exception):
    pass


def 확인(조건, 설명):
    if not 조건:
        raise 탐색문제(설명)


def 변동성():
    """구성 달(t월 말)별 126거래일 실현 변동성(연율, 소수)과 목표 변동성."""
    일 = pd.read_parquet(처리폴더 / "일별WML수익률.parquet")
    일 = 일[일["방식"] == "2x3"].set_index("날짜")["롱숏_%"].sort_index() / 100
    제곱 = (일 ** 2).rolling(창).mean()
    월말값 = 제곱.groupby(일.index.to_period("M")).last()
    시그마 = np.sqrt(연율 * 월말값)
    시그마.index = 시그마.index.astype(str)
    목표 = float(np.sqrt(연율 * (일.loc[목표구간[0]:목표구간[1]] ** 2).mean()))
    return 시그마.dropna(), 목표


def 매매량표(보유달):
    """보유 달마다 (새 비중 a, 변한 비중 b)를 같은 종목 순서로 맞춘 배열."""
    패널 = pd.read_parquet(처리폴더 / "월별패널.parquet")
    표 = 회전코드.보유표(패널)
    새 = {m: g.set_index("종목코드")["비중"] for m, g in 표.groupby("구성연월")}
    변 = {m: 회전코드.변한비중(g) for m, g in 표.groupby("구성연월")}
    쌍 = {}
    for m in 보유달:
        뒤, 앞 = str(pd.Period(m) - 1), str(pd.Period(m) - 2)
        a, b = 새[뒤], 변[앞]
        모두 = a.index.union(b.index)
        쌍[m] = (a.reindex(모두, fill_value=0).values, b.reindex(모두, fill_value=0).values)
    return 쌍


def 비용후(w, 앞w, 월, 쌍):
    """w: 보유 달 비중, 앞w: 첫 달 직전 비중. 비용 = 편도 0.5% x 매매량."""
    이전 = w.shift(1)
    이전.iloc[0] = 앞w
    매매 = pd.Series([np.abs(w[m] * 쌍[m][0] - 이전[m] * 쌍[m][1]).sum() for m in w.index], index=w.index)
    return w * 월 - 선택코드.비용률 * 매매, 매매


def 실행():
    월, 회전, 표, 단독, 보유달 = 검증코드.자료읽기()
    원래비용전 = 월["롱숏_%"]
    원래 = 선택코드.비용후(pd.Series(1.0, index=보유달), 원래비용전, 회전)
    시그마, 목표 = 변동성()
    print(f"[목표 변동성] 2000-07부터 2005-06까지 일별 2x3 WML: 연 {목표 * 100:.2f}%")

    쌍 = 매매량표(보유달)
    # 확인: 비중이 0과 1이면 기존 회전율과 같다
    하나 = pd.Series(1.0, index=보유달)
    _, 매1 = 비용후(하나, 1.0, 원래비용전, 쌍)
    확인(np.allclose(매1.values, 회전["리밸런싱"].values), "비중 1->1 매매량이 리밸런싱 회전율과 다름")
    w기존, _ = 검증코드.비중만들기(검증코드.신호만들기(표, 단독, 검증코드.기대최종), 보유달)
    r기존식, _ = 비용후(w기존, 1.0, 원래비용전, 쌍)
    r기존 = 선택코드.비용후(w기존, 원래비용전, 회전)
    확인(np.allclose(r기존식.values, r기존.values), "최종 모델의 0/1 비중에서 일반화한 비용이 기존 비용과 다름")
    print("[확인] 비중 1->1, 1->0, 0->1에서 일반화한 매매량이 기존 회전율과 같음(최종 모델 수익률 일치)")

    전비중 = 목표 / 시그마
    전비중.index = [str(pd.Period(m) + 1) for m in 전비중.index]   # t월 말 변동성 -> t+1월 비중
    전략 = {}
    for 이름, 상한 in (("변동성조절(a) 상한 없음", None), ("변동성조절(b) 1 이하", 1.0)):
        ww = 전비중 if 상한 is None else 전비중.clip(upper=상한)
        w = ww.reindex(보유달)
        확인(w.notna().all(), "보유 달 중 비중이 없는 달이 있음")
        앞 = float(ww.loc[str(pd.Period(보유달[0]) - 1)])
        r, 매 = 비용후(w, 앞, 원래비용전, 쌍)
        전략[이름] = (w, r, 매)
    _, 매기존 = 비용후(w기존, 1.0, 원래비용전, 쌍)
    전략["최종 모델(0/1)"] = (w기존, r기존, 매기존)

    줄들 = []
    for 구간이름, 구간 in 구간들.items():
        a, b = 구간
        for 이름, (w, r, 매) in 전략.items():
            e = 검증코드.평가(r, 원래, 원래비용전, w, 구간)
            줄들.append({"구간": 구간이름, "전략": 이름, "평균비중": w.loc[a:b].mean(), "최소비중": w.loc[a:b].min(),
                       "최대비중": w.loc[a:b].max(), "월평균비용_%": 선택코드.비용률 * 매.loc[a:b].mean(),
                       "월평균": e["월평균"], "샤프": e["샤프"], "최대낙폭": e["최대낙폭"], "사건합": e["사건합"],
                       "알파": e["알파"], "b": e["b"], "t": e["t"], "결론": e["결론"],
                       "원래_월평균": e["원래_월평균"], "원래_샤프": e["원래_샤프"], "원래_최대낙폭": e["원래_최대낙폭"],
                       "원래_사건합": e["원래_사건합"]})
    결과 = pd.DataFrame(줄들)
    with pd.option_context("display.width", 250):
        print(결과.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    결과폴더.mkdir(parents=True, exist_ok=True)
    결과.to_csv(결과폴더 / "변동성조절.csv", index=False, encoding="utf-8-sig")

    # 그림: 누적수익
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    날 = 선택코드.월말(보유달)
    fig, ax = plt.subplots(figsize=(11, 5.8))
    for (표지, r), 색 in zip([("원래 WML", 원래)] + [(k, v[1]) for k, v in 전략.items()],
                           [선택코드.파랑, "#eda100", "#1baf7a", 선택코드.주황]):
        누 = (1 + r / 100).cumprod()
        ax.plot(날, 누.values, color=색, linewidth=2, label=f"{표지}  {누.iloc[-1]:.2f}배")
    ax.axvline(pd.Timestamp("2016-01-01"), color=선택코드.보조글자, linewidth=1.2, linestyle="--")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([0.5, 0.75, 1, 1.5, 2, 3, 4]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}배"))
    선택코드.꾸미기(ax, "변동성 조절 WML과 최종 모델 (탐색)",
                f"목표 변동성 연 {목표 * 100:.1f}%(2000-07부터 2005-06까지). 시작값 1, 로그 눈금, 거래비용(편도 0.5%) 차감. 점선은 2016-01.")
    ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=10)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.84, bottom=0.08)
    fig.savefig(결과폴더 / "1_변동성조절_누적수익.png", dpi=200)
    plt.close(fig)
    print(f"\n[저장] {결과폴더}")


if __name__ == "__main__":
    실행()
