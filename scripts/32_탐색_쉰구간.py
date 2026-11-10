# -*- coding: utf-8 -*-
"""
32단계 (탐색, 검증 결과를 본 뒤): 최종 모델이 쉰 구간마다 원래 WML과 모델이 어떻게 달랐는지 두 기간을 이어 본다.
최종 발표(2026-11-09) 22장 "두 기간을 함께 보면"의 그림과 숫자의 근거다. 새 모델을 고르거나 판정을 바꾸지 않는다.

무엇을 하는가 (연구 기록 26절)
- 15번의 함수로 최종 모델의 비중 w(0 또는 1)와 비용 후 수익률을 만든다(편도 0.5%).
- 쉰 구간 = 비중이 0인 달이 연속된 묶음. 2016-01에서 끊어 선택 기간과 검증 기간으로 나눈다.
- 구간별로
  - 원래 WML 누적수익(비용 전 롱숏 수익의 복리),
  - 차이 = 쉰 달들에서 (모델 비용 후 - 원래 WML 비용 후) 수익의 합(%p).
- 붕괴 달 = 기간별 원래 WML(비용 전) 하위 5%(14번 사건정의). 쉬는 중이면 "피함", 투자 중이면 "놓침".
  놓친 달은 다음 달에 쉬었는지도 적는다.
- 쉰 구간 차이의 합은 모델과 원래 WML의 전체 차이와 같지 않다. 다시 들어가는 달의 구축 비용과
  나오는 달의 청산 비용은 투자한 달에 붙기 때문이다. 이 나머지(투자한 달의 차이 합)도 함께 적는다.

실행: python scripts/32_탐색_쉰구간.py   (먼저 15번까지 실행해 두어야 함)
결과: 결과/탐색/쉰구간/쉰구간.csv, 붕괴달.csv, 1_두기간_쉰구간.png
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

프로젝트 = Path(__file__).resolve().parent.parent
처리폴더 = 프로젝트 / "처리데이터"
결과폴더 = 프로젝트 / "결과" / "탐색" / "쉰구간"


def 불러오기(이름, 파일):
    명세 = importlib.util.spec_from_file_location(이름, Path(__file__).resolve().parent / 파일)
    모듈 = importlib.util.module_from_spec(명세)
    명세.loader.exec_module(모듈)
    return 모듈


검증코드 = 불러오기("검증", "15_검증.py")
선택코드 = 검증코드.선택코드
plt = 검증코드.plt
구간들 = {"선택": 검증코드.선택구간, "검증": 검증코드.검증구간}
초록, 빨강 = "#2e9e6a", "#d64545"

# 발표 22장에 쓴 값(소수 첫째 자리). 같게 나오는지 확인한다.
발표값 = {"선택합": 68.7, "검증합": -25.7, "마지막구간": -38.0}


class 탐색문제(Exception):
    pass


def 확인(조건, 설명):
    if not 조건:
        raise 탐색문제(설명)


def 자료():
    with open(처리폴더 / "최종모델.json", encoding="utf-8") as f:
        최종 = json.load(f)["최종모델"]["이름"]
    확인(최종 == 검증코드.기대최종, f"최종모델.json의 모델이 고정한 모델과 다름: {최종}")
    월, 회전, 표, 단독, 보유달 = 검증코드.자료읽기()
    전 = 월["롱숏_%"]
    원 = 선택코드.비용후(pd.Series(1.0, index=보유달), 전, 회전)
    w, _ = 검증코드.비중만들기(검증코드.신호만들기(표, 단독, 최종), 보유달)
    r = 선택코드.비용후(w, 전, 회전)
    return 최종, w, r, 원, 전


def 쉰구간표(w, r, 원, 전):
    쉼 = w == 0
    기간 = pd.Series(np.where(w.index < 검증코드.검증구간[0], "선택", "검증"), index=w.index)
    새 = (쉼 != 쉼.shift()) | (기간 != 기간.shift())
    번호 = 새.cumsum()[쉼]
    줄들 = []
    for _, 달들 in 번호.groupby(번호).groups.items():
        달들 = list(달들)
        줄들.append({"기간": 기간[달들[0]], "시작": 달들[0], "끝": 달들[-1], "개월": len(달들),
                   "원래WML_누적_비용전_%": ((1 + 전.loc[달들] / 100).prod() - 1) * 100,
                   "모델_비용후합_%": r.loc[달들].sum(), "원래_비용후합_%": 원.loc[달들].sum(),
                   "차이_%p": r.loc[달들].sum() - 원.loc[달들].sum()})
    return pd.DataFrame(줄들)


def 붕괴달표(w, 전):
    줄들 = []
    for 이름, (a, b) in 구간들.items():
        _, 붕괴달 = 선택코드.사건정의(전.loc[a:b])
        for 달 in 붕괴달:
            다음 = str(pd.Period(달) + 1)
            줄들.append({"기간": 이름, "달": 달, "결과": "피함" if w.loc[달] == 0 else "놓침",
                       "다음달_쉼": bool(w.get(다음, np.nan) == 0)})
    return pd.DataFrame(줄들)


def 그림(w, r, 원, 전, 구간, 붕괴):
    결과폴더.mkdir(parents=True, exist_ok=True)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for a in (ax, ax2):
        a.axvline(pd.Timestamp(검증코드.검증구간[0] + "-01"), color=선택코드.보조글자, linestyle="--", linewidth=1)

    날 = 선택코드.월말(w.index)
    for 값, 색, 표지 in ((원, 선택코드.파랑, "원래 WML"), (r, 선택코드.주황, "최종 모델")):
        누 = (1 + 값 / 100).cumprod()
        ax.plot(날, 누.values, color=색, linewidth=1.8,
                label=f"{표지}  2015년 말 {누.loc['2015-12']:.2f}배 → 끝 {누.iloc[-1]:.2f}배")
    ax.set_yscale("log")
    ax.set_ylim(0.6, 3.6)
    ax.yaxis.set_major_locator(FixedLocator([0.75, 1, 1.5, 2, 3]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}배"))
    ax.text(pd.Timestamp("2010-10-01"), 0.08, "선택 기간", transform=ax.get_xaxis_transform(), ha="center",
            color=선택코드.보조글자, fontsize=10)
    ax.text(pd.Timestamp("2017-10-01"), 0.08, "검증 기간", transform=ax.get_xaxis_transform(), ha="center",
            color=선택코드.보조글자, fontsize=10)

    앞끝, 내림 = None, False
    for _, g in 구간.iterrows():
        p0, p1 = pd.Period(g["시작"]), pd.Period(g["끝"])
        색 = 초록 if g["차이_%p"] > 0 else 빨강
        for a in (ax, ax2):
            a.axvspan(p0.start_time, p1.end_time, color=선택코드.쉼색, linewidth=0, zorder=0)
        x = p0.start_time + (p1.end_time - p0.start_time) / 2
        내림 = 앞끝 is not None and (p0 - 앞끝).n <= 4 and not 내림   # 가까운 구간은 숫자를 번갈아 내린다
        앞끝 = p1
        ax.text(x, 0.97 - (0.08 if 내림 else 0), f"{g['원래WML_누적_비용전_%']:+.0f}%", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8.5, color=색)
        ax2.bar(p0.start_time, g["차이_%p"], width=(p1.end_time - p0.start_time).days, align="edge", color=색, linewidth=0)
        if abs(g["차이_%p"]) >= 5:
            위 = g["차이_%p"] > 0
            ax2.text(x, g["차이_%p"] + (1.5 if 위 else -1.5), f"{g['차이_%p']:+.0f}%p", ha="center",
                     va="bottom" if 위 else "top", fontsize=8.5, color=색)

    for _, g in 붕괴.iterrows():
        p = pd.Period(g["달"])
        ax.plot([p.start_time + (p.end_time - p.start_time) / 2], [0.025], marker="v", markersize=7, linestyle="none",
                color=초록 if g["결과"] == "피함" else 빨강, transform=ax.get_xaxis_transform(), clip_on=False, zorder=5)
    h, l = ax.get_legend_handles_labels()
    h += [Line2D([], [], marker="v", color=초록, linestyle="none", markersize=7),
          Line2D([], [], marker="v", color=빨강, linestyle="none", markersize=7)]
    l += ["붕괴 달: 피함 (쉬는 중)", "붕괴 달: 놓침 (투자 중)"]
    ax.legend(h, l, loc="upper left", frameon=False, fontsize=9, bbox_to_anchor=(0, 0.83))
    선택코드.꾸미기(ax, "두 기간의 쉰 구간: 원래 WML 수익과 모델의 차이",
                 "최종 모델(시장 변동성 + 모멘텀 변동성). 회색 = 쉰 구간, 위 숫자 = 그동안 원래 WML 누적수익(비용 전). "
                 "▼ = 기간별 하위 5% 붕괴 달.")

    선택코드.꾸미기(ax2, "", "")
    ax2.axhline(0, color=선택코드.기준선, linewidth=1)
    ax2.set_ylim(-52, 48)
    ax2.yaxis.set_major_locator(FixedLocator([-40, -20, 0, 20, 40]))
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%p"))
    ax2.text(0.0, 1.02, "쉰 구간별 모델 - 원래 WML (비용 후 수익 합)", transform=ax2.transAxes, fontsize=9.5,
             color=선택코드.보조글자)
    합 = 구간.groupby("기간")["차이_%p"].sum()
    ax2.text(0.24, 0.06, f"선택 기간 합 {합['선택']:+.0f}%p", transform=ax2.transAxes, fontsize=9.5, color=초록, ha="center")
    ax2.text(0.62, 0.06, f"검증 기간 합 {합['검증']:+.0f}%p", transform=ax2.transAxes, fontsize=9.5, color=빨강, ha="center")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.05, hspace=0.12)
    fig.savefig(결과폴더 / "1_두기간_쉰구간.png", dpi=200)
    plt.close(fig)


def 실행():
    최종, w, r, 원, 전 = 자료()
    구간 = 쉰구간표(w, r, 원, 전)
    붕괴 = 붕괴달표(w, 전)

    # 확인 1: 쉰 달 수와 붕괴 포착 수가 15번 평가 함수와 같다
    for 이름, (a, b) in 구간들.items():
        평 = 검증코드.평가(r, 원, 전, w, (a, b))
        확인(구간.loc[구간["기간"] == 이름, "개월"].sum() == 평["켜진달"], f"{이름}: 쉰 달 수가 15번과 다름")
        확인((붕괴[(붕괴["기간"] == 이름) & (붕괴["결과"] == "피함")]).shape[0] == 평["붕괴포착"], f"{이름}: 붕괴 포착이 15번과 다름")

    # 확인 2: 쉰 구간 차이 합 + 투자한 달의 차이 합 = 전체 차이
    print("[기간별 차이] (모델 비용 후 - 원래 WML 비용 후, 수익 합 %p)")
    for 이름, (a, b) in 구간들.items():
        전체 = (r - 원).loc[a:b].sum()
        쉰합 = 구간.loc[구간["기간"] == 이름, "차이_%p"].sum()
        투자합 = (r - 원).loc[a:b][w.loc[a:b] == 1].sum()
        확인(np.isclose(쉰합 + 투자합, 전체), f"{이름}: 쉰 구간 + 투자한 달 차이가 전체와 다름")
        print(f"  {이름}: 쉰 구간 {쉰합:+.1f}, 투자한 달(구축·청산 비용) {투자합:+.1f}, 전체 {전체:+.1f}")

    # 확인 3: 발표 22장 숫자
    합 = 구간.groupby("기간")["차이_%p"].sum()
    마지막 = 구간.iloc[-1]
    for 이름, 값 in (("선택합", 합["선택"]), ("검증합", 합["검증"]), ("마지막구간", 마지막["차이_%p"])):
        확인(round(값, 1) == 발표값[이름], f"{이름}이 발표 값과 다름: {값:.2f} 대 {발표값[이름]}")
    print(f"[발표 22장 대조] 선택 {합['선택']:+.1f}, 검증 {합['검증']:+.1f}, "
          f"마지막 구간({마지막['시작']}부터 {마지막['끝']}까지, {마지막['개월']}개월) {마지막['차이_%p']:+.1f}: 같음")

    print("\n[쉰 구간]")
    print(구간.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    print("\n[붕괴 달]")
    print(붕괴.to_string(index=False))

    결과폴더.mkdir(parents=True, exist_ok=True)
    구간.to_csv(결과폴더 / "쉰구간.csv", index=False, encoding="utf-8-sig")
    붕괴.to_csv(결과폴더 / "붕괴달.csv", index=False, encoding="utf-8-sig")
    그림(w, r, 원, 전, 구간, 붕괴)
    print(f"\n[저장] 쉰구간.csv, 붕괴달.csv, 그림 1개: {결과폴더}")


if __name__ == "__main__":
    실행()
