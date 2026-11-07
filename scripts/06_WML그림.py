# -*- coding: utf-8 -*-
"""
6단계: WML 결과 그림 3개를 그린다. 수익률을 새로 계산하지 않고 5단계 결과 표를 읽어 그린다.

그림
- 1_2x3_FnGuide_누적수익률.png: 직접 만든 2x3과 FnGuide 모멘텀 지수의 비교
- 2_방식별_누적수익률.png: 4가지 방식의 누적수익률
- 3_2x3_월별수익률.png: 2x3의 월별 수익률과 최악의 달 5개

누적수익률은 매달 전액을 다시 투자한다고 가정하고, 거래비용은 빼지 않는다.
그림 3의 "최악의 달"은 월수익률이 가장 낮은 5개 달이며, 8절의 붕괴 기준과는 다르다.

실행: python scripts/06_WML그림.py   (먼저 scripts/05_WML계산.py를 실행해 두어야 함)
결과: 결과/WML/*.png
"""
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 창을 띄우지 않고 파일로만 저장한다.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

프로젝트 = Path(__file__).resolve().parent.parent
처리폴더 = 프로젝트 / "처리데이터"
저장폴더 = 프로젝트 / "결과" / "WML"

# NW t 계산은 05번의 함수를 그대로 쓴다. 파일 이름이 숫자로 시작해 경로로 불러온다.
_명세 = importlib.util.spec_from_file_location("WML계산", Path(__file__).resolve().parent / "05_WML계산.py")
월별코드 = importlib.util.module_from_spec(_명세)
_명세.loader.exec_module(월별코드)

시작, 끝 = "2000-01", "2026-09"
방식순서 = ["2x3", "5x5", "5분위", "10분위"]
방식이름 = {"2x3": "2×3 (기본)", "5x5": "5×5", "5분위": "5분위", "10분위": "10분위"}

# 색은 scripts 밖의 검증 도구로 확인한 순서대로 쓴다. 선 색은 대상(방식)을 따라간다.
파랑, 주황, 청록, 노랑 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
방식색 = dict(zip(방식순서, [파랑, 주황, 청록, 노랑]))
바탕, 글자, 보조글자, 약한글자 = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
격자, 기준선, 회색막대 = "#e1e0d9", "#c3c2b7", "#9ec5f4"

plt.rcParams.update({
    "font.family": "Malgun Gothic",  # 한글 표시용 글꼴
    "axes.unicode_minus": False,     # 마이너스 기호가 깨지지 않게 한다.
    "figure.facecolor": 바탕, "axes.facecolor": 바탕, "savefig.facecolor": 바탕,
    "text.color": 글자, "axes.labelcolor": 보조글자,
    "xtick.color": 약한글자, "ytick.color": 약한글자,
    "xtick.labelcolor": 보조글자, "ytick.labelcolor": 보조글자,
    "font.size": 11,
})


def 월말날짜(연월):
    return pd.PeriodIndex(연월, freq="M").to_timestamp(how="end").normalize()


def 누적(수익률):
    """월 수익률(%)을 시작값 1의 누적수익률로 바꾼다. 첫 달 직전 달 말을 1로 둔다."""
    값 = (1 + 수익률 / 100).cumprod()
    첫날 = (pd.Period(수익률.index[0], "M") - 1).strftime("%Y-%m")
    return pd.concat([pd.Series([1.0], index=[첫날]), 값])


def 틀꾸미기(ax, 제목, 부제):
    """축과 격자를 옅게 하고, 제목과 설명을 왼쪽 위에 쓴다."""
    for 면 in ["top", "right", "left"]:
        ax.spines[면].set_visible(False)
    ax.spines["bottom"].set_color(기준선)
    ax.grid(axis="y", color=격자, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", color=기준선)
    ax.set_title(제목, loc="left", fontsize=15, fontweight="bold", pad=34, color=글자)
    ax.text(0, 1.03, 부제, transform=ax.transAxes, fontsize=10.5, color=보조글자, va="bottom")


def 로그축(ax, 눈금):
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator(눈금))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}배"))


def 끝값표시(ax, 선들, 최소간격=0.075):
    """선 끝에 이름과 배수를 쓴다. 글자가 겹치면 로그 눈금에서 위아래로 벌린다."""
    줄 = sorted(선들, key=lambda x: x[1])
    위치 = [np.log10(값) for _, 값 in 줄]
    for i in range(1, len(위치)):
        위치[i] = max(위치[i], 위치[i - 1] + 최소간격)
    끝날 = pd.Timestamp("2026-09-30")
    for (이름, 값), y in zip(줄, 위치):
        ax.annotate(f"{이름}  {값:.1f}배", xy=(끝날, 10 ** y), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=10.5, color=글자, annotation_clip=False)


def 그림1(월별, 지수):
    """직접 만든 2x3과 FnGuide의 누적수익률을 같은 로그 축에 그린다."""
    내것 = 월별[월별["방식"] == "2x3"].set_index("연월")["롱숏_%"]
    fg = (지수["3FM.2M3.MOM"].pct_change(fill_method=None) * 100).reindex(내것.index)
    선들 = {"직접 계산 2×3": (누적(내것), 파랑), "FnGuide 3FM.2M3.MOM": (누적(fg), 주황)}

    fig, ax = plt.subplots(figsize=(11, 5.8))
    for 이름, (값, 색) in 선들.items():
        ax.plot(월말날짜(값.index), 값.values, color=색, linewidth=2, label=이름)
    로그축(ax, [0.5, 1, 2, 5, 10, 20])
    틀꾸미기(ax, "직접 만든 WML은 FnGuide 지수와 방향이 비슷하고 최종 누적 수익은 조금 더 높다",
         f"시작값 1, 로그 눈금, 거래비용 미반영. 월 수익률 상관 {내것.corr(fg):.2f}.")
    ax.legend(loc="upper left", frameon=False, ncol=2)
    끝값표시(ax, [(이름.split(" ")[0] if "FnGuide" in 이름 else "직접 계산", 값.iloc[-1]) for 이름, (값, _) in 선들.items()])
    fig.subplots_adjust(left=0.07, right=0.86, top=0.84, bottom=0.08)
    fig.savefig(저장폴더 / "1_2x3_FnGuide_누적수익률.png", dpi=200)
    plt.close(fig)


def 그림2(월별):
    """4가지 방식의 누적수익률을 같은 로그 축에 그린다."""
    fig, ax = plt.subplots(figsize=(11, 5.8))
    끝값들 = []
    for 방식 in 방식순서:
        값 = 누적(월별[월별["방식"] == 방식].set_index("연월")["롱숏_%"])
        ax.plot(월말날짜(값.index), 값.values, color=방식색[방식], linewidth=2, label=방식이름[방식])
        끝값들.append((방식이름[방식], 값.iloc[-1]))
    로그축(ax, [0.5, 1, 2, 5, 10, 20, 50])
    틀꾸미기(ax, "극단 종목에 집중할수록 낙폭이 커지고, 누적 수익 순위는 시점에 따라 바뀐다",
         "방식별 롱숏 누적수익률. 시작값 1, 로그 눈금, 거래비용 미반영. 모두 시가총액가중.")
    ax.legend(loc="upper left", frameon=False, ncol=4)
    끝값표시(ax, 끝값들)
    fig.subplots_adjust(left=0.07, right=0.86, top=0.84, bottom=0.08)
    fig.savefig(저장폴더 / "2_방식별_누적수익률.png", dpi=200)
    plt.close(fig)


def 그림3(월별):
    """2x3의 월별 수익률 막대를 그리고 최악의 달 5개를 강조한다."""
    값 = 월별[월별["방식"] == "2x3"].set_index("연월")["롱숏_%"]
    최악 = 값.nsmallest(5)
    날짜 = 월말날짜(값.index)
    색 = [주황 if 연월 in 최악.index else 회색막대 for 연월 in 값.index]

    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.bar(날짜, 값.values, width=24, color=색, linewidth=0)
    ax.axhline(0, color=기준선, linewidth=1)
    # 막대가 잘리지 않도록 세로축을 데이터의 최댓값에 맞춘다.
    ax.set_ylim(min(-24, 값.min() - 5), 값.max() + 2)
    ax.set_xlim(pd.Timestamp("1999-06-30"), pd.Timestamp("2027-03-31"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    틀꾸미기(ax, "2×3 WML의 큰 손실은 일부 달에 집중된다",
         "월별 롱숏 수익률, 2000-01부터 2026-09까지. 주황색은 수익률이 가장 낮은 5개 달.")

    # 가까운 달의 글자가 겹치지 않게 글자 높이를 달마다 지정한다.
    글자높이 = {"2001-01": -2.2, "2000-09": 0.0, "2009-04": -1.2, "2009-01": -3.2, "2026-07": -1.2}
    가로정렬 = {"2001-01": "left", "2000-09": "left", "2009-04": "left", "2009-01": "right", "2026-07": "right"}
    가로이동 = {"2000-09": 200}  # 2000-09 글자는 2001-01 막대 오른쪽으로 옮긴다(일 단위).
    for 연월, 수익 in 최악.items():
        x = 월말날짜([연월])[0]
        글자x = x + pd.Timedelta(days=가로이동.get(연월, 0))
        ax.annotate(f"{연월}  {수익:.1f}%", xy=(x, 수익), xytext=(글자x, 수익 + 글자높이.get(연월, -1.2)),
                    fontsize=10, color=글자, ha=가로정렬.get(연월, "left"), va="center")
    fig.subplots_adjust(left=0.06, right=0.98, top=0.83, bottom=0.08)
    fig.savefig(저장폴더 / "3_2x3_월별수익률.png", dpi=200)
    plt.close(fig)


def 실행():
    저장폴더.mkdir(parents=True, exist_ok=True)
    월별 = pd.read_parquet(처리폴더 / "WML월별수익률.parquet")
    월별 = 월별[월별["연월"].between(시작, 끝)]
    지수 = pd.read_parquet(처리폴더 / "정리표" / "08_FF팩터지수.parquet").set_index("연월")

    그림1(월별, 지수)
    그림2(월별)
    그림3(월별)

    # 그림에 쓴 값을 화면에도 남겨, 문서의 표와 대조할 수 있게 한다.
    print("[끝값] 시작값 1 기준 누적 배수")
    for 방식 in 방식순서:
        값 = 누적(월별[월별["방식"] == 방식].set_index("연월")["롱숏_%"])
        r = 월별[월별["방식"] == 방식].set_index("연월")["롱숏_%"]
        앞 = r.loc[:"2025-12"]
        print(f"  {방식}: {값.iloc[-1]:.2f}배, 고점 {값.max():.2f}배({값.idxmax()}), 2025-12 {값.loc['2025-12']:.2f}배, "
              f"2026년 {값.iloc[-1] / 값.loc['2025-12']:.2f}배, 2025-12까지 월평균 {앞.mean():.2f}% (NW t {월별코드.뉴이웨스트t(앞):.2f}), "
              f"최고의 달 {[(k, round(v, 1)) for k, v in r.nlargest(4).items()]}")
    fg = (지수["3FM.2M3.MOM"].pct_change(fill_method=None) * 100).reindex(월별[월별["방식"] == "2x3"]["연월"].values)
    print(f"  FnGuide 3FM.2M3.MOM: {누적(fg).iloc[-1]:.2f}배")
    print(f"저장 완료: {저장폴더}")


if __name__ == "__main__":
    실행()
