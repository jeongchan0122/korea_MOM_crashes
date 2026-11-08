# -*- coding: utf-8 -*-
"""
15단계: 선택 기간에 고른 최종 모델을 검증 기간(보유 달 2016-01부터 2026-09까지, 129개월)에 한 번 적용한다.

왜 필요한가
- 모델을 고를 때 쓰지 않은 기간에서 그대로 시험해야 선택 기간의 운을 걸러 낼 수 있다(OOS).

무엇을 하는가 (연구 기록 17절, 팀 문서 8.6)
- 최종모델.json의 모델을 그대로 쓴다. 신호는 13번, 비용과 성과는 14번의 함수를 그대로 부른다.
- 모든 후보의 비중은 2005-07부터 2026-09까지 한 번에 만들어, 2016-01의 비용이 2015-12의 실제 비중에서 이어지게 한다.
- 선택 기간 부분을 다시 계산해 14번의 후보 57개 t값과 같은지 확인한다(같은 계산을 쓰고 있다는 확인).
- 결론: 1단계 조건 3개(샤프, 최대낙폭, 붕괴 사건 손실)를 모두 충족하고 스패닝 알파 NW t >= 1.96이면 "개선",
  조건은 충족하지만 t < 1.96이면 "방향만 맞음", 조건 중 하나라도 못 맞추면 "실패". 결론은 최종 모델로만 낸다.
- 보고: 2025-12까지 끊은 결과, 최고 단독·최고 섞은 모델, 후보 57개 전체, 사건표, 그림 4개.

하지 않는 일
- 결과를 본 뒤 모델, 규칙, 비용, 기간 바꾸기. FF3 알파 등 보고 항목과 강건성은 다음 단계에서 한다.

실행: python scripts/15_검증.py   (먼저 11번부터 14번까지 실행해 두어야 함)
결과: 결과/검증/(검증후보표·사건표 csv, 그림 png)
"""
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

프로젝트 = Path(__file__).resolve().parent.parent
처리폴더 = 프로젝트 / "처리데이터"
결과폴더 = 프로젝트 / "결과" / "검증"


def 불러오기(이름, 파일):
    명세 = importlib.util.spec_from_file_location(이름, Path(__file__).resolve().parent / 파일)
    모듈 = importlib.util.module_from_spec(명세)
    명세.loader.exec_module(모듈)
    return 모듈


신호코드 = 불러오기("신호", "13_신호만들기.py")
선택코드 = 불러오기("선택", "14_모델선택.py")

전체시작, 전체끝 = "2005-07", "2026-09"
선택구간, 검증구간, 끊은구간 = ("2005-07", "2015-12"), ("2016-01", "2026-09"), ("2016-01", "2025-12")
기대최종 = "점수:시장변동성+모멘텀변동성"   # 태그 모델선택-2026-11-08의 최종 모델
유의기준 = 1.96
점검달들 = ["2018-10", "2020-03", "2026-06"]


class 검증문제(Exception):
    """확인에 실패했을 때 쓰는 오류. 이 오류가 나면 저장하지 않고 멈춘다."""


def 확인(조건, 설명):
    if not 조건:
        raise 검증문제(설명)


# ---------------------------------------------------------------- 자료와 비중

def 자료읽기():
    월 = pd.read_parquet(처리폴더 / "WML월별수익률.parquet")
    월 = 월[월["방식"] == "2x3"].set_index("연월")[["롱숏_%", "롱온리_%", "패자쪽_%"]].loc[전체시작:전체끝]
    회전 = pd.read_parquet(처리폴더 / "WML회전율.parquet").set_index("연월").loc[전체시작:전체끝]
    표 = 신호코드.지표읽기()
    단독 = pd.read_parquet(처리폴더 / "신호월별.parquet").set_index("연월").drop(columns="적용연월")
    보유달 = list(pd.period_range(전체시작, 전체끝, freq="M").astype(str))
    확인(list(월.index) == 보유달 and list(회전.index) == 보유달, "WML이나 회전율의 달이 맞지 않음")
    return 월, 회전, 표, 단독, 보유달


def 신호만들기(표, 단독, 이름):
    종류, 지표 = 이름.split(":")
    지표들 = tuple(지표.split("+"))
    if 종류 == "단독":
        return 단독[지표들[0]]
    return 신호코드.조합신호(표, 단독, 지표들, "과반" if 종류 == "과반" else "점수")


def 비중만들기(신호, 보유달):
    """보유 달 m의 비중 = 1 - (m-1월 말 신호). 판단할 수 없는 달은 꺼짐(비중 1)으로 채우고 구간별 수를 센다."""
    s = 신호.reindex([str(pd.Period(m) - 1) for m in 보유달])
    s.index = 보유달
    채움 = {"선택": int(s.loc[선택구간[0]:선택구간[1]].isna().sum()), "검증": int(s.loc[검증구간[0]:검증구간[1]].isna().sum())}
    return 1 - s.fillna(0), 채움


# ---------------------------------------------------------------- 평가

def 평가(r, 원래, 원래비용전, w, 구간):
    """한 구간에서 전략 r(비용 후)을 원래 WML(비용 후)과 비교한다. 붕괴 사건은 그 구간의 비용 전 원래 WML로 정한다."""
    a, b = 구간
    r, 원, 전, w = r.loc[a:b], 원래.loc[a:b], 원래비용전.loc[a:b], w.loc[a:b]
    사건, 붕괴달 = 선택코드.사건정의(전)
    전략사건, 원래사건 = 선택코드.사건수익(r, 사건), 선택코드.사건수익(원, 사건)
    알파, 기울기, t = 선택코드.알파t(r.values, 원.values)
    샤프 = r.mean() / r.std() * np.sqrt(12)
    원샤프 = 원.mean() / 원.std() * np.sqrt(12)
    낙, 원낙 = 선택코드.낙폭(r).min(), 선택코드.낙폭(원).min()
    쉼 = w == 0
    포착 = int((쉼 & w.index.isin(붕괴달)).sum())
    조건 = {"샤프통과": 샤프 > 원샤프, "낙폭통과": 낙 >= 원낙, "사건통과": sum(전략사건) > sum(원래사건)}
    t값 = float(t[0])
    if not all(조건.values()):
        결론 = "실패"
    elif t값 >= 유의기준:
        결론 = "개선"
    else:
        결론 = "방향만 맞음"
    return {"개월": len(r), "켜진달": int(쉼.sum()), "붕괴포착": 포착, "정확도": 포착 / 쉼.sum() if 쉼.sum() else np.nan,
            "월평균": r.mean(), "샤프": 샤프, "최대낙폭": 낙, "사건합": sum(전략사건), "알파": float(알파[0]),
            "b": float(기울기[0]), "t": t값, **조건, "결론": 결론,
            "원래_월평균": 원.mean(), "원래_샤프": 원샤프, "원래_최대낙폭": 원낙, "원래_사건합": sum(원래사건),
            "_사건": 사건, "_붕괴달": 붕괴달, "_전략사건": 전략사건, "_원래사건": 원래사건}


# ---------------------------------------------------------------- 점검

def 손계산(r, 원래, w, 회전, 월):
    """최종 모델의 검증 기간 비용 후 수익률과 알파 t를 반복문과 행렬 식으로 따로 계산한다."""
    a, b = 검증구간
    달들 = list(r.loc[a:b].index)
    앞 = w.loc[str(pd.Period(a) - 1)]
    직접 = []
    for 달 in 달들:
        지금 = w.loc[달]
        if 앞 == 1 and 지금 == 1:
            비용 = 회전.loc[달, "리밸런싱"]
        elif 앞 == 1 and 지금 == 0:
            비용 = 회전.loc[달, "청산"]
        elif 앞 == 0 and 지금 == 1:
            비용 = 회전.loc[달, "구축"]
        else:
            비용 = 0.0
        직접.append(지금 * 월.loc[달, "롱숏_%"] - 0.5 * 비용)
        앞 = 지금
    직접 = np.array(직접)
    확인(np.allclose(직접, r.loc[a:b].values, atol=1e-12), "검증 기간 비용 후 수익률 손계산이 다름")
    x = 원래.loc[a:b].values
    X = np.column_stack([np.ones(len(x)), x])
    β = np.linalg.lstsq(X, 직접, rcond=None)[0]
    e = 직접 - X @ β
    S = sum(np.outer(X[t] * e[t], X[t] * e[t]) for t in range(len(x)))
    for L in range(1, 선택코드.NW시차 + 1):
        Γ = sum(np.outer(X[t] * e[t], X[t - L] * e[t - L]) for t in range(L, len(x)))
        S += (1 - L / (선택코드.NW시차 + 1)) * (Γ + Γ.T)
    V = np.linalg.inv(X.T @ X) @ S @ np.linalg.inv(X.T @ X)
    return β[0] / np.sqrt(V[0, 0])


def 미래정보점검(표, 단독, 최종이름):
    for 달 in 점검달들:
        잘린표 = 표.loc[:달]
        잘린단독 = 신호코드.단독신호(잘린표)
        a = 신호만들기(표, 단독, 최종이름).loc[달]
        b = 신호만들기(잘린표, 잘린단독, 최종이름).loc[달]
        확인((np.isnan(a) and np.isnan(b)) or a == b, f"{달}: 미래 지표를 지우면 최종 모델 신호가 달라짐")
    print(f"[미래 정보 점검] {', '.join(점검달들)}: 그달 말 이후 지표를 지워도 최종 모델 신호가 같음")


# ---------------------------------------------------------------- 그림 (14번과 같은 형식, 검증 기간)

파랑, 주황 = 선택코드.파랑, 선택코드.주황


def 사건표시(ax, 사건들, y위치):
    """14번과 같지만, 6개월 안에 붙은 사건은 글자를 위아래로 번갈아 놓아 겹치지 않게 한다."""
    앞끝, 높이 = None, 0
    for i, (a, b) in enumerate(사건들, 1):
        시작, 끝 = pd.Period(a).start_time, pd.Period(b).end_time
        높이 = 0.06 if 앞끝 is not None and (pd.Period(a) - 앞끝).n < 6 and 높이 == 0 else 0
        y = y위치 - 높이
        ax.annotate("", xy=(시작, y), xytext=(끝, y), xycoords=("data", "axes fraction"),
                    arrowprops=dict(arrowstyle="-", color=선택코드.보조글자, linewidth=1.2))
        ax.text(시작 + (끝 - 시작) / 2, y + 0.02, f"사건{i}", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=9.5, color=선택코드.보조글자)
        앞끝 = pd.Period(b)


def 그림들(최종, 결과, 원래, w, r, 표, 단독):
    결과폴더.mkdir(parents=True, exist_ok=True)
    a, b = 검증구간
    w, r, 원 = w.loc[a:b], r.loc[a:b], 원래.loc[a:b]
    날 = 선택코드.월말(w.index)
    이름 = 최종
    꾸미기, 쉼음영 = 선택코드.꾸미기, 선택코드.쉼음영

    fig, ax = plt.subplots(figsize=(11, 5.8))
    쉼음영(ax, w)
    for 값, 색, 표지 in ((원, 파랑, "원래 WML"), (r, 주황, "최종 모델")):
        누 = (1 + 값 / 100).cumprod()
        ax.plot(날, 누.values, color=색, linewidth=2, label=f"{표지}  {누.iloc[-1]:.2f}배")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([0.5, 0.75, 1, 1.5, 2, 3]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}배"))
    사건표시(ax, 결과["_사건"], 0.93)
    꾸미기(ax, "검증 기간 누적수익: 원래 WML과 최종 모델",
         f"{이름}. 시작값 1, 로그 눈금, 거래비용(편도 0.5%) 차감. 회색 음영은 신호가 켜져 쉰 달.")
    ax.legend(loc="lower right", frameon=False, ncol=2)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.84, bottom=0.08)
    fig.savefig(결과폴더 / "1_누적수익.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.0))
    쉼음영(ax, w)
    for 값, 색, 표지 in ((원, 파랑, "원래 WML"), (r, 주황, "최종 모델")):
        d = 선택코드.낙폭(값)
        ax.plot(날, d.values, color=색, linewidth=2, label=f"{표지}  최대 {d.min():.1f}%")
    ax.axhline(0, color=선택코드.기준선, linewidth=1)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    꾸미기(ax, "검증 기간 낙폭: 고점 대비 하락", f"{이름}. 거래비용 차감. 회색 음영은 쉰 달.")
    ax.legend(loc="lower right", frameon=False, ncol=2)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.82, bottom=0.08)
    fig.savefig(결과폴더 / "2_낙폭.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.0))
    x = np.arange(len(결과["_사건"]))
    for 값들, 색, 표지, 이동 in ((결과["_원래사건"], 파랑, "원래 WML", -0.2), (결과["_전략사건"], 주황, "최종 모델", 0.2)):
        ax.bar(x + 이동, 값들, width=0.38, color=색, label=표지, linewidth=0)
        for xi, v in zip(x + 이동, 값들):
            ax.text(xi, v - 0.6 if v < 0 else v + 0.3, f"{v:.1f}%", ha="center", va="top" if v < 0 else "bottom",
                    fontsize=10, color=선택코드.글자)
    ax.axhline(0, color=선택코드.기준선, linewidth=1)
    ax.set_xticks(x, [f"사건{i + 1}\n{a1}부터 {b1}까지" if a1 != b1 else f"사건{i + 1}\n{a1}"
                      for i, (a1, b1) in enumerate(결과["_사건"])])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.set_ylim(min(min(결과["_원래사건"]), min(결과["_전략사건"])) - 4, max(2, max(결과["_전략사건"]) + 2))
    꾸미기(ax, "검증 기간 붕괴 사건별 누적수익률", f"{이름}. 사건 = 원래 WML 하위 5% 달을 3개월 이내끼리 묶은 구간. 거래비용 차감.")
    ax.legend(loc="lower left", frameon=False)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.82, bottom=0.14)
    fig.savefig(결과폴더 / "3_붕괴사건.png", dpi=200)
    plt.close(fig)

    # 4. 지표와 경계 (최종 모델은 위험 점수)
    지표달 = list(pd.period_range(str(pd.Period(a) - 1), str(pd.Period(b) - 1), freq="M").astype(str))
    날4 = 선택코드.월말(지표달)
    지표들 = 이름.split(":")[1].split("+")
    점수 = 신호코드.위험점수(표, 지표들)
    경계 = 선택코드.기준선값(점수, 신호코드.점수창최소)
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.plot(날4, 점수.reindex(지표달).values, color=파랑, linewidth=2, label="겹침 조정 위험 점수")
    ax.plot(날4, 경계.reindex(지표달).values, color=선택코드.약한글자, linewidth=1.5, linestyle="--", label="켜지는 경계(상위 20%)")
    for 달 in 결과["_붕괴달"]:
        p = pd.Period(달) - 1
        ax.axvspan(p.start_time, p.end_time, color="#f6c9b5", linewidth=0, zorder=0)
    꾸미기(ax, "검증 기간 최종 모델의 지표와 판단 경계", f"{이름} (높을수록 위험). 연한 주황 = 붕괴 달의 비중을 정한 지표 달(붕괴 한 달 전).")
    ax.legend(loc="upper left", frameon=False, ncol=2)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.82, bottom=0.08)
    fig.savefig(결과폴더 / "4_지표와기준선.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- 실행

def 실행():
    with open(처리폴더 / "최종모델.json", encoding="utf-8") as f:
        기록 = json.load(f)
    최종 = 기록["최종모델"]["이름"]
    확인(최종 == 기대최종, f"최종모델.json의 모델이 고정한 모델과 다름: {최종}")
    후보목록 = pd.read_csv(프로젝트 / "결과" / "모델선택" / "후보표.csv")
    확인(len(후보목록) == 57, "후보가 57개가 아님")

    월, 회전, 표, 단독, 보유달 = 자료읽기()
    원래비용전 = 월["롱숏_%"]
    원래 = 선택코드.비용후(pd.Series(1.0, index=보유달), 원래비용전, 회전)

    줄들, 보관 = [], {}
    for _, c in 후보목록.iterrows():
        이름 = c["이름"]
        w, 채움 = 비중만들기(신호만들기(표, 단독, 이름), 보유달)
        r = 선택코드.비용후(w, 원래비용전, 회전)
        선 = 평가(r, 원래, 원래비용전, w, 선택구간)
        확인(abs(선["t"] - c["t"]) < 1e-9, f"{이름}: 선택 기간 t가 14번과 다름 ({선['t']} 대 {c['t']})")
        검 = 평가(r, 원래, 원래비용전, w, 검증구간)
        끊 = 평가(r, 원래, 원래비용전, w, 끊은구간)
        보관[이름] = (w, r, 검, 끊)
        줄들.append({"이름": 이름, "종류": c["종류"], "지표수": c["지표수"], "선택_t": c["t"], "선택_심사통과": c["심사통과"],
                   "선택_거르기통과": c["거르기통과"], "검증_채운달": 채움["검증"],
                   **{k: v for k, v in 검.items() if not k.startswith(("_", "원래_"))},
                   "끊은_t": 끊["t"], "끊은_결론": 끊["결론"]})
    print(f"[선택 기간 재현] 후보 57개의 선택 기간 t값이 14번과 모두 같음(차이 1e-9 미만)")

    표결과 = pd.DataFrame(줄들)
    w, r, 검, 끊 = 보관[최종]
    t손 = 손계산(r, 원래, w, 회전, 월)
    확인(np.isclose(t손, 검["t"], rtol=1e-8), "검증 기간 알파 t 손계산이 다름")
    print(f"[손계산 대조] 최종 모델 검증 기간: 비용 후 수익률 129개월 같음, 알파 t 코드 {검['t']:.10f} 손 {t손:.10f}")
    확인(검["개월"] == 129, "검증 기간이 129개월이 아님")
    확인(표결과.loc[표결과["이름"] == 최종, "검증_채운달"].iloc[0] == 0, "최종 모델의 검증 기간에 빈 신호가 있음")
    미래정보점검(표, 단독, 최종)

    print(f"\n[원래 WML, 검증 기간 129개월] 비용 전 월평균 {원래비용전.loc[검증구간[0]:검증구간[1]].mean():.3f}%, "
          f"비용 후 {검['원래_월평균']:.3f}%, 샤프(비용 후) {검['원래_샤프']:.3f}, 최대낙폭 {검['원래_최대낙폭']:.1f}%")
    print(f"  붕괴 달 {검['_붕괴달']}, 사건 {검['_사건']}")

    print(f"\n========== 검증 결론: {최종} ==========")
    print(f"  {검['결론']}")
    print(f"  월평균 {검['월평균']:.3f}% (원래 {검['원래_월평균']:.3f}%), 샤프 {검['샤프']:.3f} (원래 {검['원래_샤프']:.3f}) -> {검['샤프통과']}")
    print(f"  최대낙폭 {검['최대낙폭']:.1f}% (원래 {검['원래_최대낙폭']:.1f}%) -> {검['낙폭통과']}")
    print(f"  붕괴 사건 합 {검['사건합']:.1f}% (원래 {검['원래_사건합']:.1f}%) -> {검['사건통과']}")
    print(f"  스패닝 알파 월 {검['알파']:.3f}%, b {검['b']:.3f}, NW t {검['t']:.3f} (기준 {유의기준})")
    print(f"  켜진 달 {검['켜진달']}, 붕괴 포착 {검['붕괴포착']}/{len(검['_붕괴달'])}, 정확도 {검['정확도']:.2f}")
    print(f"  쉰 달: {list(w.loc[검증구간[0]:검증구간[1]].index[w.loc[검증구간[0]:검증구간[1]] == 0])}")

    print(f"\n[2025-12까지 끊은 결과 (120개월)] 결론 {끊['결론']}, 월평균 {끊['월평균']:.3f}% (원래 {끊['원래_월평균']:.3f}%), "
          f"샤프 {끊['샤프']:.3f} (원래 {끊['원래_샤프']:.3f}), 최대낙폭 {끊['최대낙폭']:.1f}% (원래 {끊['원래_최대낙폭']:.1f}%), "
          f"사건 합 {끊['사건합']:.1f}% (원래 {끊['원래_사건합']:.1f}%), t {끊['t']:.3f}")

    # 사건표: 원래 WML(비용 전)의 승자 쪽, 패자 쪽과 최종 모델
    사건줄 = []
    for (a, b), 원사, 전사 in zip(검["_사건"], 검["_원래사건"], 검["_전략사건"]):
        구 = 월.loc[a:b]
        사건줄.append({"사건": f"{a}부터 {b}까지" if a != b else a, "첫달": a,
                     "원래_비용후": 원사, "최종_비용후": 전사,
                     "원래_승자쪽_비용전": ((1 + 구["롱온리_%"] / 100).prod() - 1) * 100,
                     "원래_패자쪽_비용전": ((1 + 구["패자쪽_%"] / 100).prod() - 1) * 100,
                     "최종이_쉰_달": int((w.loc[a:b] == 0).sum()), "사건_개월": len(구)})
    사건표 = pd.DataFrame(사건줄)
    print("\n[검증 기간 붕괴 사건] (패자 쪽은 패자 바구니 수익률이라 양수면 공매도 손실)")
    print(사건표.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    print("\n[함께 보고할 모델]")
    for 구분 in ("최고단독", "최고섞은"):
        for 이름 in 기록[구분]:
            _, _, 검2, 끊2 = 보관[이름]
            print(f"  {구분} {이름}: 결론 {검2['결론']}, 월평균 {검2['월평균']:.3f}%, 샤프 {검2['샤프']:.3f}, "
                  f"최대낙폭 {검2['최대낙폭']:.1f}%, 사건 합 {검2['사건합']:.1f}%, t {검2['t']:.3f} (2025-12까지 t {끊2['t']:.3f})")

    print("\n[후보 57개의 검증 결과 요약] (보고용, 결론은 최종 모델로만 냄)")
    선택통과 = 표결과[표결과["선택_심사통과"]]
    print(f"  선택 기간 심사 통과 {len(선택통과)}개 중 검증 결론: {선택통과['결론'].value_counts().to_dict()}")
    print(f"  검증 t >= 1.96: {int((표결과['t'] >= 유의기준).sum())}개, 검증 t > 0: {int((표결과['t'] > 0).sum())}개 (57개 중)")
    print(표결과.sort_values("선택_t", ascending=False)[["이름", "선택_t", "켜진달", "월평균", "샤프", "최대낙폭", "사건합", "t", "결론"]]
          .head(15).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    결과폴더.mkdir(parents=True, exist_ok=True)
    표결과.to_csv(결과폴더 / "검증후보표.csv", index=False, encoding="utf-8-sig")
    사건표.to_csv(결과폴더 / "사건표.csv", index=False, encoding="utf-8-sig")
    그림들(최종, 검, 원래, w, r, 표, 단독)
    print(f"\n[저장] 검증후보표·사건표 csv, 그림 4개: {결과폴더}")


if __name__ == "__main__":
    실행()
