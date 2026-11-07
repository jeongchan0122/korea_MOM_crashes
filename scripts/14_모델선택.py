# -*- coding: utf-8 -*-
"""
14단계: 선택 기간(보유 달 2005-07부터 2015-12까지)에서 미리 정한 규칙대로 모델 하나를 고른다.

왜 필요한가
- 검증 기간에 시험할 모델을 결과를 보기 전에 하나로 정한다.

무엇을 하는가 (연구 기록 16절, 팀 문서 8.4부터 8.6까지)
1. 수익률, 지표, 신호를 선택 기간까지만 남긴다. 2016-01 이후 값이 있으면 멈춘다.
2. 전략 수익률 = 비중 x 원래 WML - 편도 0.5% x 회전율(12번). 원래 WML도 같은 방식으로 비용을 뺀다.
3. 거르기(켜진 달 6개 이상, 50% 이하) -> 지표 정의 고르기 -> 하나씩 추가하기(합치는 방법별)
   -> 통과 심사(샤프, 최대낙폭, 붕괴 사건 손실) -> 스패닝 알파 NW t 순위 -> 1 표준오차 규칙.
4. 후보표, 경로표, 정의 비교표, 최종 모델, 그림 4개를 저장한다.

하지 않는 일
- 검증 기간 성과 계산, 강건성, 다른 3가지 방식, FF3 알파 등 보고 항목.

실행: python scripts/14_모델선택.py   (먼저 05번, 11번, 12번, 13번을 실행해 두어야 함)
결과: 처리데이터/최종모델.json, 결과/모델선택/(후보표·경로표·정의비교 csv, 그림 png)
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
결과폴더 = 프로젝트 / "결과" / "모델선택"


def 불러오기(이름, 파일):
    명세 = importlib.util.spec_from_file_location(이름, Path(__file__).resolve().parent / 파일)
    모듈 = importlib.util.module_from_spec(명세)
    명세.loader.exec_module(모듈)
    return 모듈


신호코드 = 불러오기("신호", "13_신호만들기.py")

보유시작, 보유끝 = "2005-07", "2015-12"
지표시작, 지표끝 = "2005-06", "2015-11"
비용률 = 0.5                 # 편도 %, 회전율 1당
최소켜짐, 최대비율 = 6, 0.5
NW시차 = 4
부트횟수, 묶음길이, 시드 = 5000, 6, 20261107
붕괴비율, 묶음간격 = 0.05, 3
기대사건 = [("2007-11", "2008-01"), ("2008-12", "2009-04"), ("2012-01", "2012-01")]
정의묶음 = {"약세장": ["약세장24", "약세장18"], "CI": ["CI_A", "CI_B", "CI_C", "CI_D"],
          "비유동성": ["비유동성_회전율", "비유동성_Amihud"]}


class 선택문제(Exception):
    """확인에 실패했을 때 쓰는 오류. 이 오류가 나면 저장하지 않고 멈춘다."""


def 확인(조건, 설명):
    if not 조건:
        raise 선택문제(설명)


# ---------------------------------------------------------------- 자료

def 자료읽기():
    월 = pd.read_parquet(처리폴더 / "WML월별수익률.parquet")
    월 = 월[월["방식"] == "2x3"].set_index("연월")["롱숏_%"].loc[보유시작:보유끝]
    회전 = pd.read_parquet(처리폴더 / "WML회전율.parquet").set_index("연월").loc[보유시작:보유끝]
    표 = 신호코드.지표읽기().loc[:지표끝]
    단독 = pd.read_parquet(처리폴더 / "신호월별.parquet").set_index("연월").drop(columns="적용연월").loc[:지표끝]
    보유달 = list(pd.period_range(보유시작, 보유끝, freq="M").astype(str))
    확인(list(월.index) == 보유달 and len(보유달) == 126, "선택 기간 WML이 126개월이 아님")
    확인(list(회전.index) == 보유달, "회전율의 달이 선택 기간과 다름")
    확인(월.index.max() <= 보유끝 and 표.index.max() <= 지표끝 and 단독.index.max() <= 지표끝,
         "선택 기간 뒤의 자료가 남아 있음")
    return 월, 회전, 표, 단독, 보유달


# ---------------------------------------------------------------- 성과

def 비중만들기(신호, 보유달):
    """지표 달 신호 -> 보유 달 비중. 판단할 수 없는 달은 꺼짐(비중 1)으로 채우고 그 수를 센다."""
    s = 신호.reindex(list(pd.period_range(지표시작, 지표끝, freq="M").astype(str)))
    채움 = int(s.isna().sum())
    w = (1 - s.fillna(0)).values
    return pd.Series(w, index=보유달), 채움


def 비용후(w, 월, 회전):
    """비중 w의 비용 후 수익률(%). 선택 기간 직전 달의 비중은 1로 본다."""
    앞 = w.shift(1).fillna(1.0)
    회전율 = np.select([(앞 == 1) & (w == 1), (앞 == 1) & (w == 0), (앞 == 0) & (w == 1)],
                    [회전["리밸런싱"], 회전["청산"], 회전["구축"]], 0.0)
    return w * 월 - 비용률 * 회전율


def 사건정의(월):
    n = int(round(붕괴비율 * len(월)))
    달들 = sorted(월.nsmallest(n).index)
    사건들 = []
    for 달 in 달들:
        if 사건들 and (pd.Period(달) - pd.Period(사건들[-1][1])).n <= 묶음간격:
            사건들[-1][1] = 달
        else:
            사건들.append([달, 달])
    return [tuple(x) for x in 사건들], 달들


def 사건수익(r, 사건들):
    return [((1 + r.loc[a:b] / 100).prod() - 1) * 100 for a, b in 사건들]


def 낙폭(r):
    부 = (1 + r / 100).cumprod()
    return (부 / 부.cummax() - 1) * 100


def 알파t(R, x):
    """R(개월 x 후보)을 x에 회귀한 절편, 기울기, Newey-West(시차 4, Bartlett) t값을 후보마다 한 번에 계산."""
    R = np.asarray(R, float)
    if R.ndim == 1:
        R = R[:, None]
    T = len(x)
    X = np.column_stack([np.ones(T), x])
    XtXi = np.linalg.inv(X.T @ X)
    B = XtXi @ X.T @ R
    e = R - X @ B
    u = (X @ XtXi[0])[:, None] * e
    V = (u * u).sum(axis=0)
    for L in range(1, NW시차 + 1):
        V += 2 * (1 - L / (NW시차 + 1)) * (u[L:] * u[:-L]).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):   # 잔차가 모두 0이면 t값은 빈칸(NaN)이 된다.
        t = np.where(V > 0, B[0] / np.sqrt(np.where(V > 0, V, 1)), np.nan)
    return B[0], B[1], t


def 부트위치(T):
    rng = np.random.default_rng(시드)
    묶음수 = int(np.ceil(T / 묶음길이))
    시작 = rng.integers(0, T - 묶음길이 + 1, size=(부트횟수, 묶음수))
    return (시작[:, :, None] + np.arange(묶음길이)).reshape(부트횟수, -1)[:, :T]


def 부트t(R, x, 위치):
    """같은 부트스트랩 표본에서 후보마다 t값. 결과: 반복 x 후보."""
    R = np.asarray(R, float)
    return np.vstack([알파t(R[i], x[i])[2] for i in 위치])


# ---------------------------------------------------------------- 후보

class 후보모음:
    def __init__(self, 월, 회전, 표, 단독, 보유달):
        self.월, self.회전, self.표, self.단독, self.보유달 = 월, 회전, 표, 단독, 보유달
        self.원래 = 비용후(pd.Series(1.0, index=보유달), 월, 회전)
        self.사건, self.붕괴달 = 사건정의(월)
        self.원래사건 = 사건수익(self.원래, self.사건)
        self.원래샤프 = self.원래.mean() / self.원래.std() * np.sqrt(12)
        self.원래낙폭 = 낙폭(self.원래).min()
        self.위치 = 부트위치(len(보유달))
        self.목록 = {}
        self.뺀반복 = {}

    def 추가(self, 종류, 지표들, 단계=""):
        이름 = f"{종류}:" + "+".join(지표들)
        if 이름 in self.목록:
            return self.목록[이름]
        if 종류 == "단독":
            신호 = self.단독[지표들[0]]
        else:
            신호 = 신호코드.조합신호(self.표, self.단독, tuple(지표들), "과반" if 종류 == "과반" else "점수")
        w, 채움 = 비중만들기(신호, self.보유달)
        r = 비용후(w, self.월, self.회전)
        a, b, t = 알파t(r.values, self.원래.values)
        켜짐 = int((w == 0).sum())
        # 보고용 적중률(선택에는 쓰지 않음): 붕괴 달 중 쉰 달 수, 쉰 달 중 붕괴 달 비율
        포착 = int(((w == 0) & w.index.isin(self.붕괴달)).sum())
        사건 = 사건수익(r, self.사건)
        c = {"이름": 이름, "종류": 종류, "지표": list(지표들), "지표수": len(지표들), "단계": 단계,
             "켜진달": 켜짐, "켜짐비율": 켜짐 / len(w), "채운달": 채움,
             "붕괴포착": 포착, "정확도": 포착 / 켜짐 if 켜짐 else np.nan,
             "거르기통과": 켜짐 >= 최소켜짐 and 켜짐 / len(w) <= 최대비율,
             "월평균": r.mean(), "샤프": r.mean() / r.std() * np.sqrt(12), "최대낙폭": 낙폭(r).min(),
             **{f"사건{i + 1}": v for i, v in enumerate(사건)}, "사건합": sum(사건),
             "알파": float(a[0]), "b": float(b[0]), "t": float(t[0]), "_w": w, "_r": r}
        c["샤프통과"] = c["샤프"] > self.원래샤프
        c["낙폭통과"] = c["최대낙폭"] >= self.원래낙폭
        c["사건통과"] = c["사건합"] > sum(self.원래사건)
        c["심사통과"] = c["거르기통과"] and c["샤프통과"] and c["낙폭통과"] and c["사건통과"]
        self.목록[이름] = c
        return c

    def 표준오차(self, 기준, 다른것들):
        """기준 후보와 다른 후보들의 t값 차이의 부트스트랩 표준편차."""
        R = np.column_stack([self.목록[기준]["_r"].values] + [self.목록[n]["_r"].values for n in 다른것들])
        bt = 부트t(R, self.원래.values, self.위치)
        # 쉬는 달이 적은 후보는 다시 뽑은 표본에 쉬는 달이 없으면 원래 WML과 같아져 t값을 계산할 수 없다.
        # 그런 반복은 빼고 계산 가능한 반복으로 표준편차를 구하며, 뺀 수를 기록한다.
        결과 = {}
        for i, n in enumerate(다른것들):
            d = bt[:, 0] - bt[:, i + 1]
            ok = np.isfinite(d)
            결과[n] = float(np.std(d[ok], ddof=1))
            self.뺀반복[n] = int((~ok).sum())
        return 결과


def 정의고르기(모음):
    줄들, 고른것 = [], {}
    for 묶음, 정의들 in 정의묶음.items():
        cs = [모음.추가("단독", [d]) for d in 정의들]
        통과 = [c for c in cs if c["거르기통과"]]
        기본 = cs[0]
        if not 통과:
            선택, 이유 = 정의들[0], "모두 거르기에서 빠져 기본 정의 유지"
        else:
            일위 = max(통과, key=lambda c: c["t"])
            if not 기본["거르기통과"]:
                선택, 이유 = 일위["지표"][0], "기본 정의가 거르기에서 빠져 통과한 정의 중 1위"
            elif 일위 is 기본:
                선택, 이유 = 정의들[0], "기본 정의가 1위"
            else:
                se = 모음.표준오차(일위["이름"], [기본["이름"]])[기본["이름"]]
                차이 = 일위["t"] - 기본["t"]
                if 차이 > se:
                    선택, 이유 = 일위["지표"][0], f"1위가 기본보다 t {차이:.2f} 높고 표준오차 {se:.2f}보다 큼"
                else:
                    선택, 이유 = 정의들[0], f"1위와 기본의 t 차이 {차이:.2f}가 표준오차 {se:.2f} 이내"
        고른것[묶음] = 선택
        for c in cs:
            줄들.append({"묶음": 묶음, "정의": c["지표"][0], "켜진달": c["켜진달"], "거르기통과": c["거르기통과"],
                       "t": c["t"], "선택": c["지표"][0] == 선택, "이유": 이유})
    return 고른것, pd.DataFrame(줄들)


def 재료(고른것):
    아미 = 고른것["비유동성"] == "비유동성_Amihud"
    return ["모멘텀갭", "시장변동성", 고른것["약세장"], "모멘텀변동성", "왜도", "모멘텀베타", "금융난류",
            "비유동성_Amihud" if 아미 else "비유동성_회전율", "비유동성갭_Amihud" if 아미 else "비유동성갭_회전율"]


def 하나씩추가(모음, 재료들, 방법):
    줄들 = []
    단독들 = [모음.추가("단독", [m]) for m in 재료들]
    통과 = [c for c in 단독들 if c["거르기통과"]]
    if not 통과:
        return 줄들
    현재 = max(통과, key=lambda c: c["t"])["지표"]
    줄들.append({"방법": 방법, "단계": 1, "후보": "+".join(현재), "t": 모음.목록[f"단독:{현재[0]}"]["t"],
               "거르기통과": True, "다음으로": True})
    for 단계 in (2, 3, 4):
        시험 = [모음.추가(방법, 현재 + [m], 단계=f"{방법}{단계}") for m in 재료들 if m not in 현재]
        통과 = [c for c in 시험 if c["거르기통과"]]
        일위 = max(통과, key=lambda c: c["t"]) if 통과 else None
        for c in 시험:
            줄들.append({"방법": 방법, "단계": 단계, "후보": "+".join(c["지표"]), "t": c["t"],
                       "거르기통과": c["거르기통과"], "다음으로": c is 일위})
        if 일위 is None:
            break
        현재 = 일위["지표"]
    return 줄들


def 최종고르기(모음):
    심사 = [c for c in 모음.목록.values() if c["심사통과"]]
    if not 심사:
        return None, [], {}
    일위 = max(심사, key=lambda c: c["t"])
    se = 모음.표준오차(일위["이름"], [c["이름"] for c in 심사 if c is not 일위])
    동률 = [일위] + [c for c in 심사 if c is not 일위 and 일위["t"] - c["t"] <= se[c["이름"]]]
    최종 = min(동률, key=lambda c: (c["지표수"], -c["최대낙폭"]))
    return 최종, 동률, se


# ---------------------------------------------------------------- 점검

def 손계산(모음):
    """모멘텀 갭 단독의 비중, 비용 후 수익률, 샤프, 알파 t를 반복문으로 따로 계산한다."""
    c = 모음.목록["단독:모멘텀갭"]
    신호 = 모음.단독["모멘텀갭"]
    r, 앞 = [], 1.0
    for 달 in 모음.보유달:
        지표달 = str(pd.Period(달) - 1)
        w = 0.0 if 신호.loc[지표달] == 1 else 1.0
        if 앞 == 1 and w == 1:
            비용 = 모음.회전.loc[달, "리밸런싱"]
        elif 앞 == 1 and w == 0:
            비용 = 모음.회전.loc[달, "청산"]
        elif 앞 == 0 and w == 1:
            비용 = 모음.회전.loc[달, "구축"]
        else:
            비용 = 0.0
        r.append(w * 모음.월.loc[달] - 비용률 * 비용)
        앞 = w
    r = np.array(r)
    확인(np.allclose(r, c["_r"].values, atol=1e-12), "손계산 비용 후 수익률이 다름")
    샤프 = r.mean() / r.std(ddof=1) * np.sqrt(12)
    # 알파 t: 행렬 샌드위치 식
    x = 모음.원래.values
    X = np.column_stack([np.ones(len(x)), x])
    β = np.linalg.lstsq(X, r, rcond=None)[0]
    e = r - X @ β
    S = sum(np.outer(X[t] * e[t], X[t] * e[t]) for t in range(len(x)))
    for L in range(1, NW시차 + 1):
        Γ = sum(np.outer(X[t] * e[t], X[t - L] * e[t - L]) for t in range(L, len(x)))
        S += (1 - L / (NW시차 + 1)) * (Γ + Γ.T)
    V = np.linalg.inv(X.T @ X) @ S @ np.linalg.inv(X.T @ X)
    t값 = β[0] / np.sqrt(V[0, 0])
    print(f"[손계산 대조] 모멘텀 갭 단독: 샤프 코드 {c['샤프']:.10f} 손 {샤프:.10f}, "
          f"알파 t 코드 {c['t']:.10f} 손 {t값:.10f}")
    확인(np.isclose(샤프, c["샤프"], rtol=1e-10) and np.isclose(t값, c["t"], rtol=1e-8), "손계산 샤프나 t값이 다름")


# ---------------------------------------------------------------- 그림

파랑, 주황 = "#2a78d6", "#eb6834"
바탕, 글자, 보조글자, 약한글자 = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
격자, 기준선, 쉼색 = "#e1e0d9", "#c3c2b7", "#ecebe5"
plt.rcParams.update({
    "font.family": "Malgun Gothic", "axes.unicode_minus": False,
    "figure.facecolor": 바탕, "axes.facecolor": 바탕, "savefig.facecolor": 바탕,
    "text.color": 글자, "axes.labelcolor": 보조글자, "xtick.color": 약한글자, "ytick.color": 약한글자,
    "xtick.labelcolor": 보조글자, "ytick.labelcolor": 보조글자, "font.size": 11,
})


def 월말(달들):
    return pd.PeriodIndex(list(달들), freq="M").to_timestamp(how="end").normalize()


def 꾸미기(ax, 제목, 부제):
    for 면 in ["top", "right", "left"]:
        ax.spines[면].set_visible(False)
    ax.spines["bottom"].set_color(기준선)
    ax.grid(axis="y", color=격자, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.set_title(제목, loc="left", fontsize=15, fontweight="bold", pad=34, color=글자)
    ax.text(0, 1.03, 부제, transform=ax.transAxes, fontsize=10.5, color=보조글자, va="bottom")


def 쉼음영(ax, w):
    for 달 in w.index[w == 0]:
        p = pd.Period(달)
        ax.axvspan(p.start_time, p.end_time, color=쉼색, linewidth=0, zorder=0)


def 사건표시(ax, 사건들, y위치):
    for i, (a, b) in enumerate(사건들, 1):
        시작, 끝 = pd.Period(a).start_time, pd.Period(b).end_time
        ax.annotate("", xy=(시작, y위치), xytext=(끝, y위치), xycoords=("data", "axes fraction"),
                    arrowprops=dict(arrowstyle="-", color=보조글자, linewidth=1.2))
        ax.text(시작 + (끝 - 시작) / 2, y위치 + 0.02, f"사건{i}", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=9.5, color=보조글자)


def 그림들(모음, 최종):
    결과폴더.mkdir(parents=True, exist_ok=True)
    w, r = 최종["_w"], 최종["_r"]
    날 = 월말(모음.보유달)
    이름 = 최종["이름"]

    # 1. 누적수익
    fig, ax = plt.subplots(figsize=(11, 5.8))
    쉼음영(ax, w)
    for 값, 색, 표지 in ((모음.원래, 파랑, "원래 WML"), (r, 주황, "최종 모델")):
        누 = (1 + 값 / 100).cumprod()
        ax.plot(날, 누.values, color=색, linewidth=2, label=f"{표지}  {누.iloc[-1]:.2f}배")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([0.5, 0.75, 1, 1.5, 2, 3]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}배"))
    사건표시(ax, 모음.사건, 0.93)
    꾸미기(ax, f"선택 기간 누적수익: 원래 WML과 최종 모델", f"{이름}. 시작값 1, 로그 눈금, 거래비용(편도 0.5%) 차감. "
         "회색 음영은 신호가 켜져 쉰 달.")
    ax.legend(loc="lower right", frameon=False, ncol=2)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.84, bottom=0.08)
    fig.savefig(결과폴더 / "1_누적수익.png", dpi=200)
    plt.close(fig)

    # 2. 낙폭
    fig, ax = plt.subplots(figsize=(11, 5.0))
    쉼음영(ax, w)
    for 값, 색, 표지 in ((모음.원래, 파랑, "원래 WML"), (r, 주황, "최종 모델")):
        d = 낙폭(값)
        ax.plot(날, d.values, color=색, linewidth=2, label=f"{표지}  최대 {d.min():.1f}%")
    ax.axhline(0, color=기준선, linewidth=1)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    꾸미기(ax, "선택 기간 낙폭: 고점 대비 하락", f"{이름}. 거래비용 차감. 회색 음영은 쉰 달.")
    ax.legend(loc="lower right", frameon=False, ncol=2)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.82, bottom=0.08)
    fig.savefig(결과폴더 / "2_낙폭.png", dpi=200)
    plt.close(fig)

    # 3. 붕괴 사건 막대
    fig, ax = plt.subplots(figsize=(9, 5.0))
    x = np.arange(len(모음.사건))
    전략사건 = 사건수익(r, 모음.사건)
    for 값들, 색, 표지, 이동 in ((모음.원래사건, 파랑, "원래 WML", -0.2), (전략사건, 주황, "최종 모델", 0.2)):
        ax.bar(x + 이동, 값들, width=0.38, color=색, label=표지, linewidth=0)
        for xi, v in zip(x + 이동, 값들):
            ax.text(xi, v - 0.6 if v < 0 else v + 0.3, f"{v:.1f}%", ha="center", va="top" if v < 0 else "bottom",
                    fontsize=10, color=글자)
    ax.axhline(0, color=기준선, linewidth=1)
    ax.set_xticks(x, [f"사건{i + 1}\n{a}부터 {b}까지" if a != b else f"사건{i + 1}\n{a}" for i, (a, b) in enumerate(모음.사건)])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.set_ylim(min(min(모음.원래사건), min(전략사건)) - 4, max(2, max(전략사건) + 2))
    꾸미기(ax, "붕괴 사건별 누적수익률", f"{이름}. 사건 = 원래 WML 하위 5% 달을 3개월 이내끼리 묶은 구간. 거래비용 차감.")
    ax.legend(loc="lower right", frameon=False)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.82, bottom=0.14)
    fig.savefig(결과폴더 / "3_붕괴사건.png", dpi=200)
    plt.close(fig)

    # 4. 지표와 기준선
    그림4(모음, 최종)


def 기준선값(값, 창최소):
    """위험한 쪽이 큰 계열에서 켜지는 경계(창의 k번째로 위험한 값, k = 창의 20%를 올림)."""
    v = 값.values
    결과 = np.full(len(v), np.nan)
    있음 = np.flatnonzero(np.isfinite(v))
    for k, i in enumerate(있음):
        창 = v[있음[max(0, k - 신호코드.창최대 + 1):k + 1]]
        if len(창) < 창최소:
            continue
        순서 = np.sort(창)[::-1]
        결과[i] = 순서[int(np.ceil(len(창) / 5)) - 1]
    return pd.Series(결과, index=값.index)


def 그림4(모음, 최종):
    지표달 = list(pd.period_range(지표시작, 지표끝, freq="M").astype(str))
    날 = 월말(지표달)
    종류, 지표들 = 최종["종류"], 최종["지표"]
    fig, ax = plt.subplots(figsize=(11, 5.0))
    if 종류 == "단독" and 지표들[0] in 신호코드.연속:
        열, 방향 = 신호코드.연속[지표들[0]]
        값 = 모음.표[열]
        경계 = 방향 * 기준선값(방향 * 값, 신호코드.난류창최소 if 지표들[0] == "금융난류" else 신호코드.창최대)
        ax.plot(날, 값.reindex(지표달).values, color=파랑, linewidth=2, label=지표들[0])
        ax.plot(날, 경계.reindex(지표달).values, color=약한글자, linewidth=1.5, linestyle="--",
                label="켜지는 경계(최근 60개월 중 위험한 쪽 20%)")
        설명 = "높을수록 위험" if 방향 == 1 else "낮을수록 위험"
    elif 종류 == "점수":
        점수 = 신호코드.위험점수(모음.표, 지표들)
        경계 = 기준선값(점수, 신호코드.점수창최소)
        ax.plot(날, 점수.reindex(지표달).values, color=파랑, linewidth=2, label="겹침 조정 위험 점수")
        ax.plot(날, 경계.reindex(지표달).values, color=약한글자, linewidth=1.5, linestyle="--", label="켜지는 경계(상위 20%)")
        설명 = "높을수록 위험"
    elif 종류 == "과반":
        수 = 모음.단독[지표들].sum(axis=1).reindex(지표달)
        필요 = {2: 2, 3: 2, 4: 3}[len(지표들)]
        ax.step(날, 수.values, where="mid", color=파랑, linewidth=2, label="켜진 지표 수")
        ax.axhline(필요, color=약한글자, linewidth=1.5, linestyle="--", label=f"켜지는 경계({필요}개 이상)")
        설명 = "과반 규칙"
    else:  # 규칙형 단독(약세장, CI)
        if 지표들[0].startswith("약세장"):
            열 = 신호코드.약세장[지표들[0]][0]
            ax.plot(날, 모음.표[열].reindex(지표달).values, color=파랑, linewidth=2, label=f"{열}(%)")
            ax.axhline(0, color=약한글자, linewidth=1.5, linestyle="--", label="켜지는 경계(0% 미만)")
        else:
            ax.step(날, 모음.단독[지표들[0]].reindex(지표달).values, where="mid", color=파랑, linewidth=2, label=지표들[0])
        설명 = "규칙형"
    for 달 in 모음.붕괴달:
        p = pd.Period(달) - 1   # 붕괴 달의 비중을 정한 지표 달
        ax.axvspan(p.start_time, p.end_time, color="#f6c9b5", linewidth=0, zorder=0)
    꾸미기(ax, "최종 모델의 지표와 판단 경계", f"{최종['이름']} ({설명}). 연한 주황 = 붕괴 달의 비중을 정한 지표 달(붕괴 한 달 전).")
    ax.legend(loc="upper right", frameon=False, ncol=2)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.82, bottom=0.08)
    fig.savefig(결과폴더 / "4_지표와기준선.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- 실행

def 신호품질(모음, c):
    w = c["_w"]
    붕괴 = pd.Series(w.index.isin(모음.붕괴달), index=w.index)
    켜짐 = w == 0
    원 = 모음.원래
    return {"포착률": float((켜짐 & 붕괴).sum() / 붕괴.sum()), "정확도": float((켜짐 & 붕괴).sum() / max(켜짐.sum(), 1)),
            "피한손실": float(-원[켜짐 & (원 < 0)].sum()), "놓친수익": float(원[켜짐 & (원 > 0)].sum())}


def 실행():
    월, 회전, 표, 단독, 보유달 = 자료읽기()
    모음 = 후보모음(월, 회전, 표, 단독, 보유달)
    확인([tuple(s) for s in 모음.사건] == 기대사건, f"붕괴 사건이 문서와 다름: {모음.사건}")
    전부1 = 비용후(pd.Series(1.0, index=보유달), 월, 회전)
    확인(np.allclose(전부1, 월 - 비용률 * 회전["리밸런싱"]), "비중 1 전략의 비용이 원래 WML과 다름")
    print(f"[원래 WML, 선택 기간 126개월] 비용 전 월평균 {월.mean():.3f}%, 비용 후 {모음.원래.mean():.3f}%, "
          f"샤프(비용 후) {모음.원래샤프:.3f}, 최대낙폭 {모음.원래낙폭:.1f}%")
    print(f"  붕괴 달 {모음.붕괴달}, 사건 {모음.사건}, 사건 누적수익률(비용 후) {[round(v, 1) for v in 모음.원래사건]}, "
          f"합 {sum(모음.원래사건):.1f}%")

    고른것, 정의표 = 정의고르기(모음)
    print("\n[지표 정의 고르기]")
    print(정의표.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    재료들 = 재료(고른것)
    for m in 재료들 + [고른것["CI"]]:
        모음.추가("단독", [m])
    print(f"  조합 재료: {재료들}, CI: {고른것['CI']}")

    경로 = []
    for 방법 in ("과반", "점수"):
        경로 += 하나씩추가(모음, 재료들, 방법)
    경로표 = pd.DataFrame(경로)
    print("\n[하나씩 추가하기] 단계별로 다음으로 넘긴 후보")
    print(경로표[경로표["다음으로"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    손계산(모음)
    후보 = pd.DataFrame([{k: v for k, v in c.items() if not k.startswith("_")} for c in 모음.목록.values()])
    후보 = 후보.sort_values("t", ascending=False)
    print(f"\n[후보] {len(후보)}개, 거르기 통과 {int(후보['거르기통과'].sum())}개, 통과 심사 {int(후보['심사통과'].sum())}개")
    print(f"  원래 WML: 샤프 {모음.원래샤프:.3f}, 최대낙폭 {모음.원래낙폭:.1f}%, 사건합 {sum(모음.원래사건):.1f}%")
    열 = ["이름", "켜진달", "채운달", "거르기통과", "월평균", "샤프", "최대낙폭", "사건합", "심사통과", "알파", "b", "t"]
    print(후보[열].head(25).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    최종, 동률, se = 최종고르기(모음)
    결과폴더.mkdir(parents=True, exist_ok=True)
    후보.to_csv(결과폴더 / "후보표.csv", index=False, encoding="utf-8-sig")
    경로표.to_csv(결과폴더 / "경로표.csv", index=False, encoding="utf-8-sig")
    정의표.to_csv(결과폴더 / "정의비교.csv", index=False, encoding="utf-8-sig")

    if 최종 is None:
        print("\n[결과] 통과 심사를 지난 후보가 없음: 이 지표들로는 선택 기간에서도 원래 WML을 개선하지 못했다.")
        기록 = {"최종모델": None, "정의": 고른것}
    else:
        print(f"\n[1위] {max(동률, key=lambda c: c['t'])['이름']} t {max(c['t'] for c in 동률):.3f}")
        print("[동률(1 표준오차 이내)]: " + ", ".join(f"{c['이름']}(t {c['t']:.3f}, 지표 {c['지표수']}개, 낙폭 {c['최대낙폭']:.1f}%)" for c in 동률))
        print(f"  1위와의 차이에 쓴 표준오차: " + ", ".join(f"{k} {v:.3f}" for k, v in sorted(se.items(), key=lambda x: x[1])[:10]))
        뺀 = {k: v for k, v in 모음.뺀반복.items() if v > 0}
        print(f"  계산할 수 없어 뺀 부트스트랩 반복(5,000번 중): {뺀 if 뺀 else '없음'}")
        print(f"[최종 모델] {최종['이름']}")
        품질 = 신호품질(모음, 최종)
        print(f"  켜진 달 {최종['켜진달']}, 포착률 {품질['포착률']:.2f}, 정확도 {품질['정확도']:.2f}, "
              f"피한 손실 {품질['피한손실']:.1f}%p, 놓친 수익 {품질['놓친수익']:.1f}%p, 유의(t >= 1.96): {최종['t'] >= 1.96}")
        print(f"  쉰 달(보유 달): {list(최종['_w'].index[최종['_w'] == 0])}")
        심사 = 후보[후보["심사통과"]]
        최고단독 = 심사[심사["종류"] == "단독"].head(1)
        최고섞은 = 심사[심사["종류"] != "단독"].head(1)
        기록 = {"최종모델": {"이름": 최종["이름"], "종류": 최종["종류"], "지표": 최종["지표"]},
              "최고단독": 최고단독["이름"].tolist(), "최고섞은": 최고섞은["이름"].tolist(),
              "정의": 고른것, "규칙": {"창": 신호코드.창최대, "위쪽": "20%", "비용_편도%": 비용률},
              "선택기간_t": 최종["t"], "동률": [c["이름"] for c in 동률]}
        그림들(모음, 최종)
        print(f"  그림 4개 저장: {결과폴더}")
    with open(처리폴더 / "최종모델.json", "w", encoding="utf-8") as f:
        json.dump(기록, f, ensure_ascii=False, indent=2)
    print(f"[저장] 최종모델.json, 후보표·경로표·정의비교 csv")


if __name__ == "__main__":
    실행()
