# -*- coding: utf-8 -*-
"""
11단계: 위험 지표 10개(정의 비교용 변형 포함)를 매달 말 계산해 저장한다.

왜 필요한가
- 위험 신호의 재료다. t월 말 값은 t월 말까지의 자료로만 계산하고, t+1월 비중 판단에 쓴다.

무엇을 하는가 (연구 기록 15절, 팀 문서 8.2)
- 모멘텀 갭: t월 말 투자 대상 중 점수가 있는 종목의 log(1 + 점수)의 75번째 - 25번째 분위수
- 시장 변동성, 모멘텀 변동성, 왜도, 모멘텀 베타: t월 마지막 거래일까지 126거래일의 일별 시장과 일별 2x3 WML
- 약세장: t-23월(18개월은 t-17월)부터 t월까지의 월별 시장 누적수익률. 음수면 켜짐
- 금융난류: t월의 (시장, 승자 쪽, 패자 쪽) 월 수익률이 t-1월까지의 평균·공분산에서 벗어난 마할라노비스 거리
  (24개월이 쌓인 뒤부터)
- 비유동성(회전율, Amihud): 종목별 t월 일평균을 t월 말 시가총액으로 가중평균. 거래대금 0인 날과
  일별 수익률 절댓값 35% 초과인 날은 빼고, 남은 날이 10일 미만이면 그 종목은 그달 값이 없다.
- 비유동성 갭: 종목별 t-11월부터 t-1월까지의 월 비유동성 평균(6개월 이상) -> 2x3 바구니별 시총가중 ->
  (소형 승자, 대형 승자 평균) - (소형 패자, 대형 패자 평균)
- CI A, B, C, D: 약세장 & t월 시장 수익률 > 0 & t월 말 베타 < 0 (C, D는 t-1월 시장 수익률 < 0 추가)

하지 않는 일
- 지표와 다음 달 WML 수익률의 관계 계산(상관, 붕괴 달의 값, 신호 성과). 다음 단계에서 미리 정한 규칙으로만 본다.
- 신호(상위 20%), z-score, 섞기, 모델 선택, 강건성 변형.

실행: python scripts/11_지표계산.py   (먼저 05번, 07번, 08번, 10번을 실행해 두어야 함)
결과: 처리데이터/지표월별.parquet
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

프로젝트 = Path(__file__).resolve().parent.parent
처리폴더 = 프로젝트 / "처리데이터"
저장경로 = 처리폴더 / "지표월별.parquet"

_명세 = importlib.util.spec_from_file_location("WML계산", Path(__file__).resolve().parent / "05_WML계산.py")
월별코드 = importlib.util.module_from_spec(_명세)
_명세.loader.exec_module(월별코드)
시가총액 = 월별코드.시가총액

창 = 126                 # 일별 지표의 거래일 창
극단값기준 = 35.0         # 일별 수익률 절댓값(%) 초과는 그날 빼기(08번과 같은 규칙)
최소일수 = 10             # 비유동성: 종목별 한 달 최소 유효 일수
형성개월, 최소형성개월 = 11, 6   # 비유동성 갭: t-11월부터 t-1월까지, 6개월 이상
난류최소개월 = 24
# 지표 달 기준. 2005-06 말 지표가 선택 기간 첫 보유 달(2005-07)을, 2026-08 말 지표가 마지막 보유 달(2026-09)을 정한다.
선택시작, 마지막지표달 = "2005-06", "2026-08"
점검달들 = ["2008-10", "2015-12", "2020-03"]    # 미래 정보 점검에 쓸 달
손계산달 = "2010-06"

연속지표 = ["모멘텀갭", "시장변동성", "약세장누적24", "모멘텀변동성", "왜도", "모멘텀베타", "금융난류",
         "비유동성_회전율", "비유동성갭_회전율"]   # 조합 재료 9개(기본 정의)


class 지표문제(Exception):
    """확인에 실패했을 때 쓰는 오류. 이 오류가 나면 저장하지 않고 멈춘다."""


def 확인(조건, 설명):
    if not 조건:
        raise 지표문제(설명)


# ---------------------------------------------------------------- 자료

def 자료읽기():
    일WML = pd.read_parquet(처리폴더 / "일별WML수익률.parquet")
    월WML = pd.read_parquet(처리폴더 / "WML월별수익률.parquet")
    return {
        "패널": pd.read_parquet(처리폴더 / "월별패널.parquet"),
        "일수익률": pd.read_parquet(처리폴더 / "일별수익률.parquet"),
        "일시총": pd.read_parquet(처리폴더 / "일별시가총액.parquet"),
        "일대금": pd.read_parquet(처리폴더 / "일별거래대금.parquet"),
        "일WML": 일WML[일WML["방식"] == "2x3"].set_index("날짜").sort_index()[["롱숏_%", "시장_%"]],
        "월WML": 월WML[월WML["방식"] == "2x3"].set_index("연월").sort_index()[["롱숏_%", "롱온리_%", "패자쪽_%", "시장_%"]],
    }


def 자르기(자료, 끝달):
    """끝달 말 이후의 자료를 모두 지운다(미래 정보 점검용)."""
    끝날 = pd.Period(끝달, "M").end_time.normalize()
    return {
        "패널": 자료["패널"][자료["패널"]["연월"] <= 끝달],
        "일수익률": 자료["일수익률"].loc[:끝날],
        "일시총": 자료["일시총"].loc[:끝날],
        "일대금": 자료["일대금"].loc[:끝날],
        "일WML": 자료["일WML"].loc[:끝날],
        "월WML": 자료["월WML"].loc[:끝달],
    }


# ---------------------------------------------------------------- 지표

def 월말값(일별, 달들):
    """일별 계열에서 달마다 마지막 거래일 값을 뽑는다."""
    s = 일별.groupby(일별.index.to_period("M").astype(str)).last()
    return s.reindex(달들)


def 일별지표(일WML, 달들):
    """126거래일 창의 시장 변동성, 모멘텀 변동성, 왜도, 베타."""
    w, m = 일WML["롱숏_%"], 일WML["시장_%"]
    확인(일WML.notna().all().all(), "일별 WML이나 시장에 빈 날이 있음")
    굴림 = lambda s: s.rolling(창, min_periods=창)
    return pd.DataFrame({
        "시장변동성": 월말값(굴림(m).std(), 달들),
        "모멘텀변동성": 월말값(굴림(w).std(), 달들),
        "왜도": 월말값(굴림(w).skew(), 달들),
        "모멘텀베타": 월말값(굴림(w).cov(m) / 굴림(m).var(), 달들),
    })


def 약세장지표(월WML, 달들):
    로그 = np.log1p(월WML["시장_%"] / 100)
    결과 = {}
    for 개월 in (24, 18):
        누적 = np.expm1(로그.rolling(개월, min_periods=개월).sum()) * 100
        결과[f"약세장누적{개월}"] = 누적.reindex(달들)
        결과[f"약세장{개월}"] = (누적 < 0).where(누적.notna()).reindex(달들)
    결과["시장수익률"] = 월WML["시장_%"].reindex(달들)
    결과["전월시장수익률"] = 월WML["시장_%"].shift(1).reindex(달들)
    return pd.DataFrame(결과)


def 금융난류(월WML, 달들):
    """t월 y를 t-1월까지의 평균·공분산으로 잰 마할라노비스 거리."""
    y = 월WML[["시장_%", "롱온리_%", "패자쪽_%"]].dropna()
    값 = pd.Series(np.nan, index=y.index)
    for i in range(난류최소개월, len(y)):
        과거 = y.iloc[:i].values
        차 = y.iloc[i].values - 과거.mean(axis=0)
        값.iloc[i] = 차 @ np.linalg.solve(np.cov(과거, rowvar=False), 차)
    return 값.reindex(달들)


def 종목월비유동성(자료):
    """종목별 월 비유동성(회전율, Amihud)과 유효 일수. 날짜 x 종목 -> 연월 x 종목."""
    r = 자료["일수익률"]
    대금, 시총 = 자료["일대금"], 자료["일시총"]
    유효 = r.notna() & (대금 > 0) & 시총.notna() & (r.abs() <= 극단값기준)
    절대 = (r.abs() / 100).where(유효)
    회전 = 절대 / (대금 / (시총 * 1e6))
    아미 = 절대 / (대금 / 1e6)
    달 = r.index.to_period("M").astype(str)
    일수 = 유효.groupby(달).sum()
    회전월 = 회전.groupby(달).mean().where(일수 >= 최소일수)
    아미월 = 아미.groupby(달).mean().where(일수 >= 최소일수)
    return 회전월, 아미월, 일수


def 가중평균(값, 가중):
    있음 = 값.notna() & 가중.notna() & (가중 > 0)
    return np.average(값[있음], weights=가중[있음]) if 있음.any() else np.nan


def 비유동성지표(자료, 대상, 바구니, 달들):
    회전월, 아미월, _ = 종목월비유동성(자료)
    결과 = []
    형성 = {}
    for 이름, 월값 in (("회전율", 회전월), ("Amihud", 아미월)):
        # 형성기간 평균: t-11월부터 t-1월까지, 6개월 이상
        전체달 = pd.period_range(월값.index.min(), 월값.index.max(), freq="M").astype(str)
        확인(list(전체달) == list(월값.index), "일별 자료의 달에 빠진 달이 있음")
        형성[이름] = 월값.shift(1).rolling(형성개월, min_periods=최소형성개월).mean()

    for 달 in 달들:
        줄 = {"연월": 달}
        g = 대상[대상["구성연월"] == 달]
        b = 바구니[바구니["구성연월"] == 달]
        for 이름, 월값 in (("회전율", 회전월), ("Amihud", 아미월)):
            v = 월값.loc[달].reindex(g["종목코드"]).values if 달 in 월값.index else np.full(len(g), np.nan)
            줄[f"비유동성_{이름}"] = 가중평균(pd.Series(v), pd.Series(g[시가총액].values))
            if 이름 == "회전율":
                줄["점검_대상종목수"] = len(g)
                줄["점검_비유동성종목수"] = int(np.isfinite(v).sum())
            f = 형성[이름].loc[달].reindex(b["종목코드"]).values if 달 in 형성[이름].index else np.full(len(b), np.nan)
            bb = b.assign(형성값=f)
            바구니값 = {k: 가중평균(gg["형성값"], gg[시가총액]) for k, gg in bb.groupby("2x3")}
            if all(k in 바구니값 for k in ("SW", "BW", "SL", "BL")):
                승자 = np.mean([바구니값["SW"], 바구니값["BW"]])
                패자 = np.mean([바구니값["SL"], 바구니값["BL"]])
                줄[f"비유동성갭_{이름}"] = 승자 - 패자
            else:
                줄[f"비유동성갭_{이름}"] = np.nan
            if 이름 == "회전율":
                승패 = bb[bb["2x3"].isin(["SW", "BW", "SL", "BL"])]
                줄["점검_승패종목수"] = len(승패)
                줄["점검_갭값있는종목수"] = int(승패["형성값"].notna().sum())
        결과.append(줄)
    return pd.DataFrame(결과).set_index("연월"), 형성


def 계산(자료):
    """모든 지표를 계산해 연월(지표 달) x 지표 표로 돌려준다."""
    패널 = 자료["패널"]
    대상 = 월별코드.투자대상표(패널)
    점수 = 월별코드.모멘텀점수(패널)
    표 = 대상.merge(점수, on=["구성연월", "종목코드"], how="inner")
    바구니 = 월별코드.바구니나누기(표)
    달들 = sorted(대상["구성연월"].unique())

    갭 = 표.groupby("구성연월")["점수"].apply(lambda s: np.log1p(s / 100).quantile(0.75) - np.log1p(s / 100).quantile(0.25))
    결과 = pd.DataFrame(index=pd.Index(달들, name="연월"))
    결과["적용연월"] = (pd.PeriodIndex(결과.index, freq="M") + 1).astype(str)
    결과["모멘텀갭"] = 갭.reindex(달들)
    결과["점검_갭종목수"] = 표.groupby("구성연월").size().reindex(달들)
    결과 = 결과.join(일별지표(자료["일WML"], 달들))
    결과 = 결과.join(약세장지표(자료["월WML"], 달들))
    결과["금융난류"] = 금융난류(자료["월WML"], 달들)
    비유동, 형성 = 비유동성지표(자료, 대상, 바구니, 달들)
    결과 = 결과.join(비유동)

    상승 = 결과["시장수익률"] > 0
    하락전월 = 결과["전월시장수익률"] < 0
    음베타 = 결과["모멘텀베타"] < 0
    for 정의, 개월, 전전월 in (("A", 24, False), ("B", 18, False), ("C", 24, True), ("D", 18, True)):
        켜짐 = (결과[f"약세장{개월}"] == 1) & 상승 & 음베타
        if 전전월:
            켜짐 &= 하락전월
        필요 = 결과[[f"약세장{개월}", "시장수익률", "모멘텀베타"] + (["전월시장수익률"] if 전전월 else [])].notna().all(axis=1)
        결과[f"CI_{정의}"] = 켜짐.astype(float).where(필요)
    return 결과, {"대상": 대상, "표": 표, "바구니": 바구니, "형성": 형성}


# ---------------------------------------------------------------- 점검

def 미래정보점검(자료, 결과):
    print("\n[미래 정보 점검] 그달 말 이후 자료를 지우고 다시 계산")
    지표열 = [c for c in 결과.columns if not c.startswith("점검_") and c != "적용연월"]
    for 달 in 점검달들:
        다시, _ = 계산(자르기(자료, 달))
        a, b = 결과.loc[달, 지표열].astype(float), 다시.loc[달, 지표열].astype(float)
        같음 = np.isclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True)
        확인(같음.all(), f"{달}: 미래 자료를 지우면 값이 달라짐: {list(np.array(지표열)[~같음])}")
        print(f"  {달}: 지표 {len(지표열)}개 모두 같음")


def 손계산(자료, 결과, 부속):
    """한 달을 골라 갭, 베타, 시장 비유동성을 코드와 다른 길로 계산한다."""
    달 = 손계산달
    print(f"\n[손계산 대조] {달}")
    # 갭: 월별 수익률에서 직접 11개월 누적
    패널 = 자료["패널"]
    달들 = pd.period_range(pd.Period(달) - 11, pd.Period(달) - 1, freq="M").astype(str)
    넓 = 패널[패널["연월"].isin(달들)].pivot(index="종목코드", columns="연월", values=월별코드.점수열수익률)
    대상 = 부속["대상"][부속["대상"]["구성연월"] == 달]["종목코드"]
    넓 = 넓.reindex(대상).dropna()
    로그누적 = np.log((1 + 넓 / 100).prod(axis=1))
    갭 = np.percentile(로그누적, 75) - np.percentile(로그누적, 25)
    print(f"  모멘텀 갭: 코드 {결과.loc[달, '모멘텀갭']:.10f}, 손계산 {갭:.10f}")
    확인(np.isclose(갭, 결과.loc[달, "모멘텀갭"], rtol=1e-9), "모멘텀 갭 손계산이 다름")
    # 베타: 마지막 126거래일 최소제곱
    d = 자료["일WML"].loc[:pd.Period(달).end_time].iloc[-창:]
    기울기 = np.polyfit(d["시장_%"].values, d["롱숏_%"].values, 1)[0]
    print(f"  모멘텀 베타: 코드 {결과.loc[달, '모멘텀베타']:.10f}, 손계산 {기울기:.10f}")
    확인(np.isclose(기울기, 결과.loc[달, "모멘텀베타"], rtol=1e-8), "모멘텀 베타 손계산이 다름")
    # 시장 비유동성(회전율): 종목마다 반복문
    g = 부속["대상"][부속["대상"]["구성연월"] == 달]
    날 = 자료["일수익률"].loc[달].index
    값들, 가중들 = [], []
    for 코드, 시총말 in zip(g["종목코드"], g[시가총액]):
        r = 자료["일수익률"].loc[날, 코드].values
        v = 자료["일대금"].loc[날, 코드].values
        c = 자료["일시총"].loc[날, 코드].values
        비 = [abs(ri) / 100 / (vi / (ci * 1e6)) for ri, vi, ci in zip(r, v, c)
             if np.isfinite(ri) and np.isfinite(ci) and vi > 0 and abs(ri) <= 극단값기준]
        if len(비) >= 최소일수:
            값들.append(np.mean(비)); 가중들.append(시총말)
    손 = np.average(값들, weights=가중들)
    print(f"  시장 비유동성(회전율): 코드 {결과.loc[달, '비유동성_회전율']:.10f}, 손계산 {손:.10f}")
    확인(np.isclose(손, 결과.loc[달, "비유동성_회전율"], rtol=1e-9), "시장 비유동성 손계산이 다름")


def 외부대조(결과):
    지수 = pd.read_parquet(처리폴더 / "정리표" / "44_지수_일간.parquet").set_index("날짜")["I.001"]
    코스피 = pd.to_numeric(지수, errors="coerce").pct_change(fill_method=None) * 100
    코스피 = 코스피.dropna()
    변동 = 월말값(코스피.rolling(창, min_periods=창).std(), 결과.index)
    구간 = 결과.loc[선택시작:]
    print("\n[외부 대조] 시장 변동성과 코스피 지수로 같은 식으로 계산한 변동성")
    print(f"  상관(지표 달 2005-06부터 2026-08까지) {구간['시장변동성'].corr(변동.loc[선택시작:]):.3f}, "
          f"평균 {구간['시장변동성'].mean():.3f}% 대 {변동.loc[선택시작:].mean():.3f}%")


def 덮는범위(자료, 결과, 부속):
    print("\n[비유동성이 덮는 종목]")
    비율 = 결과["점검_비유동성종목수"] / 결과["점검_대상종목수"]
    print(f"  시장 비유동성: 투자 대상 중 값이 있는 비율 최소 {비율.min() * 100:.1f}%({비율.idxmin()}), 평균 {비율.mean() * 100:.2f}%")
    갭비율 = 결과["점검_갭값있는종목수"] / 결과["점검_승패종목수"]
    print(f"  비유동성 갭: 승자·패자 바구니 종목 중 형성값이 있는 비율 최소 {갭비율.min() * 100:.1f}%({갭비율.idxmin()}), "
          f"평균 {갭비율.mean() * 100:.2f}%")
    # 형성기간 최소 개월을 바꾸면 빠지는 종목 수(승자·패자 바구니, 회전율 기준)
    회전월, _, _ = 종목월비유동성(자료)
    b = 부속["바구니"][부속["바구니"]["2x3"].isin(["SW", "BW", "SL", "BL"])]
    b = b[b["구성연월"].between(선택시작, 마지막지표달)]
    키 = pd.MultiIndex.from_arrays([b["구성연월"], b["종목코드"]])
    for 최소 in (1, 6, 11):
        f = 회전월.shift(1).rolling(형성개월, min_periods=최소).mean().stack()
        없음 = int(pd.Series(f).reindex(키).isna().sum())
        print(f"  형성기간 {최소}개월 이상 요구: 지표 달 2005-06부터 승자·패자 종목·월 {len(b):,}개 중 값 없음 {없음:,}개({없음 / len(b) * 100:.2f}%)")
    # 35% 규칙으로 빠지는 칸(투자 대상, 거래대금 있음)
    r, 대금 = 자료["일수익률"], 자료["일대금"]
    큰 = (r.abs() > 극단값기준) & (대금 > 0)
    달 = r.index.to_period("M").astype(str)
    대상집합 = set(zip(부속["대상"]["구성연월"], 부속["대상"]["종목코드"]))
    위치 = np.argwhere(큰.values)
    해당 = sum((달[i], r.columns[j]) in 대상집합 for i, j in 위치)
    print(f"  35% 규칙으로 빠지는 칸: 거래대금이 있는 칸 {len(위치):,}개 중 그 달 말 투자 대상 {해당}개")


def 모양보고(결과):
    """지표 자체의 모양만 본다. 다음 달 WML 수익률과의 관계는 계산하지 않는다."""
    지표열 = [c for c in 결과.columns if not c.startswith("점검_") and c not in ("적용연월", "시장수익률", "전월시장수익률")]
    print("\n[첫 값이 나오는 달, 지표 달 2005-06부터 2026-08까지 빈 달 수]")
    for c in 지표열:
        print(f"  {c}: {결과[c].first_valid_index()}, 빈 달 {int(결과.loc[선택시작:마지막지표달, c].isna().sum())}")
    연속 = [c for c in 지표열 if not c.startswith(("약세장2", "약세장1", "CI_"))]
    for 이름, a, b in (("선택 기간 (지표 달 2005-06부터 2015-11까지)", "2005-06", "2015-11"),
                     ("검증 기간 (지표 달 2015-12부터 2026-08까지)", "2015-12", "2026-08")):
        구 = 결과.loc[a:b]
        print(f"\n[{이름}] 연속값 지표의 분포")
        print(구[연속].describe(percentiles=[0.1, 0.5, 0.9]).T[["count", "mean", "10%", "50%", "90%"]]
              .to_string(float_format=lambda x: f"{x:,.4g}"))
        켜짐 = 구[["약세장24", "약세장18", "CI_A", "CI_B", "CI_C", "CI_D"]]
        print("  켜진 달 수: " + ", ".join(f"{c} {int(켜짐[c].sum())}" for c in 켜짐.columns) + f" (전체 {len(구)}개월)")
    상관 = 결과.loc["2005-06":"2015-11", 연속지표].corr()
    print("\n[조합 재료 9개의 상관, 선택 기간] (주성분 분석의 재료)")
    print(상관.to_string(float_format=lambda x: f"{x:.2f}"))
    고유 = np.sort(np.linalg.eigvalsh(상관.values))[::-1]
    print("  주성분 설명 비율(%): " + ", ".join(f"{v / 고유.sum() * 100:.1f}" for v in 고유))


def 실행():
    자료 = 자료읽기()
    결과, 부속 = 계산(자료)

    # 멈추는 조건
    달력 = pd.period_range(결과.index[0], 결과.index[-1], freq="M").astype(str)
    확인(list(결과.index) == list(달력), "지표 달에 빠진 달이 있음")
    확인(결과.index[-1] == 마지막지표달, f"마지막 지표 달이 {마지막지표달}이 아님: {결과.index[-1]}")
    본기간 = 결과.loc[선택시작:마지막지표달]
    지표열 = [c for c in 결과.columns if not c.startswith("점검_") and c != "적용연월"]
    빈 = 본기간[지표열].isna().sum()
    확인((빈 == 0).all(), f"{선택시작}부터 빈 지표가 있음: {빈[빈 > 0].to_dict()}")
    비율 = 결과["점검_비유동성종목수"] / 결과["점검_대상종목수"]
    확인((비율 >= 0.9).all(), f"비유동성 종목이 투자 대상의 90% 미만인 달: {비율[비율 < 0.9].round(3).to_dict()}")

    손계산(자료, 결과, 부속)
    미래정보점검(자료, 결과)
    외부대조(결과)
    덮는범위(자료, 결과, 부속)
    모양보고(결과)

    결과.reset_index().to_parquet(저장경로, index=False)
    다시 = pd.read_parquet(저장경로)
    확인(다시.equals(결과.reset_index()), "저장한 파일이 원래 표와 다름")
    print(f"\n[저장] {저장경로.name}: {len(결과)}개월({결과.index[0]}부터 {결과.index[-1]}까지) x {len(결과.columns)}열, 다시 읽어 일치 확인")


if __name__ == "__main__":
    실행()
