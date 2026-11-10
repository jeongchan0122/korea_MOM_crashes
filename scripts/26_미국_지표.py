# -*- coding: utf-8 -*-
"""
26단계 (미국 선택·검증, 연구 기록 22절): 미국 자료로 위험 지표 10개(정의 변형 포함)와 단독 신호 16개를 만든다.
정의는 11번, 13번과 같고, 미국에서 바뀌는 것만 다르다.

- 시장: CRSP 가치가중 지수(월별 msi, 일별 dsi의 vwretd).
- 모멘텀 갭: t월 말 투자 대상 중 점수가 있는 종목의 log(1 + 점수) 75번째 - 25번째 분위수.
- 시장 변동성, 모멘텀 변동성, 왜도, 모멘텀 베타: 126거래일(24번의 일별 WML과 일별 시장).
- 약세장(24, 18개월), 금융난류(시장, 승자, 패자, 24개월 뒤부터), CI A부터 D까지: 11번과 같음.
- 비유동성: 일별 |수익률| / 회전율(거래량 / 상장주식수), Amihud = |수익률| / 거래대금(백만 달러).
  거래량 0, 수익률 없음, |수익률| 35% 초과인 날을 빼고 10일 미만이면 값 없음. NASDAQ 거래량은 Gao & Ritter (2010)
  보정(2001-01까지 2.0, 2001-02부터 2001-12까지 1.8, 2002년부터 2003년까지 1.6으로 나눔).
  Amihud는 종목별 월 값에 (전달 말 투자 대상 전체 시가총액 / 1962-12 말 값)을 곱한다.
  시장 비유동성 = t월 말 시가총액 가중평균. 비유동성 갭 = 형성기간(t-11월부터 t-1월까지, 6개월 이상) 평균의
  2x3 바구니별 시총가중 -> 승자 평균 - 패자 평균.

실행: python scripts/26_미국_지표.py   (먼저 24번을 실행해 두어야 함)
결과: 처리데이터/미국/지표월별.parquet, 처리데이터/미국/신호월별.parquet
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

프로젝트 = Path(__file__).resolve().parent.parent
원본 = 프로젝트 / "미국데이터"
자료폴더 = 프로젝트 / "처리데이터" / "미국"


def 불러오기(이름, 파일):
    명세 = importlib.util.spec_from_file_location(이름, Path(__file__).resolve().parent / 파일)
    모듈 = importlib.util.module_from_spec(명세)
    명세.loader.exec_module(모듈)
    return 모듈


미국 = 불러오기("미국WML", "24_미국_WML.py")
신호코드 = 불러오기("신호", "13_신호만들기.py")
끝달 = 미국.끝달
창, 극단값, 최소일수, 형성개월, 최소형성개월, 난류최소 = 126, 0.35, 10, 11, 6, 24
아미기준달 = "1962-12"


def 확인(조건, 설명):
    if not 조건:
        raise SystemExit(f"[멈춤] {설명}")


def 대상과바구니(패널):
    """구성 달마다 투자 대상(시총), 점수, 2x3 바구니(다음 달 수익률과 무관)."""
    넓 = 패널.pivot(index="연월", columns="permno", values="ret").sort_index()
    넓 = 넓.reindex(pd.period_range(넓.index.min(), 넓.index.max(), freq="M").astype(str))
    점수 = np.expm1(np.log1p(넓).shift(1).rolling(11, min_periods=11).sum())
    점수 = 점수.stack().rename("점수").rename_axis(["구성연월", "permno"]).reset_index()
    대상 = 패널[패널["shrcd"].isin([10, 11]) & 패널["exchcd"].isin([1, 2, 3]) & (패널["시총"] > 0)
              & (패널["연월"] <= 끝달)][["연월", "permno", "시총", "exchcd"]].rename(columns={"연월": "구성연월"})
    표 = 대상.merge(점수, on=["구성연월", "permno"], how="inner")
    묶음 = []
    for 구성, g in 표.groupby("구성연월"):
        nyse = g[g["exchcd"] == 1]
        중앙 = nyse["시총"].median()
        낮음, 높음 = nyse["점수"].quantile(0.3), nyse["점수"].quantile(0.7)
        묶음.append(g.assign(바구니=np.where(g["시총"] <= 중앙, "S", "B")
                            + np.select([g["점수"] <= 낮음, g["점수"] >= 높음], ["L", "W"], "M")))
    return 대상, 표, pd.concat(묶음, ignore_index=True)


def 나스닥배수(날짜):
    return np.select([날짜 < "2001-02-01", 날짜 < "2002-01-01", 날짜 < "2004-01-01"], [2.0, 1.8, 1.6], 1.0)


def 종목월비유동성(패널):
    거래소 = 패널.set_index(["permno", "연월"])["exchcd"]
    조각 = []
    for 연도 in range(1962, int(끝달[:4]) + 1):
        d = pd.read_parquet(원본 / "daily" / f"dsf_{연도}.parquet", columns=["permno", "date", "ret", "prc", "vol", "shrout"])
        if d.empty:
            continue
        d["date"] = pd.to_datetime(d["date"])
        d["연월"] = d["date"].dt.to_period("M").astype(str)
        d["exchcd"] = 거래소.reindex(pd.MultiIndex.from_arrays([d["permno"], d["연월"]])).values
        나스닥 = d["exchcd"] == 3
        d.loc[나스닥, "vol"] = d.loc[나스닥, "vol"] / 나스닥배수(d.loc[나스닥, "date"].values.astype("datetime64[D]").astype(str))
        유효 = d["ret"].notna() & (d["vol"] > 0) & (d["shrout"] > 0) & d["prc"].notna() & (d["ret"].abs() <= 극단값)
        d = d[유효]
        절대 = d["ret"].abs()
        d = d.assign(회전=절대 / (d["vol"] / (d["shrout"] * 1000)), 아미=절대 / (d["prc"].abs() * d["vol"] / 1e6))
        g = d.groupby(["연월", "permno"]).agg(회전=("회전", "mean"), 아미=("아미", "mean"), 일수=("회전", "size"))
        조각.append(g[g["일수"] >= 최소일수])
        print(f"  비유동성 {연도} 완료", flush=True)
    전체 = pd.concat(조각)
    회전월 = 전체["회전"].unstack()
    아미월 = 전체["아미"].unstack()
    달력 = pd.period_range(회전월.index.min(), 회전월.index.max(), freq="M").astype(str)
    return 회전월.reindex(달력), 아미월.reindex(달력)


def 가중평균(값, 가중):
    있음 = np.isfinite(값) & np.isfinite(가중) & (가중 > 0)
    return np.average(값[있음], weights=가중[있음]) if 있음.any() else np.nan


def 계산():
    패널 = 미국.패널만들기()
    대상, 표, 바구니 = 대상과바구니(패널)
    달들 = sorted(대상["구성연월"].unique())
    월 = pd.read_parquet(자료폴더 / "월별WML.parquet").set_index("연월")
    일 = pd.read_parquet(자료폴더 / "일별WML.parquet").set_index("날짜")
    msi = pd.read_csv(원본 / "crsp_msi.csv")
    msi["연월"] = pd.to_datetime(msi["date"]).dt.to_period("M").astype(str)
    시장월 = msi.set_index("연월")["vwretd"] * 100

    결과 = pd.DataFrame(index=pd.Index(달들, name="연월"))
    결과["적용연월"] = (pd.PeriodIndex(결과.index, freq="M") + 1).astype(str)
    결과["모멘텀갭"] = 표.groupby("구성연월")["점수"].apply(lambda s: np.log1p(s).quantile(0.75) - np.log1p(s).quantile(0.25))

    # 일별 지표
    달 = 일.index.to_period("M").astype(str)
    굴림 = lambda s: s.rolling(창, min_periods=창)
    w, m = 일["롱숏_%"], 일["시장_%"]
    for 이름, 값 in (("시장변동성", 굴림(m).std()), ("모멘텀변동성", 굴림(w).std()), ("왜도", 굴림(w).skew()),
                   ("모멘텀베타", 굴림(w).cov(m) / 굴림(m).var())):
        결과[이름] = 값.groupby(달).last()

    # 약세장, CI 재료
    로그 = np.log1p(시장월 / 100)
    for 개월 in (24, 18):
        누적 = np.expm1(로그.rolling(개월, min_periods=개월).sum()) * 100
        결과[f"약세장누적{개월}"] = 누적
        결과[f"약세장{개월}"] = (누적 < 0).where(누적.notna())
    결과["시장수익률"] = 시장월
    결과["전월시장수익률"] = 시장월.shift(1)
    결과 = 결과.loc[달들]

    # 금융난류: (시장, 승자, 패자) 실현 달 기준
    y = 월[["시장_%", "롱온리_%", "패자쪽_%"]].dropna()
    난류 = pd.Series(np.nan, index=y.index)
    for i in range(난류최소, len(y)):
        과거 = y.iloc[:i].values
        차 = y.iloc[i].values - 과거.mean(axis=0)
        난류.iloc[i] = 차 @ np.linalg.solve(np.cov(과거, rowvar=False), 차)
    결과["금융난류"] = 난류

    # 비유동성
    회전월, 아미월 = 종목월비유동성(패널)
    총시총 = 대상.groupby("구성연월")["시총"].sum()
    배수 = pd.Series({mm: 총시총.get(str(pd.Period(mm) - 1), np.nan) for mm in 아미월.index}) / 총시총.loc[아미기준달]
    아미월 = 아미월.mul(배수, axis=0)
    형성 = {k: v.shift(1).rolling(형성개월, min_periods=최소형성개월).mean() for k, v in (("회전율", 회전월), ("Amihud", 아미월))}
    for 이름, 월값 in (("회전율", 회전월), ("Amihud", 아미월)):
        시장값, 갭값 = {}, {}
        for 구성 in 달들:
            g = 대상[대상["구성연월"] == 구성]
            v = 월값.loc[구성].reindex(g["permno"]).values if 구성 in 월값.index else np.full(len(g), np.nan)
            시장값[구성] = 가중평균(np.asarray(v, float), g["시총"].values.astype(float))
            b = 바구니[(바구니["구성연월"] == 구성) & 바구니["바구니"].isin(["SW", "BW", "SL", "BL"])]
            f = 형성[이름].loc[구성].reindex(b["permno"]).values if 구성 in 형성[이름].index else np.full(len(b), np.nan)
            값 = {k: 가중평균(np.asarray(f[(b["바구니"] == k).values], float), b.loc[b["바구니"] == k, "시총"].values.astype(float))
                 for k in ("SW", "BW", "SL", "BL")}
            갭값[구성] = (값["SW"] + 값["BW"]) / 2 - (값["SL"] + 값["BL"]) / 2
        결과[f"비유동성_{이름}"] = pd.Series(시장값)
        결과[f"비유동성갭_{이름}"] = pd.Series(갭값)

    상승, 하락전월, 음베타 = 결과["시장수익률"] > 0, 결과["전월시장수익률"] < 0, 결과["모멘텀베타"] < 0
    for 정의, 개월, 전전월 in (("A", 24, False), ("B", 18, False), ("C", 24, True), ("D", 18, True)):
        켜짐 = (결과[f"약세장{개월}"] == 1) & 상승 & 음베타
        if 전전월:
            켜짐 &= 하락전월
        필요 = 결과[[f"약세장{개월}", "시장수익률", "모멘텀베타"] + (["전월시장수익률"] if 전전월 else [])].notna().all(axis=1)
        결과[f"CI_{정의}"] = 켜짐.astype(float).where(필요)
    return 결과


def 실행():
    결과 = 계산()
    확인(list(결과.index) == list(pd.period_range(결과.index[0], 결과.index[-1], freq="M").astype(str)), "지표 달이 이어지지 않음")
    for c in ["모멘텀갭", "시장변동성", "모멘텀변동성", "왜도", "모멘텀베타", "금융난류", "비유동성_회전율",
              "비유동성_Amihud", "비유동성갭_회전율", "비유동성갭_Amihud", "약세장누적24", "CI_A"]:
        s = 결과[c].dropna()
        print(f"  {c:<14} {s.index[0]}부터, {len(s)}개월, 평균 {s.mean():.4g}, 중앙값 {s.median():.4g}")
    단독 = 신호코드.단독신호(결과)
    단독.insert(0, "적용연월", 결과["적용연월"])
    자료폴더.mkdir(parents=True, exist_ok=True)
    결과.reset_index().to_parquet(자료폴더 / "지표월별.parquet", index=False)
    단독.reset_index().to_parquet(자료폴더 / "신호월별.parquet", index=False)
    print("[단독 신호] 켜짐 비율(값 있는 달):")
    print((단독.drop(columns="적용연월").mean() * 100).round(1).to_string())
    print(f"[저장] {자료폴더}")


if __name__ == "__main__":
    실행()
