# -*- coding: utf-8 -*-
"""
24단계 (미국 시험, 연구 기록 21절): CRSP로 미국 2x3 WML(월별, 일별)과 회전율을 만든다.

무엇을 하는가
- 투자 대상: 보통주(shrcd 10, 11), NYSE·AMEX·NASDAQ(exchcd 1, 2, 3). 시가총액 = |prc| x shrout.
- 상장폐지: 상폐 달 수익률 = (1 + ret)(1 + dlret) - 1. dlret이 없고 상폐 코드가 500번대면
  NYSE·AMEX -30%, NASDAQ -55%(Shumway 1997, Shumway & Warther 1999). msf에 상폐 달 줄이 없으면 만든다.
- 점수: t-11월부터 t-1월까지 누적수익률(11개월 모두 있어야 함, 한국 05번과 같음).
- 경계: 크기 중앙값, 모멘텀 30%·70%를 NYSE 종목(exchcd 1)으로. 시가총액가중.
  t+1월 수익률이 없는 종목은 뺀다(한국 05번과 같음).
- 일별: t월 말 바구니를 t+1월 동안 고정, 매수 후 보유(비중이 주가에 따라 변함). 빈 날의 수익률은 0.
  일별 시장 = CRSP 가치가중 지수(dsi의 vwretd).
- 회전율: 12번과 같은 방식(리밸런싱, 청산, 구축).
- 확인: 비중 x 수익률 = 월별 WML, Fama-French UMD와 상관 0.95 이상, 일별을 한 달 복리한 값과 월별 WML의 상관.

실행: python scripts/24_미국_WML.py
결과: 처리데이터/미국/(월별WML, 일별WML, 회전율 parquet)
"""
from pathlib import Path

import numpy as np
import pandas as pd

프로젝트 = Path(__file__).resolve().parent.parent
원본 = 프로젝트 / "미국데이터"
저장폴더 = 프로젝트 / "처리데이터" / "미국"
끝달 = "2024-12"
쪽 = {"SW": 1, "BW": 1, "SL": -1, "BL": -1}


class 미국문제(Exception):
    pass


def 확인(조건, 설명):
    if not 조건:
        raise 미국문제(설명)


# ---------------------------------------------------------------- 월별 패널

def 패널만들기():
    msf = pd.read_csv(원본 / "crsp_msf.csv", usecols=["permno", "date", "ret", "prc", "shrout"])
    msf["date"] = pd.to_datetime(msf["date"])
    msf["연월"] = msf["date"].dt.to_period("M").astype(str)
    확인(not msf.duplicated(["permno", "연월"]).any(), "msf에 같은 종목·달이 두 번 있음")

    이름 = pd.read_csv(원본 / "crsp_msenames.csv", usecols=["permno", "namedt", "nameendt", "shrcd", "exchcd"])
    이름["namedt"] = pd.to_datetime(이름["namedt"])
    이름["nameendt"] = pd.to_datetime(이름["nameendt"])

    상폐 = pd.read_csv(원본 / "crsp_msedelist.csv")
    상폐["dlstdt"] = pd.to_datetime(상폐["dlstdt"])
    상폐 = 상폐[상폐["dlstdt"].notna()].copy()
    상폐["연월"] = 상폐["dlstdt"].dt.to_period("M").astype(str)
    상폐 = 상폐[상폐["연월"] <= 끝달]

    # 상폐 달 줄이 msf에 없으면 만든다(수익률은 상폐 수익률만으로).
    없는줄 = 상폐.merge(msf[["permno", "연월"]], on=["permno", "연월"], how="left", indicator=True)
    없는줄 = 없는줄[없는줄["_merge"] == "left_only"][["permno", "연월", "dlstdt"]].rename(columns={"dlstdt": "date"})
    msf = pd.concat([msf, 없는줄.assign(ret=np.nan, prc=np.nan, shrout=np.nan)], ignore_index=True)

    # 그 날짜의 shrcd, exchcd
    msf = msf.sort_values("date")
    이름 = 이름.sort_values("namedt")
    msf = pd.merge_asof(msf, 이름, left_on="date", right_on="namedt", by="permno", direction="backward")
    msf.loc[msf["date"] > msf["nameendt"] + pd.Timedelta(days=31), ["shrcd", "exchcd"]] = np.nan
    msf = msf.sort_values(["permno", "date"]).reset_index(drop=True)
    msf["exchcd"] = msf.groupby("permno")["exchcd"].ffill()
    msf["shrcd"] = msf.groupby("permno")["shrcd"].ffill()

    # 상폐 수익률 합치기
    msf = msf.merge(상폐[["permno", "연월", "dlret", "dlstcd"]], on=["permno", "연월"], how="left")
    성과상폐 = msf["dlstcd"].between(500, 599) & msf["dlret"].isna()
    대체 = np.where(msf["exchcd"] == 3, -0.55, -0.30)
    dlret = msf["dlret"].where(~성과상폐, 대체)
    상폐있음 = msf["dlstcd"].notna()
    합 = np.where(msf["ret"].notna() & dlret.notna(), (1 + msf["ret"]) * (1 + dlret) - 1,
                 np.where(msf["ret"].isna(), dlret, msf["ret"]))
    msf["보유수익률"] = np.where(상폐있음, 합, msf["ret"])
    msf["상폐"] = 상폐있음
    msf["시총"] = msf["prc"].abs() * msf["shrout"]
    print(f"[패널] 종목·월 {len(msf):,}줄, 상폐 줄 {int(상폐있음.sum()):,}개, "
          f"그중 dlret 없어 -30%·-55% 넣은 줄 {int(성과상폐.sum()):,}개")
    return msf[["permno", "연월", "date", "ret", "보유수익률", "시총", "shrcd", "exchcd", "상폐"]]


def 바구니표(패널):
    """구성 달 t의 2x3 바구니, 비중, t+1월 수익률."""
    넓 = 패널.pivot(index="연월", columns="permno", values="ret").sort_index()
    달력 = pd.period_range(넓.index.min(), 넓.index.max(), freq="M").astype(str)
    넓 = 넓.reindex(달력)
    점수 = np.expm1(np.log1p(넓).shift(1).rolling(11, min_periods=11).sum())
    점수 = 점수.stack().rename("점수").rename_axis(["구성연월", "permno"]).reset_index()

    대상 = 패널[패널["shrcd"].isin([10, 11]) & 패널["exchcd"].isin([1, 2, 3]) & (패널["시총"] > 0)
              & (패널["연월"] <= 끝달)]
    표 = 대상[["연월", "permno", "시총", "exchcd"]].rename(columns={"연월": "구성연월"}).merge(
        점수, on=["구성연월", "permno"], how="inner")

    묶음들 = []
    for 구성, g in 표.groupby("구성연월"):
        nyse = g[g["exchcd"] == 1]
        if len(nyse) < 20:
            continue
        중앙 = nyse["시총"].median()
        낮음, 높음 = nyse["점수"].quantile(0.3), nyse["점수"].quantile(0.7)
        이름 = (np.where(g["시총"] <= 중앙, "S", "B")
              + np.select([g["점수"] <= 낮음, g["점수"] >= 높음], ["L", "W"], "M"))
        묶음들.append(g.assign(바구니=이름))
    표 = pd.concat(묶음들, ignore_index=True)
    표 = 표[표["바구니"].isin(쪽)].copy()

    보유 = 패널.set_index(["연월", "permno"])[["보유수익률", "상폐"]]
    표["연월"] = (pd.PeriodIndex(표["구성연월"], freq="M") + 1).astype(str)
    붙임 = 보유.reindex(pd.MultiIndex.from_arrays([표["연월"], 표["permno"]]))
    표["수익률"] = 붙임["보유수익률"].values
    표["상폐"] = 붙임["상폐"].fillna(False).astype(bool).values
    표 = 표[표["수익률"].notna() & (표["연월"] <= 끝달)].copy()
    합 = 표.groupby(["구성연월", "바구니"])["시총"].transform("sum")
    표["비중"] = 표["바구니"].map(쪽) * 0.5 * 표["시총"] / 합
    return 표


def 월별WML(표, 패널):
    바 = 표.groupby(["연월", "바구니"]).apply(lambda g: np.average(g["수익률"], weights=g["시총"]),
                                         include_groups=False).unstack()
    확인(바[list(쪽)].notna().all().all(), "빈 바구니가 있는 달이 있음")
    승, 패 = 바[["SW", "BW"]].mean(axis=1), 바[["SL", "BL"]].mean(axis=1)
    msi = pd.read_csv(원본 / "crsp_msi.csv")
    msi["연월"] = pd.to_datetime(msi["date"]).dt.to_period("M").astype(str)
    시장 = msi.set_index("연월")["vwretd"]
    월 = pd.DataFrame({"롱숏_%": (승 - 패) * 100, "롱온리_%": 승 * 100, "패자쪽_%": 패 * 100,
                      "시장_%": 시장.reindex(승.index) * 100, "종목수": 표.groupby("연월").size()})
    재계산 = (표["비중"] * 표["수익률"]).groupby(표["연월"]).sum() * 100
    확인((재계산 - 월["롱숏_%"]).abs().max() < 1e-9, "비중 x 수익률이 월별 WML과 다름")
    return 월


def UMD대조(월):
    ff = pd.read_csv(원본 / "ff_factors_monthly.csv")
    ff["연월"] = pd.to_datetime(ff["date"]).dt.to_period("M").astype(str)
    umd = ff.set_index("연월")["umd"] * 100
    공통 = 월.index.intersection(umd.dropna().index)
    상관 = 월.loc[공통, "롱숏_%"].corr(umd.loc[공통])
    print(f"[UMD 대조] {공통[0]}부터 {공통[-1]}까지 {len(공통)}개월: 상관 {상관:.4f}, "
          f"월평균 우리 {월.loc[공통, '롱숏_%'].mean():.3f}% 대 UMD {umd.loc[공통].mean():.3f}%, "
          f"표준편차 {월.loc[공통, '롱숏_%'].std():.3f}% 대 {umd.loc[공통].std():.3f}%")
    확인(상관 >= 0.95, f"UMD와의 상관이 0.95 미만: {상관:.4f}")


# ---------------------------------------------------------------- 일별

def 일별WML(표):
    결과 = []
    표 = 표.assign(연도=표["연월"].str[:4])
    for 연도, 연표 in 표.groupby("연도"):
        파일 = 원본 / "daily" / f"dsf_{연도}.parquet"
        일 = pd.read_parquet(파일, columns=["permno", "date", "ret"])
        일["date"] = pd.to_datetime(일["date"])
        일["연월"] = 일["date"].dt.to_period("M").astype(str)
        일 = 일[일["permno"].isin(연표["permno"].unique())]
        for 달, 달표 in 연표.groupby("연월"):
            넓 = 일[일["연월"] == 달].pivot_table(index="date", columns="permno", values="ret", aggfunc="first")
            확인(len(넓) > 0, f"{달}: 일별 자료가 없음")
            누적 = (1 + 넓.fillna(0)).cumprod()
            바구니수익 = {}
            for 바, g in 달표.groupby("바구니"):
                시 = g.set_index("permno")["시총"]
                가치 = 누적.reindex(columns=시.index).fillna(1.0).ffill() * 시.values
                총 = 가치.sum(axis=1)
                바구니수익[바] = 총 / pd.concat([pd.Series([시.sum()]), 총.iloc[:-1]]).values - 1
            b = pd.DataFrame(바구니수익)
            결과.append(pd.DataFrame({"롱숏_%": ((b["SW"] + b["BW"]) / 2 - (b["SL"] + b["BL"]) / 2) * 100}))
        print(f"  일별 {연도} 완료", flush=True)
    일WML = pd.concat(결과).sort_index()
    dsi = pd.read_parquet(원본 / "daily" / "dsi.parquet", columns=["date", "vwretd"])
    dsi["date"] = pd.to_datetime(dsi["date"])
    일WML["시장_%"] = dsi.set_index("date")["vwretd"].reindex(일WML.index) * 100
    확인(일WML.notna().all().all(), "일별 WML이나 시장에 빈 날이 있음")
    return 일WML.rename_axis("날짜")


# ---------------------------------------------------------------- 회전율 (12번과 같은 방식)

def 회전율(표):
    새 = {m: g.set_index("permno")["비중"] for m, g in 표.groupby("구성연월")}
    변 = {}
    for m, g in 표.groupby("구성연월"):
        값 = g["비중"] * (1 + g["수익률"])
        r = pd.Series(0.0, index=g.index)
        for 부호 in (1, -1):
            안 = np.sign(g["비중"]) == 부호
            r[안] = 값[안] / abs(값[안].sum())
        r[g["상폐"]] = 0.0
        변[m] = pd.Series(r.values, index=g["permno"].values)
    달들 = sorted(새)
    줄들 = []
    for 앞, 뒤 in zip(달들, 달들[1:]):
        확인(str(pd.Period(앞) + 1) == 뒤, f"구성 달이 이어지지 않음: {앞}, {뒤}")
        a, b = 새[뒤], 변[앞]
        모두 = a.index.union(b.index)
        줄들.append({"연월": str(pd.Period(뒤) + 1),
                   "리밸런싱": float((a.reindex(모두, fill_value=0) - b.reindex(모두, fill_value=0)).abs().sum()),
                   "청산": float(b.abs().sum()), "구축": float(a.abs().sum())})
    return pd.DataFrame(줄들).set_index("연월")


def 실행():
    패널 = 패널만들기()
    표 = 바구니표(패널)
    월 = 월별WML(표, 패널)
    print(f"[월별 WML] {월.index[0]}부터 {월.index[-1]}까지 {len(월)}개월, 평균 종목 {월['종목수'].mean():.0f}개, "
          f"월평균 {월['롱숏_%'].mean():.3f}%, 최악 {월['롱숏_%'].min():.2f}% ({월['롱숏_%'].idxmin()})")
    UMD대조(월)
    회전 = 회전율(표)
    print(f"[회전율] 리밸런싱 평균 {회전['리밸런싱'].mean():.3f}, 청산 평균 {회전['청산'].mean():.3f}")
    일WML = 일별WML(표)
    복리 = (1 + 일WML["롱숏_%"] / 100).groupby(일WML.index.to_period("M").astype(str)).prod().sub(1).mul(100)
    공통 = 복리.index.intersection(월.index)
    상관 = 복리.loc[공통].corr(월.loc[공통, "롱숏_%"])
    print(f"[일별 WML] {일WML.index[0].date()}부터 {일WML.index[-1].date()}까지 {len(일WML):,}일, "
          f"한 달 복리와 월별 WML 상관 {상관:.4f}, 평균 차이 {(복리.loc[공통] - 월.loc[공통, '롱숏_%']).mean():.3f}%p")
    확인(상관 >= 0.95, f"일별 복리와 월별 WML의 상관이 0.95 미만: {상관:.4f}")

    저장폴더.mkdir(parents=True, exist_ok=True)
    월.rename_axis("연월").reset_index().to_parquet(저장폴더 / "월별WML.parquet", index=False)
    일WML.reset_index().to_parquet(저장폴더 / "일별WML.parquet", index=False)
    회전.reset_index().to_parquet(저장폴더 / "회전율.parquet", index=False)
    print(f"[저장] {저장폴더}")


if __name__ == "__main__":
    실행()
