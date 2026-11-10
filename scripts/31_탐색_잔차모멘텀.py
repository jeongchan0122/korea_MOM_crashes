# -*- coding: utf-8 -*-
"""
31단계 (탐색, 연구 기록 25절): 잔차 모멘텀 틸트(Blitz, Huij & Martens 2011, 시장 모형판)를 보통 모멘텀 틸트(24절)와 비교한다.

- 잔차 점수: t월 말, t-35월부터 t월까지 36개월 월 수익률을 시장 수익률에 회귀(36개월 모두 있어야 함),
  t-11월부터 t-1월까지 11개월 잔차의 합 / 그 표준편차.
- 틸트: 30번과 같은 MSCI 배수 방식(시가총액 x 배수). 점수가 없는 종목은 배수 1.
- 시험 기간 2005-07부터 2026-09까지, 비용 편도 0.5%(30번 매매량).
- 판정(미리 고정): 비용 후 시장 대비 월평균의 Newey-West t(시차 4) >= 3 찾음, 1.96 이상 약한 증거, 그 아래 없음.

실행: python scripts/31_탐색_잔차모멘텀.py
결과: 결과/탐색/잔차모멘텀/
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

프로젝트 = Path(__file__).resolve().parent.parent
결과폴더 = 프로젝트 / "결과" / "탐색" / "잔차모멘텀"
시작, 끝 = "2005-07", "2026-09"


def 불러오기(이름, 파일):
    명세 = importlib.util.spec_from_file_location(이름, Path(__file__).resolve().parent / 파일)
    모듈 = importlib.util.module_from_spec(명세)
    명세.loader.exec_module(모듈)
    return 모듈


틸트코드 = 불러오기("시장틸트", "30_탐색_시장틸트.py")
월별코드 = 틸트코드.월별코드
선택코드 = 틸트코드.선택코드


def NWt(x, 시차=4):
    x = np.asarray(x, float)
    e = x - x.mean()
    T = len(x)
    S = (e @ e) / T
    for L in range(1, 시차 + 1):
        S += 2 * (1 - L / (시차 + 1)) * (e[L:] @ e[:-L]) / T
    return x.mean() / np.sqrt(S / T)


def 잔차점수():
    패널 = pd.read_parquet(선택코드.처리폴더 / "월별패널.parquet")
    넓 = 패널.pivot(index="연월", columns="종목코드", values=월별코드.점수열수익률).sort_index()
    달력 = pd.period_range(넓.index.min(), 넓.index.max(), freq="M").astype(str)
    넓 = 넓.reindex(달력)
    전월 = pd.read_parquet(선택코드.처리폴더 / "WML월별수익률.parquet")
    시장 = 전월[전월["방식"] == "2x3"].set_index("연월")["시장_%"].reindex(달력)
    결과 = {}
    for i in range(35, len(달력)):
        t = 달력[i]
        m = 시장.iloc[i - 35:i + 1].values
        if np.isnan(m).any():
            continue
        R = 넓.iloc[i - 35:i + 1]
        R = R.loc[:, R.notna().all()]
        if R.shape[1] == 0:
            continue
        r = R.values
        mc = m - m.mean()
        베타 = (mc @ (r - r.mean(axis=0))) / (mc @ mc)
        알파 = r.mean(axis=0) - 베타 * m.mean()
        잔차 = r - 알파 - np.outer(m, 베타)
        형성 = 잔차[24:35]               # t-11월부터 t-1월까지
        sd = 형성.std(axis=0, ddof=1)
        점수 = np.where(sd > 0, 형성.sum(axis=0) / np.where(sd > 0, sd, 1), np.nan)
        결과[t] = pd.Series(점수, index=R.columns)
    s = pd.concat(결과, names=["구성연월", "종목코드"]).rename("잔차점수").reset_index()
    return s, 넓, 시장


def 손계산(넓, 시장, 점수표):
    t, 종목 = "2012-06", None
    i = list(넓.index).index(t)
    R = 넓.iloc[i - 35:i + 1]
    종목 = R.columns[R.notna().all()][0]
    y, x = R[종목].values, 시장.iloc[i - 35:i + 1].values
    b, a = np.polyfit(x, y, 1)
    e = (y - a - b * x)[24:35]
    손 = e.sum() / e.std(ddof=1)
    코드 = 점수표.set_index(["구성연월", "종목코드"]).loc[(t, 종목), "잔차점수"]
    print(f"[손계산] {t} {종목}: 코드 {코드:.10f}, 손 {손:.10f}")
    assert abs(손 - 코드) < 1e-8


def 실행():
    선택코드.비용률 = 틸트코드.비용률
    점수표, 넓, 시장 = 잔차점수()
    손계산(넓, 시장, 점수표)
    표 = 틸트코드.표만들기()
    표 = 표.merge(점수표, on=["구성연월", "종목코드"], how="left")
    배 = pd.Series(1.0, index=표.index)
    있음 = 표["잔차점수"].notna()
    배[있음] = 표[있음].groupby("구성연월")["잔차점수"].transform(lambda s: pd.Series(틸트코드.배수(s), index=s.index))
    시총 = 표[월별코드.시가총액]
    r = 시총 * 배
    표["B"] = r / r.groupby(표["구성연월"]).transform("sum")   # 30번 엔진의 "B" 자리를 잔차 틸트로 쓴다
    보유달 = list(pd.period_range(시작, 끝, freq="M").astype(str))
    덮음 = 표[표["구성연월"] >= "2005-06"].groupby("구성연월").apply(lambda g: g.loc[g["잔차점수"].notna(), "M"].sum())
    print(f"[덮는 비율] 잔차 점수가 있는 종목의 시총 비중 평균 {덮음.mean():.3f}(최소 {덮음.min():.3f})")

    수익, 매매 = 틸트코드.엔진(표, 보유달)
    상위 = 표[표["구성연월"] >= "2005-06"].groupby("구성연월").apply(
        lambda g: g.nlargest(10, 월별코드.시가총액)[["M", "T", "B"]].sum())
    print(f"[크기 구성] 시총 상위 10종목 평균 비중: 시장 {상위['M'].mean():.3f}, 보통 틸트 {상위['T'].mean():.3f}, 잔차 틸트 {상위['B'].mean():.3f}")
    하나 = pd.Series(1.0, index=보유달)
    액티브 = {"보통 틸트": 틸트코드.전략(하나, "T", 수익, 매매), "잔차 틸트": 틸트코드.전략(하나, "B", 수익, 매매)}
    for k, 이름 in (("T", "보통 틸트"), ("B", "잔차 틸트")):
        print(f"  {이름}: 추적오차(비용 전) 연 {(수익[k] - 수익['M']).std() * np.sqrt(12):.2f}%, 평균 매매량 {매매[f'{k}{k}'].mean():.3f}")
    print(f"  보통 틸트와 잔차 틸트 액티브 수익의 상관 {액티브['보통 틸트'].corr(액티브['잔차 틸트']):.3f}")

    줄들 = []
    구간들 = {"전체(판정)": (시작, 끝), "2016년 전": (시작, "2015-12"), "2016년부터": ("2016-01", 끝)}
    for 구간이름, (a, b) in 구간들.items():
        for 이름, x in list(액티브.items()) + [("차이(잔차 - 보통)", 액티브["잔차 틸트"] - 액티브["보통 틸트"])]:
            s = x.loc[a:b]
            t = NWt(s.values)
            결론 = "찾음" if t >= 3 else ("약한 증거" if t >= 1.96 else "없음")
            줄들.append({"구간": 구간이름, "대상": 이름, "월평균_%": s.mean(), "정보비율": s.mean() / s.std() * np.sqrt(12),
                       "NW_t": t, "최대낙폭": 선택코드.낙폭(s).min(), "판정": 결론 if 이름 != "차이(잔차 - 보통)" else "-"})
    결과 = pd.DataFrame(줄들)
    print(결과.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    결과폴더.mkdir(parents=True, exist_ok=True)
    결과.to_csv(결과폴더 / "잔차모멘텀.csv", index=False, encoding="utf-8-sig")
    print(f"[저장] {결과폴더}")


if __name__ == "__main__":
    실행()
