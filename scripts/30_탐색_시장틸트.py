# -*- coding: utf-8 -*-
"""
30단계 (탐색, 연구 기록 24절): 시장 비중 기반 모멘텀 틸트(MSCI 배수 방식)와 대형 승자에 한국 최종 모델 신호를 붙인다.
신호가 켜지면 다음 달 시장을 든다. 평가는 틸트 수익(포트폴리오 - 시장)으로 15번 평가 함수 그대로.

- 같은 표 안에서 시장과 틸트를 함께 계산한다: t월 말 투자 대상 중 t+1월 수익률이 있는 종목(05번과 같은 조건).
  시장 = 시가총액 가중. 틸트 = 시가총액 x 배수(점수 z > 0이면 1 + z, 아니면 1 / (1 - z), z는 단면 표준화 후 ±3에서 자름,
  점수 없는 종목은 배수 1). 대형 승자 = 2x3의 BW 바구니 시총가중.
- 비용(편도 0.5%): 유지 = sum|새 비중 - 변한 비중|, 포트폴리오 <-> 시장 = sum|새 쪽 비중 - 변한 비중|, 시장 유지 = 0.

실행: python scripts/30_탐색_시장틸트.py
결과: 결과/탐색/시장틸트/
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

프로젝트 = Path(__file__).resolve().parent.parent
결과폴더 = 프로젝트 / "결과" / "탐색" / "시장틸트"


def 불러오기(이름, 파일):
    명세 = importlib.util.spec_from_file_location(이름, Path(__file__).resolve().parent / 파일)
    모듈 = importlib.util.module_from_spec(명세)
    명세.loader.exec_module(모듈)
    return 모듈


검증코드 = 불러오기("검증", "15_검증.py")
선택코드 = 검증코드.선택코드
월별코드 = 불러오기("WML계산", "05_WML계산.py")
비용률 = 0.5


def 배수(점수):
    z = ((점수 - 점수.mean()) / 점수.std()).clip(-3, 3)
    return np.where(z > 0, 1 + z, 1 / (1 - z))


def 표만들기():
    패널 = pd.read_parquet(선택코드.처리폴더 / "월별패널.parquet")
    대상 = 월별코드.투자대상표(패널)
    점수 = 월별코드.모멘텀점수(패널)
    표 = 대상.merge(점수, on=["구성연월", "종목코드"], how="left")
    바 = 월별코드.바구니나누기(표[표["점수"].notna()])[["구성연월", "종목코드", "2x3"]]
    표 = 표.merge(바, on=["구성연월", "종목코드"], how="left")
    표 = 월별코드.다음달수익률붙이기(표, 패널)
    상폐 = 패널.set_index(["연월", "종목코드"])["상폐월"]
    표["상폐"] = 상폐.reindex(pd.MultiIndex.from_arrays([표["연월"], 표["종목코드"]])).fillna(False).astype(bool).values
    표 = 표[표["수익률"].notna()].copy()
    시총 = 표[월별코드.시가총액]
    표["M"] = 시총 / 시총.groupby(표["구성연월"]).transform("sum")
    표["배수"] = 1.0
    있음 = 표["점수"].notna()
    표.loc[있음, "배수"] = 표[있음].groupby("구성연월")["점수"].transform(lambda s: pd.Series(배수(s), index=s.index))
    t = 시총 * 표["배수"]
    표["T"] = t / t.groupby(표["구성연월"]).transform("sum")
    b = 시총.where(표["2x3"] == "BW", 0.0)
    표["B"] = b / b.groupby(표["구성연월"]).transform("sum")
    return 표


def 엔진(표, 보유달):
    """포트폴리오별 수익률(%)과, 쌍별 매매량."""
    새, 변 = {}, {}
    수익 = {k: {} for k in "MTB"}
    for m, g in 표.groupby("구성연월"):
        r = g["수익률"].values / 100
        for k in "MTB":
            w = g[k].values
            수익[k][str(pd.Period(m) + 1)] = float(w @ r) * 100
            v = w * (1 + r)
            v = np.where(g["상폐"].values, 0.0, v / v.sum()) if v.sum() > 0 else v
            새[(k, m)] = pd.Series(w, index=g["종목코드"].values)
            변[(k, m)] = pd.Series(v, index=g["종목코드"].values)
    수익 = pd.DataFrame(수익).reindex(보유달)

    def 차(a, b):
        모두 = a.index.union(b.index)
        return float((a.reindex(모두, fill_value=0) - b.reindex(모두, fill_value=0)).abs().sum())
    매매 = {}
    for m in 보유달:
        뒤, 앞 = str(pd.Period(m) - 1), str(pd.Period(m) - 2)
        줄 = {}
        for k in "TB":
            줄[f"{k}{k}"] = 차(새[(k, 뒤)], 변[(k, 앞)])
            줄[f"{k}M"] = 차(새[("M", 뒤)], 변[(k, 앞)])
            줄[f"M{k}"] = 차(새[(k, 뒤)], 변[("M", 앞)])
        매매[m] = 줄
    return 수익, pd.DataFrame(매매).T


def 전략(w, k, 수익, 매매):
    앞 = w.shift(1).fillna(1.0)
    회전 = np.select([(앞 == 1) & (w == 1), (앞 == 1) & (w == 0), (앞 == 0) & (w == 1)],
                   [매매[f"{k}{k}"], 매매[f"{k}M"], 매매[f"M{k}"]], 0.0)
    총 = w * 수익[k] + (1 - w) * 수익["M"] - 비용률 * 회전
    return 총 - 수익["M"]


def 실행():
    선택코드.비용률 = 비용률
    월, 회전, 지표, 단독, 보유달 = 검증코드.자료읽기()
    표 = 표만들기()
    수익, 매매 = 엔진(표, 보유달)
    전월 = pd.read_parquet(선택코드.처리폴더 / "WML월별수익률.parquet")
    파일시장 = 전월[전월["방식"] == "2x3"].set_index("연월")["시장_%"].reindex(보유달)
    print(f"[확인] 직접 계산한 시장과 05번 시장의 최대 차이 {(수익['M'] - 파일시장).abs().max():.2e}%p")
    # 손계산: 한 달
    달 = "2010-06"
    g = 표[표["구성연월"] == 달]
    손 = (g[월별코드.시가총액] * g["배수"] * g["수익률"]).sum() / (g[월별코드.시가총액] * g["배수"]).sum()
    print(f"[손계산 {달} 구성] 틸트 수익 코드 {수익.loc[str(pd.Period(달) + 1), 'T']:.6f}% 손 {손:.6f}%")
    # 크기 구성
    상위 = 표.groupby("구성연월", group_keys=False).apply(lambda g: g.nlargest(10, 월별코드.시가총액)[["M", "T", "B"]].sum())
    print(f"[크기 구성] 시총 상위 10종목의 평균 비중: 시장 {상위['M'].mean():.3f}, 틸트 {상위['T'].mean():.3f}, 대형 승자 {상위['B'].mean():.3f}")
    for k, 이름 in (("T", "틸트"), ("B", "대형 승자")):
        추적 = (수익[k] - 수익["M"]).std() * np.sqrt(12)
        print(f"  {이름}: 시장 대비 추적오차(비용 전) 연 {추적:.2f}%, 평균 매매량 유지 {매매[f'{k}{k}'].mean():.3f}, "
              f"-> 시장 {매매[f'{k}M'].mean():.3f}, 시장 -> {매매[f'M{k}'].mean():.3f}")

    w, _ = 검증코드.비중만들기(검증코드.신호만들기(지표, 단독, 검증코드.기대최종), 보유달)
    구간들 = {"선택": 검증코드.선택구간, "검증": 검증코드.검증구간, "2025-12까지": 검증코드.끊은구간}
    줄들 = []
    for k, 이름 in (("T", "시장 틸트(MSCI 배수)"), ("B", "대형 승자")):
        원 = 전략(pd.Series(1.0, index=보유달), k, 수익, 매매)
        모 = 전략(w, k, 수익, 매매)
        비용전 = 수익[k] - 수익["M"]
        for 구간이름, 구간 in 구간들.items():
            e = 검증코드.평가(모, 원, 비용전, w, 구간)
            줄들.append({"포트폴리오": 이름, "구간": 구간이름, "켜진달": e["켜진달"], "틸트_월평균": e["월평균"],
                       "원래_월평균": e["원래_월평균"], "원래_비용전": 비용전.loc[구간[0]:구간[1]].mean(),
                       "정보비율": e["샤프"], "원래_정보비율": e["원래_샤프"], "최대낙폭": e["최대낙폭"],
                       "원래_최대낙폭": e["원래_최대낙폭"], "사건합": e["사건합"], "원래_사건합": e["원래_사건합"],
                       "t": e["t"], "결론": e["결론"]})
    결과 = pd.DataFrame(줄들)
    with pd.option_context("display.width", 250):
        print(결과.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    결과폴더.mkdir(parents=True, exist_ok=True)
    결과.to_csv(결과폴더 / "시장틸트.csv", index=False, encoding="utf-8-sig")
    print(f"[저장] {결과폴더}")


if __name__ == "__main__":
    실행()
