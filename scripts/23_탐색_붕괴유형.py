# -*- coding: utf-8 -*-
"""
23단계 (탐색, 검증 결과를 본 뒤): 붕괴 달을 승자 급락형과 패자 급등형으로 나누고, 단독 신호 16개가
붕괴 직전 달 말에 켜져 있었는지 유형별로 센다. 서술 통계이며 모델을 고르지 않는다.

무엇을 하는가 (연구 기록 20.2)
- 붕괴 달: 2000-07부터 2026-09까지(315개월) 원래 2x3 WML(비용 전) 하위 10%.
- 유형: WML = 승자 쪽 - 패자 쪽. 승자 기여 = -승자 수익률, 패자 기여 = +패자 수익률.
  승자 기여가 크면 "승자 급락형", 아니면 "패자 급등형". 그 달 시장 수익률도 적는다.
- 포착률: 붕괴 달 m에 대해 m-1월 말 신호(적용연월 = m)가 켜졌는지. 신호가 있는 붕괴 달만 센다.
  비교: 붕괴가 아닌 달(신호가 있는 달) 중 켜진 비율. 최종 모델(위험 점수) 신호도 한 줄 더한다.

실행: python scripts/23_탐색_붕괴유형.py   (먼저 15번까지 실행해 두어야 함)
결과: 결과/탐색/붕괴달.csv, 결과/탐색/붕괴유형_포착.csv
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

프로젝트 = Path(__file__).resolve().parent.parent
처리폴더 = 프로젝트 / "처리데이터"
결과폴더 = 프로젝트 / "결과" / "탐색"
시작, 끝 = "2000-07", "2026-09"
붕괴비율 = 0.10


def 불러오기(이름, 파일):
    명세 = importlib.util.spec_from_file_location(이름, Path(__file__).resolve().parent / 파일)
    모듈 = importlib.util.module_from_spec(명세)
    명세.loader.exec_module(모듈)
    return 모듈


검증코드 = 불러오기("검증", "15_검증.py")


def 실행():
    월 = pd.read_parquet(처리폴더 / "WML월별수익률.parquet")
    월 = 월[월["방식"] == "2x3"].set_index("연월").loc[시작:끝]
    확인 = (월["롱온리_%"] - 월["패자쪽_%"] - 월["롱숏_%"]).abs().max()
    assert 확인 < 1e-9 and len(월) == 315, (확인, len(월))

    n = int(round(붕괴비율 * len(월)))
    붕괴 = 월.nsmallest(n, "롱숏_%").sort_index().copy()
    붕괴["승자기여"] = -붕괴["롱온리_%"]
    붕괴["패자기여"] = 붕괴["패자쪽_%"]
    붕괴["유형"] = np.where(붕괴["승자기여"] > 붕괴["패자기여"], "승자 급락형", "패자 급등형")
    붕괴["시장방향"] = np.where(붕괴["시장_%"] >= 0, "상승", "하락")

    단독 = pd.read_parquet(처리폴더 / "신호월별.parquet").set_index("적용연월").drop(columns="연월")
    _, _, 표, 단독연월, _ = 검증코드.자료읽기()
    최종 = 검증코드.신호만들기(표, 단독연월, 검증코드.기대최종)
    최종.index = [str(pd.Period(m) + 1) for m in 최종.index]
    단독["최종모델(위험점수)"] = 최종.reindex(단독.index)
    신호들 = list(단독.columns)
    for s in 신호들:
        붕괴[s] = 단독[s].reindex(붕괴.index).values

    print(f"[붕괴 달] 하위 {붕괴비율:.0%}: {n}개월 (기준 {붕괴['롱숏_%'].max():.2f}% 이하)")
    print(붕괴.groupby(["유형", "시장방향"]).size().unstack(fill_value=0))
    with pd.option_context("display.width", 200):
        print(붕괴[["롱숏_%", "롱온리_%", "패자쪽_%", "시장_%", "유형"]].round(1).to_string())

    줄들 = []
    아님 = ~월.index.isin(붕괴.index)
    for s in 신호들:
        평상 = 단독[s].reindex(월.index[아님])
        평상 = 평상[평상.notna()]
        줄 = {"신호": s, "평상_달수": len(평상), "평상_켜짐비율": 평상.mean()}
        for 유형 in ("전체", "승자 급락형", "패자 급등형"):
            g = 붕괴 if 유형 == "전체" else 붕괴[붕괴["유형"] == 유형]
            v = g[s].dropna()
            줄[f"{유형}_달수"] = len(v)
            줄[f"{유형}_포착"] = int(v.sum())
            줄[f"{유형}_포착률"] = v.mean() if len(v) else np.nan
        줄들.append(줄)
    포착 = pd.DataFrame(줄들)
    with pd.option_context("display.width", 250):
        print(포착.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    결과폴더.mkdir(parents=True, exist_ok=True)
    붕괴.to_csv(결과폴더 / "붕괴달.csv", encoding="utf-8-sig")
    포착.to_csv(결과폴더 / "붕괴유형_포착.csv", index=False, encoding="utf-8-sig")
    print(f"\n[저장] {결과폴더}")


if __name__ == "__main__":
    실행()
