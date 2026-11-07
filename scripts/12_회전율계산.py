# -*- coding: utf-8 -*-
"""
12단계: 2x3 WML의 달별 회전율을 실제 종목 비중으로 계산한다. 거래비용에 쓴다.

왜 필요한가
- 전략과 원래 WML에서 거래비용(편도 0.5% x 회전율)을 빼려면 매달 얼마나 사고팔았는지 알아야 한다.

무엇을 하는가 (연구 기록 16.3, 16.5)
- t월 말 새 비중: 승자 쪽 SW, BW에 각각 +0.5, 패자 쪽 SL, BL에 각각 -0.5. 바구니 안에서는 t월 말 시가총액으로 나눈다.
  다음 달 수익률이 없는 종목은 05번 WML 계산과 같게 뺀다. 이 비중으로 계산한 수익률이 05번 WML과 같은지 확인한다.
- 변한 비중: t월 비중이 t+1월 수익률로 변한 값. 승자 쪽과 패자 쪽을 각각 합이 1이 되도록 다시 맞춘다.
  t+1월에 상장폐지된 종목은 현금이 된 것으로 보고 0으로 둔다(매도 비용을 세지 않음).
- 보유 달 m의 비용에 쓰는 회전율 세 가지(m-1월 말의 결정에 따른 매매):
  리밸런싱 = sum |m-1월 말 새 비중 - m-2월 말 비중이 m-1월에 변한 값|  (비중 1 -> 1)
  청산     = sum |m-2월 말 비중이 m-1월에 변한 값|                    (비중 1 -> 0)
  구축     = sum |m-1월 말 새 비중| = 2                                 (비중 0 -> 1)

하지 않는 일
- 성과 계산, 신호. 2x3만 계산한다(다른 방식은 강건성 단계).

실행: python scripts/12_회전율계산.py   (먼저 05번까지 실행해 두어야 함)
결과: 처리데이터/WML회전율.parquet (보유 달 기준)
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

프로젝트 = Path(__file__).resolve().parent.parent
처리폴더 = 프로젝트 / "처리데이터"
저장경로 = 처리폴더 / "WML회전율.parquet"

_명세 = importlib.util.spec_from_file_location("WML계산", Path(__file__).resolve().parent / "05_WML계산.py")
월별코드 = importlib.util.module_from_spec(_명세)
_명세.loader.exec_module(월별코드)
시가총액 = 월별코드.시가총액

선택기간 = ("2005-07", "2015-12")
쪽 = {"SW": 1, "BW": 1, "SL": -1, "BL": -1}


class 회전율문제(Exception):
    """확인에 실패했을 때 쓰는 오류. 이 오류가 나면 저장하지 않고 멈춘다."""


def 확인(조건, 설명):
    if not 조건:
        raise 회전율문제(설명)


def 보유표(패널):
    """구성 달별 2x3 승자·패자 종목, 새 비중, 다음 달 수익률, 상폐 여부."""
    대상 = 월별코드.투자대상표(패널)
    표 = 대상.merge(월별코드.모멘텀점수(패널), on=["구성연월", "종목코드"], how="inner")
    표 = 월별코드.다음달수익률붙이기(월별코드.바구니나누기(표), 패널)
    표 = 표[표["2x3"].isin(쪽) & 표["수익률"].notna()].copy()
    상폐 = 패널.set_index(["연월", "종목코드"])["상폐월"]
    표["상폐"] = 상폐.reindex(pd.MultiIndex.from_arrays([표["연월"], 표["종목코드"]])).fillna(False).astype(bool).values
    합 = 표.groupby(["구성연월", "2x3"])[시가총액].transform("sum")
    표["비중"] = 표["2x3"].map(쪽) * 0.5 * 표[시가총액] / 합
    return 표


def 변한비중(g):
    """한 구성 달의 비중이 다음 달 수익률로 변한 값. 쪽마다 합이 1(패자 쪽은 -1)이 되게 맞추고 상폐 종목은 0."""
    값 = g["비중"] * (1 + g["수익률"] / 100)
    결과 = pd.Series(0.0, index=g.index)
    for 부호 in (1, -1):
        안 = np.sign(g["비중"]) == 부호
        결과[안] = 값[안] / abs(값[안].sum())   # 승자 쪽 합 +1, 패자 쪽 합 -1
    결과[g["상폐"]] = 0.0
    return pd.Series(결과.values, index=g["종목코드"].values)


def 계산(표):
    구성달들 = sorted(표["구성연월"].unique())
    새 = {m: g.set_index("종목코드")["비중"] for m, g in 표.groupby("구성연월")}
    변 = {m: 변한비중(g) for m, g in 표.groupby("구성연월")}
    줄들 = []
    for 앞, 뒤 in zip(구성달들, 구성달들[1:]):
        # 뒤 = m-1월 말 결정, 앞 = m-2월 말 비중이 m-1월(= 뒤)에 변한 값
        확인(str(pd.Period(앞) + 1) == 뒤, f"구성 달이 이어지지 않음: {앞}, {뒤}")
        a, b = 새[뒤], 변[앞]
        모두 = a.index.union(b.index)
        리밸 = float((a.reindex(모두, fill_value=0) - b.reindex(모두, fill_value=0)).abs().sum())
        줄들.append({"연월": str(pd.Period(뒤) + 1), "리밸런싱": 리밸, "청산": float(b.abs().sum()),
                   "구축": float(a.abs().sum()), "종목수": len(a)})
    return pd.DataFrame(줄들)


def 실행():
    패널 = pd.read_parquet(처리폴더 / "월별패널.parquet")
    월WML = pd.read_parquet(처리폴더 / "WML월별수익률.parquet")
    월WML = 월WML[월WML["방식"] == "2x3"].set_index("연월")["롱숏_%"]
    표 = 보유표(패널)

    # 확인 1: 이 비중으로 계산한 수익률이 05번 WML과 같다.
    재계산 = (표["비중"] * 표["수익률"]).groupby(표["연월"]).sum()
    차이 = (재계산 - 월WML.reindex(재계산.index)).abs()
    확인(차이.max() < 1e-9, f"비중으로 계산한 수익률이 05번 WML과 다름: 최대 {차이.max()}")
    print(f"[비중 확인] 비중 x 수익률의 합이 05번 2x3 WML과 같음({len(재계산)}개월, 최대 차이 {차이.max():.1e}%p)")
    # 확인 2: 쪽마다 새 비중의 합이 +1, -1
    쪽합 = 표.groupby(["구성연월", np.sign(표["비중"])])["비중"].sum()
    확인(np.allclose(쪽합.abs(), 1.0), "쪽마다 비중 합이 1이 아님")

    결과 = 계산(표)
    확인(np.allclose(결과["구축"], 2.0), "구축 회전율이 2가 아님")
    확인((결과["리밸런싱"] >= 0).all() and (결과["리밸런싱"] <= 4 + 1e-9).all(), "리밸런싱 회전율이 0부터 4까지 범위 밖")
    확인((결과["청산"] <= 2 + 1e-9).all(), "청산 회전율이 2를 넘음")

    print(f"\n[회전율] 보유 달 {결과['연월'].iloc[0]}부터 {결과['연월'].iloc[-1]}까지 {len(결과)}개월 "
          "(승자 쪽과 패자 쪽 합, 최대 4)")
    for 이름, (a, b) in (("전체", (결과["연월"].iloc[0], 결과["연월"].iloc[-1])), ("선택 기간", 선택기간)):
        구 = 결과[결과["연월"].between(a, b)]
        print(f"  {이름}: 리밸런싱 평균 {구['리밸런싱'].mean():.3f}(중앙값 {구['리밸런싱'].median():.3f}, "
              f"최소 {구['리밸런싱'].min():.3f}, 최대 {구['리밸런싱'].max():.3f}), 청산 평균 {구['청산'].mean():.3f}")
        print(f"    편도 0.5%면 원래 WML의 월평균 비용 {구['리밸런싱'].mean() * 0.5:.3f}%p, 연 {구['리밸런싱'].mean() * 0.5 * 12:.2f}%p")
    상폐수 = int(표["상폐"].sum())
    print(f"  상장폐지로 0이 된 종목·월: {상폐수}개")

    결과.to_parquet(저장경로, index=False)
    확인(pd.read_parquet(저장경로).equals(결과), "저장한 파일이 원래 표와 다름")
    print(f"\n[저장] {저장경로.name}: {len(결과)}개월")


if __name__ == "__main__":
    실행()
