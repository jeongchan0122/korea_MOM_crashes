# -*- coding: utf-8 -*-
"""
10단계: 일별 시가총액 엑셀 3개와 일별 거래대금 엑셀 3개를 각각 이어 날짜 x 종목 표로 저장한다.

왜 필요한가
- 비유동성 지표(시장 비유동성, 비유동성 갭)의 재료다. 종목별 일별
  |수익률| / (거래대금 / 시가총액)(기본)과 |수익률| / 거래대금(Amihud)을 계산할 때 쓴다.
- 엑셀이 커서 한 번만 읽고 parquet로 저장해 둔다. 구조는 07번과 같다.

무엇을 하는가
- 시가총액: 50번(1999-2009), 51번(2010-2019), 52번(2020-2026)을 기간 순서대로 잇는다.
- 거래대금: 53번, 54번, 55번을 같은 방식으로 잇는다.
- 항목 코드, 집계주기, 종목 순서, 날짜를 확인하고, 07번의 일별수익률과 날짜·종목 순서가 같은지 본다.
  하나라도 다르면 저장하지 않고 멈춘다.
- 월별 시가총액과의 일치, 거래대금이 0인 날, 수익률은 있는데 값이 없는 칸, 연도별 거래대금 합계를
  출력한다(멈추지 않음).

하지 않는 일
- 지표 계산, 극단값 처리, 종목 필터. 값은 원본 그대로 저장한다.
- 원본 엑셀 수정.

실행: python scripts/10_일별시총거래대금정리.py   (먼저 07번, 04번까지 실행해 두어야 함)
결과: 처리데이터/일별시가총액.parquet (백만원), 처리데이터/일별거래대금.parquet (원)
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from python_calamine import CalamineWorkbook

프로젝트 = Path(__file__).resolve().parent.parent
원본폴더 = 프로젝트 / "한국데이터"
처리폴더 = 프로젝트 / "처리데이터"

# (저장 이름, 항목 코드, 항목 이름, 기간 순서대로 적은 파일들)
항목들 = [
    ("일별시가총액", "S410001250", "시가총액(백만원)",
     ["50_전종목_일간시가총액_1999_2009.xlsx",
      "51_전종목_일간시가총액_2010_2019.xlsx",
      "52_전종목_일간시가총액_2020_2026.xlsx"]),
    ("일별거래대금", "S410000900", "거래대금(원)",
     ["53_전종목_일간거래대금_1999_2009.xlsx",
      "54_전종목_일간거래대금_2010_2019.xlsx",
      "55_전종목_일간거래대금_2020_2026.xlsx"]),
]

# DataGuide 일별 파일의 줄 위치(0부터 셈). 07번과 같다.
코드줄, 항목코드줄, 항목명줄, 집계주기줄, 데이터시작줄 = 8, 11, 12, 13, 14


class 정리문제(Exception):
    """확인에 실패했을 때 쓰는 오류. 이 오류가 나면 저장하지 않고 멈춘다."""


def 확인(조건, 설명):
    if not 조건:
        raise 정리문제(설명)


def 파일읽기(이름, 항목코드, 항목명):
    """엑셀 하나를 날짜 x 종목 표로 읽고 머리글을 확인한다."""
    시트 = CalamineWorkbook.from_path(str(원본폴더 / 이름)).get_sheet_by_name("Sheet1")
    줄들 = 시트.to_python(skip_empty_area=False)

    코드 = [str(값) for 값 in 줄들[코드줄][1:]]
    확인(len(코드) == len(set(코드)), f"{이름}: 종목 코드가 겹침")
    확인({str(값) for 값 in 줄들[항목코드줄][1:]} == {항목코드}, f"{이름}: 항목 코드가 {항목코드}가 아님")
    확인({str(값) for 값 in 줄들[항목명줄][1:]} == {항목명}, f"{이름}: 항목 이름이 {항목명}이 아님")
    확인({str(값) for 값 in 줄들[집계주기줄][1:]} == {"일간"}, f"{이름}: 집계주기가 일간이 아님")

    데이터 = 줄들[데이터시작줄:]
    날짜 = pd.to_datetime([줄[0] for 줄 in 데이터])
    # 빈칸은 글자("")로 읽히므로 숫자로 바꾸면서 결측이 된다.
    원표 = pd.DataFrame([줄[1:] for 줄 in 데이터], index=날짜, columns=코드)
    표 = 원표.apply(pd.to_numeric, errors="coerce").astype("float64")
    # 숫자로 바꾸지 못한 글자가 있으면 값이 사라지므로 멈춘다.
    글자칸 = int((원표.ne("") & 원표.notna()).sum().sum())
    확인(int(표.notna().sum().sum()) == 글자칸, f"{이름}: 숫자로 바꾸지 못한 값이 있음")
    표.index.name = "날짜"
    확인(표.index.is_monotonic_increasing and 표.index.is_unique, f"{이름}: 날짜 순서가 맞지 않거나 겹침")
    print(f"  {이름}: {표.index[0].date()}부터 {표.index[-1].date()}까지, {len(표):,}거래일, 값 {int(표.notna().sum().sum()):,}칸")
    return 표


def 항목정리(저장이름, 항목코드, 항목명, 파일들, 수익률):
    """한 항목의 파일 3개를 잇고 확인한 뒤 저장한다."""
    print(f"\n[{저장이름}: {항목명}, {항목코드}]")
    표들 = [파일읽기(이름, 항목코드, 항목명) for 이름 in 파일들]

    for 표 in 표들[1:]:
        확인(list(표.columns) == list(표들[0].columns), f"{저장이름}: 파일끼리 종목 순서가 다름")
    for 앞, 뒤 in zip(표들, 표들[1:]):
        확인(앞.index[-1] < 뒤.index[0], f"{저장이름}: 파일끼리 날짜가 겹침")

    전체 = pd.concat(표들)
    확인(전체.index.equals(수익률.index), f"{저장이름}: 날짜가 일별수익률과 다름")
    확인(list(전체.columns) == list(수익률.columns), f"{저장이름}: 종목 순서가 일별수익률과 다름")
    확인(not (전체 < 0).any().any(), f"{저장이름}: 음수 값이 있음")

    저장경로 = 처리폴더 / f"{저장이름}.parquet"
    전체.to_parquet(저장경로)
    다시 = pd.read_parquet(저장경로)
    확인(다시.equals(전체), f"{저장이름}: 저장한 파일이 원본 표와 다름")
    print(f"  [저장] {저장경로.name}: {len(전체):,}거래일 x {전체.shape[1]:,}종목, "
          f"값 {int(전체.notna().sum().sum()):,}칸, 다시 읽어 일치 확인")
    return 전체


def 월별시총비교(시총, 패널):
    """월말 거래일의 일별 시가총액이 월별 파일의 같은 달 값과 같은지 본다."""
    월말 = 시총.groupby(시총.index.to_period("M")).tail(1)
    월말.index = 월말.index.to_period("M").astype(str)
    일별 = 월말.stack().rename("일별").rename_axis(["연월", "종목코드"])
    월별 = 패널.dropna(subset=["시가총액_백만원"]).set_index(["연월", "종목코드"])["시가총액_백만원"].rename("월별")
    합 = pd.concat([일별, 월별], axis=1, join="inner")
    차이 = (합["일별"] / 합["월별"] - 1).abs()
    같음 = 차이 <= 1e-6
    print("\n[월별 시가총액과 비교] 월말 거래일의 일별 값 대 02번 월별 값")
    print(f"  둘 다 값이 있는 종목·월 {len(합):,}개 중 같음 {int(같음.sum()):,}개({같음.mean() * 100:.3f}%)")
    print(f"  월별에만 값 {len(월별.index.difference(합.index)):,}개, 일별에만 값 {len(일별.index.difference(합.index)):,}개")
    if (~같음).any():
        예 = 합[~같음].assign(차이_퍼센트=차이[~같음] * 100).sort_values("차이_퍼센트", ascending=False)
        print(f"  다른 칸의 차이: 중앙값 {예['차이_퍼센트'].median():.4f}%, 최대 {예['차이_퍼센트'].max():.2f}%")
        print("  차이가 큰 예 5개:")
        print(예.head(5).to_string(float_format=lambda x: f"{x:,.2f}"))


def 거래대금점검(수익률, 시총, 거래대금, 패널):
    """거래대금이 0이거나 없는 날, 수익률은 있는데 값이 없는 칸을 센다."""
    _명세 = importlib.util.spec_from_file_location("WML계산", Path(__file__).resolve().parent / "05_WML계산.py")
    월별코드 = importlib.util.module_from_spec(_명세)
    _명세.loader.exec_module(월별코드)
    대상 = 월별코드.투자대상표(패널)

    있음 = 수익률.notna()
    print("\n[수익률은 있는데 값이 없는 칸] (수익률이 있는 칸 기준)")
    print(f"  수익률이 있는 칸 {int(있음.sum().sum()):,}개 중 시가총액 없음 {int((있음 & 시총.isna()).sum().sum()):,}개, "
          f"거래대금 없음 {int((있음 & 거래대금.isna()).sum().sum()):,}개")

    # 그 달 말에 투자 대상인 종목의 그 달 거래일. 비유동성은 이 종목들로 계산한다.
    달 = pd.Series(수익률.index.to_period("M").astype(str), index=수익률.index)
    대상칸 = pd.DataFrame(False, index=수익률.index, columns=수익률.columns)
    열번호 = {c: i for i, c in enumerate(수익률.columns)}
    값 = 대상칸.values
    for 연월, g in 대상.groupby("구성연월"):
        행 = np.flatnonzero(달.values == 연월)
        열 = [열번호[c] for c in g["종목코드"] if c in 열번호]
        if len(행) and 열:
            값[np.ix_(행, 열)] = True
    대상칸 = pd.DataFrame(값, index=수익률.index, columns=수익률.columns) & 있음

    n = int(대상칸.sum().sum())
    영 = 대상칸 & (거래대금 == 0)
    없음 = 대상칸 & 거래대금.isna()
    print(f"\n[거래대금 0인 날] 그 달 말 투자 대상 종목의 그 달 거래일 {n:,}칸 기준")
    print(f"  거래대금 0: {int(영.sum().sum()):,}칸({영.sum().sum() / n * 100:.3f}%), "
          f"그중 수익률도 0: {int((영 & (수익률 == 0)).sum().sum()):,}칸")
    print(f"  거래대금 없음: {int(없음.sum().sum()):,}칸, 시가총액 없음: {int((대상칸 & 시총.isna()).sum().sum()):,}칸")
    연도별 = (영.sum(axis=1).groupby(영.index.year).sum() / 대상칸.sum(axis=1).groupby(대상칸.index.year).sum() * 100)
    print("  연도별 거래대금 0 비율(%): " + ", ".join(f"{y} {v:.2f}" for y, v in 연도별.items()))

    # 한 달에 거래대금이 있는 날이 10일 미만인 종목·월 (지표 계산 때 빠질 후보)
    유효 = (대상칸 & (거래대금 > 0) & 시총.notna()).groupby(달.values).sum()
    대상수 = 대상칸.groupby(달.values).sum()
    부족 = ((대상수 > 0) & (유효 < 10)).sum().sum()
    print(f"  거래대금이 있는 날이 한 달 10일 미만인 투자 대상 종목·월: {int(부족):,}개 / {int((대상수 > 0).sum().sum()):,}개")


def 연도별합계(거래대금, 시총):
    """연도별 일평균 시장 전체 거래대금과 시가총액. 단위가 중간에 바뀌지 않았는지 본다."""
    일합 = pd.DataFrame({"거래대금_조원": 거래대금.sum(axis=1) / 1e12, "시가총액_조원": 시총.sum(axis=1) / 1e6})
    연 = 일합.groupby(일합.index.year).mean()
    연["회전율_%"] = 연["거래대금_조원"] / 연["시가총액_조원"] * 100
    print("\n[연도별 일평균 시장 전체 합계] (모든 종목, 투자 대상 필터 없음)")
    print(연.to_string(float_format=lambda x: f"{x:,.2f}"))


def 실행():
    수익률 = pd.read_parquet(처리폴더 / "일별수익률.parquet")
    패널 = pd.read_parquet(처리폴더 / "월별패널.parquet")
    결과 = {저장이름: 항목정리(저장이름, 코드, 이름, 파일들, 수익률) for 저장이름, 코드, 이름, 파일들 in 항목들}
    시총, 거래대금 = 결과["일별시가총액"], 결과["일별거래대금"]

    월별시총비교(시총, 패널)
    거래대금점검(수익률, 시총, 거래대금, 패널)
    연도별합계(거래대금, 시총)


if __name__ == "__main__":
    실행()
