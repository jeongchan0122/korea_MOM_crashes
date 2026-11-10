# -*- coding: utf-8 -*-
"""
28단계 (미국 선택·검증, 연구 기록 22.2): 27번을 한국과 같은 기간으로 다시 돌린다.
- 선택 기간 2005-07부터 2015-12까지, 검증 기간 2016-01부터 2024-12까지(미국 자료 끝).
- 자료 시작을 한국처럼 2000-01로 맞춘다: 지표 표를 2000-01부터 자르고, 금융난류(t-1월까지 평균·공분산)를
  2000-01부터의 월별 (시장, 승자, 패자)로 다시 계산하고, 단독 신호를 다시 만든다. 위험 점수의 z와 상관도
  13번 함수 안에서 2000-01부터 쌓인다.

실행: python scripts/28_미국_한국기간.py   (먼저 24번, 26번을 실행해 두어야 함)
결과: 결과/미국/한국기간/
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def 불러오기(이름, 파일):
    명세 = importlib.util.spec_from_file_location(이름, Path(__file__).resolve().parent / 파일)
    모듈 = importlib.util.module_from_spec(명세)
    명세.loader.exec_module(모듈)
    return 모듈


미국 = 불러오기("미국선택검증", "27_미국_선택검증.py")
자료시작 = "2000-01"
원래자료읽기 = 미국.자료읽기


def 자료읽기():
    월, 회전, 표, _ = 원래자료읽기()
    표 = 표.loc[자료시작:].copy()
    y = 월.loc[자료시작:, ["시장_%", "롱온리_%", "패자쪽_%"]].dropna()
    난류 = pd.Series(np.nan, index=y.index)
    for i in range(24, len(y)):
        과거 = y.iloc[:i].values
        차 = y.iloc[i].values - 과거.mean(axis=0)
        난류.iloc[i] = 차 @ np.linalg.solve(np.cov(과거, rowvar=False), 차)
    표["금융난류"] = 난류.reindex(표.index)
    단독 = 미국.신호코드.단독신호(표)
    print(f"[자료] 지표 {표.index[0]}부터, 금융난류 {난류.first_valid_index()}부터, "
          f"단독 신호 첫 달: 시장변동성 {단독['시장변동성'].first_valid_index()}, 금융난류 {단독['금융난류'].first_valid_index()}")
    return 월, 회전, 표, 단독


미국.자료읽기 = 자료읽기
미국.선택구간, 미국.검증구간 = ("2005-07", "2015-12"), ("2016-01", "2024-12")
미국.결과폴더 = 미국.프로젝트 / "결과" / "미국" / "한국기간"

if __name__ == "__main__":
    미국.실행()
