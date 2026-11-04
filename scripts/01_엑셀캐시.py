# -*- coding: utf-8 -*-
"""
1단계: DataGuide 엑셀 13개를 parquet 복사본으로 저장한다.

왜 필요한가
- 엑셀이 커서 읽는 데 오래 걸린다. 한 번만 읽어 복사본을 만들어 두면 다음 단계부터 빠르다.

무엇을 하는가
- 표 모양은 그대로 두고, 모든 칸을 글자로 바꿔 저장한다.
- 글자로 바꾸는 이유: 한 열에 머리글 글자와 숫자가 섞여 있어 그대로는 parquet로 저장할 수 없다.
- 저장한 복사본을 다시 읽어 행 수, 열 수, 값이 찬 칸 수가 원본과 같은지 비교한다.

하지 않는 일
- 머리글 분리, 숫자 변환, 종목 필터, 상폐 수익률 계산은 하지 않는다.
- 원본 엑셀은 읽기만 하고 수정하지 않는다.

실행: python scripts/01_엑셀캐시.py
결과: 처리데이터/원본캐시/<엑셀 파일 이름>__<시트 이름>.parquet
"""
import sys
import time
from pathlib import Path

import pandas as pd
from python_calamine import CalamineWorkbook

# 이 코드 파일의 위치를 기준으로 폴더를 찾는다(scripts의 한 단계 위가 프로젝트 폴더).
프로젝트 = Path(__file__).resolve().parent.parent
원본폴더 = 프로젝트 / "한국데이터"
저장폴더 = 프로젝트 / "처리데이터" / "원본캐시"

# 복사할 엑셀 13개.
대상파일 = [
    # WML 재료 10개
    "02_시가총액.xlsx",
    "03_월수익률_배당포함.xlsx",
    "04_거래정지.xlsx",
    "05_관리종목.xlsx",
    "06_시장구분.xlsx",
    "29_상장일_상폐일.xlsx",
    "40_상폐종목_일간수정주가_1999_2009.xlsx",
    "41_상폐종목_일간수정주가_2010_2019.xlsx",
    "42_상폐종목_일간수정주가_2020_2026.xlsx",
    "43_누락상폐종목_일간수정주가.xlsx",
    # 완성한 WML과 비교할 FnGuide 지수 1개
    "08_FF팩터지수.xlsx",
    # 붕괴 예측 지표 후보용 일별 지수 2개(시장 변동성, 모멘텀 변동성)
    "44_지수_일간.xlsx",
    "45_FF팩터지수_일간.xlsx",
]


def 글자로(값):
    """칸 하나를 글자로 바꾼다. 빈칸은 글자로 바꾸지 않고 None(빈 값)으로 둔다."""
    if 값 is None or 값 == "":
        return None
    return str(값)


def 시트별_읽기(경로):
    """엑셀의 모든 시트를 읽어 {시트 이름: 글자로 된 표} 형태로 돌려준다."""
    엑셀 = CalamineWorkbook.from_path(str(경로))
    시트들 = {}
    for 시트이름 in 엑셀.sheet_names:
        # skip_empty_area=False: 빈 영역도 건너뛰지 않아 행 위치가 원본과 같게 유지된다.
        행들 = 엑셀.get_sheet_by_name(시트이름).to_python(skip_empty_area=False)

        # 표의 너비는 가장 긴 행에 맞추고, 짧은 행의 뒤는 빈 값으로 채운다.
        너비 = max((len(행) for 행 in 행들), default=0)
        자료 = [[글자로(값) for 값 in 행] + [None] * (너비 - len(행)) for 행 in 행들]

        # 열 이름은 c0, c1, c2, ... 로 붙인다(엑셀의 A, B, C에 해당).
        시트들[시트이름] = pd.DataFrame(
            자료, columns=[f"c{i}" for i in range(너비)], dtype="object"
        )
    return 시트들


def 실행():
    # 파일이 없거나 이름이 다르면 일부만 만들지 않도록, 시작 전에 먼저 확인한다.
    없는파일 = [이름 for 이름 in 대상파일 if not (원본폴더 / 이름).exists()]
    if 없는파일:
        sys.exit(f"원본 폴더에 없는 파일: {없는파일}")

    저장폴더.mkdir(parents=True, exist_ok=True)
    모두일치 = True

    print("복사본 | 원본 행×열 | 복사본 행×열 | 원본 찬 칸 | 복사본 찬 칸 | 결과 | 걸린 시간")

    for 이름 in 대상파일:
        시작 = time.time()
        for 시트이름, 원본표 in 시트별_읽기(원본폴더 / 이름).items():
            저장경로 = 저장폴더 / f"{Path(이름).stem}__{시트이름}.parquet"
            원본표.to_parquet(저장경로, index=False)

            # 메모리의 표가 아니라, 저장된 파일을 다시 읽어서 비교한다.
            복사본표 = pd.read_parquet(저장경로)
            원본칸 = int(원본표.notna().sum().sum())
            복사본칸 = int(복사본표.notna().sum().sum())
            일치 = 원본표.shape == 복사본표.shape and 원본칸 == 복사본칸
            모두일치 = 모두일치 and 일치

            print(f"{저장경로.stem} | "
                  f"{원본표.shape[0]:,}×{원본표.shape[1]:,} | "
                  f"{복사본표.shape[0]:,}×{복사본표.shape[1]:,} | "
                  f"{원본칸:,} | {복사본칸:,} | "
                  f"{'일치' if 일치 else '불일치'} | {time.time() - 시작:.1f}초", flush=True)

    print("\n전체 결과:", "모두 일치" if 모두일치 else "불일치가 있음. 위 표를 확인할 것")
    if not 모두일치:
        sys.exit(1)  # 불일치가 있으면 실패로 끝내서 그냥 넘어가지 않게 한다.


if __name__ == "__main__":
    실행()
