# -*- coding: utf-8 -*-
"""
8단계: 일별 WML과 일별 시장 수익률을 계산한다.

왜 필요한가
- 모멘텀 변동성, 왜도, 모멘텀 베타, 시장 변동성은 최근 126거래일의 일별 수익률로 계산한다.
- 전략과 지표의 종목을 맞추려고 FnGuide 지수 대신 직접 만든다.
- 일별 값은 지표 전용이다. 전략의 실제 수익률은 계속 기존 월별 WML을 쓴다.

계산 규칙 (팀 문서 8.3절의 "일별 WML 계산")
- 기간: 구성 1999-12부터 2026-08까지, 일별 수익률 2000-01부터 2026-09까지.
- 투자 대상과 바구니: 05_WML계산.py의 함수를 그대로 쓴다.
- 비중 드리프트: 구성 달 말 시가총액으로 시작해 한 달 동안 종목을 바꾸지 않고,
  종목별 값을 매일 수익률만큼 키운다. 바구니의 하루 수익률은 전날 값으로 가중 평균한다.
- 수익률이 없는 날: 그날 계산에서만 빼고 그 종목의 값은 그대로 둔다.
- 극단값: 모든 기간에 절댓값 31% 초과를 그날만 결측으로 둔다.
- 상폐 수익률: 일별에는 넣지 않는다. 상폐 손실은 월별 WML에 이미 반영되어 있다.

비교용으로 함께 계산하는 규칙 (저장하지 않고 보고만 함)
- 안 지움: 극단값을 그대로 둔다.
- 시기별: 시기별 가격제한폭 + 1%p를 넘으면 결측으로 둔다.

하지 않는 일
- 지표 계산, 신호, 모델 선택. 원본 엑셀과 기존 월별 결과 수정.

실행: python scripts/08_일별WML계산.py   (먼저 scripts/07_일별수익률정리.py를 실행해 두어야 함)
결과: 처리데이터/일별바구니수익률.parquet, 처리데이터/일별WML수익률.parquet
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

프로젝트 = Path(__file__).resolve().parent.parent
처리폴더 = 프로젝트 / "처리데이터"

# 파일 이름이 숫자로 시작해 일반 import를 쓸 수 없으므로 경로로 불러온다.
_명세 = importlib.util.spec_from_file_location("WML계산", Path(__file__).resolve().parent / "05_WML계산.py")
월별코드 = importlib.util.module_from_spec(_명세)
_명세.loader.exec_module(월별코드)

시가총액 = 월별코드.시가총액
방식들 = 월별코드.방식들
규칙들 = ["31%", "안지움", "시기별"]  # 첫 번째가 저장하는 기본 규칙이다.
기본규칙 = 규칙들[0]
극단값기준 = 31.0
멈춤차이 = 1.0  # 안 지운 2x3과 기존 월별 WML의 차이가 이 값(%p)을 넘는 달이 있으면 멈춘다.

# 시기별 비교 규칙의 경계. 2015-06-15는 기사로, 그 전 경계는 일별 수익률이 몰리는 값으로 확인했다.
코스닥확대일 = np.datetime64("2005-03-28")
제한폭확대일 = np.datetime64("2015-06-15")


class 계산문제(Exception):
    """확인에 실패했을 때 쓰는 오류. 이 오류가 나면 저장하지 않고 멈춘다."""


def 확인(조건, 설명):
    if not 조건:
        raise 계산문제(설명)


def 시기별기준(날짜들, 시장):
    """날짜 x 종목 크기의 기준값(%)을 돌려준다. 가격제한폭에 1%p를 더한 값이다."""
    날짜 = np.asarray(날짜들, dtype="datetime64[ns]")
    코스피 = np.where(날짜 >= 제한폭확대일, 31.0, 16.0)
    코스닥 = np.where(날짜 >= 제한폭확대일, 31.0, np.where(날짜 >= 코스닥확대일, 16.0, 13.0))
    기준 = np.where(시장[None, :] == "KS", 코스피[:, None], 코스닥[:, None])
    return 기준


def 규칙적용(값, 규칙, 날짜들, 시장):
    """규칙에 맞는 칸을 결측으로 바꾼 복사본을 돌려준다."""
    결과 = 값.copy()
    with np.errstate(invalid="ignore"):  # 결측과의 비교에서 나오는 경고만 끈다.
        if 규칙 == "31%":
            결과[np.abs(결과) > 극단값기준] = np.nan
        elif 규칙 == "시기별":
            결과[np.abs(결과) > 시기별기준(날짜들, 시장)] = np.nan
    return 결과


def 드리프트수익률(값, 비중):
    """비중 드리프트로 하루 수익률과 그날 계산에 들어간 종목 수를 구한다.
    값: 날짜 x 종목 수익률(%), 비중: 구성 달 말 시가총액."""
    현재값 = 비중.astype(float).copy()
    수익률 = np.full(len(값), np.nan)
    종목수 = np.zeros(len(값), dtype=int)
    for t, 오늘 in enumerate(값):
        있음 = np.isfinite(오늘)
        종목수[t] = 있음.sum()
        if 있음.any():
            수익률[t] = (현재값[있음] * 오늘[있음]).sum() / 현재값[있음].sum()
            현재값[있음] *= 1 + 오늘[있음] / 100  # 수익률이 없는 종목의 값은 그대로 둔다.
    return 수익률, 종목수


def 보유시장표(표, 시장구분):
    """시기별 기준에 쓸 시장. 보유 달이 코스피·코스닥이면 그 시장, 아니면 구성 달 시장을 쓴다."""
    구성 = 시장구분.reindex(pd.MultiIndex.from_arrays([표["구성연월"], 표["종목코드"]])).values
    다음달 = (pd.PeriodIndex(표["구성연월"], freq="M") + 1).astype(str)
    보유 = 시장구분.reindex(pd.MultiIndex.from_arrays([다음달, 표["종목코드"]])).values
    return np.where(np.isin(보유, ["KS", "KQ"]), 보유, 구성)


def 일별계산(대상, 표, 일별, 시장구분):
    """규칙마다 바구니와 시장의 일별 수익률을 계산하고, 기본 규칙에서 지운 칸도 모은다."""
    월표시 = 일별.index.strftime("%Y-%m")
    코드위치 = pd.Series(np.arange(일별.shape[1]), index=일별.columns)
    전체값 = 일별.values
    대상 = 대상.assign(시장=보유시장표(대상, 시장구분))

    바구니줄 = {규칙: [] for 규칙 in 규칙들}
    시장줄 = {규칙: [] for 규칙 in 규칙들}
    지운칸 = []
    칸합계 = 0

    표묶음 = {구성: g for 구성, g in 표.groupby("구성연월")}
    for 구성, 달대상 in 대상.groupby("구성연월"):
        보유 = str(pd.Period(구성, "M") + 1)
        날짜위치 = np.where(월표시 == 보유)[0]
        확인(len(날짜위치) > 0, f"{보유}: 일별 날짜가 없음")
        날짜들 = 일별.index[날짜위치]

        # 패널의 종목은 모두 일별 표에 있어야 한다.
        없는코드 = set(달대상["종목코드"]) - set(코드위치.index)
        확인(not 없는코드, f"{구성}: 일별 표에 없는 종목 {sorted(없는코드)[:5]}")
        원값 = 전체값[np.ix_(날짜위치, 코드위치[달대상["종목코드"]].values)]
        칸합계 += int(np.isfinite(원값).sum())
        비중 = 달대상[시가총액].values
        시장 = 달대상["시장"].values
        순서 = pd.Series(np.arange(len(달대상)), index=달대상["종목코드"].values)

        # 기본 규칙에서 지우는 칸을 기록한다(보고용).
        with np.errstate(invalid="ignore"):
            행, 열 = np.where(np.abs(원값) > 극단값기준)
        for a, b in zip(행, 열):
            지운칸.append({"날짜": 날짜들[a], "구성연월": 구성, "연월": 보유, "종목코드": 달대상["종목코드"].values[b],
                        "값": 원값[a, b]})

        표달 = 표묶음.get(구성)
        for 규칙 in 규칙들:
            값 = 규칙적용(원값, 규칙, 날짜들, 시장)
            수익률, _ = 드리프트수익률(값, 비중)
            시장줄[규칙].append(pd.Series(수익률, index=날짜들))
            if 표달 is None:
                continue
            for 방식 in 방식들:
                for 바구니, g in 표달.groupby(방식):
                    위치 = 순서[g["종목코드"]].values
                    수익률, 종목수 = 드리프트수익률(값[:, 위치], 비중[위치])
                    바구니줄[규칙].append(pd.DataFrame({"날짜": 날짜들, "연월": 보유, "방식": 방식, "바구니": 바구니,
                                                  "수익률_%": 수익률, "종목수": 종목수}))
    바구니 = {규칙: pd.concat(줄, ignore_index=True) for 규칙, 줄 in 바구니줄.items()}
    시장수익률 = {규칙: pd.concat(줄).sort_index().rename("시장_%") for 규칙, 줄 in 시장줄.items()}
    return 바구니, 시장수익률, pd.DataFrame(지운칸), 칸합계


def 승패(바구니, 시간열):
    """05_WML계산.py의 승패계산을 그대로 써서 시간열(날짜 또는 연월)별 WML을 만든다."""
    # 승패계산은 시간 열 이름을 "연월"로 받으므로, 날짜를 쓸 때는 원래 연월 열을 빼고 이름을 바꾼다.
    입력 = 바구니.drop(columns="연월").rename(columns={시간열: "연월"}) if 시간열 != "연월" else 바구니
    결과 =pd.concat([월별코드.승패계산(입력, 방식) for 방식 in 방식들], ignore_index=True)
    return 결과.rename(columns={"연월": 시간열}) if 시간열 != "연월" else 결과


def 월복리(바구니):
    """바구니 일별 수익률을 복리로 합쳐 월별 바구니 수익률을 만든다."""
    d = 바구니.assign(배=1 + 바구니["수익률_%"] / 100)
    결과 = d.groupby(["연월", "방식", "바구니"])["배"].prod().sub(1).mul(100).rename("수익률_%").reset_index()
    return 결과


def 비교줄(내것, 기존):
    """일별을 복리로 합친 값과 기존 월별 값을 비교한다."""
    차이 = 내것 - 기존
    return {"개월": len(내것), "상관": 내것.corr(기존), "월평균": 내것.mean(), "기존월평균": 기존.mean(),
            "최대차이": 차이.abs().max(), "최대차이달": 차이.abs().idxmax(), "1%p초과달": int((차이.abs() > 1).sum())}


def 지운칸보고(지운칸, 표, 일별, 패널, 칸합계):
    """기본 규칙에서 지운 칸의 수와 성격을 보고한다."""
    시기 = np.select([지운칸["연월"] <= "2005-03", 지운칸["연월"] <= "2015-06"], ["2005-03까지", "2005-04부터 2015-06까지"], "2015-07부터")
    방향 = np.where(지운칸["값"] > 0, "상승", "하락")
    지운칸 = 지운칸.assign(시기=시기, 방향=방향)
    print(f"\n[지운 칸: 절댓값 31% 초과] 투자 대상 {칸합계:,}칸 중 {len(지운칸):,}칸, 종목 {지운칸['종목코드'].nunique()}개")
    print(지운칸.groupby(["시기", "방향"]).size().unstack(fill_value=0).to_string())

    # 승자·패자 바구니에 속했는지와 바구니 안 구성 비중을 붙인다.
    승패이름 = ["SW", "BW", "SL", "BL"]
    합 = 표[표["2x3"].isin(승패이름)].copy()
    합["비중_%"] = 합[시가총액] / 합.groupby(["구성연월", "2x3"])[시가총액].transform("sum") * 100
    지운칸 = 지운칸.merge(합[["구성연월", "종목코드", "2x3", "비중_%"]], on=["구성연월", "종목코드"], how="left")
    print(f"  2x3 승자·패자 바구니 안: {int(지운칸['2x3'].notna().sum())}칸 "
          f"(상승 {int((지운칸['2x3'].notna() & (지운칸['방향'] == '상승')).sum())}, 하락 {int((지운칸['2x3'].notna() & (지운칸['방향'] == '하락')).sum())})")

    # 상폐와의 관계: 지운 날부터 그 종목의 마지막 수익률까지 남은 거래일을 센다.
    마지막날 = 일별.apply(lambda s: s.last_valid_index())
    날짜위치 = pd.Series(np.arange(len(일별)), index=일별.index)
    상폐 = pd.read_parquet(처리폴더 / "상폐수익률.parquet", columns=["종목코드", "상장폐지일", "출처"]).set_index("종목코드")
    지운칸["상폐종목"] = 지운칸["종목코드"].isin(상폐.index)
    지운칸["남은거래일"] = [날짜위치[마지막날[c]] - 날짜위치[d] for c, d in zip(지운칸["종목코드"], 지운칸["날짜"])]
    지운칸["상폐60일이내"] = 지운칸["상폐종목"] & (지운칸["남은거래일"] <= 60)

    # 그 달 월별 수익률과 비교한다. 월별에도 큰 움직임이 있으면 일별 값은 입력 오류가 아닐 가능성이 크다.
    월 = 패널.set_index(["연월", "종목코드"])["수정주가수익률_배당포함_상폐반영_%"]
    지운칸["월수익률"] = 월.reindex(pd.MultiIndex.from_arrays([지운칸["연월"], 지운칸["종목코드"]])).values
    이름 = 패널.drop_duplicates("종목코드", keep="last").set_index("종목코드")["종목명"]
    지운칸["종목명"] = 지운칸["종목코드"].map(이름)

    하락 = 지운칸[지운칸["방향"] == "하락"]
    print(f"\n[하락 칸 {len(하락)}개의 성격]")
    print(f"  상폐 종목 {int(하락['상폐종목'].sum())}칸, 그중 마지막 수익률까지 60거래일 이내 {int(하락['상폐60일이내'].sum())}칸")
    print(f"  −50% 미만 {int((하락['값'] < -50).sum())}칸, −90% 미만 {int((하락['값'] < -90).sum())}칸")
    print(f"  그 달 월별 수익률도 −31% 미만 {int((하락['월수익률'] < -31).sum())}칸, −31% 이상 {int((하락['월수익률'] >= -31).sum())}칸, 없음 {int(하락['월수익률'].isna().sum())}칸")

    표시열 = ["날짜", "종목코드", "종목명", "값", "월수익률", "2x3", "비중_%", "상폐종목", "남은거래일"]
    보기 = 지운칸.assign(날짜=지운칸["날짜"].dt.date)
    with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
        print("\n[구성 비중이 큰 칸 12개]")
        print(보기.sort_values("비중_%", ascending=False).head(12)[표시열].to_string(index=False))
        print("\n[값이 가장 낮은 칸 12개]")
        print(보기.sort_values("값").head(12)[표시열].to_string(index=False))
        print("\n[값이 가장 높은 칸 8개]")
        print(보기.sort_values("값", ascending=False).head(8)[표시열].to_string(index=False))
    print("\n[지운 칸이 많은 날짜] " + ", ".join(f"{d.date()} {n}칸" for d, n in 지운칸["날짜"].value_counts().head(8).items()))


def 종목대조(대상, 일별, 패널):
    """투자 대상 종목의 일별 복리와 월 수익률(상폐 미반영)을 비교한다."""
    월표시 = 일별.index.strftime("%Y-%m")
    복리 = ((1 + 일별.fillna(0) / 100).groupby(월표시).prod() - 1) * 100
    복리 = 복리.where(일별.notna().groupby(월표시).sum() > 0)
    긴 = 복리.stack().rename("일복리").rename_axis(["연월", "종목코드"]).reset_index()
    다음달 = (pd.PeriodIndex(대상["구성연월"], freq="M") + 1).astype(str)
    합 = 대상.assign(연월=다음달).merge(긴, on=["연월", "종목코드"]).merge(
        패널[["연월", "종목코드", "종목명", "수정주가수익률_배당포함_%"]], on=["연월", "종목코드"])
    합 = 합.dropna(subset=["수정주가수익률_배당포함_%"])
    합["차이"] = (합["일복리"] - 합["수정주가수익률_배당포함_%"]).abs()
    큰 = 합[합["차이"] > 1].sort_values("차이", ascending=False)
    print(f"\n[종목 대조: 일별 복리와 월 수익률, 투자 대상 {len(합):,}행] 1%p 초과 {len(큰)}행")
    if len(큰):
        with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
            print(큰.head(10)[["연월", "종목코드", "종목명", "일복리", "수정주가수익률_배당포함_%"]].to_string(index=False))


def 외부대조(일WML):
    """일별 시장을 코스피 지수와, 일별 2x3 WML을 FnGuide 일별 모멘텀 지수와 비교한다(보고만 함)."""
    일2x3 = 일WML[일WML["방식"] == "2x3"].set_index("날짜")
    지수 = pd.read_parquet(처리폴더 / "정리표" / "44_지수_일간.parquet").set_index("날짜")["I.001"]
    팩터 = pd.read_parquet(처리폴더 / "정리표" / "45_FF팩터지수_일간.parquet").set_index("날짜")["3FM.2M3.MOM"]
    코스피 = pd.to_numeric(지수, errors="coerce").pct_change(fill_method=None) * 100
    모멘텀 = pd.to_numeric(팩터, errors="coerce").pct_change(fill_method=None) * 100

    print("\n[외부 값과 대조] 기간 / 일수 / 일별 시장과 코스피 지수의 상관 / 일별 2x3 WML과 FnGuide 모멘텀의 상관")
    for 이름, 시작, 끝 in [("전체", "2000-07-01", "2026-09-30"), ("선택 기간", "2005-07-01", "2015-12-31"),
                       ("검증 기간", "2016-01-01", "2026-09-30")]:
        d = 일2x3.loc[시작:끝]
        print(f"  {이름}: {len(d):,} / {d['시장_%'].corr(코스피.reindex(d.index)):.3f} / {d['롱숏_%'].corr(모멘텀.reindex(d.index)):.3f}")


def 실행():
    패널 = pd.read_parquet(처리폴더 / "월별패널.parquet")
    일별 = pd.read_parquet(처리폴더 / "일별수익률.parquet")
    기존 = pd.read_parquet(처리폴더 / "WML월별수익률.parquet")

    # 월별과 똑같은 함수로 투자 대상과 바구니를 만든다.
    대상 = 월별코드.투자대상표(패널)
    표 = 월별코드.바구니나누기(대상.merge(월별코드.모멘텀점수(패널), on=["구성연월", "종목코드"], how="inner"))
    시장구분 = 패널.set_index(["연월", "종목코드"])["시장구분"]

    바구니, 시장수익률, 지운칸, 칸합계 = 일별계산(대상, 표, 일별, 시장구분)

    # ---------- 멈추는 조건 ----------
    for 규칙 in 규칙들:
        빈줄 = 바구니[규칙][바구니[규칙]["수익률_%"].isna()]
        확인(빈줄.empty, f"{규칙}: 일별 값이 빈 바구니가 있음 {빈줄.head(5).to_dict('records')}")
        확인(시장수익률[규칙].notna().all(), f"{규칙}: 시장 수익률이 빈 날이 있음")

    기존2x3 = 기존[기존["방식"] == "2x3"].set_index("연월")
    월별 = {규칙: 승패(월복리(바구니[규칙]), "연월") for 규칙 in 규칙들}
    안지움 = 월별["안지움"]
    안지움2x3 = 안지움[안지움["방식"] == "2x3"].set_index("연월")["롱숏_%"]
    차이 = (안지움2x3 - 기존2x3["롱숏_%"]).abs()
    확인(len(차이) == 321 and 차이.notna().all(), "안 지운 2x3과 기존 월별의 달이 321개로 맞지 않음")
    확인((차이 <= 멈춤차이).all(), f"안 지운 2x3과 기존 월별의 차이가 {멈춤차이}%p를 넘는 달: {차이[차이 > 멈춤차이].round(2).to_dict()}")

    # ---------- 저장: 기본 규칙만 ----------
    일WML = 승패(바구니[기본규칙], "날짜")
    일WML = 일WML.merge(시장수익률[기본규칙].rename_axis("날짜").reset_index(), on="날짜", how="left", validate="many_to_one")
    확인(일WML["시장_%"].notna().all(), "일별 WML에 시장 수익률이 빈 날이 있음")
    확인(일WML["날짜"].min() >= pd.Timestamp("2000-01-01") and 일WML["날짜"].max() <= pd.Timestamp("2026-09-30"), "일별 WML 기간이 다름")
    일WML = 일WML.sort_values(["방식", "날짜"], ignore_index=True)
    바구니저장 = 바구니[기본규칙].sort_values(["방식", "바구니", "날짜"], ignore_index=True)
    일WML.to_parquet(처리폴더 / "일별WML수익률.parquet", index=False)
    바구니저장.to_parquet(처리폴더 / "일별바구니수익률.parquet", index=False)

    # ---------- 결과 보고 ----------
    첫날, 끝날 = 일WML["날짜"].min().date(), 일WML["날짜"].max().date()
    print(f"[저장] 일별 WML {일WML.shape}, 일별 바구니 {바구니저장.shape}, {첫날}부터 {끝날}까지, 거래일 {일WML['날짜'].nunique():,}일")

    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print("\n[기존 월별 WML과 비교: 일별을 복리로 합친 값, 2000-07부터 2026-09까지]")
        줄 = []
        for 규칙 in 규칙들:
            for 방식 in 방식들:
                내것 = 월별[규칙][월별[규칙]["방식"] == 방식].set_index("연월")["롱숏_%"].loc["2000-07":"2026-09"]
                기준 = 기존[기존["방식"] == 방식].set_index("연월")["롱숏_%"].loc["2000-07":"2026-09"]
                줄.append({"규칙": 규칙, "방식": 방식, **비교줄(내것, 기준), "최악": 내것.min(), "최악달": 내것.idxmin()})
        print(pd.DataFrame(줄).to_string(index=False))
        print(f"  기존 2x3 최악: {기존2x3.loc['2000-07':'2026-09', '롱숏_%'].min():.2f}% "
              f"({기존2x3.loc['2000-07':'2026-09', '롱숏_%'].idxmin()})")

        print("\n[기존 시장_%와 비교: 일별을 복리로 합친 값, 2000-07부터 2026-09까지]")
        줄 = []
        기준 = 기존2x3.loc["2000-07":"2026-09", "시장_%"]
        for 규칙 in 규칙들:
            r = 시장수익률[규칙]
            월 = ((1 + r / 100).groupby(r.index.strftime("%Y-%m")).prod() - 1) * 100
            줄.append({"규칙": 규칙, **비교줄(월.reindex(기준.index), 기준)})
        print(pd.DataFrame(줄).to_string(index=False))

        print("\n[기본 규칙(±31%)의 일별 값, 2000-07 이후] 표준편차 / 왜도 / 최악의 날 / 최고의 날")
        줄 = []
        for 이름, 시계열 in ([(방식, 일WML[일WML["방식"] == 방식].set_index("날짜")["롱숏_%"]) for 방식 in 방식들]
                         + [("시장", 일WML[일WML["방식"] == "2x3"].set_index("날짜")["시장_%"])]):
            x = 시계열.loc["2000-07-01":]
            줄.append({"대상": 이름, "일수": len(x), "표준편차": x.std(), "왜도": x.skew(), "최악": x.min(),
                     "최악일": x.idxmin().date(), "최고": x.max(), "최고일": x.idxmax().date()})
        print(pd.DataFrame(줄).to_string(index=False))

        print("\n[2x3 바구니 일별 종목 수, 기본 규칙] 평균 / 최소")
        b = 바구니저장[바구니저장["방식"] == "2x3"].groupby("바구니")["종목수"]
        print("  " + ", ".join(f"{k} {a:.0f}/{m}" for (k, a), m in zip(b.mean().items(), b.min())))

    외부대조(일WML)
    지운칸보고(지운칸, 표, 일별, 패널, 칸합계)
    종목대조(대상, 일별, 패널)
    print(f"\n[A0053F0 티앤이코리아] 월별 패널에 있음: {'A0053F0' in set(패널['종목코드'])}, 투자 대상에 있음: {'A0053F0' in set(대상['종목코드'])}, "
          f"일별 값 {int(일별['A0053F0'].notna().sum()) if 'A0053F0' in 일별.columns else 0}칸")


if __name__ == "__main__":
    실행()
