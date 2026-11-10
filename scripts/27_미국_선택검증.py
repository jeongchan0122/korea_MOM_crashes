# -*- coding: utf-8 -*-
"""
27단계 (미국 선택·검증, 연구 기록 22절): 14번 선택 절차를 미국 선택 기간(1968-07부터 1996-12까지)에 그대로
적용해 모델 하나를 고르고, 미국 검증 기간(1997-01부터 2024-12까지)에 15번 평가로 판정한다.

실행: python scripts/27_미국_선택검증.py   (먼저 24번, 26번을 실행해 두어야 함)
결과: 결과/미국/선택검증/(후보표.csv, 판정.csv, 그림 png)
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

프로젝트 = Path(__file__).resolve().parent.parent
자료폴더 = 프로젝트 / "처리데이터" / "미국"
결과폴더 = 프로젝트 / "결과" / "미국" / "선택검증"
선택구간, 검증구간 = ("1968-07", "1996-12"), ("1997-01", "2024-12")


def 불러오기(이름, 파일):
    명세 = importlib.util.spec_from_file_location(이름, Path(__file__).resolve().parent / 파일)
    모듈 = importlib.util.module_from_spec(명세)
    명세.loader.exec_module(모듈)
    return 모듈


검증코드 = 불러오기("검증", "15_검증.py")
선택코드 = 검증코드.선택코드
신호코드 = 검증코드.신호코드
plt = 검증코드.plt


def 자료읽기():
    월 = pd.read_parquet(자료폴더 / "월별WML.parquet").set_index("연월")
    회전 = pd.read_parquet(자료폴더 / "회전율.parquet").set_index("연월")
    표 = pd.read_parquet(자료폴더 / "지표월별.parquet").set_index("연월")
    단독 = pd.read_parquet(자료폴더 / "신호월별.parquet").set_index("연월").drop(columns="적용연월")
    return 월, 회전, 표, 단독


def 신호(표, 단독, 이름):
    종류, 지표 = 이름.split(":")
    지표들 = tuple(지표.split("+"))
    return 단독[지표들[0]] if 종류 == "단독" else 신호코드.조합신호(표, 단독, 지표들, "과반" if 종류 == "과반" else "점수")


def 비중(s, 보유달):
    s = s.reindex([str(pd.Period(m) - 1) for m in 보유달])
    return pd.Series(1 - s.fillna(0).values, index=보유달)


def 고르기(월, 회전, 표, 단독):
    a, b = 선택구간
    선택코드.보유시작, 선택코드.보유끝 = a, b
    선택코드.지표시작, 선택코드.지표끝 = str(pd.Period(a) - 1), str(pd.Period(b) - 1)
    보유달 = list(pd.period_range(a, b, freq="M").astype(str))
    모음 = 선택코드.후보모음(월["롱숏_%"].loc[a:b], 회전.loc[a:b], 표.loc[:선택코드.지표끝], 단독.loc[:선택코드.지표끝], 보유달)
    고른것, 정의표 = 선택코드.정의고르기(모음)
    재료들 = 선택코드.재료(고른것)
    for m in 재료들 + [고른것["CI"]]:
        모음.추가("단독", [m])
    for 방법 in ("과반", "점수"):
        선택코드.하나씩추가(모음, 재료들, 방법)
    최종, 동률, _ = 선택코드.최종고르기(모음)
    return 모음, 고른것, 정의표, 최종, 동률


def 실행():
    선택코드.비용률 = 0.5
    월, 회전, 표, 단독 = 자료읽기()
    모음, 고른것, 정의표, 최종, 동률 = 고르기(월, 회전, 표, 단독)
    심사 = [c for c in 모음.목록.values() if c["심사통과"]]
    print(f"[선택 기간 {선택구간[0]}부터 {선택구간[1]}까지] 후보 {len(모음.목록)}개, 심사 통과 {len(심사)}개, "
          f"붕괴 사건 {len(모음.사건)}개, 정의 {고른것}")
    print(정의표.to_string(index=False))
    if 최종 is None:
        print("통과한 후보가 없어 모델을 고르지 못함")
        return
    일위 = max(심사, key=lambda c: c["t"])
    print(f"  t 1위 {일위['이름']} (t {일위['t']:.2f}), 1 표준오차 안 {len(동률)}개")
    print(f"  뽑힌 모델 {최종['이름']}: t {최종['t']:.2f}, 샤프 {최종['샤프']:.2f} (원래 {모음.원래샤프:.2f}), "
          f"최대낙폭 {최종['최대낙폭']:.1f}% (원래 {모음.원래낙폭:.1f}%), 쉰 달 {최종['켜진달']}")

    보유달 = list(pd.period_range(선택구간[0], 검증구간[1], freq="M").astype(str))
    원래비용전 = 월["롱숏_%"].reindex(보유달)
    회전전체 = 회전.reindex(보유달)
    줄들 = []
    for 이름, c in 모음.목록.items():
        w = 비중(신호(표, 단독, 이름), 보유달)
        원래 = 선택코드.비용후(pd.Series(1.0, index=보유달), 원래비용전, 회전전체)
        r = 선택코드.비용후(w, 원래비용전, 회전전체)
        선 = 검증코드.평가(r, 원래, 원래비용전, w, 선택구간)
        assert abs(선["t"] - c["t"]) < 1e-9, (이름, 선["t"], c["t"])
        검 = 검증코드.평가(r, 원래, 원래비용전, w, 검증구간)
        줄들.append({"이름": 이름, "지표수": c["지표수"], "선택_심사통과": c["심사통과"], "선택_t": c["t"],
                   "검증_t": 검["t"], "검증_샤프": 검["샤프"], "검증_원래샤프": 검["원래_샤프"], "검증_결론": 검["결론"]})
    후보표 = pd.DataFrame(줄들).sort_values("선택_t", ascending=False)
    print("[선택 기간 재현] 후보 전체의 선택 t가 선택 절차와 같음")

    판정줄 = []
    for 비용 in (0.5, 0.25):
        선택코드.비용률 = 비용
        원래 = 선택코드.비용후(pd.Series(1.0, index=보유달), 원래비용전, 회전전체)
        for 표지, 이름 in (("미국에서 고른 모델", 최종["이름"]), ("한국 최종 모델", 검증코드.기대최종)):
            w = 비중(신호(표, 단독, 이름), 보유달)
            r = 선택코드.비용후(w, 원래비용전, 회전전체)
            for 구간이름, 구간 in (("선택", 선택구간), ("검증", 검증구간)):
                e = 검증코드.평가(r, 원래, 원래비용전, w, 구간)
                판정줄.append({"비용": 비용, "모델": 표지, "이름": 이름, "구간": 구간이름, "쉰달": e["켜진달"],
                            "월평균": e["월평균"], "원래_월평균": e["원래_월평균"], "샤프": e["샤프"], "원래_샤프": e["원래_샤프"],
                            "최대낙폭": e["최대낙폭"], "원래_최대낙폭": e["원래_최대낙폭"], "사건합": e["사건합"],
                            "원래_사건합": e["원래_사건합"], "t": e["t"], "결론": e["결론"]})
            if 비용 == 0.5 and 표지 == "미국에서 고른 모델":
                그림재료 = (w, 원래, r, 최종["이름"])
    판정 = pd.DataFrame(판정줄)
    with pd.option_context("display.width", 260, "display.max_colwidth", 45):
        print(판정.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    검증좋음 = (후보표["검증_t"] >= 1.96).sum()
    print(f"\n[후보 {len(후보표)}개의 검증] t >= 1.96: {검증좋음}개, t > 0: {(후보표['검증_t'] > 0).sum()}개, "
          f"결론 {후보표['검증_결론'].value_counts().to_dict()}")
    with pd.option_context("display.width", 200):
        print(후보표.head(12).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    결과폴더.mkdir(parents=True, exist_ok=True)
    후보표.to_csv(결과폴더 / "후보표.csv", index=False, encoding="utf-8-sig")
    판정.to_csv(결과폴더 / "판정.csv", index=False, encoding="utf-8-sig")
    그림(*그림재료)
    print(f"[저장] {결과폴더}")


def 그림(w, 원래, r, 이름):
    from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator
    w, 원래, r = (x.loc[검증구간[0]:] for x in (w, 원래, r))
    날 = 선택코드.월말(w.index)
    fig, ax = plt.subplots(figsize=(11, 5.8))
    선택코드.쉼음영(ax, w)
    for 값, 색, 표지 in ((원래, 선택코드.파랑, "원래 WML"), (r, 선택코드.주황, "미국 선택 기간에서 고른 모델")):
        누 = (1 + 값 / 100).cumprod()
        ax.plot(날, 누.values, color=색, linewidth=1.8, label=f"{표지}  {누.iloc[-1]:.2f}배")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}배"))
    선택코드.꾸미기(ax, "미국 검증 기간: 미국 선택 기간에서 고른 모델",
                f"{이름}. {검증구간[0]}부터 {검증구간[1]}까지, 시작값 1, 로그 눈금, 비용(편도 0.5%) 차감. 회색 = 쉰 달.")
    ax.legend(loc="lower left", frameon=False)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.84, bottom=0.08)
    fig.savefig(결과폴더 / "1_검증_누적수익.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    실행()
