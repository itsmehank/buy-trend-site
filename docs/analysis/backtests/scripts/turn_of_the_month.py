"""H-031 — Turn-of-the-Month. **McConnell & Xu (2008, FAJ 64(2) 49-64) 원문 형태 그대로.**

원문 Table 1·4 주석의 정의(원문 219-222행) 그대로:

    Day −1        = 직전 월의 **마지막 거래일**
    Day +1,+2,+3  = 그 월의 **첫 세 거래일**
    Day (−1,+3)   = 직전 월 마지막 거래일 ~ 그 월 셋째 거래일 (**4거래일**)
    "Other Days"  = **Day −10 ~ −2**(월말 전) 및 **Day +4 ~ +10**(월초 후) (**16거래일**)
                    ← 전체 나머지가 **아니다.** 창 밖의 날은 어느 쪽에도 안 들어간다

  · 판정 지표는 **평균 일간수익의 차** — 원문 "Difference" 열. 거래를 수반하지 않는다.
  · 레짐 필터 없음(원문에 없음). PROTOCOL §4-(a) 항목 3-(다)에 따라 얹지 않는다.

원문 수치 (문자열 검증 완료 — 문서 §1.2):

  | 패널 | 기초자산 | (−1,+3) | Other | Difference | t |
  |---|---|---|---|---|---|
  | Table 4 South Korea (1987-09~2006-01) | **Datastream 국가 지수** | 0.29 | −0.04 | **+0.33** | 3.98 |
  | Table 1 Panel C (1926~2005)           | **CRSP VW**             | 0.16 |  0.01 | **+0.15** | 8.06 |
  | Table 1 Panel F (1926~2005)           | **CRSP EW**             | 0.23 |  0.05 | **+0.18** | 9.23 |

**KR과 US의 기초자산이 다르다** — KR은 지수, US Panel F는 EW. 그래서 판정을 둘로 나눈다.

판정은 PROTOCOL §3 개정판(2026-08-12) — **6종 추정량이 갈리면 측정 불가**.

  PYTHONPATH=.:docs/analysis .venv/bin/python -W ignore \
    docs/analysis/backtests/scripts/turn_of_the_month.py --selftest | --power

**--power 는 SE·표본·전략 회전율만 출력한다. 평균·t·부호는 출력 경로가 없다**
(PROTOCOL §3.1-2 — 점추정치를 보면 사전등록이 무효가 된다).
"""
from __future__ import annotations

import sys
from statistics import NormalDist

import numpy as np
import pandas as pd

from btlib import loading

sys.path.insert(0, "docs/analysis/backtests/scripts")

_N = NormalDist()

# ── 원문이 정한 것 (바꾸지 않는다) ─────────────────────────────────────────
TOM_PRE = 1        # Day −1 만 TOM 에 들어간다
TOM_POST = 3       # Day +1,+2,+3
OTHER_PRE = (2, 10)    # Day −2 ~ −10
OTHER_POST = (4, 10)   # Day +4 ~ +10
# 결측 허용 하한 — TOM 은 4일 전부, Other 는 16일 중 12일 이상.
# **사전등록 §3.1에 명세한다.** 이 문턱 때문에 ② US 의 2011-09 turn 이
# (Other 유효 9일) 탈락해 179 → 178 이 된다.
OTHER_MIN = 12

# 원문 수치 — **사전 고정** (문자열 검증 완료)
LIT_IDX = {"kr": 0.33, "us": 0.15}   # ① 지수: KR Table 4 · US Table 1 Panel C(VW)
LIT_EW = {"kr": 0.33, "us": 0.18}    # ② EW: US Table 1 Panel F. **KR 은 문헌 대응물이
                                     #    없어 지수값을 유용한다 — 문서에 명기**
LIT_T = {"kr": 3.98, "us_vw": 8.06, "us_ew": 9.23}
LIT_DIR = +1                          # 문헌이 예측하는 부호 — 양수 유의면 채택

# ── 저장소 규약 (PROTOCOL §2 · 실행 위생이라 §0.3에서 유지) ────────────────
PRICE_FLOOR = {"kr": 1000.0, "us": 5.0}
DV_WIN = 20
DV_MIN = {"kr": None, "us": 2e6}      # PROTOCOL §2 (PIT) · KR 은 하한 없음
COST = {"kr": 0.0014, "us": 0.0005}   # **편도** (PROTOCOL §2 왕복의 절반)

# 다중비교 가족 — **결과를 보기 전에 고정.** 원문이 두 기초자산을 따로 보고한다.
FAMILY = 2
K_CRIT = _N.inv_cdf(1.0 - 0.05 / (2.0 * FAMILY))       # 2.2414


# ────────────────────────────────────────────────────────── 일자 분류

def classify_turns(dates: pd.DatetimeIndex) -> list[dict]:
    """월 경계(turn)마다 −10…−1 · +1…+10 거래일 인덱스를 모은다.

    원문 정의는 **거래일 서수**이지 달력일이 아니다. 직전 월의 마지막 10거래일과
    그 월의 첫 10거래일을 쓰며, **어느 쪽도 10개 미만이면 그 turn 을 버린다**
    (부분 창을 평균하지 않는다).
    """
    mk = pd.PeriodIndex(dates, freq="M")
    groups: dict = {}
    for i, p in enumerate(mk):
        groups.setdefault(p, []).append(i)
    months = sorted(groups)
    out = []
    for k in range(1, len(months)):
        prev, cur = groups[months[k - 1]], groups[months[k]]
        if len(prev) < 10 or len(cur) < 10:
            continue
        pre = prev[-10:][::-1]        # pre[0] = Day −1, pre[1] = Day −2, …
        post = cur[:10]               # post[0] = Day +1, …
        out.append({
            "month": months[k],
            "tom": [pre[0]] + post[:TOM_POST],                  # −1, +1, +2, +3
            "other": pre[OTHER_PRE[0] - 1:OTHER_PRE[1]]         # −2 … −10
                     + post[OTHER_POST[0] - 1:OTHER_POST[1]],   # +4 … +10
            "d_minus1": pre[0],
            "entry_pit": None,        # PIT 근사 진입일 (아래에서 채움)
            "exit": post[TOM_POST - 1],
        })
    return out


def mark_pit_entry(turns: list[dict], dates: pd.DatetimeIndex) -> None:
    """**전략 형태의 PIT 진입일** — Day −1 은 t 시점에 알 수 없다(§2-(2-ㄴ)).

    달력상 **그 월의 마지막 영업일**(주말만 고려, 날짜 산술만으로 판정)에
    봉이 있으면 그날 종가에 진입한다. 그날이 휴장이면 **그 turn 은 진입하지 못한다**
    (Day −1 을 놓친다). 미래 봉을 보지 않으므로 look-ahead 가 없다.
    """
    by_date = {d.date(): i for i, d in enumerate(dates)}
    for t in turns:
        prev_m = (t["month"] - 1)
        last_bd = pd.Timestamp(prev_m.end_time.date())
        while last_bd.weekday() >= 5:                 # 토(5)·일(6) 이면 앞으로
            last_bd -= pd.Timedelta(days=1)
        t["entry_pit"] = by_date.get(last_bd.date())


# ──────────────────────────────────────────────────────────── 계열

def build_series(market: str) -> dict:
    px = loading.load_prices(market)
    piv = {c: px.pivot_table(index="date", columns="ticker", values=c, aggfunc="last")
           for c in ("close", "open", "volume")}
    close = piv["close"].sort_index()
    idx = pd.DatetimeIndex(pd.to_datetime(close.index))
    for k in piv:
        piv[k] = piv[k].reindex(index=close.index, columns=close.columns)

    cl = piv["close"].to_numpy(float)
    op = piv["open"].to_numpy(float)
    ok_bar = np.isfinite(cl) & (cl > 0) & np.isfinite(op) & (op > 0)   # PROTOCOL §1
    clm = np.where(ok_bar, cl, np.nan)

    dret = np.full_like(clm, np.nan)
    with np.errstate(all="ignore"):
        dret[1:] = clm[1:] / clm[:-1] - 1.0
    dret[~np.isfinite(dret)] = np.nan

    # 적격 — 가격 하한 + PIT 거래대금 (전일 기준으로 시프트해 look-ahead 제거)
    dv = (piv["close"] * piv["volume"]).rolling(DV_WIN, min_periods=DV_WIN).mean()
    dv = dv.shift(1).to_numpy(float)
    elig = ok_bar & (clm >= PRICE_FLOOR[market])
    if DV_MIN[market] is not None:
        elig &= np.isfinite(dv) & (dv >= DV_MIN[market])
    elig[0] = False

    # EW 유니버스 일간수익 (%)
    ew = np.full(len(idx), np.nan)
    for i in range(len(idx)):
        r = dret[i][elig[i] & np.isfinite(dret[i])]
        if len(r) >= 10:
            ew[i] = r.mean() * 100.0

    # 지수 일간수익 (%)
    b = loading.load_bench(market).set_index("date")["close"].reindex(close.index).ffill()
    bv = b.to_numpy(float)
    bret = np.full(len(idx), np.nan)
    with np.errstate(all="ignore"):
        bret[1:] = (bv[1:] / bv[:-1] - 1.0) * 100.0
    bret[~np.isfinite(bret)] = np.nan

    return {"dates": idx, "ew": ew, "idx": bret, "n_elig": elig.sum(axis=1),
            "market": market}


def turn_diffs(S: dict, key: str) -> tuple[pd.Series, dict]:
    """turn 별 (TOM 평균 일간수익 − Other 평균 일간수익), 단위 %p/일.

    **평균·t 를 이 함수 밖으로 내보내지 않는다** — 호출부가 SE 만 쓴다.
    """
    r = S[key]
    turns = classify_turns(S["dates"])
    vals, months, ntom, noth = [], [], [], []
    for t in turns:
        a = r[t["tom"]]
        b = r[t["other"]]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        # **결측 허용 하한 — 사전등록 §3.1에 명세됨.** TOM 은 4일 전부,
        # Other 는 16일 중 12일 이상 유효해야 그 turn 을 쓴다.
        if len(a) < TOM_PRE + TOM_POST or len(b) < OTHER_MIN:
            continue
        vals.append(a.mean() - b.mean())
        months.append(t["month"])
        ntom.append(len(a)); noth.append(len(b))
    s = pd.Series(vals, index=pd.PeriodIndex(months, freq="M"))
    return s, {"n": len(s), "n_turn_total": len(turns),
               "tom_days": float(np.mean(ntom)) if ntom else 0.0,
               "other_days": float(np.mean(noth)) if noth else 0.0}


def weight_path(turns: list[dict], n_days: int) -> np.ndarray:
    """전략의 일별 비중 경로. PIT 진입에 성공한 turn 만 1.0 을 잡는다.

    진입 실패(휴장)면 그 turn 은 **비중 0으로 지나간다** — Day −1 을 놓친 채
    뒤늦게 들어가지 않는다(사전등록 §2.1).
    """
    w = np.zeros(n_days)
    for t in turns:
        e = t["entry_pit"]
        if e is None:
            continue
        w[e:t["exit"] + 1] = 1.0
    return w


def strategy_turnover(S: dict) -> dict:
    """전략 형태의 **회전율·비용·PIT 진입 실패율만** 낸다. 수익은 내지 않는다.

    `Σ|Δw|` 를 **비중 경로에서 실제로 계산**한다(하드코딩하지 않는다 — 게이트 1차 지적).
    """
    turns = classify_turns(S["dates"])
    mark_pit_entry(turns, S["dates"])
    n = len(turns)
    miss = sum(1 for t in turns if t["entry_pit"] is None)
    w = weight_path(turns, len(S["dates"]))
    dw = np.abs(np.diff(np.concatenate([[0.0], w, [0.0]]))).sum()   # Σ|Δw| 전체
    entered = n - miss
    sigma_per_turn = dw / entered if entered else float("nan")      # 기대값 2.0
    one_way = sigma_per_turn / 2.0                                  # 편도 회전율
    cost_per_turn = COST[S["market"]] * sigma_per_turn * 100.0      # %p
    return {"turns": n, "pit_miss": miss,
            "pit_miss_pct": 100.0 * miss / n if n else float("nan"),
            "sigma_dw": dw, "sigma_per_turn": sigma_per_turn,
            "one_way": one_way, "cost_pct": cost_per_turn,
            "cost_annual": cost_per_turn * 12.0}


# ─────────────────────────────────────────────────────────────── 명령

def _pc():
    import pooled_clustering as PC
    return PC


_S: dict = {}


def _series() -> dict:
    if not _S:
        for mk in ("kr", "us"):
            _S[mk] = build_series(mk)
    return _S


def cmd_power():
    PC = _pc()
    print("=" * 100)
    print("[H-031] 사전 검출력 — **SE·표본·회전율만**. 평균·t·부호는 출력 경로가 없다.")
    print(f"  가족 {FAMILY}칸 Bonferroni → |t| > {K_CRIT:.4f}")
    print("  ① 지수 기반 (KR Table 4 · US Table 1 Panel C VW)")
    print("  ② EW 유니버스 (US Table 1 Panel F EW · KR 은 문헌 대응물 없음)")
    print("=" * 100)
    cells = (("① 지수", "idx", LIT_IDX), ("② EW 유니버스", "ew", LIT_EW))
    for tag, key, lit in cells:
        ser, meta = {}, {}
        for mk, S in _series().items():
            s, m = turn_diffs(S, key)
            ser[mk], meta[mk] = s, m
            print(f"\n── {tag} · {mk.upper()} ── turn {m['n']}개"
                  f" ({s.index[0]}~{s.index[-1]}) · 전체 turn {m['n_turn_total']}")
            print(f"   TOM 평균 {m['tom_days']:.2f}일 · Other 평균 {m['other_days']:.2f}일"
                  f" · 문헌 효과 {lit[mk]:+.2f}%p/일")
            print(f"   naive SE {PC.naive_se({'x': s})[1]:.4f}%p (참고)")
        est = PC.all_estimates(ser)
        ses = {k: est[k] for k in PC.VOTERS}
        worst = max(ses, key=lambda k: ses[k])
        n_kr, n_us = meta["kr"]["n"], meta["us"]["n"]
        e_star = (n_kr * lit["kr"] + n_us * lit["us"]) / (n_kr + n_us)
        d = e_star / est[worst]
        pw = _N.cdf(d - K_CRIT) + _N.cdf(-d - K_CRIT)
        print(f"\n[통합 {tag}] n={est['n']} · 겹치는 달 {est['overlap']} · ρ={est['rho']:+.4f}")
        print(f"   G(달)={est['G_달']} G(분기)={est['G_분기']} G(연)={est['G_연']}")
        for k in PC.VOTERS:
            print(f"   {k:<10} SE {est[k]:.4f}%p")
        print(f"   naive      SE {est['naive']:.4f}%p (판정에 투표하지 않음)")
        print(f"   표본가중 문헌 효과 e* = {e_star:.4f}%p/일")
        print(f"   **SE 최대 = {worst} ({est[worst]:.4f}) → 검정력 = {pw*100:.1f}%**")

    print("\n" + "=" * 100)
    print("[전략 형태 — 임계 없이 보고] 판정 칸이 아니다. 원문은 전략을 제안하지 않는다.")
    for mk, S in _series().items():
        t = strategy_turnover(S)
        print(f"  {mk.upper()}: turn {t['turns']}개 · **PIT 진입 실패 {t['pit_miss']}회"
              f" ({t['pit_miss_pct']:.1f}%)** · 편도 회전율 {t['one_way']*100:.0f}%/월")
        print(f"        Σ|Δw| 실측 {t['sigma_dw']:.1f} / 진입 turn"
              f" = turn 당 {t['sigma_per_turn']:.4f} (기대 2.0)")
        print(f"        비용 {t['cost_pct']:.4f}%p/월 (연 {t['cost_annual']:.2f}%p)")


def cmd_selftest():
    ok = []

    def chk(name, cond):
        ok.append((f"{len(ok) + 1:02d}", name, bool(cond)))

    # 일자 분류 — 손으로 셀 수 있는 합성 달력 (평일만, 3개월)
    d = pd.DatetimeIndex([x for x in pd.bdate_range("2024-01-01", "2024-03-29")])
    turns = classify_turns(d)
    chk("월 경계 turn 2개 (1월→2월, 2월→3월)", len(turns) == 2)
    t0 = turns[0]
    chk("TOM 은 4거래일 (−1,+1,+2,+3)", len(t0["tom"]) == 4)
    chk("Other 는 16거래일 (−2~−10 9개 + +4~+10 7개)", len(t0["other"]) == 16)
    chk("TOM 과 Other 는 겹치지 않는다", set(t0["tom"]).isdisjoint(t0["other"]))
    # Day −1 = 직전 월 마지막 거래일
    jan = [i for i, x in enumerate(d) if x.month == 1]
    chk("Day −1 = 1월 마지막 영업일", t0["d_minus1"] == jan[-1])
    feb = [i for i, x in enumerate(d) if x.month == 2]
    chk("Day +1,+2,+3 = 2월 첫 세 영업일", t0["tom"][1:] == feb[:3])
    chk("Day −2 = 1월 끝에서 둘째", t0["other"][0] == jan[-2])
    chk("Day −10 = 1월 끝에서 열째", t0["other"][8] == jan[-10])
    chk("Day +4 = 2월 넷째", t0["other"][9] == feb[3])
    chk("Day +10 = 2월 열째", t0["other"][15] == feb[9])
    chk("exit = Day +3", t0["exit"] == feb[2])

    # 거래일 10개 미만인 월은 버린다
    short = pd.DatetimeIndex(list(pd.bdate_range("2024-01-01", "2024-01-31"))
                             + list(pd.bdate_range("2024-02-01", "2024-02-07")))
    chk("한쪽 월이 10거래일 미만이면 turn 을 버린다", len(classify_turns(short)) == 0)

    # PIT 진입 — 달력 마지막 영업일에 봉이 있으면 그날, 없으면 실패
    tt = classify_turns(d); mark_pit_entry(tt, d)
    chk("PIT 진입일 = 달력상 직전 월 마지막 영업일", tt[0]["entry_pit"] == jan[-1])
    d_hole = d.delete(jan[-1])                      # 1월 마지막 영업일이 휴장
    tt2 = classify_turns(d_hole); mark_pit_entry(tt2, d_hole)
    chk("그날이 휴장이면 PIT 진입 실패(None)", tt2[0]["entry_pit"] is None)
    # **측정용 Day −1 은 실제 마지막 거래일로 남는다** — 전략만 실패한다.
    # 종전에는 `is not None` 만 봤는데 `d_minus1` 은 항상 int 라 실패 불가능한
    # 공허한 단언이었다(게이트 1차 지적). 값 자체를 검사한다.
    jan_hole = [i for i, x in enumerate(d_hole) if x.month == 1]
    chk("측정용 Day −1 = 휴장 하루 앞 실제 마지막 거래일",
        tt2[0]["d_minus1"] == jan_hole[-1] and d_hole[tt2[0]["d_minus1"]].day == 30)

    # Σ|Δw| 를 **비중 경로에서 실제로 계산**해 검사한다 (하드코딩 아님)
    st = classify_turns(d); mark_pit_entry(st, d)
    w = weight_path(st, len(d))
    chk("진입 성공 turn 은 −1~+3 4일 보유", int(w.sum()) == 4 * len(st))
    dw = np.abs(np.diff(np.concatenate([[0.0], w, [0.0]]))).sum()
    chk("Σ|Δw| = turn 당 2.0 (전량 진입 + 전량 청산)",
        np.isclose(dw / len(st), 2.0))
    st2 = classify_turns(d_hole); mark_pit_entry(st2, d_hole)
    w2 = weight_path(st2, len(d_hole))
    chk("PIT 진입 실패 turn 은 비중 0으로 지나간다(뒤늦게 들어가지 않는다)",
        int(w2.sum()) == 4 * (len(st2) - sum(1 for t in st2 if t["entry_pit"] is None)))

    # 차분 계산 — 알려진 값
    S = {"dates": d, "x": np.zeros(len(d)), "market": "kr"}
    S["x"][t0["tom"]] = 1.0
    S["x"][t0["other"]] = 0.25
    s, meta = turn_diffs(S, "x")
    chk("TOM=1.0 · Other=0.25 이면 차 = 0.75", np.isclose(s.iloc[0], 0.75))
    chk("meta 의 TOM 일수 = 4", np.isclose(meta["tom_days"], 4.0))
    chk("meta 의 Other 일수 = 16", np.isclose(meta["other_days"], 16.0))

    # 비용 — 매 turn 전량 진입·청산이면 Σ|Δw| = 2.0 → 왕복 요율
    for mk, rt in (("kr", 0.28), ("us", 0.10)):
        c = COST[mk] * 2.0 * 100.0
        chk(f"{mk.upper()} turn 당 비용 = 왕복 {rt:.2f}%", np.isclose(c, rt))
    chk("COST 는 PROTOCOL §2 왕복의 절반(편도)",
        np.isclose(COST["kr"], 0.0028 / 2) and np.isclose(COST["us"], 0.0010 / 2))

    # 상수 — 원문 사양
    chk("K_CRIT ≈ 2.2414", abs(K_CRIT - 2.2414) < 5e-4)
    chk("TOM = −1 + (+1,+2,+3)", TOM_PRE == 1 and TOM_POST == 3)
    chk("Other = −2~−10 및 +4~+10", OTHER_PRE == (2, 10) and OTHER_POST == (4, 10))
    chk("문헌 방향 +1 (양수 유의면 채택)", LIT_DIR == +1)
    chk("KR e* = +0.33 (Table 4 지수)", LIT_IDX["kr"] == 0.33)
    chk("US 지수 e* = +0.15 (Panel C VW) · EW e* = +0.18 (Panel F)",
        LIT_IDX["us"] == 0.15 and LIT_EW["us"] == 0.18)

    for num, name, good in ok:
        print(f"  {'PASS' if good else 'FAIL'}  {num}. {name}")
    bad = [n for n, _, g in ok if not g]
    print(f"\n{len(ok) - len(bad)}/{len(ok)} 통과")
    return 1 if bad else 0


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "--power"
    if arg == "--selftest":
        sys.exit(cmd_selftest())
    elif arg == "--power":
        cmd_power()
    else:
        print(__doc__)
        sys.exit(2)
