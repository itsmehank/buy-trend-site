"""H-014 — 검정력·MDE 메타 분석.

설계 문서: docs/analysis/backtests/2026-08-02-mde-power-analysis.md

이 시스템의 표준 테스트 설계에서 **검출 가능한 최소 효과크기(MDE)**를 산출하고,
**외부에서 사전 지정한 효과크기 e\\*(월 0.33%p)**와 대조한다.

**입력은 아카이브에 이미 기록된 CI뿐이다.** 새 백테스트를 돌리지 않는다.
관측 효과크기와는 비교하지 않는다 — 그것은 |t|/2.8016의 재척도라 순환이다(§2.3).

**btlib를 쓰지 않는 이유**: 가격 데이터를 전혀 쓰지 않는 순수 계산이다. 검증용
H-012 대조에서만 btlib 경유 모듈을 import한다.

실행 (저장소 루트에서):
  PYTHONPATH=.:docs/analysis .venv/bin/python docs/analysis/backtests/scripts/mde_power_analysis.py --selftest
  PYTHONPATH=.:docs/analysis .venv/bin/python docs/analysis/backtests/scripts/mde_power_analysis.py
  PYTHONPATH=.:docs/analysis:docs/analysis/backtests/scripts .venv/bin/python \
      docs/analysis/backtests/scripts/mde_power_analysis.py --verify-h012
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from statistics import NormalDist          # 표준 라이브러리 (scipy 미설치 환경)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

_N = NormalDist()


class norm:                                 # scipy.stats.norm 호환 최소 래퍼
    @staticmethod
    def ppf(p: float) -> float:
        return _N.inv_cdf(p)

# ── 문서 §2·§3 고정 파라미터 ─────────────────────────────────────────────
ALPHA, POWER = 0.05, 0.80
Z_A = norm.ppf(1 - ALPHA / 2)          # 1.959964
Z_B = norm.ppf(POWER)                  # 0.841621
K_MDE = Z_A + Z_B                      # 2.801585
E_STAR = 0.33                          # 월 %p (연 4%p) — 문헌 모멘텀 프리미엄 하단
E_STAR2 = 0.17                         # 월 %p (연 2%p) — 부품 기여 기준
TRADING_DAYS_M = 21

#: 아카이브에 기록된 CI (문서 §2.4의 20칸). 단위: %p
#: (테스트, 조건, 시장, h일 or None=월, **평균**, CI 하한, CI 상한, 표본수)
#: 평균을 함께 적는 이유 — 원 스크립트들이 CI = 평균 ± 1.96·SE로 만들므로
#: **CI 중점 == 평균**이어야 한다. selftest ⑧이 20칸 전부를 이 항등식으로 검산해
#: 손으로 옮겨 적을 때의 전사(轉寫) 오류를 잡는다. (§5.6 P1 — 실제로 잡았다)
CELLS = [
    # H-010 §5.1·§5.2 — 월별 차분, n = KR 134 / US 168
    ("H-010", "② A−C",        "KR", None, +0.153, -0.675, +0.981, 134),
    ("H-010", "③ A−D gross",  "KR", None, +0.277, -0.349, +0.904, 134),
    ("H-010", "④ A−E",        "KR", None, -0.132, -1.165, +0.902, 134),
    ("H-010", "② A−C",        "US", None, -0.625, -1.546, +0.296, 168),
    ("H-010", "③ A−D gross",  "US", None, -0.226, -0.733, +0.281, 168),
    ("H-010", "④ A−E",        "US", None, -0.289, -1.684, +1.106, 168),
    # H-012 §5.1·§5.2 — ④만 월별 초과수익(②③은 무차원 ΔSharpe라 별도)
    ("H-012", "④ A−E",        "KR", None, +0.168, +0.024, +0.312, 134),
    ("H-012", "④ A−E",        "US", None, +0.063, -0.019, +0.144, 168),
    # H-013 §5.1·§5.2 — h일 누적, T = 짝지은 리밸일 수
    ("H-013", "② SMA−RANK",    "KR", 20,   -0.02,  -2.02, +1.99, 77),
    ("H-013", "② SMA−RANK",    "KR", 60,   -2.08,  -7.29, +3.13, 77),
    ("H-013", "② SMA−RANK",    "KR", 126,  -4.00, -10.61, +2.61, 77),
    ("H-013", "③ SMA−REPLACE", "KR", 20,   -1.21,  -3.84, +1.41, 70),
    ("H-013", "③ SMA−REPLACE", "KR", 60,   -3.65,  -7.61, +0.32, 70),
    ("H-013", "③ SMA−REPLACE", "KR", 126,  -5.73,  -9.95, -1.51, 70),
    ("H-013", "② SMA−RANK",    "US", 20,   +0.91,  -2.49, +4.32, 100),
    ("H-013", "② SMA−RANK",    "US", 60,   +2.00,  -4.05, +8.05, 100),
    ("H-013", "② SMA−RANK",    "US", 126,  +2.90,  -2.30, +8.10, 100),
    ("H-013", "③ SMA−REPLACE", "US", 20,   +0.94,  -1.37, +3.25, 133),
    ("H-013", "③ SMA−REPLACE", "US", 60,   +0.72,  -3.25, +4.69, 133),
    ("H-013", "③ SMA−REPLACE", "US", 126,  -1.24,  -6.91, +4.44, 133),
]

#: 무차원 지표 — e*와 비교 불가, 절대 MDE만 보고 (§2.4)
SHARPE_CELLS = [
    ("H-012", "② ΔSharpe(A−C)", "KR", -0.0340, +0.0326),
    ("H-012", "③ ΔSharpe(A−D)", "KR", -0.0410, +0.1539),
    ("H-012", "② ΔSharpe(A−C)", "US", -0.0244, +0.0511),
    ("H-012", "③ ΔSharpe(A−D)", "US", -0.0377, +0.0416),
]

#: 층위 (B) — 각 문서가 스스로 명시한 칸 수 (§3 고정 파라미터)
K_BONF = {"H-010": 6, "H-012": 6, "H-013": 6}

#: 층위 (C) — H-013 `--ci-sim` 실측 귀무 오류율 {(h, T): α*}
CI_SIM = {(20, 148): 0.103, (20, 96): 0.125, (60, 148): 0.146, (60, 70): 0.216,
          (126, 182): 0.208, (126, 148): 0.222, (126, 70): 0.344, (126, 471): 0.132}


def se_from_ci(lo: float, hi: float) -> float:
    """CI 폭에서 SE 역산 — 각 테스트가 실제로 쓴 보정을 그대로 보존한다(§2.1)."""
    return (hi - lo) / (2 * Z_A)


def mde_nominal(se: float) -> float:
    return K_MDE * se


def mde_bonferroni(se: float, k: int) -> float:
    return (norm.ppf(1 - ALPHA / (2 * k)) + Z_B) * se


def alpha_star(h: int | None, T: int) -> float | None:
    """(h, T) 최근접 조합. 표에 없는 T는 가장 가까운 두 T 중 **작은 쪽**(보수적)."""
    if h is None:
        return None
    cands = [(hh, TT) for (hh, TT) in CI_SIM if hh == h]
    if not cands:
        return None
    below = [c for c in cands if c[1] <= T]
    pick = max(below, key=lambda c: c[1]) if below else min(cands, key=lambda c: c[1])
    return CI_SIM[pick]


def mde_alpha_corrected(se: float, a_star: float) -> float:
    """실측 오류율 α*는 SE가 s배 과소추정됐음을 뜻한다 → 참 SE = s·SE (§2.2).

    두 항 모두에 s를 곱한다. α 항에만 곱하면 "임계값은 틀렸고 SE는 옳다"는
    다른 모형이 되어 β 항과 기전이 섞인다. (§5.6 P2)
    """
    scale = Z_A / norm.ppf(1 - a_star / 2)      # α* > α 이면 > 1
    return K_MDE * scale * se


def to_monthly(mde: float, h: int | None) -> float:
    """h일 누적 MDE → 검출 가능한 최소 **월 드리프트** (§2.3)."""
    return mde if h is None else mde * TRADING_DAYS_M / h


def main():
    rows = []
    for test, cond, mkt, h, mean, lo, hi, n in CELLS:
        se = se_from_ci(lo, hi)
        a = mde_nominal(se)
        b = mde_bonferroni(se, K_BONF[test])
        ast = alpha_star(h, n)
        c = mde_alpha_corrected(se, ast) if ast else None
        rows.append({"test": test, "cond": cond, "mkt": mkt, "h": h, "n": n,
                     "se": se, "A": a, "B": b, "C": c,
                     "A_m": to_monthly(a, h), "B_m": to_monthly(b, h),
                     "C_m": to_monthly(c, h) if c else None})

    print(f"{'='*104}\n[H-014] MDE 메타 분석 — 계수 {K_MDE:.4f} (α={ALPHA}, 검정력 {POWER:.0%})"
          f" · e*={E_STAR}%p/월(연 4%p) · e**={E_STAR2}%p/월\n{'='*104}")
    print(f"{'테스트':<7}{'조건':<15}{'시장':>4}{'h':>5}{'T/n':>6}{'SE':>8}"
          f"{'MDE_A':>9}{'월환산':>9}{'>e*':>5}{'>e**':>6}{'MDE_B(월)':>11}{'MDE_C(월)':>11}")
    print("-" * 104)
    for r in rows:
        hs = "월" if r["h"] is None else str(r["h"])
        cm = f"{r['C_m']:>10.3f}" if r["C_m"] else f"{'—':>10}"
        print(f"{r['test']:<7}{r['cond']:<15}{r['mkt']:>4}{hs:>5}{r['n']:>6}"
              f"{r['se']:>8.3f}{r['A']:>9.3f}{r['A_m']:>9.3f}"
              f"{'✓' if r['A_m'] > E_STAR else '·':>5}"
              f"{'✓' if r['A_m'] > E_STAR2 else '·':>6}{r['B_m']:>11.3f}{cm}")

    # 결론 규칙 (§3)
    print(f"\n{'='*104}\n[결론 규칙] MDE_A(월환산) > e* 인 칸 비율\n{'='*104}")
    for label, sub in (("통합", rows),
                       ("KR", [r for r in rows if r["mkt"] == "KR"]),
                       ("US", [r for r in rows if r["mkt"] == "US"])):
        hit = sum(1 for r in sub if r["A_m"] > E_STAR)
        hit2 = sum(1 for r in sub if r["A_m"] > E_STAR2)
        ratio = hit / len(sub)
        verdict = ("검출 불가 우세(2/3↑)" if ratio >= 2 / 3
                   else "검출력 충분(1/3↓)" if ratio < 1 / 3 else "혼재")
        print(f"  {label:<5} e* {hit:>2}/{len(sub):<2} = {ratio:>5.1%} → **{verdict}**"
              f"   ·  e** {hit2:>2}/{len(sub):<2} = {hit2/len(sub):>5.1%}")

    print(f"\n[테스트별 집계] (§3: ②가 크게 엇갈리면 판정문에 명기)")
    for t in ("H-010", "H-012", "H-013"):
        sub = [r for r in rows if r["test"] == t]
        hit = sum(1 for r in sub if r["A_m"] > E_STAR)
        rng = (min(r["A_m"] for r in sub), max(r["A_m"] for r in sub))
        print(f"  {t}: {hit}/{len(sub)} 초과 · MDE_A 월환산 {rng[0]:.3f} ~ {rng[1]:.3f} %p")

    # e* 민감도 (참고 P1)
    print(f"\n[e* 민감도] 결론 분기를 바꾸려면 e*가 얼마여야 하는가")
    a_sorted = sorted(r["A_m"] for r in rows)
    n = len(a_sorted)
    thr_23 = a_sorted[int(np.ceil(n * (1 - 2 / 3))) - 1]   # 2/3 미만이 되는 경계
    thr_13 = a_sorted[int(np.ceil(n * (1 - 1 / 3))) - 1]
    print(f"  '2/3 이상'에서 벗어나려면 e* > {thr_23:.3f} %p/월 (연 {thr_23*12:.1f}%p)")
    print(f"  '1/3 미만'이 되려면   e* > {thr_13:.3f} %p/월 (연 {thr_13*12:.1f}%p)")

    # 필요 표본수 곡선 (§2.5) — 월별 초과수익 칸만
    print(f"\n[필요 표본수] 월 e%p를 80% 검정력으로 잡으려면 (월별 칸만)")
    print(f"  {'테스트/조건/시장':<26}" + "".join(f"{e:>9}" for e in (0.1, 0.17, 0.33, 0.5, 1.0)))
    for r in rows:
        if r["h"] is not None:
            continue
        sigma = r["se"] * np.sqrt(r["n"])
        need = [(K_MDE * sigma / e) ** 2 for e in (0.1, 0.17, 0.33, 0.5, 1.0)]
        print(f"  {r['test']+' '+r['cond']+' '+r['mkt']:<26}"
              + "".join(f"{x:>9.0f}" for x in need) + " 개월")

    # 무차원 지표 (§2.4)
    print(f"\n[무차원 지표 — e* 비교 불가, 절대 MDE만]")
    for test, cond, mkt, lo, hi in SHARPE_CELLS:
        se = se_from_ci(lo, hi)
        print(f"  {test} {cond:<16} {mkt}: SE {se:.4f} · MDE_A {mde_nominal(se):.4f} (월 Sharpe)")

    print(f"\n[제외] ① A vs B 6칸(CI 없음) · H-009 ③④ 4칸(CI·t값 도입 전) "
          f"· ΔSharpe 4칸(무차원) = 14칸")
    return rows


# ── 자체 검증 ────────────────────────────────────────────────────────────

def selftest():
    # ① 계수·역산 (문서 §3 수식 검증표)
    assert abs(K_MDE - 2.801585) < 1e-6, K_MDE
    assert abs(K_MDE / (2 * Z_A) - 0.714703) < 1e-6
    se = se_from_ci(-2.0, 2.0)                       # CI폭 4 → SE 1.0204
    assert abs(se - 4 / (2 * Z_A)) < 1e-12
    assert abs(mde_nominal(se) - 2.858813) < 1e-5

    # ② 극단값: 검정력 50%면 MDE = 1.96·SE (CI 절반폭)
    z_b50 = norm.ppf(0.50)
    assert abs(z_b50) < 1e-12
    assert abs((Z_A + z_b50) * se - Z_A * se) < 1e-12

    # ③ Bonferroni 방향 — k가 커지면 MDE 증가
    m1, m6, m20 = (mde_bonferroni(se, k) for k in (1, 6, 20))
    assert m1 < m6 < m20
    assert abs(mde_bonferroni(se, 6) / mde_nominal(se) - 1.242111) < 1e-5
    assert abs(mde_bonferroni(se, 1) - mde_nominal(se)) < 1e-12   # k=1이면 명목과 동일

    # ④ α* 보정 방향 — α*가 크면 MDE 증가, α*=0.05면 명목과 동일
    assert mde_alpha_corrected(se, 0.344) > mde_nominal(se)
    assert abs(mde_alpha_corrected(se, 0.05) - mde_nominal(se)) < 1e-9
    scale = Z_A / norm.ppf(1 - 0.344 / 2)
    assert abs(scale - 2.0712) < 0.001, scale

    # ⑤ 월 환산 — h=21이면 그대로, h=126이면 1/6
    assert abs(to_monthly(6.0, 21) - 6.0) < 1e-12
    assert abs(to_monthly(6.0, 126) - 1.0) < 1e-12
    assert to_monthly(1.0, None) == 1.0

    # ⑥ α* 최근접 규칙 — 표에 없는 T는 작은 쪽(오류율 큰 쪽)
    assert alpha_star(126, 70) == 0.344
    assert alpha_star(126, 100) == 0.344      # 70과 148 사이 → 70 채택
    assert alpha_star(126, 200) == 0.208      # 182 이상 → 182
    assert alpha_star(None, 134) is None

    # ⑦ SE 역산이 아카이브 t값을 복원하는가 (H-010 KR ② t=0.36, H-012 KR ④ t=2.28)
    t1 = 0.153 / se_from_ci(-0.675, 0.981)
    t2 = 0.168 / se_from_ci(0.024, 0.312)
    assert abs(t1 - 0.36) < 0.01, t1
    assert abs(t2 - 2.28) < 0.02, t2

    # ⑧ **전사 오류 차단** — 20칸 전부에서 CI 중점 == 아카이브 평균
    #    원 스크립트들이 CI = 평균 ± 1.96·SE로 만들므로 이 항등식은 반드시 성립한다.
    #    ⑦은 2칸만 검사해 H-010 ④ 2칸의 오등재를 놓쳤다(§5.6 P1). ⑧이 그 구멍을 막는다.
    for test, cond, mkt, h, mean, lo, hi, n in CELLS:
        mid = (lo + hi) / 2
        assert abs(mid - mean) < 0.011, (
            f"{test} {cond} {mkt} h={h}: CI 중점 {mid:+.4f} ≠ 아카이브 평균 {mean:+.4f} "
            f"— 인용을 잘못 옮겼다")
    assert len(CELLS) == 20 and len(SHARPE_CELLS) == 4

    print("selftest: 8개 항목 통과 (계수·역산·극단값·Bonferroni·α*보정·월환산·"
          "t값 복원·전사검산 20/20)")


def verify_h012():
    """아카이브에서 읽은 H-012 CI가 실제 실행과 일치하는지 대조 (문서 §4)."""
    import tsmom_vol_target as h12
    for mkt, want in (("kr", (0.024, 0.312)), ("us", (-0.019, 0.144))):
        rows, out, exposures, P, bull, a_eq, bench_eq = h12.run_market(mkt, quiet=True)
        m = lambda g: h12.monthly(out[(g, True)]["equity"].reindex(a_eq.index).ffill())
        d = (m("A") - m("E")).dropna() * 100
        se = d.std(ddof=1) / np.sqrt(len(d))
        lo, hi = d.mean() - 1.96 * se, d.mean() + 1.96 * se
        ok = abs(lo - want[0]) < 0.002 and abs(hi - want[1]) < 0.002
        print(f"[{mkt.upper()}] H-012 ④ CI 실측 [{lo:+.3f}, {hi:+.3f}] vs "
              f"아카이브 [{want[0]:+.3f}, {want[1]:+.3f}] → {'일치' if ok else '**불일치**'}")
        assert ok, "아카이브 인용값이 실제와 다르다 — 전 출처 재확인 필요"


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "--selftest":
        selftest()
    elif arg == "--verify-h012":
        verify_h012()
    else:
        main()
