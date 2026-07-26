"use strict";

// 배너 상태 판정 — 브라우저/Node 공용 순수 함수.
// 설계: docs/superpowers/specs/2026-07-22-banner-market-status-design.md

// 4일(24시간×4) 초과면 지연. 달력 날짜 차이가 아니라 built_at 이후 '경과 duration'
// 기준 — 배치 실행 간 최대 공백(정상 2.40일, KR 실행 누락 시 3.00일) 계산과 같은 단위다.
const STALE_DAYS_LIMIT = 4;

const MARKET = {
  KR: { tz: "Asia/Seoul", open: 9 * 60, close: 15 * 60 + 30 },
  US: { tz: "America/New_York", open: 9 * 60 + 30, close: 16 * 60 },
};

// "2026-07-21" → "07-21". new Date를 쓰지 않아 타임존 영향이 없다.
function fmtDateOnly(s) {
  if (!s || typeof s !== "string") return "—";
  const parts = s.slice(0, 10).split("-");
  if (parts.length !== 3) return "—";
  return `${parts[1]}-${parts[2]}`;
}

// 특정 타임존에서의 요일(0=일)과 분 단위 시각을 구한다.
function localParts(date, timeZone) {
  const fmt = new Intl.DateTimeFormat("en-US", {
    timeZone,
    weekday: "short",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
  const parts = Object.fromEntries(fmt.formatToParts(date).map((p) => [p.type, p.value]));
  const weekdayIndex = { Sun: 0, Mon: 1, Tue: 2, Wed: 3, Thu: 4, Fri: 5, Sat: 6 }[parts.weekday];
  const hour = Number(parts.hour) % 24;
  return { weekday: weekdayIndex, minutes: hour * 60 + Number(parts.minute) };
}

function bannerStatus({ market, builtAt, now }) {
  const cfg = MARKET[market];
  const staleDays = builtAt
    ? Math.floor((now.getTime() - new Date(builtAt).getTime()) / 86400000)
    : 0;

  if (staleDays > STALE_DAYS_LIMIT) return { state: "stale", staleDays };

  if (cfg) {
    const { weekday, minutes } = localParts(now, cfg.tz);
    const isWeekday = weekday >= 1 && weekday <= 5;
    if (isWeekday && minutes >= cfg.open && minutes < cfg.close) {
      return { state: "intraday", staleDays };
    }
  }
  return { state: "fresh", staleDays };
}

// 백테스트가 몇 년치·어느 구간 데이터로 계산됐는지 한 줄로 만든다.
// 8기간 성적표의 표본수는 전부 이 구간에서 나온 값이라, 구간을 모르면
// 표본수만으로 신뢰도를 판단할 수 없다(HISTORY_YEARS는 명세서 ✅가 아니라
// 자체 결정값이라 더더욱 노출이 필요하다).
// backtest_window가 없는 옛 JSON(첫 배치 전 배포)에서는 빈 문자열을 돌려준다.
const TRADING_DAYS_PER_YEAR = 252;

function btWindowText(bw) {
  if (!bw) return "";
  const ms = ["US", "KR"].filter(
    (k) => bw[k] && bw[k].start && bw[k].end && bw[k].bars_median != null);
  if (!ms.length) return "";
  const ym = (d) => String(d).slice(0, 7);
  const starts = ms.map((k) => bw[k].start).sort();
  const ends = ms.map((k) => bw[k].end).sort();
  // 설정값(HISTORY_YEARS)이 아니라 실측 봉 수로 연수를 낸다. KR은 데이터 시작이
  // 늦어 15년을 요청해도 실제로는 12년이 안 된다 — 설정값을 표기하면 표본이
  // 몇 년치인지 오해하게 만들어, 이 표기를 넣는 목적 자체가 무너진다.
  const per = ms.map((k) => {
    const b = bw[k].bars_median;
    return `${k} ${b.toLocaleString("en-US")}일≈${(b / TRADING_DAYS_PER_YEAR).toFixed(1)}년`;
  }).join(" · ");
  return ` · 백테스트 ${ym(starts[0])}~${ym(ends[ends.length - 1])}(${per})`;
}

// 승률의 95% 신뢰구간(Wilson score interval).
//
// 표본수는 n이 아니라 n_eff(겹치지 않는 보유구간 수)를 넣어야 한다. n은 신호가
// 뜬 날마다 세므로 장기 보유일수록 같은 기간을 중복해 세고, 그 값으로 구간을
//내면 실제보다 훨씬 좁게 나와 정밀도를 과장한다.
//
// 주의(근사): win_rate 자체는 n개 표본 전체로 계산된 값인데 여기서는 표본수만
// n_eff로 바꿔 넣는다. 독립 부분집합만의 승률은 어느 구간을 고르냐에 따라
// 달라져 하나로 정할 수 없기 때문이다. 점추정은 전체 표본에서, 정밀도는 실질
// 표본수에서 가져오는 방식으로, 겹치는 표본을 다룰 때 쓰는 통상적인 근사다.
const Z95 = 1.96;

function wilsonInterval(winRatePct, n) {
  if (winRatePct == null || !n || n <= 0) return null;
  const p = winRatePct / 100;
  const z2 = Z95 * Z95;
  const d = 1 + z2 / n;
  const center = (p + z2 / (2 * n)) / d;
  const half = (Z95 * Math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / d;
  return {
    lo: Math.max(0, (center - half) * 100),
    hi: Math.min(100, (center + half) * 100),
  };
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { fmtDateOnly, bannerStatus, btWindowText, wilsonInterval, STALE_DAYS_LIMIT };
} else {
  window.BannerStatus = { fmtDateOnly, bannerStatus, btWindowText, wilsonInterval, STALE_DAYS_LIMIT };
}
