"use strict";

// 배너 상태 판정 — 브라우저/Node 공용 순수 함수.
// 설계: docs/superpowers/specs/2026-07-22-banner-market-status-design.md

const STALE_DAYS_LIMIT = 4; // 4 캘린더일 초과면 지연

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

if (typeof module !== "undefined" && module.exports) {
  module.exports = { fmtDateOnly, bannerStatus, STALE_DAYS_LIMIT };
} else {
  window.BannerStatus = { fmtDateOnly, bannerStatus, STALE_DAYS_LIMIT };
}
