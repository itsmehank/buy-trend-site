const test = require("node:test");
const assert = require("node:assert");
const { fmtDateOnly, bannerStatus } = require("../site/banner-status.js");

// ── 날짜 포맷: 타임존이 달라도 절대 밀리지 않아야 한다
test("fmtDateOnly는 타임존과 무관하게 같은 값", () => {
  assert.strictEqual(fmtDateOnly("2026-07-21"), "07-21");
  assert.strictEqual(fmtDateOnly(""), "—");
  assert.strictEqual(fmtDateOnly(null), "—");
});

const BUILT = "2026-07-22T02:00:00Z";

// ── 장중 (KR 평일 11:00 KST = 02:00 UTC)
test("KR 평일 장중이면 intraday", () => {
  const r = bannerStatus({ market: "KR", builtAt: BUILT, now: new Date("2026-07-22T02:00:00Z") });
  assert.strictEqual(r.state, "intraday");
});

// ── 장 마감 후 (KR 평일 16:30 KST = 07:30 UTC)
test("KR 평일 장 마감 후면 fresh", () => {
  const r = bannerStatus({ market: "KR", builtAt: BUILT, now: new Date("2026-07-22T07:30:00Z") });
  assert.strictEqual(r.state, "fresh");
});

// ── 주말: 개장 시간대여도 장중이 아니다 (요일 조건 검증)
test("토요일 10:00 KST는 intraday가 아니다", () => {
  const r = bannerStatus({ market: "KR", builtAt: "2026-07-25T00:00:00Z", now: new Date("2026-07-25T01:00:00Z") });
  assert.strictEqual(r.state, "fresh");
});

// ── US 서머타임: 평일 10:00 ET = 14:00 UTC (EDT)
test("US 평일 장중이면 intraday", () => {
  const r = bannerStatus({ market: "US", builtAt: "2026-07-22T13:00:00Z", now: new Date("2026-07-22T14:00:00Z") });
  assert.strictEqual(r.state, "intraday");
});

// ── 지연: 임계 4일 초과
test("built_at이 5일 지나면 stale", () => {
  const r = bannerStatus({ market: "US", builtAt: "2026-07-17T02:00:00Z", now: new Date("2026-07-22T02:00:00Z") });
  assert.strictEqual(r.state, "stale");
  assert.strictEqual(r.staleDays, 5);
});

// ── 경계: KR 누락 시 최대 공백 3.00일은 stale이 아니어야 한다
test("3일 경과는 stale이 아니다 (거짓 경보 방지)", () => {
  const r = bannerStatus({ market: "US", builtAt: "2026-07-17T22:00:00Z", now: new Date("2026-07-20T22:00:00Z") });
  assert.notStrictEqual(r.state, "stale");
});

// ── 우선순위: 지연이 장중보다 우선
test("stale이 intraday보다 우선", () => {
  const r = bannerStatus({ market: "KR", builtAt: "2026-07-10T02:00:00Z", now: new Date("2026-07-22T02:00:00Z") });
  assert.strictEqual(r.state, "stale");
});
