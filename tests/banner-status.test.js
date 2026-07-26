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

// ── 백테스트 구간 표기 (표본수를 해석하려면 몇 년치인지 알아야 한다)
const { btWindowText } = require("../site/banner-status.js");

const BW = {
  years: 15,
  US: { start: "2011-07-26", end: "2026-07-24", bars_median: 3776 },
  KR: { start: "2014-04-28", end: "2026-07-24", bars_median: 2999 },
};

test("btWindowText는 구간과 시장별 실측 거래일·연수를 표기", () => {
  assert.strictEqual(
    btWindowText(BW),
    " · 백테스트 2011-07~2026-07(US 3,776일≈15.0년 · KR 2,999일≈11.9년)");
});

// 설정값 15년을 그대로 쓰면 KR(실제 11.9년)을 오해하게 된다 — 실측만 쓴다
test("btWindowText는 설정값 years를 표기에 쓰지 않는다", () => {
  const out = btWindowText(BW);
  assert.ok(!out.includes("15년 2011"), "설정 연수를 헤드라인에 쓰면 안 됨");
  assert.ok(out.includes("11.9년"), "KR 실측 연수가 드러나야 함");
});

test("btWindowText는 한 시장만 있어도 그 시장만 표기", () => {
  assert.strictEqual(
    btWindowText({ years: 15, US: BW.US, KR: {} }),
    " · 백테스트 2011-07~2026-07(US 3,776일≈15.0년)");
});

// 첫 배치 전에 프론트만 배포되면 meta에 키가 없다 — 이때 화면이 깨지면 안 된다
test("btWindowText는 데이터가 없으면 빈 문자열", () => {
  assert.strictEqual(btWindowText(undefined), "");
  assert.strictEqual(btWindowText(null), "");
  assert.strictEqual(btWindowText({}), "");
  assert.strictEqual(btWindowText({ years: 15, US: {}, KR: {} }), "");
});

test("btWindowText는 필드가 반쯤 빈 시장을 건너뛴다", () => {
  assert.strictEqual(btWindowText({ years: 15, US: { start: "2011-07-26" }, KR: {} }), "");
});

// ── 승률 신뢰구간 (Wilson) — 표본수는 반드시 n_eff 기준
const { wilsonInterval } = require("../site/banner-status.js");

const r1 = (x) => Math.round(x * 10) / 10;

test("wilsonInterval은 표본이 적을수록 구간이 넓어진다", () => {
  // 같은 승률 96.7%라도 표본수에 따라 정밀도가 다르다
  const wide = wilsonInterval(96.67, 15);   // 독립 표본 기준
  const narrow = wilsonInterval(96.67, 60); // 겹친 표본수를 그대로 쓴 경우
  assert.ok(wide.hi - wide.lo > narrow.hi - narrow.lo,
    "표본이 적으면 구간이 더 넓어야 함");
  assert.strictEqual(r1(narrow.lo), 88.6);
  assert.strictEqual(r1(narrow.hi), 99.1);
});

test("wilsonInterval은 0~100 밖으로 나가지 않는다", () => {
  const perfect = wilsonInterval(100, 5);
  assert.ok(perfect.hi <= 100 && perfect.lo >= 0);
  const zero = wilsonInterval(0, 5);
  assert.ok(zero.lo >= 0 && zero.hi <= 100);
});

test("wilsonInterval은 표본 1개면 사실상 판단 불가 수준으로 넓다", () => {
  const one = wilsonInterval(100, 1);
  assert.ok(one.hi - one.lo > 75, `표본 1개 구간이 ${one.hi - one.lo}로 너무 좁음`);
});

test("wilsonInterval은 데이터가 없으면 null", () => {
  assert.strictEqual(wilsonInterval(null, 15), null);
  assert.strictEqual(wilsonInterval(96.67, 0), null);
  assert.strictEqual(wilsonInterval(96.67, null), null);
  assert.strictEqual(wilsonInterval(96.67, undefined), null);
});
