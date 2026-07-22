"use strict";

// ── 상태
const state = {
  rows: [],
  meta: null,
  filters: { country: "all", asset: "all", signal: "all", mcap: 0 },
  sort: "stars",
};
const groupRsCache = {};   // {US: {sectors, industries}}
const detailCache = {};    // {ticker: json}

const FILT_LABEL = { L: "장기", S: "단기" };
const NHIGH_TITLE = { ATH: "역대최고(ATH)", "20": "20일신고가", "55": "55일신고가" };
const HOLD_NOTE = { 20: "1개월", 60: "3개월", 63: "3개월", 126: "6개월", 252: "1년" };

// ── 유틸
function fmtDate(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (isNaN(d)) return String(iso).slice(0, 10);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}
function money(v, country) {
  if (v == null) return "—";
  if (country === "KR") return "₩" + Math.round(v).toLocaleString("ko-KR");
  const dp = v >= 100 ? 2 : v >= 1 ? 2 : 4;
  return "$" + v.toLocaleString("en-US", { minimumFractionDigits: dp, maximumFractionDigits: dp });
}
function pct(v) { return v == null ? "—" : (v > 0 ? "+" : "") + v.toFixed(1) + "%"; }
function num(v, d = 2) { return v == null ? "—" : v.toFixed(d); }
function starStr(n) { return n == null ? "" : "★".repeat(n) + "☆".repeat(3 - n); }

// ── 데이터 로드
async function load() {
  const grid = document.getElementById("grid");
  grid.innerHTML = Array.from({ length: 6 }, () => '<div class="skeleton"></div>').join("");
  try {
    const res = await fetch("data/buy-signals.json", { cache: "no-cache" });
    const data = await res.json();
    state.rows = data.rows || [];
    state.meta = data.meta || {};
    renderBanner();
    renderFoot(data);
    render();
  } catch (e) {
    grid.innerHTML = "";
    const err = document.getElementById("empty");
    err.hidden = false;
    err.textContent = "데이터를 불러오지 못했습니다. (data/buy-signals.json)";
  }
}

const MARKET_LABEL = { US: "미국", KR: "한국" };
const STATE_LABEL = { fresh: "최신", intraday: "장중", stale: "지연" };

function renderBanner() {
  const el = document.getElementById("banner");
  const m = state.meta || {};
  const warnings = m.warnings || [];

  if (!window.BannerStatus) {
    el.classList.toggle("warn", warnings.length > 0);
    el.innerHTML =
      `<span class="dot"></span><span>신호 기준일 <span class="asof">${fmtDate(m.built_at)}</span></span>` +
      warnings.map((w) => `<span class="warn-item">⚠ ${w}</span>`).join("");
    return;
  }

  const rsAsof = (m.star && m.star.rs_asof) || {};
  const byCountry = m.by_country || {};
  const now = new Date();

  // 데이터가 있는 시장만 칩으로 표시
  const markets = ["US", "KR"].filter((k) => (byCountry[k] || 0) > 0);
  const chips = markets.map((k) => {
    const st = window.BannerStatus.bannerStatus({
      market: k, builtAt: m.built_at, now,
    });
    const date = window.BannerStatus.fmtDateOnly(rsAsof[k]);
    return {
      market: k, ...st, date,
      html: `<span class="mchip is-${st.state}">${MARKET_LABEL[k]} <b>${date}</b>` +
            `<span class="mchip-state">${STATE_LABEL[st.state]}</span></span>`,
    };
  });

  const stale = chips.find((c) => c.state === "stale");
  const intraday = chips.filter((c) => c.state === "intraday");
  const hasWarn = warnings.length > 0 || Boolean(stale);
  el.classList.toggle("warn", hasWarn);

  const notes = [];
  if (stale) {
    notes.push({ cls: "warn-item", text: `⚠ 데이터가 ${stale.staleDays}일째 갱신되지 않았습니다 — 배치를 확인하세요` });
  } else if (intraday.length) {
    const names = intraday.map((c) => MARKET_LABEL[c.market]).join("·");
    notes.push({ cls: "note", text: `지금 ${names} 장중 — 직전 완결일 기준입니다` });
  }
  warnings.forEach((w) => notes.push({ cls: "warn-item", text: `⚠ ${w}` }));

  el.innerHTML =
    `<span class="dot"></span>` +
    (chips.length
      ? `<span class="mchips">${chips.map((c) => c.html).join("")}</span>`
      : `<span>신호 기준일 <span class="asof">${fmtDate(m.built_at)}</span></span>`) +
    notes.map((n) => `<span class="${n.cls}">${n.text}</span>`).join("");
}

function renderFoot(data) {
  const m = data.meta || {};
  const el = document.getElementById("footMeta");
  el.textContent = `built ${fmtDate(m.built_at)} · 신호 ${m.grand_total ?? "?"}건 → ${data.total ?? state.rows.length}티커 · RS≥${m.rs_min} · 표본≥${m.min_sample}`;
}

// ── 필터 + 정렬
function applyFilters(rows) {
  const f = state.filters;
  return rows.filter((r) => {
    if (f.country !== "all" && r.country !== f.country) return false;
    if (f.asset !== "all" && r.asset !== f.asset) return false;
    if (f.signal !== "all" && r.signal !== f.signal) return false;
    if (f.mcap > 0 && !(r.market_cap_usd != null && r.market_cap_usd >= f.mcap)) return false;
    return true;
  });
}
function sortRows(rows) {
  const s = state.sort;
  const arr = rows.slice();
  if (s === "stars") {
    arr.sort((a, b) => {
      const sa = a.stars == null ? -1 : a.stars;
      const sb = b.stars == null ? -1 : b.stars;
      if (sb !== sa) return sb - sa;
      return b.ev - a.ev;
    });
  } else if (s === "ev") arr.sort((a, b) => b.ev - a.ev);
  else if (s === "win") arr.sort((a, b) => b.win_rate - a.win_rate);
  else if (s === "rs") arr.sort((a, b) => b.rs - a.rs);
  return arr;
}

function render() {
  const filtered = sortRows(applyFilters(state.rows));
  const top = filtered.slice(0, 100);
  document.getElementById("count").textContent = `${top.length}종목`;
  const grid = document.getElementById("grid");
  const empty = document.getElementById("empty");
  grid.innerHTML = "";
  empty.hidden = top.length > 0;
  const tpl = document.getElementById("card-tpl");
  top.forEach((r, i) => grid.appendChild(buildCard(tpl, r, i + 1)));
}

function buildCard(tpl, r, rank) {
  const node = tpl.content.firstElementChild.cloneNode(true);
  node.style.animationDelay = Math.min(rank * 18, 500) + "ms";
  node.querySelector(".rank").textContent = String(rank).padStart(2, "0");
  const st = node.querySelector(".stars");
  if (r.asset === "etf") { st.textContent = "ETF"; st.classList.add("etf"); }
  else st.textContent = starStr(r.stars);
  node.querySelector(".nm").textContent = r.name || r.ticker;
  node.querySelector(".tk").textContent = r.ticker + (r.category ? " · " + r.category : "");
  const badge = node.querySelector(".sig-badge");
  badge.textContent = r.signal;
  badge.classList.add("sig-" + r.signal);
  node.querySelector(".phrase").textContent = r.phrase || "";

  const head = node.querySelector(".card-head");
  const body = node.querySelector(".card-body");
  head.addEventListener("click", () => toggleCard(node, head, body, r));
  return node;
}

async function toggleCard(node, head, body, r) {
  const open = node.classList.toggle("open");
  head.setAttribute("aria-expanded", open ? "true" : "false");
  body.hidden = !open;
  if (open && !body.dataset.loaded) {
    body.innerHTML = '<div class="bt-loading">불러오는 중…</div>';
    await renderBody(body, r);
    body.dataset.loaded = "1";
  }
}

async function renderBody(body, r) {
  const cur = money(r.cur_price, r.country);
  const zl = money(r.zone_low, r.country);
  const zh = money(r.zone_high, r.country);
  const inZone = r.cur_price >= r.zone_low && r.cur_price <= r.zone_high;

  const chips = await groupChips(r);
  const metrics = `
    <div class="metrics">
      ${metric("종합 RS", r.rs, "")}
      ${metric("승률", r.win_rate, "%")}
      ${metric("표본수", r.n, "")}
      ${metric("평균이익", r.avg_win, "%", "pos")}
      ${metric("평균손실", r.avg_loss, "%", "neg")}
      ${metric("손익비", r.pl_ratio, "")}
    </div>`;

  let table = '<div class="bt-loading">성적표 불러오는 중…</div>';
  body.innerHTML = `
    ${chips}
    <div class="zone-box">
      <span class="zlabel">매수 구간</span>
      <span class="zval">${zl} ~ ${zh}</span>
      ${inZone ? '<span class="inzone">구간 내</span>' : ""}
      <span class="zclose">· 종가 ${cur}</span>
    </div>
    ${metrics}
    <div class="bt-wrap">${table}</div>`;

  const btWrap = body.querySelector(".bt-wrap");
  try {
    const detail = await getDetail(r.ticker);
    btWrap.innerHTML = renderTable(r, detail);
  } catch (e) {
    btWrap.innerHTML = '<div class="bt-note">8기간 성적표를 불러오지 못했습니다.</div>';
  }
}

function metric(k, v, unit, cls = "") {
  const val = v == null ? "—" : (unit === "%" ? v.toFixed(1) : v.toLocaleString()) + unit;
  return `<div class="metric"><div class="mk">${k}</div><div class="mv ${cls}">${val}</div></div>`;
}

// ── 섹터·산업 RS 칩
async function groupChips(r) {
  if (r.country !== "US" || (!r.sector && !r.industry)) return "";
  const gr = await getGroupRs(r.country);
  if (!gr) return "";
  const chips = [];
  if (r.sector && gr.sectors[r.sector]) chips.push(chip(r.sector, gr.sectors[r.sector].avg_rs));
  if (r.industry && gr.industries[r.industry]) chips.push(chip(r.industry, gr.industries[r.industry].avg_rs));
  return chips.length ? `<div class="group-chips">${chips.join("")}</div>` : "";
}
function chip(name, rs) { return `<span class="gchip">${name} <b>${Math.round(rs)}</b></span>`; }

async function getGroupRs(country) {
  if (groupRsCache[country] !== undefined) return groupRsCache[country];
  try {
    const [s, i] = await Promise.all([
      fetch(`data/rs/${country.toLowerCase()}/sectors.json`).then((x) => x.json()),
      fetch(`data/rs/${country.toLowerCase()}/industries.json`).then((x) => x.json()),
    ]);
    groupRsCache[country] = { sectors: s, industries: i };
  } catch (e) {
    groupRsCache[country] = null;
  }
  return groupRsCache[country];
}

async function getDetail(ticker) {
  if (detailCache[ticker]) return detailCache[ticker];
  const safe = ticker.replace(/\//g, "_");
  const d = await fetch(`data/detail/${safe}.json`).then((x) => x.json());
  detailCache[ticker] = d;
  return d;
}

// ── 8기간 성적표
function renderTable(r, detail) {
  let title, periods, sampleLabel;
  if (r.signal === "이평") {
    const [filt, ma] = r.detail.split("/");
    const maType = ma.startsWith("SMA") ? "SMA" : "EMA";
    const maPer = ma.slice(3);
    const key = `${filt}|${maType}|${maPer}`;
    periods = (detail.ma.stats[key] || []).map((x) => ({
      hold: x.period, win: x.win_rate, aw: x.avg_win, al: x.avg_loss, pl: x.pl_ratio, n: x.touch_count,
    }));
    title = `${ma} ${FILT_LABEL[filt]} · 이평 눌림목 8기간 성적표`;
    sampleLabel = "터치수";
  } else if (r.signal === "박스") {
    periods = (detail.box.periods_all[r.detail] || []).map((x) => ({
      hold: x.period, win: x.win_rate, aw: x.avg_win, al: x.avg_loss, pl: x.pl_ratio, n: x.cnt,
    }));
    title = `박스돌파 ${FILT_LABEL[r.detail]} 8기간 성적표`;
    sampleLabel = "돌파수";
  } else {
    const entry = detail.nhigh.entries[r.detail] || { holds: [] };
    periods = entry.holds.map((x) => ({
      hold: x.hold, win: x.win_rate, aw: x.avg_win, al: x.avg_loss, pl: x.pl_ratio, n: x.n,
    }));
    title = `${NHIGH_TITLE[r.detail]} · 신고가 8기간 성적표`;
    sampleLabel = "표본수";
  }

  const rowsHtml = periods.map((p) => {
    const opt = p.hold === r.hold_period;
    const noLoss = p.al == null;
    return `<tr class="${opt ? "optimal" : ""}">
      <td>${opt ? "★ " : ""}${p.hold}일</td>
      <td>${p.win == null ? "—" : p.win.toFixed(1)}</td>
      <td class="pos">${p.aw == null ? "—" : "+" + p.aw.toFixed(1)}</td>
      <td class="neg">${noLoss ? "—" : p.al.toFixed(1)}</td>
      <td>${noLoss ? "—" : p.pl.toFixed(2)}</td>
      <td>${p.n ?? "—"}</td>
    </tr>`;
  }).join("");

  const noteKeys = periods.map((p) => p.hold).filter((h) => HOLD_NOTE[h]);
  const note = [...new Set(noteKeys)].map((h) => `${h}일=${HOLD_NOTE[h]}`).join(" · ");

  return `
    <div class="table-title">${title}</div>
    <table class="bt-table">
      <thead><tr>
        <th>보유기간</th><th>승률</th><th>평균이익</th><th>평균손실</th><th>손익비</th><th>${sampleLabel}</th>
      </tr></thead>
      <tbody>${rowsHtml}</tbody>
    </table>
    <div class="bt-note">${note} · ★ = 최적 보유기간(표본≥${state.meta.min_sample} 중 EV 최대) · 손실 없는 기간은 —</div>`;
}

// ── 컨트롤 이벤트
function wireControls() {
  document.getElementById("filters").addEventListener("click", (e) => {
    const btn = e.target.closest(".chip");
    if (!btn) return;
    const group = btn.closest(".filter-group");
    const key = group.dataset.key;
    group.querySelectorAll(".chip").forEach((c) => c.classList.remove("is-active"));
    btn.classList.add("is-active");
    const val = btn.dataset.val;
    state.filters[key] = key === "mcap" ? Number(val) : val;
    render();
  });
  document.getElementById("sort").addEventListener("change", (e) => {
    state.sort = e.target.value;
    render();
  });
}

wireControls();
load();
