// Birdsong dashboard. Plain JavaScript, no build step, no third-party code.
"use strict";

const API = "api/v1"; // relative, so the page also works behind a reverse-proxy path prefix
const MAX_ROWS = 100;
const LEGEND_SPECIES = 7; // the rest of a day's species are grouped as "Other"
const TOP_BARS = 12;

const state = {
  tz: undefined,
  window: "24h",
  date: null,
  rows: new Map(), // detection id -> <tr>
  playing: null, // detection id
  refreshTimer: null,
};

const $ = (selector) => document.querySelector(selector);

// ---------- helpers ----------

async function api(path) {
  const res = await fetch(`${API}/${path}`, { headers: { Accept: "application/json" } });
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`;
    try {
      message = (await res.json()).error || message;
    } catch (_) {
      /* not JSON */
    }
    throw new Error(message);
  }
  return res.json();
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children) {
    if (child === null || child === undefined) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function svg(tag, attrs = {}, ...children) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  for (const child of children) node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  return node;
}

function formatter(options) {
  try {
    return new Intl.DateTimeFormat(undefined, { timeZone: state.tz, ...options });
  } catch (_) {
    return new Intl.DateTimeFormat(undefined, options); // unknown time zone: browser's own
  }
}

const fmt = {
  time: (iso) => formatter({ hour: "2-digit", minute: "2-digit", second: "2-digit" }).format(new Date(iso)),
  dateTime: (iso) => formatter({ month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }).format(new Date(iso)),
  percent: (x) => `${Math.round(x * 100)}%`,
  number: (n) => new Intl.NumberFormat().format(n),
  relative(iso) {
    const seconds = Math.round((Date.now() - new Date(iso).getTime()) / 1000);
    if (seconds < 45) return "just now";
    const minutes = Math.round(seconds / 60);
    if (minutes < 60) return `${minutes} min ago`;
    const hours = Math.round(minutes / 60);
    if (hours < 36) return `${hours} h ago`;
    return `${Math.round(hours / 24)} d ago`;
  },
};

function todayInStation() {
  // en-CA formats as YYYY-MM-DD.
  try {
    return new Intl.DateTimeFormat("en-CA", { timeZone: state.tz, year: "numeric", month: "2-digit", day: "2-digit" }).format(new Date());
  } catch (_) {
    return new Date().toISOString().slice(0, 10);
  }
}

function showMessage(container, text, isError = false) {
  container.replaceChildren(el("p", { class: isError ? "empty error" : "empty" }, text));
}

function colour(index) {
  return `var(--c${(index % 8) + 1})`;
}

// ---------- status ----------

async function loadHealth() {
  const status = $("#status");
  try {
    const h = await api("health");
    const s = h.stats;
    const parts = [];
    const since = h.seconds_since_last_chunk;
    let level = "ok";
    if (since === null) {
      parts.push(h.uptime_s > 60 ? "No audio received yet" : "Starting…");
      if (h.uptime_s > 60) level = "warn";
    } else if (since > 30) {
      parts.push(`No audio for ${fmt.number(Math.round(since))} s`);
      level = "bad";
    } else {
      parts.push("Listening");
    }
    parts.push(`${fmt.number(s.chunks_processed)} chunks`);
    if (s.mean_inference_ms !== null) parts.push(`${Math.round(s.mean_inference_ms)} ms per chunk`);
    parts.push(`${fmt.number(s.detections)} detections since start`);
    if (s.chunks_dropped > 0) {
      parts.push(`${fmt.number(s.chunks_dropped)} chunks dropped`);
      if (level === "ok") level = "warn";
    }
    status.textContent = parts.join(" · ");
    status.dataset.state = level;
  } catch (e) {
    status.textContent = `Cannot reach Birdsong: ${e.message}`;
    status.dataset.state = "bad";
  }
}

// ---------- latest birds (horizontal bars) ----------

async function loadLatest() {
  const container = $("#latest-chart");
  try {
    const data = await api(`stats/recent?window=${encodeURIComponent(state.window)}`);
    renderLatest(container, data);
  } catch (e) {
    showMessage(container, `Could not load: ${e.message}`, true);
  }
}

function renderLatest(container, data) {
  const species = data.species.slice(0, TOP_BARS);
  if (species.length === 0) {
    showMessage(container, `No detections in the last ${data.window}.`);
    return;
  }
  const max = Math.max(...species.map((s) => s.count));
  const grid = el("div", { class: "bars", role: "list" });
  for (const s of species) {
    const label = `${s.common_name}: ${s.count} detection${s.count === 1 ? "" : "s"}, last ${fmt.dateTime(s.last_seen)}`;
    grid.append(
      el("span", { class: "name", title: s.scientific_name, role: "listitem", "aria-label": label }, s.common_name),
      el("span", { class: "track", "aria-hidden": "true" }, el("span", { class: "fill", style: `width:${(100 * s.count) / max}%` })),
      el("span", { class: "count", "aria-hidden": "true" }, fmt.number(s.count)),
    );
  }
  const extra = data.species.length - species.length;
  container.replaceChildren(grid);
  if (extra > 0) container.append(el("p", { class: "muted" }, `and ${extra} more species`));
}

// ---------- by hour (stacked columns) ----------

async function loadDaily() {
  const container = $("#daily-chart");
  try {
    const data = await api(`stats/daily?date=${encodeURIComponent(state.date)}`);
    renderDaily(container, $("#daily-legend"), data);
  } catch (e) {
    $("#daily-legend").replaceChildren();
    showMessage(container, `Could not load: ${e.message}`, true);
  }
}

function niceMax(n) {
  // Smallest of 5, 10, 20, 25, 50, 100, 200, 250, 500, ... that is at least n; all divide by 5.
  if (n <= 5) return 5;
  const magnitude = 10 ** Math.floor(Math.log10(n));
  for (const step of [1, 2, 2.5, 5, 10]) {
    const candidate = step * magnitude;
    if (candidate >= n && candidate % 5 === 0) return candidate;
  }
  return 10 * magnitude;
}

function renderDaily(container, legend, data) {
  legend.replaceChildren();
  if (data.species.length === 0) {
    showMessage(container, `No detections on ${data.date}.`);
    return;
  }
  const shown = data.species.slice(0, LEGEND_SPECIES);
  const rest = data.species.slice(LEGEND_SPECIES);
  const series = shown.map((s, i) => ({ name: s.common_name, hours: s.by_hour, colour: colour(i) }));
  if (rest.length > 0) {
    const hours = Array.from({ length: 24 }, (_, h) => rest.reduce((sum, s) => sum + s.by_hour[h], 0));
    series.push({ name: `Other (${rest.length})`, hours, colour: "var(--c8)" });
  }

  const W = 480, H = 220, left = 30, right = 6, top = 10, bottom = 24;
  const plotW = W - left - right, plotH = H - top - bottom;
  const totals = Array.from({ length: 24 }, (_, h) => series.reduce((sum, s) => sum + s.hours[h], 0));
  const yMax = niceMax(Math.max(...totals));
  const colW = plotW / 24;
  const y = (v) => top + plotH - (v / yMax) * plotH;

  const chart = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `Detections per hour on ${data.date}` });
  // niceMax() returns 5, 10, 20, 25, 50, 100, ... so five intervals always give whole numbers.
  for (let i = 0; i <= 5; i++) {
    const v = (yMax * i) / 5;
    chart.append(
      svg("line", { class: "gridline", x1: left, x2: W - right, y1: y(v), y2: y(v) }),
      svg("text", { class: "axis", x: left - 5, y: y(v) + 4, "text-anchor": "end" }, String(Math.round(v))),
    );
  }
  for (let h = 0; h < 24; h++) {
    let base = 0;
    for (const s of series) {
      const v = s.hours[h];
      if (v === 0) continue;
      const rect = svg("rect", {
        x: left + h * colW + 2,
        y: y(base + v),
        width: Math.max(colW - 4, 1),
        height: Math.max(y(base) - y(base + v), 1),
        style: `fill:${s.colour}`,
        rx: 2,
      });
      rect.append(svg("title", {}, `${String(h).padStart(2, "0")}:00 ${s.name}: ${v}`));
      chart.append(rect);
      base += v;
    }
    if (h % 3 === 0) {
      chart.append(svg("text", { class: "axis", x: left + h * colW + colW / 2, y: H - 8, "text-anchor": "middle" }, String(h).padStart(2, "0")));
    }
  }
  container.replaceChildren(el("div", { class: "daily" }, chart));
  for (const s of series) {
    const total = s.hours.reduce((a, b) => a + b, 0);
    legend.append(el("li", {}, el("span", { class: "swatch", style: `background:${s.colour}` }), `${s.name} ${total}`));
  }
}

// ---------- playback ----------

function playButton(detectionId, available, label) {
  const button = el("button", {
    type: "button",
    class: "play",
    "data-id": detectionId,
    "aria-label": available ? `Play ${label}` : `No recording for ${label}`,
    "aria-pressed": "false",
    disabled: !available,
  }, "▶");
  button.addEventListener("click", () => togglePlay(detectionId));
  return button;
}

function syncPlayButtons() {
  for (const button of document.querySelectorAll("button.play")) {
    const active = Number(button.dataset.id) === state.playing;
    button.setAttribute("aria-pressed", String(active));
    button.textContent = active ? "❚❚" : "▶";
  }
}

function togglePlay(id) {
  const player = $("#player");
  if (state.playing === id && !player.paused) {
    player.pause();
    state.playing = null;
  } else {
    player.src = `${API}/detections/${id}/audio`;
    player.play().catch((e) => console.warn("playback failed", e));
    state.playing = id;
  }
  syncPlayButtons();
}

function openViewer(d) {
  $("#player").pause();
  state.playing = null;
  syncPlayButtons();
  $("#viewer-title").textContent = d.common_name;
  $("#viewer-meta").textContent = `${d.scientific_name} · ${fmt.dateTime(d.detected_at)} · ${fmt.percent(d.confidence)} · ${d.source_id}`;
  const img = $("#viewer-image");
  img.hidden = !d.spectrogram_path;
  $("#viewer-noimage").hidden = Boolean(d.spectrogram_path);
  if (d.spectrogram_path) img.src = `${API}/detections/${d.id}/spectrogram.png`;
  const audio = $("#viewer-audio");
  audio.hidden = !d.clip_path;
  if (d.clip_path) audio.src = `${API}/detections/${d.id}/audio`;
  $("#viewer").showModal();
}

// ---------- recent detections ----------

function detectionRow(d) {
  const label = `${d.common_name} at ${fmt.time(d.detected_at)}`;
  const recording = el("div", { class: "rec" }, playButton(d.id, Boolean(d.clip_path), label));
  if (d.spectrogram_path) {
    recording.append(
      el("img", {
        class: "thumb",
        src: `${API}/detections/${d.id}/spectrogram.png`,
        alt: `Spectrogram of ${label}`,
        loading: "lazy",
        width: 160,
        height: 60,
        onclick: () => openViewer(d),
      }),
    );
  } else if (!d.clip_path) {
    recording.append(el("span", { class: "pending" }, isRecent(d) ? "saving…" : "not kept"));
  }
  return el(
    "tr",
    { "data-id": d.id },
    el("td", {}, fmt.time(d.detected_at), el("span", { class: "when", "data-iso": d.detected_at }, fmt.relative(d.detected_at))),
    nameCell(d.common_name, d.scientific_name),
    el(
      "td",
      {},
      el("span", { class: "meter", title: `confidence ${d.confidence.toFixed(3)}` },
        el("span", { class: "track" }, el("span", { class: "fill", style: `width:${Math.round(d.confidence * 100)}%` })),
        fmt.percent(d.confidence)),
    ),
    el("td", {}, recording),
  );
}

function isRecent(d) {
  return Date.now() - new Date(d.detected_at).getTime() < 2 * 60 * 1000;
}

function addDetection(d, live) {
  const tbody = $("#detections tbody");
  const row = detectionRow(d);
  const existing = state.rows.get(d.id);
  if (existing) {
    existing.replaceWith(row);
  } else if (live) {
    row.classList.add("new");
    tbody.prepend(row);
  } else {
    tbody.append(row);
  }
  state.rows.set(d.id, row);
  while (state.rows.size > MAX_ROWS) {
    const oldest = Math.min(...state.rows.keys());
    state.rows.get(oldest).remove();
    state.rows.delete(oldest);
  }
  $("#detections-empty").hidden = state.rows.size > 0;
  if (live && !d.clip_path) {
    // The clip is attached a moment after the detection is stored.
    setTimeout(() => refreshDetection(d.id), 8000);
  }
  syncPlayButtons();
}

async function refreshDetection(id) {
  try {
    addDetection(await api(`detections/${id}`), false);
  } catch (e) {
    console.warn("refresh failed", e);
  }
}

async function loadDetections() {
  try {
    const page = await api("detections/latest?limit=50&kind=animal");
    for (const d of page.items) addDetection(d, false);
    $("#detections-empty").hidden = page.items.length > 0;
  } catch (e) {
    const empty = $("#detections-empty");
    empty.hidden = false;
    empty.textContent = `Could not load detections: ${e.message}`;
    empty.classList.add("error");
  }
}

function setLive(value) {
  const live = $("#live");
  live.dataset.state = value;
  live.textContent = value === "live" ? "live" : value === "offline" ? "reconnecting" : "connecting";
}

function connectStream() {
  if (!("EventSource" in window)) {
    setLive("offline");
    return;
  }
  // The browser reconnects by itself and sends Last-Event-ID, so nothing is missed.
  const stream = new EventSource(`${API}/stream`);
  stream.addEventListener("open", () => setLive("live"));
  stream.addEventListener("error", () => setLive("offline"));
  stream.addEventListener("detection", (event) => {
    try {
      const d = JSON.parse(event.data);
      // Sound events are listed in their own card, refreshed below.
      if (d.kind !== "sound_event") addDetection(d, true);
      scheduleRefresh();
    } catch (e) {
      console.warn("bad event", e);
    }
  });
}

function scheduleRefresh() {
  clearTimeout(state.refreshTimer);
  state.refreshTimer = setTimeout(() => {
    loadLatest();
    if (state.date === todayInStation()) loadDaily();
    loadSpecies();
    loadSoundEvents();
  }, 3000);
}

// ---------- species and sound events ----------

function nameCell(common, scientific) {
  const cell = el("td", {}, common);
  // Sound events have no separate scientific name ("Car_passing_by" is shown as "Car passing by").
  if (scientific.replaceAll("_", " ") !== common) cell.append(el("span", { class: "sci" }, scientific));
  return cell;
}

function summaryRow(s) {
  return el(
    "tr",
    {},
    nameCell(s.common_name, s.scientific_name),
    el("td", { class: "num" }, fmt.number(s.count)),
    el("td", {}, fmt.dateTime(s.last_seen), el("span", { class: "when", "data-iso": s.last_seen }, fmt.relative(s.last_seen))),
    el("td", { class: "num" }, fmt.percent(s.max_confidence)),
    el("td", {}, s.best_clip_detection_id
      ? el("div", { class: "rec" }, playButton(s.best_clip_detection_id, true, `best ${s.common_name} recording`))
      : el("span", { class: "pending" }, "none kept")),
  );
}

// `kind` is animal or sound_event; `id` names the table, its count and its empty message.
async function loadSummary(kind, id, noun) {
  const tbody = $(`#${id} tbody`);
  try {
    const data = await api(`species?kind=${kind}`);
    tbody.replaceChildren(...data.items.map(summaryRow));
    $(`#${id}-empty`).hidden = data.items.length > 0;
    $(`#${id}-count`).textContent = data.items.length ? `${data.items.length} total` : "";
    syncPlayButtons();
  } catch (e) {
    const empty = $(`#${id}-empty`);
    empty.hidden = false;
    empty.textContent = `Could not load ${noun}: ${e.message}`;
  }
}

function loadSpecies() {
  return loadSummary("animal", "species", "species");
}

function loadSoundEvents() {
  return loadSummary("sound_event", "sound-events", "sound events");
}

function updateRelativeTimes() {
  for (const node of document.querySelectorAll(".when[data-iso]")) node.textContent = fmt.relative(node.dataset.iso);
}

// ---------- start ----------

async function init() {
  try {
    const cfg = await api("config");
    state.tz = cfg.station.timezone;
    $("#station").textContent = cfg.station.name;
    document.title = `${cfg.station.name} · Birdsong`;
  } catch (e) {
    console.warn("config unavailable", e);
  }

  state.date = todayInStation();
  const dateInput = $("#daily-date");
  dateInput.value = state.date;
  dateInput.max = state.date;
  dateInput.addEventListener("change", () => {
    if (!dateInput.value) return;
    state.date = dateInput.value;
    loadDaily();
  });

  for (const button of document.querySelectorAll("#window-buttons button")) {
    button.addEventListener("click", () => {
      state.window = button.dataset.window;
      for (const b of document.querySelectorAll("#window-buttons button")) b.setAttribute("aria-pressed", String(b === button));
      loadLatest();
    });
  }

  const player = $("#player");
  player.addEventListener("ended", () => {
    state.playing = null;
    syncPlayButtons();
  });
  $("#viewer").addEventListener("close", () => $("#viewer-audio").pause());

  await Promise.all([loadHealth(), loadLatest(), loadDaily(), loadDetections(), loadSpecies(), loadSoundEvents()]);
  connectStream();

  setInterval(loadHealth, 15000);
  setInterval(() => {
    updateRelativeTimes();
    loadLatest();
    const today = todayInStation();
    if (dateInput.max !== today) {
      // Midnight in the station time zone: follow the new day if the old "today" was selected.
      const followToday = state.date === dateInput.max;
      dateInput.max = today;
      if (followToday) {
        state.date = today;
        dateInput.value = today;
      }
    }
    if (state.date === today) loadDaily();
  }, 60000);
}

document.addEventListener("DOMContentLoaded", init);
