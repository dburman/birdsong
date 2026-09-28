// Birdsong dashboard. Plain JavaScript, no build step, no third-party code.
"use strict";

const API = "api/v1"; // relative, so the page also works behind a reverse-proxy path prefix
const MAX_ROWS = 100;
const TOP_BARS = 12;

const state = {
  tz: undefined,
  window: "24h",
  date: null,
  rows: new Map(), // detection id -> <tr>
  playing: null, // detection id
  refreshTimer: null,
  latestData: null, // last stats/recent response, for expanding the chart without refetching
  latestExpanded: false,
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

// A species name that opens where to read about it (All About Birds, iNaturalist), in a new tab.
function speciesLink(common, url) {
  if (!url) return common;
  const site = new URL(url).hostname.replace(/^www\./, "");
  return el("a", { class: "species-link", href: url, target: "_blank", rel: "noopener noreferrer", title: `About ${common} on ${site}` }, common);
}

// ---------- latest birds (horizontal bars) ----------

async function loadLatest() {
  const container = $("#latest-chart");
  try {
    state.latestData = await api(`stats/recent?window=${encodeURIComponent(state.window)}`);
    renderLatest(container, state.latestData);
  } catch (e) {
    showMessage(container, `Could not load: ${e.message}`, true);
  }
}

function renderLatest(container, data) {
  const species = state.latestExpanded ? data.species : data.species.slice(0, TOP_BARS);
  if (species.length === 0) {
    showMessage(container, `No detections in the last ${data.window}.`);
    return;
  }
  const max = Math.max(...species.map((s) => s.count));
  const grid = el("div", { class: "bars", role: "list" });
  for (const s of species) {
    const label = `${s.common_name}: ${s.count} detection${s.count === 1 ? "" : "s"}, last ${fmt.dateTime(s.last_seen)}`;
    grid.append(
      el("span", { class: "name", title: s.scientific_name, role: "listitem", "aria-label": label }, speciesLink(s.common_name, s.info_url)),
      el("span", { class: "track", "aria-hidden": "true" }, el("span", { class: "fill", style: `width:${(100 * s.count) / max}%` })),
      el("span", { class: "count", "aria-hidden": "true" }, fmt.number(s.count)),
    );
  }
  const extra = data.species.length - species.length;
  container.replaceChildren(grid);
  const toggle = (expanded) => {
    state.latestExpanded = expanded;
    renderLatest(container, data);
    container.querySelector(".more")?.focus();
  };
  if (extra > 0) {
    container.append(
      el("button", { type: "button", class: "more", onclick: () => toggle(true) },
        `Show all ${data.species.length} species`),
    );
  } else if (state.latestExpanded && data.species.length > TOP_BARS) {
    container.append(
      el("button", { type: "button", class: "more", onclick: () => toggle(false) },
        `Show only the top ${TOP_BARS}`),
    );
  }
}

// ---------- by hour (stacked columns) ----------

async function loadDaily() {
  const container = $("#daily-chart");
  try {
    const data = await api(`stats/daily?date=${encodeURIComponent(state.date)}`);
    renderDaily(container, data);
  } catch (e) {
    showMessage(container, `Could not load: ${e.message}`, true);
  }
}

// One row per species, busiest first: a 24-hour strip and the day's total. Each row is scaled to
// that species' busiest hour, so every species' daily pattern is readable however common it is;
// the total and the hover text carry the counts.
function renderDaily(container, data) {
  if (data.species.length === 0) {
    showMessage(container, `No detections on ${data.date}.`);
    return;
  }
  const hour = (h) => `${String(h).padStart(2, "0")}:00`;
  const grid = el("div", { class: "multiples", role: "list", "aria-label": `Detections per hour on ${data.date}` });
  for (const s of data.species) {
    const peak = Math.max(1, ...s.by_hour);
    const busiest = s.by_hour.indexOf(Math.max(...s.by_hour));
    const strip = el("span", { class: "strip", "aria-hidden": "true" });
    s.by_hour.forEach((v, h) => {
      const bar = el("span", { style: v ? `height:max(2px, ${(100 * v) / peak}%)` : "height:0" });
      bar.title = `${hour(h)} ${s.common_name}: ${v}`;
      strip.append(bar);
    });
    grid.append(
      el("span", {
        class: "name",
        title: s.scientific_name,
        role: "listitem",
        "aria-label": `${s.common_name}: ${s.total} detection${s.total === 1 ? "" : "s"}, most at ${hour(busiest)}`,
      }, speciesLink(s.common_name, s.info_url)),
      strip,
      el("span", { class: "count", "aria-hidden": "true" }, fmt.number(s.total)),
    );
  }
  const axis = el("span", { class: "strip hours", "aria-hidden": "true" });
  for (let h = 0; h < 24; h++) axis.append(el("span", {}, h % 6 === 0 ? String(h).padStart(2, "0") : ""));
  grid.append(el("span", {}), axis, el("span", {}));
  container.replaceChildren(
    grid,
    el("p", { class: "muted caption" }, "Each row is scaled to that species' busiest hour; the number is the day's total."),
  );
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
  $("#viewer").dataset.id = d.id;
  showViewerReview(d);
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

// ---------- reviews ----------

// ✓ / ✗ buttons recording whether a detection was right. Pressing the active one again clears it.
function reviewButtons(d, onChange) {
  const button = (verdict, symbol, text) => el("button", {
    type: "button",
    class: `review review-${verdict}`,
    "aria-pressed": String(d.review === verdict),
    "aria-label": `${text}: ${d.common_name}`,
    title: text,
    onclick: async (event) => {
      const next = d.review === verdict ? null : verdict;
      event.currentTarget.disabled = true;
      try {
        const res = await fetch(`${API}/detections/${d.id}/review`, {
          method: "PUT",
          headers: { "Content-Type": "application/json", Accept: "application/json" },
          body: JSON.stringify({ verdict: next }),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        onChange(await res.json());
      } catch (e) {
        console.warn("review failed", e);
        event.currentTarget.disabled = false;
      }
    },
  }, symbol);
  return el("span", { class: "reviews" }, button("correct", "✓", "Correct"), button("wrong", "✗", "Wrong"));
}

function reviewed(d) {
  addDetection(d, false);
  if ($("#viewer").open && Number($("#viewer").dataset.id) === d.id) showViewerReview(d);
  loadSpecies();
}

function showViewerReview(d) {
  $("#viewer-review").replaceChildren(el("span", { class: "muted" }, "Was this right? "), reviewButtons(d, reviewed));
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
    el("td", {}, reviewButtons(d, reviewed)),
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

function nameCell(common, scientific, url) {
  const cell = el("td", {}, speciesLink(common, url));
  // Sound events have no separate scientific name ("Car_passing_by" is shown as "Car passing by").
  if (scientific.replaceAll("_", " ") !== common) cell.append(el("span", { class: "sci" }, scientific));
  return cell;
}

function reviewCell(r) {
  if (!r) return el("td", { class: "num muted" }, "–");
  const hint = r.suggested_min_confidence === null
    ? `${r.correct} of ${r.reviewed} reviewed detections were correct.`
    : `${r.correct} of ${r.reviewed} reviewed detections were correct. From ${fmt.percent(r.suggested_min_confidence)} confidence they are at least 90 % right: detection.species_min_confidence = ${r.suggested_min_confidence.toFixed(2)}`;
  const cell = el("td", { class: "num", title: hint }, `${r.correct} ✓ ${r.wrong} ✗`);
  if (r.suggested_min_confidence !== null) cell.append(el("span", { class: "sci" }, `suggest ${r.suggested_min_confidence.toFixed(2)}`));
  return cell;
}

function summaryRow(s, reviews) {
  return el(
    "tr",
    {},
    nameCell(s.common_name, s.scientific_name, s.info_url),
    el("td", { class: "num" }, fmt.number(s.count)),
    el("td", {}, fmt.dateTime(s.last_seen), el("span", { class: "when", "data-iso": s.last_seen }, fmt.relative(s.last_seen))),
    el("td", { class: "num" }, fmt.percent(s.max_confidence)),
    el("td", {}, s.best_clip_detection_id
      ? el("div", { class: "rec" }, playButton(s.best_clip_detection_id, true, `best ${s.common_name} recording`))
      : el("span", { class: "pending" }, "none kept")),
    ...(reviews ? [reviewCell(reviews.get(s.scientific_name))] : []),
  );
}

// `kind` is animal or sound_event; `id` names the table, its count and its empty message.
async function loadSummary(kind, id, noun) {
  const tbody = $(`#${id} tbody`);
  try {
    const [data, reviews] = await Promise.all([
      api(`species?kind=${kind}`),
      kind === "animal" ? api("reviews").catch(() => ({ items: [] })) : null,
    ]);
    const byName = reviews && new Map(reviews.items.map((r) => [r.scientific_name, r]));
    tbody.replaceChildren(...data.items.map((s) => summaryRow(s, byName)));
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
    const name = cfg.station.name.trim();
    document.title = !name || name.toLowerCase() === "birdsong" ? "Birdsong" : `${name} · Birdsong`;
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
