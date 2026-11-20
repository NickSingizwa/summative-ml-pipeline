// PumpSense dashboard, plain JS + Chart.js. Everything comes from the FastAPI endpoints.
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const bold = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const pct = (v, d = 1) => (v == null ? "–" : `${(v * 100).toFixed(d)}%`);
const num = (v, d = 3) => (v == null ? "–" : Number(v).toFixed(d));
const int = (v) => (v == null ? "–" : Number(v).toLocaleString());

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  let body = null;
  try { body = await res.json(); } catch { /* non-JSON */ }
  if (!res.ok) throw new Error((body && (body.detail || body.message)) || `${res.status} ${res.statusText}`);
  return body;
}

function toast(msg, ms = 4000) {
  const t = $("#toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (t.hidden = true), ms);
}

function fmtDuration(sec) {
  sec = Math.floor(sec);
  const d = Math.floor(sec / 86400), h = Math.floor((sec % 86400) / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  if (d) return `${d}d ${h}h ${m}m`;
  if (h) return `${h}h ${m}m`;
  if (m) return `${m}m ${s}s`;
  return `${s}s`;
}
const fmtTime = (iso) => (iso ? new Date(iso).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "–");

// ---- charts ----
const charts = {};
const chartBuilders = {};   // kept so charts can be redrawn when the theme changes

function baseOptions(extra = {}) {
  const muted = css("--muted"), grid = css("--grid"), axis = css("--axis");
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 250 },
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: css("--surface-raised"), titleColor: css("--ink"), bodyColor: css("--ink-2"),
        borderColor: css("--axis"), borderWidth: 1, padding: 10, boxPadding: 4, usePointStyle: true,
      },
    },
    scales: {
      x: { grid: { display: false }, border: { color: axis }, ticks: { color: muted, maxRotation: 0, autoSkipPadding: 12 } },
      y: { grid: { color: grid }, border: { display: false }, ticks: { color: muted } },
    },
    ...extra,
  };
}

function drawChart(name, canvasId, builder) {
  chartBuilders[name] = { canvasId, builder };
  if (charts[name]) charts[name].destroy();
  const canvas = document.getElementById(canvasId);
  if (!canvas || typeof Chart === "undefined") return;
  Chart.defaults.font.family = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';
  Chart.defaults.font.size = 12;
  charts[name] = new Chart(canvas, builder());
}

function redrawAll() {
  for (const [name, { canvasId, builder }] of Object.entries(chartBuilders)) drawChart(name, canvasId, builder);
}
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { redrawAll(); renderSpectrogram(); });

const lineDataset = (label, data, color, fill = false) => ({
  label, data, borderColor: color, backgroundColor: fill ? color + "1a" : color, fill,
  borderWidth: 2, pointRadius: 0, pointHoverRadius: 5, pointHoverBorderWidth: 2,
  pointHoverBorderColor: css("--surface"), pointBackgroundColor: color, tension: 0.25,
  borderCapStyle: "round", borderJoinStyle: "round",
});
const barDataset = (label, data, color) => ({
  label, data, backgroundColor: color, borderRadius: { topLeft: 4, topRight: 4 }, borderSkipped: "start",
  maxBarThickness: 24, borderWidth: 0, categoryPercentage: 0.7, barPercentage: 0.9,
});

function legend(el, items) {
  el.innerHTML = items.map(([label, color, kind]) =>
    `<span><i class="${kind === "line" ? "line" : ""}" style="background:${color}"></i>${esc(label)}</span>`).join("");
}
function classLegends() {
  $$('[data-legend="classes"]').forEach((el) => legend(el, [["Normal", css("--normal")], ["Abnormal", css("--abnormal")]]));
}

// ---- tabs ----
function showTab(name) {
  $$(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
  $$(".tab-panel").forEach((p) => (p.hidden = p.id !== `tab-${name}`));
  history.replaceState(null, "", `#${name}`);
  if (name === "insights") loadInsights();
  if (name === "retrain") loadRetrain();
  setTimeout(() => Object.values(charts).forEach((c) => c.resize()), 0);
}
$$(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));

// ---- health / uptime ----
const beats = [];
const healthLatency = [];
let lastHealth = null;

// pings /health every 5 s, this is what the uptime bar and chart show
async function pollHealth() {
  const t = performance.now();
  let ok = false;
  try {
    lastHealth = await api("/health", { cache: "no-store" });
    ok = lastHealth.status === "ok";
  } catch { ok = false; }
  const ms = performance.now() - t;
  beats.push(ok); if (beats.length > 60) beats.shift();
  healthLatency.push({ t: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }), ms: ok ? +ms.toFixed(1) : null });
  if (healthLatency.length > 60) healthLatency.shift();

  const pill = $("#status-pill");
  pill.classList.toggle("up", ok);
  pill.classList.toggle("down", !ok);
  $("#status-text").textContent = ok
    ? `Online · up ${fmtDuration(lastHealth.uptime_seconds)} · model ${lastHealth.model_version}`
    : "API unreachable";
  const okCount = beats.filter(Boolean).length;
  $("#uptime-chip").textContent = `${((okCount / beats.length) * 100).toFixed(1)}% of ${beats.length} checks`;
  $("#beats").innerHTML = Array.from({ length: 60 }, (_, i) => {
    const b = beats[i - (60 - beats.length)];
    return `<span class="${b === undefined ? "" : b ? "ok" : "fail"}" title="${b === undefined ? "" : b ? "up" : "down"}"></span>`;
  }).join("");
  drawChart("health", "chart-health", () => ({
    type: "line",
    data: { labels: healthLatency.map((d) => d.t), datasets: [lineDataset("Response time (ms)", healthLatency.map((d) => d.ms), css("--accent"), true)] },
    options: baseOptions({
      animation: false,
      scales: {
        x: { display: false },
        y: { beginAtZero: true, grid: { color: css("--grid") }, border: { display: false }, ticks: { color: css("--muted"), maxTicksLimit: 4, callback: (v) => `${v} ms` } },
      },
    }),
  }));
  if (ok && monitorState.version && lastHealth.model_version !== monitorState.version) loadMonitor();
}

// ---- monitor ----
const monitorState = { version: null };

function tiles(el, items) {
  el.innerHTML = items.map(([label, value, hint]) =>
    `<div class="tile"><div class="label">${esc(label)}</div><div class="value">${esc(value)}</div>${hint ? `<div class="hint">${esc(hint)}</div>` : ""}</div>`).join("");
}

function metricBars(el, m) {
  const rows = [["Accuracy", m.accuracy], ["Precision", m.precision], ["Recall", m.recall], ["F1 score", m.f1],
    ["Specificity", m.specificity], ["ROC-AUC", m.roc_auc], ["PR-AUC", m.pr_auc]];
  el.innerHTML = rows.map(([name, v]) =>
    `<div class="mbar"><span class="name">${name}</span><div class="track"><div style="width:${(v ?? 0) * 100}%"></div></div><span class="val">${num(v)}</span></div>`).join("")
    + `<div class="mbar"><span class="name">Log loss</span><span class="sub">lower is better</span><span class="val">${num(m.loss)}</span></div>`;
}

function confusion(el, cm) {
  const [[tn, fp], [fn, tp]] = cm;
  const max = Math.max(tn, fp, fn, tp) || 1;
  const cell = (v, label) => {
    const a = 0.1 + 0.85 * (v / max);
    const dark = a > 0.55;
    return `<div class="cell" style="background:color-mix(in srgb, var(--accent) ${Math.round(a * 100)}%, transparent);color:${dark ? "#fff" : "var(--ink)"}"><strong>${int(v)}</strong><span>${label}</span></div>`;
  };
  el.innerHTML = `<div class="cm">
      <div></div><div class="axis">Predicted normal</div><div class="axis">Predicted abnormal</div>
      <div class="axis">Actual normal</div>${cell(tn, "true negative")}${cell(fp, "false alarm")}
      <div class="axis">Actual abnormal</div>${cell(fn, "missed fault")}${cell(tp, "fault caught")}
    </div>
    <p class="cm-caption">${int(tp)} of ${int(tp + fn)} faulty clips caught; ${int(fp)} false alarms out of ${int(tn + fp)} healthy clips.</p>`;
}

async function loadMonitor() {
  try {
    const [model, stats] = await Promise.all([api("/api/model"), api("/api/stats")]);
    const cur = model.current_entry;
    monitorState.version = model.current;
    const m = cur.test_metrics;
    const lat = stats.process.predict_latency_ms;
    tiles($("#monitor-tiles"), [
      ["Uptime", fmtDuration(stats.uptime_seconds), `since ${fmtTime(stats.started_at)}`],
      ["Model in production", cur.version, `${cur.trigger} · ${fmtTime(cur.created_at)}`],
      ["Test F1 (abnormal)", num(m.f1), `threshold ${num(cur.threshold, 2)}`],
      ["Test ROC-AUC", num(m.roc_auc), `${int(m.n)} held-out clips`],
      ["Predictions logged", int(stats.predictions.total), `${int(stats.process.requests)} requests to ${stats.host}`],
      ["Prediction latency", lat.p50 == null ? "–" : `${lat.p50} ms`, lat.p95 == null ? "median, this container" : `p95 ${lat.p95} ms`],
    ]);
    $("#model-sub").textContent = `Version ${cur.version} · held-out test set (${int(m.n)} clips)`;
    metricBars($("#model-metrics"), m);
    confusion($("#confusion"), m.confusion_matrix);

    const h = cur.history || {};
    if (h.loss) {
      $("#history-sub").textContent = `${cur.version}: ${h.loss.length} epochs (early stopping keeps the best)`;
      legend($("#history-legend"), [["Training loss", css("--muted"), "line"], ["Validation loss", css("--accent"), "line"]]);
      drawChart("history", "chart-history", () => ({
        type: "line",
        data: { labels: h.loss.map((_, i) => i + 1), datasets: [lineDataset("Training loss", h.loss, css("--muted")), lineDataset("Validation loss", h.val_loss, css("--accent"))] },
        options: baseOptions({ scales: { ...baseOptions().scales, x: { ...baseOptions().scales.x, title: { display: true, text: "Epoch", color: css("--muted") } } } }),
      }));
    }

    $("#versions-table").innerHTML = `<thead><tr><th>Version</th><th>Created</th><th>Trigger</th><th>F1</th><th>ROC-AUC</th><th></th></tr></thead><tbody>${
      model.versions.map((v) => `<tr>
        <td>${v.version === model.current ? `<span class="badge current">${esc(v.version)} · live</span>` : esc(v.version)}</td>
        <td>${fmtTime(v.created_at)}</td><td>${esc(v.trigger)}</td>
        <td>${num(v.test_metrics.f1)}</td><td>${num(v.test_metrics.roc_auc)}</td>
        <td>${v.version === model.current ? "" : `<button class="btn small" data-activate="${esc(v.version)}">Activate</button>`}</td></tr>`).join("")}</tbody>`;
    $$("[data-activate]").forEach((b) => b.addEventListener("click", async () => {
      try { await api(`/api/model/${b.dataset.activate}/activate`, { method: "POST" }); toast(`Now serving ${b.dataset.activate}`); loadMonitor(); }
      catch (e) { toast(e.message); }
    }));

    const preds = stats.predictions;
    $("#pred-sub").textContent = `${int(preds.total)} in total · ${int(preds.by_label.normal || 0)} normal, ${int(preds.by_label.abnormal || 0)} abnormal`;
    $("#predictions-table").innerHTML = preds.recent.length
      ? `<thead><tr><th>Time</th><th>File</th><th>Prediction</th><th>P(fault)</th><th>Latency</th></tr></thead><tbody>${
        preds.recent.map((p) => `<tr><td>${fmtTime(p.created_at)}</td><td class="wrap">${esc(p.filename)}</td>
          <td><span class="badge"><i style="background:var(--${p.label})"></i>${esc(p.label)}</span></td>
          <td>${num(p.probability, 2)}</td><td>${num(p.latency_ms, 0)} ms</td></tr>`).join("")}</tbody>`
      : `<tbody><tr><td class="sub">No predictions yet: try the Predict tab.</td></tr></tbody>`;
  } catch (e) {
    toast(`Could not load monitor: ${e.message}`);
  }
}

$("#btn-evaluate").addEventListener("click", async (ev) => {
  const btn = ev.currentTarget;
  btn.disabled = true; btn.textContent = "Evaluating…";
  try {
    const r = await api("/api/model/evaluate", { method: "POST" });
    metricBars($("#model-metrics"), r.metrics);
    confusion($("#confusion"), r.metrics.confusion_matrix);
    $("#eval-note").textContent = `Re-evaluated ${r.model_version} on ${int(r.metrics.n)} test clips in ${r.seconds} s (${fmtTime(r.evaluated_at)}).`;
  } catch (e) { toast(e.message); }
  btn.disabled = false; btn.textContent = "Re-evaluate on test set";
});

// ---- predict ----
let predictBlob = null, predictName = null, lastSpectrogram = null;

function setPredictFile(blob, name) {
  predictBlob = blob; predictName = name;
  $("#predict-file-name").textContent = name;
  const player = $("#player");
  player.src = URL.createObjectURL(blob);
  player.hidden = false;
  $("#btn-predict").disabled = false;
}

$("#predict-file").addEventListener("change", (e) => {
  const f = e.target.files[0];
  if (f) { $$(".sample").forEach((s) => s.setAttribute("aria-pressed", "false")); setPredictFile(f, f.name); }
});
setupDrop($("#predict-drop"), (files) => { if (files[0]) setPredictFile(files[0], files[0].name); });

async function loadSamples() {
  try {
    const samples = await api("/api/samples");
    $("#samples").innerHTML = samples.map((s) =>
      `<button class="sample" aria-pressed="false" data-name="${esc(s.name)}"><i style="background:var(--${s.label})"></i>${esc(s.name.replace(".wav", ""))}</button>`).join("");
    $$(".sample").forEach((b) => b.addEventListener("click", async () => {
      $$(".sample").forEach((s) => s.setAttribute("aria-pressed", String(s === b)));
      const blob = await (await fetch(`/api/samples/${b.dataset.name}`)).blob();
      setPredictFile(blob, b.dataset.name);
    }));
  } catch { $("#samples").innerHTML = `<span class="sub">No sample clips available.</span>`; }
}

// demo clips have the real label in the file name, so we can show if the prediction is right
function trueLabelFromName(name) {
  const m = /^(abnormal|anomaly|faulty|normal)[_\-. ]/i.exec(name || "");
  if (!m) return null;
  return m[1].toLowerCase() === "normal" ? "normal" : "abnormal";
}

const ICON_OK = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="10"/><path d="m8 12 3 3 5-6"/></svg>`;
const ICON_WARN = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/><path d="M12 9v4M12 17h.01"/></svg>`;

$("#btn-predict").addEventListener("click", async (ev) => {
  if (!predictBlob) return;
  const btn = ev.currentTarget;
  btn.disabled = true; btn.textContent = "Analysing…";
  const fd = new FormData();
  fd.append("file", predictBlob, predictName);
  try {
    const r = await api("/api/predict?spectrogram=true", { method: "POST", body: fd });
    const bad = r.label === "abnormal";
    const truth = trueLabelFromName(r.filename);
    $("#result-empty").hidden = true;
    $("#result").hidden = false;
    $("#verdict").className = `verdict ${bad ? "bad" : "good"}`;
    $("#verdict").innerHTML = `${bad ? ICON_WARN : ICON_OK}<div><div class="title">${bad ? "Abnormal: likely fault" : "Normal: pump sounds healthy"}</div>
      <div class="detail">${bad ? "Schedule an inspection for this pump." : "No maintenance action needed."} Confidence ${pct(r.confidence)}.</div>
      ${truth ? `<div class="check">True label (from file name): <strong>${truth}</strong> · ${truth === r.label ? '<strong class="ok">✓ correct</strong>' : '<strong class="miss">✗ incorrect</strong>'}</div>` : ""}</div>`;
    $("#prob-value").textContent = num(r.probability_abnormal);
    const meter = $("#prob-meter");
    meter.classList.toggle("bad", bad);
    $(".fill", meter).style.width = `${r.probability_abnormal * 100}%`;
    $(".threshold", meter).style.left = `calc(${r.threshold * 100}% - 1px)`;
    $("#threshold-label").textContent = `threshold ${num(r.threshold, 2)}`;
    $("#result-facts").innerHTML = [
      ["File", r.filename], ["Duration", `${num(r.duration_s, 1)} s`], ["10 s windows", r.n_clips],
      ["Model", r.model_version], ["Latency", `${num(r.latency_ms, 0)} ms`], ["Served by", r.served_by],
    ].map(([k, v]) => `<div><dt>${k}</dt><dd>${esc(v)}</dd></div>`).join("");
    lastSpectrogram = r.spectrogram;
    renderSpectrogram();
    loadMonitor();
  } catch (e) { toast(`Prediction failed: ${e.message}`); }
  btn.disabled = false; btn.textContent = "Predict";
});

function hexToRgb(h) { const n = parseInt(h.replace("#", ""), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
// draws the spectrogram the API sends back (uint8 grid) using the blue colour ramp
function renderSpectrogram() {
  if (!lastSpectrogram) return;
  const canvas = $("#spectrogram");
  const rows = lastSpectrogram.length, cols = lastSpectrogram[0].length;
  canvas.width = cols; canvas.height = rows;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(cols, rows);
  const ramp = [0, 1, 2, 3, 4, 5].map((i) => hexToRgb(css(`--spec-${i}`)));
  for (let y = 0; y < rows; y++) for (let x = 0; x < cols; x++) {
    const v = (lastSpectrogram[y][x] / 255) * (ramp.length - 1);
    const i = Math.min(Math.floor(v), ramp.length - 2), f = v - i;
    const o = (y * cols + x) * 4;
    for (let c = 0; c < 3; c++) img.data[o + c] = ramp[i][c] + (ramp[i + 1][c] - ramp[i][c]) * f;
    img.data[o + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

// ---- insights ----
let insightsLoaded = false;
async function loadInsights() {
  if (insightsLoaded) return;
  let d;
  try { d = await api("/api/insights"); } catch (e) { toast(e.message); return; }
  insightsLoaded = true;
  const ds = d.dataset;
  tiles($("#insight-tiles"), [
    ["Recordings", int(ds.n_clips), `${ds.clip_seconds} s each at ${ds.sample_rate / 1000} kHz`],
    ["Pumps", ds.n_pumps, "industrial water pumps"],
    ["Normal clips", int(ds.class_counts.normal), pct(ds.class_counts.normal / ds.n_clips)],
    ["Abnormal clips", int(ds.class_counts.abnormal), pct(ds.class_counts.abnormal / ds.n_clips)],
    ["Training / test", `${int(ds.split_counts.train.total)} / ${int(ds.split_counts.test.total)}`, `${int(ds.split_counts.new_batch.total)} held back for retraining`],
  ]);
  $("#story").innerHTML = `<h2>What the data says</h2><ol>${d.story.map((s) => `<li>${bold(s)}</li>`).join("")}</ol>`;
  classLegends();

  drawChart("pumps", "chart-pumps", () => ({
    type: "bar",
    data: { labels: ds.pump_counts.map((p) => p.pump.replace("_", " ")), datasets: [
      barDataset("Normal", ds.pump_counts.map((p) => p.normal), css("--normal")),
      barDataset("Abnormal", ds.pump_counts.map((p) => p.abnormal), css("--abnormal"))] },
    options: baseOptions(),
  }));

  const sp = d.spectrum;
  drawChart("spectrum", "chart-spectrum", () => ({
    type: "line",
    data: { labels: sp.mel_hz.map((f) => (f >= 1000 ? `${(f / 1000).toFixed(1)}k` : Math.round(f))), datasets: [
      lineDataset("Normal", sp.normal_db, css("--normal")), lineDataset("Abnormal", sp.abnormal_db, css("--abnormal"))] },
    options: baseOptions({ scales: { ...baseOptions().scales,
      x: { ...baseOptions().scales.x, title: { display: true, text: "Frequency (Hz, mel-spaced)", color: css("--muted") }, ticks: { color: css("--muted"), maxTicksLimit: 8, maxRotation: 0 } },
      y: { ...baseOptions().scales.y, title: { display: true, text: "dB", color: css("--muted") } } } }),
  }));
  $("#spectrum-interp").innerHTML = bold(sp.interpretation);

  $("#feature-cards").innerHTML = d.features.map((f, i) => `<article class="card">
      <header class="card-head"><div><h2>${esc(f.title)}</h2><p class="sub">Share of clips per bin · median ${esc(f.fmt_normal)} vs ${esc(f.fmt_abnormal)}</p></div></header>
      <div class="legend" data-legend="classes"></div>
      <div class="chart-box"><canvas id="chart-feature-${i}"></canvas></div>
      <p class="interp">${bold(f.interpretation)}</p></article>`).join("");
  classLegends();
  d.features.forEach((f, i) => drawChart(`feature-${i}`, `chart-feature-${i}`, () => ({
    type: "line",
    data: { labels: f.bin_labels, datasets: [lineDataset("Normal", f.normal, css("--normal"), true), lineDataset("Abnormal", f.abnormal, css("--abnormal"), true)] },
    options: baseOptions({
      plugins: { ...baseOptions().plugins, tooltip: { ...baseOptions().plugins.tooltip, callbacks: { label: (c) => ` ${c.dataset.label}: ${c.parsed.y.toFixed(1)}% of clips` } } },
      scales: { ...baseOptions().scales,
        x: { ...baseOptions().scales.x, title: { display: true, text: f.unit, color: css("--muted") }, ticks: { color: css("--muted"), maxTicksLimit: 6, maxRotation: 0 } },
        y: { ...baseOptions().scales.y, beginAtZero: true, ticks: { color: css("--muted"), callback: (v) => `${v}%` } } },
    }),
  })));

  drawChart("ranking", "chart-ranking", () => ({
    type: "bar",
    data: { labels: d.ranking.map((r) => r.title), datasets: [{ ...barDataset("ROC-AUC", d.ranking.map((r) => r.auc), css("--accent")), borderRadius: { topRight: 4, bottomRight: 4 } }] },
    options: baseOptions({
      indexAxis: "y",
      interaction: { mode: "nearest", axis: "y", intersect: false },
      scales: {
        x: { min: 0, max: 1, grid: { color: css("--grid") }, border: { display: false }, ticks: { color: css("--muted") } },
        y: { grid: { display: false }, border: { color: css("--axis") }, ticks: { color: css("--ink-2") } },
      },
    }),
  }));
}

// ---- retrain ----
let uploadFiles = [];
$("#upload-files").addEventListener("change", (e) => setUploadFiles([...e.target.files]));
setupDrop($("#upload-drop"), (files) => setUploadFiles(files));

function setUploadFiles(files) {
  uploadFiles = files;
  const mb = files.reduce((s, f) => s + f.size, 0) / 1e6;
  $("#upload-file-names").textContent = files.length ? `${files.length} file(s), ${mb.toFixed(1)} MB` : "No files selected";
  $("#btn-upload").disabled = !files.length;
}

function setupDrop(zone, onFiles) {
  ["dragenter", "dragover"].forEach((ev) => zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.remove("over"); }));
  zone.addEventListener("drop", (e) => onFiles([...e.dataTransfer.files]));
}

$("#btn-upload").addEventListener("click", () => {
  if (!uploadFiles.length) return;
  const fd = new FormData();
  uploadFiles.forEach((f) => fd.append("files", f, f.webkitRelativePath || f.name));
  fd.append("label", $("#upload-label").value);
  const prog = $("#upload-progress"), bar = $(".bar", prog), btn = $("#btn-upload");
  prog.hidden = false; prog.classList.remove("indeterminate"); bar.style.width = "0";
  btn.disabled = true; btn.textContent = "Uploading…";
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/upload");
  // XHR instead of fetch so we get upload progress
  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) bar.style.width = `${(e.loaded / e.total) * 100}%`;
    if (e.loaded === e.total) { prog.classList.add("indeterminate"); btn.textContent = "Preprocessing…"; }
  };
  xhr.onload = () => {
    prog.hidden = true; btn.disabled = false; btn.textContent = "Upload";
    let r; try { r = JSON.parse(xhr.responseText); } catch { r = { detail: xhr.statusText }; }
    if (xhr.status >= 400) { toast(`Upload failed: ${r.detail}`); return; }
    renderUploadResult(r);
    loadRetrain();
    if (r.auto_retrain_run) { toast(`Automatic retraining started (run ${r.auto_retrain_run})`); pollRun(); }
  };
  xhr.onerror = () => { prog.hidden = true; btn.disabled = false; btn.textContent = "Upload"; toast("Upload failed: network error"); };
  xhr.send(fd);
});

function renderUploadResult(r) {
  const m = r.batch_metrics;
  const issues = [...(r.skipped || []), ...(r.rejected || [])];
  $("#upload-result").innerHTML = `<div class="result-box">
    <div><strong>${int(r.stored)}</strong> new recordings saved to the database · ${int(r.processed)} preprocessed into spectrograms
      ${r.duplicates ? ` · ${int(r.duplicates)} duplicates ignored` : ""}${issues.length ? ` · ${issues.length} skipped` : ""}</div>
    ${m ? `<div>Current model on this batch: accuracy <strong>${num(m.accuracy)}</strong>, F1 <strong>${num(m.f1)}</strong>${m.roc_auc != null ? `, ROC-AUC <strong>${num(m.roc_auc)}</strong>` : ""} (${int(m.n)} clips)</div>` : ""}
    <div class="sub">Trigger check: ${esc(r.trigger_check)}</div>
    ${issues.length ? `<ul>${issues.slice(0, 5).map((i) => `<li>${esc(i.file)}: ${esc(i.reason)}</li>`).join("")}</ul>` : ""}
  </div>`;
}

async function loadRetrain() {
  try {
    const [up, runs] = await Promise.all([api("/api/uploads"), api("/api/retrain/history")]);
    const s = up.settings;
    $("#pool").innerHTML = `
      <div class="stat"><div class="k">New, not yet trained on</div><div class="v">${int(up.pending_training)}</div></div>
      <div class="stat"><div class="k">Already used in training</div><div class="v">${int(up.used_in_training)}</div></div>
      <div class="stat"><div class="k">Rejected</div><div class="v">${int(up.rejected)}</div></div>
      <div class="pool-meter"><div class="meter-label"><span>Auto-trigger: ${s.auto_retrain ? "on" : "off"}</span><strong>${int(up.pending_training)} / ${int(s.retrain_threshold)} samples</strong></div>
        <div class="meter"><div class="fill" style="width:${Math.min(100, (up.pending_training / s.retrain_threshold) * 100)}%"></div></div></div>`;
    if (document.activeElement?.closest(".trigger-settings") == null) {
      $("#set-auto").checked = s.auto_retrain;
      $("#set-threshold").value = s.retrain_threshold;
      $("#set-floor").value = s.f1_floor;
    }
    $("#batches-table").innerHTML = up.batches.length
      ? `<thead><tr><th>Uploaded</th><th>Files</th><th>Saved</th><th>Label</th><th>F1 on batch</th></tr></thead><tbody>${
        up.batches.map((b) => `<tr><td>${fmtTime(b.created_at)}</td><td>${int(b.n_files)}</td><td>${int(b.n_accepted)}</td><td>${esc(b.label_mode)}</td>
          <td>${b.eval_metrics ? `${num(b.eval_metrics.f1)} <span class="sub">(${esc(b.eval_version)})</span>` : "–"}</td></tr>`).join("")}</tbody>`
      : `<tbody><tr><td class="sub">No uploads yet.</td></tr></tbody>`;
    renderRuns(runs);
    const running = runs.find((r) => r.status === "running");
    if (running) pollRun();
  } catch (e) { toast(`Could not load retraining data: ${e.message}`); }
}

function renderRuns(runs) {
  $("#runs-table").innerHTML = runs.length
    ? `<thead><tr><th>#</th><th>Started</th><th>Trigger</th><th>Status</th><th>Result</th></tr></thead><tbody>${
      runs.map((r) => `<tr><td>${r.id}</td><td>${fmtTime(r.started_at)}</td><td>${esc(r.trigger)}</td>
        <td>${esc(r.status)}</td><td class="wrap">${esc(r.message || "")}</td></tr>`).join("")}</tbody>`
    : `<tbody><tr><td class="sub">No retraining runs yet.</td></tr></tbody>`;
  const last = runs.find((r) => r.status === "completed");
  if (last && last.metrics_before && last.metrics_after) renderCompare(last);
  if (runs[0]) renderRunStatus(runs[0]);
}

function renderRunStatus(r) {
  const running = r.status === "running";
  $("#btn-retrain").disabled = running;
  $("#btn-retrain").textContent = running ? "Retraining…" : "Retrain now";
  $("#run-status").innerHTML = `<div class="run">
    <div class="top"><span>Run ${r.id} · ${esc(r.trigger)}</span><span>${esc(r.status)}${running ? ` · ${esc(r.stage)}` : ""}</span></div>
    ${running ? `<div class="progress"><div class="bar" style="width:${(r.progress || 0) * 100}%"></div></div>` : ""}
    <div class="msg">${esc(r.message || r.reason || "")}</div></div>`;
}

function renderCompare(r) {
  $("#run-compare-card").hidden = false;
  $("#run-compare-sub").textContent = `Run ${r.id}: ${r.base_version} → ${r.promoted ? r.new_version + " (promoted)" : "candidate (not promoted)"} · held-out test set`;
  const keys = [["Accuracy", "accuracy"], ["Precision", "precision"], ["Recall", "recall"], ["F1", "f1"], ["ROC-AUC", "roc_auc"], ["PR-AUC", "pr_auc"]];
  legend($("#compare-legend"), [[`Before (${r.base_version})`, css("--muted")], [`After (${r.promoted ? r.new_version : "candidate"})`, css("--accent")]]);
  drawChart("compare", "chart-compare", () => ({
    type: "bar",
    data: { labels: keys.map((k) => k[0]), datasets: [
      barDataset("Before", keys.map((k) => r.metrics_before[k[1]]), css("--muted")),
      barDataset("After", keys.map((k) => r.metrics_after[k[1]]), css("--accent"))] },
    options: baseOptions({ scales: { ...baseOptions().scales, y: { ...baseOptions().scales.y, min: 0, max: 1 } } }),
  }));
}

let runTimer = null;
function pollRun() {
  if (runTimer) return;
  runTimer = setInterval(async () => {
    try {
      const r = await api("/api/retrain/status");
      renderRunStatus(r);
      if (r.status !== "running") {
        clearInterval(runTimer); runTimer = null;
        toast(r.status === "completed" ? `Retraining finished: ${r.message}` : `Retraining failed: ${r.message}`, 7000);
        loadRetrain(); loadMonitor();
      }
    } catch { /* keep polling */ }
  }, 2000);
}

$("#btn-retrain").addEventListener("click", async () => {
  try {
    const r = await api("/api/retrain", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ reason: "manual trigger from dashboard" }) });
    toast(`Retraining started (run ${r.run_id})`);
    pollRun(); loadRetrain();
  } catch (e) { toast(e.message, 6000); }
});

$("#btn-settings").addEventListener("click", async () => {
  try {
    await api("/api/settings", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({
      auto_retrain: $("#set-auto").checked, retrain_threshold: Number($("#set-threshold").value), f1_floor: Number($("#set-floor").value) }) });
    toast("Trigger settings saved"); loadRetrain();
  } catch (e) { toast(e.message); }
});

// ---- boot ----
pollHealth();
setInterval(pollHealth, 5000);
loadMonitor();
setInterval(() => { if (!$("#tab-monitor").hidden) loadMonitor(); }, 15000);
loadSamples();
const startTab = (location.hash || "").slice(1);
showTab(["monitor", "predict", "insights", "retrain"].includes(startTab) ? startTab : "monitor");
