/* ---- family -> color-group mapping ----
   ~13 raw model families fold into 8 CVD-safe categorical slots (see
   styles.css). Every chart also direct-labels/tooltips the exact model
   name and family, so color is a secondary "lineage" cue, never the sole
   identity channel. */
const FAMILY_GROUP = {
  linear: "linear", naive_bayes: "linear",
  tree: "tree", ensemble_bagging: "tree",
  boosting: "boosting",
  kernel: "kernel",
  instance: "instance", graph: "instance", probabilistic: "instance",
  interpretable: "interpretable",
  exotic: "exotic",
  meta: "meta", imbalanced: "meta",
};
const GROUP_LABEL = {
  linear: "Linear & Naive Bayes",
  tree: "Trees & Bagging",
  boosting: "Boosting",
  kernel: "Kernel & SVM",
  instance: "Instance / Graph / Probabilistic",
  interpretable: "Interpretable (glass-box)",
  exotic: "Exotic / Experimental",
  meta: "Meta & Imbalance-Aware",
};
const GROUP_ORDER = ["linear", "tree", "boosting", "kernel", "instance", "interpretable", "exotic", "meta"];

function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
function groupOf(family) { return FAMILY_GROUP[family] || "exotic"; }
function colorForFamily(family) { return cssVar(`--grp-${groupOf(family)}`); }

const fmtPct = (v) => (v * 100).toFixed(1) + "%";
const fmtNum = (v, d = 2) => Number(v).toFixed(d);
const fmtTime = (s) => (s < 1 ? (s * 1000).toFixed(0) + " ms" : s.toFixed(2) + " s");
function fmtBigDuration(s) {
  if (s < 1) return (s * 1000).toFixed(0) + " ms";
  if (s < 60) return s.toFixed(1) + " s";
  if (s < 3600) return (s / 60).toFixed(1) + " min";
  if (s < 86400) return (s / 3600).toFixed(1) + " hr";
  const days = s / 86400;
  if (days < 365) return days.toFixed(1) + " days";
  const years = days / 365;
  return years > 100 ? "> 100 years (infeasible)" : years.toFixed(1) + " years";
}
function fmtBigBytes(b) {
  const units = ["B", "KB", "MB", "GB", "TB", "PB"];
  let i = 0;
  while (b >= 1024 && i < units.length - 1) { b /= 1024; i++; }
  return b.toFixed(b < 10 ? 1 : 0) + " " + units[i];
}

/* ---- SVG helpers ---- */
const SVG_NS = "http://www.w3.org/2000/svg";
function svgEl(tag, attrs = {}) {
  const e = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  return e;
}

/* ---- shared tooltip ---- */
const tooltipEl = document.createElement("div");
tooltipEl.className = "tooltip";
document.body.appendChild(tooltipEl);
function showTooltip(evt, html) { tooltipEl.innerHTML = html; tooltipEl.classList.add("show"); moveTooltip(evt); }
function moveTooltip(evt) {
  const pad = 14;
  let x = evt.clientX + pad, y = evt.clientY + pad;
  const rect = tooltipEl.getBoundingClientRect();
  if (x + rect.width > window.innerWidth - 8) x = evt.clientX - rect.width - pad;
  if (y + rect.height > window.innerHeight - 8) y = evt.clientY - rect.height - pad;
  tooltipEl.style.left = x + "px"; tooltipEl.style.top = y + "px";
}
function hideTooltip() { tooltipEl.classList.remove("show"); }

/* ==================================================================
   Horizontal bar panel
   ================================================================== */
function renderBarPanel(container, { title, sub, items, format = fmtPct, domainMax = 1, valueForWidth = null, sort = true }) {
  container.innerHTML = "";
  const head = document.createElement("div");
  head.innerHTML = `<p class="chart-title">${title}</p><p class="chart-sub">${sub || ""}</p>`;
  container.appendChild(head);

  const sorted = sort ? [...items].sort((a, b) => b.value - a.value) : items;
  const rowH = 26, gap = 6, leftLabelW = 190, rightPad = 56, width = 560;
  const plotW = width - leftLabelW - rightPad;
  const height = Math.max(1, sorted.length) * (rowH + gap) - gap;

  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, width: "100%", height, style: "overflow: visible; display:block;" });
  const maxVal = valueForWidth ? Math.max(...sorted.map(valueForWidth)) : domainMax;

  sorted.forEach((item, i) => {
    const y = i * (rowH + gap);
    const barW = Math.max(2, ((valueForWidth ? valueForWidth(item) : item.value) / maxVal) * plotW);
    const g = svgEl("g", { class: "bar-row" });

    const label = svgEl("text", { x: leftLabelW - 10, y: y + rowH / 2 + 4, "text-anchor": "end", "font-size": "12" });
    label.textContent = item.label;
    g.appendChild(label);

    g.appendChild(svgEl("rect", { x: leftLabelW, y, width: plotW, height: rowH, rx: 5, fill: cssVar("--surface-2") }));
    g.appendChild(svgEl("rect", { class: "bar", x: leftLabelW, y, width: barW, height: rowH, rx: 5, fill: item.color }));

    const valLabel = svgEl("text", { class: "value-label", x: leftLabelW + barW + 8, y: y + rowH / 2 + 4, "font-size": "11.5" });
    valLabel.textContent = format(item.value);
    g.appendChild(valLabel);

    const hit = svgEl("rect", { class: "hit-target", x: 0, y, width, height: rowH });
    g.appendChild(hit);
    g.addEventListener("mousemove", (evt) => {
      showTooltip(evt, `<div class="t-title">${item.label}</div><div class="t-row"><span>${title}</span><span>${format(item.value)}</span></div>${item.metaHtml || ""}`);
    });
    g.addEventListener("mouseleave", hideTooltip);
    svg.appendChild(g);
  });

  container.appendChild(svg);
}

/* ==================================================================
   Heatmap
   ================================================================== */
function seqColor(v, min = 0, max = 1) {
  const stops = ["--seq-100", "--seq-250", "--seq-400", "--seq-550", "--seq-700"];
  const t = Math.max(0, Math.min(1, (v - min) / (max - min || 1)));
  const idx = Math.min(stops.length - 1, Math.floor(t * stops.length));
  return cssVar(stops[idx]);
}
function textOnFill(v, min = 0, max = 1) {
  const t = Math.max(0, Math.min(1, (v - min) / (max - min || 1)));
  return t > 0.6 ? "#ffffff" : cssVar("--text");
}
function renderHeatmap(container, { rowLabels, colLabels, matrix, cellFormat = (v) => v.toFixed(2), rowLabelW = 170, cellW = 62, title, sub }) {
  container.innerHTML = "";
  if (title) {
    const head = document.createElement("div");
    head.innerHTML = `<p class="chart-title">${title}</p><p class="chart-sub">${sub || ""}</p>`;
    container.appendChild(head);
  }
  const cellH = 26, headH = 30;
  const width = rowLabelW + colLabels.length * cellW;
  const height = headH + rowLabels.length * cellH;
  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, width: "100%", height, style: "overflow: visible; display:block;" });

  colLabels.forEach((cl, j) => {
    const t = svgEl("text", { x: rowLabelW + j * cellW + cellW / 2, y: headH - 10, "text-anchor": "middle", "font-size": "10.5" });
    t.textContent = cl;
    svg.appendChild(t);
  });

  rowLabels.forEach((rl, i) => {
    const y = headH + i * cellH;
    const t = svgEl("text", { x: rowLabelW - 10, y: y + cellH / 2 + 4, "text-anchor": "end", "font-size": "11.5" });
    t.textContent = rl;
    svg.appendChild(t);

    matrix[i].forEach((v, j) => {
      const x = rowLabelW + j * cellW;
      const g = svgEl("g", { class: "heatmap-cell" });
      g.appendChild(svgEl("rect", { x, y, width: cellW, height: cellH, fill: seqColor(v) }));
      const label = svgEl("text", { x: x + cellW / 2, y: y + cellH / 2 + 4, "text-anchor": "middle" });
      label.style.fill = textOnFill(v);
      label.textContent = cellFormat(v);
      g.appendChild(label);
      g.addEventListener("mousemove", (evt) => {
        showTooltip(evt, `<div class="t-title">${rl}</div><div class="t-row"><span>${cl}</span><span>${cellFormat(v)}</span></div>`);
      });
      g.addEventListener("mouseleave", hideTooltip);
      svg.appendChild(g);
    });
  });
  container.appendChild(svg);
}

/* ==================================================================
   Scatter: train time (log x) vs F1 macro (y)
   ================================================================== */
function renderScatter(container, { points, xLabel, yLabel, title, sub }) {
  container.innerHTML = "";
  const head = document.createElement("div");
  head.innerHTML = `<p class="chart-title">${title}</p><p class="chart-sub">${sub || ""}</p>`;
  container.appendChild(head);

  if (points.length === 0) {
    container.appendChild(Object.assign(document.createElement("p"), { className: "loading-note", textContent: "No models match the current family filter." }));
    return;
  }

  const width = 900, height = 440;
  const padL = 46, padR = 24, padT = 16, padB = 40;
  const plotW = width - padL - padR, plotH = height - padT - padB;

  const xs = points.map((p) => p.x);
  const xMin = Math.log10(Math.min(...xs) * 0.7);
  const xMax = Math.log10(Math.max(...xs) * 1.4);
  const yMin = 0, yMax = 1;

  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, width: "100%", height, style: "overflow: visible; display:block;" });

  for (let g = 0; g <= 1; g += 0.25) {
    const y = padT + plotH - g * plotH;
    svg.appendChild(svgEl("line", { class: "grid-line", x1: padL, x2: width - padR, y1: y, y2: y }));
    const t = svgEl("text", { class: "axis-label", x: padL - 8, y: y + 4, "text-anchor": "end", "font-size": "10.5" });
    t.textContent = g.toFixed(2);
    svg.appendChild(t);
  }
  svg.appendChild(svgEl("line", { class: "baseline", x1: padL, x2: padL, y1: padT, y2: padT + plotH }));
  svg.appendChild(svgEl("line", { class: "baseline", x1: padL, x2: width - padR, y1: padT + plotH, y2: padT + plotH }));

  const tickVals = [0.001, 0.01, 0.1, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000].filter((v) => Math.log10(v) >= xMin && Math.log10(v) <= xMax);
  tickVals.forEach((v) => {
    const x = padL + ((Math.log10(v) - xMin) / (xMax - xMin)) * plotW;
    const t = svgEl("text", { class: "axis-label", x, y: height - padB + 16, "text-anchor": "middle", "font-size": "10.5" });
    t.textContent = v + "s";
    svg.appendChild(t);
  });

  svg.appendChild(Object.assign(svgEl("text", { class: "axis-label", x: padL + plotW / 2, y: height - 4, "text-anchor": "middle", "font-size": "11" }), { textContent: xLabel }));
  svg.appendChild(Object.assign(svgEl("text", { class: "axis-label", x: 12, y: padT + plotH / 2, "text-anchor": "middle", "font-size": "11", transform: `rotate(-90 12 ${padT + plotH / 2})` }), { textContent: yLabel }));

  points.forEach((p) => {
    const px = padL + ((Math.log10(p.x) - xMin) / (xMax - xMin)) * plotW;
    const py = padT + plotH - ((p.y - yMin) / (yMax - yMin)) * plotH;

    svg.appendChild(svgEl("circle", { cx: px, cy: py, r: 5.5, fill: p.color, stroke: cssVar("--surface"), "stroke-width": 1.5 }));

    const hit = svgEl("circle", { cx: px, cy: py, r: 11, class: "hit-target" });
    hit.addEventListener("mousemove", (evt) => {
      showTooltip(evt, `<div class="t-title">${p.label}</div><div class="t-row"><span>${p.familyLabel}</span></div><div class="t-row"><span>${xLabel}</span><span>${fmtTime(p.x)}</span></div><div class="t-row"><span>${yLabel}</span><span>${fmtPct(p.y)}</span></div>`);
    });
    hit.addEventListener("mouseleave", hideTooltip);
    svg.appendChild(hit);
  });

  container.appendChild(svg);
}

/* ==================================================================
   Legend / family filter chips
   ================================================================== */
function renderLegend(container, items) {
  container.innerHTML = "";
  items.forEach((it) => {
    const span = document.createElement("span");
    span.className = "item";
    span.innerHTML = `<span class="dot" style="background:${it.color}"></span>${it.label}`;
    container.appendChild(span);
  });
}

let activeGroups = new Set(GROUP_ORDER);
function renderFamilyChips(container, onChange) {
  container.innerHTML = "";
  GROUP_ORDER.forEach((g) => {
    const btn = document.createElement("button");
    btn.className = "chip" + (activeGroups.has(g) ? "" : " off");
    btn.innerHTML = `<span class="dot" style="background:${cssVar(`--grp-${g}`)}"></span>${GROUP_LABEL[g]}`;
    btn.addEventListener("click", () => {
      if (activeGroups.has(g)) activeGroups.delete(g); else activeGroups.add(g);
      btn.classList.toggle("off");
      onChange();
    });
    container.appendChild(btn);
  });
}

/* ==================================================================
   Data / state
   ================================================================== */
let DATA = null;
let currentDatasetKey = "mqttset";

function currentDS() { return DATA.datasets[currentDatasetKey]; }
function currentModels() { return currentDS().models_sorted; }
function currentClassNames() { return currentDS().dataset.classes; }
function filteredModels() { return currentModels().filter((m) => activeGroups.has(groupOf(m.family))); }

function buildDatasetSwitch() {
  const el = document.getElementById("dataset-switch");
  el.innerHTML = "";
  Object.entries(DATA.datasets).forEach(([key, d]) => {
    const btn = document.createElement("button");
    btn.textContent = d.dataset.name;
    if (key === currentDatasetKey) btn.classList.add("active");
    btn.addEventListener("click", () => {
      currentDatasetKey = key;
      renderAll();
    });
    el.appendChild(btn);
  });
}

function buildHeaderMeta() {
  const d = currentDS().dataset;
  document.getElementById("meta-row").innerHTML = `
    <span>${d.n_train.toLocaleString()} train rows</span>
    <span>${d.n_test.toLocaleString()} test rows</span>
    <span>${d.n_features} features</span>
    <span>${d.classes.length} classes</span>
    <span>${d.n_models_ok}/${d.n_models_attempted} models trained</span>
  `;
  document.getElementById("dataset-note").textContent = d.note;
}

function buildOverview() {
  const models = currentModels();
  const d = currentDS().dataset;
  const top = [...models].sort((a, b) => b.f1_macro - a.f1_macro).slice(0, 15);
  const best = top[0];
  const fastGood = [...models].filter((m) => m.f1_macro >= 0.8).sort((a, b) => a.train_time_sec - b.train_time_sec)[0];

  document.getElementById("overview-stats").innerHTML = `
    <div class="stat-card"><div class="label">Models attempted</div><div class="value">${d.n_models_attempted}</div><div class="sub">${d.n_models_ok} succeeded, ${d.n_models_failed} failed</div></div>
    <div class="stat-card"><div class="label">Best F1-macro</div><div class="value small">${best ? best.name : "—"}</div><div class="sub">${best ? fmtPct(best.f1_macro) : ""}</div></div>
    <div class="stat-card"><div class="label">Best fast model (F1 ≥ 0.80)</div><div class="value small">${fastGood ? fastGood.name : "none above 0.80"}</div><div class="sub">${fastGood ? fmtTime(fastGood.train_time_sec) + " train" : ""}</div></div>
    <div class="stat-card"><div class="label">Families represented</div><div class="value">${currentDS().families.length}</div><div class="sub">grouped into ${GROUP_ORDER.length} color groups below</div></div>
  `;

  renderBarPanel(document.getElementById("panel-top-f1"), {
    title: "Top 15 by F1-macro", sub: "Unweighted average across classes -- rewards handling rare attack classes well",
    items: top.map((m) => ({ label: m.name, value: m.f1_macro, color: colorForFamily(m.family), metaHtml: `<div class="t-row"><span>family</span><span>${GROUP_LABEL[groupOf(m.family)]}</span></div>` })),
  });

  const topSpeed = [...models].sort((a, b) => a.train_time_sec - b.train_time_sec).slice(0, 15);
  renderBarPanel(document.getElementById("panel-top-speed"), {
    title: "15 fastest to train", sub: "Log-scaled bar width; hover for exact F1-macro",
    format: fmtTime,
    items: topSpeed.map((m) => ({ label: m.name, value: m.train_time_sec, color: colorForFamily(m.family), metaHtml: `<div class="t-row"><span>f1_macro</span><span>${fmtPct(m.f1_macro)}</span></div>` })),
    valueForWidth: (it) => Math.log10(it.value + 0.01),
  });

  renderLegend(document.getElementById("overview-legend"), GROUP_ORDER.map((g) => ({ label: GROUP_LABEL[g], color: cssVar(`--grp-${g}`) })));
}

function buildFamilies() {
  const models = currentModels();
  const byFamily = {};
  models.forEach((m) => { (byFamily[m.family] ||= []).push(m); });

  const rows = Object.entries(byFamily).map(([fam, ms]) => {
    const best = [...ms].sort((a, b) => b.f1_macro - a.f1_macro)[0];
    const avg = ms.reduce((s, m) => s + m.f1_macro, 0) / ms.length;
    return { family: fam, group: groupOf(fam), count: ms.length, best, avg };
  }).sort((a, b) => b.best.f1_macro - a.best.f1_macro);

  renderBarPanel(document.getElementById("panel-family-best"), {
    title: "Best model per family (by F1-macro)", sub: "Every raw family's strongest entrant",
    items: rows.map((r) => ({ label: `${r.best.name}`, value: r.best.f1_macro, color: colorForFamily(r.family), metaHtml: `<div class="t-row"><span>family</span><span>${r.family}</span></div><div class="t-row"><span>family avg f1</span><span>${fmtPct(r.avg)}</span></div>` })),
  });

  const tbody = document.getElementById("family-table-body");
  tbody.innerHTML = rows.map((r) => `
    <tr>
      <td><span class="model-dot" style="display:inline-block;width:8px;height:8px;margin-right:6px;background:${colorForFamily(r.family)}"></span>${r.family}</td>
      <td>${GROUP_LABEL[r.group]}</td>
      <td>${r.count}</td>
      <td>${r.best.name}</td>
      <td>${fmtPct(r.best.f1_macro)}</td>
      <td>${fmtPct(r.avg)}</td>
    </tr>`).join("");
}

function buildScatter() {
  renderFamilyChips(document.getElementById("scatter-filters"), buildScatter);
  const models = filteredModels();
  renderScatter(document.getElementById("scatter-tradeoff"), {
    points: models.map((m) => ({ x: Math.max(m.train_time_sec, 0.001), y: m.f1_macro, label: m.name, familyLabel: GROUP_LABEL[groupOf(m.family)], color: colorForFamily(m.family) })),
    xLabel: "Train time (seconds, log scale)", yLabel: "F1 macro",
    title: "Speed vs. quality trade-off", sub: "Bottom-right = fast but weaker. Top-left = slower but better-balanced across classes. Click legend chips to isolate a family.",
  });
  renderLegend(document.getElementById("scatter-legend"), GROUP_ORDER.map((g) => ({ label: GROUP_LABEL[g], color: cssVar(`--grp-${g}`) })));
}

function buildPerClass() {
  const models = currentModels();
  const classNames = currentClassNames();
  const sorted = [...models].sort((a, b) => b.f1_macro - a.f1_macro);
  const matrix = sorted.map((m) => classNames.map((cn) => {
    const pc = m.per_class.find((x) => x.class === cn);
    return pc ? pc.f1 : 0;
  }));
  renderHeatmap(document.getElementById("heatmap-perclass"), {
    rowLabels: sorted.map((m) => m.name), colLabels: classNames, matrix,
    title: `Per-class F1 -- all ${sorted.length} successful models x ${classNames.length} classes`,
    sub: "Darker = higher F1. Sorted by overall F1-macro (best at top). Watch the rarest class column -- it separates models most.",
  });
}

function buildConfusionExplorer() {
  const select = document.getElementById("cm-model-select");
  const models = currentModels();
  select.innerHTML = "";
  models.forEach((m, i) => {
    const opt = document.createElement("option");
    opt.value = i;
    opt.textContent = `${m.name} (${m.family})`;
    select.appendChild(opt);
  });

  function render() {
    const m = models[Number(select.value)];
    if (!m) return;
    const classNames = currentClassNames();
    const cm = m.confusion_matrix;
    const rowSums = cm.map((row) => row.reduce((a, b) => a + b, 0));
    const norm = cm.map((row, i) => row.map((v) => (rowSums[i] ? v / rowSums[i] : 0)));
    renderHeatmap(document.getElementById("heatmap-confusion"), {
      rowLabels: classNames.map((c) => "true: " + c), colLabels: classNames, matrix: norm,
      title: `Confusion matrix -- ${m.name}`, sub: "Row-normalized: each row sums to 1.0 (recall breakdown for that true class).",
      rowLabelW: 150,
    });
  }
  select.onchange = render;
  render();
}

let tableSort = { key: "f1_macro", dir: -1 };
function buildTable() {
  const models = currentModels();
  const failed = currentDS().failed_models;
  const all = [...models, ...failed];

  const cols = [
    ["name", "Model"], ["family", "Family"], ["status", "Status"],
    ["accuracy", "Accuracy"], ["precision_macro", "Precision"], ["recall_macro", "Recall"],
    ["f1_macro", "F1 macro"], ["f1_weighted", "F1 weighted"],
    ["train_time_sec", "Train time"], ["inference_time_per_sample_ms", "Infer/sample"], ["model_size", "Size"],
  ];

  const thead = document.getElementById("full-table-head");
  thead.innerHTML = "<tr>" + cols.map(([key, label]) => `<th class="sortable${tableSort.key === key ? " sort-active" : ""}" data-key="${key}">${label}${tableSort.key === key ? (tableSort.dir === 1 ? " ↑" : " ↓") : ""}</th>`).join("") + "</tr>";
  thead.querySelectorAll("th.sortable").forEach((th) => {
    th.addEventListener("click", () => {
      const key = th.dataset.key;
      if (tableSort.key === key) tableSort.dir *= -1; else tableSort = { key, dir: 1 };
      buildTable();
    });
  });

  const sorted = [...all].sort((a, b) => {
    const av = a[tableSort.key], bv = b[tableSort.key];
    if (av === undefined || av === null) return 1;
    if (bv === undefined || bv === null) return -1;
    if (typeof av === "string") return tableSort.dir * av.localeCompare(bv);
    return tableSort.dir * (av - bv);
  });

  const tbody = document.getElementById("full-table-body");
  tbody.innerHTML = sorted.map((m) => `
    <tr class="${m.status === "failed" ? "status-failed" : ""}">
      <td><span class="model-dot" style="display:inline-block;width:8px;height:8px;margin-right:6px;background:${colorForFamily(m.family)}"></span>${m.name}</td>
      <td>${m.family}</td>
      <td>${m.status === "ok" ? "ok" : "failed"}</td>
      <td>${m.status === "ok" ? fmtPct(m.accuracy) : "—"}</td>
      <td>${m.status === "ok" ? fmtPct(m.precision_macro) : "—"}</td>
      <td>${m.status === "ok" ? fmtPct(m.recall_macro) : "—"}</td>
      <td>${m.status === "ok" ? fmtPct(m.f1_macro) : "—"}</td>
      <td>${m.status === "ok" ? fmtPct(m.f1_weighted) : "—"}</td>
      <td>${m.status === "ok" ? fmtTime(m.train_time_sec) : "—"}</td>
      <td>${m.status === "ok" ? m.inference_time_per_sample_ms.toFixed(4) + " ms" : "—"}</td>
      <td>${m.status === "ok" ? m.model_size : "—"}</td>
    </tr>`).join("");
}

function buildFailedRuns() {
  const failed = currentDS().failed_models;
  const el = document.getElementById("failed-runs-content");
  if (failed.length === 0) {
    el.innerHTML = `<div class="callout"><strong>All ${currentDS().dataset.n_models_attempted} attempted models trained successfully</strong> on this dataset -- nothing to report.</div>`;
    return;
  }
  el.innerHTML = `<div class="callout warn wide"><strong>${failed.length} of ${currentDS().dataset.n_models_attempted} attempted models failed</strong> on this dataset -- shown here for transparency rather than silently dropped.</div>` +
    `<div class="table-scroll" style="margin-top:16px;"><table><thead><tr><th>Model</th><th>Family</th><th>Error</th></tr></thead><tbody>` +
    failed.map((m) => `<tr><td>${m.name}</td><td>${m.family}</td><td style="text-align:left; white-space:normal;">${m.error}</td></tr>`).join("") +
    `</tbody></table></div>`;
}

function buildModelCards() {
  const models = currentModels();
  const grid = document.getElementById("model-cards");
  const byFamily = {};
  models.forEach((m) => { (byFamily[m.family] ||= []).push(m); });

  grid.innerHTML = "";
  Object.entries(byFamily).sort((a, b) => a[0].localeCompare(b[0])).forEach(([fam, ms]) => {
    const heading = document.createElement("div");
    heading.className = "family-heading";
    heading.textContent = `${fam} (${GROUP_LABEL[groupOf(fam)]})`;
    grid.appendChild(heading);

    const row = document.createElement("div");
    row.className = "model-grid";
    ms.forEach((m) => {
      const hp = Object.entries(m.hyperparams || {}).filter(([k]) => k !== "seed").map(([k, v]) => `<span>${k}: ${v}</span>`).join("");
      const card = document.createElement("div");
      card.className = "model-card";
      card.innerHTML = `
        <div class="model-card-head">
          <span class="model-dot" style="background:${colorForFamily(m.family)}"></span>
          <h3>${m.name}</h3>
          <span class="model-tag">F1 ${fmtPct(m.f1_macro)}</span>
        </div>
        <p class="desc">${m.description}</p>
        <div class="hp-list">${hp}</div>
      `;
      row.appendChild(card);
    });
    grid.appendChild(row);
  });
}

/* ==================================================================
   Deep Dive: feature selection, malicious/benign, scaling
   ================================================================== */
let DEEPDIVE = null;
function currentDeepDive() { return DEEPDIVE && DEEPDIVE.datasets[currentDatasetKey]; }

const DEEPDIVE_MODEL_FAMILY = { xgboost: "boosting", random_forest: "ensemble_bagging", logreg: "linear", knn5: "instance", ebm: "interpretable" };
const DEEPDIVE_MODEL_LABEL = { xgboost: "XGBoost", random_forest: "Random Forest", logreg: "Logistic Regression", knn5: "K-Nearest Neighbors", ebm: "Explainable Boosting Machine" };

function buildDeepDiveFeatures() {
  const dd = currentDeepDive();
  const panelsEl = document.getElementById("deepdive-feature-panels");
  const noteEl = document.getElementById("deepdive-feature-note");
  if (!dd) { panelsEl.innerHTML = ""; noteEl.innerHTML = ""; return; }

  const perModel = dd.feature_selection.per_model_feature_selection;
  panelsEl.innerHTML = "";
  const topSets = {};
  Object.entries(perModel).forEach(([key, entry]) => {
    const card = document.createElement("div");
    card.className = "card";
    panelsEl.appendChild(card);
    const items = (entry.permutation_importance_top15 || []).slice(0, 8);
    topSets[key] = new Set(items.slice(0, 5).map((x) => x.feature));
    renderBarPanel(card, {
      title: `${DEEPDIVE_MODEL_LABEL[key] || key} (permutation importance)`,
      sub: `family: ${entry.family} · fit ${fmtTime(entry.fit_time_sec)}`,
      items: items.map((x) => ({ label: x.feature, value: x.importance, color: colorForFamily(DEEPDIVE_MODEL_FAMILY[key]) })),
      format: (v) => v.toFixed(3), domainMax: Math.max(0.001, ...items.map((x) => x.importance)),
    });
  });

  // rough agreement: how many of xgboost's top-5 features also show up in each other model's top-5
  const ref = topSets["xgboost"];
  let agreementHtml = "";
  if (ref) {
    const overlaps = Object.entries(topSets).filter(([k]) => k !== "xgboost").map(([k, s]) => {
      const shared = [...ref].filter((f) => s.has(f)).length;
      return `${DEEPDIVE_MODEL_LABEL[k] || k} shares ${shared}/5`;
    });
    agreementHtml = `<strong>Top-5 feature overlap with XGBoost.</strong> ${overlaps.join(" · ")} of XGBoost's top-5 permutation-important features. ` +
      `Tree/boosting-family models tend to converge on a similar small set of high-signal features (they all split greedily on whatever reduces impurity most); ` +
      `linear and instance-based models, which have no notion of a "best split," spread importance across a wider, less-overlapping set -- direct evidence that ` +
      `feature selection is genuinely model-family-dependent here, not just a property of the data.`;
  }
  noteEl.innerHTML = agreementHtml;
}

function buildDeepDiveMvB() {
  const dd = currentDeepDive();
  const statsEl = document.getElementById("deepdive-mvb-stats");
  const chartEl = document.getElementById("deepdive-mvb-chart");
  const bodyEl = document.getElementById("deepdive-mvb-body");
  const featEl = document.getElementById("deepdive-mvb-features");
  if (!dd) { statsEl.innerHTML = ""; chartEl.innerHTML = ""; bodyEl.innerHTML = ""; featEl.innerHTML = ""; return; }

  const mvb = dd.malicious_vs_benign;
  const models = mvb.models;
  const best = models[0];
  const worst = models[models.length - 1];
  const avgFnr = models.reduce((s, m) => s + m.false_negative_rate, 0) / models.length;

  statsEl.innerHTML = `
    <div class="stat-card"><div class="label">Best malicious-detection F1</div><div class="value small">${best.name}</div><div class="sub">F1 ${fmtPct(best.f1_malicious)} · FNR ${fmtPct(best.false_negative_rate)} · FPR ${fmtPct(best.false_positive_rate)}</div></div>
    <div class="stat-card"><div class="label">Worst malicious-detection F1</div><div class="value small">${worst.name}</div><div class="sub">F1 ${fmtPct(worst.f1_malicious)} · FNR ${fmtPct(worst.false_negative_rate)}</div></div>
    <div class="stat-card"><div class="label">Avg. false-negative rate</div><div class="value">${fmtPct(avgFnr)}</div><div class="sub">across all ${models.length} successful models -- real attacks missed as "benign"</div></div>
    <div class="stat-card"><div class="label">Benign class</div><div class="value small">${mvb.benign_class_name}</div><div class="sub">every other class collapsed to "malicious"</div></div>
  `;

  const top15 = models.slice(0, 15);
  renderBarPanel(chartEl, {
    title: "Top 15 by malicious-detection F1", sub: "Binary view, collapsed from each model's existing multiclass confusion matrix -- no retraining",
    items: top15.map((m) => ({ label: m.name, value: m.f1_malicious, color: colorForFamily(m.family), metaHtml: `<div class="t-row"><span>FNR</span><span>${fmtPct(m.false_negative_rate)}</span></div><div class="t-row"><span>FPR</span><span>${fmtPct(m.false_positive_rate)}</span></div>` })),
  });

  bodyEl.innerHTML = models.map((m) => `
    <tr>
      <td><span class="model-dot" style="display:inline-block;width:8px;height:8px;margin-right:6px;background:${colorForFamily(m.family)}"></span>${m.name}</td>
      <td>${m.family}</td>
      <td>${fmtPct(m.f1_malicious)}</td>
      <td>${fmtPct(m.false_negative_rate)}</td>
      <td>${fmtPct(m.false_positive_rate)}</td>
      <td>${fmtPct(m.f1_macro_multiclass)}</td>
    </tr>`).join("");

  const bvm = dd.feature_selection.benign_vs_malicious_feature_selection;
  featEl.innerHTML = "";
  [["mutual_info_top15", "Mutual information", "captures any dependency, linear or not"],
   ["anova_f_top15", "ANOVA F-test", "captures linear separability only"]].forEach(([field, title, sub]) => {
    const card = document.createElement("div");
    card.className = "card";
    featEl.appendChild(card);
    const items = bvm[field].slice(0, 10);
    renderBarPanel(card, {
      title: `${title}: top 10 benign-vs-malicious features`, sub,
      items: items.map((x) => ({ label: x.feature, value: x.importance, color: cssVar("--accent") })),
      format: (v) => v.toFixed(3), domainMax: Math.max(0.001, ...items.map((x) => x.importance)),
    });
  });
}

const SCALE_ROWS = [
  { complexity: "O(n log n) — histogram-binned, near-linear", label: "Histogram gradient boosting", key: "histgbm", mode: "time_linear" },
  { complexity: "O(n log n) per tree, embarrassingly parallel", label: "Bagged trees (Random Forest)", key: "random_forest", mode: "time_linear" },
  { complexity: "O(n) per solver iteration", label: "Linear (well-chosen solver)", key: "logreg", mode: "time_linear" },
  { complexity: "O(n²)–O(n³) — OvO + balanced weights on imbalanced data", label: "Kernel SVM", key: "svc_rbf", mode: "time_quadratic" },
  { complexity: "O(n³) time, O(n²) memory — exact kernel matrix inversion", label: "Gaussian Process", key: "gaussian_process", mode: "time_cubic" },
  { complexity: "O(n²) memory — full pairwise similarity graph", label: "Graph-based (Label Spreading)", key: "label_spreading", mode: "memory_quadratic" },
  { complexity: "O(1) \"train\", O(n) per query at inference", label: "Instance-based (KNN, lazy)", key: "knn5", mode: "inference_linear" },
];
const TARGET_ROWS = 4_000_000;

function buildDeepDiveScale() {
  const tbody = document.getElementById("deepdive-scale-table");
  if (!DATA) { tbody.innerHTML = ""; return; }
  const models = currentModels();
  const d = currentDS().dataset;

  tbody.innerHTML = SCALE_ROWS.map((row) => {
    const m = models.find((x) => x.key === row.key);
    if (!m) return `<tr><td>${row.complexity}</td><td>${row.label}</td><td colspan="2">not available for this dataset</td></tr>`;
    const nUsed = m.subsample_cap || d.n_train;
    const scale = TARGET_ROWS / nUsed;
    let current, projected;
    if (row.mode === "time_linear") {
      current = `${fmtTime(m.train_time_sec)} train on ${nUsed.toLocaleString()} rows`;
      projected = `~${fmtBigDuration(m.train_time_sec * scale)} train (linear extrapolation)`;
    } else if (row.mode === "time_quadratic") {
      current = `${fmtTime(m.train_time_sec)} train, capped to ${nUsed.toLocaleString()} rows`;
      projected = `~${fmtBigDuration(m.train_time_sec * scale * scale)} train if naively scaled -- not viable`;
    } else if (row.mode === "time_cubic") {
      current = `${fmtTime(m.train_time_sec)} train, capped to ${nUsed.toLocaleString()} rows`;
      projected = `~${fmtBigDuration(m.train_time_sec * scale * scale * scale)} train if naively scaled -- not viable`;
    } else if (row.mode === "memory_quadratic") {
      const currentBytes = nUsed * nUsed * 8;
      current = `~${fmtBigBytes(currentBytes)} similarity matrix, capped to ${nUsed.toLocaleString()} rows`;
      projected = `~${fmtBigBytes(currentBytes * scale * scale)} similarity matrix -- not viable`;
    } else if (row.mode === "inference_linear") {
      current = `${m.inference_time_per_sample_ms.toFixed(4)} ms/sample, ${nUsed.toLocaleString()} rows stored`;
      projected = `~${(m.inference_time_per_sample_ms * scale).toFixed(2)} ms/sample at inference time`;
    }
    return `<tr><td>${row.complexity}</td><td>${row.label}</td><td>${current}</td><td>${projected}</td></tr>`;
  }).join("");
}

/* ---- tabs ---- */
function buildTabs() {
  const buttons = document.querySelectorAll("nav.tabs button");
  const views = document.querySelectorAll("section.view");
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      buttons.forEach((b) => b.classList.remove("active"));
      views.forEach((v) => v.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(btn.dataset.target).classList.add("active");
    });
  });
}

function renderAll() {
  buildDatasetSwitch();
  buildHeaderMeta();
  buildOverview();
  buildFamilies();
  buildScatter();
  buildPerClass();
  buildConfusionExplorer();
  buildTable();
  buildFailedRuns();
  buildModelCards();
  buildDeepDiveFeatures();
  buildDeepDiveMvB();
  buildDeepDiveScale();
}

/* ---- init ---- */
Promise.all([
  fetch("data.json").then((r) => r.json()),
  fetch("deep-dive-data.json").then((r) => (r.ok ? r.json() : null)).catch(() => null),
])
  .then(([data, deepDive]) => {
    DATA = data;
    DEEPDIVE = deepDive;
    buildTabs();
    renderAll();
  })
  .catch((err) => {
    document.querySelector(".wrap").innerHTML = `<div class="callout warn wide" style="margin-top:40px;"><strong>Couldn't load data.json</strong><br>${err}</div>`;
  });
