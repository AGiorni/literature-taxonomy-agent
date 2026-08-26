/* littax viewer.
 *
 * One filter row scopes every panel. Selecting a folder in the tree filters the
 * scatter, the histogram and the table alike; nothing has its own filters.
 */
"use strict";

const $ = (sel) => document.querySelector(sel);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

const state = {
  taxonomy: null,
  folders: new Map(),
  childrenOf: new Map(),
  selected: null,      // folder id, or null for the whole corpus
  level: 0,            // 0 = all levels
  bands: new Set(["high", "medium", "low"]),
  search: "",
  colourMode: "folder",
  onlyFlagged: false,
  sort: { key: "confidence", dir: 1 },
  expanded: new Set(),
};

/* ---------------------------------------------------------------- data --- */

function indexTaxonomy(taxonomy) {
  state.taxonomy = taxonomy;
  state.folders = new Map(taxonomy.folders.map((f) => [f.id, f]));
  state.childrenOf = new Map();
  for (const folder of taxonomy.folders) {
    const key = folder.parent_id ?? "__root__";
    if (!state.childrenOf.has(key)) state.childrenOf.set(key, []);
    state.childrenOf.get(key).push(folder);
  }
  for (const list of state.childrenOf.values()) {
    list.sort((a, b) => b.n_documents_total - a.n_documents_total);
  }
  state.expanded = new Set((state.childrenOf.get("__root__") || []).map((f) => f.id));
}

/** Every folder id at or beneath `id`. */
function subtreeIds(id) {
  const out = new Set();
  const walk = (folderId) => {
    out.add(folderId);
    for (const child of state.childrenOf.get(folderId) || []) walk(child.id);
  };
  walk(id);
  return out;
}

function ancestorAtLevel(folderId, level) {
  let folder = state.folders.get(folderId);
  while (folder && folder.level > level) folder = state.folders.get(folder.parent_id);
  return folder ? folder.id : folderId;
}

/** Documents passing the current filter row, excluding byte-duplicates. */
function visibleDocuments() {
  const scope = state.selected ? subtreeIds(state.selected) : null;
  const needle = state.search.trim().toLowerCase();

  return state.taxonomy.documents.filter((doc) => {
    if (doc.duplicate_of || !doc.assignment) return false;
    if (scope && !scope.has(doc.assignment.folder_id)) return false;
    if (!state.bands.has(doc.assignment.band)) return false;
    if (state.onlyFlagged && !needsReview(doc)) return false;
    if (needle) {
      const hay = [
        doc.title, doc.filename, (doc.authors || []).join(" "),
        (doc.keywords || []).join(" "), (doc.top_terms || []).join(" "),
        doc.assignment.folder_path,
      ].join(" ").toLowerCase();
      if (!hay.includes(needle)) return false;
    }
    return true;
  });
}

function needsReview(doc) {
  const fit = doc.assignment;
  if (!fit) return false;
  return fit.band === "low" || fit.was_noise || fit.llm_verified === false;
}

function folderPathOf(doc) {
  return doc.assignment ? doc.assignment.folder_path : "—";
}

/* -------------------------------------------------------------- colour --- */

const SEQ = () => [css("--seq-1"), css("--seq-2"), css("--seq-3"), css("--seq-4"), css("--seq-5")];

/* Categorical slots, in fixed order — assigned to entities, never cycled.
 *
 * Three, not eight. A scatter can put any two colours side by side, so it is
 * judged on all pairs rather than adjacent ones, and only the first three slots
 * of the palette clear the colourblind and normal-vision floors that way
 * (verified with the palette validator: a 4th slot fails in one mode or the
 * other — violet collides with blue under protanopia on the dark surface,
 * yellow with orange in light). Everything past the top three folds into
 * "Other", and the centroid labels carry identity for the rest. */
const CATEGORICAL = () => [css("--series-1"), css("--series-2"), css("--series-3")];
const UNSORTED_ID = "_unsorted";

/* Shape is the second channel. Three validated hues alone cap the chart at
 * three folders, which a real level-1 taxonomy already exceeds; pairing each
 * hue with a marker shape lifts that to nine without weakening a single colour
 * pair, because any two groups now differ in hue OR in shape. Glyphs mirror the
 * plot symbols in the legend. */
const SYMBOLS = ["circle", "diamond", "square"];
const SYMBOL_GLYPH = { circle: "●", diamond: "◆", square: "■" };

/** Folders at `level`, biggest first, each given a colour+shape pair.
 *  Returns the assignment plus the legend it implies. */
function folderColouring(docs, level) {
  const counts = new Map();
  for (const doc of docs) {
    const id = ancestorAtLevel(doc.assignment.folder_id, level);
    counts.set(id, (counts.get(id) || 0) + 1);
  }

  // _Unsorted is a residue, not a topic: it never earns one of three slots.
  const ranked = [...counts.entries()]
    .filter(([id]) => id !== UNSORTED_ID)
    .sort((a, b) => b[1] - a[1]);

  const hues = CATEGORICAL();
  const capacity = hues.length * SYMBOLS.length;
  const styleOf = new Map();
  ranked.slice(0, capacity).forEach(([id], i) =>
    styleOf.set(id, {
      colour: hues[i % hues.length],
      symbol: SYMBOLS[Math.floor(i / hues.length)],
    })
  );

  const other = { colour: css("--muted"), symbol: "circle" };
  const legend = ranked.slice(0, capacity).map(([id, n]) => ({
    label: state.folders.get(id)?.label ?? id,
    ...styleOf.get(id),
    n,
  }));

  const spilled = ranked.slice(capacity);
  const restCount = spilled.reduce((sum, [, n]) => sum + n, 0);
  const unsorted = counts.get(UNSORTED_ID) || 0;
  if (restCount) {
    legend.push({ label: `Other (${spilled.length} folders)`, ...other, n: restCount });
  }
  if (unsorted) legend.push({ label: "Unsorted", ...other, n: unsorted });

  return { styleOf, other, legend };
}

/** Sequential ramp as Plotly colorscale stops — one hue, light to dark. */
function seqScale() {
  const steps = SEQ();
  return steps.map((hex, i) => [i / (steps.length - 1), hex]);
}

function bandColour(band) {
  return band === "high" ? css("--good") : band === "medium" ? css("--warning") : css("--critical");
}

/* --------------------------------------------------------------- chart --- */

function baseLayout(extra = {}) {
  const ink = css("--text-secondary");
  const grid = css("--grid");
  return Object.assign({
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: 'system-ui, -apple-system, "Segoe UI", sans-serif', size: 12, color: ink },
    margin: { l: 48, r: 16, t: 8, b: 40 },
    hoverlabel: {
      bgcolor: css("--surface-1"),
      bordercolor: css("--border"),
      font: { color: css("--text-primary"), size: 12 },
      align: "left",
    },
    xaxis: { gridcolor: grid, zeroline: false, linecolor: css("--axis") },
    yaxis: { gridcolor: grid, zeroline: false, linecolor: css("--axis") },
    showlegend: false,
  }, extra);
}

const PLOT_CONFIG = { displayModeBar: false, responsive: true };

function drawScatter() {
  const docs = visibleDocuments().filter((d) => d.coords && d.coords.global);
  const sub = $("#scatter-sub");
  const legend = $("#scatter-legend");

  if (!docs.length) {
    Plotly.purge("scatter");
    sub.textContent = "No documents match the current filters.";
    legend.innerHTML = "";
    return;
  }

  // 0 means "the leaf each document was actually assigned to"; ancestorAtLevel
  // walks past the root and hands the original id back.
  const level = state.level;
  const scopeIds = state.selected ? subtreeIds(state.selected) : null;
  const x = docs.map((d) => d.coords.global[0]);
  const y = docs.map((d) => d.coords.global[1]);
  const text = docs.map((d) =>
    `<b>${escapeHtml(truncate(d.title || d.filename, 70))}</b><br>` +
    `${escapeHtml(folderPathOf(d))}<br>` +
    `confidence ${d.assignment.confidence.toFixed(2)} (${d.assignment.band})`
  );

  let marker;
  if (state.colourMode === "folder") {
    const { styleOf, other, legend: groups } = folderColouring(docs, level);
    const styles = docs.map(
      (d) => styleOf.get(ancestorAtLevel(d.assignment.folder_id, level)) || other
    );
    marker = {
      size: 9,
      color: styles.map((v) => v.colour),
      symbol: styles.map((v) => v.symbol),
      opacity: 0.85,
      line: { width: 2, color: css("--surface-1") },
    };
    legend.innerHTML = groups
      .map(
        (g) =>
          `<span><span class="glyph" style="color:${g.colour}">${SYMBOL_GLYPH[g.symbol]}</span>` +
          `${escapeHtml(truncate(g.label, 34))} <span class="muted">${g.n}</span></span>`
      )
      .join("");
  } else if (state.colourMode === "selection" && scopeIds) {
    // Two colours only: the selected branch against everything else.
    marker = {
      size: 9,
      color: docs.map((d) => scopeIds.has(d.assignment.folder_id) ? css("--series-1") : css("--muted")),
      opacity: 0.85,
      line: { width: 2, color: css("--surface-1") }, // 2px surface ring, not a border
    };
    legend.innerHTML =
      `<span><span class="swatch" style="background:${css("--series-1")}"></span>Selected</span>` +
      `<span><span class="swatch" style="background:${css("--muted")}"></span>Other</span>`;
  } else {
    marker = {
      size: 9,
      color: docs.map((d) => d.assignment.confidence),
      colorscale: seqScale(),
      cmin: 0, cmax: 1,
      showscale: false,
      opacity: 0.85,
      line: { width: 2, color: css("--surface-1") },
    };
    legend.innerHTML =
      `<span>Confidence</span><span>0</span>` +
      `<span class="ramp">${SEQ().map((c) => `<i style="background:${c}"></i>`).join("")}</span>` +
      `<span>1</span>`;
  }

  const traces = [{
    type: "scattergl", mode: "markers",
    x, y, text, hoverinfo: "text",
    marker,
    customdata: docs.map((d) => d.id),
  }];

  // Direct labels at cluster centroids: the identity channel. Colour is never
  // asked to distinguish 18 folders on its own.
  const centroids = new Map();
  for (const doc of docs) {
    const id = ancestorAtLevel(doc.assignment.folder_id, level);
    if (!centroids.has(id)) centroids.set(id, { x: 0, y: 0, n: 0 });
    const acc = centroids.get(id);
    acc.x += doc.coords.global[0];
    acc.y += doc.coords.global[1];
    acc.n += 1;
  }
  // Biggest first, so when two centroids collide the label that survives is the
  // one describing more documents.
  const ranked = [...centroids.entries()]
    .filter(([, acc]) => acc.n >= 3)
    .sort((a, b) => b[1].n - a[1].n)
    .map(([id, acc]) => ({ id, x: acc.x / acc.n, y: acc.y / acc.n, n: acc.n }));

  // Drop labels that would land on top of one another. Plotly does not
  // declutter annotations, and overlapping boxes are less readable than fewer.
  const spanX = Math.max(...x) - Math.min(...x) || 1;
  const spanY = Math.max(...y) - Math.min(...y) || 1;
  const placed = [];
  for (const candidate of ranked) {
    const clash = placed.some(
      (p) =>
        Math.abs(p.x - candidate.x) < spanX * 0.16 &&
        Math.abs(p.y - candidate.y) < spanY * 0.07
    );
    if (!clash) placed.push(candidate);
    if (placed.length === 10) break;
  }

  // Annotations rather than a text trace: they render in the SVG layer above
  // the WebGL markers, and they take a background, so a name is readable over a
  // dense cluster instead of disappearing into it. Full label, no truncation -
  // reading the name is the whole point of the layer.
  const annotations = placed.map((p) => ({
    x: p.x, y: p.y,
    text: escapeHtml(state.folders.get(p.id)?.label ?? p.id),
    showarrow: false,
    font: { size: 12, color: css("--text-primary") },
    bgcolor: css("--surface-1"),
    bordercolor: css("--border"),
    borderwidth: 1,
    borderpad: 4,
    opacity: 0.92,
    captureevents: false,
  }));

  const layout = baseLayout({
    hovermode: "closest",
    xaxis: { visible: false }, yaxis: { visible: false },
    margin: { l: 8, r: 8, t: 8, b: 8 },
    annotations,
  });

  Plotly.react("scatter", traces, layout, PLOT_CONFIG);
  const grouping = level === 0 ? "assigned folders" : `level-${level} folders`;
  sub.textContent = `${docs.length} documents · UMAP projection · labels mark ${grouping}`;

  const node = $("#scatter");
  node.removeAllListeners?.("plotly_click");
  node.on("plotly_click", (event) => {
    const id = event.points?.[0]?.customdata;
    if (id) showDetail(id);
  });
}

function drawSunburst() {
  const folders = state.taxonomy.folders.filter((f) => f.n_documents_total > 0);
  if (!folders.length) { Plotly.purge("sunburst"); return; }

  const trace = {
    type: "sunburst",
    ids: folders.map((f) => f.id),
    labels: folders.map((f) => truncate(f.label, 24)),
    parents: folders.map((f) => f.parent_id ?? ""),
    values: folders.map((f) => f.n_documents_total),
    branchvalues: "total",
    // Sequential: one hue, light to dark, encoding mean confidence.
    marker: {
      colors: folders.map((f) => f.confidence_stats?.mean ?? 0),
      colorscale: seqScale(),
      cmin: 0, cmax: 1,
      line: { width: 2, color: css("--surface-1") }, // 2px surface gap
    },
    text: folders.map((f) =>
      `${f.n_documents_total} docs · mean conf ${(f.confidence_stats?.mean ?? 0).toFixed(2)}`),
    hovertemplate: "<b>%{label}</b><br>%{text}<extra></extra>",
    insidetextorientation: "radial",
    textfont: { size: 11 },
  };

  Plotly.react("sunburst", [trace],
    baseLayout({ margin: { l: 0, r: 0, t: 0, b: 0 } }), PLOT_CONFIG);

  const node = $("#sunburst");
  node.removeAllListeners?.("plotly_sunburstclick");
  node.on("plotly_sunburstclick", (event) => {
    const id = event.points?.[0]?.id;
    if (id) { selectFolder(id); return false; }
  });
}

function drawHistogram() {
  const docs = visibleDocuments();
  const sub = $("#hist-sub");
  if (!docs.length) { Plotly.purge("histogram"); sub.textContent = ""; return; }

  const corpus = state.taxonomy.documents.filter((d) => d.assignment && !d.duplicate_of);
  const traces = [];

  // Corpus-wide distribution ghosted behind, for comparison, whenever the
  // current view is a subset of it.
  if (docs.length !== corpus.length) {
    traces.push({
      type: "histogram",
      x: corpus.map((d) => d.assignment.confidence),
      xbins: { start: 0, end: 1, size: 0.05 },
      histnorm: "percent",
      marker: { color: css("--grid") },
      hoverinfo: "skip",
      name: "Whole corpus",
    });
  }
  traces.push({
    type: "histogram",
    x: docs.map((d) => d.assignment.confidence),
    xbins: { start: 0, end: 1, size: 0.05 },
    histnorm: "percent",
    marker: { color: css("--series-1"), line: { width: 2, color: css("--surface-1") } },
    hovertemplate: "confidence %{x}<br>%{y:.1f}% of selection<extra></extra>",
    name: "Selection",
  });

  const layout = baseLayout({
    barmode: "overlay",
    bargap: 0.06,
    xaxis: { title: { text: "confidence", font: { size: 11 } }, range: [0, 1], gridcolor: css("--grid"), zeroline: false },
    yaxis: { title: { text: "% of documents", font: { size: 11 } }, gridcolor: css("--grid"), zeroline: false },
    showlegend: traces.length > 1,
    legend: { orientation: "h", y: 1.12, x: 0, font: { size: 11 } },
    margin: { l: 52, r: 12, t: 24, b: 44 },
  });
  Plotly.react("histogram", traces, layout, PLOT_CONFIG);

  const values = docs.map((d) => d.assignment.confidence).sort((a, b) => a - b);
  const median = values[Math.floor(values.length / 2)];
  const counts = { high: 0, medium: 0, low: 0 };
  for (const doc of docs) counts[doc.assignment.band] += 1;
  sub.textContent =
    `median ${median.toFixed(2)} · ${counts.high} high, ${counts.medium} medium, ${counts.low} low`;
}

/* ---------------------------------------------------------------- tiles --- */

function drawTiles() {
  const stats = state.taxonomy.stats;
  const docs = visibleDocuments();
  const scoped = state.selected ? state.folders.get(state.selected) : null;
  const values = docs.map((d) => d.assignment.confidence);
  const mean = values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0;
  const flagged = docs.filter(needsReview).length;

  const tiles = [
    { label: scoped ? "Documents in folder" : "Documents", value: docs.length,
      sub: scoped ? truncate(scoped.label, 30) : `${stats.n_duplicates} duplicates skipped` },
    { label: "Folders", value: stats.n_folders, sub: `${stats.n_leaf_folders} leaves · ${stats.levels} levels` },
    { label: "Mean confidence", value: mean.toFixed(2), sub: "0–1, corpus-calibrated" },
    { label: "Needs review", value: flagged, sub: "low confidence or flagged" },
    { label: "Abstract recall", value: `${Math.round(stats.abstract_recall * 100)}%`,
      sub: `${stats.n_degraded} degraded scans` },
  ];

  $("#tiles").innerHTML = tiles.map((t) => `
    <div class="tile">
      <div class="label">${escapeHtml(t.label)}</div>
      <div class="value">${escapeHtml(String(t.value))}</div>
      <div class="sub">${escapeHtml(t.sub)}</div>
    </div>`).join("");
}

/* ----------------------------------------------------------------- tree --- */

/* ---- folder name tooltip --------------------------------------------------
 * .node-label truncates in a narrow sidebar, so the full name is unreadable
 * without one. Built lazily and attached to <body>: .tree is a scroll
 * container, and a tip inside it would be clipped at the panel edge - exactly
 * where a long name needs to escape to.                                     */

let nodeTip = null;

function hideNodeTip() {
  if (nodeTip) nodeTip.hidden = true;
}

function showNodeTip(row, folder) {
  if (!nodeTip) {
    nodeTip = document.createElement("div");
    nodeTip.className = "node-tip";
    document.body.appendChild(nodeTip);
    // A tip anchored to a row that has scrolled away is worse than none.
    addEventListener("scroll", hideNodeTip, true);
  }
  nodeTip.innerHTML =
    escapeHtml(folder.label) +
    (folder.description ? `<span class="tip-desc">${escapeHtml(folder.description)}</span>` : "");
  nodeTip.hidden = false;

  // Measure after filling, then keep it inside the viewport: flip above the
  // row when there is no room below, and pull left when it would run off.
  const anchor = row.getBoundingClientRect();
  const tip = nodeTip.getBoundingClientRect();
  const left = Math.max(8, Math.min(anchor.left, innerWidth - tip.width - 8));
  const below = anchor.bottom + 6;
  const top = below + tip.height > innerHeight - 8 ? anchor.top - tip.height - 6 : below;

  nodeTip.style.left = `${left}px`;
  nodeTip.style.top = `${Math.max(8, top)}px`;
}

function drawTree() {
  const container = $("#tree");
  container.innerHTML = "";

  const all = document.createElement("div");
  all.className = "node-row";
  all.setAttribute("aria-selected", String(state.selected === null));
  all.innerHTML = `<span class="twisty leaf"></span>
    <span class="node-label"><strong>All documents</strong></span>
    <span class="node-count">${state.taxonomy.stats.n_documents}</span>`;
  all.onclick = () => selectFolder(null);
  container.appendChild(all);

  const build = (parentId, host) => {
    for (const folder of state.childrenOf.get(parentId) || []) {
      const children = state.childrenOf.get(folder.id) || [];
      const isOpen = state.expanded.has(folder.id);
      const mean = folder.confidence_stats?.mean ?? 0;

      const row = document.createElement("div");
      row.className = "node-row";
      row.setAttribute("aria-selected", String(state.selected === folder.id));
      row.onmouseenter = () => showNodeTip(row, folder);
      row.onmouseleave = hideNodeTip;
      row.innerHTML = `
        <button class="twisty${children.length ? "" : " leaf"}" aria-label="Toggle">${isOpen ? "▼" : "▶"}</button>
        <span class="node-label">${escapeHtml(folder.label)}</span>
        <span class="node-meter" title="mean confidence ${mean.toFixed(2)}"><i style="width:${Math.round(mean * 100)}%"></i></span>
        <span class="node-count">${folder.n_documents_total}</span>`;
      row.onclick = (event) => {
        if (event.target.classList.contains("twisty")) {
          state.expanded.has(folder.id) ? state.expanded.delete(folder.id) : state.expanded.add(folder.id);
          drawTree();
        } else {
          selectFolder(folder.id);
        }
      };
      host.appendChild(row);

      if (children.length && isOpen) {
        const nested = document.createElement("div");
        nested.className = "children";
        host.appendChild(nested);
        build(folder.id, nested);
      }
    }
  };
  build("__root__", container);
}

function selectFolder(id) {
  state.selected = id;
  if (id) {
    // Reveal the selection by opening its ancestors.
    let folder = state.folders.get(id);
    while (folder && folder.parent_id) {
      state.expanded.add(folder.parent_id);
      folder = state.folders.get(folder.parent_id);
    }
    state.expanded.add(id);
  }
  render();
}

/* ---------------------------------------------------------------- table --- */

const SORT_KEYS = {
  confidence: (d) => d.assignment.confidence,
  title: (d) => (d.title || d.filename).toLowerCase(),
  year: (d) => d.year ?? 0,
  folder: (d) => folderPathOf(d).toLowerCase(),
  runner: (d) => d.assignment.runner_up?.similarity ?? 0,
};

function drawTable() {
  const docs = visibleDocuments();
  const key = SORT_KEYS[state.sort.key] || SORT_KEYS.confidence;
  docs.sort((a, b) => {
    const x = key(a), y = key(b);
    return (x < y ? -1 : x > y ? 1 : 0) * state.sort.dir;
  });

  const rows = docs.slice(0, 400).map((doc) => {
    const fit = doc.assignment;
    const glyph = fit.band === "high" ? "●" : fit.band === "medium" ? "◐" : "○";
    const flags = [];
    if (fit.was_noise) flags.push("outlier");
    if (fit.llm_verified === false) flags.push("model disagrees");
    return `<tr data-id="${doc.id}">
      <td class="num" style="color:${bandColour(fit.band)}">${glyph} ${fit.confidence.toFixed(2)}</td>
      <td class="title">${escapeHtml(truncate(doc.title || doc.filename, 95))}
        ${flags.length ? `<div class="flag">${escapeHtml(flags.join(" · "))}</div>` : ""}</td>
      <td class="num">${doc.year ?? "—"}</td>
      <td class="folder">${escapeHtml(truncate(folderPathOf(doc), 44))}</td>
      <td class="folder">${fit.runner_up ? escapeHtml(truncate(fit.runner_up.folder_path.split("/").pop(), 26)) : "—"}</td>
    </tr>`;
  }).join("");

  $("#doc-table tbody").innerHTML = rows || `<tr><td colspan="5" class="muted">No documents match.</td></tr>`;
  $("#table-count").textContent =
    docs.length > 400 ? `Showing first 400 of ${docs.length}.` : `${docs.length} documents.`;

  for (const row of document.querySelectorAll("#doc-table tbody tr[data-id]")) {
    row.onclick = () => showDetail(row.dataset.id);
  }
}

/* --------------------------------------------------------------- detail --- */

function showDetail(documentId) {
  const doc = state.taxonomy.documents.find((d) => d.id === documentId);
  if (!doc) return;
  const fit = doc.assignment;
  const folder = fit ? state.folders.get(fit.folder_id) : null;

  $("#detail-body").innerHTML = `
    <h3>${escapeHtml(doc.title || doc.filename)}</h3>
    <p class="muted small">${escapeHtml((doc.authors || []).join(", "))}${doc.year ? ` · ${doc.year}` : ""}</p>
    <dl>
      <dt>Folder</dt><dd>${escapeHtml(fit ? fit.folder_path : "—")}</dd>
      <dt>Confidence</dt><dd style="color:${fit ? bandColour(fit.band) : ""}">
        ${fit ? `${fit.confidence.toFixed(2)} (${fit.band})` : "—"}</dd>
      ${fit ? `
      <dt>Similarity</dt><dd>${fit.similarity_to_centroid.toFixed(3)} to centroid</dd>
      <dt>Margin</dt><dd>${fit.margin_over_next.toFixed(3)} over runner-up</dd>
      ${fit.membership_probability != null ? `<dt>Membership</dt><dd>${fit.membership_probability.toFixed(2)}</dd>` : ""}
      <dt>Runner-up</dt><dd>${fit.runner_up ? escapeHtml(fit.runner_up.folder_path) : "—"}</dd>
      ${fit.llm_verified != null ? `<dt>Model check</dt><dd>${fit.llm_verified ? "agrees" : "<span class='flag'>disagrees</span>"}</dd>` : ""}` : ""}
      <dt>DOI</dt><dd>${escapeHtml(doc.doi || "—")}</dd>
      <dt>Pages</dt><dd>${doc.extraction.pages} · ${doc.extraction.title_source} title</dd>
    </dl>
    ${fit?.rationale ? `<p class="small muted"><em>${escapeHtml(fit.rationale)}</em></p>` : ""}
    ${folder?.description ? `<p class="small muted"><strong>Folder:</strong> ${escapeHtml(folder.description)}</p>` : ""}
    ${doc.abstract_excerpt ? `<p class="abstract">${escapeHtml(doc.abstract_excerpt)}…</p>` : ""}
    <div class="terms">${(doc.top_terms || []).map((t) => `<span>${escapeHtml(t)}</span>`).join("")}</div>
    ${doc.near_duplicates?.length ? `<p class="small flag">Near-duplicate of: ${escapeHtml(doc.near_duplicates.join(", "))}</p>` : ""}
    <p style="margin-top:16px"><a class="btn" href="/api/doc/${encodeURIComponent(doc.id)}/pdf" target="_blank" rel="noopener">Open PDF</a></p>`;
  $("#detail").hidden = false;
}

/* --------------------------------------------------------------- render --- */

function render() {
  drawTiles();
  drawTree();
  drawScatter();
  drawSunburst();
  drawHistogram();
  drawTable();
}

function truncate(text, n) {
  text = String(text ?? "");
  return text.length > n ? text.slice(0, n - 1) + "…" : text;
}

function escapeHtml(text) {
  return String(text ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* ---------------------------------------------------------------- wiring -- */

function wireFilters() {
  // Scatter-only: grouping changes the plot's colours and labels, nothing else.
  $("#level-select").onchange = (e) => { state.level = Number(e.target.value); drawScatter(); };
  $("#search").oninput = (e) => { state.search = e.target.value; render(); };
  $("#colour-mode").onchange = (e) => { state.colourMode = e.target.value; drawScatter(); };
  $("#only-flagged").onchange = (e) => { state.onlyFlagged = e.target.checked; render(); };

  for (const input of document.querySelectorAll(".chips input")) {
    input.onchange = () => {
      input.checked ? state.bands.add(input.value) : state.bands.delete(input.value);
      render();
    };
  }

  $("#clear-filters").onclick = () => {
    state.selected = null;
    state.level = 0;
    state.search = "";
    state.onlyFlagged = false;
    state.bands = new Set(["high", "medium", "low"]);
    $("#search").value = "";
    $("#only-flagged").checked = false;
    for (const input of document.querySelectorAll(".chips input")) input.checked = true;
    render();
  };

  for (const th of document.querySelectorAll("th.sortable")) {
    th.onclick = () => {
      const key = th.dataset.sort;
      state.sort = { key, dir: state.sort.key === key ? -state.sort.dir : 1 };
      drawTable();
    };
  }

  $("#detail-close").onclick = () => { $("#detail").hidden = true; };
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") $("#detail").hidden = true;
  });
}

function wireTheme() {
  const stored = localStorage.getItem("littax-theme");
  if (stored) document.documentElement.dataset.theme = stored;
  $("#theme-toggle").onclick = () => {
    const dark = document.documentElement.dataset.theme === "dark"
      || (!document.documentElement.dataset.theme
          && matchMedia("(prefers-color-scheme: dark)").matches);
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("littax-theme", next);
    if (state.taxonomy) render();  // Plotly colours are baked in, so redraw.
  };
}

/* ------------------------------------------------------------ run dialog -- */

/** The wizard's own state, separate from the viewer's. */
const setup = {
  config: null,      // /api/config
  providers: [],     // /api/providers
  provider: null,    // the selected provider descriptor
  hasTaxonomy: false, // does the selected folder already hold one?
};

function wireRunDialog() {
  const dialog = $("#run-dialog");

  function showStep(id) {
    for (const section of document.querySelectorAll("#run-dialog .step")) {
      section.hidden = section.id !== id;
    }
  }

  const open = async () => {
    showStep("step-folder");
    if (!dialog.open) dialog.showModal();
    const [config, providers] = await Promise.all([
      fetch("/api/config").then((r) => r.json()),
      fetch("/api/providers").then((r) => r.json()).catch(() => ({ providers: [] })),
    ]);
    setup.config = config;
    setup.providers = providers.providers || [];
    $("#env-path").textContent = `— writes to ${config.env_file}`;
    await browse(config.project_source || config.source_root);
  };
  $("#open-run").onclick = open;
  $("#empty-run").onclick = open;

  /* ---- step 1: folder ---- */

  async function browse(path) {
    try {
      const data = await fetch(`/api/browse?path=${encodeURIComponent(path)}`).then((r) => r.json());
      $("#path-input").value = data.path;
      setup.hasTaxonomy = data.has_taxonomy;

      const count = `${data.pdf_count} PDF${data.pdf_count === 1 ? "" : "s"} directly in this folder`;
      $("#path-info").textContent = data.has_taxonomy
        ? `${count} · already analysed`
        : count;

      // An analysed folder offers its result first; re-running is the
      // secondary action, since it costs minutes and possibly money.
      $("#folder-next").textContent = data.has_taxonomy ? "Open taxonomy" : "Continue →";
      $("#reanalyse").hidden = !data.has_taxonomy;

      const items = [];
      if (data.parent) items.push(`<li data-path="${escapeHtml(data.parent)}">↑ ..</li>`);
      for (const dir of data.directories) {
        const tag = dir.has_taxonomy ? `<span class="tag">✓ analysed</span>` : "";
        items.push(`<li data-path="${escapeHtml(dir.path)}">📁 ${escapeHtml(dir.name)}${tag}</li>`);
      }
      $("#dir-list").innerHTML = items.join("");
      for (const li of document.querySelectorAll("#dir-list li")) {
        li.onclick = () => browse(li.dataset.path);
      }
    } catch {
      $("#path-info").textContent = "Cannot read that folder.";
    }
  }

  $("#path-go").onclick = () => browse($("#path-input").value);

  $("#folder-next").onclick = async () => {
    if (!setup.hasTaxonomy) return toMode();
    const response = await fetch("/api/open", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source: $("#path-input").value }),
    });
    if (response.ok) {
      dialog.close();
      load();
    } else {
      $("#path-info").textContent = "Could not open that taxonomy.";
    }
  };
  $("#reanalyse").onclick = () => toMode();

  /* ---- step 2: mode ---- */

  function toMode() {
    const blocked = setup.config?.missing || [];
    $("#mode-source").textContent = $("#path-input").value;

    // No lesser path exists, so a missing dependency stops here rather than
    // starting a run that cannot finish.
    $("#mode-blocked").hidden = blocked.length === 0;
    $("#mode-blocked").textContent = blocked.length
      ? `Cannot run yet — install first: ${blocked.join("; ")}`
      : "";

    // AI-assisted needs *a* usable provider, not the configured one. Ollama
    // needs no SDK, so it keeps this path open on an install with neither
    // cloud SDK present - the daemon check happens at the provider step.
    const usable = setup.providers.filter((p) => p.sdk_installed);
    $("#mode-ai").disabled = usable.length === 0 || blocked.length > 0;
    $("#mode-ai-blocked").hidden = usable.length > 0;
    $("#mode-ai-blocked").textContent = usable.length
      ? ""
      : `Install first: ${(setup.config?.missing_for_llm || []).join("; ")}`;
    if ($("#mode-ai").disabled) $("#mode-standard").checked = true;

    syncModeButton();
    showStep("step-mode");
  }

  function syncModeButton() {
    const ai = $("#mode-ai").checked;
    $("#mode-next").textContent = ai ? "Continue →" : "Start analysis";
    $("#mode-next").disabled = (setup.config?.missing || []).length > 0;
  }
  $("#mode-standard").onchange = syncModeButton;
  $("#mode-ai").onchange = syncModeButton;
  $("#mode-back").onclick = () => showStep("step-folder");
  $("#mode-next").onclick = () => ($("#mode-ai").checked ? toProvider() : startRun(false));

  /* ---- step 3: provider ---- */

  function toProvider() {
    const usable = setup.providers.filter((p) => p.sdk_installed);
    const chosen = setup.provider || usable.find((p) => p.env_ready) || usable[0];
    $("#provider-cards").innerHTML = setup.providers
      .map(
        (p) => `<label class="card" for="prov-${p.id}">
          <input type="radio" name="provider" id="prov-${p.id}" value="${p.id}"
            ${p.id === chosen?.id ? "checked" : ""} ${p.sdk_installed ? "" : "disabled"}>
          <strong>${escapeHtml(p.name)}</strong>
          <span class="muted small">${
            p.sdk_installed
              ? p.env_ready
                ? "Credentials found on this machine"
                : "Needs an endpoint and key"
              : "SDK not installed — uv sync --extra llm"
          }</span>
        </label>`
      )
      .join("");
    for (const input of document.querySelectorAll('#provider-cards input')) {
      input.onchange = () => selectProvider(input.value);
    }
    selectProvider(chosen?.id);
    showStep("step-llm");
  }

  function selectProvider(id) {
    const provider = setup.providers.find((p) => p.id === id);
    setup.provider = provider;
    if (!provider) return;

    $("#provider-fields").innerHTML = provider.fields
      .map(
        (f) => `<label class="field">
          <span>${escapeHtml(f.label)}</span>
          <input id="cred-${f.id}" type="${f.secret ? "password" : "text"}"
            spellcheck="false" placeholder="${escapeHtml(f.placeholder || "")}"
            autocomplete="off">
        </label>`
      )
      .join("");
    $("#llm-model").value = provider.default_model;
    $("#deployment-field").hidden = !provider.needs_deployment;
    if (provider.needs_deployment) $("#llm-deployment").value = provider.default_model;

    // A local provider is addressed rather than authenticated: it shows an
    // address, and there is no secret to remember.
    $("#base-url-field").hidden = !provider.base_url;
    if (provider.base_url) {
      $("#base-url-label").textContent = provider.base_url.label;
      $("#llm-base-url").value = provider.base_url.default;
      $("#llm-base-url").placeholder = provider.base_url.placeholder;
    }
    $("#remember-field").hidden = Boolean(provider.local);
    $("#ollama-install").hidden = true;

    $("#llm-test-result").textContent =
      provider.local
        ? "Runs on this machine. Test the connection to list installed models."
        : provider.env_ready
          ? "Credentials already on this machine — leave blank to use them."
          : "";
    $("#llm-test-result").className = "small muted";
    for (const input of document.querySelectorAll("#provider-fields input")) {
      input.oninput = syncStartButton;
    }
    // Local providers auto-check, so the install panel appears without the
    // user having to guess that "Test connection" is the next thing to press.
    if (provider.local) testConnection();
    syncStartButton();
  }

  /** Install guidance for a local provider whose daemon is not answering. */
  function showInstallPanel(result) {
    const provider = setup.provider;
    const install = provider?.install || {};
    const platform = /Win/i.test(navigator.platform || "")
      ? "windows"
      : /Mac/i.test(navigator.platform || "")
        ? "macos"
        : "linux";
    const model = $("#llm-model").value.trim() || provider?.default_model;

    const steps =
      result.reason_code === "no_model"
        ? [`Pull the model: <code>ollama pull ${escapeHtml(model)}</code>`]
        : [
            `Install it: <code>${escapeHtml(install[platform] || install.url || "")}</code>`,
            `Pull a model: <code>ollama pull ${escapeHtml(model)}</code>`,
            "It starts automatically once installed.",
          ];

    $("#install-title").textContent =
      result.reason_code === "no_model"
        ? "That model is not installed yet"
        : "Ollama is not running";
    $("#install-steps").innerHTML = steps.map((s) => `<li>${s}</li>`).join("");
    $("#ollama-install").hidden = false;
  }

  /** Credentials as the server expects them; blank means "use the environment". */
  function credentials() {
    const values = { api_key: "", endpoint: "", resource: "" };
    for (const field of setup.provider?.fields || []) {
      values[field.id] = $(`#cred-${field.id}`)?.value.trim() || "";
    }
    return values;
  }

  function llmSettings() {
    return {
      provider: setup.provider?.id,
      model: $("#llm-model").value.trim() || null,
      deployment: setup.provider?.needs_deployment
        ? $("#llm-deployment").value.trim() || null
        : null,
      base_url: setup.provider?.base_url ? $("#llm-base-url").value.trim() || null : null,
      credentials: credentials(),
    };
  }

  function syncStartButton() {
    const values = credentials();
    const filled = (setup.provider?.fields || []).every((f) => values[f.id]);
    // Blank fields are fine when the environment already holds the pair, and a
    // provider with no fields at all has nothing to wait for.
    $("#start-run").disabled = !filled && !setup.provider?.env_ready;
  }

  $("#llm-back").onclick = () => showStep("step-mode");

  async function testConnection() {
    const result = $("#llm-test-result");
    result.className = "small muted";
    result.textContent = "Testing…";
    const response = await fetch("/api/llm/test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(llmSettings()),
    }).then((r) => r.json()).catch(() => ({ ok: false, reason: "Could not reach the server." }));

    // Azure calls them deployments, Ollama calls them models; the server sends
    // whichever the provider has under one name.
    if (response.deployments?.length) {
      const options = response.deployments
        .map((d) => `<option value="${escapeHtml(d)}"></option>`)
        .join("");
      $("#deployment-options").innerHTML = options;
      $("#model-options").innerHTML = options;
      // A local model must actually be pulled, so prefer one that is.
      if (setup.provider?.local && !response.deployments.includes($("#llm-model").value)) {
        $("#llm-model").value = response.deployments[0];
      }
    }

    $("#ollama-install").hidden = true;
    if (!response.ok) {
      result.className = "small error";
      result.textContent = response.reason || "Failed.";
      if (response.reason_code) showInstallPanel(response);
    } else if (!response.matches) {
      // Worse than a failure: the run would succeed and bill at another rate.
      result.className = "small error";
      result.textContent = `Requested ${response.requested} but ${response.served} replied.`;
    } else {
      result.className = "small ok";
      result.textContent = `✓ ${response.served} answered`;
    }
    syncStartButton();
  }

  $("#llm-test").onclick = testConnection;
  $("#install-retry").onclick = testConnection;

  $("#start-run").onclick = () => startRun(true);

  async function startRun(aiAssisted) {
    const body = { source: $("#path-input").value, ai_assisted: aiAssisted };
    if (aiAssisted) {
      body.llm = llmSettings();
      body.remember = $("#llm-remember").checked;
    }
    dialog.close();
    $("#progress").hidden = false;
    $("#progress-stage").textContent = "Starting";
    $("#progress-detail").textContent = "";
    $("#progress-error").textContent = "";
    $("#progress-dismiss").hidden = true;
    showProgress(0, 1);

    const response = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      $("#progress-stage").textContent = "Cannot start";
      $("#progress-error").textContent = payload.detail || "Could not start the run.";
      $("#progress-dismiss").hidden = false;
    }
  }

  $("#progress-dismiss").onclick = async () => {
    $("#progress").hidden = true;
    await fetch("/api/dismiss", { method: "POST" }).catch(() => {});
  };
}

/* The stage names the pipeline reports are internal; these are what they mean
 * to someone watching. Anything unmapped falls back to the raw name. */
const STAGE_NAMES = {
  scan: "Finding PDFs",
  ingest: "Reading PDFs",
  embed: "Embedding documents",
  cluster: "Finding clusters",
  score: "Placing documents",
  terms: "Extracting terms",
  layout: "Projecting to 2-D",
  label: "Naming folders",
  coherence: "Reviewing the tree",
  assemble: "Assembling",
  verify: "Checking uncertain documents",
  starting: "Starting",
};

/** Draw the bar, or sweep it when the stage cannot report a fraction.
 *
 *  total <= 1 means a single opaque call: score, layout and the like know only
 *  "not started" and "done". Showing 0% for one of those is what made a healthy
 *  run look hung, so they sweep instead. */
function showProgress(done, total) {
  const measurable = total > 1;
  $("#progress-track").classList.toggle("indeterminate", !measurable);
  if (measurable) {
    const pct = Math.round((done / total) * 100);
    $("#progress-fill").style.width = `${pct}%`;
    $("#progress-pct").textContent = `${pct}%`;
  } else {
    $("#progress-fill").style.width = "";
    $("#progress-pct").textContent = "";
  }
}

function wireProgress() {
  const source = new EventSource("/api/progress");
  let lastVersion = null;

  source.onmessage = (event) => {
    const p = JSON.parse(event.data);
    if (lastVersion === null) lastVersion = p.version;

    if (p.status === "running") {
      $("#progress").hidden = false;
      $("#progress-dismiss").hidden = true;
      $("#progress-stage").textContent = STAGE_NAMES[p.stage] || p.stage || "Working…";
      $("#progress-detail").textContent = p.detail || "";
      showProgress(p.done, p.total);
    } else if (p.status === "error") {
      $("#progress").hidden = false;
      $("#progress-stage").textContent = "Failed";
      $("#progress-error").textContent = p.error || "";
      $("#progress-dismiss").hidden = false;
    } else if (p.status === "done" && p.version !== lastVersion) {
      lastVersion = p.version;
      $("#progress").hidden = true;
      load();
    }
  };
}

/* ----------------------------------------------------------------- boot --- */

async function load(openPickerIfEmpty = false) {
  const response = await fetch("/api/state");
  if (!response.ok) {
    $("#empty").hidden = false;
    $("#app").hidden = true;
    const config = await fetch("/api/config").then((r) => r.json()).catch(() => null);
    if (config) $("#empty-hint").textContent = `Configured source: ${config.source_root}`;
    // Nothing to look at yet, so go straight to choosing a folder rather than
    // making the first click a formality.
    if (openPickerIfEmpty) $("#empty-run").click();
    return;
  }
  indexTaxonomy(await response.json());

  // "Leaf" is level 0 - the folder each document is actually assigned to, which
  // is a genuinely different grouping from level 1, not a synonym for "all".
  const levels = state.taxonomy.stats.levels;
  $("#level-select").innerHTML =
    Array.from({ length: levels }, (_, i) => `<option value="${i + 1}">Level ${i + 1}</option>`).join("") +
    `<option value="0">Leaf (as assigned)</option>`;
  // Level 1 is the readable default: the deeper the grouping, the more folders
  // collapse into "Other".
  if (state.level === 0) state.level = 1;
  $("#level-select").value = String(state.level);
  $("#colour-mode").value = state.colourMode;
  $("#source-root").textContent = state.taxonomy.source_root;

  $("#empty").hidden = true;
  $("#app").hidden = false;
  render();
}

/** Publish the sticky header's real height so the tree panel can sit under it
 *  without either hard-coding the other's size. */
function syncTopbarHeight() {
  const topbar = document.querySelector(".topbar");
  if (!topbar) return;
  document.documentElement.style.setProperty("--topbar-h", `${topbar.offsetHeight}px`);
}

wireFilters();
wireTheme();
wireRunDialog();
wireProgress();
syncTopbarHeight();
addEventListener("resize", syncTopbarHeight);
load(true);
