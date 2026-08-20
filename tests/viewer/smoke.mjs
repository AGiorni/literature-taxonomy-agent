/* Headless smoke test for the viewer.
 *
 * Stubs just enough DOM and Plotly to run the *real* app.js against a real
 * taxonomy.json, then drives every interaction that could throw: selecting each
 * folder, every level, both colour modes, every sort key, the empty-filter edge
 * case, and the detail panel. Catches the runtime errors a syntax check cannot.
 *
 *   node tests/viewer/smoke.mjs [path/to/taxonomy.json]
 */
import { readFileSync } from "node:fs";
import vm from "node:vm";

const taxonomyPath = process.argv[2] ?? "out/taxonomy.json";
const taxonomy = JSON.parse(readFileSync(taxonomyPath, "utf8"));
const source = readFileSync("src/littax/server/static/app.js", "utf8");

const plotCalls = [];
const errors = [];

function makeEl(tag = "div") {
  const el = {
    tagName: tag, innerHTML: "", textContent: "", value: "", hidden: false,
    dataset: {}, style: { setProperty() {} }, title: "", checked: true,
    children: [], offsetHeight: 130, className: "",
    classList: { contains: (c) => String(el.className).includes(c), add() {}, remove() {} },
    setAttribute() {}, getAttribute: () => null, appendChild(c) { el.children.push(c); },
    removeAllListeners() {}, on(evt, fn) { el._handlers = el._handlers || {}; el._handlers[evt] = fn; },
    addEventListener() {}, focus() {}, showModal() {}, close() {},
  };
  return el;
}

const registry = new Map();
const doc = {
  documentElement: makeEl("html"),
  querySelector(sel) {
    if (!registry.has(sel)) registry.set(sel, makeEl());
    return registry.get(sel);
  },
  querySelectorAll: () => [],
  createElement: (tag) => makeEl(tag),
  addEventListener() {},
};

const sandbox = {
  console,
  document: doc,
  getComputedStyle: () => ({ getPropertyValue: () => "#2a78d6" }),
  Plotly: {
    react: (id, traces, layout) => plotCalls.push({ id, traces, layout }),
    purge: (id) => plotCalls.push({ id, purged: true }),
  },
  localStorage: { getItem: () => null, setItem() {} },
  matchMedia: () => ({ matches: false }),
  EventSource: class { constructor() { this.onmessage = null; } },
  addEventListener() {},
  encodeURIComponent,
  fetch: async (url) => {
    if (url === "/api/state") return { ok: true, json: async () => taxonomy };
    if (url === "/api/config") return { ok: true, json: async () => ({ source_root: "raw" }) };
    return { ok: false, json: async () => ({}) };
  },
};
sandbox.window = sandbox;
sandbox.globalThis = sandbox;

// `const`/`let` at script top level are lexical bindings, not properties of the
// global object, so export the ones the harness drives.
const instrumented = source + `
;globalThis.state = state;
globalThis.visibleDocuments = visibleDocuments;
globalThis.selectFolder = selectFolder;
globalThis.render = render;
globalThis.drawScatter = drawScatter;
globalThis.drawTable = drawTable;
globalThis.showDetail = showDetail;
`;

vm.createContext(sandbox);
try {
  vm.runInContext(instrumented, sandbox, { filename: "app.js" });
} catch (e) {
  errors.push("load: " + e.stack);
}

// Give the async load() a chance to finish, then drive interactions.
await new Promise((r) => setTimeout(r, 300));

const st = sandbox.state;
if (!st || !st.taxonomy) errors.push("taxonomy never loaded into state");

function drive(name, fn) {
  try { fn(); } catch (e) { errors.push(`${name}: ${e.message}`); }
}

if (st && st.taxonomy) {
  const visible = sandbox.visibleDocuments();
  console.log(`visibleDocuments (unfiltered): ${visible.length}`);
  if (visible.length === 0) errors.push("no documents visible with default filters");

  // Select every folder in turn — the single most likely source of runtime errors.
  for (const folder of st.taxonomy.folders) {
    drive(`selectFolder(${folder.id})`, () => sandbox.selectFolder(folder.id));
  }
  drive("selectFolder(null)", () => sandbox.selectFolder(null));

  // Each level, colour mode, band filter and sort key.
  for (const level of [0, 1, 2, 3]) { st.level = level; drive(`level=${level}`, sandbox.render); }
  st.level = 0;
  for (const mode of ["confidence", "selection"]) {
    st.colourMode = mode;
    drive(`colour=${mode}`, sandbox.drawScatter);
  }
  // Selection mode with an active selection is the branch that reads scopeIds.
  sandbox.selectFolder(st.taxonomy.folders[0].id);
  st.colourMode = "selection";
  drive("colour=selection+scope", sandbox.drawScatter);
  sandbox.selectFolder(null);

  for (const key of ["confidence", "title", "year", "folder", "runner"]) {
    st.sort = { key, dir: 1 };
    drive(`sort=${key}`, sandbox.drawTable);
  }
  st.onlyFlagged = true; drive("onlyFlagged", sandbox.render); st.onlyFlagged = false;
  st.search = "motor unit"; drive("search", sandbox.render); st.search = "";
  st.bands = new Set(["low"]); drive("bands=low", sandbox.render);
  st.bands = new Set(); drive("bands=none (empty state)", sandbox.render);
  st.bands = new Set(["high", "medium", "low"]);

  // Detail panel for a sample of documents, including an unassigned one.
  for (const doc of st.taxonomy.documents.slice(0, 40)) {
    drive(`showDetail(${doc.id})`, () => sandbox.showDetail(doc.id));
  }
  const dup = st.taxonomy.documents.find((d) => d.duplicate_of);
  if (dup) drive(`showDetail(duplicate)`, () => sandbox.showDetail(dup.id));
}

console.log(`Plotly draw calls: ${plotCalls.length}`);
const ids = [...new Set(plotCalls.map((c) => c.id))];
console.log(`charts drawn: ${ids.join(", ")}`);

// Sanity-check the scatter payload actually carries data.
const scatter = [...plotCalls].reverse().find((c) => c.id === "scatter" && c.traces);
if (scatter) {
  console.log(`scatter points: ${scatter.traces[0].x.length}, centroid labels: ${scatter.traces[1]?.text?.length ?? 0}`);
  if (!scatter.traces[0].x.length) errors.push("scatter has no points");
} else errors.push("scatter never drawn");

const sun = [...plotCalls].reverse().find((c) => c.id === "sunburst" && c.traces);
if (sun) console.log(`sunburst segments: ${sun.traces[0].ids.length}`);
else errors.push("sunburst never drawn");

if (errors.length) {
  console.error(`\n${errors.length} ERROR(S):`);
  for (const e of errors) console.error("  - " + e);
  process.exit(1);
}
console.log("\nviewer smoke test: OK");
