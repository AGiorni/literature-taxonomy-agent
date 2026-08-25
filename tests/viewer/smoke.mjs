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
    addEventListener() {}, focus() {}, showModal() { el.open = true; }, close() { el.open = false; },
    open: false, disabled: false,
    click() { return el.onclick?.(); },
    getBoundingClientRect: () => ({ left: 20, top: 300, right: 260, bottom: 326, width: 240, height: 26 }),
  };
  return el;
}

const registry = new Map();
const doc = {
  documentElement: makeEl("html"),
  body: makeEl("body"),
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
  // Distinct values per token: a single stub colour would hide the difference
  // between categorical colouring and a constant fill.
  getComputedStyle: () => ({
    getPropertyValue: (name) =>
      ({
        "--series-1": "#2a78d6", "--series-2": "#eb6834", "--series-3": "#1baf7a",
        "--seq-1": "#86b6ef", "--seq-2": "#5598e7", "--seq-3": "#2a78d6",
        "--seq-4": "#1c5cab", "--seq-5": "#104281",
        "--muted": "#898781", "--surface-1": "#fcfcfb", "--page": "#f9f9f7",
        "--border": "#e1e0d9", "--grid": "#e1e0d9", "--axis": "#c3c2b7",
        "--text-primary": "#0b0b0b", "--text-secondary": "#52514e",
        "--good": "#0ca30c", "--warning": "#fab219", "--critical": "#d03b3b",
      })[name] ?? "#2a78d6",
  }),
  Plotly: {
    react: (id, traces, layout) => plotCalls.push({ id, traces, layout }),
    purge: (id) => plotCalls.push({ id, purged: true }),
  },
  localStorage: { getItem: () => null, setItem() {} },
  matchMedia: () => ({ matches: false }),
  innerWidth: 1280, innerHeight: 800,
  EventSource: class { constructor() { this.onmessage = null; } },
  addEventListener() {},
  encodeURIComponent,
  fetch: async (url) => {
    if (url === "/api/state") return { ok: true, json: async () => taxonomy };
    if (url === "/api/config") return {
      ok: true,
      json: async () => ({
        source_root: "raw", project_source: "raw", env_file: "/tmp/.env",
        missing: [], missing_for_llm: [], env_keys: [],
      }),
    };
    if (url === "/api/providers") return {
      ok: true,
      json: async () => ({ providers: [
        { id: "foundry", name: "Claude on Microsoft Foundry", default_model: "claude-opus-5",
          needs_deployment: false, sdk_installed: true, env_ready: false,
          fields: [{ id: "resource", label: "Resource", secret: false },
                   { id: "api_key", label: "API key", secret: true }] },
        { id: "azure-openai", name: "GPT on Azure OpenAI", default_model: "gpt-4o",
          needs_deployment: true, sdk_installed: true, env_ready: false,
          fields: [{ id: "endpoint", label: "Endpoint", secret: false },
                   { id: "api_key", label: "API key", secret: true }] },
      ] }),
    };
    if (String(url).startsWith("/api/browse")) return {
      ok: true,
      json: async () => ({
        path: "raw", parent: "/", pdf_count: 12, has_taxonomy: false,
        directories: [{ name: "sub", path: "raw/sub", has_taxonomy: true }],
      }),
    };
    if (url === "/api/llm/test") return {
      ok: true,
      json: async () => ({ ok: true, requested: "gpt-4o", served: "gpt-4o",
                           matches: true, deployments: ["gpt-4o", "gpt-4o-mini"] }),
    };
    if (url === "/api/run" || url === "/api/open" || url === "/api/dismiss") {
      return { ok: true, json: async () => ({ started: true, ok: true }) };
    }
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
  for (const mode of ["folder", "confidence", "selection"]) {
    st.colourMode = mode;
    drive(`colour=${mode}`, sandbox.drawScatter);
  }

  // Folder colouring across every grouping level, including leaf (0).
  st.colourMode = "folder";
  for (const lvl of [0, 1, 2, 3]) {
    st.level = lvl;
    drive(`folder colour @ level=${lvl}`, sandbox.drawScatter);
  }
  st.level = 1;
  drive("folder colour @ level=1", sandbox.drawScatter);

  const scatterCall = [...plotCalls].reverse().find((c) => c.id === "scatter" && c.traces);

  // Categorical means a colour per point, never a continuous colorscale.
  const mk = scatterCall?.traces?.[0]?.marker;
  if (!Array.isArray(mk?.color)) errors.push("folder mode did not colour per point");
  if (mk?.colorscale) errors.push("folder mode still carries a sequential colorscale");
  const distinct = new Set(mk?.color ?? []);
  if (distinct.size < 2) errors.push(`folder mode used ${distinct.size} colour(s)`);
  if (distinct.size > 4) {
    // 3 validated hues + one muted "Other"/Unsorted. Capacity past that comes
    // from marker shape, never from inventing a fourth hue.
    errors.push(`folder mode used ${distinct.size} colours, above the validated cap of 4`);
  }
  const shapes = new Set(mk?.symbol ?? []);
  if (!shapes.size) errors.push("folder mode set no marker symbols");
  if (shapes.size > 3) errors.push(`folder mode used ${shapes.size} symbols, above the 3 defined`);

  // Labels must be annotations (SVG, above the WebGL markers) and unabbreviated.
  const notes = scatterCall?.layout?.annotations;
  if (!notes?.length) errors.push("no centroid annotations drawn");
  else {
    if (notes.some((a) => String(a.text).includes("…"))) {
      errors.push("centroid label was truncated");
    }
    if (notes.some((a) => !a.bgcolor)) {
      errors.push("centroid label has no background, so it will vanish into the dots");
    }
    // Every drawn label must match its folder's real name exactly - the old
    // text trace clipped at 28 characters, which is the bug this replaced.
    const names = new Set(st.taxonomy.folders.map((f) => f.label));
    const clipped = notes.find((a) => !names.has(a.text));
    if (clipped) errors.push(`label is not the folder's full name: ${clipped.text}`);
  }
  const legendHtml = String(registry.get("#scatter-legend")?.innerHTML ?? "");
  if (legendHtml.indexOf("glyph") === -1) errors.push("folder mode drew no legend");
  // Identity is never colour-alone: the legend carries the marker shape too.
  if (!/[●◆■]/.test(legendHtml)) errors.push("legend omits the marker shapes");
  console.log(`folder colouring: ${distinct.size} colours x ${shapes.size} shapes, ${notes?.length ?? 0} labels`);
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

/* ---- folder name tooltip --------------------------------------------------
 * Long folder names truncate in the sidebar, so the tip is the only way to read
 * them. It touches document.body, getBoundingClientRect and the viewport size,
 * none of which the rendering tests above reach.                             */

const treeRows = (registry.get("#tree")?.children ?? []).filter((r) => r.onmouseenter);
if (!treeRows.length) errors.push("no tree row exposes a hover handler");

for (const row of treeRows.slice(0, 6)) {
  drive("hover", () => row.onmouseenter());
  drive("unhover", () => row.onmouseleave());
}

const tip = doc.body.children.find((c) => c.className === "node-tip");
if (!tip) errors.push("hover never created a tooltip");
else {
  drive("hover once more", () => treeRows[0].onmouseenter());
  const label = st?.taxonomy?.folders?.[0]?.label;
  if (label && !String(tip.innerHTML).includes(label.slice(0, 12))) {
    errors.push(`tooltip does not carry the folder name: ${tip.innerHTML}`);
  }
  if (tip.hidden) errors.push("tooltip still hidden while hovering");
  if (!/^\d+px$/.test(String(tip.style.left ?? ""))) {
    errors.push(`tooltip was not positioned: left=${tip.style.left}`);
  }
  drive("leave", () => treeRows[0].onmouseleave());
  if (!tip.hidden) errors.push("tooltip not hidden on mouseleave");
}

console.log(`tooltip: driven over ${Math.min(treeRows.length, 6)} rows`);

/* ---- the run wizard -------------------------------------------------------
 * Handlers hang off the stub elements, so the harness can drive each step the
 * way a click would. Every branch here is code that only ever runs from the
 * dialog, and so is never exercised by the rendering tests above.            */

const el = (sel) => registry.get(sel) ?? doc.querySelector(sel);

async function step(name, fn) {
  try { await fn(); } catch (e) { errors.push(`${name}: ${e.message}`); }
}

await step("open dialog", () => el("#open-run").click());
await new Promise((r) => setTimeout(r, 100));

if (!el("#run-dialog").open) errors.push("dialog never opened");
if (el("#path-input").value !== "raw") errors.push("picker did not land on the source folder");
if (!String(el("#dir-list").innerHTML).includes("analysed")) {
  errors.push("an analysed subfolder was not badged");
}

await step("browse", () => el("#path-go").click());
await new Promise((r) => setTimeout(r, 50));

// Folder has no taxonomy, so Continue advances to the mode step.
await step("folder -> mode", () => el("#folder-next").click());

// The markup ships #mode-standard checked; makeEl defaults everything to true,
// so state the radio pair explicitly rather than trusting the stub.
el("#mode-standard").checked = true;
el("#mode-ai").checked = false;
await step("mode default", () => el("#mode-standard").onchange());
if (el("#step-mode").hidden) errors.push("mode step never shown");
if (el("#mode-next").textContent !== "Start analysis") {
  errors.push(`standard mode should start directly, got ${el("#mode-next").textContent}`);
}

// Switching to AI-assisted turns the button into another step.
el("#mode-ai").checked = true;
el("#mode-standard").checked = false;
await step("mode toggle", () => el("#mode-ai").onchange());
if (el("#mode-next").textContent !== "Continue →") errors.push("AI mode should continue, not start");

await step("mode -> provider", () => el("#mode-next").click());
if (el("#step-llm").hidden) errors.push("provider step never shown");
if (!String(el("#provider-cards").innerHTML).includes("Azure OpenAI")) {
  errors.push("providers not rendered");
}

await step("select azure", () => sandbox.document.querySelector("#prov-azure-openai").onchange?.());
await step("test connection", async () => {
  await el("#llm-test").click();
  await new Promise((r) => setTimeout(r, 50));
});
if (!String(el("#llm-test-result").textContent).includes("answered")) {
  errors.push(`test connection did not report success: ${el("#llm-test-result").textContent}`);
}
if (!String(el("#deployment-options").innerHTML).includes("gpt-4o-mini")) {
  errors.push("deployment list not populated from the probe");
}

await step("start run", async () => {
  el("#start-run").disabled = false;
  await el("#start-run").click();
  await new Promise((r) => setTimeout(r, 50));
});
if (el("#progress").hidden) errors.push("progress overlay not shown after starting");
await step("dismiss", () => el("#progress-dismiss").click());

// Going back must not throw, and an analysed folder offers Open instead.
await step("back to mode", () => el("#llm-back").click());
await step("back to folder", () => el("#mode-back").click());

console.log("run wizard: three steps driven");

console.log(`Plotly draw calls: ${plotCalls.length}`);
const ids = [...new Set(plotCalls.map((c) => c.id))];
console.log(`charts drawn: ${ids.join(", ")}`);

// Sanity-check the scatter payload actually carries data.
const scatter = [...plotCalls].reverse().find((c) => c.id === "scatter" && c.traces);
if (scatter) {
  console.log(`scatter points: ${scatter.traces[0].x.length}, centroid labels: ${scatter.layout?.annotations?.length ?? 0}`);
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
