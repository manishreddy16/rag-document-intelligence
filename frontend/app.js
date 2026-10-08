// Vanilla JS UI for the RAG API. Endpoints unchanged: GET /health, POST /upload, POST /ask.
// All server text is inserted with textContent / createTextNode (no HTML injection).
const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; };
const fileInput = $("file"), dropzone = $("dropzone"), question = $("question"), askBtn = $("ask-btn");
const EXAMPLES = ["Summarize the key points of this document.", "What are the most important dates or numbers mentioned?", "Who or what is this document about?"];
let busy = false, docLoaded = false, lastQuestion = "";

/* ---------- helpers ---------- */
async function callApi(url, options) {
  let res;
  try { res = await fetch(url, options); }
  catch (e) { const err = new Error("Cannot reach the server. Is it running?"); err.code = "network"; throw err; }
  let body = null;
  try { body = await res.json(); } catch (e) { /* non-JSON error page */ }
  if (!res.ok) {
    const err = new Error(body?.error?.message || `Request failed (HTTP ${res.status}).`);
    err.code = body?.error?.code || "http_" + res.status; err.status = res.status;
    throw err;
  }
  return body;
}
function setSys(state, text) { $("sys-status").dataset.state = state; $("sys-text").textContent = text; }
function notice(kind, text, spinner) {
  const n = $("upload-status");
  n.hidden = !text; n.className = "notice " + kind; n.textContent = "";
  if (spinner) n.appendChild(el("span", "spinner"));
  n.appendChild(document.createTextNode(text || ""));
}
function show(id, on) { $(id).hidden = !on; }
function announce(t) { $("ask-status").textContent = t; }

/* ---------- theme ---------- */
function applyTheme(t) {
  document.documentElement.dataset.theme = t;
  $("theme-btn").setAttribute("aria-label", t === "dark" ? "Switch to light theme" : "Switch to dark theme");
}
let saved = null; try { saved = localStorage.getItem("theme"); } catch (e) {}
applyTheme(saved === "dark" ? "dark" : "light"); // light by default; the toggle remembers a dark choice
$("theme-btn").addEventListener("click", () => {
  const t = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  applyTheme(t); try { localStorage.setItem("theme", t); } catch (e) {}
});

/* ---------- document state ---------- */
function showDoc(d) {
  docLoaded = true;
  $("doc-name").textContent = d.filename; $("doc-name").title = d.filename;
  $("doc-pages").textContent = d.pages; $("doc-chunks").textContent = d.chunks;
  let when = "–"; try { when = new Date(d.indexed_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: false }); } catch (e) {}
  $("doc-time").textContent = when === "Invalid Date" ? "–" : when;
  show("doc-card", true); dropzone.classList.add("compact");
  $("dz-title").textContent = "Drop a PDF to replace"; $("dz-sub").textContent = "The current document will be replaced.";
  question.disabled = false; question.placeholder = "Ask a question about " + d.filename;
  show("empty-nodoc", false); if ($("answer-box").hidden && $("error-card").hidden) show("empty-ready", true);
  renderChips(); syncAsk();
}
function renderChips() {
  const c = $("chips"); c.textContent = "";
  EXAMPLES.forEach((t) => { const b = el("button", "chip", t); b.type = "button"; b.addEventListener("click", () => { question.value = t; updateCounter(); ask(); }); c.appendChild(b); });
  c.hidden = false;
}
function syncAsk() { askBtn.disabled = busy || !docLoaded || !question.value.trim(); askBtn.classList.toggle("loading", busy && docLoaded); }
function updateCounter() { $("counter").textContent = `${question.value.length} / 1000`; syncAsk(); }

async function loadHealth() {
  try {
    const h = await callApi("/health");
    if (h.document_loaded) showDoc(h.document);
    if (!h.llm_configured) setSys("warn", "LLM not configured");
    else setSys("ok", "System ready");
    if (!h.llm_configured) { const m = "Server has no LLM_API_KEY / LLM_MODEL configured yet (see README)."; notice("err", m); }
  } catch (e) { setSys("err", "Server unreachable"); notice("err", e.message); }
}

/* ---------- upload ---------- */
async function upload(file) {
  if (!file) return;
  if (!/\.pdf$/i.test(file.name) && file.type !== "application/pdf") return notice("err", "Only PDF files are supported.");
  const form = new FormData(); form.append("file", file);
  dropzone.classList.add("busy"); $("upload-btn").disabled = true; $("replace-btn").disabled = true;
  notice("busy", "Extracting, chunking, embedding and indexing… the first run downloads the embedding model.", true);
  try {
    const r = await callApi("/upload", { method: "POST", body: form });
    notice("ok", r.message + (r.replaced_previous ? " Replaced the previous document." : ""));
    showDoc(r.document);
    show("answer-box", false); show("error-card", false); show("empty-ready", true);
  } catch (e) { console.error("Upload failed:", e); notice("err", e.message); }
  finally { dropzone.classList.remove("busy"); $("upload-btn").disabled = false; $("replace-btn").disabled = false; fileInput.value = ""; }
}
const pick = () => fileInput.click();
$("upload-btn").addEventListener("click", pick); $("replace-btn").addEventListener("click", pick); $("cta-upload").addEventListener("click", pick);
fileInput.addEventListener("change", () => upload(fileInput.files[0]));
["dragenter", "dragover"].forEach((ev) => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("over"); }));
dropzone.addEventListener("drop", (e) => upload(e.dataTransfer.files[0]));
["dragover", "drop"].forEach((ev) => window.addEventListener(ev, (e) => e.preventDefault())); // don't navigate away on a missed drop

/* ---------- answer rendering ---------- */
function inline(parent, text) { // **bold** and `code`, built with DOM nodes only
  text.split(/(\*\*[^*]+\*\*|`[^`]+`)/).forEach((p) => {
    if (p.startsWith("**") && p.endsWith("**") && p.length > 4) parent.appendChild(el("strong", null, p.slice(2, -2)));
    else if (p.startsWith("`") && p.endsWith("`") && p.length > 2) parent.appendChild(el("code", null, p.slice(1, -1)));
    else if (p) parent.appendChild(document.createTextNode(p));
  });
}
function renderAnswer(text) { // paragraphs, plus bullet/numbered lists even when they follow a line of text
  const box = $("answer"); box.textContent = "";
  const isLi = (l) => /^([-*\u2022]|\d+[.)])\s+/.test(l);
  text.split(/\n\s*\n/).forEach((block) => {
    let para = null, list = null;
    block.split("\n").map((l) => l.trim()).filter(Boolean).forEach((l) => {
      if (isLi(l)) {
        const type = /^\d/.test(l) ? "ol" : "ul";
        if (!list || list.tagName.toLowerCase() !== type) { list = el(type); box.appendChild(list); }
        const li = el("li"); inline(li, l.replace(/^([-*\u2022]|\d+[.)])\s+/, "")); list.appendChild(li); para = null;
      } else {
        list = null;
        if (!para) { para = el("p"); box.appendChild(para); } else para.appendChild(el("br"));
        inline(para, l);
      }
    });
  });
}
function renderSources(sources) {
  const list = $("sources"); list.textContent = "";
  $("src-count").textContent = sources.length;
  sources.forEach((s, i) => {
    const li = el("li", "src");
    const head = el("button", "src-btn"); head.type = "button"; head.setAttribute("aria-expanded", "false");
    const idx = el("span", "src-idx", i + 1);
    const main = el("span", "src-main");
    main.append(el("span", "src-name", s.source), el("span", "src-tags"));
    main.lastChild.append(el("span", "tag", "Page " + s.page_number), el("span", "tag mono", s.chunk_id));
    const score = el("span", "score"); score.title = "Similarity score";
    const bar = el("span", "bar"), fill = el("span", "fill"); fill.style.width = Math.max(4, Math.min(100, s.score * 100)) + "%"; bar.appendChild(fill);
    score.append(bar, el("span", "score-n", s.score.toFixed(3)));
    const chev = el("span", "chev"); chev.setAttribute("aria-hidden", "true");
    head.append(idx, main, score, chev);
    const snip = el("p", "src-snippet", s.snippet + "…");
    head.addEventListener("click", () => { const open = li.classList.toggle("open"); head.setAttribute("aria-expanded", open); });
    li.append(head, snip); list.appendChild(li);
  });
}

/* ---------- ask ---------- */
function friendly(e) {
  const map = {
    llm_unavailable: ["AI service unavailable", "The AI service is temporarily unavailable. Please try again."],
    llm_rate_limited: ["Rate limit reached", "The AI service is receiving too many requests. Wait a moment and try again."],
    llm_not_configured: ["AI service not configured", "The server's LLM key or model is missing or was rejected. Check the server configuration (see README)."],
    network: ["Server unreachable", "Cannot reach the server. Is it running?"],
  };
  const m = map[e.code]; return m ? { title: m[0], text: m[1], retry: e.code !== "llm_not_configured" } : { title: "Something went wrong", text: e.message, retry: true };
}
function setBusy(on) {
  busy = on; syncAsk(); show("loading", on);
  question.readOnly = on; announce(on ? "Searching the document and asking the LLM…" : "");
}
async function ask() {
  const q = question.value.trim();
  if (!q || busy || !docLoaded) return;
  lastQuestion = q;
  show("answer-box", false); show("error-card", false); show("empty-ready", false);
  setBusy(true);
  try {
    const r = await callApi("/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: q }) });
    $("q-echo").textContent = q; renderAnswer(r.answer); renderSources(r.sources);
    const box = $("answer-box"); box.hidden = false; box.classList.remove("enter"); void box.offsetWidth; box.classList.add("enter");
    announce("Answer ready");
  } catch (e) {
    console.error("Ask failed:", e.code, e.status, e.message); // full detail stays in the console
    const f = friendly(e);
    $("err-title").textContent = f.title; $("err-text").textContent = f.text; show("retry-btn", f.retry); show("error-card", true);
  } finally { setBusy(false); }
}
$("ask-form").addEventListener("submit", (e) => { e.preventDefault(); ask(); });
question.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); ask(); } });
question.addEventListener("input", () => { updateCounter(); question.style.height = "auto"; question.style.height = Math.min(question.scrollHeight, 180) + "px"; });
$("retry-btn").addEventListener("click", () => { question.value = lastQuestion; updateCounter(); ask(); });
loadHealth();
