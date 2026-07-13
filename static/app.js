/* PolicyLens front-end.
 *
 * Jobs: manage a multi-document corpus (load/upload/remove), ask questions, and
 * wire the clickable [n] citations in the answer to the matching source card
 * (the NotebookLM move).
 */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const els = {
    llmBadge: $("llm-badge"),
    dropzone: $("dropzone"),
    browseBtn: $("browse-btn"),
    sampleBtn: $("sample-btn"),
    fileInput: $("file-input"),
    docStatus: $("doc-status"),
    docTray: $("doc-tray"),
    docList: $("doc-list"),
    clearBtn: $("clear-btn"),
    askForm: $("ask-form"),
    question: $("question"),
    askBtn: $("ask-btn"),
    answerPanel: $("answer-panel"),
    sourcesList: $("sources-list"),
  };

  let hasDocument = false;

  // --- helpers -------------------------------------------------------------

  const escapeHtml = (s) =>
    s.replace(
      /[&<>"']/g,
      (c) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;",
        })[c],
    );

  async function postJSON(url, body) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || "Request failed");
    return data;
  }

  function setDocStatus(text) {
    els.docStatus.textContent = text;
  }

  // Show a spinner + message while a document is being processed, and disable
  // the load controls so a second action can't race the first.
  function setLoading(message) {
    els.docStatus.innerHTML =
      '<span class="spinner" aria-hidden="true"></span>' + escapeHtml(message);
    [els.sampleBtn, els.browseBtn, els.clearBtn].forEach((b) => {
      if (b) b.disabled = true;
    });
  }

  function clearLoading() {
    [els.sampleBtn, els.browseBtn, els.clearBtn].forEach((b) => {
      if (b) b.disabled = false;
    });
  }

  // --- corpus rendering ----------------------------------------------------

  function renderCorpus(data) {
    const docs = data.documents || [];
    hasDocument = docs.length > 0;
    els.docTray.classList.toggle("hidden", !hasDocument);
    els.docTray.classList.toggle("flex", hasDocument);

    els.docList.innerHTML = docs
      .map(
        (d) => `
        <li class="inline-flex items-center gap-1.5 rounded-full bg-brand-light px-3 py-1 text-xs font-medium text-brand-dark">
          <span class="max-w-[16rem] truncate">${escapeHtml(d.name)}</span>
          <span class="text-brand/70">· ${d.chunks}</span>
          <button type="button" class="doc-remove ml-0.5 rounded-full px-1 leading-none hover:bg-brand hover:text-white focus:outline-none focus:ring-1 focus:ring-brand" data-doc-id="${escapeHtml(
            d.doc_id,
          )}" aria-label="Remove ${escapeHtml(d.name)}">×</button>
        </li>`,
      )
      .join("");

    if (hasDocument) {
      const total = docs.reduce((n, d) => n + d.chunks, 0);
      const mode = data.semantic ? "semantic + keyword" : "keyword";
      setDocStatus(
        `${docs.length} document${docs.length > 1 ? "s" : ""} · ${total} passages · ${mode} search`,
      );
      els.question.focus();
    } else {
      setDocStatus("");
    }
  }

  async function loadSample() {
    setLoading("Loading sample…");
    try {
      renderCorpus(await postJSON("/api/sample"));
    } catch (err) {
      setDocStatus(err.message);
    } finally {
      clearLoading();
    }
  }

  async function uploadFile(file) {
    setLoading(`Reading & indexing “${file.name}”…`);
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch("/api/ingest", { method: "POST", body: form });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error || "Upload failed");
      renderCorpus(data);
    } catch (err) {
      setDocStatus(err.message);
    } finally {
      clearLoading();
    }
  }

  async function removeDoc(docId) {
    try {
      renderCorpus(await postJSON("/api/remove", { doc_id: docId }));
    } catch (err) {
      setDocStatus(err.message);
    }
  }

  async function clearAll() {
    try {
      await postJSON("/api/reset");
      renderCorpus({ documents: [] });
      els.answerPanel.innerHTML =
        '<p class="text-sm text-slate-600">Your grounded answer will appear here.</p>';
      els.sourcesList.innerHTML = "";
    } catch (err) {
      setDocStatus(err.message);
    }
  }

  // --- asking --------------------------------------------------------------

  function renderSkeleton() {
    els.answerPanel.innerHTML = `
      <div class="space-y-2">
        <div class="skeleton-line w-11/12"></div>
        <div class="skeleton-line w-full"></div>
        <div class="skeleton-line w-4/6"></div>
      </div>`;
    els.sourcesList.innerHTML = "";
  }

  // Replace [n] markers in the answer with interactive citation buttons.
  function linkifyCitations(answer) {
    return escapeHtml(answer).replace(
      /\[(\d+)\]/g,
      (_, n) =>
        `<button type="button" class="citation" data-rank="${n}" ` +
        `aria-label="Show source ${n}">${n}</button>`,
    );
  }

  // Wrap the relevant sentence within a passage in a <mark> so the reader sees
  // exactly which part the answer came from.
  function markHighlight(text, highlight) {
    const escaped = escapeHtml(text);
    if (!highlight) return escaped;
    const escHi = escapeHtml(highlight);
    const at = escaped.indexOf(escHi);
    if (at === -1) return escaped;
    return (
      escaped.slice(0, at) +
      '<mark class="src-hl">' +
      escHi +
      "</mark>" +
      escaped.slice(at + escHi.length)
    );
  }

  function renderSources(sources) {
    if (!sources.length) {
      els.sourcesList.innerHTML =
        '<li class="text-slate-600">No matching passages.</li>';
      return;
    }
    els.sourcesList.innerHTML = sources
      .map((s) => {
        const loc = s.page ? ` · p.${s.page}` : "";
        return `
        <li id="source-${s.rank}" class="source-card">
          <div class="mb-1 flex items-center gap-2">
            <span class="rank">${s.rank}</span>
            <span class="truncate text-xs font-medium text-slate-500">${escapeHtml(
              s.doc_name,
            )}${loc}</span>
          </div>
          <p class="text-slate-700">${markHighlight(s.text, s.highlight)}</p>
        </li>`;
      })
      .join("");
  }

  function focusSource(rank) {
    document
      .querySelectorAll(".source-card.active")
      .forEach((el) => el.classList.remove("active"));
    const card = $(`source-${rank}`);
    if (!card) return;
    card.classList.add("active", "flash");
    card.scrollIntoView({ behavior: "smooth", block: "nearest" });
    card.addEventListener("animationend", () => card.classList.remove("flash"), {
      once: true,
    });
  }

  function renderAnswer(data) {
    const mode = data.grounded
      ? `<span class="text-brand">grounded answer</span>`
      : '<span class="text-slate-600">offline extract</span>';
    const sem = data.semantic
      ? ' · <span class="text-brand">semantic</span>'
      : "";
    const body = data.found
      ? linkifyCitations(data.answer)
      : `<span class="text-slate-600">${escapeHtml(data.answer)}</span>`;
    els.answerPanel.innerHTML = `
      <p class="text-[0.95rem] leading-relaxed">${body}</p>
      <p class="mt-3 text-xs text-slate-600">${mode}${sem}${
        data.model ? " · " + escapeHtml(data.model) : ""
      }</p>`;
    renderSources(data.sources || []);
  }

  async function ask(question) {
    renderSkeleton();
    const label = els.askBtn.textContent;
    els.askBtn.disabled = true;
    els.askBtn.innerHTML =
      '<span class="spinner spinner-light" aria-hidden="true"></span>Asking…';
    try {
      renderAnswer(await postJSON("/api/ask", { question }));
    } catch (err) {
      els.answerPanel.innerHTML = `<p class="text-sm text-red-600">${escapeHtml(
        err.message,
      )}</p>`;
    } finally {
      els.askBtn.disabled = false;
      els.askBtn.textContent = label;
    }
  }

  // --- status --------------------------------------------------------------

  async function refreshStatus() {
    try {
      const data = await (await fetch("/api/status")).json();
      if (data.llm_configured) {
        els.llmBadge.textContent = "NVIDIA NIM · live";
        els.llmBadge.className =
          "rounded-full bg-brand-light px-3 py-1 text-xs font-medium text-brand-dark";
      } else {
        els.llmBadge.textContent = "offline mode";
        els.llmBadge.className =
          "rounded-full bg-amber-100 px-3 py-1 text-xs font-medium text-amber-700";
      }
      renderCorpus(data); // restore any docs already in this session
    } catch {
      els.llmBadge.textContent = "offline mode";
    }
  }

  // --- wiring --------------------------------------------------------------

  els.sampleBtn.addEventListener("click", loadSample);
  els.browseBtn.addEventListener("click", () => els.fileInput.click());
  els.clearBtn.addEventListener("click", clearAll);
  els.fileInput.addEventListener("change", (e) => {
    if (e.target.files[0]) uploadFile(e.target.files[0]);
    e.target.value = "";
  });

  els.docList.addEventListener("click", (e) => {
    const btn = e.target.closest(".doc-remove");
    if (btn) removeDoc(btn.dataset.docId);
  });

  ["dragover", "dragenter"].forEach((evt) =>
    els.dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      els.dropzone.classList.add("border-brand", "bg-brand-light");
    }),
  );
  ["dragleave", "drop"].forEach((evt) =>
    els.dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      els.dropzone.classList.remove("border-brand", "bg-brand-light");
    }),
  );
  els.dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) uploadFile(file);
  });

  els.askForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const q = els.question.value.trim();
    if (!q) return;
    if (!hasDocument) {
      setDocStatus("Load or upload a document first.");
      return;
    }
    ask(q);
  });

  els.question.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      els.askForm.requestSubmit();
    }
  });

  // Event delegation: clicking any [n] citation reveals its source.
  els.answerPanel.addEventListener("click", (e) => {
    const cite = e.target.closest(".citation");
    if (cite) focusSource(cite.dataset.rank);
  });

  refreshStatus();
})();
