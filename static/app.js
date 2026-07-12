/* PolicyLens front-end.
 *
 * Three jobs: load a document, ask a question, and wire the clickable [n]
 * citations in the answer to the matching source card (the NotebookLM move).
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
      body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || "Request failed");
    return data;
  }

  function setDocStatus(text) {
    els.docStatus.textContent = text;
  }

  // --- document loading ----------------------------------------------------

  function onDocumentLoaded(data) {
    hasDocument = true;
    setDocStatus(`Loaded “${data.name}” · ${data.chunks} passages indexed`);
    els.question.focus();
  }

  async function loadSample() {
    setDocStatus("Loading sample…");
    try {
      onDocumentLoaded(await postJSON("/api/sample", {}));
    } catch (err) {
      setDocStatus(err.message);
    }
  }

  async function uploadFile(file) {
    setDocStatus(`Reading “${file.name}”…`);
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch("/api/ingest", { method: "POST", body: form });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error || "Upload failed");
      onDocumentLoaded(data);
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

  function renderSources(sources) {
    if (!sources.length) {
      els.sourcesList.innerHTML =
        '<li class="text-slate-400">No matching passages.</li>';
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
          <p class="text-slate-700">${escapeHtml(s.text)}</p>
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
    card.addEventListener(
      "animationend",
      () => card.classList.remove("flash"),
      { once: true },
    );
  }

  function renderAnswer(data) {
    const grounded = data.grounded
      ? '<span class="text-brand">grounded answer</span>'
      : '<span class="text-slate-400">offline extract</span>';
    const body = data.found
      ? linkifyCitations(data.answer)
      : `<span class="text-slate-600">${escapeHtml(data.answer)}</span>`;
    els.answerPanel.innerHTML = `
      <p class="text-[0.95rem] leading-relaxed">${body}</p>
      <p class="mt-3 text-xs text-slate-400">${grounded}${
        data.model ? " · " + escapeHtml(data.model) : ""
      }</p>`;
    renderSources(data.sources || []);
  }

  async function ask(question) {
    renderSkeleton();
    els.askBtn.disabled = true;
    try {
      renderAnswer(await postJSON("/api/ask", { question }));
    } catch (err) {
      els.answerPanel.innerHTML = `<p class="text-sm text-red-600">${escapeHtml(
        err.message,
      )}</p>`;
    } finally {
      els.askBtn.disabled = false;
    }
  }

  // --- status --------------------------------------------------------------

  async function refreshStatus() {
    try {
      const res = await fetch("/api/status");
      const data = await res.json();
      if (data.llm_configured) {
        els.llmBadge.textContent = "NVIDIA NIM · live";
        els.llmBadge.className =
          "rounded-full bg-brand-light px-3 py-1 text-xs font-medium text-brand-dark";
      } else {
        els.llmBadge.textContent = "offline mode";
        els.llmBadge.className =
          "rounded-full bg-amber-100 px-3 py-1 text-xs font-medium text-amber-700";
      }
    } catch {
      els.llmBadge.textContent = "offline mode";
    }
  }

  // --- wiring --------------------------------------------------------------

  els.sampleBtn.addEventListener("click", loadSample);
  els.browseBtn.addEventListener("click", () => els.fileInput.click());
  els.fileInput.addEventListener("change", (e) => {
    if (e.target.files[0]) uploadFile(e.target.files[0]);
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
