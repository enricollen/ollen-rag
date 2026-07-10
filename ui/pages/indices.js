// Indices page: browse raw stored documents per OpenSearch index and, when
// needed, permanently delete an index (irreversible — double confirmation).
import { api, errorMessage, escapeHtml, toast, chunkTextHtml, wireChunks, fetchIndexInfo } from "../lib.js";

const PAGE_SIZE = 20;
let currentPage = 0;
let currentIndex = null;
let currentBucket = null;  // when set, the "stored chunks" pager is scoped to this bucket

function docCardHtml(doc) {
  const chips = Object.entries(doc.metadata || {})
    .filter(([k]) => !["_node_type", "_node_content"].includes(k))
    .map(([k, v]) => `<span class="pill">${escapeHtml(k)}: ${escapeHtml(String(v).slice(0, 60))}</span>`).join("");
  return `
    <div class="result-node">
      <div class="hint" style="margin-bottom:.4rem">id: <code class="inline">${escapeHtml(doc.id)}</code></div>
      ${chunkTextHtml(doc.content)}
      <div class="chip-row">${chips}</div>
    </div>`;
}

// One clickable bucket card: 📦 icon + bucket name + document count.
function bucketCardHtml(name, count) {
  return `
    <button type="button" class="bucket-card" data-bucket="${escapeHtml(name)}">
      <span class="bucket-card-icon">📦</span>
      <span class="bucket-card-name">${escapeHtml(name)}</span>
      <span class="bucket-card-count">${count} doc${count === 1 ? "" : "s"}</span>
    </button>`;
}

// Fetch the selected index's info and render its buckets as cards. Clicking a card lists that
// bucket's 📄 file names AND refreshes the "stored chunks" pager below, scoped to that bucket;
// clicking the selected card again clears the scope (back to all chunks in the index).
// Reuses /indices/{name}/info's bucket_files map.
async function loadBuckets(view, indexName) {
  const host = document.getElementById("buckets-host");
  const docsHost = document.getElementById("bucket-docs-host");
  docsHost.innerHTML = "";
  host.innerHTML = '<div class="empty-state"><span class="spinner"></span></div>';
  try {
    const info = await fetchIndexInfo(indexName);
    const bucketFiles = info.bucket_files || {};
    const names = Object.keys(bucketFiles);
    if (!names.length) { host.innerHTML = '<div class="empty-state">No buckets in this index.</div>'; return; }
    host.innerHTML = names.map(n => bucketCardHtml(n, bucketFiles[n].length)).join("");
    host.querySelectorAll(".bucket-card").forEach(card => {
      card.onclick = () => {
        const bucket = card.dataset.bucket;
        const alreadySelected = card.classList.contains("selected");
        // Toggle: re-clicking the active bucket clears the filter; otherwise select this one.
        currentBucket = alreadySelected ? null : bucket;
        host.querySelectorAll(".bucket-card").forEach(c => c.classList.toggle("selected", !alreadySelected && c === card));
        if (currentBucket) {
          const files = bucketFiles[bucket] || [];
          docsHost.innerHTML = `
            <div class="bucket-docs-title">📦 ${escapeHtml(bucket)} — ${files.length} document${files.length === 1 ? "" : "s"}</div>
            <ul class="bucket-docs-list">${files.map(f => `<li class="bucket-doc-item">📄 ${escapeHtml(f)}</li>`).join("") || '<li class="hint">empty</li>'}</ul>`;
        } else {
          docsHost.innerHTML = "";
        }
        // Refresh the chunks list to reflect the index + bucket selection.
        currentPage = 0;
        loadDocuments(view, currentIndex, currentPage);
      };
    });
  } catch (e) {
    host.innerHTML = `<div class="card">${escapeHtml(errorMessage(e))}</div>`;
  }
}

async function loadIndicesList(select) {
  const res = await api("/api/v1/indices");
  select.innerHTML = res.indices.map(ix =>
    `<option value="${ix.index}">${ix.index} (${ix["docs.count"]} docs, ${ix["store.size"]})</option>`
  ).join("") || '<option value="">no indices yet</option>';
  return res.indices;
}

async function loadDocuments(view, indexName, page) {
  const docsHost = document.getElementById("docs-host");
  const pager = document.getElementById("pager-info");
  docsHost.innerHTML = '<div class="empty-state"><span class="spinner"></span></div>';
  try {
    const bucketQs = currentBucket ? `&bucket=${encodeURIComponent(currentBucket)}` : "";
    const res = await api(`/api/v1/indices/${encodeURIComponent(indexName)}/documents?offset=${page * PAGE_SIZE}&limit=${PAGE_SIZE}${bucketQs}`);
    if (!res.documents.length) {
      docsHost.innerHTML = `<div class="empty-state">No documents${currentBucket ? ` in bucket "${escapeHtml(currentBucket)}"` : " in this index"}.</div>`;
    } else {
      docsHost.innerHTML = res.documents.map(docCardHtml).join("");
      wireChunks(docsHost);  // reveal Expand toggles on any document body that overflows its clamp
    }
    const totalPages = Math.max(1, Math.ceil(res.total / PAGE_SIZE));
    const scope = currentBucket ? ` · bucket "${currentBucket}"` : "";
    pager.textContent = `page ${page + 1} / ${totalPages} · ${res.total} document(s) total${scope}`;
    document.getElementById("prev-page").disabled = page === 0;
    document.getElementById("next-page").disabled = page + 1 >= totalPages;
  } catch (e) {
    docsHost.innerHTML = `<div class="card">${escapeHtml(errorMessage(e))}</div>`;
    pager.textContent = "";
  }
}

export async function render(view) {
  view.innerHTML = `
    <h1 class="page-title">Indices</h1>
    <p class="page-sub">Browse the raw documents actually stored in each OpenSearch index, or permanently delete one to start clean.</p>

    <div class="card">
      <div class="row" style="align-items:flex-end">
        <label class="field" style="margin-bottom:0">
          <span class="label-text">Index</span>
          <select id="index-select"></select>
        </label>
        <div style="flex:0 0 auto">
          <button class="secondary" id="refresh-indices">Refresh list</button>
        </div>
        <div style="flex:0 0 auto">
          <button class="danger-ghost" id="delete-index-btn">Delete index…</button>
        </div>
      </div>
    </div>

    <div class="card">
      <h2 style="margin-top:0">Buckets</h2>
      <p class="page-sub" style="margin-top:0">Select a 📦 bucket to see the 📄 documents it contains.</p>
      <div id="buckets-host" class="bucket-card-grid"></div>
      <div id="bucket-docs-host"></div>
    </div>

    <div class="card">
      <h2 style="margin-top:0">All stored chunks</h2>
      <div class="btn-row" style="justify-content:space-between">
        <span id="pager-info" class="hint"></span>
        <div>
          <button class="secondary" id="prev-page">← Prev</button>
          <button class="secondary" id="next-page">Next →</button>
        </div>
      </div>
      <div id="docs-host" style="margin-top:.8rem"></div>
    </div>
  `;

  const select = document.getElementById("index-select");
  const refreshBtn = document.getElementById("refresh-indices");
  const deleteBtn = document.getElementById("delete-index-btn");

  async function refreshAll(preserveSelection) {
    const prev = preserveSelection ? select.value : null;
    const list = await loadIndicesList(select);
    if (prev && list.some(ix => ix.index === prev)) select.value = prev;
    currentIndex = select.value || null;
    currentPage = 0;
    currentBucket = null;  // switching/refreshing the index resets the bucket scope
    if (currentIndex) { await loadBuckets(view, currentIndex); await loadDocuments(view, currentIndex, currentPage); }
  }

  try {
    await refreshAll(false);
  } catch (e) {
    toast(errorMessage(e), "error");
  }

  select.onchange = () => {
    currentIndex = select.value;
    currentPage = 0;
    currentBucket = null;  // new index → clear any bucket scope
    if (currentIndex) { loadBuckets(view, currentIndex); loadDocuments(view, currentIndex, currentPage); }
  };
  refreshBtn.onclick = () => refreshAll(true);

  document.getElementById("prev-page").onclick = () => {
    if (currentPage > 0) { currentPage--; loadDocuments(view, currentIndex, currentPage); }
  };
  document.getElementById("next-page").onclick = () => {
    currentPage++; loadDocuments(view, currentIndex, currentPage);
  };

  deleteBtn.onclick = async () => {
    if (!currentIndex) { toast("No index selected", "error"); return; }
    const typed = prompt(
      `This permanently deletes ALL documents in "${currentIndex}". This cannot be undone.\n\nType the index name to confirm:`
    );
    if (typed !== currentIndex) {
      if (typed !== null) toast("Index name did not match — nothing deleted", "error");
      return;
    }
    try {
      await api(`/api/v1/indices/${encodeURIComponent(currentIndex)}`, { method: "DELETE" });
      toast(`Deleted index "${currentIndex}"`, "success");
      await refreshAll(false);
    } catch (e) {
      toast(errorMessage(e), "error");
    }
  };
}
