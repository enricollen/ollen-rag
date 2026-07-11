// Indices page: the cross-store "existing indexes" panel doubles as the picker — click an index
// (in the active store) to browse its buckets and raw stored chunks below, or 🗑 to permanently
// delete it (irreversible — typed confirmation). Inactive-store indexes are shown read-only, since
// browse/delete endpoints target the active backend only.
import { api, errorMessage, escapeHtml, toast, chunkTextHtml, wireChunks, fetchIndexInfo, indexOverviewHtml, wireIndicesOverview } from "../lib.js";

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

// One bucket entry: the clickable 📦 card (select-to-filter) plus a 🗑 delete button.
// Wrapped in a container because a <button> cannot nest inside the card <button>.
function bucketCardHtml(name, count) {
  return `
    <div class="bucket-card-wrap">
      <button type="button" class="bucket-card" data-bucket="${escapeHtml(name)}">
        <span class="bucket-card-icon">📦</span>
        <span class="bucket-card-name">${escapeHtml(name)}</span>
        <span class="bucket-card-count">${count} doc${count === 1 ? "" : "s"}</span>
      </button>
      <button type="button" class="bucket-del" data-bucket="${escapeHtml(name)}" data-count="${count}" title="Delete this bucket">🗑 Delete</button>
    </div>`;
}

// Fetch the selected index's info and render its buckets as cards. Clicking a card lists that
// bucket's 📄 file names AND refreshes the "stored chunks" pager below, scoped to that bucket;
// clicking the selected card again clears the scope (back to all chunks in the index).
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
    // Wire the 🗑 delete button on each bucket card: confirm, DELETE, then refresh
    // the buckets, stored chunks, and the overview doc counts.
    host.querySelectorAll(".bucket-del").forEach(btn => {
      btn.onclick = async (e) => {
        e.stopPropagation();  // don't trigger the card's select-to-filter click
        const bucket = btn.dataset.bucket;
        const count = btn.dataset.count;
        if (!confirm(`Delete bucket "${bucket}" and its ${count} document(s) from "${indexName}"? This cannot be undone.`)) return;
        try {
          const res = await api(`/api/v1/indices/${encodeURIComponent(indexName)}/buckets/${encodeURIComponent(bucket)}`, { method: "DELETE" });
          toast(`Deleted bucket "${bucket}" (${res.deleted} document(s))`, "success");
          if (currentBucket === bucket) currentBucket = null;  // clear a stale chunk filter
          currentPage = 0;
          await loadBuckets(view, indexName);                    // re-render bucket cards
          await loadDocuments(view, currentIndex, currentPage);  // refresh stored chunks
          refreshOverview(view);                                 // refresh overview doc counts
        } catch (err) {
          toast(errorMessage(err), "error");
        }
      };
    });
  } catch (e) {
    host.innerHTML = `<div class="card">${escapeHtml(errorMessage(e))}</div>`;
  }
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

// Re-fetch the cross-store overview and (re)render the picker panel + its selection/delete wiring.
async function refreshOverview(view) {
  const host = document.getElementById("overview-host");
  host.innerHTML = '<div class="empty-state"><span class="spinner"></span></div>';
  let overview = null;
  try { overview = await api("/api/v1/indices/overview"); } catch (e) { host.innerHTML = `<div class="card">${escapeHtml(errorMessage(e))}</div>`; return; }
  host.innerHTML = indexOverviewHtml(overview, { deletable: true, selectable: true });
  // Keep the previously-selected card highlighted across refreshes when it still exists.
  const stillExists = overview.stores.some(s => s.active && s.indices.some(i => i.index === currentIndex));
  if (!stillExists) currentIndex = null;
  markSelectedCard(host);
  wireIndicesOverview(host, overview, {
    onSelect: (store, index) => selectIndex(view, host, index),
    onDelete: (store, index) => deleteIndex(view, index),
  });
  if (currentIndex) { loadBuckets(view, currentIndex); loadDocuments(view, currentIndex, currentPage); }
  else showBrowsePlaceholder();
}

// Highlight the card matching currentIndex (active store only).
function markSelectedCard(host) {
  host.querySelectorAll(".kb-index").forEach(c => c.classList.toggle("selected", c.dataset.index === currentIndex));
}

function selectIndex(view, host, index) {
  currentIndex = index;
  currentPage = 0;
  currentBucket = null;
  markSelectedCard(host);
  loadBuckets(view, currentIndex);
  loadDocuments(view, currentIndex, currentPage);
}

function showBrowsePlaceholder() {
  document.getElementById("buckets-host").innerHTML = '<div class="empty-state">Select an index above to browse its buckets.</div>';
  document.getElementById("bucket-docs-host").innerHTML = "";
  document.getElementById("docs-host").innerHTML = '<div class="empty-state">Select an index above to browse its stored chunks.</div>';
  document.getElementById("pager-info").textContent = "";
}

async function deleteIndex(view, index) {
  const typed = prompt(
    `This permanently deletes ALL documents in "${index}". This cannot be undone.\n\nType the index name to confirm:`
  );
  if (typed !== index) {
    if (typed !== null) toast("Index name did not match — nothing deleted", "error");
    return;
  }
  try {
    await api(`/api/v1/indices/${encodeURIComponent(index)}`, { method: "DELETE" });
    toast(`Deleted index "${index}"`, "success");
    if (currentIndex === index) currentIndex = null;
    await refreshOverview(view);
  } catch (e) {
    toast(errorMessage(e), "error");
  }
}

export async function render(view) {
  currentIndex = null; currentPage = 0; currentBucket = null;
  view.innerHTML = `
    <h1 class="page-title">Indices</h1>
    <p class="page-sub">Every index across all vector stores. Click one (in the active store) to browse its documents, or 🗑 to permanently delete it.</p>

    <div class="card kb-overview">
      <div class="btn-row" style="justify-content:space-between;align-items:center;margin-bottom:.6rem">
        <h2 style="margin:0">📚 Existing indexes <span class="hint" style="font-weight:400">— across all vector stores</span></h2>
        <button class="secondary" id="refresh-indices">Refresh</button>
      </div>
      <div id="overview-host"></div>
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

  document.getElementById("refresh-indices").onclick = () => refreshOverview(view);
  document.getElementById("prev-page").onclick = () => {
    if (currentIndex && currentPage > 0) { currentPage--; loadDocuments(view, currentIndex, currentPage); }
  };
  document.getElementById("next-page").onclick = () => {
    if (currentIndex) { currentPage++; loadDocuments(view, currentIndex, currentPage); }
  };

  try {
    await refreshOverview(view);
  } catch (e) {
    toast(errorMessage(e), "error");
  }
}
