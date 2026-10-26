/**
 * Business Entity Resolution — Multi-Page SPA Controller
 * =======================================================
 * Manages:
 *  - Page navigation with fade+slide transitions
 *  - Progress indicator & navbar active state
 *  - Demo Dataset browser (Source 1/2/3, search, filter, pagination)
 *  - Live Entity Matching via Flask API
 *  - Real ML probabilities from /api/match/<id>
 */

'use strict';

// ─── State ────────────────────────────────────────────────────────────────────
const STATE = {
  currentPage: 0,
  totalPages: 8,
  transitioning: false,

  // Demo browser
  demoSource: 'source1',
  demoPage: 1,
  demoPageSize: 10,
  demoFiltered: [],

  // All loaded data
  s1Data: [],
  s2Data: [],
  s3Data: [],
};

// ─── Page Navigation ──────────────────────────────────────────────────────────
function goToPage(idx) {
  if (STATE.transitioning) return;
  if (idx < 0 || idx >= STATE.totalPages) return;
  if (idx === STATE.currentPage) return;

  STATE.transitioning = true;

  const currentEl = document.getElementById(`page-${STATE.currentPage}`);
  const nextEl    = document.getElementById(`page-${idx}`);

  // Fade out current
  currentEl.style.opacity    = '0';
  currentEl.style.transform  = idx > STATE.currentPage
    ? 'translateY(-8px)' : 'translateY(8px)';
  currentEl.style.transition = 'opacity 0.22s ease, transform 0.22s ease';

  setTimeout(() => {
    currentEl.classList.remove('active');
    currentEl.style.opacity   = '';
    currentEl.style.transform = '';
    currentEl.style.transition = '';

    STATE.currentPage = idx;

    nextEl.classList.add('active');
    nextEl.style.opacity   = '0';
    nextEl.style.transform = idx > STATE.currentPage - 1
      ? 'translateY(12px)' : 'translateY(-12px)';

    // Force reflow
    void nextEl.offsetHeight;

    nextEl.style.transition = 'opacity 0.28s ease, transform 0.28s ease';
    nextEl.style.opacity    = '1';
    nextEl.style.transform  = 'translateY(0)';

    setTimeout(() => {
      nextEl.style.transition = '';
      nextEl.style.opacity    = '';
      nextEl.style.transform  = '';
      STATE.transitioning = false;
    }, 290);

    updateNavbar();
    updateProgress();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, 230);
}

// ─── Navbar ───────────────────────────────────────────────────────────────────
function updateNavbar() {
  document.querySelectorAll('.nav-step').forEach((btn) => {
    const pg = parseInt(btn.dataset.page, 10);
    btn.classList.toggle('active', pg === STATE.currentPage);
  });
}

// ─── Progress Indicator ───────────────────────────────────────────────────────
function updateProgress() {
  const pg = STATE.currentPage;
  const pct = (pg / (STATE.totalPages - 1)) * 100;

  const fill = document.getElementById('progress-fill');
  if (fill) fill.style.width = pct + '%';

  for (let i = 0; i < STATE.totalPages; i++) {
    const dot   = document.getElementById(`pdot-${i}`);
    const label = document.getElementById(`plabel-${i}`);
    const con   = document.getElementById(`pcon-${i}`);

    if (!dot) continue;

    dot.classList.remove('active', 'done');
    if (label) label.classList.remove('active');

    if (i < pg)      { dot.classList.add('done'); }
    else if (i === pg) { dot.classList.add('active'); if (label) label.classList.add('active'); }

    if (con) con.classList.toggle('done', i < pg);
  }
}

// Wire navbar buttons
document.querySelectorAll('.nav-step').forEach((btn) => {
  btn.addEventListener('click', () => goToPage(parseInt(btn.dataset.page, 10)));
});

// Wire progress dots
document.querySelectorAll('.progress-dot-wrap').forEach((wrap) => {
  wrap.addEventListener('click', () => {
    const pg = parseInt(wrap.dataset.page, 10);
    if (!isNaN(pg)) goToPage(pg);
  });
});

// ─── Escape HTML ──────────────────────────────────────────────────────────────
function esc(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

// ─── Load All Data at Startup ─────────────────────────────────────────────────
async function loadAllData() {
  try {
    const [r1, r2, r3] = await Promise.all([
      fetch('/api/entities'),
      fetch('/api/source2'),
      fetch('/api/source3'),
    ]);

    STATE.s1Data = r1.ok ? await r1.json() : [];
    STATE.s2Data = r2.ok ? await r2.json() : [];
    STATE.s3Data = r3.ok ? await r3.json() : [];

    // Update tab counts
    setEl('dtab-count-source1', STATE.s1Data.length);
    setEl('dtab-count-source2', STATE.s2Data.length);
    setEl('dtab-count-source3', STATE.s3Data.length);

    // Populate country dropdown
    populateDemoCountries();

    // Populate entity select for matching
    renderEntityDropdown(STATE.s1Data, '');

    // Show first page of demo
    renderDemoPage();

  } catch (err) {
    console.error('Failed to load initial data:', err);
  }
}

function setEl(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

// ─── Demo Dataset Browser ─────────────────────────────────────────────────────
function getActiveDataset() {
  if (STATE.demoSource === 'source2') return STATE.s2Data;
  if (STATE.demoSource === 'source3') return STATE.s3Data;
  return STATE.s1Data;
}

function populateDemoCountries() {
  const sel = document.getElementById('demo-country');
  if (!sel) return;

  const countries = new Set();
  [...STATE.s1Data, ...STATE.s2Data, ...STATE.s3Data].forEach(r => {
    if (r.country) countries.add(r.country.trim());
  });

  const sorted = [...countries].sort();
  sel.innerHTML = '<option value="">All Countries</option>';
  sorted.forEach(c => {
    const opt = document.createElement('option');
    opt.value = c;
    opt.textContent = c;
    sel.appendChild(opt);
  });
}

function renderDemoPage() {
  const dataset  = getActiveDataset();
  const query    = (document.getElementById('demo-search')?.value || '').trim().toLowerCase();
  const country  = (document.getElementById('demo-country')?.value || '').toLowerCase();
  const isSource1 = STATE.demoSource === 'source1';

  STATE.demoFiltered = dataset.filter(r => {
    const matchQ = !query
      || r.entity_id.toLowerCase().includes(query)
      || r.business_name.toLowerCase().includes(query);
    const matchC = !country || r.country.toLowerCase() === country;
    return matchQ && matchC;
  });

  const total = STATE.demoFiltered.length;
  const pages = Math.max(1, Math.ceil(total / STATE.demoPageSize));
  if (STATE.demoPage > pages) STATE.demoPage = pages;

  const start = (STATE.demoPage - 1) * STATE.demoPageSize;
  const slice = STATE.demoFiltered.slice(start, start + STATE.demoPageSize);

  const container = document.getElementById('demo-biz-list');
  if (!container) return;

  if (slice.length === 0) {
    container.innerHTML = `<div style="text-align:center;padding:40px;color:var(--text-muted);">No records match the current filter.</div>`;
  } else {
    container.innerHTML = slice.map(r => {
      const actionBtn = isSource1
        ? `<span class="biz-card-action">→</span>`
        : '';
      return `
        <div class="biz-card ${isSource1 ? 'clickable' : ''}" ${isSource1 ? `data-s1id="${esc(r.entity_id)}"` : ''}>
          <div class="biz-card-id">${esc(r.entity_id)}</div>
          <div class="biz-card-info">
            <div class="biz-card-name">${esc(r.business_name)}</div>
            <div class="biz-card-addr">${esc(r.business_address)}</div>
          </div>
          <div class="biz-card-country">${esc(r.country)}</div>
          ${actionBtn}
        </div>
      `;
    }).join('');

    // Wire clickable Source 1 cards
    if (isSource1) {
      container.querySelectorAll('.biz-card.clickable').forEach(card => {
        card.addEventListener('click', () => {
          const id = card.dataset.s1id;
          // Preselect in matching section + navigate
          preselectMatchEntity(id);
          goToPage(7);
        });
      });
    }
  }

  // Update pagination
  const end = Math.min(start + STATE.demoPageSize, total);
  setEl('demo-page-info', total === 0 ? 'No records' : `Records ${start + 1} – ${end} of ${total}`);

  const prevBtn = document.getElementById('demo-prev');
  const nextBtn = document.getElementById('demo-next');
  if (prevBtn) prevBtn.disabled = STATE.demoPage <= 1;
  if (nextBtn) nextBtn.disabled = STATE.demoPage >= pages;
}

// Demo tab switching
document.querySelectorAll('.demo-tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    STATE.demoSource = btn.dataset.demoSrc;
    STATE.demoPage   = 1;

    document.querySelectorAll('.demo-tab-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    const search = document.getElementById('demo-search');
    if (search) search.value = '';

    renderDemoPage();
  });
});

// Demo search / filter
['demo-search', 'demo-country'].forEach(id => {
  const el = document.getElementById(id);
  if (el) el.addEventListener('input', () => { STATE.demoPage = 1; renderDemoPage(); });
  if (el) el.addEventListener('change', () => { STATE.demoPage = 1; renderDemoPage(); });
});

// Demo pagination
['demo-prev', 'demo-next'].forEach(id => {
  const el = document.getElementById(id);
  if (!el) return;
  el.addEventListener('click', () => {
    if (id === 'demo-prev') STATE.demoPage--;
    else STATE.demoPage++;
    renderDemoPage();
  });
});

// Quick sample buttons in Demo Dataset
document.querySelectorAll('[data-demo-id]').forEach(btn => {
  btn.addEventListener('click', () => {
    const id = btn.dataset.demoId;
    // Switch to Source 1 tab, filter by that ID
    STATE.demoSource = 'source1';
    STATE.demoPage   = 1;
    document.querySelectorAll('.demo-tab-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('dtab-source1')?.classList.add('active');

    const search = document.getElementById('demo-search');
    if (search) search.value = id;
    renderDemoPage();
  });
});

// ─── Live Entity Matching (Page 7) ────────────────────────────────────────────

// Build entity dropdown from S1 data
function renderEntityDropdown(entities, selectedId) {
  const sel = document.getElementById('entity-select');
  const btn = document.getElementById('match-btn');
  if (!sel) return;

  sel.innerHTML = '';

  if (!entities || entities.length === 0) {
    const opt = document.createElement('option');
    opt.value    = '';
    opt.textContent = 'No entities match filter';
    opt.disabled = true; opt.selected = true;
    sel.appendChild(opt);
    if (btn) btn.disabled = true;
    return;
  }

  let found = false;
  entities.forEach((item, i) => {
    const opt = document.createElement('option');
    opt.value = item.entity_id;
    opt.textContent = item.label || `${item.entity_id} — ${item.business_name} — ${item.country}`;
    if (selectedId && item.entity_id === selectedId) { opt.selected = true; found = true; }
    else if (!selectedId && i === 0) opt.selected = true;
    sel.appendChild(opt);
  });

  if (btn) btn.disabled = false;
}

// Filter input handler
const filterInput = document.getElementById('entity-filter');
if (filterInput) {
  filterInput.addEventListener('input', () => {
    const q = filterInput.value.trim().toLowerCase();
    const cur = document.getElementById('entity-select')?.value || '';
    if (!q) {
      renderEntityDropdown(STATE.s1Data, cur);
    } else {
      const filtered = STATE.s1Data.filter(item =>
        item.entity_id.toLowerCase().includes(q) ||
        item.business_name.toLowerCase().includes(q) ||
        item.country.toLowerCase().includes(q)
      );
      renderEntityDropdown(filtered, '');
    }
  });
}

// Quick match sample buttons
document.querySelectorAll('.mq-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const id = btn.dataset.id;
    preselectMatchEntity(id);
    triggerMatch(id);
  });
});

// Match button
const matchBtn = document.getElementById('match-btn');
if (matchBtn) {
  matchBtn.addEventListener('click', () => {
    const sel = document.getElementById('entity-select');
    if (sel && sel.value) triggerMatch(sel.value);
  });
}

function preselectMatchEntity(id) {
  if (!id) return;
  const filterEl = document.getElementById('entity-filter');
  if (filterEl) filterEl.value = '';
  renderEntityDropdown(STATE.s1Data, id);
  const sel = document.getElementById('entity-select');
  if (sel) sel.value = id;
}

async function triggerMatch(entityId) {
  if (!entityId) {
    showMatchError('Please select a Source 1 entity first.');
    return;
  }

  // UI: loading state
  hideMatchError();
  show('match-loading');
  hide('match-idle');
  hide('match-results-content');
  setMatchBtnLoading(true);

  try {
    const resp = await fetch(`/api/match/${encodeURIComponent(entityId)}`);
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.error || `HTTP ${resp.status}`);
    }
    const data = await resp.json();
    renderMatchResults(data);
  } catch (err) {
    console.error('Match error:', err);
    showMatchError(`Match failed: ${err.message}`);
    show('match-idle');
  } finally {
    hide('match-loading');
    setMatchBtnLoading(false);
  }
}

function renderMatchResults(data) {
  const s1 = data.source1 || {};

  setEl('r-s1-id',      s1.entity_id      || '—');
  setEl('r-s1-name',    s1.business_name  || '—');
  setEl('r-s1-addr',    s1.business_address || '—');
  setEl('r-s1-country', s1.country        || '—');

  const s2 = data.source2_matches || [];
  const s3 = data.source3_matches || [];

  setEl('r-s2-count', s2.length);
  setEl('r-s3-count', s3.length);

  renderMatchCards('r-s2-cards', s2, 'Source 2');
  renderMatchCards('r-s3-cards', s3, 'Source 3');

  show('match-results-content');
}

function renderMatchCards(containerId, matches, srcLabel) {
  const el = document.getElementById(containerId);
  if (!el) return;

  if (!matches || matches.length === 0) {
    el.innerHTML = `
      <div class="no-match-card">
        <div class="no-match-icon">🔇</div>
        <div class="no-match-title">No Matches Found</div>
        <div class="no-match-sub">No candidate exceeded the 0.50 prediction threshold in ${srcLabel}.</div>
      </div>`;
    return;
  }

  el.innerHTML = matches.map(item => {
    const prob    = typeof item.match_probability === 'number' ? item.match_probability : 0;
    const probPct = item.match_probability_pct || `${(prob * 100).toFixed(1)}%`;
    const barWidth = Math.min(100, Math.round(prob * 100));

    return `
      <div class="match-card">
        <div class="match-card-top">
          <div class="match-entity-id">${esc(item.entity_id)}</div>
          <div class="match-verified">✓ MATCH</div>
        </div>
        <div class="match-name">${esc(item.business_name)}</div>
        <div class="match-addr">${esc(item.business_address)}</div>
        <div class="match-country-tag">${esc(item.country)}</div>
        <div class="match-prob-section">
          <div class="match-prob-row">
            <div class="match-prob-label">Match Probability</div>
            <div class="match-prob-val">${probPct}</div>
          </div>
          <div class="prob-bar-track">
            <div class="prob-bar-fill" style="width: ${barWidth}%"></div>
          </div>
        </div>
      </div>`;
  }).join('');
}

function resetMatch() {
  hide('match-results-content');
  hide('match-loading');
  show('match-idle');
  hideMatchError();

  const filterEl = document.getElementById('entity-filter');
  if (filterEl) filterEl.value = '';
  renderEntityDropdown(STATE.s1Data, '');
}

// ─── UI Helpers ───────────────────────────────────────────────────────────────
function show(id) { const el = document.getElementById(id); if (el) el.style.display = ''; }
function hide(id) { const el = document.getElementById(id); if (el) el.style.display = 'none'; }

function showMatchError(msg) {
  const el = document.getElementById('match-error');
  if (!el) return;
  el.textContent = msg;
  el.style.display = 'block';
}

function hideMatchError() {
  const el = document.getElementById('match-error');
  if (el) el.style.display = 'none';
}

function setMatchBtnLoading(loading) {
  const btn  = document.getElementById('match-btn');
  const text = document.getElementById('match-btn-text');
  const icon = document.getElementById('match-btn-icon');
  if (!btn) return;
  btn.disabled = loading;
  if (text) text.textContent = loading ? 'Matching…' : 'FIND MATCHES';
  if (icon) icon.innerHTML = loading
    ? '<span class="btn-spin"></span>'
    : '→';
}

// ─── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  updateNavbar();
  updateProgress();
  loadAllData();
});
