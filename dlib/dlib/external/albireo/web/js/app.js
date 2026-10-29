// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

/*
    The Albireo web application: discussions and projects kept in this browser, incognito
    discussions kept in memory only, the three functions of the service (general, code,
    cybersecurity), deep reasoning shown as it streams, and an agent mode whose tools run
    in the page.

    THE CONTEXT OF A QUESTION
        The engine keeps the keys and values of each conversation and of the prompts it
        shares with others, and computes again only the tokens that differ from what it
        already read.  The prompt is therefore built so that its beginning never changes
        from one question to the next: the system message holds only what stays (the date,
        the instructions, the documents of a project when they fit whole), each message
        keeps, once sent, the excerpts of documents that were chosen for it, and the
        oldest exchanges leave in one step, well below the limit, rather than one at a
        time at each question.  Each question then costs its own tokens only, whatever
        the length of the conversation, and the conversations of one project share the
        reading of its documents.

        The documents of a project that do not fit whole, and the large documents attached
        to a message, are searched (rag.js): the excerpts closest to the question join it.
*/

import { t, setLanguage, language, languages } from './i18n.js';
import { icon } from './icons.js';
import * as store from './store.js';
import { renderMarkdown, attachCodeButtons, escapeHtml } from './markdown.js';
import * as api from './api.js';
import { extractText, maxFileBytes, extension, searchPassages } from './documents.js';
import * as rag from './rag.js';
import * as tools from './tools.js';

const roles = ['general', 'code', 'security'];
const roleIcons = { general: 'compass', code: 'code', security: 'shield' };
const projectColors = ['#5b8cff', '#f2b544', '#3fb68b', '#e5534b', '#a371f7', '#2fb9d4', '#e08a3c', '#8a90aa'];
const maxAgentSteps = 8;

// Shares of the context of the model: the documents of a project read whole, an attached
// document read whole, and the excerpts given with a question.  Beyond the first two, the
// documents are searched.
const knowledgeShare = 0.3;
const attachmentShare = 0.25;
const excerptShare = 0.2;
// The exchanges kept when the oldest leave: the conversation then takes this share of the
// room it has, so that it grows again for several questions before the next cut.
const keptShare = 0.6;

// Tokens of a text, estimated from its characters.
function estimateTokens(text) {
  return Math.ceil(String(text || '').length / 3.6);
}

const defaults = {
  theme: 'auto', language: 'auto', userName: '', defaultRole: 'general', temperature: null, topP: null,
  maxTokens: 0, sendOnEnter: true, systemPrompt: '', lastReasoning: false, lastAgent: false, sidebar: true
};

const state = {
  settings: Object.assign({}, defaults),
  conversations: [],
  projects: [],
  current: null,
  view: 'chat',
  projectId: null,
  incognito: false,
  pending: [],
  busy: false,
  abort: null,
  status: null,
  catalog: null,
  search: '',
  statusError: false
};

const ui = {};

// Helpers.
function h(tag, props, ...children) {
  const el = document.createElement(tag);
  if (props) {
    for (const [k, v] of Object.entries(props)) {
      if (v == null || v === false) continue;
      if (k === 'class') el.className = v;
      else if (k === 'html') el.innerHTML = v;
      else if (k === 'text') el.textContent = v;
      else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
      else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
      else el.setAttribute(k, v === true ? '' : v);
    }
  }
  for (const c of children.flat(Infinity)) {
    if (c == null || c === false) continue;
    el.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
  return el;
}

function iconButton(name, title, onclick, cls) {
  return h('button', { class: cls || 'icon-button', title, 'aria-label': title, type: 'button', onclick, html: icon(name) });
}

function formatSize(n) {
  if (n < 1024) return n + ' B';
  if (n < 1048576) return (n / 1024).toFixed(1) + ' KB';
  if (n < 1073741824) return (n / 1048576).toFixed(1) + ' MB';
  return (n / 1073741824).toFixed(2) + ' GB';
}

function formatDuration(ms) {
  const s = Math.round(ms / 1000);
  if (s < 60) return s + ' s';
  return Math.floor(s / 60) + ' min ' + (s % 60) + ' s';
}

function formatNumber(n) {
  return new Intl.NumberFormat(language()).format(n);
}

function toast(message, error) {
  const el = h('div', { class: 'toast' + (error ? ' error' : '') }, message);
  ui.toasts.append(el);
  setTimeout(() => el.remove(), error ? 6000 : 2600);
}

function download(name, content, type) {
  const url = URL.createObjectURL(new Blob([content], { type: type || 'text/plain;charset=utf-8' }));
  const a = h('a', { href: url, download: name });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

function mimeOf(name) {
  const ext = name.toLowerCase().split('.').pop();
  return {
    md: 'text/markdown', csv: 'text/csv', json: 'application/json', html: 'text/html', xml: 'application/xml',
    js: 'text/javascript', py: 'text/x-python', svg: 'image/svg+xml'
  }[ext] || 'text/plain';
}

// Theme and language.
function applyTheme() {
  const pref = state.settings.theme;
  const dark = pref === 'dark' || (pref === 'auto' && window.matchMedia('(prefers-color-scheme: dark)').matches);
  document.documentElement.dataset.theme = dark ? 'dark' : 'light';
  document.getElementById('hljs-light').disabled = dark;
  document.getElementById('hljs-dark').disabled = !dark;
}

function applyLanguage() {
  setLanguage(state.settings.language === 'auto' ? null : state.settings.language);
}

async function saveSetting(key, value) {
  state.settings[key] = value;
  await store.setSetting(key, value);
}

// Service status and models.
function roleModel(role) {
  const s = state.status;
  if (!s || !s.active) return null;
  return s.active[role] || s.active.general || null;
}

function modelInfo(id) {
  if (!state.catalog || !id) return null;
  return (state.catalog.models || []).find((m) => m.id === id) || null;
}

function modelName(id) {
  const m = modelInfo(id);
  return m && m.name ? m.name : id;
}

function loadedModel(id) {
  const list = (state.status && state.status.loaded_models) || [];
  return list.find((m) => m.id === id) || null;
}

function roleCapabilities(role) {
  const id = roleModel(role);
  const loaded = loadedModel(id);
  if (loaded) return { reasoning: loaded.reasoning !== false, tools: loaded.tools !== false, context: loaded.context || 0 };
  const m = modelInfo(id);
  return { reasoning: m ? m.reasoning !== false : true, tools: m ? m.tools !== false : true, context: 0 };
}

async function refreshStatus() {
  let changed = false;
  try {
    const s = await api.status();
    const before = state.status ? JSON.stringify([state.status.active, state.status.loaded_models]) : '';
    changed = before !== JSON.stringify([s.active, s.loaded_models]);
    state.status = s;
    state.statusError = false;
    if (changed || !state.catalog) state.catalog = await api.models(language()).catch(() => state.catalog);
    // The embedding model of the document search; the documents not yet given vectors are
    // indexed once it is there.
    const embedder = (s.loaded_models || []).find((m) => m.embedding && s.active && m.id === s.active.embedding);
    const previous = rag.embeddingModel();
    rag.setEmbeddingModel(embedder ? { id: embedder.id, dimensions: embedder.dimensions } : null);
    if (embedder && (!previous || previous.id !== embedder.id)) indexProjects();
  } catch (e) {
    changed = !state.statusError;
    state.statusError = true;
  }
  if (changed && state.current) {
    renderTopbar();
    renderComposerState();
    if (state.view === 'chat' && !state.current.messages.length && !state.busy) renderMain();
  }
  renderStatusLine();
}

function renderStatusLine() {
  if (!ui.statusLine) return;
  const s = state.status;
  let led = 'error', text = t('serviceDown');
  if (s && !state.statusError) {
    const stats = s.engine && s.engine.stats;
    const loading = Object.values(s.models_state || {}).some((v) => v === 'loading');
    const model = roleModel(state.current ? state.current.role : 'general');
    if (!model) {
      led = 'error';
      text = t('noModel');
    } else if (loading) {
      led = 'busy';
      text = t('modelLoading');
    } else {
      led = stats && stats.active > 0 ? 'busy' : 'ok';
      const tps = stats ? stats.generation_tokens_per_second : 0;
      text = modelName(model) + ' · ' + (s.engine.variant || '?') + (tps > 0.05 ? ' · ' + t('tokensPerSecond', { n: tps.toFixed(1) }) : '');
    }
  }
  ui.statusLine.replaceChildren(h('span', { class: 'led ' + led }), h('span', { class: 'grow', title: text }, text));
}

// Lists.
async function reloadLists() {
  state.conversations = await store.allConversations();
  state.projects = await store.allProjects();
}

function projectById(id) {
  return state.projects.find((p) => p.id === id) || null;
}

function dateGroup(ts) {
  const d = new Date(ts);
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const day = 86400000;
  if (d >= today) return 'today';
  if (d >= today - day) return 'yesterday';
  if (d >= today - 7 * day) return 'week';
  if (d >= today - 30 * day) return 'month';
  return 'older';
}

function conversationText(c) {
  const parts = [c.title || ''];
  for (const m of c.messages || []) {
    if (m.role === 'user') parts.push(m.content || '');
    else for (const s of m.steps || []) parts.push(s.content || '');
  }
  return parts.join('\n');
}

function searchConversations(query, limit) {
  const q = query.trim().toLowerCase();
  if (!q) return [];
  const out = [];
  for (const c of state.conversations) {
    const text = conversationText(c);
    const i = text.toLowerCase().indexOf(q);
    if (i < 0) continue;
    const start = Math.max(0, i - 60);
    out.push({ conversation: c, excerpt: (start ? '…' : '') + text.slice(start, i + q.length + 100).replace(/\s+/g, ' ') });
    if (out.length >= (limit || 50)) break;
  }
  return out;
}

// Sidebar.
function renderSidebar() {
  const scroll = ui.sideScroll;
  scroll.replaceChildren();
  if (state.search.trim()) {
    const hits = searchConversations(state.search);
    const section = h('div', { class: 'side-section' }, h('div', { class: 'side-title' }, t('searchIn')));
    if (!hits.length) section.append(h('div', { class: 'empty-note' }, t('noResults')));
    for (const hit of hits) {
      const item = conversationItem(hit.conversation);
      item.title = hit.excerpt;
      section.append(item);
    }
    scroll.append(section);
    return;
  }

  const projects = h('div', { class: 'side-section' },
    h('div', { class: 'side-title' }, h('span', null, t('projects')),
      h('button', { type: 'button', title: t('newProject'), onclick: () => newProject(false), html: icon('plus') })));
  if (!state.projects.length) projects.append(h('div', { class: 'empty-note' }, t('noProjects')));
  for (const p of state.projects) {
    const count = state.conversations.filter((c) => c.projectId === p.id).length;
    projects.append(h('div', {
      class: 'item' + (state.view === 'project' && state.projectId === p.id ? ' active' : ''),
      onclick: () => go('#/p/' + p.id)
    }, h('span', { class: 'dot', style: { background: p.color || projectColors[0] } }),
    h('span', { class: 'label' }, p.name), h('span', { class: 'faint', style: { fontSize: '12px', color: 'var(--text-faint)' } },
      count ? String(count) : '')));
  }
  scroll.append(projects);

  const pinned = state.conversations.filter((c) => c.pinned);
  if (pinned.length) {
    const section = h('div', { class: 'side-section' }, h('div', { class: 'side-title' }, t('pinned')));
    for (const c of pinned) section.append(conversationItem(c));
    scroll.append(section);
  }
  const groups = { today: [], yesterday: [], week: [], month: [], older: [] };
  for (const c of state.conversations) if (!c.pinned) groups[dateGroup(c.updated)].push(c);
  let any = pinned.length > 0;
  for (const g of Object.keys(groups)) {
    if (!groups[g].length) continue;
    any = true;
    const section = h('div', { class: 'side-section' }, h('div', { class: 'side-title' }, t(g)));
    for (const c of groups[g]) section.append(conversationItem(c));
    scroll.append(section);
  }
  if (!any) scroll.append(h('div', { class: 'side-section' }, h('div', { class: 'side-title' }, t('discussions')),
    h('div', { class: 'empty-note' }, t('noDiscussions'))));
}

function conversationItem(c) {
  const p = c.projectId ? projectById(c.projectId) : null;
  const active = state.view === 'chat' && state.current && state.current.id === c.id;
  return h('div', { class: 'item' + (active ? ' active' : ''), onclick: () => go('#/c/' + c.id) },
    p ? h('span', { class: 'dot', title: p.name, style: { background: p.color || projectColors[0] } })
      : h('span', { html: icon(roleIcons[c.role] || 'chat') }),
    h('span', { class: 'label' }, c.title || t('untitled')),
    c.pinned ? h('span', { class: 'pin', html: icon('pin') }) : null,
    h('button', {
      class: 'more', type: 'button', title: '…', html: icon('more'),
      onclick: (e) => { e.stopPropagation(); conversationMenu(c, e.currentTarget); }
    }));
}

function showMenu(anchor, entries) {
  closeMenu();
  const menu = h('div', { class: 'menu', role: 'menu' });
  for (const e of entries) {
    if (e === '-') { menu.append(h('div', { class: 'sep' })); continue; }
    if (e.head) { menu.append(h('div', { class: 'head' }, e.head)); continue; }
    menu.append(h('div', {
      class: 'entry' + (e.danger ? ' danger' : '') + (e.checked ? ' checked' : ''), role: 'menuitem',
      onclick: () => { closeMenu(); e.action(); }
    }, e.icon ? h('span', { html: icon(e.icon) }) : null,
    h('span', null, e.label, e.desc ? h('span', { class: 'desc' }, e.desc) : null)));
  }
  document.body.append(menu);
  const r = anchor.getBoundingClientRect();
  const mw = menu.offsetWidth, mh = menu.offsetHeight;
  let x = Math.min(r.left, window.innerWidth - mw - 8);
  let y = r.bottom + 6;
  if (y + mh > window.innerHeight - 8) y = Math.max(8, r.top - mh - 6);
  menu.style.left = Math.max(8, x) + 'px';
  menu.style.top = y + 'px';
  ui.menu = menu;
  setTimeout(() => document.addEventListener('mousedown', outsideMenu), 0);
}

function outsideMenu(e) {
  if (ui.menu && !ui.menu.contains(e.target)) closeMenu();
}

function closeMenu() {
  if (ui.menu) ui.menu.remove();
  ui.menu = null;
  document.removeEventListener('mousedown', outsideMenu);
}

function conversationMenu(c, anchor) {
  showMenu(anchor, [
    { label: t('rename'), icon: 'edit', action: () => renameConversation(c) },
    { label: c.pinned ? t('unpin') : t('pin'), icon: 'pin', action: () => togglePin(c) },
    { label: t('moveToProject'), icon: 'move', action: () => moveMenu(c, anchor) },
    c.projectId ? { label: t('removeFromProject'), icon: 'detach', action: () => setProject(c, null) } : null,
    '-',
    { label: t('exportMarkdown'), icon: 'download', action: () => exportConversation(c, 'md') },
    { label: t('exportJson'), icon: 'archive', action: () => exportConversation(c, 'json') },
    '-',
    { label: t('delete'), icon: 'trash', danger: true, action: () => removeConversation(c) }
  ].filter(Boolean));
}

function moveMenu(c, anchor) {
  const entries = [{ head: t('moveTo') }];
  for (const p of state.projects) entries.push({ label: p.name, icon: 'folder', checked: c.projectId === p.id, action: () => setProject(c, p.id) });
  entries.push('-', { label: t('newProject'), icon: 'plus', action: async () => { const p = await newProject(true); if (p) await setProject(c, p.id); } });
  if (c.projectId) entries.push({ label: t('none'), icon: 'detach', action: () => setProject(c, null) });
  showMenu(anchor, entries);
}

async function persist(c) {
  if (!c || c.incognito) return;
  await store.putConversation(c);
  const i = state.conversations.findIndex((x) => x.id === c.id);
  if (i >= 0) state.conversations[i] = c;
  else state.conversations.unshift(c);
  state.conversations.sort((a, b) => b.updated - a.updated);
}

async function renameConversation(c) {
  const name = await promptDialog(t('rename'), c.title || '');
  if (name == null) return;
  c.title = name.trim() || c.title;
  await persist(c);
  renderSidebar();
  if (state.current && state.current.id === c.id) renderTopbar();
}

async function togglePin(c) {
  c.pinned = !c.pinned;
  await persist(c);
  renderSidebar();
}

async function setProject(c, projectId) {
  c.projectId = projectId;
  if (c.incognito) { renderTopbar(); return; }
  await persist(c);
  renderSidebar();
  if (state.current && state.current.id === c.id) renderTopbar();
  if (state.view === 'project') renderMain();
}

async function removeConversation(c) {
  const ok = await confirmDialog(t('deleteDiscussion'), t('deleteDiscussionText', { title: c.title || t('untitled') }));
  if (!ok) return;
  await store.deleteConversation(c.id);
  state.conversations = state.conversations.filter((x) => x.id !== c.id);
  if (state.current && state.current.id === c.id) go('#/');
  else renderSidebar();
}

function exportConversation(c, format) {
  const safe = (c.title || 'discussion').replace(/[\\/:*?"<>|]+/g, '_').slice(0, 80);
  if (format === 'json') {
    download(safe + '.json', JSON.stringify(c, null, 2), 'application/json');
    return;
  }
  const lines = ['# ' + (c.title || t('untitled')), ''];
  for (const m of c.messages) {
    if (m.role === 'user') {
      lines.push('## ' + (state.settings.userName || 'User'), '', m.content || '');
      for (const a of m.attachments || []) lines.push('', '> ' + a.name + ' (' + formatSize(a.size || 0) + ')');
    } else {
      lines.push('## Albireo', '');
      for (const s of m.steps || []) {
        if (s.reasoning) lines.push('<details><summary>' + t('thinking') + '</summary>', '', s.reasoning, '', '</details>', '');
        for (const tc of s.toolCalls || []) lines.push('> ' + tc.name + ' ' + (tc.arguments || ''), '', '```', tc.result || '', '```', '');
        if (s.content) lines.push(s.content);
      }
    }
    lines.push('');
  }
  download(safe + '.md', lines.join('\n'), 'text/markdown');
}

// The document search: the documents being indexed, one at a time in the background.
const indexing = new Map();   // document -> {done, total}
let indexQueue = Promise.resolve();

function queueIndex(owner, doc, persistent) {
  if (indexing.has(doc.id)) return indexQueue;
  indexing.set(doc.id, { done: 0, total: 0 });
  showIndexing(doc.id);
  indexQueue = indexQueue.then(() => rag.indexDocument(owner, doc, {
    persistent,
    onProgress: (done, total) => {
      indexing.set(doc.id, { done, total });
      showIndexing(doc.id);
    }
  })).catch(() => {}).then(() => {
    indexing.delete(doc.id);
    showIndexing(doc.id);
  });
  return indexQueue;
}

// The state of a document on the page of its project.
async function indexText(owner, docId) {
  const run = indexing.get(docId);
  if (run) return run.total ? t('indexingProgress', { p: Math.floor(100 * run.done / run.total) }) : t('indexing');
  const st = (await rag.status(owner)).get(docId);
  if (!st) return '';
  if (rag.embeddingModel() && st.vectors === st.passages) return t('indexed', { n: formatNumber(st.passages) });
  return t('indexedWords', { n: formatNumber(st.passages) });
}

function showIndexing(docId) {
  for (const el of document.querySelectorAll('[data-index-doc="' + docId + '"]')) {
    indexText(el.dataset.indexOwner, docId).then((text) => { el.textContent = text; });
  }
}

// Indexes the documents of the projects that have no passages, or no vectors of the
// current embedding model.
async function indexProjects() {
  for (const p of state.projects) {
    const st = await rag.status(p.id).catch(() => new Map());
    for (const f of p.files || []) {
      const s2 = st.get(f.id);
      const model = rag.embeddingModel();
      if (!s2 || (model && s2.vectors < s2.passages)) queueIndex(p.id, f, true);
    }
  }
}

// How a project gives its documents to the model: whole, searched, or not at all.
function knowledgeMode(p, contextTokens) {
  const files = (p && p.files) || [];
  if (!files.length) return 'none';
  const total = files.reduce((n, f) => n + estimateTokens(f.text), 0);
  return total <= knowledgeShare * (contextTokens || 8192) ? 'whole' : 'search';
}

// Projects.
async function newProject(silent) {
  const name = await promptDialog(t('newProject'), '');
  if (name == null || !name.trim()) return null;
  const p = {
    id: store.uid(), name: name.trim(), description: '', instructions: '', files: [],
    color: projectColors[state.projects.length % projectColors.length], created: Date.now(), updated: Date.now()
  };
  await store.putProject(p);
  await reloadLists();
  renderSidebar();
  if (!silent) go('#/p/' + p.id);
  return p;
}

function renderProject(p) {
  const main = ui.content;
  const convs = state.conversations.filter((c) => c.projectId === p.id);
  const name = h('input', { value: p.name });
  const description = h('input', { value: p.description || '' });
  const instructions = h('textarea', { rows: 8 });
  instructions.value = p.instructions || '';
  const save = async () => {
    p.name = name.value.trim() || p.name;
    p.description = description.value;
    p.instructions = instructions.value;
    p.updated = Date.now();
    await store.putProject(p);
    await reloadLists();
    renderSidebar();
  };
  for (const el of [name, description, instructions]) el.addEventListener('change', save);
  const colors = h('div', { class: 'colors' });
  for (const col of projectColors) {
    colors.append(h('button', {
      type: 'button', class: p.color === col ? 'on' : '', style: { background: col }, 'aria-label': col,
      onclick: async () => { p.color = col; await save(); renderProject(p); }
    }));
  }
  const fileInput = h('input', { type: 'file', multiple: true, class: 'hidden' });
  fileInput.addEventListener('change', async () => {
    const added = await readFiles(Array.from(fileInput.files));
    p.files = (p.files || []).concat(added);
    fileInput.value = '';
    await save();
    renderProject(p);
    for (const f of added) queueIndex(p.id, f, true);
  });
  const fileList = h('div', { class: 'list' });
  for (const f of p.files || []) {
    const indexState = h('span', { class: 'faint index-state', 'data-index-doc': f.id, 'data-index-owner': p.id });
    fileList.append(h('div', { class: 'row' }, formatBadge(f), h('span', { class: 'label' }, f.name),
      h('span', { class: 'faint' }, f.format + ' · ' + t('tokensEstimate', { n: formatNumber(estimateTokens(f.text)) })),
      indexState,
      iconButton('trash', t('delete'), async () => {
        p.files = p.files.filter((x) => x.id !== f.id);
        await save();
        await rag.removeDocument(p.id, f.id);
        renderProject(p);
      })));
    indexText(p.id, f.id).then((text) => { indexState.textContent = text; });
  }
  // The share of the context the documents take, and how the model reads them.
  const contextTokens = roleCapabilities('general').context || 0;
  const totalTokens = (p.files || []).reduce((n, f) => n + estimateTokens(f.text), 0);
  const capacity = h('p', { class: 'help' });
  if ((p.files || []).length) {
    const mode = knowledgeMode(p, contextTokens);
    const share = contextTokens ? Math.round(100 * totalTokens / contextTokens) : 0;
    capacity.textContent = t(mode === 'whole' ? 'knowledgeWhole' : 'knowledgeSearch',
      { n: formatNumber(totalTokens), p: share }) + (rag.embeddingModel() ? '' : ' ' + t('knowledgeNoVectors'));
  }
  const convList = h('div', { class: 'list' });
  for (const c of convs) {
    convList.append(h('div', { class: 'row click', onclick: () => go('#/c/' + c.id) },
      h('span', { html: icon('chat') }), h('span', { class: 'label' }, c.title || t('untitled')),
      h('span', { class: 'faint' }, new Date(c.updated).toLocaleDateString(language()))));
  }
  if (!convs.length) convList.append(h('div', { class: 'empty-note' }, t('noDiscussions')));

  main.replaceChildren(h('div', { class: 'page' },
    h('h2', null, h('span', { class: 'dot', style: { width: '14px', height: '14px', borderRadius: '50%', display: 'inline-block', background: p.color } }), p.name),
    h('p', { class: 'sub' }, p.description || ''),
    h('div', { class: 'buttons', style: { marginBottom: '16px' } },
      h('button', { class: 'button primary', type: 'button', onclick: () => newChat(p.id), html: icon('plus') + '<span>' + escapeHtml(t('startInProject')) + '</span>' })),
    h('div', { class: 'card' },
      h('div', { class: 'field' }, h('label', null, t('projectName')), name),
      h('div', { class: 'field' }, h('label', null, t('description')), description),
      h('div', { class: 'field' }, h('label', null, t('color')), colors)),
    h('div', { class: 'card' }, h('h3', { html: icon('edit') + escapeHtml(t('instructions')) }),
      h('p', { class: 'help' }, t('instructionsHelp')), h('div', { class: 'field' }, instructions)),
    h('div', { class: 'card' }, h('h3', { html: icon('book') + escapeHtml(t('knowledge')) }),
      h('p', { class: 'help' }, t('knowledgeHelp')), capacity, fileList,
      h('div', { class: 'buttons', style: { marginTop: '10px' } },
        h('button', { class: 'button', type: 'button', onclick: () => fileInput.click(), html: icon('upload') + '<span>' + escapeHtml(t('addFiles')) + '</span>' }),
        fileInput)),
    h('div', { class: 'card' }, h('h3', { html: icon('chat') + escapeHtml(t('projectDiscussions')) }), convList),
    h('div', { class: 'buttons' },
      h('button', {
        class: 'button danger', type: 'button', html: icon('trash') + '<span>' + escapeHtml(t('deleteProject')) + '</span>',
        onclick: async () => {
          if (!await confirmDialog(t('deleteProject'), t('deleteProjectText', { name: p.name }))) return;
          await store.deleteProject(p.id);
          await reloadLists();
          go('#/');
        }
      }))));
}

// Files.
async function readFiles(files) {
  const out = [];
  for (const f of files) {
    if (f.size > maxFileBytes) {
      toast(t('tooLarge', { name: f.name, size: formatSize(f.size), limit: formatSize(maxFileBytes) }), true);
      continue;
    }
    try {
      const r = await extractText(f);
      out.push({ id: store.uid(), name: f.name, size: f.size, format: r.format, text: r.text, added: Date.now() });
    } catch (e) {
      if (e.message === 'unsupported') toast(t('unsupported', { name: f.name }), true);
      else if (e.message === 'legacy') toast(t('legacyFormat', { name: f.name }), true);
      else if (e.message === 'too-large') toast(t('tooLarge', { name: f.name, size: formatSize(f.size), limit: formatSize(maxFileBytes) }), true);
      else toast(t('readFailed', { name: f.name, error: e.message }), true);
    }
  }
  return out;
}

// Conversations.
function newConversation(projectId) {
  return {
    id: store.uid(), title: '', projectId: projectId || null, created: Date.now(), updated: Date.now(), pinned: false,
    role: state.settings.defaultRole || 'general', reasoning: !!state.settings.lastReasoning, agent: !!state.settings.lastAgent,
    messages: [], incognito: state.incognito
  };
}

function newChat(projectId) {
  if (state.busy) stopGeneration();
  state.current = newConversation(projectId);
  state.pending = [];
  state.view = 'chat';
  if (location.hash !== '#/') history.pushState(null, '', '#/');
  renderAll();
  ui.input.focus();
}

function go(hash) {
  if (location.hash === hash) route();
  else location.hash = hash;
}

async function route() {
  closeMenu();
  const hash = location.hash || '#/';
  let m;
  if ((m = /^#\/c\/([0-9a-f]+)$/.exec(hash))) {
    if (!(state.current && state.current.id === m[1])) {
      if (state.busy) stopGeneration();
      const c = await store.getConversation(m[1]);
      if (!c) { go('#/'); return; }
      state.current = c;
      state.incognito = false;
      state.pending = [];
    }
    state.view = 'chat';
  } else if ((m = /^#\/p\/([0-9a-f]+)$/.exec(hash))) {
    state.view = 'project';
    state.projectId = m[1];
  } else {
    state.view = 'chat';
    if (!state.current || state.current.messages.length || (state.current.incognito !== state.incognito)) {
      if (state.busy) stopGeneration();
      state.current = newConversation(null);
    }
  }
  renderAll();
}

// Main area.
function renderAll() {
  renderSidebar();
  renderTopbar();
  renderMain();
  renderComposerState();
  renderStatusLine();
  ui.incognitoButton.classList.toggle('on', state.incognito);
  ui.banner.classList.toggle('hidden', !(state.view === 'chat' && state.current && state.current.incognito));
}

function renderTopbar() {
  const bar = ui.topbar;
  bar.replaceChildren();
  bar.append(iconButton('sidebar', t('collapse'), toggleSidebar));
  if (state.view === 'project') {
    const p = projectById(state.projectId);
    bar.append(h('div', { class: 'title' }, p ? p.name : ''));
    return;
  }
  const c = state.current;
  const title = h('div', { class: 'title', contenteditable: c && c.messages.length ? 'true' : 'false', spellcheck: 'false' },
    c && c.title ? c.title : t('untitled'));
  title.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); title.blur(); } });
  title.addEventListener('blur', async () => {
    if (!c || !c.messages.length) return;
    const v = title.textContent.trim();
    if (v && v !== c.title) {
      c.title = v;
      await persist(c);
      renderSidebar();
    }
  });
  bar.append(title);
  if (c && c.incognito) bar.append(h('span', { class: 'chip incognito', html: icon('incognito') + escapeHtml(t('incognito')) }));
  if (c && c.projectId) {
    const p = projectById(c.projectId);
    if (p) {
      bar.append(h('button', {
        class: 'chip', type: 'button', title: t('moveToProject'),
        onclick: (e) => moveMenu(c, e.currentTarget)
      }, h('span', { class: 'swatch', style: { background: p.color } }), p.name));
    }
  }
  if (c) {
    const model = roleModel(c.role);
    bar.append(h('div', { class: 'role-picker' }, h('button', {
      class: 'chip', type: 'button', title: model ? t('roleServedBy', { model: modelName(model) }) : t('noModel'),
      onclick: (e) => roleMenu(e.currentTarget)
    }, h('span', { html: icon(roleIcons[c.role]) }), t('role_' + c.role), h('span', { html: icon('chevron') }))));
    if (c.messages.length) bar.append(iconButton('more', '…', (e) => conversationMenu(c, e.currentTarget)));
  }
}

function roleMenu(anchor) {
  const c = state.current;
  const entries = [];
  for (const r of roles) {
    const own = state.status && state.status.active && state.status.active[r];
    const model = roleModel(r);
    entries.push({
      label: t('role_' + r), icon: roleIcons[r], checked: c.role === r,
      desc: t('roleHint_' + r) + (model ? ' · ' + (own ? modelName(model) : t('roleShared')) : ''),
      action: async () => {
        c.role = r;
        await saveSetting('defaultRole', r);
        if (c.messages.length) await persist(c);
        renderTopbar();
        renderComposerState();
        if (!c.messages.length) renderMain();
        renderStatusLine();
      }
    });
  }
  showMenu(anchor, entries);
}

function toggleSidebar() {
  const open = ui.sidebar.classList.toggle('collapsed') === false;
  saveSetting('sidebar', open);
}

function renderMain() {
  if (state.view === 'project') {
    const p = projectById(state.projectId);
    ui.composerWrap.classList.add('hidden');
    if (p) renderProject(p);
    else ui.content.replaceChildren();
    return;
  }
  ui.composerWrap.classList.remove('hidden');
  const c = state.current;
  if (!c.messages.length) {
    renderWelcome();
    return;
  }
  const thread = h('div', { class: 'thread' });
  c.messages.forEach((m, i) => thread.append(messageElement(c, m, i, i === c.messages.length - 1)));
  ui.content.replaceChildren(thread);
  ui.thread = thread;
  scrollToEnd(true);
}

function renderWelcome() {
  const c = state.current;
  const name = state.settings.userName ? ', ' + state.settings.userName : '';
  const roleCards = h('div', { class: 'role-cards' });
  for (const r of roles) {
    roleCards.append(h('button', {
      type: 'button', class: 'role-card' + (c.role === r ? ' active' : ''), title: t('roleHint_' + r),
      onclick: async () => { c.role = r; await saveSetting('defaultRole', r); renderTopbar(); renderComposerState(); renderWelcome(); renderStatusLine(); }
    }, h('span', { html: icon(roleIcons[r]) }), t('role_' + r)));
  }
  const suggestions = h('div', { class: 'suggestions' });
  for (let i = 1; i <= 4; ++i) {
    const [title, prompt] = t('suggest_' + c.role + '_' + i).split('|');
    suggestions.append(h('div', {
      class: 'suggestion', onclick: () => { ui.input.value = prompt; autoGrow(); ui.input.focus(); }
    }, h('b', null, title), h('span', null, prompt)));
  }
  const p = c.projectId ? projectById(c.projectId) : null;
  const notice = !state.statusError && state.status && !roleModel(c.role) ? h('div', { class: 'error-box' }, t('noModel')) : null;
  ui.content.replaceChildren(h('div', { class: 'welcome' },
    h('img', { class: 'hero', src: 'img/albireo-mark.svg', alt: '' }),
    h('h1', { html: escapeHtml(t('welcome', { name })).replace('Albireo', '<span>Albireo</span>') }),
    h('p', null, p ? t('newProjectDiscussion', { name: p.name }) : t('welcomeSub')),
    notice, roleCards, suggestions));
  ui.thread = null;
}

function scrollToEnd(force) {
  const s = ui.content;
  const near = s.scrollHeight - s.scrollTop - s.clientHeight < 160;
  if (force || near) s.scrollTop = s.scrollHeight;
}

function messageElement(c, m, index, last) {
  if (m.role === 'user') return userElement(c, m, index, last);
  return assistantElement(c, m, index, last);
}

// The badge of a document: its format on the color of its family.
function formatBadge(a) {
  const ext = extension(a.name || '') || String(a.format || '').toLowerCase();
  const family = /^(docx?|docm|dotx|odt|rtf)$/.test(ext) ? 'word' : /^(xlsx?|xlsm|xltx|ods|csv|tsv)$/.test(ext) ? 'sheet'
    : /^(pptx?|pptm|ppsx|potx|odp)$/.test(ext) ? 'slides' : ext === 'pdf' ? 'pdf'
    : /^(md|markdown|txt|rst)$/.test(ext) ? 'text' : 'code';
  const label = (ext || 'txt').slice(0, 4).toUpperCase();
  return h('span', { class: 'format-badge ' + family, 'aria-hidden': 'true' }, label);
}

function attachmentChip(a, removable, onremove) {
  return h('span', { class: 'attachment', title: a.name },
    formatBadge(a), h('span', { class: 'name' }, a.name),
    h('span', { class: 'size' }, (a.format || '') + (a.size ? ' · ' + formatSize(a.size) : '')
      + (a.mode === 'search' ? ' · ' + t('searchedDocument') : '')),
    removable ? h('button', { type: 'button', title: t('removeAttachment'), 'aria-label': t('removeAttachment'),
      html: icon('close'), onclick: onremove }) : null);
}

function userElement(c, m, index, last) {
  const el = h('div', { class: 'message user' + (last ? ' last' : '') });
  if (m.attachments && m.attachments.length) {
    el.append(h('div', { class: 'attachments' }, m.attachments.map((a) => attachmentChip(a))));
  }
  const bubble = h('div', { class: 'bubble' }, m.content);
  el.append(bubble);
  el.append(h('div', { class: 'meta' },
    iconButton('copy', t('copy'), () => { navigator.clipboard.writeText(m.content); toast(t('copied')); }),
    iconButton('edit', t('edit'), () => editMessage(c, m, index, el))));
  return el;
}

// Edits a question: its text, and its attached documents, which can be removed or added.
function editMessage(c, m, index, el) {
  if (state.busy) return;
  let attachments = (m.attachments || []).slice();
  const area = h('textarea', { style: { width: '100%', minHeight: '90px', borderRadius: '12px', padding: '10px', border: '1px solid var(--border-strong)', background: 'var(--bg-input)' } });
  area.value = m.content;
  const chips = h('div', { class: 'attachments', style: { justifyContent: 'flex-start', marginBottom: '6px' } });
  const showChips = () => {
    chips.replaceChildren(...attachments.map((a) => attachmentChip(a, true, () => {
      attachments = attachments.filter((x) => x !== a);
      showChips();
    })));
  };
  showChips();
  const fileInput = h('input', { type: 'file', multiple: true, class: 'hidden' });
  fileInput.addEventListener('change', async () => {
    attachments = attachments.concat(await readFiles(Array.from(fileInput.files)));
    fileInput.value = '';
    showChips();
  });
  const box = h('div', { style: { width: '85%' } }, chips, area, h('div', { class: 'buttons', style: { justifyContent: 'flex-end', marginTop: '6px' } },
    h('button', { class: 'button', type: 'button', onclick: () => fileInput.click(), html: icon('clip') + '<span>' + escapeHtml(t('attach')) + '</span>' }),
    fileInput,
    h('span', { class: 'grow' }),
    h('button', { class: 'button', type: 'button', onclick: () => renderMain() }, t('cancel')),
    h('button', {
      class: 'button primary', type: 'button',
      onclick: async () => {
        const text = area.value.trim();
        if (!text && !attachments.length) return;
        c.messages = c.messages.slice(0, index);
        if ((c.contextStart || 0) > index) c.contextStart = 0;
        // The documents no longer attached leave the search of the discussion.
        const kept = new Set(attachments.map((a) => a.id));
        for (const a of m.attachments || []) if (!kept.has(a.id) && a.mode === 'search') rag.removeDocument(c.id, a.id);
        await sendMessage(text, attachments);
      }
    }, t('send'))));
  box.addEventListener('dragover', (e) => e.preventDefault());
  box.addEventListener('drop', async (e) => {
    e.preventDefault();
    if (!e.dataTransfer || !e.dataTransfer.files.length) return;
    attachments = attachments.concat(await readFiles(Array.from(e.dataTransfer.files)));
    showChips();
  });
  el.replaceChildren(box);
  area.focus();
}

function assistantElement(c, m, index, last) {
  const el = h('div', { class: 'message assistant' + (last ? ' last' : '') });
  const bubble = h('div', { class: 'bubble' });
  el.append(bubble);
  fillAssistant(bubble, m, state.busy && last);
  const meta = h('div', { class: 'meta' });
  const text = (m.steps || []).map((s) => s.content || '').join('\n\n').trim();
  meta.append(iconButton('copy', t('copy'), () => { navigator.clipboard.writeText(text); toast(t('copied')); }));
  if (last) meta.append(iconButton('refresh', t('regenerate'), () => regenerate(c)));
  const stats = statsText(m);
  if (stats) meta.append(h('span', { class: 'stats' }, stats));
  el.append(meta);
  el.dataset.index = index;
  return el;
}

function statsText(m) {
  const parts = [];
  if (m.model) parts.push(modelName(m.model));
  if (m.usage && m.usage.completion_tokens) parts.push(formatNumber(m.usage.completion_tokens) + ' tokens');
  if (m.timings && m.timings.generation_per_second) parts.push(t('tokensPerSecond', { n: m.timings.generation_per_second.toFixed(1) }));
  return parts.join(' · ');
}

// Fills the body of an answer: per step, its reasoning, its text and its tool calls.
function fillAssistant(bubble, m, live) {
  bubble.replaceChildren();
  const steps = m.steps || [];
  steps.forEach((s, k) => {
    const activeStep = live && k === steps.length - 1;
    if (s.reasoning) bubble.append(thinkingElement(s, activeStep && !s.content && !(s.toolCalls || []).length));
    if (s.content) {
      const md = h('div', { class: 'md' + (activeStep && live ? ' cursor-host' : ''), html: renderMarkdown(s.content) });
      attachCodeButtons(md, t('copy'));
      bubble.append(md);
    }
    for (const tc of s.toolCalls || []) bubble.append(toolElement(tc));
  });
  if (live && m.progress && m.progress.prompt_total > 256 && m.progress.prompt_done < m.progress.prompt_total) {
    const pct = Math.floor(100 * m.progress.prompt_done / m.progress.prompt_total);
    bubble.append(h('div', { class: 'progress-line' }, h('span', { class: 'spinner' }),
      t('readingContext', { p: pct }), h('div', { class: 'bar' }, h('div', { style: { width: pct + '%' } }))));
  } else if (live && !steps.some((s) => s.content || s.reasoning || (s.toolCalls || []).length)) {
    bubble.append(h('div', { class: 'progress-line' }, h('span', { class: 'spinner' }), h('span', { class: 'shimmer' }, t('thinking') + '…')));
  }
  if (m.trimmed) bubble.append(h('div', { class: 'progress-line' }, h('span', { html: icon('info') }), t('contextTrimmed')));
  if (m.error) bubble.append(h('div', { class: 'error-box' }, t('errorPrefix', { message: m.error })));
  if (m.cancelled) bubble.append(h('div', { class: 'progress-line' }, t('cancelled')));
  if (!live && !m.error && !m.cancelled && !steps.some((s) => s.content || (s.toolCalls || []).length)) {
    bubble.append(h('div', { class: 'progress-line' }, t('emptyAnswer')));
  }
  if (m.agentLimit) bubble.append(h('div', { class: 'progress-line' }, t('agentLimit', { n: maxAgentSteps })));
}

function thinkingElement(s, live) {
  const d = h('details', { class: 'thinking' + (live ? ' live' : '') });
  // Open while the model thinks, unless the user closed it; closed afterwards, unless the
  // user opened it.
  const open = live ? !s.collapsed : !!s.expanded;
  if (open) d.open = true;
  d.addEventListener('toggle', () => {
    if (d.open === open) return;
    if (live) s.collapsed = !d.open;
    else s.expanded = d.open;
  });
  let label;
  if (live) label = h('span', { class: 'shimmer' }, t('thinking') + '…' + (s.reasoningStart ? ' ' + formatDuration(Date.now() - s.reasoningStart) : ''));
  else label = h('span', null, s.reasoningMs ? t('thoughtFor', { s: formatDuration(s.reasoningMs) }) : t('thoughtBriefly'));
  d.append(h('summary', null, h('span', { html: icon('spark', 'spark') }), label, h('span', { html: icon('chevron', 'chev') })),
    h('div', { class: 'thought' }, s.reasoning));
  return d;
}

const toolIcons = {
  calculator: 'calc', run_javascript: 'terminal', search_documents: 'search', read_document: 'book',
  search_discussions: 'history', current_datetime: 'clock', create_file: 'file'
};

function toolElement(tc) {
  const d = h('details', { class: 'tool-call' });
  const open = !!tc.open;
  if (open) d.open = true;
  d.addEventListener('toggle', () => { if (d.open !== open) tc.open = d.open; });
  let brief = '';
  try {
    const a = JSON.parse(tc.arguments || '{}');
    brief = a.expression || a.query || a.name || (a.code ? a.code.split('\n')[0] : '') || '';
  } catch (e) {
    brief = tc.arguments || '';
  }
  const stateLabel = tc.status === 'running' ? h('span', { class: 'spinner' })
    : h('span', { class: 'state ' + (tc.ok === false ? 'error' : 'ok') }, tc.ok === false ? t('toolError') : t('toolDone'));
  let args = tc.arguments || '';
  try { args = JSON.stringify(JSON.parse(args), null, 2); } catch (e) { /* kept as sent */ }
  d.append(h('summary', null, h('span', { html: icon(toolIcons[tc.name] || 'tool', 'tool-icon') }),
    h('span', { class: 'tool-name' }, tc.name), h('span', { class: 'tool-brief' }, brief), stateLabel),
  h('div', { class: 'io' }, h('h6', null, t('toolInput')), h('pre', null, h('code', null, args)),
    tc.result != null ? [h('h6', null, t('toolOutput')), h('pre', null, h('code', null, tc.result))] : null));
  const wrap = h('div', null, d);
  if (tc.file) {
    wrap.append(h('div', { class: 'attachments', style: { justifyContent: 'flex-start' } },
      h('span', {
        class: 'attachment file-out', title: t('download'),
        onclick: () => download(tc.file.name, tc.file.content, mimeOf(tc.file.name))
      }, h('span', { html: icon('download') }), h('span', { class: 'name' }, tc.file.name),
      h('span', { class: 'size' }, t('generatedFile')))));
  }
  return wrap;
}

// Composer.
function renderComposerState() {
  if (!ui.input) return;
  const c = state.current;
  const caps = roleCapabilities(c ? c.role : 'general');
  ui.reasonButton.classList.toggle('on', !!(c && c.reasoning && caps.reasoning));
  ui.reasonButton.disabled = !caps.reasoning;
  ui.agentButton.classList.toggle('on', !!(c && c.agent && caps.tools));
  ui.agentButton.disabled = !caps.tools;
  ui.sendButton.classList.toggle('stop', state.busy);
  ui.sendButton.innerHTML = icon(state.busy ? 'stop' : 'send');
  ui.sendButton.title = state.busy ? t('stop') : t('send');
  ui.sendButton.disabled = !state.busy && !ui.input.value.trim() && !state.pending.length;
  const p = c && c.projectId ? projectById(c.projectId) : null;
  ui.input.placeholder = p ? t('placeholderProject', { project: p.name }) : t('placeholder');
  ui.pendingBox.replaceChildren(...state.pending.map((a) => attachmentChip(a, true, () => {
    state.pending = state.pending.filter((x) => x !== a);
    renderComposerState();
  })));
  ui.pendingBox.classList.toggle('hidden', !state.pending.length);
}

function autoGrow() {
  ui.input.style.height = 'auto';
  ui.input.style.height = Math.min(ui.input.scrollHeight, 260) + 'px';
  ui.sendButton.disabled = !state.busy && !ui.input.value.trim() && !state.pending.length;
}

async function addPending(files) {
  const added = await readFiles(files);
  state.pending = state.pending.concat(added);
  renderComposerState();
}

function stopGeneration() {
  if (state.abort) state.abort.abort();
}

async function onSend() {
  if (state.busy) { stopGeneration(); return; }
  const text = ui.input.value.trim();
  if (!text && !state.pending.length) return;
  const attachments = state.pending;
  state.pending = [];
  ui.input.value = '';
  autoGrow();
  await sendMessage(text, attachments);
}

async function sendMessage(text, attachments) {
  const c = state.current;
  const m = { id: store.uid(), role: 'user', content: text, attachments: attachments || [], created: Date.now() };
  c.messages.push(m);
  await prepareQuestion(c, m);
  if (!c.title) c.title = (text || (attachments[0] && attachments[0].name) || t('untitled')).replace(/\s+/g, ' ').slice(0, 70);
  c.updated = Date.now();
  await persist(c);
  if (!c.incognito && location.hash !== '#/c/' + c.id) history.replaceState(null, '', '#/c/' + c.id);
  renderSidebar();
  renderTopbar();
  await runTurn(c);
}

async function regenerate(c) {
  if (state.busy) return;
  while (c.messages.length && c.messages[c.messages.length - 1].role === 'assistant') c.messages.pop();
  if (!c.messages.length) return;
  await runTurn(c);
}

/*
    Prepares a question once, when it is sent: its large documents are indexed for the
    search of the discussion, and the excerpts of the searched documents closest to it are
    kept with it, so that the prompt of the following questions starts the same way.
*/
async function prepareQuestion(c, m) {
  const contextTokens = roleCapabilities(c.role).context || 8192;
  const owners = [];
  for (const a of m.attachments || []) {
    if (!a.mode) a.mode = estimateTokens(a.text) <= attachmentShare * contextTokens ? 'whole' : 'search';
    if (a.mode === 'search') {
      // Indexed at once, beside the documents of the projects that may be in progress.
      toast(t('indexingDocument', { name: a.name }));
      await rag.indexDocument(c.id, a, { persistent: !c.incognito }).catch(() => {});
    }
  }
  for (const x of c.messages) {
    if (x.role === 'user' && (x.attachments || []).some((a) => a.mode === 'search')) {
      owners.push(c.id);
      break;
    }
  }
  const p = c.projectId ? projectById(c.projectId) : null;
  if (p && knowledgeMode(p, contextTokens) === 'search') owners.unshift(p.id);
  m.context = '';
  if (!owners.length) return;
  // A short question takes the previous one with it: "and in 2025?" is searched with what
  // it continues.
  let question = m.content || '';
  if (question.length < 80) {
    const users = c.messages.filter((x) => x.role === 'user' && x !== m);
    if (users.length) question = users[users.length - 1].content + '\n' + question;
  }
  if (!question.trim()) question = (m.attachments || []).map((a) => a.name).join(' ');
  try {
    const hits = await rag.search(owners, question, { budget: Math.floor(excerptShare * contextTokens * 3.6) });
    if (hits.length) {
      m.context = '<excerpts>\n' + hits.map((x) => '<excerpt document="' + x.name + '"'
        + (x.where ? ' location="' + x.where + '"' : '') + '>\n' + x.text + '\n</excerpt>').join('\n') + '\n</excerpts>';
    }
  } catch (e) {
    m.context = '';
  }
}

/*
    The passages of the documents of a discussion and of its project closest to a query: the
    documents indexed by their passages (rag.js), and those read whole in the messages by
    their words.
*/
async function searchDocumentsOf(c, query, limit) {
  const owners = [];
  const p = c.projectId ? projectById(c.projectId) : null;
  if (p && (p.files || []).length) owners.push(p.id);
  const whole = [];
  let searched = false;
  for (const m of c.messages) {
    if (m.role !== 'user') continue;
    for (const a of m.attachments || []) {
      if (a.mode === 'search') searched = true;
      else whole.push({ name: a.name, text: a.text });
    }
  }
  if (searched) owners.push(c.id);
  const hits = owners.length ? await rag.search(owners, query, { limit, budget: 1500 * limit }) : [];
  const words = whole.length ? searchPassages(whole, query, limit).map((x) => ({ name: x.name, where: '', text: x.text })) : [];
  return hits.concat(words).slice(0, limit);
}

// The context the model receives.
function documentsOf(c) {
  const docs = [];
  const p = c.projectId ? projectById(c.projectId) : null;
  if (p) for (const f of p.files || []) docs.push({ name: f.name, text: f.text });
  for (const m of c.messages) if (m.role === 'user') for (const a of m.attachments || []) docs.push({ name: a.name, text: a.text });
  return docs;
}

function systemPrompt(c, contextTokens) {
  const parts = [];
  const now = new Date();
  parts.push('Current date: ' + now.toLocaleDateString('en-CA') + ' (' + now.toLocaleDateString(language(), { weekday: 'long' }) + ').');
  if (state.settings.userName) parts.push('The user is named ' + state.settings.userName + '.');
  if (state.settings.systemPrompt) parts.push(state.settings.systemPrompt);
  const p = c.projectId ? projectById(c.projectId) : null;
  if (p) {
    parts.push('This discussion belongs to the project "' + p.name + '".' + (p.description ? ' ' + p.description : ''));
    if (p.instructions) parts.push('Instructions of the project:\n' + p.instructions);
    const files = p.files || [];
    // The documents enter whole while they fit in their share of the context; beyond, the
    // excerpts closest to each question come with it.  This message stays the same from one
    // question to the next.
    if (knowledgeMode(p, contextTokens) === 'whole') {
      parts.push('Documents of the project:\n' + files.map((f) => '<document name="' + f.name + '">\n' + f.text + '\n</document>').join('\n'));
    } else if (files.length) {
      parts.push('The project holds ' + files.length + ' documents, too long to be read whole: '
        + files.map((f) => f.name + ' (' + f.format + ')').join(', ') + '. The excerpts of these documents closest to '
        + 'each question come with it, in <excerpts>; say so when they do not hold the answer.'
        + (c.agent ? ' Use search_documents, document_outline and read_document to read more of them.' : ''));
    }
  }
  if (c.agent) parts.push(tools.agentInstructions);
  return parts.join('\n\n');
}

function userContent(m) {
  let s = '';
  for (const a of m.attachments || []) {
    if (a.mode === 'search') {
      s += '<document name="' + a.name + '" format="' + (a.format || '') + '">\n(Too long to be read whole: its excerpts '
        + 'closest to the questions come with them.)\n</document>\n\n';
    } else {
      s += '<document name="' + a.name + '">\n' + a.text + '\n</document>\n\n';
    }
  }
  if (m.context) s += m.context + '\n\n';
  return s + (m.content || '');
}

function apiMessages(c, system) {
  const out = [];
  if (system) out.push({ role: 'system', content: system });
  // The exchanges before contextStart left the context in an earlier question.
  for (const m of c.messages.slice(c.contextStart || 0)) {
    if (m.role === 'user') {
      out.push({ role: 'user', content: userContent(m) });
      continue;
    }
    for (const s of m.steps || []) {
      const msg = { role: 'assistant', content: s.content || '' };
      if (s.reasoning) msg.reasoning_content = s.reasoning;
      const calls = (s.toolCalls || []).filter((tc) => tc.result != null);
      if (calls.length) {
        msg.tool_calls = calls.map((tc) => ({ id: tc.id, type: 'function', function: { name: tc.name, arguments: tc.arguments || '{}' } }));
      }
      if (!msg.content && !msg.tool_calls) continue;
      out.push(msg);
      for (const tc of calls) out.push({ role: 'tool', tool_call_id: tc.id, name: tc.name, content: tc.result });
    }
  }
  return out;
}

/*
    Fits the conversation in the context of the model.  When it no longer fits, the oldest
    exchanges leave in one step, until the conversation takes keptShare of its room, and
    the point where it now starts stays with the conversation: the following questions
    start from that same point, so that the engine reuses what it already read, and the
    next cut comes only after several questions.  The system message and the last
    question always stay.  Returns {messages, trimmed, context, tokens}.
*/
async function fitContext(c, build, toolDefs, reserve) {
  let trimmed = false;
  for (let attempt = 0; attempt < 6; ++attempt) {
    const list = build();
    let r;
    try {
      r = await api.count('albireo-' + c.role, list, toolDefs);
    } catch (e) {
      return { messages: list, trimmed, context: 0 };
    }
    if (!r || typeof r.tokens !== 'number' || !r.context) return { messages: list, trimmed, context: 0 };
    const room = r.context - Math.min(reserve, Math.floor(r.context / 2));
    if (r.tokens <= room) return { messages: list, trimmed, context: r.context, tokens: r.tokens };
    // The questions after the current start, the last one excepted, are where it may move.
    const from = c.contextStart || 0;
    const users = [];
    for (let i = from + 1; i < c.messages.length; ++i) if (c.messages[i].role === 'user') users.push(i);
    const last = users.pop();
    if (last == null) return { messages: list, trimmed, context: r.context, tokens: r.tokens, overflow: true };
    // The tokens of each candidate start, from the characters scaled to the measure.
    const size = (msgs) => msgs.reduce((n, m) => n + JSON.stringify(m).length, 0);
    const scale = r.tokens / Math.max(1, size(list));
    let next = last;
    for (const i of users.concat([last])) {
      c.contextStart = i;
      if (size(build()) * scale <= keptShare * room) {
        next = i;
        break;
      }
    }
    c.contextStart = next;
    trimmed = true;
  }
  return { messages: build(), trimmed, context: 0 };
}

let renderPending = false;

function scheduleRender(el, m) {
  if (renderPending) return;
  renderPending = true;
  requestAnimationFrame(() => {
    renderPending = false;
    const bubble = el.querySelector('.bubble');
    const thought = bubble.querySelector('.thinking.live .thought');
    const keepScroll = thought ? thought.scrollHeight - thought.scrollTop - thought.clientHeight < 40 : true;
    fillAssistant(bubble, m, true);
    const t2 = bubble.querySelector('.thinking.live .thought');
    if (t2 && keepScroll) t2.scrollTop = t2.scrollHeight;
    scrollToEnd(false);
  });
}

async function runTurn(c) {
  const model = roleModel(c.role);
  const turn = { id: store.uid(), role: 'assistant', steps: [], created: Date.now(), model: model || '' };
  c.messages.push(turn);
  state.busy = true;
  state.abort = new AbortController();
  renderMain();
  renderComposerState();
  const el = ui.thread.lastChild;
  const caps = roleCapabilities(c.role);
  const useTools = c.agent && caps.tools;
  const toolDefs = useTools ? tools.definitions : null;
  try {
    if (!model) throw new Error(t('noModel'));
    for (let step = 0; step < maxAgentSteps; ++step) {
      const s = { reasoning: '', content: '', toolCalls: [] };
      turn.steps.push(s);
      const system = systemPrompt(c, caps.context || 8192);
      // The conversation, with the steps of this answer already done (the agent's).
      const build = () => apiMessages(Object.assign({}, c, {
        messages: c.messages.slice(0, -1).concat([Object.assign({}, turn, { steps: turn.steps.slice(0, -1) })])
      }), system);
      const fitted = await fitContext(c, build, toolDefs, state.settings.maxTokens || 2048);
      if (fitted.trimmed) turn.trimmed = true;
      const body = {
        model: 'albireo-' + c.role, messages: fitted.messages, reasoning: !!(c.reasoning && caps.reasoning)
      };
      if (toolDefs) body.tools = toolDefs;
      if (state.settings.temperature != null) body.temperature = state.settings.temperature;
      if (state.settings.topP != null) body.top_p = state.settings.topP;
      if (state.settings.maxTokens > 0) body.max_tokens = state.settings.maxTokens;
      let finish = null;
      for await (const ev of api.streamChat(body, state.abort.signal)) {
        if (ev.type === 'progress') {
          turn.progress = ev.data;
          scheduleRender(el, turn);
          continue;
        }
        const chunk = ev.data;
        if (chunk.usage) {
          turn.usage = turn.usage ? {
            prompt_tokens: chunk.usage.prompt_tokens, completion_tokens: (turn.usage.completion_tokens || 0) + chunk.usage.completion_tokens
          } : chunk.usage;
          turn.timings = chunk.timings || turn.timings;
        }
        const choice = chunk.choices && chunk.choices[0];
        if (!choice) continue;
        const d = choice.delta || {};
        if (d.reasoning_content) {
          if (!s.reasoningStart) s.reasoningStart = Date.now();
          s.reasoning += d.reasoning_content;
        }
        if (d.content) {
          if (s.reasoningStart && !s.reasoningMs) s.reasoningMs = Date.now() - s.reasoningStart;
          s.content += d.content;
        }
        for (const tc of d.tool_calls || []) {
          if (s.reasoningStart && !s.reasoningMs) s.reasoningMs = Date.now() - s.reasoningStart;
          const existing = tc.index != null ? s.toolCalls.find((x) => x.index === tc.index) : null;
          if (existing) existing.arguments += (tc.function && tc.function.arguments) || '';
          else s.toolCalls.push({
            index: tc.index, id: tc.id || ('call_' + store.uid()), name: tc.function ? tc.function.name : '',
            arguments: (tc.function && tc.function.arguments) || '', status: 'pending'
          });
        }
        if (choice.finish_reason) finish = choice.finish_reason;
        turn.progress = null;
        scheduleRender(el, turn);
      }
      if (s.reasoningStart && !s.reasoningMs) s.reasoningMs = Date.now() - s.reasoningStart;
      delete s.reasoningStart;
      if (!(useTools && s.toolCalls.length && (finish === 'tool_calls' || finish === 'stop'))) break;
      // The tools of the step run, then the model continues with their results.
      for (const tc of s.toolCalls) {
        tc.status = 'running';
        scheduleRender(el, turn);
        const r = await tools.run(tc.name, tc.arguments, {
          documents: documentsOf(c),
          searchDocuments: (query, limit) => searchDocumentsOf(c, query, limit),
          searchDiscussions: async (q) => searchConversations(q, 6).filter((x) => x.conversation.id !== c.id).map((x) => ({
            title: x.conversation.title, date: new Date(x.conversation.updated).toLocaleDateString(), excerpt: x.excerpt
          })),
          createFile: (name, content) => ({ name, content })
        });
        tc.status = 'done';
        tc.ok = r.ok;
        tc.result = r.text;
        if (r.file) tc.file = r.file;
        delete tc.index;
        scheduleRender(el, turn);
      }
      if (step === maxAgentSteps - 1) turn.agentLimit = true;
    }
  } catch (e) {
    if (e.name === 'AbortError') turn.cancelled = true;
    else turn.error = e.message || String(e);
  } finally {
    for (const s of turn.steps) delete s.reasoningStart;
    turn.progress = null;
    state.busy = false;
    state.abort = null;
    c.updated = Date.now();
    await persist(c).catch(() => {});
    renderMain();
    renderComposerState();
    renderSidebar();
  }
}

// Dialogs.
function dialog(content, small) {
  return new Promise((resolve) => {
    const box = h('div', { class: 'dialog' + (small ? ' small' : ''), role: 'dialog', 'aria-modal': 'true' });
    const overlay = h('div', { class: 'overlay' }, box);
    const close = (v) => { overlay.remove(); document.removeEventListener('keydown', onKey); resolve(v); };
    const onKey = (e) => { if (e.key === 'Escape') close(null); };
    overlay.addEventListener('mousedown', (e) => { if (e.target === overlay) close(null); });
    document.addEventListener('keydown', onKey);
    content(box, close);
    document.body.append(overlay);
    const f = box.querySelector('input, textarea, button.primary');
    if (f) f.focus();
  });
}

function confirmDialog(title, text) {
  return dialog((box, close) => {
    box.append(h('h2', null, title), h('p', null, text), h('div', { class: 'actions' },
      h('button', { class: 'button', type: 'button', onclick: () => close(false) }, t('cancel')),
      h('button', { class: 'button primary', type: 'button', onclick: () => close(true) }, t('confirm'))));
  }, true);
}

function promptDialog(title, value) {
  return dialog((box, close) => {
    const input = h('input', { value: value || '', style: { width: '100%' } });
    input.addEventListener('keydown', (e) => { if (e.key === 'Enter') close(input.value); });
    box.append(h('h2', null, title), h('div', { class: 'field' }, input), h('div', { class: 'actions' },
      h('button', { class: 'button', type: 'button', onclick: () => close(null) }, t('cancel')),
      h('button', { class: 'button primary', type: 'button', onclick: () => close(input.value) }, t('save'))));
    setTimeout(() => input.select(), 0);
  }, true);
}

function settingsDialog(initialTab) {
  dialog((box, close) => {
    const tabs = ['general', 'model', 'data', 'about'];
    const tabBar = h('div', { class: 'tabs' });
    const body = h('div');
    const show = (tab) => {
      for (const b of tabBar.children) b.classList.toggle('on', b.dataset.tab === tab);
      body.replaceChildren();
      if (tab === 'general') generalTab(body, close);
      else if (tab === 'model') modelTab(body);
      else if (tab === 'data') dataTab(body, close);
      else aboutTab(body);
    };
    for (const tab of tabs) tabBar.append(h('button', { type: 'button', 'data-tab': tab, onclick: () => show(tab) }, tab === 'about' ? t('about') : t(tab)));
    box.append(h('h2', { html: icon('gear') + escapeHtml(t('settings')) }), tabBar, body,
      h('div', { class: 'actions' }, h('button', { class: 'button primary', type: 'button', onclick: () => close(true) }, t('close'))));
    show(initialTab || 'general');
  });
}

function field(label, control, help) {
  return h('div', { class: 'field' }, h('label', null, label), control, help ? h('p', { class: 'help', style: { margin: 0, color: 'var(--text-faint)', fontSize: '12.5px' } }, help) : null);
}

function generalTab(body, close) {
  const theme = h('select', null, ['auto', 'light', 'dark'].map((v) =>
    h('option', { value: v, selected: state.settings.theme === v }, t('theme' + v[0].toUpperCase() + v.slice(1)))));
  theme.addEventListener('change', async () => { await saveSetting('theme', theme.value); applyTheme(); });
  const lang = h('select', null, h('option', { value: 'auto', selected: state.settings.language === 'auto' }, t('languageAuto')),
    languages.map((l) => h('option', { value: l.code, selected: state.settings.language === l.code }, l.name)));
  lang.addEventListener('change', async () => {
    await saveSetting('language', lang.value);
    applyLanguage();
    close(true);
    state.catalog = await api.models(language()).catch(() => state.catalog);
    buildLayout();
    renderAll();
    settingsDialog('general');
  });
  const name = h('input', { value: state.settings.userName || '' });
  name.addEventListener('change', () => saveSetting('userName', name.value.trim()));
  const role = h('select', null, roles.map((r) => h('option', { value: r, selected: state.settings.defaultRole === r }, t('role_' + r))));
  role.addEventListener('change', () => saveSetting('defaultRole', role.value));
  const enter = h('input', { type: 'checkbox', checked: state.settings.sendOnEnter });
  enter.addEventListener('change', () => saveSetting('sendOnEnter', enter.checked));
  const sys = h('textarea', { rows: 4 });
  sys.value = state.settings.systemPrompt || '';
  sys.addEventListener('change', () => saveSetting('systemPrompt', sys.value));
  body.append(field(t('theme'), theme), field(t('language'), lang), field(t('yourName'), name, t('yourNameHelp')),
    field(t('defaultRole'), role), h('div', { class: 'field' }, h('label', { class: 'inline' }, enter, ' ', t('sendOnEnter'))),
    field(t('systemPrompt'), sys, t('systemPromptHelp')));
}

function modelTab(body) {
  const slider = (key, min, max, stepValue, def) => {
    const value = state.settings[key];
    const input = h('input', { type: 'range', min, max, step: stepValue, value: value == null ? def : value });
    const out = h('span', { style: { minWidth: '64px', textAlign: 'right' } }, value == null ? t('maxTokensAuto') : String(value));
    input.addEventListener('input', () => { out.textContent = input.value; });
    input.addEventListener('change', () => saveSetting(key, parseFloat(input.value)));
    const reset = h('button', {
      class: 'button', type: 'button', onclick: async () => { await saveSetting(key, null); out.textContent = t('maxTokensAuto'); input.value = def; }
    }, t('maxTokensAuto'));
    return h('div', { class: 'inline' }, input, out, reset);
  };
  const max = h('input', { type: 'number', min: 0, max: 131072, step: 256, value: state.settings.maxTokens || 0 });
  max.addEventListener('change', () => saveSetting('maxTokens', Math.max(0, parseInt(max.value, 10) || 0)));
  body.append(field(t('temperature'), slider('temperature', 0, 2, 0.05, 0.7), t('temperatureHelp')),
    field(t('topP'), slider('topP', 0.05, 1, 0.05, 0.95)), field(t('maxTokens'), max, '0 = ' + t('maxTokensAuto')));
}

function dataTab(body, close) {
  const input = h('input', { type: 'file', accept: '.json,application/json', class: 'hidden' });
  input.addEventListener('change', async () => {
    try {
      const data = JSON.parse(await input.files[0].text());
      const r = await store.importAll(data);
      await reloadLists();
      renderSidebar();
      toast(t('imported', { n: r.conversations, p: r.projects }));
      indexProjects();
    } catch (e) {
      toast(t('importFailed'), true);
    }
    input.value = '';
  });
  body.append(h('p', null, t('storageUsage', { n: state.conversations.length, p: state.projects.length })),
    h('div', { class: 'buttons' },
      h('button', {
        class: 'button', type: 'button', html: icon('download') + '<span>' + escapeHtml(t('exportAll')) + '</span>',
        onclick: async () => download('albireo-' + new Date().toISOString().slice(0, 10) + '.json', JSON.stringify(await store.exportAll()), 'application/json')
      }),
      h('button', { class: 'button', type: 'button', html: icon('upload') + '<span>' + escapeHtml(t('importAll')) + '</span>', onclick: () => input.click() }),
      input,
      h('button', {
        class: 'button danger', type: 'button', html: icon('trash') + '<span>' + escapeHtml(t('clearAll')) + '</span>',
        onclick: async () => {
          if (!await confirmDialog(t('clearAll'), t('clearAllText'))) return;
          await store.clearAll();
          await reloadLists();
          close(true);
          go('#/');
        }
      })));
}

function memoryBar(stats) {
  const v = stats && stats.memory;
  if (!v || !v.vram_budget) return null;
  const total = Math.max(v.vram_budget, 1);
  const pct = (x) => Math.min(100, 100 * x / total) + '%';
  return h('div', null, h('div', { class: 'mem-bar' },
    h('div', { style: { width: pct(v.device_bytes || 0), background: 'var(--blue)' }, title: t('vramResident') }),
    h('div', { style: { width: pct(v.scratch_bytes || 0), background: 'var(--gold)' } })),
  h('div', { class: 'legend' },
    h('span', { style: { '--c': 'var(--blue)' } }, t('vramResident') + ' ' + formatSize(v.device_bytes || 0)),
    h('span', { style: { '--c': 'var(--gold)' } }, 'workspace ' + formatSize(v.scratch_bytes || 0)),
    h('span', { style: { '--c': 'var(--text-faint)' } }, t('vramHost') + ' ' + formatSize(v.pinned_bytes || 0)),
    h('span', { style: { '--c': 'var(--incognito)' } }, t('vramStore') + ' ' + formatSize(v.store_bytes || 0))));
}

async function aboutTab(body) {
  const s = state.status || {};
  const art = h('div', { class: 'about-art' }, h('img', { src: 'img/about.svg', alt: '' }));
  const kv = h('div', { class: 'kv' });
  const row = (k, v) => kv.append(h('div', null, k), h('div', null, v));
  row(t('version'), (s.product || 'Albireo') + ' ' + (s.version || ''));
  row(t('engine'), s.engine ? (s.engine.variant || '-') + (s.engine.device ? ' (' + s.engine.device + ')' : '') : '-');
  if (s.host) {
    row(t('processor'), s.host.cpu.name + ' · ' + s.host.cpu.cores + ' / ' + s.host.cpu.threads);
    row(t('memory'), formatSize(s.host.memory.total));
  }
  const models = Object.entries(s.active || {}).map(([r, id]) => t('role_' + r) + ': ' + modelName(id)).join(' · ');
  row(t('models'), models || '-');
  if (s.usage) row(t('requests'), formatNumber(s.usage.requests) + ' · ' + t('tokensIn') + ' ' + formatNumber(s.usage.prompt_tokens) +
    ' · ' + t('tokensOut') + ' ' + formatNumber(s.usage.completion_tokens));
  body.append(art, h('p', null, t('aboutText')), kv);
  const mem = memoryBar(s.engine && s.engine.stats);
  if (mem) body.append(h('h4', null, t('v2ram')), mem);
  try {
    const u = await api.usage();
    const byDay = new Map((u.days || []).map((d) => [d.day, (d.prompt_tokens || 0) + (d.generated_tokens || 0)]));
    if (byDay.size) {
      // The last thirty days, the days without use included.
      const days = [];
      for (let k = 29; k >= 0; --k) {
        const d = new Date(Date.now() - k * 86400000).toISOString().slice(0, 10);
        days.push([d, byDay.get(d) || 0]);
      }
      const max = Math.max(...days.map((d) => d[1]), 1);
      body.append(h('h4', null, t('usage')), h('div', { class: 'usage-bars' }, days.map(([day, v]) =>
        h('div', { style: { height: Math.max(2, 80 * v / max) + 'px', opacity: v ? 0.9 : 0.25 }, title: day + ': ' + formatNumber(v) }))));
    }
  } catch (e) {
    // Usage figures are optional.
  }
  body.append(h('p', { style: { marginTop: '14px' } }, t('createdBy', { author: s.author || 'Aldric Pierrain' }) + ' · ' +
    t('publishedBy', { company: s.company || 'Cydral Technology' })),
  h('p', null, h('a', { href: s.project || 'https://github.com/Cydral/Dlib-Transformer-extensions', target: '_blank', rel: 'noopener' }, 'Dlib-Transformer-extensions'),
    ' · ', h('a', { href: '/docs/', target: '_blank' }, t('apiDocs')), ' · ', h('a', { href: 'vendor/README.txt', target: '_blank' }, t('thirdParty'))));
}

// Layout.
function buildLayout() {
  const app = document.getElementById('app');
  app.replaceChildren();
  ui.sidebar = h('aside', { class: 'sidebar' + (state.settings.sidebar === false ? ' collapsed' : '') });
  ui.sideScroll = h('div', { class: 'side-scroll' });
  ui.statusLine = h('div', { class: 'status-line' });
  const search = h('input', { type: 'search', placeholder: t('search'), value: state.search });
  search.addEventListener('input', () => { state.search = search.value; renderSidebar(); });
  ui.incognitoButton = h('button', {
    class: 'incognito-toggle' + (state.incognito ? ' on' : ''), type: 'button', title: t('incognitoStart'), html: icon('incognito'),
    onclick: () => {
      state.incognito = !state.incognito;
      if (state.busy) stopGeneration();
      state.current = newConversation(null);
      state.view = 'chat';
      if (location.hash !== '#/') history.pushState(null, '', '#/');
      renderAll();
    }
  });
  ui.sidebar.append(
    h('div', { class: 'brand' }, h('img', { src: 'img/albireo-mark.svg', alt: '' }),
      h('div', { class: 'name', html: '<span>Albireo</span>' }), h('div', { class: 'spacer' })),
    h('div', { class: 'sidebar-actions' },
      h('button', {
        class: 'new-chat', type: 'button', html: icon('plus') + '<span>' + escapeHtml(t('newChat')) + '</span>',
        onclick: () => { state.incognito = false; newChat(null); }
      }), ui.incognitoButton),
    h('div', { class: 'search' }, h('span', { html: icon('search') }), search),
    ui.sideScroll,
    h('div', { class: 'sidebar-footer' }, ui.statusLine, h('div', { class: 'footer-buttons' },
      h('button', { type: 'button', onclick: () => settingsDialog('general'), html: icon('gear') + '<span>' + escapeHtml(t('settings')) + '</span>' }),
      h('button', { type: 'button', onclick: () => window.open('/docs/', '_blank'), html: icon('book') + '<span>' + escapeHtml(t('api')) + '</span>' }),
      h('button', { type: 'button', onclick: () => settingsDialog('about'), html: icon('info') + '<span>' + escapeHtml(t('about')) + '</span>' }))));

  ui.topbar = h('header', { class: 'topbar' });
  ui.banner = h('div', { class: 'banner hidden', html: icon('incognito') + '<span>' + escapeHtml(t('incognitoBanner')) + '</span>' });
  ui.content = h('div', { class: 'scroller' });
  ui.input = h('textarea', { rows: 1, 'aria-label': t('placeholder') });
  ui.input.addEventListener('input', autoGrow);
  ui.input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing && state.settings.sendOnEnter) {
      e.preventDefault();
      onSend();
    }
  });
  ui.input.addEventListener('paste', (e) => {
    const files = Array.from(e.clipboardData ? e.clipboardData.files : []);
    if (files.length) { e.preventDefault(); addPending(files); }
  });
  const fileInput = h('input', { type: 'file', multiple: true, class: 'hidden' });
  fileInput.addEventListener('change', () => { addPending(Array.from(fileInput.files)); fileInput.value = ''; });
  ui.reasonButton = h('button', {
    class: 'toggle reason', type: 'button', title: t('reasoningHint'), html: icon('spark') + '<span>' + escapeHtml(t('reasoning')) + '</span>',
    onclick: async () => {
      const c = state.current;
      c.reasoning = !c.reasoning;
      await saveSetting('lastReasoning', c.reasoning);
      if (c.messages.length) await persist(c);
      renderComposerState();
    }
  });
  ui.agentButton = h('button', {
    class: 'toggle', type: 'button', title: t('agentHint'), html: icon('tool') + '<span>' + escapeHtml(t('agent')) + '</span>',
    onclick: async () => {
      const c = state.current;
      c.agent = !c.agent;
      await saveSetting('lastAgent', c.agent);
      if (c.messages.length) await persist(c);
      renderComposerState();
    }
  });
  ui.sendButton = h('button', { class: 'send', type: 'button', onclick: onSend });
  ui.pendingBox = h('div', { class: 'composer-attachments hidden' });
  const composer = h('div', { class: 'composer' }, ui.pendingBox, ui.input, h('div', { class: 'row' },
    iconButton('clip', t('attach'), () => fileInput.click()), fileInput, ui.reasonButton, ui.agentButton,
    h('span', { class: 'grow' }), ui.sendButton));
  for (const ev of ['dragenter', 'dragover']) composer.addEventListener(ev, (e) => { e.preventDefault(); composer.classList.add('drag'); });
  for (const ev of ['dragleave', 'drop']) composer.addEventListener(ev, () => composer.classList.remove('drag'));
  composer.addEventListener('drop', (e) => {
    e.preventDefault();
    if (e.dataTransfer && e.dataTransfer.files.length) addPending(Array.from(e.dataTransfer.files));
  });
  ui.composerWrap = h('div', { class: 'composer-wrap' }, composer,
    h('div', { class: 'hint' }, 'Albireo · ' + (state.status ? state.status.version : '')));
  ui.toasts = h('div', { class: 'toast-zone' });
  const main = h('main', { class: 'main' }, ui.topbar, ui.banner, ui.content, ui.composerWrap);
  app.append(ui.sidebar, main, ui.toasts);
}

async function init() {
  try {
    await store.open();
    state.settings = Object.assign({}, defaults, await store.getSettings());
  } catch (e) {
    // Without IndexedDB (a private window of some browsers) the application keeps its
    // data in memory, as in an incognito discussion.
    console.warn('Albireo: the database of the browser is unavailable', e);
    state.incognito = true;
  }
  applyLanguage();
  applyTheme();
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', applyTheme);
  await reloadLists().catch(() => {});
  buildLayout();
  window.addEventListener('hashchange', route);
  await route();
  await refreshStatus();
  setInterval(refreshStatus, 4000);
}

init();
