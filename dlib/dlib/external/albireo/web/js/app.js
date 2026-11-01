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

    THE IMAGES
        A question may carry images, for a model that reads them (images.js).  They go before
        its text while the question stays in the context, maxContextImages at most for the
        discussion: the images of the exchanges that a compression summarized leave the
        context with them and free their places.  The images of a project are described by
        the background model, and their descriptions are searched with its documents; a
        question that finds one receives the image with it.  A model that reads text only
        receives the names of the images.
*/

import { t, setLanguage, language, languages } from './i18n.js';
import { icon } from './icons.js';
import * as store from './store.js';
import { renderMarkdown, attachCodeButtons, escapeHtml } from './markdown.js';
import * as api from './api.js';
import { extractText, maxFileBytes, extension, searchPassages, documentAccept, fileKind, pdfPages } from './documents.js';
import * as rag from './rag.js';
import * as tools from './tools.js';
import * as images from './images.js';

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
// An answer after which the discussion takes this share of its room leads to the compression
// of its oldest exchanges, in the background, before the next question needs it.
const compactShare = 0.8;
// The summary of the exchanges that left the context takes at most this share of the room.
const summaryShare = 0.15;
/*
    The images the context of a discussion holds at most, those of its project that a question receives with their
    descriptions, the most relevant first, and those a project holds.  A request of the service carries 16 images
    at most.
*/
const maxContextImages = 10;
const maxQuestionProjectImages = 4;
const maxProjectImages = 10;

// Tokens of a text, estimated from its characters.
function estimateTokens(text) {
  return Math.ceil(String(text || '').length / 3.6);
}

const defaults = {
  theme: 'auto', language: 'auto', userName: '', defaultRole: 'general', temperature: null, topP: null, topK: null,
  maxTokens: 0, sendOnEnter: true, systemPrompt: '', lastReasoning: false, lastAgent: false, sidebar: true,
  memoryEnabled: true, memorySensitive: false, memory: '', memoryUpdated: 0, memoryLanguage: '', storagePersisted: null
};

// The memory of the discussions: short notes on the user, which the model writes from the
// discussions, private ones excepted, and which each discussion receives at its start.  Its
// size follows the context of the model: memoryShare of it, memoryWords words at most.
const memoryWords = 120;
const memoryShare = 0.04;

function memoryLimitWords(contextTokens) {
  return Math.max(30, Math.min(memoryWords, Math.floor(memoryShare * (contextTokens || 8192) * 0.7)));
}

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

// A short duration, such as the time a tool took, in milliseconds below one second.
function formatShortDuration(ms) {
  if (ms < 1000) return Math.max(1, Math.round(ms)) + ' ms';
  if (ms >= 60000) return formatDuration(ms);
  const n = new Intl.NumberFormat(language(), { minimumFractionDigits: 1, maximumFractionDigits: 1 });
  return n.format(ms / 1000) + ' s';
}

function toast(message, error, ms) {
  if (!ui.toasts) return;
  const el = h('div', { class: 'toast' + (error ? ' error' : ''), role: error ? 'alert' : 'status' }, message);
  ui.toasts.append(el);
  setTimeout(() => el.remove(), ms || (error ? 6000 : 2600));
}

// Copies a text to the clipboard, and tells whether it could.
function copyText(text) {
  const failed = () => toast(t('copyFailed'), true);
  try {
    navigator.clipboard.writeText(String(text || '')).then(() => toast(t('copied')), failed);
  } catch (e) {
    // The clipboard is out of reach outside of a secure page.
    failed();
  }
}

/*
    A change the database of the browser refused, for instance on a full disk.  The user is told
    at most once every few seconds, and the work goes on with the copies in memory.
*/
let storageWarned = 0;

function storageFailed(e) {
  console.warn('Albireo: the database of the browser refused a change', e);
  if (Date.now() - storageWarned < 15000) return;
  storageWarned = Date.now();
  toast(t(e && e.name === 'QuotaExceededError' ? 'errStorageFull' : 'errStorage'), true);
}

function download(name, content, type) {
  const url = URL.createObjectURL(new Blob([content], { type: type || 'text/plain;charset=utf-8' }));
  const a = h('a', { href: url, download: name });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

/*
    The type of a file the agent created.  The pages, images and scripts a browser would run are
    given as plain data, so that the file is saved and never opened in the origin of the application.
*/
function mimeOf(name) {
  const ext = name.toLowerCase().split('.').pop();
  return {
    md: 'text/markdown', csv: 'text/csv', json: 'application/json', py: 'text/x-python', txt: 'text/plain'
  }[ext] || 'application/octet-stream';
}

// Theme and language.
function applyTheme() {
  const pref = state.settings.theme;
  const dark = pref === 'dark' || (pref === 'auto' && window.matchMedia('(prefers-color-scheme: dark)').matches);
  document.documentElement.dataset.theme = dark ? 'dark' : 'light';
  document.getElementById('hljs-light').disabled = dark;
  document.getElementById('hljs-dark').disabled = !dark;
}

/*
    The language of the interface: the one chosen here, else the one of Albireo, chosen in
    the window of the computer of its icon, else the one of the browser, as the windows of
    Albireo follow the one of Windows.
*/
function applyLanguage() {
  const own = state.settings.language;
  const system = state.status && state.status.settings ? state.status.settings.language : 'auto';
  setLanguage(own && own !== 'auto' ? own : system && system !== 'auto' ? system : null);
}

// Saves a setting and tells the other tabs; a setting the database refuses stays for this visit.
async function saveSetting(key, value) {
  state.settings[key] = value;
  await store.setSetting(key, value).catch(storageFailed);
  broadcast({ type: 'setting', key, value });
  if (/^memory/.test(key) && ui.memoryView) ui.memoryView();
}

/*
    The tabs of the application tell each other the settings they change, the reset of the
    memory, the discussions and the projects they save or delete, and the discussions they are
    answering, so that each tab shows the current state at once.  The messages go by a
    BroadcastChannel, else by the storage event of localStorage.
*/
let channel = null;
try {
  channel = new BroadcastChannel('albireo');
  channel.onmessage = (e) => receive(e.data);
} catch (e) {
  channel = null;
  window.addEventListener('storage', (ev) => {
    if (ev.key !== 'albireo-message' || !ev.newValue) return;
    try {
      receive(JSON.parse(ev.newValue).message);
    } catch (e2) {
      // A value that another application of the origin wrote.
    }
  });
}

function broadcast(message) {
  try {
    if (channel) channel.postMessage(message);
    else localStorage.setItem('albireo-message', JSON.stringify({ message, n: Math.random() }));
  } catch (e) {
    // The other tabs read the settings again at their next start.
  }
}

// A message of another tab.
function receive(message) {
  if (!message || typeof message !== 'object') return;
  if (message.type === 'setting' && Object.prototype.hasOwnProperty.call(defaults, message.key)) {
    state.settings[message.key] = message.value;
    settingChanged(message.key);
  } else if (message.type === 'memoryReset') {
    forgetMemory();
  } else if (message.type === 'conversation' && typeof message.id === 'string') {
    syncConversation(message.id);
  } else if (message.type === 'deleted' && typeof message.id === 'string') {
    conversationDeleted(message.id);
  } else if (message.type === 'cleared') {
    everythingDeleted();
  } else if (message.type === 'lists') {
    syncLists();
  } else if (message.type === 'busy' && typeof message.id === 'string' && typeof message.tab === 'string') {
    busyElsewhere(message.id, message.tab, !!message.busy);
  }
}

// Shows a setting changed in this tab or in another one.
function settingChanged(key) {
  if (key === 'theme') applyTheme();
  else if (key === 'language') languageChanged().catch(() => {});
  else if (key === 'userName' || key === 'defaultRole') refreshViews();
  if (/^memory/.test(key) && ui.memoryView) ui.memoryView();
}

/*
    Applies the language of the interface after a change of the setting here or in another tab, or
    of the language of Albireo.  The texts are drawn again, and the memory is written in the new
    language at its next update.  force draws them again even when the language stays.
*/
async function languageChanged(force) {
  const before = language();
  applyLanguage();
  if (language() === before && !force) return;
  state.catalog = await api.models(language()).catch(() => state.catalog);
  buildLayout();
  renderAll();
  if (language() !== before) rememberLater();
}

/*
    Draws again the side panel, the top bar, the composer, the status line, and the pages without
    messages (welcome, project) unless the user is typing in them, so that they show the state of
    the discussions and of the projects.  The messages of the discussion shown stay as they are.
*/
function refreshViews() {
  if (!ui.sideScroll) return;
  renderSidebar();
  if (!ui.topbar.contains(document.activeElement)) renderTopbar();
  renderComposerState();
  renderStatusLine();
  const page = state.view === 'project' || (state.view === 'chat' && state.current && !state.current.messages.length);
  if (page && !editingIn(ui.content)) renderMain();
}

// True when the user is typing in a field of el.
function editingIn(el) {
  const a = document.activeElement;
  return !!(a && el && el.contains(a) && (a.tagName === 'INPUT' || a.tagName === 'TEXTAREA' || a.isContentEditable));
}

// The messages of a discussion as they are drawn, to tell whether a saved copy changes the thread.
function threadKey(c) {
  const messages = (c && c.messages) || [];
  const last = messages[messages.length - 1];
  const steps = last && last.steps
    ? last.steps.map((s) => (s.content || '').length + '.' + (s.toolCalls || []).length) : [];
  const ended = last && (last.error || last.cancelled) ? 1 : 0;
  return messages.map((m) => m.id).join(',') + '|' + steps.join(',') + '|' + ended + '|' + (c.contextStart || 0);
}

// The fields of a discussion another tab may change while this one answers in it.
const sharedFields = ['title', 'titleSource', 'pinned', 'projectId'];

// The discussions to read again from the database, each once after a burst of messages.
const syncTimers = new Map();

function syncConversation(id) {
  clearTimeout(syncTimers.get(id));
  syncTimers.set(id, setTimeout(() => {
    syncTimers.delete(id);
    applyStored(id).catch(storageFailed);
  }, 50));
}

/*
    Takes the copy of a discussion another tab saved.  The discussion this tab is answering keeps
    its messages and takes the title, the pin and the project; the one shown otherwise takes the
    saved copy whole, and its messages are drawn again when they changed.
*/
async function applyStored(id) {
  if (deleted.has(id)) return;
  const stored = await store.getConversation(id);
  if (!stored) {
    if (state.conversations.some((x) => x.id === id)) conversationDeleted(id);
    return;
  }
  let copy = stored;
  const shown = state.current && state.current.id === id ? state.current : null;
  let redraw = false;
  // A question being edited keeps the copy it belongs to.
  if (shown && (state.busyId === id || editingIn(ui.content) || (ui.thread && ui.thread.querySelector('textarea')))) {
    for (const k of sharedFields) shown[k] = stored[k];
    copy = shown;
  } else if (shown) {
    redraw = threadKey(shown) !== threadKey(stored);
    state.current = stored;
  }
  const i = state.conversations.findIndex((x) => x.id === id);
  if (i >= 0) state.conversations[i] = copy;
  else state.conversations.unshift(copy);
  state.conversations.sort((a, b) => b.updated - a.updated);
  if (redraw && state.view === 'chat') renderMain();
  refreshViews();
}

// Forgets a discussion deleted in another tab, stops its answer in progress here, and never saves it again.
function conversationDeleted(id) {
  deleted.add(id);
  othersBusy.delete(id);
  state.conversations = state.conversations.filter((x) => x.id !== id);
  if (state.current && state.current.id === id) {
    if (state.busy) stopGeneration();
    go('#/');
  } else {
    refreshViews();
  }
}

// Every discussion and every project deleted in another tab.
function everythingDeleted() {
  for (const c of state.conversations) deleted.add(c.id);
  if (state.current && state.current.messages.length && !state.current.incognito) deleted.add(state.current.id);
  if (state.busy) stopGeneration();
  cancelIndex();
  rag.removeAll();
  images.removeAll();
  syncLists().then(() => go('#/'));
}

/*
    The lists read again from the database, after an import or a change of the projects in another
    tab, once for a burst of messages.  The discussion shown stays the copy of this tab.
*/
let listsSync = null;

function syncLists() {
  if (!listsSync) {
    listsSync = new Promise((resolve) => setTimeout(resolve, 50)).then(async () => {
      listsSync = null;
      await reloadLists().catch(storageFailed);
      const c = state.current;
      const i = c ? state.conversations.findIndex((x) => x.id === c.id) : -1;
      if (i >= 0) state.conversations[i] = c;
      // A discussion whose project another tab deleted leaves it, as the stored discussions did.
      if (c && c.projectId && !projectById(c.projectId)) c.projectId = null;
      refreshViews();
    });
  }
  return listsSync;
}

/*
    The discussions other tabs are answering: id -> {tab, until}.  A tab says it again every
    busyBeat milliseconds while it answers, so that the mark of a tab closed meanwhile expires.
*/
const othersBusy = new Map();
const busyBeat = 20000;

function busyElsewhere(id, tab, busy) {
  if (busy) othersBusy.set(id, { tab, until: Date.now() + 3 * busyBeat });
  else if (othersBusy.has(id) && othersBusy.get(id).tab === tab) othersBusy.delete(id);
  if (state.current && state.current.id === id) {
    renderComposerState();
    lockThread();
  }
}

// True when another tab is answering in the discussion c.
function answeredElsewhere(c) {
  const mark = c ? othersBusy.get(c.id) : null;
  if (mark && mark.until < Date.now()) othersBusy.delete(c.id);
  return !!(mark && mark.until >= Date.now());
}

// Forgets the marks of the tabs that stopped saying they answer; true when one expired.
function expireBusy() {
  let expired = false;
  for (const [id, mark] of othersBusy) {
    if (mark.until >= Date.now()) continue;
    othersBusy.delete(id);
    expired = true;
  }
  return expired;
}

let busyTimer = 0;

// Tells the other tabs that this tab answers, or no longer answers, in the discussion c.
function announceBusy(c, busy) {
  clearInterval(busyTimer);
  busyTimer = 0;
  if (!c || c.incognito) return;
  const say = () => broadcast({ type: 'busy', id: c.id, tab: tabId, busy });
  say();
  if (busy) busyTimer = setInterval(say, busyBeat);
}

// Service status and models.
function roleModel(role) {
  const s = state.status;
  if (!s || !s.active) return null;
  return s.active[role] || s.active.general || null;
}

function modelInfo(id) {
  if (!state.catalog || !id) return null;
  const same = (m) => m.id === id;
  return (state.catalog.models || []).find(same) || (state.catalog.embedding_models || []).find(same) || null;
}

function modelName(id) {
  const m = modelInfo(id);
  return m && m.name ? m.name : id;
}

function loadedModel(id) {
  const list = (state.status && state.status.loaded_models) || [];
  return list.find((m) => m.id === id) || null;
}

// What a model reads: "text", and "image" or "audio" for the models that read them, as the
// engine or the catalog tells.
function modelInputs(id) {
  const loaded = loadedModel(id);
  if (loaded && Array.isArray(loaded.inputs) && loaded.inputs.length) return loaded.inputs;
  const m = modelInfo(id);
  return m && Array.isArray(m.inputs) && m.inputs.length ? m.inputs : ['text'];
}

// What the model of a role reads; the attachments follow.
function roleInputs(role) {
  return modelInputs(roleModel(role));
}

function readsImages(role) {
  return roleInputs(role).includes('image');
}

function inputsLabel(inputs) {
  return inputs.map((i) => t('input_' + i)).join(', ');
}

/*
    The file picker of a role: the documents, read as text, and the media its model reads, the
    images while room, the places left for them, is not 0.
*/
function acceptFor(role, room) {
  const inputs = roleInputs(role);
  let accept = documentAccept();
  if (inputs.includes('image') && room !== 0) accept += ',image/*';
  if (inputs.includes('audio')) accept += ',audio/*';
  return accept;
}

function roleCapabilities(role) {
  const id = roleModel(role);
  const loaded = loadedModel(id);
  if (loaded) return { reasoning: loaded.reasoning !== false, tools: loaded.tools !== false, context: loaded.context || 0 };
  const m = modelInfo(id);
  return { reasoning: m ? m.reasoning !== false : true, tools: m ? m.tools !== false : true, context: 0 };
}

/*
    The work done in the background (titles, summaries, memory) goes to the model that comes
    with Albireo, which the service keeps loaded, without reasoning; the service gives it to
    the general model when there is none.
*/
const backgroundModel = 'albireo-background';

function backgroundContext() {
  const s = state.status;
  const loaded = s && s.background ? loadedModel(s.background) : null;
  return (loaded && loaded.context) || roleCapabilities('general').context || 8192;
}

// True when the background model reads images, which the descriptions of the images of the projects need.
function backgroundReadsImages() {
  const s = state.status;
  return modelInputs((s && s.background) || roleModel('general')).includes('image');
}

// The text the background model writes for an instruction and a message.
function backgroundText(system, user, maxTokens, signal) {
  return backgroundChat([{ role: 'system', content: system }, { role: 'user', content: user }], maxTokens, signal);
}

// The text the background model writes after a list of messages.
async function backgroundChat(messages, maxTokens, signal, temperature) {
  const body = {
    model: backgroundModel, reasoning: false, temperature: temperature == null ? 0.2 : temperature,
    max_tokens: maxTokens, messages
  };
  let text = '';
  let finish = null;
  for await (const ev of api.streamChat(body, signal)) {
    const choice = ev.type === 'chunk' && ev.data.choices && ev.data.choices[0];
    if (!choice) continue;
    if (choice.delta && choice.delta.content) text += choice.delta.content;
    if (choice.finish_reason) finish = choice.finish_reason;
  }
  return { text: text.trim(), finish };
}

// The reading of the status in progress, which the calls made meanwhile share.
let statusPoll = null;

function refreshStatus() {
  if (!statusPoll) statusPoll = pollStatus().finally(() => { statusPoll = null; });
  return statusPoll;
}

/*
    Reads the status of the service, with the models of the functions, which the window of the
    notification area may change, and whether the service answers.  A change draws the displays
    that show it again at once, and the return of the service resumes the titles left waiting.
*/
async function pollStatus() {
  let changed = false;
  let back = false;
  try {
    const s = await api.status();
    const key = (x) => JSON.stringify([x.active, x.loaded_models, x.state, x.background]);
    changed = !state.status || key(state.status) !== key(s);
    back = state.statusError;
    state.status = s;
    state.statusError = false;
    if (changed || !state.catalog) state.catalog = await api.models(language()).catch(() => state.catalog);
    // The tokens of the images were measured for the models of the functions as they were.
    if (changed) imageTokenCache.clear();
    // A language chosen in Albireo meanwhile, which this interface follows.
    await languageChanged();
    if (ui.hint) ui.hint.textContent = hintText();
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
  if ((changed || back || expireBusy()) && state.current) {
    refreshViews();
    lockThread();
  } else {
    renderStatusLine();
  }
  if (back) namePending();
  if (back || changed) describePending();
}

function renderStatusLine() {
  if (!ui.statusLine) return;
  const s = state.status;
  let led = 'error', text = t('serviceDown');
  if (s && !state.statusError) {
    const stats = s.engine && s.engine.stats;
    const loading = Object.values(s.models_state || {}).some((v) => v === 'loading');
    const model = roleModel(state.current ? state.current.role : 'general');
    if (s.state === 'starting') {
      led = 'busy';
      text = t('serviceStarting');
    } else if (!model) {
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

// The lists of the sidebar show this many projects and discussions at most; the others are
// found by the search.
const sideLimit = 20;
// The searches run once the user stops typing for this many milliseconds.
const searchDelay = 200;

// Text as the search compares it: lower case, without accents.
function fold(text) {
  return String(text || '').normalize('NFD').replace(/[̀-ͯ]/g, '').toLowerCase();
}

function queryTerms(query) {
  return fold(query).split(/\s+/).filter(Boolean);
}

// The languages of the interface, named in English for the instructions of the model.
const languageNames = { en: 'English', fr: 'French', de: 'German', it: 'Italian', es: 'Spanish' };

// Frequent short words of each language, without accents, and letters that mostly belong to one.
const languageWords = {
  en: 'the a an and or but of to in on at for with from by about is are was were be been am it its this that these '
    + 'those what how why which who when where can could would should will you your i my me we our they their he she '
    + 'his her do does did not no please write explain tell give name list make show find use works prefers likes '
    + 'uses wants as one line if than then also there',
  fr: 'le la les un une des du de et ou mais est sont etait suis es que qui quoi comment pourquoi quel quelle quels '
    + 'quelles quand dans pour avec sans sur sous par chez je tu il elle nous vous ils elles mon ma mes ton ta tes son '
    + 'sa ses notre votre leur ce cet cette ces au aux en pas ne plus tres peux peut faire fais explique ecris calcule '
    + 'donne dis resume liste c l d j qu n aime prefere utilise travaille veut souhaite comme ligne me se si non',
  de: 'der die das den dem des ein eine einen einem einer und oder aber ist sind was war bin nicht kein keine ich du '
    + 'er sie es wir ihr mein meine dein wie warum welche welcher welches wann wo mit fur auf von vom zu zum zur im '
    + 'in bei aus nach uber kann konnen bitte erklare schreibe berechne gib zeige als also auch noch sehr arbeitet '
    + 'bevorzugt mag nutzt verwendet mochte zeile',
  it: 'il lo la gli le un una uno di del della dei delle degli e o ma che chi come cosa perche quale quali quando dove '
    + 'per con su tra fra nel nella nei sono sei mi ti ci non piu molto puoi spiegami spiega scrivi calcola dammi '
    + 'dimmi lavora preferisce usa vuole riga tu me se si',
  es: 'el la los las un una unos unas de del y o pero que quien como cual cuales cuando donde por para con sin sobre '
    + 'entre en es son soy estoy mi tu su no muy puedes explica explicame escribe calcula dame dime trabaja prefiere '
    + 'usa quiere linea me se si'
};
const languageLetters = { fr: /[çœêâîôûë]/g, de: /[äöüß]/g, es: /[ñ¿¡áíóú]/g, it: /[ìò]/g };
let languageIndex = null;

/*
    The language of a text among those of the interface, from its frequent short words and its
    letters, or null when the text is too short or too mixed to tell.  strict asks for a clearer
    lead of one language over the others.
*/
function textLanguage(text, strict) {
  if (!languageIndex) {
    languageIndex = new Map();
    for (const [code, words] of Object.entries(languageWords)) {
      for (const w of new Set(words.split(' '))) languageIndex.set(w, (languageIndex.get(w) || []).concat([code]));
    }
  }
  const score = { en: 0, fr: 0, de: 0, it: 0, es: 0 };
  const lower = String(text || '').toLowerCase();
  for (const w of fold(lower).split(/[^a-z]+/)) {
    const codes = languageIndex.get(w);
    if (codes) for (const code of codes) score[code] += 1 / codes.length;
  }
  for (const [code, re] of Object.entries(languageLetters)) {
    score[code] += Math.min(2, (lower.match(re) || []).length / 2);
  }
  const ranked = Object.entries(score).sort((a, b) => b[1] - a[1]);
  const floor = strict ? 2 : 1.5;
  return ranked[0][1] >= floor && ranked[0][1] >= floor * ranked[1][1] ? ranked[0][0] : null;
}

/*
    The folded texts of the search, kept with the text they were made from (WeakMap: message or
    discussion -> {text, folded}), so that a search folds only what changed since the last one.
*/
const messageFolds = new WeakMap();
const titleFolds = new WeakMap();

function foldedOf(cache, owner, text) {
  const hit = cache.get(owner);
  if (hit && hit.text === text) return hit.folded;
  const folded = fold(text);
  cache.set(owner, { text, folded });
  return folded;
}

// The text of the messages of a discussion, and its folded form made of the folded messages.
function conversationFolded(c) {
  const messages = c.messages || [];
  const texts = messages.map(messageText);
  return { text: texts.join('\n'), folded: messages.map((m, i) => foldedOf(messageFolds, m, texts[i])).join('\n') };
}

// The first position of a term of terms in text, or -1 when one of them is missing.  A term
// matches the beginning, the middle or the end of a word.
function findTerms(text, terms) {
  return findFolded(fold(text), terms);
}

// findTerms for a text already folded.
function findFolded(f, terms) {
  let first = -1;
  for (const term of terms) {
    const i = f.indexOf(term);
    if (i < 0) return -1;
    if (first < 0 || i < first) first = i;
  }
  return first;
}

// A part of text around a position, with the terms marked.
function excerptHtml(text, at, terms, width) {
  text = String(text || '').replace(/\s+/g, ' ');
  const w = width || 160;
  const start = Math.max(0, at - Math.floor(w / 3));
  const piece = text.slice(start, start + w);
  return (start > 0 ? '…' : '') + markTerms(piece, terms) + (start + w < text.length ? '…' : '');
}

// text as HTML, the terms of the search in <mark>.
function markTerms(text, terms) {
  text = String(text || '');
  const f = fold(text);
  // Folding keeps the length of most texts; the marks are left out of the others.
  if (f.length !== text.length || !terms.length) return escapeHtml(text);
  const marked = new Array(text.length).fill(false);
  for (const term of terms) {
    for (let i = f.indexOf(term); i >= 0; i = f.indexOf(term, i + 1)) marked.fill(true, i, i + term.length);
  }
  let out = '';
  let open = false;
  for (let i = 0; i < text.length; ++i) {
    if (marked[i] !== open) {
      out += open ? '</mark>' : '<mark>';
      open = marked[i];
    }
    out += escapeHtml(text[i]);
  }
  return out + (open ? '</mark>' : '');
}

function messageText(m) {
  if (m.role === 'user') return m.content || '';
  return (m.steps || []).map((s) => s.content || '').join('\n');
}

/*
    The discussions that hold every term of a query, whatever their case and their accents,
    a term matching the beginning, the middle or the end of a word: those whose title holds
    them first, then those whose messages hold them, each group the most recent first.  Each
    result tells where the terms are: {conversation, where: 'title', 'question', 'answer' or
    'messages' (several of them), at, text, excerpt}.
*/
function searchConversations(query, limit) {
  const terms = queryTerms(query);
  if (!terms.length) return [];
  const titles = [];
  const contents = [];
  for (const c of state.conversations) {
    const title = c.title || '';
    const at = findFolded(foldedOf(titleFolds, c, title), terms);
    if (at >= 0) {
      titles.push({ conversation: c, where: 'title', at, text: title });
      continue;
    }
    // The first message that holds every term, else the discussion as a whole.
    let hit = null;
    for (const m of c.messages || []) {
      const text = messageText(m);
      const i = findFolded(foldedOf(messageFolds, m, text), terms);
      if (i >= 0) {
        hit = { conversation: c, where: m.role === 'user' ? 'question' : 'answer', at: i, text, created: m.created };
        break;
      }
    }
    if (!hit) {
      const all = conversationFolded(c);
      if (findFolded(all.folded, terms) >= 0) {
        hit = { conversation: c, where: 'messages', at: all.folded.indexOf(terms[0]), text: all.text };
      }
    }
    if (hit) contents.push(hit);
  }
  const out = titles.concat(contents).slice(0, limit || 100);
  for (const x of out) x.excerpt = String(x.text).replace(/\s+/g, ' ').slice(Math.max(0, x.at - 60), x.at + 120);
  return out;
}

// The time of the last activity of a project: its own changes and those of its discussions.
function projectActivity(p) {
  let t0 = p.updated || p.created || 0;
  for (const c of state.conversations) if (c.projectId === p.id && c.updated > t0) t0 = c.updated;
  return t0;
}

/*
    The projects that hold every term of a query, as searchConversations: those whose name
    holds them first, then those whose description, instructions, documents, images or
    discussions hold them.  where: 'name', 'description', 'instructions', 'document', 'image'
    or 'discussion', with the name of the document or of the image or the title of the
    discussion in label.
*/
function searchProjects(query, limit) {
  const terms = queryTerms(query);
  if (!terms.length) return [];
  const names = [];
  const contents = [];
  for (const p of state.projects) {
    const at = findTerms(p.name, terms);
    if (at >= 0) {
      names.push({ project: p, where: 'name', at, text: p.name });
      continue;
    }
    // Each field with its folded text, the long ones folded once (foldedOf).
    const fields = [['description', p.description || '', ''], ['instructions', p.instructions || '', '']]
      .map(([where, text, label]) => [where, text, label, fold(text)]);
    for (const f of p.files || []) {
      const text = f.name + '\n' + (f.text || '');
      fields.push(['document', text, f.name, foldedOf(messageFolds, f, text)]);
    }
    for (const img of p.images || []) {
      const text = img.name + '\n' + (img.description || '');
      fields.push(['image', text, img.name, foldedOf(messageFolds, img, text)]);
    }
    for (const c of state.conversations) {
      if (c.projectId !== p.id) continue;
      const all = conversationFolded(c);
      fields.push(['discussion', (c.title || '') + '\n' + all.text, c.title || t('untitled'),
        fold(c.title || '') + '\n' + all.folded]);
    }
    for (const [where, text, label, folded] of fields) {
      const i = findFolded(folded, terms);
      if (i >= 0) {
        contents.push({ project: p, where, at: i, text, label });
        break;
      }
    }
  }
  const byActivity = (a, b) => projectActivity(b.project) - projectActivity(a.project);
  return names.sort(byActivity).concat(contents.sort(byActivity)).slice(0, limit || 100);
}

/*
    The window of the search of the projects (kind 'projects') or of the discussions: each
    result shows its title, where the words were found, and the passage that holds them.
*/
function searchDialog(kind) {
  const projects = kind === 'projects';
  return dialog((box, close) => {
    const input = h('input', { type: 'search', placeholder: t('searchPlaceholder'), 'aria-label': t('searchPlaceholder') });
    const count = h('div', { class: 'help' });
    const list = h('div', { class: 'search-results', role: 'listbox' });
    const open = (hit) => {
      close(true);
      go(projects ? '#/p/' + hit.project.id : '#/c/' + hit.conversation.id);
    };
    const show = () => {
      const terms = queryTerms(input.value);
      const hits = projects ? searchProjects(input.value) : searchConversations(input.value);
      list.replaceChildren();
      if (!terms.length) {
        count.textContent = t('searchHint');
        return;
      }
      count.textContent = t('resultsCount', { n: hits.length });
      for (const hit of hits) {
        const title = projects ? hit.project.name : (hit.conversation.title || t('untitled'));
        const date = new Date(projects ? projectActivity(hit.project) : hit.conversation.updated);
        const p = projects ? hit.project : (hit.conversation.projectId ? projectById(hit.conversation.projectId) : null);
        const inTitle = hit.where === 'title' || hit.where === 'name';
        const where = t('found_' + hit.where, { name: hit.label || '' });
        list.append(h('div', { class: 'search-result', role: 'option', tabindex: '0', onclick: () => open(hit),
          onkeydown: (e) => { if (e.key === 'Enter') open(hit); } },
          h('div', { class: 'line' },
            p ? h('span', { class: 'dot', style: { background: p.color || projectColors[0] } })
              : h('span', { html: icon(roleIcons[hit.conversation.role] || 'chat') }),
            h('span', { class: 'title', html: inTitle ? markTerms(title, terms) : escapeHtml(title) }),
            h('span', { class: 'date' }, date.toLocaleDateString(language()))),
          h('div', { class: 'where' }, h('span', { class: 'badge' }, where),
            inTitle ? null : h('span', { class: 'excerpt', html: excerptHtml(hit.text, hit.at, terms) }))));
      }
      if (!hits.length) list.append(h('div', { class: 'empty-note' }, t('noResults')));
    };
    let timer = 0;
    input.addEventListener('input', () => { clearTimeout(timer); timer = setTimeout(show, searchDelay); });
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        const first = list.querySelector('.search-result');
        if (first) first.click();
      }
    });
    box.classList.add('search-dialog');
    box.append(h('h2', null, projects ? t('searchProjects') : t('searchDiscussions')), input, count, list,
      h('div', { class: 'actions' }, h('button', { class: 'button', type: 'button', onclick: () => close(null) }, t('close'))));
    show();
    setTimeout(() => input.focus(), 0);
  });
}

// Sidebar.
function renderSidebar() {
  const scroll = ui.sideScroll;
  scroll.replaceChildren();
  if (state.search.trim()) {
    const terms = queryTerms(state.search);
    const hits = searchConversations(state.search);
    const section = h('div', { class: 'side-section' }, h('div', { class: 'side-title' }, t('searchIn')));
    if (!hits.length) section.append(h('div', { class: 'empty-note' }, t('noResults')));
    for (const hit of hits) {
      const item = conversationItem(hit.conversation);
      item.title = hit.where === 'title' ? hit.text : t('found_' + hit.where) + ' : ' + hit.excerpt;
      if (hit.where !== 'title') {
        item.querySelector('.label').innerHTML = escapeHtml(hit.conversation.title || t('untitled'))
          + '<span class="side-excerpt">' + excerptHtml(hit.text, hit.at, terms, 60) + '</span>';
      } else {
        item.querySelector('.label').innerHTML = markTerms(hit.text, terms);
      }
      section.append(item);
    }
    scroll.append(section);
    return;
  }

  const projects = h('div', { class: 'side-section' },
    h('div', { class: 'side-title' }, h('span', null, t('projects')),
      h('span', { class: 'side-tools' },
        h('button', { type: 'button', title: t('searchProjects'), onclick: () => searchDialog('projects'), html: icon('search') }),
        h('button', { type: 'button', title: t('newProject'), onclick: () => newProject(false), html: icon('plus') }))));
  // A project is created from here as well as from a discussion.
  projects.append(h('div', { class: 'item new-item', onclick: () => newProject(false) },
    h('span', { html: icon('plus') }), h('span', { class: 'label' }, t('newProject'))));
  const recent = state.projects.slice().sort((a, b) => projectActivity(b) - projectActivity(a));
  for (const p of recent.slice(0, sideLimit)) {
    const count = state.conversations.filter((c) => c.projectId === p.id).length;
    projects.append(h('div', {
      class: 'item' + (state.view === 'project' && state.projectId === p.id ? ' active' : ''),
      onclick: () => go('#/p/' + p.id)
    }, h('span', { class: 'dot', style: { background: p.color || projectColors[0] } }),
    h('span', { class: 'label' }, p.name), h('span', { class: 'faint', style: { fontSize: '12px', color: 'var(--text-faint)' } },
      count ? String(count) : '')));
  }
  if (recent.length > sideLimit) {
    projects.append(h('div', { class: 'item more-item', onclick: () => searchDialog('projects') },
      h('span', { html: icon('search') }), h('span', { class: 'label' }, t('searchMoreProjects', { n: recent.length }))));
  }
  scroll.append(projects);

  // The discussions: the pinned ones, then by date, sideLimit of them at most.
  scroll.append(h('div', { class: 'side-section side-head' }, h('div', { class: 'side-title' }, h('span', null, t('discussions')),
    h('span', { class: 'side-tools' },
      h('button', { type: 'button', title: t('searchDiscussions'), onclick: () => searchDialog('discussions'), html: icon('search') })))));
  const pinned = state.conversations.filter((c) => c.pinned);
  const shown = pinned.slice(0, sideLimit);
  if (shown.length) {
    const section = h('div', { class: 'side-section' }, h('div', { class: 'side-title' }, t('pinned')));
    for (const c of shown) section.append(conversationItem(c));
    scroll.append(section);
  }
  let left = sideLimit - shown.length;
  const groups = { today: [], yesterday: [], week: [], month: [], older: [] };
  for (const c of state.conversations) if (!c.pinned) groups[dateGroup(c.updated)].push(c);
  for (const g of Object.keys(groups)) {
    if (!groups[g].length || left <= 0) continue;
    const section = h('div', { class: 'side-section' }, h('div', { class: 'side-title' }, t(g)));
    for (const c of groups[g].slice(0, left)) section.append(conversationItem(c));
    left -= Math.min(left, groups[g].length);
    scroll.append(section);
  }
  if (!state.conversations.length) scroll.append(h('div', { class: 'empty-note' }, t('noDiscussions')));
  if (state.conversations.length > sideLimit) {
    scroll.append(h('div', { class: 'side-section' }, h('div', { class: 'item more-item', onclick: () => searchDialog('discussions') },
      h('span', { html: icon('search') }), h('span', { class: 'label' },
        t('searchMoreDiscussions', { n: state.conversations.length })))));
  }
}

// True when the discussion c is being answered, in this tab or in another one.
function answering(c) {
  return !!c && (state.busyId === c.id || answeredElsewhere(c));
}

function conversationItem(c) {
  const p = c.projectId ? projectById(c.projectId) : null;
  const active = state.view === 'chat' && state.current && state.current.id === c.id;
  return h('div', { class: 'item' + (active ? ' active' : ''), onclick: () => go('#/c/' + c.id) },
    p ? h('span', { class: 'dot', title: p.name, style: { background: p.color || projectColors[0] } })
      : h('span', { html: icon(roleIcons[c.role] || 'chat') }),
    h('span', { class: 'label' }, c.title || t('untitled')),
    answering(c) ? h('span', { class: 'spinner small', title: t('answering'), 'aria-label': t('answering') }) : null,
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
    // An entry that does not apply in the current state shows why, and does nothing.
    menu.append(h('div', {
      class: 'entry' + (e.danger ? ' danger' : '') + (e.checked ? ' checked' : '') + (e.disabled ? ' disabled' : ''),
      role: 'menuitem', 'aria-disabled': e.disabled ? 'true' : null, title: e.disabled ? e.hint : null,
      onclick: () => {
        if (e.disabled) return;
        closeMenu();
        e.action();
      }
    }, e.icon ? h('span', { html: icon(e.icon) }) : null,
    h('span', null, e.label, e.desc ? h('span', { class: 'desc' }, e.desc) : null)));
  }
  placeMenu(menu, anchor);
}

// A panel of explanations under a button, which closes as a menu does.
function showPopover(anchor, children) {
  closeMenu();
  placeMenu(h('div', { class: 'menu popover', role: 'dialog' }, children), anchor);
}

// Shows a menu or a panel under its anchor, above it when the window lacks room below.
function placeMenu(menu, anchor) {
  document.body.append(menu);
  const r = anchor.getBoundingClientRect();
  const mw = menu.offsetWidth, mh = menu.offsetHeight;
  let x = Math.min(r.left, window.innerWidth - mw - 8);
  let y = r.bottom + 6;
  if (y + mh > window.innerHeight - 8) y = Math.max(8, r.top - mh - 6);
  menu.style.left = Math.max(8, x) + 'px';
  menu.style.top = y + 'px';
  ui.menu = menu;
  // The listeners follow the click that opened the menu, unless the menu closed meanwhile.
  setTimeout(() => {
    if (ui.menu !== menu) return;
    document.addEventListener('mousedown', outsideMenu);
    document.addEventListener('keydown', escapeMenu);
  }, 0);
}

function outsideMenu(e) {
  if (ui.menu && !ui.menu.contains(e.target)) closeMenu();
}

function escapeMenu(e) {
  if (e.key === 'Escape') closeMenu();
}

function closeMenu() {
  if (ui.menu) ui.menu.remove();
  ui.menu = null;
  document.removeEventListener('mousedown', outsideMenu);
  document.removeEventListener('keydown', escapeMenu);
}

// What the agent mode does, and the tools it may call, each with what it is for.
function agentHelp(anchor) {
  showPopover(anchor, [
    h('div', { class: 'head' }, t('agentHelpTitle')),
    h('p', null, t('agentHelp')),
    h('div', { class: 'popover-tools' }, tools.definitions.map((d) => h('div', { class: 'popover-tool' },
      h('span', { html: icon(toolIcons[d.function.name] || 'tool') }),
      h('span', null, h('b', null, toolLabel(d.function.name)),
        h('span', { class: 'desc' }, t('toolHint_' + d.function.name))))))
  ]);
}

// The open discussion is the copy that answers and saves: an action of the sidebar on it
// changes that copy, so that the next save keeps the action.
function liveCopy(c) {
  return state.current && state.current.id === c.id ? state.current : c;
}

function conversationMenu(entry, anchor) {
  const c = liveCopy(entry);
  // The export waits for the answer in progress, which it would hold in part.
  const busy = answering(c);
  const hint = t('afterAnswer');
  showMenu(anchor, [
    { label: t('rename'), icon: 'edit', action: () => renameConversation(liveCopy(c)) },
    { label: c.pinned ? t('unpin') : t('pin'), icon: 'pin', action: () => togglePin(liveCopy(c)) },
    { label: t('moveToProject'), icon: 'move', action: () => moveMenu(liveCopy(c), anchor) },
    c.projectId ? { label: t('removeFromProject'), icon: 'detach', action: () => setProject(liveCopy(c), null) } : null,
    '-',
    { label: t('exportMarkdown'), icon: 'download', disabled: busy, hint,
      action: () => exportConversation(liveCopy(c), 'md') },
    { label: t('exportJson'), icon: 'archive', disabled: busy, hint,
      action: () => exportConversation(liveCopy(c), 'json') },
    '-',
    { label: t('delete'), icon: 'trash', danger: true, action: () => removeConversation(liveCopy(c)) }
  ].filter(Boolean));
}

function moveMenu(c, anchor) {
  const entries = [{ head: t('moveTo') }];
  for (const p of state.projects) {
    entries.push({
      label: p.name, icon: 'folder', checked: c.projectId === p.id, action: () => setProject(liveCopy(c), p.id)
    });
  }
  entries.push('-', {
    label: t('newProject'), icon: 'plus',
    action: async () => {
      const p = await newProject(true);
      if (p) await setProject(liveCopy(c), p.id);
    }
  });
  if (c.projectId) entries.push({ label: t('none'), icon: 'detach', action: () => setProject(liveCopy(c), null) });
  showMenu(anchor, entries);
}

// The discussions deleted in this tab: the work that ends after their deletion (an answer, a
// title, a summary, the marks of the memory) does not save them again.
const deleted = new Set();

/*
    Saves a discussion and tells the other tabs.  Returns false when there is nothing to save, or
    when the database refused it, the user being told; the copy in memory then stays the one of
    this tab.
*/
async function persist(c) {
  if (!c || c.incognito || deleted.has(c.id)) return false;
  // The answer being written is saved once it ends, whole.
  const record = c.messages.includes(liveTurn)
    ? Object.assign({}, c, { messages: c.messages.filter((x) => x !== liveTurn) }) : c;
  try {
    await store.putConversation(record);
    // A deletion that came while the discussion was being saved stays.
    if (deleted.has(c.id)) {
      await store.deleteConversation(c.id);
      return false;
    }
  } catch (e) {
    storageFailed(e);
    return false;
  }
  const i = state.conversations.findIndex((x) => x.id === c.id);
  if (i >= 0) state.conversations[i] = c;
  else state.conversations.unshift(c);
  state.conversations.sort((a, b) => b.updated - a.updated);
  broadcast({ type: 'conversation', id: c.id });
  return true;
}

async function renameConversation(c) {
  const name = await promptDialog(t('rename'), c.title || '');
  if (name == null || !name.trim()) return;
  const live = liveCopy(c);
  live.title = name.trim();
  // A title of the user stays: the model no longer names the discussion.
  live.titleSource = 'user';
  await persist(live);
  refreshViews();
}

async function togglePin(c) {
  c.pinned = !c.pinned;
  await persist(c);
  refreshViews();
}

async function setProject(c, projectId) {
  c.projectId = projectId;
  // A discussion without messages, not stored yet, takes the project with its first question.
  if (c.messages.length) await persist(c);
  refreshViews();
}

async function removeConversation(c) {
  const ok = await confirmDialog(t('deleteDiscussion'), t('deleteDiscussionText', { title: c.title || t('untitled') }));
  if (!ok) return;
  deleted.add(c.id);
  // The answer in progress stops, and the passages of the documents leave the memory of the
  // search with those of the database.
  if (state.busy && state.busyId === c.id) stopGeneration();
  await forgetOwner(c.id).catch(() => {});
  await store.deleteConversation(c.id).catch(storageFailed);
  await images.removeOwner(c.id).catch(() => {});
  broadcast({ type: 'deleted', id: c.id });
  state.conversations = state.conversations.filter((x) => x.id !== c.id);
  if (state.current && state.current.id === c.id) go('#/');
  else refreshViews();
}

// The name of the file of a discussion, its title without the characters a file name cannot hold.
function fileName(c) {
  const name = (c.title || '').replace(/[\\/:*?"<>|\u0000-\u001f\u007f]+/g, '_')
    .replace(/^[\s.]+|[\s.]+$/g, '').slice(0, 80);
  return name || 'discussion';
}

// The messages of a discussion in Markdown, under its title.
function markdownOf(title, messages) {
  const lines = ['# ' + title, ''];
  for (const m of messages) {
    if (m.role === 'user') {
      lines.push('## ' + (state.settings.userName || 'User'), '', m.content || '');
      for (const a of m.attachments || []) lines.push('', '> ' + a.name + ' (' + formatSize(a.size || 0) + ')');
      for (const img of m.images || []) {
        lines.push('', '> ' + img.name + ' (' + img.width + ' × ' + img.height + ', ' + formatSize(img.size || 0) + ')');
      }
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
  return lines.join('\n');
}

function exportConversation(c, format) {
  if (format === 'json') download(fileName(c) + '.json', JSON.stringify(c, null, 2), 'application/json');
  else download(fileName(c) + '.md', markdownOf(c.title || t('untitled'), c.messages), 'text/markdown');
}

// One answer of a discussion, with the question it answers.
function exportAnswer(c, index, format) {
  let q = index - 1;
  while (q >= 0 && c.messages[q].role !== 'user') --q;
  const messages = (q >= 0 ? [c.messages[q]] : []).concat([c.messages[index]]);
  const n = c.messages.slice(0, index + 1).filter((m) => m.role !== 'user').length;
  const name = fileName(c) + '-' + n;
  if (format === 'json') {
    const doc = { title: c.title || '', conversation: c.id, exported: new Date().toISOString(), messages };
    download(name + '.json', JSON.stringify(doc, null, 2), 'application/json');
  } else {
    download(name + '.md', markdownOf(c.title || t('untitled'), messages), 'text/markdown');
  }
}

/*
  Deletes a question with its answer, the messages up to the next question, and takes the
  documents attached to the question out of the search of the discussion.  The start of the
  context follows the messages that remain.
*/
async function deleteQuestion(c, index) {
  if (threadLocked()) return;
  const id = c.messages[index] && c.messages[index].id;
  if (!await confirmDialog(t('deleteQuestionTitle'), t('deleteQuestionText'))) return;
  // The discussion as it is once the deletion is confirmed, which another tab may have changed.
  c = liveCopy(c);
  index = c.messages.findIndex((x) => x.id === id);
  if (index < 0 || threadLocked()) return;
  // A compression in progress moves the start of the discussion: it ends first.
  if (compactions.has(c.id)) await compactions.get(c.id).catch(() => {});
  const m = c.messages[index];
  let end = index + 1;
  while (end < c.messages.length && c.messages[end].role !== 'user') ++end;
  const removed = end - index;
  // The documents and the images of the question leave, unless another question, a replay, shares them.
  const others = attachmentIds(c.messages.slice(0, index).concat(c.messages.slice(end)));
  for (const a of (m && m.attachments) || []) if (a.mode === 'search' && !others.has(a.id)) forgetDocument(c.id, a.id);
  const otherImages = imageIds(c.messages.slice(0, index).concat(c.messages.slice(end)));
  const gone = ((m && m.images) || []).filter((img) => !otherImages.has(img.id)).map((img) => img.id);
  c.messages.splice(index, removed);
  // The memory reads again from the question that took its place.
  if ((c.memoryMark || 0) > index) c.memoryMark = Math.max(index, (c.memoryMark || 0) - removed);
  const start = c.contextStart || 0;
  if (start >= end) c.contextStart = start - removed;
  else if (start > index) c.contextStart = index;
  // A question already compressed: the summary is written again from the exchanges before the
  // start, without it.
  if (c.summary && index < start) await rewriteSummary(c);
  c.updated = Date.now();
  await persist(c);
  await images.remove(gone).catch(() => {});
  if (state.current === c) renderMain();
  refreshViews();
}

// The documents attached to the questions of a list of messages, by their ids.
function attachmentIds(messages) {
  const ids = new Set();
  for (const x of messages) for (const a of x.attachments || []) ids.add(a.id);
  return ids;
}

async function rewriteSummary(c) {
  const start = c.contextStart || 0;
  c.summary = null;
  if (start === 0 || !roleModel(c.role)) return;
  const context = roleCapabilities(c.role).context || 8192;
  const room = context - Math.min(state.settings.maxTokens || 2048, Math.floor(context / 2));
  toast(t('compacting'));
  await compact(c, 0, start, room, 1 / 3.6, null).then((r) => applyCompaction(c, r)).catch(() => { c.summary = null; });
}

// The document search: the documents being indexed, one at a time in the background.
const indexing = new Map();   // document -> {owner, done, total, abort}
let indexQueue = Promise.resolve();

function queueIndex(owner, doc, persistent) {
  if (indexing.has(doc.id)) return indexQueue;
  const run = { owner, done: 0, total: 0, abort: new AbortController() };
  indexing.set(doc.id, run);
  showIndexing(doc.id);
  indexQueue = indexQueue.then(() => {
    // A document removed while it waited is not indexed.
    if (run.abort.signal.aborted) return null;
    return rag.indexDocument(owner, doc, {
      persistent, signal: run.abort.signal,
      onProgress: (done, total) => {
        run.done = done;
        run.total = total;
        showIndexing(doc.id);
      }
    });
  }).catch(() => {}).then(() => {
    if (indexing.get(doc.id) === run) indexing.delete(doc.id);
    showIndexing(doc.id);
  });
  return indexQueue;
}

// Stops the indexing, waiting or in progress, of the documents of an owner, of one of them,
// or of all of them.
function cancelIndex(owner, docId) {
  for (const [id, run] of indexing) {
    if ((owner == null || run.owner === owner) && (docId == null || id === docId)) {
      run.abort.abort();
      indexing.delete(id);
    }
  }
}

/*
    Takes a document, or the documents of an owner (a project or a discussion), out of the
    search: their indexing stops, and the passages it was computing are never written.
*/
function forgetDocument(owner, docId) {
  cancelIndex(owner, docId);
  return rag.removeDocument(owner, docId);
}

function forgetOwner(owner) {
  cancelIndex(owner);
  return rag.removeOwner(owner);
}

/*
    Deletes the passages of the documents that no longer belong to their project or to their
    discussion, such as those another tab wrote back while this one removed the document.
    indexed lists the documents that had passages before the lists were read: a document is
    saved in its project or its discussion before it is indexed, so that one indexed meanwhile
    is never taken for a removed one.
*/
async function pruneIndex(indexed) {
  if (!indexed.length) return;
  const documents = new Map();
  for (const p of state.projects) {
    documents.set(p.id, new Set((p.files || []).map((f) => f.id).concat((p.images || []).map((x) => x.id))));
  }
  for (const c of state.conversations) {
    const ids = new Set();
    for (const m of c.messages || []) for (const a of m.attachments || []) ids.add(a.id);
    documents.set(c.id, ids);
  }
  for (const { owner, doc } of indexed) {
    const ids = documents.get(owner);
    if (!ids || !ids.has(doc)) await rag.removeDocument(owner, doc);
  }
}

/*
    Deletes the stored images that neither a discussion nor a project refers to any longer, such as
    those of a question deleted in a tab closed meanwhile.  listed are the images of the database,
    read before the lists (images.prune).
*/
function pruneImages(listed) {
  if (!listed.length) return Promise.resolve(0);
  const references = new Map();
  for (const p of state.projects) references.set(p.id, new Set((p.images || []).map((x) => x.id)));
  for (const c of state.conversations) references.set(c.id, imageIds(c.messages || []));
  return images.prune(listed, references);
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
    const model = rag.embeddingModel();
    for (const f of p.files || []) {
      const s2 = st.get(f.id);
      if (!s2 || (model && s2.vectors < s2.passages)) queueIndex(p.id, f, true);
    }
    // An image joins the search once its description is written, or has failed.
    for (const img of p.images || []) {
      const s2 = st.get(img.id);
      if (img.state !== 'pending' && (!s2 || (model && s2.vectors < s2.passages))) queueIndex(p.id, imageDocument(img), true);
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

/*
    The descriptions of the images of the projects, which the background model writes one at a
    time (describeImage): image -> project, for the images waiting or being described by this tab.
*/
const describing = new Map();
let describeQueue = Promise.resolve();

// The text of an image of a project for the search: its description, else its name.
function imageDocument(img) {
  return { id: img.id, name: img.name, text: img.description || img.name };
}

function queueDescription(projectId, imageId) {
  if (describing.has(imageId)) return;
  describing.set(imageId, projectId);
  showProjectImage(projectId, imageId);
  describeQueue = describeQueue.then(() => describeImage(projectId, imageId)).catch(() => {}).then(() => {
    describing.delete(imageId);
    showProjectImage(projectId, imageId);
  });
}

// Asks the description of an image again, from the button of its line.
async function retryDescription(projectId, imageId) {
  if (await updateProjectImage(projectId, imageId, { state: 'pending' })) queueDescription(projectId, imageId);
}

/*
    Describes the images of the projects that have no description: those whose description failed,
    and those a closed tab left in progress, once the service answers with a background model that
    reads images.
*/
function describePending() {
  if (!serviceReady() || !backgroundReadsImages()) return;
  for (const p of state.projects) {
    for (const img of p.images || []) if (img.state !== 'ready') queueDescription(p.id, img.id);
  }
}

// The request of the description of an image, written in the language of the interface.
function describeMessages(url, name) {
  const lang = languageNames[language()] || 'English';
  return [{
    role: 'system',
    content: 'You describe images for the document search of a project. Write in ' + lang + ', as plain text without '
      + 'introduction, in at most 220 words: first the kind of image (photograph, screenshot, chart, diagram, table, '
      + 'scanned document, drawing...) and its subject, in one sentence; then the visible text, transcribed exactly '
      + 'as written, in its own language; then the elements and the data the image shows: objects, places, values, '
      + 'labels, axes and trends. Describe only what is visible.'
  }, {
    role: 'user',
    content: [{ type: 'image_url', image_url: { url, detail: 'auto' } },
      { type: 'text', text: 'Describe this image ("' + name + '") in ' + lang + '.' }]
  }];
}

// A description as the model writes it, without a heading, within 4,000 characters.
function cleanDescription(text) {
  const s = String(text || '').replace(/^\s*(description|beschreibung|descrizione|descripci[oó]n)\s*:\s*/i, '').trim();
  return s.length > 4000 ? keepStart(s, 4000) : s;
}

/*
    Changes an image of a project in the copy of this tab, which its page shows and saves, and in
    the stored project, read again so that the changes of the other tabs stay.  Returns the image,
    null when the project or the image was removed meanwhile.
*/
async function updateProjectImage(projectId, imageId, change) {
  const p = projectById(projectId);
  const img = p && (p.images || []).find((x) => x.id === imageId);
  if (!img) return null;
  const stored = await store.getProject(projectId).catch(() => null);
  const kept = stored && (stored.images || []).find((x) => x.id === imageId);
  if (stored && !kept) return null;
  Object.assign(img, change);
  try {
    if (kept) Object.assign(kept, change);
    await store.putProject(stored || p);
  } catch (e) {
    storageFailed(e);
  }
  broadcast({ type: 'lists' });
  showProjectImage(projectId, imageId);
  return img;
}

/*
    Asks the background model for the description of an image of a project, in the language of the
    interface: the kind of image, its visible text, its elements and its data.  The description is
    kept with the image and takes its place in the search of the project.  Without a background
    model that reads images, or after an error, the name of the image stands for its description,
    and the description is asked again later (describePending) or with the button of the image.
    One tab describes an image at a time.
*/
async function describeImage(projectId, imageId) {
  await exclusive('albireo-describe-' + imageId, async () => {
    const p = projectById(projectId);
    const img = p && (p.images || []).find((x) => x.id === imageId);
    if (!img) return;
    // Another tab may have described it meanwhile.
    const stored = await store.getProject(projectId).catch(() => null);
    const kept = stored && (stored.images || []).find((x) => x.id === imageId);
    if (kept && kept.state === 'ready') {
      Object.assign(img, kept);
      return;
    }
    let text = '';
    if (serviceReady() && backgroundReadsImages()) {
      const url = await images.dataUrl(imageId).catch(() => null);
      if (url) {
        try {
          text = cleanDescription((await backgroundChat(describeMessages(url, img.name), 1024, null, 0.2)).text);
        } catch (e) {
          console.warn('Albireo: the description of an image failed', e);
          text = '';
        }
      }
    }
    // The earlier passages of the image leave the search before its new state is saved, so that a tab closed
    // meanwhile leaves the image to index again (indexProjects).
    await forgetDocument(projectId, imageId).catch(() => {});
    const saved = await updateProjectImage(projectId, imageId, text
      ? { state: 'ready', description: text, language: language() } : { state: 'failed', description: '' });
    if (saved) queueIndex(projectId, imageDocument(saved), true);
  });
}

// The state of the description of an image of a project: {cls, label, spin, retry}.
function imageState(img) {
  if (describing.has(img.id) || img.state === 'pending') return { cls: 'pending', label: t('imageDescribing'), spin: true };
  if (img.state === 'ready') return { cls: 'ready', label: t('imageDescribed') };
  return { cls: 'failed', label: t('imageDescriptionFailed'), retry: true };
}

/*
    The line of an image on the page of its project: its thumbnail, its name and its size, the
    state of its description (in progress, ready, failed with the button that asks it again), the
    description, and the button that removes the image.
*/
function projectImageRow(p, img) {
  const st = imageState(img);
  const row = h('div', { class: 'row image-row', 'data-project-image': img.id },
    imageThumb(img, { list: p.images || [] }),
    h('div', { class: 'image-text' },
      h('div', { class: 'image-name' }, h('span', { class: 'label' }, img.name),
        h('span', { class: 'faint' }, img.width + ' × ' + img.height + ' · ' + formatSize(img.size || 0))),
      h('div', { class: 'image-state ' + st.cls, title: st.retry ? t('imageDescriptionFailedHint') : null },
        st.spin ? h('span', { class: 'spinner small' }) : h('span', { html: icon(st.retry ? 'info' : 'check') }),
        h('span', null, st.label),
        st.retry ? h('button', {
          class: 'button small', type: 'button', onclick: () => retryDescription(p.id, img.id)
        }, t('retry')) : null),
      img.state === 'ready' && img.description
        ? h('p', { class: 'image-description', title: img.description }, img.description) : null),
    iconButton('trash', t('delete'), () => {
      if (ui.projectPage && ui.projectPage.id === p.id) ui.projectPage.removeImage(img);
    }));
  return row;
}

// Shows the state of an image on the page of its project, when it is shown.
function showProjectImage(projectId, imageId) {
  if (state.view !== 'project' || state.projectId !== projectId || !ui.content) return;
  const p = projectById(projectId);
  const img = p && (p.images || []).find((x) => x.id === imageId);
  const row = ui.content.querySelector('[data-project-image="' + imageId + '"]');
  if (row && img) row.replaceWith(projectImageRow(p, img));
}

// Projects.
async function newProject(silent) {
  const name = await promptDialog(t('newProject'), '');
  if (name == null || !name.trim()) return null;
  const p = {
    id: store.uid(), name: name.trim(), description: '', instructions: '', files: [],
    color: projectColors[state.projects.length % projectColors.length], created: Date.now(), updated: Date.now()
  };
  try {
    await store.putProject(p);
  } catch (e) {
    storageFailed(e);
    return null;
  }
  state.projects = state.projects.concat([p]).sort((a, b) => a.name.localeCompare(b.name));
  broadcast({ type: 'lists' });
  refreshViews();
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
  const heading = h('h2', null, h('span', { class: 'dot', style: { background: p.color } }), h('span', null, p.name));
  const sub = h('p', { class: 'sub' }, p.description || '');
  // The name, the description and the color show at once in the page, the top bar, the side
  // panel and the other tabs.
  const save = async () => {
    p.name = name.value.trim() || p.name;
    p.description = description.value;
    p.instructions = instructions.value;
    p.updated = Date.now();
    try {
      await store.putProject(p);
    } catch (e) {
      storageFailed(e);
    }
    const i = state.projects.findIndex((x) => x.id === p.id);
    if (i >= 0) state.projects[i] = p;
    state.projects.sort((a, b) => a.name.localeCompare(b.name));
    heading.lastChild.textContent = p.name;
    heading.firstChild.style.background = p.color;
    sub.textContent = p.description || '';
    broadcast({ type: 'lists' });
    renderSidebar();
    renderTopbar();
  };
  for (const el of [name, description, instructions]) el.addEventListener('change', save);
  const colors = h('div', { class: 'colors' });
  for (const col of projectColors) {
    colors.append(h('button', {
      type: 'button', class: p.color === col ? 'on' : '', style: { background: col }, 'aria-label': col,
      onclick: async () => { p.color = col; await save(); renderProject(p); }
    }));
  }
  // The documents and the images, maxProjectImages of them at most: the documents are indexed, the images described.
  const places = () => Math.max(0, maxProjectImages - (p.images || []).length);
  const fileInput = h('input', {
    type: 'file', multiple: true, class: 'hidden', accept: documentAccept() + (places() ? ',image/*' : '')
  });
  fileInput.addEventListener('change', async () => {
    const files = Array.from(fileInput.files);
    fileInput.value = '';
    const added = await readFiles(files, null, places());
    const docs = added.filter((a) => a.kind !== 'image');
    const pictures = added.filter((a) => a.kind === 'image')
      .map((a) => Object.assign(imageRecord(a), { state: 'pending', description: '' }));
    p.files = (p.files || []).concat(docs);
    if (pictures.length) p.images = (p.images || []).concat(pictures);
    await save();
    // The images are stored once the project that holds them is (pruneImages).
    try {
      await images.keep(p.id, pictures, true);
    } catch (e) {
      storageFailed(e);
    }
    renderProject(p);
    for (const f of docs) queueIndex(p.id, f, true);
    for (const img of pictures) queueDescription(p.id, img.id);
  });
  const fileList = h('div', { class: 'list' });
  for (const f of p.files || []) {
    const indexState = h('span', { class: 'faint index-state', 'data-index-doc': f.id, 'data-index-owner': p.id });
    fileList.append(h('div', { class: 'row' }, formatBadge(f), h('span', { class: 'label' }, f.name),
      h('span', { class: 'faint' }, f.format + ' · ' + t('tokensEstimate', { n: formatNumber(estimateTokens(f.text)) })),
      indexState,
      iconButton('trash', t('delete'), async () => {
        const removed = forgetDocument(p.id, f.id);
        p.files = p.files.filter((x) => x.id !== f.id);
        await save();
        await removed;
        renderProject(p);
      })));
    indexText(p.id, f.id).then((text) => { indexState.textContent = text; });
  }
  // The images, each with the state of its description.
  ui.projectPage = {
    id: p.id,
    removeImage: async (img) => {
      const removed = forgetDocument(p.id, img.id);
      p.images = (p.images || []).filter((x) => x.id !== img.id);
      await save();
      await removed;
      await images.remove([img.id]);
      renderProject(p);
    }
  };
  const pictures = p.images || [];
  const imagesBlock = h('div', { class: 'knowledge-images' },
    h('div', { class: 'images-head' }, h('span', { html: icon('image') }), h('span', null, t('projectImages')),
      h('span', { class: 'count', title: t('projectImagesCount', { n: pictures.length, max: maxProjectImages }) },
        pictures.length + ' / ' + maxProjectImages)),
    h('p', { class: 'help' }, t('knowledgeImagesHelp')),
    pictures.length ? h('div', { class: 'list image-list' }, pictures.map((img) => projectImageRow(p, img))) : null);
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

  main.replaceChildren(h('div', { class: 'page' }, heading, sub,
    h('div', { class: 'buttons', style: { marginBottom: '16px' } },
      h('button', { class: 'button primary', type: 'button', onclick: () => newChat(p.id), html: icon('plus') + '<span>' + escapeHtml(t('startInProject')) + '</span>' })),
    h('div', { class: 'card' },
      h('div', { class: 'field' }, h('label', null, t('projectName')), name),
      h('div', { class: 'field' }, h('label', null, t('description')), description),
      h('div', { class: 'field' }, h('label', null, t('color')), colors)),
    h('div', { class: 'card' }, h('h3', { html: icon('edit') + escapeHtml(t('instructions')) }),
      h('p', { class: 'help' }, t('instructionsHelp')), h('div', { class: 'field' }, instructions)),
    h('div', { class: 'card' }, h('h3', { html: icon('book') + escapeHtml(t('knowledge')) }),
      h('p', { class: 'help' }, t('knowledgeHelp')), capacity, fileList, imagesBlock,
      h('div', { class: 'buttons', style: { marginTop: '10px' } },
        h('button', { class: 'button', type: 'button', onclick: () => fileInput.click(), html: icon('upload') + '<span>' + escapeHtml(t('addFiles')) + '</span>' }),
        fileInput)),
    h('div', { class: 'card' }, h('h3', { html: icon('chat') + escapeHtml(t('projectDiscussions')) }), convList),
    h('div', { class: 'buttons' },
      h('button', {
        class: 'button danger', type: 'button', html: icon('trash') + '<span>' + escapeHtml(t('deleteProject')) + '</span>',
        onclick: async () => {
          if (!await confirmDialog(t('deleteProject'), t('deleteProjectText', { name: p.name }))) return;
          // The documents leave the search, in memory and in the database, and the discussion
          // shown leaves the project as the stored ones do.
          await forgetOwner(p.id).catch(() => {});
          await store.deleteProject(p.id).catch(storageFailed);
          await images.removeOwner(p.id).catch(() => {});
          if (state.current && state.current.projectId === p.id) state.current.projectId = null;
          await reloadLists().catch(storageFailed);
          broadcast({ type: 'lists' });
          go('#/');
        }
      }))));
}

// Files.
/*
    Reads the files attached for the model of role (the documents and the images of a project
    for none): the documents become text; an image is attached when the model reads images,
    always to a project, and prepared (images.js) within room places, the images beyond them
    refused with a message that says why; a sound is attached only when the model reads it.  A
    PDF whose pages are images without text, a scan, is left out with a dialog, or offered as
    images of its pages to a model that reads images; a PDF some of whose pages are such images
    is added with a notice that counts them.  The images come back with kind 'image'.
*/
async function readFiles(files, role, room) {
  const out = [];
  const scans = [];
  const refused = [];
  const inputs = role ? roleInputs(role) : ['text', 'image'];
  let left = room || 0;
  for (const f of files) {
    if (f.size > maxFileBytes) {
      toast(t('tooLarge', { name: f.name, size: formatSize(f.size), limit: formatSize(maxFileBytes) }), true);
      continue;
    }
    const kind = fileKind(f);
    if (kind !== 'document' && !inputs.includes(kind)) {
      const model = role ? modelName(roleModel(role)) : '';
      toast(role ? t('inputRefused', { name: f.name, model, inputs: inputsLabel(inputs) })
        : t('unsupported', { name: f.name }), true);
      continue;
    }
    if (kind === 'image') {
      if (left <= 0) {
        refused.push(f.name);
        continue;
      }
      try {
        out.push(Object.assign({ kind: 'image' }, await images.prepare(f)));
        --left;
      } catch (e) {
        toast(t('imageUnreadable', { name: f.name }), true);
      }
      continue;
    }
    try {
      const r = await extractText(f);
      out.push({ id: store.uid(), name: f.name, size: f.size, format: r.format, text: r.text, added: Date.now() });
      if (r.unreadablePages) {
        toast(t(r.unreadablePages === 1 ? 'pdfPageWithoutText' : 'pdfPagesWithoutText',
          { name: f.name, n: r.unreadablePages, total: r.pages }), false, 9000);
      }
    } catch (e) {
      if (e.message === 'pdf-images') scans.push({ file: f, pages: e.pages || 0 });
      else if (e.message === 'unsupported') toast(t('unsupported', { name: f.name }), true);
      else if (e.message === 'legacy') toast(t('legacyFormat', { name: f.name }), true);
      else if (e.message === 'too-large') toast(t('tooLarge', { name: f.name, size: formatSize(f.size), limit: formatSize(maxFileBytes) }), true);
      else if (e.message === 'too-large-content') toast(t('tooLargeContent', { name: f.name }), true);
      else if (e.message === 'pdf-password') toast(t('pdfPassword', { name: f.name }), true, 9000);
      else if (e.message === 'pdf-invalid') toast(t('pdfInvalid', { name: f.name }), true);
      else if (e.message === 'pdf-reader') toast(t('pdfReader', { name: f.name }), true, 12000);
      else toast(t('readFailed', { name: f.name, error: e.message }), true);
    }
  }
  if (refused.length) toast(imageLimitText(refused, left, !role), true, 10000);
  for (const scan of scans) {
    if (role && inputs.includes('image')) {
      const pages = await scanPages(scan, left);
      out.push(...pages);
      left -= pages.length;
    } else {
      scanDialog(scan.file.name);
    }
  }
  return out;
}

/*
    Offers to attach the pages of a scanned PDF as images, to a model that reads them, within the
    places left for images: the pages are drawn by pdf.js, their longest side at images.maxSide
    pixels, the first ones when the places do not cover them all.  Returns the images of the
    pages, none when the user declines or when no place is left, which the dialog says.
*/
async function scanPages(scan, left) {
  const name = scan.file.name;
  const count = Math.min(scan.pages || left, left);
  const accepted = await dialog((box, close) => {
    const text = left > 0 ? t('scanOffer', { name, pages: scan.pages || '?', left })
      + (scan.pages > left ? ' ' + t('scanFirstPages', { n: count }) : '')
      : t('scanNoRoom', { name, max: maxContextImages });
    box.append(h('h2', { html: icon('image') + escapeHtml(t('pdfImagesTitle')) }), h('p', null, text),
      h('div', { class: 'actions' }, left > 0 ? [
        h('button', { class: 'button', type: 'button', onclick: () => close(false) }, t('cancel')),
        h('button', { class: 'button primary', type: 'button', onclick: () => close(true) }, t('scanAttach'))
      ] : h('button', { class: 'button primary', type: 'button', onclick: () => close(false) }, t('close'))));
  }, true);
  if (!accepted || count <= 0) return [];
  toast(t('scanRendering', { name }));
  try {
    const r = await pdfPages(scan.file, images.maxSide, count);
    const out = [];
    const base = name.replace(/\.pdf$/i, '');
    for (let i = 0; i < r.canvases.length; ++i) {
      const meta = await images.prepareCanvas(r.canvases[i], t('scanPageName', { name: base, n: i + 1 }));
      out.push(Object.assign({ kind: 'image' }, meta));
    }
    return out;
  } catch (e) {
    toast(t('readFailed', { name, error: e.message }), true);
    return [];
  }
}

// The dialog that tells a PDF of images was not added, shown once the previous one is closed.
let scanDialogs = Promise.resolve();

function scanDialog(name) {
  scanDialogs = scanDialogs.then(() => dialog((box, close) => {
    box.append(h('h2', { html: icon('image') + escapeHtml(t('pdfImagesTitle')) }),
      h('p', null, t('pdfImagesText', { name })), h('div', { class: 'actions' },
        h('button', { class: 'button primary', type: 'button', onclick: () => close(true) }, t('close'))));
  }, true));
}

// Images.
// The ids of the images of the questions of a list of messages.
function imageIds(messages) {
  const ids = new Set();
  for (const x of messages) for (const img of x.images || []) ids.add(img.id);
  return ids;
}

// The images of the questions of c from the message from on, the start of its context by default.
function contextImages(c, from) {
  let n = 0;
  if (!c) return n;
  for (let i = from == null ? c.contextStart || 0 : from; i < c.messages.length; ++i) {
    n += (c.messages[i].images || []).length;
  }
  return n;
}

function pendingImageCount() {
  return state.pending.filter((a) => a.kind === 'image').length;
}

// The places left for images in the context of c, the images waiting in the composer taken.
function imageRoom(c) {
  return c ? Math.max(0, maxContextImages - contextImages(c) - pendingImageCount()) : 0;
}

// Why images were not added: their names, the limit of a discussion or of a project, and the places left.
function imageLimitText(names, left, project) {
  return t(project ? 'projectImagesNotAdded' : 'imagesNotAdded',
    { names: names.join(', '), max: project ? maxProjectImages : maxContextImages, left: Math.max(0, left) });
}

// The description of an image as a question or a project keeps it.
function imageRecord(a) {
  return { id: a.id, name: a.name, type: a.type, size: a.size, width: a.width, height: a.height, added: a.added };
}

function imageLabel(img) {
  return img.name + ' · ' + img.width + ' × ' + img.height + ' · ' + formatSize(img.size || 0);
}

// The tooltip of a thumbnail, with the reason why its image no longer reaches the model.
function thumbTitle(button, outside) {
  button.classList.toggle('outside', outside);
  button.title = button.dataset.label + (outside ? '\n' + t('imageOutside') : '');
  button.setAttribute('aria-label', t('imageOpen', { name: button.dataset.name }) + (outside ? '. ' + t('imageOutside') : ''));
}

/*
    The thumbnail of an image, a button that opens it in the viewer.  options: list, the images the
    viewer goes through; outside, for an image whose exchange left the context, shown dimmed;
    onremove, which adds the button that removes the image.
*/
function imageThumb(img, options) {
  const o = options || {};
  const list = o.list || [img];
  const pic = h('img', { alt: '', draggable: 'false' });
  const button = h('button', { type: 'button', class: 'image-thumb', onclick: () => openViewer(list, list.indexOf(img), button) },
    pic);
  button.dataset.image = img.id;
  button.dataset.name = img.name;
  button.dataset.label = imageLabel(img);
  thumbTitle(button, !!o.outside);
  images.thumbUrl(img.id).then((url) => {
    if (url) {
      pic.src = url;
      return;
    }
    // An image this browser does not hold, such as one whose storage failed.
    button.classList.add('missing');
    button.dataset.label = img.name + ' · ' + t('imageMissing');
    thumbTitle(button, !!o.outside);
    pic.replaceWith(h('span', { html: icon('image') }));
  });
  if (!o.onremove) return button;
  return h('span', { class: 'image-pending' }, button, h('button', {
    type: 'button', class: 'image-remove', title: t('removeImage'), 'aria-label': t('removeImage') + ': ' + img.name,
    html: icon('close'), onclick: o.onremove
  }));
}

/*
    Shows images whole over the page, from the one at index in list.  Escape, a click beside the
    image and the close button close the viewer, the arrows go through the images, and the focus
    stays in the viewer, then goes back to opener.
*/
function openViewer(list, index, opener) {
  closeMenu();
  let at = Math.max(0, index);
  let sequence = 0;
  const pic = h('img', { class: 'viewer-image', alt: '' });
  const missing = h('div', { class: 'viewer-missing hidden' }, h('span', { html: icon('image') }), t('imageMissing'));
  const name = h('span', { class: 'viewer-name' });
  const info = h('span', { class: 'viewer-info' });
  const stage = h('div', { class: 'viewer-stage' }, pic, missing);
  const buttons = [];
  const show = async (k) => {
    at = (k + list.length) % list.length;
    const img = list[at];
    const mine = ++sequence;
    name.textContent = img.name;
    info.textContent = img.width + ' × ' + img.height + ' · ' + formatSize(img.size || 0)
      + (list.length > 1 ? ' · ' + (at + 1) + ' / ' + list.length : '');
    box.setAttribute('aria-label', img.name);
    pic.alt = img.name;
    const url = await images.fullUrl(img.id);
    if (mine !== sequence) return;
    pic.classList.toggle('hidden', !url);
    missing.classList.toggle('hidden', !!url);
    if (url) pic.src = url;
    else pic.removeAttribute('src');
  };
  if (list.length > 1) {
    buttons.push(iconButton('chevron', t('previousImage'), () => show(at - 1), 'icon-button viewer-prev'),
      iconButton('chevron', t('nextImage'), () => show(at + 1), 'icon-button viewer-next'));
  }
  const closeButton = iconButton('close', t('close'), () => done(), 'icon-button viewer-close');
  buttons.push(closeButton);
  const box = h('div', { class: 'viewer', role: 'dialog', 'aria-modal': 'true', tabindex: '-1' },
    h('div', { class: 'viewer-bar' }, h('span', { class: 'viewer-title' }, name, info), buttons), stage);
  const overlay = h('div', { class: 'overlay viewer-overlay' }, box);
  // The keys reach the viewer first, wherever the focus is, and go no further.
  const onKey = (e) => {
    if (e.key === 'Escape') done();
    else if (e.key === 'ArrowLeft' && list.length > 1) show(at - 1);
    else if (e.key === 'ArrowRight' && list.length > 1) show(at + 1);
    else if (e.key === 'Tab') {
      const i = buttons.indexOf(document.activeElement);
      buttons[i < 0 ? 0 : (i + (e.shiftKey ? buttons.length - 1 : 1)) % buttons.length].focus();
    } else return;
    e.preventDefault();
    e.stopPropagation();
  };
  const done = () => {
    document.removeEventListener('keydown', onKey, true);
    overlay.remove();
    if (opener && opener.isConnected) opener.focus();
  };
  overlay.addEventListener('mousedown', (e) => {
    if (e.target === overlay || e.target === stage) done();
  });
  document.addEventListener('keydown', onKey, true);
  document.body.append(overlay);
  show(at);
  closeButton.focus();
}

// Dims the images of the questions that left the context, in the thread shown.
function markOutside() {
  const c = state.current;
  if (!c || !ui.thread || !ui.thread.isConnected) return;
  const start = c.contextStart || 0;
  for (const el of ui.thread.querySelectorAll('.message.user[data-index]')) {
    const outside = Number(el.dataset.index) < start;
    for (const b of el.querySelectorAll('.image-thumb')) thumbTitle(b, outside);
  }
}

/*
    The files of a paste.  An image of the clipboard, a screenshot for instance, takes a name with
    the time of the paste instead of the one the browser gives all of them.
*/
function pastedFiles(e) {
  const files = Array.from(e.clipboardData ? e.clipboardData.files : []);
  const d = new Date();
  const pad = (n) => String(n).padStart(2, '0');
  const date = d.getFullYear() + '-' + pad(d.getMonth() + 1) + '-' + pad(d.getDate()) + ' ' + pad(d.getHours()) + '.'
    + pad(d.getMinutes()) + '.' + pad(d.getSeconds());
  return files.map((f, i) => {
    if (fileKind(f) !== 'image' || !/^(image\.[a-z0-9]+)?$/i.test(f.name || '')) return f;
    const ext = ((f.type || '').split('/')[1] || 'png').replace('jpeg', 'jpg');
    const name = t('pastedImage', { date }) + (files.length > 1 ? ' (' + (i + 1) + ')' : '') + '.' + ext;
    return new File([f], name, { type: f.type, lastModified: f.lastModified });
  });
}

// Empties the composer, the images prepared for it forgotten.
function clearPending() {
  images.drop(state.pending.filter((a) => a.kind === 'image').map((a) => a.id));
  state.pending = [];
}

// The discussion shown gives way to another one: an incognito discussion, which no list keeps, takes its images out of memory.
function forgetIncognito() {
  const c = state.current;
  if (c && c.incognito && c.messages.length) images.removeOwner(c.id).catch(() => {});
}

// Keeps the images of a question of c, in the database of the browser, or in memory for an incognito discussion.
async function keepImages(c, list) {
  if (!list || !list.length) return;
  try {
    await images.keep(c.id, list, !c.incognito);
  } catch (e) {
    storageFailed(e);
  }
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
  forgetIncognito();
  clearPending();
  state.current = newConversation(projectId);
  state.view = 'chat';
  if (location.hash !== '#/') history.pushState(null, '', '#/');
  renderAll();
  ui.input.focus();
}

function go(hash) {
  if (location.hash === hash) route();
  else location.hash = hash;
}

/*
    The count of the navigations.  A discussion read from the database is shown only when no
    later navigation took over meanwhile.
*/
let routeSequence = 0;

async function route() {
  closeMenu();
  const sequence = ++routeSequence;
  const hash = location.hash || '#/';
  let m;
  if ((m = /^#\/c\/([0-9a-f]+)$/.exec(hash))) {
    if (!(state.current && state.current.id === m[1])) {
      if (state.busy) stopGeneration();
      const c = deleted.has(m[1]) ? null : await store.getConversation(m[1]).catch(() => null);
      if (sequence !== routeSequence) return;
      if (!c) { go('#/'); return; }
      forgetIncognito();
      clearPending();
      state.current = c;
      state.incognito = false;
    }
    state.view = 'chat';
  } else if ((m = /^#\/p\/([0-9a-f]+)$/.exec(hash))) {
    state.view = 'project';
    state.projectId = m[1];
  } else if (hash === '#/licenses') {
    state.view = 'licenses';
  } else {
    state.view = 'chat';
    if (!state.current || state.current.messages.length || (state.current.incognito !== state.incognito)) {
      if (state.busy) stopGeneration();
      forgetIncognito();
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
  if (state.view === 'licenses') {
    bar.append(h('div', { class: 'title' }, t('licensesTitle')));
    return;
  }
  const c = state.current;
  const title = h('div', { class: 'title', contenteditable: c && c.messages.length ? 'true' : 'false', spellcheck: 'false' },
    c && c.title ? c.title : t('untitled'));
  title.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); title.blur(); } });
  title.addEventListener('blur', async () => {
    if (!c || !c.messages.length) return;
    const live = liveCopy(c);
    const v = title.textContent.trim();
    if (v && v !== live.title) {
      live.title = v;
      // A title of the user stays: the model no longer names the discussion.
      live.titleSource = 'user';
      await persist(live);
      refreshViews();
    } else if (!v) {
      title.textContent = live.title || t('untitled');
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
  // The functions without a model of their own share the general one, or the model that comes
  // with Albireo.
  const chosen = state.status && (state.status.selected || state.status.active);
  for (const r of roles) {
    const own = chosen && chosen[r];
    const model = roleModel(r);
    entries.push({
      label: t('role_' + r), icon: roleIcons[r], checked: c.role === r,
      desc: t('roleHint_' + r) + (model ? ' · ' + (own ? modelName(model) : t('roleShared')) : ''),
      // An answer in progress keeps the function it started with; the next question takes this one.
      action: async () => {
        const live = liveCopy(c);
        live.role = r;
        await saveSetting('defaultRole', r);
        if (live.messages.length) await persist(live);
        refreshViews();
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
  stopClock();
  if (state.view === 'project') {
    const p = projectById(state.projectId);
    ui.composerWrap.classList.add('hidden');
    ui.thread = null;
    if (p) renderProject(p);
    // A project deleted meanwhile, in this tab or in another one, leaves its page.
    else go('#/');
    return;
  }
  if (state.view === 'licenses') {
    ui.composerWrap.classList.add('hidden');
    ui.thread = null;
    renderLicenses();
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
  showElsewhere();
  startClock();
  scrollToEnd(true);
}

function renderWelcome() {
  const c = state.current;
  const name = state.settings.userName ? ', ' + state.settings.userName : '';
  // The function comes from the picker of the top bar; the suggestions follow it.
  const suggestions = h('div', { class: 'suggestions' });
  for (let i = 1; i <= 4; ++i) {
    const [title, prompt] = t('suggest_' + c.role + '_' + i).split('|');
    suggestions.append(h('div', {
      class: 'suggestion', onclick: () => { ui.input.value = prompt; autoGrow(); ui.input.focus(); }
    }, h('b', null, title), h('span', null, prompt)));
  }
  const p = c.projectId ? projectById(c.projectId) : null;
  let notice = null;
  if (state.statusError) notice = h('div', { class: 'error-box' }, t('serviceDown'));
  else if (state.status && !roleModel(c.role)) notice = h('div', { class: 'error-box' }, t('noModel'));
  // With the agent on, what it does and the tools it has.
  const agent = c.agent && roleCapabilities(c.role).tools
    ? h('div', { class: 'agent-note' }, h('span', { html: icon('tool') }), h('span', null, t('agentHelp'))) : null;
  ui.content.replaceChildren(h('div', { class: 'welcome' },
    h('img', { class: 'hero', src: 'img/albireo-mark.svg', alt: '' }),
    h('h1', { html: escapeHtml(t('welcome', { name })).replace('Albireo', '<span>Albireo</span>') }),
    h('p', null, p ? t('newProjectDiscussion', { name: p.name }) : t('welcomeSub')),
    notice, agent, suggestions));
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
  el.dataset.index = index;
  // The images of a question that left the context show dimmed: the model no longer receives them.
  const pictures = m.images || [];
  if (pictures.length) {
    const outside = index < (c.contextStart || 0);
    el.append(h('div', { class: 'attachments image-thumbs' }, pictures.map((img) => imageThumb(img, { list: pictures, outside }))));
  }
  if (m.attachments && m.attachments.length) {
    el.append(h('div', { class: 'attachments' }, m.attachments.map((a) => attachmentChip(a))));
  }
  // A question made of images alone shows no empty bubble.
  if (m.content || !pictures.length) el.append(h('div', { class: 'bubble' }, m.content));
  // The time of the question shows before its buttons, which appear when the pointer is over it.
  el.append(h('div', { class: 'meta' }, timeElement(m),
    iconButton('copy', t('copy'), () => copyText(m.content)),
    lockable(iconButton('history', t('replayQuestion'), () => replayQuestion(c, m)), 'replay', pictures.length),
    lockable(iconButton('edit', t('edit'), () => editMessage(c, m, index, el)), 'busy'),
    lockable(iconButton('trash', t('deleteQuestion'), () => deleteQuestion(c, index)), 'busy')));
  return el;
}

/*
    The time of a message in the language of the interface: relative within a day ("5 minutes
    ago", "just now" within a minute), the short date and the time beyond, the full date and time
    in its tooltip.  A message saved without its time shows none.
*/
function timeText(ts) {
  const age = Date.now() - ts;
  if (age < 60000) return t('justNow');
  if (age < 86400000) {
    const relative = new Intl.RelativeTimeFormat(language(), { numeric: 'auto' });
    return age < 3600000 ? relative.format(-Math.floor(age / 60000), 'minute')
      : relative.format(-Math.floor(age / 3600000), 'hour');
  }
  return new Intl.DateTimeFormat(language(), { dateStyle: 'short', timeStyle: 'short' }).format(new Date(ts));
}

function timeElement(m) {
  if (typeof m.created !== 'number' || !Number.isFinite(m.created)) return null;
  const full = new Intl.DateTimeFormat(language(), { dateStyle: 'full', timeStyle: 'medium' })
    .format(new Date(m.created));
  return h('time', { class: 'msg-time', datetime: new Date(m.created).toISOString(), title: full,
    'data-time': String(m.created) }, timeText(m.created));
}

// The relative times of the thread shown, written again every minute while it stays shown.
let clockTimer = 0;

function startClock() {
  clearInterval(clockTimer);
  clockTimer = setInterval(() => {
    if (!ui.thread || !ui.thread.isConnected) {
      stopClock();
      return;
    }
    for (const el of ui.thread.querySelectorAll('.msg-time[data-time]')) {
      el.textContent = timeText(Number(el.dataset.time));
    }
  }, 60000);
}

function stopClock() {
  clearInterval(clockTimer);
  clockTimer = 0;
}

/*
    Asks a question of the thread again: the same question, with its documents and its images,
    joins the end of the discussion as a new question, which the model answers there.  The earlier
    exchange stays as it is, the documents keep their passages and the images their Blobs, which
    the two questions share; the images take places in the context again.
*/
async function replayQuestion(c, m) {
  const live = liveCopy(c);
  if (threadLocked() || state.current !== live) return;
  const pictures = m.images || [];
  if (pictures.length > imageRoom(live)) {
    toast(replayLimitText(pictures.length), true, 10000);
    return;
  }
  await sendMessage(m.content || '', (m.attachments || []).map((a) => Object.assign({}, a))
    .concat(pictures.map((img) => Object.assign({ kind: 'image' }, img))));
}

// Why a question with n images cannot be asked again: the places left for images in the context.
function replayLimitText(n) {
  return t('replayImagesFull', { n, left: imageRoom(state.current), max: maxContextImages });
}

/*
    The buttons that change a discussion wait while it is being answered, here or in another tab
    (kind 'busy'), the buttons of an answer wait until it is written (kind 'turn'), and the button
    that asks a question again also waits for places for its images in the context (kind
    'replay').  They stay disabled meanwhile, and say why.
*/
function threadLocked() {
  return state.busy || answeredElsewhere(state.current);
}

/*
    Marks a button of kind 'busy', of kind 'turn' for an answer that live tells is being written,
    or of kind 'replay' for a question with live images.
*/
function lockable(button, kind, live) {
  button.dataset.lock = kind;
  button.dataset.title = button.title;
  if (kind === 'replay') button.dataset.images = String(live || 0);
  lockButton(button, kind === 'turn' ? !!live : threadLocked());
  return button;
}

function lockButton(button, busy) {
  if (button.dataset.lock === 'replay' && !busy) {
    const n = Number(button.dataset.images || 0);
    if (n > imageRoom(state.current)) {
      applyLock(button, true, replayLimitText(n));
      return;
    }
  }
  applyLock(button, busy);
}

function applyLock(button, locked, reason) {
  button.disabled = locked;
  button.title = locked ? reason || t('afterAnswer') : button.dataset.title;
}

// Gives the buttons of the thread shown the state of its discussion.
function lockThread() {
  if (!ui.content) return;
  const busy = threadLocked();
  for (const b of ui.content.querySelectorAll('button[data-lock]')) {
    lockButton(b, b.dataset.lock === 'turn' ? !!(liveTurn && b.closest('.message.live')) : busy);
  }
  showElsewhere();
}

// The note at the end of the thread while another tab answers in its discussion.
function showElsewhere() {
  if (!ui.thread || !ui.thread.isConnected) return;
  const note = ui.thread.querySelector(':scope > .elsewhere');
  const on = !state.busy && answeredElsewhere(state.current);
  if (on && !note) {
    const spin = h('span', { class: 'spinner' });
    ui.thread.append(h('div', { class: 'progress-line elsewhere' }, spin, t('busyElsewhere')));
  } else if (!on && note) {
    note.remove();
  }
}

// Edits a question: its text, and its attached documents and images, which can be removed or added.
function editMessage(c, m, index, el) {
  if (threadLocked()) return;
  let attachments = (m.attachments || []).concat((m.images || []).map((img) => Object.assign({ kind: 'image' }, img)));
  // The images prepared for this edition, forgotten when it is cancelled.
  const prepared = new Set();
  /*
      The places for the images of the edited question: the questions after it leave, the exchanges
      before it come back into the context when it was summarized, and the images waiting in the
      composer keep theirs.
  */
  const places = () => maxContextImages - pendingImageCount()
    - (contextImages(c, (c.contextStart || 0) > index ? 0 : c.contextStart || 0) - contextImages(c, index));
  const room = () => Math.max(0, places() - attachments.filter((a) => a.kind === 'image').length);
  const area = h('textarea', { style: { width: '100%', minHeight: '90px', borderRadius: '12px', padding: '10px', border: '1px solid var(--border-strong)', background: 'var(--bg-input)' } });
  area.value = m.content;
  const chips = h('div', { class: 'attachments', style: { justifyContent: 'flex-start', marginBottom: '6px' } });
  const fileInput = h('input', { type: 'file', multiple: true, class: 'hidden' });
  const remove = (a) => {
    attachments = attachments.filter((x) => x !== a);
    if (prepared.has(a.id)) images.drop([a.id]);
    showChips();
  };
  const showChips = () => {
    const pictures = attachments.filter((a) => a.kind === 'image');
    chips.replaceChildren(...pictures.map((a) => imageThumb(a, { list: pictures, onremove: () => remove(a) })),
      ...attachments.filter((a) => a.kind !== 'image').map((a) => attachmentChip(a, true, () => remove(a))));
    fileInput.accept = acceptFor(c.role, room());
  };
  const add = async (files) => {
    const added = await readFiles(files, c.role, room());
    for (const a of added) if (a.kind === 'image') prepared.add(a.id);
    attachments = attachments.concat(added);
    showChips();
  };
  showChips();
  fileInput.addEventListener('change', () => {
    const files = Array.from(fileInput.files);
    fileInput.value = '';
    add(files);
  });
  area.addEventListener('paste', (e) => {
    const files = pastedFiles(e);
    if (files.length) {
      e.preventDefault();
      add(files);
    }
  });
  const box = h('div', { style: { width: '85%' } }, chips, area, h('div', { class: 'buttons', style: { justifyContent: 'flex-end', marginTop: '6px' } },
    h('button', { class: 'button', type: 'button', onclick: () => fileInput.click(), html: icon('clip') + '<span>' + escapeHtml(t('attach')) + '</span>' }),
    fileInput,
    h('span', { class: 'grow' }),
    h('button', {
      class: 'button', type: 'button',
      onclick: () => {
        images.drop(Array.from(prepared));
        renderMain();
      }
    }, t('cancel')),
    h('button', {
      class: 'button primary', type: 'button',
      onclick: async () => {
        if (threadLocked()) return;
        const text = area.value.trim();
        if (!text && !attachments.length) return;
        if (compactions.has(c.id)) await compactions.get(c.id).catch(() => {});
        // A question sent meanwhile is being answered: the messages stay as they are.
        if (threadLocked() || state.current !== c) return;
        // The images keep to the places of the context, which another tab may have taken meanwhile.
        const pictures = attachments.filter((a) => a.kind === 'image');
        const left = places();
        if (pictures.length > left) {
          toast(imageLimitText(pictures.slice(Math.max(0, left)).map((a) => a.name), left), true, 10000);
          return;
        }
        /*
            The documents no longer attached, to this question or to the questions after it,
            which leave with it, leave the search of the discussion, unless an earlier question
            shares them.  The memory reads the discussion again from the edited question.
        */
        const kept = attachmentIds(c.messages.slice(0, index));
        for (const a of attachments) kept.add(a.id);
        // The images of the questions that leave go too, unless a question kept or the edited one holds them.
        const keptImages = imageIds(c.messages.slice(0, index));
        for (const a of pictures) keptImages.add(a.id);
        const gone = [];
        for (const x of c.messages.slice(index)) {
          for (const a of x.attachments || []) if (!kept.has(a.id) && a.mode === 'search') forgetDocument(c.id, a.id);
          for (const img of x.images || []) if (!keptImages.has(img.id)) gone.push(img.id);
        }
        images.remove(gone).catch(() => {});
        c.messages = c.messages.slice(0, index);
        if ((c.memoryMark || 0) > index) c.memoryMark = index;
        if ((c.contextStart || 0) > index) {
          // The exchanges come back into the context, without the summary of what follows.
          c.contextStart = 0;
          c.summary = null;
        }
        await sendMessage(text, attachments);
      }
    }, t('send'))));
  box.addEventListener('dragover', (e) => e.preventDefault());
  box.addEventListener('drop', (e) => {
    e.preventDefault();
    if (!e.dataTransfer || !e.dataTransfer.files.length) return;
    add(Array.from(e.dataTransfer.files));
  });
  el.replaceChildren(box);
  area.focus();
}

// The text of an answer, as it is when it is copied.
function answerText(m) {
  return (m.steps || []).map((s) => s.content || '').join('\n\n').trim();
}

function assistantElement(c, m, index, last) {
  const live = m === liveTurn;
  const el = h('div', { class: 'message assistant' + (last ? ' last' : '') + (live ? ' live' : '') });
  const bubble = h('div', { class: 'bubble' });
  el.append(bubble);
  fillAssistant(bubble, m, live);
  // The buttons of an answer wait until it is written; regenerate also waits for the discussion.
  const meta = h('div', { class: 'meta' });
  meta.append(lockable(iconButton('copy', t('copy'), () => copyText(answerText(m))), 'turn', live));
  if (last) meta.append(lockable(iconButton('refresh', t('regenerate'), () => regenerate(liveCopy(c))), 'busy'));
  const exporter = lockable(iconButton('download', t('exportAnswer'), () => showMenu(exporter, [
    { label: t('exportMarkdown'), icon: 'download', action: () => exportAnswer(liveCopy(c), index, 'md') },
    { label: t('exportJson'), icon: 'archive', action: () => exportAnswer(liveCopy(c), index, 'json') }
  ])), 'turn', live);
  meta.append(exporter);
  // The time of the answer shows after its buttons.
  const when = timeElement(m);
  if (when) meta.append(when);
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

// The HTML of the finished texts of the answers, with the text it was made from (WeakMap: step
// -> {text, html}), so that a thread is drawn again without parsing its answers again.
const markdownCache = new WeakMap();

function stepHtml(s, live) {
  if (live) return renderMarkdown(s.content, true);
  const hit = markdownCache.get(s);
  if (hit && hit.text === s.content) return hit.html;
  const html = renderMarkdown(s.content, false);
  markdownCache.set(s, { text: s.content, html });
  return html;
}

// What a step shows, to tell whether it must be drawn again while the answer is written.
function stepKey(s, active) {
  const calls = (s.toolCalls || []).map((tc) => [tc.status, tc.ok, (tc.arguments || '').length,
    tc.result == null ? -1 : tc.result.length, tc.ms || 0, !!tc.file]);
  return JSON.stringify([active, (s.reasoning || '').length, (s.content || '').length, calls]);
}

function stepElement(s, active, firstTool) {
  const el = h('div', { class: 'step' });
  if (s.reasoning) el.append(thinkingElement(s, active && !s.content && !(s.toolCalls || []).length));
  if (s.content) {
    const md = h('div', { class: 'md' + (active ? ' cursor-host' : ''), html: stepHtml(s, active) });
    attachCodeButtons(md, t('copy'));
    el.append(md);
  }
  (s.toolCalls || []).forEach((tc, j) => el.append(toolElement(tc, firstTool + j + 1)));
  return el;
}

/*
    Fills the body of an answer: per step, its reasoning, its text and its tool calls, then the
    notes on its state.  While the answer is written, the steps already drawn that did not change
    stay as they are, and only the step being written is drawn again.
*/
function fillAssistant(bubble, m, live) {
  const steps = m.steps || [];
  const drawn = Array.from(bubble.children).filter((x) => x.classList.contains('step'));
  for (const x of Array.from(bubble.children)) if (!x.classList.contains('step')) x.remove();
  let tools = 0;
  steps.forEach((s, k) => {
    const active = live && k === steps.length - 1;
    const key = stepKey(s, active);
    const first = tools;
    tools += (s.toolCalls || []).length;
    const old = drawn[k];
    if (old && old.dataset.key === key) return;
    const el = stepElement(s, active, first);
    el.dataset.key = key;
    if (old) old.replaceWith(el);
    else bubble.append(el);
  });
  for (let k = steps.length; k < drawn.length; ++k) drawn[k].remove();
  const note = (...children) => bubble.append(h('div', { class: 'progress-line' }, ...children));
  if (live && m.progress && m.progress.prompt_total > 256 && m.progress.prompt_done < m.progress.prompt_total) {
    const pct = Math.floor(100 * m.progress.prompt_done / m.progress.prompt_total);
    note(h('span', { class: 'spinner' }), t('readingContext', { p: pct }),
      h('div', { class: 'bar' }, h('div', { style: { width: pct + '%' } })));
  } else if (live && m.compacting) {
    note(h('span', { class: 'spinner' }), h('span', { class: 'shimmer' }, t('compacting')));
  } else if (live && m.preparing) {
    note(h('span', { class: 'spinner' }), h('span', { class: 'shimmer' }, t('preparingContext')));
  } else if (live && !steps.some((s) => s.content || s.reasoning || (s.toolCalls || []).length)) {
    note(h('span', { class: 'spinner' }), h('span', { class: 'shimmer' }, t('thinking') + '…'));
  }
  if (m.compacted) note(h('span', { html: icon('info') }), t('contextCompacted'));
  else if (m.trimmed) note(h('span', { html: icon('info') }), t('contextTrimmed'));
  if (m.imagesNamed) note(h('span', { html: icon('image') }), t('imagesNamed', { model: modelName(m.model) }));
  if (m.error) bubble.append(h('div', { class: 'error-box' }, t('errorPrefix', { message: m.error })));
  if (m.cancelled) note(t('cancelled'));
  if (!live && !m.error && !m.cancelled && !steps.some((s) => s.content || (s.toolCalls || []).length)) {
    note(t('emptyAnswer'));
  }
  if (m.agentLimit) note(t('agentLimit', { n: maxAgentSteps }));
}

function thinkingElement(s, live) {
  const d = h('details', { class: 'thinking' + (live ? ' live' : '') });
  // Closed, while the model thinks and afterwards: the user opens it, and it then stays as
  // the user left it.
  const open = !!s.expanded;
  if (open) d.open = true;
  d.addEventListener('toggle', () => { if (d.open !== open) s.expanded = d.open; });
  let label;
  if (live) label = h('span', { class: 'shimmer' }, t('thinking') + '…' + (s.reasoningStart ? ' ' + formatDuration(Date.now() - s.reasoningStart) : ''));
  else label = h('span', null, s.reasoningMs ? t('thoughtFor', { s: formatDuration(s.reasoningMs) }) : t('thoughtBriefly'));
  d.append(h('summary', null, h('span', { html: icon('spark', 'spark') }), label, h('span', { html: icon('chevron', 'chev') })),
    h('div', { class: 'thought' }, s.reasoning));
  return d;
}

const toolIcons = {
  calculator: 'calc', run_javascript: 'terminal', search_documents: 'search', document_outline: 'archive',
  read_document: 'book', search_discussions: 'history', current_datetime: 'clock', create_file: 'file'
};

// The name of a tool in the language of the interface, its own name for a tool without one.
function toolLabel(name) {
  const key = 'tool_' + name;
  const label = t(key);
  return label === key ? String(name || '') : label;
}

// The words that sum up the arguments of a tool call, for the line of the call.
function toolBrief(tc) {
  let a = null;
  try {
    a = JSON.parse(tc.arguments || '{}');
  } catch (e) {
    return String(tc.arguments || '').replace(/\s+/g, ' ').slice(0, 160);
  }
  if (!a || typeof a !== 'object') return '';
  const text = (v) => (typeof v === 'string' ? v : v == null ? '' : JSON.stringify(v));
  const quoted = (v) => (v ? '"' + text(v) + '"' : '');
  switch (tc.name) {
    case 'calculator': return text(a.expression);
    case 'run_javascript': {
      const lines = text(a.code).trim().split('\n');
      return lines[0] + (lines.length > 1 ? ' …' : '');
    }
    case 'search_documents': case 'search_discussions': return quoted(a.query);
    case 'read_document': return [a.name, a.part].filter(Boolean).map(text).join(' · ');
    case 'document_outline': case 'create_file': return text(a.name);
    default: return Object.values(a).map(text).join(', ').slice(0, 160);
  }
}

// The state of a tool call: being written by the model, running, done, failed or stopped.
function toolState(tc) {
  if (tc.status === 'pending') return { cls: 'pending', label: t('toolPending'), spin: true };
  if (tc.status === 'running') return { cls: 'running', label: t('toolRunning'), spin: true };
  const time = tc.ms != null ? ' · ' + formatShortDuration(tc.ms) : '';
  if (tc.status === 'stopped') return { cls: 'stopped', label: t('toolStopped') + time };
  return tc.ok === false ? { cls: 'error', label: t('toolError') + time } : { cls: 'ok', label: t('toolDone') + time };
}

/*
    A tool call of the agent, with its number in the answer, the name of the tool in the language
    of the interface, the words of its arguments, a short result on the same line, its state and
    its duration.  Unfolded, it shows its arguments and its whole result.
*/
function toolElement(tc, number) {
  const st = toolState(tc);
  const d = h('details', { class: 'tool-call ' + st.cls });
  const open = !!tc.open;
  if (open) d.open = true;
  d.addEventListener('toggle', () => { if (d.open !== open) tc.open = d.open; });
  const brief = toolBrief(tc);
  // The value of a result of the interpreter shows without its arrow.
  const result = tc.result == null ? '' : String(tc.result).trim().replace(/^=> /, '');
  // A result of one short line shows on the line of the call.
  const inline = st.cls === 'ok' && result && result.length <= 60 && !result.includes('\n') ? result : '';
  let args = tc.arguments || '';
  try { args = JSON.stringify(JSON.parse(args), null, 2); } catch (e) { /* kept as sent */ }
  d.append(h('summary', { title: tc.name },
    number ? h('span', { class: 'tool-number' }, String(number)) : null,
    h('span', { html: icon(toolIcons[tc.name] || 'tool', 'tool-icon') }),
    h('span', { class: 'tool-name' }, toolLabel(tc.name)),
    h('span', { class: 'tool-brief' }, brief, inline ? h('span', { class: 'tool-result' }, ' → ' + inline) : null),
    h('span', { class: 'state ' + st.cls }, st.spin ? h('span', { class: 'spinner small' }) : null, st.label),
    h('span', { html: icon('chevron', 'chev') })),
  h('div', { class: 'io' }, h('h6', null, t('toolInput') + ' · ' + tc.name), h('pre', null, h('code', null, args)),
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
  const reasoning = !!(c && c.reasoning && caps.reasoning);
  const agent = !!(c && c.agent && caps.tools);
  ui.reasonButton.classList.toggle('on', reasoning);
  ui.reasonButton.setAttribute('aria-pressed', String(reasoning));
  ui.reasonButton.disabled = !caps.reasoning;
  ui.agentButton.classList.toggle('on', agent);
  ui.agentButton.setAttribute('aria-pressed', String(agent));
  ui.agentButton.disabled = !caps.tools;
  // With the agent on, the tools it may call show under the prompt.
  ui.agentTools.classList.toggle('hidden', !agent);
  // The attachments the model of the function reads, the images while the context has places for them.
  const role = c ? c.role : 'general';
  const room = imageRoom(c);
  if (ui.fileInput) ui.fileInput.accept = acceptFor(role, room);
  if (ui.attachButton) {
    let hint = t('attachInputs', { inputs: inputsLabel(roleInputs(role)) });
    if (readsImages(role)) hint += ' · ' + t('imagePlaces', { left: room, max: maxContextImages });
    ui.attachButton.title = hint;
    ui.attachButton.setAttribute('aria-label', hint);
  }
  // A discussion another tab is answering takes its next question once that answer is written.
  const elsewhere = !state.busy && answeredElsewhere(c);
  ui.sendButton.classList.toggle('stop', state.busy);
  ui.sendButton.innerHTML = icon(state.busy ? 'stop' : 'send');
  ui.sendButton.title = state.busy ? t('stop') : elsewhere ? t('busyElsewhere') : t('send');
  ui.sendButton.setAttribute('aria-label', ui.sendButton.title);
  ui.sendButton.disabled = sendDisabled();
  const p = c && c.projectId ? projectById(c.projectId) : null;
  ui.input.placeholder = p ? t('placeholderProject', { project: p.name }) : t('placeholder');
  // The images waiting, then the documents; the images show how many the context holds with them.
  const pictures = state.pending.filter((a) => a.kind === 'image');
  const count = pictures.length ? maxContextImages - room : 0;
  ui.pendingBox.replaceChildren(...pictures.map((a) => imageThumb(a, { list: pictures, onremove: () => removePending(a) })),
    ...state.pending.filter((a) => a.kind !== 'image').map((a) => attachmentChip(a, true, () => removePending(a))),
    pictures.length ? h('span', { class: 'image-count', title: t('imageCountTitle', { n: count, max: maxContextImages }) },
      count + ' / ' + maxContextImages) : null);
  ui.pendingBox.classList.toggle('hidden', !state.pending.length);
}

// Takes an attachment out of the composer; the places of an image go back to the context.
function removePending(a) {
  state.pending = state.pending.filter((x) => x !== a);
  if (a.kind === 'image') images.drop([a.id]);
  renderComposerState();
  lockThread();
  if (ui.input) ui.input.focus();
}

/*
    True when the send button can do nothing, without a question or while another tab answers in
    the discussion.  While this tab answers, the button stops the answer.
*/
function sendDisabled() {
  if (state.busy) return false;
  if (answeredElsewhere(state.current)) return true;
  return !ui.input.value.trim() && !state.pending.length;
}

function autoGrow() {
  ui.input.style.height = 'auto';
  ui.input.style.height = Math.min(ui.input.scrollHeight, 260) + 'px';
  ui.sendButton.disabled = sendDisabled();
}

/*
    Adds files to the composer of the discussion shown, the images within the places its context
    has left.  The files read while another discussion took its place are forgotten.
*/
async function addPending(files) {
  const c = state.current;
  let added = await readFiles(files, c ? c.role : 'general', imageRoom(c));
  if (state.current !== c) {
    images.drop(added.filter((a) => a.kind === 'image').map((a) => a.id));
    return;
  }
  // Two additions read together keep to the places as well.
  const room = imageRoom(c);
  const pictures = added.filter((a) => a.kind === 'image');
  if (pictures.length > room) {
    const extra = pictures.slice(room);
    images.drop(extra.map((a) => a.id));
    added = added.filter((a) => !extra.includes(a));
    toast(imageLimitText(extra.map((a) => a.name), 0), true, 10000);
  }
  state.pending = state.pending.concat(added);
  renderComposerState();
  lockThread();
}

function stopGeneration() {
  if (state.abort) state.abort.abort();
}

async function onSend() {
  if (state.busy) { stopGeneration(); return; }
  if (sendDisabled()) return;
  // The images keep to the places of the context, which another tab may have taken meanwhile.
  const pictures = state.pending.filter((a) => a.kind === 'image');
  const left = maxContextImages - contextImages(state.current);
  if (pictures.length > left) {
    toast(imageLimitText(pictures.slice(Math.max(0, left)).map((a) => a.name), left), true, 10000);
    return;
  }
  const text = ui.input.value.trim();
  const attachments = state.pending;
  state.pending = [];
  ui.input.value = '';
  autoGrow();
  await sendMessage(text, attachments);
}

/*
    Starts an answer in the discussion c.  The composer offers to stop it, the buttons that change
    the discussion wait, the other tabs know it, and the questions wait for its end.  Returns the
    controller that stops it.
*/
function beginTurn(c) {
  state.busy = true;
  state.busyId = c.id;
  state.abort = new AbortController();
  announceBusy(c, true);
  renderComposerState();
  lockThread();
  return state.abort;
}

// A new answer of the discussion c, drawn live from now on.
function newTurn(c) {
  const turn = { id: store.uid(), role: 'assistant', steps: [], created: Date.now(), model: roleModel(c.role) || '' };
  c.messages.push(turn);
  liveTurn = turn;
  return turn;
}

async function sendMessage(text, attachments) {
  const c = state.current;
  if (state.busy || answeredElsewhere(c)) return;
  const abort = beginTurn(c);
  const all = attachments || [];
  const m = {
    id: store.uid(), role: 'user', content: text, attachments: all.filter((a) => a.kind !== 'image'), created: Date.now()
  };
  const pictures = all.filter((a) => a.kind === 'image').map(imageRecord);
  if (pictures.length) m.images = pictures;
  // The update of the memory gives way to the question, and comes back once the user is quiet.
  if (memoryAbort) memoryAbort.abort();
  c.messages.push(m);
  // A new discussion starts with the memory as it is now.
  if (c.messages.length === 1 && !c.incognito && state.settings.memoryEnabled && state.settings.memory) {
    c.memory = state.settings.memory;
  }
  attachmentModes(c, m);
  /*
      The beginning of the first question stands for the title until the model writes one, and
      the title of a discussion whose first question is edited stays.
  */
  if (c.messages.length === 1 && !c.title) {
    c.title = provisionalTitle(m);
    c.titleSource = 'question';
  }
  c.updated = Date.now();
  // The answer shows at once, its buttons disabled, while its context is prepared.
  const turn = newTurn(c);
  turn.preparing = true;
  try {
    // The question is saved before its documents are indexed (pruneIndex) and before its images are stored
    // (pruneImages).
    renderMain();
    await persist(c);
    await keepImages(c, pictures);
    if (!c.incognito && state.current === c && location.hash !== '#/c/' + c.id) history.replaceState(null, '', '#/c/' + c.id);
    refreshViews();
    await prepareQuestion(c, m, abort.signal);
    await persist(c);
  } catch (e) {
    // A question that could not be saved is answered all the same, and saved with its answer.
  }
  turn.preparing = false;
  await runTurn(c, abort, turn);
}

async function regenerate(c) {
  if (state.busy || answeredElsewhere(c)) return;
  while (c.messages.length && c.messages[c.messages.length - 1].role === 'assistant') c.messages.pop();
  if (!c.messages.length) return;
  const abort = beginTurn(c);
  await runTurn(c, abort, newTurn(c));
}

// How the model reads the documents of a question: whole while they fit in their share of the
// context, searched beyond.
function attachmentModes(c, m) {
  const contextTokens = roleCapabilities(c.role).context || 8192;
  for (const a of m.attachments || []) {
    if (!a.mode) a.mode = estimateTokens(a.text) <= attachmentShare * contextTokens ? 'whole' : 'search';
  }
}

/*
    Prepares a question once, when it is sent: its large documents are indexed for the
    search of the discussion, and the excerpts of the searched documents closest to it are
    kept with it, so that the prompt of the following questions starts the same way.  signal
    stops the vectors of the documents, then searched by their words, and the search.
*/
async function prepareQuestion(c, m, signal) {
  const contextTokens = roleCapabilities(c.role).context || 8192;
  const owners = [];
  attachmentModes(c, m);
  for (const a of m.attachments || []) {
    if (a.mode === 'search') {
      // Indexed at once, beside the documents of the projects that may be in progress.
      toast(t('indexingDocument', { name: a.name }));
      await rag.indexDocument(c.id, a, { persistent: !c.incognito, signal }).catch(() => {});
    }
  }
  for (const x of c.messages) {
    if (x.role === 'user' && (x.attachments || []).some((a) => a.mode === 'search')) {
      owners.push(c.id);
      break;
    }
  }
  const p = c.projectId ? projectById(c.projectId) : null;
  // The descriptions of the images of the project are searched, even when its documents are read whole.
  const pictures = new Map(((p && p.images) || []).map((x) => [x.id, x]));
  let skip = null;
  if (p && knowledgeMode(p, contextTokens) === 'search') owners.unshift(p.id);
  else if (p && pictures.size) {
    owners.unshift(p.id);
    skip = new Set((p.files || []).map((f) => f.id));
  }
  m.context = '';
  delete m.projectImages;
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
    const hits = await rag.search(owners, question, { budget: Math.floor(excerptShare * contextTokens * 3.6), signal, skip });
    if (hits.length) {
      m.context = '<excerpts>\n' + hits.map((x) => (pictures.has(x.doc) ? '<excerpt image="' + x.name + '"'
        : '<excerpt document="' + x.name + '"' + (x.where ? ' location="' + x.where + '"' : '')) + '>\n' + x.text
        + '\n</excerpt>').join('\n') + '\n</excerpts>';
      // The images whose descriptions were found come with the question, the most relevant first.
      const found = hits.filter((x) => pictures.has(x.doc)).sort((a, b) => b.score - a.score)
        .slice(0, maxQuestionProjectImages).map((x) => imageRecord(pictures.get(x.doc)));
      if (found.length) m.projectImages = found;
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
  if (p && ((p.files || []).length || (p.images || []).length)) owners.push(p.id);
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
  if (p) for (const img of p.images || []) docs.push({ name: img.name, text: imageDocument(img).text });
  for (const m of c.messages) if (m.role === 'user') for (const a of m.attachments || []) docs.push({ name: a.name, text: a.text });
  return docs;
}

// The system message of a discussion; agent tells whether the tools come with the question.
function systemPrompt(c, contextTokens, agent) {
  const parts = [];
  const now = new Date();
  parts.push('Current date: ' + now.toLocaleDateString('en-CA') + ' (' + now.toLocaleDateString(language(), { weekday: 'long' }) + ').');
  if (state.settings.userName) parts.push('The user is named ' + state.settings.userName + '.');
  if (state.settings.systemPrompt) parts.push(state.settings.systemPrompt);
  // The memory as it was when the discussion started: its prompt keeps the same beginning.
  if (c.memory && state.settings.memoryEnabled && !c.incognito) {
    parts.push('What you remember from earlier discussions with the user, to use when it helps, without mentioning it '
      + 'otherwise:\n' + c.memory);
  }
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
        + (agent ? ' Use search_documents, document_outline and read_document to read more of them.' : ''));
    }
    const pictures = p.images || [];
    if (pictures.length) {
      parts.push('Images of the project: ' + pictures.map((x) => x.name).join(', ') + '. The description of an image '
        + 'comes with the questions it relates to, in <excerpt image="...">, and the image itself with it when you '
        + 'read images.');
    }
  }
  if (agent) parts.push(tools.agentInstructions);
  return parts.join('\n\n');
}

// The text of a question for the model; cuts (fitContext) shortens the oldest texts.
function userContent(m, cuts) {
  const cut = (key, text) => {
    const kept = cuts && cuts.texts.get(key);
    return kept == null ? text : keepEnd(text, kept);
  };
  let s = '';
  for (const a of m.attachments || []) {
    if (a.mode === 'search') {
      s += '<document name="' + a.name + '" format="' + (a.format || '') + '">\n(Too long to be read whole: its excerpts '
        + 'closest to the questions come with them.)\n</document>\n\n';
    } else {
      s += '<document name="' + a.name + '">\n' + cut(a.id, a.text || '') + '\n</document>\n\n';
    }
  }
  if (m.context) s += m.context + '\n\n';
  return s + cut(m.id, m.content || '');
}

// The parts of the requests that carry an image, with the description of the image, for the estimates of the fitting.
const imageParts = new WeakMap();

function imagePart(url, img) {
  const part = { type: 'image_url', image_url: { url, detail: 'auto' } };
  imageParts.set(part, img);
  return part;
}

// The images of a list of messages the model receives, "message/position": the newest maxContextImages of them.
function sentImages(messages) {
  const out = new Set();
  for (let i = messages.length - 1; i >= 0; --i) {
    const list = messages[i].images || [];
    for (let k = list.length - 1; k >= 0; --k) {
      if (out.size >= maxContextImages) return out;
      out.add(messages[i].id + '/' + k);
    }
  }
  return out;
}

/*
    The messages of a request.  media gives the images: read, whether the model reads images, and
    url(id), the data URL of an image, null when this browser does not hold it.  The images of a
    question go before its text, the newest maxContextImages of the discussion, and the images of
    the project found for the last question come with it; a model that reads text only, a request
    without media, and the images missing receive their names ("[Image: name]").
*/
function apiMessages(c, system, cuts, media) {
  const out = [];
  if (system) out.push({ role: 'system', content: system });
  // The exchanges before contextStart left the context in an earlier question; their summary
  // comes with the first question that stays.
  const start = c.contextStart || 0;
  const messages = c.messages.slice(start);
  const read = !!(media && media.read);
  const sent = read ? sentImages(messages) : null;
  let last = null;
  for (const m of messages) if (m.role === 'user') last = m;
  let summary = start > 0 && c.summary && c.summary.text ? c.summary.text : '';
  if (summary && cuts && cuts.summary != null) summary = keepEnd(summary, cuts.summary);
  for (const m of messages) {
    if (m.role === 'user') {
      const parts = [];
      const named = [];
      (m.images || []).forEach((img, k) => {
        const url = sent && sent.has(m.id + '/' + k) ? media.url(img.id) : null;
        if (url != null) parts.push(imagePart(url, img));
        else named.push('[Image: ' + img.name + ']');
      });
      if (read && m === last) {
        for (const img of (m.projectImages || []).slice(0, maxQuestionProjectImages)) {
          const url = media.url(img.id);
          if (url != null) parts.push(imagePart(url, img));
        }
      }
      let content = [named.join('\n'), userContent(m, cuts)].filter(Boolean).join('\n\n');
      if (summary) {
        content = '<summary_of_the_earlier_discussion>\n' + summary + '\n</summary_of_the_earlier_discussion>\n\n' + content;
        summary = '';
      }
      out.push({ role: 'user', content: parts.length ? parts.concat(content ? [{ type: 'text', text: content }] : []) : content });
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
    The images the request of a question of c carries, as data URLs read from their Blobs when it
    is sent: those of the questions of its context and those of the project found for its last
    question, for a model of role that reads images.  Returns {read, url(id)}, url giving null for
    an image this browser does not hold.
*/
async function requestMedia(c, role) {
  const read = readsImages(role);
  const urls = new Map();
  if (read) {
    const ids = imageIds(c.messages.slice(c.contextStart || 0));
    const questions = c.messages.filter((x) => x.role === 'user');
    const last = questions[questions.length - 1];
    for (const img of (last && last.projectImages) || []) ids.add(img.id);
    for (const id of ids) urls.set(id, await images.dataUrl(id).catch(() => null));
  }
  return { read, url: (id) => urls.get(id) || null };
}

// The images of the estimates, which hold no data: the fitting weighs them by their tokens.
function estimateMedia(role) {
  return { read: readsImages(role), url: () => '' };
}

// The tokens of each image for a model, measured by the service: model -> Map(image -> tokens).
const imageTokenCache = new Map();

// The tokens of an image for a model, as measured, else from its size: one token per 32 by 32 pixels.
function imageTokens(model, img) {
  const known = imageTokenCache.get(model);
  if (known && known.has(img.id)) return known.get(img.id);
  return Math.ceil((img.width || 512) / 32) * Math.ceil((img.height || 512) / 32) + 2;
}

/*
    Measures with the service the tokens of the images of a list of messages that it has not
    measured yet for model: the tokens of a question that holds the image, less those of the same
    question without it.  An image the service cannot count keeps its estimate.
*/
async function measureImages(model, list) {
  let known = imageTokenCache.get(model);
  if (!known) imageTokenCache.set(model, known = new Map());
  const parts = [];
  for (const m of list) {
    if (!Array.isArray(m.content)) continue;
    for (const part of m.content) {
      const img = imageParts.get(part);
      if (img && !known.has(img.id) && part.image_url.url) parts.push([img, part]);
    }
  }
  if (!parts.length) return;
  const base = await api.count(model, [{ role: 'user', content: '.' }]).catch(() => null);
  if (!base || typeof base.tokens !== 'number') return;
  for (const [img, part] of parts) {
    if (known.has(img.id)) continue;
    const r = await api.count(model, [{ role: 'user', content: [part, { type: 'text', text: '.' }] }]).catch(() => null);
    if (r && typeof r.tokens === 'number') known.set(img.id, Math.max(1, r.tokens - base.tokens));
  }
}

/*
    The weight of a list of messages for the estimates of the fitting: {chars, tokens}, the
    characters of its texts and the tokens of its images (imageTokens) for model.
*/
function weigh(list, model) {
  let chars = 0;
  let tokens = 0;
  for (const m of list) {
    if (!Array.isArray(m.content)) {
      chars += JSON.stringify(m).length;
      continue;
    }
    chars += JSON.stringify(Object.assign({}, m, { content: '' })).length;
    for (const part of m.content) {
      const img = imageParts.get(part);
      if (img) tokens += imageTokens(model, img);
      else chars += JSON.stringify(part).length;
    }
  }
  return { chars, tokens };
}

/*
    The estimate of the tokens of lists of messages, from a list measured at tokens: its images by
    their tokens, its texts by their characters, scaled.  Returns {scale, of(list)}, scale giving
    the tokens per character of the texts.
*/
function tokenEstimate(list, tokens, model) {
  const w = weigh(list, model);
  const scale = Math.max(0.02, (tokens - Math.min(w.tokens, 0.9 * tokens)) / Math.max(1, w.chars));
  return {
    scale,
    of: (msgs) => {
      const x = weigh(msgs, model);
      return x.chars * scale + x.tokens;
    }
  };
}

// Where text may be cut: after the end of a sentence followed by a space, or a blank line.
const sentenceBreak = /[.!?\u2026\u3002\uff01\uff1f]["'\u00bb\u201d)\]]*\s+|\n\s*\n/g;

/*
    The end of text, without the beginning that takes it over max characters: the cut comes
    after the end of a sentence, so that the oldest information leaves first and no sentence
    is left in part.  A text without a sentence break there is cut at a space.
*/
function keepEnd(text, max) {
  text = String(text || '');
  if (text.length <= max) return text;
  const from = text.length - Math.max(0, max - 2);
  sentenceBreak.lastIndex = from;
  const m = sentenceBreak.exec(text);
  let start = m ? m.index + m[0].length : text.indexOf(' ', from) + 1;
  if (start <= 0 || start >= text.length) start = from;
  return '\u2026 ' + text.slice(start);
}

// The beginning of text, at most max characters, ending with its last whole sentence.
function keepStart(text, max) {
  text = String(text || '');
  if (text.length <= max) return text;
  const head = text.slice(0, max);
  let end = 0;
  sentenceBreak.lastIndex = 0;
  for (let m; (m = sentenceBreak.exec(head)); ) end = m.index + m[0].length;
  if (end < max / 2) end = Math.max(head.lastIndexOf(' '), max / 2);
  return head.slice(0, end).trim() + ' \u2026';
}

// The exchanges as plain text, for their summary: the questions with their images named
// ("[Image: name]") and the names of their documents, the answers without their reasoning, the
// results of the tools in short.
function transcriptOf(messages) {
  const lines = [];
  for (const m of messages) {
    if (m.role === 'user') {
      const docs = (m.attachments || []).map((a) => a.name);
      const pictures = (m.images || []).map((img) => '[Image: ' + img.name + ']');
      lines.push('User: ' + (pictures.length ? pictures.join(' ') + '\n' : '') + (m.content || '')
        + (docs.length ? '\n(documents: ' + docs.join(', ') + ')' : ''));
      continue;
    }
    const parts = [];
    for (const s of m.steps || []) {
      for (const tc of s.toolCalls || []) {
        parts.push('[' + tc.name + ' ' + (tc.arguments || '') + ': ' + keepStart(tc.result || '', 600) + ']');
      }
      if (s.content) parts.push(s.content);
    }
    if (parts.length) lines.push('Assistant: ' + parts.join('\n'));
  }
  return lines.join('\n\n');
}

function summaryInstructions(words) {
  return 'You keep the memory of a long discussion between a user and an assistant. Rewrite the previous summary, '
    + 'when there is one, together with the new exchanges, into a single summary that the assistant will read in '
    + 'place of these exchanges. Keep, in the order they came: the facts, figures, names, decisions and '
    + 'conclusions; the requests, preferences and constraints of the user; the code, commands and values that may '
    + 'be needed again, in short; the documents and the images by their names; the questions left open. Leave out '
    + 'greetings, repetitions and reasoning. Write in the language of the latest exchanges, as a compact list of '
    + 'points, in at most ' + words + ' words.';
}

// A summary written by the background model from the previous one and a transcript.
async function summarize(c, previous, transcript, limit, signal) {
  const r = await backgroundText(summaryInstructions(Math.floor(limit * 0.6)),
    (previous ? '<previous_summary>\n' + previous + '\n</previous_summary>\n\n' : '')
      + '<exchanges>\n' + transcript + '\n</exchanges>', limit, signal);
  const text = r.text;
  const finish = r.finish;
  if (!text) throw new Error('empty summary');
  // A summary stopped by its length ends with its last whole sentence.
  return finish === 'length' ? keepStart(text, text.length - 1) : text;
}

/*
    Compresses the exchanges [from, to) of a discussion into its summary: the background model
    rewrites the summary with them, in parts that fit its context, oldest first.
    Returns {summary, to}, which applyCompaction gives to the discussion, then started at
    to, its summary given with its first question.  A summary grown beyond its share of the
    room loses its oldest points.
*/
async function compact(c, from, to, room, scale, signal) {
  const limit = Math.max(128, Math.min(1024, Math.floor(summaryShare * room)));
  // The parts fit the context of the background model, which writes the summary.
  room = Math.min(room, backgroundContext() - 64);
  let summary = (c.summary && c.summary.text) || '';
  // Characters of exchanges per part: the room, less the summary, its instructions and the
  // summary to write.
  const chars = (n) => Math.floor(n / Math.max(scale, 1e-6));
  let i = from;
  while (i < to) {
    const available = Math.max(chars(Math.floor(room / 4)), chars(room - limit - 400) - summary.length);
    let j = i;
    let length = 0;
    while (j < to) {
      const piece = transcriptOf([c.messages[j]]).length + 2;
      if (j > i && length + piece > available) break;
      length += piece;
      ++j;
    }
    summary = await summarize(c, summary, keepEnd(transcriptOf(c.messages.slice(i, j)), available), limit, signal);
    i = j;
  }
  return { summary: { text: keepEnd(summary, chars(limit)), upTo: to, updated: Date.now() }, to,
    last: c.messages[to - 1] && c.messages[to - 1].id };
}

function applyCompaction(c, r) {
  c.summary = r.summary;
  c.contextStart = r.to;
}

// The discussion as it is now, for the work that ends in the background: the one shown, else
// the one stored.  Null when it was deleted.
async function currentVersion(c) {
  if (deleted.has(c.id)) return null;
  if (state.current && state.current.id === c.id) return state.current;
  if (c.incognito) return c;
  return (await store.getConversation(c.id)) || null;
}

const compactions = new Map();   // discussion -> its compression in the background

/*
    The start of a discussion that leaves to the exchanges after it keptShare of the room of
    the context: the first question after the current start for which the conversation built
    by build fits, its tokens given by estimate, the last question at most.  Returns -1 when no
    question can leave.
*/
function nextStart(c, build, estimate, room) {
  const from = c.contextStart || 0;
  const users = [];
  for (let i = from + 1; i < c.messages.length; ++i) if (c.messages[i].role === 'user') users.push(i);
  // The question being answered stays: the last one of the messages, or the one before the
  // answer in progress.
  const lastUser = c.messages.map((m) => m.role).lastIndexOf('user');
  const candidates = users.filter((i) => i <= lastUser);
  if (!candidates.length) return -1;
  const saved = c.contextStart;
  let next = candidates[candidates.length - 1];
  for (const i of candidates) {
    c.contextStart = i;
    if (estimate(build()) <= keptShare * room) {
      next = i;
      break;
    }
  }
  c.contextStart = saved;
  return next;
}

/*
    Fits the conversation in the context of the model.  When it no longer fits, the oldest
    exchanges leave in one step, until the conversation takes keptShare of its room, and
    the point where it now starts stays with the conversation: the following questions
    start from that same point, so that the engine reuses what it already read, and the
    next cut comes only after several questions.  The exchanges that leave are compressed
    into the summary of the discussion, which the model receives with the first question
    that stays (compact); the exchanges leave without it when the model cannot write it.
    When the last question alone does not fit, the oldest texts lose their beginnings,
    at the end of a sentence: the summary first, then the documents of the questions, then
    the questions.  The system message and the last question always stay.  onCompact tells
    when a compression starts, and model names the model the tokens are counted for.  Returns
    {messages, trimmed, compacted, context, tokens}.
*/
async function fitContext(c, build, toolDefs, reserve, signal, onCompact, model) {
  let trimmed = false;
  let compacted = false;
  if (compactions.has(c.id)) {
    if (onCompact) onCompact();
    await compactions.get(c.id).catch(() => {});
  }
  const cuts = { summary: null, texts: new Map() };
  const counted = model || 'albireo-' + c.role;
  for (let attempt = 0; attempt < 12; ++attempt) {
    const list = build(cuts);
    // The images the service has not measured yet for this model, once.
    if (!attempt) await measureImages(counted, list);
    let r;
    try {
      r = await api.count(counted, list, toolDefs);
    } catch (e) {
      return { messages: list, trimmed, compacted, context: 0 };
    }
    if (!r || typeof r.tokens !== 'number' || !r.context) return { messages: list, trimmed, compacted, context: 0 };
    const room = r.context - Math.min(reserve, Math.floor(r.context / 2));
    if (r.tokens <= room) return { messages: list, trimmed, compacted, context: r.context, tokens: r.tokens };
    // The tokens of a list of messages: its images by their tokens, its texts from their characters scaled to the
    // measure.
    const estimate = tokenEstimate(list, r.tokens, counted);
    const scale = estimate.scale;
    const next = nextStart(c, () => build(cuts), estimate.of, room);
    if (next > (c.contextStart || 0)) {
      try {
        if (onCompact) onCompact();
        applyCompaction(c, await compact(c, c.contextStart || 0, next, room, scale, signal));
        compacted = true;
      } catch (e) {
        if (e.name === 'AbortError') throw e;
        c.contextStart = next;
        trimmed = true;
      }
      continue;
    }
    // Nothing more can leave whole: the oldest texts lose their beginnings.
    const excess = Math.ceil((r.tokens - room) / scale) + 200;
    if (!shortenOldest(c, cuts, excess)) {
      return { messages: list, trimmed, compacted, context: r.context, tokens: r.tokens, overflow: true };
    }
    trimmed = true;
  }
  return { messages: build(cuts), trimmed, compacted, context: 0 };
}

/*
    Takes excess characters from the beginnings of the oldest texts the model receives, the
    summary first, then the documents and the texts of the questions in their order: cuts
    holds the characters each of them keeps.  Returns false when nothing is left to take.
*/
function shortenOldest(c, cuts, excess) {
  const texts = [];
  if (c.summary && c.summary.text && (c.contextStart || 0) > 0) texts.push({ key: 'summary', text: c.summary.text });
  for (const m of c.messages.slice(c.contextStart || 0)) {
    if (m.role !== 'user') continue;
    for (const a of m.attachments || []) if (a.mode !== 'search') texts.push({ key: a.id, text: a.text || '' });
    texts.push({ key: m.id, text: m.content || '' });
  }
  for (const x of texts) {
    if (excess <= 0) break;
    const kept = x.key === 'summary' ? cuts.summary : cuts.texts.get(x.key);
    const length = kept == null ? x.text.length : kept;
    // A text keeps a short end: what remains of the question, its last sentences.
    const floor = Math.min(length, 200);
    const take = Math.min(excess, length - floor);
    if (take <= 0) continue;
    if (x.key === 'summary') cuts.summary = length - take;
    else cuts.texts.set(x.key, length - take);
    excess -= take;
  }
  return excess <= 0;
}

// The answer being written, drawn live.
let liveTurn = null;

/*
    The answer in progress is drawn again in the next frame, while its discussion is the one
    shown, at most once every renderInterval milliseconds, and less often when a drawing takes
    long, so that a long answer leaves the page responsive.
*/
const renderInterval = 100;
let renderSpacing = renderInterval;
let renderTimer = 0;
let renderFrame = 0;
let renderedAt = 0;

function scheduleRender(c, m) {
  if (renderTimer || renderFrame) return;
  const draw = () => {
    renderFrame = 0;
    const t0 = performance.now();
    renderedAt = t0;
    const el = answerElement(c, m);
    if (!el) return;
    const bubble = el.querySelector('.bubble');
    const thought = bubble.querySelector('.thinking.live .thought');
    const keepScroll = thought ? thought.scrollHeight - thought.scrollTop - thought.clientHeight < 40 : true;
    const top = thought ? thought.scrollTop : 0;
    fillAssistant(bubble, m, true);
    const t2 = bubble.querySelector('.thinking.live .thought');
    if (t2) t2.scrollTop = keepScroll ? t2.scrollHeight : top;
    scrollToEnd(false);
    renderSpacing = Math.min(1000, Math.max(renderInterval, 4 * (performance.now() - t0)));
  };
  const wait = renderSpacing - (performance.now() - renderedAt);
  if (wait > 0) {
    renderTimer = setTimeout(() => {
      renderTimer = 0;
      renderFrame = requestAnimationFrame(draw);
    }, wait);
  } else {
    renderFrame = requestAnimationFrame(draw);
  }
}

// Drops the drawing scheduleRender planned, once the answer is drawn whole.
function cancelRender() {
  clearTimeout(renderTimer);
  cancelAnimationFrame(renderFrame);
  renderTimer = 0;
  renderFrame = 0;
}

// The element of the message m of the discussion c, while c is the discussion shown.
function answerElement(c, m) {
  if (state.view !== 'chat' || state.current !== c || !ui.thread || !ui.thread.isConnected) return null;
  const index = c.messages.indexOf(m);
  const el = index >= 0 ? ui.thread.children[index] : null;
  return el && el.dataset.index === String(index) ? el : null;
}

/*
    Answers the last question of c; abort, from beginTurn, stops the answer.  The answer is
    drawn while c is the discussion shown, and saved with it at its end.
*/
async function runTurn(c, abort, turn) {
  // The function, the reasoning and the agent of the answer stay those of its start.
  const role = c.role;
  const model = roleModel(role);
  turn.model = model || '';
  const signal = abort.signal;
  const caps = roleCapabilities(role);
  const reasoning = !!(c.reasoning && caps.reasoning);
  const useTools = !!(c.agent && caps.tools);
  const toolDefs = useTools ? tools.definitions : null;
  try {
    if (state.current === c && !answerElement(c, turn)) renderMain();
    else scheduleRender(c, turn);
    if (signal.aborted) throw new DOMException('The answer was stopped.', 'AbortError');
    if (!model) throw new Error(t('noModel'));
    // The images of the context as data URLs for a model that reads them; a model that reads text only receives
    // their names, which the discussion tells once.
    const media = await requestMedia(c, role);
    if (!media.read && contextImages(c) && !c.imagesNamed) {
      c.imagesNamed = true;
      turn.imagesNamed = true;
      toast(t('imagesNamed', { model: modelName(model) }), false, 9000);
    }
    for (let step = 0; step < maxAgentSteps; ++step) {
      const s = { reasoning: '', content: '', toolCalls: [] };
      turn.steps.push(s);
      const system = systemPrompt(c, caps.context || 8192, useTools);
      // The conversation, with the steps of this answer already done (the agent's).
      const build = (cuts) => apiMessages(Object.assign({}, c, {
        messages: c.messages.slice(0, -1).concat([Object.assign({}, turn, { steps: turn.steps.slice(0, -1) })])
      }), system, cuts, media);
      const fitted = await fitContext(c, build, toolDefs, state.settings.maxTokens || 2048, signal, () => {
        turn.compacting = true;
        scheduleRender(c, turn);
      }, 'albireo-' + role);
      turn.compacting = false;
      if (fitted.trimmed) turn.trimmed = true;
      if (fitted.compacted) turn.compacted = true;
      if (fitted.context) {
        turn.contextTokens = fitted.context;
        turn.promptTokens = fitted.tokens;
      }
      const body = { model: 'albireo-' + role, messages: fitted.messages, reasoning };
      if (toolDefs) body.tools = toolDefs;
      if (state.settings.temperature != null) body.temperature = state.settings.temperature;
      if (state.settings.topP != null) body.top_p = state.settings.topP;
      if (state.settings.topK != null) body.top_k = state.settings.topK;
      if (state.settings.maxTokens > 0) body.max_tokens = state.settings.maxTokens;
      let finish = null;
      for await (const ev of api.streamChat(body, signal)) {
        if (ev.type === 'progress') {
          turn.progress = ev.data;
          scheduleRender(c, turn);
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
        scheduleRender(c, turn);
      }
      if (s.reasoningStart && !s.reasoningMs) s.reasoningMs = Date.now() - s.reasoningStart;
      delete s.reasoningStart;
      if (!(useTools && s.toolCalls.length && (finish === 'tool_calls' || finish === 'stop'))) break;
      // The tools of the step run, each timed, then the model continues with their results.
      for (const tc of s.toolCalls) {
        if (signal.aborted) throw new DOMException('The answer was stopped.', 'AbortError');
        tc.status = 'running';
        tc.started = Date.now();
        scheduleRender(c, turn);
        const r = await tools.run(tc.name, tc.arguments, {
          documents: documentsOf(c),
          searchDocuments: (query, limit) => searchDocumentsOf(c, query, limit),
          searchDiscussions: async (q) => searchConversations(q, 6).filter((x) => x.conversation.id !== c.id).map((x) => ({
            title: x.conversation.title, date: new Date(x.conversation.updated).toLocaleDateString(), excerpt: x.excerpt
          })),
          createFile: (name, content) => ({ name, content }),
          signal
        });
        tc.ms = Date.now() - tc.started;
        delete tc.started;
        // A call the stop of the answer ended keeps what it returned, as stopped.
        tc.status = signal.aborted ? 'stopped' : 'done';
        tc.ok = r.ok;
        tc.result = r.text;
        if (r.file) tc.file = r.file;
        delete tc.index;
        scheduleRender(c, turn);
      }
      if (step === maxAgentSteps - 1) turn.agentLimit = true;
    }
  } catch (e) {
    if (e.name === 'AbortError') turn.cancelled = true;
    else turn.error = errorText(e);
  } finally {
    cancelRender();
    liveTurn = null;
    // The calls an answer stopped or failed before running stay as stopped, without a result.
    for (const s of turn.steps) {
      delete s.reasoningStart;
      for (const tc of s.toolCalls || []) {
        if (tc.status === 'done') continue;
        tc.status = 'stopped';
        delete tc.started;
        delete tc.index;
      }
    }
    turn.progress = null;
    turn.compacting = false;
    turn.preparing = false;
    state.busy = false;
    state.busyId = null;
    state.abort = null;
    announceBusy(c, false);
    c.updated = Date.now();
    await persist(c);
    if (state.current === c) renderMain();
    else lockThread();
    refreshViews();
  }
  if (deleted.has(c.id)) return;
  // A discussion still named after its first question receives a title from the model, whatever
  // the end of its answer, since the title follows the question.
  if (needsTitle(c)) nameConversation(c);
  if (!turn.error && !turn.cancelled) {
    compactLater(c, turn);
    if (!c.incognito) rememberLater();
  }
}

/*
    The memory reads the questions of the discussions that it has not read yet, each discussion
    from its mark, in one pass once they are enough (memoryBatchQuestions questions or
    memoryBatchChars characters) and the user has been idle for memoryIdle: the model writes
    the new facts only, which join the memory without a rewriting, and the memory is rewritten
    as a whole only when it outgrows its size.  One tab updates it at a time, by a lease that
    expires: a tab closed in the middle of an update holds nothing, and the questions it was
    reading are read again by the next update, since the marks move only once the memory is
    saved.  The memory is written in the language of the interface, whatever the languages of
    the discussions, and an update first rewrites in it the facts written in another language,
    after a change of the language among others.
*/
const memoryBatchQuestions = 3;
const memoryBatchChars = 1200;
const memoryIdle = 20000;
const memoryLeaseMs = 180000;
const tabId = store.uid();
let memoryTimer = null;
let memoryRunning = false;
let memoryAbort = null;   // the update in progress, which a new question stops

function rememberLater() {
  if (!state.settings.memoryEnabled) return;
  clearTimeout(memoryTimer);
  memoryTimer = setTimeout(() => {
    memoryTimer = null;
    updateMemory().catch(() => {});
  }, memoryIdle);
}

function sensitiveRule(sensitive) {
  return sensitive ? 'Information the user shares about themselves may be kept, sensitive or not. '
    : 'Leave out sensitive information: health, finances, passwords, keys and identifiers, political, religious '
      + 'or philosophical opinions, sexual orientation, private matters, and information about other people. ';
}

/*
    The language the memory is written in, named in English for the model.  It is the one of the
    interface, whatever the languages of the discussions the memory learns from.
*/
function memoryLanguage() {
  return languageNames[language()] || 'English';
}

/*
    The two facts of the example of the delta in each language of the interface, for a message in
    French (in English for an interface in French), so that the example shows a fact in the language
    of the interface beside the words of the user in their own language.
*/
const memoryExamples = {
  en: ['Works as a nurse in Lyon', 'Prefers short answers'],
  fr: ['Travaille comme infirmière à Lyon', 'Préfère les réponses courtes'],
  de: ['Arbeitet als Krankenpflegerin in Lyon', 'Bevorzugt kurze Antworten'],
  it: ['Lavora come infermiera a Lione', 'Preferisce risposte brevi'],
  es: ['Trabaja como enfermera en Lyon', 'Prefiere respuestas cortas']
};

/*
    The new facts the questions teach, which join the memory, each followed by the words of the user
    that say it, which the update checks in the questions it read (groundedFacts).
*/
function memoryDeltaInstructions(words, sensitive) {
  const code = memoryExamples[language()] ? language() : 'en';
  const [a, b] = memoryExamples[code];
  const english = code === 'fr';
  const message = english ? 'I am a nurse in Lyon and I prefer short answers. What does triage mean?'
    : 'Je suis infirmière à Lyon et je préfère les réponses courtes. Que veut dire triage ?';
  const [qa, qb] = english ? ['I am a nurse in Lyon', 'I prefer short answers']
    : ['Je suis infirmière à Lyon', 'je préfère les réponses courtes'];
  return 'You keep a short memory about the user for the future discussions of an assistant. From the new messages '
    + 'of the user, write the lasting facts about the user that the memory does not hold yet: profession, projects, '
    + 'preferences, habits of work, the way they want to be answered. A topic the user asks about once teaches '
    + 'nothing lasting: keep only what the user says about themselves. Leave out what serves one discussion only, '
    + 'the content of documents, and what the memory already holds, even in other words. ' + sensitiveRule(sensitive)
    + 'Write each new fact on a line of its own: "- ", the fact in a few words in ' + memoryLanguage() + ', " | ", '
    + 'then the statement of the user that says it, copied exactly as the user wrote it, in its language, between '
    + 'double quotes; a question of the user states no fact. A fact the user did not state in so many words is left '
    + 'out. The facts take at most ' + words + ' words in all. For example, for the message "' + message + '":\n'
    + '- ' + a + ' | "' + qa + '"\n'
    + '- ' + b + ' | "' + qb + '"\n'
    + 'When there is no new fact, answer NONE.';
}

/*
    The words of a text as the memory compares a quote of the user with the questions: lower case,
    one kind of apostrophe, without double quotes, the spaces reduced to one.
*/
function evidenceText(text) {
  return String(text || '').toLowerCase().replace(/[\u2018\u2019\u201a\u201b\u2032`\u00b4]/g, '\'')
    .replace(/["\u201c\u201d\u201e\u201f\u00ab\u00bb\u2033]/g, '').replace(/\s+/g, ' ').trim();
}

/*
    Whether the words of a quote, normalized, appear in the questions read, normalized too, with 3
    words or 12 characters at least within a part of a sentence that asks nothing.  The parts end at
    the marks that end a sentence, at colons and at semicolons, and a part that ends with a question
    mark supports no fact, as in "One line: what is RAII?".
*/
function statedIn(quote, where) {
  const at = quote ? where.indexOf(quote) : -1;
  if (at < 0) return false;
  const end = at + quote.length;
  for (const s of where.matchAll(/[^.!?:;\u2026]+[.!?:;\u2026]*/g)) {
    const from = Math.max(at, s.index);
    const to = Math.min(end, s.index + s[0].length);
    if (from >= to || /\?\s*$/.test(s[0])) continue;
    const part = where.slice(from, to).replace(/[.!?;:,\u2026]+$/, '').trim();
    if (part.split(' ').filter(Boolean).length >= 3 || part.length >= 12) return true;
  }
  return false;
}

/*
    The new facts of a delta that the questions read by the update support.  Each line holds a fact,
    " | " and the words of the user that say it, which must lie in a statement of the questions read
    (statedIn).  The words of the user support one fact, and a later fact that cites them again is
    left out.  The facts keep no quote.  Returns {facts, proposed, discarded}.
*/
function groundedFacts(text, read, sensitive) {
  const where = evidenceText(read);
  const cited = new Set();
  const facts = [];
  let proposed = 0;
  let discarded = 0;
  for (const raw of String(text || '').split('\n')) {
    const line = raw.trim();
    if (!/^[-*\u2022]\s*\S/.test(line)) continue;
    ++proposed;
    const m = /^[-*\u2022]\s*(.+?)\s*\|\s*(.+)$/.exec(line);
    const quote = m ? evidenceText(m[2]).replace(/[.!?;:,\u2026]+$/, '').trim() : '';
    if (!m || !m[1].trim() || cited.has(quote) || !statedIn(quote, where)) {
      ++discarded;
      continue;
    }
    cited.add(quote);
    facts.push(...memoryLineList('- ' + m[1].trim(), sensitive));
  }
  return { facts, proposed, discarded };
}

// The memory rewritten within its size, when the new facts made it outgrow it.
function memoryCompactInstructions(words, sensitive) {
  return 'You keep a short memory about the user for the future discussions of an assistant. Rewrite it in at most '
    + words + ' words: merge the lines that say the same thing, keep the most lasting facts, the newest last, each '
    + 'as a short line that starts with "- ". ' + sensitiveRule(sensitive) + 'Write every line in ' + memoryLanguage()
    + ', the lines written in another language included.';
}

// The memory written again in the language of the interface, each of its facts kept.
function memoryTranslateInstructions() {
  return 'You keep a short memory about the user for the future discussions of an assistant. Rewrite the memory in '
    + memoryLanguage() + ': translate the lines written in another language, keep the lines already written in '
    + memoryLanguage() + ' as they are, keep every fact and the order of the lines, one short line per fact that '
    + 'starts with "- ". Answer with the lines alone.';
}

// True when a line of a memory is written in another language than the one of the interface.
function memoryForeign(text) {
  const target = language();
  return memoryLineList(text, true).some((l) => {
    const found = textLanguage(l.slice(2), true);
    return !!found && found !== target;
  });
}

// The credentials, keys and payment identifiers, which the memory keeps only when the settings
// keep sensitive information, whatever the model wrote.
const secretPatterns = [
  /\b(mots? de passe|passwords?|passw[o\u00f6]rt|kennwort|contrase[n\u00f1]as?|parol[ae] d'ordine|passphrase|passcode)\b/i,
  /\b(pin|cvv|cvc|iban|bic|swift)\b/i,
  /\b(api[ _-]?keys?|cl[e\u00e9]s? d'api|access[ _-]?tokens?|bearer|secret[ _-]?keys?|private[ _-]?keys?|cl[e\u00e9]s? priv[e\u00e9]es?)\b/i,
  /\b(?:\d[ -]?){13,19}\b/,
  /[A-Za-z0-9+\/=_-]{32,}/
];

// The lines of a text written by the model, "- " lines only, without the secrets unless the
// settings keep sensitive information.
function memoryLineList(text, sensitive) {
  return String(text || '').split('\n').map((l) => l.trim()).filter((l) => /^[-*\u2022]\s*\S/.test(l))
    .map((l) => '- ' + l.replace(/^[-*\u2022]\s*/, ''))
    .filter((l) => sensitive || !secretPatterns.some((re) => re.test(l)));
}

const wordCount = (lines) => lines.join(' ').split(/\s+/).filter(Boolean).length;

// The lines of a memory written by the model, within words words: the oldest leave first.
function memoryLines(text, words, sensitive) {
  let lines = memoryLineList(text, sensitive);
  while (lines.length > 1 && wordCount(lines) > words) lines = lines.slice(1);
  return lines.join('\n');
}

// The words of a line, for the comparison of two facts: their first five letters, which
// bring the forms of a word together ("courtes", "court").
function factWords(line) {
  return new Set(line.toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '').replace(/[^\p{L}\p{N}]+/gu, ' ')
    .split(' ').filter((w) => w.length > 2).map((w) => w.slice(0, 5)));
}

// True when two lines say nearly the same thing: most of their words are shared.
function sameFact(a, b) {
  const x = factWords(a);
  const y = factWords(b);
  if (!x.size || !y.size) return false;
  let shared = 0;
  for (const w of x) if (y.has(w)) ++shared;
  return shared / (x.size + y.size - shared) >= 0.7;
}

// The questions the memory has not read yet, each discussion from its mark.
function pendingMemory() {
  const items = [];
  let questions = 0;
  let chars = 0;
  const all = state.conversations.slice();
  if (state.current && !state.current.incognito && !all.includes(state.current)) all.push(state.current);
  for (const c of all) {
    if (c.incognito || !Array.isArray(c.messages) || c.messages.length <= (c.memoryMark || 0)) continue;
    const asked = c.messages.slice(c.memoryMark || 0).filter((m) => m.role === 'user' && (m.content || '').trim())
      .map((m) => m.content);
    items.push({ id: c.id, to: c.messages.length, asked });
    questions += asked.length;
    chars += asked.reduce((n, a) => n + a.length, 0);
  }
  return { items, questions, chars };
}

// The lease of this tab on the memory: false when another tab holds it and it has not expired.
function takeMemoryLease() {
  return store.claimLease('memoryLease', tabId, Date.now() + memoryLeaseMs);
}

function releaseMemoryLease() {
  return store.releaseLease('memoryLease', tabId);
}

// The copies of a discussion in this tab: the one shown and the one of the lists.
function copiesOf(id) {
  const out = state.conversations.filter((x) => x.id === id);
  if (state.current && state.current.id === id && !out.includes(state.current)) out.push(state.current);
  return out;
}

/*
    The discussions read up to the questions the memory took.  The mark changes on the stored
    discussion, read again from the database, so that the messages another tab saved stay,
    and on the copies of this tab, so that their next save keeps it.
*/
async function markRead(items) {
  for (const item of items) {
    if (deleted.has(item.id)) continue;
    const mark = (x) => {
      const to = Math.min(item.to, Array.isArray(x.messages) ? x.messages.length : 0);
      if ((x.memoryMark || 0) >= to) return false;
      x.memoryMark = to;
      return true;
    };
    await store.updateConversations(mark, item.id).catch(() => {});
    copiesOf(item.id).forEach(mark);
    broadcast({ type: 'conversation', id: item.id });
  }
}

// The memory and its settings as stored, which another tab may have changed.
async function storedMemory() {
  const s = Object.assign({}, defaults, await store.getSettings());
  return {
    enabled: !!s.memoryEnabled, sensitive: !!s.memorySensitive, memory: s.memory || '', language: s.memoryLanguage
  };
}

// The copies of the discussions in this tab, after a reset of the memory in this tab or in another one.
function forgetMemory() {
  const all = state.conversations.slice();
  if (state.current && !all.includes(state.current)) all.push(state.current);
  for (const c of all) {
    delete c.memory;
    c.memoryMark = c.messages.length;
  }
  clearTimeout(memoryTimer);
  memoryTimer = null;
}

/*
    Updates the memory with the questions it has not read yet, when they are enough or force is
    true: the background model writes the new facts, which join the memory unless it holds them
    already, then the memory is rewritten within its size when it outgrew it, or loses its
    oldest lines when the rewriting fails.  A notification says how many facts joined.
*/
async function updateMemory(force) {
  if (memoryRunning || !state.settings.memoryEnabled || state.incognito) return;
  const pending = pendingMemory();
  const enough = pending.items.length > 0
    && (force || pending.questions >= memoryBatchQuestions || pending.chars >= memoryBatchChars);
  // A memory written in another language than the one of the interface is rewritten in it, with
  // new questions to read or without them.
  const memory = state.settings.memory || '';
  const foreign = !!memory && (memoryForeign(memory)
    || (!!state.settings.memoryLanguage && state.settings.memoryLanguage !== language()));
  if (!enough && !foreign) return;
  // The answer in progress comes first.
  if (state.busy) {
    rememberLater();
    return;
  }
  memoryRunning = true;
  memoryAbort = new AbortController();
  const signal = memoryAbort.signal;
  let leased = false;
  try {
    leased = await takeMemoryLease();
    if (!leased) return;
    // The memory as stored: another tab may have reset or edited it.
    const stored = await storedMemory();
    if (!stored.enabled) return;
    const sensitive = stored.sensitive;
    const context = backgroundContext();
    const words = memoryLimitWords(context);
    const current = stored.memory;
    const target = language();
    // True when the memory was reset, edited or turned off meanwhile, in this tab or in another
    // one, as stored now: the update is then dropped.
    const changed = async () => {
      const now = await storedMemory();
      return !now.enabled || now.memory !== current;
    };
    let lines = memoryLineList(current, true);
    let rewritten = false;
    /*
        The facts written in another language are rewritten in the language of the interface
        first, so that the new facts are compared with them in one language.  A rewriting that
        lost facts is left out, and the next update tries again.
    */
    if (lines.length && (memoryForeign(current) || (!!stored.language && stored.language !== target))) {
      const r = await backgroundText(memoryTranslateInstructions(),
        '<memory>\n' + lines.join('\n') + '\n</memory>\n\nRewrite every line in ' + memoryLanguage() + '.',
        Math.ceil(wordCount(lines) * 2.5) + 32, signal).catch((e) => {
        if (e.name === 'AbortError') throw e;
        return null;
      });
      if (await changed()) return;
      const translated = r ? memoryLineList(r.text, true) : [];
      if (translated.length >= Math.ceil(lines.length * 0.7)) {
        lines = translated;
        rewritten = true;
      }
    }
    let added = 0;
    if (enough && pending.chars >= 40) {
      // What the user wrote: the memory is about the user, and the answers would fill it with
      // what the assistant said.
      const asked = [];
      for (const item of pending.items) for (const a of item.asked) asked.push('User: ' + keepStart(a, 1500));
      // The questions as the model reads them, in which the words of each new fact are looked for.
      const read = keepEnd(asked.join('\n\n'), Math.floor(context * 3.6 * 0.5));
      const r = await backgroundText(memoryDeltaInstructions(words, sensitive),
        '<memory>\n' + (lines.join('\n') || '(empty)') + '\n</memory>\n\n<messages>\n' + read
          + '\n</messages>\n\nWrite the new facts in ' + memoryLanguage() + ', each with the words of the user.',
        Math.ceil(words * 3.2) + 32, signal);
      if (await changed()) return;
      const delta = groundedFacts(r.text, read, sensitive);
      console.info('Albireo: memory: ' + delta.proposed + ' facts proposed, ' + delta.discarded
        + ' left out, unsupported by the words of the user');
      for (const fact of delta.facts) {
        if (lines.some((l) => sameFact(l, fact))) continue;
        lines.push(fact);
        ++added;
      }
      if (added && wordCount(lines) > words) {
        const compact = await backgroundText(memoryCompactInstructions(words, sensitive),
          '<memory>\n' + lines.join('\n') + '\n</memory>', Math.ceil(words * 2.2) + 16, signal).catch(() => null);
        if (signal.aborted) return;
        if (await changed()) return;
        const compacted = compact ? memoryLineList(compact.text, sensitive) : [];
        if (compacted.length && wordCount(compacted) <= words * 1.2) lines = compacted;
        while (lines.length > 1 && wordCount(lines) > words) lines = lines.slice(1);
      }
    }
    if (added || rewritten) {
      await saveSetting('memory', lines.join('\n'));
      await saveSetting('memoryUpdated', Date.now());
      await saveSetting('memoryLanguage', target);
    }
    if (enough) await markRead(pending.items);
    if (added) toast(t(added === 1 ? 'memoryLearnedOne' : 'memoryLearned', { n: added }), false, 4500);
    else if (rewritten) toast(t('memoryRewritten'), false, 4500);
  } finally {
    memoryRunning = false;
    memoryAbort = null;
    if (leased) await releaseMemoryLease().catch(() => {});
  }
}

// An error of the service in the language of the interface, by its code when it has one.
function errorText(e) {
  const known = { model_not_found: 'errModelNotFound', context_length_exceeded: 'errContextLength', unloaded: 'errUnloaded',
    starting: 'errStarting', model_loading: 'modelLoading' };
  if (e && known[e.code]) return t(known[e.code]);
  // fetch rejects with a TypeError when the service does not answer.
  if (e instanceof TypeError) return t('errService');
  return (e && e.message) || String(e);
}

// The title of a discussion until the model names it: the beginning of its first question.
function provisionalTitle(m) {
  const first = m.content || ((m.attachments || [])[0] || (m.images || [])[0] || {}).name || t('untitled');
  return first.replace(/\s+/g, ' ').slice(0, 70);
}

// A title as the model writes it, without its quotes, marks and final punctuation.
function cleanTitle(text) {
  let title = String(text || '').split('\n').map((l) => l.trim()).find((l) => l) || '';
  title = title.replace(/^(title|titre|titel|titolo|t[ií]tulo)\s*:\s*/i, '').replace(/[*#_`]+/g, '')
    .replace(/^["'\u00ab\u201c\u201e\s]+|["'\u00bb\u201d\s]+$/g, '').replace(/[.!?:;,\u2026]+$/, '').trim();
  // An identifier written as a title (ReverseWordsInSentence) becomes words.
  if (!/\s/.test(title) && /[a-z][A-Z]/.test(title)) title = title.replace(/([a-z0-9])([A-Z])/g, '$1 $2');
  if (title) title = title.charAt(0).toLocaleUpperCase() + title.slice(1);
  return title.length > 70 ? keepStart(title, 70).replace(/\s*\u2026$/, '') : title;
}

// True when a discussion still bears the beginning of its first question as its title.
function needsTitle(c) {
  if (!c || deleted.has(c.id) || c.titleSource === 'user' || c.titleSource === 'model') return false;
  const first = (c.messages || []).find((m) => m.role === 'user');
  if (!first) return false;
  if (c.titleSource === 'question' || c.titleSource === 'pending') return true;
  return !c.title || c.title === provisionalTitle(first);
}

/*
    Examples of titles, one per language of the interface, which show a small model that a title
    names the topic of the message in a few words and in its language, without its answer or its
    first words.
*/
const titleExamples = [
  ['English', 'I\'m a nurse in Leeds. In one line, what is triage?', 'Meaning of triage'],
  ['French', 'Peux-tu m\'expliquer comment fonctionne la photosynthèse ?', 'Fonctionnement de la photosynthèse'],
  ['German', 'Berechne 15 % von 80 und erkläre den Rechenweg.', 'Prozentrechnung mit Rechenweg'],
  ['Spanish', 'Escribe un poema corto sobre el mar.', 'Poema sobre el mar'],
  ['Italian', 'Come faccio a invertire una lista collegata in Java?', 'Inversione di una lista in Java']
];

/*
    The request of a title, with the examples as earlier exchanges, then the question with the name
    of its language when it can be told.  again asks for other words than those of the question.
*/
function titleMessages(question, again) {
  const code = textLanguage(question);
  const name = code ? languageNames[code] : 'the language of the message';
  const out = [{
    role: 'system',
    content: 'You give a short title to a discussion from its first message. The title names the topic in two to six '
      + 'words, as a noun phrase, in the language given. It never answers the message and never copies it. '
      + 'Answer with the title alone.'
  }];
  for (const [language, message, title] of titleExamples) {
    out.push({ role: 'user', content: 'Language: ' + language + '\nMessage: ' + message },
      { role: 'assistant', content: title });
  }
  out.push({
    role: 'user',
    content: 'Language: ' + name + '\nMessage: ' + question
      + (again ? '\nThe title uses other words than the message: it names the kind of request.' : '')
  });
  return out;
}

// The words of a text as the titles are compared, folded and cut to their first five letters.
function titleWords(text) {
  return fold(text).split(/[^\p{L}\p{N}]+/u).filter(Boolean).map((w) => w.slice(0, 5));
}

/*
    True when a title is empty, holds more than ten words, or repeats the first words of a question
    of three words or more, in other forms ("Calculer 17,5 %" for "Calcule 17,5 %").
*/
function poorTitle(title, question) {
  const words = titleWords(title);
  if (!words.length || words.length > 10) return true;
  const asked = titleWords(question);
  return asked.length >= 3 && words.join(' ') === asked.slice(0, words.length).join(' ');
}

/*
    The title the background model writes for a question, asked a second time with other words
    when the first one only repeats the question.
*/
async function askTitle(question) {
  const first = cleanTitle((await backgroundChat(titleMessages(question, false), 24)).text);
  if (!poorTitle(first, question)) return first;
  const second = cleanTitle((await backgroundChat(titleMessages(question, true), 24, null, 0.7)).text);
  return poorTitle(second, question) && first ? first : second;
}

/*
    The errors after which the request of a title is sent again, while the service starts, while
    its model loads or is missing, or while the service does not answer.
*/
function retryable(e) {
  if (e instanceof TypeError) return true;
  return ['starting', 'model_loading', 'unloaded', 'model_not_found'].includes(e && e.code)
    || /^5\d\d\b/.test(String(e && e.message));
}

// The pauses between the attempts of a title, in seconds, about two minutes in all.
const titleRetries = [2, 4, 8, 15, 30, 30, 30];

// True when the service answers and has a model, which a title needs.
function serviceReady() {
  return !!state.status && !state.statusError && state.status.state !== 'starting' && !!roleModel('general');
}

/*
    Runs fn under a lock the tabs share, so that one tab only does the work; does nothing when
    another tab holds it.  Where the browser has no such locks, fn runs.
*/
function exclusive(name, fn) {
  if (!navigator.locks || typeof navigator.locks.request !== 'function') return fn();
  return navigator.locks.request(name, { ifAvailable: true }, (lock) => (lock ? fn() : null));
}

/*
    The discussions this tab is naming.  The mark stays out of the discussions, so that none is
    saved with it, and a discussion whose tab closed while it was named is named at a later visit.
*/
const naming = new Set();

/*
    Names a discussion after its first question.  The background model writes a short title, in
    the language of the question, which replaces the beginning of the question that stood for it
    until then, unless the user renames the discussion meanwhile.  The request waits for the
    service and its model, and is sent again after an error that passes, a bounded number of
    times; it does not depend on the discussion shown, nor on the questions that follow.  The
    title shows at once in the side panel, the top bar and the other tabs.
*/
async function nameConversation(c) {
  if (naming.has(c.id)) return;
  naming.add(c.id);
  try {
    await exclusive('albireo-title-' + c.id, async () => {
      for (let attempt = 0; attempt <= titleRetries.length; ++attempt) {
        if (attempt) await new Promise((r) => setTimeout(r, titleRetries[attempt - 1] * 1000));
        const now = await currentVersion(c).catch(() => null);
        if (!needsTitle(now)) return;
        if (!serviceReady()) continue;
        const first = now.messages.find((m) => m.role === 'user');
        const names = (first.attachments || []).concat(first.images || []).map((a) => a.name);
        const question = keepStart((first.content || '') + (names.length ? '\n(' + names.join(', ') + ')' : ''), 1500);
        let title = '';
        try {
          title = await askTitle(question);
        } catch (e) {
          if (retryable(e)) continue;
          return;
        }
        // Without a title, the beginning of the question stays, and a later answer tries again.
        const latest = await currentVersion(c).catch(() => null);
        if (!title || !needsTitle(latest)) return;
        latest.title = title;
        latest.titleSource = 'model';
        await persist(latest);
        refreshViews();
        return;
      }
    });
  } finally {
    naming.delete(c.id);
  }
}

/*
    Names the stored discussions that a closed tab, a reload or a service out of reach left with
    the beginning of their question as title, the most recent first, one after the other.
*/
let namingPending = false;

async function namePending() {
  if (namingPending || state.incognito) return;
  namingPending = true;
  try {
    const waiting = state.conversations.filter((c) => needsTitle(c) && c.messages.some((m) => m.role === 'assistant'));
    for (const c of waiting.slice(0, 5)) {
      if (!serviceReady()) break;
      await nameConversation(c);
    }
  } finally {
    namingPending = false;
  }
}

/*
    Compresses the oldest exchanges of a discussion in the background when its last answer
    filled compactShare of the room of the context: the next question then starts from the
    summary, without waiting for it, unless it comes before the summary is written.
*/
function compactLater(c, turn) {
  const context = turn.contextTokens;
  if (!context || compactions.has(c.id)) return;
  const room = context - Math.min(state.settings.maxTokens || 2048, Math.floor(context / 2));
  const used = (turn.promptTokens || 0) + ((turn.usage && turn.usage.completion_tokens) || 0);
  if (used < compactShare * room) return;
  const system = systemPrompt(c, context, !!(c.agent && roleCapabilities(c.role).tools));
  const media = estimateMedia(c.role);
  const build = () => apiMessages(c, system, null, media);
  const estimate = tokenEstimate(build(), used, 'albireo-' + c.role);
  // The questions after the start, the last one included: the next question comes after it.
  const saved = c.messages;
  c.messages = saved.concat([{ id: 'next', role: 'user', content: '' }]);
  const next = nextStart(c, build, estimate.of, room);
  c.messages = saved;
  if (next <= (c.contextStart || 0)) return;
  const job = compact(c, c.contextStart || 0, next, room, estimate.scale, null)
    .then(async (r) => {
      // The summary goes to the discussion as it is now, when its compressed messages are
      // still the same.
      const now = await currentVersion(c);
      const last = now && now.messages[r.to - 1];
      if (!now || !last || last.id !== r.last || (now.contextStart || 0) > r.to) return;
      applyCompaction(now, r);
      await persist(now);
      // The images of the exchanges summarized left the context: their places go to new ones.
      if (state.current === now) {
        markOutside();
        renderComposerState();
        lockThread();
      }
    })
    .catch(() => {})
    .finally(() => compactions.delete(c.id));
  compactions.set(c.id, job);
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
    const tabs = ['general', 'model', 'memoryTab', 'data', 'about'];
    const tabBar = h('div', { class: 'tabs' });
    const body = h('div');
    const show = (tab) => {
      for (const b of tabBar.children) b.classList.toggle('on', b.dataset.tab === tab);
      ui.memoryView = null;
      body.replaceChildren();
      if (tab === 'general') generalTab(body, close);
      else if (tab === 'model') modelTab(body);
      else if (tab === 'memoryTab') memoryTab(body);
      else if (tab === 'data') dataTab(body, close);
      else aboutTab(body, close);
    };
    for (const tab of tabs) tabBar.append(h('button', { type: 'button', 'data-tab': tab, onclick: () => show(tab) }, t(tab)));
    box.append(h('h2', { html: icon('gear') + escapeHtml(t('settings')) }), tabBar, body,
      h('div', { class: 'actions' }, h('button', { class: 'button primary', type: 'button', onclick: () => close(true) }, t('close'))));
    show(initialTab || 'general');
  }).then(() => { ui.memoryView = null; });
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
    close(true);
    await languageChanged(true);
    settingsDialog('general');
  });
  const name = h('input', { value: state.settings.userName || '' });
  name.addEventListener('change', async () => {
    await saveSetting('userName', name.value.trim());
    refreshViews();
  });
  const role = h('select', null, roles.map((r) => h('option', { value: r, selected: state.settings.defaultRole === r }, t('role_' + r))));
  // A new discussion, without messages yet, takes the default function at once.
  role.addEventListener('change', async () => {
    await saveSetting('defaultRole', role.value);
    if (state.current && !state.current.messages.length) state.current.role = role.value;
    refreshViews();
  });
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
  /*
      top-k: a slider and its value, which may also be typed, from 0 (no limit) to 1024, the bounds of the
      service; empty, the service takes its own value.
  */
  const topK = () => {
    const value = state.settings.topK;
    const range = h('input', { type: 'range', min: 0, max: 1024, step: 1, value: value == null ? 40 : value,
      'aria-label': t('topK') });
    const number = h('input', { type: 'number', class: 'top-k', min: 0, max: 1024, step: 1, 'aria-label': t('topK'),
      value: value == null ? '' : String(value), placeholder: t('maxTokensAuto') });
    range.addEventListener('input', () => { number.value = range.value; });
    range.addEventListener('change', () => saveSetting('topK', parseInt(range.value, 10)));
    number.addEventListener('change', () => {
      const typed = parseInt(number.value, 10);
      const v = number.value.trim() === '' || Number.isNaN(typed) ? null : Math.max(0, Math.min(1024, typed));
      number.value = v == null ? '' : String(v);
      if (v != null) range.value = v;
      saveSetting('topK', v);
    });
    const reset = h('button', {
      class: 'button', type: 'button', onclick: async () => {
        await saveSetting('topK', null);
        number.value = '';
        range.value = 40;
      }
    }, t('maxTokensAuto'));
    return h('div', { class: 'inline' }, range, number, reset);
  };
  const max = h('input', { type: 'number', min: 0, max: 131072, step: 256, value: state.settings.maxTokens || 0 });
  max.addEventListener('change', () => saveSetting('maxTokens', Math.max(0, parseInt(max.value, 10) || 0)));
  body.append(field(t('temperature'), slider('temperature', 0, 2, 0.05, 0.7), t('temperatureHelp')),
    field(t('topP'), slider('topP', 0.05, 1, 0.05, 0.95), t('topPHelp')), field(t('topK'), topK(), t('topKHelp')),
    field(t('maxTokens'), max, '0 = ' + t('maxTokensAuto')));
}

// The memory of the discussions: its use, its content, which the user may edit, and its reset.
function memoryTab(body) {
  const context = roleCapabilities('general').context || 8192;
  const words = memoryLimitWords(context);
  const enabled = h('input', { type: 'checkbox', checked: !!state.settings.memoryEnabled });
  const sensitive = h('input', { type: 'checkbox', checked: !!state.settings.memorySensitive });
  const text = h('textarea', { rows: 8, placeholder: t('memoryEmpty') });
  text.value = state.settings.memory || '';
  const size = h('p', { class: 'help', style: { margin: 0, color: 'var(--text-faint)', fontSize: '12.5px' } });
  const showSize = () => {
    const n = text.value.split(/\s+/).filter(Boolean).length;
    size.textContent = t('memorySize', { n, max: words })
      + (state.settings.memoryUpdated ? ' · ' + t('memoryUpdated', { date: new Date(state.settings.memoryUpdated).toLocaleString(language()) }) : '');
  };
  showSize();
  enabled.addEventListener('change', async () => {
    await saveSetting('memoryEnabled', enabled.checked);
    sensitive.disabled = text.disabled = !enabled.checked;
  });
  sensitive.addEventListener('change', () => saveSetting('memorySensitive', sensitive.checked));
  text.addEventListener('input', showSize);
  text.addEventListener('change', async () => {
    // What the user writes stays, sensitive or not, within twice the size of the memory.
    await saveSetting('memory', memoryLines(text.value.split('\n').map((l) => l.trim() && !/^[-*\u2022]/.test(l.trim())
      ? '- ' + l.trim() : l).join('\n'), words * 2, true));
    await saveSetting('memoryUpdated', Date.now());
    text.value = state.settings.memory;
    showSize();
  });
  sensitive.disabled = text.disabled = !enabled.checked;
  // The memory as an update or another tab changes it, unless the user is writing in it.
  ui.memoryView = () => {
    if (!text.isConnected) return;
    enabled.checked = !!state.settings.memoryEnabled;
    sensitive.checked = !!state.settings.memorySensitive;
    sensitive.disabled = text.disabled = !enabled.checked;
    if (document.activeElement !== text) text.value = state.settings.memory || '';
    showSize();
  };
  body.append(h('p', null, t('memoryIntro')),
    h('div', { class: 'field' }, h('label', { class: 'inline' }, enabled, ' ', t('memoryEnabled'))),
    h('div', { class: 'field' }, h('label', { class: 'inline' }, sensitive, ' ', t('memorySensitive')),
      h('p', { class: 'help', style: { margin: 0, color: 'var(--text-faint)', fontSize: '12.5px' } }, t('memorySensitiveHelp'))),
    field(t('memoryText'), text), size,
    h('div', { class: 'buttons', style: { marginTop: '10px' } }, h('button', {
      class: 'button danger', type: 'button', html: icon('trash') + '<span>' + escapeHtml(t('memoryReset')) + '</span>',
      onclick: async () => {
        if (!await confirmDialog(t('memoryReset'), t('memoryResetText'))) return;
        await saveSetting('memory', '');
        await saveSetting('memoryUpdated', 0);
        // The discussions forget the memory they started with, and what they already said
        // stays out of the next updates: the stored discussions change in place, read again
        // from the database with the messages other tabs saved, and the copies of the tabs follow.
        await store.updateConversations((x) => {
          const mark = Array.isArray(x.messages) ? x.messages.length : 0;
          if (x.memory === undefined && x.memoryMark === mark) return false;
          delete x.memory;
          x.memoryMark = mark;
          return true;
        }).catch(() => {});
        forgetMemory();
        broadcast({ type: 'memoryReset' });
        text.value = '';
        showSize();
      }
    })));
}

function dataTab(body, close) {
  const input = h('input', { type: 'file', accept: '.json,application/json', class: 'hidden' });
  input.addEventListener('change', async () => {
    try {
      const data = JSON.parse(await input.files[0].text());
      const r = await store.importAll(data);
      // A discussion deleted earlier comes back with the import, and is saved again.
      for (const c of data.conversations) if (c && typeof c.id === 'string') deleted.delete(c.id);
      broadcast({ type: 'lists' });
      await syncLists();
      toast(t('imported', { n: r.conversations, p: r.projects }));
      indexProjects();
      describePending();
    } catch (e) {
      toast(t('importFailed'), true);
    }
    input.value = '';
  });
  const storage = h('div', { class: 'storage-state' });
  // The discussions, the projects and the images stored, with the room the images take.
  const stored = h('p', { class: 'storage-usage' });
  const showUsage = (u) => {
    stored.textContent = t('storageUsage', {
      n: state.conversations.length, p: state.projects.length, i: u.count, size: formatSize(u.bytes)
    });
  };
  showUsage({ count: 0, bytes: 0 });
  images.usage().then(showUsage).catch(() => {});
  body.append(stored, storage,
    h('div', { class: 'buttons' },
      h('button', {
        class: 'button', type: 'button', html: icon('download') + '<span>' + escapeHtml(t('exportAll')) + '</span>',
        onclick: async () => {
          try {
            const data = JSON.stringify(await store.exportAll());
            download('albireo-' + new Date().toISOString().slice(0, 10) + '.json', data, 'application/json');
          } catch (e) {
            storageFailed(e);
          }
        }
      }),
      h('button', { class: 'button', type: 'button', html: icon('upload') + '<span>' + escapeHtml(t('importAll')) + '</span>', onclick: () => input.click() }),
      input,
      h('button', {
        class: 'button danger', type: 'button', html: icon('trash') + '<span>' + escapeHtml(t('clearAll')) + '</span>',
        onclick: async () => {
          if (!await confirmDialog(t('clearAll'), t('clearAllText'))) return;
          // The answer in progress stops, the stored discussions are not saved again, and the
          // indexing in progress writes no passage.  A new discussion, not yet stored, stays.
          for (const c of state.conversations) deleted.add(c.id);
          if (state.current && state.current.messages.length) deleted.add(state.current.id);
          if (state.busy) stopGeneration();
          cancelIndex();
          rag.removeAll();
          images.removeAll();
          await store.clearAll().catch(storageFailed);
          broadcast({ type: 'cleared' });
          await reloadLists().catch(storageFailed);
          close(true);
          go('#/');
        }
      })));
  showStorage(storage);
}

/*
    Asks the browser to keep the database of the application when the disk runs short, once the
    database is open, when it is not kept yet.  The answer, which the browser gives without asking
    the user, stays in the settings (storagePersisted) and shows in the Data tab of the settings.
*/
async function requestPersistence() {
  const s = navigator.storage;
  if (!s || typeof s.persisted !== 'function' || typeof s.persist !== 'function') return;
  try {
    let kept = await s.persisted();
    if (!kept) kept = await s.persist();
    if (state.settings.storagePersisted !== kept) await saveSetting('storagePersisted', kept);
  } catch (e) {
    // The browser keeps its answer to itself; the Data tab shows what it tells then.
  }
}

/*
    The state of the storage of this browser, kept or not, with the space used and the space
    available, and the advice to export the data regularly when the browser may delete them.
*/
async function showStorage(el) {
  const s = navigator.storage;
  let kept = state.settings.storagePersisted;
  let estimate = null;
  try {
    if (s && typeof s.persisted === 'function') kept = await s.persisted();
  } catch (e) {
    // The last answer stays.
  }
  try {
    if (s && typeof s.estimate === 'function') estimate = await s.estimate();
  } catch (e) {
    estimate = null;
  }
  if (typeof kept === 'boolean' && kept !== state.settings.storagePersisted) saveSetting('storagePersisted', kept);
  const parts = [t(kept === true ? 'storagePersistent' : kept === false ? 'storageNotPersistent' : 'storageUnknown')];
  if (estimate && estimate.quota) {
    parts.push(t('storageEstimate', { used: formatSize(estimate.usage || 0), quota: formatSize(estimate.quota) }));
  }
  el.dataset.persistent = String(kept);
  el.replaceChildren(h('p', null, h('b', null, t('storageState')), ' ', parts.join(' · ')),
    kept === true ? null : h('p', { class: 'help storage-hint' }, h('span', { html: icon('info') }), t('storageHint')));
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

async function aboutTab(body, close) {
  const s = state.status || {};
  const art = h('div', { class: 'about-art' }, h('img', { src: 'img/about.svg', alt: '' }));
  const kv = h('div', { class: 'kv' });
  const row = (k, v) => kv.append(h('div', null, k), h('div', null, v));
  row(t('version'), [s.product || 'Albireo', s.edition, s.version].filter(Boolean).join(' '));
  row(t('engine'), s.engine ? (s.engine.variant || '-') + (s.engine.device ? ' (' + s.engine.device + ')' : '') : '-');
  if (s.host) {
    row(t('processor'), s.host.cpu.name + ' · ' + s.host.cpu.cores + ' / ' + s.host.cpu.threads);
    row(t('memory'), formatSize(s.host.memory.total));
  }
  // The functions of each model, in the order of the picker, the document search last.
  const byModel = new Map();
  for (const r of roles.concat(['embedding'])) {
    const id = (s.active || {})[r];
    if (id) byModel.set(id, (byModel.get(id) || []).concat([t('role_' + r)]));
  }
  const models = Array.from(byModel, ([id, names]) => names.join(', ') + ': ' + modelName(id)).join(' · ');
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
  body.append(h('p', { style: { marginTop: '14px' } }, t('createdBy', {
    author: (s.author || 'Aldric Pierrain') + ' (' + (s.email || 'cydraltechnology@gmail.com') + ')'
  }) + ' · ' + (s.copyright || 'Copyright (C) 2026 Cydral Technology')),
  h('p', null, h('a', { href: s.project || 'https://github.com/Cydral/Dlib-Transformer-extensions', target: '_blank', rel: 'noopener' }, 'Dlib-Transformer-extensions'),
    ' · ', h('a', { href: '/docs/', target: '_blank' }, t('apiDocs')), ' · ',
    h('a', { href: '#/licenses', onclick: () => close(true) }, t('thirdParty'))));
}

/*
    The page of the components and of their licenses: those of the web application, which
    vendor/components.json lists; those of the programs and of the engines, whose texts the
    installer copies into the licenses folder of the programs; and the installed models, with
    the license of their entry in the catalog.  A text opens in the page, under its line.
*/
let licensesSequence = 0;

// The role and the license of each text of the licenses folder, by its file.
const programTexts = {
  'albireo.txt': ['albireo', 'Boost Software License 1.0'], 'dlib.txt': ['dlib', 'Boost Software License 1.0'],
  'mbedtls.txt': ['mbedtls', 'Apache License 2.0 or GPL 2.0 or later'], 'cpp-httplib.txt': ['httplib', 'MIT License'],
  'zlib.txt': ['zlib', 'zlib License'], 'sqlite.txt': ['sqlite', null],
  'libpng.txt': ['libpng', 'PNG Reference Library License 2'], 'libjpeg.txt': ['libjpeg', 'IJG License'],
  'openvino-license.txt': ['openvino', 'Apache License 2.0'], 'cuda-eula.txt': ['cuda', 'NVIDIA CUDA EULA'],
  'cudnn-license.txt': ['cudnn', 'NVIDIA cuDNN License Agreement']
};

async function licenseText(url) {
  const r = await fetch(url, { cache: 'no-store' });
  if (!r.ok) throw new Error('HTTP ' + r.status);
  return r.text();
}

function licenseHead(columns) {
  return h('div', { class: 'lic-row lic-head', 'aria-hidden': 'true' }, columns.map((c) => h('div', null, c)),
    h('div'));
}

/*
    A line of the page of the licenses: its cells, then a button that opens its texts under it
    when load is given, load returning [[file, text]...], or the element action.
*/
function licenseRow(cells, load, action) {
  const box = h('div', { class: 'lic-text hidden' });
  let loaded = false;
  let button = null;
  if (load) {
    button = h('button', {
      class: 'button small', type: 'button', 'aria-expanded': 'false',
      html: icon('chevron') + '<span>' + escapeHtml(t('licenseShow')) + '</span>'
    });
    button.addEventListener('click', async () => {
      const open = box.classList.toggle('hidden') === false;
      button.setAttribute('aria-expanded', String(open));
      button.classList.toggle('open', open);
      if (!open || loaded) return;
      loaded = true;
      box.replaceChildren(h('div', { class: 'empty-note' }, t('licenseLoading')));
      try {
        const parts = await load();
        box.replaceChildren(...parts.flatMap(([file, text]) => [parts.length > 1 ? h('div', { class: 'lic-file' }, file)
          : null, h('pre', null, text)]).filter(Boolean));
      } catch (e) {
        loaded = false;
        box.replaceChildren(h('div', { class: 'error-box' }, t('licenseReadFailed')));
      }
    });
  }
  return [h('div', { class: 'lic-row' }, cells.map(([cls, v]) => h('div', { class: cls }, v)),
    h('div', { class: 'lic-action' }, button || action)), box];
}

async function renderLicenses() {
  const sequence = ++licensesSequence;
  const head = [h('h2', { html: icon('book') + escapeHtml(t('licensesTitle')) }),
    h('p', { class: 'sub' }, t('licensesIntro'))];
  ui.content.replaceChildren(h('div', { class: 'page licenses' }, head,
    h('div', { class: 'empty-note' }, t('licenseLoading'))));
  const [web, programs, models] = await Promise.all([
    api.getJSON('/vendor/components.json').catch(() => null),
    api.getJSON('/albireo/v1/licenses').catch(() => null),
    api.models(language()).catch(() => null)
  ]);
  if (sequence !== licensesSequence || state.view !== 'licenses') return;
  const columns = [t('licenseComponent'), t('licenseVersion'), t('licenseRole'), t('licenseName')];
  const card = (iconName, title, help, rows, empty) => h('div', { class: 'card' },
    h('h3', { html: icon(iconName) + escapeHtml(title) }), help ? h('p', { class: 'help' }, help) : null,
    rows.length ? h('div', { class: 'lic-table' }, rows) : h('div', { class: 'empty-note' }, empty));

  // The web application, from the list its updates keep.
  const webRows = [];
  for (const c of (web && Array.isArray(web.components)) ? web.components : []) {
    const key = 'component_' + c.id;
    webRows.push(licenseRow([['lic-name', c.name], ['lic-version', c.version || '-'],
      ['lic-role', t(key) === key ? c.role || '' : t(key)], ['lic-license', c.license_name || c.license || '-']],
    () => Promise.all((c.license_files || []).map(async (f) => [f, await licenseText('/' + f)]))));
  }
  if (webRows.length) webRows.unshift(licenseHead(columns));

  // The programs and the engines, from the licenses folder of the installation.
  const programRows = [];
  for (const p of Array.isArray(programs) ? programs : []) {
    const [id, license] = programTexts[p.file] || ['runtime', '-'];
    programRows.push(licenseRow([['lic-name', p.name || p.file], ['lic-version', p.version || '-'],
      ['lic-role', t('program_' + id)], ['lic-license', license === null ? t('licensePublicDomain') : license]],
    () => licenseText('/albireo/v1/licenses/' + encodeURIComponent(p.file)).then((text) => [[p.file, text]])));
  }
  if (programRows.length) programRows.unshift(licenseHead(columns));

  // The installed models, the language models then the embedding models, with the functions they serve.
  const modelRows = [];
  const active = (models && models.active) || {};
  const installed = models
    ? (models.models || []).filter((m) => m.installed).concat(models.embedding_models || []) : [];
  for (const m of installed) {
    const functions = roles.concat(['embedding']).filter((r) => active[r] === m.id).map((r) => t('role_' + r));
    const source = /^https:\/\//.test(m.source || '') ? h('a', {
      class: 'button small', href: m.source, target: '_blank', rel: 'noopener',
      html: icon('external') + '<span>' + escapeHtml(t('licenseSource')) + '</span>'
    }) : null;
    modelRows.push(licenseRow([['lic-name', m.name || m.id], ['lic-version', m.publisher || '-'],
      ['lic-role', functions.join(', ') || '-'], ['lic-license', m.license || t('licenseNotStated')]], null, source));
  }
  if (modelRows.length) {
    modelRows.unshift(licenseHead([t('licenseModel'), t('licensePublisher'), t('licenseFunctions'), t('licenseName')]));
  }

  ui.content.replaceChildren(h('div', { class: 'page licenses' }, head,
    card('code', t('licensesWeb'), t('licensesWebHelp'), webRows, t('licenseReadFailed')),
    card('cpu', t('licensesPrograms'), programRows.length ? t('licensesProgramsHelp') : null, programRows,
      programs ? t('licensesProgramsNone') : t('serviceDown')),
    card('spark', t('licensesModels'), t('licensesModelsHelp'), modelRows,
      models ? t('licensesModelsNone') : t('serviceDown'))));
}

// The line under the composer: what the answers are; the version and the copyright are in About.
function hintText() {
  return t('aiNotice');
}

// Layout.
function buildLayout() {
  const app = document.getElementById('app');
  // The question being typed stays when the layout is built again in another language.
  const draft = ui.input ? ui.input.value : '';
  app.replaceChildren();
  ui.sidebar = h('aside', { class: 'sidebar' + (state.settings.sidebar === false ? ' collapsed' : '') });
  ui.sideScroll = h('div', { class: 'side-scroll' });
  ui.statusLine = h('div', { class: 'status-line' });
  const search = h('input', { type: 'search', placeholder: t('search'), value: state.search });
  let searchTimer = 0;
  search.addEventListener('input', () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => {
      state.search = search.value;
      renderSidebar();
    }, searchDelay);
  });
  ui.incognitoButton = h('button', {
    class: 'incognito-toggle' + (state.incognito ? ' on' : ''), type: 'button', title: t('incognitoStart'), html: icon('incognito'),
    onclick: () => {
      state.incognito = !state.incognito;
      if (state.busy) stopGeneration();
      forgetIncognito();
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
  ui.input.value = draft;
  ui.input.addEventListener('input', autoGrow);
  ui.input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing && state.settings.sendOnEnter) {
      e.preventDefault();
      onSend();
    }
  });
  ui.input.addEventListener('paste', (e) => {
    const files = pastedFiles(e);
    if (files.length) { e.preventDefault(); addPending(files); }
  });
  const fileInput = h('input', { type: 'file', multiple: true, class: 'hidden' });
  fileInput.addEventListener('change', () => { addPending(Array.from(fileInput.files)); fileInput.value = ''; });
  ui.fileInput = fileInput;
  ui.reasonButton = h('button', {
    class: 'toggle reason', type: 'button', title: t('reasoningHint'), html: icon('spark') + '<span>' + escapeHtml(t('reasoning')) + '</span>',
    onclick: async () => {
      const c = state.current;
      c.reasoning = !c.reasoning;
      renderComposerState();
      await saveSetting('lastReasoning', c.reasoning);
      if (c.messages.length) await persist(c);
      refreshViews();
    }
  });
  ui.agentButton = h('button', {
    class: 'toggle', type: 'button', title: t('agentHint'), html: icon('tool') + '<span>' + escapeHtml(t('agent')) + '</span>',
    onclick: async () => {
      const c = state.current;
      c.agent = !c.agent;
      renderComposerState();
      await saveSetting('lastAgent', c.agent);
      if (c.messages.length) await persist(c);
      refreshViews();
    }
  });
  // What the agent mode does, from a button beside it, and the tools it may call, under the prompt.
  ui.agentHelp = h('button', {
    class: 'help-button', type: 'button', title: t('agentHelpTitle'), 'aria-label': t('agentHelpTitle'),
    html: icon('info'), onclick: (e) => agentHelp(e.currentTarget)
  });
  ui.agentTools = h('div', { class: 'agent-tools hidden', role: 'list', 'aria-label': t('agentTools') },
    h('span', { class: 'label' }, t('agentTools')),
    tools.definitions.map((d) => h('span', {
      class: 'tool-chip', role: 'listitem', tabindex: '0', title: t('toolHint_' + d.function.name),
      html: icon(toolIcons[d.function.name] || 'tool') + '<span>' + escapeHtml(toolLabel(d.function.name)) + '</span>'
    })));
  ui.sendButton = h('button', { class: 'send', type: 'button', onclick: onSend });
  ui.pendingBox = h('div', { class: 'composer-attachments hidden' });
  ui.attachButton = iconButton('clip', t('attach'), () => fileInput.click());
  const composer = h('div', { class: 'composer' }, ui.pendingBox, ui.input, ui.agentTools, h('div', { class: 'row' },
    ui.attachButton, fileInput, ui.reasonButton, ui.agentButton, ui.agentHelp, h('span', { class: 'grow' }),
    ui.sendButton));
  for (const ev of ['dragenter', 'dragover']) composer.addEventListener(ev, (e) => { e.preventDefault(); composer.classList.add('drag'); });
  for (const ev of ['dragleave', 'drop']) composer.addEventListener(ev, () => composer.classList.remove('drag'));
  composer.addEventListener('drop', (e) => {
    e.preventDefault();
    if (e.dataTransfer && e.dataTransfer.files.length) addPending(Array.from(e.dataTransfer.files));
  });
  ui.hint = h('div', { class: 'hint' }, hintText());
  ui.composerWrap = h('div', { class: 'composer-wrap' }, composer, ui.hint);
  ui.toasts = h('div', { class: 'toast-zone' });
  const main = h('main', { class: 'main' }, ui.topbar, ui.banner, ui.content, ui.composerWrap);
  app.append(ui.sidebar, main, ui.toasts);
}

async function init() {
  // A failure the code does not handle is logged, and a refusal of the database is told.
  window.addEventListener('unhandledrejection', (e) => {
    const r = e.reason;
    if (r && r.name === 'AbortError') {
      e.preventDefault();
      return;
    }
    console.warn('Albireo: an operation failed', r);
    if (r && (r.name === 'QuotaExceededError' || r.name === 'UnknownError' || r.name === 'InvalidStateError')) {
      storageFailed(r);
    }
  });
  try {
    await store.open();
    state.settings = Object.assign({}, defaults, await store.getSettings());
    // The browser is asked to keep the database, without waiting for its answer.
    requestPersistence();
  } catch (e) {
    // Without IndexedDB (a private window of some browsers) the application keeps its
    // data in memory, as in an incognito discussion.
    console.warn('Albireo: the database of the browser is unavailable', e);
    state.incognito = true;
  }
  // The language of Albireo, when this interface follows it, before the first texts.
  state.status = await api.status().catch(() => null);
  applyLanguage();
  applyTheme();
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', applyTheme);
  // The documents that have passages, listed before the lists are read (pruneIndex), which
  // prunes them only once the lists are read whole.
  const indexed = state.incognito ? [] : await store.chunkDocuments().catch(() => []);
  const pictures = state.incognito ? [] : await store.imageKeys().catch(() => []);
  if (await reloadLists().then(() => true, () => false)) {
    pruneIndex(indexed).catch(() => {});
    pruneImages(pictures).catch(() => {});
  }
  buildLayout();
  window.addEventListener('hashchange', route);
  await route();
  await refreshStatus();
  setInterval(refreshStatus, 4000);
  /*
      The questions an earlier visit left to the memory, once the service answers, and the titles
      it left to write.
  */
  rememberLater();
  namePending();
  describePending();
  // A tab that closes gives back the lease of the memory and tells that it no longer answers.
  window.addEventListener('pagehide', () => {
    if (memoryRunning) releaseMemoryLease().catch(() => {});
    if (state.busyId) broadcast({ type: 'busy', id: state.busyId, tab: tabId, busy: false });
  });
}

init();
