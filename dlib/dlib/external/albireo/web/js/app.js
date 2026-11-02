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
import * as web from './web.js';
import * as images from './images.js';
import * as speech from './speech.js';

const roles = ['general', 'code', 'security'];
const roleIcons = { general: 'compass', code: 'code', security: 'shield' };
const projectColors = ['#5b8cff', '#f2b544', '#3fb68b', '#e5534b', '#a371f7', '#2fb9d4', '#e08a3c', '#8a90aa'];
// The steps of an answer with tools at most: each step is a request of the model, then the calls it asks for.
const maxAgentSteps = 100;

// Shares of the context of the model: the documents of a project read whole, an attached
// document read whole, and the excerpts given with a question.  Beyond the first two, the
// documents are searched.
const knowledgeShare = 0.3;
const attachmentShare = 0.25;
const excerptShare = 0.2;
/*
    The room of the exchanges is the room of the context less the system message and the tools.  When the oldest
    exchanges leave, the exchanges kept, with the summary of those that left, take keptShare of that room, so that
    they grow again for several questions before the next cut.  Once an answer ends, or a discussion is opened, with
    the exchanges taking compactShare of their room, their oldest part is compressed in the background: the next
    questions still fit meanwhile, and find the summary written.
*/
const keptShare = 0.3;
const compactShare = 0.5;
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

/*
    The memory of the discussions: short sentences on the user, one after the other, which the model writes from the
    discussions, private ones excepted, and which each discussion receives at its start.  Its size follows the
    context of the model: memoryShare of it, memoryWords words at most.
*/
const memoryWords = 200;
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
      else if (k === 'style' && typeof v === 'object') {
        // The custom properties ("--c") are set by their name.
        for (const [name, value] of Object.entries(v)) {
          if (name.startsWith('--')) el.style.setProperty(name, value);
          else el.style[name] = value;
        }
      }
      else el.setAttribute(k, v === true ? '' : v);
    }
  }
  for (const c of children.flat(Infinity)) {
    if (c == null || c === false) continue;
    el.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
  return el;
}

// Puts children in el in place of its own, the null and false ones left out, as h() leaves them out.
function fill(el, ...children) {
  el.replaceChildren(...children.flat(Infinity).filter((c) => c != null && c !== false));
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

// A limit of size in whole megabytes, "100 MB".
function formatLimit(n) {
  return formatNumber(Math.round(n / 1048576)) + ' MB';
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
  } else if (message.type === 'projects') {
    syncProjects();
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

/*
    The fields of a discussion another tab may change while this one answers in it or edits one of its questions: the
    title, the pin, the project, the function and the buttons of the composer, which the next question takes.  The
    answer being written keeps the function and the tools of its start.
*/
const sharedFields = ['title', 'titleSource', 'titleQuestion', 'pinned', 'projectId', 'role', 'reasoning', 'agent',
  'web'];

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
    Takes the copy of a discussion another tab saved.  A discussion this tab is answering, shown or
    not, keeps its messages and takes the title, the pin and the project; the one shown otherwise
    takes the saved copy whole, and its messages are drawn again when they changed.
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
  const live = generations.get(id);
  let redraw = false;
  // A question being edited keeps the copy it belongs to.
  if (live || (shown && (editingIn(ui.content) || (ui.thread && ui.thread.querySelector('textarea'))))) {
    copy = live ? live.c : shown;
    for (const k of sharedFields) copy[k] = stored[k];
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

/*
    Forgets a discussion deleted in another tab, stops its answer in progress and the compression of its exchanges
    here, forgets its draft, and never saves it again.
*/
function conversationDeleted(id) {
  deleted.add(id);
  othersBusy.delete(id);
  unread.delete(id);
  stopGeneration(id);
  cancelCompaction(id);
  cancelWarmup(id);
  web.forgetPages(id);
  forgetDraft(id);
  state.conversations = state.conversations.filter((x) => x.id !== id);
  if (state.current && state.current.id === id) go('#/');
  else refreshViews();
}

// Every discussion and every project deleted in another tab; the draft of "New chat" stays.
function everythingDeleted() {
  for (const c of state.conversations) deleted.add(c.id);
  if (state.current && state.current.messages.length && !state.current.incognito) deleted.add(state.current.id);
  for (const g of generations.values()) if (!g.c.incognito) deleted.add(g.c.id);
  stopAllGenerations();
  unread.clear();
  cancelIndex();
  cancelBackground();
  rag.removeAll();
  images.removeAll(waitingImages());
  forgetStoredDrafts();
  for (const id of deleted) web.forgetPages(id);
  syncLists().then(() => go('#/'));
}

/*
    The projects read again from the database after another tab changed one of them, once for a burst of
    messages: the discussions stay as they are.
*/
let projectsSync = null;

function syncProjects() {
  if (!projectsSync) {
    projectsSync = new Promise((resolve) => setTimeout(resolve, 50)).then(async () => {
      projectsSync = null;
      try {
        state.projects = await store.allProjects();
      } catch (e) {
        storageFailed(e);
      }
      refreshViews();
    });
  }
  return projectsSync;
}

/*
    The lists read again from the database, after an import, the deletion of a project or everything
    deleted in another tab, once for a burst of messages.  The discussion shown and the discussions this
    tab is answering stay the copies of this tab.
*/
let listsSync = null;

function syncLists() {
  if (!listsSync) {
    listsSync = new Promise((resolve) => setTimeout(resolve, 50)).then(async () => {
      listsSync = null;
      await reloadLists().catch(storageFailed);
      keepOwnCopies();
      refreshViews();
    });
  }
  return listsSync;
}

/*
    Puts the copies of this tab, the discussion shown and the discussions it is answering, in the list read from the
    database in place of the stored ones.  A discussion whose project was deleted leaves it, as the stored discussions
    did.
*/
function keepOwnCopies() {
  const own = Array.from(generations.values(), (g) => g.c);
  if (state.current) own.push(state.current);
  for (const c of own) {
    const i = state.conversations.findIndex((x) => x.id === c.id);
    if (i >= 0) state.conversations[i] = c;
    if (c.projectId && !projectById(c.projectId)) c.projectId = null;
  }
}

/*
    The discussions other tabs are answering: id -> {tab, until}.  A tab says it again every
    busyBeat milliseconds while it answers, so that the mark of a tab closed meanwhile expires.
*/
const othersBusy = new Map();
const busyBeat = 20000;

function busyElsewhere(id, tab, busy) {
  const before = othersBusy.has(id);
  if (busy) othersBusy.set(id, { tab, until: Date.now() + 3 * busyBeat });
  else if (othersBusy.has(id) && othersBusy.get(id).tab === tab) othersBusy.delete(id);
  if (state.current && state.current.id === id) {
    renderComposerState();
    lockThread();
  }
  // The dot of the discussion in the side panel follows.
  if (before !== othersBusy.has(id) && ui.sideScroll) renderSidebar();
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

/*
    Tells the other tabs that this tab answers, or no longer answers, in the discussion c, an incognito discussion
    excepted.  While this tab answers in stored discussions, it says so again for each of them every busyBeat
    milliseconds.
*/
function announceBusy(c, busy) {
  if (c && !c.incognito) broadcast({ type: 'busy', id: c.id, tab: tabId, busy });
  const stored = () => Array.from(generations.values()).filter((g) => !g.c.incognito);
  if (stored().length && !busyTimer) {
    busyTimer = setInterval(() => {
      for (const g of stored()) broadcast({ type: 'busy', id: g.c.id, tab: tabId, busy: true });
    }, busyBeat);
  } else if (!stored().length && busyTimer) {
    clearInterval(busyTimer);
    busyTimer = 0;
  }
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
    Why the web access cannot serve a discussion whose model has the capabilities caps, '' when it can: a model that
    calls no tool, a service set offline, a web access the administrator turned off (the "web" member of the status),
    or a status without it.
*/
function webUnavailable(caps) {
  if (!caps.tools) return t('webNoTools');
  const s = state.status;
  if (!s || !s.web || typeof s.web !== 'object') return t('webUnavailable');
  if (s.web.available) return '';
  return s.offline ? t('webOffline') : t('webOff');
}

// The texts of the providers of the web search that the status names (web.search).
const searchProviders = { wikipedia: 'provider_wikipedia', brave: 'provider_brave', searxng: 'provider_searxng' };

/*
    The provider of the web search of the service in the language of the interface, '' while the web access is
    unavailable or gives no search.  The help of the agent mode and the About window name it, as Brave asks of the
    projects that use the credit of its API.
*/
function searchProvider() {
  const w = state.status && state.status.web;
  if (!w || typeof w !== 'object' || !w.available) return '';
  const key = searchProviders[w.search];
  return key ? t(key) : '';
}

/*
    The tools of the requests of c and the parts of its system message that come with them: those of the agent while
    it is on, then those of the web while Web is on and the service offers it, web_search when the service names a
    provider of search.  The list and its order stay the same while the toggles and the provider do not change, so
    that the beginning of the prompt stays the same from one question to the next.
*/
function toolSet(c, caps) {
  const agent = !!(c.agent && caps.tools);
  const webOn = !!(c.web && !webUnavailable(caps));
  const provider = webOn ? state.status.web.search : '';
  const search = typeof provider === 'string' && provider !== '' && provider !== 'none';
  const definitions = (agent ? tools.definitions : []).concat(webOn ? web.definitions(search) : []);
  return { agent, web: webOn, search, definitions: definitions.length ? definitions : null };
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

/*
    The time a request of the background model may take: beyond it the request ends with a TimeoutError, which
    frees the work waiting behind it (the descriptions of the images, the next question after a compression).
*/
const backgroundTimeout = 10 * 60 * 1000;

// The text the background model writes after a list of messages; signal, when given, stops it too.
async function backgroundChat(messages, maxTokens, signal, temperature) {
  const body = {
    model: backgroundModel, reasoning: false, temperature: temperature == null ? 0.2 : temperature,
    max_tokens: maxTokens, messages
  };
  let text = '';
  let finish = null;
  const limit = api.limitedSignal(signal, backgroundTimeout);
  try {
    for await (const ev of api.streamChat(body, limit.signal)) {
      const choice = ev.type === 'chunk' && ev.data.choices && ev.data.choices[0];
      if (!choice) continue;
      if (choice.delta && choice.delta.content) text += choice.delta.content;
      if (choice.finish_reason) finish = choice.finish_reason;
    }
  } finally {
    limit.done();
  }
  return { text: text.trim(), finish };
}

// The reading of the status in progress, which the calls made meanwhile share.
let statusPoll = null;

function refreshStatus() {
  if (!statusPoll) statusPoll = pollStatus().finally(() => { statusPoll = null; });
  return statusPoll;
}

// Whether the service was ready at the previous reading of its status (serviceReady).
let serviceWasReady = false;

/*
    Reads the status of the service, with the models of the functions, which the window of the
    notification area may change, and whether the service answers.  A change draws the displays
    that show it again at once.  When the service becomes ready, at the opening while it starts,
    or after it did not answer, the work an earlier visit or a closed tab left resumes: the titles
    left waiting, the memory, and the descriptions of the images.
*/
async function pollStatus() {
  let changed = false;
  let back = false;
  let voices = false;
  try {
    const s = await api.status();
    // The web access and the offline mode change the Web button of the composer and the tools of the next question.
    const key = (x) => JSON.stringify([x.active, x.loaded_models, x.state, x.background, x.web, x.offline]);
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
    // The voices of the speech synthesis, asked again while the service gave none.
    if (!speechVoices.length) {
      await askSpeechVoices();
      voices = speechVoices.length > 0;
      changed = changed || voices;
    }
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
  // The answers drawn before the voices came receive their buttons, unless a question is being edited.
  if (voices && state.view === 'chat' && ui.thread && ui.thread.isConnected && !ui.thread.querySelector('textarea')
    && !editingIn(ui.content)) {
    renderMain();
  }
  const ready = serviceReady();
  const became = ready && !serviceWasReady;
  serviceWasReady = ready;
  if (became) {
    namePending();
    rememberLater();
  }
  if (back || changed || became) describePending();
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
  return !!c && (generations.has(c.id) || answeredElsewhere(c));
}

/*
    A discussion of the side panel.  At its left, a dot blinks while it is being answered, in this tab or in another
    one, and stays blue once its answer ended while another discussion was shown, until it is opened.
*/
function conversationItem(c) {
  const p = c.projectId ? projectById(c.projectId) : null;
  const active = state.view === 'chat' && state.current && state.current.id === c.id;
  const mark = answering(c) ? 'live' : unread.has(c.id) ? 'unread' : '';
  const markText = mark ? t(mark === 'live' ? 'answering' : 'unreadAnswer') : '';
  return h('div', { class: 'item' + (active ? ' active' : ''), onclick: () => go('#/c/' + c.id) },
    mark ? h('span', { class: 'gen-dot ' + mark, title: markText, role: 'img', 'aria-label': markText }) : null,
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

/*
    What the agent mode and the web access do, and the tools each of them gives the model, with what it is for; the
    provider of the web search last, while the service offers one.
*/
function agentHelp(anchor) {
  const row = (name) => h('div', { class: 'popover-tool' },
    h('span', { html: icon(toolIcons[name] || 'tool') }),
    h('span', null, h('b', null, toolLabel(name)), h('span', { class: 'desc' }, t('toolHint_' + name))));
  const provider = searchProvider();
  showPopover(anchor, [
    h('div', { class: 'head' }, t('agentHelpTitle')),
    h('p', null, t('agentHelp')),
    h('div', { class: 'popover-tools' }, tools.definitions.map((d) => row(d.function.name))),
    h('p', null, t('webHelp')),
    h('div', { class: 'popover-tools' }, web.definitions(true).map((d) => row(d.function.name))),
    provider ? h('p', { class: 'search-provider' }, t('webSearchBy', { provider })) : null
  ]);
}

/*
    The discussion this tab answers, shown or not, and the open discussion, are the copies that answer and save: an
    action of the sidebar on them changes that copy, so that the next save keeps the action.
*/
function liveCopy(c) {
  const g = generations.get(c.id);
  if (g) return g.c;
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
  const g = generations.get(c.id);
  const live = g && g.turn;
  const record = live && c.messages.includes(live)
    ? Object.assign({}, c, { messages: c.messages.filter((x) => x !== live) }) : c;
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
  delete live.titleQuestion;
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
  unread.delete(c.id);
  // The answer in progress, shown or written in the background, and the compression of the exchanges stop, and the
  // draft leaves.  The record of the discussion leaves the database first, then the passages of its documents, from
  // the memory of the search and from the database.
  stopGeneration(c.id);
  cancelCompaction(c.id);
  cancelWarmup(c.id);
  web.forgetPages(c.id);
  forgetDraft(c.id);
  await store.deleteConversation(c.id).catch(storageFailed);
  await forgetOwner(c.id).catch(() => {});
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
      lines.push('## ' + (state.settings.userName || t('exportUser')), '', m.content || '');
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
  if (compactions.has(c.id)) await compactions.get(c.id).done;
  const m = c.messages[index];
  let end = index + 1;
  while (end < c.messages.length && c.messages[end].role !== 'user') ++end;
  const removed = end - index;
  stopReadingOf(c, c.messages.slice(index, end), index);
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

/*
    Writes again the summary of the exchanges before the start of the context, after a deletion among them.  The job
    takes its place in compactions, marked always, so that a question sent meanwhile waits for the new summary.
*/
async function rewriteSummary(c) {
  const start = c.contextStart || 0;
  c.summary = null;
  if (start === 0 || !roleModel(c.role)) return;
  const context = roleCapabilities(c.role).context || 8192;
  const room = context - Math.min(state.settings.maxTokens || 2048, Math.floor(context / 2));
  toast(t('compacting'));
  cancelWarmup(c.id);
  const abort = new AbortController();
  const done = compact(c, 0, start, room, 1 / 3.6, abort.signal).then((r) => applyCompaction(c, r))
    .catch(() => { c.summary = null; });
  const job = { done, abort, always: true };
  compactions.set(c.id, job);
  try {
    await done;
  } finally {
    if (compactions.get(c.id) === job) compactions.delete(c.id);
  }
  if (c.summary) warmBeginning(c);
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

/*
    Indexes the searched documents of the questions of a discussion that have no passages, such as those of a
    discussion imported, since an export carries no passages, or those a tab closed while they were written.
*/
async function indexAttachments(c) {
  if (!c || !Array.isArray(c.messages)) return;
  const docs = new Map();
  for (const m of c.messages) {
    if (m && m.role === 'user') for (const a of m.attachments || []) if (a.mode === 'search') docs.set(a.id, a);
  }
  if (!docs.size) return;
  const st = await rag.status(c.id);
  for (const a of docs.values()) if (!st.has(a.id)) queueIndex(c.id, a, !c.incognito);
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
    time (describeImage): image -> project, for the images waiting or being described by this tab,
    and image -> the AbortController of the description being written.
*/
const describing = new Map();
const describeJobs = new Map();
let describeQueue = Promise.resolve();

// Stops the description of an image being written, when its image or its project leaves.
function cancelDescription(imageId) {
  const job = describeJobs.get(imageId);
  if (job) job.abort();
}

// Stops the work of the background model in progress in this tab: the compressions and the descriptions.
function cancelBackground() {
  for (const id of Array.from(compactions.keys())) cancelCompaction(id);
  for (const id of Array.from(describeJobs.keys())) cancelDescription(id);
  for (const id of Array.from(warmups.keys())) cancelWarmup(id);
}

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
  broadcast({ type: 'projects' });
  showProjectImage(projectId, imageId);
  return img;
}

/*
    Asks the background model for the description of an image of a project, in the language of the
    interface: the kind of image, its visible text, its elements and its data.  The description is
    kept with the image and takes its place in the search of the project.  Without a background
    model that reads images, or after an error, the name of the image stands for its description,
    and the description is asked again later (describePending) or with the button of the image.
    One tab describes an image at a time.  The request ends when the image or its project leaves,
    and after backgroundTimeout, so that the descriptions waiting behind it and its lock are freed.
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
    const job = new AbortController();
    describeJobs.set(imageId, job);
    try {
      if (serviceReady() && backgroundReadsImages()) {
        const url = await images.dataUrl(imageId).catch(() => null);
        if (url) {
          try {
            const r = await backgroundChat(describeMessages(url, img.name), 1024, job.signal, 0.2);
            text = cleanDescription(r.text);
          } catch (e) {
            console.warn('Albireo: the description of an image failed', e);
            text = '';
          }
        }
      }
    } finally {
      describeJobs.delete(imageId);
    }
    // An image or a project removed meanwhile keeps no state.
    if (job.signal.aborted) return;
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
  broadcast({ type: 'projects' });
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
  /*
      Saves an edit of the project, edit(project) changing the fields it concerns.  The edit applies to the
      project as stored, read again, so that what another tab or the descriptions of the images wrote meanwhile
      stays; a project deleted meanwhile is not written again.  The name, the description and the color show at
      once in the page, the top bar, the side panel and the other tabs.
  */
  const save = async (edit) => {
    let stored = p;
    try {
      stored = await store.getProject(p.id);
    } catch (e) {
      // The database is out of reach: the copy of this tab takes the edit.
    }
    if (!stored) return;
    if (stored !== p) {
      for (const k of Object.keys(p)) if (!(k in stored)) delete p[k];
      Object.assign(p, stored);
    }
    edit(p);
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
    broadcast({ type: 'projects' });
    renderSidebar();
    renderTopbar();
  };
  name.addEventListener('change', () => save((x) => { x.name = name.value.trim() || x.name; }));
  description.addEventListener('change', () => save((x) => { x.description = description.value; }));
  instructions.addEventListener('change', () => save((x) => { x.instructions = instructions.value; }));
  const colors = h('div', { class: 'colors' });
  for (const col of projectColors) {
    colors.append(h('button', {
      type: 'button', class: p.color === col ? 'on' : '', style: { background: col }, 'aria-label': col,
      onclick: async () => {
        await save((x) => { x.color = col; });
        renderProject(p);
      }
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
    await save((x) => {
      x.files = (x.files || []).concat(docs);
      if (pictures.length) x.images = (x.images || []).concat(pictures);
    });
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
        await save((x) => { x.files = (x.files || []).filter((y) => y.id !== f.id); });
        await removed;
        renderProject(p);
      })));
    indexText(p.id, f.id).then((text) => { indexState.textContent = text; });
  }
  // The images, each with the state of its description.
  ui.projectPage = {
    id: p.id,
    removeImage: async (img) => {
      cancelDescription(img.id);
      const removed = forgetDocument(p.id, img.id);
      await save((x) => { x.images = (x.images || []).filter((y) => y.id !== img.id); });
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
          // The descriptions of its images stop, the documents leave the search, in memory and in the
          // database, and the discussion shown leaves the project as the stored ones do.
          for (const img of p.images || []) cancelDescription(img.id);
          await forgetOwner(p.id).catch(() => {});
          await store.deleteProject(p.id).catch(storageFailed);
          await images.removeOwner(p.id).catch(() => {});
          if (state.current && state.current.projectId === p.id) state.current.projectId = null;
          for (const g of generations.values()) if (g.c.projectId === p.id) g.c.projectId = null;
          await reloadLists().catch(storageFailed);
          keepOwnCopies();
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
    is added with a notice that counts them.  The images come back with kind 'image'.  A document of more than
    maxFileBytes and an image of more than images.maxImageBytes are refused before they are read, with a message
    that names the file and the limit.
*/
async function readFiles(files, role, room) {
  const out = [];
  const scans = [];
  const refused = [];
  const inputs = role ? roleInputs(role) : ['text', 'image'];
  let left = room || 0;
  for (const f of files) {
    const kind = fileKind(f);
    if (kind !== 'document' && !inputs.includes(kind)) {
      const model = role ? modelName(roleModel(role)) : '';
      toast(role ? t('inputRefused', { name: f.name, model, inputs: inputsLabel(inputs) })
        : t('unsupported', { name: f.name }), true);
      continue;
    }
    const limit = kind === 'image' ? images.maxImageBytes : maxFileBytes;
    if (f.size > limit) {
      toast(t(kind === 'image' ? 'imageTooLarge' : 'tooLarge',
        { name: f.name, size: formatSize(f.size), limit: formatLimit(limit) }), true, 9000);
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
      else if (e.message === 'too-large') {
        toast(t('tooLarge', { name: f.name, size: formatSize(f.size), limit: formatLimit(maxFileBytes) }), true, 9000);
      }
      else if (e.message === 'too-large-content') toast(t('tooLargeContent', { name: f.name }), true);
      else if (e.message === 'pdf-password') toast(t('pdfPassword', { name: f.name }), true, 9000);
      else if (e.message === 'pdf-invalid') toast(t('pdfInvalid', { name: f.name }), true);
      else if (e.message === 'pdf-reader') toast(t('pdfReader', { name: f.name }), true, 12000);
      else toast(t('readFailed', { name: f.name, error: errorText(e) }), true);
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
    toast(t('readFailed', { name, error: errorText(e) }), true);
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
    // The image the viewer shows keeps its object URL (releaseHiddenImages).
    pic.dataset.image = img.id;
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

/*
    The drafts of the composer, one per discussion: the text typed and the documents and images waiting to be sent.
    The composer holds the draft of the discussion shown (ui.input, state.pending), drafts those of the other
    discussions of this tab, by their key (draftKey).  The discussion that "New chat" opens keeps its draft under
    newDraft until its first question creates the discussion.  The texts stay across a reload of the page in
    localStorage, one key per discussion, removed once its question is sent or the discussion deleted; the files stay
    in the memory of the tab.  An incognito discussion keeps no draft.
*/
const drafts = new Map();
const newDraft = 'new';
const draftPrefix = 'albireo-draft-';
// The text typed is kept this many milliseconds after the last key.
const draftDelay = 400;
let draftTimer = 0;

// The key of the draft of c: its id, newDraft while it is not stored, '' for an incognito or deleted discussion.
function draftKey(c) {
  if (!c || c.incognito || deleted.has(c.id)) return '';
  return c.messages.length || state.conversations.some((x) => x.id === c.id) ? c.id : newDraft;
}

// The text of the draft key that localStorage keeps, '' without one.
function storedDraft(key) {
  if (!key) return '';
  try {
    return localStorage.getItem(draftPrefix + key) || '';
  } catch (e) {
    return '';
  }
}

// Keeps the text of the draft key in localStorage, or removes it when the text holds nothing to send.
function storeDraft(key, text) {
  if (!key) return;
  try {
    if (String(text || '').trim()) localStorage.setItem(draftPrefix + key, text);
    else localStorage.removeItem(draftPrefix + key);
  } catch (e) {
    // Without localStorage, or beyond its room, the text stays in the memory of the tab.
  }
}

// Keeps the text of the composer for the discussion shown once the user stops typing, at once with now.
function saveDraft(now) {
  clearTimeout(draftTimer);
  draftTimer = 0;
  if (!now) {
    draftTimer = setTimeout(() => saveDraft(true), draftDelay);
    return;
  }
  if (ui.input) storeDraft(draftKey(state.current), ui.input.value);
}

// The ids of the images of a list of attachments.
function imageIdsOf(list) {
  return list.filter((a) => a.kind === 'image').map((a) => a.id);
}

// The images waiting in the drafts that stay once everything is deleted: those of the composer and of "New chat".
function waitingImages() {
  const d = drafts.get(newDraft);
  return new Set(imageIdsOf(state.pending.concat(d ? d.files : [])));
}

// Forgets the draft key, once its question is sent or its discussion deleted, here or in another tab.
function forgetDraft(key) {
  if (!key) return;
  const d = drafts.get(key);
  if (d) images.drop(imageIdsOf(d.files));
  drafts.delete(key);
  storeDraft(key, '');
}

// Forgets the drafts of the stored discussions, once everything was deleted; the one of "New chat" stays.
function forgetStoredDrafts() {
  for (const key of Array.from(drafts.keys())) if (key !== newDraft) forgetDraft(key);
  try {
    for (let i = localStorage.length - 1; i >= 0; --i) {
      const k = localStorage.key(i);
      if (k && k.startsWith(draftPrefix) && k !== draftPrefix + newDraft) localStorage.removeItem(k);
    }
  } catch (e) {
    // Without localStorage, no text was kept.
  }
}

// Removes the texts kept for discussions that no longer exist, such as those deleted while no tab was open.
function pruneDrafts() {
  const ids = new Set(state.conversations.map((c) => c.id));
  try {
    for (let i = localStorage.length - 1; i >= 0; --i) {
      const k = localStorage.key(i);
      if (!k || !k.startsWith(draftPrefix)) continue;
      const id = k.slice(draftPrefix.length);
      if (id !== newDraft && !ids.has(id)) localStorage.removeItem(k);
    }
  } catch (e) {
    // Without localStorage, no text was kept.
  }
}

/*
    Shows the discussion c in place of the one shown, each with its own draft.  The draft of the one shown leaves the
    composer for drafts and localStorage, or leaves the memory for an incognito or deleted discussion; the draft of c
    takes its place: the one this tab keeps, else the text of localStorage, else nothing.
*/
function showDiscussion(c) {
  clearTimeout(draftTimer);
  draftTimer = 0;
  const key = draftKey(state.current);
  const text = ui.input ? ui.input.value : '';
  if (key) {
    if (text.trim() || state.pending.length) drafts.set(key, { text, files: state.pending });
    else drafts.delete(key);
    storeDraft(key, text);
  } else {
    images.drop(imageIdsOf(state.pending));
  }
  state.current = c;
  const next = draftKey(c);
  const d = next ? drafts.get(next) : null;
  if (d) drafts.delete(next);
  state.pending = d ? d.files : [];
  if (ui.input) {
    ui.input.value = d ? d.text : storedDraft(next);
    autoGrow();
  }
}

/*
    The discussion shown gives way to another one: an incognito discussion, which no list keeps, takes its images, the
    passages of its documents and the web pages it read out of memory.
*/
function forgetIncognito() {
  const c = state.current;
  if (!c || !c.incognito || !c.messages.length) return;
  cancelWarmup(c.id);
  web.forgetPages(c.id);
  images.removeOwner(c.id).catch(() => {});
  forgetOwner(c.id).catch(() => {});
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

// Conversations.  The web access starts off in every discussion, whatever the previous ones did.
function newConversation(projectId) {
  return {
    id: store.uid(), title: '', projectId: projectId || null, created: Date.now(), updated: Date.now(), pinned: false,
    role: state.settings.defaultRole || 'general', reasoning: !!state.settings.lastReasoning, agent: !!state.settings.lastAgent,
    web: false, messages: [], incognito: state.incognito
  };
}

/*
    The discussion shown gives way to another one.  The answer of a stored discussion goes on in the background; an
    incognito discussion, which no list keeps, stops its answer, and its images and the passages of its documents
    leave the memory.
*/
function leaveCurrent() {
  const c = state.current;
  if (c && c.incognito) stopGeneration(c.id);
  forgetIncognito();
}

// A new discussion, which the draft of "New chat" fills.
function newChat(projectId) {
  // The reading aloud ends with the discussion it belongs to, as route() ends it.
  speech.stopReading();
  leaveCurrent();
  showDiscussion(newConversation(projectId));
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
  // A reading aloud ends with the page it belongs to.
  speech.stopReading();
  closeMenu();
  const sequence = ++routeSequence;
  const hash = location.hash || '#/';
  let m;
  if ((m = /^#\/c\/([0-9a-f]+)$/.exec(hash))) {
    if (!(state.current && state.current.id === m[1])) {
      // A discussion this tab is answering in the background is shown with its answer being written.
      const live = generations.get(m[1]);
      const c = live ? live.c : deleted.has(m[1]) ? null : await store.getConversation(m[1]).catch(() => null);
      if (sequence !== routeSequence) return;
      if (!c) { go('#/'); return; }
      leaveCurrent();
      showDiscussion(c);
      state.incognito = false;
      // The searched documents without passages, those of an imported discussion for instance, are indexed.
      indexAttachments(c).catch(() => {});
      // A discussion whose exchanges take the share of their room that leads to a compression starts it now,
      // unless another tab is answering in it.
      if (!live && !answeredElsewhere(c) && serviceReady()) compactLater(c);
    }
    // The answer written while another discussion was shown is read now.
    unread.delete(m[1]);
    state.view = 'chat';
  } else if ((m = /^#\/p\/([0-9a-f]+)$/.exec(hash))) {
    state.view = 'project';
    state.projectId = m[1];
  } else if (hash === '#/licenses') {
    state.view = 'licenses';
  } else {
    state.view = 'chat';
    if (!state.current || state.current.messages.length || (state.current.incognito !== state.incognito)) {
      leaveCurrent();
      showDiscussion(newConversation(null));
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
  // The text the title shows when the user starts to edit it: leaving it unchanged renames nothing, and shows the
  // title the model may have written meanwhile.
  let shownText = '';
  title.addEventListener('focus', () => { shownText = title.textContent.trim(); });
  title.addEventListener('blur', async () => {
    if (!c || !c.messages.length) return;
    const live = liveCopy(c);
    const v = title.textContent.trim();
    if (v && v !== shownText && v !== live.title) {
      live.title = v;
      // A title of the user stays: the model no longer names the discussion.
      live.titleSource = 'user';
      delete live.titleQuestion;
      await persist(live);
      refreshViews();
    } else {
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

/*
    Draws the main area for the view.  Afterwards, the reading aloud of an answer the page no longer shows ends, and
    the images it no longer shows give back their object URLs.
*/
function renderMain() {
  drawMain();
  keepReadingShown();
  releaseHiddenImages();
}

// Ends the reading aloud of an answer that the thread shown no longer holds, such as one deleted or regenerated.
function keepReadingShown() {
  const key = speech.readingKey();
  if (!key) return;
  const buttons = ui.thread && ui.thread.isConnected ? Array.from(ui.thread.querySelectorAll('[data-speech]')) : [];
  if (!buttons.some((b) => b.dataset.speech === key)) speech.stopReading();
}

// Revokes the object URLs of the images that no element of the page shows (data-image), the viewer included.
function releaseHiddenImages() {
  images.releaseExcept(new Set(Array.from(document.querySelectorAll('[data-image]'), (el) => el.dataset.image)));
}

function drawMain() {
  stopClock();
  // The buttons that go to the ends of the thread wait for the next scroll of the user.
  if (ui.jump) ui.jump.classList.remove('on');
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
  // The composer shown again takes the height of its text, which it could not measure while hidden.
  const hidden = ui.composerWrap.classList.contains('hidden');
  ui.composerWrap.classList.remove('hidden');
  if (hidden) autoGrow();
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
      class: 'suggestion', onclick: () => { ui.input.value = prompt; autoGrow(); saveDraft(true); ui.input.focus(); }
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

/*
    The thread follows its end while the user leaves it there (ui.stick): an answer being written, and a composer
    that grows, keep its last lines in view.  The position is written without reading it first; the scrolls the
    application makes are told apart from those of the user by the position they leave (ui.ownTop).
*/
function scrollToEnd(force) {
  if (!force && !ui.stick) return;
  const s = ui.content;
  s.scrollTop = s.scrollHeight;
  ui.ownTop = s.scrollTop;
  ui.stick = true;
}

// A scroll of the thread: whether it stays at its end, and, for a scroll of the user, the buttons that go to its ends.
function threadScrolled() {
  const s = ui.content;
  const own = Math.abs(s.scrollTop - ui.ownTop) < 2;
  if (!own) ui.stick = s.scrollHeight - s.scrollTop - s.clientHeight < 80;
  if (!own) showJump(true);
  else if (ui.jump.classList.contains('on')) showJump(false);
}

/*
    The buttons that go to the start and to the end of the thread show while the user scrolls a thread longer than the
    view, each one away from its end, and leave a moment after the scroll unless the pointer or the focus is on them.
    keep tells that the user scrolls, which shows them again.
*/
const jumpDelay = 1600;

function showJump(keep) {
  const s = ui.content;
  const thread = state.view === 'chat' && ui.thread && ui.thread.isConnected;
  const fits = !thread || s.scrollHeight <= s.clientHeight + 4;
  const atStart = s.scrollTop < 8;
  const atEnd = s.scrollHeight - s.scrollTop - s.clientHeight < 8;
  ui.jumpStart.classList.toggle('hidden', fits || atStart);
  ui.jumpEnd.classList.toggle('hidden', fits || atEnd);
  const on = !fits && !(atStart && atEnd) && (keep || ui.jump.classList.contains('on'));
  ui.jump.classList.toggle('on', on);
  clearTimeout(ui.jumpTimer);
  if (on) ui.jumpTimer = setTimeout(hideJump, jumpDelay);
}

function hideJump() {
  clearTimeout(ui.jumpTimer);
  if (ui.jump.matches(':hover') || ui.jump.contains(document.activeElement)) {
    ui.jumpTimer = setTimeout(hideJump, jumpDelay);
    return;
  }
  ui.jump.classList.remove('on');
}

// Goes at once to the start or to the end of the thread.
function jumpTo(end) {
  const s = ui.content;
  if (end) {
    scrollToEnd(true);
  } else {
    s.scrollTop = 0;
    ui.ownTop = s.scrollTop;
    ui.stick = false;
  }
  showJump(true);
}

/*
    The composer lies over the bottom of the thread: the height it takes, whatever its text, its attachments and the
    tools of the agent, goes to --composer-room, which the bottom of the thread and the buttons over it leave free.
    A thread at its end stays at its end.
*/
function watchComposer() {
  if (ui.composerObserver) ui.composerObserver.disconnect();
  ui.composerObserver = null;
  if (typeof ResizeObserver !== 'function') return;
  ui.composerObserver = new ResizeObserver(() => {
    const height = ui.composerWrap.offsetHeight;
    if (height === ui.composerHeight) return;
    ui.composerHeight = height;
    ui.main.style.setProperty('--composer-room', height + 'px');
    if (ui.stick && ui.thread && ui.thread.isConnected) scrollToEnd(true);
  });
  ui.composerObserver.observe(ui.composerWrap);
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
  el.dataset.id = m.id || '';
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
  return busyHere() || answeredElsewhere(state.current);
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
    lockButton(b, b.dataset.lock === 'turn' ? !!b.closest('.message.live') : busy);
  }
  showElsewhere();
}

// The note at the end of the thread while another tab answers in its discussion.
function showElsewhere() {
  if (!ui.thread || !ui.thread.isConnected) return;
  const note = ui.thread.querySelector(':scope > .elsewhere');
  const on = !busyHere() && answeredElsewhere(state.current);
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
        // A compression in progress moves the start of the discussion: it ends first.
        if (compactions.has(c.id)) await compactions.get(c.id).done;
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
        // The reading aloud of an answer that leaves with the edited question ends.
        stopReadingOf(c, c.messages.slice(index), index);
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

// The languages the speech synthesis of the service reads, empty while it gives none.
let speechVoices = [];

async function askSpeechVoices() {
  if (!speechVoices.length) speechVoices = await speech.speechLanguages();
}

/*
    The key of the reading aloud of the message m of c, at index: the identifier of the message, which deletions,
    edits and regenerations leave to it, else its position for a message saved without one.
*/
function speechKey(c, m, index) {
  return c.id + ':' + (m.id || 'at' + index);
}

// Ends the reading aloud of one of the messages of c that leave the discussion, the first of them at from.
function stopReadingOf(c, messages, from) {
  const key = speech.readingKey();
  if (key && messages.some((m, k) => speechKey(c, m, from + k) === key)) speech.stopReading();
}

// The buttons of the answer key show whether it is being read.
function markReading(key, on) {
  for (const b of document.querySelectorAll('[data-speech]')) {
    if (b.dataset.speech !== key) continue;
    b.innerHTML = icon(on ? 'stop' : 'speaker');
    b.title = t(on ? 'stopReading' : 'readAloud');
    b.setAttribute('aria-label', b.title);
    b.setAttribute('aria-pressed', on ? 'true' : 'false');
  }
}

// Reads an answer aloud in the language of its words, or stops the reading of this answer.
function toggleReading(m, key) {
  if (speech.readingKey() === key) {
    speech.stopReading();
    return;
  }
  const text = speech.spokenText(answerText(m));
  const fallback = speechVoices.includes(language()) ? language() : speechVoices[0];
  const lang = speech.textLanguage(text, fallback);
  speech.readAloud(text, speechVoices.includes(lang) ? lang : fallback, key, {
    start: (k) => markReading(k, true),
    end: (k) => markReading(k, false),
    error: (k, e) => toast(t('readAloudFailed', { reason: errorText(e) }), true)
  });
}

/*
    An answer, with its buttons.  Copy, read aloud and export wait until the answer is written, whether it is shown
    while it streams or written in the background.
*/
function assistantElement(c, m, index, last) {
  const live = liveMessage(c, m);
  const el = h('div', { class: 'message assistant' + (last ? ' last' : '') + (live ? ' live' : '') });
  el.dataset.id = m.id || '';
  const bubble = h('div', { class: 'bubble' });
  el.append(bubble);
  fillAssistant(bubble, m, live);
  // The buttons of an answer wait until it is written; regenerate also waits for the discussion.
  const meta = h('div', { class: 'meta' });
  meta.append(lockable(iconButton('copy', t('copy'), () => copyText(answerText(m))), 'turn', live));
  if (speechVoices.length) {
    // The answer read aloud, once it is written.
    const key = speechKey(c, m, index);
    const on = speech.readingKey() === key;
    const reader = lockable(iconButton(on ? 'stop' : 'speaker', t(on ? 'stopReading' : 'readAloud'),
      () => toggleReading(m, key)), 'turn', live);
    reader.dataset.speech = key;
    reader.setAttribute('aria-pressed', on ? 'true' : 'false');
    meta.append(reader);
  }
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
  if (m.usage && m.usage.completion_tokens) {
    parts.push(t('tokensCount', { n: formatNumber(m.usage.completion_tokens) }));
  }
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

// What a step shows, to tell whether it must be drawn again.
function stepKey(s, active) {
  const calls = (s.toolCalls || []).map(toolKey);
  return JSON.stringify([active, (s.reasoning || '').length, (s.content || '').length, calls]);
}

// What a tool call shows, to tell whether it must be drawn again.
function toolKey(tc) {
  return JSON.stringify([tc.status, tc.ok, (tc.arguments || '').length, tc.result == null ? -1 : tc.result.length,
    tc.ms || 0, !!tc.file]);
}

function stepElement(s, active, firstTool) {
  const el = h('div', { class: 'step' });
  if (s.reasoning) el.append(thinkingElement(s, active && !s.content && !(s.toolCalls || []).length));
  if (s.content) {
    const md = h('div', { class: 'md', html: stepHtml(s, active) });
    attachCodeButtons(md, t('copy'));
    el.append(md);
  }
  (s.toolCalls || []).forEach((tc, j) => el.append(toolElement(tc, firstTool + j + 1)));
  return el;
}

/*
    The step of an answer being written, drawn in place.  el is the element the previous drawing made, or null.  The
    reasoning receives the text that came since then, the text of the answer its new blocks (liveMarkdown), and a
    tool call is drawn again when its state changes, so that a drawing costs the new tokens only, whatever the length
    of the answer.  Returns the element of the step.
*/
function liveStep(el, s, firstTool) {
  if (!el || !el.live) {
    el = h('div', { class: 'step' });
    el.live = { thinking: null, thinkingLive: false, label: null, text: null, shown: 0, md: null, cut: 0, tail: null,
      tailText: null, calls: [] };
  }
  const st = el.live;
  const calls = s.toolCalls || [];
  if (s.reasoning) {
    const thinking = !s.content && !calls.length;
    if (!st.thinking || st.thinkingLive !== thinking) {
      const d = thinkingElement(s, thinking);
      if (st.thinking) st.thinking.replaceWith(d);
      else el.prepend(d);
      st.thinking = d;
      st.thinkingLive = thinking;
      st.label = d.querySelector('summary > .label');
      st.text = d.querySelector('.thought').firstChild;
      st.shown = s.reasoning.length;
    } else {
      if (s.reasoning.length > st.shown) {
        st.text.appendData(s.reasoning.slice(st.shown));
        st.shown = s.reasoning.length;
      }
      if (thinking) st.label.textContent = thinkingLabel(s);
    }
  }
  if (s.content) {
    if (!st.md) {
      st.md = h('div', { class: 'md' });
      el.insertBefore(st.md, st.calls.length ? st.calls[0].el : null);
    }
    liveMarkdown(st, s.content);
  }
  calls.forEach((tc, j) => {
    const key = toolKey(tc);
    const old = st.calls[j];
    if (old && old.key === key) return;
    const wrap = toolElement(tc, firstTool + j + 1);
    if (old) old.el.replaceWith(wrap);
    else el.append(wrap);
    st.calls[j] = { key, el: wrap };
  });
  return el;
}

/*
    The text of the step being written, drawn by blocks in st.md: the part that later tokens no longer change
    (settledEnd) is drawn once, block after block, and the rest, the block being written, is drawn again when it
    changes.  The answer once written is drawn whole again, from its whole text.
*/
function liveMarkdown(st, text) {
  const cut = settledEnd(text, st.cut);
  if (cut > st.cut) {
    const block = h('div', { class: 'md-block', html: renderMarkdown(text.slice(st.cut, cut), true) });
    attachCodeButtons(block, t('copy'));
    st.md.insertBefore(block, st.tail);
    st.cut = cut;
  }
  const rest = text.slice(st.cut);
  if (rest === st.tailText) return;
  if (!st.tail) {
    st.tail = h('div', { class: 'md-block' });
    st.md.append(st.tail);
  }
  st.tail.innerHTML = renderMarkdown(rest, true);
  attachCodeButtons(st.tail, t('copy'));
  st.tailText = rest;
}

/*
    The end of the part of a text being written that later tokens no longer change: the start of the last line,
    after from, that follows a blank line and begins at the margin, outside a code block and outside a formula
    between $$ or \[ \].  A line that begins with a space may continue a list item, and stays with what follows.
    Returns from when there is no such line.  The text before from ends outside a code block and a formula.
*/
function settledEnd(text, from) {
  let end = from;
  let fence = '';
  let math = false;
  let blank = false;
  for (let at = from; at < text.length;) {
    const nl = text.indexOf('\n', at);
    const line = text.slice(at, nl < 0 ? text.length : nl);
    if (!fence && !math && blank && /^\S/.test(line)) end = at;
    // The last line, not ended yet, only tells where a block starts.
    if (nl < 0) break;
    if (fence) {
      const close = /^\s*(`{3,}|~{3,})\s*$/.exec(line);
      if (close && close[1][0] === fence[0] && close[1].length >= fence.length) fence = '';
      blank = false;
    } else {
      const open = /^\s*(`{3,}|~{3,})/.exec(line);
      if (open) {
        fence = open[1];
      } else if (((line.match(/\$\$/g) || []).length + (line.match(/\\\[|\\\]/g) || []).length) % 2) {
        math = !math;
      }
      blank = !open && /^\s*$/.test(line);
    }
    at = nl + 1;
  }
  return end;
}

/*
    Fills the body of an answer: per step, its reasoning, its text and its tool calls, then the
    notes on its state.  While the answer is written, the steps already drawn that did not change
    stay as they are, the step being written is drawn in place (liveStep), and the notes change
    only when what they say changes.  A step that ended no longer changes while the answer is
    written: once drawn whole, it is left as it is, so that a drawing costs the same whatever the
    number of steps.
*/
function fillAssistant(bubble, m, live) {
  const steps = m.steps || [];
  const drawn = Array.from(bubble.children).filter((x) => x.classList.contains('step'));
  // The steps come before the notes.
  const place = (el) => bubble.insertBefore(el, bubble.querySelector(':scope > :not(.step)'));
  let tools = 0;
  steps.forEach((s, k) => {
    const active = live && k === steps.length - 1;
    const first = tools;
    tools += (s.toolCalls || []).length;
    const old = drawn[k] || null;
    if (active) {
      const el = liveStep(old, s, first);
      if (el !== old) {
        if (old) old.replaceWith(el);
        else place(el);
      }
      return;
    }
    if (live && old && !old.live && old.dataset.key) return;
    const key = stepKey(s, false);
    if (old && !old.live && old.dataset.key === key) return;
    const el = stepElement(s, false, first);
    el.dataset.key = key;
    if (old) old.replaceWith(el);
    else place(el);
  });
  for (let k = steps.length; k < drawn.length; ++k) drawn[k].remove();
  const notes = noteElements(m, live, steps);
  const key = notes.map((x) => x.outerHTML).join('');
  if (bubble.notesKey === key) return;
  for (const x of Array.from(bubble.children)) if (!x.classList.contains('step')) x.remove();
  bubble.append(...notes);
  bubble.notesKey = key;
}

// The notes on the state of an answer, under its steps.
function noteElements(m, live, steps) {
  const out = [];
  const note = (...children) => out.push(h('div', { class: 'progress-line' }, ...children));
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
  if (m.error) out.push(h('div', { class: 'error-box' }, t('errorPrefix', { message: m.error })));
  if (m.cancelled) note(t('cancelled'));
  if (!live && !m.error && !m.cancelled && !steps.some((s) => s.content || (s.toolCalls || []).length)) {
    note(t('emptyAnswer'));
  }
  if (m.agentLimit) note(t('agentLimit', { n: maxAgentSteps }));
  return out;
}

// The label of a reasoning being written, with the time it has taken.
function thinkingLabel(s) {
  return t('thinking') + '…' + (s.reasoningStart ? ' ' + formatDuration(Date.now() - s.reasoningStart) : '');
}

function thinkingElement(s, live) {
  const d = h('details', { class: 'thinking' + (live ? ' live' : '') });
  // Closed, while the model thinks and afterwards: the user opens it, and it then stays as
  // the user left it.
  if (s.expanded) d.open = true;
  d.addEventListener('toggle', () => { s.expanded = d.open; });
  let label;
  if (live) {
    label = h('span', { class: 'label shimmer' }, thinkingLabel(s));
  } else {
    label = h('span', { class: 'label' },
      s.reasoningMs ? t('thoughtFor', { s: formatDuration(s.reasoningMs) }) : t('thoughtBriefly'));
  }
  d.append(h('summary', null, h('span', { html: icon('spark', 'spark') }), label, h('span', { html: icon('chevron', 'chev') })),
    h('div', { class: 'thought' }, s.reasoning));
  return d;
}

const toolIcons = {
  calculator: 'calc', run_javascript: 'terminal', search_documents: 'search', document_outline: 'archive',
  read_document: 'book', search_discussions: 'history', current_datetime: 'clock', create_file: 'file',
  web_search: 'globe', fetch_page: 'link'
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
    case 'search_documents': case 'search_discussions': case 'web_search': return quoted(a.query);
    case 'read_document': return [a.name, a.part].filter(Boolean).map(text).join(' · ');
    case 'fetch_page': return [a.url, a.start].filter(Boolean).map(text).join(' · ');
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
    The words of a call of fetch_page that read its page (tc.web), which the rule of the addresses allowed: the address
    as a link that opens in a new tab without the page of Albireo as its opener or referrer, then the start of the part
    read.  null for the other calls, whose words are text.
*/
function pageBrief(tc) {
  const url = tc.name === 'fetch_page' && tc.web ? web.addressKey(tc.web.url) : '';
  if (!url) return null;
  let start = 0;
  try {
    start = parseInt(JSON.parse(tc.arguments || '{}').start, 10) || 0;
  } catch (e) {
    start = 0;
  }
  return [h('a', { href: url, target: '_blank', rel: 'noopener noreferrer', title: url }, url),
    start > 0 ? ' · ' + start : null];
}

/*
    A tool call of the agent or of the web, with its number in the answer, the name of the tool in
    the language of the interface, the words of its arguments, a short result on the same line, its
    state and its duration.  Unfolded, it shows its arguments and its whole result.
*/
function toolElement(tc, number) {
  const st = toolState(tc);
  const d = h('details', { class: 'tool-call ' + st.cls });
  const open = !!tc.open;
  if (open) d.open = true;
  d.addEventListener('toggle', () => { if (d.open !== open) tc.open = d.open; });
  const brief = pageBrief(tc) || toolBrief(tc);
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
  // The Web button is off and disabled, with the reason as its title, while the model calls no tool or the service
  // gives no web access.
  const reason = webUnavailable(caps);
  const webOn = !!(c && c.web && !reason);
  ui.webButton.classList.toggle('on', webOn);
  ui.webButton.setAttribute('aria-pressed', String(webOn));
  ui.webButton.disabled = !!reason;
  ui.webButton.title = reason || t('webHint');
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
  // A discussion another tab is answering takes its next question once that answer is written; the answer this tab
  // writes in the discussion shown can be stopped.
  const busy = busyHere();
  const elsewhere = !busy && answeredElsewhere(c);
  ui.sendButton.classList.toggle('stop', busy);
  ui.sendButton.innerHTML = icon(busy ? 'stop' : 'send');
  ui.sendButton.title = busy ? t('stop') : elsewhere ? t('busyElsewhere') : t('send');
  ui.sendButton.setAttribute('aria-label', ui.sendButton.title);
  ui.sendButton.disabled = sendDisabled();
  const p = c && c.projectId ? projectById(c.projectId) : null;
  ui.input.placeholder = p ? t('placeholderProject', { project: p.name }) : t('placeholder');
  // The images waiting, then the documents; the images show how many the context holds with them.
  const pictures = state.pending.filter((a) => a.kind === 'image');
  const count = pictures.length ? maxContextImages - room : 0;
  fill(ui.pendingBox, pictures.map((a) => imageThumb(a, { list: pictures, onremove: () => removePending(a) })),
    state.pending.filter((a) => a.kind !== 'image').map((a) => attachmentChip(a, true, () => removePending(a))),
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
  if (busyHere()) return false;
  if (answeredElsewhere(state.current)) return true;
  return !ui.input.value.trim() && !state.pending.length;
}

function autoGrow() {
  ui.input.style.height = 'auto';
  // A composer hidden, behind the page of a project, has no height to measure: it takes it once shown (drawMain).
  if (ui.input.scrollHeight) ui.input.style.height = Math.min(ui.input.scrollHeight, 260) + 'px';
  ui.sendButton.disabled = sendDisabled();
}

/*
    Adds files to the composer of the discussion shown, the images within the places its context
    has left.  The files read while another discussion took its place join the draft of their own
    discussion, and leave the memory with an incognito or deleted one.
*/
async function addPending(files) {
  const c = state.current;
  let added = await readFiles(files, c ? c.role : 'general', imageRoom(c));
  const key = draftKey(c);
  const shown = state.current === c || (key !== '' && draftKey(state.current) === key);
  const draft = shown || !key ? null : drafts.get(key) || { text: storedDraft(key), files: [] };
  if (!shown && !draft) {
    images.drop(imageIdsOf(added));
    return;
  }
  // Two additions read together keep to the places as well.
  const waiting = shown ? state.pending : draft.files;
  const room = Math.max(0, maxContextImages - contextImages(shown ? state.current : c) - imageIdsOf(waiting).length);
  const pictures = added.filter((a) => a.kind === 'image');
  if (pictures.length > room) {
    const extra = pictures.slice(room);
    images.drop(extra.map((a) => a.id));
    added = added.filter((a) => !extra.includes(a));
    toast(imageLimitText(extra.map((a) => a.name), 0), true, 10000);
  }
  if (!shown) {
    draft.files = draft.files.concat(added);
    drafts.set(key, draft);
    return;
  }
  state.pending = state.pending.concat(added);
  renderComposerState();
  lockThread();
}

/*
    The answers this tab is writing, one per discussion at most: discussion id -> {c, abort, turn, answering,
    compaction}.  An answer goes on while another discussion is shown, written into its own discussion, which is
    saved as usual and which the other tabs follow; the discussion shown draws the answer it holds.  answering tells
    that the model has started to answer, and compaction holds a compression of the exchanges that ended since then,
    applied once the answer ends, or when a step of the agent no longer fits without it (fitContext).
*/
const generations = new Map();
// The discussions whose answer ended while another one was shown, marked in the side panel until they are opened.
const unread = new Set();

// The answer this tab writes in the discussion c, or null.
function generationOf(c) {
  return (c && generations.get(c.id)) || null;
}

// True when this tab writes an answer in the discussion shown.
function busyHere() {
  return !!generationOf(state.current);
}

// True when m is the answer this tab writes in the discussion c.
function liveMessage(c, m) {
  const g = generationOf(c);
  return !!g && g.turn === m;
}

// Stops the answer this tab writes in the discussion id, the one shown when id is not given.
function stopGeneration(id) {
  const g = generations.get(id == null ? state.current && state.current.id : id);
  if (g) g.abort.abort();
}

function stopAllGenerations() {
  for (const g of generations.values()) g.abort.abort();
}

async function onSend() {
  if (busyHere()) { stopGeneration(); return; }
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
  // The draft leaves with its question: the discussion, created by it or not, goes on with an empty composer.
  clearTimeout(draftTimer);
  forgetDraft(draftKey(state.current));
  await sendMessage(text, attachments);
}

/*
    Starts an answer in the discussion c.  While c is shown, the composer offers to stop it and the
    buttons that change the discussion wait; the side panel marks it, the other tabs know it, and the
    questions of c wait for its end.  Returns the controller that stops it.
*/
function beginTurn(c) {
  const g = { c, abort: new AbortController(), turn: null, answering: false, compaction: null };
  generations.set(c.id, g);
  // The question reads the beginning itself: a warm-up of it stops, and the engine keeps what it read.
  cancelWarmup(c.id);
  announceBusy(c, true);
  unread.delete(c.id);
  renderComposerState();
  lockThread();
  if (ui.sideScroll) renderSidebar();
  return g.abort;
}

// A new answer of the discussion c, drawn live from now on.
function newTurn(c) {
  const turn = { id: store.uid(), role: 'assistant', steps: [], created: Date.now(), model: roleModel(c.role) || '' };
  c.messages.push(turn);
  const g = generations.get(c.id);
  if (g) g.turn = turn;
  return turn;
}

// The day of the computer, "2026-11-01".
function today() {
  return new Date().toLocaleDateString('en-CA');
}

// A day as the model receives it, in English as the rest of the instructions: "2026-11-01 (Sunday)".
function dayText(day) {
  const at = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(day || ''));
  if (!at) return String(day || '');
  const weekday = new Intl.DateTimeFormat('en-US', { weekday: 'long', timeZone: 'UTC' })
    .format(new Date(Date.UTC(Number(at[1]), Number(at[2]) - 1, Number(at[3]))));
  return day + ' (' + weekday + ')';
}

// The last day the model of c was given: the one of the latest question that carries one, else the one of c.
function lastDay(c) {
  for (let i = c.messages.length - 1; i >= 0; --i) if (c.messages[i].day) return c.messages[i].day;
  return c.day || '';
}

async function sendMessage(text, attachments) {
  const c = state.current;
  if (generations.has(c.id) || answeredElsewhere(c)) return;
  const abort = beginTurn(c);
  const all = attachments || [];
  const m = {
    id: store.uid(), role: 'user', content: text, attachments: all.filter((a) => a.kind !== 'image'), created: Date.now()
  };
  const pictures = all.filter((a) => a.kind === 'image').map(imageRecord);
  if (pictures.length) m.images = pictures;
  // The update of the memory gives way to the question, and comes back once the user is quiet.
  if (memoryAbort) memoryAbort.abort();
  /*
      The system message of a discussion keeps the day it started, so that the prompt of each question keeps the
      same beginning, which the engine reads once; a question asked on a later day carries the day it is asked, at
      its own place in the prompt.
  */
  const day = today();
  if (!c.day) c.day = day;
  else if (day !== lastDay(c)) m.day = day;
  c.messages.push(m);
  // A new discussion starts with the memory as it is now.
  if (c.messages.length === 1 && !c.incognito && state.settings.memoryEnabled && state.settings.memory) {
    c.memory = state.settings.memory;
  }
  attachmentModes(c, m);
  /*
      The beginning of the first question stands for the title until the model writes one, and
      the text of that question stays with it, the only text the model names the discussion from
      (nameConversation).  The title of a discussion whose first question is edited or deleted
      stays.
  */
  if (c.messages.length === 1 && !c.title) {
    c.title = provisionalTitle(m);
    c.titleSource = 'question';
    c.titleQuestion = titleQuestionOf(m);
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
  if (generations.has(c.id) || answeredElsewhere(c)) return;
  const removed = [];
  while (c.messages.length && c.messages[c.messages.length - 1].role === 'assistant') removed.unshift(c.messages.pop());
  // The reading aloud of an answer that leaves ends with it: the new answer is drawn in the place of the old one.
  stopReadingOf(c, removed, c.messages.length);
  if (!c.messages.length) return;
  const abort = beginTurn(c);
  const turn = newTurn(c);
  // The new answer takes the place of the old one in the thread, its buttons waiting until it is written.
  if (state.current === c) renderMain();
  await runTurn(c, abort, turn);
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

/*
    The system message of a discussion; set, from toolSet(), tells which tools come with the question, and their
    instructions close the message, those of the agent, then those of the web.  It holds what stays the same from one
    question to the next, so that the engine reads it once: the day the discussion started, in English whatever the
    language of the interface, and the memory as it was then.
*/
function systemPrompt(c, contextTokens, set) {
  const agent = !!(set && set.agent);
  const parts = [];
  parts.push('Current date: ' + dayText(c.day || today()) + '.');
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
  if (set && set.web) parts.push(web.instructions(set.search));
  return parts.join('\n\n');
}

/*
    The text of a question for the model, with the day it was asked when it differs from the day the model knew
    before it; cuts (fitContext) shortens the oldest texts.
*/
function userContent(m, cuts) {
  const cut = (key, text) => {
    const kept = cuts && cuts.texts.get(key);
    return kept == null ? text : keepEnd(text, kept);
  };
  let s = m.day ? 'Current date: ' + dayText(m.day) + '.\n\n' : '';
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
    without media, and the images missing receive their names ("[Image: name]").  cuts (fitContext)
    shortens the oldest texts, a result of a tool keeping its beginning, leaves out the reasoning of
    the steps it gives a length of 0, and leaves out the cuts.steps oldest steps of the answer being
    written, the last message, which a line then names.
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
    const kept = (key) => (cuts ? cuts.texts.get(key) : null);
    let steps = m.steps || [];
    let left = '';
    if (cuts && cuts.steps && m === messages[messages.length - 1]) {
      left = '[The first ' + cuts.steps + ' steps of this answer, their tool calls and their results, left the '
        + 'context.]';
      steps = steps.slice(cuts.steps);
    }
    for (const s of steps) {
      const msg = { role: 'assistant', content: s.content || '' };
      if (left) {
        msg.content = left + (msg.content ? '\n\n' + msg.content : '');
        left = '';
      }
      if (s.reasoning && kept(s) !== 0) msg.reasoning_content = s.reasoning;
      const calls = (s.toolCalls || []).filter((tc) => tc.result != null);
      if (calls.length) {
        msg.tool_calls = calls.map((tc) => ({ id: tc.id, type: 'function', function: { name: tc.name, arguments: tc.arguments || '{}' } }));
      }
      if (!msg.content && !msg.tool_calls) continue;
      out.push(msg);
      for (const tc of calls) {
        const n = kept(tc);
        const content = n == null ? tc.result : keepStart(tc.result, n);
        out.push({ role: 'tool', tool_call_id: tc.id, name: tc.name, content });
      }
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

/*
    The discussion as it is now, for the work that ends in the background: the one this tab is answering, shown or
    not, else the one shown, else the one stored.  Null when it was deleted.
*/
async function currentVersion(c) {
  if (deleted.has(c.id)) return null;
  const g = generations.get(c.id);
  if (g) return g.c;
  if (state.current && state.current.id === c.id) return state.current;
  if (c.incognito) return c;
  return (await store.getConversation(c.id)) || null;
}

// Gives a compression of the exchanges to the discussion c, when the exchanges it compressed are still those of c.
function settleCompaction(c, r) {
  const last = c.messages[r.to - 1];
  if (!last || last.id !== r.last || (c.contextStart || 0) > r.to) return false;
  applyCompaction(c, r);
  return true;
}

/*
    The compressions of the exchanges in progress: discussion -> {done, abort, always}, done a promise that never
    rejects, abort the AbortController of its requests, always true for a summary written again after a deletion
    (rewriteSummary), which every question waits for.
*/
const compactions = new Map();

// Stops the compression of the exchanges of a discussion in progress, when the discussion leaves.
function cancelCompaction(id) {
  const job = compactions.get(id);
  if (job) job.abort.abort();
}

// Waits for promise, which never rejects, or throws an AbortError as soon as signal stops the answer.
function untilDone(promise, signal) {
  if (!signal) return promise;
  return new Promise((resolve, reject) => {
    const stop = () => reject(new DOMException('The answer was stopped.', 'AbortError'));
    if (signal.aborted) {
      stop();
      return;
    }
    signal.addEventListener('abort', stop, { once: true });
    promise.then((v) => {
      signal.removeEventListener('abort', stop);
      resolve(v);
    });
  });
}

/*
    The start of a discussion that brings the conversation built by build to target tokens at most, its tokens given
    by estimate: the first question after the current start for which it fits, the last question at most.  Returns
    -1 when no question can leave.
*/
function nextStart(c, build, estimate, target) {
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
    if (estimate(build()) <= target) {
      next = i;
      break;
    }
  }
  c.contextStart = saved;
  return next;
}

/*
    The tokens a conversation keeps once its oldest exchanges left: the system message and the tools (fixed), and
    keptShare of the room they leave to the exchanges.
*/
function keptTokens(fixed, room) {
  return fixed + keptShare * Math.max(0, room - fixed);
}

/*
    Fits the conversation in the context of the model.  When it no longer fits, the oldest
    exchanges leave in one step, until they take keptShare of the room the system message and
    the tools leave them, and the point where it now starts stays with the conversation: the
    following questions start from that same point, so that the engine reuses what it already
    read, and the next cut comes only after several questions.  The exchanges that leave are compressed
    into the summary of the discussion, which the model receives with the first question
    that stays (compact); the exchanges leave without it when the model cannot write it.
    When the last question alone does not fit, the oldest texts lose their beginnings,
    at the end of a sentence, and then the oldest steps of the answer being written leave
    (shortenOldest).  The system message and the last question always stay.  onCompact tells
    when a compression starts, and model names the model the tokens are counted for.  Returns
    {messages, trimmed, compacted, context, tokens}.

    The answer of the agent fits its context before each of its steps, with the cuts of its
    earlier steps (kept, which a compression empties), so that the beginning of the prompt stays
    from one step to the next; once its steps are cut, a quarter of the room of the exchanges is
    freed beyond what is needed, so that the next steps fit without another cut.

    A compression the background started after an earlier answer (compactLater) is waited for
    only when the conversation does not fit without it, and one that ended while the answer was
    written (generations) applies then; a summary being written again after a deletion
    (rewriteSummary) is waited for in every case.  signal ends these waits too.
*/
async function fitContext(c, build, toolDefs, reserve, signal, onCompact, model, kept) {
  let trimmed = false;
  let compacted = false;
  const waited = new Set();
  const wait = async (job) => {
    waited.add(job);
    if (onCompact) onCompact();
    await untilDone(job.done, signal);
  };
  const first = compactions.get(c.id);
  if (first && first.always) await wait(first);
  const cuts = kept || { summary: null, texts: new Map(), steps: 0 };
  // A compression moves the start of the conversation, and the texts are measured again whole.
  const restart = () => {
    cuts.summary = null;
    cuts.texts.clear();
    cuts.steps = 0;
  };
  // The steps of the answer being written that the request holds, the last message of c holding the one asked for.
  const answer = c.messages[c.messages.length - 1];
  let spare = answer && answer.role === 'assistant' && (answer.steps || []).length > 1 ? 0.25 : 0;
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
    // The conversation does not fit: the compression that ended or is in progress moves its start, and it is
    // measured again.
    const g = generations.get(c.id);
    if (g && g.compaction) {
      const done = g.compaction;
      g.compaction = null;
      if (settleCompaction(c, done)) {
        compacted = true;
        restart();
        continue;
      }
    }
    const job = compactions.get(c.id);
    if (job && !waited.has(job)) {
      await wait(job);
      continue;
    }
    // The tokens of a list of messages: its images by their tokens, its texts from their characters scaled to the
    // measure.  The system message and the tools are measured apart, as they stay.
    const estimate = tokenEstimate(list, r.tokens, counted);
    const scale = estimate.scale;
    const base = await api.count(counted, list.slice(0, 1), toolDefs).catch(() => null);
    if (signal && signal.aborted) throw new DOMException('The answer was stopped.', 'AbortError');
    const fixed = base && typeof base.tokens === 'number' ? base.tokens : estimate.of(list.slice(0, 1));
    const next = nextStart(c, () => build(cuts), estimate.of, keptTokens(fixed, room));
    if (next > (c.contextStart || 0)) {
      try {
        if (onCompact) onCompact();
        cancelWarmup(c.id);
        applyCompaction(c, await compact(c, c.contextStart || 0, next, room, scale, signal));
        compacted = true;
      } catch (e) {
        if (e.name === 'AbortError') throw e;
        c.contextStart = next;
        trimmed = true;
      }
      restart();
      continue;
    }
    // Nothing more can leave whole: the oldest texts lose their beginnings, with the spare room once.
    const excess = Math.ceil((r.tokens - room + spare * Math.max(0, room - fixed)) / scale) + 200;
    spare = 0;
    if (!shortenOldest(c, cuts, excess)) {
      return { messages: list, trimmed, compacted, context: r.context, tokens: r.tokens, overflow: true };
    }
    trimmed = true;
  }
  return { messages: build(cuts), trimmed, compacted, context: 0 };
}

/*
    Takes excess characters from the texts the model receives, the oldest first: the summary,
    the documents and the texts of the earlier questions with the results of the tools of their
    answers, then the reasoning and the results of the tools of the steps of the answer being
    written, the last message of c, but those of its last step, then the documents and the text
    of the question it answers, and the results of the last step of the answer last.  cuts holds
    the characters each of them keeps: a text keeps its end and a result its beginning, 200
    characters at least, and the reasoning of a step leaves whole.  When the texts before the
    question have nothing more to give, the oldest steps of the answer leave whole, the last step
    of the request staying (cuts.steps).  Returns false when nothing is left to take.
*/
function shortenOldest(c, cuts, excess) {
  const messages = c.messages.slice(c.contextStart || 0);
  const lastUser = messages.map((m) => m.role).lastIndexOf('user');
  const end = messages[messages.length - 1];
  const answer = end && end.role === 'assistant' && messages.length - 1 > lastUser ? end : null;
  // The request holds the steps of the answer before its last one, which is being asked for.
  const latest = answer && answer.steps.length > 1 ? answer.steps[answer.steps.length - 2] : null;
  const early = [];
  const reasoning = [];
  const results = [];
  const question = [];
  const last = [];
  if (c.summary && c.summary.text && (c.contextStart || 0) > 0) early.push({ key: 'summary', text: c.summary.text });
  messages.forEach((m, i) => {
    if (m.role === 'user') {
      const list = i === lastUser ? question : early;
      for (const a of m.attachments || []) if (a.mode !== 'search') list.push({ key: a.id, text: a.text || '' });
      list.push({ key: m.id, text: m.content || '' });
      return;
    }
    for (const s of m.steps || []) {
      if (m === answer && s.reasoning) reasoning.push({ key: s, text: s.reasoning, whole: true });
      for (const tc of s.toolCalls || []) {
        if (tc.result == null) continue;
        const list = m !== answer ? early : s === latest ? last : results;
        list.push({ key: tc, text: tc.result });
      }
    }
  });
  const keptOf = (key, text) => {
    const kept = key === 'summary' ? cuts.summary : cuts.texts.get(key);
    return kept == null ? text.length : kept;
  };
  const take = (list) => {
    for (const x of list) {
      if (excess <= 0) return;
      const length = keptOf(x.key, x.text);
      // A text keeps 200 characters at least: the last sentences of a question, the beginning of a result.
      const n = x.whole ? length : Math.min(excess, length - Math.min(length, 200));
      if (n <= 0) continue;
      if (x.key === 'summary') cuts.summary = length - n;
      else cuts.texts.set(x.key, length - n);
      excess -= n;
    }
  };
  take(early);
  take(reasoning);
  take(results);
  if (excess > 0 && answer) {
    const steps = answer.steps.length - 1;
    while (excess > 0 && cuts.steps < steps - 1) {
      const s = answer.steps[cuts.steps];
      let n = 60 + (s.content || '').length + (s.reasoning ? keptOf(s, s.reasoning) : 0);
      for (const tc of s.toolCalls || []) {
        if (tc.result != null) n += 80 + (tc.arguments || '').length + keptOf(tc, tc.result);
      }
      excess -= n;
      cuts.steps += 1;
    }
  }
  take(question);
  take(last);
  return excess <= 0;
}

/*
    The answer being written in the discussion shown is drawn again in the next frame, at most once
    every renderInterval milliseconds, and less often when a drawing takes long, so that a long
    answer leaves the page responsive.  An answer written in the background is drawn whole when its
    discussion is shown again.
*/
const renderInterval = 100;
let renderSpacing = renderInterval;
let renderTimer = 0;
let renderFrame = 0;
let renderedAt = 0;

function scheduleRender(c) {
  if (state.view !== 'chat' || state.current !== c || renderTimer || renderFrame) return;
  const draw = () => {
    renderFrame = 0;
    const g = generationOf(state.current);
    if (!g || !g.turn) return;
    const t0 = performance.now();
    renderedAt = t0;
    drawLive(g.c, g.turn);
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

/*
    Draws the answer m of c being written, while c is shown.  The positions of the scrolls are read
    before the drawing changes the page and written after it, so that the page is laid out once.
*/
function drawLive(c, m) {
  const el = answerElement(c, m);
  if (!el) return;
  const bubble = el.querySelector('.bubble');
  const thought = bubble.querySelector('.thinking.live .thought');
  const follow = !thought || thought.scrollHeight - thought.scrollTop - thought.clientHeight < 40;
  fillAssistant(bubble, m, true);
  const now = bubble.querySelector('.thinking.live .thought');
  if (now && follow) now.scrollTop = now.scrollHeight;
  scrollToEnd(false);
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
  return el && el.dataset.index === String(index) && el.dataset.id === String(m.id || '') ? el : null;
}

/*
    Answers the last question of c; abort, from beginTurn, stops the answer.  The answer is
    drawn while c is the discussion shown, goes on in the background while another one is
    shown, and is saved with c at its end; an answer that ends while c is not shown marks c in
    the side panel until it is opened.
*/
async function runTurn(c, abort, turn) {
  const g = generationOf(c);
  // The function, the reasoning and the agent of the answer stay those of its start.
  const role = c.role;
  const model = roleModel(role);
  turn.model = model || '';
  const signal = abort.signal;
  const caps = roleCapabilities(role);
  const reasoning = !!(c.reasoning && caps.reasoning);
  // The tools of the agent and of the web.  The calls of a step run while one of them is on, and a call of a tool
  // that the answer does not offer is refused.
  const set = toolSet(c, caps);
  const toolDefs = set.definitions;
  const useTools = !!toolDefs;
  const offered = new Set((toolDefs || []).map((d) => d.function.name));
  try {
    if (state.current === c && !answerElement(c, turn)) renderMain();
    else scheduleRender(c);
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
    // The cuts of the texts that the steps of the answer share (fitContext).
    const cuts = { summary: null, texts: new Map(), steps: 0 };
    for (let step = 0; step < maxAgentSteps; ++step) {
      const s = { reasoning: '', content: '', toolCalls: [] };
      turn.steps.push(s);
      const system = systemPrompt(c, caps.context || 8192, set);
      // The conversation, with the steps of this answer already done (the agent's).
      const build = (x) => apiMessages(Object.assign({}, c, {
        messages: c.messages.slice(0, -1).concat([Object.assign({}, turn, { steps: turn.steps.slice(0, -1) })])
      }), system, x, media);
      const fitted = await fitContext(c, build, toolDefs, state.settings.maxTokens || 2048, signal, () => {
        turn.compacting = true;
        scheduleRender(c);
      }, 'albireo-' + role, cuts);
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
      // A compression of the exchanges that ends from now on waits for the end of the answer, whose steps keep
      // the beginning of their prompt.
      if (g) g.answering = true;
      for await (const ev of api.streamChat(body, signal)) {
        if (ev.type === 'progress') {
          turn.progress = ev.data;
          scheduleRender(c);
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
        scheduleRender(c);
      }
      if (s.reasoningStart && !s.reasoningMs) s.reasoningMs = Date.now() - s.reasoningStart;
      delete s.reasoningStart;
      if (!(useTools && s.toolCalls.length && (finish === 'tool_calls' || finish === 'stop'))) break;
      // The tools of the step run, each timed, then the model continues with their results.
      for (const tc of s.toolCalls) {
        if (signal.aborted) throw new DOMException('The answer was stopped.', 'AbortError');
        tc.status = 'running';
        tc.started = Date.now();
        scheduleRender(c);
        let r;
        if (!offered.has(tc.name)) {
          r = { ok: false, text: 'The tool ' + tc.name + ' is not available in this answer.' };
        } else if (web.isTool(tc.name)) {
          // fetch_page reads the addresses of the discussion as it is when the call runs, the results of the
          // calls of this answer included.
          r = await web.run(tc.name, tc.arguments, {
            id: c.id, language: language(), signal, allowed: () => web.allowedAddresses(c)
          });
        } else {
          r = await tools.run(tc.name, tc.arguments, {
            documents: documentsOf(c),
            searchDocuments: (query, limit) => searchDocumentsOf(c, query, limit),
            searchDiscussions: async (q) => searchConversations(q, 6).filter((x) => x.conversation.id !== c.id)
              .map((x) => ({
                title: x.conversation.title, date: new Date(x.conversation.updated).toLocaleDateString(),
                excerpt: x.excerpt
              })),
            createFile: (name, content) => ({ name, content }),
            signal
          });
        }
        tc.ms = Date.now() - tc.started;
        delete tc.started;
        // A call the stop of the answer ended keeps what it returned, as stopped.
        tc.status = signal.aborted ? 'stopped' : 'done';
        tc.ok = r.ok;
        tc.result = r.text;
        if (r.file) tc.file = r.file;
        // The addresses a web tool adds to the discussion, which fetch_page may read from then on.
        if (r.web) tc.web = r.web;
        delete tc.index;
        scheduleRender(c);
      }
      if (step === maxAgentSteps - 1) turn.agentLimit = true;
    }
  } catch (e) {
    if (e.name === 'AbortError') turn.cancelled = true;
    else turn.error = errorText(e);
  } finally {
    if (state.current === c) cancelRender();
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
    if (g && generations.get(c.id) === g) generations.delete(c.id);
    // A compression of the exchanges that ended while the answer was written applies now.
    const settled = !!(g && g.compaction && settleCompaction(c, g.compaction));
    announceBusy(c, false);
    c.updated = Date.now();
    await persist(c);
    if (settled) warmBeginning(c);
    if (!(state.view === 'chat' && state.current && state.current.id === c.id) && !c.incognito && !deleted.has(c.id)) {
      unread.add(c.id);
    }
    if (state.current === c) renderMain();
    else lockThread();
    refreshViews();
  }
  if (deleted.has(c.id)) return;
  // A discussion still named after its first question receives a title from the model, whatever
  // the end of its answer, since the title follows the question.
  if (needsTitle(c)) nameConversation(c);
  if (!turn.error && !turn.cancelled) {
    compactLater(c);
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
    of the interface beside the words of the user in their own language.  A fact is a short sentence
    whose subject, the user, is left out.
*/
const memoryExamples = {
  en: ['Works as a nurse in Lyon.', 'Prefers short answers.'],
  fr: ['Travaille comme infirmière à Lyon.', 'Préfère les réponses courtes.'],
  de: ['Arbeitet als Krankenpflegerin in Lyon.', 'Bevorzugt kurze Antworten.'],
  it: ['Lavora come infermiera a Lione.', 'Preferisce risposte brevi.'],
  es: ['Trabaja como enfermera en Lyon.', 'Prefiere respuestas cortas.']
};

/*
    The new facts the questions teach, which join the memory, each followed by the words of the user
    that say it, which the update checks in the questions it read (groundedFacts).  The examples show
    a message with two facts, and a request, which teaches none.
*/
function memoryDeltaInstructions(words, sensitive) {
  const code = memoryExamples[language()] ? language() : 'en';
  const [a, b] = memoryExamples[code];
  const english = code === 'fr';
  const message = english ? 'I am a nurse in Lyon and I prefer short answers. What does triage mean?'
    : 'Je suis infirmière à Lyon et je préfère les réponses courtes. Que veut dire triage ?';
  const [qa, qb] = english ? ['I am a nurse in Lyon', 'I prefer short answers']
    : ['Je suis infirmière à Lyon', 'je préfère les réponses courtes'];
  const request = english ? 'Write a Python program that computes the integral of x squared between 0 and 1.'
    : 'Crée un programme Python pour calculer l\'intégrale de x au carré entre 0 et 1.';
  return 'You keep a short memory about the user for the future discussions of an assistant. From the new messages '
    + 'of the user, write the lasting facts about the user that the memory does not hold yet: profession, projects, '
    + 'preferences, habits of work, the way they want to be answered. A fact says who the user is, what the user '
    + 'does, uses or wants over time, as the user says it about themselves. A request of the user (write, create, '
    + 'explain, translate or compute something) and the topic of a question are not facts about the user: leave '
    + 'them out, whatever their words. Leave out what serves one discussion only, the content of documents, and '
    + 'what the memory already holds, even in other words. ' + sensitiveRule(sensitive)
    + 'Write each new fact on a line of its own: "- ", the fact as a short sentence in ' + memoryLanguage()
    + ' about the user, without naming the user, " | ", then the statement of the user that says it, copied '
    + 'exactly as the user wrote it, in its language, between double quotes; a question or a request of the user '
    + 'states no fact. A fact the user did not state in so many words is left out. The facts take at most ' + words
    + ' words in all. For example, for the message "' + message + '":\n'
    + '- ' + a + ' | "' + qa + '"\n'
    + '- ' + b + ' | "' + qb + '"\n'
    + 'For the message "' + request + '": NONE\n'
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
    The first words of the requests in the languages of the interface, folded (fold): the verbs of the frequent
    requests in the imperative, and the openings of a polite request.  A statement of the user that starts with one
    of them asks for a task, which says nothing lasting about the user.
*/
const requestOpeners = new Set(('write create make build generate compute calculate explain describe give show '
  + 'list summarize summarise translate convert fix correct rewrite draft compose draw find search tell help '
  + 'implement add remove check review solve analyze analyse compare prepare design develop optimize optimise '
  + 'refactor debug sort define recommend suggest propose estimate extract complete simplify improve update please '
  + 'can could would '
  + 'ecris ecrivez cree creez fais faites genere generez calcule calculez explique expliquez decris decrivez donne '
  + 'donnez montre montrez liste listez resume resumez traduis traduisez convertis convertissez corrige corrigez '
  + 'reecris reecrivez redige redigez dessine dessinez trouve trouvez cherche cherchez dis dites aide aidez '
  + 'implemente implementez ajoute ajoutez supprime supprimez verifie verifiez resous resolvez analysez comparez '
  + 'preparez concois concevez developpe developpez optimisez proposez suggere suggerez completez simplifie '
  + 'simplifiez ameliore ameliorez peux pourrais pouvez pourriez '
  + 'schreib schreibe erstelle erstellen mach mache generiere berechne berechnen erklar erklare erklaren beschreibe '
  + 'gib zeig zeige fasse ubersetze ubersetzen konvertiere korrigiere finde suche sag sage hilf implementiere fuge '
  + 'entferne prufe uberprufe analysiere vergleiche bereite programmiere entwickle optimiere schlage vereinfache '
  + 'verbessere kannst konntest konnen bitte '
  + 'scrivi scrivete crea calcola calcolate spiega spiegate descrivi dammi mostra mostrami elenca riassumi traduci '
  + 'converti correggi riscrivi disegna trova cerca dimmi aiuta aiutami implementa aggiungi rimuovi togli verifica '
  + 'controlla risolvi analizza confronta prepara progetta sviluppa ottimizza proponi suggerisci completa semplifica '
  + 'migliora puoi potresti '
  + 'escribe escriba haz haga genera calcula explica dame muestra enumera traduce traduzca convierte reescribe '
  + 'redacta dibuja encuentra busca dime ayuda ayudame agrega anade elimina quita comprueba revisa resuelve analiza '
  + 'compara disena desarrolla optimiza propon sugiere mejora puedes podrias puede podria').split(' '));

// The pronouns that follow an imperative after a hyphen in French ("écris-moi", "donne-le-moi").
const requestPronouns = new Set(['moi', 'toi', 'nous', 'vous', 'le', 'la', 'les', 'lui', 'leur', 'en', 'y']);

/*
    True when a clause asks for a task: its first word, after a polite opening, is a request opener, alone, before
    the pronouns of a French imperative, or before the pronoun an Italian or Spanish imperative takes at its end
    ("scrivimi", "explícame").
*/
function asksForTask(clause) {
  const words = fold(clause).replace(/[^\p{L}\s'-]+/gu, ' ').trim().split(/\s+/);
  let k = 0;
  if ((words[0] === 'por' && words[1] === 'favor') || (words[0] === 'per' && words[1] === 'favore')) k = 2;
  if (words[0] === 's\'il' && (words[1] === 'te' || words[1] === 'vous')) k = 3;
  const w = words[k] || '';
  if (requestOpeners.has(w)) return true;
  const parts = w.split('-');
  if (parts.length > 1 && requestOpeners.has(parts[0]) && parts.slice(1).every((x) => requestPronouns.has(x))) {
    return true;
  }
  const enclitic = /^(\p{L}{3,}?)(?:me|mi|ci|ti|te|lo|la|le|los|las|nos|gli)$/u.exec(w);
  return !!enclitic && requestOpeners.has(enclitic[1]);
}

/*
    Whether the words of a quote, normalized, appear in the questions read, normalized too, with 3
    words or 12 characters at least within a part of a sentence that asks nothing.  The parts end at
    the marks that end a sentence, at colons and at semicolons, and a part that ends with a question
    mark supports no fact, as in "One line: what is RAII?".  A quote whose clause starts with a request
    ("Write a program that...") supports no fact either (asksForTask).
*/
function statedIn(quote, where) {
  const at = quote ? where.indexOf(quote) : -1;
  if (at < 0) return false;
  let clause = at;
  while (clause > 0 && !/[.!?:;,\u2026]/.test(where[clause - 1])) --clause;
  if (asksForTask(where.slice(clause, at + quote.length))) return false;
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
    left out.  The facts keep no quote, and become sentences.  Returns {facts, proposed, discarded}.
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
    facts.push(...memorySentences(sentenceOf(m[1]), sensitive));
  }
  return { facts, proposed, discarded };
}

// The memory rewritten within its size, when the new facts made it outgrow it.
function memoryCompactInstructions(words, sensitive) {
  return 'You keep a short memory about the user for the future discussions of an assistant. Rewrite it in at most '
    + words + ' words, as short sentences one after the other in a single paragraph, without a list, a bullet or a '
    + 'line break: merge the sentences that say the same thing, keep the most lasting facts about the user, the '
    + 'newest last, and leave out the requests and the tasks of past discussions. ' + sensitiveRule(sensitive)
    + 'Write every sentence in ' + memoryLanguage() + ', the sentences written in another language included. '
    + 'Answer with the paragraph alone.';
}

// The memory written again in the language of the interface, each of its facts kept.
function memoryTranslateInstructions() {
  return 'You keep a short memory about the user for the future discussions of an assistant. Rewrite the memory in '
    + memoryLanguage() + ': translate the sentences written in another language, keep the sentences already written '
    + 'in ' + memoryLanguage() + ' as they are, and keep every fact and their order, as short sentences one after '
    + 'the other in a single paragraph, without a list or a line break. Answer with the paragraph alone.';
}

// True when a sentence of a memory is written in another language than the one of the interface.
function memoryForeign(text) {
  const target = language();
  return memorySentences(text, true).some((x) => {
    const found = textLanguage(x, true);
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

// A fact as a sentence of the memory: without the marks of a list or quotes around it, its first letter in capital,
// with its final point.
function sentenceOf(text) {
  let x = String(text || '').replace(/\s+/g, ' ').trim().replace(/^[-*\u2022]\s*/, '')
    .replace(/^["'\u00ab\u201c\u201e]+|["'\u00bb\u201d]+$/g, '').trim();
  if (!x) return '';
  x = x.charAt(0).toLocaleUpperCase() + x.slice(1);
  return /[.!?\u2026]$/.test(x) ? x : x + '.';
}

/*
    The sentences of a memory: the lines of the notes written as a list by an earlier version, each one a sentence,
    or the sentences of a paragraph, without the secrets unless the settings keep sensitive information.
*/
function memorySentences(text, sensitive) {
  const out = [];
  for (const line of String(text || '').split('\n').map((l) => l.trim()).filter(Boolean)) {
    if (/^[-*\u2022]\s*\S/.test(line)) out.push(sentenceOf(line));
    else for (const x of line.split(/(?<=[.!?\u2026])\s+(?=\S)/)) out.push(sentenceOf(x));
  }
  return out.filter((x) => x && (sensitive || !secretPatterns.some((re) => re.test(x))));
}

const wordCount = (lines) => lines.join(' ').split(/\s+/).filter(Boolean).length;

// The memory as it is kept: its sentences one after the other, within words words, the oldest leaving first.
function memoryParagraph(sentences, words) {
  let list = sentences.slice();
  while (list.length > 1 && wordCount(list) > words) list = list.slice(1);
  return list.join(' ');
}

// True when a memory holds notes written as a list, which become sentences (memorySentences).
function listMemory(text) {
  return /^\s*[-*\u2022]\s*\S/m.test(String(text || ''));
}

// The words of a sentence, for the comparison of two facts: their first five letters, which
// bring the forms of a word together ("courtes", "court").
function factWords(line) {
  return new Set(line.toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '').replace(/[^\p{L}\p{N}]+/gu, ' ')
    .split(' ').filter((w) => w.length > 2).map((w) => w.slice(0, 5)));
}

// True when two sentences say nearly the same thing: most of their words are shared.
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
    oldest sentences when the rewriting fails.  The memory is kept as short sentences one after
    the other.  A notification says how many facts joined.
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
  // The answers in progress come first.
  if (generations.size) {
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
    // The notes written as a list by an earlier version become sentences, and are saved as such.
    let lines = memorySentences(current, true);
    let rewritten = false;
    /*
        The facts written in another language are rewritten in the language of the interface
        first, so that the new facts are compared with them in one language.  A rewriting that
        lost facts is left out, and the next update tries again.
    */
    if (lines.length && (memoryForeign(current) || (!!stored.language && stored.language !== target))) {
      const r = await backgroundText(memoryTranslateInstructions(),
        '<memory>\n' + lines.join(' ') + '\n</memory>\n\nRewrite every sentence in ' + memoryLanguage() + '.',
        Math.ceil(wordCount(lines) * 2.5) + 32, signal).catch((e) => {
        if (e.name === 'AbortError') throw e;
        return null;
      });
      if (await changed()) return;
      const translated = r ? memorySentences(r.text, true) : [];
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
        '<memory>\n' + (lines.join(' ') || '(empty)') + '\n</memory>\n\n<messages>\n' + read
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
          '<memory>\n' + lines.join(' ') + '\n</memory>', Math.ceil(words * 2.2) + 16, signal).catch(() => null);
        if (signal.aborted) return;
        if (await changed()) return;
        const compacted = compact ? memorySentences(compact.text, sensitive) : [];
        if (compacted.length && wordCount(compacted) <= words * 1.2) lines = compacted;
      }
    }
    // The memory the user wrote may take twice its size, which the facts the model adds bring back to it.
    if (added || rewritten || listMemory(current)) {
      await saveSetting('memory', memoryParagraph(lines, added ? words : words * 2));
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

/*
    An error in the language of the interface: an error of the service by its code, an error of the reading of a
    document (documents.js) by its message, else its message as it is.
*/
function errorText(e) {
  const known = { model_not_found: 'errModelNotFound', context_length_exceeded: 'errContextLength', unloaded: 'errUnloaded',
    starting: 'errStarting', model_loading: 'modelLoading', speech_unavailable: 'errSpeechUnavailable',
    speech_failed: 'errSpeechFailed', too_many_images: 'errTooManyImages', invalid_image_format: 'errImageFormat',
    engine_busy: 'errEngineBusy', busy: 'errBusy' };
  if (e && known[e.code]) return t(known[e.code]);
  const reading = {
    'zip-invalid': 'errZipInvalid', 'zip-compression': 'errZipCompression', 'xml-invalid': 'errXmlInvalid'
  };
  if (e && Object.prototype.hasOwnProperty.call(reading, e.message)) return t(reading[e.message]);
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

// The text the model names a discussion from: its first question, with the names of its documents and images.
function titleQuestionOf(m) {
  const names = (m.attachments || []).concat(m.images || []).map((a) => a.name);
  return keepStart((m.content || '') + (names.length ? '\n(' + names.join(', ') + ')' : ''), 1500);
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
    its model loads, is missing or is busy with images, while the bodies it reads at once hold too
    many values, or while the service does not answer.  The next attempt waits at least the
    seconds the Retry-After header of the answer gives (e.retryAfter, api.js).
*/
function retryable(e) {
  if (e instanceof TypeError) return true;
  return ['starting', 'model_loading', 'unloaded', 'model_not_found', 'engine_busy', 'busy'].includes(e && e.code)
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
    times; it does not depend on the discussion shown, nor on the questions that follow: the text
    of the first question is the one kept with the provisional title (titleQuestion), which an
    edition or a deletion of the question leaves as it was, else the first question of a
    discussion saved before it was kept.  The title shows at once in the side panel, the top bar
    and the other tabs.
*/
async function nameConversation(c) {
  if (naming.has(c.id)) return;
  naming.add(c.id);
  try {
    await exclusive('albireo-title-' + c.id, async () => {
      // The seconds the last refusal of the service asked to wait (Retry-After), which the next pause lasts at least.
      let asked = 0;
      for (let attempt = 0; attempt <= titleRetries.length; ++attempt) {
        if (attempt) await new Promise((r) => setTimeout(r, Math.max(titleRetries[attempt - 1], asked) * 1000));
        asked = 0;
        const now = await currentVersion(c).catch(() => null);
        if (!needsTitle(now)) return;
        if (!serviceReady()) continue;
        const question = typeof now.titleQuestion === 'string' && now.titleQuestion.trim() ? now.titleQuestion
          : titleQuestionOf(now.messages.find((m) => m.role === 'user'));
        let title = '';
        try {
          title = await askTitle(question);
        } catch (e) {
          if (!retryable(e)) return;
          asked = e.retryAfter || 0;
          continue;
        }
        // Without a title, the beginning of the question stays, and a later answer tries again.
        const latest = await currentVersion(c).catch(() => null);
        if (!title || !needsTitle(latest)) return;
        latest.title = title;
        latest.titleSource = 'model';
        delete latest.titleQuestion;
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
    the beginning of their question as title, the most recent first, one after the other.  A
    discussion whose first answer did not end is named too: the title follows the question only.
*/
let namingPending = false;

async function namePending() {
  if (namingPending || state.incognito) return;
  namingPending = true;
  try {
    const waiting = state.conversations.filter((c) => needsTitle(c));
    for (const c of waiting.slice(0, 5)) {
      if (!serviceReady()) break;
      await nameConversation(c);
    }
  } finally {
    namingPending = false;
  }
}

/*
    Compresses the oldest exchanges of a discussion in the background once they take compactShare of the room the
    system message and the tools leave them, as the next question will receive them: after an answer, and when the
    discussion is opened.  The job takes its place in compactions at once, with its own AbortController, which the
    deletion of the discussion uses (cancelCompaction); a question that comes before the summary is written waits
    for it only when it does not fit without it (fitContext).  One tab compresses a discussion at a time.
*/
function compactLater(c) {
  if (!c || !c.messages.length || compactions.has(c.id) || !roleModel(c.role)) return;
  const abort = new AbortController();
  const job = { done: null, abort, always: false };
  const work = () => exclusive('albireo-compact-' + c.id, () => compactInBackground(c, abort.signal));
  job.done = Promise.resolve().then(work).catch(() => {}).finally(() => {
    if (compactions.get(c.id) === job) compactions.delete(c.id);
  });
  compactions.set(c.id, job);
}

/*
    The work of compactLater, on the messages of c as they are when it starts.  The service measures the text of the
    conversation, and the system message with the tools apart; the images count by their tokens (imageTokens).  The
    exchanges that leave are those that bring the conversation back to keptTokens.  The summary goes to the
    discussion as it is once written, when the exchanges it compressed are still there; an answer being written in
    the discussion keeps the beginning of its prompt, and the summary applies at its end.
*/
async function compactInBackground(c, signal) {
  const view = Object.assign({}, c, { messages: c.messages.slice() });
  const role = view.role;
  const model = 'albireo-' + role;
  const caps = roleCapabilities(role);
  const set = toolSet(view, caps);
  const toolDefs = set.definitions;
  const system = systemPrompt(view, caps.context || 8192, set);
  const text = apiMessages(view, system, null, { read: false, url: () => null });
  const [all, base] = await Promise.all([api.count(model, text, toolDefs),
    api.count(model, text.slice(0, 1), toolDefs).catch(() => null)]);
  if (signal.aborted || !all || typeof all.tokens !== 'number' || !all.context) return;
  const room = all.context - Math.min(state.settings.maxTokens || 2048, Math.floor(all.context / 2));
  const estimate = tokenEstimate(text, all.tokens, model);
  const media = estimateMedia(role);
  const build = () => apiMessages(view, system, null, media);
  const fixed = base && typeof base.tokens === 'number' ? base.tokens : estimate.of(text.slice(0, 1));
  if (estimate.of(build()) - fixed < compactShare * Math.max(0, room - fixed)) return;
  // The questions after the start, the last one included: the next question comes after it.
  view.messages.push({ id: 'next', role: 'user', content: '' });
  const next = nextStart(view, build, estimate.of, keptTokens(fixed, room));
  view.messages.pop();
  if (next <= (view.contextStart || 0)) return;
  cancelWarmup(c.id);
  const r = await compact(view, view.contextStart || 0, next, room, estimate.scale, signal);
  const now = await currentVersion(c);
  if (!now) return;
  const g = generations.get(now.id);
  if (g && g.answering) {
    g.compaction = r;
    return;
  }
  if (!settleCompaction(now, r)) return;
  await persist(now);
  // The images of the exchanges summarized left the context: their places go to new ones.
  if (state.current === now) {
    markOutside();
    renderComposerState();
    lockThread();
  }
  warmBeginning(now);
}

/*
    The warm-ups of the beginning of a discussion after a compression, one per discussion at most: discussion id ->
    AbortController.  A question sent in the discussion, its deletion and a new compression of it stop the warm-up.
*/
const warmups = new Map();

function cancelWarmup(id) {
  const abort = warmups.get(id);
  if (!abort) return;
  warmups.delete(id);
  abort.abort();
}

/*
    Reads ahead, in the background, the beginning of the prompt of the next question of c once a summary changed it:
    the system message, the summary and the exchanges kept, as the code of a question builds them (apiMessages),
    followed by a question of one character, so that the engine renders the last answer as a turn of the history, as
    the next question will.  The request goes to the background model, which the service serves behind the answers,
    without reasoning and for one token; the engine keeps the keys and values it read, which the next question takes
    over, and computes then its own tokens only.  The engine reads the history of each answer ahead by itself; the
    page asks it only after a summary, which changes the beginning.  The request is sent when the background model is
    the model of the function of c, whose keys and values the next question reads, by the tab that applied the summary,
    while no answer is written in c.  Its failure is told in the console only.
*/
async function warmBeginning(c) {
  if (!c || deleted.has(c.id) || generations.has(c.id) || !c.messages.length || !serviceReady()) return;
  const model = roleModel(c.role);
  const s = state.status;
  if (!model || ((s && s.background) || roleModel('general')) !== model) return;
  cancelWarmup(c.id);
  const abort = new AbortController();
  warmups.set(c.id, abort);
  let limit = null;
  try {
    const caps = roleCapabilities(c.role);
    const set = toolSet(c, caps);
    const media = await requestMedia(c, c.role);
    if (abort.signal.aborted) return;
    const next = Object.assign({}, c, { messages: c.messages.concat([{ id: 'next', role: 'user', content: '.' }]) });
    const body = { model: backgroundModel, reasoning: false, max_tokens: 1, stream: false,
      messages: apiMessages(next, systemPrompt(c, caps.context || 8192, set), null, media) };
    if (set.definitions) body.tools = set.definitions;
    limit = api.limitedSignal(abort.signal, backgroundTimeout);
    await api.postJSON('/v1/chat/completions', body, limit.signal);
  } catch (e) {
    if (!abort.signal.aborted) console.info('Albireo: the beginning of a discussion was not read ahead', e);
  } finally {
    if (limit) limit.done();
    if (warmups.get(c.id) === abort) warmups.delete(c.id);
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
    // What the user writes stays, sensitive or not, within twice the size of the memory, as sentences one after
    // the other: a line becomes a sentence.
    await saveSetting('memory', memoryParagraph(memorySentences(text.value, true), words * 2));
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
      for (const id of r.ids) deleted.delete(id);
      broadcast({ type: 'lists' });
      await syncLists();
      toast(t('imported', { n: r.conversations, p: r.projects }));
      // The documents of the projects and the searched documents of the discussions are indexed, since an export
      // carries no passages; the images of the projects are described, and the discussions left waiting are named.
      indexProjects();
      for (const id of r.ids) {
        const c = state.conversations.find((x) => x.id === id);
        if (c) indexAttachments(c).catch(() => {});
      }
      describePending();
      namePending();
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
            download('albireo-' + new Date().toISOString().slice(0, 10) + '.json', await store.exportAll(),
              'application/json');
          } catch (e) {
            // The export reads the database of the browser only: the error is that of the browser, as it is.
            toast(t('exportFailed', { reason: (e && e.message) || String(e) }), true);
          }
        }
      }),
      h('button', { class: 'button', type: 'button', html: icon('upload') + '<span>' + escapeHtml(t('importAll')) + '</span>', onclick: () => input.click() }),
      input,
      h('button', {
        class: 'button danger', type: 'button', html: icon('trash') + '<span>' + escapeHtml(t('clearAll')) + '</span>',
        onclick: async () => {
          if (!await confirmDialog(t('clearAll'), t('clearAllText'))) return;
          // The answers in progress, shown or written in the background, and the work of the background
          // model stop, the stored discussions are not saved again, and the indexing in progress writes
          // no passage.  A new discussion, not yet stored, stays with its draft.
          for (const c of state.conversations) deleted.add(c.id);
          if (state.current && state.current.messages.length) deleted.add(state.current.id);
          for (const g of generations.values()) deleted.add(g.c.id);
          stopAllGenerations();
          unread.clear();
          cancelIndex();
          cancelBackground();
          rag.removeAll();
          images.removeAll(waitingImages());
          forgetStoredDrafts();
          for (const id of deleted) web.forgetPages(id);
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
  fill(el, h('p', null, h('b', null, t('storageState')), ' ', parts.join(' · ')),
    kept === true ? null : h('p', { class: 'help storage-hint' }, h('span', { html: icon('info') }), t('storageHint')));
}

function memoryBar(stats) {
  const v = stats && stats.memory;
  if (!v || !v.vram_budget) return null;
  const total = Math.max(v.vram_budget, 1);
  const pct = (x) => Math.min(100, 100 * x / total) + '%';
  return h('div', null, h('div', { class: 'mem-bar' },
    h('div', { style: { width: pct(v.device_bytes || 0), background: 'var(--blue)' }, title: t('vramResident') }),
    h('div', { style: { width: pct(v.scratch_bytes || 0), background: 'var(--gold)' }, title: t('vramWorkspace') })),
  h('div', { class: 'legend' },
    h('span', { style: { '--c': 'var(--blue)' } }, t('vramResident') + ' ' + formatSize(v.device_bytes || 0)),
    h('span', { style: { '--c': 'var(--gold)' } }, t('vramWorkspace') + ' ' + formatSize(v.scratch_bytes || 0)),
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
  // The provider of the web search, while the service offers one: Brave asks the projects that use its credit to
  // name its API there.
  const provider = searchProvider();
  if (provider) row(t('tool_web_search'), provider);
  if (s.usage) row(t('requests'), formatNumber(s.usage.requests) + ' · ' + t('tokensIn') + ' ' + formatNumber(s.usage.prompt_tokens) +
    ' · ' + t('tokensOut') + ' ' + formatNumber(s.usage.completion_tokens));
  body.append(art, h('p', null, t('aboutText')), kv);
  const mem = memoryBar(s.engine && s.engine.stats);
  if (mem) body.append(h('h4', null, t('v2ram')), mem);
  try {
    const u = await api.usage();
    const byDay = new Map((u.days || []).map((d) => [d.day, d]));
    if (byDay.size) {
      /*
          The last thirty days, the days without use included, each a bar of the tokens generated over the tokens
          read, whose tooltip gives its date and its figures in the language of the interface.  The days of the
          service are those of UTC.
      */
      const dateOf = (day) => new Intl.DateTimeFormat(language(), { dateStyle: 'medium', timeZone: 'UTC' })
        .format(new Date(day + 'T00:00:00Z'));
      const days = [];
      for (let k = 29; k >= 0; --k) {
        const day = new Date(Date.now() - k * 86400000).toISOString().slice(0, 10);
        const d = byDay.get(day) || {};
        days.push({ day, read: d.prompt_tokens || 0, generated: d.generated_tokens || 0 });
      }
      const max = Math.max(...days.map((d) => d.read + d.generated), 1);
      body.append(h('h4', null, t('usage')),
        h('div', { class: 'usage-bars', role: 'img', 'aria-label': t('usageChart') }, days.map((d) => h('div', {
          class: 'usage-day', style: { height: Math.max(2, 80 * (d.read + d.generated) / max) + 'px' },
          title: t('usageDay',
            { date: dateOf(d.day), generated: formatNumber(d.generated), read: formatNumber(d.read) })
        }, h('div', { class: 'generated', style: { flexGrow: String(d.generated) } }),
        h('div', { class: 'read', style: { flexGrow: String(d.read) } })))),
        h('div', { class: 'usage-axis' }, h('span', null, dateOf(days[0].day)), h('span', null, dateOf(days[29].day))),
        h('div', { class: 'legend' }, h('span', { style: { '--c': 'var(--blue)' } }, t('usageGenerated')),
          h('span', { style: { '--c': 'var(--gold)' } }, t('usageRead'))));
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
  'svox-pico.txt': ['pico', 'Apache License 2.0'],
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
      // The reading aloud ends with the discussion it belongs to, as route() ends it.
      speech.stopReading();
      leaveCurrent();
      showDiscussion(newConversation(null));
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
  ui.input.addEventListener('input', () => {
    autoGrow();
    saveDraft();
  });
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
  /*
      The web access of the discussion shown, off in every new discussion: the choice belongs to its discussion and
      never becomes the default of the next ones.
  */
  ui.webButton = h('button', {
    class: 'toggle web', type: 'button', title: t('webHint'), 'aria-label': t('webAccess'), 'aria-pressed': 'false',
    html: icon('globe') + '<span>' + escapeHtml(t('web')) + '</span>',
    onclick: async () => {
      const c = state.current;
      c.web = !c.web;
      renderComposerState();
      if (c.messages.length) await persist(c);
      refreshViews();
    }
  });
  // What the agent mode and the web access do, and the tools they give, from a button beside them.
  ui.agentHelp = h('button', {
    class: 'help-button', type: 'button', title: t('agentHelpTitle'), 'aria-label': t('agentHelpTitle'),
    html: icon('info'), onclick: (e) => agentHelp(e.currentTarget)
  });
  ui.sendButton = h('button', { class: 'send', type: 'button', onclick: onSend });
  ui.pendingBox = h('div', { class: 'composer-attachments hidden' });
  ui.attachButton = iconButton('clip', t('attach'), () => fileInput.click());
  const composer = h('div', { class: 'composer' }, ui.pendingBox, ui.input, h('div', { class: 'row' },
    ui.attachButton, fileInput, ui.reasonButton, ui.agentButton, ui.webButton, ui.agentHelp,
    h('span', { class: 'grow' }), ui.sendButton));
  for (const ev of ['dragenter', 'dragover']) composer.addEventListener(ev, (e) => { e.preventDefault(); composer.classList.add('drag'); });
  for (const ev of ['dragleave', 'drop']) composer.addEventListener(ev, () => composer.classList.remove('drag'));
  composer.addEventListener('drop', (e) => {
    e.preventDefault();
    if (e.dataTransfer && e.dataTransfer.files.length) addPending(Array.from(e.dataTransfer.files));
  });
  ui.hint = h('div', { class: 'hint' }, hintText());
  ui.composerWrap = h('div', { class: 'composer-wrap' }, composer, ui.hint);
  ui.toasts = h('div', { class: 'toast-zone' });
  // The buttons that go to the start and to the end of the thread, over its bottom right corner.
  ui.jumpStart = iconButton('toStart', t('scrollStart'), () => jumpTo(false), 'icon-button jump-start');
  ui.jumpEnd = iconButton('toEnd', t('scrollEnd'), () => jumpTo(true), 'icon-button jump-end');
  ui.jump = h('div', { class: 'jump-buttons' }, ui.jumpStart, ui.jumpEnd);
  const later = () => {
    clearTimeout(ui.jumpTimer);
    ui.jumpTimer = setTimeout(hideJump, jumpDelay);
  };
  ui.jump.addEventListener('mouseleave', later);
  ui.jump.addEventListener('focusout', later);
  ui.stick = true;
  ui.ownTop = 0;
  ui.content.addEventListener('scroll', threadScrolled, { passive: true });
  ui.main = h('main', { class: 'main' }, ui.topbar, ui.banner, ui.content, ui.jump, ui.composerWrap);
  app.append(ui.sidebar, ui.main, ui.toasts);
  ui.composerHeight = -1;
  watchComposer();
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
  let blocked = false;
  try {
    await store.open();
    state.settings = Object.assign({}, defaults, await store.getSettings());
    // The notes of the memory written as a list by an earlier version become sentences one after the other.
    if (listMemory(state.settings.memory)) {
      state.settings.memory = memoryParagraph(memorySentences(state.settings.memory, true), memoryWords * 2);
      await store.setSetting('memory', state.settings.memory);
    }
    // The browser is asked to keep the database, without waiting for its answer.
    requestPersistence();
  } catch (e) {
    // Without IndexedDB (a private window of some browsers), or while a tab of an earlier version
    // keeps it open, the application keeps its data in memory, as in an incognito discussion.
    console.warn('Albireo: the database of the browser is unavailable', e);
    blocked = !!e && e.name === 'BlockedError';
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
    if (!state.incognito) pruneDrafts();
  }
  buildLayout();
  if (blocked) toast(t('dbBlocked'), true, 30000);
  window.addEventListener('hashchange', route);
  // The text being typed stays when the page is left, reloaded or hidden.
  window.addEventListener('pagehide', () => saveDraft(true));
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden') saveDraft(true);
  });
  await askSpeechVoices();
  await route();
  /*
      The first reading of the status, once the service is ready now or later (pollStatus), plans the
      questions an earlier visit left to the memory, the titles it left to write and the descriptions
      of the images.
  */
  await refreshStatus();
  setInterval(refreshStatus, 4000);
  // The documents of the projects without passages are indexed, by their words without an embedding
  // model; with one, the status has already started their indexing.
  if (!rag.embeddingModel()) indexProjects();
  // A tab that closes gives back the lease of the memory and tells that it no longer answers.
  window.addEventListener('pagehide', () => {
    speech.stopReading();
    if (memoryRunning) releaseMemoryLease().catch(() => {});
    for (const g of generations.values()) {
      if (!g.c.incognito) broadcast({ type: 'busy', id: g.c.id, tab: tabId, busy: false });
    }
  });
}

init();
