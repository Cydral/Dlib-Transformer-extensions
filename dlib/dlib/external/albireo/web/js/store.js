// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

/*
    The data of the application, kept in the IndexedDB database of this browser for the
    origin of the local server: the discussions with their messages, the projects with
    their instructions and documents, the passages of the documents with their vectors for
    the document search, the images of the discussions and of the projects, and the
    preferences.  Nothing is sent anywhere; an incognito discussion never reaches this module.
*/

const DB_NAME = 'albireo';
const DB_VERSION = 3;
let db = null;

function request(r) {
  return new Promise((resolve, reject) => {
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
  });
}

// The opening in progress, which the calls made meanwhile share.
let opening = null;

/*
    The error of an opening that a tab of an earlier version blocked.  The browser queues every later opening behind
    the blocked one, so that the calls of this visit fail at once with this error instead of waiting.
*/
let blocked = null;

export function open() {
  if (db) return Promise.resolve(db);
  if (blocked) return Promise.reject(blocked);
  if (!opening) {
    opening = connect().then((d) => {
      /*
          When a newer version of the application upgrades the database in another tab, this tab
          lets it go and opens it again at its next access.
      */
      d.onversionchange = () => {
        d.close();
        if (db === d) db = null;
      };
      db = d;
      return d;
    }).finally(() => { opening = null; });
  }
  return opening;
}

async function connect() {
  const r = indexedDB.open(DB_NAME, DB_VERSION);
  r.onupgradeneeded = () => {
    const d = r.result;
    if (!d.objectStoreNames.contains('conversations')) {
      const s = d.createObjectStore('conversations', { keyPath: 'id' });
      s.createIndex('updated', 'updated');
      s.createIndex('project', 'projectId');
    }
    if (!d.objectStoreNames.contains('projects')) d.createObjectStore('projects', { keyPath: 'id' });
    if (!d.objectStoreNames.contains('settings')) d.createObjectStore('settings', { keyPath: 'key' });
    // The passages of the documents: {id, owner (project or discussion), doc, ordinal, text,
    // where, model, vector}.
    if (!d.objectStoreNames.contains('chunks')) {
      const c = d.createObjectStore('chunks', { keyPath: 'id' });
      c.createIndex('owner', 'owner');
      c.createIndex('doc', 'doc');
    }
    // The images of the discussions and of the projects (images.js): {id, owner, name, type, size, width,
    // height, added, blob, thumb}, the image and its thumbnail as Blobs.
    if (!d.objectStoreNames.contains('images')) {
      const i = d.createObjectStore('images', { keyPath: 'id' });
      i.createIndex('owner', 'owner');
    }
  };
  /*
      A tab of an earlier version that keeps the database open blocks its upgrade: the opening then fails with a
      BlockedError, and so do the next calls of this visit (open).  The connection the browser opens once that tab
      closed, with the database upgraded, is closed at once: the next visit reads the database.
  */
  return new Promise((resolve, reject) => {
    r.onsuccess = () => {
      if (blocked) r.result.close();
      else resolve(r.result);
    };
    r.onerror = () => reject(r.error);
    r.onblocked = () => {
      const e = new Error('another tab keeps the database open in an earlier version');
      e.name = 'BlockedError';
      blocked = e;
      reject(e);
    };
  });
}

function store(name, mode) {
  return db.transaction(name, mode).objectStore(name);
}

// The end of a transaction.
function completed(tx) {
  return new Promise((resolve, reject) => {
    tx.oncomplete = resolve;
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error);
  });
}

export function uid() {
  const a = new Uint8Array(12);
  crypto.getRandomValues(a);
  return Array.from(a, (b) => b.toString(16).padStart(2, '0')).join('');
}

export async function allConversations() {
  await open();
  const list = await request(store('conversations', 'readonly').getAll());
  list.sort((a, b) => b.updated - a.updated);
  return list;
}

export async function getConversation(id) {
  await open();
  return request(store('conversations', 'readonly').get(id));
}

export async function putConversation(c) {
  await open();
  return request(store('conversations', 'readwrite').put(c));
}

/*
    Changes stored discussions in place, in one transaction, so that the messages another tab
    saved meanwhile stay: fn(record) changes the record read from the database and returns
    true to save it.  The discussion of id only, when id is given.
*/
export async function updateConversations(fn, id) {
  await open();
  const tx = db.transaction('conversations', 'readwrite');
  const r = tx.objectStore('conversations').openCursor(id == null ? null : IDBKeyRange.only(id));
  r.onsuccess = () => {
    const c = r.result;
    if (!c) return;
    if (fn(c.value)) c.update(c.value);
    c.continue();
  };
  await completed(tx);
}

/*
    Deletes a discussion: its record first, in a transaction of its own, then its passages and its images.  The
    passages and the images that a tab closed meanwhile leaves are pruned at the next start (app.js, pruneIndex and
    pruneImages).
*/
export async function deleteConversation(id) {
  await open();
  const tx = db.transaction('conversations', 'readwrite');
  tx.objectStore('conversations').delete(id);
  await completed(tx);
  await deleteChunks('owner', id);
  await deleteImages(id);
}

export async function allProjects() {
  await open();
  const list = await request(store('projects', 'readonly').getAll());
  list.sort((a, b) => a.name.localeCompare(b.name));
  return list;
}

export async function getProject(id) {
  await open();
  return request(store('projects', 'readonly').get(id));
}

export async function putProject(p) {
  await open();
  return request(store('projects', 'readwrite').put(p));
}

export async function deleteProject(id) {
  await open();
  /*
      The discussions of the project stay, outside of any project.  They change in place, in one
      transaction with the deletion of the project, through the index of their projects.
  */
  const tx = db.transaction(['conversations', 'projects'], 'readwrite');
  const r = tx.objectStore('conversations').index('project').openCursor(IDBKeyRange.only(id));
  r.onsuccess = () => {
    const c = r.result;
    if (!c) return;
    c.value.projectId = null;
    c.update(c.value);
    c.continue();
  };
  tx.objectStore('projects').delete(id);
  await completed(tx);
  await deleteChunks('owner', id);
  await deleteImages(id);
}

/*
    Replaces the passages of the document docId with records, in one transaction: a tab closed meanwhile leaves
    either the former passages or the new ones.
*/
export async function replaceChunks(docId, records) {
  await open();
  const tx = db.transaction('chunks', 'readwrite');
  const s = tx.objectStore('chunks');
  const cursor = s.index('doc').openKeyCursor(IDBKeyRange.only(docId));
  cursor.onsuccess = () => {
    const c = cursor.result;
    if (c) {
      s.delete(c.primaryKey);
      c.continue();
      return;
    }
    // The former passages deleted, the new ones are written by the same transaction.
    for (const r of records) s.put(r);
  };
  await completed(tx);
}

export async function chunksOf(owner) {
  await open();
  return request(store('chunks', 'readonly').index('owner').getAll(owner));
}

// The passages of one document.
export async function chunksOfDocument(docId) {
  await open();
  return request(store('chunks', 'readonly').index('doc').getAll(docId));
}

/*
    The state of the documents of an owner, read passage by passage without keeping the passages: Map doc ->
    {passages, vectors, model}, vectors counting the passages that hold a vector of model.
*/
export async function chunkStatus(owner, model) {
  await open();
  const tx = db.transaction('chunks', 'readonly');
  const out = new Map();
  const r = tx.objectStore('chunks').index('owner').openCursor(IDBKeyRange.only(owner));
  r.onsuccess = () => {
    const c = r.result;
    if (!c) return;
    const v = c.value;
    const s = out.get(v.doc) || { passages: 0, vectors: 0, model: v.model };
    s.passages += 1;
    if (v.vector && model && v.model === model) s.vectors += 1;
    out.set(v.doc, s);
    c.continue();
  };
  await completed(tx);
  return out;
}

// The documents that have passages: one {owner, doc} per document, read from one passage of each.
export async function chunkDocuments() {
  await open();
  const tx = db.transaction('chunks', 'readonly');
  const out = [];
  const r = tx.objectStore('chunks').index('doc').openCursor(null, 'nextunique');
  r.onsuccess = () => {
    const c = r.result;
    if (!c) return;
    out.push({ owner: c.value.owner, doc: c.value.doc });
    c.continue();
  };
  await completed(tx);
  return out;
}

// Deletes the passages of an owner ('owner') or of a document ('doc').
export async function deleteChunks(index, key) {
  await open();
  const tx = db.transaction('chunks', 'readwrite');
  const s = tx.objectStore('chunks');
  const cursor = s.index(index).openKeyCursor(IDBKeyRange.only(key));
  cursor.onsuccess = () => {
    const c = cursor.result;
    if (!c) return;
    s.delete(c.primaryKey);
    c.continue();
  };
  await completed(tx);
}

// The images.
export async function putImages(records) {
  await open();
  if (!records.length) return;
  const tx = db.transaction('images', 'readwrite');
  const s = tx.objectStore('images');
  for (const r of records) s.put(r);
  await completed(tx);
}

export async function getImage(id) {
  await open();
  return request(store('images', 'readonly').get(id));
}

// The images of an owner, or of every owner: [{owner, id}], read from the index without the images.
export async function imageKeys(owner) {
  await open();
  const tx = db.transaction('images', 'readonly');
  const out = [];
  const r = tx.objectStore('images').index('owner').openKeyCursor(owner == null ? null : IDBKeyRange.only(owner));
  r.onsuccess = () => {
    const c = r.result;
    if (!c) return;
    out.push({ owner: c.key, id: c.primaryKey });
    c.continue();
  };
  await completed(tx);
  return out;
}

// Deletes the images of an owner.
export async function deleteImages(owner) {
  await open();
  const tx = db.transaction('images', 'readwrite');
  const s = tx.objectStore('images');
  const r = s.index('owner').openKeyCursor(IDBKeyRange.only(owner));
  r.onsuccess = () => {
    const c = r.result;
    if (!c) return;
    s.delete(c.primaryKey);
    c.continue();
  };
  await completed(tx);
}

export async function deleteImageIds(ids) {
  await open();
  if (!ids.length) return;
  const tx = db.transaction('images', 'readwrite');
  const s = tx.objectStore('images');
  for (const id of ids) s.delete(id);
  await completed(tx);
}

// The images stored: {count, bytes}, the bytes of the images and of their thumbnails.
export async function imageUsage() {
  await open();
  const tx = db.transaction('images', 'readonly');
  let count = 0, bytes = 0;
  const r = tx.objectStore('images').openCursor();
  r.onsuccess = () => {
    const c = r.result;
    if (!c) return;
    count += 1;
    for (const b of [c.value.blob, c.value.thumb]) if (b instanceof Blob) bytes += b.size;
    c.continue();
  };
  await completed(tx);
  return { count, bytes };
}

// A Blob as a data URL ("data:image/jpeg;base64,...").
export function dataUrlOf(blob) {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = () => resolve(typeof reader.result === 'string' ? reader.result : null);
    reader.onerror = () => resolve(null);
    reader.readAsDataURL(blob);
  });
}

// The Blob of a data URL of an image, decoded here, since the page reads no address outside its origin; null for
// another text.
function blobOfDataUrl(text) {
  const m = /^data:(image\/[a-z0-9.+-]+);base64,([A-Za-z0-9+/=]*)$/i.exec(String(text || ''));
  if (!m) return null;
  const bin = atob(m[2]);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; ++i) bytes[i] = bin.charCodeAt(i);
  return new Blob([bytes], { type: m[1].toLowerCase() });
}

export async function getSettings() {
  await open();
  const rows = await request(store('settings', 'readonly').getAll());
  const s = {};
  for (const r of rows) s[r.key] = r.value;
  return s;
}

export async function setSetting(key, value) {
  await open();
  return request(store('settings', 'readwrite').put({ key, value }));
}

/*
    Puts the settings of values in one transaction, unless unless(settings), given the settings as the transaction
    reads them, returns true, so that no other tab changes them between the check and the change.  Returns whether
    the settings were put.
*/
export async function setSettings(values, unless) {
  await open();
  const tx = db.transaction('settings', 'readwrite');
  const s = tx.objectStore('settings');
  let put = false;
  const r = s.getAll();
  r.onsuccess = () => {
    const now = {};
    for (const row of r.result) now[row.key] = row.value;
    if (unless && unless(now)) return;
    for (const key of Object.keys(values)) s.put({ key, value: values[key] });
    put = true;
  };
  await completed(tx);
  return put;
}

/*
    Takes the lease of key for owner until the time until, in one transaction, so that two tabs
    never both take it.  Returns true when the lease was free, expired or already held by owner.
*/
export async function claimLease(key, owner, until) {
  await open();
  const tx = db.transaction('settings', 'readwrite');
  const s = tx.objectStore('settings');
  let taken = false;
  const r = s.get(key);
  r.onsuccess = () => {
    const lease = r.result && r.result.value;
    if (lease && lease.owner !== owner && lease.until > Date.now()) return;
    s.put({ key, value: { owner, until } });
    taken = true;
  };
  await completed(tx);
  return taken;
}

// Gives the lease of key back, when owner holds it.
export async function releaseLease(key, owner) {
  await open();
  const tx = db.transaction('settings', 'readwrite');
  const s = tx.objectStore('settings');
  const r = s.get(key);
  r.onsuccess = () => {
    const lease = r.result && r.result.value;
    if (lease && lease.owner === owner) s.put({ key, value: null });
  };
  await completed(tx);
}

/*
    Everything stored, as a JSON file in a Blob built record by record, the images as data URLs, so that no single
    string holds the whole export.
*/
export async function exportAll() {
  await open();
  const parts = ['{"format":"albireo-export","version":1,"exported":' + Date.now() + ',"conversations":['];
  (await allConversations()).forEach((c, i) => parts.push((i ? ',' : '') + JSON.stringify(c)));
  parts.push('],"projects":[');
  (await allProjects()).forEach((p, i) => parts.push((i ? ',' : '') + JSON.stringify(p)));
  parts.push('],"settings":' + JSON.stringify(await getSettings()) + ',"images":[');
  const records = await request(store('images', 'readonly').getAll());
  let first = true;
  for (const r of records) {
    if (!(r.blob instanceof Blob)) continue;
    const { blob, thumb, ...rest } = r;
    const picture = Object.assign(rest, {
      data: await dataUrlOf(blob), thumbnail: thumb instanceof Blob ? await dataUrlOf(thumb) : null
    });
    parts.push((first ? '' : ',') + JSON.stringify(picture));
    first = false;
  }
  parts.push(']}');
  return new Blob(parts, { type: 'application/json' });
}

// The identifiers the application gives its records, which its addresses (#/c/..., #/p/...) take.
const recordId = /^[0-9a-f]{1,64}$/;

// The objects of a list, none for another value.
function objects(list) {
  return Array.isArray(list) ? list.filter((x) => x !== null && typeof x === 'object' && !Array.isArray(x)) : [];
}

function text(value) {
  return typeof value === 'string' ? value : '';
}

/*
    A discussion of an export as the application reads it: null without a valid identifier or messages; the
    messages other than questions and answers are left out, and the lists and the texts the pages read take their
    types.
*/
function cleanConversation(c) {
  if (!c || typeof c !== 'object' || typeof c.id !== 'string' || !recordId.test(c.id) || !Array.isArray(c.messages)) {
    return null;
  }
  const messages = [];
  for (const m of objects(c.messages)) {
    if (m.role === 'user') {
      messages.push(Object.assign({}, m, {
        content: text(m.content),
        attachments: objects(m.attachments).filter((a) => typeof a.id === 'string' && typeof a.name === 'string')
          .map((a) => Object.assign({}, a, { text: text(a.text) })),
        images: objects(m.images).filter((x) => typeof x.id === 'string' && typeof x.name === 'string')
      }));
    } else if (m.role === 'assistant') {
      messages.push(Object.assign({}, m, {
        steps: objects(m.steps).map((s) => Object.assign({}, s, {
          content: text(s.content), reasoning: text(s.reasoning), toolCalls: objects(s.toolCalls)
        }))
      }));
    }
  }
  const now = Date.now();
  const index = (v) => Math.min(Math.max(0, Number.isInteger(v) ? v : 0), messages.length);
  return Object.assign({}, c, {
    title: text(c.title), projectId: typeof c.projectId === 'string' ? c.projectId : null, messages,
    created: Number.isFinite(c.created) ? c.created : now, updated: Number.isFinite(c.updated) ? c.updated : now,
    contextStart: index(c.contextStart), memoryMark: index(c.memoryMark),
    summary: c.summary && typeof c.summary.text === 'string' ? c.summary : null, incognito: false
  });
}

// A project of an export as the application reads it, null without a valid identifier or name.
function cleanProject(p) {
  if (!p || typeof p !== 'object' || typeof p.id !== 'string' || !recordId.test(p.id) || typeof p.name !== 'string') {
    return null;
  }
  return Object.assign({}, p, {
    description: text(p.description), instructions: text(p.instructions),
    color: typeof p.color === 'string' ? p.color : '#5b8cff',
    files: objects(p.files).filter((f) => typeof f.id === 'string' && typeof f.name === 'string')
      .map((f) => Object.assign({}, f, { text: text(f.text) })),
    images: objects(p.images).filter((x) => typeof x.id === 'string' && typeof x.name === 'string')
  });
}

/*
    Imports an export: {conversations, projects, images, ids}, the counts of the records kept and the identifiers
    of the discussions imported.  The records the pages could not read are left out.
*/
export async function importAll(data) {
  if (!data || data.format !== 'albireo-export' || !Array.isArray(data.conversations)) throw new Error('format');
  let n = 0, p = 0;
  const ids = [];
  for (const raw of data.conversations) {
    const c = cleanConversation(raw);
    if (!c) continue;
    await putConversation(c);
    ids.push(c.id);
    ++n;
  }
  for (const raw of Array.isArray(data.projects) ? data.projects : []) {
    const pr = cleanProject(raw);
    if (!pr) continue;
    await putProject(pr);
    ++p;
  }
  // The images, each with the owner it belongs to; an export without images (an earlier version) has none.
  const pictures = [];
  for (const x of Array.isArray(data.images) ? data.images : []) {
    if (!x || typeof x.id !== 'string' || typeof x.owner !== 'string') continue;
    const blob = blobOfDataUrl(x.data);
    if (!blob) continue;
    const record = Object.assign({}, x, { blob, thumb: blobOfDataUrl(x.thumbnail) || blob });
    delete record.data;
    delete record.thumbnail;
    pictures.push(record);
  }
  await putImages(pictures);
  return { conversations: n, projects: p, images: pictures.length, ids };
}

/*
    Deletes every discussion, project, passage and image and puts the settings of reset, in one transaction; the other
    settings stay.
*/
export async function clearAll(reset) {
  await open();
  const names = ['conversations', 'projects', 'chunks', 'images'];
  const tx = db.transaction(names.concat(['settings']), 'readwrite');
  for (const name of names) tx.objectStore(name).clear();
  for (const key of Object.keys(reset || {})) tx.objectStore('settings').put({ key, value: reset[key] });
  await completed(tx);
}
