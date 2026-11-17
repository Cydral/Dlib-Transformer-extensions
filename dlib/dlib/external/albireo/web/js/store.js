// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

/*
    The data of the application, kept in the IndexedDB database of this browser for the
    origin of the local server: the discussions with their messages, the projects with
    their instructions and documents, the passages of the documents with their vectors for
    the document search, the images of the discussions and of the projects, and the
    preferences.  Nothing is sent anywhere; an incognito discussion never reaches this module.

    The browser may close this database under the application, when it deletes the data of the
    origin or finds one of their files damaged: the next access then opens it again, and the
    application is told (watchClosed).  Each change of a discussion, of a project, of a setting or
    of an image is copied outside of the database (backup.js), from which the records come back
    when the browser lost the database (copyFound, restoreCopy).  A setting marks the database
    the copy follows (followedKey): the database the browser creates in place of a deleted one
    lacks it, which tells the records it lost from the records deleted in it.
*/

import * as backup from './backup.js';

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

// What the application asked to be told when the browser closes the database (watchClosed).
let closedListener = null;

/*
    Calls listener each time the browser closes the database under the application, as it does when it
    deletes the data of the origin or finds them damaged.  The next access opens the database again.
*/
export function watchClosed(listener) {
  closedListener = listener;
}

/*
    Lets go of a connection the browser no longer answers on, when one of its accesses failed because the
    connection was closing: the next access opens the database again.
*/
export function forget() {
  const d = db;
  db = null;
  if (!d) return;
  try {
    d.close();
  } catch (e) {
    // The connection is already closed.
  }
}

export function open() {
  if (db) return Promise.resolve(db);
  if (blocked) return Promise.reject(blocked);
  if (!opening) {
    opening = connect().then((d) => {
      /*
          When a newer version of the application upgrades the database in another tab, this tab
          lets it go and opens it again at its next access.  A change to no version is the deletion
          of the database, by the tools of the browser for instance: the application is told, as
          for a database the browser closed.
      */
      d.onversionchange = (e) => {
        d.close();
        const held = db === d;
        if (held) db = null;
        if (held && e.newVersion === null && closedListener) closedListener();
      };
      // The browser closed the connection itself: the database was deleted, or one of its files failed.
      d.onclose = () => {
        if (db !== d) return;
        db = null;
        if (closedListener) closedListener();
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

// True when the request of the setting of the mark (followedKey), read in a transaction, found the mark.
function marked(r) {
  return !!r.result && r.result.value === true;
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
  const tx = db.transaction(['conversations', 'settings'], 'readwrite');
  const mark = tx.objectStore('settings').get(followedKey);
  const r = tx.objectStore('conversations').put(c);
  // The copy holds the discussion as the database receives it here, whatever changes it before the copy is written.
  const shot = backup.snapshot(c, c.updated);
  const key = await request(r);
  // Beside a database the copy does not follow, a later state the copy holds stays (copyFound).
  backup.putSnapshot('conversations', c.id, shot, !marked(mark));
  return key;
}

/*
    Changes stored discussions in place, in one transaction, so that the messages another tab
    saved meanwhile stay: fn(record) changes the record read from the database and returns
    true to save it.  The discussion of id only, when id is given.
*/
export async function updateConversations(fn, id) {
  await open();
  const tx = db.transaction(['conversations', 'settings'], 'readwrite');
  const mark = tx.objectStore('settings').get(followedKey);
  const r = tx.objectStore('conversations').openCursor(id == null ? null : IDBKeyRange.only(id));
  const changed = [];
  r.onsuccess = () => {
    const c = r.result;
    if (!c) return;
    const value = c.value;
    if (fn(value)) {
      c.update(value);
      changed.push(value);
    }
    c.continue();
  };
  await completed(tx);
  for (const value of changed) {
    backup.putSnapshot('conversations', value.id, backup.snapshot(value, value.updated), !marked(mark));
  }
}

/*
    Deletes a discussion: its record first, in a transaction of its own, then its passages and its images.  The
    passages and the images that a tab closed meanwhile leaves are pruned at the next start (app.js, pruneIndex and
    pruneImages).
*/
export async function deleteConversation(id) {
  /*
      The copy forgets the discussion and its images first, whatever the database answers, and the discussion again
      once the database did, for a save of the discussion that was on its way to the copy.
  */
  backup.remove('conversations', id);
  backup.removeImagesOf(id);
  await open();
  const tx = db.transaction('conversations', 'readwrite');
  tx.objectStore('conversations').delete(id);
  await completed(tx);
  backup.remove('conversations', id);
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
  const tx = db.transaction(['projects', 'settings'], 'readwrite');
  const mark = tx.objectStore('settings').get(followedKey);
  const r = tx.objectStore('projects').put(p);
  const shot = backup.snapshot(p, p.updated);
  const key = await request(r);
  backup.putSnapshot('projects', p.id, shot, !marked(mark));
  return key;
}

export async function deleteProject(id) {
  // The copy forgets the project and its images first, whatever the database answers (deleteConversation).
  backup.remove('projects', id);
  backup.removeImagesOf(id);
  await open();
  /*
      The discussions of the project stay, outside of any project.  They change in place, in one
      transaction with the deletion of the project, through the index of their projects.
  */
  const tx = db.transaction(['conversations', 'projects', 'settings'], 'readwrite');
  const mark = tx.objectStore('settings').get(followedKey);
  const r = tx.objectStore('conversations').index('project').openCursor(IDBKeyRange.only(id));
  const changed = [];
  r.onsuccess = () => {
    const c = r.result;
    if (!c) return;
    const value = c.value;
    value.projectId = null;
    c.update(value);
    changed.push(value);
    c.continue();
  };
  tx.objectStore('projects').delete(id);
  await completed(tx);
  backup.remove('projects', id);
  for (const value of changed) {
    backup.putSnapshot('conversations', value.id, backup.snapshot(value, value.updated), !marked(mark));
  }
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
  for (const r of records) backup.putImage(r);
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
  // The copy finds the images of the owner itself: the database may have lost them, or may not answer.
  backup.removeImagesOf(owner);
  await open();
  const tx = db.transaction('images', 'readwrite');
  const s = tx.objectStore('images');
  const r = s.index('owner').openKeyCursor(IDBKeyRange.only(owner));
  const ids = [];
  r.onsuccess = () => {
    const c = r.result;
    if (!c) return;
    ids.push(c.primaryKey);
    s.delete(c.primaryKey);
    c.continue();
  };
  await completed(tx);
  for (const id of ids) backup.removeImage(id);
}

export async function deleteImageIds(ids) {
  await open();
  if (!ids.length) return;
  const tx = db.transaction('images', 'readwrite');
  const s = tx.objectStore('images');
  for (const id of ids) s.delete(id);
  await completed(tx);
  for (const id of ids) backup.removeImage(id);
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

/*
    Puts a setting.  It reaches the copy from the database the copy follows only (followedKey): beside a new
    database, the settings of the copy are those of the database the browser deleted, and wait there for their
    restoration (copyFound, restoreCopy).
*/
export async function setSetting(key, value) {
  await open();
  const tx = db.transaction('settings', 'readwrite');
  const s = tx.objectStore('settings');
  const mark = s.get(followedKey);
  const r = s.put({ key, value });
  await completed(tx);
  if (mark.result && mark.result.value === true) backup.put('settings', key, { value }, 0);
  return r.result;
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
  let followed = false;
  const r = s.getAll();
  r.onsuccess = () => {
    const now = {};
    for (const row of r.result) now[row.key] = row.value;
    followed = now[followedKey] === true;
    if (unless && unless(now)) return;
    for (const key of Object.keys(values)) s.put({ key, value: values[key] });
    put = true;
  };
  await completed(tx);
  // As for one setting (setSetting): the copy takes the settings of the database it follows.
  if (put && followed) for (const key of Object.keys(values)) backup.put('settings', key, { value: values[key] }, 0);
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
  const settings = await getSettings();
  delete settings[followedKey];
  parts.push('],"settings":' + JSON.stringify(settings) + ',"images":[');
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
    The setting that marks the database the copy follows.  A database receives it once it holds every discussion
    and every project of the copy, with every image of theirs that the browser reads in the copy (copyFound), and
    the copy follows each of its changes from then on: what such a database lacks was deleted in it.  The database
    the browser creates after it deleted that one carries no mark: what the copy holds beyond it was lost.  The
    mark stays out of the copy and of the exports.  A database that carries the mark and went back to an earlier
    state, its last changes lost when the browser stopped, is followed as it is: the copies of those changes leave
    at the next start.
*/
const followedKey = 'copyFollows';

// The keys of the settings the copy leaves out: those that only one visit of the browser needs, and the mark.
const uncopied = /Lease$|^copyFollows$/;

/*
    Deletes every discussion, project, passage and image and puts the settings of reset, in one transaction; the other
    settings stay.
*/
export async function clearAll(reset) {
  // A database that does not open refuses here, before the copy is touched.
  await open();
  /*
      The copy is emptied first.  The transaction then marks the database it empties as the one the copy follows:
      what a copy the browser refused to delete still holds leaves at the next comparison (alignCopy), and is never
      offered again.
  */
  await backup.clear();
  await open();
  const names = ['conversations', 'projects', 'chunks', 'images'];
  const tx = db.transaction(names.concat(['settings']), 'readwrite');
  for (const name of names) tx.objectStore(name).clear();
  for (const key of Object.keys(reset || {})) tx.objectStore('settings').put({ key, value: reset[key] });
  tx.objectStore('settings').put({ key: followedKey, value: true });
  await completed(tx);
  await copySettings();
}

// Copies the settings of the database that the copy lacks or holds with another value.
async function copySettings() {
  await open();
  const rows = await request(store('settings', 'readonly').getAll());
  const copied = await backup.settings() || {};
  for (const row of rows) {
    if (uncopied.test(row.key)) continue;
    const has = Object.prototype.hasOwnProperty.call(copied, row.key);
    if (has && JSON.stringify(copied[row.key]) === JSON.stringify(row.value)) continue;
    backup.put('settings', row.key, { value: row.value }, 0);
  }
}

/*
    The identifiers of the discussions, of the projects and of the images stored, {conversations, projects, images},
    and followed, true for a database that carries the mark of the copy (followedKey), all read in one transaction.
*/
export async function keys() {
  await open();
  const tx = db.transaction(['conversations', 'projects', 'images', 'settings'], 'readonly');
  const read = (name) => request(tx.objectStore(name).getAllKeys());
  const [conversations, projects, images, mark] = await Promise.all([
    read('conversations'), read('projects'), read('images'), request(tx.objectStore('settings').get(followedKey))
  ]);
  return {
    conversations: new Set(conversations), projects: new Set(projects), images: new Set(images),
    followed: !!mark && mark.value === true
  };
}

/*
    Writes records back into the database: {conversations, projects, settings}, the discussions and the projects as
    lists, the settings as an object.  A stored record as recent as the one given stays, so that the copy of
    another tab that answered since is kept; the settings given are put as they are.  Returns the numbers of
    discussions and of projects written, {conversations, projects}.
*/
export async function putBack(records) {
  await open();
  const tx = db.transaction(['conversations', 'projects', 'settings'], 'readwrite');
  const mark = tx.objectStore('settings').get(followedKey);
  const written = { conversations: [], projects: [] };
  for (const name of ['conversations', 'projects']) {
    const s = tx.objectStore(name);
    for (const value of records[name] || []) {
      const r = s.get(value.id);
      r.onsuccess = () => {
        if (r.result && (r.result.updated || 0) >= (value.updated || 0)) return;
        s.put(value);
        written[name].push(value);
      };
    }
  }
  const settings = records.settings || {};
  for (const key of Object.keys(settings)) {
    if (!uncopied.test(key)) tx.objectStore('settings').put({ key, value: settings[key] });
  }
  await completed(tx);
  for (const name of ['conversations', 'projects']) {
    for (const value of written[name]) {
      backup.putSnapshot(name, value.id, backup.snapshot(value, value.updated), !marked(mark));
    }
  }
  for (const key of Object.keys(settings)) {
    if (!uncopied.test(key)) backup.put('settings', key, { value: settings[key] }, 0);
  }
  return { conversations: written.conversations.length, projects: written.projects.length };
}

/*
    What the copy holds and a new database lacks: {conversations, projects, updated, settings}, the numbers of the
    discussions and of the projects, the time of the latest of them, and the number of settings taken back; null
    without any, and for a database the copy follows (followedKey).  The discussions and the projects that the copy
    holds beyond a database without the mark, or in a later state than it, were lost with the database the browser
    deleted: they stay in the copy until the user restores them or deletes the copy (restoreCopy, clearCopy).  The
    new database changes the copy of one of them in two cases only: it saves that record in a later state than the
    copy holds, which then takes its place, or the user deletes that record.  A database without the mark that
    lacks none of them takes back
    the images of the copy that belong to its records, then receives the mark with the settings of the copy that it
    lacks: the first start of this version, a copy the browser emptied, the end of a restoration, and a database
    deleted while it held neither discussion nor project.  The mark is given in the transaction that reads the
    database, so that it never reaches another database than the one compared, and is withheld while an image of
    the copy that the browser reads could not be stored: the copy then stays as it is.  The list of the copy is
    read before that transaction.
*/
export async function copyFound() {
  const held = await backup.list();
  if (!held) return null;
  if ((await keys()).followed) return null;
  const kept = held.settings.size ? await backup.settings() : null;
  // null when the database refused an image: the comparison goes on, and the mark waits.
  const unread = await bringImages(held).catch(() => null);
  await open();
  const tx = db.transaction(['conversations', 'projects', 'images', 'settings'], 'readwrite');
  const found = { conversations: 0, projects: 0, updated: 0, settings: 0 };
  let followed = false;
  const owners = new Set();
  let images = null;
  let pending = 3;
  const settings = tx.objectStore('settings');
  const compared = () => {
    pending -= 1;
    if (pending || found.conversations + found.projects > 0 || !unread) return;
    // An image of a stored record that the copy alone holds: the mark waits for it.
    for (const [id, owner] of held.images) if (owners.has(owner) && !images.has(id) && !unread.has(id)) return;
    const present = settings.getAllKeys();
    present.onsuccess = () => {
      const has = new Set(present.result);
      for (const key of Object.keys(kept || {})) {
        if (has.has(key) || uncopied.test(key)) continue;
        settings.put({ key, value: kept[key] });
        found.settings += 1;
      }
      settings.put({ key: followedKey, value: true });
    };
  };
  const mark = settings.get(followedKey);
  mark.onsuccess = () => {
    followed = marked(mark);
    if (followed) return;
    for (const name of ['conversations', 'projects']) {
      const r = tx.objectStore(name).getAll();
      r.onsuccess = () => {
        const times = new Map(r.result.map((x) => [x.id, Number(x.updated) || 0]));
        for (const id of times.keys()) owners.add(id);
        for (const [id, time] of held[name]) {
          if (times.has(id) && times.get(id) >= time) continue;
          found[name] += 1;
          found.updated = Math.max(found.updated, time);
        }
        compared();
      };
    }
    const k = tx.objectStore('images').getAllKeys();
    k.onsuccess = () => {
      images = new Set(k.result);
      compared();
    };
  };
  await completed(tx);
  return !followed && found.conversations + found.projects + found.settings > 0 ? found : null;
}

// The numbers of the records of the copy, {conversations, projects, images}; null without a copy.
export async function copyCounts() {
  const held = await backup.list();
  if (!held) return null;
  return { conversations: held.conversations.size, projects: held.projects.size, images: held.images.size };
}

/*
    Brings the records of the copy back into the database: the discussions, the projects and the settings (putBack),
    then the images the database lacks.  Returns {conversations, projects, images}, the numbers written, unread,
    the number of discussions and of projects of the copy that could not be read, which stay in it, and settings,
    the settings of the copy; null when the copy cannot be read.
*/
export async function restoreCopy() {
  const held = await backup.list();
  const records = await backup.records();
  if (!held || !records) return null;
  const clean = {
    conversations: records.conversations.map(cleanConversation).filter(Boolean),
    projects: records.projects.map(cleanProject).filter(Boolean), settings: records.settings
  };
  const unread = Math.max(0, held.conversations.size - clean.conversations.length)
    + Math.max(0, held.projects.size - clean.projects.length);
  const written = await putBack(clean);
  const images = await restoreImages();
  return Object.assign(written, { images, unread, settings: clean.settings });
}

/*
    Stores again the images of the copy held, as its list gave it, that the database lacks and whose owner, a
    discussion or a project, is stored.  Returns the identifiers of those the browser could not read in the copy;
    restored.count is the number stored.  A refusal of the database is thrown.
*/
async function bringImages(held, restored) {
  const unread = new Set();
  if (!held.images.size) return unread;
  const stored = await keys();
  for (const [id, owner] of held.images) {
    if (stored.images.has(id) || !(stored.conversations.has(owner) || stored.projects.has(owner))) continue;
    const record = await backup.image(id);
    if (!record) {
      unread.add(id);
      continue;
    }
    await open();
    const tx = db.transaction('images', 'readwrite');
    tx.objectStore('images').put(record);
    await completed(tx);
    if (restored) restored.count += 1;
  }
  return unread;
}

/*
    Stores again the images of the copy that the database lacks and whose owner, a discussion or a project, is
    stored.  Returns their number.
*/
export async function restoreImages() {
  const held = await backup.list();
  const restored = { count: 0 };
  if (held) await bringImages(held, restored);
  return restored.count;
}

/*
    An export (exportAll) built without the database, for one that does not open or lost its records: the
    discussions, the projects and the settings of records, {conversations, projects, settings}, those of the copy
    without them, and the images of the copy.  null without a copy.
*/
export async function exportCopy(records) {
  const held = await backup.list();
  const own = records || await backup.records();
  if (!held || !own) return null;
  const parts = ['{"format":"albireo-export","version":1,"exported":' + Date.now() + ',"conversations":['];
  own.conversations.forEach((c, i) => parts.push((i ? ',' : '') + JSON.stringify(c)));
  parts.push('],"projects":[');
  own.projects.forEach((p, i) => parts.push((i ? ',' : '') + JSON.stringify(p)));
  parts.push('],"settings":' + JSON.stringify(own.settings) + ',"images":[');
  /*
      The images of the discussions and of the projects exported: the copy may still hold those of a record deleted
      while the database did not answer.
  */
  const owners = new Set(own.conversations.concat(own.projects).map((x) => x.id));
  let first = true;
  for (const [id, owner] of held.images) {
    if (!owners.has(owner)) continue;
    const r = await backup.image(id);
    if (!r) continue;
    const { blob, thumb, ...rest } = r;
    const picture = Object.assign(rest, { data: await dataUrlOf(blob), thumbnail: await dataUrlOf(thumb) });
    parts.push((first ? '' : ',') + JSON.stringify(picture));
    first = false;
  }
  parts.push(']}');
  return new Blob(parts, { type: 'application/json' });
}

/*
    Brings the copy in line with the database it follows, once the lists were read at the start of a visit: lists is
    {conversations, projects}, the records as read, of which the identifiers and the times count, taken at the call.
    The records the copy lacks or holds with another time are copied, each read again from the database when it is
    copied, so that a change made meanwhile is the one kept; so are the settings whose copy differs and the images
    the copy lacks.  A change that keeps the time of its record, a title or a pin for instance, and whose copy the
    browser refused or the end of the visit left behind, waits for the next change of that record.  The copies of
    records and of images the database no longer holds leave: they are listed before the database is read for that
    step, so that a record another tab saves meanwhile, in the database then in the copy, never counts as a copy
    without a record.  A database without the mark (followedKey) leaves the copy as it is, at the start and again
    at the step that removes copies, which reads the mark and the identifiers in one transaction: the database the
    browser creates in place of a deleted one never empties the copy.
*/
export async function alignCopy(lists) {
  const listed = {};
  for (const name of ['conversations', 'projects']) {
    listed[name] = (lists[name] || []).map((x) => ({ id: x.id, updated: Number(x.updated) || 0 }));
  }
  const held = await backup.list();
  if (!held) return;
  const stored = await keys();
  if (!stored.followed) return;
  for (const name of ['conversations', 'projects']) {
    for (const entry of listed[name]) {
      if (held[name].get(entry.id) === entry.updated) continue;
      await open();
      const value = await request(store(name, 'readonly').get(entry.id));
      if (value) await backup.put(name, value.id, value, value.updated);
    }
  }
  await copySettings();
  for (const id of stored.images) {
    if (held.images.has(id)) continue;
    await open();
    const record = await request(store('images', 'readonly').get(id));
    if (record) await backup.putImage(record);
  }
  const copied = await backup.list();
  if (!copied) return;
  const now = await keys();
  if (!now.followed) return;
  for (const name of ['conversations', 'projects']) {
    for (const id of copied[name].keys()) if (!now[name].has(id)) await backup.removeListed(name, id);
  }
  for (const id of copied.parts) if (!now.images.has(id)) await backup.removeImage(id);
}

// Empties the copy of the records, the database staying as it is; false when the browser refused.
export function clearCopy() {
  return backup.clear();
}
