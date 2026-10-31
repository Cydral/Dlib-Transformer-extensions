// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

/*
    The data of the application, kept in the IndexedDB database of this browser for the
    origin of the local server: the discussions with their messages, the projects with
    their instructions and documents, the passages of the documents with their vectors for
    the document search, and the preferences.  Nothing is sent anywhere; an incognito
    discussion never reaches this module.
*/

const DB_NAME = 'albireo';
const DB_VERSION = 2;
let db = null;

function request(r) {
  return new Promise((resolve, reject) => {
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
  });
}

// The opening in progress, which the calls made meanwhile share.
let opening = null;

export function open() {
  if (db) return Promise.resolve(db);
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
  };
  return request(r);
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

export async function deleteConversation(id) {
  await open();
  await deleteChunks('owner', id);
  return request(store('conversations', 'readwrite').delete(id));
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
}

// The passages of the document search.
export async function putChunks(records) {
  await open();
  if (!records.length) return;
  const tx = db.transaction('chunks', 'readwrite');
  const s = tx.objectStore('chunks');
  for (const r of records) s.put(r);
  await new Promise((resolve, reject) => {
    tx.oncomplete = resolve;
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error);
  });
}

export async function chunksOf(owner) {
  await open();
  return request(store('chunks', 'readonly').index('owner').getAll(owner));
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
  const range = IDBKeyRange.only(key);
  await new Promise((resolve, reject) => {
    const cursor = tx.objectStore('chunks').index(index).openKeyCursor(range);
    cursor.onsuccess = () => {
      const c = cursor.result;
      if (!c) return;
      tx.objectStore('chunks').delete(c.primaryKey);
      c.continue();
    };
    tx.oncomplete = resolve;
    tx.onerror = () => reject(tx.error);
  });
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

export async function exportAll() {
  return {
    format: 'albireo-export', version: 1, exported: Date.now(),
    conversations: await allConversations(), projects: await allProjects(), settings: await getSettings()
  };
}

export async function importAll(data) {
  if (!data || data.format !== 'albireo-export' || !Array.isArray(data.conversations)) throw new Error('format');
  let n = 0, p = 0;
  for (const c of data.conversations) {
    if (c && typeof c.id === 'string' && Array.isArray(c.messages)) {
      await putConversation(c);
      ++n;
    }
  }
  for (const pr of data.projects || []) {
    if (pr && typeof pr.id === 'string' && typeof pr.name === 'string') {
      await putProject(pr);
      ++p;
    }
  }
  return { conversations: n, projects: p };
}

export async function clearAll() {
  await open();
  await request(store('conversations', 'readwrite').clear());
  await request(store('projects', 'readwrite').clear());
  await request(store('chunks', 'readwrite').clear());
}
