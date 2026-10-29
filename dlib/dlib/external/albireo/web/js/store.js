// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

/*
    The data of the application, kept in the IndexedDB database of this browser for the
    origin of the local server: the discussions with their messages, the projects with
    their instructions and documents, and the preferences.  Nothing is sent anywhere; an
    incognito discussion never reaches this module.
*/

const DB_NAME = 'albireo';
const DB_VERSION = 1;
let db = null;

function request(r) {
  return new Promise((resolve, reject) => {
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
  });
}

export async function open() {
  if (db) return db;
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
  };
  db = await request(r);
  return db;
}

function store(name, mode) {
  return db.transaction(name, mode).objectStore(name);
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

export async function deleteConversation(id) {
  await open();
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
  // The discussions of the project stay, outside of any project.
  const convs = await allConversations();
  for (const c of convs) {
    if (c.projectId === id) {
      c.projectId = null;
      await putConversation(c);
    }
  }
  return request(store('projects', 'readwrite').delete(id));
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
}
