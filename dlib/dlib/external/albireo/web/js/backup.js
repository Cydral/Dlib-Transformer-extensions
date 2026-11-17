// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

/*
    A second copy of the records of the application, kept by this browser outside of its IndexedDB
    database, in the cache storage of the origin: the discussions, the projects, the settings and
    the images.  A browser that finds a damaged file in the database of an origin deletes that
    database whole and starts again with an empty one; the cache storage, kept in other files, then
    still holds the records, which the application offers to restore (app.js).

    The copy follows each change of the database (store.js): a record saved is copied, a record
    deleted leaves the copy, and the deletion of everything empties it.  It leaves with the data
    of the site when the user clears them in the browser, and never holds an incognito discussion.
    Nothing leaves the browser.

    Each record is the body of an answer the cache keeps under an address of the origin that the
    service never serves: /albireo-copy/<kind>/~<id>, with the time of the record for a discussion
    or a project (?u=...), so that the list of the addresses tells which copies are out of date.
    An image is its Blob under images, its thumbnail under thumbs and its description under
    pictures, whose address names the owner of the image (?o=...), so that the images of a
    discussion or of a project are found without reading them.  A record is copied as it is when
    its copy is asked, whatever changes it before the cache receives it.

    Every function returns instead of throwing, false or null when the browser has no cache
    storage or refused the operation: the database stays the reference, and the copy of a record
    whose time changed since catches up at the next start (store.js, alignCopy).
*/

const cacheName = 'albireo-copy';
const root = '/albireo-copy/';

// The operations of this tab run one after the other, so that its copies of one record never cross.
let tail = Promise.resolve();

function queued(task) {
  const run = tail.then(task, task);
  tail = run.catch(() => {});
  return run;
}

/*
    The addresses this tab knows in the copy for each record, by kind and identifier: those of the last list, and
    those it wrote since.  A new copy of a record then replaces the earlier ones by their addresses, without a
    search of the cache, whose time grows with the number of its entries.  An address another tab wrote since the
    list is not known here: the next list finds it, and the latest copy of a record is the one read meanwhile.
*/
const known = new Map();

function knownOf(kind, id) {
  const key = kind + '/' + id;
  if (!known.has(key)) known.set(key, new Set());
  return known.get(key);
}

// The cache of the copy, null when the browser has none or refuses it.
async function opened() {
  if (typeof caches !== 'object' || !caches || typeof caches.open !== 'function') return null;
  try {
    return await caches.open(cacheName);
  } catch (e) {
    return null;
  }
}

// The mark before an identifier keeps the identifiers made of dots from being read as steps of the path.
function address(kind, id) {
  return root + kind + '/~' + encodeURIComponent(id);
}

/*
    What an address of the copy holds: {kind, id, updated, owner}, the kind and the identifier of the record, its
    time and, for the description of an image, the owner of the image; null for another address.
*/
function parsed(url) {
  let u;
  try {
    u = new URL(url);
  } catch (e) {
    return null;
  }
  if (!u.pathname.startsWith(root)) return null;
  const parts = u.pathname.slice(root.length).split('/');
  if (parts.length !== 2 || !parts[0] || parts[1].length < 2 || parts[1][0] !== '~') return null;
  let id;
  try {
    id = decodeURIComponent(parts[1].slice(1));
  } catch (e) {
    return null;
  }
  return { kind: parts[0], id, updated: Number(u.searchParams.get('u')) || 0, owner: u.searchParams.get('o') };
}

/*
    The record value as the copy will hold it, read now: {text, updated}, the text of the record and its time, 0 for
    a setting.  A record that changes afterwards is copied as it was here.  null for a value without a text.
*/
export function snapshot(value, updated) {
  try {
    const text = JSON.stringify(value);
    return typeof text === 'string' ? { text, updated: Number(updated) || 0 } : null;
  } catch (e) {
    return null;
  }
}

/*
    Keeps answer in the cache c under the address of the record of kind and of identifier id that ends with query,
    in place of the copies of that record this tab knows.  With later true, the cache is searched first for every
    copy of the record, and answer is left out when one of them has a later time than updated, the time of answer.
*/
async function keep(c, kind, id, query, answer, later, updated) {
  const request = new Request(address(kind, id) + query);
  const others = knownOf(kind, id);
  /*
      A record written with an earlier time than a copy this tab knows went back, as an import of an older file
      does: the cache is searched then too, so that no copy another tab wrote stays as the latest.
  */
  const back = Array.from(others).some((url) => (parsed(url) || { updated: 0 }).updated > updated);
  if (later || back) {
    for (const r of await c.keys(address(kind, id), { ignoreSearch: true })) {
      const p = parsed(r.url);
      if (later && p && p.updated > updated) return;
      others.add(r.url);
    }
  }
  await c.put(request, answer);
  // The copies of the earlier states of the record leave once this one is kept.
  for (const url of others) if (url !== request.url) await c.delete(url);
  others.clear();
  others.add(request.url);
}

/*
    Copies the record value of kind ('conversations', 'projects' or 'settings') and of identifier id, in place of
    its earlier copy.  updated is the time of the record, 0 for a setting.
*/
export function put(kind, id, value, updated) {
  return putSnapshot(kind, id, snapshot(value, updated));
}

/*
    Copies a record read earlier (snapshot), in place of its earlier copy.  With later true, a copy of the record
    that has a later time stays in place of this one: beside a database that lost its records, the copy is the only
    place that holds their latest state (store.js, copyFound).
*/
export function putSnapshot(kind, id, shot, later) {
  if (!shot) return Promise.resolve(false);
  return queued(async () => {
    const c = await opened();
    if (!c) return false;
    try {
      const answer = new Response(shot.text, { headers: { 'Content-Type': 'application/json' } });
      await keep(c, kind, id, '?u=' + shot.updated, answer, !!later, shot.updated);
      return true;
    } catch (e) {
      return false;
    }
  });
}

// Removes every copy of a record, those another tab wrote included.
export function remove(kind, id) {
  return queued(async () => {
    const c = await opened();
    if (!c) return false;
    try {
      known.delete(kind + '/' + id);
      await c.delete(address(kind, id), { ignoreSearch: true });
      return true;
    } catch (e) {
      return false;
    }
  });
}

/*
    Removes the copies of a record that the last list holds, by their addresses: the step of an alignment that
    removes the copies without a record (store.js, alignCopy), right after its list.
*/
export function removeListed(kind, id) {
  return queued(async () => {
    const c = await opened();
    if (!c) return false;
    try {
      const urls = Array.from(known.get(kind + '/' + id) || []);
      known.delete(kind + '/' + id);
      for (const url of urls) await c.delete(url);
      return true;
    } catch (e) {
      return false;
    }
  });
}

/*
    Copies an image of the database (store.js): its description, its thumbnail, then its Blob, whose copy tells
    that the image is whole.
*/
export function putImage(record) {
  return queued(async () => {
    const c = await opened();
    if (!c || !record || !(record.blob instanceof Blob)) return false;
    try {
      const { blob, thumb, ...description } = record;
      const described = new Response(JSON.stringify(description), { headers: { 'Content-Type': 'application/json' } });
      await keep(c, 'pictures', record.id, '?o=' + encodeURIComponent(String(record.owner)), described, false, 0);
      if (thumb instanceof Blob) {
        await c.put(new Request(address('thumbs', record.id)), new Response(thumb, {
          headers: { 'Content-Type': thumb.type || 'application/octet-stream' }
        }));
      } else {
        await c.delete(address('thumbs', record.id));
      }
      await c.put(new Request(address('images', record.id)), new Response(blob, {
        headers: { 'Content-Type': blob.type || 'application/octet-stream' }
      }));
      return true;
    } catch (e) {
      return false;
    }
  });
}

// Removes the entries of an image from the cache c: its Blob first, then its thumbnail and its descriptions.
async function dropImage(c, id) {
  known.delete('pictures/' + id);
  await c.delete(address('images', id));
  await c.delete(address('thumbs', id));
  await c.delete(address('pictures', id), { ignoreSearch: true });
}

export function removeImage(id) {
  return queued(async () => {
    const c = await opened();
    if (!c) return false;
    try {
      await dropImage(c, id);
      return true;
    } catch (e) {
      return false;
    }
  });
}

/*
    Removes the images of an owner, a discussion or a project, found by the addresses of their descriptions: the
    database is not asked, which may have lost them or may not answer.
*/
export function removeImagesOf(owner) {
  return queued(async () => {
    const c = await opened();
    if (!c) return false;
    try {
      for (const r of await c.keys()) {
        const p = parsed(r.url);
        if (p && p.kind === 'pictures' && p.owner === String(owner)) await dropImage(c, p.id);
      }
      return true;
    } catch (e) {
      return false;
    }
  });
}

/*
    What the copy holds: {conversations, projects}, each a Map of the identifiers to the times of their records;
    {images}, a Map of the identifiers of the images that are whole to their owners; {parts}, a Set of the
    identifiers of the images of which the copy holds an entry, whole or not; and {settings}, a Set of identifiers.
    null without a copy.  The addresses read here are those this tab knows from now on.
*/
export function list() {
  return queued(async () => {
    const c = await opened();
    if (!c) return null;
    const out = {
      conversations: new Map(), projects: new Map(), images: new Map(), parts: new Set(), settings: new Set()
    };
    const whole = new Set();
    const owners = new Map();
    try {
      const requests = await c.keys();
      known.clear();
      for (const r of requests) {
        const p = parsed(r.url);
        if (!p) continue;
        if (p.kind === 'conversations' || p.kind === 'projects') {
          out[p.kind].set(p.id, Math.max(p.updated, out[p.kind].get(p.id) || 0));
          knownOf(p.kind, p.id).add(r.url);
        } else if (p.kind === 'settings') {
          out.settings.add(p.id);
          knownOf(p.kind, p.id).add(r.url);
        } else if (p.kind === 'pictures') {
          owners.set(p.id, p.owner || '');
          out.parts.add(p.id);
          knownOf(p.kind, p.id).add(r.url);
        } else if (p.kind === 'images' || p.kind === 'thumbs') {
          if (p.kind === 'images') whole.add(p.id);
          out.parts.add(p.id);
        }
      }
    } catch (e) {
      return null;
    }
    for (const id of whole) if (owners.has(id)) out.images.set(id, owners.get(id));
    return out;
  });
}

/*
    The records of the copy: {conversations, projects, settings}, the discussions and the projects as lists, each in
    its latest copy, and the settings as an object of their values.  A record the browser cannot read is left out.
    null without a copy.
*/
export function records() {
  return queued(async () => {
    const c = await opened();
    if (!c) return null;
    const out = { conversations: [], projects: [], settings: {} };
    const latest = new Map();
    try {
      for (const r of await c.keys()) {
        const p = parsed(r.url);
        if (!p || !Object.prototype.hasOwnProperty.call(out, p.kind)) continue;
        const key = p.kind + '/' + p.id;
        const best = latest.get(key);
        if (!best || p.updated > best.p.updated) latest.set(key, { p, r });
      }
    } catch (e) {
      return null;
    }
    for (const { p, r } of latest.values()) {
      try {
        const answer = await c.match(r);
        const value = answer ? await answer.json() : null;
        if (value === null || typeof value !== 'object') continue;
        if (p.kind === 'settings') out.settings[p.id] = value.value;
        else out[p.kind].push(value);
      } catch (e) {
        // A copy the browser no longer reads.
      }
    }
    return out;
  });
}

// The settings of the copy as an object of their values, those the browser cannot read left out; null without a copy.
export function settings() {
  return queued(async () => {
    const c = await opened();
    if (!c) return null;
    const out = {};
    try {
      for (const r of await c.keys()) {
        const p = parsed(r.url);
        if (!p || p.kind !== 'settings') continue;
        const answer = await c.match(r);
        const value = answer ? await answer.json().catch(() => null) : null;
        if (value !== null && typeof value === 'object') out[p.id] = value.value;
      }
    } catch (e) {
      return null;
    }
    return out;
  });
}

// The image of identifier id as the database keeps it, its thumbnail the image itself without one; null without it.
export function image(id) {
  return queued(async () => {
    const c = await opened();
    if (!c) return null;
    try {
      const answer = await c.match(address('images', id));
      if (!answer) return null;
      // The address of the description, known since the last list, else searched for.
      const [url] = Array.from(known.get('pictures/' + id) || []);
      const described = (url && await c.match(url)) || await c.match(address('pictures', id), { ignoreSearch: true });
      if (!described) return null;
      const description = await described.json();
      if (!description || typeof description.id !== 'string' || typeof description.owner !== 'string') return null;
      const blob = await answer.blob();
      const small = await c.match(address('thumbs', id));
      return Object.assign({}, description, { blob, thumb: small ? await small.blob() : blob });
    } catch (e) {
      return null;
    }
  });
}

/*
    Empties the copy.  Returns true once the copy is gone, and for a browser without cache storage, where no copy
    was ever kept; false when the browser refused to tell whether it holds a copy or to delete it.
*/
export function clear() {
  return queued(async () => {
    if (typeof caches !== 'object' || !caches || typeof caches.delete !== 'function') return true;
    try {
      known.clear();
      if (!await caches.has(cacheName)) return true;
      await caches.delete(cacheName);
      return true;
    } catch (e) {
      return false;
    }
  });
}
