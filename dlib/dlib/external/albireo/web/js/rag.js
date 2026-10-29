// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

/*
    The document search of the web application, run in this browser.

    The documents of a project, and the large documents attached to a discussion, are cut
    into passages of a few hundred tokens, overlapping by a fifth so that no idea is lost
    at a boundary, each named after its page, slide or sheet.  The embedding model of the
    service computes the vector of each passage once (/v1/embeddings, the passages of
    several requests sharing the passes of the engine), and the passages are kept with
    their vectors in the database of the browser, or in memory for an incognito discussion.

    A question is answered by the passages closest to it in meaning, by their vectors, and
    in words, by the BM25 score of its terms, the two rankings fused, then chosen so that
    they do not repeat each other (maximal marginal relevance) within a budget of
    characters.  The consecutive passages of a document merge into one excerpt.

    Without an embedding model, the search works on the words alone.
*/

import * as store from './store.js';
import { terms } from './documents.js';

export const passageChars = 1200;
export const overlapChars = 240;
const embedBatch = 16;

// The embedding model of the service: {id, dimensions}, or null.
let model = null;

export function setEmbeddingModel(info) {
  const changed = (model && model.id) !== (info && info.id);
  model = info && info.id ? { id: info.id, dimensions: info.dimensions || 0 } : null;
  if (changed) cache.clear();
}

export function embeddingModel() {
  return model;
}

/*
    The text a model receives for a question.  Qwen3-Embedding reads its questions with an
    instruction, and its passages as they are; the other models read both as they are.
*/
function questionText(id, question) {
  if (/qwen3-embedding/i.test(id)) {
    return 'Instruct: Given a question, retrieve the passages of the documents that answer it\nQuery:' + question;
  }
  return question;
}

function passageText(name, where, text) {
  return name + (where ? ' (' + where + ')' : '') + '\n' + text;
}

function decodeVector(base64) {
  const bin = atob(base64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; ++i) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

// The vectors of texts, from the embedding model of the service.
async function embed(texts, signal) {
  const r = await fetch('/v1/embeddings', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, signal,
    body: JSON.stringify({ model: model.id, input: texts, encoding_format: 'base64' })
  });
  if (!r.ok) {
    let message = r.status + ' ' + r.statusText;
    try {
      const j = await r.json();
      if (j && j.error) message = j.error.message || message;
    } catch (e) {
      // The body was not JSON.
    }
    throw new Error(message);
  }
  const j = await r.json();
  const out = new Array(texts.length);
  for (const d of j.data) out[d.index] = decodeVector(d.embedding);
  return out;
}

/*
    Cuts a text into passages: [{start, end, where, text}].  The "## " lines of the text
    name the places (pages, slides, sheets) and the sections of the passages; the rows of a
    sheet repeat its first row, the names of its columns, at the head of each passage.
*/
export function chunk(text) {
  const out = [];
  const place = /^## (Page \d+|Slide \d+.*|Sheet: .*)$/;
  let location = '', section = '', header = '';
  let buffer = '', bufferStart = 0, bufferWhere = '';
  let pos = 0;

  const where = () => (location && section ? location + ' · ' + section : location || section);
  const emit = (keepTail) => {
    const body = buffer.trim();
    if (body) {
      const prefix = header && !buffer.startsWith(header) ? header + '\n' : '';
      out.push({ start: bufferStart, end: pos, where: bufferWhere, text: prefix + body });
    }
    // The end of the passage starts the next one, cut at a line or a sentence.
    let tail = '';
    if (keepTail && buffer.length > overlapChars) {
      tail = buffer.slice(-overlapChars);
      const cut = tail.search(/[\n.!?;]\s/);
      if (cut >= 0 && cut < tail.length - 40) tail = tail.slice(cut + 1).trimStart();
    }
    buffer = tail;
    bufferStart = pos - tail.length;
    bufferWhere = where();
  };
  const add = (piece) => {
    if (!buffer) {
      bufferStart = pos;
      bufferWhere = where();
    }
    buffer += piece;
  };

  const lines = text.split('\n');
  for (let li = 0; li < lines.length; ++li) {
    const line = lines[li] + (li + 1 < lines.length ? '\n' : '');
    const m = line.trimEnd().match(place);
    if (m) {
      // A new place: the passage in progress ends there, without overlap.
      emit(false);
      location = m[1];
      section = '';
      header = '';
      pos += line.length;
      bufferStart = pos;
      bufferWhere = where();
      continue;
    }
    const heading = line.match(/^#{1,6} (.+)$/);
    if (heading) {
      if (buffer.length > passageChars / 3) emit(true);
      section = heading[1].trim().slice(0, 80);
      if (!buffer.trim()) bufferWhere = where();
    }
    if (location.startsWith('Sheet: ') && !header && line.trim()) header = line.trimEnd();
    let rest = line;
    while (rest.length) {
      const room = passageChars - buffer.length;
      if (rest.length <= room) {
        add(rest);
        pos += rest.length;
        rest = '';
      } else {
        // A long line is cut at the last sentence or space that fits.
        let cut = room;
        const slice = rest.slice(0, room);
        const sentence = Math.max(slice.lastIndexOf('. '), slice.lastIndexOf('! '), slice.lastIndexOf('? '));
        if (sentence > room / 2) cut = sentence + 2;
        else {
          const space = slice.lastIndexOf(' ');
          if (space > room / 2) cut = space + 1;
        }
        if (cut <= 0) cut = room > 0 ? room : 0;
        if (cut > 0) {
          add(rest.slice(0, cut));
          pos += cut;
          rest = rest.slice(cut);
        }
        emit(true);
      }
    }
    if (buffer.length >= passageChars) emit(true);
  }
  emit(false);
  return out;
}

// Owners whose passages stay in memory: the incognito discussions.
const memory = new Map();

// The passages of each owner, read once: {records, sparse}.
const cache = new Map();

async function recordsOf(owner) {
  if (memory.has(owner)) return memory.get(owner);
  try {
    return await store.chunksOf(owner);
  } catch (e) {
    return [];
  }
}

async function load(owner) {
  if (cache.has(owner)) return cache.get(owner);
  const records = (await recordsOf(owner)).slice().sort((a, b) => (a.doc < b.doc ? -1 : a.doc > b.doc ? 1 : a.ordinal - b.ordinal));
  // The terms of each passage, for the search by words.
  const sparse = records.map((r) => {
    const tf = new Map();
    const words = terms(r.text);
    for (const w of words) tf.set(w, (tf.get(w) || 0) + 1);
    return { tf, length: words.length };
  });
  for (const r of records) if (r.vector && !(r.vector instanceof Float32Array)) r.vector = new Float32Array(r.vector);
  const entry = { records, sparse };
  cache.set(owner, entry);
  return entry;
}

function forget(owner) {
  cache.delete(owner);
}

/*
    The state of the documents of an owner: Map doc -> {passages, vectors, model}, vectors
    counting the passages that hold a vector of the current model.
*/
export async function status(owner) {
  const { records } = await load(owner);
  const out = new Map();
  for (const r of records) {
    const s = out.get(r.doc) || { passages: 0, vectors: 0, model: r.model };
    s.passages += 1;
    if (r.vector && model && r.model === model.id) s.vectors += 1;
    out.set(r.doc, s);
  }
  return out;
}

/*
    Indexes a document {id, name, text} for an owner: its passages, and their vectors when
    the service has an embedding model.  A document already indexed with the current model
    is left as it is.  onProgress(done, total) follows the vectors computed.  Returns
    {passages, vectors}.
*/
export async function indexDocument(owner, doc, options = {}) {
  const { persistent = true, onProgress = null, signal = null } = options;
  const pieces = chunk(doc.text || '');
  const existing = (await load(owner)).records.filter((r) => r.doc === doc.id);
  const current = model ? model.id : null;
  const complete = existing.length === pieces.length && existing.every((r) => r.model === current && (!current || r.vector));
  if (complete && existing.length) return { passages: existing.length, vectors: current ? existing.length : 0 };

  const records = pieces.map((p, i) => ({
    id: doc.id + ':' + i, owner, doc: doc.id, name: doc.name, ordinal: i, start: p.start, end: p.end,
    where: p.where, text: p.text, model: null, vector: null
  }));
  let vectors = 0;
  if (current) {
    for (let i = 0; i < records.length; i += embedBatch) {
      if (signal && signal.aborted) break;
      const batch = records.slice(i, i + embedBatch);
      try {
        const vs = await embed(batch.map((r) => passageText(r.name, r.where, r.text)), signal);
        batch.forEach((r, k) => {
          r.vector = vs[k];
          r.model = current;
        });
        vectors += batch.length;
      } catch (e) {
        if (e.name === 'AbortError') break;
        // The service lost its model, or stopped: the passages are searched by their words.
        break;
      }
      if (onProgress) onProgress(Math.min(i + embedBatch, records.length), records.length);
    }
  }
  if (persistent) {
    await store.deleteChunks('doc', doc.id);
    await store.putChunks(records.map((r) => Object.assign({}, r, { vector: r.vector ? r.vector.buffer : null })));
    memory.delete(owner);
  } else {
    const kept = (memory.get(owner) || []).filter((r) => r.doc !== doc.id);
    memory.set(owner, kept.concat(records));
  }
  forget(owner);
  return { passages: records.length, vectors };
}

export async function removeDocument(owner, docId) {
  if (memory.has(owner)) memory.set(owner, memory.get(owner).filter((r) => r.doc !== docId));
  else await store.deleteChunks('doc', docId).catch(() => {});
  forget(owner);
}

export async function removeOwner(owner) {
  memory.delete(owner);
  await store.deleteChunks('owner', owner).catch(() => {});
  forget(owner);
}

// The longest end of a that begins b, for the merge of overlapping passages.
function overlap(a, b) {
  const max = Math.min(a.length, b.length, overlapChars + 40);
  for (let k = max; k >= 16; --k) if (a.endsWith(b.slice(0, k))) return k;
  return 0;
}

/*
    The excerpts of the documents of owners closest to a question:
    [{doc, name, where, text, score}], in the order of the documents, within budget
    characters, limit passages at most.
*/
export async function search(owners, question, options = {}) {
  const { limit = 12, budget = 12000, signal = null } = options;
  const all = [];
  const sparse = [];
  for (const o of owners) {
    const entry = await load(o);
    for (let i = 0; i < entry.records.length; ++i) {
      all.push(entry.records[i]);
      sparse.push(entry.sparse[i]);
    }
  }
  if (!all.length || !String(question).trim()) return [];

  // By words: BM25 over the passages of the owners.
  const q = Array.from(new Set(terms(question)));
  const words = new Float64Array(all.length);
  if (q.length) {
    const df = new Map();
    for (const s of sparse) for (const w of q) if (s.tf.has(w)) df.set(w, (df.get(w) || 0) + 1);
    const avg = sparse.reduce((x, s) => x + s.length, 0) / sparse.length || 1;
    const n = all.length;
    for (let i = 0; i < n; ++i) {
      let score = 0;
      for (const w of q) {
        const f = sparse[i].tf.get(w) || 0;
        if (!f) continue;
        const d = df.get(w) || 0;
        score += Math.log(1 + (n - d + 0.5) / (d + 0.5)) * (f * 2.2) / (f + 1.2 * (0.25 + 0.75 * sparse[i].length / avg));
      }
      words[i] = score;
    }
  }

  // By meaning: the cosine of the vectors, all of unit length.
  const meaning = new Float64Array(all.length).fill(-2);
  let questionVector = null;
  if (model && all.some((r) => r.vector && r.model === model.id)) {
    try {
      questionVector = (await embed([questionText(model.id, question)], signal))[0];
    } catch (e) {
      if (e.name === 'AbortError') throw e;
      questionVector = null;
    }
  }
  if (questionVector) {
    const d = questionVector.length;
    for (let i = 0; i < all.length; ++i) {
      const v = all[i].vector;
      if (!v || all[i].model !== model.id || v.length !== d) continue;
      let dot = 0;
      for (let k = 0; k < d; ++k) dot += v[k] * questionVector[k];
      meaning[i] = dot;
    }
  }

  // Reciprocal rank fusion of the two rankings.
  const fused = new Float64Array(all.length);
  const rank = (scores, floor) => {
    const order = Array.from(scores.keys()).filter((i) => scores[i] > floor).sort((a, b) => scores[b] - scores[a]);
    order.slice(0, 60).forEach((i, r) => { fused[i] += 1 / (60 + r); });
  };
  rank(words, 0);
  if (questionVector) rank(meaning, -1);
  const candidates = Array.from(fused.keys()).filter((i) => fused[i] > 0).sort((a, b) => fused[b] - fused[a]).slice(0, 60);
  if (!candidates.length) return [];

  // Maximal marginal relevance: each passage chosen for its relevance, less its likeness to
  // the passages already chosen.
  const top = fused[candidates[0]];
  const chosen = [];
  let used = 0;
  const similarity = (a, b) => {
    const va = all[a].vector, vb = all[b].vector;
    if (va && vb && va.length === vb.length) {
      let dot = 0;
      for (let k = 0; k < va.length; ++k) dot += va[k] * vb[k];
      return dot;
    }
    return all[a].doc === all[b].doc && Math.abs(all[a].ordinal - all[b].ordinal) <= 1 ? 0.6 : 0;
  };
  const pool = candidates.slice();
  while (pool.length && chosen.length < limit) {
    let best = -1, bestScore = -Infinity;
    for (let k = 0; k < pool.length; ++k) {
      const i = pool[k];
      let like = 0;
      for (const j of chosen) like = Math.max(like, similarity(i, j));
      const score = 0.75 * (fused[i] / top) - 0.25 * like;
      if (score > bestScore) {
        bestScore = score;
        best = k;
      }
    }
    const i = pool.splice(best, 1)[0];
    if (used + all[i].text.length > budget && chosen.length) continue;
    chosen.push(i);
    used += all[i].text.length;
  }

  // The excerpts in the order of the documents, the consecutive passages merged.
  chosen.sort((a, b) => (all[a].doc < all[b].doc ? -1 : all[a].doc > all[b].doc ? 1 : all[a].ordinal - all[b].ordinal));
  const out = [];
  for (const i of chosen) {
    const r = all[i];
    const last = out[out.length - 1];
    if (last && last.doc === r.doc && last.lastOrdinal + 1 === r.ordinal) {
      const k = overlap(last.text, r.text);
      last.text += (k ? '' : '\n') + r.text.slice(k);
      last.lastOrdinal = r.ordinal;
      last.score = Math.max(last.score, fused[i]);
      continue;
    }
    out.push({ doc: r.doc, name: r.name, where: r.where, text: r.text, score: fused[i], lastOrdinal: r.ordinal });
  }
  return out.map(({ doc, name, where, text, score }) => ({ doc, name, where, text, score }));
}
