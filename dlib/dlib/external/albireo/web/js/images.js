// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

/*
    The images of the discussions and of the projects, prepared in this browser.  The browser
    decodes the file (PNG, JPEG, WebP, the first frame of a GIF, BMP, AVIF and the other formats
    it reads), the image is scaled so that its longest side is maxSide pixels at most, then
    encoded as JPEG, or as PNG when it has transparency or when the PNG is smaller, with a
    thumbnail.

    The discussions and the projects hold the description of each image only (id, name, type,
    size, width, height); the images themselves are Blobs kept in the database of the browser
    (store.js), by their owner, the discussion or the project, or in memory for an incognito
    discussion.  The requests receive them as data URLs, built from these Blobs when they are
    sent.  Nothing leaves the browser but these requests to the local service.
*/

import * as store from './store.js';

export const maxSide = 1536;
const thumbSide = 192;
const jpegQuality = 0.9;
const thumbQuality = 0.82;

// The images of this tab: id -> {blob, thumb, owner, stored}, those prepared and not sent yet included.
const held = new Map();
// The images kept in memory only, those of the incognito discussions: id -> record.
const memory = new Map();
// The object URLs of the images shown: id -> {thumb, full}.
const urls = new Map();

// A canvas of width by height pixels, outside the page where the browser offers one.
function surface(width, height) {
  if (typeof OffscreenCanvas === 'function') return new OffscreenCanvas(width, height);
  const c = document.createElement('canvas');
  c.width = width;
  c.height = height;
  return c;
}

function encode(canvas, type, quality) {
  if (typeof canvas.convertToBlob === 'function') return canvas.convertToBlob({ type, quality });
  return new Promise((resolve, reject) => canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('encode'))), type,
    quality));
}

// True when a pixel of the canvas is not fully opaque.
function transparent(ctx, width, height) {
  const data = ctx.getImageData(0, 0, width, height).data;
  for (let i = 3; i < data.length; i += 4) if (data[i] !== 255) return true;
  return false;
}

/*
    The image drawn from source (an ImageBitmap or a canvas) of width by height pixels, scaled to
    maxSide and encoded: {blob, thumb, width, height}.  opaque tells that the source has no
    transparency, as a JPEG file or a page of a PDF.
*/
async function encodeImage(source, width, height, opaque) {
  const scale = Math.min(1, maxSide / Math.max(width, height));
  const w = Math.max(1, Math.round(width * scale));
  const h = Math.max(1, Math.round(height * scale));
  const canvas = surface(w, h);
  const ctx = canvas.getContext('2d', { willReadFrequently: !opaque });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(source, 0, 0, w, h);
  const alpha = !opaque && transparent(ctx, w, h);
  let blob = await encode(canvas, alpha ? 'image/png' : 'image/jpeg', jpegQuality);
  if (!alpha) {
    // A screenshot or a drawing of few colors is often smaller as PNG, and keeps its edges sharp.
    const png = await encode(canvas, 'image/png');
    if (png.size < blob.size) blob = png;
  }
  const ts = Math.min(1, thumbSide / Math.max(w, h));
  const tw = Math.max(1, Math.round(w * ts));
  const th = Math.max(1, Math.round(h * ts));
  const small = surface(tw, th);
  const sctx = small.getContext('2d');
  sctx.imageSmoothingEnabled = true;
  sctx.imageSmoothingQuality = 'high';
  sctx.drawImage(canvas, 0, 0, tw, th);
  const thumb = await encode(small, alpha ? 'image/png' : 'image/jpeg', thumbQuality);
  return { blob, thumb, width: w, height: h };
}

// The description of an image that the discussions and the projects keep.
function metaOf(name, r, source) {
  return {
    id: store.uid(), name, type: r.blob.type, size: r.blob.size, width: r.width, height: r.height,
    added: Date.now(), source
  };
}

/*
    Prepares an image file: decoded by the browser with its orientation, scaled, encoded, and
    held by this tab until it is kept (keep) or dropped (drop).  Returns its description
    {id, name, type, size, width, height, added, source: {type, size, width, height}}, or throws
    "image-unreadable" when the browser cannot decode the file (HEIC or TIFF for instance).
*/
export async function prepare(file, name) {
  let bitmap;
  try {
    bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
  } catch (e) {
    throw new Error('image-unreadable');
  }
  try {
    const opaque = /^image\/jpe?g$/i.test(file.type || '');
    const r = await encodeImage(bitmap, bitmap.width, bitmap.height, opaque);
    const meta = metaOf(name || file.name || 'image', r,
      { type: file.type || '', size: file.size, width: bitmap.width, height: bitmap.height });
    held.set(meta.id, { blob: r.blob, thumb: r.thumb, owner: null, stored: false });
    return meta;
  } finally {
    bitmap.close();
  }
}

// Prepares an image drawn in a canvas, such as a page of a PDF rendered by pdf.js, which is opaque.
export async function prepareCanvas(canvas, name) {
  const r = await encodeImage(canvas, canvas.width, canvas.height, true);
  const meta = metaOf(name, r, { type: 'application/pdf', size: 0, width: canvas.width, height: canvas.height });
  held.set(meta.id, { blob: r.blob, thumb: r.thumb, owner: null, stored: false });
  return meta;
}

/*
    Keeps the images of an owner (a discussion or a project) that this tab holds and has not kept
    yet: in the database of the browser when persistent is true, in memory otherwise.  The
    images already kept, such as those of a question asked again, are left as they are.  Throws
    the error of the database, the images staying in memory for this visit.
*/
export async function keep(owner, metas, persistent) {
  const records = [];
  for (const m of metas || []) {
    const h = held.get(m.id);
    if (!h || h.stored) continue;
    h.owner = owner;
    const record = {
      id: m.id, owner, name: m.name, type: m.type, size: m.size, width: m.width, height: m.height,
      added: m.added || Date.now(), blob: h.blob, thumb: h.thumb
    };
    if (persistent) records.push(record);
    else {
      memory.set(m.id, record);
      h.stored = true;
    }
  }
  if (!records.length) return;
  await store.putImages(records);
  for (const r of records) held.get(r.id).stored = true;
}

// The record of an image: {blob, thumb, ...}, from this tab, from memory or from the database; null when missing.
async function recordOf(id) {
  const h = held.get(id);
  if (h) return h;
  if (memory.has(id)) return memory.get(id);
  try {
    return (await store.getImage(id)) || null;
  } catch (e) {
    return null;
  }
}

// The Blob of an image, null when it is missing from this browser.
export async function blobOf(id) {
  const r = await recordOf(id);
  return r && r.blob instanceof Blob ? r.blob : null;
}

// True when the image is available in this browser.
export async function available(id) {
  return !!(await blobOf(id));
}

async function objectUrl(id, kind) {
  const u = urls.get(id);
  if (u && u[kind]) return u[kind];
  const r = await recordOf(id);
  const blob = r && (kind === 'thumb' ? r.thumb : r.blob);
  if (!(blob instanceof Blob)) return null;
  const entry = urls.get(id) || {};
  if (!entry[kind]) entry[kind] = URL.createObjectURL(blob);
  urls.set(id, entry);
  return entry[kind];
}

// The address of the thumbnail of an image, for the page; null when the image is missing.
export function thumbUrl(id) {
  return objectUrl(id, 'thumb');
}

// The address of the image itself, for the viewer; null when the image is missing.
export function fullUrl(id) {
  return objectUrl(id, 'full');
}

// The image as a data URL, as a request carries it; null when the image is missing.
export async function dataUrl(id) {
  const blob = await blobOf(id);
  return blob ? store.dataUrlOf(blob) : null;
}

function release(id) {
  const u = urls.get(id);
  if (u) {
    if (u.thumb) URL.revokeObjectURL(u.thumb);
    if (u.full) URL.revokeObjectURL(u.full);
    urls.delete(id);
  }
}

// Forgets the images prepared and not kept, such as those removed from the composer.
export function drop(ids) {
  for (const id of ids || []) {
    const h = held.get(id);
    if (h && h.stored) continue;
    held.delete(id);
    release(id);
  }
}

// Deletes images, wherever they are kept.
export async function remove(ids) {
  const list = Array.from(new Set(ids || []));
  if (!list.length) return;
  for (const id of list) {
    held.delete(id);
    memory.delete(id);
    release(id);
  }
  await store.deleteImageIds(list).catch(() => {});
}

// Deletes the images of an owner, a discussion or a project.
export async function removeOwner(owner) {
  const ids = [];
  for (const [id, h] of held) if (h.owner === owner) ids.push(id);
  for (const [id, r] of memory) if (r.owner === owner) ids.push(id);
  for (const id of ids) {
    held.delete(id);
    memory.delete(id);
    release(id);
  }
  const stored = await store.imageKeys(owner).catch(() => []);
  for (const k of stored) release(k.id);
  await store.deleteImages(owner).catch(() => {});
}

// Forgets every image, after the database was emptied.
export function removeAll() {
  for (const id of Array.from(urls.keys())) release(id);
  held.clear();
  memory.clear();
}

/*
    Deletes the stored images that no owner refers to any longer.  listed are the images of the
    database {owner, id} read before the discussions and the projects were: an image is stored
    after the description of its owner, so that one stored meanwhile is never taken for a
    forgotten one.  references maps each owner to the ids it refers to.
*/
export async function prune(listed, references) {
  const gone = listed.filter((k) => !(references.get(k.owner) || new Set()).has(k.id)).map((k) => k.id);
  if (gone.length) await remove(gone);
  return gone.length;
}

// The images stored in this browser: {count, bytes}.
export function usage() {
  return store.imageUsage();
}
