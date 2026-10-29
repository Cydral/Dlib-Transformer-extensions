// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

/*
    Text of the files attached to a discussion or to a project, extracted in the browser:
    plain text and source code, Word, Excel and PowerPoint documents, OpenDocument files,
    PDF (through pdf.js), RTF and HTML.  The archives of the Office and OpenDocument formats
    are read with the decompression of the browser; nothing is sent to the service but the
    text, inside the messages.
*/

export const maxFileBytes = 64 * 1024 * 1024;
export const maxTextChars = 4 * 1024 * 1024;

const textExtensions = new Set(('txt md markdown rst csv tsv json jsonl ndjson xml yaml yml toml ini cfg conf log env ' +
  'c h cc cpp cxx hpp hh hxx inl cu cuh py pyw pyi js mjs cjs ts tsx jsx java kt kts scala go rs swift m mm cs fs vb ' +
  'php rb pl pm lua r jl sh bash zsh fish ps1 psm1 bat cmd sql graphql proto cmake mk make gradle properties tex bib ' +
  'css scss less vue svelte dart hs ml clj ex exs erl asm s v sv vhd tf hcl dockerfile gitignore editorconfig nix srt ' +
  'vtt diff patch').split(' '));

function extension(name) {
  const base = String(name).toLowerCase();
  if (base === 'dockerfile' || base === 'makefile' || base === 'cmakelists.txt') return 'txt';
  const i = base.lastIndexOf('.');
  return i < 0 ? '' : base.slice(i + 1);
}

function decodeText(bytes) {
  try {
    return new TextDecoder('utf-8', { fatal: true }).decode(bytes);
  } catch (e) {
    // Not UTF-8: the Windows code page of Western Europe, the most frequent other one.
    return new TextDecoder('windows-1252').decode(bytes);
  }
}

function looksBinary(bytes) {
  const n = Math.min(bytes.length, 8192);
  let zero = 0;
  for (let i = 0; i < n; ++i) if (bytes[i] === 0) ++zero;
  return zero > 0;
}

// The entries of a ZIP archive, read on demand.
async function unzip(buffer) {
  const dv = new DataView(buffer);
  let end = -1;
  for (let i = buffer.byteLength - 22; i >= Math.max(0, buffer.byteLength - 65557); --i) {
    if (dv.getUint32(i, true) === 0x06054b50) {
      end = i;
      break;
    }
  }
  if (end < 0) throw new Error('not a ZIP archive');
  const count = dv.getUint16(end + 10, true);
  let p = dv.getUint32(end + 16, true);
  const entries = new Map();
  const names = new TextDecoder();
  for (let k = 0; k < count && p + 46 <= buffer.byteLength; ++k) {
    if (dv.getUint32(p, true) !== 0x02014b50) break;
    const method = dv.getUint16(p + 10, true);
    const csize = dv.getUint32(p + 20, true);
    const nlen = dv.getUint16(p + 28, true);
    const elen = dv.getUint16(p + 30, true);
    const clen = dv.getUint16(p + 32, true);
    const offset = dv.getUint32(p + 42, true);
    entries.set(names.decode(new Uint8Array(buffer, p + 46, nlen)), { method, csize, offset });
    p += 46 + nlen + elen + clen;
  }
  return {
    names: () => Array.from(entries.keys()),
    async text(name) {
      const e = entries.get(name);
      if (!e) return null;
      const nlen = dv.getUint16(e.offset + 26, true);
      const elen = dv.getUint16(e.offset + 28, true);
      const data = new Uint8Array(buffer, e.offset + 30 + nlen + elen, e.csize);
      if (e.method === 0) return decodeText(data);
      if (e.method !== 8) throw new Error('unsupported compression');
      const stream = new Blob([data]).stream().pipeThrough(new DecompressionStream('deflate-raw'));
      return decodeText(new Uint8Array(await new Response(stream).arrayBuffer()));
    }
  };
}

function parseXml(text) {
  const doc = new DOMParser().parseFromString(text, 'application/xml');
  if (doc.getElementsByTagName('parsererror').length) throw new Error('invalid XML');
  return doc;
}

function byLocalName(root, name) {
  return Array.from(root.getElementsByTagNameNS('*', name));
}

function numericOrder(a, b) {
  const na = parseInt((a.match(/(\d+)\.xml$/) || [0, 0])[1], 10);
  const nb = parseInt((b.match(/(\d+)\.xml$/) || [0, 0])[1], 10);
  return na - nb;
}

async function docxText(zip) {
  const doc = parseXml(await zip.text('word/document.xml'));
  const out = [];
  const body = byLocalName(doc, 'body')[0] || doc.documentElement;
  const paragraphText = (p) => {
    let s = '';
    const walk = (n) => {
      for (const c of n.childNodes) {
        if (c.nodeType !== 1) continue;
        if (c.localName === 't') s += c.textContent;
        else if (c.localName === 'tab') s += '\t';
        else if (c.localName === 'br' || c.localName === 'cr') s += '\n';
        else walk(c);
      }
    };
    walk(p);
    return s;
  };
  for (const node of body.childNodes) {
    if (node.nodeType !== 1) continue;
    if (node.localName === 'p') out.push(paragraphText(node));
    else if (node.localName === 'tbl') {
      for (const row of byLocalName(node, 'tr')) {
        out.push(byLocalName(row, 'tc').map((c) => byLocalName(c, 'p').map(paragraphText).join(' ').trim()).join(' | '));
      }
      out.push('');
    }
  }
  return out.join('\n');
}

async function pptxText(zip) {
  const slides = zip.names().filter((n) => /^ppt\/slides\/slide\d+\.xml$/.test(n)).sort(numericOrder);
  const out = [];
  let i = 0;
  for (const s of slides) {
    const doc = parseXml(await zip.text(s));
    const paragraphs = byLocalName(doc, 'p').map((p) => byLocalName(p, 't').map((t) => t.textContent).join(''));
    out.push('## Slide ' + (++i) + '\n' + paragraphs.filter((p) => p.trim()).join('\n'));
  }
  return out.join('\n\n');
}

function columnIndex(ref) {
  const letters = String(ref).replace(/[^A-Z]/g, '');
  let n = 0;
  for (const c of letters) n = n * 26 + (c.charCodeAt(0) - 64);
  return n - 1;
}

async function xlsxText(zip) {
  const shared = [];
  const sst = await zip.text('xl/sharedStrings.xml');
  if (sst) {
    for (const si of byLocalName(parseXml(sst), 'si')) shared.push(byLocalName(si, 't').map((t) => t.textContent).join(''));
  }
  const names = [];
  const wb = await zip.text('xl/workbook.xml');
  if (wb) for (const s of byLocalName(parseXml(wb), 'sheet')) names.push(s.getAttribute('name'));
  const sheets = zip.names().filter((n) => /^xl\/worksheets\/sheet\d+\.xml$/.test(n)).sort(numericOrder);
  const out = [];
  sheets.forEach((s, k) => out.push(s));
  const result = [];
  for (let k = 0; k < out.length; ++k) {
    const doc = parseXml(await zip.text(out[k]));
    const lines = [];
    for (const row of byLocalName(doc, 'row')) {
      const cells = [];
      for (const c of byLocalName(row, 'c')) {
        const type = c.getAttribute('t');
        const v = byLocalName(c, 'v')[0];
        let value = '';
        if (type === 's' && v) value = shared[parseInt(v.textContent, 10)] || '';
        else if (type === 'inlineStr') value = byLocalName(c, 't').map((t) => t.textContent).join('');
        else if (v) value = v.textContent;
        cells[columnIndex(c.getAttribute('r'))] = value.includes(',') || value.includes('"')
          ? '"' + value.replace(/"/g, '""') + '"' : value;
      }
      lines.push(Array.from(cells, (x) => x || '').join(','));
    }
    result.push('## ' + (names[k] || 'Sheet ' + (k + 1)) + '\n' + lines.join('\n'));
  }
  return result.join('\n\n');
}

async function odfText(zip) {
  const doc = parseXml(await zip.text('content.xml'));
  const out = [];
  const walk = (n) => {
    for (const c of n.childNodes) {
      if (c.nodeType !== 1) continue;
      if (c.localName === 'h' || c.localName === 'p') out.push(c.textContent);
      else if (c.localName === 'table-row') {
        out.push(byLocalName(c, 'table-cell').map((cell) => cell.textContent.trim()).join(' | '));
      } else walk(c);
    }
  };
  walk(doc.documentElement);
  return out.join('\n');
}

let pdfjs = null;

async function pdfText(buffer) {
  if (!pdfjs) {
    pdfjs = await import('../vendor/pdfjs/pdf.min.mjs');
    pdfjs.GlobalWorkerOptions.workerSrc = new URL('../vendor/pdfjs/pdf.worker.min.mjs', import.meta.url).href;
  }
  const doc = await pdfjs.getDocument({ data: new Uint8Array(buffer), isEvalSupported: false, useSystemFonts: true }).promise;
  const pages = [];
  for (let i = 1; i <= doc.numPages; ++i) {
    const page = await doc.getPage(i);
    const content = await page.getTextContent();
    let s = '';
    for (const item of content.items) {
      if (typeof item.str !== 'string') continue;
      s += item.str;
      if (item.hasEOL) s += '\n';
      else if (item.str && !item.str.endsWith(' ')) s += ' ';
    }
    pages.push('## Page ' + i + '\n' + s.trim());
    if (pages.join('').length > maxTextChars) break;
  }
  await doc.destroy();
  return pages.join('\n\n');
}

function rtfText(s) {
  return s.replace(/\\par[d]?/g, '\n').replace(/\{\\\*[^{}]*\}/g, '').replace(/\\'([0-9a-f]{2})/gi,
    (m, h) => String.fromCharCode(parseInt(h, 16))).replace(/\\[a-z]+-?\d* ?/gi, '').replace(/[{}]/g, '').trim();
}

function htmlText(s) {
  const doc = new DOMParser().parseFromString(s, 'text/html');
  for (const n of doc.querySelectorAll('script, style, noscript')) n.remove();
  return (doc.body ? doc.body.innerText || doc.body.textContent : '').replace(/\n{3,}/g, '\n\n').trim();
}

/*
    The text of a file: {text, format}, or an exception whose message says why it cannot
    be read.
*/
export async function extractText(file) {
  if (file.size > maxFileBytes) throw new Error('too-large');
  const ext = extension(file.name);
  const buffer = await file.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  let text = null;
  let format = ext.toUpperCase() || 'TXT';
  if (ext === 'docx' || ext === 'docm') text = await docxText(await unzip(buffer));
  else if (ext === 'pptx' || ext === 'pptm') text = await pptxText(await unzip(buffer));
  else if (ext === 'xlsx' || ext === 'xlsm') text = await xlsxText(await unzip(buffer));
  else if (ext === 'odt' || ext === 'ods' || ext === 'odp') text = await odfText(await unzip(buffer));
  else if (ext === 'pdf') text = await pdfText(buffer);
  else if (ext === 'rtf') text = rtfText(decodeText(bytes));
  else if (ext === 'html' || ext === 'htm' || ext === 'xhtml') text = htmlText(decodeText(bytes));
  else if (textExtensions.has(ext) || (file.type || '').startsWith('text/') || !looksBinary(bytes)) {
    text = decodeText(bytes);
    format = ext ? ext.toUpperCase() : 'TXT';
  } else {
    throw new Error('unsupported');
  }
  if (text.length > maxTextChars) text = text.slice(0, maxTextChars);
  return { text, format };
}

// Splits a text into overlapping passages for the search of the documents.
export function passages(text, size = 1200, overlap = 200) {
  const out = [];
  for (let i = 0; i < text.length; i += size - overlap) {
    out.push({ start: i, text: text.slice(i, i + size) });
    if (i + size >= text.length) break;
  }
  return out;
}

const stopWords = new Set(('the a an and or of to in on for with is are was were be by as at it this that from not ' +
  'le la les un une des et ou de du en dans pour par sur avec est sont au aux ce cette qui que ' +
  'der die das und oder ein eine zu im mit ist sind von den dem il lo gli e di da per con che ' +
  'el los las y o del para por con es son que').split(' '));

function terms(s) {
  return String(s).toLowerCase().normalize('NFKD').replace(/[̀-ͯ]/g, '').split(/[^\p{L}\p{N}_]+/u)
    .filter((w) => w.length > 1 && !stopWords.has(w));
}

/*
    The passages of a set of documents closest to a query, by the BM25 score of their
    terms: [{name, start, text, score}], best first.
*/
export function searchPassages(documents, query, limit = 5) {
  const q = Array.from(new Set(terms(query)));
  if (!q.length) return [];
  const all = [];
  for (const d of documents) for (const p of passages(d.text || '')) all.push({ name: d.name, start: p.start, text: p.text });
  if (!all.length) return [];
  const df = new Map();
  const tfs = all.map((p) => {
    const tf = new Map();
    const words = terms(p.text);
    for (const w of words) tf.set(w, (tf.get(w) || 0) + 1);
    for (const w of tf.keys()) df.set(w, (df.get(w) || 0) + 1);
    p.length = words.length;
    return tf;
  });
  const avg = all.reduce((s, p) => s + p.length, 0) / all.length || 1;
  const k1 = 1.2, b = 0.75, n = all.length;
  all.forEach((p, i) => {
    let score = 0;
    for (const w of q) {
      const f = tfs[i].get(w) || 0;
      if (!f) continue;
      const idf = Math.log(1 + (n - (df.get(w) || 0) + 0.5) / ((df.get(w) || 0) + 0.5));
      score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * p.length / avg));
    }
    p.score = score;
  });
  return all.filter((p) => p.score > 0).sort((a, b) => b.score - a.score).slice(0, limit)
    .map(({ name, start, text, score }) => ({ name, start, text, score }));
}
