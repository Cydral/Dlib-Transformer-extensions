// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

/*
    Text of the files attached to a discussion or to a project, extracted in the browser:
    plain text and source code, CSV, Markdown, Word, Excel and PowerPoint documents,
    OpenDocument files, PDF (through pdf.js), RTF and HTML.  The archives of the Office and
    OpenDocument formats are read with the decompression of the browser; nothing is sent to
    the service but the text, inside the messages.

    The text keeps the structure of its document in lines that start with "## ": one per
    page of a PDF and per page break of a Word document, per slide of a presentation, in
    the order of the presentation, per sheet of a workbook, in the order of its tabs.  The
    search of the documents names its passages after them, and the tools of the agent read
    a document part by part with them (outline() and partText()).
*/

export const maxFileBytes = 64 * 1024 * 1024;
export const maxTextChars = 4 * 1024 * 1024;
// Rows of a sheet kept in the text; the following ones are counted.
const maxSheetRows = 20000;

const textExtensions = new Set(('txt md markdown rst csv tsv json jsonl ndjson xml yaml yml toml ini cfg conf log env ' +
  'c h cc cpp cxx hpp hh hxx inl cu cuh py pyw pyi js mjs cjs ts tsx jsx java kt kts scala go rs swift m mm cs fs vb ' +
  'php rb pl pm lua r jl sh bash zsh fish ps1 psm1 bat cmd sql graphql proto cmake mk make gradle properties tex bib ' +
  'css scss less vue svelte dart hs ml clj ex exs erl asm s v sv vhd tf hcl dockerfile gitignore editorconfig nix srt ' +
  'vtt diff patch').split(' '));

// The binary formats of the Office versions before 2007, which the browser cannot read.
const legacyExtensions = new Set(['doc', 'dot', 'xls', 'xlt', 'ppt', 'pot', 'pps']);

// The formats this module reads, for the file pickers: its documents and the text files.
export function documentAccept() {
  const office = ['pdf', 'docx', 'docm', 'dotx', 'pptx', 'pptm', 'ppsx', 'potx', 'xlsx', 'xlsm', 'xltx', 'odt', 'ods',
    'odp', 'rtf', 'html', 'htm', 'xhtml'];
  return office.concat(Array.from(textExtensions)).map((e) => '.' + e).join(',') + ',text/*';
}

/*
    What a file brings to a model: "image", "audio" or "video" by its type or its extension,
    "document" otherwise, whose text extractText() reads.
*/
export function fileKind(file) {
  const type = (file.type || '').toLowerCase();
  const ext = extension(file.name);
  if (type.startsWith('image/') && type !== 'image/svg+xml') return 'image';
  if (type.startsWith('audio/')) return 'audio';
  if (type.startsWith('video/')) return 'video';
  if (['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp', 'tif', 'tiff', 'heic', 'avif'].includes(ext)) return 'image';
  if (['mp3', 'wav', 'ogg', 'oga', 'flac', 'm4a', 'aac', 'opus', 'wma'].includes(ext)) return 'audio';
  if (['mp4', 'mov', 'avi', 'mkv', 'webm', 'wmv'].includes(ext)) return 'video';
  return 'document';
}

export function extension(name) {
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
  for (let i = 0; i < n; ++i) if (bytes[i] === 0) return true;
  return false;
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
    has: (name) => entries.has(name),
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

function child(node, name) {
  for (const c of node.childNodes) if (c.nodeType === 1 && c.localName === name) return c;
  return null;
}

// The value of an attribute whatever its namespace prefix (w:val, w:type...).
function attr(node, name) {
  if (!node || !node.attributes) return null;
  for (const a of node.attributes) if (a.localName === name) return a.value;
  return null;
}

// The relationship an element names (r:id), distinct from its own id attribute.
function relId(node) {
  if (!node || !node.attributes) return null;
  for (const a of node.attributes) {
    if (a.localName === 'id' && a.namespaceURI && a.namespaceURI.includes('relationships')) return a.value;
  }
  return null;
}

function numericOrder(a, b) {
  const na = parseInt((a.match(/(\d+)\.xml$/) || [0, 0])[1], 10);
  const nb = parseInt((b.match(/(\d+)\.xml$/) || [0, 0])[1], 10);
  return na - nb;
}

// The relationships of a part: id -> {target, type}, the targets resolved from the folder of the part.
async function relationships(zip, part) {
  const slash = part.lastIndexOf('/');
  const dir = part.slice(0, slash + 1);
  const file = dir + '_rels/' + part.slice(slash + 1) + '.rels';
  const out = new Map();
  const text = await zip.text(file);
  if (!text) return out;
  for (const r of byLocalName(parseXml(text), 'Relationship')) {
    let target = r.getAttribute('Target') || '';
    if (r.getAttribute('TargetMode') === 'External') continue;
    if (target.startsWith('/')) target = target.slice(1);
    else {
      const parts = (dir + target).split('/');
      const resolved = [];
      for (const s of parts) {
        if (s === '..') resolved.pop();
        else if (s !== '.' && s !== '') resolved.push(s);
      }
      target = resolved.join('/');
    }
    out.set(r.getAttribute('Id'), { target, type: r.getAttribute('Type') || '' });
  }
  return out;
}

// Word.

const headingStyle = /^(heading|titre|berschrift|überschrift|titolo|título|titulo)\s*([1-9])$/i;

async function docxText(zip) {
  const out = [];
  let page = 1;
  let pageMarks = false;
  const lines = [];
  // A page starts at a break, unless nothing was written since the previous one: an explicit
  // break and the break Word records where it rendered a page fall at the same place.
  const pageBreak = () => {
    let text = false;
    for (let k = lines.length - 1; k >= 0 && !lines[k].page; --k) text = text || !!String(lines[k]).trim();
    if (!text) return;
    page += 1;
    pageMarks = true;
    lines.push({ page: true, n: page });
  };

  // The text of a paragraph, its page breaks as markers between its pieces.
  const paragraph = (p) => {
    const ppr = child(p, 'pPr');
    let prefix = '';
    const before = ppr ? child(ppr, 'pageBreakBefore') : null;
    if (before && !/^(0|false|off)$/i.test(attr(before, 'val') || '')) pageBreak();
    if (ppr) {
      const style = attr(child(ppr, 'pStyle'), 'val') || '';
      const outline = attr(child(ppr, 'outlineLvl'), 'val');
      const m = style.match(headingStyle);
      if (m) prefix = '#'.repeat(Math.min(6, parseInt(m[2], 10) + 1)) + ' ';
      else if (/^(title|titre|titel|titolo|título)$/i.test(style)) prefix = '# ';
      else if (outline != null && /^[0-8]$/.test(outline)) prefix = '#'.repeat(Math.min(6, parseInt(outline, 10) + 2)) + ' ';
      else if (child(ppr, 'numPr')) prefix = '- ';
    }
    let s = '';
    const flush = () => {
      if (s.trim()) lines.push(prefix + s);
      s = '';
    };
    const walk = (n) => {
      for (const c of n.childNodes) {
        if (c.nodeType !== 1) continue;
        const name = c.localName;
        if (name === 't') s += c.textContent;
        else if (name === 'tab') s += '\t';
        else if (name === 'br') {
          if (attr(c, 'type') === 'page') {
            flush();
            pageBreak();
          } else s += '\n';
        } else if (name === 'cr') s += '\n';
        else if (name === 'lastRenderedPageBreak') {
          flush();
          pageBreak();
        }
        // Deleted text and field codes are not text; Fallback repeats the content of Choice.
        else if (name === 'delText' || name === 'instrText' || name === 'Fallback' || name === 'pPr' || name === 'rPr') continue;
        else if (name === 'p') {
          // A paragraph of a text box, inside the paragraph that anchors it.
          s += '\n';
          walk(c);
          s += '\n';
        } else walk(c);
      }
    };
    walk(p);
    flush();
    // A section that starts on a new page.
    const sect = ppr ? child(ppr, 'sectPr') : null;
    const kind = sect ? attr(child(sect, 'type'), 'val') : null;
    if (sect && (!kind || kind === 'nextPage' || kind === 'oddPage' || kind === 'evenPage')) pageBreak();
  };

  const table = (tbl) => {
    for (const row of Array.from(tbl.childNodes).filter((n) => n.nodeType === 1 && n.localName === 'tr')) {
      const cells = [];
      for (const cell of Array.from(row.childNodes).filter((n) => n.nodeType === 1 && n.localName === 'tc')) {
        const before = lines.length;
        blocks(cell);
        const text = lines.splice(before).filter((l) => !l.page).join(' ').replace(/\s+/g, ' ').trim();
        cells.push(text);
      }
      if (cells.some((c) => c)) lines.push(cells.join(' | '));
    }
    lines.push('');
  };

  // The paragraphs and tables of a container, the content controls opened.
  const blocks = (container) => {
    for (const node of container.childNodes) {
      if (node.nodeType !== 1) continue;
      if (node.localName === 'p') paragraph(node);
      else if (node.localName === 'tbl') table(node);
      else if (node.localName === 'sdt') {
        const content = child(node, 'sdtContent');
        if (content) blocks(content);
      } else if (node.localName === 'customXml' || node.localName === 'smartTag') blocks(node);
    }
  };

  const doc = parseXml(await zip.text('word/document.xml'));
  blocks(byLocalName(doc, 'body')[0] || doc.documentElement);
  // The pages are marked when the document records where they break.
  let current = 1;
  if (pageMarks) out.push('## Page 1');
  for (const l of lines) {
    if (l.page) {
      // A break at the very end of the document opens no page.
      current = l.n;
      out.push('', '## Page ' + current);
    } else out.push(l);
  }
  while (out.length && /^## Page \d+$/.test(out[out.length - 1])) out.pop();

  // The parts outside the body: headers and footers, notes, comments.
  const extra = async (title, pattern, item) => {
    const texts = [];
    for (const name of zip.names().filter((n) => pattern.test(n)).sort(numericOrder)) {
      const part = parseXml(await zip.text(name));
      lines.length = 0;
      const items = item ? byLocalName(part, item) : [part.documentElement];
      for (const it of items) {
        // The separators of the notes carry no text.
        if (/separator|continuation/i.test(attr(it, 'type') || '')) continue;
        blocks(it);
      }
      const text = lines.filter((l) => !l.page).join('\n').trim();
      if (text && !texts.includes(text)) texts.push(text);
    }
    if (texts.length) out.push('', '## ' + title, texts.join('\n\n'));
  };
  await extra('Headers and footers', /^word\/(header|footer)\d*\.xml$/, null);
  await extra('Footnotes', /^word\/footnotes\.xml$/, 'footnote');
  await extra('Endnotes', /^word\/endnotes\.xml$/, 'endnote');
  await extra('Comments', /^word\/comments\.xml$/, 'comment');
  return out.join('\n').replace(/\n{3,}/g, '\n\n');
}

// PowerPoint.

function shapeTexts(root) {
  // The paragraphs of each shape and table, in the order of the slide; the tables cell by cell.
  const out = [];
  const walk = (n) => {
    for (const c of n.childNodes) {
      if (c.nodeType !== 1) continue;
      if (c.localName === 'tbl') {
        for (const row of byLocalName(c, 'tr')) {
          const cells = byLocalName(row, 'tc').map((cell) =>
            byLocalName(cell, 'p').map((p) => byLocalName(p, 't').map((t) => t.textContent).join('')).join(' ').trim());
          if (cells.some((x) => x)) out.push(cells.join(' | '));
        }
      } else if (c.localName === 'txBody') {
        for (const p of Array.from(c.childNodes).filter((x) => x.nodeType === 1 && x.localName === 'p')) {
          const text = byLocalName(p, 't').map((t) => t.textContent).join('');
          if (text.trim()) out.push(text);
        }
      } else if (c.localName !== 'Fallback') walk(c);
    }
  };
  walk(root);
  return out;
}

function placeholderType(sp) {
  const nv = byLocalName(sp, 'ph')[0];
  return nv ? nv.getAttribute('type') || 'body' : null;
}

async function chartText(zip, part) {
  const text = await zip.text(part);
  if (!text) return '';
  const doc = parseXml(text);
  const title = byLocalName(doc, 'title').map((t) => byLocalName(t, 't').map((x) => x.textContent).join('')).join(' ').trim();
  const series = [];
  for (const ser of byLocalName(doc, 'ser')) {
    const name = byLocalName(child(ser, 'tx') || ser, 'v').map((v) => v.textContent).join(' ').trim();
    if (name && !series.includes(name)) series.push(name);
  }
  const categories = [];
  for (const cat of byLocalName(doc, 'cat')) {
    for (const v of byLocalName(cat, 'v')) if (!categories.includes(v.textContent)) categories.push(v.textContent);
    break;
  }
  const parts = [];
  if (title) parts.push(title);
  if (series.length) parts.push('series: ' + series.join(', '));
  if (categories.length) parts.push('categories: ' + categories.slice(0, 40).join(', '));
  return parts.length ? 'Chart: ' + parts.join('; ') : '';
}

async function pptxText(zip) {
  // The slides in the order of the presentation, not of their files.
  let slides = [];
  const pres = await zip.text('ppt/presentation.xml');
  if (pres) {
    const rels = await relationships(zip, 'ppt/presentation.xml');
    for (const id of byLocalName(parseXml(pres), 'sldId')) {
      const r = rels.get(relId(id));
      if (r && zip.has(r.target)) slides.push(r.target);
    }
  }
  if (!slides.length) slides = zip.names().filter((n) => /^ppt\/slides\/slide\d+\.xml$/.test(n)).sort(numericOrder);
  const out = [];
  let i = 0;
  for (const s of slides) {
    const doc = parseXml(await zip.text(s));
    const hidden = doc.documentElement.getAttribute('show') === '0';
    let title = '';
    for (const sp of byLocalName(doc, 'sp')) {
      const type = placeholderType(sp);
      if (type === 'title' || type === 'ctrTitle') {
        title = shapeTexts(sp).join(' ').trim();
        break;
      }
    }
    const body = shapeTexts(doc.documentElement);
    if (title && body[0] && body[0].trim() === title) body.shift();
    const rels = await relationships(zip, s);
    const extras = [];
    for (const r of rels.values()) {
      if (/\/chart$/.test(r.type)) {
        const c = await chartText(zip, r.target);
        if (c) extras.push(c);
      } else if (/\/diagramData$/.test(r.type)) {
        // The text of a SmartArt graphic.
        const text = await zip.text(r.target);
        if (text) {
          const items = byLocalName(parseXml(text), 'p').map((p) => byLocalName(p, 't').map((t) => t.textContent).join(''))
            .filter((x) => x.trim());
          if (items.length) extras.push(items.join('\n'));
        }
      }
    }
    let notes = '';
    for (const r of rels.values()) {
      if (!/\/notesSlide$/.test(r.type)) continue;
      const text = await zip.text(r.target);
      if (!text) continue;
      const nd = parseXml(text);
      const parts = [];
      for (const sp of byLocalName(nd, 'sp')) {
        const type = placeholderType(sp);
        if (type === 'sldImg' || type === 'sldNum' || type === 'hdr' || type === 'ftr' || type === 'dt') continue;
        parts.push(...shapeTexts(sp));
      }
      notes = parts.join('\n').trim();
    }
    const head = '## Slide ' + (++i) + (title ? ': ' + title : '') + (hidden ? ' (hidden)' : '');
    const lines = [head].concat(body, extras);
    if (notes) lines.push('Notes: ' + notes);
    out.push(lines.join('\n'));
  }
  return out.join('\n\n');
}

// Excel.

const builtinDateFormats = new Set([14, 15, 16, 17, 18, 19, 20, 21, 22, 45, 46, 47]);

function isDateFormat(code) {
  // The quoted texts, the escaped characters and the bracketed colors and locales hold no date.
  const s = String(code).replace(/"[^"]*"/g, '').replace(/\\./g, '').replace(/\[(?!h\]|m\]|mm\]|s\]|ss\])[^\]]*\]/gi, '');
  return /[dmyhs]/i.test(s) && !/^general$/i.test(s.trim());
}

function serialDate(serial, date1904) {
  const base = date1904 ? Date.UTC(1904, 0, 1) : Date.UTC(1899, 11, 30);
  const ms = Math.round(serial * 86400000);
  const d = new Date(base + ms);
  const iso = d.toISOString();
  const time = iso.slice(11, 19);
  if (serial < 1) return time;
  return time === '00:00:00' ? iso.slice(0, 10) : iso.slice(0, 10) + ' ' + time.replace(/:00$/, '');
}

function csvField(value) {
  return /[",\n\r]/.test(value) ? '"' + value.replace(/"/g, '""') + '"' : value;
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
    for (const si of byLocalName(parseXml(sst), 'si')) {
      // The phonetic guides (rPh) repeat the text in another script.
      shared.push(byLocalName(si, 't').filter((t) => t.parentNode.localName !== 'rPh').map((t) => t.textContent).join(''));
    }
  }
  // The styles that show a date.
  const dateStyles = new Set();
  const styles = await zip.text('xl/styles.xml');
  if (styles) {
    const sd = parseXml(styles);
    const custom = new Map();
    for (const f of byLocalName(sd, 'numFmt')) custom.set(parseInt(f.getAttribute('numFmtId'), 10), f.getAttribute('formatCode'));
    const xfs = byLocalName(sd, 'cellXfs')[0];
    if (xfs) {
      Array.from(xfs.childNodes).filter((n) => n.nodeType === 1 && n.localName === 'xf').forEach((xf, k) => {
        const id = parseInt(xf.getAttribute('numFmtId') || '0', 10);
        if (builtinDateFormats.has(id) || (custom.has(id) && isDateFormat(custom.get(id)))) dateStyles.add(k);
      });
    }
  }
  // The sheets in the order of their tabs, through the relationships of the workbook.
  const sheets = [];
  let date1904 = false;
  const wb = await zip.text('xl/workbook.xml');
  if (wb) {
    const wd = parseXml(wb);
    const pr = byLocalName(wd, 'workbookPr')[0];
    date1904 = !!pr && /^(1|true)$/i.test(pr.getAttribute('date1904') || '');
    const rels = await relationships(zip, 'xl/workbook.xml');
    for (const s of byLocalName(wd, 'sheet')) {
      const r = rels.get(relId(s));
      sheets.push({ name: s.getAttribute('name'), state: s.getAttribute('state') || 'visible', part: r ? r.target : null });
    }
  }
  if (!sheets.some((s) => s.part && zip.has(s.part))) {
    zip.names().filter((n) => /^xl\/worksheets\/sheet\d+\.xml$/.test(n)).sort(numericOrder)
      .forEach((n, k) => sheets.push({ name: 'Sheet ' + (k + 1), state: 'visible', part: n }));
  }
  const result = [];
  for (const sheet of sheets) {
    if (!sheet.part || !zip.has(sheet.part)) continue;
    if (!/worksheets\//.test(sheet.part)) {
      // A chart sheet: its chart.
      const rels = await relationships(zip, sheet.part);
      const charts = [];
      for (const r of rels.values()) {
        if (/\/drawing$/.test(r.type)) {
          const drels = await relationships(zip, r.target);
          for (const d of drels.values()) if (/\/chart$/.test(d.type)) charts.push(await chartText(zip, d.target));
        }
      }
      result.push('## Sheet: ' + sheet.name + '\n' + charts.filter((c) => c).join('\n'));
      continue;
    }
    const doc = parseXml(await zip.text(sheet.part));
    const lines = [];
    let rows = 0, skipped = 0;
    for (const row of byLocalName(doc, 'row')) {
      const cells = [];
      for (const c of Array.from(row.childNodes).filter((n) => n.nodeType === 1 && n.localName === 'c')) {
        const type = c.getAttribute('t') || 'n';
        const v = child(c, 'v');
        let value = '';
        if (type === 's' && v) value = shared[parseInt(v.textContent, 10)] || '';
        else if (type === 'inlineStr') value = byLocalName(c, 't').map((t) => t.textContent).join('');
        else if (type === 'b' && v) value = v.textContent === '1' ? 'TRUE' : 'FALSE';
        else if (v) {
          value = v.textContent;
          const style = parseInt(c.getAttribute('s') || '0', 10);
          if (type === 'n' && dateStyles.has(style) && value !== '' && isFinite(Number(value))) {
            value = serialDate(Number(value), date1904);
          }
        } else if (child(c, 'f') && child(c, 'f').textContent) {
          // A formula never computed by a spreadsheet: its expression.
          value = '=' + child(c, 'f').textContent;
        }
        const ref = c.getAttribute('r');
        const k = ref ? columnIndex(ref) : cells.length;
        if (k >= 0 && k < 16384) cells[k] = csvField(value);
      }
      if (!cells.some((x) => x)) continue;
      if (rows >= maxSheetRows) {
        ++skipped;
        continue;
      }
      lines.push(Array.from(cells, (x) => x || '').join(','));
      ++rows;
    }
    if (skipped) lines.push('(' + skipped + ' more rows)');
    const state = sheet.state !== 'visible' ? ' (hidden)' : '';
    result.push('## Sheet: ' + sheet.name + state + '\n' + lines.join('\n'));
  }
  return result.join('\n\n');
}

// OpenDocument.

async function odfText(zip) {
  const doc = parseXml(await zip.text('content.xml'));
  const out = [];
  let slide = 0;
  const walk = (n) => {
    for (const c of n.childNodes) {
      if (c.nodeType !== 1) continue;
      if (c.localName === 'h' || c.localName === 'p') out.push(c.textContent);
      else if (c.localName === 'table-row') {
        out.push(byLocalName(c, 'table-cell').map((cell) => cell.textContent.trim()).join(' | '));
      } else if (c.localName === 'table' && c.namespaceURI && /table/.test(c.namespaceURI) && attr(c, 'name')) {
        out.push('', '## Sheet: ' + attr(c, 'name'));
        walk(c);
      } else if (c.localName === 'page' && attr(c, 'name')) {
        out.push('', '## Slide ' + (++slide) + ': ' + attr(c, 'name'));
        walk(c);
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
  let total = 0;
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
    total += s.length;
    page.cleanup();
    if (total > maxTextChars) break;
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
    be read: too-large, legacy (an Office format before 2007), unsupported.
*/
export async function extractText(file) {
  if (file.size > maxFileBytes) throw new Error('too-large');
  const ext = extension(file.name);
  if (legacyExtensions.has(ext)) throw new Error('legacy');
  const buffer = await file.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  let text = null;
  let format = ext.toUpperCase() || 'TXT';
  if (ext === 'docx' || ext === 'docm' || ext === 'dotx') text = await docxText(await unzip(buffer));
  else if (ext === 'pptx' || ext === 'pptm' || ext === 'ppsx' || ext === 'potx') text = await pptxText(await unzip(buffer));
  else if (ext === 'xlsx' || ext === 'xlsm' || ext === 'xltx') text = await xlsxText(await unzip(buffer));
  else if (ext === 'odt' || ext === 'ods' || ext === 'odp') text = await odfText(await unzip(buffer));
  else if (ext === 'pdf') text = await pdfText(buffer);
  else if (ext === 'rtf') text = rtfText(decodeText(bytes));
  else if (ext === 'html' || ext === 'htm' || ext === 'xhtml') text = htmlText(decodeText(bytes));
  else if (textExtensions.has(ext) || (file.type || '').startsWith('text/') || !looksBinary(bytes)) {
    text = decodeText(bytes);
    // A CSV keeps its rows; the separators other than the comma are read as they are.
    if (text.charCodeAt(0) === 0xfeff) text = text.slice(1);
    format = ext ? ext.toUpperCase() : 'TXT';
  } else {
    throw new Error('unsupported');
  }
  if (text.length > maxTextChars) text = text.slice(0, maxTextChars);
  return { text, format };
}

/*
    The parts of a text: its "## " lines (pages, slides, sheets, sections), each with the
    characters it covers.  A text without such lines is one part.
*/
export function outline(text) {
  const parts = [];
  const re = /^## (.+)$/gm;
  let m;
  while ((m = re.exec(text)) !== null) parts.push({ title: m[1].trim(), start: m.index });
  if (!parts.length || parts[0].start > 0 && text.slice(0, parts[0].start).trim()) {
    parts.unshift({ title: '', start: 0 });
  }
  for (let i = 0; i < parts.length; ++i) parts[i].end = i + 1 < parts.length ? parts[i + 1].start : text.length;
  return parts;
}

/*
    The text of a part named as outline() names it: "Slide 3", "Page 12", "Sheet: Budget",
    or the beginning of a section title; null when no part matches.
*/
export function partText(text, name) {
  const wanted = String(name || '').trim().toLowerCase().replace(/\s+/g, ' ');
  if (!wanted) return null;
  const parts = outline(text).filter((p) => p.title);
  const exact = parts.find((p) => p.title.toLowerCase() === wanted
    || p.title.toLowerCase().split(':')[0].replace(' (hidden)', '') === wanted);
  const found = exact || parts.find((p) => p.title.toLowerCase().startsWith(wanted))
    || parts.find((p) => p.title.toLowerCase().includes(wanted));
  return found ? text.slice(found.start, found.end).trim() : null;
}

const stopWords = new Set(('the a an and or of to in on for with is are was were be by as at it this that from not ' +
  'le la les un une des et ou de du en dans pour par sur avec est sont au aux ce cette qui que ' +
  'der die das und oder ein eine zu im mit ist sind von den dem il lo gli e di da per con che ' +
  'el los las y o del para por con es son que').split(' '));

export function terms(s) {
  return String(s).toLowerCase().normalize('NFKD').replace(/[̀-ͯ]/g, '').split(/[^\p{L}\p{N}_]+/u)
    .filter((w) => w.length > 1 && !stopWords.has(w));
}

/*
    BM25 scores of a query over passages [{text}]: an array of scores, in their order.  The
    statistics of the terms are computed over the passages given.
*/
export function bm25Scores(passages, query) {
  const q = Array.from(new Set(terms(query)));
  const scores = new Float64Array(passages.length);
  if (!q.length || !passages.length) return scores;
  const df = new Map();
  const tfs = [];
  const lengths = [];
  for (const p of passages) {
    const tf = new Map();
    const words = terms(p.text);
    for (const w of words) tf.set(w, (tf.get(w) || 0) + 1);
    for (const w of q) if (tf.has(w)) df.set(w, (df.get(w) || 0) + 1);
    tfs.push(tf);
    lengths.push(words.length);
  }
  const avg = lengths.reduce((s, x) => s + x, 0) / lengths.length || 1;
  const k1 = 1.2, b = 0.75, n = passages.length;
  for (let i = 0; i < passages.length; ++i) {
    let score = 0;
    for (const w of q) {
      const f = tfs[i].get(w) || 0;
      if (!f) continue;
      const d = df.get(w) || 0;
      const idf = Math.log(1 + (n - d + 0.5) / (d + 0.5));
      score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * lengths[i] / avg));
    }
    scores[i] = score;
  }
  return scores;
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

/*
    The passages of a set of documents closest to a query, by the BM25 score of their
    terms: [{name, start, text, score}], best first.  The search of the documents without
    an embedding model.
*/
export function searchPassages(documents, query, limit = 5) {
  const all = [];
  for (const d of documents) for (const p of passages(d.text || '')) all.push({ name: d.name, start: p.start, text: p.text });
  const scores = bm25Scores(all, query);
  all.forEach((p, i) => { p.score = scores[i]; });
  return all.filter((p) => p.score > 0).sort((a, b) => b.score - a.score).slice(0, limit)
    .map(({ name, start, text, score }) => ({ name, start, text, score }));
}
