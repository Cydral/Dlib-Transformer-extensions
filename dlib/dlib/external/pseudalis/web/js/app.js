// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

// The web application of Pseudalis: files and texts protected and restored by the local service.
(function () {
  'use strict';

  const texts = window.PSEUDALIS_TEXTS;
  let language = 'en';
  // The text card is ready: its marks follow the language (renderMarks()).
  let textReady = false;

  /* The language of the page: the one chosen in its menu, else the first language of the
     browser, which follows the system unless its user chose otherwise, that the application
     knows, else English. */
  function pick() {
    let saved = null;
    try { saved = localStorage.getItem('pseudalis.language'); } catch (e) { /* storage unavailable */ }
    if (saved && texts[saved]) return saved;
    const wanted = navigator.languages && navigator.languages.length ? navigator.languages
      : [navigator.language || 'en'];
    for (let i = 0; i < wanted.length; ++i) {
      const code = String(wanted[i] || '').slice(0, 2).toLowerCase();
      if (texts[code]) return code;
    }
    return 'en';
  }

  function t(key, values) {
    let s = (texts[language] && texts[language][key]) || texts.en[key] || key;
    if (values) Object.keys(values).forEach(function (k) { s = s.split('{' + k + '}').join(values[k]); });
    return s;
  }

  /* A text that counts n things: its singular form, key_one, for one thing, and for none in French,
     else its plural form. */
  function tn(key, n, values) {
    const one = n === 1 || (language === 'fr' && n === 0);
    const k = one && texts[language] && texts[language][key + '_one'] ? key + '_one' : key;
    return t(k, Object.assign({ n: n }, values || {}));
  }

  function translate() {
    document.documentElement.lang = language;
    document.querySelectorAll('[data-t]').forEach(function (e) { e.textContent = t(e.getAttribute('data-t')); });
    if (textReady) renderMarks();
    refreshStatus();
    if (admin) {
      renderRecords();
      renderTerms();
    }
  }

  /* The administration token, which the icon gives in the fragment of the address it opens: the
     browser never sends a fragment, and the address loses it at once. */
  let admin = '';
  (function () {
    const m = /(?:^#|&)admin=([0-9A-Za-z_-]+)/.exec(location.hash || '');
    try {
      if (m) sessionStorage.setItem('pseudalis.admin', m[1]);
      admin = sessionStorage.getItem('pseudalis.admin') || '';
    } catch (e) {
      admin = m ? m[1] : '';
    }
    if (m) history.replaceState(null, '', location.pathname + location.search);
  })();

  function adminFetch(route, options) {
    const o = Object.assign({ cache: 'no-store' }, options || {});
    o.headers = Object.assign({}, o.headers || {}, { 'X-Pseudalis-Admin': admin });
    return fetch(route, o);
  }

  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  }

  // ---------------------------------------------------------------------------------------
  // The status of the service.

  function pill(label, value, ok) {
    const p = el('span', 'pill' + (ok === true ? ' ok' : ok === false ? ' off' : ''));
    p.appendChild(document.createTextNode(label + ' '));
    p.appendChild(el('b', '', value));
    return p;
  }

  async function refreshStatus() {
    const box = document.getElementById('status');
    try {
      const r = await fetch('v1/info', { cache: 'no-store' });
      const info = await r.json();
      box.textContent = '';
      document.getElementById('version').textContent = [info.edition, info.version].filter(Boolean).join(' ');
      box.appendChild(pill(t('s_version'), info.version));
      box.appendChild(pill(t('s_engine'), (info.engine && info.engine.variant) || '?'));
      const d = info.drive || {};
      box.appendChild(pill(t('s_drive'), d.mounted ? d.letter + ':' : t('s_drive_off'), !!d.mounted));
      const a = info.albireo || {};
      box.appendChild(pill(t('s_albireo'), a.found ? a.model : t('s_albireo_off'), !!a.found));
      box.appendChild(pill(t('s_records'), String(info.records || 0)));
      const h = info.history || {};
      if (h.limit) box.appendChild(pill(t('s_history'), size(h.bytes || 0) + ' / ' + size(h.limit)));
    } catch (e) {
      box.textContent = '';
      box.appendChild(pill('', t('s_offline'), false));
    }
  }

  // ---------------------------------------------------------------------------------------
  // Files.

  function bytesOf(base64) {
    const s = atob(base64);
    const b = new Uint8Array(s.length);
    for (let i = 0; i < s.length; ++i) b[i] = s.charCodeAt(i);
    return b;
  }

  /* The failure of a request in the language of the page: the text of its code, with its values
     (a date written in the manner of the language, sizes in megabytes), else the English text
     of the service. */
  function errorText(body, status) {
    if (body && body.code && texts.en['err_' + body.code]) {
      const values = { e: body.error || '' };
      const args = body.args || {};
      Object.keys(args).forEach(function (k) { values[k] = String(args[k]); });
      if (/^\d{4}-\d{2}-\d{2}$/.test(values.date || '')) {
        const d = new Date(values.date + 'T12:00:00');
        if (!isNaN(d.getTime())) values.date = d.toLocaleDateString(language);
      }
      const sizes = body.sizes || {};
      Object.keys(sizes).forEach(function (k) { values[k] = String(Math.ceil((Number(sizes[k]) || 0) / 1048576)); });
      return t('err_' + body.code, values);
    }
    if (body && body.error) return body.error;
    return t('err_http', { status: status });
  }

  /* Sends an item, {name, read}, whose read() gives its bytes: a file, or the ZIP archive of a
     folder, which the service processes whole and gives back as an archive. */
  async function send(item, route, list, describe) {
    const li = el('li');
    const busy = el('span', 'busy');
    const grow = el('div', 'grow');
    grow.appendChild(el('div', 'name', item.name));
    const detail = el('div', 'detail', t('working'));
    grow.appendChild(detail);
    li.appendChild(busy);
    li.appendChild(grow);
    list.prepend(li);
    let data;
    try {
      data = await item.read();
    } catch (e) {
      busy.remove();
      detail.className = 'error';
      detail.textContent = t(e && e.message === 'folder_too_big' ? 'folder_too_big' : 'read_failed');
      return;
    }
    try {
      const r = await fetch(route + '&name=' + encodeURIComponent(item.name), {
        method: 'POST',
        headers: { 'Content-Type': 'application/octet-stream' },
        body: data
      });
      const body = await r.json().catch(function () { return null; });
      busy.remove();
      if (!r.ok || !body) {
        detail.className = 'error';
        detail.textContent = errorText(body, r.status);
        return;
      }
      grow.firstChild.textContent = body.name;
      detail.textContent = describe(body);
      const url = URL.createObjectURL(new Blob([bytesOf(body.data)], { type: 'application/octet-stream' }));
      const a = el('a', 'button', t('download'));
      a.href = url;
      a.download = body.name;
      li.appendChild(a);
    } catch (e) {
      // The request did not reach the service, or its answer was cut.
      busy.remove();
      detail.className = 'error';
      detail.textContent = t('s_offline');
    }
  }

  // ---------------------------------------------------------------------------------------
  // Folders: their files and their folders, with their paths, in a ZIP archive.

  const crcTable = (function () {
    const table = new Uint32Array(256);
    for (let n = 0; n < 256; ++n) {
      let c = n;
      for (let k = 0; k < 8; ++k) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1;
      table[n] = c >>> 0;
    }
    return table;
  })();

  function crc32(bytes) {
    let c = 0xFFFFFFFF;
    for (let i = 0; i < bytes.length; ++i) c = crcTable[(c ^ bytes[i]) & 0xFF] ^ (c >>> 8);
    return (c ^ 0xFFFFFFFF) >>> 0;
  }

  /* A ZIP archive of the entries {path, bytes, folder}, stored without compression, their names
     in UTF-8: the service reads it, and writes the archive it gives back compressed. */
  function zipOf(entries) {
    const encoder = new TextEncoder();
    const now = new Date();
    const time = (now.getHours() << 11) | (now.getMinutes() << 5) | (now.getSeconds() >> 1);
    const date = ((now.getFullYear() - 1980) << 9) | ((now.getMonth() + 1) << 5) | now.getDate();
    const parts = [];
    const central = [];
    let offset = 0, size = 0;
    entries.forEach(function (e) {
      const name = encoder.encode(e.path);
      const crc = crc32(e.bytes);
      const local = new DataView(new ArrayBuffer(30));
      local.setUint32(0, 0x04034b50, true);
      local.setUint16(4, 20, true);
      local.setUint16(6, 0x0800, true);
      local.setUint16(10, time, true);
      local.setUint16(12, date, true);
      local.setUint32(14, crc, true);
      local.setUint32(18, e.bytes.length, true);
      local.setUint32(22, e.bytes.length, true);
      local.setUint16(26, name.length, true);
      parts.push(local.buffer, name, e.bytes);
      const c = new DataView(new ArrayBuffer(46));
      c.setUint32(0, 0x02014b50, true);
      c.setUint16(4, 20, true);
      c.setUint16(6, 20, true);
      c.setUint16(8, 0x0800, true);
      c.setUint16(12, time, true);
      c.setUint16(14, date, true);
      c.setUint32(16, crc, true);
      c.setUint32(20, e.bytes.length, true);
      c.setUint32(24, e.bytes.length, true);
      c.setUint16(28, name.length, true);
      c.setUint32(38, e.folder ? 0x10 : 0, true);
      c.setUint32(42, offset, true);
      central.push(c.buffer, name);
      offset += 30 + name.length + e.bytes.length;
      size += 46 + name.length;
    });
    const end = new DataView(new ArrayBuffer(22));
    end.setUint32(0, 0x06054b50, true);
    end.setUint16(8, entries.length, true);
    end.setUint16(10, entries.length, true);
    end.setUint32(12, size, true);
    end.setUint32(16, offset, true);
    return new Blob(parts.concat(central, [end.buffer]), { type: 'application/zip' });
  }

  // The names that a folder holds for the system or for a program at work, left out.
  function passed(name) {
    const n = name.toLowerCase();
    return n === 'desktop.ini' || n === 'thumbs.db' || n === 'ehthumbs.db' || n === '.ds_store' ||
      n.indexOf('~$') === 0;
  }

  // The archive of the items {path, file} or {path, folder: true}, as an ArrayBuffer.
  async function archiveOf(items) {
    // The limits of a ZIP archive without its 64-bit extensions.
    if (items.length > 65535) throw new Error('folder_too_big');
    const entries = [];
    let total = 0;
    for (let i = 0; i < items.length; ++i) {
      const it = items[i];
      if (it.folder) {
        entries.push({ path: it.path, bytes: new Uint8Array(0), folder: true });
        continue;
      }
      total += it.file.size + 76 + 2 * it.path.length * 3;
      if (total > 0xFFF00000) throw new Error('folder_too_big');
      entries.push({ path: it.path, bytes: new Uint8Array(await it.file.arrayBuffer()) });
    }
    return zipOf(entries).arrayBuffer();
  }

  // The files and the folders under a folder dropped, the folder itself first.
  async function collect(entry, prefix, out) {
    if (passed(entry.name)) return;
    if (entry.isFile) {
      const file = await new Promise(function (resolve, reject) { entry.file(resolve, reject); });
      out.push({ path: prefix + entry.name, file: file });
      return;
    }
    const path = prefix + entry.name + '/';
    out.push({ path: path, folder: true });
    const reader = entry.createReader();
    for (;;) {
      const batch = await new Promise(function (resolve, reject) { reader.readEntries(resolve, reject); });
      if (!batch.length) break;
      for (let i = 0; i < batch.length; ++i) await collect(batch[i], path, out);
    }
  }

  function fileItem(file) {
    return { name: file.name, read: function () { return file.arrayBuffer(); } };
  }

  function folderItem(entry) {
    return {
      name: entry.name + '.zip',
      read: async function () {
        const items = [];
        await collect(entry, '', items);
        return archiveOf(items);
      }
    };
  }

  // A folder chosen in the dialog: its files, which keep their paths under it.
  function chosenFolderItem(files) {
    const top = (files[0].webkitRelativePath || files[0].name).split('/')[0];
    return {
      name: top + '.zip',
      read: function () {
        return archiveOf(files.filter(function (f) { return !passed(f.name); }).map(function (f) {
          return { path: f.webkitRelativePath || f.name, file: f };
        }));
      }
    };
  }

  function protectDetail(body) {
    const report = body.report || {};
    let s = tn('values', report.total || 0);
    const dropped = (report.dropped || []).length;
    if (dropped) s += ', ' + tn('dropped', dropped);
    return s;
  }

  function restoreDetail(body) {
    if (body.restored === 'original') return t('restored_original');
    if (body.restored === 'picture') return t('restored_picture');
    return tn('restored_tags', (body.report && body.report.tags_restored) || 0);
  }

  function wire(dropId, inputId, folderId, run) {
    const drop = document.getElementById(dropId);
    const input = document.getElementById(inputId);
    const folder = document.getElementById(folderId);
    input.addEventListener('change', function () {
      Array.from(input.files).forEach(function (f) { run(fileItem(f)); });
      input.value = '';
    });
    folder.addEventListener('change', function () {
      const files = Array.from(folder.files);
      folder.value = '';
      if (files.length) run(chosenFolderItem(files));
    });
    ['dragenter', 'dragover'].forEach(function (k) {
      drop.addEventListener(k, function (e) { e.preventDefault(); drop.classList.add('over'); });
    });
    ['dragleave', 'drop'].forEach(function (k) {
      drop.addEventListener(k, function () { drop.classList.remove('over'); });
    });
    drop.addEventListener('drop', function (e) {
      e.preventDefault();
      // The entries are taken during the event: a folder among them becomes an archive.
      const files = Array.from(e.dataTransfer.files);
      const entries = Array.from(e.dataTransfer.items || []).filter(function (it) { return it.kind === 'file'; })
        .map(function (it) { return it.webkitGetAsEntry ? it.webkitGetAsEntry() : null; });
      if (entries.length !== files.length || !entries.every(Boolean)) {
        files.forEach(function (f) { run(fileItem(f)); });
        return;
      }
      entries.forEach(function (entry, i) { run(entry.isDirectory ? folderItem(entry) : fileItem(files[i])); });
    });
  }

  wire('protect-drop', 'protect-input', 'protect-folder', function (item) {
    const keep = document.getElementById('keep').checked;
    send(item, 'v1/anonymize?format=json&source=web&record=' + (keep ? 'true' : 'false'),
      document.getElementById('protect-results'), protectDetail).then(refreshStatus);
  });
  wire('restore-drop', 'restore-input', 'restore-folder', function (item) {
    send(item, 'v1/restore?format=json', document.getElementById('restore-results'), restoreDetail);
  });

  // ---------------------------------------------------------------------------------------
  // Texts.

  /* The text card: Protect and Restore write their result in place of the text, whose tags
     and values restored are coloured by kind on marks drawn under the text area; Show the values
     found lists them below. */
  const textArea = document.getElementById('text');
  const textMarks = document.getElementById('text-marks');
  const textLegend = document.getElementById('text-legend');
  // The tags of the pseudonymizer: [[KEYWORD:body#CCC]].
  const tagPattern = new RegExp('\\[\\[(PERSON|ORG|LOCATION|ADDRESS|EMAIL|PHONE|IBAN|CARD|NIR|SIRET|ID|IPV4|IPV6|' +
    'MAC|URL|DATE|PLATE|GEO|TITLE|GENDER|NATIONALITY):[^\\[\\]]{1,4096}?#[0-9A-Z]{3}\\]\\]', 'g');
  // The values just restored, in UTF-16 units of the text, until the text changes.
  let restoredMarks = null;

  function markKind(type) {
    if (type === 'ORG') return 'org';
    if (type === 'LOCATION' || type === 'ADDRESS' || type === 'GEO' || type === 'PLATE') return 'place';
    if (type === 'EMAIL' || type === 'PHONE' || type === 'URL' || type === 'IPV4' || type === 'IPV6' ||
        type === 'MAC') return 'contact';
    if (type === 'ID' || type === 'IBAN' || type === 'CARD' || type === 'NIR' || type === 'SIRET') return 'id';
    if (type === 'DATE') return 'date';
    return 'person';
  }

  /* Byte offsets of the UTF-8 text, as the service gives them, in UTF-16 units of the text as the
     text area holds it, every line break a LF: the LF of a CR LF takes no room of its own. */
  function unitSpans(text, spans) {
    const at = new Map([[0, 0]]);
    let bytes = 0, units = 0, previous = '';
    for (const ch of text) {
      const c = ch.codePointAt(0);
      bytes += c < 0x80 ? 1 : c < 0x800 ? 2 : c < 0x10000 ? 3 : 4;
      if (!(ch === '\n' && previous === '\r')) units += ch.length;
      previous = ch;
      at.set(bytes, units);
    }
    const out = [];
    (spans || []).forEach(function (s) {
      const b = at.get(s.begin), e = at.get(s.end);
      if (b !== undefined && e !== undefined && e > b) out.push({ begin: b, end: e, type: s.type });
    });
    return out;
  }

  function syncMarks() {
    // The scroll bar of the text area narrows its text: the marks take the same width.
    const bar = Math.max(0, textArea.offsetWidth - textArea.clientWidth - 2);
    textMarks.style.paddingRight = (10 + bar) + 'px';
    textMarks.scrollTop = textArea.scrollTop;
    textMarks.scrollLeft = textArea.scrollLeft;
  }

  function renderMarks() {
    const text = textArea.value;
    let spans = [];
    if (restoredMarks) {
      spans = restoredMarks;
    } else {
      tagPattern.lastIndex = 0;
      for (let m = tagPattern.exec(text); m; m = tagPattern.exec(text))
        spans.push({ begin: m.index, end: m.index + m[0].length, type: m[1] });
    }
    spans.sort(function (a, b) { return a.begin - b.begin; });
    textMarks.textContent = '';
    let pos = 0;
    spans.forEach(function (s) {
      if (s.begin < pos) return;
      textMarks.appendChild(document.createTextNode(text.slice(pos, s.begin)));
      const mark = el('mark', 'k-' + markKind(s.type) + (restoredMarks ? ' restored' : ''), text.slice(s.begin, s.end));
      textMarks.appendChild(mark);
      pos = s.end;
    });
    // A final line break keeps the height of a last empty line of the text area.
    textMarks.appendChild(document.createTextNode(text.slice(pos) + '\n'));
    textLegend.hidden = spans.length === 0;
    document.getElementById('text-legend-how').textContent =
      restoredMarks ? t('legend_restored') : t('legend_protected');
    syncMarks();
  }

  textArea.addEventListener('input', function () {
    restoredMarks = null;
    renderMarks();
  });
  textArea.addEventListener('scroll', syncMarks);
  if (window.ResizeObserver) new ResizeObserver(syncMarks).observe(textArea);
  textReady = true;

  async function textRoute(route) {
    const out = document.getElementById('text-result');
    let r;
    try {
      r = await fetch(route, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: textArea.value })
      });
    } catch (e) {
      out.hidden = false;
      out.textContent = t('s_offline');
      return;
    }
    const body = await r.json().catch(function () { return null; });
    if (!r.ok || !body) {
      out.hidden = false;
      out.textContent = errorText(body, r.status);
      return;
    }
    if (body.spans) {
      out.hidden = false;
      out.textContent = body.spans.length ? body.spans.map(function (s) {
        return s.type + '  ' + s.text + '  (' + Math.round(s.confidence * 100) + ' %)';
      }).join('\n') : t('none_found');
      return;
    }
    // Protect and Restore: the result takes the place of the text.
    out.hidden = true;
    textArea.value = body.text.replace(/\r\n?/g, '\n');
    restoredMarks = route === 'v1/text/restore' ? unitSpans(body.text, body.restored) : null;
    if (restoredMarks && !restoredMarks.length) restoredMarks = null;
    renderMarks();
  }

  document.getElementById('text-protect').addEventListener('click', function () { textRoute('v1/text/anonymize'); });
  document.getElementById('text-restore').addEventListener('click', function () { textRoute('v1/text/restore'); });
  document.getElementById('text-detect').addEventListener('click', function () { textRoute('v1/text/detect'); });

  // ---------------------------------------------------------------------------------------
  // Administration: the originals kept, the key, the settings.

  const records = [];
  let recordsTotal = 0;
  let fingerprint = '';

  function size(bytes) {
    const mb = bytes >= 1048576;
    try {
      return new Intl.NumberFormat(language, { style: 'unit', unit: mb ? 'megabyte' : 'kilobyte',
        maximumFractionDigits: 1 }).format(mb ? bytes / 1048576 : Math.max(0.1, bytes / 1024));
    } catch (e) {
      return mb ? (bytes / 1048576).toFixed(1) + ' MB' : (bytes / 1024).toFixed(1) + ' KB';
    }
  }

  function message(id, text, bad) {
    const p = document.getElementById(id);
    p.hidden = !text;
    p.textContent = text || '';
    p.className = 'message' + (bad ? ' bad' : '');
  }

  function save(blob, name) {
    const a = el('a');
    a.href = URL.createObjectURL(blob);
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(function () { URL.revokeObjectURL(a.href); }, 60000);
  }

  function renderRecords() {
    const body = document.getElementById('records');
    body.textContent = '';
    records.forEach(function (r) {
      const tr = el('tr');
      tr.appendChild(el('td', 'name', r.original_name));
      tr.appendChild(el('td', '', new Date(r.created).toLocaleString(language)));
      const source = el('td', '', t('source_' + r.source));
      // A file that left the drive: its original stays until a day, or without limit of time.
      if (r.forgotten) {
        const until = new Date(r.forgotten).toLocaleDateString(language);
        source.appendChild(el('div', 'note', t('records_removed', { d: until })));
      } else if (r.removed) {
        source.appendChild(el('div', 'note', t('records_removed_always')));
      }
      tr.appendChild(source);
      tr.appendChild(el('td', 'number', size(r.original_size)));
      const actions = el('td', 'actions');
      const get = el('button', 'secondary', t('btn_original'));
      get.addEventListener('click', function () { original(r); });
      const forget = el('button', 'secondary', t('btn_forget'));
      forget.addEventListener('click', function () { forgetRecord(r); });
      actions.appendChild(get);
      actions.appendChild(forget);
      tr.appendChild(actions);
      body.appendChild(tr);
    });
    document.getElementById('records-none').hidden = recordsTotal > 0;
    document.getElementById('records-more').hidden = records.length >= recordsTotal;
  }

  async function loadRecords(more) {
    let r;
    try {
      r = await adminFetch('v1/records?limit=50&offset=' + (more ? records.length : 0));
    } catch (e) {
      return false;
    }
    if (!r.ok) return false;
    const body = await r.json().catch(function () { return null; });
    if (!body || !Array.isArray(body.records)) return false;
    if (!more) records.length = 0;
    body.records.forEach(function (x) { records.push(x); });
    recordsTotal = body.total;
    renderRecords();
    return true;
  }

  async function original(r) {
    let answer;
    try {
      answer = await adminFetch('v1/records/' + r.id + '/original');
    } catch (e) {
      window.alert(t('s_offline'));
      return;
    }
    if (!answer.ok) {
      const body = await answer.json().catch(function () { return null; });
      window.alert(errorText(body, answer.status));
      return;
    }
    save(await answer.blob(), r.original_name);
  }

  async function forgetRecord(r) {
    if (!window.confirm(t('confirm_forget', { f: r.original_name }))) return;
    let answer;
    try {
      answer = await adminFetch('v1/records/' + r.id, { method: 'DELETE' });
    } catch (e) {
      window.alert(t('s_offline'));
      return;
    }
    if (!answer.ok) {
      const body = await answer.json().catch(function () { return null; });
      window.alert(errorText(body, answer.status));
    }
    await loadRecords(false);
    refreshStatus();
  }

  // A request of the administration with a JSON body: its answer, or status 0 when the service did not answer.
  async function sendJson(route, value, method) {
    let r;
    try {
      r = await adminFetch(route, {
        method: method || 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(value)
      });
    } catch (e) {
      return { ok: false, status: 0, body: null };
    }
    const body = await r.json().catch(function () { return null; });
    return { ok: r.ok && !!body, status: r.status, body: body };
  }

  function failure(r) {
    return r.status === 0 ? t('s_offline') : errorText(r.body, r.status);
  }

  async function exportKey() {
    const p1 = document.getElementById('pass1').value, p2 = document.getElementById('pass2').value;
    if (p1 !== p2) return message('key-message', t('passphrase_mismatch'), true);
    const r = await sendJson('v1/key/export', { passphrase: p1 });
    if (!r.ok) return message('key-message', failure(r), true);
    save(new Blob([r.body.key], { type: 'application/json' }), 'pseudalis-key-' + (fingerprint || 'export') + '.json');
    document.getElementById('pass1').value = document.getElementById('pass2').value = '';
    message('key-message', t('key_exported'));
  }

  async function importKey(file) {
    const passphrase = document.getElementById('pass1').value;
    const r = await sendJson('v1/key/import', { key: await file.text(), passphrase: passphrase });
    if (!r.ok) return message('key-message', failure(r), true);
    document.getElementById('pass1').value = document.getElementById('pass2').value = '';
    message('key-message', t('key_imported'));
  }

  async function loadSettings() {
    let r;
    try {
      r = await adminFetch('v1/settings');
    } catch (e) {
      return;
    }
    if (!r.ok) return;
    const s = await r.json().catch(function () { return null; });
    if (!s) return;
    document.getElementById('set-language').value = s.language || 'auto';
    document.getElementById('set-drive').value = s.drive_letter || '';
    document.getElementById('set-dates').value = s.date_order === 'mdy' ? 'mdy' : 'dmy';
    document.getElementById('set-max').value = s.max_file_mb;
    document.getElementById('set-removed').value = s.removed_days;
    document.getElementById('set-history').value = s.history_mb;
    document.getElementById('set-albireo').checked = s.albireo !== 'off';
    document.getElementById('set-updates').checked = !!s.check_updates;
  }

  // The settings that the card shows, by their name in the interface, with the key of their label.
  const settingLabels = {
    language: 'set_language', drive_letter: 'set_drive', date_order: 'set_dates', max_file_mb: 'set_max',
    removed_days: 'set_removed', history_mb: 'set_history', albireo: 'set_albireo', check_updates: 'set_updates'
  };

  // A whole number of a field, or null when it holds none: the service refuses it with its range.
  function wholeNumber(id) {
    const text = document.getElementById(id).value.trim();
    return /^\d+$/.test(text) ? parseInt(text, 10) : null;
  }

  async function saveSettings() {
    const r = await sendJson('v1/settings', {
      language: document.getElementById('set-language').value,
      drive_letter: document.getElementById('set-drive').value.trim().toUpperCase(),
      date_order: document.getElementById('set-dates').value,
      max_file_mb: wholeNumber('set-max'),
      removed_days: wholeNumber('set-removed'),
      history_mb: wholeNumber('set-history'),
      albireo: document.getElementById('set-albireo').checked ? 'auto' : 'off',
      check_updates: document.getElementById('set-updates').checked
    }, 'PUT');
    if (!r.ok) {
      const b = r.body || {};
      if (b.code === 'settings_value' && b.args && settingLabels[b.args.field]) {
        const range = b.args.min !== undefined ? t('range', { min: b.args.min, max: b.args.max }) : '';
        return message('settings-message', t('err_settings_value', { field: t(settingLabels[b.args.field]) }) +
          range, true);
      }
      return message('settings-message', failure(r), true);
    }
    message('settings-message', t(r.body.restart ? 'saved_restart' : 'saved'));
    loadSettings();
    refreshStatus();
  }

  // ---------------------------------------------------------------------------------------
  // The imposed terms: replaced with the tag of their type before anything else of a text.

  let terms = [];
  let termTypes = ['ORG', 'PERSON', 'LOCATION', 'ADDRESS', 'ID', 'EMAIL', 'PHONE', 'URL'];
  let termsMax = 100, termLength = 200;

  function renderTerms() {
    const body = document.getElementById('terms');
    if (!body) return;
    body.textContent = '';
    terms.forEach(function (term, i) {
      const tr = el('tr');
      if (term.bad) tr.className = 'bad';
      const cell = el('td');
      const input = el('input');
      input.type = 'text';
      input.maxLength = termLength;
      input.value = term.term;
      input.setAttribute('aria-label', t('col_term'));
      input.addEventListener('input', function () {
        term.term = input.value;
        term.bad = false;
        tr.className = '';
      });
      cell.appendChild(input);
      tr.appendChild(cell);
      const check = el('td', 'check');
      const box = el('input');
      box.type = 'checkbox';
      box.checked = !!term.case_sensitive;
      box.setAttribute('aria-label', t('col_case'));
      box.addEventListener('change', function () { term.case_sensitive = box.checked; });
      check.appendChild(box);
      tr.appendChild(check);
      const tag = el('td');
      const select = el('select');
      select.setAttribute('aria-label', t('col_tag'));
      termTypes.forEach(function (k) {
        const o = el('option', '', t('tag_' + k));
        o.value = k;
        select.appendChild(o);
      });
      select.value = termTypes.indexOf(term.type) >= 0 ? term.type : termTypes[0];
      term.type = select.value;
      select.addEventListener('change', function () { term.type = select.value; });
      tag.appendChild(select);
      tr.appendChild(tag);
      const actions = el('td', 'actions');
      const remove = el('button', 'secondary', t('btn_remove'));
      remove.addEventListener('click', function () {
        terms.splice(i, 1);
        renderTerms();
      });
      actions.appendChild(remove);
      tr.appendChild(actions);
      body.appendChild(tr);
    });
    document.getElementById('terms-none').hidden = terms.length > 0;
    document.getElementById('terms-add').disabled = terms.length >= termsMax;
    document.getElementById('terms-count').textContent = t('terms_count', { n: terms.length, max: termsMax });
  }

  async function loadTerms() {
    let r;
    try {
      r = await adminFetch('v1/terms');
    } catch (e) {
      return;
    }
    if (!r.ok) return;
    const body = await r.json().catch(function () { return null; });
    if (!body) return;
    if (Array.isArray(body.types) && body.types.length) termTypes = body.types;
    termsMax = body.max || termsMax;
    termLength = body.max_length || termLength;
    terms = (body.terms || []).map(function (x) {
      return { term: x.term, type: x.type, case_sensitive: !!x.case_sensitive };
    });
    renderTerms();
  }

  function addTerm() {
    if (terms.length >= termsMax) return;
    terms.push({ term: '', type: termTypes[0], case_sensitive: false });
    renderTerms();
    const inputs = document.querySelectorAll('#terms input[type=text]');
    if (inputs.length) inputs[inputs.length - 1].focus();
  }

  async function saveTerms() {
    // The rows left empty are no terms; the others keep their place for the errors.
    const rows = [];
    const list = [];
    terms.forEach(function (x, i) {
      x.bad = false;
      const text = (x.term || '').trim();
      if (!text) return;
      rows.push(i);
      list.push({ term: text, type: x.type, case_sensitive: !!x.case_sensitive });
    });
    const r = await sendJson('v1/terms', { terms: list }, 'PUT');
    if (!r.ok) {
      const b = r.body || {};
      const at = b.args && b.args.index !== undefined ? Number(b.args.index) : -1;
      const row = at >= 0 && at < rows.length ? rows[at] : -1;
      if (row >= 0) terms[row].bad = true;
      renderTerms();
      const known = ['terms_invalid', 'terms_too_many', 'terms_type', 'terms_duplicate'].indexOf(b.code) >= 0;
      return message('terms-message', known ? t('err_' + b.code, { n: row + 1, max: termsMax, length: termLength })
        : failure(r), true);
    }
    terms = (r.body.terms || []).map(function (x) {
      return { term: x.term, type: x.type, case_sensitive: !!x.case_sensitive };
    });
    renderTerms();
    message('terms-message', t('terms_saved', { n: terms.length }));
  }

  async function startAdmin() {
    if (!admin || !(await loadRecords(false))) return;
    ['records-card', 'key-card', 'settings-card', 'terms-card'].forEach(function (id) {
      document.getElementById(id).hidden = false;
    });
    const info = await fetch('v1/info', { cache: 'no-store' }).then(function (r) { return r.json(); })
      .catch(function () { return {}; });
    fingerprint = info.key_fingerprint || '';
    document.getElementById('fingerprint').textContent = fingerprint;
    loadSettings();
    loadTerms();
  }

  document.getElementById('records-more').addEventListener('click', function () { loadRecords(true); });
  document.getElementById('key-export').addEventListener('click', exportKey);
  document.getElementById('key-import').addEventListener('change', function (e) {
    const f = e.target.files[0];
    e.target.value = '';
    if (f) importKey(f);
  });
  document.getElementById('settings-save').addEventListener('click', saveSettings);
  document.getElementById('terms-add').addEventListener('click', addTerm);
  document.getElementById('terms-save').addEventListener('click', saveTerms);

  const select = document.getElementById('language');
  language = pick();
  select.value = language;
  select.addEventListener('change', function () {
    language = select.value;
    try { localStorage.setItem('pseudalis.language', language); } catch (e) { /* storage unavailable */ }
    translate();
  });
  translate();
  startAdmin();
  setInterval(refreshStatus, 15000);
})();
