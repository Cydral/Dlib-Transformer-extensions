// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

// The web application of Pseudalis: files and texts protected and restored by the local service.
(function () {
  'use strict';

  const texts = window.PSEUDALIS_TEXTS;
  let language = 'en';

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

  function translate() {
    document.documentElement.lang = language;
    document.querySelectorAll('[data-t]').forEach(function (e) { e.textContent = t(e.getAttribute('data-t')); });
    refreshStatus();
    if (admin) renderRecords();
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
      const off = d.error_code === 'winfsp_missing' ? t('s_drive_winfsp') : t('s_drive_off');
      box.appendChild(pill(t('s_drive'), d.mounted ? d.letter + ':' : off, !!d.mounted));
      const a = info.albireo || {};
      box.appendChild(pill(t('s_albireo'), a.found ? a.model : t('s_albireo_off'), !!a.found));
      box.appendChild(pill(t('s_records'), String(info.records || 0)));
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

  async function send(file, route, list, describe) {
    const li = el('li');
    const busy = el('span', 'busy');
    const grow = el('div', 'grow');
    grow.appendChild(el('div', 'name', file.name));
    const detail = el('div', 'detail', t('working'));
    grow.appendChild(detail);
    li.appendChild(busy);
    li.appendChild(grow);
    list.prepend(li);
    try {
      const data = await file.arrayBuffer();
      const r = await fetch(route + '&name=' + encodeURIComponent(file.name), {
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

  function protectDetail(body) {
    const report = body.report || {};
    let s = t('values', { n: report.total || 0 });
    const dropped = (report.dropped || []).length;
    if (dropped) s += ', ' + t('dropped', { n: dropped });
    return s;
  }

  function restoreDetail(body) {
    if (body.restored === 'original') return t('restored_original');
    if (body.restored === 'picture') return t('restored_picture');
    return t('restored_tags', { n: (body.report && body.report.tags_restored) || 0 });
  }

  function wire(dropId, inputId, run) {
    const drop = document.getElementById(dropId);
    const input = document.getElementById(inputId);
    input.addEventListener('change', function () {
      Array.from(input.files).forEach(run);
      input.value = '';
    });
    ['dragenter', 'dragover'].forEach(function (k) {
      drop.addEventListener(k, function (e) { e.preventDefault(); drop.classList.add('over'); });
    });
    ['dragleave', 'drop'].forEach(function (k) {
      drop.addEventListener(k, function () { drop.classList.remove('over'); });
    });
    drop.addEventListener('drop', function (e) {
      e.preventDefault();
      Array.from(e.dataTransfer.files).forEach(run);
    });
  }

  wire('protect-drop', 'protect-input', function (f) {
    const keep = document.getElementById('keep').checked;
    send(f, 'v1/anonymize?format=json&source=web&record=' + (keep ? 'true' : 'false'),
      document.getElementById('protect-results'), protectDetail).then(refreshStatus);
  });
  wire('restore-drop', 'restore-input', function (f) {
    send(f, 'v1/restore?format=json', document.getElementById('restore-results'), restoreDetail);
  });

  // ---------------------------------------------------------------------------------------
  // Texts.

  async function textRoute(route) {
    const out = document.getElementById('text-result');
    let r;
    try {
      r = await fetch(route, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: document.getElementById('text').value })
      });
    } catch (e) {
      out.hidden = false;
      out.textContent = t('s_offline');
      return;
    }
    const body = await r.json().catch(function () { return null; });
    out.hidden = false;
    if (!r.ok || !body) {
      out.textContent = errorText(body, r.status);
      return;
    }
    if (body.spans) {
      out.textContent = body.spans.length ? body.spans.map(function (s) {
        return s.type + '  ' + s.text + '  (' + Math.round(s.confidence * 100) + ' %)';
      }).join('\n') : t('none_found');
      return;
    }
    out.textContent = body.text;
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
      if (r.forgotten) {
        const until = new Date(r.forgotten).toLocaleDateString(language);
        source.appendChild(el('div', 'note', t('records_removed', { d: until })));
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
    const r = await adminFetch('v1/records?limit=50&offset=' + (more ? records.length : 0));
    if (!r.ok) return false;
    const body = await r.json();
    if (!more) records.length = 0;
    body.records.forEach(function (x) { records.push(x); });
    recordsTotal = body.total;
    renderRecords();
    return true;
  }

  async function original(r) {
    const answer = await adminFetch('v1/records/' + r.id + '/original');
    if (!answer.ok) {
      const body = await answer.json().catch(function () { return null; });
      window.alert(errorText(body, answer.status));
      return;
    }
    save(await answer.blob(), r.original_name);
  }

  async function forgetRecord(r) {
    if (!window.confirm(t('confirm_forget', { f: r.original_name }))) return;
    const answer = await adminFetch('v1/records/' + r.id, { method: 'DELETE' });
    if (!answer.ok) {
      const body = await answer.json().catch(function () { return null; });
      window.alert(errorText(body, answer.status));
    }
    await loadRecords(false);
    refreshStatus();
  }

  async function postJson(route, value) {
    const r = await adminFetch(route, {
      method: route === 'v1/settings' ? 'PUT' : 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(value)
    });
    const body = await r.json().catch(function () { return null; });
    return { ok: r.ok && !!body, status: r.status, body: body };
  }

  async function exportKey() {
    const p1 = document.getElementById('pass1').value, p2 = document.getElementById('pass2').value;
    if (p1 !== p2) return message('key-message', t('passphrase_mismatch'), true);
    const r = await postJson('v1/key/export', { passphrase: p1 });
    if (!r.ok) return message('key-message', errorText(r.body, r.status), true);
    save(new Blob([r.body.key], { type: 'application/json' }), 'pseudalis-key-' + (fingerprint || 'export') + '.json');
    document.getElementById('pass1').value = document.getElementById('pass2').value = '';
    message('key-message', t('key_exported'));
  }

  async function importKey(file) {
    const passphrase = document.getElementById('pass1').value;
    const r = await postJson('v1/key/import', { key: await file.text(), passphrase: passphrase });
    if (!r.ok) return message('key-message', errorText(r.body, r.status), true);
    document.getElementById('pass1').value = document.getElementById('pass2').value = '';
    message('key-message', t('key_imported'));
  }

  async function loadSettings() {
    const r = await adminFetch('v1/settings');
    if (!r.ok) return;
    const s = await r.json();
    document.getElementById('set-language').value = s.language || 'auto';
    document.getElementById('set-drive').value = s.drive_letter || '';
    document.getElementById('set-max').value = s.max_file_mb;
    document.getElementById('set-removed').value = s.removed_days;
    document.getElementById('set-albireo').checked = s.albireo !== 'off';
    document.getElementById('set-updates').checked = !!s.check_updates;
  }

  async function saveSettings() {
    const r = await postJson('v1/settings', {
      language: document.getElementById('set-language').value,
      drive_letter: document.getElementById('set-drive').value.trim().toUpperCase(),
      max_file_mb: parseInt(document.getElementById('set-max').value, 10) || 256,
      removed_days: parseInt(document.getElementById('set-removed').value, 10) || 30,
      albireo: document.getElementById('set-albireo').checked ? 'auto' : 'off',
      check_updates: document.getElementById('set-updates').checked
    });
    if (!r.ok) return message('settings-message', errorText(r.body, r.status), true);
    message('settings-message', t(r.body.restart ? 'saved_restart' : 'saved'));
    loadSettings();
  }

  async function startAdmin() {
    if (!admin || !(await loadRecords(false))) return;
    ['records-card', 'key-card', 'settings-card'].forEach(function (id) {
      document.getElementById(id).hidden = false;
    });
    const info = await fetch('v1/info', { cache: 'no-store' }).then(function (r) { return r.json(); })
      .catch(function () { return {}; });
    fingerprint = info.key_fingerprint || '';
    document.getElementById('fingerprint').textContent = fingerprint;
    loadSettings();
  }

  document.getElementById('records-more').addEventListener('click', function () { loadRecords(true); });
  document.getElementById('key-export').addEventListener('click', exportKey);
  document.getElementById('key-import').addEventListener('change', function (e) {
    const f = e.target.files[0];
    e.target.value = '';
    if (f) importKey(f);
  });
  document.getElementById('settings-save').addEventListener('click', saveSettings);

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
