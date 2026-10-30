// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

/*
    The interpreter of the agent.  This page runs in a sandboxed frame of its own origin,
    without network (policy of the server) and without access to the data of the
    application; each run gets a new worker, terminated when it ends or exceeds its time.
*/
(function () {
  'use strict';
  var workerSource = [
    '"use strict";',
    // The interfaces that reach outside the worker are removed before any code runs.
    '["fetch","XMLHttpRequest","WebSocket","EventSource","importScripts","indexedDB","caches","BroadcastChannel",',
    ' "Worker","SharedWorker","WebTransport","RTCPeerConnection","navigator"].forEach(function (name) {',
    '  try { var o = self; while (o) { if (Object.prototype.hasOwnProperty.call(o, name)) delete o[name];',
    '        o = Object.getPrototypeOf(o); } } catch (e) {}',
    '  try { Object.defineProperty(self, name, { value: undefined, configurable: false, writable: false }); } catch (e) {}',
    '});',
    'function show(v) {',
    '  if (typeof v === "string") return v;',
    '  if (typeof v === "undefined") return "undefined";',
    '  if (typeof v === "function") return v.toString();',
    '  if (typeof v === "bigint") return v.toString() + "n";',
    '  try { var s = JSON.stringify(v, null, 2); return s === undefined ? String(v) : s; } catch (e) { return String(v); }',
    '}',
    // The output stops at limit characters, the text the page keeps of a run.
    'var limit = 20000;',
    'function cut(s) { return s.length > limit ? s.slice(0, limit) + " [...]" : s; }',
    'self.onmessage = function (e) {',
    '  var logs = [], size = 0;',
    '  var print = function () {',
    '    if (size > limit || logs.length >= 2000) return;',
    '    var line = cut(Array.prototype.map.call(arguments, show).join(" "));',
    '    size += line.length + 1;',
    '    logs.push(line);',
    '    if (size > limit) logs.push("[...]");',
    '  };',
    '  self.console = { log: print, info: print, warn: print, error: print, debug: print, table: print };',
    '  var done = function (ok, value, error) {',
    '    try {',
    '      postMessage({ ok: ok, logs: logs, result: ok && value !== undefined ? cut(show(value)) : undefined,',
    '                    error: error === undefined ? undefined : cut(String(error)) });',
    '    } catch (err) {',
    '      postMessage({ ok: false, logs: logs, error: "The result of the code could not be read." });',
    '    }',
    '  };',
    '  var code = String(e.data.code);',
    '  var value;',
    '  try {',
    '    value = (0, eval)(code);',
    '  } catch (err) {',
    '    if (err instanceof SyntaxError && /await|return/.test(code)) {',
    '      try { value = (0, eval)("(async function () {\\n" + code + "\\n})()"); }',
    '      catch (err2) { done(false, undefined, String(err2 && err2.stack || err2)); return; }',
    '    } else { done(false, undefined, String(err && err.stack || err)); return; }',
    '  }',
    '  Promise.resolve(value).then(function (v) { done(true, v); },',
    '    function (err) { done(false, undefined, String(err && err.stack || err)); });',
    '};'
  ].join('\n');
  var url = URL.createObjectURL(new Blob([workerSource], { type: 'text/javascript' }));
  var maxChars = 20000;
  var text = function (v) {
    return typeof v === 'string' ? (v.length > maxChars ? v.slice(0, maxChars) : v) : undefined;
  };
  /*
      The reply to the page, built from the fields of what the worker posted that are strings,
      truncated: the code it runs may post anything itself, a number, null, a reply of its own
      or objects that cannot be sent on.
  */
  var reply = function (id, data) {
    try {
      var d = data !== null && typeof data === 'object' ? data : {};
      var logs = [];
      var size = 0;
      if (Array.isArray(d.logs)) {
        for (var i = 0; i < d.logs.length && logs.length < 2000 && size <= maxChars; ++i) {
          var line = text(d.logs[i]);
          if (line === undefined) continue;
          size += line.length + 1;
          logs.push(line);
        }
      }
      var ok = d.ok === true;
      return {
        id: id, ok: ok, logs: logs, result: ok ? text(d.result) : undefined,
        error: ok ? undefined : text(d.error) || 'The code posted a message that is not a result.'
      };
    } catch (err) {
      return { id: id, ok: false, logs: [], error: 'The result of the code could not be read.' };
    }
  };
  window.addEventListener('message', function (e) {
    if (e.source !== window.parent || !e.data || typeof e.data.code !== 'string') return;
    var id = e.data.id;
    var worker = new Worker(url);
    var finished = false;
    var finish = function (data) {
      if (finished) return;
      finished = true;
      clearTimeout(timer);
      worker.terminate();
      window.parent.postMessage(reply(id, data), '*');
    };
    var timer = setTimeout(function () {
      finish({ ok: false, logs: [], error: 'The code did not finish within the time limit.' });
    }, Math.min(Math.max(e.data.timeout || 15000, 1000), 60000));
    worker.onmessage = function (m) { finish(m.data); };
    worker.onmessageerror = function () { finish(null); };
    worker.onerror = function (ev) {
      ev.preventDefault();
      finish({ ok: false, logs: [], error: ev.message || 'error' });
    };
    worker.postMessage({ code: e.data.code });
  });
  window.parent.postMessage({ ready: true }, '*');
})();
