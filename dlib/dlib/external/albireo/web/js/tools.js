// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

/*
    The tools of the agent mode, all executed in this page, without network: a calculator,
    a JavaScript interpreter isolated in a sandboxed frame, the search, the outline and the
    reading of the documents of the discussion and of its project (a presentation slide by
    slide, a Word document page by page, a workbook sheet by sheet), the search of the
    other discussions, the date, and the creation of files offered for download.
*/

import { outline, partText } from './documents.js';

export const definitions = [
  {
    type: 'function',
    function: {
      name: 'calculator',
      description: 'Evaluates an arithmetic expression exactly as written and returns its value. Supports + - * / % ^, ' +
        'parentheses, and the functions sqrt, cbrt, abs, exp, ln, log10, log2, sin, cos, tan, asin, acos, atan, ' +
        'floor, ceil, round, min, max, pow, and the constants pi and e.',
      parameters: {
        type: 'object',
        properties: { expression: { type: 'string', description: 'The expression, for example "(3.5 + 2) * 4 ^ 2".' } },
        required: ['expression']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'run_javascript',
      description: 'Runs JavaScript code in an isolated interpreter without network and returns what it prints with ' +
        'console.log and the value of its last expression. Use it for computations, data processing, parsing and ' +
        'checking code. The code may use async/await. Time limit: 15 seconds.',
      parameters: {
        type: 'object',
        properties: { code: { type: 'string', description: 'The JavaScript code to run.' } },
        required: ['code']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'search_documents',
      description: 'Searches the documents attached to this discussion and to its project, by meaning and by words, and ' +
        'returns the most relevant passages with their document and their place (page, slide, sheet).',
      parameters: {
        type: 'object',
        properties: {
          query: { type: 'string', description: 'The question or the words to look for.' },
          limit: { type: 'integer', description: 'Number of passages, 5 by default, at most 10.' }
        },
        required: ['query']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'document_outline',
      description: 'Lists the parts of a document attached to this discussion or to its project: the pages of a PDF or ' +
        'of a Word document, the slides of a presentation with their titles, the sheets of a workbook, the sections ' +
        'of a text, each with its size. Use it to go through a document part by part with read_document.',
      parameters: {
        type: 'object',
        properties: { name: { type: 'string', description: 'Name of the document.' } },
        required: ['name']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'read_document',
      description: 'Reads a document attached to this discussion or to its project: one part named as document_outline ' +
        'names it ("Slide 3", "Page 12", "Sheet: Budget"), or the characters from an offset.',
      parameters: {
        type: 'object',
        properties: {
          name: { type: 'string', description: 'Name of the document.' },
          part: { type: 'string', description: 'The part to read, for example "Slide 3", "Page 12" or "Sheet: Budget".' },
          offset: { type: 'integer', description: 'First character to read, within the part when one is given, 0 by default.' },
          length: { type: 'integer', description: 'Number of characters, 6000 by default, at most 20000.' }
        },
        required: ['name']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'search_discussions',
      description: 'Searches the previous discussions of the user and returns matching excerpts with their title and date.',
      parameters: {
        type: 'object',
        properties: { query: { type: 'string', description: 'Words to look for.' } },
        required: ['query']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'current_datetime',
      description: 'Returns the current date, time and time zone of the computer.',
      parameters: { type: 'object', properties: {} }
    }
  },
  {
    type: 'function',
    function: {
      name: 'create_file',
      description: 'Creates a text file (code, CSV, Markdown, JSON, HTML...) that the user can download from the discussion.',
      parameters: {
        type: 'object',
        properties: {
          name: { type: 'string', description: 'File name with its extension, for example "report.md".' },
          content: { type: 'string', description: 'The full content of the file.' }
        },
        required: ['name', 'content']
      }
    }
  }
];

export const agentInstructions = 'You can call tools. Call a tool when it gives a more reliable or more precise result ' +
  'than answering directly: arithmetic with the calculator, computations and data processing with run_javascript, ' +
  'facts from the user documents with search_documents, document_outline and read_document (to go through a whole ' +
  'document, list its parts, then read them one after the other), earlier work with search_discussions. ' +
  'After the tool results, answer the user directly.';

// Arithmetic without eval: a recursive descent over numbers, operators and functions.
function calculate(expression) {
  const s = String(expression).replace(/\*\*/g, '^').replace(/×/g, '*').replace(/÷/g, '/').replace(/,/g, '.');
  let i = 0;
  const functions = {
    sqrt: Math.sqrt, cbrt: Math.cbrt, abs: Math.abs, exp: Math.exp, ln: Math.log, log: Math.log10, log10: Math.log10,
    log2: Math.log2, sin: Math.sin, cos: Math.cos, tan: Math.tan, asin: Math.asin, acos: Math.acos, atan: Math.atan,
    floor: Math.floor, ceil: Math.ceil, round: Math.round, min: Math.min, max: Math.max, pow: Math.pow, sign: Math.sign
  };
  const constants = { pi: Math.PI, e: Math.E, tau: 2 * Math.PI };
  const space = () => { while (i < s.length && /\s/.test(s[i])) ++i; };
  const fail = (m) => { throw new Error(m + ' at position ' + i); };
  function primary() {
    space();
    if (s[i] === '(') {
      ++i;
      const v = sum();
      space();
      if (s[i] !== ')') fail('")" expected');
      ++i;
      return v;
    }
    if (s[i] === '-') { ++i; return -power(); }
    if (s[i] === '+') { ++i; return power(); }
    const num = /^(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?/.exec(s.slice(i));
    if (num) { i += num[0].length; return parseFloat(num[0]); }
    const id = /^[a-zA-Z_][a-zA-Z0-9_]*/.exec(s.slice(i));
    if (id) {
      i += id[0].length;
      const name = id[0].toLowerCase();
      space();
      if (s[i] === '(') {
        ++i;
        const args = [];
        space();
        if (s[i] !== ')') {
          for (;;) {
            args.push(sum());
            space();
            if (s[i] === ',' || s[i] === ';') { ++i; continue; }
            break;
          }
        }
        if (s[i] !== ')') fail('")" expected');
        ++i;
        if (!functions[name]) fail('unknown function ' + name);
        return functions[name](...args);
      }
      if (name in constants) return constants[name];
      fail('unknown name ' + name);
    }
    fail('number expected');
    return NaN;
  }
  function postfix() {
    let v = primary();
    space();
    while (s[i] === '!') {
      ++i;
      if (v < 0 || v > 170 || v !== Math.floor(v)) fail('factorial of a non-integer');
      let f = 1;
      for (let k = 2; k <= v; ++k) f *= k;
      v = f;
      space();
    }
    return v;
  }
  function power() {
    const base = postfix();
    space();
    if (s[i] === '^') { ++i; return Math.pow(base, power()); }
    return base;
  }
  function product() {
    let v = power();
    for (;;) {
      space();
      const c = s[i];
      if (c === '*') { ++i; v *= power(); } else if (c === '/') { ++i; v /= power(); } else if (c === '%') { ++i; v %= power(); } else return v;
    }
  }
  function sum() {
    let v = product();
    for (;;) {
      space();
      const c = s[i];
      if (c === '+') { ++i; v += product(); } else if (c === '-') { ++i; v -= product(); } else return v;
    }
  }
  const v = sum();
  space();
  if (i < s.length) fail('unexpected "' + s[i] + '"');
  return v;
}

// The sandboxed frame of run_javascript, created at its first use.
let frame = null;
let frameReady = null;
let nextRun = 1;
const runs = new Map();

// The time limit of a run in the frame, and the one of the page, longer, which also ends the
// runs of a frame that does not answer.
const runLimit = 15000;
const pageLimit = runLimit + 5000;

function sandbox() {
  if (frameReady) return frameReady;
  frameReady = new Promise((resolve) => {
    frame = document.createElement('iframe');
    frame.setAttribute('sandbox', 'allow-scripts');
    frame.className = 'sandbox-frame';
    frame.src = 'sandbox.html';
    window.addEventListener('message', (e) => {
      if (!frame || e.source !== frame.contentWindow) return;
      const d = e.data !== null && typeof e.data === 'object' ? e.data : {};
      // The frame says it is ready in a message of its own, without the id of a run.
      if (d.id === undefined) {
        if (d.ready === true) resolve(frame);
        return;
      }
      const r = runs.get(d.id);
      if (r) r(d);
    });
    document.body.appendChild(frame);
  });
  return frameReady;
}

// Runs code in the frame; the run ends with its result, the time limit of the page, or signal.
async function runJavascript(code, signal) {
  const id = nextRun++;
  const result = await new Promise((resolve) => {
    let timer = 0;
    const end = (r) => {
      if (!runs.has(id)) return;
      runs.delete(id);
      clearTimeout(timer);
      if (signal) signal.removeEventListener('abort', stopped);
      resolve(r);
    };
    const stopped = () => end({ ok: false, error: 'The run was stopped.' });
    runs.set(id, end);
    timer = setTimeout(() => end({ ok: false, error: 'The interpreter did not answer within the time limit.' }), pageLimit);
    if (signal) {
      if (signal.aborted) {
        stopped();
        return;
      }
      signal.addEventListener('abort', stopped);
    }
    sandbox().then((f) => {
      // The frame has an opaque origin, which no target origin can name.
      if (runs.has(id)) f.contentWindow.postMessage({ id, code: String(code), timeout: runLimit }, '*');
    });
  });
  const ok = result.ok === true;
  const parts = [];
  if (Array.isArray(result.logs) && result.logs.length) parts.push(result.logs.join('\n'));
  if (ok && typeof result.result === 'string') parts.push('=> ' + result.result);
  if (!ok) parts.push('Error: ' + (typeof result.error === 'string' && result.error ? result.error : 'The interpreter sent no result.'));
  return { ok, text: parts.join('\n').slice(0, 20000) || '(no output)' };
}

function documentByName(documents, name) {
  const n = String(name || '').toLowerCase();
  return documents.find((d) => d.name.toLowerCase() === n) || documents.find((d) => d.name.toLowerCase().includes(n));
}

/*
    Runs a tool call.  context = { documents: [{name, text}], searchDocuments(query, limit),
    searchDiscussions(query), createFile(name, content), signal }, signal stopping the run of
    code with the answer.  Returns {ok, text, file?}.
*/
export async function run(name, argumentsText, context) {
  let args = {};
  try {
    args = argumentsText ? JSON.parse(argumentsText) : {};
  } catch (e) {
    return { ok: false, text: 'The arguments are not valid JSON: ' + e.message };
  }
  try {
    switch (name) {
      case 'calculator': {
        const v = calculate(args.expression);
        return { ok: Number.isFinite(v), text: Number.isFinite(v) ? String(+v.toPrecision(15)) : 'Not a finite number' };
      }
      case 'run_javascript':
        return await runJavascript(args.code || '', context.signal);
      case 'search_documents': {
        if (!context.documents.length) return { ok: true, text: 'No document is attached to this discussion or its project.' };
        const hits = await context.searchDocuments(args.query || '', Math.min(Math.max(args.limit || 5, 1), 10));
        if (!hits.length) return { ok: true, text: 'No passage matches. Documents: ' + context.documents.map((d) => d.name).join(', ') };
        return {
          ok: true,
          text: hits.map((h) => '[' + h.name + (h.where ? ', ' + h.where : '') + ']\n' + h.text.trim()).join('\n\n---\n\n')
        };
      }
      case 'document_outline': {
        const d = documentByName(context.documents, args.name);
        if (!d) return { ok: false, text: 'No document named "' + args.name + '". Documents: ' + context.documents.map((x) => x.name).join(', ') };
        const parts = outline(d.text);
        const lines = parts.map((p) => (p.title || '(beginning)') + ': ' + (p.end - p.start) + ' characters');
        return { ok: true, text: '[' + d.name + ', ' + d.text.length + ' characters, ' + parts.length + ' parts]\n' + lines.join('\n') };
      }
      case 'read_document': {
        const d = documentByName(context.documents, args.name);
        if (!d) return { ok: false, text: 'No document named "' + args.name + '". Documents: ' + context.documents.map((x) => x.name).join(', ') };
        let source = d.text;
        let label = d.name;
        if (args.part) {
          const part = partText(d.text, args.part);
          if (part == null) {
            const names = outline(d.text).map((p) => p.title).filter((x) => x).slice(0, 60);
            return { ok: false, text: 'No part "' + args.part + '" in ' + d.name + '. Parts: ' + (names.join(', ') || 'none') };
          }
          source = part;
          label += ', ' + args.part;
        }
        const offset = Math.max(0, args.offset | 0);
        const length = Math.min(Math.max(args.length || 6000, 100), 20000);
        const piece = source.slice(offset, offset + length);
        return {
          ok: true,
          text: '[' + label + ', characters ' + offset + ' to ' + (offset + piece.length) + ' of ' + source.length + ']\n' + piece
        };
      }
      case 'search_discussions': {
        const hits = await context.searchDiscussions(args.query || '');
        if (!hits.length) return { ok: true, text: 'No discussion matches.' };
        return { ok: true, text: hits.map((h) => '[' + h.title + ', ' + h.date + ']\n' + h.excerpt).join('\n\n') };
      }
      case 'current_datetime': {
        const now = new Date();
        return {
          ok: true,
          text: now.toString() + '\nISO 8601: ' + now.toISOString() + '\nTime zone: ' +
            Intl.DateTimeFormat().resolvedOptions().timeZone
        };
      }
      case 'create_file': {
        const fileName = String(args.name || 'file.txt').replace(/[\\/:*?"<>|]+/g, '_').slice(0, 120);
        const content = String(args.content || '');
        const file = context.createFile(fileName, content);
        return { ok: true, text: 'The file ' + fileName + ' (' + content.length + ' characters) is offered to the user.', file };
      }
      default:
        return { ok: false, text: 'Unknown tool ' + name };
    }
  } catch (e) {
    return { ok: false, text: 'Error: ' + (e && e.message ? e.message : String(e)) };
  }
}
