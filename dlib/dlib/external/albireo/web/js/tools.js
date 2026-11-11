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
      description: 'Evaluates an arithmetic expression exactly as written and returns its value. Supports + - * / ^, ' +
        'parentheses, percentages ("17.5% of 2340", "2340 * 17.5%"), mod for the remainder of a division, ' +
        'and the functions sqrt, cbrt, abs, exp, ln, log10, log2, sin, cos, tan, asin, acos, atan, ' +
        'floor, ceil, round, min, max, pow, and the constants pi and e.',
      parameters: {
        type: 'object',
        properties: {
          expression: {
            type: 'string', description: 'The expression, for example "(3.5 + 2) * 4 ^ 2" or "17.5% of 2340".'
          }
        },
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

/*
    The guard of an answer against the calls that a model repeats, as a small model does when the results do not hold
    what it looks for.  A call that gives a tool the arguments of an earlier call of the answer, whose result the model
    still reads whole, is not run ('repeat'), a failed call being run once more.  A search is not run when its words,
    common words, elisions and placeholders of identifiers (CVE-2026-XXXX) aside, hold those of sameSubject earlier
    searches of its tool or are held by them, making half of them at least, as the same search with a word or a year
    more does ('repeat'), nor beyond searchBudget searches of its tool in the answer ('budget').  A search that changes
    a word for another, as the same question about another city, another year or another vitamin does, runs.  check()
    tells why a call is not run, or '' when it runs; record() keeps a call that ran and whether it succeeded.
*/
export const searchBudget = 8;
const sameSubject = 2;
const searchTools = ['web_search', 'search_documents', 'search_discussions'];
/*
    The words that tell nothing of the subject of a search, in English and in the other languages of the interface.
    The letters alone stay, since they name things (vitamin D, plan B), the elided articles and prepositions of French
    and Italian (l', d', qu', dell') leaving with their apostrophe.
*/
const commonWords = new Set(('an and are as at be by for from how in is it of on the to what when where which '
  + 'who why with au aux ce ces dans de des du en est et la le les ou par pour que quel quelle quels quelles qui sur '
  + 'un une am auf das dem den der die ein eine einem einen einer fur im ist mit oder und vom von was wie zu zum zur '
  + 'al alla che come con da dei del della delle di gli il lo nel nella per quale su uno como cual el las los para '
  + 'por una unas unos').split(' '));
const elision = /(^|[^a-z])(?:c|d|j|l|m|n|qu|s|t|all|dall|dell|nell|sull)['\u2019]/g;

// The words of a search, in any script, without its common words, its elisions, its placeholders and its accents.
function searchWords(query) {
  return new Set(String(query || '').toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '')
    .replace(elision, '$1 ').split(/[^\p{L}\p{M}\p{N}]+/u)
    .filter((w) => w && !commonWords.has(w) && !/^x{3,}$/.test(w)));
}

// The text of a call that compares it with the others, its tool and its arguments.
export function callKey(name, argumentsText) {
  return name + ' ' + argumentsKey(argumentsText);
}

// The result of a call that repeats an earlier call of its step, which runs once.
export const duplicateText = 'Not run: this call is the same as an earlier call of this step, whose result is above.';

// Whether two searches are on one subject: the words of one hold those of the other, half of its words at least.
function sameWords(a, b) {
  const small = a.size <= b.size ? a : b;
  const large = small === a ? b : a;
  for (const w of small) if (!large.has(w)) return false;
  return 2 * small.size >= large.size;
}

// The arguments of a call as one text, their keys in order and a search query without its case and its spaces.
function argumentsKey(argumentsText) {
  let args;
  try {
    args = JSON.parse(argumentsText || '{}');
  } catch (e) {
    return String(argumentsText || '');
  }
  if (!args || typeof args !== 'object' || Array.isArray(args)) return JSON.stringify(args);
  const out = {};
  for (const k of Object.keys(args).sort()) {
    out[k] = k === 'query' && typeof args[k] === 'string' ? args[k].toLowerCase().replace(/\s+/g, ' ').trim() : args[k];
  }
  return JSON.stringify(out);
}

function queryOf(argumentsText) {
  try {
    const args = JSON.parse(argumentsText || '{}');
    return args && typeof args.query === 'string' ? args.query : '';
  } catch (e) {
    return '';
  }
}

export function callGuard() {
  // The calls that ran, by their tool and their arguments: {call, ok, retried}.
  const calls = new Map();
  const searches = new Map();
  return {
    // seen(call) tells whether the request of the model holds the whole result of an earlier call.
    check(name, argumentsText, seen) {
      const earlier = calls.get(callKey(name, argumentsText));
      if (earlier && (earlier.ok || earlier.retried) && seen(earlier.call)) return 'repeat';
      if (!searchTools.includes(name)) return '';
      const done = searches.get(name) || [];
      if (done.length >= searchBudget) return 'budget';
      // A failed search run once more, or one whose result left the request, runs as it is.
      if (earlier) return '';
      const words = searchWords(queryOf(argumentsText));
      return done.filter((w) => sameWords(w, words)).length >= sameSubject ? 'repeat' : '';
    },
    record(name, argumentsText, call, ok) {
      const key = callKey(name, argumentsText);
      calls.set(key, { call, ok, retried: calls.has(key) });
      if (!searchTools.includes(name)) return;
      const done = searches.get(name) || [];
      done.push(searchWords(queryOf(argumentsText)));
      searches.set(name, done);
    }
  };
}

/*
    The result of a call that the guard does not run, which asks the model to answer with the results it has and to
    take its identifiers from them, as a small model otherwise invents some (CVE numbers among others).
*/
export function refusalText(reason) {
  return (reason === 'budget' ? 'Not run: this answer has run its ' + searchBudget + ' searches.'
    : 'Not run: this call repeats earlier calls of this answer, whose results are above.')
    + ' Answer the user now with the results above, and take from them the identifiers, names and dates you give. '
    + 'When they do not hold what was asked, say so, then answer with what they hold that comes closest, and with '
    + 'what you know only when you are sure of it, saying that it may be out of date.';
}

/*
    An expression in the notation the calculator reads.  The signs of multiplication, division and
    subtraction become *, / and -, the groups of thousands written with spaces join ("2 340"), a
    comma between two digits outside of the arguments of a function is a decimal comma ("17,5"),
    and the word that follows a percentage in the languages of the interface ("17.5% of 2340",
    "17,5 % de 2 340") becomes a product.
*/
function normalizeExpression(text) {
  const s = String(text).replace(/\*\*/g, '^').replace(/[×·]/g, '*').replace(/÷/g, '/').replace(/[−–]/g, '-')
    .replace(/[   ]/g, ' ')
    .replace(/%\s*(of|de|du|des|von|vom|der|di|del|della|dello|dei|delle)(?![\p{L}\p{N}_])/giu, '%*')
    .replace(/(\d) (?=\d{3}(?!\d))/g, '$1');
  let out = '';
  // For each open parenthesis, true when it opens the arguments of a function.
  const calls = [];
  for (let k = 0; k < s.length; ++k) {
    const c = s[k];
    if (c === '(') calls.push(/[A-Za-z_][A-Za-z0-9_]*\s*$/.test(out));
    else if (c === ')') calls.pop();
    else if (c === ',' && !calls[calls.length - 1] && /\d$/.test(out) && /\d/.test(s[k + 1] || '')) {
      out += '.';
      continue;
    }
    out += c;
  }
  return out;
}

/*
    Arithmetic without eval: a recursive descent over numbers, operators and functions.  A % after
    a value is a percentage, unless a value follows it, which makes it the remainder of a division.
*/
function calculate(expression) {
  const s = normalizeExpression(expression);
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
  // True when the % at the position is a percentage, followed by nothing, an operator or a parenthesis.
  function percentage() {
    if (s[i] !== '%') return false;
    let j = i + 1;
    while (j < s.length && /\s/.test(s[j])) ++j;
    return j >= s.length || /[)+\-*\/^%,;!]/.test(s[j]);
  }
  function postfix() {
    let v = primary();
    for (;;) {
      space();
      if (s[i] === '!') {
        ++i;
        if (v < 0 || v > 170 || v !== Math.floor(v)) fail('factorial of a non-integer');
        let f = 1;
        for (let k = 2; k <= v; ++k) f *= k;
        v = f;
      } else if (percentage()) {
        ++i;
        v /= 100;
      } else {
        return v;
      }
    }
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
      if (c === '*') {
        ++i;
        v *= power();
      } else if (c === '/') {
        ++i;
        v /= power();
      } else if (c === '%' || /^mod(?![A-Za-z0-9_])/i.test(s.slice(i, i + 4))) {
        i += c === '%' ? 1 : 3;
        v %= power();
      } else {
        return v;
      }
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
      // The frame has an opaque origin, which its messages carry as "null".
      if (!frame || e.source !== frame.contentWindow || e.origin !== 'null') return;
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

// The extensions of the files that Windows runs when they are opened, which a file of the agent never keeps.
const runnableExtensions = new Set(('bat cmd com exe scr pif cpl msi msp msc ps1 psm1 psd1 ps1xml vbs vbe js jse ' +
  'wsf wsh hta jar lnk url reg inf scf appref-ms application gadget').split(' '));

/*
    The name of a file the agent creates: without the characters a file name cannot hold, at most 120 of them, and
    ending with ".txt" when its extension is one Windows runs, so that opening it shows its text.  The dots and the
    spaces that end a name, which Windows drops, are dropped first, so that the extension checked is the one the file
    keeps.
*/
export function fileNameOf(name) {
  let fileName = String(name || '').replace(/[\\/:*?"<>|\u0000-\u001f\u007f]+/g, '_').slice(0, 120)
    .replace(/[. ]+$/, '');
  if (!fileName) fileName = 'file.txt';
  const dot = fileName.lastIndexOf('.');
  const ext = dot < 0 ? '' : fileName.slice(dot + 1).trim().toLowerCase();
  if (runnableExtensions.has(ext)) fileName += '.txt';
  return fileName;
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
        const fileName = fileNameOf(args.name);
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
