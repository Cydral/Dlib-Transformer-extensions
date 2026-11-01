// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

/*
    Markdown of the answers: GitHub flavoured markdown, code highlighted by highlight.js,
    formulas typeset by KaTeX between $...$, $$...$$, \(...\) and \[...\].  The HTML a
    model writes outside of code is shown as text, and the result is sanitized before it
    reaches the page.  An answer loads nothing from another origin, and shows an image of
    another origin as a link to it, which the browser follows only when the user does.
*/

import { Marked } from '../vendor/marked/marked.esm.js';
import DOMPurify from '../vendor/dompurify/purify.es.mjs';
import { icon } from './icons.js';

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

// Whether a URL names the origin of the application, or data the page holds (data:, blob:).
function localSource(url) {
  try {
    const u = new URL(String(url), document.baseURI);
    return u.origin === location.origin || u.protocol === 'data:' || u.protocol === 'blob:';
  } catch (e) {
    return false;
  }
}

function tex(text, display) {
  if (!window.katex) return escapeHtml(display ? '$$' + text + '$$' : '$' + text + '$');
  try {
    return window.katex.renderToString(text, { displayMode: display, throwOnError: false, output: 'html', strict: false });
  } catch (e) {
    return '<code>' + escapeHtml(text) + '</code>';
  }
}

const math = [
  {
    name: 'blockMath',
    level: 'block',
    start(src) {
      const i = src.search(/\$\$|\\\[/);
      return i < 0 ? undefined : i;
    },
    tokenizer(src) {
      const m = /^\$\$([\s\S]+?)\$\$/.exec(src) || /^\\\[([\s\S]+?)\\\]/.exec(src);
      if (m) return { type: 'blockMath', raw: m[0], text: m[1].trim() };
      return undefined;
    },
    renderer(token) {
      return '<div class="math-block">' + tex(token.text, true) + '</div>';
    }
  },
  {
    name: 'inlineMath',
    level: 'inline',
    start(src) {
      const i = src.search(/\$|\\\(/);
      return i < 0 ? undefined : i;
    },
    tokenizer(src) {
      let m = /^\$\$([\s\S]+?)\$\$/.exec(src);
      if (m) return { type: 'inlineMath', raw: m[0], text: m[1].trim(), display: true };
      // A dollar followed by a digit after its closing one is a price, not a formula.
      m = /^\$(?![\s$])((?:\\.|[^\\$\n])+?)\$(?!\d)/.exec(src);
      if (m && !/\s$/.test(m[1])) return { type: 'inlineMath', raw: m[0], text: m[1], display: false };
      m = /^\\\(([\s\S]+?)\\\)/.exec(src);
      if (m) return { type: 'inlineMath', raw: m[0], text: m[1].trim(), display: false };
      return undefined;
    },
    renderer(token) {
      return tex(token.text, token.display);
    }
  }
];

// True while renderMarkdown draws an answer that streams: the code without a language waits
// for the end of the answer to have its language detected, and a code block whose closing
// fence has not come yet is highlighted once it comes.
let streaming = false;

// True for a fenced code block whose closing fence is missing, the block being written.
function openFence(raw) {
  const text = String(raw || '').replace(/\s+$/, '');
  const open = /^ {0,3}(`{3,}|~{3,})/.exec(text);
  if (!open) return false;
  const lines = text.split('\n');
  if (lines.length < 2) return true;
  const last = /^ {0,3}(`{3,}|~{3,})\s*$/.exec(lines[lines.length - 1]);
  return !(last && last[1][0] === open[1][0] && last[1].length >= open[1].length);
}

const marked = new Marked({ gfm: true, breaks: false });
marked.use({
  extensions: math,
  renderer: {
    code({ text, lang, raw }) {
      const language = String(lang || '').trim().split(/\s+/)[0].toLowerCase();
      let body = escapeHtml(text);
      const hl = window.hljs;
      // A block marked as plain text or as the output of a program stays as it is.
      const plain = /^(text|txt|plain|plaintext|output)$/.test(language) || (streaming && openFence(raw));
      if (hl && !plain) {
        try {
          if (language && hl.getLanguage(language)) body = hl.highlight(text, { language, ignoreIllegals: true }).value;
          else if (!streaming && text.length < 20000) body = hl.highlightAuto(text).value;
        } catch (e) {
          body = escapeHtml(text);
        }
      }
      return '<pre><div class="code-head"><span>' + escapeHtml(language || 'text') + '</span>' +
        '<button type="button" data-action="copy-code">' + icon('copy') + '<span>Copy</span></button></div>' +
        '<code class="hljs">' + body + '</code></pre>';
    },
    html({ text }) {
      return escapeHtml(text);
    },
    // An image of the application or of data keeps the default rendering; another one becomes a link.
    image({ href, title, text, tokens }) {
      if (localSource(href)) return false;
      const label = tokens ? this.parser.parseInline(tokens, this.parser.textRenderer) : text;
      return '<a href="' + escapeHtml(href) + '"' + (title ? ' title="' + escapeHtml(title) + '"' : '') + '>' +
        escapeHtml(label || href) + '</a>';
    }
  }
});

DOMPurify.addHook('afterSanitizeAttributes', (node) => {
  if (node.tagName === 'A') {
    node.setAttribute('target', '_blank');
    node.setAttribute('rel', 'noopener noreferrer');
  }
  // The attributes that load a resource keep the origin of the application, whatever wrote them.
  if (!node.hasAttribute) return;
  if (node.hasAttribute('srcset')) node.removeAttribute('srcset');
  for (const name of ['src', 'poster', 'background']) {
    if (node.hasAttribute(name) && !localSource(node.getAttribute(name))) node.removeAttribute(name);
  }
});

const purifyOptions = {
  FORBID_TAGS: ['style', 'form', 'input', 'textarea', 'select', 'iframe', 'object', 'embed', 'link', 'meta'],
  FORBID_ATTR: ['action', 'formaction'],
  ADD_ATTR: ['data-action', 'aria-hidden']
};

// The HTML of a text; live for an answer that streams, drawn again at each of its pieces.
export function renderMarkdown(text, live) {
  streaming = !!live;
  try {
    return DOMPurify.sanitize(marked.parse(String(text || '')), purifyOptions);
  } finally {
    streaming = false;
  }
}

// The copy buttons of the code blocks of an element.
export function attachCodeButtons(root, label) {
  for (const b of root.querySelectorAll('button[data-action="copy-code"]')) {
    if (b.dataset.bound) continue;
    b.dataset.bound = '1';
    if (label) b.lastChild.textContent = label;
    b.addEventListener('click', () => {
      const code = b.closest('pre').querySelector('code');
      navigator.clipboard.writeText(code ? code.innerText : '').catch(() => {});
      const span = b.lastChild;
      const before = span.textContent;
      span.textContent = '✓';
      setTimeout(() => { span.textContent = before; }, 1200);
    });
  }
}

export { escapeHtml };
