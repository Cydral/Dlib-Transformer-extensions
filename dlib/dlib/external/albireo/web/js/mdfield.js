// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

/*
    The fields where the user writes for the model: the question of a discussion, the instructions of the settings
    and of a project, and the memory.  A field keeps as its text the markdown the user writes, which goes to the model
    as it is, and draws it as it is written, as the answers draw it: bold, italic, struck and highlighted text
    (==text==), code, headings, lists with their bullets and their indentation, quotes, code blocks and links, their
    marks shown faint.  The field is an element whose text can be edited (contenteditable), drawn again from its text
    after each change, the selection kept at the same characters, and it keeps the part of the interface of a textarea
    that the application uses: value, placeholder, disabled, focus(), and the events input, change, keydown and paste.
    A field drawn again leaves the browser without the history of its edits, so the field keeps its own, which Ctrl+Z
    and Ctrl+Y (or Ctrl+Shift+Z) follow.  Enter in a list item starts the next item, and on an empty item ends the
    list; Tab and Shift+Tab indent the items of the selection; Ctrl+B and Ctrl+I put the marks of bold and italic
    around the selection, or take them away.
*/

const escapes = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' };
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => escapes[c]);
const faint = (s) => s ? '<span class="md-mark">' + esc(s) + '</span>' : '';
const span = (cls, html) => '<span class="' + cls + '">' + html + '</span>';

// The inline marks of markdown, tried at the position of their first character (lastIndex).
const codeSpan = /(`+)([^`]|[^`][\s\S]*?[^`])\1(?!`)/y;
const strong = /(\*\*|__)(?=\S)([\s\S]*?\S)\1/y;
const em = /([*_])(?=[^\s*_])([\s\S]*?[^\s*_])\1(?![*_])/y;
const struck = /~~(?=\S)([\s\S]*?\S)~~/y;
const highlighted = /==(?=[^\s=])([\s\S]*?[^\s=])==/y;
const link = /(!?\[)([^\]\n]+)(\]\([^()\s]*(?:\([^()\s]*\)[^()\s]*)*(?:\s+"[^"\n]*")?\))/y;
const word = /[A-Za-z0-9]/;

function at(re, s, i) {
  re.lastIndex = i;
  return re.exec(s);
}

/*
    The HTML of a line of text with its inline marks drawn.  A mark whose search for its end found none, from a
    position where it can start, finds none from a later position either, and a link whose search failed fails from
    every position before the first ] after it, so the line is searched once for each kind of mark.
*/
function inline(s) {
  let out = '';
  let from = 0;
  let i = 0;
  const unclosed = {};
  let linkless = -1;
  const flush = (to) => {
    if (to > from) out += esc(s.slice(from, to));
  };
  // The mark of key that re finds at i, when it can still end; starts is whether the mark can start at i.
  const mark = (key, re, starts) => {
    if (!starts || unclosed[key]) return null;
    const r = at(re, s, i);
    if (!r) unclosed[key] = true;
    return r;
  };
  while (i < s.length) {
    const c = s[i];
    let m = null;
    let html = '';
    if (c === '\\' && i + 1 < s.length && /[!-/:-@[-`{-~]/.test(s[i + 1])) {
      html = faint('\\') + esc(s[i + 1]);
      m = { length: 2 };
    } else if (c === '`') {
      const r = at(codeSpan, s, i);
      if (r) {
        html = span('md-code', faint(r[1]) + esc(r[2]) + faint(r[1]));
        m = { length: r[0].length };
      } else {
        // A run of backquotes without its closing run stays as it is.
        let n = 1;
        while (s[i + n] === '`') ++n;
        flush(i);
        out += esc(s.slice(i, i + n));
        i += n;
        from = i;
        continue;
      }
    } else if ((c === '[' || (c === '!' && s[i + 1] === '[')) && i >= linkless) {
      const r = at(link, s, i);
      if (r) {
        html = span('md-link', faint(r[1]) + inline(r[2]) + faint(r[3]));
        m = { length: r[0].length };
      } else {
        const close = s.indexOf(']', i);
        linkless = close < 0 ? s.length : close;
      }
    } else if (c === '*' || c === '_') {
      // The marks of _ stand between words only, as the answers read them.
      const intraword = c === '_' && i > 0 && word.test(s[i - 1]);
      let r = mark('strong' + c, strong, !intraword && s[i + 1] === c && /\S/.test(s[i + 2] || ''));
      if (r && (c !== '_' || !word.test(s[i + r[0].length] || ''))) {
        html = span('md-strong', faint(r[1]) + inline(r[2]) + faint(r[1]));
        m = { length: r[0].length };
      } else {
        r = mark('em' + c, em, !intraword && /[^\s*_]/.test(s[i + 1] || ''));
        if (r && (c !== '_' || !word.test(s[i + r[0].length] || ''))) {
          html = span('md-em', faint(r[1]) + inline(r[2]) + faint(r[1]));
          m = { length: r[0].length };
        }
      }
    } else if (c === '~') {
      const r = mark('~', struck, s[i + 1] === '~' && /\S/.test(s[i + 2] || ''));
      if (r) {
        html = faint('~~') + span('md-del', inline(r[1])) + faint('~~');
        m = { length: r[0].length };
      }
    } else if (c === '=') {
      const r = mark('=', highlighted, s[i + 1] === '=' && /[^\s=]/.test(s[i + 2] || ''));
      if (r) {
        html = span('md-hl', faint('==') + inline(r[1]) + faint('=='));
        m = { length: r[0].length };
      }
    }
    if (m) {
      flush(i);
      out += html;
      i += m.length;
      from = i;
    } else {
      ++i;
    }
  }
  flush(s.length);
  return out;
}

// The width of the indentation of a line, a tab counting four spaces.
function indentWidth(s) {
  let n = 0;
  for (const c of s) n += c === '\t' ? 4 - (n % 4) : 1;
  return n;
}

// The HTML of a line outside the code blocks.
function line(s) {
  let m = /^( {0,3})(#{1,6})([ \t]+|$)(.*)$/.exec(s);
  if (m) return span('md-h md-h' + m[2].length, faint(m[1] + m[2] + m[3]) + inline(m[4]));
  if (/^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$/.test(s)) return span('md-hr', esc(s));
  m = /^( {0,3}(?:>[ \t]?)+)(.*)$/.exec(s);
  if (m) return span('md-quote', faint(m[1]) + line(m[2]));
  m = /^([ \t]*)([-*+])([ \t]+)(\[[ xX]\](?:[ \t]+|$))?(.*)$/.exec(s);
  if (m) {
    const level = Math.floor(indentWidth(m[1]) / 2) % 3;
    return esc(m[1]) + span('md-bullet md-level' + level, esc(m[2])) + esc(m[3]) + faint(m[4] || '')
      + inline(m[5]);
  }
  m = /^([ \t]*)(\d{1,9}[.)])([ \t]+)(.*)$/.exec(s);
  if (m) return esc(m[1]) + span('md-number', esc(m[2])) + esc(m[3]) + inline(m[4]);
  // A row of a table: its cells, between their bars.
  if (/^\s*\|.*\|\s*$/.test(s)) return s.split('|').map((cell) => inline(cell)).join(faint('|'));
  return inline(s);
}

// The HTML of a text of markdown, its lines and their line breaks.
export function drawMarkdown(text) {
  const lines = String(text).split('\n');
  const out = [];
  let fence = '';
  for (let k = 0; k < lines.length; ++k) {
    const s = lines[k];
    const end = k + 1 < lines.length ? '\n' : '';
    if (fence) {
      const close = /^ {0,3}(`{3,}|~{3,})[ \t]*$/.exec(s);
      if (close && close[1][0] === fence[0] && close[1].length >= fence.length) {
        out.push(span('md-fence', esc(s)) + end);
        fence = '';
      } else {
        out.push(span('md-pre', esc(s)) + end);
      }
      continue;
    }
    const open = /^ {0,3}(`{3,}|~{3,})(.*)$/.exec(s);
    if (open && !(open[1][0] === '`' && open[2].includes('`'))) {
      fence = open[1];
      out.push(span('md-fence', esc(s)) + end);
      continue;
    }
    out.push(line(s) + end);
  }
  return out.join('');
}

const blocks = /^(DIV|P|LI|UL|OL|PRE|BLOCKQUOTE|H[1-6]|TABLE|TR)$/;

/*
    The text of the field as the browser left it after an edit, and the positions in it of the selection, -1 when
    the selection lies elsewhere.  A line break is a character of the text or an element <br>, and a block an edit
    inserted starts a line; the <br> the field draws after its text, so that a last empty line shows, stays out.
*/
function readField(el) {
  let text = '';
  let start = -1;
  let end = -1;
  const sel = el.ownerDocument.getSelection();
  const range = sel && sel.rangeCount ? sel.getRangeAt(0) : null;
  const mark = (node, offset) => {
    if (!range) return;
    if (node === range.startContainer && offset === range.startOffset) start = text.length;
    if (node === range.endContainer && offset === range.endOffset) end = text.length;
  };
  const walk = (parent) => {
    let i = 0;
    for (let n = parent.firstChild; n; n = n.nextSibling, ++i) {
      mark(parent, i);
      if (n.nodeType === 3) {
        if (range && n === range.startContainer) start = text.length + Math.min(range.startOffset, n.data.length);
        if (range && n === range.endContainer) end = text.length + Math.min(range.endOffset, n.data.length);
        text += n.data;
      } else if (n.nodeName === 'BR') {
        text += '\n';
      } else if (n.nodeType === 1) {
        if (blocks.test(n.nodeName) && text && !text.endsWith('\n')) text += '\n';
        walk(n);
      }
    }
    mark(parent, i);
  };
  walk(el);
  if (el.lastChild && el.lastChild.nodeName === 'BR' && text.endsWith('\n')) text = text.slice(0, -1);
  return { text, start: Math.min(start, text.length), end: Math.min(end, text.length) };
}

/*
    The text after an edit, the non-breaking spaces that the browser puts for the spaces it inserts made plain spaces
    in the part the edit changed, those of the text before it kept.
*/
function plainSpaces(before, after) {
  const n = Math.min(before.length, after.length);
  let a = 0;
  while (a < n && before[a] === after[a]) ++a;
  let b = 0;
  while (b < n - a && before[before.length - 1 - b] === after[after.length - 1 - b]) ++b;
  return after.slice(0, a) + after.slice(a, after.length - b).replace(/\u00a0/g, ' ') + after.slice(after.length - b);
}

// The position of the DOM of a character of the text of a field just drawn.
function locate(el, offset) {
  let count = 0;
  const walker = el.ownerDocument.createTreeWalker(el, NodeFilter.SHOW_TEXT);
  for (let n = walker.nextNode(); n; n = walker.nextNode()) {
    if (offset <= count + n.data.length) return [n, offset - count];
    count += n.data.length;
  }
  // An empty text: before the line break drawn after it.
  return [el, Math.max(0, el.childNodes.length - 1)];
}

function selectRange(el, start, end) {
  const sel = el.ownerDocument.getSelection();
  if (!sel) return;
  const range = el.ownerDocument.createRange();
  const [a, ao] = locate(el, start);
  const [b, bo] = locate(el, end);
  range.setStart(a, ao);
  range.setEnd(b, bo);
  sel.removeAllRanges();
  sel.addRange(range);
}

// Brings the caret into view in a field that scrolls.
function reveal(el) {
  const sel = el.ownerDocument.getSelection();
  if (!sel || !sel.rangeCount || el.scrollHeight <= el.clientHeight) return;
  const rects = sel.getRangeAt(0).getClientRects();
  const r = rects.length ? rects[rects.length - 1] : null;
  if (!r) return;
  const box = el.getBoundingClientRect();
  if (r.bottom > box.bottom) el.scrollTop += r.bottom - box.bottom + 4;
  else if (r.top < box.top) el.scrollTop -= box.top - r.top + 4;
}

const listItem = /^([ \t]*)([-*+]|(\d{1,9})([.)]))([ \t]+)(\[[ xX]\][ \t]+)?/;
// The edits that join the entry of the history before them, typing within this time of the previous one.
const joinMs = 1000;
const historyLength = 200;
// The characters of a text whose marks the field draws, beyond which it draws the text alone.
const drawnLength = 30000;
/*
    The characters of a text beyond which the field draws its marks again once the typing pauses for settleMs, the
    browser showing the characters typed meanwhile, since drawing a long text at each key slows the typing.
*/
const quickLength = 4000;
const settleMs = 150;

/*
    A field of markdown.  options: placeholder, label (its name for the screen readers), className.
*/
export function markdownField(options = {}) {
  const el = document.createElement('div');
  el.className = 'md-field md-text' + (options.className ? ' ' + options.className : '');
  el.setAttribute('role', 'textbox');
  el.setAttribute('aria-multiline', 'true');
  if (options.label) el.setAttribute('aria-label', options.label);
  el.spellcheck = true;
  let text = '';
  let placeholder = '';
  let disabled = false;
  let composing = false;
  let focusText = '';
  let history = [{ text: '', start: 0, end: 0 }];
  let at = 0;
  let lastKind = '';
  let lastTime = 0;
  // A text given by the application, whose caret goes to its end at the next focus, as in a textarea.
  let fresh = false;
  // The drawing that waits for a pause of the typing (quickLength), and a text moved by a drag (deleteByDrag).
  let settle = 0;
  let dragged = null;

  const editable = () => {
    if (disabled) {
      el.contentEditable = 'false';
      return;
    }
    // The browsers that edit plain text alone keep the rich text of a paste or a drop out of the field.
    try {
      el.contentEditable = 'plaintext-only';
    } catch (e) {
      el.contentEditable = 'true';
    }
    if (el.contentEditable !== 'plaintext-only') el.contentEditable = 'true';
  };

  const draw = (start, end) => {
    clearTimeout(settle);
    settle = 0;
    el.innerHTML = (text.length > drawnLength ? esc(text) : drawMarkdown(text)) + '<br>';
    el.classList.toggle('empty', !text);
    if (start != null && el.ownerDocument.activeElement === el) {
      selectRange(el, start, end == null ? start : end);
      reveal(el);
    }
  };

  const remember = (start, end, kind) => {
    const now = Date.now();
    const group = (k) => (k === 'insertText' ? 'insert' : k.startsWith('delete') ? 'delete' : k);
    const join = (kind === 'insertText' || kind.startsWith('delete')) && group(kind) === group(lastKind)
      && now - lastTime < joinMs && at > 0;
    history.length = at + 1;
    if (join) {
      history[at] = { text, start, end };
    } else if (history[at].text !== text) {
      history.push({ text, start, end });
      if (history.length > historyLength) history.shift();
      at = history.length - 1;
    }
    lastKind = kind;
    lastTime = now;
  };

  // An input event of the field itself, after an edit it made in place of the browser.
  let own = false;
  const fire = (kind, data) => {
    own = true;
    try {
      el.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: kind, data: data == null ? null : data }));
    } finally {
      own = false;
    }
  };

  // Replaces the characters from start to end with insert, the caret after it unless caret says where.
  const replace = (start, end, insert, kind, caret) => {
    const before = text;
    text = text.slice(0, start) + insert + text.slice(end);
    const c = caret == null ? start + insert.length : caret;
    draw(c, c);
    if (text !== before) {
      remember(c, c, kind);
      fire(kind, insert);
    }
  };

  const selection = () => {
    const r = readField(el);
    if (r.start < 0 || r.end < 0) return { start: text.length, end: text.length };
    return { start: Math.min(r.start, r.end), end: Math.max(r.start, r.end) };
  };

  // A line break, which continues a list: the next item, or the end of the list after an empty item.
  const newline = (kind) => {
    const { start, end } = selection();
    const lineStart = text.lastIndexOf('\n', start - 1) + 1;
    const m = start === end ? listItem.exec(text.slice(lineStart, start)) : null;
    if (!m) {
      replace(start, end, '\n', kind);
      return;
    }
    const lineEnd = text.indexOf('\n', start) < 0 ? text.length : text.indexOf('\n', start);
    if (lineStart + m[0].length === start && lineEnd === start) {
      // An empty item ends the list.
      replace(lineStart, lineEnd, '', kind);
      return;
    }
    const marker = m[3] ? String(Number(m[3]) + 1) + m[4] : m[2];
    replace(start, end, '\n' + m[1] + marker + m[5] + (m[6] ? '[ ] ' : ''), kind);
  };

  // The lines from the one of start to the one of end, as [line start, line end].
  const linesOf = (start, end) => {
    const out = [];
    let s = text.lastIndexOf('\n', start - 1) + 1;
    for (;;) {
      const e = text.indexOf('\n', s) < 0 ? text.length : text.indexOf('\n', s);
      out.push([s, e]);
      if (e >= end || e >= text.length) break;
      s = e + 1;
    }
    return out;
  };

  /*
      Indents or outdents the list items of the selection, by the width of their marker (two characters for a
      bullet, three for a number); false when the selection holds none.
  */
  const indent = (outdent) => {
    const { start, end } = selection();
    // [line start, characters removed, characters added], the last line first.
    const edits = [];
    for (const [s0, e0] of linesOf(start, end)) {
      const item = text.slice(s0, e0);
      if (!listItem.test(item)) continue;
      const width = /^[ \t]*\d/.test(item) ? 3 : 2;
      if (!outdent) {
        edits.unshift([s0, 0, width]);
        continue;
      }
      const lead = /^[ \t]*/.exec(item)[0];
      let cut = 0;
      while (cut < lead.length && cut < width && lead[cut] === ' ') ++cut;
      if (cut === 0 && lead[0] === '\t') cut = 1;
      if (cut) edits.unshift([s0, cut, 0]);
    }
    if (!edits.length) return !outdent ? false : linesOf(start, end).some(([a, b]) => listItem.test(text.slice(a, b)));
    const move = (p) => {
      let q = p;
      for (const [s0, removed, added] of edits) {
        if (p >= s0) q += added - Math.min(removed, p - s0);
      }
      return q;
    };
    const a = move(start);
    const b = move(end);
    for (const [s0, removed, added] of edits) text = text.slice(0, s0) + ' '.repeat(added) + text.slice(s0 + removed);
    draw(a, b);
    remember(a, b, 'indent');
    fire('insertText', null);
    return true;
  };

  // Puts the marks of a format around the selection, or takes them away when they surround it already.
  const surround = (marks) => {
    const { start, end } = selection();
    const n = marks.length;
    if (start >= n && text.slice(start - n, start) === marks && text.slice(end, end + n) === marks
        && !(n === 1 && (text[start - n - 1] === marks || text[end + n] === marks))) {
      const before = text;
      text = text.slice(0, start - n) + text.slice(start, end) + text.slice(end + n);
      draw(start - n, end - n);
      if (text !== before) {
        remember(start - n, end - n, 'format');
        fire('insertText', null);
      }
      return;
    }
    const before = text;
    text = text.slice(0, start) + marks + text.slice(start, end) + marks + text.slice(end);
    draw(start + n, end + n);
    if (text !== before) {
      remember(start + n, end + n, 'format');
      fire('insertText', null);
    }
  };

  const restore = (to) => {
    if (to < 0 || to >= history.length) return;
    const kind = to < at ? 'historyUndo' : 'historyRedo';
    at = to;
    const h = history[at];
    text = h.text;
    draw(h.start, h.end);
    lastKind = 'history';
    fire(kind, null);
  };

  el.addEventListener('beforeinput', (e) => {
    if (disabled) return;
    const kind = e.inputType || '';
    if (kind === 'insertParagraph' || kind === 'insertLineBreak') {
      e.preventDefault();
      newline(kind);
    } else if (kind === 'insertFromPaste' || kind === 'insertFromPasteAsQuotation' || kind === 'insertFromYank') {
      const data = e.dataTransfer ? e.dataTransfer.getData('text/plain') : null;
      if (data == null) return;
      e.preventDefault();
      const { start, end } = selection();
      replace(start, end, data.replace(/\r\n?/g, '\n'), 'insertFromPaste');
    } else if (kind === 'historyUndo') {
      e.preventDefault();
      restore(at - 1);
    } else if (kind === 'historyRedo') {
      e.preventDefault();
      restore(at + 1);
    } else if (kind === 'formatBold') {
      e.preventDefault();
      surround('**');
    } else if (kind === 'formatItalic') {
      e.preventDefault();
      surround('*');
    } else if (kind.startsWith('format')) {
      // The other rich formats have no markdown: the text stays as it is.
      e.preventDefault();
    }
  });

  el.addEventListener('input', (e) => {
    if (own || composing || e.isComposing) return;
    const r = readField(el);
    text = plainSpaces(text, r.text);
    const start = r.start < 0 ? text.length : r.start;
    const end = r.end < 0 ? start : r.end;
    const kind = e.inputType || 'insertText';
    if (kind === 'deleteByDrag') {
      /*
          The drop that follows in the field places the text on the elements the browser left, so the field draws
          once the drop is done, or after the drag when the text went to another place.
      */
      dragged = { start, end };
      setTimeout(() => {
        if (!dragged) return;
        const d = dragged;
        dragged = null;
        draw(d.start, d.end);
        remember(d.start, d.end, 'deleteByDrag');
      }, 0);
      return;
    }
    dragged = null;
    if (text.length > quickLength && (kind === 'insertText' || kind.startsWith('delete'))) {
      clearTimeout(settle);
      settle = setTimeout(() => {
        settle = 0;
        if (composing) return;
        const s = selection();
        draw(s.start, s.end);
      }, settleMs);
    } else {
      draw(start, end);
    }
    remember(start, end, kind);
  });

  el.addEventListener('compositionstart', () => { composing = true; });
  el.addEventListener('compositionend', () => {
    composing = false;
    const r = readField(el);
    text = plainSpaces(text, r.text);
    const start = r.start < 0 ? text.length : r.start;
    draw(start, r.end < 0 ? start : r.end);
    remember(start, r.end < 0 ? start : r.end, 'insertCompositionText');
  });

  el.addEventListener('keydown', (e) => {
    if (disabled || e.isComposing) return;
    const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
    const command = (e.ctrlKey || e.metaKey) && !e.altKey;
    if (command && key === 'z') {
      e.preventDefault();
      restore(e.shiftKey ? at + 1 : at - 1);
    } else if (command && key === 'y' && !e.shiftKey) {
      e.preventDefault();
      restore(at + 1);
    } else if (command && !e.shiftKey && (key === 'b' || key === 'i')) {
      e.preventDefault();
      surround(key === 'b' ? '**' : '*');
    } else if (e.key === 'Tab' && !e.ctrlKey && !e.altKey && !e.metaKey) {
      // Tab leaves a field whose selection holds no list item, as it leaves a textarea.
      if (indent(e.shiftKey)) e.preventDefault();
    }
  });

  el.addEventListener('focus', () => { focusText = text; });
  el.addEventListener('blur', () => {
    if (text !== focusText) {
      focusText = text;
      el.dispatchEvent(new Event('change', { bubbles: true }));
    }
  });

  Object.defineProperties(el, {
    value: {
      get: () => text,
      set: (v) => {
        text = String(v == null ? '' : v).replace(/\r\n?/g, '\n');
        history = [{ text, start: text.length, end: text.length }];
        at = 0;
        lastKind = '';
        focusText = text;
        fresh = true;
        const focused = el.ownerDocument.activeElement === el;
        draw(focused ? text.length : null, focused ? text.length : null);
      }
    },
    placeholder: {
      get: () => placeholder,
      set: (v) => {
        placeholder = String(v == null ? '' : v);
        el.setAttribute('data-placeholder', placeholder);
        if (!options.label) el.setAttribute('aria-label', placeholder);
      }
    },
    disabled: {
      get: () => disabled,
      set: (v) => {
        disabled = !!v;
        editable();
        el.classList.toggle('disabled', disabled);
        el.setAttribute('aria-disabled', String(disabled));
      }
    }
  });

  /*
      The focus puts the caret at the end of a text the application gave, and of a field whose selection lies
      elsewhere, as a textarea does; a field the user edited keeps its selection.
  */
  const focus = HTMLElement.prototype.focus;
  el.focus = function (opts) {
    focus.call(el, opts);
    const r = readField(el);
    if (fresh || r.start < 0) {
      selectRange(el, text.length, text.length);
      reveal(el);
    }
    fresh = false;
  };
  el.addEventListener('pointerdown', () => { fresh = false; });

  editable();
  el.placeholder = options.placeholder || '';
  draw(null, null);
  return el;
}
