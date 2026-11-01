// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

/*
    Reading the answers aloud with the speech synthesis of the service, which needs no
    network: the text of an answer without its markup, the language its words show, then
    groups of sentences read by the service one after the other, the next one asked while
    the current one plays.  One answer is read at a time.
*/

import { getJSON, postAudio } from './api.js';

const languages = ['en', 'fr', 'de', 'it', 'es'];

// Frequent short words of each language, which tell the language of a text.
const markers = {
  en: ['the', 'and', 'is', 'are', 'of', 'to', 'in', 'that', 'it', 'for', 'with', 'this', 'you', 'not', 'be', 'was',
    'have', 'can', 'or', 'which'],
  fr: ['le', 'la', 'les', 'et', 'est', 'des', 'une', 'un', 'que', 'pour', 'dans', 'qui', 'pas', 'vous', 'sur', 'au',
    'du', 'ce', 'il', 'sont'],
  de: ['der', 'die', 'das', 'und', 'ist', 'nicht', 'ein', 'eine', 'zu', 'mit', 'den', 'sie', 'es', 'auf', 'für',
    'von', 'dem', 'sind', 'auch', 'wird'],
  it: ['il', 'di', 'che', 'e', 'la', 'per', 'non', 'un', 'una', 'sono', 'del', 'della', 'con', 'gli', 'è', 'le',
    'si', 'nel', 'questo', 'anche'],
  es: ['el', 'la', 'de', 'que', 'y', 'en', 'los', 'las', 'es', 'un', 'una', 'por', 'con', 'para', 'del', 'se', 'no',
    'su', 'como', 'más']
};
const markerSets = {};
for (const l of languages) markerSets[l] = new Set(markers[l]);

// The language of a text by its frequent words, or fallback when the words do not tell.
export function textLanguage(text, fallback) {
  const words = (text.toLowerCase().match(/\p{L}+/gu) || []).slice(0, 600);
  let best = null, top = 0, second = 0;
  for (const l of languages) {
    let score = 0;
    for (const w of words) if (markerSets[l].has(w)) score++;
    if (score > top) {
      second = top;
      top = score;
      best = l;
    } else if (score > second) second = score;
  }
  return best && top >= 3 && top >= second * 1.25 ? best : fallback;
}

/*
    The text of an answer as it is read: the code blocks and the reasoning left out, the
    links and the images by their text, the headings and the items of the lists as
    sentences, the marks of emphasis, of tables and of quotes removed.
*/
export function spokenText(markdown) {
  let t = String(markdown || '');
  t = t.replace(/<think>[\s\S]*?(<\/think>|$)/g, ' ');
  t = t.replace(/```[\s\S]*?(```|$)/g, '\n');
  t = t.replace(/!\[([^\]]*)\]\([^)]*\)/g, '$1');
  t = t.replace(/\[([^\]]+)\]\([^)]*\)/g, '$1');
  t = t.replace(/https?:\/\/\S+/g, ' ');
  t = t.replace(/`([^`\n]*)`/g, '$1');
  t = t.replace(/<[^>\n]*>/g, ' ');
  t = t.replace(/\$\$?([^$\n]+)\$\$?/g, '$1');
  const lines = [];
  for (let line of t.split('\n')) {
    line = line.replace(/^\s{0,3}>\s?/, '');
    if (/^\s*[-:|\s]{3,}$/.test(line)) continue;
    const heading = /^\s{0,3}#{1,6}\s+(.*)$/.exec(line);
    const item = /^\s*(?:[-*+]|\d+[.)])\s+(.*)$/.exec(line);
    if (heading) line = heading[1];
    else if (item) line = item[1];
    line = line.replace(/\|/g, ', ').replace(/(\*\*|__|~~|\*)/g, '').trim();
    if (!line) continue;
    if ((heading || item) && !/[.!?:;,]$/.test(line)) line += '.';
    lines.push(line);
  }
  return lines.join('\n');
}

/*
    The text cut into groups of sentences of at most max characters, the first one shorter
    so that the reading starts soon.
*/
export function speechParts(text, max = 700) {
  const sentences = text.match(/[^.!?…\n]+[.!?…]*[\s]*|\n+/g) || [];
  const parts = [];
  let part = '';
  for (const s of sentences) {
    const limit = parts.length ? max : Math.min(max, 220);
    if (part && part.length + s.length > limit) {
      parts.push(part.trim());
      part = '';
    }
    // A sentence longer than a group goes in pieces cut at its spaces.
    let rest = s;
    while (rest.length > max) {
      let cut = rest.lastIndexOf(' ', max);
      if (cut < max / 2) cut = max;
      parts.push(rest.slice(0, cut).trim());
      rest = rest.slice(cut);
    }
    part += rest;
  }
  if (part.trim()) parts.push(part.trim());
  return parts.filter((p) => /\p{L}|\p{N}/u.test(p));
}

let voicesAsked = null;

/*
    The languages the service reads aloud, empty when it has no voice.  An empty list is asked again at the
    next call, so that voices installed after the page opened show.
*/
export function speechLanguages() {
  if (!voicesAsked) {
    voicesAsked = getJSON('/albireo/v1/speech/voices')
      .then((r) => {
        const list = Array.from(new Set((r.voices || []).map((v) => v.language)));
        if (!list.length) voicesAsked = null;
        return list;
      })
      .catch(() => {
        voicesAsked = null;
        return [];
      });
  }
  return voicesAsked;
}

let current = null;

// The key of the answer being read, or null.
export function readingKey() {
  return current ? current.key : null;
}

// Ends the reading in progress, if any.
export function stopReading() {
  if (!current) return;
  const c = current;
  current = null;
  c.abort.abort();
  c.audio.pause();
  c.audio.removeAttribute('src');
  c.audio.load();
  if (c.url) URL.revokeObjectURL(c.url);
  c.events.end(c.key);
}

function play(audio, signal) {
  return new Promise((resolve, reject) => {
    const cleanup = () => {
      audio.removeEventListener('ended', done);
      audio.removeEventListener('error', failed);
      signal.removeEventListener('abort', aborted);
    };
    const done = () => {
      cleanup();
      resolve();
    };
    const failed = () => {
      cleanup();
      reject(new Error((audio.error && audio.error.message) || 'audio'));
    };
    const aborted = () => {
      cleanup();
      const e = new Error('aborted');
      e.name = 'AbortError';
      reject(e);
    };
    audio.addEventListener('ended', done);
    audio.addEventListener('error', failed);
    signal.addEventListener('abort', aborted);
    audio.play().catch((e) => {
      cleanup();
      reject(e);
    });
  });
}

/*
    Reads text aloud in language; key names the answer for the buttons, events.start(key),
    events.end(key) and events.error(key, error) follow the reading.  A reading in progress
    ends first.
*/
export async function readAloud(text, language, key, events) {
  stopReading();
  const parts = speechParts(text);
  if (!parts.length) return;
  const state = { key, events, abort: new AbortController(), audio: new Audio(), url: null };
  current = state;
  events.start(key);
  const ask = (i) => {
    const p = postAudio('/v1/audio/speech', { model: 'albireo-speech', input: parts[i], voice: language,
      response_format: 'wav' }, state.abort.signal);
    p.catch(() => {});
    return p;
  };
  try {
    let next = ask(0);
    for (let i = 0; i < parts.length; i++) {
      const blob = await next;
      if (current !== state) return;
      next = i + 1 < parts.length ? ask(i + 1) : null;
      if (state.url) URL.revokeObjectURL(state.url);
      state.url = URL.createObjectURL(blob);
      state.audio.src = state.url;
      await play(state.audio, state.abort.signal);
      if (current !== state) return;
    }
    if (current === state) stopReading();
  } catch (e) {
    if (current !== state) return;
    stopReading();
    if (e.name !== 'AbortError') events.error(key, e);
  }
}
