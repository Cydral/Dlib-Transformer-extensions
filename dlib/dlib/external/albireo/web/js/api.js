// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

// Calls to the local service: its OpenAI compatible interface and its own interface.

async function errorOf(r) {
  try {
    const j = await r.json();
    if (j && j.error) return typeof j.error === 'string' ? j.error : j.error.message || r.statusText;
  } catch (e) {
    // The body was not JSON.
  }
  return r.status + ' ' + r.statusText;
}

export async function getJSON(url) {
  const r = await fetch(url, { cache: 'no-store' });
  if (!r.ok) throw new Error(await errorOf(r));
  return r.json();
}

export async function postJSON(url, body, signal) {
  const r = await fetch(url, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal
  });
  if (!r.ok) throw new Error(await errorOf(r));
  return r.json();
}

/*
    Streams a chat completion.  Yields {type: 'chunk', data} for each chunk of the
    interface, and {type: 'progress', data} for the progress of a long prompt, which the
    service sends as comment lines.
*/
export async function* streamChat(body, signal) {
  const r = await fetch('/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify(Object.assign({}, body, { stream: true, stream_options: { include_usage: true } })),
    signal
  });
  if (!r.ok) throw new Error(await errorOf(r));
  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let finished = false;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    if (finished) continue;
    buffer += decoder.decode(value, { stream: true });
    let i;
    while (!finished && (i = buffer.indexOf('\n\n')) >= 0) {
      const block = buffer.slice(0, i);
      buffer = buffer.slice(i + 2);
      for (const line of block.split('\n')) {
        if (line.startsWith('data: ')) {
          const payload = line.slice(6);
          // The response is read to its end, so that the connection stays reusable.
          if (payload === '[DONE]') { finished = true; break; }
          const data = JSON.parse(payload);
          if (data.error) throw new Error(data.error.message || 'error');
          yield { type: 'chunk', data };
        } else if (line.startsWith(': progress ')) {
          try {
            yield { type: 'progress', data: JSON.parse(line.slice(11)) };
          } catch (e) {
            // A malformed progress line is ignored.
          }
        }
      }
    }
  }
}

export function status() {
  return getJSON('/albireo/v1/status');
}

export function models(lang) {
  return getJSON('/albireo/v1/models?lang=' + encodeURIComponent(lang || 'en'));
}

export function usage() {
  return getJSON('/albireo/v1/usage');
}

// Number of tokens of a list of messages for the model of a role, and its context.
export function count(model, messages, tools) {
  const body = { model, messages };
  if (tools) body.tools = tools;
  return postJSON('/albireo/v1/count', body);
}
