// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

// Calls to the local service: its OpenAI compatible interface and its own interface.

// An error of the service, with the code of its answer (model_not_found,
// context_length_exceeded, unloaded...), which the interface explains in its language.
function serviceError(message, code) {
  const e = new Error(message);
  if (code) e.code = code;
  return e;
}

/*
    The error of an answer of the service, with retryAfter, the seconds its Retry-After header asks to wait before the
    request is sent again (busy, engine_busy, starting, model_loading), when it gives them.
*/
async function errorOf(r) {
  let e = null;
  try {
    const j = await r.json();
    if (j && j.error) {
      e = typeof j.error === 'string' ? serviceError(j.error, j.code) : serviceError(j.error.message || r.statusText,
        j.error.code);
    }
  } catch (x) {
    // The body was not JSON.
  }
  if (!e) e = serviceError(r.status + ' ' + r.statusText);
  const wait = Number(r.headers.get('Retry-After'));
  if (wait > 0) e.retryAfter = wait;
  return e;
}

export async function getJSON(url, signal) {
  const r = await fetch(url, { cache: 'no-store', signal });
  if (!r.ok) throw await errorOf(r);
  return r.json();
}

// A signal that ends a request after ms milliseconds, where the browser offers one.
function timeout(ms) {
  return typeof AbortSignal.timeout === 'function' ? AbortSignal.timeout(ms) : undefined;
}

/*
    A signal that ends a request after ms milliseconds (a TimeoutError), or as soon as signal ends it (its
    reason): {signal, done}, done forgetting the timer and the listener once the request ended.
*/
export function limitedSignal(signal, ms) {
  const controller = new AbortController();
  const timer = setTimeout(() => {
    controller.abort(new DOMException('The request did not end within ' + Math.round(ms / 1000) + ' seconds.',
      'TimeoutError'));
  }, ms);
  const stop = () => controller.abort(signal.reason);
  if (signal) {
    if (signal.aborted) stop();
    else signal.addEventListener('abort', stop, { once: true });
  }
  return {
    signal: controller.signal,
    done() {
      clearTimeout(timer);
      if (signal) signal.removeEventListener('abort', stop);
    }
  };
}

// A sound of the service (the speech synthesis), as a Blob.
export async function postAudio(url, body, signal) {
  const r = await fetch(url, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal
  });
  if (!r.ok) throw await errorOf(r);
  return r.blob();
}

export async function postJSON(url, body, signal) {
  const r = await fetch(url, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal
  });
  if (!r.ok) throw await errorOf(r);
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
  if (!r.ok) throw await errorOf(r);
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
          if (data.error) throw serviceError(data.error.message || 'error', data.error.code);
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

// The status of the service, which a service that no longer answers leaves unknown after a while.
export function status() {
  return getJSON('/albireo/v1/status', timeout(8000));
}

// The models of the service and of its catalog, in the language lang, read within the time given to the status, so
// that the opening of the application and the readings of the status end when the service does not answer.
export function models(lang) {
  return getJSON('/albireo/v1/models?lang=' + encodeURIComponent(lang || 'en'), timeout(8000));
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
