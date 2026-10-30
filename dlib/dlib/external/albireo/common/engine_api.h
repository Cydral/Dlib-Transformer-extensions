// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_ENGINE_API_H_
#define ALBIREO_ENGINE_API_H_

/*
    C interface between the Albireo host and its inference engines.

    An engine is a shared library built from the sources of engine/ for one class of
    machine: plain x86-64, x86-64 with AVX2 and FMA, Intel NPU and GPU through OpenVINO,
    or NVIDIA GPU through CUDA.  Every variant exports the functions below with the same
    signatures, so the host loads whichever suits the machine and talks to it through this
    single entry point; replacing an engine never requires rebuilding the host.

    Conventions
      - Only C types cross the boundary.  Structured data travels as UTF-8 JSON, either
        in a buffer owned by the caller (functions taking json, capacity and needed), or
        as a pointer and a size valid only during a callback.
      - A function writing JSON into a caller buffer returns ALBIREO_E_BUFFER_TOO_SMALL
        when the document does not fit, and sets *needed to the size required, null
        terminator included, so that the caller can retry with a larger buffer.
      - Error messages are written NUL-terminated into the caller's error buffer, truncated
        when it is too small.
      - Every function is thread-safe.  Event callbacks run on threads of the engine, must
        return quickly and may call albireo_engine_cancel(), and no other function, on the
        engine that invoked them.

    Lifetime
      albireo_engine_create() starts the engine with the configuration of the host;
      models are then loaded and unloaded by identifier; requests are submitted against a
      loaded model and report their progress through events until a single
      ALBIREO_EVENT_DONE or ALBIREO_EVENT_ERROR, after which the callback is never invoked
      again for that request.  albireo_engine_destroy() cancels the requests still running,
      waits for their final event and releases everything.
*/

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(ALBIREO_ENGINE_BUILD)
#    define ALBIREO_ENGINE_EXPORT __declspec(dllexport)
#  else
#    define ALBIREO_ENGINE_EXPORT
#  endif
#  define ALBIREO_ENGINE_CALL __cdecl
#else
#  define ALBIREO_ENGINE_EXPORT __attribute__((visibility("default")))
#  define ALBIREO_ENGINE_CALL
#endif

#define ALBIREO_ENGINE_ABI_VERSION 1u

#ifdef __cplusplus
extern "C" {
#endif

typedef struct albireo_engine albireo_engine;

/* Return codes. */
enum
{
    ALBIREO_OK = 0,
    ALBIREO_E_INVALID_ARGUMENT = 1,
    ALBIREO_E_NOT_FOUND = 2,
    ALBIREO_E_BUSY = 3,
    ALBIREO_E_OUT_OF_MEMORY = 4,
    ALBIREO_E_UNSUPPORTED = 5,
    ALBIREO_E_INTERNAL = 6,
    ALBIREO_E_BUFFER_TOO_SMALL = 7,
    ALBIREO_E_CONTEXT_OVERFLOW = 8
};

/* Kinds of the events delivered to a request callback. */
enum
{
    ALBIREO_EVENT_CONTENT = 1,    /* UTF-8 text appended to the answer */
    ALBIREO_EVENT_REASONING = 2,  /* UTF-8 text appended to the reasoning trace */
    ALBIREO_EVENT_TOOL_CALL = 3,  /* JSON {"index","id","name","arguments"} of a complete call */
    ALBIREO_EVENT_PROGRESS = 4,   /* JSON {"prompt_done","prompt_total","cached"} while reading */
    ALBIREO_EVENT_DONE = 5,       /* JSON {"finish_reason","usage","timings"}, final event */
    ALBIREO_EVENT_ERROR = 6       /* JSON {"code","message"}, final event */
};

/* Severities passed to the log callback. */
enum
{
    ALBIREO_LOG_DEBUG = 0,
    ALBIREO_LOG_INFO = 1,
    ALBIREO_LOG_WARNING = 2,
    ALBIREO_LOG_ERROR = 3
};

typedef void (ALBIREO_ENGINE_CALL *albireo_event_callback)(void* user, int kind, const char* data, size_t size);
typedef void (ALBIREO_ENGINE_CALL *albireo_log_callback)(void* user, int level, const char* message);

/* ABI version the library implements, ALBIREO_ENGINE_ABI_VERSION when it matches the
   host.  The host checks it before calling anything else. */
ALBIREO_ENGINE_EXPORT uint32_t ALBIREO_ENGINE_CALL albireo_engine_abi_version(void);

/* JSON description of the engine and of the devices it found on this machine:
   {"variant","version","backend","build","kernels","devices":[...],"usable":true|false,
   "reason","reason_code","reason_args"}.  An engine that cannot run here gives the reason
   in English, and a code with its values for the windows, which explain it in the language
   of the user: isa_avx2, isa_avx_vnni, isa_avx512 (the processor lacks the instructions of
   the kernels), no_nvidia_driver, old_nvidia_driver {driver, runtime}, no_nvidia_card,
   old_nvidia_card {card, cc}, no_intel_device.
   Safe to call before albireo_engine_create(); the host runs it in a child process to
   rank the engines without risking its own process. */
ALBIREO_ENGINE_EXPORT int ALBIREO_ENGINE_CALL albireo_engine_describe(char* json, size_t capacity, size_t* needed);

/* Starts an engine.  config_json holds the settings of the host (threads, context length,
   number of sequences, memory budgets, devices); missing members take their defaults.
   Returns NULL on failure, with the reason in error. */
ALBIREO_ENGINE_EXPORT albireo_engine* ALBIREO_ENGINE_CALL albireo_engine_create(const char* config_json,
    albireo_log_callback log, void* log_user, char* error, size_t error_capacity);

ALBIREO_ENGINE_EXPORT void ALBIREO_ENGINE_CALL albireo_engine_destroy(albireo_engine* engine);

/* Loads the GGUF model at path (UTF-8) under the identifier model_id.  options_json may
   override the context length and other settings for this model. */
ALBIREO_ENGINE_EXPORT int ALBIREO_ENGINE_CALL albireo_engine_load_model(albireo_engine* engine,
    const char* model_id, const char* path, const char* options_json, char* error, size_t error_capacity);

ALBIREO_ENGINE_EXPORT int ALBIREO_ENGINE_CALL albireo_engine_unload_model(albireo_engine* engine, const char* model_id);

/* Synchronous queries, request_json naming the operation in "op":
     {"op":"stats"}                                   activity, throughput and memory
     {"op":"models"}                                  loaded models and their properties
     {"op":"tokenize","model":id,"text":s}            token identifiers of a text
     {"op":"count","model":id,"messages":[...],...}   tokens of a rendered conversation */
ALBIREO_ENGINE_EXPORT int ALBIREO_ENGINE_CALL albireo_engine_query(albireo_engine* engine,
    const char* request_json, char* json, size_t capacity, size_t* needed);

/* Submits a request and returns at once with its identifier.  Events reach callback,
   invoked with user, until the final one.

   A language model takes the members of an OpenAI chat completion request:
     "model", "messages" (or "prompt", a text read as it is), "tools", "tool_choice",
     "reasoning" (true or false) or "reasoning_effort" or
     "chat_template_kwargs":{"enable_thinking"}, "max_tokens" or "max_completion_tokens",
     "temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty",
     "repeat_penalty", "repeat_last_n", "seed", "stop", "logit_bias".
   Its events are CONTENT, REASONING, TOOL_CALL and PROGRESS, then DONE.

   An embedding model takes "model", "input" (a text, a list of texts, a list of token
   identifiers or a list of such lists, 2048 inputs at most) and "dimensions" (0 or absent
   for the width of the model; otherwise the leading dimensions, normalized again).  Its
   only event is DONE, whose JSON holds "data":[{"index","embedding"}] with each vector as
   the base64 form of its little-endian float32 values at unit length, "dimensions",
   "usage" and "timings"; or ERROR when any input fails. */
ALBIREO_ENGINE_EXPORT int ALBIREO_ENGINE_CALL albireo_engine_submit(albireo_engine* engine,
    const char* request_json, albireo_event_callback callback, void* user, uint64_t* request_id,
    char* error, size_t error_capacity);

/* Asks a request to stop.  Its final event, ALBIREO_EVENT_DONE with the finish reason
   "cancelled", follows shortly.  Unknown or finished identifiers are ignored. */
ALBIREO_ENGINE_EXPORT int ALBIREO_ENGINE_CALL albireo_engine_cancel(albireo_engine* engine, uint64_t request_id);

#ifdef __cplusplus
}
#endif

#endif /* ALBIREO_ENGINE_API_H_ */
