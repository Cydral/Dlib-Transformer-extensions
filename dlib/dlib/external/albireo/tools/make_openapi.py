# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
#
# Writes web/openapi.json, the description of the interface of the service shown by the
# documentation page (/docs/).  Run it after a change of the routes of host/server.cpp:
#
#   python3 tools/make_openapi.py

import json
import os
import re


def ref(name):
    return {"$ref": "#/components/schemas/" + name}


def json_body(schema, required=True):
    return {"required": required, "content": {"application/json": {"schema": schema}}}


def reply(description, schema=None):
    if not schema:
        return {"description": description}
    return {"description": description, "content": {"application/json": {"schema": schema}}}


error_replies = {
    "400": reply("Invalid request", ref("Error")),
    "404": reply("Unknown or inactive model", ref("Error")),
    "503": reply("The service starts (code starting), the model loads (code model_loading), or the values of the "
                 "JSON bodies being read at once reach their budget (code busy, Retry-After: 2); the Retry-After "
                 "header gives the seconds to wait", ref("Error")),
}
# The routes whose request the engine computes, which may fail once it started; a stream ends with StreamError then.
engine_replies = dict(error_replies, **{
    "500": reply("The engine failed the request, which was not streamed: type server_error, and code the reason "
                 "the engine gives, such as out_of_memory", ref("Error")),
})
# The routes that generate text, which the engine also refuses for a while when the images already queued for the
# model hold the memory kept for them.
generation_replies = dict(engine_replies, **{
    "503": reply("The service starts (code starting), the model loads (code model_loading), the values of the JSON "
                 "bodies being read at once reach their budget (code busy, Retry-After: 2), or the engine refuses the "
                 "request for a while because the images already queued for the model hold the memory kept for them "
                 "(code engine_busy, Retry-After: 5); the Retry-After header gives the seconds to wait", ref("Error")),
})
admin = [{"adminToken": []}]
offline_reply = {"409": reply("Refused in offline mode", {"type": "object", "properties": {
    "ok": {"type": "boolean"}, "error": {"type": "string"}}})}
model_id = {"name": "id", "in": "path", "required": True, "schema": {"type": "string", "pattern": "^[A-Za-z0-9._-]+$"},
            "description": "Identifier of the model, as listed by /albireo/v1/models."}
ok = {"type": "object", "properties": {"ok": {"type": "boolean"}, "error": {"type": "string"}}}
# The codes of the errors of the updates (host/updater.h), and of the refusals of the imports by path
# (service::import_folder() in host/service.h).
update_codes = ["update_check", "update_record", "update_feed_missing", "update_feed_missing_folder",
                "update_signature", "update_feed_mismatch", "update_outside", "update_installer_mismatch",
                "update_feed_invalid", "update_download", "update_not_published", "update_not_ready",
                "update_invalid", "update_start", "update_package_manager", "stopping"]
import_codes = ["import_path_relative", "import_path_network", "import_path_device", "import_path_invalid",
                "import_use_upload", "import_not_bundle", "import_not_found", "import_path_link",
                "import_folder_insecure"]
import_refusal = {"allOf": [ref("Refusal"), {"type": "object", "properties": {"code": {"type": "string",
                  "enum": import_codes}}}]}

schemas = {
    "Error": {"type": "object", "properties": {"error": {"type": "object", "properties": {
        "message": {"type": "string"}, "type": {"type": "string"}, "param": {"nullable": True},
        "code": {"type": "string", "nullable": True}}}}},
    "Message": {
        "type": "object", "required": ["role"],
        "properties": {
            "role": {"type": "string", "enum": ["system", "developer", "user", "assistant", "tool"]},
            "content": {"description": "Text, or a list of parts: text parts, and for a model that reads images "
                                       "(inputs of /v1/models) image_url parts, 16 images per request at most.",
                        "oneOf": [{"type": "string"}, {"type": "array", "items": ref("ContentPart")}],
                        "nullable": True},
            "reasoning_content": {"type": "string", "description": "Reasoning of an earlier assistant message."},
            "tool_calls": {"type": "array", "items": ref("ToolCall")},
            "tool_call_id": {"type": "string", "description": "For a tool message: the call it answers."},
            "name": {"type": "string"}}},
    "ContentPart": {"type": "object", "required": ["type"], "properties": {
        "type": {"type": "string", "enum": ["text", "image_url", "input_audio"]},
        "text": {"type": "string"},
        "image_url": {"description": "The image: an object with its url, or its url alone as a string.", "oneOf": [
            {"type": "string", "description": "The data URL of the image, as url in the object form."},
            {"type": "object", "required": ["url"], "properties": {
                "url": {"type": "string", "description": "A PNG or JPEG file as a data URL, data:image/png;base64,... "
                        "or data:image/jpeg;base64,...; the service fetches no address (code invalid_image_url), "
                        "reads no other format (invalid_image_format), and refuses more than 20 MB, 8192 pixels on "
                        "a side or 40 megapixels."},
                "detail": {"type": "string", "enum": ["low", "auto", "high"], "default": "auto",
                           "description": "The tokens an image takes at most: for Qwen3-VL 256, 576 or 1024 tokens "
                                          "of 32 pixels square; for a LLaVA-NeXT model (EuroVLM) the whole image and "
                                          "1, 2 or 4 tiles of 384 pixels."}}}]},
        "input_audio": {"type": "object", "description": "A sound, for a model whose inputs list audio; a model "
                        "that reads no sound refuses the request (code audio_input_not_supported).",
                        "properties": {"data": {"type": "string", "description": "The sound in base64."},
                                       "format": {"type": "string", "example": "wav"}}}},
        "description": "A part of the content of a message. A model that reads text only refuses the images (code "
                       "image_input_not_supported), a model that reads no sound refuses the sounds (code "
                       "audio_input_not_supported), and a request carries 16 images at most (too_many_images)."},
    "ToolCall": {"type": "object", "properties": {
        "index": {"type": "integer", "description": "Position of the call, in the chunks of a stream."},
        "id": {"type": "string"}, "type": {"type": "string", "enum": ["function"]},
        "function": {"type": "object", "properties": {"name": {"type": "string"},
                                                      "arguments": {"type": "string", "description": "JSON text."}}}}},
    "Tool": {"type": "object", "required": ["type", "function"], "properties": {
        "type": {"type": "string", "enum": ["function"]},
        "function": {"type": "object", "required": ["name"], "properties": {
            "name": {"type": "string"}, "description": {"type": "string"},
            "parameters": {"type": "object", "description": "JSON Schema of the arguments."}}}}},
    "Sampling": {"type": "object", "description": "A setting the request leaves out takes the value the authors of the "
                 "family of the model advise, for a request of text alone or one with images: temperature 1, top_p "
                 "0.95 and top_k 20 for the thinking variants of Qwen3-VL, with presence_penalty 1.5 over the whole "
                 "answer for text alone, and temperature 0.7, top_p 0.8, top_k 20 and presence_penalty 1.5 when the "
                 "engine ends their reasoning at once; for its instruct variants, temperature 1, top_p 1, top_k 40 and "
                 "presence_penalty 2 for text alone, temperature 0.7, top_p 0.8, top_k 20 and presence_penalty 1.5 "
                 "with images; temperature 0.3, top_p 0.8 and top_k 20 for CyberSecQwen-4B; temperature 0.7, top_p 0.8 "
                 "and top_k 20 for Qwen3 without reasoning; the same with repeat_penalty 1.05 for Qwen2.5; temperature "
                 "0.3, top_p 0.95 and top_k 20 for Foundation-Sec 1.1 8B Instruct and Foundation-Sec-8B-Reasoning; "
                 "temperature 1 and top_p 0.95 without top_k for Granite 4.2, with and without reasoning; temperature "
                 "0.1 for Ministral 3 and 0.15 for Devstral and the other Mistral models without reasoning, without "
                 "top_p or top_k; temperature 0.6, top_p 0.95 and top_k 20 for the other models that reason, and 0.7, "
                 "0.9 and 40 for the others; min_p 0 for all of them.",
                 "properties": {
        "max_tokens": {"type": "integer",
                       "description": "Tokens to generate at most; by default up to the end of the context."},
        "max_completion_tokens": {"type": "integer"},
        "temperature": {"type": "number", "minimum": 0, "maximum": 2},
        "top_p": {"type": "number"},
        "top_k": {"type": "integer", "minimum": 0, "maximum": 1024,
                  "description": "Keeps the k most likely tokens, among which top_p and min_p then choose; 0 sets "
                                 "no limit, and the 1024 most likely tokens are kept. A value outside 0 to 1024 is "
                                 "refused (400). Without effect at temperature 0, which takes the most likely "
                                 "token."},
        "min_p": {"type": "number"},
        "presence_penalty": {"type": "number"}, "frequency_penalty": {"type": "number"},
        "repeat_penalty": {"type": "number"},
        "repetition_penalty": {"type": "number", "description": "The same setting as repeat_penalty, under the "
                               "name other clients give it."},
        "repeat_last_n": {"type": "integer", "minimum": -1,
                          "description": "The tokens the penalties read, the last ones, the end of the prompt "
                                         "included; -1 reads the whole answer alone, without the prompt and "
                                         "without the end of a reasoning the engine writes. Without it, 64, or "
                                         "the whole answer for a model whose authors advise a presence "
                                         "penalty."},
        "seed": {"type": "integer"},
        "stop": {"oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}, "maxItems": 16}]},
        "logit_bias": {"type": "object", "additionalProperties": {"type": "number"},
                       "description": "Bias added to the logits of tokens, keyed by token identifiers of the "
                                      "vocabulary of the model; another key is refused."}}},
    "ChatCompletionRequest": {"allOf": [ref("Sampling"), {
        "type": "object", "required": ["messages"],
        "properties": {
            "model": {"type": "string",
                      "description": "A model identifier, a function (albireo-general, albireo-code, "
                                     "albireo-security), albireo-background or auto. A function without a model "
                                     "of its own is served by the general model; albireo-background, the work done "
                                     "in the background (titles, summaries, memory), by the model that comes with "
                                     "Albireo, else the general one.",
                      "example": "albireo-general"},
            "messages": {"type": "array", "items": ref("Message")},
            "stream": {"type": "boolean", "default": False},
            "stream_options": {"type": "object", "properties": {"include_usage": {"type": "boolean"}}},
            "tools": {"type": "array", "items": ref("Tool")},
            "tool_choice": {"description": "none disables the tools.",
                            "oneOf": [{"type": "string"}, {"type": "object"}]},
            "reasoning": {"type": "boolean", "description": "Enables or disables the reasoning of the models that "
                          "support both modes. The reasoning streams in reasoning_content. A model whose template "
                          "opens the reasoning of every answer (reasoning_always in the description of the model: "
                          "the thinking variants of Qwen3 and Qwen3-VL, Foundation-Sec-8B-Reasoning) has that "
                          "reasoning opened whatever the request asks; without reasoning, the engine ends it at once "
                          "and the model answers directly (reasoning_budget)."},
            "think": {"type": "boolean", "description": "The same setting as reasoning, under the name other "
                      "clients give it; when both are given, think applies."},
            "reasoning_effort": {"type": "string", "enum": ["none", "low", "medium", "high"],
                                 "description": "none disables the reasoning; low gives it the budget of a short "
                                                "reasoning, high lifts its bound by default (reasoning_budget)."},
            "reasoning_budget": {"type": "integer", "minimum": -1, "nullable": True,
                                 "description": "The tokens the reasoning may take: beyond them, the engine ends it "
                                                "with a sentence that asks the model to answer from what it has "
                                                "thought, then the marker that closes the reasoning; that end counts "
                                                "among the tokens of the answer and stays out of reasoning_content. "
                                                "-1 or null sets no limit, 0 ends the reasoning at once. Without it, "
                                                "the reasoning takes half of max_tokens at most (half of the room "
                                                "the context leaves to the answer when the request gives no "
                                                "max_tokens), 32768 tokens at most, so that the answer keeps room "
                                                "after it, and the effort high sets no limit; a model that always "
                                                "reasons asked for no reasoning has its reasoning ended at once, and "
                                                "the effort low gives it 512 tokens, 256 for the work in the "
                                                "background (albireo-background). The answer of a model asked for no "
                                                "reasoning receives max_tokens once its reasoning ends, and the stop "
                                                "sequences apply to that answer alone. A reasoning whose last 384 "
                                                "tokens repeat, for nine tenths of them, passages of 16 tokens it "
                                                "wrote twice before ends in the same way."},
            "priority": {"type": "string", "enum": ["background"],
                         "description": "background marks work done in the background, as the model "
                                        "albireo-background does: while it writes its answer, the passes keep their "
                                        "whole room for the prompts of the other requests, where an answer that a "
                                        "user waits for limits the tokens of prompts a pass reads. The web "
                                        "application asks it to read ahead the beginning of a discussion after a "
                                        "summary."},
            "chat_template_kwargs": {"type": "object", "properties": {"enable_thinking": {"type": "boolean"}}},
            "n": {"type": "integer", "enum": [1]}}}],
        "example": {"model": "albireo-general", "messages": [{"role": "user", "content": "Hello!"}],
                    "max_tokens": 200}},
    "Usage": {"type": "object", "properties": {
        "prompt_tokens": {"type": "integer"}, "completion_tokens": {"type": "integer"},
        "total_tokens": {"type": "integer"},
        "prompt_tokens_details": {"type": "object", "properties": {"cached_tokens": {"type": "integer"}}}}},
    "Timings": {"type": "object", "properties": {
        "queued_ms": {"type": "number"}, "prompt_ms": {"type": "number"}, "generation_ms": {"type": "number"},
        "prompt_per_second": {"type": "number"}, "generation_per_second": {"type": "number"}}},
    "ChatCompletion": {"type": "object", "properties": {
        "id": {"type": "string"}, "object": {"type": "string", "enum": ["chat.completion"]},
        "created": {"type": "integer"}, "model": {"type": "string"},
        "choices": {"type": "array", "items": {"type": "object", "properties": {
            "index": {"type": "integer"}, "message": ref("Message"),
            "finish_reason": {"type": "string", "enum": ["stop", "length", "tool_calls", "cancelled"]}}}},
        "repetition": {"type": "boolean", "description": "Present and true when the engine stopped the answer "
                       "because it repeated itself (finish_reason length): nine tenths of its last 384 tokens "
                       "repeated passages of 16 tokens it had written twice before."},
        "usage": ref("Usage"), "timings": ref("Timings"), "system_fingerprint": {"type": "string"}}},
    "ChatCompletionChunk": {"type": "object", "description": "One event of a stream (text/event-stream), as "
                            "'data: <chunk>' lines ended by 'data: [DONE]'. With stream_options.include_usage, a "
                            "last chunk before [DONE] carries usage and timings and no choice. An error once the "
                            "stream has started ends it with a last event 'data: {\"error\": {\"message\": ..., "
                            "\"type\": \"server_error\", \"code\": ...}}' (StreamError), without [DONE]. Comment "
                            "lines (': keep-alive', ': progress {...}') may come between the events.",
                            "properties": {
        "id": {"type": "string"}, "object": {"type": "string", "enum": ["chat.completion.chunk"]},
        "created": {"type": "integer"}, "model": {"type": "string"},
        "choices": {"type": "array", "items": {"type": "object", "properties": {
            "index": {"type": "integer"},
            "delta": {"type": "object", "properties": {"role": {"type": "string"}, "content": {"type": "string"},
                                                       "reasoning_content": {"type": "string"},
                                                       "tool_calls": {"type": "array", "items": ref("ToolCall")}}},
            "logprobs": {"nullable": True, "description": "Always null."},
            "finish_reason": {"type": "string", "nullable": True}}}},
        "repetition": {"type": "boolean", "description": "In the chunk of the finish_reason, present and true when "
                       "the engine stopped the answer because it repeated itself."},
        "usage": ref("Usage"), "timings": ref("Timings")}},
    "StreamError": {"type": "object", "description": "The last event of a stream that an error ended, without "
                    "'data: [DONE]' after it.", "properties": {"error": {"type": "object", "properties": {
                        "message": {"type": "string"}, "type": {"type": "string", "enum": ["server_error"]},
                        "code": {"type": "string", "nullable": True,
                                 "description": "The code of the engine, such as out_of_memory."}}}}},
    "CompletionRequest": {"allOf": [ref("Sampling"), {"type": "object", "required": ["prompt"], "properties": {
        "model": {"type": "string", "example": "albireo-general"}, "prompt": {"type": "string"},
        "stream": {"type": "boolean"}}}]},
    "Completion": {"type": "object", "properties": {
        "id": {"type": "string"}, "object": {"type": "string", "enum": ["text_completion"]},
        "choices": {"type": "array", "items": {"type": "object", "properties": {
            "index": {"type": "integer"}, "text": {"type": "string"}, "finish_reason": {"type": "string"}}}},
        "usage": ref("Usage")}},
    "SpeechRequest": {"type": "object", "required": ["input"], "properties": {
        "model": {"type": "string", "description": "Not checked: one synthesizer serves every model name."},
        "input": {"type": "string",
                  "description": "The text read aloud, at most 8192 bytes of UTF-8, counted in bytes and not in "
                                 "characters: a longer text is refused (code string_above_max_length). The letters "
                                 "of the Latin scripts, the digits and the usual punctuation are read; any other "
                                 "character is a pause."},
        "voice": {"type": "string",
                  "description": "A voice of Albireo, en-US, en-GB, fr-FR, de-DE, it-IT or es-ES, or its "
                                 "language, en, fr, de, it or es; another name (the voices of other services) "
                                 "takes the voice of language."},
        "language": {"type": "string", "enum": ["en", "fr", "de", "it", "es"], "default": "en",
                     "description": "Albireo: the language of the voice when voice names none of Albireo."},
        "speed": {"type": "number", "minimum": 0.25, "maximum": 4, "default": 1,
                  "description": "The speed of the reading, applied from 0.5 to 2."},
        "response_format": {"type": "string", "enum": ["wav", "pcm"], "default": "wav",
                            "description": "A WAVE file, or its 16-bit mono samples at 16 kHz alone."}}},
    "EmbeddingRequest": {"type": "object", "required": ["input"], "properties": {
        "model": {"type": "string", "description": "albireo-embedding, auto, or the identifier of an active embedding "
                                                   "model.", "example": "albireo-embedding"},
        "input": {"description": "A text, a list of texts, or token lists.", "oneOf": [
            {"type": "string"}, {"type": "array", "items": {"type": "string"}, "maxItems": 2048},
            {"type": "array", "items": {"type": "integer"}},
            {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}}]},
        "encoding_format": {"type": "string", "enum": ["float", "base64"], "default": "float",
                            "description": "base64: the little-endian float32 values of each vector."},
        "dimensions": {"type": "integer", "minimum": 1, "description": "Keeps the leading dimensions of each "
                       "vector, normalized again, for the models trained to keep their meaning when shortened "
                       "(Qwen3 Embedding: 32 to 1024)."},
        "user": {"type": "string", "description": "Accepted and ignored."}},
        "example": {"model": "albireo-embedding", "input": ["The service runs on this computer.",
                                                            "Where does the service run?"]}},
    "EmbeddingList": {"type": "object", "properties": {
        "object": {"type": "string", "enum": ["list"]}, "model": {"type": "string"},
        "data": {"type": "array", "items": {"type": "object", "properties": {
            "object": {"type": "string", "enum": ["embedding"]}, "index": {"type": "integer"},
            "embedding": {"description": "Unit vector: numbers, or a base64 string.",
                          "oneOf": [{"type": "array", "items": {"type": "number"}}, {"type": "string"}]}}}},
        "usage": {"type": "object", "properties": {"prompt_tokens": {"type": "integer"},
                                                   "total_tokens": {"type": "integer"}}}}},
    "ModelList": {"type": "object", "properties": {"object": {"type": "string"}, "data": {"type": "array", "items": {
        "type": "object", "properties": {"id": {"type": "string"}, "object": {"type": "string"},
                                         "owned_by": {"type": "string"},
                                         "roles": {"type": "array", "items": {"type": "string"}},
                                         "alias_of": {"type": "string"},
                                         "inputs": {"type": "array", "items": {"type": "string",
                                                                              "enum": ["text", "image", "audio"]},
                                                    "description": "What the model reads, text and images; an "
                                                                   "alias gives those of its model."},
                                         "publisher": {"type": "string",
                                                       "description": "The company that made the model, empty when "
                                                                      "unknown."}}}}}},
    "Settings": {"type": "object", "description": "Settings the tray may change; a change of port, engine or memory "
                 "restarts the service.", "properties": {
        "port": {"type": "integer", "minimum": 1024, "maximum": 65535},
        "language": {"type": "string", "enum": ["auto", "en", "fr", "de", "it", "es"]},
        "engine": {"type": "string", "enum": ["auto", "best", "cpu", "cpu-avx2", "cpu-avxvnni", "openvino", "cuda"],
                   "description": "auto: the accelerator of the computer, the NVIDIA card (cuda), else the NPU and "
                   "the graphics processor of Intel together, else the one of them present, with the processor "
                   "computing the products its device does not take; the processor engine when no accelerator is "
                   "usable or when a measure found the accelerator slower. best: the fastest measured engine. A "
                   "variant: that engine, with the device of accelerator for the engines that have several."},
        "accelerator": {"type": "string", "enum": ["auto", "none", "npu", "gpu", "both"]},
        "v2ram": {"type": "boolean", "description": "Extended device memory of the graphics engines."},
        "v2ram_max_mb": {"type": "integer", "description": "Limit of the extended memory, 0 for automatic."},
        "device_mb": {"type": "integer", "description": "Memory of the device for the copies of the weights of "
                      "the models, in megabytes, 0 for automatic."},
        "threads": {"type": "integer"},
        "context": {"type": "integer", "description": "Context of the language models, 0 for the context each "
                    "model was trained for, which the memory the system can commit (its physical memory and its "
                    "paging file or swap) reduces when it cannot hold the keys and values of one full context."},
        "max_sequences": {"type": "integer", "minimum": 1, "maximum": 64},
        "kv_mb": {"type": "integer", "description": "Memory of the keys and values of each language model, in "
                  "megabytes, 0 for two full contexts within the memory the system can commit less a reserve."},
        "kv_type": {"type": "string", "enum": ["auto", "q8", "f16"], "default": "auto",
                    "description": "Storage of the keys and values of the language models: q8, 8-bit integers by "
                                   "groups of 32 values with a scale, the keys rotated; f16, half precision. auto "
                                   "takes q8 for the models whose heads hold whole groups."},
        "draft_tokens": {"type": "integer", "minimum": 0, "maximum": 6, "default": 4,
                         "description": "Tokens of the drafts of the answers a step of a language model verifies at "
                                        "most: the tokens that followed the last ones in the context, which the "
                                        "model keeps when it would have drawn them itself, the answers keeping the "
                                        "distribution of their tokens. 0 turns the drafts off."},
        "integer_prompt": {"type": "boolean", "default": True,
                           "description": "Products of the prompts on 8-bit integers for the weights in Q4_0, Q4_K, "
                                          "Q5_K, Q6_K and Q8_0, the activations rounded to 8 bits by groups of 32 "
                                          "values; false computes them on floats."},
        "check_updates": {"type": "boolean"},
        "offline": {"type": "boolean", "description": "No access to the network: the catalog, the models and the "
                    "updates come from imports, and the service opens no connection."},
        "proxy": {"type": "string",
                  "description": "Proxy of the downloads: empty for the proxy of the system, none for a direct "
                                 "connection, or http://[user:password@]host:port.  The settings give the password "
                                 "as ********; that value sent back keeps the stored password when the user, the "
                                 "host and the port stay the same, and is refused (code proxy_password) otherwise."},
        "web_access": {"type": "boolean", "default": True,
                       "description": "The web access of the Web button, /albireo/v1/web/search and "
                                      "/albireo/v1/web/fetch; false turns it off."},
        "web_search": {"type": "string", "enum": ["wikipedia", "none"], "default": "wikipedia",
                       "description": "The provider of /albireo/v1/web/search: wikipedia, the REST interface of "
                                      "Wikipedia; or none, which turns the search off while /albireo/v1/web/fetch "
                                      "stays available. A settings.json of an earlier version that names another "
                                      "provider is read as wikipedia."}}},
    "WebSearchRequest": {"type": "object", "required": ["query"], "properties": {
        "query": {"type": "string", "minLength": 1, "maxLength": 400,
                  "description": "The text searched, 1 to 400 characters once the spaces at its ends are removed."},
        "count": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5, "description": "Results at most."},
        "language": {"type": "string", "pattern": "^[a-z]{2,3}(-([A-Z]{2}|[0-9]{3}))?$", "example": "fr",
                     "description": "The language of the results: the site of Wikipedia of that language (en "
                                    "without one)."}},
        "example": {"query": "Albireo double star", "count": 5, "language": "en"}},
    "WebSearchResults": {"type": "object", "properties": {
        "provider": {"type": "string", "enum": ["wikipedia"]},
        "results": {"type": "array", "description": "The results whose address is http or https, count at most.",
                    "items": {"type": "object", "properties": {
                        "title": {"type": "string"},
                        "url": {"type": "string", "description": "An http or https address; for Wikipedia, "
                                "https://<language>.wikipedia.org/wiki/<key of the page>."},
                        "snippet": {"type": "string", "description": "The excerpt of Wikipedia without its tags "
                                    "and entities, else the description of the page."}}}}}},
    "WebFetchRequest": {"type": "object", "required": ["url"], "properties": {
        "url": {"type": "string", "description": "An http or https address of a public host, without user "
                "information.", "example": "https://en.wikipedia.org/wiki/Albireo"},
        "language": {"type": "string", "pattern": "^[a-z]{2,3}(-([A-Z]{2}|[0-9]{3}))?$",
                     "description": "Sent as Accept-Language."}}},
    "WebPage": {"type": "object", "properties": {
        "url": {"type": "string", "description": "The address asked."},
        "final_url": {"type": "string", "description": "The address that answered, after the redirections."},
        "status": {"type": "integer", "description": "The status of the site, 4xx and 5xx included, with their "
                   "body."},
        "content_type": {"type": "string", "description": "The media type of the Content-Type of the site, in "
                         "lowercase, without its parameters.", "example": "text/html"},
        "charset": {"type": "string", "description": "The charset of the Content-Type, empty without one.",
                    "example": "utf-8"},
        "size": {"type": "integer", "description": "The bytes read."},
        "truncated": {"type": "boolean", "description": "True when a text was cut at 2 MB."},
        "data": {"type": "string", "format": "byte", "description": "The bytes read, in base64: the web "
                 "application decodes a text with TextDecoder and the charset, HTML with DOMParser, PDF with its "
                 "reader."}}},
    "Engines": {"type": "object", "properties": {
        "starting": {"type": "boolean", "description": "True while the engines are being described."},
        "engines": {"type": "array", "items": {"type": "object", "properties": {
            "variant": {"type": "string"}, "path": {"type": "string"}, "usable": {"type": "boolean"},
            "version": {"type": "string"}, "build": {"type": "string"}, "current": {"type": "boolean"},
            "devices": {"type": "array", "items": {"type": "object", "properties": {
                "type": {"type": "string", "enum": ["cpu", "gpu", "npu"]}, "id": {"type": "string"},
                "name": {"type": "string"}}}},
            "reason": {"type": "string", "description": "Why the engine cannot run here, in English."},
            "reason_code": {"type": "string", "description": "The code of the reason: isa_avx2, isa_avx_vnni, "
                            "no_nvidia_driver, old_nvidia_driver, no_nvidia_card, old_nvidia_card, "
                            "no_intel_device, intel_device_unusable, library_failed, probe_timeout."},
            "reason_args": {"type": "object", "additionalProperties": {"type": "string"}},
            "missing": {"type": "string", "description": "For an engine that runs: why its runtime left out the "
                        "other devices it looks for, such as the message of a plugin of OpenVINO."}}}},
        "options": {"type": "array", "description": "The choices of engine and device.", "items": {
            "type": "object", "properties": {
                "key": {"type": "string", "example": "openvino:gpu"}, "variant": {"type": "string"},
                "device": {"type": "string"}, "current": {"type": "boolean"},
                "kind": {"type": "string", "enum": ["cpu", "gpu", "npu", "both"]},
                "names": {"type": "array", "items": {"type": "string"},
                          "description": "The names of the devices of the choice."},
                "measure": {"type": "object", "description": "The last measure of the choice."}}}},
        "current": {"type": "string", "description": "The key of the choice in use."},
        "setting": {"type": "string"}, "accelerator": {"type": "string"},
        "accelerators_available": {"type": "boolean"},
        "accelerators_reason": {"type": "string", "enum": ["not_installed", "unusable"]}}},
    "UpdateState": {"type": "object", "description": "The updates: their state, the versions, the notes by language, "
                    "and an error with its code and its values, which the tray translates.", "properties": {
        "state": {"type": "string", "enum": ["idle", "checking", "up_to_date", "downloading", "ready", "error"]},
        "current": {"type": "string"}, "latest": {"type": "string"}, "file": {"type": "string"},
        "notes": {"type": "object", "additionalProperties": {"type": "string"}},
        "received": {"type": "integer"}, "total": {"type": "integer"}, "checked_at": {"type": "integer"},
        "error": {"type": "string", "description": "The error in English."},
        "code": {"type": "string", "enum": update_codes,
                 "description": "update_check {e}, update_record {version}, update_feed_missing {version}, "
                                "update_feed_missing_folder {folder}, update_signature {file}, update_feed_mismatch "
                                "{file, version}, update_outside {file}, update_installer_mismatch {version}, "
                                "update_feed_invalid, update_download {e}, update_not_published {file}, "
                                "update_not_ready, update_invalid {version}, update_start {code}, "
                                "update_package_manager, stopping."},
        "args": {"type": "object", "additionalProperties": {"type": "string"}}}},
    "Refusal": {"type": "object", "description": "A refused operation, with the code of its reason and its values.",
                "properties": {"ok": {"type": "boolean", "enum": [False]}, "error": {"type": "string"},
                               "code": {"type": "string"},
                               "args": {"type": "object", "additionalProperties": {"type": "string"}}}},
}

paths = {
    "/v1/models": {"get": {"tags": ["OpenAI"], "summary": "Models of the engine and aliases of the functions",
                           "responses": {"200": reply("The models", ref("ModelList"))}}},
    "/v1/chat/completions": {"post": {
        "tags": ["OpenAI"], "summary": "Chat completion, streamed or not",
        "description": "Compatible with the OpenAI interface. Requests from several clients are computed together. "
                       "A stream also carries ': progress {...}' comment lines while a long prompt is read, and "
                       "': keep-alive' lines while no token comes. An error once the stream has started ends it "
                       "with a last event 'data: {\"error\": {...}}' (StreamError) instead of 'data: [DONE]'.",
        "requestBody": json_body(ref("ChatCompletionRequest")),
        "responses": dict({"200": {"description": "The completion, or a stream of chunks", "content": {
            "application/json": {"schema": ref("ChatCompletion")},
            "text/event-stream": {"schema": {"oneOf": [ref("ChatCompletionChunk"), ref("StreamError")]}}}}},
            **generation_replies)}},
    "/v1/completions": {"post": {
        "tags": ["OpenAI"], "summary": "Text completion of a raw prompt",
        "description": "A stream gives text_completion events ended by 'data: [DONE]', or by a StreamError event "
                       "after an error.",
        "requestBody": json_body(ref("CompletionRequest")),
        "responses": dict({"200": reply("The completion", ref("Completion"))}, **generation_replies)}},
    "/v1/embeddings": {"post": {
        "tags": ["OpenAI"], "summary": "Vectors of texts, for search and comparison",
        "description": "Computed by the embedding model of the service, which the search in the documents of the "
                       "web application uses. The inputs of every client are computed together, and the passes of "
                       "the embedding model stay short while a conversation is being answered.",
        "requestBody": json_body(ref("EmbeddingRequest")),
        "responses": dict({"200": reply("One vector per input, in the order of the inputs", ref("EmbeddingList"))},
                          **engine_replies)}},
    "/v1/audio/speech": {"post": {
        "tags": ["OpenAI"], "summary": "A text read aloud",
        "description": "The speech synthesis that comes with Albireo, which needs no network and no voice of the "
                       "system, in six voices of five languages. The program albireo-speech reads the text, "
                       "started by the service with the rights of a process of low integrity; two read at a time "
                       "and the others wait up to 30 seconds.",
        "requestBody": json_body(ref("SpeechRequest")),
        "responses": {
            "200": {"description": "The sound, 16-bit mono samples at 16 kHz",
                    "content": {"audio/wav": {"schema": {"type": "string", "format": "binary"}},
                                "audio/pcm": {"schema": {"type": "string", "format": "binary"}}}},
            "400": reply("Invalid request: input, voice, language, speed or response_format", ref("Error")),
            "500": reply("The synthesis failed (code speech_failed)", ref("Error")),
            "503": reply("The speech program or its voices are not installed (code speech_unavailable)",
                         ref("Error"))}}},
    "/albireo/v1/speech/voices": {"get": {
        "tags": ["Service"], "summary": "Voices of the speech synthesis",
        "description": "The voices whose files the installation holds, which /v1/audio/speech reads with; none "
                       "when the speech program is missing.",
        "responses": {"200": reply("The voices", {"type": "object", "properties": {
            "voices": {"type": "array", "items": {"type": "object", "properties": {
                "id": {"type": "string", "description": "en-US, en-GB, fr-FR, de-DE, it-IT or es-ES."},
                "language": {"type": "string", "enum": ["en", "fr", "de", "it", "es"]}}}},
            "format": {"type": "string", "enum": ["wav"]},
            "sample_rate": {"type": "integer", "enum": [16000]}}})}}},
    "/albireo/v1/status": {"get": {
        "tags": ["Service"], "summary": "State of the service, of the engine and of the models",
        "description": "The server answers while the service starts: state is \"starting\" and starting names the "
                       "step in progress, \"engines\" then \"models\", until the service is \"ready\".",
        "responses": {"200": reply("The state", {"type": "object", "properties": {
            "product": {"type": "string", "example": "Albireo"},
            "edition": {"type": "string", "description": "The edition, Community Edition."},
            "version": {"type": "string"},
            "author": {"type": "string"}, "email": {"type": "string"}, "copyright": {"type": "string"},
            "project": {"type": "string", "description": "The address of the source code of Albireo."},
            "settings": {"type": "object", "description": "The settings of the service that the web application "
                         "follows, such as the language of the interface.", "properties": {
                             "language": {"type": "string", "enum": ["auto", "en", "fr", "de", "it", "es"]},
                             "engine": {"type": "string"}, "accelerator": {"type": "string"},
                             "v2ram": {"type": "boolean"}, "v2ram_max_mb": {"type": "integer"},
                             "device_mb": {"type": "integer"}, "check_updates": {"type": "boolean"},
                             "offline": {"type": "boolean"}}},
            "engine": {"type": "object", "description": "The engine running.", "properties": {
                "running": {"type": "boolean"}, "variant": {"type": "string", "example": "cpu-avx2"},
                "key": {"type": "string"}, "device": {"type": "string"}, "error": {"type": "string"},
                "description": {"type": "string"},
                "stats": {"type": "object", "nullable": True, "description": "Null while no engine runs.",
                          "properties": {
                              "active": {"type": "integer", "description": "Requests being computed."},
                              "waiting": {"type": "integer"},
                              "generation_tokens_per_second": {"type": "number",
                                                               "description": "Over the last five seconds."},
                              "prompt_tokens_per_second": {"type": "number"},
                              "memory": {"type": "object", "description": "The memory of the graphics engines: "
                                         "vram_budget, device_bytes, scratch_bytes, pinned_bytes, store_bytes.",
                                         "additionalProperties": {"type": "integer"}}}}}},
            "host": {"type": "object", "description": "The computer.", "properties": {
                "name": {"type": "string"}, "os": {"type": "string"},
                "cpu": {"type": "object", "properties": {
                    "name": {"type": "string"}, "cores": {"type": "integer"}, "threads": {"type": "integer"},
                    "features": {"type": "array", "items": {"type": "string"}}}},
                "memory": {"type": "object", "properties": {
                    "total": {"type": "integer"}, "available": {"type": "integer"},
                    "commit_total": {"type": "integer", "description": "The memory the system can commit, its "
                                     "physical memory and its paging file or swap together."},
                    "commit_available": {"type": "integer"}}}}},
            "usage": {"type": "object", "description": "The requests and the tokens since the service started.",
                      "properties": {"requests": {"type": "integer"}, "prompt_tokens": {"type": "integer"},
                                     "completion_tokens": {"type": "integer"}}},
            "state": {"type": "string", "enum": ["starting", "ready"]},
            "starting": {"type": "string", "enum": ["engines", "models"]},
            "active": {"type": "object", "additionalProperties": {"type": "string"},
                       "description": "The model of each function: general, code, security, embedding. A "
                                      "function without a model chosen for it takes the general model, else the "
                                      "model that comes with Albireo."},
            "selected": {"type": "object", "additionalProperties": {"type": "string"},
                         "description": "The models chosen for the functions, which Deactivate gives back."},
            "background": {"type": "string",
                           "description": "The model that comes with Albireo and serves albireo-background, which "
                                          "stays loaded; empty without one."},
            "loaded_models": {"type": "array", "items": {"type": "object"},
                              "description": "The models of the engine, with their context and capabilities: "
                                             "context, trained_context, kv_bytes_per_token, kv_type (q8 or f16), "
                                             "and the speeds measured on their answers, prompt_tokens_per_second "
                                             "and generation_tokens_per_second, 0 until they answered."},
            "models_state": {"type": "object", "additionalProperties": {"type": "string"},
                             "description": "loading, ready, unloading, removing, or the error of a load."},
            "web": {"type": "object", "description": "The web access of the Web button, which the web application "
                    "reads at its start: available while the administrator leaves it on (web_access) and the "
                    "service is not offline, and the provider of /albireo/v1/web/search, wikipedia, or none when "
                    "the search is unavailable while /albireo/v1/web/fetch stays available.",
                    "properties": {
                        "available": {"type": "boolean"},
                        "search": {"type": "string", "enum": ["wikipedia", "none"]}}}}})}}},
    "/albireo/v1/models": {"get": {
        "tags": ["Service"], "summary": "Catalog, installed models, downloads and active models",
        "description": "The language models: those that come with Albireo first (builtin), then the catalog and "
                       "the imports. The embedding models serve the document search and are given apart.",
        "parameters": [{"name": "lang", "in": "query",
                        "schema": {"type": "string", "enum": ["auto", "en", "fr", "de", "it", "es"]},
                        "description": "Language of the descriptions; auto, or no value, gives English."}],
        "responses": {"200": reply("The models", {"type": "object", "properties": {
            "free": {"type": "integer", "description": "The free bytes of the disk of the models folder."},
            "offline": {"type": "boolean", "description": "True when the service is set offline: the models come "
                        "from imports."},
            "models": {"type": "array", "items": {"type": "object", "properties": {
                "id": {"type": "string"}, "name": {"type": "string"},
                "roles": {"type": "array", "items": {"type": "string"}},
                "inputs": {"type": "array", "items": {"type": "string", "enum": ["text", "image", "audio"]},
                           "description": "What the model reads; the attachments of the web application follow."},
                "installed": {"type": "boolean"},
                "builtin": {"type": "boolean", "description": "Comes with Albireo: it cannot be deleted."},
                "size": {"type": "integer"}, "quantization": {"type": "string"},
                "total_size": {"type": "integer", "description": "The bytes of the files of the model, its "
                               "projector included."},
                "projector": {"type": "object", "description": "The image encoder of a model that reads images, a "
                              "second file downloaded with the decoder: its size, SHA-256 and quantization, and "
                              "its file once installed.", "properties": {
                                  "file": {"type": "string"}, "size": {"type": "integer"},
                                  "sha256": {"type": "string"}, "quantization": {"type": "string"}}},
                "description": {"type": "string"}, "publisher": {"type": "string"},
                "license": {"type": "string", "description": "The license of the entry of the model in the catalog, "
                            "or in the manifest of the models that come with Albireo; absent for a file imported "
                            "without an entry."}}}},
            "background": {"type": "string", "description": "The model of albireo-background, empty without one."},
            "embedding_models": {"type": "array", "items": {"type": "object", "properties": {
                "id": {"type": "string"}, "name": {"type": "string"},
                "source": {"type": "string", "enum": ["builtin", "catalog", "import"]},
                "size": {"type": "integer"}, "publisher": {"type": "string"},
                "license": {"type": "string", "description": "As for the language models."}}},
                "description": "The installed embedding models."},
            "downloads": {"type": "array", "items": {"type": "object"}},
            "pending_projectors": {"type": "array", "description": "The projectors imported without their decoder, "
                                   "which wait for it.", "items": {"type": "object", "properties": {
                                       "name": {"type": "string"}, "size": {"type": "integer"},
                                       "sha256": {"type": "string"}, "projector_type": {"type": "string"},
                                       "projection_width": {"type": "integer"}}}},
            "active": {"type": "object", "additionalProperties": {"type": "string"},
                       "description": "The model of each function, as in the status."},
            "selected": {"type": "object", "additionalProperties": {"type": "string"},
                         "description": "The models chosen for the functions."},
            "loaded": {"type": "array", "items": {"type": "string"}, "description": "Models of the engine."},
            "leaving": {"type": "object", "additionalProperties": {"type": "string", "enum": ["unloading",
                                                                                            "removing"]},
                        "description": "Models waiting for their requests before they leave the engine."}}})}}},
    "/albireo/v1/usage": {"get": {"tags": ["Service"], "summary": "Usage per day over 30 days",
                                  "responses": {"200": reply("The figures", {"type": "object", "properties": {
                                      "days": {"type": "array", "description": "The days with a use, 30 at most, the "
                                               "most recent first.", "items": {"type": "object", "properties": {
                                                   "day": {"type": "string", "format": "date",
                                                           "example": "2026-11-01"},
                                                   "prompt_tokens": {"type": "integer"},
                                                   "generated_tokens": {"type": "integer"},
                                                   "requests": {"type": "integer"}}}}}})}}},
    "/albireo/v1/engines": {"get": {"tags": ["Service"], "summary": "Engines found, their devices and measures",
                                    "responses": {"200": reply("The engines", ref("Engines"))}}},
    "/albireo/v1/updates": {"get": {"tags": ["Service"], "summary": "State of the updates",
                                    "responses": {"200": reply("The state", ref("UpdateState"))}}},
    "/albireo/v1/licenses": {"get": {
        "tags": ["Service"], "summary": "License texts of the installation",
        "description": "The texts of the licenses folder of the programs, which the installer fills: those of the "
                       "components of the programs (Albireo, dlib, Mbed TLS, cpp-httplib, zlib, SQLite) and of the "
                       "runtimes of the engines (OpenVINO, CUDA, cuDNN). The list is empty without that folder, as "
                       "in a build folder. The components of the web application are listed by "
                       "/vendor/components.json.",
        "responses": {"200": reply("The texts, by the names of their files", {"type": "array", "items": {
            "type": "object", "properties": {
                "file": {"type": "string", "example": "mbedtls.txt"},
                "name": {"type": "string", "example": "Mbed TLS"}, "size": {"type": "integer"},
                "version": {"type": "string", "description": "The version the service was built with, when it "
                            "knows it."}}}})}}},
    "/albireo/v1/licenses/{file}": {"get": {
        "tags": ["Service"], "summary": "A license text of the installation",
        "description": "Only the files the folder lists, read through no link: a link, a file with several names "
                       "(hard links) or a text of more than 1 MB is not served.",
        "parameters": [{"name": "file", "in": "path", "required": True, "schema": {"type": "string"},
                        "description": "A file listed by /albireo/v1/licenses."}],
        "responses": {"200": {"description": "The text", "content": {"text/plain": {"schema": {"type": "string"}}}},
                      "400": reply("A name with a separator, a colon, a control character or a leading dot (code "
                                   "invalid_license)", ref("Error")),
                      "404": reply("No such text (code license_not_found)", ref("Error"))}}},
    "/albireo/v1/count": {"post": {
        "tags": ["Service"], "summary": "Tokens of messages, or of a text, for a model",
        "requestBody": json_body({"type": "object", "properties": {
            "model": {"type": "string"}, "messages": {"type": "array", "items": ref("Message")},
            "tools": {"type": "array", "items": ref("Tool")}, "text": {"type": "string"}}}),
        "responses": {"200": reply("The count and the context of the model", {"type": "object", "properties": {
            "tokens": {"type": "integer"}, "context": {"type": "integer"}}}),
                      "400": reply("The engine refused the count, such as for a model that is not loaded: error "
                                   "gives the reason, as a text", {"type": "object", "properties": {
                                       "error": {"type": "string"}}})}}},
    "/albireo/v1/web/search": {"post": {
        "tags": ["Web access"], "summary": "Searches Wikipedia",
        "description": "The tool web_search of the Web button, through the REST interface of Wikipedia (GET "
                       "https://<language>.wikipedia.org/w/rest.php/v1/search/page, with a User-Agent that gives a "
                       "contact, as the policy of Wikimedia asks), unless the setting web_search is none. The log "
                       "names the provider and the number of results, never the query.",
        "requestBody": json_body(ref("WebSearchRequest")),
        "responses": {
            "200": reply("The results", ref("WebSearchResults")),
            "400": reply("An invalid query, count or language (codes invalid_query, invalid_count, "
                         "invalid_language)", ref("Error")),
            "403": reply("The service works offline (code offline), the administrator turned the web access off "
                         "(web_access_off), or the setting web_search is none (no_search_provider)", ref("Error")),
            "429": reply("Wikipedia limits the rate of the anonymous requests (code rate_limited); Retry-After "
                         "gives the delay it asks for", ref("Error")),
            "502": reply("Wikipedia answers with another status or another answer than its interface "
                         "(search_failed), or cannot be reached (unreachable, tls_failed, proxy_failed)",
                         ref("Error")),
            "503": reply("Three requests of the web access run already (code busy, Retry-After: 2), or the service "
                         "stops (cancelled)", ref("Error")),
            "504": reply("The provider did not answer in time (code timeout)", ref("Error"))}}},
    "/albireo/v1/web/fetch": {"post": {
        "tags": ["Web access"], "summary": "Reads a page of the web at a public address",
        "description": "The tool fetch_page of the Web button. The address is http or https, without user "
                       "information; its host is refused when it names this computer or a local network "
                       "(localhost, *.localhost, *.local, *.internal, *.home.arpa, a name of a single label) or "
                       "when one of its addresses is not public (loopback, private, link-local, shared, "
                       "documentation, multicast and reserved ranges, and the IPv4 addresses that IPv6 embeds), "
                       "the decimal, hexadecimal and shortened forms of IPv4 included. The connection goes to the "
                       "address checked, the name keeping the server name of TLS and the check of the certificate. "
                       "The service follows five redirections at most, each address checked again. A text "
                       "(text/<subtype>, XML, JSON) is read up to 2 MB and cut beyond (truncated), a PDF file up to "
                       "20 MB. A request takes 10 s to connect, 15 s for each wait for data and 30 s in all, and "
                       "keeps no cookie. Behind a proxy, the check applies to the addresses this computer resolves "
                       "and the proxy resolves the name again. The log names the host, the status and the size, "
                       "never the path or the query.",
        "requestBody": json_body(ref("WebFetchRequest")),
        "responses": {
            "200": reply("The page, whatever the status of the site", ref("WebPage")),
            "400": reply("An invalid address (code invalid_url): another scheme, user information, no host, an "
                         "invalid port; or an invalid language (invalid_language)", ref("Error")),
            "403": reply("An address that is local or private (code address_refused), the service offline "
                         "(offline), or the web access turned off (web_access_off)", ref("Error")),
            "413": reply("A PDF file above 20 MB (code too_large)", ref("Error")),
            "415": reply("A type other than text/<subtype>, application/xhtml+xml, application/xml, "
                         "application/<name>+xml, application/json, application/<name>+json and application/pdf "
                         "(code unsupported_type), named in the message", ref("Error")),
            "502": reply("The site cannot be reached (code unreachable), its TLS connection fails (tls_failed), the "
                         "proxy does not open the connection (proxy_failed), more than five redirections "
                         "(too_many_redirects), or a redirection to an address that is not http or https "
                         "(invalid_redirect)", ref("Error")),
            "503": reply("Three requests of the web access run already (code busy, Retry-After: 2), or the service "
                         "stops (cancelled)", ref("Error")),
            "504": reply("The site did not answer in time (code timeout)", ref("Error"))}}},
    "/albireo/v1/models/refresh": {"post": {"tags": ["Administration"], "security": admin,
                                            "summary": "Fetches the catalog again, in the background",
                                            "responses": dict({"202": reply("Started", {"type": "object"})},
                                                              **offline_reply)}},
    "/albireo/v1/models/{id}/download": {
        "post": {"tags": ["Administration"], "security": admin, "summary": "Starts or resumes the download of a model",
                 "parameters": [model_id],
                 "responses": dict({"202": reply("Started", ok), "400": reply("Refused", ok)}, **offline_reply)},
        "delete": {"tags": ["Administration"], "security": admin, "summary": "Cancels a download",
                   "parameters": [model_id], "responses": {"200": reply("Cancelled", ok)}}},
    "/albireo/v1/models/{id}/activate": {"post": {
        "tags": ["Administration"], "security": admin, "summary": "Activates a model for a function",
        "parameters": [model_id],
        "requestBody": json_body({"type": "object", "properties": {
            "role": {"type": "string", "enum": ["general", "code", "security", "embedding"], "default": "general",
                     "description": "embedding takes an embedding model only, the other functions a language "
                                    "model."}}}),
        "responses": {"200": reply("Activated", ok), "400": reply("Refused", ok)}}},
    "/albireo/v1/models/{id}/deactivate": {"post": {
        "tags": ["Administration"], "security": admin, "summary": "Deactivates a model for a function or for all",
        "description": "The functions taken from the model go back to the model that comes with Albireo; the "
                       "functions the model serves through its general function stay with it when the general "
                       "function alone is taken. The model leaves the engine once its requests in progress and "
                       "waiting are served, when no function uses it any more; an activation meanwhile keeps it "
                       "loaded. The model of the document search is replaced by the activation of another "
                       "embedding model (code embedding_needed). A function the model does not serve is refused "
                       "with the code role_not_served, and an unknown one with unknown_role.",
        "parameters": [model_id],
        "requestBody": json_body({"type": "object", "properties": {
            "role": {"type": "string", "enum": ["general", "code", "security", "embedding"],
                     "description": "The function taken from the model; without it, every function."}}},
            required=False),
        "responses": {"200": reply("Deactivated", ok), "400": reply("Refused", ok)}}},
    "/albireo/v1/models/{id}/verify": {"post": {
        "tags": ["Administration"], "security": admin, "summary": "Recomputes the hash of an installed model",
        "parameters": [model_id], "responses": {"200": reply("Result", {"type": "object"})}}},
    "/albireo/v1/models/{id}": {"delete": {
        "tags": ["Administration"], "security": admin, "summary": "Deletes an installed model",
        "description": "A model of the engine is deleted once its requests are served and it left the engine: "
                       "pending is then true, and the model is listed under leaving until then. A model that "
                       "comes with Albireo is refused with the code builtin_model.",
        "parameters": [model_id],
        "responses": {"200": reply("Deleted, or to be deleted", {"type": "object", "properties": {
            "ok": {"type": "boolean"}, "pending": {"type": "boolean"}}}), "400": reply("Refused", ok)}}},
    "/albireo/v1/import": {"post": {
        "tags": ["Administration"], "security": admin,
        "summary": "Imports a signed offline bundle, or the content of the import folder",
        "description": "For a computer without network: a memory card or a disk brought to the machine. The "
                       "service reads with its own rights, so that it accepts two kinds of paths: a signed offline "
                       "bundle on a local drive (a folder whose catalog.json or feed.json bears the signature of "
                       "Albireo, as tools/make_offline_bundle.py prepares it), of which it reads only the signed "
                       "files, the models the signed catalog lists, by their names, sizes and hashes, and the "
                       "installer the signed feed names; and a folder or a file of the import folder of the data "
                       "folder. Every file is read through no link. A model file of the user is sent through "
                       "/albireo/v1/models/upload. An older catalog never replaces a newer one, a model installed "
                       "already is listed under already, and the update is installed at the next start of the "
                       "service.",
        "requestBody": json_body({"type": "object", "required": ["path"], "properties": {"path": {"type": "string"}}}),
        "responses": {"200": reply("Imported models, catalog, update and errors", {"type": "object", "properties": {
            "imported": {"type": "array", "items": {"type": "string"}},
            "already": {"type": "array", "items": {"type": "string"},
                        "description": "The installed models whose content the files repeat."},
            "errors": {"type": "array", "items": {"type": "string"}},
            "catalog": {"type": "object"}, "update": ref("UpdateState")}}),
                      "400": reply("The path is refused before anything of it is read; the code is the same "
                                   "whether a folder outside the import folder is missing, unreadable, unsigned or "
                                   "reached through a link (import_not_bundle)", import_refusal)}}},
    "/albireo/v1/models/import": {"post": {
        "tags": ["Administration"], "security": admin, "summary": "Same as /albireo/v1/import",
        "requestBody": json_body({"type": "object", "required": ["path"], "properties": {"path": {"type": "string"}}}),
        "responses": {"200": reply("Imported files and errors", {"type": "object"}),
                      "400": reply("Refused", import_refusal)}}},
    "/albireo/v1/models/upload": {"post": {
        "tags": ["Administration"], "security": admin, "summary": "Installs a GGUF file sent in the body",
        "description": "The body is the file, of the length its Content-Length header gives: a body without length, "
                       "or sent in chunks (Transfer-Encoding), is refused (400). The size limit of the JSON bodies "
                       "does not apply.",
        "parameters": [{"name": "name", "in": "query", "required": True, "schema": {"type": "string"}}],
        "requestBody": {"required": True, "content": {
            "application/octet-stream": {"schema": {"type": "string", "format": "binary"}}}},
        "responses": {"200": reply("Imported", {"type": "object"}),
                      "400": reply("Refused: no name or no length, a transfer that did not complete, or a file that "
                                   "is not a model", {"type": "object"}),
                      "500": reply("The file cannot be created in the models folder", ref("Error")),
                      "507": reply("The disk of the models folder lacks the size of the file and 256 MB",
                                   ref("Error"))}}},
    "/albireo/v1/settings": {
        "get": {"tags": ["Administration"], "security": admin, "summary": "Settings of the service",
                "responses": {"200": reply("The settings, the password of the proxy given as ********",
                                           {"type": "object"})}},
        "put": {"tags": ["Administration"], "security": admin, "summary": "Changes settings",
                "description": "A new port is opened once before it is saved.",
                "requestBody": json_body(ref("Settings")),
                "responses": {"200": reply("Saved; restart tells whether the service restarts", {"type": "object"}),
                              "400": reply("A value out of range, or the masked password of the proxy with another "
                                           "user, host or port (code proxy_password)", ref("Error")),
                              "409": reply("The new port is in use or cannot be opened (code port_unavailable)",
                                           ref("Error"))}}},
    "/albireo/v1/updates/check": {"post": {"tags": ["Administration"], "security": admin,
                                           "summary": "Checks for an update and downloads it, in the background",
                                           "responses": dict({"202": reply("State", ref("UpdateState"))},
                                                             **offline_reply)}},
    "/albireo/v1/updates/apply": {"post": {"tags": ["Administration"], "security": admin,
                                           "summary": "Restarts the service to install a downloaded update",
                                           "responses": {"202": reply("Restarting", ok),
                                                         "400": reply("No update is ready (code update_not_ready)",
                                                                      ref("Refusal")),
                                                         "409": reply("The package manager of the system applies "
                                                                      "the updates (code update_package_manager)",
                                                                      ref("Refusal"))}}},
    "/albireo/v1/restart": {"post": {"tags": ["Administration"], "security": admin, "summary": "Restarts the service",
                                     "responses": {"202": reply("Restarting", ok)}}},
    "/albireo/v1/engines/benchmark": {"post": {"tags": ["Administration"], "security": admin,
                                               "summary": "Measures every usable engine on this machine",
                                               "description": "One measure runs at a time.",
                                               "responses": {
                                                   "200": reply("The measures", {"type": "object"}),
                                                   "409": reply("A measure is already running (code "
                                                                "benchmark_running)", {"type": "object", "properties": {
                                                                    "error": {"type": "string"},
                                                                    "code": {"type": "string",
                                                                             "enum": ["benchmark_running"]}}})}}},
    "/albireo/v1/logs": {"get": {"tags": ["Administration"], "security": admin, "summary": "Last lines of the log",
                                 "responses": {"200": reply("The lines", {"type": "object"})}}},
}

# Every administration route answers 401 without the administration token.
unauthorized = reply("Without the administration token, or with another value (code authentication_error)",
                     ref("Error"))
for operations in paths.values():
    for operation in operations.values():
        if operation.get("security") == admin:
            operation["responses"].setdefault("401", unauthorized)

here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "..", "common", "version.h"), encoding="utf-8") as f:
    version = re.search(r'#define ALBIREO_VERSION "([0-9.]+)"', f.read()).group(1)

doc = {
    "openapi": "3.0.3",
    "info": {
        "title": "Albireo Community Edition local interface",
        "version": version,
        "description": "The interface of the Albireo service, served on https://localhost (port 8443 unless changed in "
                       "the tray). The routes under /v1 follow the OpenAI interface, so that its client libraries "
                       "work with base_url https://localhost:8443/v1 and any API key. The functions albireo-general, "
                       "albireo-code and albireo-security name the models activated in the tray, "
                       "albireo-background the model that comes with Albireo, which the web application uses for its "
                       "work in the background, and albireo-embedding the embedding model of the search in "
                       "documents.\n\n"
                       "The administration routes require the token of the data folder of the service "
                       "(admin.token) in the X-Albireo-Admin header.\n\n"
                       "Requests carry JSON bodies of 64 MB and 524,288 values at most, with their length: a body "
                       "without length is refused with 411, a longer one with 413, another type with 415. GET, HEAD, "
                       "DELETE and OPTIONS take no body (400). While the JSON bodies being read at once hold the "
                       "values of their budget, a route that reads one answers 503 with the code busy and a "
                       "Retry-After of 2 seconds. An answer being written whose model has no memory left "
                       "for its keys and values ends with the error code out_of_memory.\n\n"
                       "Before any route, the server refuses, as plain text: a request whose Host header names "
                       "another server than this computer (421), a request whose Origin header names another "
                       "origin than the local server (403), so that no web site drives the service through a "
                       "browser, and a body of more than 8 MB while four such bodies are being read (503, with a "
                       "Retry-After header).",
        "license": {"name": "Boost Software License 1.0", "url": "https://www.boost.org/LICENSE_1_0.txt"},
        "contact": {"name": "Aldric Pierrain", "email": "cydraltechnology@gmail.com",
                    "url": "https://github.com/Cydral/Dlib-Transformer-extensions"}},
    "servers": [{"url": "/"}],
    "tags": [{"name": "OpenAI", "description": "Compatible with the OpenAI interface"},
             {"name": "Service", "description": "State of the service"},
             {"name": "Web access", "description": "The search and the pages of the web for the Web button"},
             {"name": "Administration", "description": "Routes of the tray, protected by the administration token"}],
    "paths": paths,
    "components": {"schemas": schemas, "securitySchemes": {
        "adminToken": {"type": "apiKey", "in": "header", "name": "X-Albireo-Admin"}}},
}

with open(os.path.join(here, "..", "web", "openapi.json"), "w", encoding="utf-8") as f:
    json.dump(doc, f, indent=1, ensure_ascii=False)
    f.write("\n")
