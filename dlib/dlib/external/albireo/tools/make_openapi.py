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
    "503": reply("The service starts (code starting) or the model loads (code model_loading); the Retry-After "
                 "header gives the seconds to wait", ref("Error")),
}
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
                "update_feed_invalid", "update_download", "update_not_ready", "update_invalid", "update_start",
                "update_package_manager", "stopping"]
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
            "content": {"description": "Text, or a list of parts of type text.",
                        "oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "object"}}],
                        "nullable": True},
            "reasoning_content": {"type": "string", "description": "Reasoning of an earlier assistant message."},
            "tool_calls": {"type": "array", "items": ref("ToolCall")},
            "tool_call_id": {"type": "string", "description": "For a tool message: the call it answers."},
            "name": {"type": "string"}}},
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
    "Sampling": {"type": "object", "properties": {
        "max_tokens": {"type": "integer",
                       "description": "Tokens to generate at most; by default up to the end of the context."},
        "max_completion_tokens": {"type": "integer"},
        "temperature": {"type": "number", "minimum": 0, "maximum": 2},
        "top_p": {"type": "number"},
        "top_k": {"type": "integer", "minimum": 0, "maximum": 1024, "default": 40,
                  "description": "Keeps the k most likely tokens, among which top_p and min_p then choose; 0 sets "
                                 "no limit, and the 1024 most likely tokens are kept. Without it, 40, or 20 for a "
                                 "model of the ChatML format that reasons. A value outside 0 to 1024 is refused "
                                 "(400). Without effect at temperature 0, which takes the most likely token."},
        "min_p": {"type": "number"},
        "presence_penalty": {"type": "number"}, "frequency_penalty": {"type": "number"},
        "repeat_penalty": {"type": "number"}, "repeat_last_n": {"type": "integer"},
        "seed": {"type": "integer"},
        "stop": {"oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}, "maxItems": 16}]},
        "logit_bias": {"type": "object", "additionalProperties": {"type": "number"},
                       "description": "Bias added to the logits of tokens, keyed by token identifiers of the "
                                      "vocabulary of the model; another key is refused."}}},
    "ChatCompletionRequest": {"allOf": [ref("Sampling"), {
        "type": "object", "required": ["messages"],
        "properties": {
            "model": {"type": "string",
                      "description": "A model identifier, a function (albireo-general, albireo-code, albireo-security), "
                                     "albireo-background or auto. A function without a model of its own is served by "
                                     "the general model; albireo-background, the work done in the background (titles, "
                                     "summaries, memory), by the model that comes with Albireo, else the general one.",
                      "example": "albireo-general"},
            "messages": {"type": "array", "items": ref("Message")},
            "stream": {"type": "boolean", "default": False},
            "stream_options": {"type": "object", "properties": {"include_usage": {"type": "boolean"}}},
            "tools": {"type": "array", "items": ref("Tool")},
            "tool_choice": {"description": "none disables the tools.",
                            "oneOf": [{"type": "string"}, {"type": "object"}]},
            "reasoning": {"type": "boolean", "description": "Enables or disables the reasoning of the models that "
                          "support both modes. The reasoning streams in reasoning_content."},
            "reasoning_effort": {"type": "string", "enum": ["none", "low", "medium", "high"]},
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
        "usage": ref("Usage"), "timings": ref("Timings"), "system_fingerprint": {"type": "string"}}},
    "ChatCompletionChunk": {"type": "object", "description": "One event of a stream (text/event-stream), as "
                            "'data: <chunk>' lines ended by 'data: [DONE]'.", "properties": {
        "id": {"type": "string"}, "object": {"type": "string", "enum": ["chat.completion.chunk"]},
        "choices": {"type": "array", "items": {"type": "object", "properties": {
            "index": {"type": "integer"},
            "delta": {"type": "object", "properties": {"role": {"type": "string"}, "content": {"type": "string"},
                                                       "reasoning_content": {"type": "string"},
                                                       "tool_calls": {"type": "array", "items": ref("ToolCall")}}},
            "finish_reason": {"type": "string", "nullable": True}}}},
        "usage": ref("Usage")}},
    "CompletionRequest": {"allOf": [ref("Sampling"), {"type": "object", "required": ["prompt"], "properties": {
        "model": {"type": "string", "example": "albireo-general"}, "prompt": {"type": "string"},
        "stream": {"type": "boolean"}}}]},
    "Completion": {"type": "object", "properties": {
        "id": {"type": "string"}, "object": {"type": "string", "enum": ["text_completion"]},
        "choices": {"type": "array", "items": {"type": "object", "properties": {
            "index": {"type": "integer"}, "text": {"type": "string"}, "finish_reason": {"type": "string"}}}},
        "usage": ref("Usage")}},
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
                                         "alias_of": {"type": "string"}}}}}},
    "Settings": {"type": "object", "description": "Settings the tray may change; a change of port, engine or memory "
                 "restarts the service.", "properties": {
        "port": {"type": "integer", "minimum": 1024, "maximum": 65535},
        "language": {"type": "string", "enum": ["auto", "en", "fr", "de", "it", "es"]},
        "engine": {"type": "string", "enum": ["auto", "best", "cpu", "cpu-avx2", "cpu-avxvnni", "cpu-avx512",
                                              "openvino", "cuda"]},
        "accelerator": {"type": "string", "enum": ["auto", "none", "npu", "gpu", "both"]},
        "v2ram": {"type": "boolean", "description": "Extended device memory of the graphics engines."},
        "v2ram_max_mb": {"type": "integer", "description": "Limit of the extended memory, 0 for automatic."},
        "device_mb": {"type": "integer", "description": "Memory of the device for the copies of the weights of "
                      "the models, in megabytes, 0 for automatic."},
        "threads": {"type": "integer"}, "context": {"type": "integer"},
        "max_sequences": {"type": "integer", "minimum": 1, "maximum": 64}, "kv_mb": {"type": "integer"},
        "check_updates": {"type": "boolean"},
        "offline": {"type": "boolean", "description": "No access to the network: the catalog, the models and the "
                    "updates come from imports, and the service opens no connection."},
        "proxy": {"type": "string",
                  "description": "Proxy of the downloads: empty for the proxy of the system, none for a direct "
                                 "connection, or http://[user:password@]host:port.  The settings give the password "
                                 "as ********; that value sent back keeps the stored password when the user, the "
                                 "host and the port stay the same, and is refused (code proxy_password) otherwise."}}},
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
                            "isa_avx512, no_nvidia_driver, old_nvidia_driver, no_nvidia_card, old_nvidia_card, "
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
                                "update_feed_invalid, update_download {e}, update_not_ready, update_invalid "
                                "{version}, update_start {code}, update_package_manager, stopping."},
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
                       "A stream also carries ': progress {...}' comment lines while a long prompt is read.",
        "requestBody": json_body(ref("ChatCompletionRequest")),
        "responses": dict({"200": {"description": "The completion, or a stream of chunks", "content": {
            "application/json": {"schema": ref("ChatCompletion")},
            "text/event-stream": {"schema": ref("ChatCompletionChunk")}}}}, **error_replies)}},
    "/v1/completions": {"post": {
        "tags": ["OpenAI"], "summary": "Text completion of a raw prompt",
        "requestBody": json_body(ref("CompletionRequest")),
        "responses": dict({"200": reply("The completion", ref("Completion"))}, **error_replies)}},
    "/v1/embeddings": {"post": {
        "tags": ["OpenAI"], "summary": "Vectors of texts, for search and comparison",
        "description": "Computed by the embedding model of the service, which the search in the documents of the "
                       "web application uses. The inputs of every client are computed together, and the passes of "
                       "the embedding model stay short while a conversation is being answered.",
        "requestBody": json_body(ref("EmbeddingRequest")),
        "responses": dict({"200": reply("One vector per input, in the order of the inputs", ref("EmbeddingList"))},
                          **error_replies)}},
    "/albireo/v1/status": {"get": {
        "tags": ["Service"], "summary": "State of the service, of the engine and of the models",
        "description": "The server answers while the service starts: state is \"starting\" and starting names the "
                       "step in progress, \"engines\" then \"models\", until the service is \"ready\".",
        "responses": {"200": reply("The state", {"type": "object", "properties": {
            "edition": {"type": "string", "description": "The edition, Community Edition."},
            "version": {"type": "string"},
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
                              "description": "The models of the engine, with their context and capabilities."},
            "models_state": {"type": "object", "additionalProperties": {"type": "string"},
                             "description": "loading, ready, unloading, removing, or the error of a load."}}})}}},
    "/albireo/v1/models": {"get": {
        "tags": ["Service"], "summary": "Catalog, installed models, downloads and active models",
        "description": "The language models: those that come with Albireo first (builtin), then the catalog and "
                       "the imports. The embedding models serve the document search and are given apart.",
        "parameters": [{"name": "lang", "in": "query",
                        "schema": {"type": "string", "enum": ["en", "fr", "de", "it", "es"]},
                        "description": "Language of the descriptions."}],
        "responses": {"200": reply("The models", {"type": "object", "properties": {
            "models": {"type": "array", "items": {"type": "object", "properties": {
                "id": {"type": "string"}, "name": {"type": "string"},
                "roles": {"type": "array", "items": {"type": "string"}},
                "inputs": {"type": "array", "items": {"type": "string", "enum": ["text", "image", "audio"]},
                           "description": "What the model reads; the attachments of the web application follow."},
                "installed": {"type": "boolean"},
                "builtin": {"type": "boolean", "description": "Comes with Albireo: it cannot be deleted."},
                "size": {"type": "integer"}, "quantization": {"type": "string"},
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
            "active": {"type": "object", "additionalProperties": {"type": "string"},
                       "description": "The model of each function, as in the status."},
            "selected": {"type": "object", "additionalProperties": {"type": "string"},
                         "description": "The models chosen for the functions."},
            "loaded": {"type": "array", "items": {"type": "string"}, "description": "Models of the engine."},
            "leaving": {"type": "object", "additionalProperties": {"type": "string", "enum": ["unloading",
                                                                                            "removing"]},
                        "description": "Models waiting for their requests before they leave the engine."}}})}}},
    "/albireo/v1/usage": {"get": {"tags": ["Service"], "summary": "Usage per day over 30 days",
                                  "responses": {"200": reply("The figures", {"type": "object"})}}},
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
            "tokens": {"type": "integer"}, "context": {"type": "integer"}}})}}},
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
        "tags": ["Administration"], "security": admin, "summary": "Deactivates a model for every function",
        "description": "The model leaves the engine once its requests in progress and waiting are served; an "
                       "activation meanwhile keeps it loaded. The model of the document search is replaced by "
                       "the activation of another embedding model.",
        "parameters": [model_id], "responses": {"200": reply("Deactivated", ok), "400": reply("Refused", ok)}}},
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
        "parameters": [{"name": "name", "in": "query", "required": True, "schema": {"type": "string"}}],
        "requestBody": {"required": True, "content": {
            "application/octet-stream": {"schema": {"type": "string", "format": "binary"}}}},
        "responses": {"200": reply("Imported", {"type": "object"}), "400": reply("Refused", {"type": "object"})}}},
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
                                               "responses": {"200": reply("The measures", {"type": "object"})}}},
    "/albireo/v1/logs": {"get": {"tags": ["Administration"], "security": admin, "summary": "Last lines of the log",
                                 "responses": {"200": reply("The lines", {"type": "object"})}}},
}

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
                       "DELETE and OPTIONS take no body (400). An answer being written whose model has no memory left "
                       "for its keys and values ends with the error code out_of_memory.",
        "license": {"name": "Boost Software License 1.0", "url": "https://www.boost.org/LICENSE_1_0.txt"},
        "contact": {"name": "Aldric Pierrain", "email": "cydraltechnology@gmail.com",
                    "url": "https://github.com/Cydral/Dlib-Transformer-extensions"}},
    "servers": [{"url": "/"}],
    "tags": [{"name": "OpenAI", "description": "Compatible with the OpenAI interface"},
             {"name": "Service", "description": "State of the service"},
             {"name": "Administration", "description": "Routes of the tray, protected by the administration token"}],
    "paths": paths,
    "components": {"schemas": schemas, "securitySchemes": {
        "adminToken": {"type": "apiKey", "in": "header", "name": "X-Albireo-Admin"}}},
}

with open(os.path.join(here, "..", "web", "openapi.json"), "w", encoding="utf-8") as f:
    json.dump(doc, f, indent=1, ensure_ascii=False)
    f.write("\n")
