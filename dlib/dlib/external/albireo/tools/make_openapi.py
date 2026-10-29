# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
#
# Writes web/openapi.json, the description of the interface of the service shown by the
# documentation page (/docs/).  Run it after a change of the routes of host/server.cpp:
#
#   python3 tools/make_openapi.py

import json
import os


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
}
admin = [{"adminToken": []}]
model_id = {"name": "id", "in": "path", "required": True, "schema": {"type": "string", "pattern": "^[A-Za-z0-9._-]+$"},
            "description": "Identifier of the model, as listed by /albireo/v1/models."}
ok = {"type": "object", "properties": {"ok": {"type": "boolean"}, "error": {"type": "string"}}}

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
        "top_p": {"type": "number"}, "top_k": {"type": "integer"}, "min_p": {"type": "number"},
        "presence_penalty": {"type": "number"}, "frequency_penalty": {"type": "number"},
        "repeat_penalty": {"type": "number"}, "repeat_last_n": {"type": "integer"},
        "seed": {"type": "integer"},
        "stop": {"oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}, "maxItems": 16}]},
        "logit_bias": {"type": "object", "additionalProperties": {"type": "number"}}}},
    "ChatCompletionRequest": {"allOf": [ref("Sampling"), {
        "type": "object", "required": ["messages"],
        "properties": {
            "model": {"type": "string",
                      "description": "A model identifier, a function (albireo-general, albireo-code, albireo-security) "
                                     "or auto. A function without a model of its own is served by the general model.",
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
    "ModelList": {"type": "object", "properties": {"object": {"type": "string"}, "data": {"type": "array", "items": {
        "type": "object", "properties": {"id": {"type": "string"}, "object": {"type": "string"},
                                         "owned_by": {"type": "string"},
                                         "roles": {"type": "array", "items": {"type": "string"}},
                                         "alias_of": {"type": "string"}}}}}},
    "Settings": {"type": "object", "description": "Settings the tray may change; a change of port, engine or memory "
                 "restarts the service.", "properties": {
        "port": {"type": "integer", "minimum": 1024, "maximum": 65535},
        "language": {"type": "string", "enum": ["auto", "en", "fr", "de", "it", "es"]},
        "engine": {"type": "string", "enum": ["auto", "best", "cpu", "cpu-avx2", "openvino", "cuda"]},
        "accelerator": {"type": "string", "enum": ["auto", "none", "npu", "gpu", "both"]},
        "v2ram": {"type": "boolean", "description": "Extended device memory of the graphics engines."},
        "v2ram_max_mb": {"type": "integer", "description": "Limit of the extended memory, 0 for automatic."},
        "threads": {"type": "integer"}, "context": {"type": "integer"},
        "max_sequences": {"type": "integer", "minimum": 1, "maximum": 64}, "kv_mb": {"type": "integer"},
        "check_updates": {"type": "boolean"},
        "proxy": {"type": "string",
                  "description": "Proxy of the downloads: empty for the proxy of the system, none for a direct "
                                 "connection, or http://[user:password@]host:port."}}},
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
    "/albireo/v1/status": {"get": {
        "tags": ["Service"], "summary": "State of the service, of the engine and of the models",
        "responses": {"200": reply("The state", {"type": "object"})}}},
    "/albireo/v1/models": {"get": {
        "tags": ["Service"], "summary": "Catalog, installed models, downloads and active models",
        "parameters": [{"name": "lang", "in": "query",
                        "schema": {"type": "string", "enum": ["en", "fr", "de", "it", "es"]},
                        "description": "Language of the descriptions."}],
        "responses": {"200": reply("The models", {"type": "object"})}}},
    "/albireo/v1/usage": {"get": {"tags": ["Service"], "summary": "Usage per day over 30 days",
                                  "responses": {"200": reply("The figures", {"type": "object"})}}},
    "/albireo/v1/engines": {"get": {"tags": ["Service"], "summary": "Engines found, their devices and measures",
                                    "responses": {"200": reply("The engines", {"type": "object"})}}},
    "/albireo/v1/updates": {"get": {"tags": ["Service"], "summary": "State of the updates",
                                    "responses": {"200": reply("The state", {"type": "object"})}}},
    "/albireo/v1/count": {"post": {
        "tags": ["Service"], "summary": "Tokens of messages, or of a text, for a model",
        "requestBody": json_body({"type": "object", "properties": {
            "model": {"type": "string"}, "messages": {"type": "array", "items": ref("Message")},
            "tools": {"type": "array", "items": ref("Tool")}, "text": {"type": "string"}}}),
        "responses": {"200": reply("The count and the context of the model", {"type": "object", "properties": {
            "tokens": {"type": "integer"}, "context": {"type": "integer"}}})}}},
    "/albireo/v1/models/refresh": {"post": {"tags": ["Administration"], "security": admin,
                                            "summary": "Fetches the catalog again",
                                            "responses": {"200": reply("Result", ok)}}},
    "/albireo/v1/models/{id}/download": {
        "post": {"tags": ["Administration"], "security": admin, "summary": "Starts or resumes the download of a model",
                 "parameters": [model_id], "responses": {"202": reply("Started", ok), "400": reply("Refused", ok)}},
        "delete": {"tags": ["Administration"], "security": admin, "summary": "Cancels a download",
                   "parameters": [model_id], "responses": {"200": reply("Cancelled", ok)}}},
    "/albireo/v1/models/{id}/activate": {"post": {
        "tags": ["Administration"], "security": admin, "summary": "Activates a model for a function",
        "parameters": [model_id],
        "requestBody": json_body({"type": "object", "properties": {
            "role": {"type": "string", "enum": ["general", "code", "security"], "default": "general"}}}),
        "responses": {"200": reply("Activated", ok), "400": reply("Refused", ok)}}},
    "/albireo/v1/models/{id}/deactivate": {"post": {
        "tags": ["Administration"], "security": admin, "summary": "Deactivates a model for every function",
        "parameters": [model_id], "responses": {"200": reply("Deactivated", ok)}}},
    "/albireo/v1/models/{id}/verify": {"post": {
        "tags": ["Administration"], "security": admin, "summary": "Recomputes the hash of an installed model",
        "parameters": [model_id], "responses": {"200": reply("Result", {"type": "object"})}}},
    "/albireo/v1/models/{id}": {"delete": {
        "tags": ["Administration"], "security": admin, "summary": "Deletes an installed model",
        "parameters": [model_id], "responses": {"200": reply("Deleted", ok)}}},
    "/albireo/v1/models/import": {"post": {
        "tags": ["Administration"], "security": admin, "summary": "Imports the GGUF files of a local folder or file",
        "description": "For an installation without network: a memory card or a disk brought to the machine.",
        "requestBody": json_body({"type": "object", "required": ["path"], "properties": {"path": {"type": "string"}}}),
        "responses": {"200": reply("Imported files and errors", {"type": "object"})}}},
    "/albireo/v1/models/upload": {"post": {
        "tags": ["Administration"], "security": admin, "summary": "Installs a GGUF file sent in the body",
        "parameters": [{"name": "name", "in": "query", "required": True, "schema": {"type": "string"}}],
        "requestBody": {"required": True, "content": {
            "application/octet-stream": {"schema": {"type": "string", "format": "binary"}}}},
        "responses": {"200": reply("Imported", {"type": "object"}), "400": reply("Refused", {"type": "object"})}}},
    "/albireo/v1/settings": {
        "get": {"tags": ["Administration"], "security": admin, "summary": "Settings of the service",
                "responses": {"200": reply("The settings", {"type": "object"})}},
        "put": {"tags": ["Administration"], "security": admin, "summary": "Changes settings",
                "requestBody": json_body(ref("Settings")),
                "responses": {"200": reply("Saved; restart tells whether the service restarts", {"type": "object"})}}},
    "/albireo/v1/updates/check": {"post": {"tags": ["Administration"], "security": admin,
                                           "summary": "Checks for an update and downloads it",
                                           "responses": {"202": reply("State", {"type": "object"})}}},
    "/albireo/v1/updates/apply": {"post": {"tags": ["Administration"], "security": admin,
                                           "summary": "Restarts the service to install a downloaded update",
                                           "responses": {"202": reply("Restarting", ok)}}},
    "/albireo/v1/restart": {"post": {"tags": ["Administration"], "security": admin, "summary": "Restarts the service",
                                     "responses": {"202": reply("Restarting", ok)}}},
    "/albireo/v1/engines/benchmark": {"post": {"tags": ["Administration"], "security": admin,
                                               "summary": "Measures every usable engine on this machine",
                                               "responses": {"200": reply("The measures", {"type": "object"})}}},
    "/albireo/v1/logs": {"get": {"tags": ["Administration"], "security": admin, "summary": "Last lines of the log",
                                 "responses": {"200": reply("The lines", {"type": "object"})}}},
}

doc = {
    "openapi": "3.0.3",
    "info": {
        "title": "Albireo local interface",
        "version": "1.0.0",
        "description": "The interface of the Albireo service, served on https://localhost (port 8443 unless changed in "
                       "the tray). The routes under /v1 follow the OpenAI interface, so that its client libraries "
                       "work with base_url https://localhost:8443/v1 and any API key. The functions albireo-general, "
                       "albireo-code and albireo-security name the models activated in the tray.\n\n"
                       "The administration routes require the token of the data folder of the service "
                       "(admin.token) in the X-Albireo-Admin header.",
        "license": {"name": "Boost Software License 1.0", "url": "https://www.boost.org/LICENSE_1_0.txt"},
        "contact": {"name": "Cydral Technology", "url": "https://github.com/Cydral/Dlib-Transformer-extensions"}},
    "servers": [{"url": "/"}],
    "tags": [{"name": "OpenAI", "description": "Compatible with the OpenAI interface"},
             {"name": "Service", "description": "State of the service"},
             {"name": "Administration", "description": "Routes of the tray, protected by the administration token"}],
    "paths": paths,
    "components": {"schemas": schemas, "securitySchemes": {
        "adminToken": {"type": "apiKey", "in": "header", "name": "X-Albireo-Admin"}}},
}

here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "..", "web", "openapi.json"), "w", encoding="utf-8") as f:
    json.dump(doc, f, indent=1, ensure_ascii=False)
    f.write("\n")
