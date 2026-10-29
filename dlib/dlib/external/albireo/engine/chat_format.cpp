// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "chat_format.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <random>

namespace albireo
{

    // Helpers

    size_t json_extent(const std::string& text, size_t start)
    {
        size_t i = start;
        while (i < text.size() && (text[i] == ' ' || text[i] == '\n' || text[i] == '\r' || text[i] == '\t')) ++i;
        if (i >= text.size()) return 0;
        const char first = text[i];
        if (first != '{' && first != '[')
        {
            // A scalar is complete when a delimiter follows it; the parse decides.
            size_t j = i;
            if (first == '"')
            {
                for (++j; j < text.size(); ++j)
                {
                    if (text[j] == '\\') { ++j; continue; }
                    if (text[j] == '"') return j + 1 - start;
                }
                return 0;
            }
            while (j < text.size() && std::strchr(",]}\n\r\t ", text[j]) == nullptr) ++j;
            return j < text.size() ? j - start : 0;
        }
        int depth = 0;
        bool in_string = false;
        for (size_t j = i; j < text.size(); ++j)
        {
            const char c = text[j];
            if (in_string)
            {
                if (c == '\\') ++j;
                else if (c == '"') in_string = false;
                continue;
            }
            if (c == '"') in_string = true;
            else if (c == '{' || c == '[') ++depth;
            else if (c == '}' || c == ']')
            {
                if (--depth == 0) return j + 1 - start;
            }
        }
        return 0;
    }

    void current_dates(render_options& options)
    {
        static const char* months[] = { "January", "February", "March", "April", "May", "June", "July", "August",
            "September", "October", "November", "December" };
        const std::time_t now = std::time(nullptr);
        char buf[64];
        std::tm t;
#if defined(_WIN32)
        localtime_s(&t, &now);
#else
        localtime_r(&now, &t);
#endif
        std::strftime(buf, sizeof(buf), "%Y-%m-%d", &t);
        options.today = buf;
        std::snprintf(buf, sizeof(buf), "%s %02d, %d", months[t.tm_mon], t.tm_mday, t.tm_year + 1900);
        options.today_long = buf;
        std::snprintf(buf, sizeof(buf), "%02d %.3s %d", t.tm_mday, months[t.tm_mon], t.tm_year + 1900);
        options.today_short = buf;
        const std::time_t before = now - 24 * 3600;
#if defined(_WIN32)
        localtime_s(&t, &before);
#else
        localtime_r(&before, &t);
#endif
        std::strftime(buf, sizeof(buf), "%Y-%m-%d", &t);
        options.yesterday = buf;
    }

    namespace
    {
        std::string new_call_id()
        {
            static std::atomic<unsigned long long> counter{0};
            static const unsigned long long seed = static_cast<unsigned long long>(
                std::chrono::steady_clock::now().time_since_epoch().count());
            std::mt19937_64 rng(seed + counter.fetch_add(1) * 0x9E3779B97F4A7C15ull);
            static const char* alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
            std::string id = "call_";
            for (int i = 0; i < 24; ++i) id += alphabet[rng() % 62];
            return id;
        }

        bool starts_with(const std::string& s, const std::string& p)
        {
            return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
        }

        bool ends_with(const std::string& s, const std::string& p)
        {
            return s.size() >= p.size() && s.compare(s.size() - p.size(), p.size(), p) == 0;
        }

        std::string trim_newlines(const std::string& s)
        {
            size_t a = 0, b = s.size();
            while (a < b && s[a] == '\n') ++a;
            while (b > a && s[b - 1] == '\n') --b;
            return s.substr(a, b - a);
        }

        std::string lstrip_newlines(const std::string& s)
        {
            size_t a = 0;
            while (a < s.size() && s[a] == '\n') ++a;
            return s.substr(a);
        }

        std::string strip_spaces(const std::string& s)
        {
            size_t a = 0, b = s.size();
            while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
            while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
            return s.substr(a, b - a);
        }

        void replace_all(std::string& s, const std::string& from, const std::string& to)
        {
            size_t pos = 0;
            while ((pos = s.find(from, pos)) != std::string::npos)
            {
                s.replace(pos, from.size(), to);
                pos += to.size();
            }
        }

        // The arguments of a call as JSON text: an object given as a string is kept as
        // written, an empty one becomes {}.
        std::string arguments_text(const json& args)
        {
            if (args.is_string()) return args.as_string().empty() ? std::string("{}") : args.as_string();
            if (args.is_null()) return "{}";
            return args.dump();
        }

        /*  The string literal a template assigns to a variable, as in
            {%- set default_system_message = '...' %}, with its escapes resolved; empty
            when the template has no such assignment.  */
        std::string template_string(const std::string& tmpl, const std::string& variable)
        {
            size_t pos = tmpl.find("set " + variable);
            if (pos == std::string::npos) return std::string();
            pos = tmpl.find('=', pos);
            if (pos == std::string::npos) return std::string();
            ++pos;
            while (pos < tmpl.size() && tmpl[pos] == ' ') ++pos;
            if (pos >= tmpl.size() || (tmpl[pos] != '\'' && tmpl[pos] != '"')) return std::string();
            const char quote = tmpl[pos++];
            std::string out;
            for (; pos < tmpl.size(); ++pos)
            {
                const char c = tmpl[pos];
                if (c == '\\' && pos + 1 < tmpl.size())
                {
                    const char e = tmpl[++pos];
                    out += e == 'n' ? '\n' : e == 't' ? '\t' : e;
                    continue;
                }
                if (c == quote) return out;
                out += c;
            }
            return std::string();
        }

        struct piece_writer
        {
            std::vector<prompt_piece>& out;
            void markup(const std::string& s)
            {
                if (s.empty()) return;
                if (!out.empty() && out.back().markup) out.back().text += s;
                else out.push_back(prompt_piece{ s, true });
            }
            void text(const std::string& s)
            {
                if (s.empty()) return;
                if (!out.empty() && !out.back().markup) out.back().text += s;
                else out.push_back(prompt_piece{ s, false });
            }
        };
    }

    // Output parsing

    namespace
    {
        enum class call_style
        {
            none,
            tagged_object,   // <tool_call>{"name":..,"arguments":..}</tool_call>
            mistral,         // [TOOL_CALLS]name[ARGS]{...} or [TOOL_CALLS][{...}]
            json_list,       // <|tool_call|>[{"name":..,"arguments":..}]
            whole_json       // the whole answer is {"name":..,"parameters":..}
        };

        struct parser_config
        {
            std::string think_open, think_close;
            bool starts_in_reasoning = false;
            call_style calls = call_style::none;
            std::string call_open, call_close;
            std::string call_open_alt;   // a second spelling of the opening marker
        };

        class tag_parser : public output_parser
        {
        public:
            explicit tag_parser(const parser_config& c) : c_(c)
            {
                state_ = c_.starts_in_reasoning ? state::reasoning : state::content;
                if (!c_.think_open.empty()) opens_.push_back(c_.think_open);
                if (!c_.call_open.empty()) opens_.push_back(c_.call_open);
                if (!c_.call_open_alt.empty()) opens_.push_back(c_.call_open_alt);
            }

            void feed(const std::string& text, std::vector<output_event>& events) override
            {
                buffer_ += text;
                process(events, false);
            }

            void finish(std::vector<output_event>& events) override
            {
                process(events, true);
                if (c_.calls == call_style::whole_json && !held_.empty())
                {
                    if (!emit_whole_json(held_, events)) emit(output_kind::content, held_, events);
                    held_.clear();
                }
                if (!buffer_.empty())
                {
                    // A call left unfinished, or an unterminated reasoning block.
                    emit(state_ == state::reasoning ? output_kind::reasoning : output_kind::content,
                        state_ == state::call ? call_prefix_ + buffer_ : buffer_, events);
                    buffer_.clear();
                }
            }

        private:
            enum class state { content, reasoning, call };

            void emit(output_kind k, const std::string& t, std::vector<output_event>& events)
            {
                if (t.empty()) return;
                if (k == output_kind::content && c_.calls == call_style::whole_json && !decided_whole_)
                {
                    // The answer is held until it is known not to be a call.
                    held_ += t;
                    const size_t first = held_.find_first_not_of(" \n\r\t");
                    if (first == std::string::npos) return;
                    if (held_[first] == '{' || starts_with(held_.substr(first), "<|python_tag|>")) return;
                    decided_whole_ = true;
                    const std::string all = held_;
                    held_.clear();
                    emit(k, all, events);
                    return;
                }
                if (!events.empty() && events.back().kind == k && k != output_kind::tool_call)
                {
                    events.back().text += t;
                    return;
                }
                output_event e;
                e.kind = k;
                e.text = t;
                events.push_back(e);
            }

            void emit_call(const std::string& name, const std::string& args, std::vector<output_event>& events)
            {
                output_event e;
                e.kind = output_kind::tool_call;
                e.call.id = new_call_id();
                e.call.name = name;
                e.call.arguments = args.empty() ? "{}" : args;
                events.push_back(e);
            }

            bool emit_call_object(const json& j, std::vector<output_event>& events)
            {
                const json& fn = j.contains("function") ? j["function"] : j;
                const std::string name = fn["name"].as_string();
                if (name.empty()) return false;
                const json& args = fn.contains("arguments") ? fn["arguments"] : fn["parameters"];
                emit_call(name, arguments_text(args), events);
                return true;
            }

            bool emit_whole_json(const std::string& text, std::vector<output_event>& events)
            {
                std::string t = strip_spaces(text);
                if (starts_with(t, "<|python_tag|>")) t = strip_spaces(t.substr(14));
                json j;
                if (!json::parse(t, j)) return false;
                if (j.is_array())
                {
                    for (const auto& e : j.elements())
                        if (!e.is_object() || !e["name"].is_string()) return false;
                    for (const auto& e : j.elements()) emit_call_object(e, events);
                    return true;
                }
                return j.is_object() && emit_call_object(j, events);
            }

            // Length of the longest suffix of buffer_ that begins one of the markers.
            size_t partial_suffix(const std::vector<std::string>& markers) const
            {
                size_t best = 0;
                for (const auto& m : markers)
                    for (size_t len = std::min(m.size() - 1, buffer_.size()); len > best; --len)
                        if (buffer_.compare(buffer_.size() - len, len, m, 0, len) == 0)
                        {
                            best = len;
                            break;
                        }
                return best;
            }

            void process(std::vector<output_event>& events, bool final)
            {
                for (;;)
                {
                    if (state_ == state::content)
                    {
                        // The blank lines that separate the reasoning from the answer.
                        if (skip_content_newlines_)
                        {
                            const size_t a = buffer_.find_first_not_of('\n');
                            if (a == std::string::npos)
                            {
                                if (final) buffer_.clear();
                                return;
                            }
                            buffer_.erase(0, a);
                            skip_content_newlines_ = false;
                        }
                        // Earliest opening marker.
                        size_t at = std::string::npos;
                        const std::string* which = nullptr;
                        for (const auto& m : opens_)
                        {
                            const size_t p = buffer_.find(m);
                            if (p != std::string::npos && p < at) { at = p; which = &m; }
                        }
                        if (which == nullptr)
                        {
                            const size_t keep = final ? 0 : partial_suffix(opens_);
                            emit(output_kind::content, buffer_.substr(0, buffer_.size() - keep), events);
                            buffer_.erase(0, buffer_.size() - keep);
                            return;
                        }
                        emit(output_kind::content, buffer_.substr(0, at), events);
                        buffer_.erase(0, at + which->size());
                        if (*which == c_.think_open)
                        {
                            state_ = state::reasoning;
                            strip_leading_ = true;
                        }
                        else
                        {
                            state_ = state::call;
                            call_prefix_ = *which;
                        }
                        continue;
                    }

                    if (state_ == state::reasoning)
                    {
                        if (strip_leading_)
                        {
                            const size_t a = buffer_.find_first_not_of('\n');
                            if (a == std::string::npos) { if (final) buffer_.clear(); return; }
                            buffer_.erase(0, a);
                            strip_leading_ = false;
                        }
                        const size_t p = buffer_.find(c_.think_close);
                        if (p == std::string::npos)
                        {
                            // Trailing newlines may precede the closing marker: held too.
                            size_t keep = final ? 0 : partial_suffix({ c_.think_close });
                            if (!final)
                                while (keep < buffer_.size() && buffer_[buffer_.size() - keep - 1] == '\n') ++keep;
                            emit(output_kind::reasoning, buffer_.substr(0, buffer_.size() - keep), events);
                            buffer_.erase(0, buffer_.size() - keep);
                            return;
                        }
                        std::string r = buffer_.substr(0, p);
                        while (!r.empty() && r.back() == '\n') r.pop_back();
                        emit(output_kind::reasoning, r, events);
                        buffer_.erase(0, p + c_.think_close.size());
                        state_ = state::content;
                        skip_content_newlines_ = true;
                        continue;
                    }

                    // state::call
                    if (!parse_call(events, final)) return;
                }
            }

            // Consumes one call from buffer_; false when more text is needed.
            bool parse_call(std::vector<output_event>& events, bool final)
            {
                if (c_.calls == call_style::tagged_object)
                {
                    const size_t p = buffer_.find(c_.call_close);
                    if (p == std::string::npos)
                    {
                        if (final) { emit(output_kind::content, call_prefix_ + buffer_, events); buffer_.clear(); }
                        return false;
                    }
                    const std::string body = buffer_.substr(0, p);
                    json j;
                    if (!json::parse(strip_spaces(body), j) || !j.is_object() || !emit_call_object(j, events))
                        emit(output_kind::content, call_prefix_ + body + c_.call_close, events);
                    buffer_.erase(0, p + c_.call_close.size());
                    state_ = state::content;
                    return true;
                }
                if (c_.calls == call_style::json_list || (c_.calls == call_style::mistral && starts_json_list()))
                {
                    const size_t n = json_extent(buffer_);
                    if (n == 0)
                    {
                        if (final) { emit(output_kind::content, call_prefix_ + buffer_, events); buffer_.clear(); }
                        return false;
                    }
                    json j;
                    if (json::parse(strip_spaces(buffer_.substr(0, n)), j) && (j.is_array() || j.is_object()))
                    {
                        if (j.is_object()) emit_call_object(j, events);
                        else for (const auto& e : j.elements()) if (e.is_object()) emit_call_object(e, events);
                    }
                    else
                    {
                        emit(output_kind::content, call_prefix_ + buffer_.substr(0, n), events);
                    }
                    buffer_.erase(0, n);
                    state_ = state::content;
                    return true;
                }
                // mistral: name[ARGS]{...}
                static const std::string args_marker = "[ARGS]";
                const size_t a = buffer_.find(args_marker);
                if (a == std::string::npos)
                {
                    if (final) { emit(output_kind::content, call_prefix_ + buffer_, events); buffer_.clear(); }
                    return false;
                }
                const size_t n = json_extent(buffer_, a + args_marker.size());
                if (n == 0)
                {
                    if (final) { emit(output_kind::content, call_prefix_ + buffer_, events); buffer_.clear(); }
                    return false;
                }
                const std::string name = strip_spaces(buffer_.substr(0, a));
                const std::string args = strip_spaces(buffer_.substr(a + args_marker.size(), n));
                json j;
                if (!name.empty() && json::parse(args, j)) emit_call(name, args, events);
                else emit(output_kind::content, call_prefix_ + buffer_.substr(0, a + args_marker.size() + n), events);
                buffer_.erase(0, a + args_marker.size() + n);
                state_ = state::content;
                return true;
            }

            bool starts_json_list() const
            {
                const size_t first = buffer_.find_first_not_of(" \n\r\t");
                return first != std::string::npos && buffer_[first] == '[';
            }

            parser_config c_;
            std::vector<std::string> opens_;
            state state_ = state::content;
            std::string buffer_;
            std::string call_prefix_;
            std::string held_;
            bool decided_whole_ = false;
            bool strip_leading_ = false;
            bool skip_content_newlines_ = false;
        };
    }

    // ChatML

    namespace
    {
        class chatml_format : public chat_format
        {
        public:
            chatml_format(const dlib::hf_tokenizer& tok, const std::string& tmpl)
            {
                const int end = tok.token_id("<|im_end|>");
                if (end >= 0) stop_tokens_.push_back(end);
                const int eot = tok.token_id("<|endoftext|>");
                if (eot >= 0) stop_tokens_.push_back(eot);
                tools_ = tmpl.find("<tool_call>") != std::string::npos || tmpl.find("tools") != std::string::npos;
                const bool think_tokens = tok.token_id("<think>") >= 0 && tok.token_id("</think>") >= 0;
                always_thinks_ = think_tokens && tmpl.find("<|im_start|>assistant\\n<think>\\n'") != std::string::npos;
                switchable_ = think_tokens && tmpl.find("enable_thinking") != std::string::npos;
                const std::string qwen = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
                const std::string smollm = "You are a helpful AI assistant named SmolLM, trained by Hugging Face";
                if (tmpl.find(qwen) != std::string::npos) default_system_ = qwen;
                else if (tmpl.find(smollm) != std::string::npos) default_system_ = smollm;
                separate_pieces_ = tok.type() == dlib::hf_tokenizer::kind::bpe;
                add_markers(tok, { "<think>", "</think>", "<tool_call>", "</tool_call>" });
            }

            const char* name() const override { return "chatml"; }
            bool supports_tools() const override { return tools_; }
            bool supports_reasoning() const override { return always_thinks_ || switchable_; }

            void render(const std::vector<chat_message>& messages, const render_options& opt,
                std::vector<prompt_piece>& out) const override
            {
                piece_writer w{ out };
                const bool has_system = !messages.empty() && messages[0].role == "system";
                const bool tools = tools_ && opt.tools.is_array() && !opt.tools.empty();
                if (tools)
                {
                    w.markup("<|im_start|>system\n");
                    if (has_system) { w.text(messages[0].content); w.markup("\n\n"); }
                    else if (!default_system_.empty()) { w.markup(default_system_); w.markup("\n\n"); }
                    w.markup("# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
                        "You are provided with function signatures within <tools></tools> XML tags:\n<tools>");
                    for (const auto& t : opt.tools.elements())
                    {
                        w.markup("\n");
                        w.text(t.dump_formatted());
                    }
                    w.markup("\n</tools>\n\nFor each function call, return a json object with function name and "
                        "arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, "
                        "\"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n");
                }
                else if (has_system)
                {
                    w.markup("<|im_start|>system\n");
                    w.text(messages[0].content);
                    w.markup("<|im_end|>\n");
                }
                else if (!default_system_.empty())
                {
                    w.markup("<|im_start|>system\n" + default_system_ + "<|im_end|>\n");
                }

                // The last real query: reasoning is kept for the turns after it only.
                long last_query = static_cast<long>(messages.size()) - 1;
                for (long i = static_cast<long>(messages.size()) - 1; i >= 0; --i)
                {
                    const chat_message& m = messages[static_cast<size_t>(i)];
                    const bool tool_response = starts_with(m.content, "<tool_response>")
                        && ends_with(m.content, "</tool_response>");
                    if (m.role == "user" && !tool_response)
                    {
                        last_query = i;
                        break;
                    }
                }

                for (size_t i = 0; i < messages.size(); ++i)
                {
                    const chat_message& m = messages[i];
                    if (m.role == "user" || (m.role == "system" && i != 0))
                    {
                        w.markup("<|im_start|>" + m.role + "\n");
                        w.text(m.content);
                        w.markup("<|im_end|>\n");
                    }
                    else if (m.role == "assistant")
                    {
                        std::string content = m.content;
                        std::string reasoning = m.reasoning;
                        if (reasoning.empty())
                        {
                            const size_t close = content.find("</think>");
                            if (close != std::string::npos)
                            {
                                std::string before = content.substr(0, close);
                                while (!before.empty() && before.back() == '\n') before.pop_back();
                                const size_t open = before.rfind("<think>");
                                if (open != std::string::npos) before = before.substr(open + 7);
                                reasoning = lstrip_newlines(before);
                                content = lstrip_newlines(content.substr(close + 8));
                            }
                        }
                        w.markup("<|im_start|>assistant\n");
                        const bool last = i + 1 == messages.size();
                        if (supports_reasoning() && static_cast<long>(i) > last_query &&
                            (last || !strip_spaces(reasoning).empty()))
                        {
                            w.markup("<think>\n");
                            w.text(trim_newlines(reasoning));
                            w.markup("\n</think>\n\n");
                            w.text(lstrip_newlines(content));
                        }
                        else
                        {
                            w.text(content);
                        }
                        for (size_t k = 0; k < m.tool_calls.size(); ++k)
                        {
                            if (k > 0 || !content.empty()) w.markup("\n");
                            w.markup("<tool_call>\n{\"name\": \"");
                            w.text(m.tool_calls[k].name);
                            w.markup("\", \"arguments\": ");
                            w.text(m.tool_calls[k].arguments.empty() ? std::string("{}") : m.tool_calls[k].arguments);
                            w.markup("}\n</tool_call>");
                        }
                        w.markup("<|im_end|>\n");
                    }
                    else if (m.role == "tool")
                    {
                        if (i == 0 || messages[i - 1].role != "tool") w.markup("<|im_start|>user");
                        w.markup("\n<tool_response>\n");
                        w.text(m.content);
                        w.markup("\n</tool_response>");
                        if (i + 1 == messages.size() || messages[i + 1].role != "tool") w.markup("<|im_end|>\n");
                    }
                }
                if (opt.add_generation_prompt)
                {
                    w.markup("<|im_start|>assistant\n");
                    if (always_thinks_) w.markup("<think>\n");
                    else if (switchable_ && !opt.reasoning) w.markup("<think>\n\n</think>\n\n");
                }
            }

            std::unique_ptr<output_parser> make_parser(const render_options& opt) const override
            {
                parser_config c;
                if (supports_reasoning())
                {
                    c.think_open = "<think>";
                    c.think_close = "</think>";
                    c.starts_in_reasoning = always_thinks_ && opt.add_generation_prompt;
                }
                if (tools_ && opt.tools.is_array() && !opt.tools.empty())
                {
                    c.calls = call_style::tagged_object;
                    c.call_open = "<tool_call>";
                    c.call_close = "</tool_call>";
                }
                return std::unique_ptr<output_parser>(new tag_parser(c));
            }

        private:
            bool tools_ = false;
            bool always_thinks_ = false;
            bool switchable_ = false;
            std::string default_system_;
        };

        // Mistral

        class mistral_format : public chat_format
        {
        public:
            mistral_format(const dlib::hf_tokenizer& tok, const std::string& tmpl)
            {
                bos_ = tok.bos_id() >= 0 ? tok.id_to_token(tok.bos_id()) : std::string();
                eos_ = tok.eos_id() >= 0 ? tok.id_to_token(tok.eos_id()) : std::string("</s>");
                reasoning_ = tok.token_id("[THINK]") >= 0 && tmpl.find("[THINK]") != std::string::npos;
                default_system_ = template_string(tmpl, "default_system_message");
                separate_pieces_ = tok.type() == dlib::hf_tokenizer::kind::bpe;
                add_markers(tok, { "[TOOL_CALLS]", "[ARGS]", "[CALL_ID]", "[THINK]", "[/THINK]" });
            }

            const char* name() const override { return "mistral"; }
            bool supports_tools() const override { return true; }
            bool supports_reasoning() const override { return reasoning_; }

            void render(const std::vector<chat_message>& messages, const render_options& opt,
                std::vector<prompt_piece>& out) const override
            {
                piece_writer w{ out };
                w.markup(bos_);
                size_t first = 0;
                if (!messages.empty() && messages[0].role == "system")
                {
                    w.markup("[SYSTEM_PROMPT]");
                    w.text(messages[0].content);
                    w.markup("[/SYSTEM_PROMPT]");
                    first = 1;
                }
                else if (!default_system_.empty())
                {
                    std::string s = default_system_;
                    replace_all(s, "{today}", opt.today);
                    replace_all(s, "{yesterday}", opt.yesterday);
                    w.markup("[SYSTEM_PROMPT]");
                    w.text(s);
                    w.markup("[/SYSTEM_PROMPT]");
                }
                if (opt.tools.is_array() && !opt.tools.empty())
                {
                    w.markup("[AVAILABLE_TOOLS]");
                    w.text(opt.tools.dump_formatted());
                    w.markup("[/AVAILABLE_TOOLS]");
                }
                for (size_t i = first; i < messages.size(); ++i)
                {
                    const chat_message& m = messages[i];
                    if (m.role == "user")
                    {
                        w.markup("[INST]");
                        w.text(m.content);
                        w.markup("[/INST]");
                    }
                    else if (m.role == "system")
                    {
                        w.markup("[SYSTEM_PROMPT]");
                        w.text(m.content);
                        w.markup("[/SYSTEM_PROMPT]");
                    }
                    else if (m.role == "assistant")
                    {
                        if (reasoning_ && !m.reasoning.empty())
                        {
                            w.markup("[THINK]");
                            w.text(m.reasoning);
                            w.markup("[/THINK]");
                        }
                        w.text(m.content);
                        for (const auto& call : m.tool_calls)
                        {
                            w.markup("[TOOL_CALLS]");
                            w.text(call.name);
                            w.markup("[ARGS]");
                            w.text(call.arguments.empty() ? std::string("{}") : call.arguments);
                        }
                        w.markup(eos_);
                    }
                    else if (m.role == "tool")
                    {
                        w.markup("[TOOL_RESULTS]");
                        w.text(m.content);
                        w.markup("[/TOOL_RESULTS]");
                    }
                }
            }

            std::unique_ptr<output_parser> make_parser(const render_options&) const override
            {
                parser_config c;
                if (reasoning_)
                {
                    c.think_open = "[THINK]";
                    c.think_close = "[/THINK]";
                }
                c.calls = call_style::mistral;
                c.call_open = "[TOOL_CALLS]";
                return std::unique_ptr<output_parser>(new tag_parser(c));
            }

        private:
            std::string bos_, eos_;
            bool reasoning_ = false;
            std::string default_system_;
        };

        // Llama 3

        class llama3_format : public chat_format
        {
        public:
            llama3_format(const dlib::hf_tokenizer& tok, const std::string& tmpl)
            {
                for (const char* t : { "<|eot_id|>", "<|eom_id|>", "<|end_of_text|>" })
                {
                    const int id = tok.token_id(t);
                    if (id >= 0) stop_tokens_.push_back(id);
                }
                dated_ = tmpl.find("Cutting Knowledge Date") != std::string::npos;
                add_markers(tok, { "<|python_tag|>" });
            }

            const char* name() const override { return "llama3"; }
            bool supports_tools() const override { return true; }
            bool supports_reasoning() const override { return false; }

            void render(const std::vector<chat_message>& messages, const render_options& opt,
                std::vector<prompt_piece>& out) const override
            {
                piece_writer w{ out };
                w.markup("<|begin_of_text|>");
                size_t first = 0;
                const bool tools = opt.tools.is_array() && !opt.tools.empty();
                std::string system;
                // The templates of the family trim the messages, the results of the tools aside.
                if (!messages.empty() && messages[0].role == "system")
                {
                    system = strip_spaces(messages[0].content);
                    first = 1;
                }
                if (dated_ || !system.empty())
                {
                    w.markup("<|start_header_id|>system<|end_header_id|>\n\n");
                    if (tools) w.markup("Environment: ipython\n");
                    if (dated_)
                        w.markup("Cutting Knowledge Date: December 2023\nToday Date: " + opt.today_short + "\n\n");
                    w.text(system);
                    w.markup("<|eot_id|>");
                }
                bool tools_given = !tools;
                for (size_t i = first; i < messages.size(); ++i)
                {
                    const chat_message& m = messages[i];
                    if (m.role == "assistant")
                    {
                        w.markup("<|start_header_id|>assistant<|end_header_id|>\n\n");
                        if (!m.tool_calls.empty())
                        {
                            const tool_call_data& c = m.tool_calls.front();
                            w.markup("{\"name\": \"");
                            w.text(c.name);
                            w.markup("\", \"parameters\": ");
                            w.text(c.arguments.empty() ? std::string("{}") : c.arguments);
                            w.markup("}");
                        }
                        else
                        {
                            w.text(strip_spaces(m.content));
                        }
                        w.markup("<|eot_id|>");
                        continue;
                    }
                    const std::string role = m.role == "tool" ? "ipython" : m.role;
                    w.markup("<|start_header_id|>" + role + "<|end_header_id|>\n\n");
                    if (m.role == "user" && !tools_given)
                    {
                        // Llama 3.1 puts the definitions in the first user message.
                        w.markup("Given the following functions, please respond with a JSON for a function call with "
                            "its proper arguments that best answers the given prompt.\n\nRespond in the format "
                            "{\"name\": function name, \"parameters\": dictionary of argument name and its value}."
                            "Do not use variables.\n\n");
                        for (const auto& t : opt.tools.elements())
                        {
                            w.text(t.dump_formatted(4));
                            w.markup("\n\n");
                        }
                        tools_given = true;
                    }
                    w.text(m.role == "tool" ? m.content : strip_spaces(m.content));
                    w.markup("<|eot_id|>");
                }
                if (opt.add_generation_prompt) w.markup("<|start_header_id|>assistant<|end_header_id|>\n\n");
            }

            std::unique_ptr<output_parser> make_parser(const render_options& opt) const override
            {
                parser_config c;
                if (opt.tools.is_array() && !opt.tools.empty()) c.calls = call_style::whole_json;
                return std::unique_ptr<output_parser>(new tag_parser(c));
            }

        private:
            bool dated_ = false;
        };

        // Granite

        class granite_format : public chat_format
        {
        public:
            granite_format(const dlib::hf_tokenizer& tok, const std::string&)
            {
                const int end = tok.token_id("<|end_of_text|>");
                if (end >= 0) stop_tokens_.push_back(end);
                add_markers(tok, { "<|tool_call|>" });
            }

            const char* name() const override { return "granite"; }
            bool supports_tools() const override { return true; }
            bool supports_reasoning() const override { return false; }

            void render(const std::vector<chat_message>& messages, const render_options& opt,
                std::vector<prompt_piece>& out) const override
            {
                piece_writer w{ out };
                const bool tools = opt.tools.is_array() && !opt.tools.empty();
                size_t first = 0;
                w.markup("<|start_of_role|>system<|end_of_role|>");
                if (!messages.empty() && messages[0].role == "system")
                {
                    w.text(messages[0].content);
                    first = 1;
                }
                else
                {
                    // The default system message of the current reference template.
                    w.markup("Knowledge Cutoff Date: April 2024.\nToday's Date: " + opt.today_long
                        + ".\nYou are Granite, developed by IBM." + (tools
                        ? " You are a helpful AI assistant with access to the following tools. When a tool is "
                          "required to answer the user's query, respond with <|tool_call|> followed by a JSON list "
                          "of tools used. If a tool does not exist in the provided list of tools, notify the user "
                          "that you do not have the ability to fulfill the request."
                        : " You are a helpful AI assistant."));
                }
                w.markup("<|end_of_text|>\n");
                if (tools)
                {
                    w.markup("<|start_of_role|>tools<|end_of_role|>");
                    w.text(opt.tools.dump_formatted(4));
                    w.markup("<|end_of_text|>\n");
                }
                for (size_t i = first; i < messages.size(); ++i)
                {
                    const chat_message& m = messages[i];
                    w.markup("<|start_of_role|>" + m.role + "<|end_of_role|>");
                    w.text(m.content);
                    // The reference template drops the calls of an assistant message; they are
                    // written back as the model wrote them.
                    if (m.role == "assistant" && !m.tool_calls.empty())
                    {
                        json list = json::array();
                        for (const auto& c : m.tool_calls)
                        {
                            json e = json::object();
                            e["name"] = c.name;
                            e["arguments"] = json::parse_or_null(c.arguments.empty() ? std::string("{}") : c.arguments);
                            list.push_back(e);
                        }
                        w.markup("<|tool_call|>");
                        w.text(list.dump_formatted());
                    }
                    w.markup("<|end_of_text|>\n");
                }
                if (opt.add_generation_prompt) w.markup("<|start_of_role|>assistant<|end_of_role|>");
            }

            std::unique_ptr<output_parser> make_parser(const render_options& opt) const override
            {
                parser_config c;
                if (opt.tools.is_array() && !opt.tools.empty())
                {
                    c.calls = call_style::json_list;
                    c.call_open = "<|tool_call|>";
                    // The smaller models of the family write the marker as plain text.
                    c.call_open_alt = "<tool_call>";
                }
                return std::unique_ptr<output_parser>(new tag_parser(c));
            }

        };

        // Plain

        class plain_format : public chat_format
        {
        public:
            explicit plain_format(const dlib::hf_tokenizer& tok)
            {
                separate_pieces_ = tok.type() == dlib::hf_tokenizer::kind::bpe;
            }

            const char* name() const override { return "plain"; }
            bool supports_tools() const override { return false; }
            bool supports_reasoning() const override { return false; }

            void render(const std::vector<chat_message>& messages, const render_options& opt,
                std::vector<prompt_piece>& out) const override
            {
                piece_writer w{ out };
                for (const auto& m : messages)
                {
                    std::string role = m.role;
                    if (!role.empty()) role[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(role[0])));
                    w.markup(role + ": ");
                    w.text(m.content);
                    w.markup("\n\n");
                }
                if (opt.add_generation_prompt) w.markup("Assistant: ");
            }

            std::unique_ptr<output_parser> make_parser(const render_options&) const override
            {
                return std::unique_ptr<output_parser>(new tag_parser(parser_config()));
            }
        };
    }

    void chat_format::add_markers(const dlib::hf_tokenizer& tok, const std::vector<std::string>& markers)
    {
        for (const auto& m : markers)
        {
            const int id = tok.token_id(m);
            if (id >= 0) marker_tokens_.push_back(id);
        }
        std::sort(marker_tokens_.begin(), marker_tokens_.end());
        marker_tokens_.erase(std::unique(marker_tokens_.begin(), marker_tokens_.end()), marker_tokens_.end());
    }

    std::unique_ptr<chat_format> chat_format::detect(const dlib::hf_tokenizer& tok, const std::string& tmpl)
    {
        auto has = [&](const char* s) { return tmpl.find(s) != std::string::npos; };
        if (has("[INST]") && (tok.token_id("[INST]") >= 0))
            return std::unique_ptr<chat_format>(new mistral_format(tok, tmpl));
        if (has("<|start_header_id|>") && tok.token_id("<|start_header_id|>") >= 0)
            return std::unique_ptr<chat_format>(new llama3_format(tok, tmpl));
        if (has("<|start_of_role|>") && tok.token_id("<|start_of_role|>") >= 0)
            return std::unique_ptr<chat_format>(new granite_format(tok, tmpl));
        if (tok.token_id("<|im_start|>") >= 0 && tok.token_id("<|im_end|>") >= 0)
            return std::unique_ptr<chat_format>(new chatml_format(tok, tmpl));
        // A file published without its template: the control tokens of the vocabulary name
        // the family of the model.
        if (tmpl.empty())
        {
            if (tok.token_id("<|start_header_id|>") >= 0 && tok.token_id("<|eot_id|>") >= 0)
                return std::unique_ptr<chat_format>(new llama3_format(tok, tmpl));
            if (tok.token_id("<|start_of_role|>") >= 0 && tok.token_id("<|end_of_text|>") >= 0)
                return std::unique_ptr<chat_format>(new granite_format(tok, tmpl));
            if (tok.token_id("[INST]") >= 0 && tok.token_id("[/INST]") >= 0)
                return std::unique_ptr<chat_format>(new mistral_format(tok, tmpl));
        }
        return std::unique_ptr<chat_format>(new plain_format(tok));
    }

    std::vector<int> chat_format::tokenize(const dlib::hf_tokenizer& tok, const std::vector<prompt_piece>& pieces,
        bool separate_pieces)
    {
        std::vector<int> ids;
        if (!separate_pieces)
        {
            // SentencePiece: the whole prompt at once, its markers recognized.
            std::string all;
            for (const auto& p : pieces) all += p.text;
            return tok.encode(all, false, false, true);
        }
        // The markup is searched for control tokens; the text between two of them, markup
        // and messages together, is encoded in one go so that the pre-tokenizer sees it as
        // the reference tokenizer would see the whole prompt.
        std::string pending;
        auto flush = [&]() {
            if (pending.empty()) return;
            const std::vector<int> part = tok.encode(pending, false, false, false);
            ids.insert(ids.end(), part.begin(), part.end());
            pending.clear();
        };
        for (const auto& p : pieces)
        {
            if (!p.markup)
            {
                pending += p.text;
                continue;
            }
            const std::vector<int> marked = tok.encode(p.text, false, false, true);
            // Split the markup on its control tokens, by encoding it with them recognized
            // and spelling the remaining stretches back.
            std::string stretch;
            for (int id : marked)
            {
                const int type = tok.token_type(id);
                if (type == dlib::hf_tokenizer::TT_CONTROL || type == dlib::hf_tokenizer::TT_USER_DEFINED)
                {
                    pending += stretch;
                    stretch.clear();
                    flush();
                    ids.push_back(id);
                }
                else
                {
                    stretch += tok.token_bytes(id);
                }
            }
            pending += stretch;
        }
        flush();
        return ids;
    }

}
