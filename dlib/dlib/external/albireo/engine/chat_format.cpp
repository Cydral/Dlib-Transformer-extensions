// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "chat_format.h"

#include <algorithm>
#include <atomic>
#include <cctype>
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
        // written, an empty one becomes {}, and an object is written as the templates write
        // it (json.dumps), with a space after each separator.
        std::string arguments_text(const json& args)
        {
            if (args.is_string()) return args.as_string().empty() ? std::string("{}") : args.as_string();
            if (args.is_null()) return "{}";
            return args.dump_formatted();
        }

        /*
            Parses the JSON of a tool call as a model writes it.  The models write the code
            and the texts they pass to a tool with raw line breaks and tabs in its strings,
            which the grammar forbids.  When the text does not parse as it is, these characters
            are escaped and the result parsed again, and repaired receives it.
        */
        bool parse_call_json(const std::string& text, json& out, std::string& repaired)
        {
            repaired.clear();
            if (json::parse(text, out)) return true;
            static const char hex[] = "0123456789abcdef";
            std::string fixed;
            fixed.reserve(text.size() + 16);
            bool in_string = false, changed = false;
            for (size_t i = 0; i < text.size(); ++i)
            {
                const char c = text[i];
                if (in_string)
                {
                    if (c == '\\' && i + 1 < text.size())
                    {
                        fixed += c;
                        fixed += text[++i];
                        continue;
                    }
                    if (c == '"')
                    {
                        in_string = false;
                    }
                    else if (static_cast<unsigned char>(c) < 0x20)
                    {
                        changed = true;
                        if (c == '\n') fixed += "\\n";
                        else if (c == '\r') fixed += "\\r";
                        else if (c == '\t') fixed += "\\t";
                        else
                        {
                            fixed += "\\u00";
                            fixed += hex[(c >> 4) & 15];
                            fixed += hex[c & 15];
                        }
                        continue;
                    }
                }
                else if (c == '"')
                {
                    in_string = true;
                }
                fixed += c;
            }
            if (!changed || !json::parse(fixed, out)) return false;
            repaired.swap(fixed);
            return true;
        }

        /*
            The text of the value of member key in the JSON object written in text, or of
            the member of its "function" object, as written; empty when it has none.  A
            model writes the arguments of a call as it was trained to, which this keeps.
        */
        std::string member_text(const std::string& text, const std::string& key)
        {
            auto skip = [&](size_t i) {
                while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i]))) ++i;
                return i;
            };
            size_t i = skip(0);
            if (i >= text.size() || text[i] != '{') return std::string();
            i = skip(i + 1);
            while (i < text.size() && text[i] == '"')
            {
                const size_t name_length = json_extent(text, i);
                if (name_length == 0) return std::string();
                json name;
                if (!json::parse(text.substr(i, name_length), name)) return std::string();
                i = skip(i + name_length);
                if (i >= text.size() || text[i] != ':') return std::string();
                const size_t value_start = skip(i + 1);
                const size_t value_length = json_extent(text, value_start);
                if (value_length == 0) return std::string();
                const std::string value = text.substr(value_start, value_length);
                if (name.as_string() == key) return value;
                if (name.as_string() == "function")
                {
                    const std::string inner = member_text(value, key);
                    if (!inner.empty()) return inner;
                }
                i = skip(value_start + value_length);
                if (i < text.size() && text[i] == ',') i = skip(i + 1);
            }
            return std::string();
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

        /*
            The end of a reasoning that reached its budget (chat_format::reasoning_close()): the
            sentence the authors of Qwen3 give for it, then the closing marker of the format and
            the separation of the answer.
        */
        std::vector<prompt_piece> closing(const std::string& marker, const std::string& after)
        {
            std::vector<prompt_piece> out;
            piece_writer w{ out };
            w.text("\n\nConsidering the limited time by the user, I have to give the solution based on the thinking "
                "directly now.\n");
            w.markup(marker);
            w.markup(after);
            return out;
        }
    }

    // Output parsing

    namespace
    {
        enum class call_style
        {
            none,
            tagged_object,   // <tool_call>{"name":..,"arguments":..}</tool_call>
            mistral,         // [TOOL_CALLS]name[ARGS]{...}, name[CALL_ID]id[ARGS]{...} or [TOOL_CALLS][{...}]
            json_list,       // <|tool_call|>[{"name":..,"arguments":..}]
            whole_json,      // the whole answer is {"name":..,"parameters":..} naming a tool of the request
            xml_function     // <tool_call><function=name><parameter=key>value</parameter></function></tool_call>
        };

        struct parser_config
        {
            std::string think_open, think_close;
            bool starts_in_reasoning = false;
            call_style calls = call_style::none;
            std::string call_open, call_close;
            std::string call_open_alt;   // a second spelling of the opening marker
            json tools;                  // the definitions of the tools: the types of the XML parameters, the
                                         // names a whole JSON answer may call
        };

        class tag_parser : public output_parser
        {
        public:
            explicit tag_parser(const parser_config& c) : c_(c)
            {
                state_ = c_.starts_in_reasoning ? state::reasoning : state::content;
                // A reasoning the prompt opens loses its leading line breaks, and the opening
                // marker a model writes again.
                strip_leading_ = c_.starts_in_reasoning;
                reopen_ = c_.starts_in_reasoning;
                if (!c_.think_open.empty()) opens_.push_back(c_.think_open);
                if (!c_.call_open.empty()) opens_.push_back(c_.call_open);
                if (!c_.call_open_alt.empty()) opens_.push_back(c_.call_open_alt);
            }

            void feed(const std::string& text, std::vector<output_event>& events) override
            {
                buffer_ += text;
                process(events, false);
            }

            bool in_reasoning() const override { return state_ == state::reasoning; }

            void finish(std::vector<output_event>& events) override
            {
                process(events, true);
                if (c_.calls == call_style::whole_json && !held_.empty())
                {
                    // The held answer is a call, or the answer as the model wrote it.
                    std::string held;
                    held.swap(held_);
                    decided_whole_ = true;
                    if (!emit_whole_json(held, events)) emit(output_kind::content, held, events);
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
                    // The answer is held until it is known not to be a call, the beginning of the
                    // python tag included.
                    static const std::string tag = "<|python_tag|>";
                    held_ += t;
                    const size_t first = held_.find_first_not_of(" \n\r\t");
                    if (first == std::string::npos) return;
                    const size_t length = std::min(held_.size() - first, tag.size());
                    if (held_[first] == '{' || held_.compare(first, length, tag, 0, length) == 0) return;
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

            /*
                A call, with the identifier the model gave it when it is one of the identifiers
                the formats use (letters, digits, '_' and '-', 64 at most), else a new one.
            */
            void emit_call(const std::string& name, const std::string& args, std::vector<output_event>& events,
                const std::string& id = std::string())
            {
                bool usable = !id.empty() && id.size() <= 64;
                for (size_t i = 0; usable && i < id.size(); ++i)
                    usable = std::isalnum(static_cast<unsigned char>(id[i])) || id[i] == '_' || id[i] == '-';
                output_event e;
                e.kind = output_kind::tool_call;
                e.call.id = usable ? id : new_call_id();
                e.call.name = name;
                e.call.arguments = args.empty() ? "{}" : args;
                events.push_back(e);
            }

            // Whether the definitions of the tools of the request name a function name.
            bool declared_tool(const std::string& name) const
            {
                if (name.empty()) return false;
                for (const auto& t : c_.tools.elements())
                {
                    const json& fn = t.contains("function") ? t["function"] : t;
                    if (fn["name"].as_string() == name) return true;
                }
                return false;
            }

            /*
                A call given as an object {"name", "arguments"} or {"function": {...}}; text,
                when given, is the object as the model wrote it, whose arguments are then
                passed on as written when they are an object.
            */
            bool emit_call_object(const json& j, std::vector<output_event>& events, const std::string& text = "")
            {
                const json& fn = j.contains("function") ? j["function"] : j;
                const std::string name = fn["name"].as_string();
                if (name.empty()) return false;
                const char* key = fn.contains("arguments") ? "arguments" : "parameters";
                const json& args = fn[key];
                std::string written;
                if (args.is_object() && !text.empty())
                {
                    written = strip_spaces(member_text(text, key));
                    json check;
                    if (!json::parse(written, check) || !(check == args)) written.clear();
                }
                emit_call(name, written.empty() ? arguments_text(args) : written, events);
                return true;
            }

            /*
                The calls of the text between two call markers, a JSON object or several in a
                row, possibly in a fenced block.  The closing braces a model writes beyond its
                last object, after code with braces of its own, are left aside.  False, emitting
                nothing, when the text holds anything else.
            */
            bool emit_calls(const std::string& body, std::vector<output_event>& events)
            {
                std::string text = strip_spaces(body);
                if (starts_with(text, "```"))
                {
                    const size_t line = text.find('\n');
                    text = line == std::string::npos ? std::string() : text.substr(line + 1);
                    text = strip_spaces(text);
                    if (ends_with(text, "```")) text = strip_spaces(text.substr(0, text.size() - 3));
                }
                std::vector<output_event> calls;
                size_t pos = 0;
                while (pos < text.size())
                {
                    if (!calls.empty() && text.find_first_not_of("}] \n\r\t", pos) == std::string::npos) break;
                    const size_t n = json_extent(text, pos);
                    if (n == 0 || text[pos] != '{') return false;
                    std::string one = text.substr(pos, n);
                    json j;
                    std::string repaired;
                    if (!parse_call_json(one, j, repaired) || !j.is_object()) return false;
                    if (!repaired.empty()) one.swap(repaired);
                    if (!emit_call_object(j, calls, one)) return false;
                    pos += n;
                    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) ++pos;
                }
                if (calls.empty()) return false;
                events.insert(events.end(), calls.begin(), calls.end());
                return true;
            }

            // The type the definition of a tool gives to one of its parameters, or "".
            std::string parameter_type(const std::string& function, const std::string& key) const
            {
                for (const auto& t : c_.tools.elements())
                {
                    const json& fn = t.contains("function") ? t["function"] : t;
                    if (fn["name"].as_string() == function)
                        return fn["parameters"]["properties"][key]["type"].as_string();
                }
                return std::string();
            }

            /*
                The value of a parameter of the XML form: a string when the definition of the
                tool declares a string, else the JSON value the text spells, with True, False
                and None as Python writes them, else the text as a string.
            */
            json xml_value(const std::string& function, const std::string& key, const std::string& text) const
            {
                const std::string type = parameter_type(function, key);
                if (type == "string") return json(text);
                const std::string t = strip_spaces(text);
                if (t == "True") return json(true);
                if (t == "False") return json(false);
                if (t == "None") return json();
                json v;
                if (!t.empty() && json::parse(t, v)) return v;
                return json(text);
            }

            /*
                The calls of the text between two call markers in the XML form of Granite 4.2
                and Qwen3-Coder: <function=name> elements that hold <parameter=key> elements,
                whose value is the text between their markers, less the line break after the
                opening marker and the one before the closing marker.  A model may stop before
                the closing marker of its last function.  False, emitting nothing, when the text
                holds anything else.
            */
            bool emit_xml_calls(const std::string& body, std::vector<output_event>& events)
            {
                static const std::string fn_open = "<function=", fn_close = "</function>";
                static const std::string par_open = "<parameter=", par_close = "</parameter>";
                std::vector<output_event> calls;
                size_t pos = 0;
                while (pos < body.size())
                {
                    const size_t f = body.find(fn_open, pos);
                    if (f == std::string::npos || !strip_spaces(body.substr(pos, f - pos)).empty()) break;
                    const size_t name_end = body.find('>', f);
                    if (name_end == std::string::npos) return false;
                    const size_t name_at = f + fn_open.size();
                    const std::string name = strip_spaces(body.substr(name_at, name_end - name_at));
                    if (name.empty() || name.find_first_of("<\n") != std::string::npos) return false;
                    const size_t close = body.find(fn_close, name_end);
                    const size_t stop = close == std::string::npos ? body.size() : close;
                    json args = json::object();
                    size_t q = name_end + 1;
                    for (;;)
                    {
                        const size_t a = body.find(par_open, q);
                        if (a == std::string::npos || a >= stop)
                        {
                            if (!strip_spaces(body.substr(q, stop - q)).empty()) return false;
                            break;
                        }
                        if (!strip_spaces(body.substr(q, a - q)).empty()) return false;
                        const size_t key_end = body.find('>', a);
                        const size_t b = key_end == std::string::npos ? key_end : body.find(par_close, key_end);
                        if (b == std::string::npos || b > stop) return false;
                        const size_t key_at = a + par_open.size();
                        const std::string key = strip_spaces(body.substr(key_at, key_end - key_at));
                        std::string value = body.substr(key_end + 1, b - key_end - 1);
                        if (!value.empty() && value[0] == '\n') value.erase(0, 1);
                        if (!value.empty() && value.back() == '\n') value.pop_back();
                        args[key] = xml_value(name, key, value);
                        q = b + par_close.size();
                    }
                    emit_call(name, args.dump_formatted(), calls);
                    pos = close == std::string::npos ? body.size() : close + fn_close.size();
                }
                if (calls.empty() || !strip_spaces(body.substr(std::min(pos, body.size()))).empty()) return false;
                events.insert(events.end(), calls.begin(), calls.end());
                return true;
            }

            /*
                The calls of an answer that is a whole JSON value: an object, or a list of
                objects, each naming one of the tools of the request.  False, emitting nothing,
                for any other answer, which is then the text of the answer.
            */
            bool emit_whole_json(const std::string& text, std::vector<output_event>& events)
            {
                std::string t = strip_spaces(text);
                if (starts_with(t, "<|python_tag|>")) t = strip_spaces(t.substr(14));
                json j;
                if (!json::parse(t, j)) return false;
                std::vector<const json*> objects;
                if (j.is_array())
                    for (const auto& e : j.elements()) objects.push_back(&e);
                else
                    objects.push_back(&j);
                if (objects.empty()) return false;
                for (const json* e : objects)
                {
                    if (!e->is_object()) return false;
                    const json& fn = e->contains("function") ? (*e)["function"] : *e;
                    if (!declared_tool(fn["name"].as_string())) return false;
                }
                std::vector<output_event> calls;
                for (const json* e : objects)
                    if (!emit_call_object(*e, calls)) return false;
                events.insert(events.end(), calls.begin(), calls.end());
                return true;
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
                            reset_scans();
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
                            if (reopen_)
                            {
                                const std::string& open = c_.think_open;
                                const bool partial = buffer_.size() < open.size()
                                    && open.compare(0, buffer_.size(), buffer_) == 0;
                                if (!final && partial) return;
                                reopen_ = false;
                                if (starts_with(buffer_, open))
                                {
                                    buffer_.erase(0, open.size());
                                    continue;
                                }
                            }
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
                if (c_.calls == call_style::tagged_object || c_.calls == call_style::xml_function)
                {
                    const bool xml = c_.calls == call_style::xml_function;
                    const size_t p = find_marker(c_.call_close);
                    if (p == std::string::npos)
                    {
                        // A model may end its turn after the object, without the closing marker.
                        if (final)
                        {
                            if (!(xml ? emit_xml_calls(buffer_, events) : emit_calls(buffer_, events)))
                                emit(output_kind::content, call_prefix_ + buffer_, events);
                            buffer_.clear();
                            reset_scans();
                        }
                        return false;
                    }
                    const std::string body = buffer_.substr(0, p);
                    if (!(xml ? emit_xml_calls(body, events) : emit_calls(body, events)))
                        emit(output_kind::content, call_prefix_ + body + c_.call_close, events);
                    buffer_.erase(0, p + c_.call_close.size());
                    state_ = state::content;
                    reset_scans();
                    return true;
                }
                if (c_.calls == call_style::json_list || (c_.calls == call_style::mistral && starts_json_list()))
                {
                    const size_t n = value_extent(0);
                    if (n == 0)
                    {
                        if (final)
                        {
                            emit(output_kind::content, call_prefix_ + buffer_, events);
                            buffer_.clear();
                            reset_scans();
                        }
                        return false;
                    }
                    json j;
                    std::string repaired;
                    const std::string value = strip_spaces(buffer_.substr(0, n));
                    if (parse_call_json(value, j, repaired) && (j.is_array() || j.is_object()))
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
                    reset_scans();
                    return true;
                }
                // mistral: name[ARGS]{...}, or name[CALL_ID]id[ARGS]{...} with the identifier of the call
                static const std::string args_marker = "[ARGS]", id_marker = "[CALL_ID]";
                const size_t a = find_marker(args_marker);
                if (a == std::string::npos)
                {
                    if (final)
                    {
                        emit(output_kind::content, call_prefix_ + buffer_, events);
                        buffer_.clear();
                        reset_scans();
                    }
                    return false;
                }
                const size_t n = value_extent(a + args_marker.size());
                if (n == 0)
                {
                    if (final)
                    {
                        emit(output_kind::content, call_prefix_ + buffer_, events);
                        buffer_.clear();
                        reset_scans();
                    }
                    return false;
                }
                std::string name = strip_spaces(buffer_.substr(0, a)), id;
                const size_t marked = name.find(id_marker);
                if (marked != std::string::npos)
                {
                    id = strip_spaces(name.substr(marked + id_marker.size()));
                    name = strip_spaces(name.substr(0, marked));
                }
                const std::string args = strip_spaces(buffer_.substr(a + args_marker.size(), n));
                json j;
                std::string repaired;
                if (!name.empty() && parse_call_json(args, j, repaired) && j.is_object())
                    emit_call(name, repaired.empty() ? args : repaired, events, id);
                else
                    emit(output_kind::content, call_prefix_ + buffer_.substr(0, a + args_marker.size() + n), events);
                buffer_.erase(0, a + args_marker.size() + n);
                state_ = state::content;
                reset_scans();
                return true;
            }

            bool starts_json_list() const
            {
                const size_t first = buffer_.find_first_not_of(" \n\r\t");
                return first != std::string::npos && buffer_[first] == '[';
            }

            /*
                Position of a marker in buffer_, the search starting where the previous one
                ended without finding it, since a call grows by a token at a time.
            */
            size_t find_marker(const std::string& marker)
            {
                const size_t from = scanned_ >= marker.size() ? scanned_ - marker.size() + 1 : 0;
                const size_t p = buffer_.find(marker, from);
                if (p == std::string::npos) scanned_ = buffer_.size();
                return p;
            }

            /*
                Length of the complete JSON value at start in buffer_, as json_extent() gives
                it.  The scan of an object or a list resumes where the previous call left it.
            */
            size_t value_extent(size_t start)
            {
                if (extent_.start != start)
                {
                    extent_ = extent_state();
                    extent_.start = start;
                    extent_.pos = start;
                }
                if (!extent_.begun)
                {
                    size_t i = extent_.pos;
                    while (i < buffer_.size() && (buffer_[i] == ' ' || buffer_[i] == '\n' || buffer_[i] == '\r'
                        || buffer_[i] == '\t'))
                        ++i;
                    extent_.pos = i;
                    if (i >= buffer_.size()) return 0;
                    if (buffer_[i] != '{' && buffer_[i] != '[') return json_extent(buffer_, start);
                    extent_.begun = true;
                }
                for (size_t j = extent_.pos; j < buffer_.size(); ++j)
                {
                    const char c = buffer_[j];
                    if (extent_.escape)
                    {
                        extent_.escape = false;
                        continue;
                    }
                    if (extent_.in_string)
                    {
                        if (c == '\\') extent_.escape = true;
                        else if (c == '"') extent_.in_string = false;
                        continue;
                    }
                    if (c == '"') extent_.in_string = true;
                    else if (c == '{' || c == '[') ++extent_.depth;
                    else if ((c == '}' || c == ']') && --extent_.depth == 0) return j + 1 - start;
                }
                extent_.pos = buffer_.size();
                return 0;
            }

            // Forgets the scans of buffer_, whose text changed at its start.
            void reset_scans()
            {
                scanned_ = 0;
                extent_ = extent_state();
            }

            // Where the scan of a JSON value at start stopped: its depth, inside a string or after
            // a backslash of one.
            struct extent_state
            {
                size_t start = std::string::npos;
                size_t pos = 0;
                int depth = 0;
                bool begun = false;
                bool in_string = false;
                bool escape = false;
            };

            parser_config c_;
            std::vector<std::string> opens_;
            state state_ = state::content;
            std::string buffer_;
            std::string call_prefix_;
            std::string held_;
            bool decided_whole_ = false;
            bool strip_leading_ = false;
            bool reopen_ = false;
            bool skip_content_newlines_ = false;
            // The bytes of buffer_ searched for a marker of the call without finding it.
            size_t scanned_ = 0;
            extent_state extent_;
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
                switchable_ = think_tokens && tmpl.find("enable_thinking") != std::string::npos;
                // A template whose switch opens the reasoning in one of its branches reasons on demand.
                opens_ = think_tokens && tmpl.find("<|im_start|>assistant\\n<think>\\n'") != std::string::npos;
                always_thinks_ = opens_ && !switchable_;
                const std::string qwen = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
                const std::string smollm = "You are a helpful AI assistant named SmolLM, trained by Hugging Face";
                if (tmpl.find(qwen) != std::string::npos) default_system_ = qwen;
                else if (tmpl.find(smollm) != std::string::npos) default_system_ = smollm;
                // The templates that trim the text of the messages (EuroLLM).
                trims_ = tmpl.find("message['content'] | trim") != std::string::npos;
                // The templates that give a system message the turn of the user (CyberSecQwen-4B).
                system_role_ = tmpl.find("!= 'system' else 'user'") != std::string::npos ? "user" : "system";
                /*
                    The template of Qwen2.5 writes the form of a call between double braces in a
                    string of the template, which Jinja leaves as they are: the models of that
                    family read "{{" and "}}" there.
                */
                doubled_ = tmpl.find("{{\\\"name\\\": <function-name>") != std::string::npos;
                add_markers(tok, { "<think>", "</think>", "<tool_call>", "</tool_call>" });
            }

            const char* name() const override { return "chatml"; }
            bool supports_tools() const override { return tools_; }
            bool supports_reasoning() const override { return always_thinks_ || switchable_; }
            bool always_reasons() const override { return always_thinks_; }

            std::vector<prompt_piece> reasoning_close() const override
            {
                return supports_reasoning() ? closing("</think>", "\n\n") : std::vector<prompt_piece>();
            }

            void render(const std::vector<chat_message>& messages, const render_options& opt,
                std::vector<prompt_piece>& out) const override
            {
                piece_writer w{ out };
                const bool has_system = !messages.empty() && messages[0].role == "system";
                const bool tools = tools_ && opt.tools.is_array() && !opt.tools.empty();
                auto body = [this](const std::string& s) { return trims_ ? strip_spaces(s) : s; };
                if (tools)
                {
                    w.markup("<|im_start|>system\n");
                    if (has_system) { w.text(body(messages[0].content)); w.markup("\n\n"); }
                    else if (!default_system_.empty()) { w.markup(default_system_); w.markup("\n\n"); }
                    w.markup("# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
                        "You are provided with function signatures within <tools></tools> XML tags:\n<tools>");
                    for (const auto& t : opt.tools.elements())
                    {
                        w.markup("\n");
                        w.text(t.dump_formatted());
                    }
                    const std::string open = doubled_ ? "{{" : "{", close = doubled_ ? "}}" : "}";
                    w.markup("\n</tools>\n\nFor each function call, return a json object with function name and "
                        "arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n" + open + "\"name\": "
                        "<function-name>, \"arguments\": <args-json-object>" + close + "\n</tool_call><|im_end|>\n");
                }
                else if (has_system)
                {
                    w.markup("<|im_start|>" + system_role_ + "\n");
                    w.text(body(messages[0].content));
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
                        w.markup("<|im_start|>" + (m.role == "system" ? system_role_ : m.role) + "\n");
                        w.text(body(m.content));
                        w.markup("<|im_end|>\n");
                    }
                    else if (m.role == "assistant")
                    {
                        std::string content = body(m.content);
                        std::string reasoning = m.reasoning;
                        /*
                            The templates that reason split an answer at its </think> markers, the
                            reasoning ending at the first and the answer following the last; the
                            others keep the answer as written.
                        */
                        if (reasoning.empty() && supports_reasoning())
                        {
                            const size_t close = content.find("</think>");
                            if (close != std::string::npos)
                            {
                                std::string before = content.substr(0, close);
                                while (!before.empty() && before.back() == '\n') before.pop_back();
                                const size_t open = before.rfind("<think>");
                                if (open != std::string::npos) before = before.substr(open + 7);
                                reasoning = lstrip_newlines(before);
                                content = lstrip_newlines(content.substr(content.rfind("</think>") + 8));
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
                    /*
                        A template that always opens the reasoning (the thinking variants) opens
                        it whatever the request asks, the engine bounding its length; a
                        switchable template closes it at once when the request asks for none.
                    */
                    if (opens(opt)) w.markup("<think>\n");
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
                    c.starts_in_reasoning = opens(opt) && opt.add_generation_prompt;
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
            // Whether the prompt opens the reasoning: always, or when asked for a switchable template that does.
            bool opens(const render_options& opt) const
            {
                return always_thinks_ || (switchable_ && opens_ && opt.reasoning);
            }

            std::string system_role_;
            bool opens_ = false;
            bool always_thinks_ = false;
            bool switchable_ = false;
            bool trims_ = false;
            bool doubled_ = false;
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
                add_markers(tok, { "[TOOL_CALLS]", "[ARGS]", "[CALL_ID]", "[THINK]", "[/THINK]" });
            }

            const char* name() const override { return "mistral"; }
            bool supports_tools() const override { return true; }
            bool supports_reasoning() const override { return reasoning_; }

            std::vector<prompt_piece> reasoning_close() const override
            {
                return reasoning_ ? closing("[/THINK]", "") : std::vector<prompt_piece>();
            }

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
                if (opt.tools.is_array() && !opt.tools.empty())
                {
                    // A whole answer in JSON is a call only when it names a tool of the request.
                    c.calls = call_style::whole_json;
                    c.tools = opt.tools;
                }
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

        // Granite 4.2

        /*
            The ChatML of Granite 4.2: a system turn in every prompt, the tools in a <tools>
            element of it with calls in the XML form (<function=name> and <parameter=key>
            elements in <tool_call>), and a reasoning that the prompt opens with "<think>\n"
            or closes at once with "<think></think>".  The assistant turns before the last
            query keep their answer only.
        */
        class granite4_format : public chat_format
        {
        public:
            explicit granite4_format(const dlib::hf_tokenizer& tok)
            {
                for (const char* t : { "<|im_end|>", "<|end_of_text|>" })
                {
                    const int id = tok.token_id(t);
                    if (id >= 0) stop_tokens_.push_back(id);
                }
                add_markers(tok, { "<think>", "</think>", "<tool_call>", "</tool_call>" });
            }

            const char* name() const override { return "granite4"; }
            bool supports_tools() const override { return true; }
            bool supports_reasoning() const override { return true; }
            // The template writes a line break after </think>, before the answer.
            std::vector<prompt_piece> reasoning_close() const override { return closing("</think>", "\n"); }

            void render(const std::vector<chat_message>& messages, const render_options& opt,
                std::vector<prompt_piece>& out) const override
            {
                piece_writer w{ out };
                const bool tools = opt.tools.is_array() && !opt.tools.empty();
                const bool has_system = !messages.empty() && messages[0].role == "system";
                const size_t first = has_system ? 1 : 0;
                w.markup("<|im_start|>system\n");
                if (has_system) w.text(messages[0].content);
                if (tools)
                {
                    if (has_system && !messages[0].content.empty()) w.markup("\n\n");
                    w.markup("# Tools\n\nYou have access to the following functions:\n\n<tools>");
                    for (const auto& t : opt.tools.elements())
                    {
                        json fn = t.contains("function") ? t["function"] : t;
                        if (fn["defer_loading"].as_bool(false)) continue;
                        fn.erase("defer_loading");
                        fn.erase("strict");
                        w.markup("\n");
                        w.text(fn.dump_formatted());
                    }
                    w.markup("\n</tools>\n\nIf you choose to call a function ONLY reply in the following format "
                        "with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n"
                        "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\n"
                        "This is the value for the second "
                        "parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n"
                        "<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner "
                        "<function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
                        "- Required parameters MUST be specified\n- You may provide optional reasoning for your "
                        "function call in natural language BEFORE the function call, but NOT after\n- If there is no "
                        "function call available, answer the question like normal with your current knowledge and do "
                        "not tell the user about function calls\n</IMPORTANT>");
                }
                w.markup("<|im_end|>\n");

                size_t last_user = std::string::npos;
                for (size_t i = first; i < messages.size(); ++i)
                    if (messages[i].role == "user") last_user = i;

                for (size_t i = first; i < messages.size(); ++i)
                {
                    const chat_message& m = messages[i];
                    if (m.role == "assistant")
                    {
                        const bool earlier = last_user != std::string::npos && i < last_user;
                        const bool given = !strip_spaces(m.reasoning).empty();
                        const size_t open = m.content.find("<think>");
                        const size_t close = m.content.rfind("</think>");
                        w.markup("<|im_start|>assistant\n");
                        if (!m.tool_calls.empty())
                        {
                            if (earlier)
                            {
                                // The text after the reasoning, or before a reasoning left open.
                                std::string c = m.content;
                                if (!given && close != std::string::npos) c = c.substr(close + 8);
                                else if (!given && open != std::string::npos) c = c.substr(0, open);
                                w.markup("<think></think>");
                                w.text(strip_spaces(c));
                            }
                            else
                            {
                                write_answer(w, m, given, open, close);
                            }
                            w.markup("\n");
                            for (const auto& call : m.tool_calls)
                            {
                                w.markup("<tool_call>\n<function=");
                                w.text(call.name);
                                w.markup(">\n");
                                const json args = json::parse_or_null(call.arguments);
                                for (const auto& kv : args.items())
                                {
                                    w.markup("<parameter=");
                                    w.text(kv.first);
                                    w.markup(">\n");
                                    w.text(python_text(kv.second));
                                    w.markup("\n</parameter>\n");
                                }
                                w.markup("</function>\n</tool_call>\n");
                            }
                            w.markup("<|im_end|>\n");
                            continue;
                        }
                        const bool none = open == std::string::npos && close == std::string::npos;
                        const bool both = open != std::string::npos && close != std::string::npos;
                        if (earlier && (given || none || both))
                        {
                            // The earlier answers keep the text after their reasoning.
                            std::string c = close != std::string::npos ? m.content.substr(close + 8)
                                : given ? "\n" + m.content : m.content;
                            w.markup("<think></think>");
                            w.text(rstrip(c));
                        }
                        else
                        {
                            write_answer(w, m, given, open, close);
                        }
                        w.markup("<|im_end|>\n");
                    }
                    else if (m.role == "tool")
                    {
                        if (i > first && messages[i - 1].role != "tool") w.markup("<|im_start|>user\n");
                        w.markup("<tool_response>\n");
                        w.text(m.content);
                        w.markup("\n</tool_response>\n");
                        if (i + 1 == messages.size() || messages[i + 1].role != "tool") w.markup("<|im_end|>\n");
                    }
                    else
                    {
                        w.markup("<|im_start|>" + m.role + "\n");
                        w.text(m.content);
                        w.markup("<|im_end|>\n");
                    }
                }
                if (opt.add_generation_prompt)
                    w.markup(opt.reasoning ? "<|im_start|>assistant\n<think>\n"
                        : "<|im_start|>assistant\n<think></think>");
            }

            std::unique_ptr<output_parser> make_parser(const render_options& opt) const override
            {
                parser_config c;
                c.think_open = "<think>";
                c.think_close = "</think>";
                c.starts_in_reasoning = opt.add_generation_prompt && opt.reasoning;
                if (opt.tools.is_array() && !opt.tools.empty())
                {
                    c.calls = call_style::xml_function;
                    c.call_open = "<tool_call>";
                    c.call_close = "</tool_call>";
                    c.tools = opt.tools;
                }
                return std::unique_ptr<output_parser>(new tag_parser(c));
            }

        private:
            // A value as the string filter of the templates writes it: str() of Python for the
            // strings, numbers and constants, JSON for the objects and the arrays.
            static std::string python_text(const json& v)
            {
                if (v.is_string()) return v.as_string();
                if (v.is_bool()) return v.as_bool() ? "True" : "False";
                if (v.is_null()) return "None";
                return v.dump_formatted();
            }

            static std::string rstrip(std::string s)
            {
                while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
                return s;
            }

            /*
                The answer of the turn being worked on, trimmed as the template trims it: the
                reasoning given apart between its markers, an empty reasoning before an answer
                that has none, or the content as the model wrote it.
            */
            static void write_answer(piece_writer& w, const chat_message& m, bool given, size_t open, size_t close)
            {
                if (given)
                {
                    w.markup("<think>\n");
                    w.text(m.reasoning);
                    const std::string c = rstrip(m.content);
                    w.markup(c.empty() ? "\n</think>" : "\n</think>\n");
                    w.text(c);
                }
                else if (open == std::string::npos && close == std::string::npos)
                {
                    w.markup("<think></think>");
                    w.text(rstrip(m.content));
                }
                else
                {
                    w.text(strip_spaces(m.content));
                }
            }
        };

        /*
            The format of Foundation-Sec-8B-Reasoning, as its template lays it out once rendered
            with the line breaks after the tags and the spaces before them removed, as the
            libraries that apply chat templates render it: <|system|> and <|user|> followed by a
            line break and their text, <|assistant|> followed at once by its answer and the end of
            text token, each turn after the previous one without a line break between them.  The
            template gives its own system prompt when the conversation has none, and keeps of the
            earlier answers the text after their reasoning.  The other roles are left out, as the
            template leaves them.  The prompt opens the reasoning with <think> after <|assistant|>
            and a line break, as the template of the model does since its update of February 2026,
            everything before </think> being the reasoning.  The template of the GGUF file of the
            catalog, an earlier one, leaves that marker to the model, which then may answer
            without reasoning, an answer read as a reasoning never closed.  The template opens the
            reasoning of every answer: the model always reasons, the engine bounding the length of
            the reasoning of a request that asks for none.
        */
        class foundation_sec_format : public chat_format
        {
        public:
            foundation_sec_format(const dlib::hf_tokenizer& tok, const std::string& tmpl)
            {
                for (const char* t : { "<|end_of_text|>", "<|eot_id|>" })
                {
                    const int id = tok.token_id(t);
                    if (id >= 0) stop_tokens_.push_back(id);
                }
                bos_ = tok.bos_id() >= 0 ? tok.id_to_token(tok.bos_id()) : std::string();
                eos_ = tok.eos_id() >= 0 ? tok.id_to_token(tok.eos_id()) : std::string("<|end_of_text|>");
                // The system prompt of the template, in the branch without a system message.
                const size_t branch = tmpl.find("if not has_system");
                const size_t open = branch == std::string::npos ? branch : tmpl.find("<|system|>", branch);
                const size_t end = open == std::string::npos ? open : tmpl.find("{%", open);
                if (end != std::string::npos) default_system_ = strip_spaces(tmpl.substr(open + 10, end - open - 10));
                add_markers(tok, { "<think>", "</think>" });
            }

            const char* name() const override { return "foundation-sec"; }
            bool supports_tools() const override { return false; }
            bool supports_reasoning() const override { return true; }
            bool always_reasons() const override { return true; }
            std::vector<prompt_piece> reasoning_close() const override { return closing("</think>", "\n\n"); }

            void render(const std::vector<chat_message>& messages, const render_options& opt,
                std::vector<prompt_piece>& out) const override
            {
                piece_writer w{ out };
                w.markup(bos_);
                const bool has_system = !messages.empty() && messages[0].role == "system";
                if (!has_system && !default_system_.empty()) w.markup("<|system|>\n" + default_system_);
                for (size_t i = 0; i < messages.size(); ++i)
                {
                    const chat_message& m = messages[i];
                    if (i == 0 && has_system)
                    {
                        w.markup("<|system|>\n");
                        w.text(m.content);
                    }
                    else if (m.role == "user")
                    {
                        w.markup("<|user|>\n");
                        w.text(m.content);
                    }
                    else if (m.role == "assistant")
                    {
                        std::string content = strip_spaces(m.content);
                        const size_t close = content.rfind("</think>");
                        if (i + 1 < messages.size() && close != std::string::npos)
                            content = strip_spaces(content.substr(close + 8));
                        w.markup("<|assistant|>");
                        w.text(content);
                        w.markup(eos_);
                    }
                }
                if (opt.add_generation_prompt) w.markup("<|assistant|>\n<think>");
            }

            std::unique_ptr<output_parser> make_parser(const render_options& opt) const override
            {
                parser_config c;
                c.think_open = "<think>";
                c.think_close = "</think>";
                c.starts_in_reasoning = opt.add_generation_prompt;
                return std::unique_ptr<output_parser>(new tag_parser(c));
            }

        private:
            std::string bos_, eos_, default_system_;
        };

        /*
            The format of Foundation-Sec 1.1 8B Instruct, as the template of the repository of
            the model lays it out (its GGUF files come without one, published_template()): each
            message after <|system|>, <|user|> or <|assistant|> and a line break, followed by a
            line break, an answer ended by the end of text token, and <|assistant|> with a line
            break to let the model answer.  These markers are plain text of the Llama 3.1
            vocabulary of the model, so that a model that goes on past its answer writes one of
            them as text, which ends its turn (stop_strings()).  The other roles are left out, as
            the template leaves them.
        */
        class role_tags_format : public chat_format
        {
        public:
            explicit role_tags_format(const dlib::hf_tokenizer& tok)
            {
                for (const char* t : { "<|end_of_text|>", "<|eot_id|>" })
                {
                    const int id = tok.token_id(t);
                    if (id >= 0) stop_tokens_.push_back(id);
                }
                eos_ = tok.eos_id() >= 0 ? tok.id_to_token(tok.eos_id()) : std::string("<|end_of_text|>");
                stop_strings_ = { "<|user|>", "<|system|>", "<|assistant|>" };
            }

            const char* name() const override { return "role-tags"; }
            bool supports_tools() const override { return false; }
            bool supports_reasoning() const override { return false; }

            void render(const std::vector<chat_message>& messages, const render_options& opt,
                std::vector<prompt_piece>& out) const override
            {
                piece_writer w{ out };
                for (size_t i = 0; i < messages.size(); ++i)
                {
                    const chat_message& m = messages[i];
                    const bool last = i + 1 == messages.size();
                    if (m.role == "system" || m.role == "user")
                    {
                        w.markup("<|" + m.role + "|>\n");
                        w.text(m.content);
                        w.markup("\n");
                    }
                    else if (m.role == "assistant")
                    {
                        w.markup("<|assistant|>\n");
                        w.text(m.content);
                        w.markup(last ? eos_ : eos_ + "\n");
                    }
                }
                if (opt.add_generation_prompt) w.markup("<|assistant|>\n");
            }

            std::unique_ptr<output_parser> make_parser(const render_options&) const override
            {
                return std::unique_ptr<output_parser>(new tag_parser(parser_config()));
            }

        private:
            std::string eos_;
        };

        // Plain

        /*
            The format of Phi-4-mini: each message between <|role|> and <|end|>, with no line
            break, and <|assistant|> to let the model answer.  The tool messages are given as
            the user turns that carry their results; the tools themselves are not offered.
        */
        class phi_format : public chat_format
        {
        public:
            explicit phi_format(const dlib::hf_tokenizer& tok)
            {
                for (const char* t : { "<|end|>", "<|endoftext|>" })
                {
                    const int id = tok.token_id(t);
                    if (id >= 0) stop_tokens_.push_back(id);
                }
            }

            const char* name() const override { return "phi"; }
            bool supports_tools() const override { return false; }
            bool supports_reasoning() const override { return false; }

            void render(const std::vector<chat_message>& messages, const render_options& opt,
                std::vector<prompt_piece>& out) const override
            {
                piece_writer w{ out };
                for (const auto& m : messages)
                {
                    const std::string role = m.role == "system" || m.role == "assistant" ? m.role : std::string("user");
                    w.markup("<|" + role + "|>");
                    w.text(m.content);
                    w.markup("<|end|>");
                }
                if (opt.add_generation_prompt) w.markup("<|assistant|>");
            }

            std::unique_ptr<output_parser> make_parser(const render_options&) const override
            {
                return std::unique_ptr<output_parser>(new tag_parser(parser_config()));
            }
        };

        class plain_format : public chat_format
        {
        public:
            plain_format() {}

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
        const bool chatml = tok.token_id("<|im_start|>") >= 0 && tok.token_id("<|im_end|>") >= 0;
        if (chatml && has("<function=") && has("<think></think>"))
            return std::unique_ptr<chat_format>(new granite4_format(tok));
        if (chatml)
            return std::unique_ptr<chat_format>(new chatml_format(tok, tmpl));
        if (has("'<|' + message['role'] + '|>'") && has("<|end|>") && tok.token_id("<|end|>") >= 0
            && tok.token_id("<|assistant|>") >= 0)
            return std::unique_ptr<chat_format>(new phi_format(tok));
        if (has("<|system|>") && has("<|user|>") && has("<|assistant|>") && !has("<|end|>")
            && tok.token_id("<|system|>") >= 0 && tok.token_id("<|user|>") >= 0 && tok.token_id("<|assistant|>") >= 0)
            return std::unique_ptr<chat_format>(new foundation_sec_format(tok, tmpl));
        // The same markers as plain text of the vocabulary (Foundation-Sec 1.1 8B Instruct).
        if (has("'<|system|>\n' + message['content']") && has("'<|assistant|>\n'") && !has("<|end|>")
            && tok.token_id("<|system|>") < 0)
            return std::unique_ptr<chat_format>(new role_tags_format(tok));
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
        return std::unique_ptr<chat_format>(new plain_format());
    }

    std::vector<int> chat_format::tokenize(const dlib::hf_tokenizer& tok, const std::vector<prompt_piece>& pieces)
    {
        /*
            The markup is searched for control tokens; the text between two of them, markup
            and messages together, is encoded in one go so that the pre-tokenizer sees it as
            the reference tokenizer would see the whole prompt.  The text of the messages is
            never searched for control tokens.  A SentencePiece vocabulary that adds a leading
            space puts it before each of these stretches, as the reference tokenizers do when
            they split a prompt on its control tokens (the Prepend normalizer of EuroLLM).
        */
        std::vector<int> ids;
        std::string pending;
        auto flush = [&]() {
            if (pending.empty()) return;
            const std::vector<int> part = tok.encode(pending, false, false, false, true);
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
            // Without the leading space of SentencePiece, so that the stretches spell the
            // markup back as written.
            const std::vector<int> marked = tok.encode(p.text, false, false, true, false);
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

    std::string published_template(const std::string& general_name)
    {
        /*
            Foundation-Sec 1.1 8B Instruct: the chat_template.jinja of fdtn-ai/Foundation-Sec-1.1-8B-Instruct,
            whose GGUF files carry none.
        */
        if (general_name.find("Foundation Sec 1.1") != std::string::npos
            && general_name.find("Instruct") != std::string::npos)
            return "{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|system|>\n' + "
                "message['content'] + '\n' }}{% elif message['role'] == 'user' %}{{ '<|user|>\n' + "
                "message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{% if not loop.last %}{{ "
                "'<|assistant|>\n'  + message['content'] + eos_token + '\n' }}{% else %}{{ '<|assistant|>\n'  + "
                "message['content'] + eos_token }}{% endif %}{% endif %}{% if loop.last and add_generation_prompt %}"
                "{{ '<|assistant|>\n' }}{% endif %}{% endfor %}";
        return std::string();
    }

}
