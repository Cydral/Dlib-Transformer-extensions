// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
/*
    Checks the ChatML format of the engine on the conversations of an agent: the prompt of
    a conversation with tools, tool calls and tool results, as the chat template of Qwen3
    writes it, and the reading of the calls a model writes, whole or token by token.

        test_chat_format
*/

#include <cstdio>
#include <string>
#include <vector>

#include <dlib/tokenizer/hf_tokenizer.h>

#include "../engine/chat_format.h"

using namespace albireo;

namespace
{
    int failures = 0;

    void expect(bool ok, const std::string& what)
    {
        if (!ok)
        {
            ++failures;
            std::printf("FAILED: %s\n", what.c_str());
        }
    }

    // A byte-level vocabulary with the control tokens of ChatML and the markers of Qwen3.
    dlib::hf_tokenizer tokenizer(
        const std::vector<std::string>& control = { "<|endoftext|>", "<|im_start|>", "<|im_end|>" },
        const std::vector<std::string>& markers = { "<tool_call>", "</tool_call>", "<tool_response>",
            "</tool_response>", "<think>", "</think>" })
    {
        dlib::hf_tokenizer::vocab_data v;
        v.family = dlib::hf_tokenizer::kind::bpe;
        v.pretokenizer = "qwen2";
        int extra = 0;
        for (int b = 0; b < 256; ++b)
        {
            const bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
            const int cp = printable ? b : 256 + extra++;
            std::string s;
            if (cp < 0x80) s += static_cast<char>(cp);
            else
            {
                s += static_cast<char>(0xC0 | (cp >> 6));
                s += static_cast<char>(0x80 | (cp & 0x3F));
            }
            v.tokens.push_back(s);
            v.types.push_back(dlib::hf_tokenizer::TT_NORMAL);
        }
        for (const auto& t : control)
        {
            v.tokens.push_back(t);
            v.types.push_back(dlib::hf_tokenizer::TT_CONTROL);
        }
        for (const auto& t : markers)
        {
            v.tokens.push_back(t);
            v.types.push_back(dlib::hf_tokenizer::TT_USER_DEFINED);
        }
        for (size_t i = 0; i < control.size(); ++i)
        {
            if (control[i] == "<|begin_of_text|>") v.bos_id = 256 + static_cast<int>(i);
            if (control[i] == "<|im_end|>" || control[i] == "<|end_of_text|>") v.eos_id = 256 + static_cast<int>(i);
        }
        dlib::hf_tokenizer tok;
        tok.load(v);
        return tok;
    }

    struct parsed
    {
        std::string content, reasoning;
        std::vector<tool_call_data> calls;
    };

    // The events of an answer fed to a parser in pieces of the given size (0: whole).
    parsed read(const chat_format& f, const render_options& opt, const std::string& answer, size_t piece)
    {
        std::unique_ptr<output_parser> p = f.make_parser(opt);
        std::vector<output_event> events;
        if (piece == 0) p->feed(answer, events);
        else
            for (size_t i = 0; i < answer.size(); i += piece) p->feed(answer.substr(i, piece), events);
        p->finish(events);
        parsed out;
        for (const auto& e : events)
        {
            if (e.kind == output_kind::content) out.content += e.text;
            else if (e.kind == output_kind::reasoning) out.reasoning += e.text;
            else out.calls.push_back(e.call);
        }
        return out;
    }

    // Every way of feeding the answer gives the same events.
    parsed read_all_ways(const chat_format& f, const render_options& opt, const std::string& answer,
        const std::string& name)
    {
        const parsed whole = read(f, opt, answer, 0);
        for (size_t piece : { size_t(1), size_t(3), size_t(7) })
        {
            const parsed p = read(f, opt, answer, piece);
            bool same = p.content == whole.content && p.reasoning == whole.reasoning
                && p.calls.size() == whole.calls.size();
            for (size_t i = 0; same && i < p.calls.size(); ++i)
                same = p.calls[i].name == whole.calls[i].name && p.calls[i].arguments == whole.calls[i].arguments;
            expect(same, name + ": the answer read in pieces of " + std::to_string(piece) + " differs");
        }
        return whole;
    }

    std::string render(const chat_format& f, const std::vector<chat_message>& messages, const render_options& opt)
    {
        std::vector<prompt_piece> pieces;
        f.render(messages, opt, pieces);
        std::string text;
        for (const auto& p : pieces) text += p.text;
        return text;
    }
}

int main()
{
    const dlib::hf_tokenizer tok = tokenizer();
    const std::string tmpl = "{%- if tools %}<tool_call>{%- endif %}<|im_start|>{%- if enable_thinking %}{%- endif %}";
    const std::unique_ptr<chat_format> f = chat_format::detect(tok, tmpl);
    expect(std::string(f->name()) == "chatml" && f->supports_tools() && f->supports_reasoning(), "ChatML detected");

    render_options opt;
    opt.tools = json::parse_or_null("[{\"type\": \"function\", \"function\": {\"name\": \"calculator\", "
        "\"description\": \"Evaluates an expression.\", \"parameters\": {\"type\": \"object\", \"properties\": "
        "{\"expression\": {\"type\": \"string\"}}, \"required\": [\"expression\"]}}}]");

    // The prompt of the second step of an agent, as the template of Qwen3 writes it.
    {
        std::vector<chat_message> m(4);
        m[0].role = "system";
        m[0].content = "You are a helpful assistant.";
        m[1].role = "user";
        m[1].content = "What is 17.5% of 2340?";
        m[2].role = "assistant";
        tool_call_data c;
        c.id = "call_1";
        c.name = "calculator";
        c.arguments = "{\"expression\": \"2340 * 0.175\"}";
        m[2].tool_calls.push_back(c);
        m[3].role = "tool";
        m[3].tool_call_id = "call_1";
        m[3].content = "409.5";
        const std::string expected = "<|im_start|>system\nYou are a helpful assistant.\n\n# Tools\n\nYou may call one "
            "or more functions to assist with the user query.\n\nYou are provided with function signatures within "
            "<tools></tools> XML tags:\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"calculator\", "
            "\"description\": \"Evaluates an expression.\", \"parameters\": {\"type\": \"object\", \"properties\": "
            "{\"expression\": {\"type\": \"string\"}}, \"required\": [\"expression\"]}}}\n</tools>\n\nFor each "
            "function call, return a json object with function name and arguments within <tool_call></tool_call> "
            "XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
            "</tool_call><|im_end|>\n<|im_start|>user\nWhat is 17.5% of 2340?<|im_end|>\n<|im_start|>assistant\n"
            "<tool_call>\n{\"name\": \"calculator\", \"arguments\": {\"expression\": \"2340 * 0.175\"}}\n"
            "</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n409.5\n</tool_response><|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n";
        const std::string got = render(*f, m, opt);
        expect(got == expected, "prompt of an agent step:\n" + got);

        // The markers of the template become their tokens, the text of the messages stays text.
        std::vector<prompt_piece> pieces;
        f->render(m, opt, pieces);
        const std::vector<int> ids = chat_format::tokenize(tok, pieces);
        long calls = 0, responses = 0;
        for (int id : ids)
        {
            calls += id == tok.token_id("<tool_call>");
            responses += id == tok.token_id("<tool_response>");
        }
        expect(calls == 3 && responses == 1, "tokens of the markers in the prompt");
    }

    // A call as Qwen3 writes it: the arguments keep the form the model gave them.
    {
        const parsed p = read_all_ways(*f, opt, "<tool_call>\n{\"name\": \"calculator\", \"arguments\": "
            "{\"expression\": \"2340 * 0.175\"}}\n</tool_call>", "a call");
        expect(p.calls.size() == 1 && p.content.empty(), "one call, no text");
        if (p.calls.size() == 1)
        {
            expect(p.calls[0].name == "calculator", "name of the call");
            expect(p.calls[0].arguments == "{\"expression\": \"2340 * 0.175\"}", "arguments as written: "
                + p.calls[0].arguments);
            expect(p.calls[0].id.compare(0, 5, "call_") == 0, "identifier of the call");
        }
    }

    // Code with raw line breaks in a string, which JSON forbids and the models write.
    {
        const parsed p = read_all_ways(*f, opt, "<tool_call>\n{\"name\": \"run_javascript\", \"arguments\": "
            "{\"code\": \"let a = 1;\n\tconsole.log(a);\"}}\n</tool_call>", "raw line breaks");
        expect(p.calls.size() == 1 && p.content.empty(), "a call with raw line breaks is read");
        if (p.calls.size() == 1)
        {
            json args;
            const bool ok = json::parse(p.calls[0].arguments, args);
            expect(ok && args["code"].as_string() == "let a = 1;\n\tconsole.log(a);",
                "arguments with escaped line breaks: " + p.calls[0].arguments);
        }
    }

    // Reasoning, a sentence, then two calls in one block; a call left open at the end.
    {
        render_options think = opt;
        think.reasoning = true;
        const parsed p = read_all_ways(*f, think, "<think>\nI should compute it.\n</think>\n\nLet me compute.\n"
            "<tool_call>\n{\"name\": \"calculator\", \"arguments\": {\"expression\": \"1+1\"}}\n"
            "{\"name\": \"calculator\", \"arguments\": {\"expression\": \"2+2\"}}\n</tool_call>", "two calls");
        expect(p.reasoning == "I should compute it.", "reasoning: " + p.reasoning);
        expect(p.content == "Let me compute.\n", "text before the calls: " + p.content);
        expect(p.calls.size() == 2, "two calls in one block");

        const parsed open = read_all_ways(*f, opt, "<tool_call>\n{\"name\": \"current_datetime\", \"arguments\": {}}\n",
            "an open call");
        expect(open.calls.size() == 1 && open.content.empty(), "a call without its closing marker");

        const parsed fenced = read_all_ways(*f, opt, "<tool_call>\n```json\n{\"name\": \"calculator\", \"arguments\": "
            "{\"expression\": \"3*3\"}}\n```\n</tool_call>", "a fenced call");
        expect(fenced.calls.size() == 1, "a call in a fenced block");

        // A closing brace too many, after code whose braces the model closed outside the string.
        const parsed extra = read_all_ways(*f, opt, "<tool_call>\n{\"name\": \"run_javascript\", \"arguments\": "
            "{\"code\": \"for (let i = 0; i < 3; i++) {\\n  console.log(i);\"}}}\n</tool_call>", "a brace too many");
        expect(extra.calls.size() == 1 && extra.content.empty(), "a call followed by a closing brace too many");
    }

    // Text that is no call stays the text of the answer, as written.
    {
        const std::string answer = "Here: <tool_call>not json</tool_call> done.";
        const parsed p = read_all_ways(*f, opt, answer, "no call");
        expect(p.calls.empty() && p.content == answer, "text kept: " + p.content);
        const std::string unfinished = "Here: <tool_call>{\"name\": \"calculator\", \"arguments\": {\"expr";
        const parsed q = read_all_ways(*f, opt, unfinished, "an unfinished call");
        expect(q.calls.empty() && q.content == unfinished, "unfinished call kept as text: " + q.content);
    }

    // A template that always opens the reasoning (the thinking variants of Qwen3 and Qwen3-VL).
    {
        const std::unique_ptr<chat_format> t = chat_format::detect(tok, "{{- '<|im_start|>assistant\\n<think>\\n' }}");
        expect(t->supports_reasoning(), "thinking template detected");
        std::vector<chat_message> m(1);
        m[0].role = "user";
        m[0].content = "Capital of France?";
        render_options off;
        expect(render(*t, m, off) == "<|im_start|>user\nCapital of France?<|im_end|>\n<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n", "an empty reasoning when the request asks for none");
        render_options on;
        on.reasoning = true;
        expect(render(*t, m, on) == "<|im_start|>user\nCapital of France?<|im_end|>\n<|im_start|>assistant\n"
            "<think>\n", "the reasoning opened by the prompt");
        const parsed p = read_all_ways(*t, on, "\nThe capital is Paris.\n</think>\n\nParis.", "opened reasoning");
        expect(p.reasoning == "The capital is Paris." && p.content == "Paris.", "reasoning then answer: "
            + p.reasoning + " / " + p.content);
        const parsed q = read_all_ways(*t, off, "Paris.", "no reasoning");
        expect(q.reasoning.empty() && q.content == "Paris.", "answer without reasoning: " + q.content);
    }

    // Granite 4.2: ChatML with the XML form of the calls.
    {
        const std::unique_ptr<chat_format> g = chat_format::detect(tok,
            "{{- '<tool_call>\\n<function=' ~ tool_call.name ~ '>\\n' -}}"
            "{{- '<|im_start|>assistant\\n<think></think>' }}");
        expect(std::string(g->name()) == "granite4" && g->supports_tools() && g->supports_reasoning(),
            "Granite 4.2 detected");
        render_options gopt;
        gopt.tools = json::parse_or_null("[{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", "
            "\"description\": \"Weather of a city.\", \"parameters\": {\"type\": \"object\", \"properties\": "
            "{\"city\": {\"type\": \"string\"}, \"days\": {\"type\": \"integer\"}, \"metric\": {\"type\": "
            "\"boolean\"}}, \"required\": [\"city\"]}}}]");
        std::vector<chat_message> m(6);
        m[0].role = "system";
        m[0].content = "You are a helpful assistant.";
        m[1].role = "user";
        m[1].content = "What is the weather in Paris?";
        m[2].role = "assistant";
        tool_call_data c;
        c.id = "call_1";
        c.name = "get_weather";
        c.arguments = "{\"city\": \"Paris\", \"days\": 2, \"metric\": true}";
        m[2].tool_calls.push_back(c);
        m[3].role = "tool";
        m[3].content = "{\"temperature\": 12}";
        m[4].role = "assistant";
        m[4].content = "It is 12 degrees in Paris.";
        m[5].role = "user";
        m[5].content = "And in Rome?";
        // As the template of granite-4.2-3b-GGUF writes it.
        const std::string expected = "<|im_start|>system\nYou are a helpful assistant.\n\n# Tools\n\nYou have access "
            "to the following functions:\n\n<tools>\n{\"name\": \"get_weather\", \"description\": \"Weather of a "
            "city.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}, "
            "\"days\": {\"type\": \"integer\"}, \"metric\": {\"type\": \"boolean\"}}, \"required\": [\"city\"]}}\n"
            "</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
            "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
            "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple "
            "lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow "
            "the specified format: an inner <function=...></function> block must be nested within "
            "<tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional "
            "reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If "
            "there is no function call available, answer the question like normal with your current knowledge and "
            "do not tell the user about function calls\n</IMPORTANT><|im_end|>\n<|im_start|>user\nWhat is the "
            "weather in Paris?<|im_end|>\n<|im_start|>assistant\n<think></think>\n<tool_call>\n"
            "<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n2\n</parameter>\n"
            "<parameter=metric>\nTrue\n</parameter>\n</function>\n</tool_call>\n<|im_end|>\n<|im_start|>user\n"
            "<tool_response>\n{\"temperature\": 12}\n</tool_response>\n<|im_end|>\n<|im_start|>assistant\n"
            "<think></think>It is 12 degrees in Paris.<|im_end|>\n<|im_start|>user\nAnd in Rome?<|im_end|>\n"
            "<|im_start|>assistant\n<think></think>";
        const std::string got = render(*g, m, gopt);
        expect(got == expected, "Granite 4.2 prompt:\n" + got);

        const parsed p = read_all_ways(*g, gopt, "\n<tool_call>\n<function=get_weather>\n<parameter=city>\n123\n"
            "</parameter>\n<parameter=days>\n3\n</parameter>\n<parameter=metric>\nFalse\n</parameter>\n</function>\n"
            "</tool_call>", "an XML call");
        expect(p.calls.size() == 1, "one XML call");
        if (p.calls.size() == 1)
            expect(p.calls[0].name == "get_weather" && p.calls[0].arguments ==
                "{\"city\": \"123\", \"days\": 3, \"metric\": false}", "XML arguments typed: " + p.calls[0].arguments);
        const parsed open = read_all_ways(*g, gopt, "<tool_call>\n<function=get_weather>\n<parameter=city>\nRome\n"
            "</parameter>\n", "an XML call left open");
        expect(open.calls.size() == 1 && open.content.empty(), "an XML call without its closing markers");
        const std::string text = "Use <tool_call>no function</tool_call> here.";
        const parsed none = read_all_ways(*g, gopt, text, "no XML call");
        expect(none.calls.empty() && none.content == text, "text kept: " + none.content);
    }

    // Foundation-Sec-8B-Reasoning: <|system|>, <|user|>, <|assistant|> turns, an answer after the reasoning.
    {
        const dlib::hf_tokenizer ftok = tokenizer({ "<|begin_of_text|>", "<|end_of_text|>", "<|system|>",
            "<|user|>", "<|assistant|>" }, { "<think>", "</think>" });
        const std::string ftmpl = "{{ bos_token }}{%- if not has_system %}\n<|system|>\nYou are Metis.\n\n"
            "{%- endif %}<|user|>\n<|assistant|>";
        const std::unique_ptr<chat_format> f2 = chat_format::detect(ftok, ftmpl);
        expect(std::string(f2->name()) == "foundation-sec" && f2->supports_reasoning(), "Foundation-Sec detected");
        std::vector<chat_message> m(4);
        m[0].role = "system";
        m[0].content = "You are a security analyst.";
        m[1].role = "user";
        m[1].content = "What is XSS?";
        m[2].role = "assistant";
        m[2].content = "<think>\nThe user asks.\n</think>\n\nCross-site scripting is an injection.  ";
        m[3].role = "user";
        m[3].content = "Give an example.";
        render_options fopt;
        expect(render(*f2, m, fopt) == "<|begin_of_text|><|system|>\nYou are a security analyst.<|user|>\nWhat is "
            "XSS?<|assistant|>Cross-site scripting is an injection.<|end_of_text|><|user|>\nGive an example."
            "<|assistant|>", "Foundation-Sec prompt");
        std::vector<chat_message> q(m.begin() + 3, m.end());
        expect(render(*f2, q, fopt) == "<|begin_of_text|><|system|>\nYou are Metis.<|user|>\nGive an example."
            "<|assistant|>", "the system prompt of the template");
        const parsed p = read_all_ways(*f2, fopt, "\nThe user wants an example.</think>\n\n<script>", "reasoning");
        expect(p.reasoning == "The user wants an example." && p.content == "<script>", "reasoning, answer: "
            + p.reasoning + " / " + p.content);
        const parsed r = read_all_ways(*f2, fopt, "<think>\nShort.\n</think>\nDone.", "reasoning opened again");
        expect(r.reasoning == "Short." && r.content == "Done.", "opening marker written again: " + r.reasoning);
    }

    // Phi-4-mini.
    {
        const dlib::hf_tokenizer ptok = tokenizer({ "<|endoftext|>", "<|system|>", "<|user|>", "<|assistant|>",
            "<|end|>" }, {});
        const std::unique_ptr<chat_format> p = chat_format::detect(ptok,
            "{{ '<|' + message['role'] + '|>' + message['content'] + '<|end|>' }}");
        expect(std::string(p->name()) == "phi" && !p->supports_tools(), "Phi detected");
        std::vector<chat_message> m(2);
        m[0].role = "system";
        m[0].content = "Be brief.";
        m[1].role = "user";
        m[1].content = "Hi";
        render_options popt;
        expect(render(*p, m, popt) == "<|system|>Be brief.<|end|><|user|>Hi<|end|><|assistant|>", "Phi prompt");
    }

    if (failures)
    {
        std::printf("%d checks FAILED\n", failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
