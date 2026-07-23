/*!
    @file slm_gguf_runtime_ex.cpp
    @brief Dynamic GGUF inference: load any supported model without recompiling.

    Counterpart of slm_gguf_import_ex for the runtime engine: the architecture is
    resolved at load time from the GGUF metadata, so this program compiles once and
    probes any supported model. It shares the validation protocol of the static path,
    same options, same output format, so the per-position dumps of the two engines
    and of an external reference implementation are directly comparable.

    The interactive loop and the served endpoint drive the same generation core
    (dlib/dnn/text_generation.h): they differ only in how they collect the conversation
    and how they render the answer, never in what they compute.

    Usage:
      slm_gguf_runtime_ex --input model.gguf --probe-logits --prompt "The capital of France is"
      slm_gguf_runtime_ex --input model.gguf --probe-ids "1 450 7483 310 3444 338"
      slm_gguf_runtime_ex --input model.gguf --save-dat model.dat
      slm_gguf_runtime_ex --input model.dat --chat
      slm_gguf_runtime_ex --input a.gguf,b.gguf --serve 5000
!*/

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <algorithm>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <sys/ioctl.h>
#  include <unistd.h>
#endif

#include <chrono>
#include <memory>
#include <thread>

#include <dlib/cmd_line_parser.h>
#include <dlib/misc_api.h>
#include <dlib/dnn.h>
#include <dlib/tokenizer/hf_tokenizer.h>
#include <dlib/tokenizer/chat_template.h>
#include <dlib/server/server_chat.h>

using namespace std;
using namespace dlib;

/* Print the top-k next-token candidates of the last position and the argmax of every
   position, in the exact format of the static-path probes. */
static void report_logits(const tensor& logits, const std::vector<int>& toks, hf_tokenizer& tok)
{
    const long N = logits.nr();
    const long V = logits.nc();

    /* Softmax of the last position, on the host: probe-scale work. */
    std::vector<double> probs(static_cast<size_t>(V));
    {
        const float* row = last_logits_row(logits);
        const float mx = *std::max_element(row, row + V);
        double sum = 0.0;
        for (long v = 0; v < V; ++v) { probs[v] = std::exp(row[v] - mx); sum += probs[v]; }
        for (long v = 0; v < V; ++v) probs[v] /= sum;
    }
    std::vector<long> order(static_cast<size_t>(V));
    for (long v = 0; v < V; ++v) order[v] = v;
    std::partial_sort(order.begin(), order.begin() + 5, order.end(),
        [&](long a, long b) { return probs[a] > probs[b]; });

    cout << "Most probable next tokens:\n";
    for (int i = 0; i < 5; ++i)
    {
        const long id = order[static_cast<size_t>(i)];
        cout << "  " << probs[id] << "  id " << id << "  \""
             << tok.decode({ static_cast<int>(id) }, false) << "\"\n";
    }

    cout << "Token ids fed:";
    for (int t : toks) cout << ' ' << t;
    cout << "\nPer-position argmax (pos: predicted_id 'tok' prob):\n";
    for (long p = 0; p < N; ++p)
    {
        const float* row = logits_row(logits, p);
        const long am = static_cast<long>(std::max_element(row, row + V) - row);
        const float mx = row[am];
        double sum = 0.0;
        for (long v = 0; v < V; ++v) sum += std::exp(static_cast<double>(row[v]) - mx);
        cout << "  " << p << ": " << am << " '" << tok.decode({ static_cast<int>(am) }, false)
             << "' " << (1.0 / sum) << "\n";
    }
}

/* Width of the terminal, for the physical-row accounting of the erase sequences. */
static long terminal_width()
{
#ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO info;
    if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &info))
        return info.srWindow.Right - info.srWindow.Left + 1;
#else
    winsize w{};
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0 && w.ws_col > 0)
        return w.ws_col;
#endif
    return 80;
}

/* --trace-prompt output, identical on both front ends: the decoded stream with its
   special tokens, the ids, and the sampling line actually resolved for the call.
   Comparing two runs is only conclusive when both print the same thing here. */
static void trace_stream(const hf_tokenizer& tok, const std::vector<int>& ids,
    const sampling_params& sp, const std::string& what)
{
    cout << "---- " << what << " (" << ids.size() << " tokens) ----\n"
         << tok.decode(ids, false) << "\n---- ids:";
    for (int id : ids) cout << ' ' << id;
    cout << "\n---- sampling: temp " << (sp.greedy ? 0.0 : sp.temperature)
         << (sp.greedy ? " (greedy)" : "")
         << ", top_k " << sp.top_k << ", top_p " << sp.top_p
         << ", min_p " << sp.min_p << ", repeat " << sp.repeat_penalty
         << " ----" << std::endl;
}

/* Interactive chat: the first turn goes through the prefill, every later turn is fed
   token by token to the incremental step, and the KV cache lives across turns. Only
   the display policy is local; the token stream and the generation loop come from the
   shared core, so this loop and the served endpoint compute the same thing. */
static int chat_loop(runtime_transformer& rt, hf_tokenizer& tok, const chat_template_formatter& fmt,
    long ctx_len, const sampling_params& sp, long max_new, const std::string& system_prompt,
    bool trace_prompt = false)
{
    rt.set_context(ctx_len, system_keep_length(tok, fmt, system_prompt));

    token_sampler sampler;
    const int eos = tok.eos_id();
    const long tick_width = std::max<long>(20, terminal_width() - 12);

    cout << "Chat template: " << chat_template_formatter::name(fmt.kind()) << "\n";
    cout << "Ready. Type 'quit' or 'exit' to stop.\n";

    bool first = true;
    for (;;)
    {
        cout << "You: " << std::flush;
        std::string user;
        if (!std::getline(std::cin, user)) break;
        if (user == "quit" || user == "exit") break;
        if (user.empty()) continue;

        const std::vector<int> toks = encode_turn(tok, fmt, system_prompt, user, first);
        if (trace_prompt) trace_stream(tok, toks, sp, "chat turn");

        const tensor* logits = nullptr;
        if (first)
        {
            logits = &rt.forward_prefill(toks);
            first = false;
        }
        else
        {
            for (int t : toks) logits = &rt.step(t);
        }

        /* Two display regimes. Text destined to disappear (an open reasoning span,
           template residue) is shown as a one-line ticker rewritten in place with the
           tail of the trace: it never wraps nor scrolls, so it can always be erased,
           unlike a multi-row region whose top may leave the screen, beyond the reach of
           any cursor sequence. The visible answer arrives already held back by the
           generation core, streams by stable suffix, and persists. */
        bool ticker = false;
        std::string shown;
        std::vector<int> recent;

        generation_options opt;
        opt.max_new_tokens = max_new;
        opt.is_cancelled = []() { return dlib::signal_handler::is_triggered(); };
        opt.on_token = [&](const generation_event& ev)
        {
            if (!ev.clean_delta.empty())
            {
                if (ticker) { cout << "\r\033[KModel: "; ticker = false; }
                cout << ev.clean_delta << std::flush;
                shown += ev.clean_delta;
            }
            else if (ev.reasoning_open && !ev.answer.empty())
            {
                std::string tail = ev.answer.size() > static_cast<size_t>(tick_width)
                    ? ev.answer.substr(ev.answer.size() - tick_width) : ev.answer;
                for (char& c : tail) if (c == '\n' || c == '\r') c = ' ';
                cout << "\r\033[KModel: \033[2m" << tail << "\033[0m" << std::flush;
                ticker = true;
            }
        };

        cout << "Model: " << std::flush;
        const generation_result res =
            generate_reply(rt, tok, fmt, *logits, sampler, sp, recent, opt);

        /* Epilogue: settle the display on the final cleaned answer. Only a genuine
           divergence reprints; the display merely being ahead by held-back or trimmed
           trailing characters is left as is. */
        if (ticker) cout << "\r\033[KModel: ";
        if (res.text.size() > shown.size() && res.text.compare(0, shown.size(), shown) == 0)
            cout << res.text.substr(shown.size());
        else if (res.text != shown
            && !(res.text.size() <= shown.size() && shown.compare(0, res.text.size(), res.text) == 0))
            cout << (shown.empty() ? "" : "\nModel: ") << res.text;
        cout << "\n";
        if (res.truncated) cout << "[stopped at " << max_new << " tokens]\n";
        if (res.cancelled) break;

        if (eos >= 0) rt.step(eos);   // close the assistant turn in the token stream
    }
    return 0;
}

/* One model served by the chat endpoint: the engine, its tokenizer and the chat
   template detected for it. Several can be loaded side by side; the request's
   "model" field selects one, the first being the default. */
struct served_model
{
    std::string name;
    runtime_transformer* rt;
    hf_tokenizer* tok;
    chat_template_formatter fmt;
};

/* OpenAI-compatible service over the runtime engine. Each request carries the whole
   conversation, replayed into the token stream the interactive loop would have produced
   turn by turn, then handed to the same generation core: serving and interactive chat
   share one numeric path by construction rather than by parallel maintenance. The
   server is stateless and server_chat serializes the calls, which matches the engine's
   single-generation-thread assumption. */
class runtime_chat_server : public dlib::server_chat
{
public:
    runtime_chat_server(std::vector<served_model> models, long ctx,
        double forced_temp, bool temp_forced, bool deterministic, bool trace_prompt)
        : models_(std::move(models)), ctx_(ctx), temp_(forced_temp),
          temp_forced_(temp_forced), det_(deterministic), trace_prompt_(trace_prompt)
    {
        std::vector<dlib::chat_model_info> infos;
        for (const served_model& m : models_)
            infos.push_back(dlib::chat_model_info{ m.name, m.fmt.supports_reasoning() });
        set_models(infos);
    }

private:
    served_model& select(const std::string& id)
    {
        for (served_model& m : models_)
            if (m.name == id) return m;
        return models_.front();   // first declared model is the default
    }

    /* Sampling resolved per request against the target model's own presets, never
       against another served model's; each request override applies on top. Unset
       overrides arrive negative. */
    sampling_params resolve_sampling(const dlib::chat_request& req,
        const chat_template_formatter& fmt) const
    {
        sampling_params sp;
        sp.temperature = req.temperature >= 0.0 ? req.temperature
            : (temp_forced_ ? temp_ : fmt.default_temperature());
        sp.top_k = req.top_k >= 0 ? static_cast<size_t>(req.top_k) : fmt.default_top_k();
        sp.top_p = req.top_p >= 0.0 ? static_cast<float>(req.top_p) : fmt.default_top_p();
        sp.min_p = req.min_p >= 0.0 ? static_cast<float>(req.min_p) : fmt.default_min_p();
        sp.repeat_penalty = req.repeat_penalty >= 0.0
            ? static_cast<float>(req.repeat_penalty) : fmt.default_repeat_penalty();
        sp.greedy = det_ || sp.temperature <= 0.0;
        return sp;
    }

    dlib::chat_result on_chat_completion(const dlib::chat_request& req,
        const std::function<void(const std::string&)>& emit) override
    {
        served_model& use = select(req.model);
        runtime_transformer& rt = *use.rt;
        hf_tokenizer& tok = *use.tok;
        chat_template_formatter fmt = use.fmt;
        if (req.reasoning >= 0 && fmt.supports_reasoning())
            fmt.set_reasoning(req.reasoning == 1);

        /* Split the wire messages: every system part joins the system block, a user
           message opens a turn, an assistant message closes the turn before it. */
        std::string sys;
        std::vector<dlib::chat_turn> turns;
        for (const dlib::chat_message& m : req.messages)
        {
            if (m.role == "system")
            {
                if (!sys.empty()) sys += "\n";
                sys += m.content;
            }
            else if (m.role == "user")
            {
                std::string text = m.content;
                for (size_t k = 0; k < m.image_urls.size(); ++k)
                    text += "\n[attached image: not visible to this text-only model]";
                turns.push_back(dlib::chat_turn{ text, std::string() });
            }
            else if (m.role == "assistant" && !turns.empty())
            {
                turns.back().assistant = m.content;
            }
        }
        if (turns.empty())
            throw std::runtime_error("the conversation contains no user message");
        turns.back().assistant.clear();   // the turn being answered carries no reply yet

        const std::vector<int> ids = encode_conversation(tok, fmt, sys, turns);
        const long max_new = req.max_tokens > 0 ? req.max_tokens : 512;
        if (static_cast<long>(ids.size()) + 8 >= ctx_)
            throw std::runtime_error("prompt exceeds the context capacity; reduce the "
                "context budget in the interface settings");

        const sampling_params sp = resolve_sampling(req, fmt);
        if (trace_prompt_) trace_stream(tok, ids, sp, "prompt to " + use.name);

        /* Same capacity and same pinned prefix as the interactive loop: an eviction
           mid-answer must drop the same rows on both paths. set_context() reallocates
           only when the geometry changes, so the per-request cost is the cache reset. */
        rt.set_context(ctx_, system_keep_length(tok, fmt, sys));

        std::vector<int> recent;
        std::string streamed;
        generation_options opt;
        opt.max_new_tokens = max_new;
        opt.is_cancelled = [&req]() {
            return dlib::signal_handler::is_triggered()
                || (req.is_cancelled && req.is_cancelled());
        };
        /* Stable suffixes only: the client never receives a fragment of the stop marker
           nor a trailing blank that the final answer will have trimmed. An open
           reasoning span still streams raw, so the interface keeps showing the trace
           live; the server's streaming path buffers any incomplete UTF-8 tail on top. */
        opt.on_token = [&](const generation_event& ev)
        {
            if (!ev.clean_delta.empty()) emit(ev.clean_delta);
            else if (ev.reasoning_open && ev.answer.size() > streamed.size())
            {
                emit(ev.answer.substr(streamed.size()));
                streamed = ev.answer;
            }
        };

        const generation_result gen =
            generate_reply(rt, tok, fmt, rt.forward_prefill(ids), sampler_, sp, recent, opt);

        dlib::chat_result res;
        res.prompt_tokens = static_cast<long>(ids.size());
        res.completion_tokens = static_cast<long>(gen.tokens.size());
        res.finish_reason = gen.truncated ? "length" : "stop";
        res.content = gen.text;
        return res;
    }

    /* Declaration order follows the constructor's initializer list. */
    std::vector<served_model> models_;
    long ctx_;
    double temp_;
    bool temp_forced_;
    bool det_;
    bool trace_prompt_;
    token_sampler sampler_;
};

int main(int argc, char** argv)
{
    try
    {
#ifdef _WIN32
        SetConsoleOutputCP(CP_UTF8);
        SetConsoleCP(CP_UTF8);
        {
            HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
            DWORD mode = 0;
            if (h != INVALID_HANDLE_VALUE && GetConsoleMode(h, &mode))
                SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
        }
#endif
        command_line_parser parser;
        parser.add_option("input", "Path to the source model: .gguf, or a runtime .dat archive", 1);
        parser.add_option("save-dat", "After loading a GGUF, write a self-contained runtime archive (model + tokenizer)", 1);
        parser.add_option("probe-logits", "Run a prefill and report the logits");
        parser.add_option("probe-ids", "Feed explicit token ids (space or comma separated)", 1);
        parser.add_option("probe-step", "Self-consistency check: last-position logits of a full prefill versus prefill(N-1) + step(last)");
        parser.add_option("prompt", "Prompt for --probe-logits (default: capital of France)", 1);
        parser.add_option("rope-permute", "Map split-half (NeoX) Q/K layouts to the interleaved kernel");
        parser.add_option("chat", "Interactive chat with the loaded model");
        parser.add_option("serve", "Serve the web chat interface and an OpenAI-compatible API on the given port", 1);
        parser.add_option("trace-prompt", "Print the rebuilt token stream and the resolved sampling of every generation");
        parser.add_option("system", "System prompt for chat mode", 1);
        parser.add_option("context", "KV cache capacity in tokens (default: 2048)", 1);
        parser.add_option("max-tokens", "Maximum number of generated tokens per answer (default: 512)", 1);
        parser.add_option("template", "Chat template override: auto, zephyr, chatml, guanaco, granite", 1);
        parser.add_option("temp", "Sampling temperature (default: template preset)", 1);
        parser.add_option("deterministic", "Greedy decoding");
        parser.add_option("think", "Let thinking-capable models produce their reasoning trace (streamed, then hidden)");
        parser.add_option("resident", "Dequantize all weights at load time (fastest forward, full f32 footprint); default keeps them quantized at rest");
        parser.parse(argc, argv);

        if (!parser.option("input"))
        {
            cout << "Dynamic GGUF inference: load any supported model without recompiling.\n\n";
            parser.print_options();
            return 0;
        }

        std::vector<string> inputs;
        {
            /* The shell only expands a leading tilde on the first word, so the
               second and later comma-separated paths arrive verbatim; expand it
               here for every entry. */
            const char* home = std::getenv("HOME");
#ifdef _WIN32
            if (!home) home = std::getenv("USERPROFILE");
#endif
            string all = parser.option("input").argument();
            std::istringstream iss(all);
            string p;
            while (std::getline(iss, p, ','))
            {
                if (p.empty()) continue;
                if (home && p.size() >= 2 && p[0] == '~' && (p[1] == '/' || p[1] == '\\'))
                    p = string(home) + p.substr(1);
                inputs.push_back(p);
            }
        }
        const string input = inputs.front();
        if (inputs.size() > 1 && !parser.option("serve"))
            cout << "note: extra --input models are only loaded in --serve mode\n";
        runtime_transformer rt;
        hf_tokenizer tok;

        const bool from_archive = input.size() > 4 && input.substr(input.size() - 4) == ".dat";
        if (from_archive)
        {
            cout << "Loading runtime archive: " << input << "\n";
            std::ifstream in(input, std::ios::binary);
            if (!in) { cerr << "Cannot open " << input << "\n"; return 1; }
            rt.load(in);
            deserialize(tok, in);
            cout << describe(rt.spec()) << "\n";
        }
        else
        {
            cout << "Reading GGUF: " << input << "\n";
            gguf_reader g(input);
            const model_spec spec = detect_model(g);
            cout << describe(spec) << "\n";

            const compat_result rep = check_compatibility(spec, /*fused_attention_path=*/false);
            for (const string& n : rep.notes) cout << "note: " << n << "\n";
            if (!rep.blockers.empty())
            {
                for (const string& b : rep.blockers) cerr << "blocker: " << b << "\n";
                return 1;
            }

            gguf_load_options lopt;
            lopt.rope_permute = parser.option("rope-permute");
            lopt.verbose = true;

            rt.set_quantized_at_rest(!parser.option("resident"));
            cout << "Loading weights (runtime engine"
                 << (rt.quantized_at_rest() ? ", quantized at rest" : "") << ")...\n";
            rt.load(g, spec, lopt);
            tok.load_from_gguf(g);

            if (parser.option("save-dat"))
            {
                const string path = parser.option("save-dat").argument();
                std::ofstream out(path, std::ios::binary);
                if (!out) { cerr << "Cannot write " << path << "\n"; return 1; }
                rt.save(out);
                serialize(tok, out);
                cout << "Runtime archive written: " << path << "\n";
            }
        }

        const long ctx = parser.option("context")
            ? std::stol(parser.option("context").argument()) : 2048;
        const long max_new = parser.option("max-tokens")
            ? std::stol(parser.option("max-tokens").argument()) : 512;

        if (parser.option("chat"))
        {
            chat_template_formatter fmt = parser.option("template")
                ? chat_template_formatter::for_tokenizer(tok,
                    chat_template_formatter::from_name(parser.option("template").argument()))
                : chat_template_formatter::for_tokenizer(tok, rt.spec().model_name);
            if (parser.option("think"))
            {
                if (fmt.supports_reasoning()) fmt.set_reasoning(true);
                else cout << "note: this model exposes no reasoning mode; --think ignored\n";
            }
            const std::string sys = parser.option("system")
                ? parser.option("system").argument()
                : std::string("You are a helpful assistant.");

            /* The presets of the template family, overridden by --temp when given.
               greedy also covers a zero temperature, which the sampler must never
               divide by. */
            sampling_params sp;
            sp.temperature = parser.option("temp")
                ? std::stod(parser.option("temp").argument()) : fmt.default_temperature();
            sp.top_k = fmt.default_top_k();
            sp.top_p = fmt.default_top_p();
            sp.min_p = fmt.default_min_p();
            sp.repeat_penalty = fmt.default_repeat_penalty();
            sp.greedy = parser.option("deterministic") || sp.temperature <= 0.0;

            return chat_loop(rt, tok, fmt, ctx, sp, max_new, sys, parser.option("trace-prompt"));
        }

        if (parser.option("serve"))
        {
            chat_template_formatter fmt = parser.option("template")
                ? chat_template_formatter::for_tokenizer(tok,
                    chat_template_formatter::from_name(parser.option("template").argument()))
                : chat_template_formatter::for_tokenizer(tok, rt.spec().model_name);
            if (parser.option("think") && fmt.supports_reasoning()) fmt.set_reasoning(true);
            const int port = std::stoi(parser.option("serve").argument());

            /* First model: the one loaded by the main flow. Extra models come from
               the comma-separated --input list; each gets its own auto-detected
               template unless --template forces one for all. */
            std::vector<std::unique_ptr<runtime_transformer>> extra_rt;
            std::vector<std::unique_ptr<hf_tokenizer>> extra_tok;
            std::vector<served_model> models;
            /* Display identity: the file stem is always descriptive (metadata
               general.name sometimes is not), normalized by the shared cleaner
               which also strips the quantization and container markers. */
            auto label = [](const runtime_transformer& r, const string& path) {
                const size_t sl = path.find_last_of("/\\");
                string stem = sl == string::npos ? path : path.substr(sl + 1);
                if (stem.size() > 4 && stem.compare(stem.size() - 4, 4, ".dat") == 0)
                    stem.erase(stem.size() - 4);
                stem = clean_model_name(stem);
                return stem.empty() ? clean_model_name(r.spec().model_name) : stem;
            };
            models.push_back(served_model{ label(rt, input), &rt, &tok, fmt });
            for (size_t i = 1; i < inputs.size(); ++i)
            {
                extra_rt.emplace_back(new runtime_transformer());
                extra_tok.emplace_back(new hf_tokenizer());
                runtime_transformer& r = *extra_rt.back();
                hf_tokenizer& t = *extra_tok.back();
                const string& path = inputs[i];
                cout << "Loading additional model: " << path << "\n";
                if (path.size() > 4 && path.substr(path.size() - 4) == ".dat")
                {
                    std::ifstream in2(path, std::ios::binary);
                    if (!in2) { cerr << "Cannot open " << path << "\n"; return 1; }
                    r.load(in2);
                    deserialize(t, in2);
                }
                else
                {
                    gguf_reader g2(path);
                    const model_spec sp2 = detect_model(g2);
                    const compat_result rep2 = check_compatibility(sp2, false);
                    if (!rep2.blockers.empty())
                    {
                        for (const string& b : rep2.blockers) cerr << "blocker: " << b << "\n";
                        return 1;
                    }
                    gguf_load_options lo2;
                    lo2.rope_permute = parser.option("rope-permute");
                    lo2.verbose = false;
                    r.set_quantized_at_rest(!parser.option("resident"));
                    r.load(g2, sp2, lo2);
                    t.load_from_gguf(g2);
                }
                chat_template_formatter f2 = parser.option("template")
                    ? chat_template_formatter::for_tokenizer(t,
                        chat_template_formatter::from_name(parser.option("template").argument()))
                    : chat_template_formatter::for_tokenizer(t, r.spec().model_name);
                models.push_back(served_model{ label(r, path), &r, &t, f2 });
            }

            cout << "Serving " << models.size() << " model(s):\n";
            for (const served_model& m : models)
                cout << "  " << m.name
                     << "  [template " << chat_template_formatter::name(m.fmt.kind())
                     << (m.fmt.supports_reasoning() ? ", reasoning" : "") << "]\n";

            runtime_chat_server srv(std::move(models), ctx,
                parser.option("temp") ? std::stod(parser.option("temp").argument()) : 0.0,
                parser.option("temp") != 0,
                parser.option("deterministic"), parser.option("trace-prompt"));
            srv.set_listening_port(port);
            cout << "Serving http://localhost:" << port
                 << "  (chat interface on /, API on /v1/chat/completions; Ctrl-C to stop)\n";
            dlib::signal_handler::setup();
            srv.start_async();
            while (!dlib::signal_handler::is_triggered() && srv.is_running())
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            cout << "\nShutting down (waiting for in-flight requests)...\n";
            srv.clear();   // shuts connections, waits for handlers, releases the port
            cout << "Server stopped.\n";
            return 0;
        }

        std::vector<int> toks;
        if (parser.option("probe-ids"))
        {
            string s = parser.option("probe-ids").argument();
            for (char& c : s) if (c == ',') c = ' ';
            std::istringstream iss(s);
            long v;
            while (iss >> v) toks.push_back(static_cast<int>(v));
        }
        else
        {
            const string prompt = parser.option("prompt")
                ? parser.option("prompt").argument()
                : string("The capital of France is");
            toks = tok.encode(prompt);
            cout << "Prompt (" << toks.size() << " tokens): \"" << prompt << "\"\n";
        }
        if (toks.empty()) { cerr << "Nothing to feed.\n"; return 1; }

        if (parser.option("probe-step"))
        {
            /* The prefill path is the reference (validated by external parity); the
               incremental path must reproduce its last-position logits through the KV
               cache machinery: append, window gather, rotation on read, QK-norm
               ordering. Differences beyond float accumulation indicate a step bug. */
            if (toks.size() < 2) { cerr << "probe-step needs at least 2 tokens.\n"; return 1; }
            const tensor& full = rt.forward_prefill(toks);
            const long V = full.nc();
            const float* full_last = last_logits_row(full);
            std::vector<float> ref(full_last, full_last + V);

            rt.set_context(static_cast<long>(toks.size()) + 8);
            std::vector<int> head(toks.begin(), toks.end() - 1);
            rt.forward_prefill(head);
            const tensor& inc = rt.step(toks.back());

            const float* a = last_logits_row(inc);
            double dmax = 0.0;
            long amax = 0;
            for (long v = 0; v < V; ++v)
            {
                const double dv = std::abs(static_cast<double>(a[v]) - ref[static_cast<size_t>(v)]);
                if (dv > dmax) { dmax = dv; amax = v; }
            }
            const long am_full = static_cast<long>(std::max_element(ref.begin(), ref.end()) - ref.begin());
            const long am_inc = static_cast<long>(std::max_element(a, a + V) - a);
            cout << "Self-consistency prefill vs incremental (last position):\n"
                 << "  max |logit diff| " << dmax << " at id " << amax << "\n"
                 << "  argmax full " << am_full << " '" << tok.decode({ static_cast<int>(am_full) }, false)
                 << "'  vs incremental " << am_inc << " '" << tok.decode({ static_cast<int>(am_inc) }, false) << "'\n"
                 << ((am_full == am_inc && dmax < 1e-3) ? "  CONSISTENT\n" : "  DIVERGENT\n");
            return 0;
        }

        const tensor& logits = rt.forward_prefill(toks);
        report_logits(logits, toks, tok);
        return 0;
    }
    catch (const std::exception& e)
    {
        cerr << "Exception thrown: " << e.what() << "\n";
        return 1;
    }
}
