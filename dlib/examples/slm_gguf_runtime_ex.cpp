/*!
    @file slm_gguf_runtime_ex.cpp
    @brief Dynamic GGUF inference: load any supported model without recompiling.

    Counterpart of slm_gguf_import_ex for the runtime engine: the architecture is
    resolved at load time from the GGUF metadata, so this program compiles once and
    probes any supported model. It shares the validation protocol of the static path,
    same options, same output format, so the per-position dumps of the two engines
    and of an external reference implementation are directly comparable.

    Usage:
      slm_gguf_runtime_ex --input model.gguf --probe-logits --prompt "The capital of France is"
      slm_gguf_runtime_ex --input model.gguf --probe-ids "1 450 7483 310 3444 338"
      slm_gguf_runtime_ex --input model.gguf --save-dat model.dat
      slm_gguf_runtime_ex --input model.dat --chat
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
#include <random>
#include <thread>

#include <dlib/cmd_line_parser.h>
#include <dlib/misc_api.h>
#include <dlib/dnn/runtime_transformer.h>
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
    const float* lg = logits.host();

    /* Softmax of the last position, on the host: probe-scale work. */
    std::vector<double> probs(static_cast<size_t>(V));
    {
        const float* row = lg + (N - 1) * V;
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
        const float* row = lg + p * V;
        const long am = static_cast<long>(std::max_element(row, row + V) - row);
        const float mx = row[am];
        double sum = 0.0;
        for (long v = 0; v < V; ++v) sum += std::exp(static_cast<double>(row[v]) - mx);
        cout << "  " << p << ": " << am << " '" << tok.decode({ static_cast<int>(am) }, false)
             << "' " << (1.0 / sum) << "\n";
    }
}

/* Sample the next token from a [1,1,1,vocab] logits tensor: repeat penalty over the
   recent window, temperature, top-k, min-p and top-p nucleus filtering. Deterministic
   mode returns the argmax. */
static int pick_next(const tensor& logits, const std::vector<int>& recent,
    double temperature, size_t top_k, float top_p, float min_p, float repeat_penalty,
    bool deterministic, std::mt19937& rng)
{
    const long V = logits.nc();
    const float* row = logits.host();
    std::vector<double> lg(row, row + V);

    for (int t : recent)
    {
        double& v = lg[static_cast<size_t>(t)];
        v = v > 0 ? v / repeat_penalty : v * repeat_penalty;
    }
    if (deterministic)
        return static_cast<int>(std::max_element(lg.begin(), lg.end()) - lg.begin());

    std::vector<long> order(static_cast<size_t>(V));
    for (long v = 0; v < V; ++v) order[static_cast<size_t>(v)] = v;
    const size_t k = std::min<size_t>(top_k ? top_k : static_cast<size_t>(V), static_cast<size_t>(V));
    std::partial_sort(order.begin(), order.begin() + k, order.end(),
        [&](long a, long b) { return lg[a] > lg[b]; });

    std::vector<double> p(k);
    const double mx = lg[order[0]];
    double sum = 0.0;
    for (size_t i = 0; i < k; ++i) { p[i] = std::exp((lg[order[i]] - mx) / temperature); sum += p[i]; }
    for (size_t i = 0; i < k; ++i) p[i] /= sum;

    size_t last = k;
    double kept = 0.0;
    for (size_t i = 0; i < k; ++i)
    {
        if (p[i] < min_p * p[0]) { last = i; break; }
        kept += p[i];
        if (kept >= top_p) { last = i + 1; break; }
    }
    if (last == 0) last = 1;

    std::discrete_distribution<size_t> dist(p.begin(), p.begin() + last);
    return static_cast<int>(order[dist(rng)]);
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

/* Interactive chat: first turn through the prefill, every later token through the
   incremental step, continuous tokenization across turns. */
static int chat_loop(runtime_transformer& rt, hf_tokenizer& tok, const chat_template_formatter& fmt,
    long ctx_len, double temperature, size_t top_k, float top_p, float min_p,
    float repeat_penalty, bool deterministic, const std::string& system_prompt,
    bool trace_prompt = false)
{
    const int eos = tok.eos_id();
    const std::string stop = fmt.stop_string();
    std::mt19937 rng(std::random_device{}());

    /* Pin the immutable BOS + system prefix across cache evictions. */
    const std::string prefix = fmt.system_prefix(system_prompt);
    long keep = fmt.add_bos_on_first_turn() ? 1 : 0;
    if (!prefix.empty())
        keep += static_cast<long>(tok.encode(prefix, false, false, true, false).size());
    rt.set_context(ctx_len, keep);

    cout << "Chat template: " << chat_template_formatter::name(fmt.kind()) << "\n";
    cout << "Ready. Type 'quit' or 'exit' to stop.\n";

    bool first = true;
    std::vector<int> recent;
    resizable_tensor last_logits;   // last-row copy of the prefill logits
    for (;;)
    {
        cout << "You: " << std::flush;
        std::string user;
        if (!std::getline(std::cin, user)) break;
        if (user == "quit" || user == "exit") break;
        if (user.empty()) continue;

        const std::string turn = first ? fmt.first_turn(system_prompt, user) : fmt.next_turn(user);
        std::vector<int> toks = tok.encode(turn, first && fmt.add_bos_on_first_turn(), false, true, false);
        if (trace_prompt)
        {
            std::cout << "---- chat turn (" << toks.size() << " tokens) ----\n"
                      << tok.decode(toks, false) << "\n---- ids:";
            for (int id : toks) std::cout << ' ' << id;
            std::cout << "\n---- sampling: temp " << (deterministic ? 0.0 : temperature)
                      << (deterministic ? " (greedy)" : "")
                      << ", top_k " << top_k << ", top_p " << top_p
                      << ", min_p " << min_p << ", repeat " << repeat_penalty
                      << " ----" << std::endl;
        }

        const tensor* logits = nullptr;
        if (first)
        {
            logits = &rt.forward_prefill(toks);
            /* The prefill returns every position; sampling reads the last row. */
            alias_tensor last_row(1, 1, 1, logits->nc());
            last_logits.set_size(1, 1, 1, logits->nc());
            auto lr = last_row(const_cast<tensor&>(*logits),
                static_cast<size_t>((logits->nr() - 1)) * logits->nc());
            memcpy(last_logits, lr);
            logits = &last_logits;
            first = false;
        }
        else
        {
            for (int t : toks) logits = &rt.step(t);
        }

        /* Token-by-token streaming, in two regimes. Text destined to disappear (the
           reasoning span, template residues) is shown as a one-line ticker rewritten
           in place with the tail of the trace: it never wraps nor scrolls, so it can
           always be erased, unlike a multi-row region whose top may leave the screen,
           beyond the reach of any cursor sequence. The visible answer, obtained by
           cleaning the partial decode at every step (an open reasoning span cleans to
           its start), streams normally by stable suffix and persists. The whole
           sequence is re-decoded at each step because SentencePiece carries the
           inter-word spaces in the piece markers. */
        std::vector<int> out;
        std::string answer, shown_clean;
        const long tick_width = std::max<long>(20, terminal_width() - 12);
        bool ticker_active = false;
        cout << "Model: " << std::flush;
        for (;;)
        {
            const int next = pick_next(*logits, recent, temperature, top_k, top_p,
                min_p, repeat_penalty, deterministic, rng);
            if (next == eos) break;
            out.push_back(next);
            answer = tok.decode(out, true);
            recent.push_back(next);
            if (recent.size() > 64) recent.erase(recent.begin());
            bool stopped = false;
            if (!stop.empty() && answer.find(stop) != std::string::npos)
            {
                answer.erase(answer.find(stop));
                stopped = true;
            }
            std::string clean = fmt.clean_answer(answer);
            if (!stopped && !stop.empty())
            {
                /* Hold back the longest tail that is a prefix of the stop marker:
                   it may complete into it at the next token, and a terminal cannot
                   unprint. The held characters are flushed by the epilogue when
                   they turn out to be genuine answer text. */
                const size_t maxh = std::min(clean.size(), stop.size() - 1);
                for (size_t h = maxh; h > 0; --h)
                    if (clean.compare(clean.size() - h, h, stop, 0, h) == 0)
                    {
                        clean.erase(clean.size() - h);
                        break;
                    }
            }
            if (!stopped)
            {
                /* Likewise, hold back trailing whitespace: it prints later, once
                   followed by visible text, and the final clean_answer() trims it
                   when it turns out to be terminal, so printing it early would
                   leave the display ahead of the definitive answer. */
                while (!clean.empty() && (clean.back() == '\n' || clean.back() == '\r'
                    || clean.back() == ' ' || clean.back() == '\t'))
                    clean.pop_back();
            }
            if (clean.size() > shown_clean.size()
                && clean.compare(0, shown_clean.size(), shown_clean) == 0)
            {
                if (ticker_active)
                {
                    cout << "\r\033[KModel: ";
                    ticker_active = false;
                }
                cout << clean.substr(shown_clean.size()) << std::flush;
                shown_clean = clean;
            }
            else if (clean == shown_clean && shown_clean.empty())
            {
                /* Reasoning in progress: rolling tail of the hidden trace, dimmed,
                   newlines flattened, on the single rewritten line. */
                std::string tail = answer.size() > static_cast<size_t>(tick_width)
                    ? answer.substr(answer.size() - tick_width) : answer;
                for (char& c : tail) if (c == '\n' || c == '\r') c = ' ';
                cout << "\r\033[KModel: \033[2m" << tail << "\033[0m" << std::flush;
                ticker_active = true;
            }
            if (stopped) break;
            logits = &rt.step(next);
        }
        /* Epilogue: settle the display on the final cleaned answer. */
        const std::string clean = fmt.clean_answer(answer);
        if (ticker_active)
            cout << "\r\033[KModel: ";
        if (clean.size() > shown_clean.size()
            && clean.compare(0, shown_clean.size(), shown_clean) == 0)
            cout << clean.substr(shown_clean.size());
        else if (clean != shown_clean
            && !(clean.size() <= shown_clean.size()
                 && shown_clean.compare(0, clean.size(), clean) == 0))
            /* Only a genuine divergence reprints; the display merely being ahead
               by held-back or trimmed trailing characters is left as is. */
            cout << (shown_clean.empty() ? "" : "\nModel: ") << clean;
        cout << "\n";
        rt.step(eos);   // close the assistant turn in the token stream
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

/* OpenAI-compatible service over the runtime engine: each request carries the whole
   conversation; the token stream is rebuilt exactly as the interactive loop would
   have produced it turn by turn (system + first user turn, then per exchange the
   assistant text, the eos that closed it, and the next user turn), so serving and
   interactive chat share one numeric path. The server is stateless and server_chat
   serializes the calls, matching the engine's single-generation-thread assumption. */
class runtime_chat_server : public dlib::server_chat
{
public:
    runtime_chat_server(std::vector<served_model> models, long ctx,
        double forced_temp, bool temp_forced, bool deterministic, bool trace_prompt)
        : models_(std::move(models)), ctx_(ctx), temp_(forced_temp),
          temp_forced_(temp_forced), det_(deterministic),
          trace_prompt_(trace_prompt), rng_(std::random_device{}())
    {
        std::vector<dlib::chat_model_info> infos;
        for (const served_model& m : models_)
            infos.push_back(dlib::chat_model_info{ m.name, m.fmt.supports_reasoning() });
        set_models(infos);
    }

private:
    dlib::chat_result on_chat_completion(const dlib::chat_request& req,
        const std::function<void(const std::string&)>& emit) override
    {
        served_model* selp = &models_.front();   // first declared model is the default
        for (served_model& m : models_)
            if (m.name == req.model) { selp = &m; break; }
        served_model& use = *selp;
        runtime_transformer& rt_ = *use.rt;
        hf_tokenizer& tok_ = *use.tok;
        chat_template_formatter fmt_ = use.fmt;
        if (req.reasoning >= 0 && fmt_.supports_reasoning())
            fmt_.set_reasoning(req.reasoning == 1);

        /* Split the wire messages: system content joins the system block, then the
           user/assistant alternation drives the template turns. */
        std::string sys;
        std::vector<int> ids;
        bool first_user_seen = false;
        std::string pending_assistant;
        for (const dlib::chat_message& m : req.messages)
        {
            if (m.role == "system")
            {
                if (!sys.empty()) sys += "\n";
                sys += m.content;
                continue;
            }
            if (m.role == "assistant")
            {
                if (first_user_seen) pending_assistant = m.content;
                continue;
            }
            if (m.role != "user") continue;
            std::string user_text = m.content;
            for (size_t k = 0; k < m.image_urls.size(); ++k)
                user_text += "\n[attached image: not visible to this text-only model]";
            if (!first_user_seen)
            {
                /* Encoded exactly as the interactive loop encodes its turns: in
                   particular no space prefix, which would malform the role header
                   right after the opening special token. */
                const std::vector<int> t = tok_.encode(fmt_.first_turn(sys, user_text),
                    fmt_.add_bos_on_first_turn(), false, true, false);
                ids.insert(ids.end(), t.begin(), t.end());
                first_user_seen = true;
            }
            else
            {
                if (!pending_assistant.empty())
                {
                    const std::vector<int> a = tok_.encode(pending_assistant,
                        false, false, false, false);
                    ids.insert(ids.end(), a.begin(), a.end());
                    if (tok_.eos_id() >= 0) ids.push_back(tok_.eos_id());
                    pending_assistant.clear();
                }
                const std::vector<int> t = tok_.encode(fmt_.next_turn(user_text),
                    false, false, true, false);
                ids.insert(ids.end(), t.begin(), t.end());
            }
        }
        if (!first_user_seen)
            throw std::runtime_error("the conversation contains no user message");

        const long max_new = req.max_tokens > 0 ? req.max_tokens : 512;
        const long need = static_cast<long>(ids.size()) + max_new + 8;
        if (static_cast<long>(ids.size()) + 8 >= ctx_)
            throw std::runtime_error("prompt exceeds the context capacity; reduce the "
                "context budget in the interface settings");

        /* Sampling parameters resolve per request against the target model's own
           presets (each template family carries its recommended defaults), never
           against another served model's. */
        const double temp = req.temperature >= 0.0 ? req.temperature
            : (temp_forced_ ? temp_ : fmt_.default_temperature());
        const float top_p = req.top_p >= 0.0 ? static_cast<float>(req.top_p)
            : fmt_.default_top_p();
        const size_t top_k = req.top_k >= 0 ? static_cast<size_t>(req.top_k)
            : fmt_.default_top_k();
        const float min_p = req.min_p >= 0.0 ? static_cast<float>(req.min_p)
            : fmt_.default_min_p();
        const float repeat = req.repeat_penalty >= 0.0
            ? static_cast<float>(req.repeat_penalty) : fmt_.default_repeat_penalty();
        const bool greedy = det_ || temp <= 0.0;

        if (trace_prompt_)
        {
            std::cout << "---- prompt to " << use.name << " (" << ids.size()
                      << " tokens) ----\n" << tok_.decode(ids, false)
                      << "\n---- ids:";
            for (int id : ids) std::cout << ' ' << id;
            std::cout << "\n---- sampling: temp " << temp << (greedy ? " (greedy)" : "")
                      << ", top_k " << top_k << ", top_p " << top_p
                      << ", min_p " << min_p << ", repeat " << repeat
                      << " ----" << std::endl;
        }

        rt_.set_context(std::min(need, ctx_));
        const tensor* logits = &rt_.forward_prefill(ids);

        dlib::chat_result res;
        res.prompt_tokens = static_cast<long>(ids.size());
        /* Token pieces go out as produced; the server's streaming path buffers any
           incomplete UTF-8 tail, so multi-byte characters split across byte-fallback
           tokens still reach the client whole. */
        std::vector<int> answer, recent;
        const std::string stop = fmt_.stop_string();
        std::string text;
        for (long n = 0; n < max_new; ++n)
        {
            if (dlib::signal_handler::is_triggered()
                || (req.is_cancelled && req.is_cancelled()))
                break;
            const int next = pick_next(*logits, recent, greedy ? 1.0 : temp, top_k,
                top_p, min_p, repeat, greedy, rng_);
            if (next == tok_.eos_id()) break;
            if (rt_.cache_length() + 1 >= ctx_) { res.finish_reason = "length"; break; }
            answer.push_back(next);
            emit(tok_.decode({ next }));
            recent.push_back(next);
            if (recent.size() > 64) recent.erase(recent.begin());
            if (!stop.empty())
            {
                text = tok_.decode(answer);
                const size_t b = text.find(stop);
                if (b != std::string::npos) { text.erase(b); break; }
            }
            logits = &rt_.step(next);
        }
        if (static_cast<long>(answer.size()) >= max_new) res.finish_reason = "length";
        if (stop.empty() || text.empty()) text = tok_.decode(answer);
        res.completion_tokens = static_cast<long>(answer.size());
        res.content = fmt_.clean_answer(text);
        return res;
    }

    std::vector<served_model> models_;
    bool trace_prompt_ = false;
    long ctx_;
    double temp_;
    bool temp_forced_;
    bool det_;
    std::mt19937 rng_;
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
        parser.add_option("trace-prompt", "In serve mode, print the rebuilt prompt of every request (specials included)");
        parser.add_option("system", "System prompt for chat mode", 1);
        parser.add_option("context", "KV cache capacity in tokens (default: 2048)", 1);
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
            const double temp = parser.option("temp")
                ? std::stod(parser.option("temp").argument()) : fmt.default_temperature();
            const long ctx = parser.option("context")
                ? std::stol(parser.option("context").argument()) : 2048;
            const std::string sys = parser.option("system")
                ? parser.option("system").argument()
                : std::string("You are a helpful assistant.");
            return chat_loop(rt, tok, fmt, ctx, temp, fmt.default_top_k(), fmt.default_top_p(),
                fmt.default_min_p(), fmt.default_repeat_penalty(),
                parser.option("deterministic"), sys, parser.option("trace-prompt"));
        }

        if (parser.option("serve"))
        {
            chat_template_formatter fmt = parser.option("template")
                ? chat_template_formatter::for_tokenizer(tok,
                    chat_template_formatter::from_name(parser.option("template").argument()))
                : chat_template_formatter::for_tokenizer(tok, rt.spec().model_name);
            if (parser.option("think") && fmt.supports_reasoning()) fmt.set_reasoning(true);
            const long ctx = parser.option("context")
                ? std::stol(parser.option("context").argument()) : 2048;
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
            std::vector<float> ref(full.host() + (full.nr() - 1) * V,
                                   full.host() + full.nr() * V);

            rt.set_context(static_cast<long>(toks.size()) + 8);
            std::vector<int> head(toks.begin(), toks.end() - 1);
            rt.forward_prefill(head);
            const tensor& inc = rt.step(toks.back());

            const float* a = inc.host();
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
