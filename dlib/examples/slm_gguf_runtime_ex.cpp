/*!
    @file slm_gguf_runtime_ex.cpp
    @brief Dynamic GGUF inference: load any supported model without recompiling.

    Counterpart of slm_gguf_import_ex for the runtime engine: the architecture is
    resolved at load time from the GGUF metadata, so this program compiles once and
    probes any supported model. It shares the validation protocol of the static path,
    same options, same output format, so the per-position dumps of the two engines
    and of the llama.cpp reference are directly comparable.

    Usage:
      slm_gguf_runtime_ex --input model.gguf --probe-logits --prompt "The capital of France is"
      slm_gguf_runtime_ex --input model.gguf --probe-ids "1 450 7483 310 3444 338"
!*/

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#endif

#include <dlib/cmd_line_parser.h>
#include <dlib/dnn/runtime_transformer.h>
#include <dlib/tokenizer/hf_tokenizer.h>

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

int main(int argc, char** argv)
{
    try
    {
#ifdef _WIN32
        SetConsoleOutputCP(CP_UTF8);
        SetConsoleCP(CP_UTF8);
#endif
        command_line_parser parser;
        parser.add_option("input", "Path to the source .gguf model", 1);
        parser.add_option("probe-logits", "Run a prefill and report the logits");
        parser.add_option("probe-ids", "Feed explicit token ids (space or comma separated)", 1);
        parser.add_option("prompt", "Prompt for --probe-logits (default: capital of France)", 1);
        parser.add_option("rope-permute", "Map split-half (NeoX) Q/K layouts to the interleaved kernel");
        parser.parse(argc, argv);

        if (!parser.option("input"))
        {
            cout << "Dynamic GGUF inference: load any supported model without recompiling.\n\n";
            parser.print_options();
            return 0;
        }

        const string input = parser.option("input").argument();
        cout << "Reading GGUF: " << input << "\n";
        gguf_reader g(input);
        const model_spec spec = detect_model(g);
        cout << describe(spec) << "\n";

        const compat_result rep = check_compatibility(spec);
        for (const string& n : rep.notes) cout << "note: " << n << "\n";
        if (!rep.blockers.empty())
        {
            for (const string& b : rep.blockers) cerr << "blocker: " << b << "\n";
            return 1;
        }

        gguf_load_options lopt;
        lopt.rope_permute = parser.option("rope-permute");
        lopt.verbose = true;

        runtime_transformer rt;
        cout << "Loading weights (runtime engine)...\n";
        rt.load(g, spec, lopt);

        hf_tokenizer tok;
        tok.load_from_gguf(g);

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
