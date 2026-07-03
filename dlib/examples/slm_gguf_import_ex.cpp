/*!
    @file slm_gguf_import_ex.cpp
    @brief Import a GGUF model (Llama 2 first) into the Dlib transformer stack.

    This example runs the front of the import pipeline:
      stage 0  read the GGUF container (gguf_reader)
      stage 1  detect the architecture into a neutral model_spec
      stage 2  check compatibility against the available Dlib layers
      stage 3  emit a Dlib model header for the detected model
      stage 4  extract the tokenizer, round-trip test it, serialize it

    Weight dequantization and repacking is the next increment.

    Usage:
      slm_gguf_import_ex --input model.gguf --out-prefix imported_model
      slm_gguf_import_ex --input model.gguf --out-prefix m --probe "Hello, world!"

    Two-phase build (resolves the chicken-and-egg of needing the generated header to
    compile the model-using code):
      Phase 1  compile WITHOUT WITH_IMPORTED_MODEL. No model header is included and no
               network is instantiated, so the file always compiles. Run it to detect the
               model, emit its header (use --out-prefix imported_model so the file is named
               imported_model.h and declares namespace imported_model) and extract the
               tokenizer.
      Phase 2  recompile WITH WITH_IMPORTED_MODEL defined. The generated header is included
               and --chat / --probe-logits / --convert become available. Enable it at configure
               time through the CMake option, for example from dlib/examples:
                 cmake -S . -B build -DWITH_IMPORTED_MODEL=ON
                 cmake --build build --config Release
               A deep network type is instantiated, so this phase needs /bigobj on MSVC (added
               by the example CMakeLists), and imported_model.h must sit next to this file or on
               the include path.

    The utility headers gguf_reader.h, gguf_dequantize.h, gguf_model_spec.h and
    gguf_weight_loader.h live under dlib/data_io; hf_tokenizer.h lives under dlib/tokenizer.
!*/

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <sstream>

#include <dlib/cmd_line_parser.h>
#include <dlib/data_io/gguf_reader.h>
#include <dlib/data_io/gguf_dequantize.h>
#include <dlib/data_io/gguf_model_spec.h>
#include <dlib/tokenizer/hf_tokenizer.h>

#define WITH_IMPORTED_MODEL 1

#ifdef WITH_IMPORTED_MODEL
#  include <random>
#  include <ctime>
#  include <dlib/dnn.h>
#  include <dlib/data_io/gguf_weight_loader.h>
#  ifndef IMPORTED_MODEL_HEADER
#    define IMPORTED_MODEL_HEADER "imported_model.h"
#  endif
#  include IMPORTED_MODEL_HEADER
#endif

using namespace std;
using namespace dlib;

/* Extract the tokenizer from the GGUF metadata, run a round-trip sanity check, and
   serialize it. The round-trip is the cheap local validation: encode then decode a few
   strings and confirm the text is recovered. For exact parity, compare the token ids
   against llama.cpp on the same strings. */
void extract_tokenizer(const gguf_reader& g, const string& out_path, const string& probe)
{
    hf_tokenizer tok;
    tok.load_from_gguf(g);

    cout << "Tokenizer family    : "
         << (tok.type() == hf_tokenizer::kind::spm ? "SentencePiece" : "byte-level BPE") << "\n"
         << "Vocab size          : " << tok.size() << "\n"
         << "Special ids         : bos=" << tok.bos_id() << " eos=" << tok.eos_id()
         << " unk=" << tok.unk_id() << " pad=" << tok.pad_id() << "\n";

    std::vector<string> samples = {
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "def add(a, b):\n    return a + b"
    };
    if (!probe.empty()) samples.insert(samples.begin(), probe);

    cout << "\nRound-trip check:\n";
    bool all_ok = true;
    for (const string& s : samples)
    {
        const std::vector<int> ids = tok.encode(s, /*add_bos=*/true, /*add_eos=*/false);
        const string back = tok.decode(ids, /*skip_special=*/true);
        const bool ok = (back == s);
        all_ok = all_ok && ok;
        cout << "  [" << (ok ? "ok " : "MISMATCH") << "] " << ids.size() << " tokens : \""
             << s << "\"\n";
        if (!ok) cout << "             decoded : \"" << back << "\"\n";
    }
    cout << (all_ok ? "Round-trip passed.\n" : "Round-trip MISMATCH (see above).\n");

    /* Show how the chat-template markers tokenize: a single id means a dedicated special
       token, several ids mean an ordinary subword sequence (the case for the standard
       Llama-2 vocabulary, which has no dedicated chat markers). */
    cout << "\nChat-template markers:\n";
    for (const string& m : { string("<|user|>"), string("<|assistant|>"), string("<|system|>") })
    {
        const std::vector<int> ids = tok.encode(m, /*add_bos=*/false, /*add_eos=*/false);
        cout << "  \"" << m << "\" -> " << ids.size() << (ids.size() == 1 ? " token  [" : " tokens [");
        for (size_t i = 0; i < ids.size(); ++i) cout << (i ? " " : "") << ids[i];
        cout << "]\n";
    }

    ofstream out(out_path, ios::binary);
    if (!out) throw runtime_error("cannot write " + out_path);
    serialize(tok, out);
    cout << "Tokenizer written   : " << out_path << "\n";
}

/* Validate the model weights against the detected architecture: confirm every expected
   tensor is present with the right element count, exercise the dequantizer on the real
   file, and report the total parameter count. This is the precondition for the repacking
   stage; it does not yet write the model. */
struct expected_tensor { string name; long long elems; };

void build_expected_tensors(const model_spec& s, std::vector<expected_tensor>& out)
{
    const long long d = s.d_model;
    const long long q = static_cast<long long>(s.n_heads) * s.head_dim;       // query projection width
    const long long kv = static_cast<long long>(s.n_kv_heads) * s.head_dim;   // key/value projection width
    const long long ff = s.d_ffn;

    out.push_back({ "token_embd.weight", static_cast<long long>(s.vocab_size) * d });
    out.push_back({ "output_norm.weight", d });
    if (!s.tied_embeddings) out.push_back({ "output.weight", static_cast<long long>(s.vocab_size) * d });

    for (long i = 0; i < s.n_layers; ++i)
    {
        const string p = "blk." + std::to_string(i) + ".";
        out.push_back({ p + "attn_norm.weight",   d });
        out.push_back({ p + "attn_q.weight",      d * q });
        out.push_back({ p + "attn_k.weight",      d * kv });
        out.push_back({ p + "attn_v.weight",      d * kv });
        out.push_back({ p + "attn_output.weight", q * d });
        if (s.qk_norm)
        {
            out.push_back({ p + "attn_q_norm.weight", s.head_dim });
            out.push_back({ p + "attn_k_norm.weight", s.head_dim });
        }
        out.push_back({ p + "ffn_norm.weight",    d });
        out.push_back({ p + "ffn_gate.weight",    d * ff });
        out.push_back({ p + "ffn_up.weight",      d * ff });
        out.push_back({ p + "ffn_down.weight",    ff * d });
    }
}

void validate_weights(gguf_reader& g, const model_spec& spec)
{
    std::vector<expected_tensor> expected;
    build_expected_tensors(spec, expected);

    size_t missing = 0, mismatched = 0;
    long long total_params = 0;
    for (const expected_tensor& e : expected)
    {
        const gguf_tensor_info* t = g.find_tensor(e.name);
        if (!t) { cout << "  MISSING : " << e.name << "\n"; ++missing; continue; }
        const long long got = static_cast<long long>(t->n_elements());
        if (got != e.elems)
        {
            cout << "  SHAPE   : " << e.name << " expected " << e.elems << " got " << got << "\n";
            ++mismatched;
        }
        total_params += got;
    }

    cout << "Tensors expected    : " << expected.size() << "\n"
         << "Missing             : " << missing << "\n"
         << "Shape mismatches    : " << mismatched << "\n"
         << "Total parameters    : " << total_params << "\n";

    /* Exercise the dequantizer on a representative weight and report basic statistics,
       a cheap sanity check that the values land in a plausible range. */
    const gguf_tensor_info* sample = g.find_tensor("blk.0.attn_q.weight");
    if (sample)
    {
        std::vector<float> w;
        gguf_read_dequantized(g, *sample, w);
        float mn = w.empty() ? 0.f : w[0], mx = mn;
        double sum = 0.0, sumsq = 0.0;
        for (float v : w) { mn = std::min(mn, v); mx = std::max(mx, v); sum += v; sumsq += double(v) * v; }
        const double mean = w.empty() ? 0.0 : sum / w.size();
        const double rms = w.empty() ? 0.0 : std::sqrt(sumsq / w.size());
        cout << "Dequantized sample  : blk.0.attn_q.weight (" << w.size() << " values)\n"
             << "  min " << mn << "  max " << mx << "  mean " << mean << "  rms " << rms << "\n";
    }

    if (missing == 0 && mismatched == 0)
        cout << "Weight inventory and shapes match the detected architecture.\n";
    else
        cout << "Weight inventory does not fully match; repacking should not proceed yet.\n";
}

#ifdef WITH_IMPORTED_MODEL

/* The chat and probe modes use the network type compiled in from the generated header, so
   the GGUF dimensions must match that header. */
bool model_matches_header(const model_spec& s)
{
    return s.vocab_size == imported_model::VOCAB_SIZE
        && s.n_layers == imported_model::NUM_LAYERS
        && s.n_heads == imported_model::NUM_HEADS
        && s.n_kv_heads == imported_model::NUM_KV_HEADS
        && s.d_model == imported_model::EMBEDDING_DIM;
}

using infer_net = imported_model::network_type<false>;
using generator_type = softmaxm<multiply<infer_net::subnet_type>>;

/* Greedy or nucleus pick over the probabilities at the last sequence position. */
int pick_next(const tensor& probs, const std::vector<int>& recent, bool deterministic,
    size_t top_k, float top_p, float min_p, float repeat_penalty, dlib::rand& rng)
{
    const long seq_len = probs.nr();
    const long V = probs.nc();
    const float* row = probs.host() + tensor_index(probs, 0, 0, seq_len - 1, 0);
    if (deterministic)
        return static_cast<int>(std::max_element(row, row + V) - row);

    std::vector<float> p(row, row + V);
    if (repeat_penalty > 1.0f)
    {
        const size_t span = std::min<size_t>(recent.size(), 64);
        for (size_t i = recent.size() - span; i < recent.size(); ++i)
            if (recent[i] >= 0 && recent[i] < V) p[recent[i]] /= repeat_penalty;
    }
    const float maxp = *std::max_element(p.begin(), p.end());
    std::vector<std::pair<int, float>> cand;
    for (long i = 0; i < V; ++i) if (p[i] >= maxp * min_p) cand.push_back({ static_cast<int>(i), p[i] });
    const size_t k = std::min(top_k, cand.size());
    std::partial_sort(cand.begin(), cand.begin() + k, cand.end(),
        [](const std::pair<int, float>& a, const std::pair<int, float>& b) { return a.second > b.second; });
    float cum = 0.0f; size_t cutoff = 0;
    for (size_t i = 0; i < k; ++i) { cum += cand[i].second; cutoff = i; if (cum >= top_p) break; }
    float total = 0.0f;
    for (size_t i = 0; i <= cutoff; ++i) total += cand[i].second;
    float r = rng.get_random_float() * total, cs = 0.0f;
    for (size_t i = 0; i <= cutoff; ++i) { cs += cand[i].second; if (r <= cs) return cand[i].first; }
    return cand.empty() ? 0 : cand[0].first;
}

int run_chat(gguf_reader& g, const model_spec& spec, const gguf_load_options& lopt,
    double temperature, size_t top_k, float top_p,
    float min_p, float repeat_penalty, bool deterministic, long ctx_len, bool use_template,
    const std::string& system_prompt)
{
    if (!model_matches_header(spec))
    { cerr << "Error: model does not match the compiled-in header. Regenerate and recompile.\n"; return 1; }

    infer_net net;
    cout << "Importing weights into the network...\n";
    import_gguf_weights(net, g, spec, lopt);

    hf_tokenizer tok;
    tok.load_from_gguf(g);
    const int eos = tok.eos_id();

    const float temp = deterministic ? 1.0f : static_cast<float>(temperature);
    generator_type generator(multiply_(1.0 / temp));
    generator.subnet().subnet() = net.subnet();

    network_context::reset();
    network_context::set_kv_cache_capacity(ctx_len);
    /* Clear the KV cache before the first prefill. The weight-import allocation pass
       leaves one dummy token in the attention caches; run_probe clears it the same way.
       Without this, the first turn runs on a polluted cache (shifted RoPE positions and a
       stale token seen by attention), which is what made the chat degenerate into
       repetition and spurious role markers. */
    network_context::request_kv_cache_clear();
    dlib::rand rng(std::time(nullptr));
    const int max_response = 512;

    cout << "\nReady. Type 'quit' or 'exit' to stop.\n\n";

    /* Validated KV-cache pattern: a single prefill on the first turn, then everything else,
       the response and every later turn's tokens, is fed one token at a time in incremental
       mode, never clearing or re-prefilling. This is the only path the cache is known to
       reproduce exactly; the attention layer slides its window automatically when the
       capacity is reached. */
    network_context::set_inference_mode(network_context::inference_mode::prefill);
    network_context::clear_padding();
    bool primed = false;
    std::vector<int> ctx;
    while (true)
    {
        cout << "You: " << std::flush;
        std::string line;
        if (!std::getline(std::cin, line)) break;
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        if (!line.empty()) line.erase(line.find_last_not_of(" \t\r\n") + 1);
        if (line.empty()) continue;
        if (line == "quit" || line == "exit") break;

        std::vector<int> turn;
        if (use_template)
        {
            /* TinyLlama / Zephyr chat template. The first turn carries the system block ahead
               of the user turn; later turns begin with the newline that follows the assistant's
               closing </s>, so the running token stream matches a single continuous tokenization
               of the whole conversation. </s> is parsed as the eos special token, and the
               SentencePiece dummy space prefix is applied to every fragment, including those
               after a special token, exactly as llama.cpp does. */
            const bool first = !primed;
            const std::string turn_text = first
                ? "<|system|>\n" + system_prompt + "</s>\n<|user|>\n" + line + "</s>\n<|assistant|>\n"
                : "\n<|user|>\n" + line + "</s>\n<|assistant|>\n";
            turn = tok.encode(turn_text, /*add_bos=*/first, /*add_eos=*/false,
                /*parse_special=*/true, /*allow_space_prefix=*/true);
        }
        else
        {
            turn = tok.encode(line, /*add_bos=*/!primed, /*add_eos=*/false);
        }

        int nxt = 0;
        if (!primed)
        {
            /* First turn: a single prefill over the whole turn. */
            matrix<int, 0, 1> pf(static_cast<long>(turn.size()), 1);
            for (long i = 0; i < static_cast<long>(turn.size()); ++i) pf(i) = turn[static_cast<size_t>(i)];
            const tensor& pr = generator(pf);
            nxt = pick_next(pr, ctx, deterministic, top_k, top_p, min_p, repeat_penalty, rng);
            /* The prefill consumed the clear request and every layer reset its cache. Reset the
               flag now: consume_kv_cache_clear_request does not clear it (so all layers in a pass
               see the same value), and if it stayed set, every incremental step below would wipe
               the cache, the model would lose all context, and the output would be blank. */
            network_context::clear_kv_cache_request();
            ctx.insert(ctx.end(), turn.begin(), turn.end());
            network_context::set_inference_mode(network_context::inference_mode::incremental);
            network_context::clear_padding();
            primed = true;
        }
        else
        {
            /* Later turns: feed the new tokens incrementally, continuing the same cache; the
               last one yields the first response token. */
            for (size_t j = 0; j < turn.size(); ++j)
            {
                matrix<int, 0, 1> step(1, 1);
                step(0) = turn[j];
                const tensor& out = generator(step);
                ctx.push_back(turn[j]);
                if (j + 1 == turn.size())
                    nxt = pick_next(out, ctx, deterministic, top_k, top_p, min_p, repeat_penalty, rng);
            }
        }

        cout << "Model: " << std::flush;
        std::vector<int> out_toks;
        size_t printed = 0;
        for (int i = 0; i < max_response; ++i)
        {
            if (nxt == eos) break;
            ctx.push_back(nxt);
            out_toks.push_back(nxt);
            std::string full = tok.decode(out_toks, true);
            if (full.size() > printed) { cout << full.substr(printed) << std::flush; printed = full.size(); }
            matrix<int, 0, 1> step(1, 1);
            step(0) = nxt;
            nxt = pick_next(generator(step), ctx, deterministic, top_k, top_p, min_p, repeat_penalty, rng);
        }
        if (use_template)
        {
            /* Close the assistant turn with </s> in the cache so the next turn continues
               cleanly; advance the cache past it without sampling. */
            matrix<int, 0, 1> step(1, 1);
            step(0) = eos;
            generator(step);
            ctx.push_back(eos);
        }
        cout << "\n\n";
    }
    network_context::reset();
    return 0;
}

/* Print the most probable next tokens for a prompt's last position. Compare these with a
   reference (for example llama.cpp) to validate the weight repacking. */
int run_probe(gguf_reader& g, const model_spec& spec, const gguf_load_options& lopt,
    const std::string& prompt)
{
    if (!model_matches_header(spec))
    { cerr << "Error: model does not match the compiled-in header. Regenerate and recompile.\n"; return 1; }

    infer_net net;
    cout << "Importing weights into the network...\n";
    import_gguf_weights(net, g, spec, lopt);

    hf_tokenizer tok;
    tok.load_from_gguf(g);
    generator_type generator(multiply_(1.0));
    generator.subnet().subnet() = net.subnet();

    std::vector<int> toks = tok.encode(prompt, /*add_bos=*/true, /*add_eos=*/false);
    matrix<int, 0, 1> in(static_cast<long>(toks.size()), 1);
    for (long i = 0; i < static_cast<long>(toks.size()); ++i) in(i) = toks[i];

    network_context::reset();
    network_context::set_kv_cache_capacity(static_cast<long>(toks.size()));
    network_context::request_kv_cache_clear();
    network_context::clear_padding();
    network_context::set_inference_mode(network_context::inference_mode::prefill);

    const tensor& probs = generator(in);
    const long seq_len = probs.nr();
    const long V = probs.nc();
    const float* row = probs.host() + tensor_index(probs, 0, 0, seq_len - 1, 0);

    std::vector<std::pair<int, float>> cand(V);
    for (long i = 0; i < V; ++i) cand[i] = { static_cast<int>(i), row[i] };
    std::partial_sort(cand.begin(), cand.begin() + 5, cand.end(),
        [](const std::pair<int, float>& a, const std::pair<int, float>& b) { return a.second > b.second; });

    cout << "\nPrompt (" << toks.size() << " tokens): \"" << prompt << "\"\n"
         << "Most probable next tokens:\n";
    for (int i = 0; i < 5; ++i)
    {
        std::vector<int> one{ cand[i].first };
        cout << "  " << cand[i].second << "  id " << cand[i].first
             << "  \"" << tok.decode(one, false) << "\"\n";
    }

    // Echo the token ids and the per-position argmax. Feeding the same ids to a reference
    // implementation localizes the first diverging position when validating an import.
    cout << "\nToken ids fed:";
    for (long i = 0; i < static_cast<long>(toks.size()); ++i) cout << " " << toks[i];
    cout << "\nPer-position argmax (pos: predicted_id 'tok' prob):\n";
    for (long p = 0; p < seq_len; ++p)
    {
        const float* rp = probs.host() + tensor_index(probs, 0, 0, p, 0);
        long am = 0; float mx = rp[0];
        for (long i = 1; i < V; ++i) if (rp[i] > mx) { mx = rp[i]; am = i; }
        std::vector<int> one{ static_cast<int>(am) };
        cout << "  " << p << ": " << am << " '" << tok.decode(one, false) << "' " << mx << "\n";
    }

    network_context::reset();
    return 0;
}

/* Probe on an explicit token-id sequence, bypassing the tokenizer. This isolates forward-pass
   behavior for a chosen sequence, for instance to measure the effect of a control token such
   as the eos </s> in the middle of a prompt. The ids must already include BOS if wanted. */
int run_probe_ids(gguf_reader& g, const model_spec& spec, const gguf_load_options& lopt,
    const std::string& id_string)
{
    if (!model_matches_header(spec))
    { cerr << "Error: model does not match the compiled-in header. Regenerate and recompile.\n"; return 1; }

    infer_net net;
    cout << "Importing weights into the network...\n";
    import_gguf_weights(net, g, spec, lopt);

    hf_tokenizer tok;
    tok.load_from_gguf(g);
    generator_type generator(multiply_(1.0));
    generator.subnet().subnet() = net.subnet();

    std::vector<int> toks;
    {
        std::string s = id_string;
        for (char& c : s) if (c == ',') c = ' ';
        std::istringstream iss(s);
        long v;
        while (iss >> v) toks.push_back(static_cast<int>(v));
    }
    if (toks.empty()) { cerr << "Error: --probe-ids received no ids.\n"; return 1; }

    matrix<int, 0, 1> in(static_cast<long>(toks.size()), 1);
    for (long i = 0; i < static_cast<long>(toks.size()); ++i) in(i) = toks[static_cast<size_t>(i)];

    network_context::reset();
    network_context::set_kv_cache_capacity(static_cast<long>(toks.size()));
    network_context::request_kv_cache_clear();
    network_context::clear_padding();
    network_context::set_inference_mode(network_context::inference_mode::prefill);

    const tensor& probs = generator(in);
    const long seq_len = probs.nr();
    const long V = probs.nc();
    const float* row = probs.host() + tensor_index(probs, 0, 0, seq_len - 1, 0);

    std::vector<std::pair<int, float>> cand(static_cast<size_t>(V));
    for (long i = 0; i < V; ++i) cand[static_cast<size_t>(i)] = { static_cast<int>(i), row[i] };
    std::partial_sort(cand.begin(), cand.begin() + 8, cand.end(),
        [](const std::pair<int, float>& a, const std::pair<int, float>& b) { return a.second > b.second; });

    cout << "\nFed " << toks.size() << " explicit token ids (seq=" << seq_len << ").\n"
         << "Most probable next tokens:\n";
    for (int i = 0; i < 8; ++i)
    {
        std::vector<int> one{ cand[static_cast<size_t>(i)].first };
        cout << "  " << cand[static_cast<size_t>(i)].second << "  id " << cand[static_cast<size_t>(i)].first
             << "  \"" << tok.decode(one, false) << "\"\n";
    }

    // Per-position argmax. In a causal model the prediction at position p depends only on
    // tokens [0, p], so comparing this dump against a reference implementation fed with the
    // exact same ids localizes the first diverging position when validating an import.
    cout << "\nPer-position argmax (pos: predicted_id 'tok' prob):\n";
    for (long p = 0; p < seq_len; ++p)
    {
        const float* rp = probs.host() + tensor_index(probs, 0, 0, p, 0);
        long am = 0; float mx = rp[0];
        for (long i = 1; i < V; ++i) if (rp[i] > mx) { mx = rp[i]; am = i; }
        std::vector<int> one{ static_cast<int>(am) };
        cout << "  " << p << ": " << am << " '" << tok.decode(one, false) << "' " << mx << "\n";
    }

    network_context::reset();
    return 0;
}

/* Load the weights and serialize the model (and tokenizer) to a dlib .dat. This needs the
   whole network resident in memory: on a CUDA build that is GPU memory (so a model larger
   than VRAM requires a CPU build, which uses system RAM instead). */
int run_convert(gguf_reader& g, const model_spec& spec, const gguf_load_options& lopt,
    const std::string& out_path)
{
    if (!model_matches_header(spec))
    { cerr << "Error: model does not match the compiled-in header. Regenerate and recompile.\n"; return 1; }

    infer_net net;
    cout << "Importing weights into the network...\n";
    import_gguf_weights(net, g, spec, lopt);

    hf_tokenizer tok;
    tok.load_from_gguf(g);

    cout << "Serializing model to " << out_path << " ...\n";
    serialize(out_path) << net << tok;
    cout << "Done. Wrote " << out_path << "\n";
    return 0;
}

#endif // WITH_IMPORTED_MODEL

int main(int argc, char** argv)
{
    try
    {
        command_line_parser parser;
        parser.add_option("input", "Path to the source .gguf model", 1);
        parser.add_option("out-prefix", "Output prefix for generated files (default: imported_model)", 1);
        parser.add_option("probe", "Extra string to round-trip through the tokenizer", 1);
        parser.add_option("chat", "Load the model and start an interactive completion session");
        parser.add_option("convert", "Load the model and serialize it to <out-prefix>.dat");
        parser.add_option("probe-logits", "Print the most probable next tokens for --prompt (weight validation)");
        parser.add_option("prompt", "Prompt used by --probe-logits (default: 'The capital of France is')", 1);
        parser.add_option("probe-ids", "Print next-token predictions for an explicit space-separated id list", 1);
        parser.add_option("context", "KV cache length for --chat (default: 512)", 1);
        parser.add_option("temperature", "Sampling temperature (default: 0.8)", 1);
        parser.add_option("top-k", "Top-k filter (default: 40)", 1);
        parser.add_option("top-p", "Nucleus threshold (default: 0.9)", 1);
        parser.add_option("min-p", "Relative min-p threshold (default: 0.05)", 1);
        parser.add_option("repeat-penalty", "Repetition penalty (default: 1.1)", 1);
        parser.add_option("deterministic", "Greedy decoding (argmax)");
        parser.add_option("raw", "Chat without the chat template (raw text completion)");
        parser.add_option("system", "System prompt used by --chat (default: a helpful assistant)", 1);
        parser.add_option("rope-permute", "Permute Q/K rows from split-half (NeoX) to interleaved RoPE ordering; leave off for llama-family GGUFs, expected for NeoX-convention architectures");
        parser.add_option("swap-gate-up", "Swap ffn_gate / ffn_up assignment (weight-loader knob)");
        parser.parse(argc, argv);

        if (!parser.option("input"))
        {
            cout << "Import a GGUF model into the Dlib transformer stack.\n\n";
            parser.print_options();
            cout << "\nExamples:\n"
                 << "  Phase 1 (generate header + tokenizer, any build):\n    " << argv[0]
                 << " --input tinyllama-1.1b-chat-v1.0.Q8_0.gguf --out-prefix imported_model\n"
                 << "  Phase 2 (built with WITH_IMPORTED_MODEL):\n    " << argv[0]
                 << " --input tinyllama-1.1b-chat-v1.0.Q8_0.gguf --probe-logits --prompt \"The capital of France is\"\n    " << argv[0]
                 << " --input tinyllama-1.1b-chat-v1.0.Q8_0.gguf --chat\n";
            return 0;
        }

        const string input  = parser.option("input").argument();
        const string prefix = get_option(parser, "out-prefix", string("imported_model"));

        cout << "Reading GGUF: " << input << "\n";
        gguf_reader g(input);
        cout << "GGUF version " << g.version() << ", "
             << g.metadata().size() << " metadata keys, "
             << g.tensors().size() << " tensors\n\n";

        const model_spec spec = detect_model(g);
        cout << describe(spec) << "\n";

        const compat_result compat = check_compatibility(spec);
        for (const auto& n : compat.notes)    cout << "note: "    << n << "\n";
        for (const auto& b : compat.blockers) cout << "BLOCKER: " << b << "\n";
        if (!compat.ok)
        {
            cout << "\nModel not yet importable with the current layers. Stopping.\n";
            return 1;
        }

        if (parser.option("chat") || parser.option("probe-logits") || parser.option("convert") || parser.option("probe-ids"))
        {
#ifdef WITH_IMPORTED_MODEL
            gguf_load_options lopt;
            lopt.rope_permute = parser.option("rope-permute");
            lopt.swap_gate_up = parser.option("swap-gate-up");

            if (parser.option("chat"))
                return run_chat(g, spec, lopt,
                    get_option(parser, "temperature", 0.8),
                    get_option(parser, "top-k", size_t(40)),
                    get_option(parser, "top-p", 0.9f),
                    get_option(parser, "min-p", 0.05f),
                    get_option(parser, "repeat-penalty", 1.1f),
                    parser.option("deterministic"),
                    get_option(parser, "context", long(512)),
                    /*use_template=*/!parser.option("raw"),
                    get_option(parser, "system", std::string("You are a helpful assistant.")));

            if (parser.option("convert"))
                return run_convert(g, spec, lopt, prefix + ".dat");

            if (parser.option("probe-ids"))
                return run_probe_ids(g, spec, lopt, parser.option("probe-ids").argument());

            return run_probe(g, spec, lopt,
                get_option(parser, "prompt", std::string("The capital of France is")));
#else
            cerr << "This build has no model header compiled in.\n"
                 << "Generate it first (run with --out-prefix imported_model), then reconfigure\n"
                 << "with the CMake option enabled and rebuild:\n"
                 << "    cmake -S . -B build -DWITH_IMPORTED_MODEL=ON\n"
                 << "    cmake --build build --config Release\n"
                 << "(/bigobj is added by the example CMakeLists on MSVC).\n";
            return 1;
#endif
        }

        const string header_path = prefix + ".h";
        emit_header(spec, header_path);
        cout << "\nGenerated model header: " << header_path << "\n\n";

        const string probe = parser.option("probe") ? parser.option("probe").argument() : "";
        extract_tokenizer(g, prefix + "_tokenizer.dat", probe);

        cout << "\nValidating weights:\n";
        validate_weights(g, spec);

        cout << "\nNext increment:\n"
             << "  - repack into the model network and serialize -> " << prefix << ".dat\n";
        return 0;
    }
    catch (exception& e)
    {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
}
