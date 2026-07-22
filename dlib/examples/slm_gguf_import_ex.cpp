/*!
    @file slm_gguf_import_ex.cpp
    @brief Import a GGUF open-weight model into the Dlib transformer stack.

    The program covers the whole import pipeline:
      stage 0  read the GGUF container (gguf_reader)
      stage 1  detect the architecture into a neutral model_spec
      stage 2  check compatibility against the available Dlib layers
      stage 3  emit a Dlib model header for the detected model
      stage 4  extract the tokenizer, round-trip test it, serialize it
      stage 5  dequantize and repack the weights into the network, then probe the
               logits, convert to a self-contained .dat, or chat

    Usage:
      slm_gguf_import_ex --input model.gguf --out-prefix slm_imported_model
      slm_gguf_import_ex --input model.gguf --probe-logits --prompt "The capital of France is"
      slm_gguf_import_ex --input model.gguf --convert
      slm_gguf_import_ex --load model.dat --chat

    Two-phase build (resolves the chicken-and-egg of needing the generated header to
    compile the model-using code):
      Phase 1  no slm_imported_model.h exists yet: the model half is skipped by the
               __has_include detection, so the file always compiles. Run it to detect
               the model, emit its header (use --out-prefix slm_imported_model so the
               file is named slm_imported_model.h) and extract the tokenizer.
      Phase 2  rebuild the target: the generated header is now detected and included,
               and --chat / --probe-logits / --convert become available. A deep network
               type is instantiated, so this phase needs /bigobj on MSVC.

    The utility headers gguf_reader.h, gguf_dequantize.h, gguf_model_spec.h and
    gguf_weight_loader.h live under dlib/data_io; hf_tokenizer.h and chat_template.h
    live under dlib/tokenizer.
!*/

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <sstream>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#endif

#include <dlib/cmd_line_parser.h>
#include <dlib/data_io/gguf_reader.h>
#include <dlib/data_io/gguf_dequantize.h>
#include <dlib/data_io/gguf_model_spec.h>
#include <dlib/tokenizer/hf_tokenizer.h>
#include <dlib/tokenizer/chat_template.h>

/* The model-dependent half of this program (probes, conversion, chat) compiles only
   when the generated model header is present. __has_include acts as the build switch:
   phase 1, before any header exists, always compiles; once slm_imported_model.h has
   been generated next to this file (or anywhere on the include path), the next build
   of the target enables the model commands automatically. Note that a header created
   after a previous build is not tracked as a dependency of the old object file, so
   rebuild the target explicitly after the first generation. An external definition of
   WITH_IMPORTED_MODEL or IMPORTED_MODEL_HEADER keeps priority over the detection. */
#ifndef IMPORTED_MODEL_HEADER
#  define IMPORTED_MODEL_HEADER "slm_imported_model.h"
#endif
#if !defined(WITH_IMPORTED_MODEL) && defined(__has_include)
#  if __has_include(IMPORTED_MODEL_HEADER)
#    define WITH_IMPORTED_MODEL 1
#  endif
#endif

#ifdef WITH_IMPORTED_MODEL
#  include <random>
#  include <ctime>
#  include <dlib/dnn.h>
#  include <dlib/data_io/gguf_weight_loader.h>
#  include IMPORTED_MODEL_HEADER
#endif

using namespace std;
using namespace dlib;

/* Display and file identity of the imported model: the container's general.name run
   through the shared cleaner, which drops a redundant organization prefix and any
   quantization or container marker left in that field. model_spec::model_name keeps the
   raw value, so describe() still reports what the container actually declares. */
string model_display_name(const model_spec& s)
{
    const string cleaned = clean_model_name(s.model_name);
    return cleaned.empty() ? s.arch_name : cleaned;
}

/* Extract the tokenizer from the GGUF metadata, run a round-trip sanity check, and
   serialize it. The round-trip is the cheap local validation: encode then decode a few
   strings and confirm the text is recovered. For exact parity, compare the token ids
   against an external reference on the same strings. */
void extract_tokenizer(const gguf_reader& g, const string& out_path, const string& probe,
    const string& model_name)
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
    /* Report the conversation-format detection: the template the model declares when
       the container carries one, otherwise the eos-piece fallback, refined by the model
       name. The name matters for the families that leave no signature in the tokenizer:
       a Guanaco fine-tune inherits the declared template of its base model, so the name
       hint has to win over what the container claims. The marker tokenizations show how
       the detected family's delimiters map onto this vocabulary (single ids for genuine
       special tokens, several for plain text). */
    const chat_template_formatter fmt = chat_template_formatter::for_tokenizer(tok, model_name);
    cout << "\nChat template       : " << chat_template_formatter::name(fmt.kind())
         << (tok.chat_template().empty()
             ? " (fallback from the eos piece; none declared by the model)"
             : " (declared by the model)") << "\n";
    std::vector<string> markers;
    switch (fmt.kind())
    {
    case chat_template_kind::zephyr:
        markers = { "<|user|>", "<|assistant|>", "<|system|>", "</s>" };
        break;
    case chat_template_kind::chatml:
        markers = { "<|im_start|>", "<|im_end|>", "<think>" };
        break;
    case chat_template_kind::guanaco:
        markers = { "### Human:", "### Assistant:" };
        break;
    case chat_template_kind::granite:
        markers = { "<|start_of_role|>", "<|end_of_role|>", "<|end_of_text|>" };
        break;
    default:
        break;
    }
    if (!markers.empty())
    {
        cout << "Template markers:\n";
        for (const string& m : markers)
        {
            const std::vector<int> ids = tok.encode(m, /*add_bos=*/false, /*add_eos=*/false,
                /*parse_special=*/true, /*allow_space_prefix=*/false);
            cout << "  \"" << m << "\" -> " << ids.size() << (ids.size() == 1 ? " token  [" : " tokens [");
            for (size_t i = 0; i < ids.size(); ++i) cout << (i ? " " : "") << ids[i];
            cout << "]\n";
        }
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

    /* Exercise the dequantizer on tensors spread across the whole data section and
       report basic statistics. Weight values of a trained model land in a narrow,
       plausible range (rms typically 1e-2..1, |min|,|max| below ~30, mean near 0);
       a misread data offset corrupts every tensor past the drift point, so sampling
       the first, middle and last blocks plus the embedding and output tensors makes
       positional file-reading faults directly visible. */
    {
        const long last = spec.n_layers - 1;
        const long mid = spec.n_layers / 2;
        const std::string names[] = {
            "token_embd.weight",
            "blk.0.attn_q.weight",
            "blk." + std::to_string(mid) + ".attn_output.weight",
            "blk." + std::to_string(mid) + ".ffn_down.weight",
            "blk." + std::to_string(last) + ".ffn_up.weight",
            "blk." + std::to_string(last) + ".attn_v.weight",
            "output_norm.weight",
            spec.tied_embeddings ? std::string() : std::string("output.weight")
        };
        cout << "Dequantized samples (type, count, min, max, mean, rms):\n";
        for (const std::string& name : names)
        {
            if (name.empty()) continue;
            const gguf_tensor_info* t = g.find_tensor(name);
            if (!t) continue;
            std::vector<float> w;
            gguf_read_dequantized(g, *t, w);
            float mn = w.empty() ? 0.f : w[0], mx = mn;
            double sum = 0.0, sumsq = 0.0;
            bool finite = true;
            for (float v : w)
            {
                if (!std::isfinite(v)) finite = false;
                mn = std::min(mn, v); mx = std::max(mx, v);
                sum += v; sumsq += double(v) * v;
            }
            const double mean = w.empty() ? 0.0 : sum / w.size();
            const double rms = w.empty() ? 0.0 : std::sqrt(sumsq / w.size());
            cout << "  " << name << " : type " << static_cast<uint32_t>(t->type)
                 << ", " << w.size() << " values, min " << mn << ", max " << mx
                 << ", mean " << mean << ", rms " << rms
                 << (finite ? "" : "  [NON-FINITE VALUES]") << "\n";
        }
    }

    if (missing == 0 && mismatched == 0)
        cout << "Weight inventory and shapes match the detected architecture.\n";
    else
        cout << "Weight inventory does not fully match; repacking should not proceed yet.\n";
}

#ifdef WITH_IMPORTED_MODEL

/* The chat and probe modes use the network type compiled in from the generated header, so
   the GGUF geometry must match that header. Every shape the network type is built from is
   compared, not only the outer dimensions: a derivative sharing the layer count, the head
   geometry and the width but carrying a different feed-forward ratio or head dimension
   would otherwise pass the check and have its weights repacked into the wrong slots. */
bool model_matches_header(const model_spec& s)
{
    return s.vocab_size == imported_model::VOCAB_SIZE
        && s.n_layers == imported_model::NUM_LAYERS
        && s.n_heads == imported_model::NUM_HEADS
        && s.n_kv_heads == imported_model::NUM_KV_HEADS
        && s.d_model == imported_model::EMBEDDING_DIM
        && s.head_dim == imported_model::HEAD_DIM
        && s.qk_norm == imported_model::USE_QK_NORM
        && s.ffn_num == imported_model::FFN_NUM
        && s.ffn_den == imported_model::FFN_DEN;
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

/* Interactive chat loop over an already-loaded generator and tokenizer. The callers
   below provide the two loading paths: run_chat imports the weights from the GGUF,
   run_chat_dat reads back a previously converted .dat archive. The generator is loaded
   by the caller so that exactly one copy of the parameters is ever resident: the
   temperature scaling is applied here through the multiply layer's setter. */
int chat_loop(generator_type& generator, hf_tokenizer& tok,
    double temperature, size_t top_k, float top_p,
    float min_p, float repeat_penalty, bool deterministic, long ctx_len, bool use_template,
    const std::string& system_prompt, const std::string& template_override,
    const std::string& model_name, bool offload_params)
{
    const int eos = tok.eos_id();

    /* Model-aware conversation formatting. The family is detected from the chat template
       the model declares, falling back to the eos piece, and refined by the model name;
       the same logic covers the GGUF import path and the .dat loading path, which carries
       the name in its archive. The name hint is what identifies the families that leave no
       signature in the tokenizer: a Guanaco fine-tune inherits the declared template of
       its base model, so trusting the container alone selects the wrong family. The
       override still forces a family explicitly. Sampling values left unset on the command
       line fall back to the family's published presets. */
    const chat_template_formatter fmt = !use_template
        ? chat_template_formatter(chat_template_kind::raw)
        : (template_override.empty() || template_override == "auto")
            ? chat_template_formatter::for_tokenizer(tok, model_name)
            : chat_template_formatter::for_tokenizer(tok,
                  chat_template_formatter::from_name(template_override));
    if (use_template)
        cout << "Chat template: " << chat_template_formatter::name(fmt.kind()) << "\n";

    if (temperature < 0.0)     temperature    = fmt.default_temperature();
    if (top_k == 0)            top_k          = fmt.default_top_k();
    if (top_p < 0.0f)          top_p          = fmt.default_top_p();
    if (min_p < 0.0f)          min_p          = fmt.default_min_p();
    if (repeat_penalty < 0.0f) repeat_penalty = fmt.default_repeat_penalty();

    const float temp = deterministic ? 1.0f : static_cast<float>(temperature);
    layer<1>(generator).layer_details().set_multiply_value(1.0f / temp);

    network_context::reset();
    /* Host residency for the layer parameters (simulated unified memory): must be set
       after reset() and after the generator holds the weights, and before the first
       forward, so the capture happens on real weights during inference only. */
    if (offload_params)
        network_context::set_parameter_residency(network_context::parameter_residency::host_f32);
    network_context::set_kv_cache_capacity(ctx_len);
    /* Clear the KV cache before the first prefill. The weight-import allocation pass
       leaves one dummy token in the attention caches; run_probe clears it the same way.
       Without this, the first turn runs on a polluted cache (shifted RoPE positions and a
       stale token seen by attention), which is what made the chat degenerate into
       repetition and spurious role markers. */
    network_context::request_kv_cache_clear();

    /* Attention sinks: the conversation's immutable prefix (system block, plus BOS for
       the families that use one) is pinned in the KV cache and survives window
       evictions. Small decoder models concentrate a large share of their attention
       mass on the first positions; letting them slide out once the window is full
       collapses generation into repetitive output. The keep length is measured on the
       exact token prefix the first prefill produces. In raw mode only the BOS is
       pinned. */
    if (use_template)
    {
        const std::vector<int> sink = tok.encode(fmt.system_prefix(system_prompt),
            /*add_bos=*/fmt.add_bos_on_first_turn(), /*add_eos=*/false,
            /*parse_special=*/true, /*allow_space_prefix=*/true);
        network_context::set_kv_cache_keep_length(static_cast<long>(sink.size()));
    }
    else
    {
        network_context::set_kv_cache_keep_length(1);
    }

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
    /* Surface runtime failures explicitly: without this handler, a CUDA error thrown
       mid-generation unwinds through the tensor destructors, whose own failure logs
       (cudaFree / cudaStreamDestroy on a sticky-error device) flood the console and
       bury the primary message that names the failing call. */
    try
    {
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
            /* The turn strings are designed so the running token stream matches a single
               continuous tokenization of the whole conversation: the first turn carries
               the system block ahead of the user turn; later turns begin with the newline
               that follows the assistant's closing eos. Special markers are parsed as
               special tokens, and the SentencePiece dummy space prefix is applied to
               every fragment for the families that use it, exactly as the reference implementations do. */
            const bool first = !primed;
            const std::string turn_text = first
                ? fmt.first_turn(system_prompt, line)
                : fmt.next_turn(line);
            turn = tok.encode(turn_text, /*add_bos=*/first && fmt.add_bos_on_first_turn(),
                /*add_eos=*/false, /*parse_special=*/true, /*allow_space_prefix=*/true);
        }
        else
        {
            turn = tok.encode(line, /*add_bos=*/!primed, /*add_eos=*/false);
        }

        /* A single status line covers the whole turn: the (potentially long) prefill or
           delta feed, then the token-by-token generation. Tokens are not streamed; the
           complete answer replaces the indicator once generation finishes. */
        cout << "Model: thinking" << std::flush;

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

        std::vector<int> out_toks;
        const std::string stop = fmt.stop_string();
        for (int i = 0; i < max_response; ++i)
        {
            if (nxt == eos) break;
            ctx.push_back(nxt);
            out_toks.push_back(nxt);
            /* Some template families end a turn by starting the next one instead of
               emitting eos; stop as soon as the marker appears in the decoded answer.
               The marker tokens already fed remain in the KV cache; the eos closing
               below still seals the turn, and clean_answer trims the display. */
            if (!stop.empty() && tok.decode(out_toks, true).find(stop) != std::string::npos)
                break;
            static const char* const dots[] = { ".  ", ".. ", "..." };
            cout << "\rModel: thinking" << dots[(i / 8) % 3] << std::flush;
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

        /* Erase the indicator and print the complete answer in its place. */
        cout << "\r" << std::string(20, ' ') << "\r";
        cout << "Model: " << fmt.clean_answer(tok.decode(out_toks, true)) << "\n\n";
    }
    }
    catch (const std::exception& e)
    {
        /* Print the primary error first: for CUDA faults, e.what() carries the failing
           call with its file and line, which identifies the kernel or library call at
           the origin. The device is left in an undefined (sticky-error) state, so the
           destructor logs that follow are secondary noise. */
        cout << "\n";
        cerr << "\nFATAL during generation: " << e.what() << "\n"
             << "The CUDA device is now in an undefined state; restart the program.\n";
        network_context::reset();
        return 1;
    }
    network_context::reset();
    return 0;
}

/* Chat after importing the weights from the GGUF container. The weights are imported
   directly into the generator (the softmax/multiply head carries no parameters, so the
   layer visit order is identical), keeping a single resident copy of the model. */
int run_chat(gguf_reader& g, const model_spec& spec, const gguf_load_options& lopt,
    double temperature, size_t top_k, float top_p,
    float min_p, float repeat_penalty, bool deterministic, long ctx_len, bool use_template,
    const std::string& system_prompt, const std::string& template_override, bool offload_params)
{
    if (!model_matches_header(spec))
    { cerr << "Error: model does not match the compiled-in header. Regenerate and recompile.\n"; return 1; }

    generator_type generator(multiply_(1.0));
    cout << "Importing weights into the network...\n";
    import_gguf_weights(generator, g, spec, lopt);

    hf_tokenizer tok;
    tok.load_from_gguf(g);
    return chat_loop(generator, tok, temperature, top_k, top_p, min_p, repeat_penalty,
        deterministic, ctx_len, use_template, system_prompt, template_override,
        spec.model_name, offload_params);
}

/* Chat over a previously converted model: the parameters and the tokenizer are read
   back from the .dat archive written by --convert, skipping the GGUF import entirely.
   The archive carries the parameter subnet directly, deserialized straight
   into the generator: no temporary network exists at any point, so peak memory equals
   a single copy of the model. The archive must have been produced by a build compiled
   with the same model header. */
int run_chat_dat(const std::string& dat_path,
    double temperature, size_t top_k, float top_p,
    float min_p, float repeat_penalty, bool deterministic, long ctx_len, bool use_template,
    const std::string& system_prompt, const std::string& template_override, bool offload_params)
{
    generator_type generator(multiply_(1.0));
    hf_tokenizer tok;
    std::string name_hint;
    cout << "Loading converted model from " << dat_path << " ...\n";
    {
        std::ifstream fin(dat_path, std::ios::binary);
        if (!fin) { cerr << "Error: cannot open " << dat_path << "\n"; return 1; }
        std::string tag, model_name;
        deserialize(tag, fin);
        if (tag != "gguf_import_model")
        {
            cerr << "Error: '" << dat_path << "' is not a model archive produced by --convert; regenerate it.\n";
            return 1;
        }
        deserialize(model_name, fin);
        deserialize(generator.subnet().subnet(), fin);
        deserialize(tok, fin);
        /* Cleaned on read as well, so archives written before the identity was
           normalized at conversion time display and detect the same way. */
        model_name = clean_model_name(model_name);
        cout << "Model: " << model_name << "\n";
        name_hint = model_name;
    }
    return chat_loop(generator, tok, temperature, top_k, top_p, min_p, repeat_penalty,
        deterministic, ctx_len, use_template, system_prompt, template_override,
        name_hint, offload_params);
}

/* Print the most probable next tokens for a prompt's last position. Compare these with a
   reference (for example an external GGUF runtime) to validate the weight repacking. */
int run_probe(gguf_reader& g, const model_spec& spec, const gguf_load_options& lopt,
    const std::string& prompt)
{
    if (!model_matches_header(spec))
    { cerr << "Error: model does not match the compiled-in header. Regenerate and recompile.\n"; return 1; }

    generator_type generator(multiply_(1.0));
    cout << "Importing weights into the network...\n";
    import_gguf_weights(generator, g, spec, lopt);

    hf_tokenizer tok;
    tok.load_from_gguf(g);

    /* Tokenize with the model's own declared conventions (tokenizer.ggml.add_bos_token
       and friends). Forcing a BOS is wrong for models that declare none: SmolLM2's id 1
       is <|im_start|>, a chat control token never seen followed by raw text during
       training, so prepending it puts the probe out of distribution. */
    std::vector<int> toks = tok.encode(prompt);
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

    generator_type generator(multiply_(1.0));
    cout << "Importing weights into the network...\n";
    import_gguf_weights(generator, g, spec, lopt);

    hf_tokenizer tok;
    tok.load_from_gguf(g);

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

    /* Archive format: a format tag, the model name, the parameter-bearing subnet
       (the loss head carries no parameters) and the tokenizer. Serializing the subnet
       rather than the full loss network lets --load deserialize straight into the
       generator, so a single copy of the parameters is ever allocated; the alternative of
       serializing the full network would need a temporary at load time, transiently
       doubling the pinned host memory. */
    cout << "Serializing model to " << out_path << " ...\n";
    serialize(out_path) << std::string("gguf_import_model") << model_display_name(spec)
                        << net.subnet() << tok;
    cout << "Done. Wrote " << out_path << "\n";
    return 0;
}

#endif // WITH_IMPORTED_MODEL

int main(int argc, char** argv)
{
    try
    {
#ifdef _WIN32
        /* The Windows console defaults to the OEM code page (CP850 on French systems),
           which garbles the UTF-8 byte stream the tokenizer emits and reads. Switch both
           directions to UTF-8 so accented output and input display correctly. */
        SetConsoleOutputCP(CP_UTF8);
        SetConsoleCP(CP_UTF8);
#endif
        command_line_parser parser;
        parser.add_option("input", "Path to the source .gguf model", 1);
        parser.add_option("out-prefix", "Output prefix for generated files (default: derived from the model name)", 1);
        parser.add_option("load", "Path to a converted .dat model; with --chat, skips the GGUF import entirely", 1);
        parser.add_option("probe", "Extra string to round-trip through the tokenizer", 1);
        parser.add_option("chat", "Load the model and start an interactive completion session");
        parser.add_option("convert", "Load the model and serialize it to <out-prefix>.dat");
        parser.add_option("probe-logits", "Print the most probable next tokens for --prompt (weight validation)");
        parser.add_option("prompt", "Prompt used by --probe-logits (default: 'The capital of France is')", 1);
        parser.add_option("probe-ids", "Print next-token predictions for an explicit space-separated id list", 1);
        parser.add_option("context", "KV cache length for --chat (default: 512)", 1);
        parser.add_option("temperature", "Sampling temperature (default: model template preset)", 1);
        parser.add_option("top-k", "Top-k filter (default: model template preset)", 1);
        parser.add_option("top-p", "Nucleus threshold (default: model template preset)", 1);
        parser.add_option("min-p", "Relative min-p threshold (default: 0.05)", 1);
        parser.add_option("repeat-penalty", "Repetition penalty (default: 1.1)", 1);
        parser.add_option("deterministic", "Greedy decoding (argmax)");
        parser.add_option("raw", "Chat without the chat template (raw text completion)");
        parser.add_option("system", "System prompt used by --chat (default: a helpful assistant)", 1);
        parser.add_option("template", "Chat template override: auto, zephyr, chatml, guanaco, granite (default: auto)", 1);
        parser.add_option("offload-params", "Keep supported layer parameters in host memory and materialize them per layer (lowers VRAM)");
        parser.add_option("rope-permute", "Permute Q/K rows from split-half (NeoX) to interleaved RoPE ordering; leave off for llama-family GGUFs, expected for NeoX-convention architectures");
        parser.add_option("swap-gate-up", "Swap ffn_gate / ffn_up assignment (weight-loader knob)");
        parser.parse(argc, argv);

        /* Chat over an already-converted model: no GGUF needed, the .dat archive carries
           both the network weights and the tokenizer. */
        if (parser.option("load") && parser.option("chat"))
        {
#ifdef WITH_IMPORTED_MODEL
            return run_chat_dat(parser.option("load").argument(),
                get_option(parser, "temperature", -1.0),
                get_option(parser, "top-k", size_t(0)),
                get_option(parser, "top-p", -1.0f),
                get_option(parser, "min-p", -1.0f),
                get_option(parser, "repeat-penalty", -1.0f),
                parser.option("deterministic"),
                get_option(parser, "context", long(512)),
                /*use_template=*/!parser.option("raw"),
                get_option(parser, "system", std::string("You are a helpful assistant.")),
                get_option(parser, "template", std::string("auto")),
                parser.option("offload-params"));
#else
            cerr << "This build has no model header compiled in; generate slm_imported_model.h\n"
                 << "(run with --out-prefix slm_imported_model) and rebuild the target.\n";
            return 1;
#endif
        }

        if (!parser.option("input"))
        {
            cout << "Import a GGUF model into the Dlib transformer stack.\n\n";
            parser.print_options();
            cout << "\nExamples:\n"
                 << "  Phase 1 (generate header + tokenizer, any build):\n    " << argv[0]
                 << " --input tinyllama-1.1b-chat-v1.0.Q8_0.gguf --out-prefix slm_imported_model\n"
                 << "  Phase 2 (built with WITH_IMPORTED_MODEL):\n    " << argv[0]
                 << " --input tinyllama-1.1b-chat-v1.0.Q8_0.gguf --probe-logits --prompt \"The capital of France is\"\n    " << argv[0]
                 << " --input tinyllama-1.1b-chat-v1.0.Q8_0.gguf --convert\n    " << argv[0]
                 << " --load tinyllama_1_1b_chat_v1_0.dat --chat\n";
            return 0;
        }

        const string input  = parser.option("input").argument();

        cout << "Reading GGUF: " << input << "\n";
        gguf_reader g(input);
        cout << "GGUF version " << g.version() << ", "
             << g.metadata().size() << " metadata keys, "
             << g.tensors().size() << " tensors\n\n";

        const model_spec spec = detect_model(g);
        cout << describe(spec) << "\n";

        /* Every produced file defaults to the model identity (the cleaned general.name,
           sanitized into an identifier), so successive imports of different models do not
           overwrite one another. The header used by this example's own build is
           regenerated with an explicit --out-prefix slm_imported_model. */
        const string prefix = parser.option("out-prefix")
            ? parser.option("out-prefix").argument()
            : sanitize_identifier(model_display_name(spec));

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
                    get_option(parser, "temperature", -1.0),
                    get_option(parser, "top-k", size_t(0)),
                    get_option(parser, "top-p", -1.0f),
                    get_option(parser, "min-p", -1.0f),
                    get_option(parser, "repeat-penalty", -1.0f),
                    parser.option("deterministic"),
                    get_option(parser, "context", long(512)),
                    /*use_template=*/!parser.option("raw"),
                    get_option(parser, "system", std::string("You are a helpful assistant.")),
                    get_option(parser, "template", std::string("auto")),
                parser.option("offload-params"));

            if (parser.option("convert"))
                return run_convert(g, spec, lopt, prefix + ".dat");

            if (parser.option("probe-ids"))
                return run_probe_ids(g, spec, lopt, parser.option("probe-ids").argument());

            return run_probe(g, spec, lopt,
                get_option(parser, "prompt", std::string("The capital of France is")));
#else
            cerr << "This build has no model header compiled in.\n"
                 << "Generate it first (run with --out-prefix slm_imported_model), then rebuild\n"
                 << "the target: the header is detected and included automatically\n"
                 << "(/bigobj is required on MSVC).\n";
            return 1;
#endif
        }

        const string header_path = prefix + ".h";
        /* Same identity for the file, the include guard and the namespace: the cleaned
           model name sanitized into an identifier. Left to itself emit_header derives the
           namespace from the raw general.name, which would drift from the file name
           whenever the cleaner has something to strip. */
        emit_header(spec, header_path, sanitize_identifier(model_display_name(spec)));
        cout << "\nGenerated model header: " << header_path << "\n\n";

        const string probe = parser.option("probe") ? parser.option("probe").argument() : "";
        extract_tokenizer(g, prefix + "_tokenizer.dat", probe, spec.model_name);

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
