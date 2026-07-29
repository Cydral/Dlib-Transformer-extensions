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
      slm_gguf_import_ex --input model.gguf --mmproj mmproj-model.gguf --out-prefix m
      slm_gguf_import_ex --input model.gguf --mmproj mmproj-model.gguf --convert
      slm_gguf_import_ex --load model.dat --probe-logits
      slm_gguf_import_ex --load model.dat --image photo.png
      slm_gguf_import_ex --load model.dat --serve 8080

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

    Images. A multimodal model ships a second container, conventionally named
    mmproj-*.gguf, holding the vision tower and the projector that brings its output into
    the decoder's embedding space. Give it with --mmproj when generating the header and the
    emitted configuration becomes a multimodal_transformer_config: the tower is then part
    of the network type, its weights are imported like any others, the archive written by
    --convert carries the whole stack, and a gradient can reach the tower. That is the
    difference with the runtime engine, which keeps the two files apart and encodes images
    outside the graph.

    An image then enters through network_context's vision slot, as prepared pixels rather
    than as vectors, and the prompt reserves one position per vector the tower will produce
    with a placeholder token. Nothing in the stack below the fusion layer knows an image
    went through.

    The archive written by --convert is self-contained: decoder, tower, tokenizer and the
    pixel normalization the tower was trained with. Neither container is needed again, for
    chatting, describing, serving or training.
!*/

#include <iostream>
#include <string>
#include <thread>
#include <chrono>
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
#include <dlib/image_io.h>
#include <dlib/image_transforms.h>
#include <dlib/data_io/gguf_reader.h>
#include <dlib/data_io/gguf_dequantize.h>
#include <dlib/data_io/gguf_model_spec.h>
#include <dlib/data_io/gguf_vision_spec.h>
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
#  include <dlib/data_io/gguf_vision_loader.h>
#  include <dlib/data_io/model_archive.h>
#  include <dlib/server/chat_service.h>
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

/* Low-rank adaptation requested on the command line. Applied after the weights are in
   place, never before: DoRA initializes its magnitudes from the column norms of the base,
   so an adapter configured on an untrained network would carry the norms of the random
   initialization instead. */
using adapter_request = dlib::adapter_plan;

/* Configures the adapters and reports what the optimizer would be allowed to move.

   The report is printed whether or not adapters were asked for, because the number it
   carries is the one thing a training log never says: a parameter-efficient method that
   silently left the whole network trainable produces a plausible loss curve and an hour of
   wasted compute. */
template <typename net_type>
void apply_adapters(net_type& net, const adapter_request& req)
{
    if (!req.active()) return;

    const size_t layers = configure_network_adapters(net, req);
    freeze_all_but_adapters(net);

    const trainable_counts counts = count_trainable_parameters(net);
    cout << "Adapters            : " << adapter_method_name(req.method)
         << ", rank " << req.rank << ", alpha " << req.alpha
         << " on " << (req.attention_query ? "Q" : "") << (req.attention_value ? "V" : "")
         << (req.projection ? "+FFN" : "")
         << " over " << layers << " layers\n"
         << counts.describe() << "\n";
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

/* Reads a projector container and reports its geometry. Depends on no compiled-in model,
   so this runs in both build phases and is the cheapest way to tell whether a given
   container is one this pipeline can serve. */
int report_vision(const std::string& mmproj_path, vision_spec& out)
{
    cout << "Reading projector: " << mmproj_path << "\n";
    gguf_reader gv(mmproj_path);
    out = detect_vision(gv);
    cout << describe(out);
    const vision_compat_result rep = check_vision_compatibility(out, gv);
    for (const string& n : rep.notes)    cout << "note: "    << n << "\n";
    for (const string& b : rep.blockers) cerr << "BLOCKER: " << b << "\n";
    if (!rep.usable())
    {
        cout << "This projector cannot be served by the current vision path.\n";
        return 1;
    }
    return 0;
}

#ifdef WITH_IMPORTED_MODEL

/* Vision operations of a model that has a tower, and their absence for one that has not.

   The selection is a preprocessor one because it has to be. A text-only header does not
   declare vision_tower or VISUAL_TOKENS, and code naming them cannot be compiled against
   it at all; a template would not help, those names being non-dependent and therefore
   resolved where the template is written rather than where it is used. The generated
   header defines DLIB_IMPORTED_MODEL_HAS_VISION when it carries a tower, and that is what
   keeps the two worlds apart. */
struct vision
{
#ifdef DLIB_IMPORTED_MODEL_HAS_VISION

    static constexpr bool available = true;
    static long tokens_per_image() { return imported_model::VISUAL_TOKENS; }

    /* Finds the fusion layer by its type rather than by its index. The index depends on the
       depth of the stack above it, which is a poor thing to hardcode; the type is exactly
       what makes the layer the one we want. */
    template <typename net_type>
    static void import_tower(net_type& net, gguf_reader& g, const vision_spec& spec)
    {
        bool found = false;
        visit_computational_layers(net, [&](auto& layer) {
            import_into_fusion(layer, g, spec, found);
        });
        if (!found)
            throw std::runtime_error("no fusion layer was found in the compiled network");
    }

private:

    template <typename layer_type>
    static void import_into_fusion(layer_type&, gguf_reader&, const vision_spec&, bool&) {}

    template <typename E, long SLOT, long TK, long W>
    static void import_into_fusion(modality_fusion_<E, SLOT, TK, W>& l, gguf_reader& g,
        const vision_spec& spec, bool& found)
    {
        import_gguf_vision_weights(l.get_encoder(), g, spec, imported_model::vision_tower());
        found = true;
    }

public:

#else

    static constexpr bool available = false;
    static long tokens_per_image() { return 0; }

    template <typename net_type>
    static void import_tower(net_type&, gguf_reader&, const vision_spec&)
    {
        throw std::runtime_error("this build has no vision tower compiled in; regenerate "
            "the header with --mmproj and rebuild");
    }

#endif

    /* The prepared pixels: the tower's own normalization, read from the container rather
       than assumed. The same function serves the shape-dynamic encoder, so both paths see
       exactly the same pixels for the same file. */
    static void prepare(const matrix<rgb_pixel>& img, const vision_spec& spec,
        resizable_tensor& out)
    {
        prepare_vision_image(img, spec, out);
    }
};

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

/* Engine adapter for the chat service: the compiled network seen through the small
   interface chat_service.h expects, so that this program and the shape-dynamic one serve
   the same endpoint from one implementation.

   Two things are worth noting. The generation core wants logits, while the generator built
   above ends in a softmax for the interactive loop's benefit, so the adapter drives the
   subnet underneath it instead: the same weights, one layer short of the normalization.
   And staging an image here means normalizing pixels, not encoding them; the tower is a
   layer of this network and will run during the prefill, which is exactly what lets a
   gradient reach it. */
class static_engine_adapter
{
public:

    explicit static_engine_adapter(generator_type& net) : net_(net) {}

    void set_context(long capacity, long keep)
    {
        network_context::set_kv_cache_capacity(capacity);
        network_context::set_kv_cache_keep_length(keep);
        network_context::request_kv_cache_clear();
        network_context::clear_padding();
        network_context::set_inference_mode(network_context::inference_mode::prefill);
    }

    const tensor& forward_prefill(const std::vector<int>& ids)
    {
        matrix<int, 0, 1> pf(static_cast<long>(ids.size()), 1);
        for (long i = 0; i < static_cast<long>(ids.size()); ++i)
            pf(i) = ids[static_cast<size_t>(i)];
        const tensor& out = logits(pf);
        network_context::clear_kv_cache_request();
        network_context::set_inference_mode(network_context::inference_mode::incremental);
        network_context::clear_padding();
        return out;
    }

    const tensor& step(int token)
    {
        matrix<int, 0, 1> one(1, 1);
        one(0) = token;
        return logits(one);
    }

    bool vision_available() const { return vision::available; }
    long visual_tokens() const { return vision::tokens_per_image(); }

    void set_vision_spec(const vision_spec& vs) { vspec_ = vs; }

    bool stage_image(const matrix<rgb_pixel>& img, std::string& why)
    {
        if (!vision::available) { why = "this build has no vision tower"; return false; }
        if (vspec_.image_size <= 0)
        { why = "no pixel normalization is known for this model"; return false; }
        try
        {
            vision::prepare(img, vspec_, staged_);
            return true;
        }
        catch (const std::exception& e)
        { why = std::string("preparing: ") + e.what(); return false; }
    }

    void commit_images(const std::vector<long>& positions)
    {
        std::vector<modality_input> in(1);
        in[0].payload = staged_;
        in[0].positions = positions;
        in[0].sequence = 0;
        network_context::set_modality_inputs(modality_slot::vision, std::move(in));
    }

private:

    const tensor& logits(const matrix<int, 0, 1>& x)
    {
        return net_.subnet().subnet()(x);
    }

    generator_type& net_;
    vision_spec vspec_;
    resizable_tensor staged_;
};

/* Loads the model, and the tower with it when the build has one. Both come from their own
   container; what makes the result one model rather than two is that the tower ends up
   inside the network, in the fusion layer, and is written to the archive with everything
   else. */
int load_model(generator_type& generator, gguf_reader& g, const model_spec& spec,
    const gguf_load_options& lopt, const adapter_request& adapters,
    const std::string& mmproj_path, vision_spec& vspec)
{
    if (!model_matches_header(spec))
    { cerr << "Error: model does not match the compiled-in header. Regenerate and recompile.\n"; return 1; }

    cout << "Importing weights into the network...\n";
    import_gguf_weights(generator, g, spec, lopt);

    if (vision::available)
    {
        if (mmproj_path.empty())
        { cerr << "Error: this build carries a vision tower and needs --mmproj to fill it.\n"; return 1; }
        if (report_vision(mmproj_path, vspec) != 0) return 1;
        gguf_reader gv(mmproj_path);
        cout << "Importing the vision tower...\n";
        vision::import_tower(generator, gv, vspec);
    }
    else if (!mmproj_path.empty())
    {
        cerr << "Error: --mmproj was given but this build has no vision tower; regenerate\n"
             << "the header with --mmproj and rebuild.\n";
        return 1;
    }

    apply_adapters(generator, adapters);
    return 0;
}

/* Reads a converted archive. The tower, when there is one, travels inside it: nothing else
   is needed to serve or to fine-tune the model. */
int load_archive(generator_type& generator, hf_tokenizer& tok, const std::string& dat_path,
    std::string& name_out, vision_spec& vspec_out)
{
    cout << "Loading converted model from " << dat_path << " ...\n";
    model_archive_info info;
    try
    {
        load_model_archive(dat_path, generator.subnet().subnet(), tok, info,
            vision::available);
    }
    catch (const std::exception& e)
    { cerr << "Error: " << e.what() << "\n"; return 1; }

    name_out = clean_model_name(info.model_name);
    vspec_out = info.vision;
    cout << "Model: " << name_out
         << (info.has_vision ? " (vision tower included)" : "") << "\n";
    if (info.tail_missing)
        cout << "note: this archive carries no pixel normalization, so it predates that\n"
                "      block. Text works; images will refuse rather than guess.\n";
    return 0;
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
    const std::string& system_prompt, const std::string& template_override, bool offload_params,
    const adapter_request& adapters, const std::string& mmproj_path)
{
    generator_type generator(multiply_(1.0));
    vision_spec vspec;
    if (load_model(generator, g, spec, lopt, adapters, mmproj_path, vspec) != 0) return 1;

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
    vision_spec vspec;
    if (load_archive(generator, tok, dat_path, name_hint, vspec) != 0) return 1;
    return chat_loop(generator, tok, temperature, top_k, top_p, min_p, repeat_penalty,
        deterministic, ctx_len, use_template, system_prompt, template_override,
        name_hint, offload_params);
}

/* Describes one image through the compiled network.

   Everything happens inside the graph: the pixels go into the vision slot of the context,
   the fusion layer runs the tower it owns and writes its vectors over the positions the
   prompt reserved, and the decoder never learns that an image went through. The tower's
   weights came from the archive or from the container at import time, like any others. */
int describe_image(generator_type& generator, hf_tokenizer& tok, const std::string& model_name,
    const std::string& image_path, const std::string& question, const vision_spec& vspec,
    double temperature, size_t top_k, float top_p, float min_p, float repeat_penalty,
    bool deterministic, long ctx_len, const std::string& system_prompt,
    const std::string& template_override, bool offload_params)
{
    if (!vision::available)
    { cerr << "This build has no vision tower compiled in.\n"; return 1; }

    matrix<rgb_pixel> img;
    load_image(img, image_path);
    cout << "Image              : " << image_path
         << " (" << img.nc() << "x" << img.nr() << ")\n";
    resizable_tensor pixels;
    vision::prepare(img, vspec, pixels);

    const chat_template_formatter fmt = (template_override.empty() || template_override == "auto")
        ? chat_template_formatter::for_tokenizer(tok, model_name)
        : chat_template_formatter::for_tokenizer(tok,
              chat_template_formatter::from_name(template_override));
    cout << "Chat template      : " << chat_template_formatter::name(fmt.kind()) << "\n";
    if (fmt.kind() != chat_template_kind::idefics3)
        cout << "note: this template declares no image markers; the description will "
                "likely be poor\n";

    if (temperature < 0.0)     temperature    = fmt.default_temperature();
    if (top_k == 0)            top_k          = fmt.default_top_k();
    if (top_p < 0.0f)          top_p          = fmt.default_top_p();
    if (min_p < 0.0f)          min_p          = fmt.default_min_p();
    if (repeat_penalty < 0.0f) repeat_penalty = fmt.default_repeat_penalty();

    const long tokens = vision::tokens_per_image();
    const string turn = fmt.first_turn(system_prompt,
        idefics3_markers::image_block(tokens) + question);
    std::vector<int> ids = tok.encode(turn, fmt.add_bos_on_first_turn(), false, true, false);

    /* The reserved positions are found by scanning for the placeholder rather than by
       counting characters: the tokenizer decides where they land, and a single token of
       drift would put the image on the wrong words. */
    const std::vector<int> mark = tok.encode(idefics3_markers::image_placeholder(),
        false, false, true, false);
    if (mark.size() != 1)
    { cerr << "The image placeholder is not a single token of this vocabulary.\n"; return 1; }
    std::vector<long> positions;
    for (size_t i = 0; i < ids.size(); ++i)
        if (ids[i] == mark[0]) positions.push_back(static_cast<long>(i));
    cout << "Prompt             : " << ids.size() << " tokens, "
         << positions.size() << " reserved for the image\n";
    if (static_cast<long>(positions.size()) != tokens)
    { cerr << "The reserved positions do not match what the tower produces.\n"; return 1; }

    const float temp = deterministic ? 1.0f : static_cast<float>(temperature);
    layer<1>(generator).layer_details().set_multiply_value(1.0f / temp);

    network_context::reset();
    if (offload_params)
        network_context::set_parameter_residency(network_context::parameter_residency::host_f32);
    network_context::set_kv_cache_capacity(std::max<long>(ctx_len,
        static_cast<long>(ids.size()) + 256));
    /* The whole prompt is pinned: the image occupies most of it, and letting the visual
       positions slide out of the window mid-description leaves the model answering about a
       picture it can no longer see. */
    network_context::set_kv_cache_keep_length(static_cast<long>(ids.size()));
    network_context::request_kv_cache_clear();
    network_context::clear_padding();
    network_context::set_inference_mode(network_context::inference_mode::prefill);

    /* Posted last, after the reset that would otherwise discard it. Consumed by the
       prefill; the single-token steps that follow see an empty slot. */
    {
        std::vector<modality_input> in(1);
        in[0].payload = pixels;
        in[0].positions = positions;
        in[0].sequence = 0;
        network_context::set_modality_inputs(modality_slot::vision, std::move(in));
    }

    const int eos = tok.eos_id();
    dlib::rand rng(std::time(nullptr));
    std::vector<int> ctx;
    cout << "\nModel: thinking" << std::flush;
    try
    {
        matrix<int, 0, 1> pf(static_cast<long>(ids.size()), 1);
        for (long i = 0; i < static_cast<long>(ids.size()); ++i) pf(i) = ids[static_cast<size_t>(i)];
        int nxt = pick_next(generator(pf), ctx, deterministic, top_k, top_p, min_p,
            repeat_penalty, rng);
        network_context::clear_kv_cache_request();
        ctx.insert(ctx.end(), ids.begin(), ids.end());
        network_context::set_inference_mode(network_context::inference_mode::incremental);
        network_context::clear_padding();

        std::vector<int> out_toks;
        const std::string stop = fmt.stop_string();
        for (int i = 0; i < 512; ++i)
        {
            if (nxt == eos) break;
            ctx.push_back(nxt);
            out_toks.push_back(nxt);
            if (!stop.empty() && tok.decode(out_toks, true).find(stop) != std::string::npos)
                break;
            static const char* const dots[] = { ".  ", ".. ", "..." };
            cout << "\rModel: thinking" << dots[(i / 8) % 3] << std::flush;
            matrix<int, 0, 1> step(1, 1);
            step(0) = nxt;
            nxt = pick_next(generator(step), ctx, deterministic, top_k, top_p, min_p,
                repeat_penalty, rng);
        }
        cout << "\r" << std::string(20, ' ') << "\r";
        cout << "Model: " << fmt.clean_answer(tok.decode(out_toks, true)) << "\n";
    }
    catch (const std::exception& e)
    {
        cerr << "\nFATAL during generation: " << e.what() << "\n";
        network_context::reset();
        return 1;
    }
    network_context::reset();
    return 0;
}

/* Description after importing from the containers. */
int run_describe(gguf_reader& g, const model_spec& spec, const gguf_load_options& lopt,
    const adapter_request& adapters, const std::string& mmproj_path,
    const std::string& image_path, const std::string& question,
    double temperature, size_t top_k, float top_p, float min_p, float repeat_penalty,
    bool deterministic, long ctx_len, const std::string& system_prompt,
    const std::string& template_override, bool offload_params)
{
    generator_type generator(multiply_(1.0));
    vision_spec vspec;
    if (load_model(generator, g, spec, lopt, adapters, mmproj_path, vspec) != 0) return 1;

    hf_tokenizer tok;
    tok.load_from_gguf(g);
    return describe_image(generator, tok, spec.model_name, image_path, question, vspec,
        temperature, top_k, top_p, min_p, repeat_penalty, deterministic, ctx_len,
        system_prompt, template_override, offload_params);
}

/* Description over a converted archive. Everything comes from that one file: the decoder,
   the tower, the tokenizer and the pixel normalization. */
int run_describe_dat(const std::string& dat_path,
    const std::string& image_path, const std::string& question,
    double temperature, size_t top_k, float top_p, float min_p, float repeat_penalty,
    bool deterministic, long ctx_len, const std::string& system_prompt,
    const std::string& template_override, bool offload_params)
{
    generator_type generator(multiply_(1.0));
    hf_tokenizer tok;
    std::string name;
    vision_spec vspec;
    if (load_archive(generator, tok, dat_path, name, vspec) != 0) return 1;
    return describe_image(generator, tok, name, image_path, question, vspec,
        temperature, top_k, top_p, min_p, repeat_penalty, deterministic, ctx_len,
        system_prompt, template_override, offload_params);
}

/* Serves the compiled model over the shared chat endpoint.

   The service, its request handling and its streaming come from dlib/server/chat_service.h,
   which the shape-dynamic program uses too: the two answer identically because they run one
   implementation, not because two were kept in step. */
int run_serve(generator_type& generator, hf_tokenizer& tok, const std::string& name,
    const vision_spec* vspec, unsigned short port, long ctx_len,
    double forced_temp, bool temp_forced, bool deterministic, bool trace_prompt,
    const std::string& template_override, bool offload_params)
{
    const chat_template_formatter fmt = (template_override.empty() || template_override == "auto")
        ? chat_template_formatter::for_tokenizer(tok, name)
        : chat_template_formatter::for_tokenizer(tok,
              chat_template_formatter::from_name(template_override));

    network_context::reset();
    if (offload_params)
        network_context::set_parameter_residency(network_context::parameter_residency::host_f32);
    /* The interactive loop scales the logits through the multiply layer; the service
       resolves temperature inside the sampler, so that layer must be neutral here. */
    layer<1>(generator).layer_details().set_multiply_value(1.0f);

    static_engine_adapter engine(generator);
    if (vspec) engine.set_vision_spec(*vspec);

    std::vector<dlib::served_model<static_engine_adapter>> models;
    models.push_back(dlib::served_model<static_engine_adapter>{ name, &engine, &tok, fmt });

    cout << "Serving 1 model:\n  " << name
         << "  [template " << chat_template_formatter::name(fmt.kind())
         << (fmt.supports_reasoning() ? ", reasoning" : "")
         << (engine.vision_available() ? ", vision" : "") << "]\n";

    dlib::chat_service<static_engine_adapter> srv(std::move(models), ctx_len,
        forced_temp, temp_forced, deterministic, trace_prompt);
    srv.set_listening_port(port);
    cout << "Serving http://localhost:" << port
         << "  (chat interface on /, API on /v1/chat/completions; Ctrl-C to stop)\n";
    /* Started asynchronously and polled, exactly as the shape-dynamic program does.
       start() would block inside the server's own accept loop, where the interrupt has no
       way to reach us: the handler would set its flag and nothing would read it. */
    dlib::signal_handler::setup();
    srv.start_async();
    while (!dlib::signal_handler::is_triggered() && srv.is_running())
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    cout << "\nShutting down (waiting for in-flight requests)...\n";
    srv.clear();   // shuts connections, waits for handlers, releases the port
    cout << "Server stopped.\n";
    network_context::reset();
    return 0;
}

int run_serve_gguf(gguf_reader& g, const model_spec& spec, const gguf_load_options& lopt,
    const adapter_request& adapters, const std::string& mmproj_path, unsigned short port,
    long ctx_len, double forced_temp, bool temp_forced, bool deterministic,
    bool trace_prompt, const std::string& template_override, bool offload_params)
{
    generator_type generator(multiply_(1.0));
    vision_spec vspec;
    if (load_model(generator, g, spec, lopt, adapters, mmproj_path, vspec) != 0) return 1;

    hf_tokenizer tok;
    tok.load_from_gguf(g);
    return run_serve(generator, tok, model_display_name(spec),
        vision::available ? &vspec : nullptr, port, ctx_len, forced_temp, temp_forced,
        deterministic, trace_prompt, template_override, offload_params);
}

int run_serve_dat(const std::string& dat_path,
    unsigned short port, long ctx_len, double forced_temp, bool temp_forced,
    bool deterministic, bool trace_prompt, const std::string& template_override,
    bool offload_params)
{
    generator_type generator(multiply_(1.0));
    hf_tokenizer tok;
    std::string name;
    vision_spec vspec;
    if (load_archive(generator, tok, dat_path, name, vspec) != 0) return 1;
    return run_serve(generator, tok, name, vision::available ? &vspec : nullptr, port,
        ctx_len, forced_temp, temp_forced, deterministic, trace_prompt, template_override,
        offload_params);
}

/* The probe itself, shared by the two ways of getting a loaded network.

   This is the cheapest check there is on a set of weights: five probabilities that can be
   read against a reference implementation, and a per-position argmax that says where two
   runs first diverge. Worth having on an archive and not only on a container, since an
   archive is what a fine-tuning run consumes and produces. */
int probe_prompt(generator_type& generator, hf_tokenizer& tok, const std::string& prompt)
{
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

/* Print the most probable next tokens for a prompt's last position. Compare these with a
   reference (for example an external GGUF runtime) to validate the weight repacking. */
int run_probe(gguf_reader& g, const model_spec& spec, const gguf_load_options& lopt,
    const std::string& prompt, const adapter_request& adapters)
{
    if (!model_matches_header(spec))
    { cerr << "Error: model does not match the compiled-in header. Regenerate and recompile.\n"; return 1; }

    generator_type generator(multiply_(1.0));
    cout << "Importing weights into the network...\n";
    import_gguf_weights(generator, g, spec, lopt);
    apply_adapters(generator, adapters);

    hf_tokenizer tok;
    tok.load_from_gguf(g);
    return probe_prompt(generator, tok, prompt);
}

/* Probe over a converted archive, so that the weights a fine-tuning run reads and writes
   can be checked without going back to the source container. */
int run_probe_dat(const std::string& dat_path, const std::string& prompt)
{
    generator_type generator(multiply_(1.0));
    hf_tokenizer tok;
    std::string name;
    vision_spec vspec;
    if (load_archive(generator, tok, dat_path, name, vspec) != 0) return 1;
    return probe_prompt(generator, tok, prompt);
}

/* Probe on an explicit token-id sequence, bypassing the tokenizer. This isolates forward-pass
   behavior for a chosen sequence, for instance to measure the effect of a control token such
   as the eos </s> in the middle of a prompt. The ids must already include BOS if wanted. */
int run_probe_ids(gguf_reader& g, const model_spec& spec, const gguf_load_options& lopt,
    const std::string& id_string, const adapter_request& adapters)
{
    if (!model_matches_header(spec))
    { cerr << "Error: model does not match the compiled-in header. Regenerate and recompile.\n"; return 1; }

    generator_type generator(multiply_(1.0));
    cout << "Importing weights into the network...\n";
    import_gguf_weights(generator, g, spec, lopt);
    apply_adapters(generator, adapters);

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
    const std::string& out_path, const std::string& mmproj_path)
{
    if (!model_matches_header(spec))
    { cerr << "Error: model does not match the compiled-in header. Regenerate and recompile.\n"; return 1; }

    infer_net net;
    cout << "Importing weights into the network...\n";
    import_gguf_weights(net, g, spec, lopt);

    /* The tower joins the archive here. This is the point of the whole static path: what
       comes out is one file holding the complete model, which Dlib can serve, fine-tune or
       keep training without a second container anywhere in sight. */
    vision_spec vspec;
    if (vision::available)
    {
        if (mmproj_path.empty())
        { cerr << "Error: this build carries a vision tower and needs --mmproj to fill it.\n"; return 1; }
        if (report_vision(mmproj_path, vspec) != 0) return 1;
        gguf_reader gv(mmproj_path);
        cout << "Importing the vision tower...\n";
        vision::import_tower(net, gv, vspec);
    }

    hf_tokenizer tok;
    tok.load_from_gguf(g);

    /* Archive format: a format tag, the model name, the parameter-bearing subnet
       (the loss head carries no parameters) and the tokenizer. Serializing the subnet
       rather than the full loss network lets --load deserialize straight into the
       generator, so a single copy of the parameters is ever allocated; the alternative of
       serializing the full network would need a temporary at load time, transiently
       doubling the pinned host memory. */
    /* The pixel normalization travels with the archive. It is geometry rather than
       weights, but a tower fed pictures centred differently from the ones it was trained
       on sees other images, so the archive is not self-contained without it. This is what
       lets --load serve or describe with no projector container anywhere in sight. */
    cout << "Serializing model to " << out_path << " ...\n";
    {
        model_archive_info info;
        info.model_name = model_display_name(spec);
        info.has_vision = vision::available;
        info.vision = vspec;
        save_model_archive(out_path, info, net.subnet(), tok);
    }
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
        parser.add_option("mmproj", "Multimodal projector container: makes the generated header carry a vision tower", 1);
        parser.add_option("image", "Image described through the compiled network; needs a multimodal build", 1);
        parser.add_option("serve", "Serve the model over HTTP on the given port", 1);
        parser.add_option("trace-prompt", "Print the token stream handed to each generation");
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
        parser.add_option("lora-rank", "Rank of the low-rank adapters; 0 leaves the network untouched (default: 0)", 1);
        parser.add_option("lora-method", "Adaptation method: lora or dora (default: lora)", 1);
        parser.add_option("lora-alpha", "Adapter alpha; the effective scale is alpha / rank (default: 16)", 1);
        parser.add_option("lora-targets", "Projections to adapt, as letters among q, v and f for the feed-forward (default: qv)", 1);
        parser.add_option("lora-max-width", "Widest projection an adapter may attach to; keeps the vocabulary head out (default: 16384)", 1);
        parser.parse(argc, argv);

        /* Adapter request, resolved once and passed to whichever mode runs. Targets are
           given as one string rather than as two flags so that a sweep script varies one
           argument instead of a combination. */
        adapter_request adapters;
        adapters.rank = get_option(parser, "lora-rank", long(0));
        adapters.alpha = get_option(parser, "lora-alpha", 16.0);
        adapters.method = adapter_method_from_name(
            get_option(parser, "lora-method", std::string("lora")));
        {
            const std::string targets = get_option(parser, "lora-targets", std::string("qv"));
            adapters.attention_query = targets.find('q') != std::string::npos;
            adapters.attention_value = targets.find('v') != std::string::npos;
            adapters.projection = targets.find('f') != std::string::npos;
            /* The output head projects onto the vocabulary; an adapter there would cost
               more than every other adapter combined and is rarely what a fine-tune
               needs, so plain projections are bounded well below it. The bound is a plain
               width rather than a multiple of the model dimension, since this block runs
               before any model is known. */
            adapters.max_width = get_option(parser, "lora-max-width", long(16384));
            if (adapters.rank > 0 && !adapters.attention_query && !adapters.attention_value
                && !adapters.projection)
            { cerr << "Error: --lora-targets selects no projection.\n"; return 1; }
            if (adapters.rank > 0 && adapters.method == adapter_method::none)
            { cerr << "Error: --lora-method must be lora or dora.\n"; return 1; }
        }

        /* A projector reported on its own. It depends on no compiled-in model, so this
           runs in both build phases and is the cheapest way to tell whether a container is
           one this pipeline can serve. */
        if (parser.option("mmproj") && !parser.option("image") && !parser.option("input")
            && !parser.option("load"))
        {
            vision_spec vs;
            return report_vision(parser.option("mmproj").argument(), vs);
        }

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

        /* Serving an already-converted model. */
        if (parser.option("load") && parser.option("serve"))
        {
#ifdef WITH_IMPORTED_MODEL
            return run_serve_dat(parser.option("load").argument(),
                static_cast<unsigned short>(get_option(parser, "serve", 8080)),
                get_option(parser, "context", long(2048)),
                get_option(parser, "temperature", 0.0),
                parser.option("temperature") != 0,
                parser.option("deterministic"), parser.option("trace-prompt"),
                get_option(parser, "template", std::string("auto")),
                parser.option("offload-params"));
#else
            cerr << "This build has no model header compiled in.\n";
            return 1;
#endif
        }

        /* The probe over an archive: same five probabilities, no container needed. */
        if (parser.option("load") && parser.option("probe-logits"))
        {
#ifdef WITH_IMPORTED_MODEL
            return run_probe_dat(parser.option("load").argument(),
                get_option(parser, "prompt", std::string("The capital of France is")));
#else
            cerr << "This build has no model header compiled in.\n";
            return 1;
#endif
        }

        /* An image described through an already-converted model. */
        if (parser.option("load") && parser.option("image"))
        {
#ifdef WITH_IMPORTED_MODEL
            return run_describe_dat(parser.option("load").argument(),
                parser.option("image").argument(),
                get_option(parser, "prompt", std::string("What is in this image?")),
                get_option(parser, "temperature", -1.0),
                get_option(parser, "top-k", size_t(0)),
                get_option(parser, "top-p", -1.0f),
                get_option(parser, "min-p", -1.0f),
                get_option(parser, "repeat-penalty", -1.0f),
                parser.option("deterministic"),
                get_option(parser, "context", long(512)),
                get_option(parser, "system", std::string()),
                get_option(parser, "template", std::string("auto")),
                parser.option("offload-params"));
#else
            cerr << "This build has no model header compiled in; generate slm_imported_model.h\n"
                 << "(run with --out-prefix slm_imported_model --mmproj ...) and rebuild.\n";
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

        if (parser.option("chat") || parser.option("probe-logits") || parser.option("convert")
            || parser.option("probe-ids") || parser.option("image") || parser.option("serve"))
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
                    parser.option("offload-params"), adapters,
                    get_option(parser, "mmproj", std::string()));

            if (parser.option("serve"))
                return run_serve_gguf(g, spec, lopt, adapters,
                    get_option(parser, "mmproj", std::string()),
                    static_cast<unsigned short>(get_option(parser, "serve", 8080)),
                    get_option(parser, "context", long(2048)),
                    get_option(parser, "temperature", 0.0),
                    parser.option("temperature") != 0,
                    parser.option("deterministic"), parser.option("trace-prompt"),
                    get_option(parser, "template", std::string("auto")),
                    parser.option("offload-params"));

            if (parser.option("image"))
                return run_describe(g, spec, lopt, adapters,
                    get_option(parser, "mmproj", std::string()),
                    parser.option("image").argument(),
                    get_option(parser, "prompt", std::string("What is in this image?")),
                    get_option(parser, "temperature", -1.0),
                    get_option(parser, "top-k", size_t(0)),
                    get_option(parser, "top-p", -1.0f),
                    get_option(parser, "min-p", -1.0f),
                    get_option(parser, "repeat-penalty", -1.0f),
                    parser.option("deterministic"),
                    get_option(parser, "context", long(512)),
                    get_option(parser, "system", std::string()),
                    get_option(parser, "template", std::string("auto")),
                    parser.option("offload-params"));

            if (parser.option("convert"))
                return run_convert(g, spec, lopt, prefix + ".dat",
                    get_option(parser, "mmproj", std::string()));

            if (parser.option("probe-ids"))
                return run_probe_ids(g, spec, lopt,
                    parser.option("probe-ids").argument(), adapters);

            return run_probe(g, spec, lopt,
                get_option(parser, "prompt", std::string("The capital of France is")),
                adapters);
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
        vision_spec vspec;
        const bool with_vision = parser.option("mmproj");
        if (with_vision && report_vision(parser.option("mmproj").argument(), vspec) != 0)
            return 1;
        emit_header(spec, header_path, sanitize_identifier(model_display_name(spec)),
            with_vision ? &vspec : nullptr);
        if (with_vision)
            cout << "The generated header carries the vision tower: the compiled network\n"
                 << "holds it, the archive written by --convert carries it, and a gradient\n"
                 << "can reach it.\n";
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
