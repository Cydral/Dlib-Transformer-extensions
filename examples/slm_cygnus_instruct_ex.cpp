/*!
    @file slm_cygnus_instruct_ex.cpp
    @brief Instruction fine-tuning and interactive chat for the Cygnus model series.

    This program is the instruct stage that follows slm_cygnus_foundation_ex.cpp. It loads
    a Cygnus foundation model (GQA + MoE), specializes it on conversational question/answer
    pairs, and exposes an interactive chat mode for immediate testing after fine-tuning.

      --fine-tune   Specialize the foundation model on Q/A pairs (Instruct).
      --chat        Interactive multi-turn conversation with the fine-tuned model.

    Fine-tuning method:
      The conversation is formatted with the chat template
        <question><text> ... </text><answer><text> ... </text>
      and used as a full-sequence training target. The per-token loss
      (loss_cross_entropy_per_token) supervises every position via teacher forcing, so a
      single sample per conversation supervises the whole response at once. Only <pad> is
      ignored; in particular </text> closing the answer is NOT ignored, since it is the
      stop token the model must learn to emit. Conversations are left-padded for batching,
      which is the natural place for padding (variable-length instruct examples) and is
      compatible with the attention layer's leading-padding mask.

      Note on completion-only masking: masking the prompt by position is not expressible
      with this loss, which masks by target token value. Fine-tuning here is therefore
      full-sequence language modeling over the formatted conversation. The structural
      markers are deterministic and learned trivially; the model still learns the
      prompt -> response transition, the response content, and the stop token.

    Stabilization:
      - Lower learning rate than pre-training and conservative AdamW hyperparameters.
      - Relaxed freeze: a global learning-rate multiplier of 0.3 with the final projection
        and final RMSNorm kept at 1.0, so the output distribution adapts to the Q/A format
        while general language structure is largely preserved.

    The architecture constants below MUST match the foundation model exactly, otherwise
    deserialization fails or silently produces wrong results.
!*/

#ifdef _WIN32
// For the console code page, so that non-Latin output is not reinterpreted byte by byte.
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#endif
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <fstream>
#include <chrono>
#include <numeric>

#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/cmd_line_parser.h>
#include <dlib/tokenizer/bpe_tokenizer.h>
#include <dlib/misc_api.h>

#include "slm_data.h"

using namespace std;
using namespace dlib;

/* The Cygnus-M target topology. MUST match slm_cygnus_foundation_ex.cpp exactly.

   A mismatch is a deserialization error rather than a silent degradation, which is the
   behaviour to want: the two programs share weights, and a foundation model loaded into a
   differently shaped network would fail loudly rather than train on nonsense. See the
   foundation example for why each number is what it is. */
struct pipeline_constants
{
    static constexpr long num_tokens = 24576;
    static constexpr long num_layers = 18;
    static constexpr long num_heads = 10;
    static constexpr long num_kv_heads = 2;
    static constexpr long embedding_dim = 640;
    static constexpr long num_experts = 4;
    static constexpr long top_k = 2;
    static constexpr long max_seq_len = 1024;
};

constexpr double Z_LOSS_WEIGHT = 1e-4;

/* Special token ids of the chat template, resolved once from the tokenizer. */
struct special_ids
{
    int question = -1, answer = -1, text_open = -1, text_close = -1, pad = -1;
};

special_ids resolve_special_ids(bpe_tokenizer& tok)
{
    auto get = [&](const std::string& tag) -> int {
        try { return tok.get_special_token_id(tag); }
        catch (...) { return -1; }
        };
    special_ids sp;
    sp.question = get("<question>");
    sp.answer = get("<answer>");
    sp.text_open = get("<text>");
    sp.text_close = get("</text>");
    sp.pad = get("<pad>");
    return sp;
}

bool special_ids_valid(const special_ids& sp)
{
    return sp.question >= 0 && sp.answer >= 0 && sp.text_open >= 0 && sp.text_close >= 0 && sp.pad >= 0;
}

/* Full token sequence for one (question, answer) pair under the chat template. The same
   layout is reused at inference, so the model sees an identical structure in both phases. */
std::vector<int> format_qa(bpe_tokenizer& tok, const special_ids& sp,
    const std::string& q, const std::string& a)
{
    std::vector<int> t;
    t.reserve(64);
    t.push_back(sp.question);
    t.push_back(sp.text_open);
    auto qe = tok.encode(q);
    t.insert(t.end(), qe.begin(), qe.end());
    t.push_back(sp.text_close);
    t.push_back(sp.answer);
    t.push_back(sp.text_open);
    auto ae = tok.encode(a);
    t.insert(t.end(), ae.begin(), ae.end());
    t.push_back(sp.text_close);
    return t;
}

/* Inference prompt: everything up to and including <answer><text>. The model then
   generates the answer content and the closing </text> that terminates the turn. */
std::vector<int> format_question_prompt(bpe_tokenizer& tok, const special_ids& sp, const std::string& q)
{
    std::vector<int> t;
    t.push_back(sp.question);
    t.push_back(sp.text_open);
    auto qe = tok.encode(q);
    t.insert(t.end(), qe.begin(), qe.end());
    t.push_back(sp.text_close);
    t.push_back(sp.answer);
    t.push_back(sp.text_open);
    return t;
}

/* Build instruct samples: one left-padded sample per conversation. The per-token loss
   supervises every position via teacher forcing, so the whole response is supervised in a
   single forward. The scalar label is <pad>: the last position holds the closing </text>
   and has nothing meaningful to predict after it, while the stop token itself is supervised
   one position earlier through the input shift. Conversations longer than the window are
   skipped (Q/A pairs are short). */
void build_sft_samples(
    const std::vector<std::pair<std::string, std::string>>& qa_pairs,
    bpe_tokenizer& tok, const special_ids& sp, long max_seq_len,
    std::vector<matrix<int, 0, 1>>& samples,
    std::vector<matrix<unsigned long, 0, 1>>& labels,
    size_t& skipped)
{
    samples.clear();
    labels.clear();
    skipped = 0;

    for (const auto& qa : qa_pairs)
    {
        std::vector<int> t = format_qa(tok, sp, qa.first, qa.second);
        if (t.size() < 2) { ++skipped; continue; }
        if (static_cast<long>(t.size()) > max_seq_len) { ++skipped; continue; }

        const long pad_n = max_seq_len - static_cast<long>(t.size());
        matrix<int, 0, 1> window(max_seq_len, 1);
        for (long i = 0; i < pad_n; ++i) window(i) = sp.pad;
        for (long i = 0; i < static_cast<long>(t.size()); ++i) window(pad_n + i) = t[i];

        /* One target per position, which is what the loss reads: inside the window each
           position is supervised against the token that follows it, and the last one
           against the padding that closes the sequence. */
        matrix<unsigned long, 0, 1> target(max_seq_len, 1);
        for (long i = 0; i + 1 < max_seq_len; ++i)
            target(i) = static_cast<unsigned long>(window(i + 1));
        target(max_seq_len - 1) = static_cast<unsigned long>(sp.pad);

        samples.push_back(std::move(window));
        labels.push_back(std::move(target));
    }
}

void display_sample_questions(size_t num_samples = 3)
{
    try {
        auto qa = get_dataset_as_pairs({ dataset_id::BLACK_HOLE_QA_PARTA });
        if (qa.empty()) return;
        dlib::rand rng(std::time(0));
        cout << "Sample questions the model was tuned on:\n";
        num_samples = std::min(num_samples, qa.size());
        for (size_t i = 0; i < num_samples; ++i)
            cout << "  - " << qa[rng.get_random_32bit_number() % qa.size()].first << "\n";
        cout << "\n";
    }
    catch (...) {}
}

int run_fine_tune(const std::string& model_file, const std::string& tokenizer_file,
    double learning_rate, long batch_size, long max_epochs, long patience,
    double weight_decay, double beta1, double beta2, std::vector<int>& gpus)
{
    using my_transformer = gqa_moe_transformer_config<
        pipeline_constants::num_tokens, pipeline_constants::num_layers,
        pipeline_constants::num_heads, pipeline_constants::num_kv_heads,
        pipeline_constants::embedding_dim, pipeline_constants::num_experts,
        pipeline_constants::top_k>;
    using train_net = my_transformer::network_type<true>;

    cout << "=== Cygnus instruct fine-tuning ===\n";
    cout << my_transformer::model_info::describe() << "\n";

    const std::string finetuned_model = model_file.substr(0, model_file.find_last_of('.')) + "_instruct.dat";

    train_net net;
    bpe_tokenizer tokenizer;

    if (file_exists(finetuned_model) && !file_exists("chkpt-" + finetuned_model)) {
        deserialize(finetuned_model) >> net >> tokenizer;
        cout << "Resuming from fine-tuned model: " << finetuned_model << "\n";
    }
    else if (file_exists(model_file)) {
        deserialize(model_file) >> net >> tokenizer;
        cout << "Foundation model loaded: " << model_file << "\n";
    }
    else {
        cerr << "Error: foundation model not found (" << model_file << ").\n"
            << "       Run slm_cygnus_foundation_ex --train first.\n";
        return 1;
    }

    if (tokenizer.get_vocab_size() != static_cast<size_t>(pipeline_constants::num_tokens)) {
        cerr << "Error: tokenizer vocab size (" << tokenizer.get_vocab_size()
            << ") does not match num_tokens (" << pipeline_constants::num_tokens << ").\n";
        return 1;
    }

    const special_ids sp = resolve_special_ids(tokenizer);
    if (!special_ids_valid(sp)) {
        cerr << "Error: the chat template special tokens are missing from the tokenizer.\n";
        return 1;
    }

    /* Single-index masking on <pad> only. The pad query mask is enabled implicitly because
       the pad index follows the ignore index in single-index mode, so left-padding rows are
       skipped. </text> is deliberately left unmasked so the stop token is learned. */
    layer<0>(net).loss_details().set_ignore_index(sp.pad);
    layer<0>(net).loss_details().set_z_loss_weight(Z_LOSS_WEIGHT);

    /* Relaxed freeze: keep general structure, let the output head adapt to the Q/A format.
       Topology: loss=0, linear=1, rms_norm=2. */
    set_all_learning_rate_multipliers(net, 0.3);
    layer<1>(net).layer_details().set_learning_rate_multiplier(1.0);
    layer<2>(net).layer_details().set_learning_rate_multiplier(1.0);

    cout << "Loading Q/A datasets...\n";
    auto qa_pairs = get_dataset_as_pairs({
        dataset_id::BLACK_HOLE_QA_PARTA,
        dataset_id::BLACK_HOLE_QA_PARTB,
        dataset_id::BLACK_HOLE_QA_PARTC });
    cout << "Loaded " << qa_pairs.size() << " Q/A pairs\n";

    std::vector<matrix<int, 0, 1>> samples;
    std::vector<matrix<unsigned long, 0, 1>> labels;
    size_t skipped = 0;
    build_sft_samples(qa_pairs, tokenizer, sp, pipeline_constants::max_seq_len, samples, labels, skipped);
    cout << "Instruct samples: " << samples.size() << " (skipped " << skipped
        << " too long or too short)\n";
    if (samples.empty()) { cerr << "Error: no fine-tuning samples produced.\n"; return 1; }

    network_context::set_optimizer_params(weight_decay, beta1, beta2);

    dnn_trainer<train_net, adamw> trainer(net, adamw(weight_decay, beta1, beta2), gpus);
    trainer.set_learning_rate(learning_rate);
    trainer.set_min_learning_rate(1e-7);
    trainer.set_learning_rate_shrink_factor(0.1);
    trainer.set_mini_batch_size(batch_size);
    trainer.set_iterations_without_progress_threshold(patience);
    trainer.set_synchronization_file("chkpt-" + finetuned_model, std::chrono::minutes(5));
    trainer.be_quiet();

    cout << "Starting fine-tuning (lr=" << learning_rate << ", batch=" << batch_size
        << ", max_epochs=" << max_epochs << ")...\n";

    size_t epoch = 0, batches_count = 0;
    const int pad_token = sp.pad;

    while (trainer.get_learning_rate() >= trainer.get_min_learning_rate()
        && epoch < static_cast<size_t>(max_epochs) && !signal_handler::is_triggered())
    {
        double total_loss = 0.0;
        size_t batches_seen = 0, samples_seen = 0;
        auto epoch_start = std::chrono::high_resolution_clock::now();

        shuffle_training_dataset(samples, labels);

        for (size_t i = 0; i < samples.size() && !signal_handler::is_triggered(); i += batch_size)
        {
            size_t batch_end = std::min(i + static_cast<size_t>(batch_size), samples.size());
            std::vector<matrix<int, 0, 1>> batch_X(samples.begin() + i, samples.begin() + batch_end);
            std::vector<matrix<unsigned long, 0, 1>> batch_Y(
                labels.begin() + i, labels.begin() + batch_end);

            /* Left-padded batch: the attention layer needs the per-sample leading padding
               lengths, which forces the synchronization barrier each step; propagating the
               learning rate on the same barrier is then free. */
            std::vector<long> pad_lengths(batch_X.size());
            for (size_t j = 0; j < batch_X.size(); ++j)
                pad_lengths[j] = count_leading_padding(batch_X[j], pad_token);

            trainer.get_net(force_flush_to_disk::no);
            network_context::set_learning_rate(trainer.get_learning_rate());
            network_context::set_padding_from_lengths(pad_lengths);

            trainer.train_one_step(batch_X, batch_Y);

            total_loss += trainer.get_average_loss();
            batches_seen++;
            samples_seen += batch_X.size();

            if (batches_count++ % 25 == 0) {
                double avg_loss = total_loss / batches_seen;
                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - epoch_start).count();
                double sps = (ms > 0) ? (samples_seen * 1000.0 / ms) : 0.0;

                std::ostringstream line;
                line << "epoch#: " << std::setw(4) << std::right << (epoch + 1)
                    << "/" << std::left << std::setw(4) << max_epochs
                    << "  loss: " << std::fixed << std::setprecision(4)
                    << std::setw(8) << std::right << avg_loss
                    << "  lr: " << std::scientific << std::setprecision(2)
                    << std::setw(9) << trainer.get_learning_rate()
                    << "  speed: " << std::fixed << std::setprecision(0)
                    << std::setw(6) << std::right << sps << " samples/sec";

                std::string s = line.str();
                if (s.size() < 90) s.append(90 - s.size(), ' ');
                cout << "\r" << s << std::flush;
            }
        }
        epoch++;
    }
    cout << "\n";

    /* Restore unit multipliers so a subsequent pass starts from a clean state. */
    trainer.get_net();
    set_all_learning_rate_multipliers(net, 1.0);
    net.clean();
    network_context::reset();

    serialize(finetuned_model) << net << tokenizer;
    cout << "Fine-tuning complete. Model saved to: " << finetuned_model << "\n";

/* What the run actually needed, printed here rather than at the end of main because
   the network is still alive and the block inventory therefore still means
   something. A sizing pass is a minute of training and a Ctrl-C. */
if (extended_memory_enabled())
{
    cout << "\n";
    print_extended_memory_stats(cout);
    cout << "\n";
    print_extended_memory_blocks(cout);
}
    return 0;
}

int run_chat(const std::string& model_file, double temperature, size_t top_k, float top_p,
    float repeat_penalty, float min_p, bool deterministic)
{
    using my_transformer = gqa_moe_transformer_config<
        pipeline_constants::num_tokens, pipeline_constants::num_layers,
        pipeline_constants::num_heads, pipeline_constants::num_kv_heads,
        pipeline_constants::embedding_dim, pipeline_constants::num_experts,
        pipeline_constants::top_k>;
    using infer_net = my_transformer::network_type<false>;

    const std::string finetuned_model = model_file.substr(0, model_file.find_last_of('.')) + "_instruct.dat";
    if (!file_exists(finetuned_model)) {
        cerr << "Error: fine-tuned model not found: " << finetuned_model << "\nRun --fine-tune first.\n";
        return 1;
    }

    infer_net net;
    bpe_tokenizer tokenizer;
    deserialize(finetuned_model) >> net >> tokenizer;
    cout << "Model loaded: " << finetuned_model << "\n\n";

    if (tokenizer.get_vocab_size() != static_cast<size_t>(pipeline_constants::num_tokens)) {
        cerr << "Error: tokenizer vocab size mismatch with model.\n";
        return 1;
    }

    const special_ids sp = resolve_special_ids(tokenizer);
    if (!special_ids_valid(sp)) { cerr << "Error: chat template special tokens missing.\n"; return 1; }

    const float effective_temp = deterministic ? 1.0f : static_cast<float>(temperature);

    /* Probability head: the inference subnet produces logits, scaled by 1/T then softmaxed
       per position. The KV cache lives inside this subnet and is driven entirely through
       this object, so cache state stays consistent across prefill and incremental calls. */
    softmaxm<multiply<infer_net::subnet_type>> generator(multiply_(1.0 / effective_temp));
    generator.subnet().subnet() = net.subnet();

    display_sample_questions(3);
    cout << "Type 'quit' or 'exit' to stop.\n\n";

    dlib::rand rng(std::time(0));
    const long max_seq_len = pipeline_constants::max_seq_len;
    const int max_response_tokens = 3 * max_seq_len;

    /* Sampler over the next-token probabilities at the last position: repetition penalty on
       the recent context, then min-p, top-k and nucleus (top-p) filtering. */
    auto sample_next = [&](const float* probs, size_t N, const std::vector<int>& recent) -> int {
        std::vector<float> p(probs, probs + N);

        if (repeat_penalty > 1.0f) {
            const size_t span = std::min<size_t>(recent.size(), 64);
            for (size_t i = recent.size() - span; i < recent.size(); ++i) {
                int id = recent[i];
                if (id >= 0 && static_cast<size_t>(id) < N) p[id] /= repeat_penalty;
            }
            float s = 0.0f;
            for (float v : p) s += v;
            if (s > 1e-8f) for (float& v : p) v /= s;
        }

        const float max_prob = *std::max_element(p.begin(), p.end());
        const float thresh = max_prob * min_p;
        std::vector<std::pair<size_t, float>> cand;
        cand.reserve(N);
        for (size_t i = 0; i < N; ++i) if (p[i] >= thresh) cand.push_back({ i, p[i] });
        if (cand.empty()) return sp.text_close;

        size_t k = std::min(top_k, cand.size());
        std::partial_sort(cand.begin(), cand.begin() + k, cand.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

        float cumsum = 0.0f;
        size_t cutoff = 0;
        for (size_t i = 0; i < k; ++i) { cumsum += cand[i].second; cutoff = i; if (cumsum >= top_p) break; }

        float total = 0.0f;
        for (size_t i = 0; i <= cutoff; ++i) total += cand[i].second;
        if (total < 1e-8f) return static_cast<int>(cand[0].first);

        float r = rng.get_random_float() * total, cs = 0.0f;
        for (size_t i = 0; i <= cutoff; ++i) { cs += cand[i].second; if (r <= cs) return static_cast<int>(cand[i].first); }
        return static_cast<int>(cand[0].first);
        };

    auto pick_next = [&](const tensor& probs_tensor, const std::vector<int>& recent) -> int {
        const long seq_len = probs_tensor.nr();
        const long v_size = probs_tensor.nc();
        const float* probs = probs_tensor.host() + tensor_index(probs_tensor, 0, 0, seq_len - 1, 0);
        if (deterministic) {
            const float* m = std::max_element(probs, probs + v_size);
            return static_cast<int>(std::distance(probs, m));
        }
        return sample_next(probs, static_cast<size_t>(v_size), recent);
        };

    network_context::reset();
    network_context::set_kv_cache_capacity(max_seq_len);

    std::vector<int> dialogue;

    while (!signal_handler::is_triggered())
    {
        cout << "You: ";
        cout.flush();

        std::string user_input;
        if (!std::getline(std::cin, user_input)) break;
        user_input.erase(0, user_input.find_first_not_of(" \t\n\r"));
        user_input.erase(user_input.find_last_not_of(" \t\n\r") + 1);
        if (user_input.empty()) continue;
        if (user_input == "quit" || user_input == "exit") { cout << "Goodbye!\n"; break; }

        auto prompt = format_question_prompt(tokenizer, sp, user_input);
        dialogue.insert(dialogue.end(), prompt.begin(), prompt.end());

        cout << "Cygnus: ";
        cout.flush();

        /* Re-prefill the recent dialogue (last <= max_seq_len tokens), fed cold at positions
           0..L-1 with no padding. This matches how training exposes contexts at low positions
           and avoids the prefill/incremental position discrepancy of a left-padded prefill. */
        const size_t start = dialogue.size() > static_cast<size_t>(max_seq_len)
            ? dialogue.size() - max_seq_len : 0;
        const long win_len = static_cast<long>(dialogue.size() - start);
        matrix<int, 0, 1> prefill_input(win_len, 1);
        for (long i = 0; i < win_len; ++i) prefill_input(i) = dialogue[start + i];

        network_context::request_kv_cache_clear();
        network_context::clear_padding();
        network_context::set_inference_mode(network_context::inference_mode::prefill);

        int next_tok = pick_next(generator(prefill_input), dialogue);

        network_context::clear_kv_cache_request();
        network_context::set_inference_mode(network_context::inference_mode::incremental);

        std::vector<int> answer_tokens;
        size_t printed = 0;
        auto stream_emit = [&](int tok_id) {
            answer_tokens.push_back(tok_id);
            std::string full = tokenizer.decode(answer_tokens, false);
            if (full.size() > printed) { cout << full.substr(printed); cout.flush(); printed = full.size(); }
            };

        if (next_tok != sp.text_close) { dialogue.push_back(next_tok); stream_emit(next_tok); }

        for (int i = 1; i < max_response_tokens && !signal_handler::is_triggered(); ++i)
        {
            if (next_tok == sp.text_close) break;
            matrix<int, 0, 1> incr_input(1, 1);
            incr_input(0) = next_tok;
            next_tok = pick_next(generator(incr_input), dialogue);
            if (next_tok == sp.text_close) { dialogue.push_back(next_tok); break; }
            dialogue.push_back(next_tok);
            stream_emit(next_tok);
        }

        /* Keep the recorded dialogue template-consistent: the answer must end with the
           closing tag even when it is the very first sampled token or when the response
           is truncated by max_response_tokens. The next turn re-prefills from `dialogue`,
           so a missing close tag would expose an unclosed answer to the model. */
        if (dialogue.empty() || dialogue.back() != sp.text_close)
            dialogue.push_back(sp.text_close);

        cout << "\n\n";
    }

    network_context::reset();
    return 0;
}

// ----------------------------------------------------------------------------------------

/* Makes the console able to show what the model actually produces.

   A model trained on seven writing systems will sooner or later print Cyrillic or Japanese,
   and where that lands depends entirely on the terminal. Linux terminals have used UTF-8 for
   twenty years and need nothing. A Windows console still starts in a legacy code page,
   commonly 850 or 437, in which every byte above 127 is reinterpreted as a different
   character: the output is not merely ugly, it is a different text, and anyone reading it
   would conclude the model is broken rather than the console.

   Both ends are set. The output page decides how bytes written are interpreted; the input
   page decides how typed characters are delivered, which matters as soon as someone asks a
   question with an accent in it. The previous pages are restored on the way out, because a
   program that leaves a shell in a different state than it found it is a nuisance.

   Nothing here is conditional on the model or the language: a console that can show UTF-8
   shows ASCII correctly too, so the safe thing is to do it always. */
class console_utf8
{
public:
    console_utf8()
    {
#ifdef _WIN32
        previous_out = GetConsoleOutputCP();
        previous_in = GetConsoleCP();
        SetConsoleOutputCP(CP_UTF8);
        SetConsoleCP(CP_UTF8);
#endif
    }

    ~console_utf8()
    {
#ifdef _WIN32
        if (previous_out) SetConsoleOutputCP(previous_out);
        if (previous_in) SetConsoleCP(previous_in);
#endif
    }

private:
#ifdef _WIN32
    unsigned int previous_out = 0;
    unsigned int previous_in = 0;
#endif
};

int main(int argc, char** argv)
{
    // Restored when this object goes out of scope, at the end of main.
    const console_utf8 console;

    try
    {
        signal_handler::setup();

        command_line_parser parser;
        parser.add_option("fine-tune", "Specialize the foundation model on Q/A pairs (Instruct)");
        parser.add_option("chat", "Interactive multi-turn conversation");

        parser.add_option("learning-rate", "Fine-tuning learning rate (default: 5e-5)", 1);
        parser.add_option("batch-size", "Mini-batch size (default: 64)", 1);
        parser.add_option("max-epochs", "Maximum number of epochs (default: 15)", 1);
        parser.add_option("patience", "Steps without progress before LR reduction (default: 8000)", 1);

        parser.add_option("model-file", "Foundation model path (default: cygnus_model.dat)", 1);
        parser.add_option("tokenizer-file", "Tokenizer path (default: cygnus_tokenizer.vocab)", 1);

        parser.add_option("temperature", "Sampling temperature (default: 0.8)", 1);
        parser.add_option("top-k", "Top-k filter (default: 50)", 1);
        parser.add_option("top-p", "Nucleus sampling threshold (default: 0.9)", 1);
        parser.add_option("repeat-penalty", "Repetition penalty (default: 1.2)", 1);
        parser.add_option("min-p", "Relative min-p threshold (default: 0.05)", 1);
        parser.add_option("deterministic", "Deterministic decoding (strict argmax)");
        /* Extended device memory. Off unless asked for, so a run that does not mention it
           behaves exactly as before, and inert in a build without CUDA, where enabling it
           returns false and nothing else changes. */
        parser.add_option("extended-memory", "Stream tensors through the device under a budget "
            "when the working set does not fit");
        parser.add_option("vram-budget", "Device memory the extension may use, in MiB (default: 8192)", 1);
        parser.add_option("vram-store", "Directory holding the store, or \"none\" to keep evicted blocks "
            "in host memory only (default: DLIB_XMEM_STORE, else the temp dir)", 1);
        parser.add_option("host-limit", "Pinned host memory the extension may take for evicted "
            "blocks, in MiB. Gigabytes of overflow belong in a store instead: its pages are "
            "ordinary memory the system can reclaim (default: the same as the budget)", 1);
        parser.add_option("blocks", "list the managed blocks by size when the run ends");

        parser.parse(argc, argv);

        /* Before any tensor exists, which is why this sits at the top of main rather than
           beside the code that builds a network. The subsystem cannot be switched on later
           and cannot be switched off at all: turning it off would mean bringing every block
           back at once, which is the operation it exists because one cannot perform. */
        if (parser.option("extended-memory"))
        {
            extended_memory_options xopts;
            xopts.vram_budget = (size_t)get_option(parser, "vram-budget", 8192) << 20;
            /* The store is the tier that carries the overflow. Its pages are ordinary
               memory the system can reclaim, which is what makes it the right place for
               gigabytes; pinned host memory cannot be paged out and is bounded for that
               reason. Point it at a volume with room. */
            xopts.store_path  = get_option(parser, "vram-store", default_extended_memory_store_path());
            if (xopts.store_path == "none")
                xopts.store_path.clear();
            xopts.max_pinned_bytes = (size_t)get_option(parser, "host-limit",
                                        (unsigned long)(xopts.vram_budget >> 20)) << 20;
            xopts.verbose     = true;

            if (!enable_extended_memory(xopts))
                cout << "Extended memory was requested but is unavailable in this build; "
                        "continuing without it.\n";
        }

        if (!parser.option("fine-tune") && !parser.option("chat")) {
            cout << "Cygnus instruct fine-tuning and chat\n\n";
            parser.print_options();
            cout << "\nExample usage:\n"
                << "  Fine-tune : " << argv[0] << " --fine-tune\n"
                << "  Chat      : " << argv[0] << " --chat\n"
                << "\n  Add --extended-memory to any of the above when the batch or the window\n"
                   "  no longer fits the card. --vram-budget sets what the extension may use,\n"
                   "  in MiB, and defaults to 8192. Raise it until the working set fits and the\n"
                   "  extension goes quiet; that is the regime it is meant to run in.\n";
        return 0;
        }

        const std::string model_file = get_option(parser, "model-file", std::string("cygnus_model.dat"));
        const std::string tokenizer_file = get_option(parser, "tokenizer-file", std::string("cygnus_tokenizer.vocab"));

        std::vector<int> gpus{ 0 };

        if (parser.option("fine-tune")) {
            const double learning_rate = get_option(parser, "learning-rate", 5e-5);
            const long batch_size = get_option(parser, "batch-size", 16);
            const long max_epochs = get_option(parser, "max-epochs", 15);
            const long patience = get_option(parser, "patience", 8000);
            const double weight_decay = 0.004;
            const double beta1 = 0.9;
            const double beta2 = 0.98;
            return run_fine_tune(model_file, tokenizer_file, learning_rate, batch_size, max_epochs,
                patience, weight_decay, beta1, beta2, gpus);
        }

        const double temperature = get_option(parser, "temperature", 0.8);
        const size_t top_k = get_option(parser, "top-k", 50);
        const float  top_p = get_option(parser, "top-p", 0.9f);
        const float  repeat_penalty = get_option(parser, "repeat-penalty", 1.2f);
        const float  min_p = get_option(parser, "min-p", 0.05f);
        const bool   deterministic = parser.option("deterministic");
        return run_chat(model_file, temperature, top_k, top_p, repeat_penalty, min_p, deterministic);
    }
    catch (exception& e)
    {
        cerr << "Exception thrown: " << e.what() << endl;
        return 1;
    }
}
