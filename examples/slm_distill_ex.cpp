// The contents of this file are in the public domain.
// See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This is an example illustrating how to raise a new model by distillation: a student of
    a shape you choose, trained on what a larger teacher predicts over a corpus of your own.

    WHY DISTILL RATHER THAN TRAIN FROM SCRATCH

        A hard label says the next token is X. A teacher's distribution says X, but Y was
        nearly as good, Z was plausible, and the remaining fifty thousand were not. That
        ordering over the vocabulary is not present in the corpus and no student could
        derive it from the text alone, which is why a student learns from a teacher far
        faster than from the same corpus on its own.

        The transfer is bounded by the corpus, though, and it is worth being clear about
        that: the teacher is only ever asked what it would predict at the positions the
        corpus contains. What survives is the intersection of what the teacher knows and
        what the corpus probes. Distillation carries generalization, not an encyclopedia.

    THE THREE STEPS, AND WHY THERE ARE THREE

        The student's shape is a set of compile-time constants, so it cannot be chosen at
        run time any more than an imported model's can. The flow is therefore the same one
        slm_gguf_import_ex established: emit a header, rebuild, then work.

        1. Emit the student's header. Its vocabulary is the teacher's, which is not a
           choice: distillation compares distributions position by position, so the two
           models must number their tokens identically.

               slm_distill_ex --teacher t.gguf --emit-student --layers 12 --width 512
                              --heads 8 --kv-heads 2 --ffn 1536 --out-prefix student

        2. Put the header next to this file, rebuild, then record what the teacher
           predicts. This
           pass is the expensive one, hours on a large teacher, and it is paid once per
           corpus rather than once per student: the trace file is an asset, and several
           students of different shapes can be raised on it without the teacher running
           again.

               slm_distill_ex --teacher t.gguf --corpus corpus.txt --record traces.bin
                              --window 512 --top-k 64

        3. Train the student on the traces. Neither the teacher nor the corpus is needed
           again.

               slm_distill_ex --traces traces.bin --train --epochs 3 --out student.dat

        What comes out is an archive slm_gguf_import_ex reads, so the student can be
        served, chatted with, and fine-tuned with everything already in this library.

    THE WINDOW IS THE STUDENT'S, NOT THE TEACHER'S

        A teacher that reads four thousand tokens can teach a student that reads one
        thousand, but the recorded windows must fit what the student can read. The window
        is therefore fixed before the recording pass and travels in the trace header, where
        step 3 checks it against the compiled student and refuses a mismatch rather than
        training on windows it will truncate.
*/

#include <chrono>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <dlib/cmd_line_parser.h>
#include <dlib/data_io.h>
#include <dlib/dnn.h>

/* The student header is detected rather than declared, as slm_gguf_import_ex does for the
   imported model: step 1, before any header exists, always compiles; once the header sits
   next to this file, or anywhere on the include path, the next build of the target enables
   the training command by itself. A header created after a previous build is not tracked as
   a dependency of the old object file, so rebuild the target explicitly after generating
   it. An external definition of either macro keeps priority over the detection. */
#ifndef STUDENT_MODEL_HEADER
#  define STUDENT_MODEL_HEADER "slm_student_model.h"
#endif
#if !defined(WITH_STUDENT_MODEL) && defined(__has_include)
#  if __has_include(STUDENT_MODEL_HEADER)
#    define WITH_STUDENT_MODEL 1
#  endif
#endif

#ifdef WITH_STUDENT_MODEL
#  include STUDENT_MODEL_HEADER
#endif

using namespace std;
using namespace dlib;

const unsigned long IGNORE_LABEL = 0xFFFFFFFF;

// ---------------------------------------------------------------------------------------

/* The student's shape, as the command line describes it.

   Everything here is free except the vocabulary, which the teacher fixes. That is the whole
   point of a model factory: the objective and the data are given, the architecture is the
   variable. */
struct student_shape
{
    long layers = 12;
    long width = 512;
    long heads = 8;
    long kv_heads = 2;
    long ffn = 0;          // 0 derives the usual 8/3 ratio
    double rope_base = 10000.0;
    double rms_eps = 1e-5;
    long experts = 0;      // 0 keeps the dense stack, > 1 routes to a mixture
    long experts_used = 0; // 0 lets the configuration derive it
};

static model_spec spec_for_student(const student_shape& s, const model_spec& teacher)
{
    model_spec out;
    out.arch_name = "llama";
    out.model_name = "Distilled " + std::to_string(s.width) + "x" + std::to_string(s.layers);
    out.n_layers = s.layers;
    out.n_heads = s.heads;
    out.n_kv_heads = s.kv_heads > 0 ? s.kv_heads : s.heads;
    out.d_model = s.width;
    out.head_dim = s.width / s.heads;
    out.d_ffn = s.ffn > 0 ? s.ffn : (s.width * 8 / 3);
    out.rms_eps = s.rms_eps;
    out.rope_freq_base = s.rope_base;

    /* The vocabulary is the teacher's, and so are the special tokens: the student will be
       driven by the teacher's tokenizer for the rest of its life. */
    out.vocab_size = teacher.vocab_size;
    out.tied_embeddings = false;
    out.quantized = false;

    const long g = gcd_long(out.d_ffn, out.d_model);
    out.ffn_num = out.d_ffn / g;
    out.ffn_den = out.d_model / g;
    return out;
}

/* Writes the student's header by hand when it is a mixture of experts.

   emit_header knows one shape, the dense decoder, because that is the only one a set of
   GGUF weights can be repacked into. A student has no weights to repack: it starts empty,
   and nothing prevents it from having an inside its teacher does not share. That is the
   whole appeal of distilling on logits rather than on hidden states, and refusing it here
   would throw away the only architectural freedom the method offers. */
static void emit_moe_student_header(const model_spec& s, const std::string& path,
    const student_shape& shape, const std::string& ns)
{
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write " + path);

    out << "// Generated by slm_distill_ex. Do not edit.\n"
        << "// Student of " << s.model_name << ", raised by distillation.\n"
        << "#ifndef SLM_STUDENT_MODEL_H_\n#define SLM_STUDENT_MODEL_H_\n\n"
        << "#include <dlib/dnn.h>\n\n"
        << "#define DLIB_STUDENT_IS_MOE 1\n\n"
        << "namespace " << ns << "\n{\n"
        << "    constexpr long VOCAB_SIZE    = " << s.vocab_size << ";\n"
        << "    constexpr long NUM_LAYERS    = " << s.n_layers << ";\n"
        << "    constexpr long NUM_HEADS     = " << s.n_heads << ";\n"
        << "    constexpr long NUM_KVHEADS   = " << s.n_kv_heads << ";\n"
        << "    constexpr long EMBEDDING_DIM = " << s.d_model << ";\n"
        << "    constexpr long NUM_EXPERTS   = " << shape.experts << ";\n"
        << "    constexpr long EXPERTS_USED  = " << shape.experts_used << ";\n\n"
        << "    using config = dlib::gqa_moe_transformer_config<\n"
        << "        VOCAB_SIZE, NUM_LAYERS, NUM_HEADS, NUM_KVHEADS, EMBEDDING_DIM,\n"
        << "        NUM_EXPERTS, EXPERTS_USED>;\n"
        << "}\n\n#endif // SLM_STUDENT_MODEL_H_\n";
}

static void check_shape(const student_shape& s)
{
    if (s.width % s.heads != 0)
        throw std::runtime_error("the width must divide evenly among the heads: "
            + std::to_string(s.width) + " over " + std::to_string(s.heads));
    if (s.kv_heads > 0 && s.heads % s.kv_heads != 0)
        throw std::runtime_error("the query heads must group evenly over the key-value "
            "heads: " + std::to_string(s.heads) + " over " + std::to_string(s.kv_heads));
    if (s.layers <= 0 || s.width <= 0 || s.heads <= 0)
        throw std::runtime_error("layers, width and heads must all be positive");
    if (s.experts == 1)
        throw std::runtime_error("a mixture of one expert is a dense stack written the "
            "long way; pass --experts 0 or --experts 4");
    if (s.experts > 1 && s.experts_used > s.experts)
        throw std::runtime_error("more experts used than there are: "
            + std::to_string(s.experts_used) + " of " + std::to_string(s.experts));
}

// ---------------------------------------------------------------------------------------

/* Step 1: describe the student and write its header. */
static int run_emit_student(const std::string& teacher_path, const student_shape& shape,
    const std::string& prefix, const std::string& ns)
{
    check_shape(shape);

    cout << "Reading teacher: " << teacher_path << "\n";
    gguf_reader g(teacher_path);
    const model_spec teacher = detect_model(g);
    cout << "  name       : " << teacher.model_name << "\n"
         << "  vocabulary : " << teacher.vocab_size << "\n"
         << "  layers     : " << teacher.n_layers
         << ", width " << teacher.d_model << "\n";

    hf_tokenizer tok;
    tok.load_from_gguf(g);

    model_spec student = spec_for_student(shape, teacher);
    student.n_experts = shape.experts;
    student.n_experts_used = shape.experts_used;
    const std::string header = prefix + ".h";
    if (shape.experts > 1) emit_moe_student_header(student, header, shape, ns);
    else                   emit_header(student, header, ns);
    cout << "\nStudent header : " << header << "\n"
         << "  layers     : " << student.n_layers << "\n"
         << "  width      : " << student.d_model
         << " (head dim " << student.head_dim << ")\n"
         << "  heads      : " << student.n_heads
         << " (key-value " << student.n_kv_heads << ")\n"
         << "  ffn hidden : " << student.d_ffn << "\n"
         << "  vocabulary : " << student.vocab_size << ", from the teacher\n"
         << "  feed-forward: "
         << (shape.experts > 1
                ? "mixture of " + std::to_string(shape.experts) + " experts"
                : std::string("dense")) << "\n";

    const std::string tok_path = prefix + "_tokenizer.dat";
    serialize(tok_path) << tok;
    cout << "Tokenizer      : " << tok_path << "\n";
    cout << "  fingerprint  : " << tokenizer_fingerprint(tok) << "\n";

    /* A rough count, so that a shape can be judged before an hour is spent on it. The
       embedding table dominates a small student and is easy to forget. */
    const long emb = student.vocab_size * student.d_model;
    const long per_layer = 4 * student.d_model * student.d_model + 3 * student.d_model * student.d_ffn;
    cout << "\nAbout " << (2 * emb + student.n_layers * per_layer)
         << " parameters, of which " << (2 * emb) << " in the embedding table and the "
            "output projection.\n";

    /* Serving the student is a second build, not a second format.

       A program that runs a compiled model is compiled for one geometry. slm_gguf_import_ex
       built for a teacher cannot read a student's archive any more than it could read a
       stranger's: the weights are fine, the network they are poured into is not. Emitting
       the same student a second time under the name that program expects makes it
       readable, which is why the namespace is an option rather than a constant. */
    if (ns != "imported_model")
        cout << "\nTo serve this student later, emit it again for the serving program:\n"
             << "  --emit-student ... --header-namespace imported_model "
                "--out-prefix slm_imported_model\n"
             << "then rebuild slm_gguf_import_ex against that header.\n";
    cout << "\nCopy " << header << " next to the examples and rebuild to record.\n";
    return 0;
}

// ---------------------------------------------------------------------------------------

/* Step 2: run the teacher over the corpus and keep what it predicted. */
static int run_record(const std::string& teacher_path, const std::string& corpus_path,
    const std::string& out_path, long window, long top_k, long limit, bool supervised,
    bool mask_prompt, const std::string& system_prompt)
{
    cout << "Reading teacher: " << teacher_path << "\n";
    gguf_reader g(teacher_path);
    const model_spec spec = detect_model(g);
    hf_tokenizer tok;
    tok.load_from_gguf(g);
    cout << "  " << spec.model_name << ", vocabulary " << spec.vocab_size << "\n";

    std::vector<matrix<int, 0, 1>> X;
    std::vector<matrix<unsigned long, 0, 1>> Y;

    if (supervised)
    {
        const chat_template_formatter fmt = chat_template_formatter::for_tokenizer(
            tok, spec.model_name);
        std::vector<chat_record> records;
        load_chat_records(corpus_path, records);
        if (limit > 0 && static_cast<long>(records.size()) > limit)
            records.resize(static_cast<size_t>(limit));
        if (!system_prompt.empty())
            for (chat_record& r : records) r.system = system_prompt;
        cout << "  records    : " << records.size() << "\n";

        const std::vector<supervised_example> examples =
            encode_supervised_examples(tok, fmt, records);

        if (mask_prompt)
        {
            build_supervised_finetuning_dataset(examples, window,
                tok.pad_id() >= 0 ? tok.pad_id() : 0, IGNORE_LABEL,
                sequence_overflow_policy::truncate_prompt_head, X, Y);
        }
        else
        {
            /* Rendered through the teacher's chat template, then read straight through.

               Masking the prompt is right for supervised fine-tuning, where a model must
               not be taught to produce the user's question. It is wrong here. The teacher's
               prediction at a prompt position is language modelling of exactly the same
               quality as anywhere else, and dropping it throws away two fifths of a
               recording that costs hours. Worse, the markers that open and close a turn sit
               in the masked part: masked, the student never sees the teacher predict them,
               and never learns where an assistant's answer begins or ends.

               So the template is applied, and then the whole exchange is treated as text. */
            std::vector<std::vector<int>> flat;
            flat.reserve(examples.size());
            for (const supervised_example& e : examples)
            {
                std::vector<int> one;
                one.reserve(e.prompt.size() + e.response.size());
                one.insert(one.end(), e.prompt.begin(), e.prompt.end());
                one.insert(one.end(), e.response.begin(), e.response.end());
                flat.push_back(std::move(one));
            }
            build_causal_lm_dataset(flat, window, window,
                tok.pad_id() >= 0 ? tok.pad_id() : 0, IGNORE_LABEL, false, X, Y);
        }
    }
    else
    {
        std::vector<std::string> documents;
        load_document_corpus(corpus_path, documents);
        if (limit > 0 && static_cast<long>(documents.size()) > limit)
            documents.resize(static_cast<size_t>(limit));
        const std::vector<std::vector<int>> ids = tokenize_documents(tok, documents, true);
        size_t tokens = 0;
        for (const auto& d : ids) tokens += d.size();
        cout << "  documents  : " << documents.size() << ", " << tokens << " tokens\n";
        build_causal_lm_dataset(ids, window, window,
            tok.pad_id() >= 0 ? tok.pad_id() : 0, IGNORE_LABEL, true, X, Y);
    }

    if (X.empty()) { cerr << "Error: the corpus yielded no window.\n"; return 1; }

    const double bytes = distillation_bytes_per_window(window, top_k) * X.size();
    cout << "  windows    : " << X.size() << " of " << window << " tokens\n"
         << "  top-k      : " << top_k << "\n"
         << "  file size  : about " << (bytes / (1024.0 * 1024.0)) << " MB\n";

    cout << "\nLoading the teacher's weights...\n";
    runtime_transformer teacher;
    teacher.load(g, spec, gguf_load_options());

    distillation_header head;
    head.tokenizer_id = tokenizer_fingerprint(tok);
    head.teacher_name = spec.model_name;
    head.vocab_size = spec.vocab_size;
    head.top_k = top_k;
    head.window_len = window;

    const auto started = std::chrono::steady_clock::now();
    distillation_writer out(out_path, head);
    long last_percent = -1;
    record_distillation_traces(teacher, X, Y, top_k, IGNORE_LABEL, out,
        [&](long done, long total)
        {
            const long pct = 100 * done / total;
            if (pct != last_percent)
            {
                last_percent = pct;
                cout << "\r  recording  : " << pct << "%   " << std::flush;
            }
        });
    const long written = out.finish();
    const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - started).count();

    cout << "\r  recorded   : " << written << " windows in " << elapsed << " s\n"
         << "Written to " << out_path << "\n";
    return 0;
}

// ---------------------------------------------------------------------------------------

#ifdef WITH_STUDENT_MODEL

/* The student's own network, with the distillation loss in place of the ordinary one.

   config::network_type would give the cross-entropy head, which is what a plain training
   run wants; here the head has to read soft targets. Everything below it is identical, so
   what this network serializes is exactly what slm_gguf_import_ex deserializes: the
   student is a model of this library from its first step, not a special case. */
#ifdef DLIB_STUDENT_IS_MOE
/* The mixture keeps its own stack; only the head changes, and it changes the same way. */
using student_head = linear<student_model::VOCAB_SIZE,
    rms_norm<typename student_model::config::template transformer_stack<
        student_model::NUM_LAYERS, dlib::training_mode_tag,
        embeddings<student_model::VOCAB_SIZE, student_model::EMBEDDING_DIM,
        input<matrix<int, 0, 1>>>>>>;
#else
using student_head = linear_no_bias<student_model::VOCAB_SIZE,
    rms_norm<typename student_model::config::subnet>>;
#endif
using train_net = loss_distillation_per_token<student_head>;

/* Step 3: train the student on the recording. */
static int run_train(const std::vector<std::string>& trace_paths, const std::string& tok_path,
    const std::string& out_path, long epochs, long batch, double lr, double temperature,
    double alpha, long patience, const std::string& sync_file)
{
    hf_tokenizer tok;
    { std::ifstream fin(tok_path, std::ios::binary);
      if (!fin) { cerr << "Error: cannot open " << tok_path << "\n"; return 1; }
      deserialize(tok, fin); }

    /* Several recordings are read side by side rather than one after the other.

       A student raised on prose and then on conversation forgets the prose while it learns
       the conversation: that is the ordinary catastrophic forgetting, and chaining two
       corpora walks straight into it. Interleaving them means every part of the training
       sees both, so the student ends up able to do both.

       Batches are taken from each recording in turn. The overall proportion is therefore
       exactly the ratio of the file sizes, which is how the mixture is chosen: record more
       of what should weigh more. */
    std::vector<std::unique_ptr<distillation_reader>> traces;
    for (const std::string& path : trace_paths)
    {
        traces.emplace_back(new distillation_reader(path));
        const distillation_header& h = traces.back()->header();
        cout << "Traces      : " << path << "\n"
             << "  teacher   : " << h.teacher_name << "\n"
             << "  vocabulary: " << h.vocab_size << "\n"
             << "  window    : " << h.window_len << ", top-k " << h.top_k << "\n";

        try { traces.back()->check_tokenizer(tok); }
        catch (const std::exception& e) { cerr << "Error: " << e.what() << "\n"; return 1; }

        if (h.vocab_size != student_model::VOCAB_SIZE)
        {
            cerr << "Error: '" << path << "' carries a vocabulary of " << h.vocab_size
                 << " and the compiled student expects " << student_model::VOCAB_SIZE
                 << ".\nRegenerate the student header from the same teacher.\n";
            return 1;
        }
        if (h.vocab_size != traces.front()->header().vocab_size)
        {
            cerr << "Error: '" << path << "' and '" << trace_paths.front()
                 << "' disagree on the vocabulary; they cannot be mixed.\n";
            return 1;
        }
    }
    const distillation_header& head = traces.front()->header();
    /* No window check here, and it is worth saying why rather than leaving a silence.

       Neither configuration carries a compile-time context length: the positional tables
       are sized when the network first runs, so a student reads whatever window it is
       given. What bounds the window is memory and the quadratic cost of attention, not the
       type. The trace header still records the window, because a reader has to know what
       shape the file holds, but there is nothing to refuse it against. */

    train_net net;
    net.loss_details().set_ignore_index(static_cast<long>(IGNORE_LABEL));
    net.loss_details().set_temperature(temperature);
    net.loss_details().set_alpha(alpha);

    cout << "\nStudent     : " << student_model::config::model_info::describe() << "\n"
         << "  soft/hard : alpha " << alpha << ", temperature " << temperature << "\n";

    dnn_trainer<train_net, adamw> trainer(net, adamw(0.01f, 0.9f, 0.95f));
    trainer.set_learning_rate(lr);
    trainer.set_min_learning_rate(lr * 1e-3);
    trainer.set_mini_batch_size(static_cast<unsigned long>(batch));
    trainer.set_learning_rate_shrink_factor(0.1);
    trainer.set_iterations_without_progress_threshold(patience);
    trainer.be_verbose();
    if (!sync_file.empty())
        trainer.set_synchronization_file(sync_file, std::chrono::minutes(10));

    const auto started = std::chrono::steady_clock::now();
    std::vector<distillation_window> chunk;
    std::vector<matrix<int, 0, 1>> bx;
    std::vector<distillation_target> by;

    for (long e = 0; e < epochs; ++e)
    {
        for (auto& r : traces) r->rewind();
        std::vector<bool> live(traces.size(), true);
        size_t turn = 0;
        long seen = 0;
        /* Read a batch's worth at a time: a recording of a real corpus does not fit in
           memory, and holding one batch is all the training loop ever needs. */
        while (std::find(live.begin(), live.end(), true) != live.end())
        {
            const size_t who = turn % traces.size();
            turn++;
            if (!live[who]) continue;
            if (traces[who]->read(batch, chunk) == 0) { live[who] = false; continue; }

            bx.clear();
            by.clear();
            for (const distillation_window& w : chunk)
            {
                bx.push_back(w.tokens);
                distillation_target t;
                t.hard = w.hard;
                t.ids = w.ids;
                t.logits = w.logits;
                by.push_back(std::move(t));
            }
            network_context::set_learning_rate(trainer.get_learning_rate());
            trainer.train_one_step(bx, by);
            seen += static_cast<long>(chunk.size());
            if (trainer.get_learning_rate() < trainer.get_min_learning_rate()) break;
        }
        trainer.get_net(force_flush_to_disk::no);
        cout << "  epoch " << (e + 1) << "/" << epochs
             << "  windows " << seen
             << "  learning rate " << trainer.get_learning_rate()
             << "  average loss " << trainer.get_average_loss() << "\n";
        trainer.clear_average_loss();
        if (trainer.get_learning_rate() < trainer.get_min_learning_rate())
        {
            cout << "  stopped early: the learning rate reached its floor\n";
            break;
        }
    }

    trainer.get_net();
    net.clean();
    const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - started).count();
    cout << "  trained in  : " << elapsed << " s\n";

    model_archive_info info;
    info.model_name = std::string("Distilled from ") + head.teacher_name;
    info.has_vision = false;
    save_model_archive(out_path, info, net.subnet(), tok);
    cout << "\nWritten to " << out_path << "\n";
#ifdef DLIB_STUDENT_IS_MOE
    cout << "This student routes to a mixture of experts, which slm_gguf_import_ex cannot\n"
            "read: its compiled network is dense. Serving a mixture needs a serving program\n"
            "built against this same header, which does not exist yet.\n";
#else
    cout << "It loads with slm_gguf_import_ex --load " << out_path << " --chat\n";
#endif
    return 0;
}

#endif // WITH_STUDENT_MODEL

// ---------------------------------------------------------------------------------------

int main(int argc, char** argv)
{
    try
    {
        command_line_parser parser;
        parser.add_option("teacher", "Teacher model, a .gguf container", 1);
        parser.add_option("emit-student", "Write the student header and stop");
        parser.add_option("out-prefix", "Prefix of the generated files (default: slm_student_model)", 1);
        parser.add_option("header-namespace", "Namespace of the generated header; use "
            "imported_model to build a serving program against it (default: student_model)", 1);
        parser.add_option("layers", "Student blocks (default: 12)", 1);
        parser.add_option("width", "Student embedding width (default: 512)", 1);
        parser.add_option("heads", "Student attention heads (default: 8)", 1);
        parser.add_option("kv-heads", "Student key-value heads; 0 means as many as heads (default: 2)", 1);
        parser.add_option("ffn", "Student feed-forward width; 0 derives the usual 8/3 ratio", 1);
        parser.add_option("rope-base", "RoPE frequency base of the student (default: 10000)", 1);
        parser.add_option("experts", "Student experts per block; 0 keeps a dense stack (default: 0)", 1);
        parser.add_option("experts-used", "Experts routed to per token; 0 lets the configuration derive it", 1);

        parser.add_option("corpus", "Text the teacher will be questioned over", 1);
        parser.add_option("supervised", "Read the corpus as question-and-answer records");
        parser.add_option("mask-prompt", "Record only the answer positions of a supervised corpus; off by default");
        parser.add_option("system", "System block forced on every supervised record", 1);
        parser.add_option("record", "Where to write the traces", 1);
        parser.add_option("window", "Tokens per window; must fit the student (default: 512)", 1);
        parser.add_option("top-k", "Teacher entries kept per position (default: 64)", 1);
        parser.add_option("limit", "Stop after this many documents or records; 0 reads them all", 1);

        parser.add_option("traces", "Recording to train on; repeat it to mix several, "
            "which is how a student learns prose and conversation at once instead of "
            "forgetting one while it learns the other", 1);
        parser.add_option("tokenizer", "Tokenizer written beside the student header", 1);
        parser.add_option("train", "Train the student on the traces");
        parser.add_option("out", "Where to write the student (default: student.dat)", 1);
        parser.add_option("epochs", "Passes over the traces (default: 3)", 1);
        parser.add_option("batch-size", "Windows per step (default: 4)", 1);
        parser.add_option("learning-rate", "Initial learning rate (default: 3e-4)", 1);
        parser.add_option("temperature", "Softens both distributions (default: 2)", 1);
        parser.add_option("alpha", "Weight of the teacher against the corpus (default: 0.9)", 1);
        parser.add_option("patience", "Steps without progress before the rate is lowered (default: 2000)", 1);
        parser.add_option("sync", "Trainer synchronization file", 1);
        parser.add_option("h", "Display this help message");
        parser.parse(argc, argv);

        if (parser.option("h") || argc == 1)
        {
            cout << "Raise a model by distillation, in three steps.\n\n";
            parser.print_options();
            cout << "\nSteps:\n"
                 << "  1. slm_distill_ex --teacher t.gguf --emit-student --layers 12 "
                    "--width 512 --out-prefix slm_student_model\n"
                 << "     then copy slm_student_model.h next to the examples and rebuild\n"
                 << "  2. slm_distill_ex --teacher t.gguf --corpus corpus.txt "
                    "--record traces.bin --window 512 --top-k 64\n"
                 << "  3. slm_distill_ex --traces flat.bin --traces chat.bin --tokenizer "
                    "slm_student_model_tokenizer.dat --train --out student.dat\n";
            return 0;
        }

        const std::string prefix = get_option(parser, "out-prefix",
            std::string("slm_student_model"));

        if (parser.option("emit-student"))
        {
            if (!parser.option("teacher"))
            { cerr << "Error: --emit-student needs --teacher for the vocabulary.\n"; return 1; }
            student_shape shape;
            shape.layers = get_option(parser, "layers", 12L);
            shape.width = get_option(parser, "width", 512L);
            shape.heads = get_option(parser, "heads", 8L);
            shape.kv_heads = get_option(parser, "kv-heads", 2L);
            shape.ffn = get_option(parser, "ffn", 0L);
            shape.rope_base = get_option(parser, "rope-base", 10000.0);
            shape.experts = get_option(parser, "experts", 0L);
            shape.experts_used = get_option(parser, "experts-used", 0L);
            return run_emit_student(parser.option("teacher").argument(), shape, prefix,
                get_option(parser, "header-namespace", std::string("student_model")));
        }

        if (parser.option("record"))
        {
            if (!parser.option("teacher") || !parser.option("corpus"))
            { cerr << "Error: --record needs --teacher and --corpus.\n"; return 1; }
            return run_record(parser.option("teacher").argument(),
                parser.option("corpus").argument(),
                parser.option("record").argument(),
                get_option(parser, "window", 512L),
                get_option(parser, "top-k", 64L),
                get_option(parser, "limit", 0L),
                parser.option("supervised") != 0,
                parser.option("mask-prompt") != 0,
                get_option(parser, "system", std::string()));
        }

        if (parser.option("train"))
        {
#ifdef WITH_STUDENT_MODEL
            if (!parser.option("traces") || !parser.option("tokenizer"))
            { cerr << "Error: --train needs --traces and --tokenizer.\n"; return 1; }
            std::vector<std::string> trace_paths;
            for (unsigned long i = 0; i < parser.option("traces").count(); ++i)
                trace_paths.push_back(parser.option("traces").argument(0, i));
            return run_train(trace_paths,
                parser.option("tokenizer").argument(),
                get_option(parser, "out", std::string("student.dat")),
                get_option(parser, "epochs", 3L),
                get_option(parser, "batch-size", 4L),
                get_option(parser, "learning-rate", 3e-4),
                get_option(parser, "temperature", 2.0),
                get_option(parser, "alpha", 0.9),
                get_option(parser, "patience", 2000L),
                get_option(parser, "sync", std::string()));
#else
            cerr << "This build has no student header compiled in.\n"
                    "Emit one with --emit-student, copy slm_student_model.h next to the\n"
                    "examples, and rebuild this target.\n";
            return 1;
#endif
        }

        cout << "Nothing to do. Run with -h for the three steps.\n";
        return 0;
    }
    catch (const std::exception& e)
    {
        cerr << "\nFATAL: " << e.what() << "\n";
        return 1;
    }
}
