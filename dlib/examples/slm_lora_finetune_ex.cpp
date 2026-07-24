/*!
    @file slm_lora_finetune_ex.cpp
    @brief Parameter-efficient fine-tuning of an imported model, LoRA or DoRA.

    Two stages, each useful on its own and chainable into a pipeline. Which one to run is
    decided by which corpus is supplied, not by a mode switch, because the choice really is
    a property of the objective:

      knowledge alignment (--corpus)
          A causal pass over plain domain text. Teaches the model vocabulary, phrasing and
          facts it did not have, without touching the way it answers. This is the stage for
          scattered internal documents, a standards corpus, a product catalogue. Run alone,
          it leaves an instruct model's conversational machinery intact.

      task alignment (--dataset)
          A supervised pass over question-and-answer records, scored on the answer only.
          Teaches the model how to respond rather than what to know: answer shape, register,
          refusal policy, a security guardrail. Run alone, it leaves the model's general
          knowledge where it was.

      both
          The full pipeline, in that order. The adapters of the first stage are merged into
          the weights before the second starts, which is the one step a sequential
          fine-tuning cannot skip: resuming on saved adapters alone trains the adapters and
          leaves the base exactly where the first stage found it, so its knowledge is never
          carried forward.

    The corpora come from the two preparation programs of this series:

      nist_corpus_prepare.py  ->  the plain document corpus read by --corpus
      cve_qa_prepare.py       ->  the question-and-answer records read by --dataset

    Both write the sentinel-separated format that language_model_data.h reads, and neither
    applies a conversation template: that is done here, with the model's own formatter, so
    that the prompt trained on is the byte-for-byte prompt the inference path will build.

    Usage:
      slm_lora_finetune_ex --load model.dat --dataset cve_qa.txt --dry-run
      slm_lora_finetune_ex --load model.dat --dataset cve_qa.txt --valid cve_qa_valid.txt \
                           --lora-rank 8 --epochs 2 --out model_cve.dat
      slm_lora_finetune_ex --load model.dat --corpus nist_corpus.txt --dataset cve_qa.txt \
                           --lora-method dora --out model_full.dat
!*/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <dlib/cmd_line_parser.h>
#include <dlib/dnn.h>
#include <dlib/tokenizer/hf_tokenizer.h>
#include <dlib/tokenizer/chat_template.h>

#if __has_include("slm_imported_model.h")
#  include "slm_imported_model.h"
#  define WITH_IMPORTED_MODEL 1
#endif

using namespace std;
using namespace dlib;

#ifdef WITH_IMPORTED_MODEL

/* Training network of the imported architecture. network_type<true> and network_type<false>
   are the same type: the attention layer selects its behaviour at run time, so the archive
   written by the import program loads here unchanged. */
using train_net = imported_model::network_type<true>;

/* Label reserved for the positions that carry no gradient: the prompt of a supervised
   example, and the padding of a short window. It has to be outside the vocabulary, since
   any in-vocabulary value would be a legitimate target somewhere. */
static const unsigned long IGNORE_LABEL = static_cast<unsigned long>(imported_model::VOCAB_SIZE);

/* The report structures embed newlines, so their continuation lines have to be indented
   here or the output loses its alignment as soon as one of them grows a second line. */
static std::string indent(const std::string& text, const std::string& prefix = "  ")
{
    std::string out = prefix;
    for (char c : text) { out += c; if (c == '\n') out += prefix; }
    return out;
}

struct stage_settings
{
    long   window = 0;            // 0 asks for a suggestion from the data
    double coverage = 0.95;
    long   max_window = 2048;
    long   epochs = 1;
    long   batch_size = 4;
    double learning_rate = 2e-5;
    double min_learning_rate = 1e-6;
    long   valid_batch = 1;       // validation runs one window at a time by default
    long   patience = 0;          // 0 lets the trainer derive one from the set size
    long   limit = 0;             // 0 keeps the whole set
    double weight_decay = 0.0;    // decoupled, and off by default on adapters
    double beta1 = 0.9;
    double beta2 = 0.999;
    double shrink = 0.1;
    double valid_fraction = 0.05;   // held out when no validation file is supplied
    bool   dry_run = false;
};

using adapter_settings = adapter_plan;

// ---------------------------------------------------------------------------------------

/* Positions processed per epoch, which is what an epoch actually costs: the transformer
   spends the same work on a padded position as on a scored one. Printing it lets a run be
   budgeted from a timed subset instead of being discovered. */
static void report_cost(size_t windows, const stage_settings& st, size_t parameters)
{
    const double positions = static_cast<double>(windows) * st.window;
    cout << "  cost        : " << static_cast<long long>(positions)
         << " positions per epoch, " << (windows / std::max<long>(1, st.batch_size))
         << " steps\n";

    /* The steady state is what decides whether a run survives, and it is set by the whole
       parameter blob rather than by the trainable share: the trainer allocates a gradient
       for every layer, and the optimizer two moments for every layer it may move. Printed
       because the failure it causes is a process kill with no message of its own. */
    const double gb = 4.0 / 1e9;
    cout << "  memory      : about "
         << static_cast<long>(parameters * gb + 0.5) << " GB of weights, as much again in\n"
         << "                gradients, and two optimizer moments on the adapted layers,\n"
         << "                before activations. Lower --window or --batch-size first.\n";
}

static void report_adapters(train_net& net, const adapter_settings& ad)
{
    const size_t layers = configure_network_adapters(net, ad);
    freeze_all_but_adapters(net);

    const trainable_counts counts = count_trainable_parameters(net);
    cout << "  adapters    : " << adapter_method_name(ad.method)
         << ", rank " << ad.rank << ", alpha " << ad.alpha
         << " on " << (ad.attention_query ? "Q" : "") << (ad.attention_value ? "V" : "")
         << (ad.projection ? "+FFN" : "")
         << " over " << layers << " layers\n"
         << indent(counts.describe()) << "\n";
    if (counts.trainable == 0)
        cout << "  warning     : nothing is trainable; check the rank and the targets\n";
}

/* One training run over an already built dataset. The trainer is configured the same way
   for both stages: what distinguishes them is the labels, not the optimization. */
static void run_training(train_net& net,
    std::vector<matrix<int, 0, 1>>& X,
    std::vector<matrix<unsigned long, 0, 1>>& Y,
    std::vector<matrix<int, 0, 1>>& VX,
    std::vector<matrix<unsigned long, 0, 1>>& VY,
    const stage_settings& st,
    const std::string& sync_file)
{
    net.loss_details().set_ignore_index(static_cast<long>(IGNORE_LABEL));

    /* AdamW rather than Adam: its weight decay is decoupled from the adaptive step, so the
       decay a layer receives no longer depends on the scale of its gradients. On an
       adapter that matters, since A and B carry gradients of very different magnitudes.
       The default decay here is zero: an adapter is already a low-rank constraint, and
       pulling B back towards the origin pulls the whole update towards doing nothing. */
    dnn_trainer<train_net, adamw> trainer(net,
        adamw(static_cast<float>(st.weight_decay),
              static_cast<float>(st.beta1), static_cast<float>(st.beta2)));
    trainer.set_learning_rate(st.learning_rate);
    trainer.set_min_learning_rate(st.min_learning_rate);
    trainer.set_mini_batch_size(st.batch_size);
    trainer.set_learning_rate_shrink_factor(st.shrink);
    trainer.set_iterations_without_progress_threshold(
        st.patience > 0 ? st.patience : std::max<long>(200, static_cast<long>(X.size() / 4)));
    trainer.be_verbose();
    if (!sync_file.empty())
        trainer.set_synchronization_file(sync_file, std::chrono::minutes(10));

    /* network_context carries the inference-time switches. Training must see them off, or
       the attention layers would try to read a KV cache that no generation loop is
       filling. The optimizer parameters go the same way, for the layers that read them
       from the context rather than from the solver. */
    network_context::reset();
    network_context::set_optimizer_params(st.weight_decay, st.beta1, st.beta2);
    network_context::set_learning_rate(st.learning_rate);

    /* Explicit epochs over shuffled mini-batches rather than trainer.train(), which runs
       until the learning rate bottoms out: a fine-tuning stage is budgeted in passes over
       a known set, and the caller wants to see each pass land. The trainer still lowers
       the rate on plateau and still stops the stage when it reaches the floor. */
    const auto started = std::chrono::steady_clock::now();
    const size_t batch = static_cast<size_t>(std::max<long>(1, st.batch_size));

    for (long epoch = 0; epoch < st.epochs; ++epoch)
    {
        shuffle_training_dataset(X, Y, 1 + static_cast<unsigned long>(epoch));
        for (size_t i = 0; i + batch <= X.size(); i += batch)
        {
            trainer.train_one_step(X.begin() + i, X.begin() + i + batch, Y.begin() + i);
            network_context::set_learning_rate(trainer.get_learning_rate());
            if (trainer.get_learning_rate() < trainer.get_min_learning_rate()) break;
        }
        const double train_loss = trainer.get_average_loss();
        cout << "  epoch " << (epoch + 1) << "/" << st.epochs
             << "  learning rate " << trainer.get_learning_rate()
             << "  average loss " << train_loss << "\n";
        trainer.clear_average_loss();

        if (!VX.empty())
        {
            /* Validation loss through the same loss layer, which is the only comparison
               that means anything: a perplexity computed elsewhere would not share the
               ignore index and would score the prompt positions too. get_net() doubles as
               the synchronization barrier with the trainer's background thread. */
            /* Batched like the training steps. compute_loss over the whole validation set
               forwards it as a single batch, and the output head alone then allocates
               samples x window x vocabulary floats: on a 150k vocabulary that is gigabytes
               for a handful of windows, on top of the weights, their gradients and the two
               optimizer moments. Chunking keeps the peak at the size of a training step. */
            train_net& current = trainer.get_net(force_flush_to_disk::no);
            const size_t vbatch = static_cast<size_t>(std::max<long>(1, st.valid_batch));
            double total = 0.0;
            size_t counted = 0;
            for (size_t i = 0; i < VX.size(); i += vbatch)
            {
                const size_t upto = std::min(i + vbatch, VX.size());
                total += current.compute_loss(VX.begin() + i, VX.begin() + upto,
                    VY.begin() + i) * static_cast<double>(upto - i);
                counted += upto - i;
            }
            const double v = counted ? total / counted : 0.0;
            cout << "  validation loss " << v;
            if (v > 1.5 * train_loss)
                cout << "  (well above the training loss: the stage is memorizing)";
            cout << "\n";
        }

        if (trainer.get_learning_rate() < trainer.get_min_learning_rate())
        {
            cout << "  stopped early: the learning rate reached its floor\n";
            break;
        }
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - started).count();

    trainer.get_net();   // flush the trainer's background thread into the network
    cout << "  trained in  : " << elapsed << " s\n";
}

// ---------------------------------------------------------------------------------------

static bool stage_knowledge(train_net& net, const hf_tokenizer& tok,
    const std::string& corpus_path, const adapter_settings& ad, stage_settings st,
    const std::string& sync_file)
{
    cout << "\n=== Knowledge alignment: " << corpus_path << "\n";

    std::vector<std::string> documents;
    load_document_corpus(corpus_path, documents);
    if (documents.empty()) { cerr << "  the corpus holds no document\n"; return false; }

    const std::vector<std::vector<int>> ids = tokenize_documents(tok, documents, true);
    size_t tokens = 0;
    for (const auto& d : ids) tokens += d.size();
    cout << "  documents   : " << documents.size() << ", " << tokens << " tokens\n";

    /* A plain corpus has no natural example length: the windows are cut from a continuous
       stream, so the window is a pure cost knob rather than something the data dictates. */
    if (st.window <= 0) st.window = std::min<long>(st.max_window, 512);

    std::vector<matrix<int, 0, 1>> X;
    std::vector<matrix<unsigned long, 0, 1>> Y;
    const dataset_report rep = build_causal_lm_dataset(ids, st.window, st.window,
        tok.pad_id() >= 0 ? tok.pad_id() : 0, IGNORE_LABEL, /*pack_documents=*/true, X, Y);
    cout << "  window      : " << st.window << "\n" << indent(rep.describe()) << "\n";
    if (X.empty()) { cerr << "  no window could be built\n"; return false; }

    report_cost(X.size(), st, count_trainable_parameters(net).total);
    report_adapters(net, ad);
    if (st.dry_run) { cout << "  dry run, no training\n"; return true; }

    std::vector<matrix<int, 0, 1>> VX;
    std::vector<matrix<unsigned long, 0, 1>> VY;
    run_training(net, X, Y, VX, VY, st, sync_file);
    return true;
}

static bool stage_task(train_net& net, const hf_tokenizer& tok,
    const chat_template_formatter& fmt, const std::string& system_prompt,
    const std::string& dataset_path, const std::string& valid_path,
    const adapter_settings& ad, stage_settings st, const std::string& sync_file)
{
    cout << "\n=== Task alignment: " << dataset_path << "\n";

    std::vector<chat_record> records;
    load_chat_records(dataset_path, records);
    if (records.empty()) { cerr << "  the dataset holds no record\n"; return false; }
    if (st.limit > 0 && static_cast<long>(records.size()) > st.limit)
    {
        /* Subsampled here rather than in the preparation script, so that a timing run and
           the full run read the same file and differ by one argument. */
        records.resize(static_cast<size_t>(st.limit));
        cout << "  limited to  : " << records.size() << " records\n";
    }
    if (!system_prompt.empty())
        for (chat_record& r : records) r.system = system_prompt;

    const std::vector<supervised_example> examples =
        encode_supervised_examples(tok, fmt, records);
    const length_profile profile = profile_lengths(examples);
    cout << indent(profile.describe()) << "\n";

    /* The window comes from the data rather than from the model's context capacity: an
       answer of two hundred tokens gains nothing from being trained in a window of two
       thousand, and attention cost grows with the square of that choice. */
    if (st.window <= 0)
    {
        st.window = suggest_window_length(profile, st.coverage, 64, st.max_window);
        cout << "  window      : " << st.window << " (suggested for "
             << static_cast<long>(100.0 * st.coverage + 0.5) << "% coverage, "
             << static_cast<long>(100.0 * profile.coverage_at(st.window) + 0.5)
             << "% actually covered)\n";
    }
    else
    {
        cout << "  window      : " << st.window << " (given, "
             << static_cast<long>(100.0 * profile.coverage_at(st.window) + 0.5)
             << "% covered)\n";
    }

    std::vector<matrix<int, 0, 1>> X;
    std::vector<matrix<unsigned long, 0, 1>> Y;
    const dataset_report rep = build_supervised_finetuning_dataset(examples, st.window,
        tok.pad_id() >= 0 ? tok.pad_id() : 0, IGNORE_LABEL,
        sequence_overflow_policy::truncate_prompt_head, X, Y);
    cout << indent(rep.describe()) << "\n";
    if (X.empty()) { cerr << "  no window could be built\n"; return false; }

    std::vector<matrix<int, 0, 1>> VX;
    std::vector<matrix<unsigned long, 0, 1>> VY;
    if (!valid_path.empty())
    {
        std::vector<chat_record> vrecords;
        load_chat_records(valid_path, vrecords);
        if (!system_prompt.empty())
            for (chat_record& r : vrecords) r.system = system_prompt;
        build_supervised_finetuning_dataset(
            encode_supervised_examples(tok, fmt, vrecords), st.window,
            tok.pad_id() >= 0 ? tok.pad_id() : 0, IGNORE_LABEL,
            sequence_overflow_policy::truncate_prompt_head, VX, VY);
        cout << "  validation  : " << VX.size() << " windows, from " << valid_path << "\n";
    }
    else if (st.valid_fraction > 0.0 && X.size() > 8)
    {
        /* Held out from the training set when no validation file is given, because the
           failure this stage is most likely to produce is invisible otherwise: a training
           loss that keeps falling while the model memorizes the answer shape and loses the
           ability to say anything else. The split is deterministic and shuffles first, so
           an ordered corpus does not put one range of the data on one side of it. */
        std::vector<matrix<int, 0, 1>> TX;
        std::vector<matrix<unsigned long, 0, 1>> TY;
        split_training_dataset(X, Y, st.valid_fraction, 1, TX, TY, VX, VY);
        X.swap(TX);
        Y.swap(TY);
        cout << "  validation  : " << VX.size() << " windows, held out from the training set\n";
    }

    report_cost(X.size(), st, count_trainable_parameters(net).total);
    report_adapters(net, ad);
    if (st.dry_run) { cout << "  dry run, no training\n"; return true; }

    run_training(net, X, Y, VX, VY, st, sync_file);
    return true;
}

#endif // WITH_IMPORTED_MODEL

// ---------------------------------------------------------------------------------------

int main(int argc, char** argv)
{
    try
    {
        command_line_parser parser;
        parser.add_option("load", "Converted .dat model to fine-tune (required)", 1);
        parser.add_option("corpus", "Plain document corpus; runs the knowledge-alignment stage", 1);
        parser.add_option("dataset", "Question-and-answer records; runs the task-alignment stage", 1);
        parser.add_option("valid", "Validation records for the task-alignment stage", 1);
        parser.add_option("out", "Where to write the fine-tuned model (default: <load>_tuned.dat)", 1);
        parser.add_option("system", "System block forced on every record of the dataset", 1);
        parser.add_option("template", "Chat template override: auto, zephyr, chatml, guanaco, granite", 1);
        parser.add_option("lora-rank", "Adapter rank (default: 8)", 1);
        parser.add_option("lora-method", "Adaptation method: lora or dora (default: lora)", 1);
        parser.add_option("lora-alpha", "Adapter alpha; the effective scale is alpha / rank (default: 16)", 1);
        parser.add_option("lora-targets", "Projections to adapt, as letters among q, v and f for the feed-forward (default: qv)", 1);
        parser.add_option("lora-max-width", "Widest projection an adapter may attach to; keeps the vocabulary head out (default: 16384)", 1);
        parser.add_option("window", "Training window in tokens; 0 derives it from the data (default: 0)", 1);
        parser.add_option("coverage", "Fraction of examples a derived window must keep whole (default: 0.95)", 1);
        parser.add_option("max-window", "Upper bound on a derived window (default: 2048)", 1);
        parser.add_option("epochs", "Passes over the training set, per stage (default: 1)", 1);
        parser.add_option("batch-size", "Mini-batch size (default: 4)", 1);
        parser.add_option("learning-rate", "Initial learning rate (default: 2e-5)", 1);
        parser.add_option("min-learning-rate", "Learning rate at which a stage stops (default: 1e-6)", 1);
        parser.add_option("patience", "Steps without progress before the rate is lowered; 0 derives one", 1);
        parser.add_option("limit", "Keep only this many records of the dataset; 0 keeps them all", 1);
        parser.add_option("weight-decay", "Decoupled weight decay of AdamW (default: 0, adapters need none)", 1);
        parser.add_option("beta1", "AdamW first moment decay (default: 0.9)", 1);
        parser.add_option("beta2", "AdamW second moment decay (default: 0.999)", 1);
        parser.add_option("shrink", "Factor the learning rate is multiplied by on a plateau (default: 0.1)", 1);
        parser.add_option("valid-fraction", "Fraction held out for validation when --valid is absent (default: 0.05)", 1);
        parser.add_option("valid-batch", "Windows per validation forward; 1 keeps the peak lowest (default: 1)", 1);
        parser.add_option("sync", "Trainer synchronization file, for resuming an interrupted stage", 1);
        parser.add_option("keep-adapters", "Write the model with its adapters unmerged");
        parser.add_option("dry-run", "Build the datasets and report, without training");
        parser.add_option("h", "Display this help message");
        parser.parse(argc, argv);

        if (parser.option("h") || !parser.option("load"))
        {
            cout << "Parameter-efficient fine-tuning of an imported model.\n\n"
                 << "Supply --corpus for the knowledge-alignment stage, --dataset for the\n"
                 << "task-alignment stage, or both to chain them. Corpora are produced by\n"
                 << "nist_corpus_prepare.py and cve_qa_prepare.py.\n\n";
            parser.print_options();
            return parser.option("h") ? 0 : 1;
        }

#ifndef WITH_IMPORTED_MODEL
        cerr << "This build has no model header compiled in.\n"
             << "Generate it with slm_gguf_import_ex --out-prefix slm_imported_model,\n"
             << "then rebuild this target.\n";
        return 1;
#else
        auto number = [&](const char* name, double fallback) {
            return parser.option(name) ? std::stod(parser.option(name).argument()) : fallback;
        };

        adapter_settings ad;
        ad.rank = static_cast<long>(number("lora-rank", 8));
        /* The output head projects onto the vocabulary; bounding plain projections well
           below it keeps an adapter off it without any layer knowing its own role. */
        ad.max_width = static_cast<long>(number("lora-max-width", 16384));
        ad.alpha = number("lora-alpha", 16.0);
        ad.method = adapter_method_from_name(parser.option("lora-method")
            ? parser.option("lora-method").argument() : std::string("lora"));
        {
            const std::string targets = parser.option("lora-targets")
                ? parser.option("lora-targets").argument() : std::string("qv");
            ad.attention_query = targets.find('q') != std::string::npos;
            ad.attention_value = targets.find('v') != std::string::npos;
            ad.projection = targets.find('f') != std::string::npos;
        }
        if (ad.rank <= 0 || ad.method == adapter_method::none
            || (!ad.attention_query && !ad.attention_value && !ad.projection))
        { cerr << "Error: the adapter settings would train nothing.\n"; return 1; }

        stage_settings st;
        st.window = static_cast<long>(number("window", 0));
        st.coverage = number("coverage", 0.95);
        st.max_window = static_cast<long>(number("max-window", 2048));
        st.epochs = static_cast<long>(number("epochs", 1));
        st.batch_size = static_cast<long>(number("batch-size", 4));
        st.learning_rate = number("learning-rate", 2e-5);
        st.min_learning_rate = number("min-learning-rate", 1e-6);
        st.patience = static_cast<long>(number("patience", 0));
        st.limit = static_cast<long>(number("limit", 0));
        st.weight_decay = number("weight-decay", 0.0);
        st.beta1 = number("beta1", 0.9);
        st.beta2 = number("beta2", 0.999);
        st.shrink = number("shrink", 0.1);
        st.valid_fraction = number("valid-fraction", 0.05);
        st.valid_batch = static_cast<long>(number("valid-batch", 1));
        st.dry_run = parser.option("dry-run");

        const std::string in_path = parser.option("load").argument();
        const std::string corpus = parser.option("corpus")
            ? parser.option("corpus").argument() : std::string();
        const std::string dataset = parser.option("dataset")
            ? parser.option("dataset").argument() : std::string();
        if (corpus.empty() && dataset.empty())
        { cerr << "Error: supply --corpus, --dataset, or both.\n"; return 1; }

        std::string out_path = parser.option("out") ? parser.option("out").argument()
            : (in_path.size() > 4 ? in_path.substr(0, in_path.size() - 4) : in_path) + "_tuned.dat";
        const std::string sync_file = parser.option("sync")
            ? parser.option("sync").argument() : std::string();

        /* The archive carries the parameter-bearing subnet and the tokenizer, in the
           layout the import program writes. */
        train_net net;
        hf_tokenizer tok;
        std::string model_name;
        {
            cout << "Loading " << in_path << " ...\n";
            std::ifstream fin(in_path, std::ios::binary);
            if (!fin) { cerr << "Cannot open " << in_path << "\n"; return 1; }
            std::string tag;
            deserialize(tag, fin);
            if (tag != "gguf_import_model")
            { cerr << "Not a converted model archive: " << in_path << "\n"; return 1; }
            deserialize(model_name, fin);
            try
            {
                deserialize(net.subnet(), fin);
                deserialize(tok, fin);
            }
            catch (const serialization_error& e)
            {
                /* The library carries no serialization versioning by design, so a layout
                   change makes existing archives unreadable rather than silently wrong.
                   The raw message names a class and says nothing actionable, hence this
                   one. */
                cerr << "Cannot read " << in_path << ": " << e.what() << "\n"
                     << "This archive predates a change in the network layout. Convert it\n"
                     << "again from the source model:\n"
                     << "  slm_gguf_import_ex --input <model>.gguf --out-prefix "
                     << "slm_imported_model --probe x\n"
                     << "  (rebuild, then)\n"
                     << "  slm_gguf_import_ex --input <model>.gguf --convert\n";
                return 1;
            }
            model_name = clean_model_name(model_name);
            cout << "Model       : " << model_name << "\n";
        }

        chat_template_formatter fmt = parser.option("template")
            ? chat_template_formatter::for_tokenizer(tok,
                chat_template_formatter::from_name(parser.option("template").argument()))
            : chat_template_formatter::for_tokenizer(tok, model_name);
        cout << "Chat template: " << chat_template_formatter::name(fmt.kind()) << "\n";

        const std::string system_prompt = parser.option("system")
            ? parser.option("system").argument() : std::string();

        if (!corpus.empty())
        {
            if (!stage_knowledge(net, tok, corpus, ad, st, sync_file)) return 1;
            /* Merging is mandatory before a second stage and harmless before none: the
               adapters of a stage have to become part of the weights, or the next stage
               starts from a base that never learned anything. */
            if (!st.dry_run && !dataset.empty())
            {
                const size_t merged = merge_network_adapters(net);
                cout << "  merged      : " << merged << " layers folded into the weights\n";
            }
        }

        if (!dataset.empty())
        {
            if (!stage_task(net, tok, fmt, system_prompt, dataset,
                parser.option("valid") ? parser.option("valid").argument() : std::string(),
                ad, st, sync_file)) return 1;
        }

        if (st.dry_run) { cout << "\nDry run complete; nothing was written.\n"; return 0; }

        if (!parser.option("keep-adapters"))
        {
            const size_t merged = merge_network_adapters(net);
            cout << "\nMerged " << merged << " layers into the weights.\n";
        }

        {
            /* add_layer::serialize writes each layer's x_grad, cached_output and
               params_grad alongside its parameters, so an archive written straight after a
               training pass carries a whole batch of activations and their gradients. On a
               decoder that is several times the weights themselves; clean() drops them,
               and the attention layers drop their saved forward state and KV cache with
               it. */
            net.clean();
            cout << "Writing " << out_path << " ...\n";
            serialize(out_path) << std::string("gguf_import_model") << model_name
                                << net.subnet() << tok;
            cout << "Done. The result loads with slm_gguf_import_ex --load "
                 << out_path << " --chat\n";
        }
        return 0;
#endif
    }
    catch (const std::exception& e)
    {
        cerr << "Exception thrown: " << e.what() << "\n";
        return 1;
    }
}
