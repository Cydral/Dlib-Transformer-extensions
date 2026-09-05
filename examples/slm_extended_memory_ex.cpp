/*
    @file slm_extended_memory_ex.cpp
    @brief Running a network whose tensors do not all fit in VRAM at once

    A dlib network normally holds every one of its tensors on the device from the moment it
    is sized until it is destroyed, so the footprint of a run is the sum of everything the
    graph contains. The extended memory subsystem breaks that link: residency becomes a
    decision the runtime makes under a budget, and a block that is not being read right now
    can sit in host memory or in a file on disk instead.

    Nothing in the program below is aware of any of this. The network is declared, built and
    run exactly as it would be without the subsystem. The only difference is the call to
    enable_extended_memory() in the first statements of main, and the fact that the budget
    passed to it can be smaller than the model.

    That placement is a requirement rather than a convention. A block settles where its host
    copy lives when it is sized, so a network built before the call would keep pinned mirrors
    for the rest of the run; enable_extended_memory() therefore refuses to start once any
    tensor exists. There is no matching call to switch it off, because switching off means
    bringing everything back to VRAM at once, which is precisely what the subsystem exists
    because you cannot do.

    What makes this workable rather than merely possible is that the access pattern of a
    network is periodic. Training and inference walk the same blocks in the same order on
    every step, so an observation thread can find the period of the sequence and then know,
    at any point, both what is coming and how far away each block's next use is. The first
    tells it what to prefetch, the second tells it what to evict, and the second is the
    optimal replacement policy rather than an approximation of one.

    The store, when one is configured, is a sparse file mapped once at startup. That choice
    is what lets the run stop depending on host memory as well: the mapping's pages are page
    cache, so they stay resident while there is room and are written back when there is not,
    and a block sitting in the store can be read on the host with no copy at all. Loading a
    model therefore writes through to the store rather than into pinned buffers that would
    have to be spilled afterwards.

    Try it both ways. With --budget set to a fraction of what the model needs the run still
    completes, and print_extended_memory_stats() shows where the tensors went and how much
    traffic that cost. With --off the program behaves exactly as it did before, which is the
    point of the design: the subsystem is a runtime option, not a fork of the code.

    Usage:
      ./slm_extended_memory_ex --infer --budget 64
      ./slm_extended_memory_ex --infer --budget 64 --store        (uses DLIB_XMEM_STORE or the
                                                                   platform temporary directory)
      ./slm_extended_memory_ex --infer --budget 64 --store-path /fast/disk
      ./slm_extended_memory_ex --infer --budget 64 --no-predict     (compare against LRU)
      ./slm_extended_memory_ex --train --budget 128 --hot-window 128
      ./slm_extended_memory_ex --infer --off                        (stock behaviour)
      ./slm_extended_memory_ex --infer --budget 64 --store --blocks (where the memory went)
*/

#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <dlib/cmd_line_parser.h>
#include <dlib/dnn.h>

using namespace std;
using namespace dlib;

/*
    A grouped-query transformer, which is what a real model here is built from, deliberately
    sized above the budgets suggested above so that the subsystem has something to do.

    The choice of configuration matters more than it looks. fused_transformer_config builds
    its projections from fc<>, whose setup() takes num_inputs from k()*nr()*nc(), so the
    sequence stays inside the weight shape and the parameter count is multiplied by the
    sequence length: at 64 positions the same topology carries sixty-four times the weights
    it should, and cannot be run at any other length. gqa_transformer_config builds from
    linear<> and gqa_attention_, both of which size on nc() alone, so the weights describe a
    position and the sequence passes through them. Streaming a graph inflated by the first
    mistake would measure the transport of an artefact.
*/
const long VOCAB_SIZE    = 16384;
const long NUM_LAYERS    = 8;
const long NUM_HEADS     = 8;
const long NUM_KV_HEADS  = 2;
const long EMBEDDING_DIM = 512;
const long MAX_SEQ_LEN   = 64;
const int  PAD_TOKEN     = 0;

/* Two configurations of one topology, differing only in whether the feed-forward is wrapped
   in adaptive computation. The layer is worth isolating here because its halting logic reads
   results the device has just produced, and a host access that finds the device ahead pulls
   the block back over the link whatever the budget is. That is the one pattern no residency
   manager can help with, so being able to measure the same network with and without it says
   how much of a run's link traffic the manager is responsible for and how much it is not. */
using act_config = gqa_transformer_config<VOCAB_SIZE, NUM_LAYERS, NUM_HEADS,
                                          NUM_KV_HEADS, EMBEDDING_DIM,
                                          attention_impl::unified, true>;
using flat_config = gqa_transformer_config<VOCAB_SIZE, NUM_LAYERS, NUM_HEADS,
                                           NUM_KV_HEADS, EMBEDDING_DIM,
                                           attention_impl::unified, false>;

using transformer_config = act_config;
using train_net_type     = act_config::network_type<true>;
using infer_net_type     = act_config::network_type<false>;
using flat_train_net_type = flat_config::network_type<true>;
using flat_infer_net_type = flat_config::network_type<false>;

// ----------------------------------------------------------------------------------------

/*
    Synthetic sequences. What matters here is the shape of the traffic through the network,
    not what the model learns from it, so a reproducible stream of tokens is enough and
    keeps the example free of any data file.
*/
void build_synthetic_dataset(
    size_t count,
    std::vector<matrix<int, 0, 1> >& samples,
    std::vector<matrix<unsigned long, 0, 1> >& labels
)
{
    std::mt19937 rnd(42);
    std::uniform_int_distribution<int> token(1, VOCAB_SIZE - 1);

    samples.clear();
    labels.clear();
    samples.reserve(count);
    labels.reserve(count);

    for (size_t i = 0; i < count; ++i)
    {
        /* The grouped-query head scores every position, so a sample is answered by a
           sequence of labels rather than by one. */
        matrix<int, 0, 1> s(MAX_SEQ_LEN, 1);
        matrix<unsigned long, 0, 1> l(MAX_SEQ_LEN, 1);
        for (long j = 0; j < MAX_SEQ_LEN; ++j)
        {
            s(j, 0) = token(rnd);
            l(j, 0) = (unsigned long)token(rnd);
        }
        samples.push_back(s);
        labels.push_back(l);
    }
}

// ----------------------------------------------------------------------------------------

static size_t steps_run = 1;

void report(const std::string& label, double setup, double seconds, bool per_block)
{
    /* Setting the network up and running it are two different costs and they must be
       reported apart. Construction runs a first forward pass, which sizes and allocates
       every tensor in the graph, and without a store that means pinning several hundred
       megabytes of host memory, which can take longer than the whole generation loop. A
       single figure covering both cannot be compared between two configurations that do not
       allocate the same way, and comparing it anyway is how a startup cost gets mistaken for
       a throughput gain. */
    cout << "\n" << label << ": " << setup << " s to build and size the network, "
         << seconds << " s to run, " << (1000.0*seconds/(double)steps_run) << " ms per step\n\n";
    print_extended_memory_stats(cout);
    if (per_block)
    {
        cout << "\n";
        print_extended_memory_blocks(cout);
    }
}

// ----------------------------------------------------------------------------------------

template <typename net_type>
void run_training(
    const command_line_parser& parser,
    size_t steps,
    size_t batch_size,
    const std::chrono::steady_clock::time_point& started
)
{
        /* The trainer runs a worker thread of its own, so two threads hand out device
           pointers on the same card. The hot window is shared between them, which is
           why --hot-window is worth raising here even though the default is ample for
           single threaded inference. */
        cout << "=== TRAINING ===\n";

        std::vector<matrix<int, 0, 1> > samples;
        std::vector<matrix<unsigned long, 0, 1> > labels;
        build_synthetic_dataset(steps * batch_size, samples, labels);

        net_type net;
        dnn_trainer<net_type, adam> trainer(net, adam(1e-4, 0.9, 0.999));
        const auto built = std::chrono::steady_clock::now();
        trainer.set_learning_rate(1e-4);
        trainer.set_mini_batch_size(batch_size);
        trainer.be_verbose();

        for (size_t i = 0; i < steps; ++i)
        {
            std::vector<matrix<int, 0, 1> > batch(samples.begin() + i*batch_size,
                                                  samples.begin() + (i+1)*batch_size);
            std::vector<matrix<unsigned long, 0, 1> > batch_labels(
                labels.begin() + i*batch_size, labels.begin() + (i+1)*batch_size);
            trainer.train_one_step(batch, batch_labels);
        }
        trainer.get_net();

        const auto now = std::chrono::steady_clock::now();
        report("training",
               std::chrono::duration<double>(built - started).count(),
               std::chrono::duration<double>(now - built).count(),
               parser.option("blocks"));
}

template <typename net_type>
void run_generation(
    const command_line_parser& parser,
    size_t steps,
    size_t batch_size,
    const std::chrono::steady_clock::time_point& started
)
{
        /* Inference is where the subsystem is at its most comfortable. The weights are
           read and never written, so once a block has been written to the store that
           copy stays valid and every later eviction is free: the device buffer is simply
           dropped and read back when its turn comes round again. */
        cout << "=== GENERATION ===\n";

        net_type net;
        /* This runs a forward pass, so it is what actually sizes and allocates every
           tensor in the graph. It belongs to setup, not to the loop below. */
        cout << "Model parameters: " << count_network_parameters(net, MAX_SEQ_LEN) << "\n";
        const auto built = std::chrono::steady_clock::now();

        std::vector<matrix<int, 0, 1> > samples;
        std::vector<matrix<unsigned long, 0, 1> > labels;
        build_synthetic_dataset(1, samples, labels);

        inference_context ctx(MAX_SEQ_LEN, 1, PAD_TOKEN);
        std::vector<int> prompt;
        for (long j = 0; j < MAX_SEQ_LEN; ++j)
            prompt.push_back(samples[0](j, 0));
        ctx.add_tokens(prompt);

        for (size_t i = 0; i < steps; ++i)
        {
            auto window = ctx.get_input_window();
            const unsigned long next = (unsigned long)net(window);
            ctx.add_token(next);
            if ((i + 1) % 10 == 0)
                cout << "  " << (i + 1) << " tokens\n";
        }

        const auto now = std::chrono::steady_clock::now();
        report("generation",
               std::chrono::duration<double>(built - started).count(),
               std::chrono::duration<double>(now - built).count(),
               parser.option("blocks"));
}

int main(int argc, char** argv)
{
    try
    {
        command_line_parser parser;
        parser.add_option("train", "run a short training loop");
        parser.add_option("infer", "run a short generation loop");
        parser.add_option("off", "leave the extended memory subsystem disabled");
        parser.add_option("budget", "device budget in MiB, 0 to derive it from free memory", 1);
        parser.add_option("store", "enable the store tier");
        parser.add_option("store-path", "directory holding the store, overriding DLIB_XMEM_STORE", 1);
        parser.add_option("store-size", "size of the store mapping in MiB, 0 for the default", 1);
        parser.add_option("staging", "size of the pinned staging buffer in MiB", 1);
        parser.add_option("min-block", "blocks below this size in KiB are never evicted", 1);
        parser.add_option("hot-window", "number of recently handed out pointers kept resident", 1);
        parser.add_option("lookahead", "how many future accesses the prefetcher tries to cover", 1);
        parser.add_option("advise", "how many future accesses the store warms its pages for", 1);
        parser.add_option("idle-release", "ms of inactivity after which residency is relaxed", 1);
        parser.add_option("idle-keep", "fraction of the budget kept at the first idle step", 1);
        parser.add_option("idle-purge", "ms of inactivity after which the card is handed back", 1);
        parser.add_option("no-predict", "evict by least recent use instead of by next use");
        parser.add_option("no-fingerprint", "read every block back instead of hashing it on the card");
        parser.add_option("fingerprint-min", "smallest block worth hashing in KiB, 0 to measure it", 1);
        parser.add_option("blocks", "list the managed blocks by size at the end of the run");
        parser.add_option("no-act", "build the feed-forward without adaptive computation");
        parser.add_option("paranoid", "fill every released device block with NaN");
        parser.add_option("steps", "number of steps to run", 1);
        parser.add_option("batch", "mini batch size for the training loop", 1);
        parser.add_option("h", "display this help message");
        parser.parse(argc, argv);

        if (parser.option("h") || (!parser.option("train") && !parser.option("infer")))
        {
            cout << "Usage: slm_extended_memory_ex (--train | --infer) [options]\n";
            parser.print_options();
            return 0;
        }

        /* Everything the subsystem needs to decide is settled here, before the first
           tensor exists.  Nothing after this point can turn it on, and nothing at all can
           turn it off. */
        if (!parser.option("off"))
        {
            extended_memory_options opts;
            opts.vram_budget     = (size_t)get_option(parser, "budget", 0) * 1024 * 1024;
            opts.store_bytes     = (size_t)get_option(parser, "store-size", 0) * 1024 * 1024;
            opts.staging_bytes   = (size_t)get_option(parser, "staging", 64) * 1024 * 1024;
            opts.min_block_bytes = (size_t)get_option(parser, "min-block", 1024) * 1024;
            opts.hot_window      = (unsigned)get_option(parser, "hot-window", 64);
            opts.lookahead       = (unsigned)get_option(parser, "lookahead", 24);
            opts.advise_horizon  = (unsigned)get_option(parser, "advise", 96);
            opts.idle_release_ms   = (unsigned)get_option(parser, "idle-release", 0);
            opts.idle_release_keep = get_option(parser, "idle-keep", 0.5);
            opts.idle_purge_ms     = (unsigned)get_option(parser, "idle-purge", 0);
            opts.predictive      = !parser.option("no-predict");
            opts.fingerprint     = !parser.option("no-fingerprint");
            opts.fingerprint_min_bytes = (size_t)get_option(parser, "fingerprint-min", 0) * 1024;
            opts.paranoid        = parser.option("paranoid");
            opts.verbose         = true;

            /* With a store the large blocks lose their pinned mirror as well: their host
               copy becomes a window onto the mapping, so host memory stops following the
               size of the model and follows the staging buffer instead. */
            /* Two options rather than one with an optional argument: command_line_parser
               has no notion of an optional argument, so a --store that declared one would
               quietly swallow whatever flag came after it. */
            if (parser.option("store") || parser.option("store-path"))
                opts.store_path = get_option(parser, "store-path", default_extended_memory_store_path());

            if (!enable_extended_memory(opts))
                cout << "extended memory is unavailable in this build, continuing without it\n";
        }

        const size_t steps      = get_option(parser, "steps", 20);
        const size_t batch_size = get_option(parser, "batch", 8);

        cout << (parser.option("no-act") ? flat_config::model_info::describe()
                                        : act_config::model_info::describe()) << "\n"
             << "- maximum sequence length: " << MAX_SEQ_LEN << "\n\n";

        steps_run = steps;
        const auto started = std::chrono::steady_clock::now();

        if (parser.option("train"))
        {
            if (parser.option("no-act"))
                run_training<flat_train_net_type>(parser, steps, batch_size, started);
            else
                run_training<train_net_type>(parser, steps, batch_size, started);
        }

        if (parser.option("infer"))
        {
            if (parser.option("no-act"))
                run_generation<flat_infer_net_type>(parser, steps, batch_size, started);
            else
                run_generation<infer_net_type>(parser, steps, batch_size, started);
        }

        return 0;
    }
    catch (std::exception& e)
    {
        cerr << "Exception thrown: " << e.what() << endl;
        return 1;
    }
}

/*
 * WHAT TO LOOK FOR IN THE OUTPUT
 *
 * The line reporting the access cycle is the one that matters. Until it appears, victims
 * are chosen by age and the prefetcher has nothing to work from, so the run is correct but
 * slower than it needs to be. Once the period is reported, restores should start being
 * counted as anticipated, and the ratio between those two numbers is the honest measure of
 * whether the schedule is being followed.
 *
 * The transfer figures are the other half of the picture. A generation loop with a store
 * should settle into a state where the number of store writes stops growing while the
 * number of reads keeps climbing: the weights are written once and read back on every pass,
 * because nothing modifies them. A training loop writes on every step, since the optimizer
 * touches everything, and its transfer volume is correspondingly higher.
 *
 * The two lines under store traffic are where the weights show up. An eviction of a block
 * the card says is unchanged moves nothing at all, so on a generation loop the count of
 * evictions settled by fingerprint should approach the count of evictions, and the transfers
 * going down the bus should fall away. Running with --no-fingerprint puts the read back in
 * place, which is a quick way to see what the hash is worth on your own model.
 *
 * Watch the pinned mirrors line too. With a store it should stay small whatever the size of
 * the model, because only the tensors below --min-block keep a buffer of their own. That
 * number staying flat as the model grows is the whole argument for mapping the store rather
 * than reading and writing it.
 *
 * A NOTE ON WHAT THIS BUYS
 *
 * Extending capacity is not the same as gaining speed. A model of this size has a low
 * arithmetic intensity, so with residency streamed across the bus the run becomes limited
 * by PCIe long before it is limited by the card. The subsystem is there to let something
 * run at all that otherwise would not, and to make the cost of doing so predictable rather
 * than to make it disappear.
 */
