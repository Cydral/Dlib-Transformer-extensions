// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
/*!
    @file slm_federated_ex.cpp
    @brief Federated training: a master, students in threads of their own, a relay between
           them, and the consolidation of what the students learn, shown on a text that no
           student holds whole and that the consolidated model recites from its first token.

    The program runs a federation in one process, with the roles and the rules a federation
    of separate hosts follows:
      - every exchange goes through a relay that the master and the students call; nothing
        calls them back, and each one only reads and writes the compartment of its task;
      - the master never sends a text to a student: it sends commands, model states and
        update directions, all of them numbers;
      - a student never sends its text: it sends the update its text produced, a compressed
        summary of the curvature it measured and the losses of the states it is asked to
        measure.

    The demonstration splits a text into as many parts as there are students, one part
    each, at the line ends nearest to equal shares. Each student holds the tokens of its
    part, preceded by the window of tokens before it, which only serves as context, so that
    the passage from one part to the next is learned too. Once the federation ends, the program, standing for the owner
    of the text and outside the master, checks what the consolidated model recites from the
    start token alone. As a control, the same students train apart for as many steps and
    their models are averaged once at the end.

    Two stages use the same protocol and the same check:
      recite  a network compiled into the program learns the first text from scratch: a
              perceptron over a window of bytes (--arch mlp, the default, which holds the
              text after a few hundred steps) or a small decoder (--arch transformer,
              decoder_transformer_config, which needs more rounds and whose consolidated
              state the master also writes as a GGUF container). The federation only
              handles vectors of parameters, so any network whose parameters
              visit_learnable_parameters() reaches can take their place;
      adapt   (--stage adapt or both) the consolidated decoder, written as a GGUF
              container, is loaded at run time by runtime_transformer, which trains
              low-rank adapters so that the model learns a second text; the container with
              the adapters merged recites it from its title. Any dense decoder in a GGUF
              file can take its place through --model, its own tokenizer splitting the
              text.

    A round:
      1. the master posts a command to train from the state of the round;
      2. each student trains for --inner-steps steps from that state and posts its update,
         the difference between that state and the one it reached, compressed with error
         feedback, together with the factored curvature of its gradients and the loss of
         the state of the round on its part;
      3. the master consolidates the updates into a direction (delta_consolidator);
      4. with the curvature_consensus method, the students measure the loss of a few step
         lengths along that direction on their parts and the master keeps the best one, or
         rejects the round when none improves on the current state;
      5. the master posts its decision; every replica applies the same step to the same
         state and checks the digest of the result against the master's.

    Methods compared, each run from the same initial state:
      average              FedAvg: the weighted mean of the updates, applied in full;
      outer_momentum       DiLoCo: the same mean through an outer Nesterov momentum;
      curvature_consensus  updates weighted per coordinate by the curvature each student
                           measured, shrunk where the students disagree, each block bounded
                           by the median norm of the students when three or more take part,
                           with the step length chosen on the losses of the students.
    average and outer_momentum trust every student, as FedAvg and DiLoCo do, and serve as
    references.

    The master keeps its last --pool-size consistent states in --pool, each one as a Dlib
    archive (.dat) and, for a decoder, as a GGUF container that Albireo imports. The master
    can pause the students and resume them, or stop at any time: the state it keeps is then
    the last one every replica agreed on.

    Usage:
      slm_federated_ex
      slm_federated_ex --students 3 --pause-at 3 --pause-for 2
      slm_federated_ex --text poem.txt --rounds 20 --methods curvature_consensus
      slm_federated_ex --arch transformer --stage both
      slm_federated_ex --stage adapt --model model.gguf --stop-at 6
!*/

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <dlib/cmd_line_parser.h>
#include <dlib/dir_nav.h>
#include <dlib/dnn.h>
#include <dlib/data_io/gguf_weight_loader.h>
#include <dlib/tokenizer/hf_tokenizer.h>

#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
#include <xmmintrin.h>
#define SLM_FEDERATED_MXCSR 1
#endif

using namespace dlib;

namespace
{
    std::atomic<bool> interrupted(false);

    std::vector<std::string> split_list(const std::string& s);

    void on_interrupt(int)
    {
        interrupted = true;
    }

    /* Values below the smallest normal float, which optimizer states and weights reach after
       many steps, slow the arithmetic of a processor down by orders of magnitude: every
       thread that computes takes them as zeros. */
    void flush_denormals()
    {
#ifdef SLM_FEDERATED_MXCSR
        _mm_setcsr(_mm_getcsr() | 0x8040);
#endif
    }

    /* Each student computes in a thread of its own, so the library runs its parallel loops in
       the thread that calls them rather than in its shared pool: the students do not wait on
       one another, and the gradient of an embedding table, which the shared pool would add up
       in the order its threads finish, adds up in the same order at every run. A value of
       DLIB_NUM_THREADS set beforehand is kept. Call before the first computation. */
    void compute_in_calling_threads()
    {
#ifdef _WIN32
        if (!std::getenv("DLIB_NUM_THREADS"))
            _putenv("DLIB_NUM_THREADS=0");
#else
        setenv("DLIB_NUM_THREADS", "0", 0);
#endif
    }

    // ----------------------------------------------------------------------------------------
    // The texts, and a vocabulary of bytes
    // ----------------------------------------------------------------------------------------

    /* Two sonnets of Shakespeare (1609), in the public domain: the compiled network learns
       the first, the adapters the second. */
    const char* const first_text =
        "Sonnet 18\n"
        "Shall I compare thee to a summer's day?\n"
        "Thou art more lovely and more temperate:\n"
        "Rough winds do shake the darling buds of May,\n"
        "And summer's lease hath all too short a date;\n"
        "Sometime too hot the eye of heaven shines,\n"
        "And often is his gold complexion dimm'd;\n"
        "And every fair from fair sometime declines,\n"
        "By chance or nature's changing course untrimm'd;\n"
        "But thy eternal summer shall not fade,\n"
        "Nor lose possession of that fair thou ow'st;\n"
        "Nor shall Death brag thou wander'st in his shade,\n"
        "When in eternal lines to time thou grow'st:\n"
        "So long as men can breathe or eyes can see,\n"
        "So long lives this, and this gives life to thee.\n";

    const char* const second_text =
        "Sonnet 116\n"
        "Let me not to the marriage of true minds\n"
        "Admit impediments. Love is not love\n"
        "Which alters when it alteration finds,\n"
        "Or bends with the remover to remove.\n"
        "O no! it is an ever-fixed mark\n"
        "That looks on tempests and is never shaken;\n"
        "It is the star to every wand'ring bark,\n"
        "Whose worth's unknown, although his height be taken.\n"
        "Love's not Time's fool, though rosy lips and cheeks\n"
        "Within his bending sickle's compass come;\n"
        "Love alters not with his brief hours and weeks,\n"
        "But bears it out even to the edge of doom.\n"
        "If this be error and upon me proved,\n"
        "I never writ, nor no man ever loved.\n";

    /* The bytes, the start token every text opens with, the two tokens of the chat template of
       the container (257 opens a turn, 258 ends it), and padding. ignore_label, a token no
       text holds, labels the positions a student does not learn. */
    const long vocab_size = 264;
    const int end_of_text = 256, turn_end = 258;
    const unsigned long ignore_label = vocab_size - 1;

    /* The bytes of a text after a window of start tokens, so that every byte, the first
       ones included, has a full window of context before it, in training as in recitation. */
    std::vector<int> byte_tokens(const std::string& text, long window)
    {
        std::vector<int> out(static_cast<size_t>(window), end_of_text);
        for (unsigned char c : text)
            out.push_back(c);
        return out;
    }

    /* The metadata of a GPT-2 byte-level tokenizer over the same identifiers, so that a
       container written by this program can be imported and served: the printable bytes
       stand for themselves and the others take the code points from 256 on. */
    void write_tokenizer(gguf_writer& w)
    {
        std::vector<std::string> tokens(256);
        int extra = 0;
        for (int b = 0; b < 256; ++b)
        {
            const bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
            const int cp = printable ? b : 256 + extra++;
            std::string s;
            if (cp < 0x80)
            {
                s += static_cast<char>(cp);
            }
            else
            {
                s += static_cast<char>(0xC0 | (cp >> 6));
                s += static_cast<char>(0x80 | (cp & 0x3F));
            }
            tokens[static_cast<size_t>(b)] = s;
        }
        std::vector<int32_t> types(256, 1);
        for (const char* t : { "<|endoftext|>", "<|im_start|>", "<|im_end|>" })
        {
            tokens.push_back(t);
            types.push_back(3);
        }
        while (static_cast<long>(tokens.size()) < vocab_size)
        {
            tokens.push_back("[PAD" + std::to_string(tokens.size()) + "]");
            types.push_back(5);
        }
        w.set_string("tokenizer.ggml.model", "gpt2");
        w.set_string("tokenizer.ggml.pre", "default");
        w.set_string_array("tokenizer.ggml.tokens", tokens);
        w.set_int32_array("tokenizer.ggml.token_type", types);
        w.set_string_array("tokenizer.ggml.merges", std::vector<std::string>());
        w.set_uint32("tokenizer.ggml.bos_token_id", end_of_text);
        w.set_uint32("tokenizer.ggml.eos_token_id", turn_end);
        w.set_bool("tokenizer.ggml.add_bos_token", false);
        w.set_string("tokenizer.chat_template",
            "{%- for message in messages %}{{- '<|im_start|>' + message.role + '\\n' + message.content + "
            "'<|im_end|>\\n' }}{%- endfor %}{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}"
            "{%- endif %}");
    }

    // ----------------------------------------------------------------------------------------
    // The networks of the recite stage
    // ----------------------------------------------------------------------------------------

    // A small decoder, enough to hold a page by heart.
    using decoder = decoder_transformer_config<vocab_size, 2, 2, 1, 32, 4, 1>;
    using transformer_net = decoder::network_type<true>;
    const long transformer_window = 32;

    model_spec decoder_spec()
    {
        model_spec s;
        s.arch_name = "llama";
        s.model_name = "slm-federated";
        s.n_layers = decoder::NUM_LAYERS;
        s.n_heads = decoder::NUM_HEADS;
        s.n_kv_heads = decoder::NUM_KV_HEADS;
        s.d_model = decoder::EMBEDDING_DIM;
        s.head_dim = decoder::HEAD_DIM;
        s.d_ffn = decoder::FFN_HIDDEN;
        s.vocab_size = vocab_size;
        s.rms_eps = 1e-5;
        s.rope_freq_base = 10000.0;
        return s;
    }

    /* A perceptron that predicts a byte from the mlp_window bytes before it: an embedding
       of each byte, one hidden layer, and the scores of the next byte. */
    const long mlp_window = 16;
    using mlp_net = loss_multiclass_log<fc<vocab_size, relu<fc<384, embeddings<vocab_size, 16,
        input<matrix<int, 0, 1>>>>>>>;

    // One forward pass allocates the parameters of a network.
    template <typename net_type>
    void allocate(net_type& net, long length)
    {
        std::vector<matrix<int, 0, 1>> x(1, matrix<int, 0, 1>(length));
        x[0] = 0;
        resizable_tensor in;
        net.to_tensor(x.begin(), x.end(), in);
        net.forward(in);
    }

    // ----------------------------------------------------------------------------------------
    // The relay
    // ----------------------------------------------------------------------------------------

    /* What the master may post (commands, the task, proposals, decisions, states) and
       what a student may post (results, evaluations, statuses). A payload is a serialized
       block of numbers; the relay never interprets it. */
    enum class kind { task, command, proposal, decision, state, result, evaluation, status };

    bool posted_by_master(kind k)
    {
        return k == kind::task || k == kind::command || k == kind::proposal || k == kind::decision || k == kind::state;
    }

    struct message
    {
        long seq = 0;
        kind what = kind::status;
        long round = 0;
        long sender = -1;     // -1 for the master
        long recipient = -1;  // -1 for every subscriber
        std::string payload;
    };

    /* An exchange point in the manner of the relay of the website: compartments, one per
       task, subscriptions, a size bound on every payload, and the direction of each kind
       of message checked at the door. Every call returns within its timeout, and no lock
       is held while a caller does anything else. */
    class relay
    {
    public:
        explicit relay(size_t max_payload) : max_payload_(max_payload) {}

        void open(const std::string& task)
        {
            std::lock_guard<std::mutex> lock(m_);
            tasks_[task];
        }

        void subscribe(long student, const std::string& task)
        {
            std::lock_guard<std::mutex> lock(m_);
            compartment& c = at(task);
            c.subscribers.insert(student);
        }

        void post_from_master(const std::string& task, kind what, long round, std::string payload, long recipient = -1)
        {
            if (!posted_by_master(what))
                throw error("relay: the master cannot post this kind of message");
            post(task, -1, recipient, what, round, std::move(payload), true);
        }

        void post_from_student(const std::string& task, long student, kind what, long round, std::string payload)
        {
            if (posted_by_master(what))
                throw error("relay: a student cannot post this kind of message");
            post(task, student, -1, what, round, std::move(payload), false);
        }

        // The next message of the master for this student after cursor, waiting up to timeout.
        bool next_for_student(const std::string& task, long student, long& cursor, message& out,
            std::chrono::milliseconds timeout)
        {
            std::unique_lock<std::mutex> lock(m_);
            compartment& c = at(task);
            if (!c.subscribers.count(student))
                throw error("relay: a student reads only the tasks it subscribed to");
            const auto deadline = std::chrono::steady_clock::now() + timeout;
            while (true)
            {
                for (const message& msg : c.down)
                    if (msg.seq > cursor && (msg.recipient < 0 || msg.recipient == student))
                    {
                        cursor = msg.seq;
                        out = msg;
                        bytes_down_ += msg.payload.size();
                        return true;
                    }
                if (cv_.wait_until(lock, deadline) == std::cv_status::timeout)
                    return false;
            }
        }

        /* The messages of a kind and a round posted by the students, waiting up to timeout
           until needed() of them arrived. The statuses posted meanwhile go to on_status,
           called without the lock held, so that the caller can answer them. */
        template <typename needed_t, typename abort_t, typename status_t>
        std::vector<message> collect(const std::string& task, kind what, long round, needed_t needed,
            std::chrono::milliseconds timeout, abort_t aborted, status_t on_status)
        {
            const auto deadline = std::chrono::steady_clock::now() + timeout;
            while (true)
            {
                std::vector<message> statuses;
                bool done = false;
                {
                    std::unique_lock<std::mutex> lock(m_);
                    compartment& c = at(task);
                    std::set<long> senders;
                    for (const message& msg : c.up)
                    {
                        if (msg.what == what && msg.round == round)
                            senders.insert(msg.sender);
                        if (msg.what == kind::status && msg.seq > c.status_cursor)
                        {
                            statuses.push_back(msg);
                            c.status_cursor = msg.seq;
                        }
                    }
                    done = senders.size() >= needed() || aborted() || std::chrono::steady_clock::now() >= deadline;
                    if (statuses.empty() && !done)
                    {
                        cv_.wait_for(lock, std::chrono::milliseconds(50));
                        continue;
                    }
                    if (done)
                    {
                        // The latest message of each sender, copied once.
                        std::map<long, const message*> latest;
                        for (const message& msg : c.up)
                            if (msg.what == what && msg.round == round)
                                latest[msg.sender] = &msg;
                        std::vector<message> out;
                        for (const auto& kv : latest)
                        {
                            bytes_up_ += kv.second->payload.size();
                            out.push_back(*kv.second);
                        }
                        lock.unlock();
                        for (const message& msg : statuses)
                            on_status(msg);
                        return out;
                    }
                }
                for (const message& msg : statuses)
                    on_status(msg);
            }
        }

        // Drops the messages of rounds older than round, so that a long run keeps a bounded memory.
        void purge(const std::string& task, long round)
        {
            std::lock_guard<std::mutex> lock(m_);
            compartment& c = at(task);
            const long answered = c.status_cursor;
            auto old = [round, answered](const message& msg)
            {
                if (msg.what == kind::status)
                    return msg.seq <= answered;
                return msg.round < round && msg.what != kind::task;
            };
            c.down.erase(std::remove_if(c.down.begin(), c.down.end(), old), c.down.end());
            c.up.erase(std::remove_if(c.up.begin(), c.up.end(), old), c.up.end());
        }

        size_t bytes_up() const { std::lock_guard<std::mutex> lock(m_); return bytes_up_; }
        size_t bytes_down() const { std::lock_guard<std::mutex> lock(m_); return bytes_down_; }

    private:
        struct compartment
        {
            std::set<long> subscribers;
            std::deque<message> down, up;
            long status_cursor = 0;
        };

        compartment& at(const std::string& task)
        {
            auto it = tasks_.find(task);
            if (it == tasks_.end())
                throw error("relay: unknown task '" + task + "'");
            return it->second;
        }

        void post(const std::string& task, long sender, long recipient, kind what, long round, std::string payload,
            bool down)
        {
            if (payload.size() > max_payload_)
                throw error("relay: payload over the size bound");
            {
                std::lock_guard<std::mutex> lock(m_);
                compartment& c = at(task);
                if (!down && !c.subscribers.count(sender))
                    throw error("relay: a student posts only to the tasks it subscribed to");
                message msg;
                msg.seq = ++seq_;
                msg.what = what;
                msg.round = round;
                msg.sender = sender;
                msg.recipient = recipient;
                msg.payload = std::move(payload);
                (down ? c.down : c.up).push_back(std::move(msg));
            }
            cv_.notify_all();
        }

        mutable std::mutex m_;
        std::condition_variable cv_;
        std::map<std::string, compartment> tasks_;
        long seq_ = 0;
        size_t max_payload_;
        size_t bytes_up_ = 0, bytes_down_ = 0;
    };

    inline void write_items(std::ostream&) {}

    template <typename T, typename... R>
    void write_items(std::ostream& out, const T& first, const R&... rest)
    {
        using dlib::serialize;
        serialize(first, out);
        write_items(out, rest...);
    }

    template <typename T>
    void read_item(std::istream& in, T& item)
    {
        using dlib::deserialize;
        deserialize(item, in);
    }

    // The lengths of vectors from another participant are checked against the payload.
    template <typename T>
    void read_item(std::istream& in, std::vector<T>& item)
    {
        deserialize_bounded(item, in);
    }

    inline void read_items(std::istream&) {}

    template <typename T, typename... R>
    void read_items(std::istream& in, T& first, R&... rest)
    {
        read_item(in, first);
        read_items(in, rest...);
    }

    // A payload: numbers only, in the order given.
    template <typename... T>
    std::string pack(const T&... items)
    {
        std::ostringstream out;
        write_items(out, items...);
        return out.str();
    }

    template <typename... T>
    void unpack(const std::string& payload, T&... items)
    {
        std::istringstream in(payload);
        read_items(in, items...);
    }

    // Commands of the master.
    enum : int { cmd_train = 1, cmd_pause = 2, cmd_resume = 3, cmd_stop = 4, cmd_evaluate = 5 };

    // ----------------------------------------------------------------------------------------
    // What a student holds, and what it trains
    // ----------------------------------------------------------------------------------------

    /* The tokens of a part of a text, preceded by those of the window before it, which only
       serve as context: the student learns to predict the tokens from first_target on. */
    struct text_part
    {
        std::vector<int> tokens;
        long first_target = 1;
    };

    // The tokens a window of a part holds: window, or fewer when the part is shorter.
    long window_length(const text_part& part, long window)
    {
        return std::min<long>(window, static_cast<long>(part.tokens.size()) - 1);
    }

    /* A random window of a part that holds at least one of its targets, given by its first
       position. */
    long random_start(const text_part& part, long length, dlib::rand& rnd)
    {
        const long lo = std::max<long>(0, part.first_target - length);
        const long hi = static_cast<long>(part.tokens.size()) - 1 - length;
        return lo + static_cast<long>(rnd.get_random_64bit_number() % static_cast<uint64>(hi - lo + 1));
    }

    /* The windows that score every target of a part once, as pairs of a first position and
       of the first target the window scores. Past the first window, a window scores the
       targets of its second half, so that every scored target has half a window of context
       before it, or the whole text before it when that is shorter. */
    std::vector<std::pair<long, long>> scoring_windows(const text_part& part, long length)
    {
        std::vector<std::pair<long, long>> out;
        const long n = static_cast<long>(part.tokens.size());
        long next = part.first_target;
        while (next <= n - 1)
        {
            const long start = std::max<long>(0, std::min<long>(next - 1 - length / 2, n - 1 - length));
            out.emplace_back(start, next);
            next = start + length + 1;
        }
        return out;
    }

    // A window and its targets: the next tokens from the position of scored_from on, ignore_label before.
    void make_window(const text_part& part, long start, long length, long scored_from, matrix<int, 0, 1>& x,
        matrix<unsigned long, 0, 1>& y)
    {
        x.set_size(length);
        y.set_size(length);
        const long from = std::max(part.first_target, scored_from);
        for (long t = 0; t < length; ++t)
        {
            x(t) = part.tokens[static_cast<size_t>(start + t)];
            const long target = start + t + 1;
            y(t) = target >= from ? static_cast<unsigned long>(part.tokens[static_cast<size_t>(target)]) : ignore_label;
        }
    }

    class replica
    {
    public:
        virtual ~replica() = default;
        virtual void get_parameters(std::vector<float>& p) = 0;
        virtual void set_parameters(const std::vector<float>& p) = 0;
        // One optimizer step; adds the squared gradients to squares when given one. Returns the loss.
        virtual double train_step(dlib::rand& rnd, std::vector<float>* squares) = 0;
        // The mean loss over every target of the part, which states are compared on.
        virtual double evaluation_loss() = 0;
        virtual double tokens_per_step() const = 0;
        // The most likely token after a context, read from the last window of it.
        virtual int next_token(const std::vector<int>& context) = 0;
    };

    // The decoder compiled into the program, trained on windows of the part.
    class transformer_replica : public replica
    {
    public:
        transformer_replica(const text_part& part, long batch, double learning_rate)
            : part_(part), length_(window_length(part, transformer_window)), batch_(batch),
              learning_rate_(learning_rate), solvers_(transformer_net::num_computational_layers, adamw(0.0f, 0.9f, 0.99f))
        {
            DLIB_CASSERT(length_ > 0 && part_.first_target < static_cast<long>(part_.tokens.size()));
            allocate(net_, length_);
            net_.loss_details().set_ignore_index(static_cast<long>(ignore_label));
            for (const auto& w : scoring_windows(part_, length_))
            {
                ex_.emplace_back();
                ey_.emplace_back();
                make_window(part_, w.first, length_, w.second, ex_.back(), ey_.back());
            }
        }

        void get_parameters(std::vector<float>& p) override { get_parameter_vector(net_, p); }
        void set_parameters(const std::vector<float>& p) override { set_parameter_vector(net_, p); }
        double tokens_per_step() const override { return static_cast<double>(length_ * batch_); }

        double train_step(dlib::rand& rnd, std::vector<float>* squares) override
        {
            x_.resize(static_cast<size_t>(batch_));
            y_.resize(static_cast<size_t>(batch_));
            for (long b = 0; b < batch_; ++b)
                make_window(part_, random_start(part_, length_, rnd), length_, 0, x_[static_cast<size_t>(b)],
                    y_[static_cast<size_t>(b)]);
            net_.to_tensor(x_.begin(), x_.end(), in_);
            const double loss = net_.compute_parameter_gradients(in_, y_.begin());
            if (squares)
                accumulate_squared_gradients(net_, *squares);
            net_.update_parameters(make_sstack(solvers_), learning_rate_);
            return loss;
        }

        double evaluation_loss() override
        {
            // One batch: the loss is the mean over the targets the windows score.
            net_.to_tensor(ex_.begin(), ex_.end(), in_);
            return net_.compute_loss(in_, ey_.begin());
        }

        int next_token(const std::vector<int>& context) override
        {
            const long n = std::min<long>(transformer_window, static_cast<long>(context.size()));
            matrix<int, 0, 1> x(n);
            for (long t = 0; t < n; ++t)
                x(t) = context[context.size() - static_cast<size_t>(n) + static_cast<size_t>(t)];
            return static_cast<int>(net_(x));
        }

    private:
        const text_part& part_;
        long length_, batch_;
        double learning_rate_;
        transformer_net net_;
        std::vector<adamw> solvers_;
        std::vector<matrix<int, 0, 1>> x_, ex_;
        std::vector<matrix<unsigned long, 0, 1>> y_, ey_;
        resizable_tensor in_;
    };

    // The perceptron compiled into the program, trained on the bytes of the part one by one.
    class mlp_replica : public replica
    {
    public:
        mlp_replica(const text_part& part, long batch, double learning_rate)
            : part_(part), batch_(batch), learning_rate_(learning_rate),
              solvers_(mlp_net::num_computational_layers, adamw(0.0f, 0.9f, 0.99f))
        {
            DLIB_CASSERT(part_.first_target < static_cast<long>(part_.tokens.size()));
            allocate(net_, mlp_window);
            for (long j = part_.first_target; j < static_cast<long>(part_.tokens.size()); ++j)
            {
                ex_.push_back(context_of(part_.tokens, static_cast<size_t>(j)));
                ey_.push_back(static_cast<unsigned long>(part_.tokens[static_cast<size_t>(j)]));
            }
        }

        void get_parameters(std::vector<float>& p) override { get_parameter_vector(net_, p); }
        void set_parameters(const std::vector<float>& p) override { set_parameter_vector(net_, p); }
        double tokens_per_step() const override { return static_cast<double>(batch_); }

        double train_step(dlib::rand& rnd, std::vector<float>* squares) override
        {
            x_.resize(static_cast<size_t>(batch_));
            y_.resize(static_cast<size_t>(batch_));
            for (long b = 0; b < batch_; ++b)
            {
                const size_t i = static_cast<size_t>(rnd.get_random_64bit_number() % ex_.size());
                x_[static_cast<size_t>(b)] = ex_[i];
                y_[static_cast<size_t>(b)] = ey_[i];
            }
            net_.to_tensor(x_.begin(), x_.end(), in_);
            const double loss = net_.compute_parameter_gradients(in_, y_.begin());
            if (squares)
                accumulate_squared_gradients(net_, *squares);
            net_.update_parameters(make_sstack(solvers_), learning_rate_);
            return loss;
        }

        double evaluation_loss() override
        {
            net_.to_tensor(ex_.begin(), ex_.end(), in_);
            return net_.compute_loss(in_, ey_.begin());
        }

        int next_token(const std::vector<int>& context) override
        {
            return static_cast<int>(net_(context_of(context, context.size())));
        }

    private:
        // The mlp_window tokens before position j, the start token standing for those before the text.
        static matrix<int, 0, 1> context_of(const std::vector<int>& tokens, size_t j)
        {
            matrix<int, 0, 1> x(mlp_window);
            for (long t = 0; t < mlp_window; ++t)
            {
                const long i = static_cast<long>(j) - mlp_window + t;
                x(t) = i >= 0 ? tokens[static_cast<size_t>(i)] : end_of_text;
            }
            return x;
        }

        const text_part& part_;
        long batch_;
        double learning_rate_;
        mlp_net net_;
        std::vector<adamw> solvers_;
        std::vector<matrix<int, 0, 1>> x_, ex_;
        std::vector<unsigned long> y_, ey_;
        resizable_tensor in_;
    };

    /* Loads a decoder from a GGUF container into the run-time engine, in floats, after the
       check the engine requires: a container it cannot run is refused with the reasons. */
    void load_decoder(runtime_transformer& rt, gguf_reader& g)
    {
        const model_spec spec = detect_model(g);
        const compat_result c = check_compatibility(spec, false);
        if (!c.blockers.empty())
        {
            std::string reasons;
            for (const std::string& b : c.blockers)
                reasons += "\n  " + b;
            throw error("the run-time engine cannot run this model:" + reasons);
        }
        gguf_load_options opt;
        opt.verbose = false;
        rt.set_quantized_at_rest(false);
        rt.load(g, spec, opt);
    }

    // The most likely token after a context, read by the run-time engine from its last window.
    int most_likely_next(runtime_transformer& rt, const std::vector<int>& context, long window)
    {
        const size_t n = std::min(static_cast<size_t>(window), context.size());
        const std::vector<int> x(context.end() - static_cast<std::ptrdiff_t>(n), context.end());
        const tensor& logits = rt.forward_prefill(x);
        const long vocab = logits.nc();
        const float* last = logits.host() + (static_cast<long>(n) - 1) * vocab;
        return static_cast<int>(std::max_element(last, last + vocab) - last);
    }

    /* A decoder loaded at run time from a GGUF container, whose low-rank adapters are what
       the student trains. */
    class adapter_replica : public replica
    {
    public:
        adapter_replica(const std::string& model_path, const adapter_plan& plan, const text_part& part, long window,
            long batch, double learning_rate)
            : part_(part), window_(window), length_(window_length(part, window)), batch_(batch),
              learning_rate_(learning_rate)
        {
            DLIB_CASSERT(length_ > 0 && part_.first_target < static_cast<long>(part_.tokens.size()));
            gguf_reader g(model_path);
            load_decoder(rt_, g);
            rt_.set_context(0);
            rt_.configure_adapters(plan, 1);
            m_.copy_size(rt_.adapter_parameters());
            m_ = 0;
            v_.copy_size(m_);
            v_ = 0;
            s_.copy_size(m_);
            s_ = 0;
            for (const auto& w : scoring_windows(part_, length_))
                evaluation_.push_back(sequence(w.first, w.second));
        }

        void get_parameters(std::vector<float>& p) override
        {
            const tensor& t = rt_.adapter_parameters();
            p.assign(t.host(), t.host() + t.size());
        }

        void set_parameters(const std::vector<float>& p) override
        {
            tensor& t = rt_.adapter_parameters();
            DLIB_CASSERT(p.size() == t.size());
            std::copy(p.begin(), p.end(), t.host_write_only());
        }

        double tokens_per_step() const override { return static_cast<double>(length_ * batch_); }

        double train_step(dlib::rand& rnd, std::vector<float>* squares) override
        {
            std::vector<window_targets> batch;
            long supervised = 0;
            for (long b = 0; b < batch_; ++b)
            {
                batch.push_back(sequence(random_start(part_, length_, rnd), 0));
                supervised += batch.back().supervised;
            }
            rt_.zero_adapter_gradients();
            double loss = 0;
            for (const window_targets& w : batch)
            {
                long s = 0;
                loss += rt_.accumulate_adapter_gradients(w.tokens, w.targets, s, 1.0 / std::max<long>(1, supervised));
            }
            // AdamW with a global norm bound on the gradient.
            tensor& params = rt_.adapter_parameters();
            const tensor& grads = rt_.adapter_gradients();
            const float* g = grads.host();
            double norm = 0;
            for (size_t i = 0; i < grads.size(); ++i)
                norm += static_cast<double>(g[i]) * g[i];
            norm = std::sqrt(norm);
            const float clip = norm > 1.0 ? static_cast<float>(1.0 / norm) : 1.0f;
            clipped_.copy_size(grads);
            tt::affine_transform(clipped_, grads, clip, 0.0f);
            if (squares)
            {
                const float* c = clipped_.host();
                if (squares->size() != clipped_.size())
                    squares->assign(clipped_.size(), 0.0f);
                for (size_t i = 0; i < clipped_.size(); ++i)
                    (*squares)[i] += c[i] * c[i];
            }
            ++steps_;
            tt::compute_adam_update(0, params.size(), s_, m_, v_, static_cast<float>(steps_),
                static_cast<float>(learning_rate_), 0.0f, 0.9f, 0.99f, params, clipped_);
            tt::add(params, params, s_);
            return loss / std::max<long>(1, supervised);
        }

        double evaluation_loss() override
        {
            double total = 0;
            long count = 0;
            for (const window_targets& w : evaluation_)
            {
                long s = 0;
                total += rt_.evaluate_loss(w.tokens, w.targets, s);
                count += s;
            }
            return count ? total / count : 0;
        }

        int next_token(const std::vector<int>& context) override { return most_likely_next(rt_, context, window_); }

    private:
        // A window as the runtime engine takes it: the tokens, and -1 for the positions not scored.
        struct window_targets
        {
            std::vector<int> tokens, targets;
            long supervised = 0;
        };

        window_targets sequence(long start, long scored_from) const
        {
            window_targets w;
            const long from = std::max(part_.first_target, scored_from);
            for (long t = 0; t < length_; ++t)
            {
                const long target = start + t + 1;
                w.tokens.push_back(part_.tokens[static_cast<size_t>(start + t)]);
                w.targets.push_back(target >= from ? part_.tokens[static_cast<size_t>(target)] : -1);
                if (target >= from)
                    ++w.supervised;
            }
            return w;
        }

        const text_part& part_;
        long window_, length_, batch_;
        double learning_rate_;
        runtime_transformer rt_;
        resizable_tensor m_, v_, s_, clipped_;
        long steps_ = 0;
        std::vector<window_targets> evaluation_;
    };

    // ----------------------------------------------------------------------------------------
    // The settings of a run, sent to the students as numbers
    // ----------------------------------------------------------------------------------------

    struct run_settings
    {
        consolidation_options consolidation;
        delta_codec_options codec;
        long inner_steps = 25;
        long curvature_every = 2;
    };

    void serialize(const run_settings& s, std::ostream& out)
    {
        dlib::serialize(s.consolidation, out);
        dlib::serialize(s.codec, out);
        dlib::serialize(s.inner_steps, out);
        dlib::serialize(s.curvature_every, out);
    }

    void deserialize(run_settings& s, std::istream& in)
    {
        dlib::deserialize(s.consolidation, in);
        dlib::deserialize(s.codec, in);
        dlib::deserialize(s.inner_steps, in);
        dlib::deserialize(s.curvature_every, in);
        if (s.inner_steps <= 0 || s.inner_steps > 1000000 || s.curvature_every <= 0)
            throw serialization_error("run_settings: invalid values");
    }

    // ----------------------------------------------------------------------------------------
    // The student
    // ----------------------------------------------------------------------------------------

    struct student_report
    {
        long resyncs = 0;
        std::string failure;
    };

    void run_student(long id, relay& r, const std::string& task, replica& model, const parameter_layout& layout,
        unsigned long seed, student_report& report, std::atomic<long>& failed)
    {
        try
        {
            flush_denormals();
            dlib::rand rnd(seed);
            r.subscribe(id, task);
            long cursor = 0;
            message msg;
            run_settings settings;
            std::vector<float> state, local, delta, squares, direction;
            delta_encoder encoder;
            long state_round = -1;
            bool paused = false;
            bool have_proposal = false;
            long proposal_round = -1;
            message pending_command;
            bool has_pending_command = false;

            auto apply_state = [&](const std::vector<float>& s)
            {
                state = s;
                model.set_parameters(state);
            };

            /* Reads the commands that arrive while it trains: a pause holds the student
               between two steps, a stop ends it at once. */
            std::deque<message> deferred;
            auto poll_control = [&](bool& stop) -> void
            {
                while (r.next_for_student(task, id, cursor, msg, std::chrono::milliseconds(0)))
                {
                    if (msg.what == kind::command)
                    {
                        int code = 0;
                        long round = 0;
                        uint64 digest = 0;
                        unpack(msg.payload, code, round, digest);
                        if (code == cmd_pause) { paused = true; continue; }
                        if (code == cmd_resume) { paused = false; continue; }
                        if (code == cmd_stop) { stop = true; return; }
                    }
                    deferred.push_back(msg);
                }
            };

            while (true)
            {
                if (interrupted)
                    return;
                if (!deferred.empty())
                {
                    msg = deferred.front();
                    deferred.pop_front();
                }
                else if (!r.next_for_student(task, id, cursor, msg, std::chrono::milliseconds(100)))
                {
                    continue;
                }

                if (msg.what == kind::task || msg.what == kind::state)
                {
                    std::vector<float> s;
                    long round = 0;
                    if (msg.what == kind::task)
                    {
                        unpack(msg.payload, settings, round, s);
                        encoder.set_options(settings.codec);
                    }
                    else
                    {
                        unpack(msg.payload, round, s);
                    }
                    if (s.size() != layout.size())
                        throw error("student: a state of a wrong size");
                    apply_state(s);
                    state_round = round;
                    // The command it could not follow out of step comes back once in step.
                    if (has_pending_command)
                    {
                        deferred.push_front(pending_command);
                        has_pending_command = false;
                    }
                    continue;
                }

                if (msg.what == kind::command)
                {
                    int code = 0;
                    long round = 0;
                    uint64 digest = 0;
                    unpack(msg.payload, code, round, digest);
                    if (code == cmd_stop)
                        return;
                    if (code == cmd_pause) { paused = true; continue; }
                    if (code == cmd_resume) { paused = false; continue; }
                    if (code != cmd_train && code != cmd_evaluate)
                        continue;
                    if (round != state_round || parameter_digest(state) != digest)
                    {
                        // Out of step with the master: ask for its state and wait for it.
                        ++report.resyncs;
                        pending_command = msg;
                        has_pending_command = true;
                        r.post_from_student(task, id, kind::status, round, pack(round, 1));
                        continue;
                    }
                    if (code == cmd_evaluate)
                    {
                        model.set_parameters(state);
                        const std::vector<double> losses = { model.evaluation_loss() };
                        r.post_from_student(task, id, kind::evaluation, round, pack(round, losses));
                        continue;
                    }

                    model.set_parameters(state);
                    const double before = model.evaluation_loss();
                    squares.assign(layout.size(), 0.0f);
                    double train_loss = 0;
                    long done = 0;
                    bool stop = false;
                    for (long step = 0; step < settings.inner_steps; ++step)
                    {
                        poll_control(stop);
                        while (paused && !stop && !interrupted)
                        {
                            std::this_thread::sleep_for(std::chrono::milliseconds(50));
                            poll_control(stop);
                        }
                        if (stop || interrupted)
                            return;
                        const bool measure = step % settings.curvature_every == 0;
                        train_loss += model.train_step(rnd, measure ? &squares : nullptr);
                        ++done;
                    }
                    model.get_parameters(local);
                    delta.resize(local.size());
                    for (size_t i = 0; i < local.size(); ++i)
                        delta[i] = state[i] - local[i];
                    compressed_delta packed;
                    encoder.encode(layout, delta, packed);
                    curvature_summary curvature;
                    curvature.build(layout, squares);
                    const double weight = model.tokens_per_step() * done;
                    r.post_from_student(task, id, kind::result, round,
                        pack(round, weight, before, train_loss / std::max<long>(1, done), packed, curvature));
                    have_proposal = false;
                    continue;
                }

                if (msg.what == kind::proposal)
                {
                    long round = 0;
                    compressed_delta packed;
                    std::vector<double> multipliers;
                    unpack(msg.payload, round, packed, multipliers);
                    if (round != state_round || packed.dimension() != layout.size())
                        continue;
                    decode_delta(packed, direction);
                    have_proposal = true;
                    proposal_round = round;
                    // Candidate 0 is the state of the round itself.
                    std::vector<double> losses;
                    model.set_parameters(state);
                    losses.push_back(model.evaluation_loss());
                    for (double m : multipliers)
                    {
                        local = state;
                        apply_consolidated_step(layout, local, direction,
                            settings.consolidation.method == consolidation_method::average ? m
                                : settings.consolidation.outer_learning_rate * m,
                            settings.consolidation.trust_ratio);
                        model.set_parameters(local);
                        losses.push_back(model.evaluation_loss());
                    }
                    model.set_parameters(state);
                    r.post_from_student(task, id, kind::evaluation, round, pack(round, losses));
                    continue;
                }

                if (msg.what == kind::decision)
                {
                    long round = 0;
                    int accepted = 0;
                    double step = 0;
                    uint64 digest = 0;
                    compressed_delta packed;
                    unpack(msg.payload, round, accepted, step, digest, packed);
                    if (round != state_round || (!packed.empty() && packed.dimension() != layout.size()))
                        continue;
                    if (accepted)
                    {
                        if (!packed.empty())
                        {
                            decode_delta(packed, direction);
                            have_proposal = true;
                            proposal_round = round;
                        }
                        if (!have_proposal || proposal_round != round)
                            throw error("student: a decision without its direction");
                        apply_consolidated_step(layout, state, direction, step, settings.consolidation.trust_ratio);
                    }
                    model.set_parameters(state);
                    state_round = round + 1;
                    have_proposal = false;
                    continue;
                }
            }
        }
        catch (std::exception& e)
        {
            report.failure = e.what();
            ++failed;
        }
    }

    // ----------------------------------------------------------------------------------------
    // The master
    // ----------------------------------------------------------------------------------------

    struct round_log
    {
        long round = 0;
        double loss = 0;
        double train = 0;
        double multiplier = 0;
        bool accepted = false;
    };

    struct master_controls
    {
        std::atomic<bool> stop{ false };
    };

    /* The last states every replica agreed on, as a Dlib archive and as a GGUF container,
       each written under a temporary name then renamed, the oldest dropped beyond size. */
    class state_pool
    {
    public:
        state_pool(const std::string& folder, size_t size, bool with_gguf)
            : folder_(folder), size_(size), with_gguf_(with_gguf)
        {
            create_directory(folder_);
        }

        /* A state enters the pool when the master accepts it; its loss is known one round
           later, when the students measure it, and set_loss() records it. */
        template <typename writer_dat, typename writer_gguf>
        void add(long round, writer_dat write_dat, writer_gguf write_gguf)
        {
            entry e;
            e.round = round;
            e.stem = folder_ + "/round_" + std::to_string(round);
            write_dat(e.stem + ".dat.part");
            replace(e.stem + ".dat.part", e.stem + ".dat");
            if (with_gguf_)
                write_gguf(e.stem + ".gguf");
            entries_.push_back(e);
            while (entries_.size() > size_)
            {
                std::remove((entries_.front().stem + ".dat").c_str());
                std::remove((entries_.front().stem + ".gguf").c_str());
                entries_.pop_front();
            }
            write_index();
        }

        void set_loss(long round, double loss)
        {
            for (entry& e : entries_)
                if (e.round == round)
                {
                    e.loss = loss;
                    write_index();
                }
        }

        // The container of the state added as round, empty when the pool holds none.
        std::string gguf_of(long round) const
        {
            if (with_gguf_)
                for (const entry& e : entries_)
                    if (e.round == round)
                        return e.stem + ".gguf";
            return std::string();
        }

    private:
        struct entry
        {
            long round = 0;
            double loss = -1;
            std::string stem;
        };

        void write_index() const
        {
            std::ofstream index(folder_ + "/pool.txt.part");
            for (const entry& x : entries_)
            {
                index << x.round << ' ';
                if (x.loss >= 0)
                    index << std::setprecision(6) << x.loss;
                else
                    index << '-';
                index << ' ' << x.stem << ".dat";
                if (with_gguf_)
                    index << ' ' << x.stem << ".gguf";
                index << '\n';
            }
            index.close();
            replace(folder_ + "/pool.txt.part", folder_ + "/pool.txt");
        }

        static void replace(const std::string& from, const std::string& to)
        {
            // Where a rename keeps no file under the target name, the target goes first.
            if (std::rename(from.c_str(), to.c_str()) == 0)
                return;
            std::remove(to.c_str());
            if (std::rename(from.c_str(), to.c_str()) != 0)
                throw error("state_pool: cannot rename " + from);
        }

        std::string folder_;
        size_t size_;
        bool with_gguf_;
        std::deque<entry> entries_;
    };

    struct federation_result
    {
        std::vector<float> state;   // the last state every replica agreed on
        std::vector<round_log> rounds;
        double final_loss = 0;
        long accepted = 0;
        size_t bytes_up = 0, bytes_down = 0;
        double seconds = 0;
        std::string state_gguf;     // the container of that state, when the pool holds one
    };

    /* Runs one federation: the master in this thread, the students in theirs. make_replica
       builds the model of student k; save_dat and save_gguf write a state of the pool. */
    template <typename make_replica_t, typename save_dat_t, typename save_gguf_t>
    federation_result federate(const std::string& task, long students, long rounds, const run_settings& settings,
        const parameter_layout& layout, const std::vector<float>& initial, make_replica_t make_replica,
        save_dat_t save_dat, save_gguf_t save_gguf, state_pool& pool, master_controls& controls,
        long pause_at, long pause_for, long stop_at, unsigned long seed, bool verbose)
    {
        const auto t0 = std::chrono::steady_clock::now();
        /* The bound on a payload follows the size of the model: a state travels as serialized
           floats, a few bytes each, which 8 bytes per parameter and 1 MiB for the rest cover. */
        relay r(std::max<size_t>(size_t(64) << 20, 8 * layout.size() + (size_t(1) << 20)));
        r.open(task);
        std::vector<std::unique_ptr<replica>> models;
        for (long k = 0; k < students; ++k)
            models.push_back(make_replica(k));
        std::vector<student_report> reports(static_cast<size_t>(students));
        std::atomic<long> failed(0);
        std::vector<std::thread> threads;
        for (long k = 0; k < students; ++k)
            threads.emplace_back(run_student, k, std::ref(r), std::cref(task), std::ref(*models[static_cast<size_t>(k)]),
                std::cref(layout), seed + static_cast<unsigned long>(k), std::ref(reports[static_cast<size_t>(k)]),
                std::ref(failed));

        /* Whatever ends the master, an error included, the students receive the stop command
           and are joined before their threads are destroyed. */
        bool stopped = false;
        auto stop_students = [&]()
        {
            if (stopped)
                return;
            stopped = true;
            try { r.post_from_master(task, kind::command, 0, pack(int(cmd_stop), 0L, uint64(0))); }
            catch (std::exception&) { interrupted = true; }
            for (std::thread& t : threads)
                if (t.joinable())
                    t.join();
        };
        struct guard
        {
            std::function<void()> f;
            ~guard() { f(); }
        } join_on_exit{ stop_students };

        federation_result result;
        std::vector<float> state = initial;
        delta_consolidator consolidator(layout, settings.consolidation);
        delta_encoder downlink(settings.codec);
        const auto wait = std::chrono::minutes(30);
        r.post_from_master(task, kind::task, 0, pack(settings, 0L, state));

        auto command = [&](int code, long round)
        {
            r.post_from_master(task, kind::command, round, pack(code, round, parameter_digest(state)));
        };
        // The name of the current state in the pool, 0 for the initial state, which it does not hold.
        long saved_as = 0;
        auto aborted = [&]() { return controls.stop || interrupted; };
        // A student that stopped on an error is no longer waited for.
        auto live = [&]() { return static_cast<size_t>(std::max<long>(0, students - failed)); };
        // A student out of step receives the state of the master.
        long state_round = 0;
        auto answer = [&](const message& m)
        {
            r.post_from_master(task, kind::state, state_round, pack(state_round, state), m.sender);
        };

        for (long round = 0; round < rounds && !controls.stop && !interrupted; ++round)
        {
            if (round == stop_at)
            {
                controls.stop = true;
                break;
            }
            command(cmd_train, round);
            if (round == pause_at && pause_for > 0)
            {
                /* The operator holds the federation: the students stop between two steps
                   and keep their state until the master resumes them. */
                command(cmd_pause, round);
                if (verbose)
                    std::cout << "  paused during round " << round << " for " << pause_for << " s\n";
                const auto resume_at = std::chrono::steady_clock::now() + std::chrono::seconds(pause_for);
                while (!interrupted && std::chrono::steady_clock::now() < resume_at)
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                command(cmd_resume, round);
            }

            state_round = round;
            const std::vector<message> results = r.collect(task, kind::result, round, live, wait, aborted, answer);
            if (controls.stop || interrupted)
                break;
            round_log log;
            log.round = round;
            std::vector<double> weights;
            double measured = 0, train = 0, total = 0;
            for (const message& m : results)
            {
                try
                {
                    long rr = 0;
                    double weight = 0, before = 0, train_loss = 0;
                    compressed_delta packed;
                    curvature_summary curvature;
                    unpack(m.payload, rr, weight, before, train_loss, packed, curvature);
                    /* The weight a student declares is its own claim: the students run the same
                       number of steps on batches of the same size, and the master weighs them
                       alike. */
                    weight = 1.0;
                    consolidator.add(packed, weight, curvature, rr);
                    if (!(std::abs(before) < 1e30) || !(std::abs(train_loss) < 1e30))
                        continue;
                    measured += weight * before;
                    train += weight * train_loss;
                    total += weight;
                    weights.push_back(weight);
                }
                catch (std::exception& e)
                {
                    // A malformed contribution is left out of the round.
                    if (verbose)
                        std::cout << "  contribution of student " << m.sender << " refused: " << e.what() << "\n";
                }
            }
            log.loss = total > 0 ? measured / total : 0;
            log.train = total > 0 ? train / total : 0;
            // The students measured the state of the round, saved by the last round that changed it.
            if (total > 0 && saved_as > 0)
                pool.set_loss(saved_as, log.loss);
            bool consolidated = false;
            try { consolidated = consolidator.consolidate(); }
            catch (std::exception& e)
            {
                if (verbose)
                    std::cout << "  round " << round << " not consolidated: " << e.what() << "\n";
            }

            double multiplier = 0.0;
            if (consolidated)
            {
                // A rejected direction leaves nothing of itself in the residual of the downlink.
                const delta_encoder downlink_before = downlink;
                // The direction every replica applies is the one decoded from what is sent.
                compressed_delta packed;
                std::vector<float> direction;
                bool encoded = true;
                try
                {
                    downlink.encode(layout, consolidator.proposal(), packed);
                    decode_delta(packed, direction);
                }
                catch (std::exception& e)
                {
                    encoded = false;
                    if (verbose)
                        std::cout << "  round " << round << " not encoded: " << e.what() << "\n";
                }

                bool sent = false;
                if (encoded)
                {
                    multiplier = 1.0;
                    if (!settings.consolidation.step_multipliers.empty())
                    {
                        r.post_from_master(task, kind::proposal, round, pack(round, packed, settings.consolidation.step_multipliers));
                        sent = true;
                        const std::vector<message> evaluations = r.collect(task, kind::evaluation, round, live, wait,
                            aborted, answer);
                        if (controls.stop || interrupted)
                            break;
                        std::vector<std::vector<double>> losses;
                        std::vector<double> w;
                        for (const message& m : evaluations)
                        {
                            long rr = 0;
                            std::vector<double> l;
                            try { unpack(m.payload, rr, l); }
                            catch (std::exception&) { continue; }
                            if (rr != round || l.size() != settings.consolidation.step_multipliers.size() + 1)
                                continue;
                            losses.push_back(l);
                            w.push_back(1.0);
                        }
                        const size_t best = losses.empty() ? 0 : delta_consolidator::choose_multiplier(losses, w);
                        multiplier = best == 0 ? 0.0 : settings.consolidation.step_multipliers[best - 1];
                    }
                }

                if (multiplier > 0)
                {
                    const double step = consolidator.step_size(multiplier);
                    apply_consolidated_step(layout, state, direction, step, settings.consolidation.trust_ratio);
                    consolidator.commit();
                    log.accepted = true;
                    ++result.accepted;
                    r.post_from_master(task, kind::decision, round,
                        pack(round, 1, step, parameter_digest(state), sent ? compressed_delta() : packed));
                }
                else
                {
                    downlink = downlink_before;
                }
            }
            if (!log.accepted)
            {
                // The state stays where it was, whether no direction was formed or none was chosen.
                consolidator.reject();
                r.post_from_master(task, kind::decision, round,
                    pack(round, 0, 0.0, parameter_digest(state), compressed_delta()));
            }
            log.multiplier = multiplier;
            result.rounds.push_back(log);
            if (log.accepted)
            {
                saved_as = round + 1;
                pool.add(saved_as,
                    [&](const std::string& path) { save_dat(state, path); },
                    [&](const std::string& path) { save_gguf(state, path); });
            }
            r.purge(task, round);
            if (verbose)
                std::cout << "  round " << std::setw(3) << round << "  loss " << std::fixed << std::setprecision(4)
                          << log.loss << "  train " << log.train << "  step x" << std::setprecision(2) << multiplier
                          << (log.accepted ? "" : "  rejected") << "\n" << std::defaultfloat;
        }

        // A final evaluation of the state every replica agrees on, unless the run was stopped.
        if (!interrupted)
        {
            command(cmd_evaluate, static_cast<long>(result.rounds.size()));
            state_round = static_cast<long>(result.rounds.size());
            const std::vector<message> finals = r.collect(task, kind::evaluation, state_round, live,
                std::chrono::minutes(5), [&]() { return static_cast<bool>(interrupted); }, answer);
            double sum = 0;
            size_t counted = 0;
            for (const message& m : finals)
            {
                long rr = 0;
                std::vector<double> l;
                try { unpack(m.payload, rr, l); }
                catch (std::exception&) { continue; }
                if (!l.empty() && l[0] >= 0 && l[0] < 1e30)
                {
                    sum += l[0];
                    ++counted;
                }
            }
            result.final_loss = counted ? sum / counted : 0;
            if (counted && saved_as > 0)
                pool.set_loss(saved_as, result.final_loss);
        }
        stop_students();
        for (long k = 0; k < students; ++k)
        {
            const student_report& rep = reports[static_cast<size_t>(k)];
            if (!rep.failure.empty())
                std::cout << "  student " << k << " failed: " << rep.failure << "\n";
            if (rep.resyncs > 0)
                std::cout << "  student " << k << " received the state of the master " << rep.resyncs << " times\n";
        }
        result.bytes_up = r.bytes_up();
        result.bytes_down = r.bytes_down();
        result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        result.state_gguf = saved_as > 0 ? pool.gguf_of(saved_as) : std::string();
        result.state = state;
        return result;
    }

    /* The settings of a method: the defaults of consolidation_options for
       curvature_consensus, those of DiLoCo for outer_momentum, a plain mean applied in full
       for average, then the overrides of the command line. */
    run_settings settings_for(const std::string& method, const delta_codec_options& codec, long inner_steps,
        const command_line_parser& parser)
    {
        run_settings s;
        s.codec = codec;
        s.inner_steps = inner_steps;
        consolidation_options& c = s.consolidation;
        c.method = consolidation_method_from_name(method);
        if (c.method == consolidation_method::average)
        {
            c.outer_learning_rate = 1.0;
            c.outer_momentum = 0.0;
            c.step_multipliers.clear();
            return s;
        }
        if (c.method == consolidation_method::outer_momentum)
        {
            c.outer_learning_rate = 0.7;
            c.outer_momentum = 0.9;
            c.step_multipliers.clear();
        }
        c.outer_learning_rate = get_option(parser, "outer-lr", c.outer_learning_rate);
        c.outer_momentum = get_option(parser, "outer-momentum", c.outer_momentum);
        if (c.method != consolidation_method::curvature_consensus)
            return s;
        c.curvature_mix = get_option(parser, "curvature-mix", c.curvature_mix);
        c.consensus_power = get_option(parser, "consensus", c.consensus_power);
        c.spectral_clip = get_option(parser, "spectral", c.spectral_clip);
        c.norm_bound = get_option(parser, "norm-bound", c.norm_bound);
        if (parser.option("multipliers"))
        {
            c.step_multipliers.clear();
            for (const std::string& m : split_list(parser.option("multipliers").argument()))
                if (m != "none") c.step_multipliers.push_back(std::stod(m));
        }
        return s;
    }

    // ----------------------------------------------------------------------------------------
    // The parts of a text, and what a model recites
    // ----------------------------------------------------------------------------------------

    /* Cuts the targets of a text, its tokens from first on, into parts of near equal sizes,
       and returns the first target of each part followed by the size of the text. A part ends
       at the line end nearest to its share of the text, within half a share, and at the share
       itself when no line end lies that close. Part k holds, before its targets, the context
       tokens before them. */
    std::vector<long> cut_text(const std::vector<int>& tokens, long first, long parts,
        const std::function<bool(int)>& ends_line)
    {
        const long n = static_cast<long>(tokens.size());
        DLIB_CASSERT(parts >= 1 && first >= 1 && n >= first + parts, "the text is too short for so many parts");
        const long reach = std::max<long>(1, (n - first) / (2 * parts));
        std::vector<long> cuts(1, first);
        for (long k = 1; k < parts; ++k)
        {
            const long ideal = first + (n - first) * k / parts;
            const long last = n - (parts - k);
            long best = -1;
            for (long d = 0; d <= reach && best < 0; ++d)
                for (long c : { ideal - d, ideal + d })
                    if (best < 0 && c > cuts.back() && c <= last && ends_line(tokens[static_cast<size_t>(c - 1)]))
                        best = c;
            if (best < 0)
                best = std::max(cuts.back() + 1, std::min(ideal, last));
            cuts.push_back(best);
        }
        cuts.push_back(n);
        return cuts;
    }

    std::vector<text_part> make_parts(const std::vector<int>& tokens, const std::vector<long>& cuts, long context)
    {
        std::vector<text_part> parts(cuts.size() - 1);
        for (size_t k = 0; k + 1 < cuts.size(); ++k)
        {
            const long start = std::max<long>(0, cuts[k] - context);
            parts[k].tokens.assign(tokens.begin() + start, tokens.begin() + cuts[k + 1]);
            parts[k].first_target = cuts[k] - start;
        }
        return parts;
    }

    void print_parts(const std::vector<int>& tokens, const std::vector<long>& cuts, const std::function<bool(int)>& ends_line,
        long context)
    {
        // The line of each target, counted from the line ends before it.
        std::vector<long> first(cuts.size() - 1), last(cuts.size() - 1);
        long line = 1;
        size_t k = 0;
        for (long j = cuts[0]; j < cuts.back(); ++j)
        {
            while (j >= cuts[k + 1])
                ++k;
            if (j == cuts[k])
                first[k] = line;
            last[k] = line;
            if (ends_line(tokens[static_cast<size_t>(j)]))
                ++line;
        }
        for (k = 0; k + 1 < cuts.size(); ++k)
        {
            std::cout << "  student " << k << ": lines " << first[k] << " to " << last[k] << ", "
                      << cuts[k + 1] - cuts[k] << " tokens to learn";
            if (k > 0)
                std::cout << ", after the " << std::min(context, cuts[k]) << " before them as context";
            std::cout << "\n";
        }
    }

    // A recited sequence of bytes as text, a token beyond the bytes shown by its number.
    std::string bytes_text(const std::vector<int>& tokens, size_t from)
    {
        std::string out;
        for (size_t i = from; i < tokens.size(); ++i)
            out += tokens[i] >= 0 && tokens[i] < 256 ? std::string(1, static_cast<char>(tokens[i]))
                                                      : "<" + std::to_string(tokens[i]) + ">";
        return out;
    }

    // The most likely next token after a context.
    using next_fn = std::function<int(const std::vector<int>&)>;

    next_fn next_of(replica& model)
    {
        return [&model](const std::vector<int>& context) { return model.next_token(context); };
    }

    next_fn next_of(runtime_transformer& rt, long window)
    {
        return [&rt, window](const std::vector<int>& context) { return most_likely_next(rt, context, window); };
    }

    /* What a model recites after the first prompt tokens of a text: the tokens it predicts
       one after the other, each one added to the context of the next, as many as the text
       holds. */
    std::vector<int> recite(const next_fn& next, const std::vector<int>& text, size_t prompt)
    {
        std::vector<int> out(text.begin(), text.begin() + static_cast<std::ptrdiff_t>(prompt));
        while (out.size() < text.size() && !interrupted)
            out.push_back(next(out));
        return out;
    }

    // The tokens recited after the prompt before the first one that differs from the text.
    long correct_prefix(const std::vector<int>& recited, const std::vector<int>& text, size_t prompt)
    {
        size_t i = prompt;
        while (i < recited.size() && i < text.size() && recited[i] == text[i])
            ++i;
        return static_cast<long>(i - prompt);
    }

    /* For each part, the share of its targets the model predicts from the text before them,
       the right tokens and not its own, so that one error does not hide what follows. */
    std::vector<double> part_accuracy(const next_fn& next, const std::vector<int>& text, const std::vector<long>& cuts)
    {
        std::vector<double> out;
        std::vector<int> context(text.begin(), text.begin() + cuts[0]);
        for (size_t k = 0; k + 1 < cuts.size(); ++k)
        {
            long right = 0;
            for (long j = cuts[k]; j < cuts[k + 1] && !interrupted; ++j)
            {
                if (next(context) == text[static_cast<size_t>(j)])
                    ++right;
                context.push_back(text[static_cast<size_t>(j)]);
            }
            out.push_back(static_cast<double>(right) / static_cast<double>(cuts[k + 1] - cuts[k]));
        }
        return out;
    }

    struct recitation
    {
        std::string name;
        long recited = 0;               // tokens recited from the prompt before the first error
        std::vector<double> parts;      // share of each part predicted from the text before it
        std::vector<int> tokens;        // what the model recited
    };

    recitation check_model(const std::string& name, const next_fn& next, const std::vector<int>& text,
        const std::vector<long>& cuts, size_t prompt)
    {
        recitation r;
        r.name = name;
        r.tokens = recite(next, text, prompt);
        r.recited = correct_prefix(r.tokens, text, prompt);
        r.parts = part_accuracy(next, text, cuts);
        return r;
    }

    void print_recitations(const std::vector<recitation>& checks, long total, size_t parts)
    {
        std::cout << "  " << std::left << std::setw(40) << "model" << std::right << std::setw(16) << "recited";
        for (size_t k = 0; k < parts; ++k)
            std::cout << std::setw(9) << ("part " + std::to_string(k + 1));
        std::cout << "\n";
        for (const recitation& r : checks)
        {
            std::cout << "  " << std::left << std::setw(40) << r.name << std::right << std::setw(16)
                      << (std::to_string(r.recited) + " of " + std::to_string(total));
            for (double a : r.parts)
                std::cout << std::setw(8) << std::fixed << std::setprecision(1) << 100 * a << "%" << std::defaultfloat;
            std::cout << "\n";
        }
    }

    void print_summary(const std::string& stage, const std::vector<std::string>& methods,
        const std::vector<federation_result>& results)
    {
        std::cout << "\n" << stage << ": loss of the consolidated state on the parts of the students\n";
        std::cout << "  " << std::left << std::setw(22) << "method" << std::right << std::setw(10) << "final"
                  << std::setw(10) << "accepted" << std::setw(12) << "up (KiB)" << std::setw(12) << "down (KiB)"
                  << std::setw(10) << "seconds" << "\n";
        for (size_t i = 0; i < results.size(); ++i)
        {
            const federation_result& r = results[i];
            std::cout << "  " << std::left << std::setw(22) << methods[i] << std::right << std::fixed
                      << std::setprecision(4) << std::setw(10) << r.final_loss << std::setw(10) << r.accepted
                      << std::setprecision(1) << std::setw(12) << r.bytes_up / 1024.0 << std::setw(12)
                      << r.bytes_down / 1024.0 << std::setw(10) << r.seconds << "\n" << std::defaultfloat;
        }
    }

    /* The control: each student trains on its part alone, from the same initial state and for
       as many steps as in the federation, and the models are averaged once at the end. */
    std::vector<std::vector<float>> train_apart(const std::function<std::unique_ptr<replica>(long)>& make_replica,
        const std::vector<float>& initial, long students, long steps, unsigned long seed)
    {
        std::vector<std::vector<float>> states(static_cast<size_t>(students));
        std::vector<std::thread> threads;
        for (long k = 0; k < students; ++k)
            threads.emplace_back([&, k]()
            {
                flush_denormals();
                std::unique_ptr<replica> model = make_replica(k);
                model->set_parameters(initial);
                dlib::rand rnd(seed + static_cast<unsigned long>(k));
                for (long s = 0; s < steps && !interrupted; ++s)
                    model->train_step(rnd, nullptr);
                model->get_parameters(states[static_cast<size_t>(k)]);
            });
        for (std::thread& t : threads)
            t.join();
        return states;
    }

    std::vector<float> average_of(const std::vector<std::vector<float>>& states)
    {
        std::vector<float> out(states[0].size(), 0.0f);
        for (const std::vector<float>& s : states)
            for (size_t i = 0; i < out.size(); ++i)
                out[i] += s[i] / static_cast<float>(states.size());
        return out;
    }

    std::string read_file(const std::string& path)
    {
        std::ifstream in(path, std::ios::binary);
        if (!in)
            throw error("cannot read " + path);
        return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }

    std::vector<std::string> split_list(const std::string& s)
    {
        std::vector<std::string> out;
        std::stringstream in(s);
        std::string item;
        while (std::getline(in, item, ','))
            if (!item.empty()) out.push_back(item);
        return out;
    }
}

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{
    try
    {
        command_line_parser parser;
        parser.add_option("stage", "recite, adapt or both (default: recite)", 1);
        parser.add_option("arch", "Network of the recite stage: mlp or transformer (default: mlp)", 1);
        parser.add_option("text", "File of the text the recite stage learns (default: a sonnet)", 1);
        parser.add_option("adapt-text", "File of the text the adapt stage learns (default: another sonnet)", 1);
        parser.add_option("students", "Number of students, and of parts of the text (default: 2)", 1);
        parser.add_option("rounds", "Rounds of the recite stage (default: 12 for the mlp, 40 for the transformer)", 1);
        parser.add_option("adapt-rounds", "Rounds of the adapt stage (default: 30)", 1);
        parser.add_option("inner-steps", "Optimizer steps of a student per round (default: 25)", 1);
        parser.add_option("lr", "Learning rate of the students (default: 0.005 for the networks, 0.003 for the adapters)", 1);
        parser.add_option("methods", "Comma separated list among average, outer_momentum, curvature_consensus", 1);
        parser.add_option("no-control", "Skip the control of the students trained apart");
        parser.add_option("model", "GGUF container of a dense decoder for the adapt stage, instead of the recite result", 1);
        parser.add_option("rank", "Rank of the adapters of the adapt stage (default: 32)", 1);
        parser.add_option("pool", "Folder of the state pool (default: slm_federated_pool)", 1);
        parser.add_option("pool-size", "States kept in the pool (default: 3)", 1);
        parser.add_option("bits", "Bits of the compressed updates: 4, 8 or 32 (default: 8)", 1);
        parser.add_option("keep", "Fraction of each update block sent (default: 1)", 1);
        parser.add_option("pause-at", "Round during which the master pauses the students", 1);
        parser.add_option("pause-for", "Seconds of that pause (default: 2)", 1);
        parser.add_option("stop-at", "Round at which the master stops", 1);
        parser.add_option("outer-lr", "Outer learning rate of outer_momentum and curvature_consensus", 1);
        parser.add_option("outer-momentum", "Outer Nesterov momentum of outer_momentum and curvature_consensus", 1);
        parser.add_option("curvature-mix", "Weight of the curvature, curvature_consensus (0 disables it)", 1);
        parser.add_option("consensus", "Exponent of the agreement, curvature_consensus (0 disables it)", 1);
        parser.add_option("spectral", "Spectral clipping level, curvature_consensus (0 disables it)", 1);
        parser.add_option("norm-bound", "Bound on the norm of an update, curvature_consensus (0 disables it)", 1);
        parser.add_option("multipliers", "Step multipliers evaluated by the students, curvature_consensus "
            "(comma separated, none for no evaluation)", 1);
        parser.add_option("seed", "Seed of the sampling of the students (default: 0)", 1);
        parser.add_option("quiet", "Print the summaries only");
        parser.add_option("h", "Display this help message");
        parser.parse(argc, argv);
        if (parser.option("h"))
        {
            parser.print_options();
            return 0;
        }
        std::signal(SIGINT, on_interrupt);
        compute_in_calling_threads();
        flush_denormals();

        const std::string stage = get_option(parser, "stage", "recite");
        const std::string arch = get_option(parser, "arch", "mlp");
        const long students = get_option(parser, "students", 2);
        const long rounds = get_option(parser, "rounds", arch == "mlp" ? 12 : 40);
        const long adapt_rounds = get_option(parser, "adapt-rounds", 30);
        const long inner_steps = get_option(parser, "inner-steps", 25);
        const std::vector<std::string> methods = split_list(get_option(parser, "methods",
            "average,outer_momentum,curvature_consensus"));
        const std::string pool_dir = get_option(parser, "pool", "slm_federated_pool");
        const size_t pool_size = static_cast<size_t>(get_option(parser, "pool-size", 3));
        const long pause_at = get_option(parser, "pause-at", -1);
        const long pause_for = get_option(parser, "pause-for", 2);
        const long stop_at = get_option(parser, "stop-at", -1);
        const bool verbose = !parser.option("quiet");
        const bool control = !parser.option("no-control");
        const unsigned long seed = 1000 + 100 * static_cast<unsigned long>(get_option(parser, "seed", 0));
        delta_codec_options codec;
        codec.bits = static_cast<int>(get_option(parser, "bits", 8));
        codec.keep_fraction = get_option(parser, "keep", 1.0);
        if (students < 1 || students > 64 || rounds < 1 || adapt_rounds < 1 || inner_steps < 1 || pool_size < 1 ||
            methods.empty() || (arch != "transformer" && arch != "mlp") ||
            (stage != "recite" && stage != "adapt" && stage != "both"))
            throw error("invalid arguments, see --h");
        if (!parser.option("model") && (stage == "adapt" || (stage == "both" && arch != "transformer")))
            throw error("the adapt stage needs --model, or --stage both with --arch transformer");
        for (const std::string& m : methods)
            consolidation_method_from_name(m);
        create_directory(pool_dir);

        std::string adapt_model = get_option(parser, "model", "");
        master_controls controls;

        if (stage == "recite" || stage == "both")
        {
            const std::string text = parser.option("text") ? read_file(parser.option("text").argument()) : first_text;
            const double lr = get_option(parser, "lr", 0.005);
            const bool transformer = arch == "transformer";
            const long context = transformer ? transformer_window : mlp_window;
            const std::vector<int> tokens = byte_tokens(text, context);
            if (static_cast<long>(tokens.size()) - context < students)
                throw error("the text of the recite stage holds fewer bytes than there are students");
            const auto ends_line = [](int t) { return t == '\n'; };
            const std::vector<long> cuts = cut_text(tokens, context, students, ends_line);
            const std::vector<text_part> parts = make_parts(tokens, cuts, context);
            text_part whole;
            whole.tokens = tokens;
            whole.first_target = context;

            std::function<std::unique_ptr<replica>(const text_part&)> make_model;
            parameter_layout layout;
            std::vector<float> initial;
            if (transformer)
            {
                make_model = [&](const text_part& p) { return std::unique_ptr<replica>(new transformer_replica(p, 8, lr)); };
                transformer_net seed_net;
                allocate(seed_net, transformer_window);
                layout = decoder_parameter_layout(seed_net, decoder_spec());
                get_parameter_vector(seed_net, initial);
            }
            else
            {
                make_model = [&](const text_part& p) { return std::unique_ptr<replica>(new mlp_replica(p, 32, lr)); };
                mlp_net seed_net;
                allocate(seed_net, mlp_window);
                layout = get_parameter_layout(seed_net);
                get_parameter_vector(seed_net, initial);
            }
            std::cout << "Recite stage: the " << arch << " network, " << initial.size() << " parameters, learns a text of "
                      << tokens.size() - static_cast<size_t>(context) << " bytes by heart; " << students << " students, "
                      << rounds << " rounds of " << inner_steps << " steps\n";
            print_parts(tokens, cuts, ends_line, context);

            auto save_dat = [&](const std::vector<float>& state, const std::string& path)
            {
                if (transformer)
                {
                    transformer_net net;
                    allocate(net, transformer_window);
                    set_parameter_vector(net, state);
                    serialize(path) << net;
                }
                else
                {
                    mlp_net net;
                    allocate(net, mlp_window);
                    set_parameter_vector(net, state);
                    serialize(path) << net;
                }
            };
            auto save_gguf = [&](const std::vector<float>& state, const std::string& path)
            {
                transformer_net net;
                allocate(net, transformer_window);
                set_parameter_vector(net, state);
                gguf_writer w;
                write_tokenizer(w);
                export_gguf_weights(net, decoder_spec(), w, ggml_type::F32);
                w.write(path);
            };
            // The program, which holds the text, checks a state outside the master, which never does.
            std::unique_ptr<replica> checker = make_model(whole);
            auto check_state = [&](const std::string& name, const std::vector<float>& state)
            {
                checker->set_parameters(state);
                return check_model(name, next_of(*checker), tokens, cuts, static_cast<size_t>(context));
            };

            std::vector<federation_result> results;
            std::vector<recitation> checks;
            for (const std::string& method : methods)
            {
                if (verbose)
                    std::cout << "\n[" << method << "]\n";
                state_pool pool(pool_dir + "/recite_" + method, pool_size, transformer);
                federation_result r = federate("recite/" + method, students, rounds,
                    settings_for(method, codec, inner_steps, parser), layout, initial,
                    [&](long k) { return make_model(parts[static_cast<size_t>(k)]); },
                    save_dat, save_gguf, pool, controls, pause_at, pause_for, stop_at, seed, verbose);
                if (interrupted)
                    break;
                results.push_back(r);
                const recitation check = check_state("federation, " + method, r.state);
                if (interrupted)
                    break;
                checks.push_back(check);
                if (controls.stop)
                    break;
            }
            print_summary("Recite stage", methods, results);
            /* The adapt stage starts from the container of the state the check recited, that of
               the method that recites the most, the lowest loss deciding between equals. */
            const size_t federated = checks.size();
            size_t chosen = federated;
            for (size_t i = 0; i < federated; ++i)
                if (!results[i].state_gguf.empty() && (chosen == federated ||
                    checks[i].recited > checks[chosen].recited || (checks[i].recited == checks[chosen].recited &&
                    results[i].final_loss < results[chosen].final_loss)))
                    chosen = i;
            if (adapt_model.empty() && chosen < federated)
                adapt_model = results[chosen].state_gguf;

            if (control && !controls.stop && !interrupted)
            {
                const std::vector<std::vector<float>> apart = train_apart(
                    [&](long k) { return make_model(parts[static_cast<size_t>(k)]); }, initial, students,
                    rounds * inner_steps, seed);
                for (long k = 0; k < students && !interrupted; ++k)
                {
                    const recitation check = check_state("student " + std::to_string(k) + " trained alone",
                        apart[static_cast<size_t>(k)]);
                    if (!interrupted)
                        checks.push_back(check);
                }
                if (!interrupted)
                {
                    const recitation check = check_state("average of the students alone", average_of(apart));
                    if (!interrupted)
                        checks.push_back(check);
                }
            }
            if (!checks.empty())
            {
                std::cout << "\nWhat each model recites from the start token, and the share of each part it "
                             "predicts from the text before it:\n";
                print_recitations(checks, static_cast<long>(tokens.size()) - context, parts.size());
            }
            if (federated > 0)
            {
                const recitation& best = *std::max_element(checks.begin(), checks.begin() +
                    static_cast<std::ptrdiff_t>(federated),
                    [](const recitation& a, const recitation& b) { return a.recited < b.recited; });
                const std::string recited = bytes_text(best.tokens, static_cast<size_t>(context));
                std::cout << "\nRecited by the " << best.name << ":\n" << recited
                          << (recited.empty() || recited.back() != '\n' ? "\n" : "");
            }
        }

        if ((stage == "adapt" || stage == "both") && !controls.stop && !interrupted)
        {
            if (adapt_model.empty())
                throw error("the recite stage left no container for the adapt stage");
            const std::string text = parser.option("adapt-text") ? read_file(parser.option("adapt-text").argument())
                                                                 : second_text;
            gguf_reader source(adapt_model);
            hf_tokenizer tokenizer;
            tokenizer.load_from_gguf(source);
            const long window = 32;
            const double lr = get_option(parser, "lr", 0.003);
            /* The tokens of the text after a window of start tokens, when the tokenizer has one,
               as in the recite stage. The recitation starts from the title, the first line, so
               that the model knows which text to recite. */
            const long pad = tokenizer.bos_id() >= 0 ? window : 0;
            const long first = std::max<long>(pad, 1);
            std::vector<int> tokens(static_cast<size_t>(pad), tokenizer.bos_id());
            // The text is read as plain content: a string that names a control token stays text.
            const std::vector<int> body = tokenizer.encode(text, false, false, false);
            tokens.insert(tokens.end(), body.begin(), body.end());
            if (static_cast<long>(tokens.size()) - first < students)
                throw error("the text of the adapt stage holds fewer tokens than there are students");
            const std::vector<int> title = tokenizer.encode(text.substr(0, text.find('\n') + 1), false, false, false);
            const bool titled = title.size() < body.size() && std::equal(title.begin(), title.end(), body.begin());
            const size_t prompt = std::max<size_t>(1, static_cast<size_t>(pad) + (titled ? title.size() : 0));
            const auto ends_line = [&tokenizer](int t)
            {
                const std::string s = tokenizer.decode({ t }, false);
                return !s.empty() && s.back() == '\n';
            };
            const std::vector<long> cuts = cut_text(tokens, first, students, ends_line);
            const std::vector<text_part> parts = make_parts(tokens, cuts, window);

            adapter_plan plan;
            plan.rank = get_option(parser, "rank", 32);
            plan.method = adapter_method::lora;
            plan.alpha = 2.0 * plan.rank;
            plan.attention_query = true;
            plan.attention_value = true;
            plan.projection = true;
            runtime_transformer master_model;
            load_decoder(master_model, source);
            master_model.configure_adapters(plan, 1);
            const parameter_layout layout = master_model.adapter_layout();
            const tensor& p0 = master_model.adapter_parameters();
            const std::vector<float> initial(p0.host(), p0.host() + p0.size());
            std::cout << "\nAdapt stage: low-rank adapters on " << adapt_model << ", loaded at run time, learn a text of "
                      << tokens.size() - static_cast<size_t>(first) << " tokens, " << initial.size()
                      << " adapter parameters on " << master_model.adapter_sites().size() << " projections\n";
            print_parts(tokens, cuts, ends_line, window);

            std::vector<federation_result> results;
            std::vector<recitation> checks;
            for (const std::string& method : methods)
            {
                if (verbose)
                    std::cout << "\n[" << method << "]\n";
                state_pool pool(pool_dir + "/adapt_" + method, pool_size, true);
                auto save_dat = [&](const std::vector<float>& state, const std::string& path)
                {
                    std::copy(state.begin(), state.end(), master_model.adapter_parameters().host_write_only());
                    std::ofstream out(path, std::ios::binary);
                    master_model.save_adapters(out);
                };
                auto save_gguf = [&](const std::vector<float>& state, const std::string& path)
                {
                    std::copy(state.begin(), state.end(), master_model.adapter_parameters().host_write_only());
                    master_model.write_merged_gguf(source, path, ggml_type::F32);
                };
                federation_result r = federate("adapt/" + method, students, adapt_rounds,
                    settings_for(method, codec, inner_steps, parser), layout, initial,
                    [&](long k) { return std::unique_ptr<replica>(new adapter_replica(adapt_model, plan,
                        parts[static_cast<size_t>(k)], window, 4, lr)); },
                    save_dat, save_gguf, pool, controls, pause_at, pause_for, stop_at, seed, verbose);
                if (interrupted)
                    break;
                results.push_back(r);
                // The container of the last agreed state with the adapters merged, as Albireo would import it.
                if (!r.state_gguf.empty())
                {
                    gguf_reader merged(r.state_gguf);
                    runtime_transformer rt;
                    load_decoder(rt, merged);
                    rt.set_context(0);
                    const recitation check = check_model("merged container, " + method, next_of(rt, window), tokens,
                        cuts, prompt);
                    if (interrupted)
                        break;
                    checks.push_back(check);
                }
                if (controls.stop)
                    break;
            }
            print_summary("Adapt stage", methods, results);
            if (!checks.empty())
            {
                std::cout << "\nWhat each merged container recites from the title, and the share of each part it "
                             "predicts from the text before it:\n";
                print_recitations(checks, static_cast<long>(tokens.size() - prompt), parts.size());
                const recitation& best = *std::max_element(checks.begin(), checks.end(),
                    [](const recitation& a, const recitation& b) { return a.recited < b.recited; });
                const std::string recited = tokenizer.decode(best.tokens, true);
                std::cout << "\nRecited by the " << best.name << ":\n" << recited
                          << (recited.empty() || recited.back() != '\n' ? "\n" : "");
            }
        }
        if (controls.stop)
            std::cout << "\nStopped by the master: the pool holds the last states every replica agreed on.\n";
        if (interrupted)
        {
            std::cout << "\nInterrupted: the pool holds the last states every replica agreed on.\n";
            return 130;
        }
        return 0;
    }
    catch (std::exception& e)
    {
        std::cerr << "error: " << e.what() << std::endl;
        return 1;
    }
}
