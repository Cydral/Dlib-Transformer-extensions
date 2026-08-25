// The contents of this file are in the public domain.
// See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This is an example illustrating how to turn text into vectors with an embedding model,
    build a searchable index from them, and answer a question with the passage that best
    matches it.

    WHAT AN EMBEDDING MODEL IS, IN THIS LIBRARY'S TERMS

        It is the decoder you already know with its last step removed. An ordinary language
        model ends by projecting the final hidden state onto the vocabulary to obtain
        logits; an embedding model stops one step earlier and returns that hidden state.
        The container reflects this literally: jina-embeddings-v5-text-small-retrieval
        declares itself a qwen3, carries token_embd, twenty-eight blocks and output_norm,
        and has no output.weight at all.

        Nothing else in the stack changes. The attention stays causal, the rotary encoding
        stays as it is, the tokenizer is the model's own.

    WHICH POSITION BECOMES THE VECTOR

        A causal decoder has exactly one position that has seen the whole text: the last
        one. Averaging over positions would mix a vector that saw everything with vectors
        that saw a prefix, and would produce something the model was never trained to
        produce. The container states its own convention in <arch>.pooling_type, and this
        program follows it rather than assuming.

    THE THREE CHOICES THAT DECIDE RETRIEVAL QUALITY

        Chunk size. A vector summarizes whatever it is given, so a long chunk yields a
        vector that is about everything and therefore about nothing in particular. The
        passage you want to return also has to be short enough to be worth reading. A few
        hundred tokens is the usual range, and it is not a limitation of the model, which
        here accepts forty thousand.

        Overlap. A chunk boundary that falls in the middle of an argument leaves neither
        half retrievable. Overlapping consecutive chunks by a fifth costs storage and
        removes the boundary problem.

        Asymmetry. A retrieval model is trained on question-and-passage pairs, not on pairs
        of similar sentences, so the two sides are encoded differently. This model uses the
        prefixes "Query: " and "Document: ", and using the wrong one degrades results
        without producing any error.

    Usage:
      slm_embed_ex --model v5-small-retrieval-Q8_0.gguf --index docs/ --out index.dat
      slm_embed_ex --model v5-small-retrieval-Q8_0.gguf --load index.dat --query "..."
      slm_embed_ex --model v5-small-retrieval-Q8_0.gguf --embed "one text" --embed "another"

    The reference model is jina-embeddings-v5-text-small-retrieval, published under
    CC-BY-NC-4.0, which is not this project's licence: it is named as a reference and is
    not distributed here.

      curl -L -o v5-small-retrieval-Q8_0.gguf \
        https://huggingface.co/jinaai/jina-embeddings-v5-text-small-retrieval-GGUF/resolve/main/v5-small-retrieval-Q8_0.gguf
*/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <dlib/cmd_line_parser.h>
#include <dlib/graph_utils.h>
#include <dlib/lsh.h>
#include <dlib/data_io.h>
#include <dlib/dir_nav.h>
#include <dlib/dnn.h>

using namespace std;
using namespace dlib;

// ---------------------------------------------------------------------------------------

/* One indexed passage: where it came from, what it says, and its vector. */
struct passage
{
    std::string source;      // file it was read from
    long ordinal = 0;        // its rank within that file
    std::string text;        // the passage itself, returned to the user
    matrix<float, 0, 1> vec; // unit-length embedding
    hash_similar_angles_256::result_type sketch;  // 256-bit angular hash, for the shortlist
};

inline void serialize(const passage& item, std::ostream& out)
{
    dlib::serialize(item.source, out);
    dlib::serialize(item.ordinal, out);
    dlib::serialize(item.text, out);
    dlib::serialize(item.vec, out);
    dlib::serialize(item.sketch.first.first, out);
    dlib::serialize(item.sketch.first.second, out);
    dlib::serialize(item.sketch.second.first, out);
    dlib::serialize(item.sketch.second.second, out);
}

inline void deserialize(passage& item, std::istream& in)
{
    dlib::deserialize(item.source, in);
    dlib::deserialize(item.ordinal, in);
    dlib::deserialize(item.text, in);
    dlib::deserialize(item.vec, in);
    dlib::deserialize(item.sketch.first.first, in);
    dlib::deserialize(item.sketch.first.second, in);
    dlib::deserialize(item.sketch.second.first, in);
    dlib::deserialize(item.sketch.second.second, in);
}

/* The index.

   A flat list scanned linearly. On a corpus of a few tens of thousands of passages that is
   a few milliseconds, and it is exact: an approximate structure would trade recall for a
   speed nobody needs at this scale, and would obscure what the example is about. The model
   name and the dimension travel with it, because an index searched with another model, or
   truncated to another width, returns confident nonsense rather than an error. */
struct embedding_index
{
    std::string model_name;
    std::string tokenizer_id;
    long dimensions = 0;
    uint64 sketch_seed = 1;   // the hasher that produced the passage sketches
    std::vector<passage> passages;
};

/* Above this many passages the search shortlists before it ranks.

   A linear scan compares the question with every vector, which is exact and costs a few
   milliseconds per ten thousand passages. It stays the right answer until it does not: past
   a hundred thousand it becomes the slowest part of answering, and past a million it is the
   only part anyone notices.

   The threshold is the program's business rather than the caller's. Nobody indexing a
   directory knows how many passages it will yield, and asking them to choose a search
   strategy afterwards is asking them to know what this file knows. */
const size_t LINEAR_SCAN_LIMIT = 50000;

inline void serialize(const embedding_index& item, std::ostream& out)
{
    dlib::serialize(std::string("dlib_embedding_index"), out);
    dlib::serialize(item.model_name, out);
    dlib::serialize(item.tokenizer_id, out);
    dlib::serialize(item.dimensions, out);
    dlib::serialize(item.sketch_seed, out);
    dlib::serialize(item.passages, out);
}

inline void deserialize(embedding_index& item, std::istream& in)
{
    std::string tag;
    dlib::deserialize(tag, in);
    if (tag != "dlib_embedding_index")
        throw std::runtime_error("this file is not an embedding index");
    dlib::deserialize(item.model_name, in);
    dlib::deserialize(item.tokenizer_id, in);
    dlib::deserialize(item.dimensions, in);
    dlib::deserialize(item.sketch_seed, in);
    dlib::deserialize(item.passages, in);
}

// ---------------------------------------------------------------------------------------

/* Splits text into overlapping chunks, cutting on sentence boundaries where it can.

   Cutting on characters would split words; cutting on words would split arguments. The
   compromise is to fill up to the target size and then extend to the next sentence end,
   which keeps a chunk readable when it is returned as an answer. */
static std::vector<std::string> chunk_text(const std::string& text, size_t target_chars,
    size_t overlap_chars, size_t min_chars)
{
    std::vector<std::string> out;
    if (text.empty()) return out;

    size_t pos = 0;
    while (pos < text.size())
    {
        size_t end = std::min(pos + target_chars, text.size());

        if (end < text.size())
        {
            /* Extend to the end of the sentence, but not indefinitely: a text without
               punctuation would otherwise become one chunk. */
            const size_t limit = std::min(end + target_chars / 4, text.size());
            size_t cut = end;
            while (cut < limit && !(text[cut] == '.' || text[cut] == '!' || text[cut] == '?'
                                    || text[cut] == '\n'))
                ++cut;
            if (cut < limit) end = cut + 1;
        }

        std::string piece = text.substr(pos, end - pos);
        // Trim, since a chunk starting on a newline reads badly when returned.
        const size_t first = piece.find_first_not_of(" \t\r\n");
        const size_t last = piece.find_last_not_of(" \t\r\n");
        if (first != std::string::npos)
        {
            /* Fragments shorter than min_chars are dropped rather than embedded.

               A vector summarizes what it is given, and given three words it summarizes
               three words: the result is a point that sits close to a great many unrelated
               things and pollutes a ranking. Trailing fragments of a file are the usual
               source, and they are never the passage anyone wanted.

               The threshold belongs to indexing and not to questioning. "What is a CVE?"
               is fourteen characters and a perfectly good question, so a caller encoding a
               question passes zero here. Applying the same floor to both would discard the
               shortest questions, which are the ones users actually type. */
            const std::string kept = piece.substr(first, last - first + 1);
            if (kept.size() >= min_chars) out.push_back(kept);
        }

        if (end >= text.size()) break;
        pos = end > overlap_chars ? end - overlap_chars : end;
    }
    return out;
}

static std::string read_file(const std::string& path)
{
    std::ifstream fin(path, std::ios::binary);
    if (!fin) throw std::runtime_error("cannot read " + path);
    return std::string((std::istreambuf_iterator<char>(fin)),
                       std::istreambuf_iterator<char>());
}

// ---------------------------------------------------------------------------------------

/* Encodes one text into a unit-length vector.

   The prefix is not decoration. A retrieval model is trained on pairs where one side is a
   question and the other a passage, so the two are encoded into different regions of the
   space on purpose. Encoding a question as a document, or the reverse, degrades results
   quietly.

   truncate_dim implements Matryoshka truncation: the model is trained so that the leading
   coordinates carry most of the information, so keeping a prefix and renormalizing gives a
   shorter vector that still works. Truncating happens before normalizing, as the reference
   implementation does; the other order gives a vector that is not unit length. */
static matrix<float, 0, 1> encode_one(runtime_transformer& model, const hf_tokenizer& tok,
    const model_spec& spec, const std::string& text, bool as_query, long truncate_dim,
    long max_tokens)
{
    const std::string prefixed = (as_query ? "Query: " : "Document: ") + text;
    std::vector<int> ids = tok.encode(prefixed, true, false, true, false);
    if (ids.empty()) ids.push_back(tok.bos_id() >= 0 ? tok.bos_id() : 0);
    if (static_cast<long>(ids.size()) > max_tokens)
        ids.resize(static_cast<size_t>(max_tokens));

    model.set_context(static_cast<long>(ids.size()), 0);
    model.forward_prefill(ids);
    const tensor& hidden = model.hidden_states();

    const long width = hidden.nc();
    const long keep = truncate_dim > 0 ? std::min(truncate_dim, width) : width;

    /* The container declares its pooling; 3 is the last token, which is the only position a
       causal model has let see the whole text. Anything else is honoured as declared. */
    long row = hidden.nr() - 1;
    if (spec.pooling_type == 2) row = 0;

    matrix<float, 0, 1> v(keep);
    const float* h = hidden.host();
    if (spec.pooling_type == 1)
    {
        v = 0;
        for (long t = 0; t < hidden.nr(); ++t)
            for (long c = 0; c < keep; ++c)
                v(c) += h[tensor_index(hidden, 0, 0, t, c)];
        v /= static_cast<float>(hidden.nr());
    }
    else
    {
        for (long c = 0; c < keep; ++c) v(c) = h[tensor_index(hidden, 0, 0, row, c)];
    }

    const double n = length(v);
    if (n > 0) v /= static_cast<float>(n);
    return v;
}

/* Encodes a text of any length, as one vector per piece.

   A question is not necessarily short. A pasted paragraph, a log extract or a specification
   clause can exceed what one forward pass reads, and truncating it silently is the worst of
   the available answers: the discarded half may be the half that mattered, and nothing in
   the output says so.

   The text is therefore cut with the same chunker the corpus uses, and each piece becomes a
   vector. What the caller does with several vectors is a scoring question, answered below.
   A short text yields exactly one, so the ordinary case costs nothing. */
static std::vector<matrix<float, 0, 1>> encode_text(runtime_transformer& model,
    const hf_tokenizer& tok, const model_spec& spec, const std::string& text, bool as_query,
    long truncate_dim, long max_tokens, size_t chunk_chars, size_t overlap_chars)
{
    std::vector<matrix<float, 0, 1>> out;
    /* No floor: whatever the caller hands over is encoded, however short. */
    std::vector<std::string> pieces = chunk_text(text, chunk_chars, overlap_chars, 0);
    if (pieces.empty()) pieces.push_back(text);
    for (const std::string& piece : pieces)
        out.push_back(encode_one(model, tok, spec, piece, as_query, truncate_dim, max_tokens));
    return out;
}

// ---------------------------------------------------------------------------------------

static std::vector<std::string> list_inputs(const std::string& path)
{
    std::vector<std::string> files;
    /* A directory or a single file, decided by whether it opens as one. dir_nav throws on
       a path that is not a directory, which is the cheapest reliable test available. */
    try
    {
        for (const file& f : directory(path).get_files())
        {
            const std::string n = f.full_name();
            const size_t dot = n.rfind('.');
            if (dot != std::string::npos)
            {
                const std::string ext = n.substr(dot);
                if (ext == ".txt" || ext == ".md") files.push_back(n);
            }
        }
        std::sort(files.begin(), files.end());
    }
    catch (const directory::dir_not_found&)
    {
        files.push_back(path);
    }
    return files;
}

static int run_index(const std::string& model_path, const std::string& input,
    const std::string& out_path, size_t chunk_chars, size_t overlap_chars,
    size_t min_chunk, long truncate_dim, long max_tokens)
{
    cout << "Reading model: " << model_path << "\n";
    gguf_reader g(model_path);
    const model_spec spec = detect_model(g);
    cout << describe(spec);
    if (!spec.embedding_model)
        cout << "\nnote: this container declares no pooling type, so it is a generative\n"
                "      model. The last hidden state is still a usable representation, but\n"
                "      it was not trained to be one.\n";

    hf_tokenizer tok;
    tok.load_from_gguf(g);
    runtime_transformer model;
    cout << "\nLoading weights...\n";
    model.load(g, spec, gguf_load_options());

    const std::vector<std::string> files = list_inputs(input);
    if (files.empty()) { cerr << "Error: no .txt or .md file found at " << input << "\n"; return 1; }
    cout << "Files       : " << files.size() << "\n";

    embedding_index index;
    index.model_name = spec.model_name;
    index.tokenizer_id = tokenizer_fingerprint(tok);
    index.dimensions = truncate_dim > 0 ? std::min(truncate_dim, spec.d_model) : spec.d_model;

    /* Every passage carries a 128-bit angular sketch alongside its vector.

       Two unit vectors that point the same way agree on nearly every bit of this sketch,
       and comparing two of them is four exclusive-ors and a population count rather than a
       thousand multiplications. That is what lets a large index be narrowed to a few hundred
       candidates before anything expensive happens.

       Two hundred and fifty-six bits rather than a hundred and twenty-eight. On plainly
       similar passages both are perfect, but where the similarity is moderate, around a
       cosine of one half, the shorter sketch starts losing genuine neighbours out of a
       shortlist while the longer one keeps all of them. The difference is sixteen bytes per
       passage, against four thousand for the vector it accompanies. */
    const hash_similar_angles_256 hasher(index.sketch_seed);

    dlib::signal_handler::setup();
    const auto started = std::chrono::steady_clock::now();
    size_t chunks_total = 0;

    for (const std::string& path : files)
    {
        const std::vector<std::string> chunks =
            chunk_text(read_file(path), chunk_chars, overlap_chars, min_chunk);
        for (size_t i = 0; i < chunks.size(); ++i)
        {
            passage p;
            p.source = path;
            p.ordinal = static_cast<long>(i);
            p.text = chunks[i];
            p.vec = encode_one(model, tok, spec, chunks[i], false, truncate_dim, max_tokens);
            p.sketch = hasher(p.vec);
            index.passages.push_back(std::move(p));
            ++chunks_total;
            if (chunks_total % 10 == 0)
                cout << "\r  embedded  : " << chunks_total << " passages   " << std::flush;
            if (dlib::signal_handler::is_triggered()) break;
        }
        if (dlib::signal_handler::is_triggered())
        {
            cout << "\nInterrupted; writing what was indexed so far.\n";
            break;
        }
    }

    const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - started).count();
    cout << "\r  embedded  : " << index.passages.size() << " passages in "
         << elapsed << " s\n"
         << "  dimensions: " << index.dimensions << "\n";

    serialize(out_path) << index;
    cout << "Written to " << out_path << "\n";
    return 0;
}

// ---------------------------------------------------------------------------------------

static int run_query(const std::string& model_path, const std::string& index_path,
    const std::string& question, long top, long truncate_dim, long max_tokens,
    size_t chunk_chars, size_t overlap_chars)
{
    embedding_index index;
    { std::ifstream fin(index_path, std::ios::binary);
      if (!fin) { cerr << "Error: cannot open " << index_path << "\n"; return 1; }
      deserialize(index, fin); }
    if (index.passages.empty()) { cerr << "Error: the index is empty.\n"; return 1; }

    gguf_reader g(model_path);
    const model_spec spec = detect_model(g);
    hf_tokenizer tok;
    tok.load_from_gguf(g);

    /* An index searched with another model returns confident nonsense: the vectors are the
       right shape and mean something else entirely. The fingerprint makes that a refusal. */
    if (tokenizer_fingerprint(tok) != index.tokenizer_id)
    {
        cerr << "Error: '" << index_path << "' was built with another model ("
             << index.model_name << ").\nAn index and a query must come from the same one; "
                "the vectors are otherwise\nthe right shape and unrelated.\n";
        return 1;
    }

    runtime_transformer model;
    model.load(g, spec, gguf_load_options());

    const std::vector<matrix<float, 0, 1>> qs = encode_text(model, tok, spec, question,
        true, index.dimensions, max_tokens, chunk_chars, overlap_chars);
    if (qs.size() > 1)
        cout << "note: the question was cut into " << qs.size()
             << " pieces; a passage scores on its best match.\n";

    /* Scored with dlib's own cosine distance rather than a dot product written here.

       Both sides are unit length, so the two agree, and the library's functor is the one
       its nearest-neighbour machinery already takes: keeping it means the linear scan below
       can be swapped for find_k_nearest_neighbors_lsh on a large index without changing
       what "close" means.

       A question cut into several pieces scores a passage on its best piece rather than on
       an average. A long question usually asks several things, and a passage answering one
       of them is relevant; averaging would dilute that match into the pieces it does not
       answer. */
    const cosine_distance distance;

    /* Which passages are worth comparing exactly, and how that is decided.

       Below the threshold every passage is compared: exact, simple, and fast enough. Above
       it, the sketches shortlist first. The question is hashed, the Hamming distance to
       every passage sketch is computed, and only the closest few hundred are then ranked by
       the real cosine.

       The shortlist is deliberately far larger than what is returned. Angular hashing is a
       probabilistic filter: a genuinely close passage occasionally lands a few bits away by
       chance, and asking for fifty times the requested count makes that accident harmless
       while still discarding almost everything. What comes out is ranked exactly, so the
       approximation affects which passages were considered and never the order of those
       that were. */
    std::vector<size_t> candidates;
    const bool shortlist = index.passages.size() > LINEAR_SCAN_LIMIT;

    if (shortlist)
    {
        const hash_similar_angles_256 hasher(index.sketch_seed);
        std::vector<std::pair<unsigned int, size_t>> by_bits;
        by_bits.reserve(index.passages.size());
        for (size_t i = 0; i < index.passages.size(); ++i)
        {
            unsigned int nearest = 257;
            for (const matrix<float, 0, 1>& q : qs)
                nearest = std::min(nearest, hasher.distance(hasher(q), index.passages[i].sketch));
            by_bits.emplace_back(nearest, i);
        }
        const size_t take = std::min(index.passages.size(),
            std::max<size_t>(static_cast<size_t>(top) * 50, 500));
        std::partial_sort(by_bits.begin(), by_bits.begin() + take, by_bits.end());
        candidates.reserve(take);
        for (size_t i = 0; i < take; ++i) candidates.push_back(by_bits[i].second);
        cout << "search      : " << index.passages.size() << " passages, shortlisted to "
             << take << " by angular sketch, then ranked exactly\n";
    }
    else
    {
        candidates.resize(index.passages.size());
        for (size_t i = 0; i < candidates.size(); ++i) candidates[i] = i;
    }

    std::vector<std::pair<float, size_t>> scored;
    scored.reserve(candidates.size());
    for (size_t i : candidates)
    {
        double best = -1.0;
        for (const matrix<float, 0, 1>& q : qs)
            best = std::max(best, 1.0 - distance(q, index.passages[i].vec));
        scored.emplace_back(static_cast<float>(best), i);
    }
    std::partial_sort(scored.begin(),
        scored.begin() + std::min<size_t>(top, scored.size()), scored.end(),
        [](const std::pair<float, size_t>& a, const std::pair<float, size_t>& b)
        { return a.first > b.first; });

    cout << "Question    : " << question << "\n"
         << "Index       : " << index.passages.size() << " passages, "
         << index.dimensions << " dimensions, from " << index.model_name << "\n\n";

    for (long k = 0; k < std::min<long>(top, static_cast<long>(scored.size())); ++k)
    {
        const passage& p = index.passages[scored[static_cast<size_t>(k)].second];
        cout << "[" << (k + 1) << "] " << scored[static_cast<size_t>(k)].first
             << "  " << p.source << " #" << p.ordinal << "\n";
        std::string t = p.text;
        if (t.size() > 600) t = t.substr(0, 600) + " ...";
        cout << t << "\n\n";
    }
    return 0;
}

// ---------------------------------------------------------------------------------------

static int run_embed(const std::string& model_path, const std::vector<std::string>& texts,
    bool as_query, long truncate_dim, long max_tokens, bool pairwise)
{
    gguf_reader g(model_path);
    const model_spec spec = detect_model(g);
    cout << describe(spec) << "\n";

    hf_tokenizer tok;
    tok.load_from_gguf(g);
    runtime_transformer model;
    model.load(g, spec, gguf_load_options());

    std::vector<matrix<float, 0, 1>> vecs;
    for (const std::string& t : texts)
    {
        const std::vector<matrix<float, 0, 1>> parts = encode_text(model, tok, spec, t,
            as_query, truncate_dim, max_tokens, 1200, 240);
        if (parts.size() > 1)
            cout << "  (cut into " << parts.size() << " pieces; the first is shown)\n";
        vecs.push_back(parts.front());
        cout << "\"" << (t.size() > 48 ? t.substr(0, 48) + "..." : t) << "\"\n"
             << "  " << vecs.back().size() << " dimensions, norm " << length(vecs.back())
             << ", first five:";
        for (long i = 0; i < std::min<long>(5, vecs.back().size()); ++i)
            cout << " " << vecs.back()(i);
        cout << "\n";
    }

    if (pairwise && vecs.size() > 1)
    {
        cout << "\nCosine similarity\n";
        for (size_t i = 0; i < vecs.size(); ++i)
        {
            cout << "  ";
            for (size_t j = 0; j < vecs.size(); ++j)
                cout << (j ? "  " : "") << dot(vecs[i], vecs[j]);
            cout << "\n";
        }
    }
    return 0;
}

// ---------------------------------------------------------------------------------------

int main(int argc, char** argv)
{
    try
    {
        command_line_parser parser;
        parser.add_option("model", "Embedding model, a .gguf container (required)", 1);
        parser.add_option("index", "File or directory of .txt and .md to index", 1);
        parser.add_option("out", "Where to write the index (default: embeddings.dat)", 1);
        parser.add_option("load", "Index to search", 1);
        parser.add_option("query", "Question to answer from the index", 1);
        parser.add_option("embed", "Text to embed; repeat it for several", 1);
        parser.add_option("as-query", "Encode --embed texts as questions rather than passages");
        parser.add_option("similarity", "Print the cosine matrix between --embed texts");
        parser.add_option("top", "Passages returned by --query (default: 3)", 1);
        parser.add_option("chunk", "Characters per passage (default: 1200)", 1);
        parser.add_option("overlap", "Characters shared by consecutive passages (default: 240)", 1);
        parser.add_option("min-chunk", "Shortest passage worth indexing; questions are never subject to it (default: 40)", 1);
        parser.add_option("dimensions", "Truncate embeddings to this width; 0 keeps them whole", 1);
        parser.add_option("max-tokens", "Tokens read per passage (default: 1024)", 1);
        parser.add_option("h", "Display this help message");
        parser.parse(argc, argv);

        if (parser.option("h") || argc == 1 || !parser.option("model"))
        {
            cout << "Turn text into vectors, index it, and search it.\n\n";
            parser.print_options();
            cout << "Examples:\n"
                 << "  " << argv[0] << " --model m.gguf --index docs/ --out index.dat\n"
                 << "  " << argv[0] << " --model m.gguf --load index.dat --query \"how does X work\"\n"
                 << "  " << argv[0] << " --model m.gguf --embed \"a cat\" --embed \"a kitten\" --similarity\n";
            return 0;
        }

        const std::string model_path = parser.option("model").argument();
        const long truncate_dim = get_option(parser, "dimensions", 0L);
        const long max_tokens = get_option(parser, "max-tokens", 1024L);

        if (parser.option("index"))
            return run_index(model_path, parser.option("index").argument(),
                get_option(parser, "out", std::string("embeddings.dat")),
                static_cast<size_t>(get_option(parser, "chunk", 1200L)),
                static_cast<size_t>(get_option(parser, "overlap", 240L)),
                static_cast<size_t>(get_option(parser, "min-chunk", 40L)),
                truncate_dim, max_tokens);

        if (parser.option("query"))
        {
            if (!parser.option("load"))
            { cerr << "Error: --query needs --load to name an index.\n"; return 1; }
            return run_query(model_path, parser.option("load").argument(),
                parser.option("query").argument(), get_option(parser, "top", 3L),
                truncate_dim, max_tokens,
                static_cast<size_t>(get_option(parser, "chunk", 1200L)),
                static_cast<size_t>(get_option(parser, "overlap", 240L)));
        }

        if (parser.option("embed"))
        {
            std::vector<std::string> texts;
            for (unsigned long i = 0; i < parser.option("embed").count(); ++i)
                texts.push_back(parser.option("embed").argument(0, i));
            return run_embed(model_path, texts, parser.option("as-query") != 0,
                truncate_dim, max_tokens, parser.option("similarity") != 0);
        }

        cout << "Nothing to do. Run with -h for the three modes.\n";
        return 0;
    }
    catch (const std::exception& e)
    {
        cerr << "\nFATAL: " << e.what() << "\n";
        return 1;
    }
}
