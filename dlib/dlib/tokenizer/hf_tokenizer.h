// Copyright (C) 2025 Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_HF_TOKENIZER_Hh_
#define DLIB_HF_TOKENIZER_Hh_

#include <string>
#include <vector>
#include <array>
#include <queue>
#include <unordered_map>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "hf_tokenizer_abstract.h"
#include "../data_io.h"
#include "../serialize.h"

namespace dlib
{
    namespace hf_tok_impl
    {
        inline int utf8_char_len(unsigned char c)
        {
            if (c < 0x80) return 1;
            if ((c >> 5) == 0x6) return 2;
            if ((c >> 4) == 0xE) return 3;
            if ((c >> 3) == 0x1E) return 4;
            return 1;
        }

        inline uint32_t utf8_decode(const std::string& s, size_t pos, size_t& len)
        {
            const unsigned char c = static_cast<unsigned char>(s[pos]);
            len = utf8_char_len(c);
            if (pos + len > s.size()) { len = 1; return c; }
            switch (len)
            {
            case 2: return ((c & 0x1Fu) << 6) | (static_cast<unsigned char>(s[pos + 1]) & 0x3Fu);
            case 3: return ((c & 0x0Fu) << 12) | ((static_cast<unsigned char>(s[pos + 1]) & 0x3Fu) << 6)
                | (static_cast<unsigned char>(s[pos + 2]) & 0x3Fu);
            case 4: return ((c & 0x07u) << 18) | ((static_cast<unsigned char>(s[pos + 1]) & 0x3Fu) << 12)
                | ((static_cast<unsigned char>(s[pos + 2]) & 0x3Fu) << 6)
                | (static_cast<unsigned char>(s[pos + 3]) & 0x3Fu);
            default: return c;
            }
        }

        inline std::string utf8_encode(uint32_t cp)
        {
            std::string o;
            if (cp < 0x80) o.push_back(static_cast<char>(cp));
            else if (cp < 0x800)
            {
                o.push_back(static_cast<char>(0xC0 | (cp >> 6)));
                o.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            }
            else if (cp < 0x10000)
            {
                o.push_back(static_cast<char>(0xE0 | (cp >> 12)));
                o.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                o.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            }
            else
            {
                o.push_back(static_cast<char>(0xF0 | (cp >> 18)));
                o.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
                o.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                o.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            }
            return o;
        }
    }

    // ----------------------------------------------------------------------------------------

    class hf_tokenizer
    {
    public:
        enum class kind { spm, bpe };

        // GGUF token-type values (the llama.cpp convention).
        enum token_type
        {
            TT_UNDEFINED = 0, TT_NORMAL = 1, TT_UNKNOWN = 2, TT_CONTROL = 3,
            TT_USER_DEFINED = 4, TT_UNUSED = 5, TT_BYTE = 6
        };

        // Source-independent description of a pretrained vocabulary.
        struct vocab_data
        {
            kind family = kind::spm;
            std::vector<std::string> tokens;
            std::vector<float> scores;          // SentencePiece merge scores (optional)
            std::vector<int> types;             // per-token token_type (optional)
            std::vector<std::string> merges;    // byte-level BPE merge rules (optional)
            int bos_id = -1, eos_id = -1, unk_id = -1, pad_id = -1;
            bool add_bos = false, add_eos = false, add_space_prefix = true;
            std::string pretokenizer;           // tokenizer.ggml.pre, for byte-level BPE
        };

        hf_tokenizer() = default;

        void load(const vocab_data& data)
        {
            kind_ = data.family;
            id_to_token_ = data.tokens;

            const size_t n = id_to_token_.size();
            scores_ = data.scores;
            scores_.resize(n, 0.0f);
            token_types_ = data.types;
            token_types_.resize(n, TT_NORMAL);
            bpe_merges_ = data.merges;

            bos_ = data.bos_id; eos_ = data.eos_id; unk_ = data.unk_id; pad_ = data.pad_id;
            add_bos_ = data.add_bos; add_eos_ = data.add_eos; add_space_prefix_ = data.add_space_prefix;
            pre_ = data.pretokenizer;

            finalize();
        }

        void load_from_gguf(const gguf_reader& g)
        {
            vocab_data v;
            const std::string model = g.get_str("tokenizer.ggml.model", "llama");
            v.family = (model == "gpt2" || model == "bpe") ? kind::bpe : kind::spm;

            if (!g.has("tokenizer.ggml.tokens"))
                throw std::runtime_error("hf_tokenizer: missing tokenizer.ggml.tokens");
            v.tokens = g.at("tokenizer.ggml.tokens").arr_str;

            if (g.has("tokenizer.ggml.scores"))
            {
                const auto& sc = g.at("tokenizer.ggml.scores").arr_float;
                v.scores.reserve(sc.size());
                for (double x : sc) v.scores.push_back(static_cast<float>(x));
            }
            if (g.has("tokenizer.ggml.token_type"))
            {
                const auto& tt = g.at("tokenizer.ggml.token_type").arr_int;
                v.types.reserve(tt.size());
                for (int64_t x : tt) v.types.push_back(static_cast<int>(x));
            }
            if (g.has("tokenizer.ggml.merges")) v.merges = g.at("tokenizer.ggml.merges").arr_str;

            v.bos_id = static_cast<int>(g.get_int("tokenizer.ggml.bos_token_id", -1));
            v.eos_id = static_cast<int>(g.get_int("tokenizer.ggml.eos_token_id", -1));
            v.unk_id = static_cast<int>(g.get_int("tokenizer.ggml.unknown_token_id", -1));
            v.pad_id = static_cast<int>(g.get_int("tokenizer.ggml.padding_token_id", -1));

            v.add_bos = g.has("tokenizer.ggml.add_bos_token")
                ? (g.get_int("tokenizer.ggml.add_bos_token") != 0) : (v.family == kind::spm);
            v.add_eos = g.has("tokenizer.ggml.add_eos_token")
                ? (g.get_int("tokenizer.ggml.add_eos_token") != 0) : false;
            v.add_space_prefix = g.has("tokenizer.ggml.add_space_prefix")
                ? (g.get_int("tokenizer.ggml.add_space_prefix") != 0) : true;
            v.pretokenizer = g.get_str("tokenizer.ggml.pre", "");

            load(v);
        }

        size_t size() const { return id_to_token_.size(); }
        kind type() const { return kind_; }
        int bos_id() const { return bos_; }
        int eos_id() const { return eos_; }
        int unk_id() const { return unk_; }
        int pad_id() const { return pad_; }
        bool add_bos_default() const { return add_bos_; }
        bool add_eos_default() const { return add_eos_; }
        const std::string& pretokenizer() const { return pre_; }
        const std::string& id_to_token(int id) const { return id_to_token_.at(static_cast<size_t>(id)); }

        std::vector<int> encode(const std::string& text) const
        {
            return encode(text, add_bos_, add_eos_, true);
        }

        std::vector<int> encode(const std::string& text, bool add_bos, bool add_eos,
            bool parse_special = true) const
        {
            std::vector<int> out;
            if (add_bos && bos_ >= 0) out.push_back(bos_);
            encode_text(text, parse_special, out);
            if (add_eos && eos_ >= 0) out.push_back(eos_);
            return out;
        }

        std::string decode(const std::vector<int>& ids, bool skip_special = true) const
        {
            return (kind_ == kind::spm) ? decode_spm(ids, skip_special) : decode_bpe(ids, skip_special);
        }

        friend void serialize(const hf_tokenizer& t, std::ostream& out)
        {
            dlib::serialize(std::string("hf_tokenizer_v2"), out);
            dlib::serialize(static_cast<int>(t.kind_), out);
            dlib::serialize(t.id_to_token_, out);
            dlib::serialize(t.scores_, out);
            dlib::serialize(t.token_types_, out);
            dlib::serialize(t.bpe_merges_, out);
            dlib::serialize(t.bos_, out);
            dlib::serialize(t.eos_, out);
            dlib::serialize(t.unk_, out);
            dlib::serialize(t.pad_, out);
            dlib::serialize(t.add_bos_, out);
            dlib::serialize(t.add_eos_, out);
            dlib::serialize(t.add_space_prefix_, out);
            dlib::serialize(t.pre_, out);
        }

        friend void deserialize(hf_tokenizer& t, std::istream& in)
        {
            std::string version;
            dlib::deserialize(version, in);
            if (version != "hf_tokenizer_v2")
                throw serialization_error("hf_tokenizer: unexpected version '" + version + "'");
            int k = 0;
            dlib::deserialize(k, in);
            t.kind_ = static_cast<kind>(k);
            dlib::deserialize(t.id_to_token_, in);
            dlib::deserialize(t.scores_, in);
            dlib::deserialize(t.token_types_, in);
            dlib::deserialize(t.bpe_merges_, in);
            dlib::deserialize(t.bos_, in);
            dlib::deserialize(t.eos_, in);
            dlib::deserialize(t.unk_, in);
            dlib::deserialize(t.pad_, in);
            dlib::deserialize(t.add_bos_, in);
            dlib::deserialize(t.add_eos_, in);
            dlib::deserialize(t.add_space_prefix_, in);
            dlib::deserialize(t.pre_, in);
            t.finalize();
        }

    private:
        /* Rebuild every lookup table derived from the serialized data. */
        void finalize()
        {
            token_to_id_.clear();
            token_to_id_.reserve(id_to_token_.size() * 2);
            for (int i = 0; i < static_cast<int>(id_to_token_.size()); ++i)
                token_to_id_.emplace(id_to_token_[i], i);

            byte_to_id_.fill(-1);
            byte_value_.assign(id_to_token_.size(), -1);
            special_pieces_.clear();
            for (int i = 0; i < static_cast<int>(id_to_token_.size()); ++i)
            {
                const int tt = (i < static_cast<int>(token_types_.size())) ? token_types_[i] : TT_NORMAL;
                if (tt == TT_BYTE)
                {
                    const int b = parse_byte_token(id_to_token_[i]);
                    if (b >= 0) { byte_to_id_[static_cast<size_t>(b)] = i; byte_value_[i] = b; }
                }
                else if ((tt == TT_CONTROL || tt == TT_USER_DEFINED) && !id_to_token_[i].empty())
                {
                    special_pieces_.emplace_back(id_to_token_[i], i);
                }
            }
            /* Longest pieces first so matching is greedy and unambiguous. */
            std::sort(special_pieces_.begin(), special_pieces_.end(),
                [](const std::pair<std::string, int>& a, const std::pair<std::string, int>& b)
                { return a.first.size() > b.first.size(); });

            if (kind_ == kind::bpe)
            {
                build_byte_unicode_tables();
                bpe_ranks_.clear();
                bpe_ranks_.reserve(bpe_merges_.size() * 2);
                for (int i = 0; i < static_cast<int>(bpe_merges_.size()); ++i)
                    bpe_ranks_.emplace(bpe_merges_[i], i);
            }
        }

        static int parse_byte_token(const std::string& s)
        {
            if (s.size() != 6 || s[0] != '<' || s[1] != '0' || s[2] != 'x' || s[5] != '>') return -1;
            auto hex = [](char c) -> int {
                if (c >= '0' && c <= '9') return c - '0';
                if (c >= 'A' && c <= 'F') return c - 'A' + 10;
                if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                return -1;
            };
            const int hi = hex(s[3]), lo = hex(s[4]);
            if (hi < 0 || lo < 0) return -1;
            return (hi << 4) | lo;
        }

        bool is_special(int id) const
        {
            if (id == bos_ || id == eos_ || id == pad_) return true;
            if (id >= 0 && id < static_cast<int>(token_types_.size()))
                return token_types_[id] == TT_CONTROL;
            return false;
        }

        /* Longest special piece that is a prefix of text at pos; -1 if none. */
        bool match_special(const std::string& text, size_t pos, int& id, size_t& len) const
        {
            for (const auto& sp : special_pieces_)
            {
                const std::string& piece = sp.first;
                if (pos + piece.size() <= text.size() && text.compare(pos, piece.size(), piece) == 0)
                {
                    id = sp.second; len = piece.size(); return true;
                }
            }
            return false;
        }

        /* Split the raw text on special pieces, encoding the spans in between. The dummy
           space prefix (SentencePiece) is applied only to the first text fragment. */
        void encode_text(const std::string& text, bool parse_special, std::vector<int>& out) const
        {
            if (!parse_special || special_pieces_.empty())
            {
                encode_fragment(text, true, out);
                return;
            }
            size_t pos = 0, frag_start = 0;
            bool first = true;
            while (pos < text.size())
            {
                int sid = -1; size_t slen = 0;
                if (match_special(text, pos, sid, slen))
                {
                    if (pos > frag_start)
                    {
                        encode_fragment(text.substr(frag_start, pos - frag_start), first, out);
                        first = false;
                    }
                    out.push_back(sid);
                    pos += slen;
                    frag_start = pos;
                }
                else ++pos;
            }
            if (frag_start < text.size())
                encode_fragment(text.substr(frag_start), first, out);
        }

        void encode_fragment(const std::string& frag, bool first, std::vector<int>& out) const
        {
            std::vector<int> ids = (kind_ == kind::spm)
                ? encode_spm(frag, first && add_space_prefix_)
                : encode_bpe(frag);
            out.insert(out.end(), ids.begin(), ids.end());
        }

        // --- SentencePiece -------------------------------------------------------------

        struct spm_symbol { int prev; int next; const char* text; size_t n; };
        struct spm_bigram { int left; int right; float score; size_t size; };
        struct spm_bigram_cmp
        {
            bool operator()(const spm_bigram& a, const spm_bigram& b) const
            {
                return (a.score < b.score) || (a.score == b.score && a.left > b.left);
            }
        };

        std::vector<int> encode_spm(const std::string& text, bool dummy_prefix) const
        {
            static const char meta_space[] = "\xe2\x96\x81";   // U+2581
            std::string norm;
            norm.reserve(text.size() + 4);
            if (dummy_prefix) norm += meta_space;
            for (char c : text)
            {
                if (c == ' ') norm += meta_space;
                else norm.push_back(c);
            }

            std::vector<spm_symbol> syms;
            for (size_t i = 0; i < norm.size();)
            {
                size_t len = hf_tok_impl::utf8_char_len(static_cast<unsigned char>(norm[i]));
                if (i + len > norm.size()) len = 1;
                spm_symbol s;
                s.text = norm.data() + i;
                s.n = len;
                s.prev = static_cast<int>(syms.size()) - 1;
                s.next = -1;
                if (!syms.empty()) syms.back().next = static_cast<int>(syms.size());
                syms.push_back(s);
                i += len;
            }
            if (syms.empty()) return {};

            std::priority_queue<spm_bigram, std::vector<spm_bigram>, spm_bigram_cmp> work;
            auto try_add = [&](int left, int right)
            {
                if (left == -1 || right == -1) return;
                const std::string token(syms[left].text, syms[left].n + syms[right].n);
                auto it = token_to_id_.find(token);
                if (it == token_to_id_.end()) return;
                spm_bigram b;
                b.left = left; b.right = right;
                b.score = scores_[static_cast<size_t>(it->second)];
                b.size = token.size();
                work.push(b);
            };

            for (int i = 1; i < static_cast<int>(syms.size()); ++i) try_add(i - 1, i);

            while (!work.empty())
            {
                const spm_bigram b = work.top();
                work.pop();
                spm_symbol& left = syms[b.left];
                spm_symbol& right = syms[b.right];
                if (left.n == 0 || right.n == 0 || left.n + right.n != b.size) continue;

                left.n += right.n;
                right.n = 0;
                left.next = right.next;
                if (right.next != -1) syms[right.next].prev = b.left;

                try_add(left.prev, b.left);
                try_add(b.left, left.next);
            }

            std::vector<int> out;
            for (int i = 0; i != -1; i = syms[i].next)
            {
                const spm_symbol& s = syms[i];
                if (s.n == 0) continue;
                const std::string token(s.text, s.n);
                auto it = token_to_id_.find(token);
                if (it != token_to_id_.end()) { out.push_back(it->second); continue; }
                for (size_t k = 0; k < s.n; ++k)   // byte fallback
                {
                    const unsigned char b = static_cast<unsigned char>(s.text[k]);
                    const int id = byte_to_id_[b];
                    if (id >= 0) out.push_back(id);
                    else if (unk_ >= 0) out.push_back(unk_);
                }
            }
            return out;
        }

        std::string decode_spm(const std::vector<int>& ids, bool skip_special) const
        {
            static const char meta_space[] = "\xe2\x96\x81";
            std::string out;
            for (int id : ids)
            {
                if (id < 0 || id >= static_cast<int>(id_to_token_.size())) continue;
                if (skip_special && is_special(id)) continue;
                if (byte_value_[static_cast<size_t>(id)] >= 0)
                {
                    out.push_back(static_cast<char>(byte_value_[static_cast<size_t>(id)]));
                    continue;
                }
                const std::string& tok = id_to_token_[static_cast<size_t>(id)];
                size_t pos = 0;
                while (pos < tok.size())
                {
                    if (tok.compare(pos, 3, meta_space) == 0) { out.push_back(' '); pos += 3; }
                    else out.push_back(tok[pos++]);
                }
            }
            if (add_space_prefix_ && !out.empty() && out[0] == ' ') out.erase(0, 1);
            return out;
        }

        // --- Byte-level BPE -------------------------------------------------------------

        void build_byte_unicode_tables()
        {
            bool printable[256] = { false };
            for (int b = 33; b <= 126; ++b) printable[b] = true;
            for (int b = 161; b <= 172; ++b) printable[b] = true;
            for (int b = 174; b <= 255; ++b) printable[b] = true;

            byte_to_uni_.fill(0);
            uni_to_byte_.clear();
            int n = 0;
            for (int b = 0; b < 256; ++b)
            {
                const uint32_t cp = printable[b] ? static_cast<uint32_t>(b) : static_cast<uint32_t>(256 + n++);
                byte_to_uni_[static_cast<size_t>(b)] = cp;
                uni_to_byte_[cp] = static_cast<unsigned char>(b);
            }
        }

        static bool is_ws(uint32_t c)
        {
            return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
        }
        static int classify(uint32_t c)
        {
            if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) return 0;   // letter
            if (c >= '0' && c <= '9') return 1;                              // digit
            if (c >= 128) return 0;                                         // non-ASCII -> letter (approx)
            return 2;                                                       // other
        }

        /* Approximate GPT-2 pre-tokenizer. Exact for ASCII; non-ASCII codepoints are
           treated as letters. Exact parity for a specific model needs that model's
           pre-tokenizer regex (see pretokenizer()). */
        std::vector<std::string> pretokenize(const std::string& text) const
        {
            struct cp_info { uint32_t cp; size_t off; };
            std::vector<cp_info> cps;
            for (size_t i = 0; i < text.size();)
            {
                size_t len = 0;
                const uint32_t cp = hf_tok_impl::utf8_decode(text, i, len);
                cps.push_back({ cp, i });
                i += len;
            }
            const size_t n = cps.size();

            auto slice = [&](size_t a, size_t b) -> std::string {
                if (a >= b) return std::string();
                const size_t bs = cps[a].off;
                const size_t be = (b < n) ? cps[b].off : text.size();
                return text.substr(bs, be - bs);
            };

            std::vector<std::string> words;
            size_t i = 0;
            while (i < n)
            {
                const uint32_t c = cps[i].cp;

                if (c == '\'' && i + 1 < n)   // contractions
                {
                    const uint32_t c1 = cps[i + 1].cp;
                    if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd')
                    {
                        words.push_back(slice(i, i + 2)); i += 2; continue;
                    }
                    if (i + 2 < n)
                    {
                        const uint32_t c2 = cps[i + 2].cp;
                        if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l'))
                        {
                            words.push_back(slice(i, i + 3)); i += 3; continue;
                        }
                    }
                }

                if (is_ws(c))
                {
                    size_t j = i;
                    while (j < n && is_ws(cps[j].cp)) ++j;
                    const bool next_word = (j < n) && !is_ws(cps[j].cp);
                    if (next_word)
                    {
                        if (j - 1 > i) words.push_back(slice(i, j - 1));
                        const size_t start = j - 1;
                        const int cl = classify(cps[j].cp);
                        size_t k = j;
                        while (k < n && classify(cps[k].cp) == cl) ++k;
                        words.push_back(slice(start, k));
                        i = k;
                    }
                    else { words.push_back(slice(i, j)); i = j; }
                }
                else
                {
                    const int cl = classify(c);
                    size_t k = i;
                    while (k < n && classify(cps[k].cp) == cl) ++k;
                    words.push_back(slice(i, k));
                    i = k;
                }
            }
            return words;
        }

        std::vector<int> encode_bpe(const std::string& text) const
        {
            std::vector<int> out;
            for (const std::string& word : pretokenize(text))
            {
                std::vector<std::string> symbols;
                symbols.reserve(word.size());
                for (unsigned char b : word)
                    symbols.push_back(hf_tok_impl::utf8_encode(byte_to_uni_[b]));
                if (symbols.empty()) continue;

                for (;;)
                {
                    int best = -1, best_rank = std::numeric_limits<int>::max();
                    for (size_t i = 0; i + 1 < symbols.size(); ++i)
                    {
                        auto it = bpe_ranks_.find(symbols[i] + " " + symbols[i + 1]);
                        if (it != bpe_ranks_.end() && it->second < best_rank)
                        {
                            best_rank = it->second;
                            best = static_cast<int>(i);
                        }
                    }
                    if (best < 0) break;
                    symbols[best] += symbols[best + 1];
                    symbols.erase(symbols.begin() + best + 1);
                }

                for (const std::string& s : symbols)
                {
                    auto it = token_to_id_.find(s);
                    if (it != token_to_id_.end()) out.push_back(it->second);
                    else if (unk_ >= 0) out.push_back(unk_);
                }
            }
            return out;
        }

        std::string decode_bpe(const std::vector<int>& ids, bool skip_special) const
        {
            std::string uni;
            for (int id : ids)
            {
                if (id < 0 || id >= static_cast<int>(id_to_token_.size())) continue;
                if (skip_special && is_special(id)) continue;
                uni += id_to_token_[static_cast<size_t>(id)];
            }
            std::string out;
            out.reserve(uni.size());
            for (size_t i = 0; i < uni.size();)
            {
                size_t len = 0;
                const uint32_t cp = hf_tok_impl::utf8_decode(uni, i, len);
                auto it = uni_to_byte_.find(cp);
                if (it != uni_to_byte_.end()) out.push_back(static_cast<char>(it->second));
                i += len;
            }
            return out;
        }

        // --- data ----------------------------------------------------------------------

        kind kind_ = kind::spm;
        std::vector<std::string> id_to_token_;
        std::vector<float> scores_;
        std::vector<int> token_types_;
        std::vector<std::string> bpe_merges_;
        int bos_ = -1, eos_ = -1, unk_ = -1, pad_ = -1;
        bool add_bos_ = false, add_eos_ = false, add_space_prefix_ = true;
        std::string pre_;

        /* Derived, rebuilt by finalize(); not serialized. */
        std::unordered_map<std::string, int> token_to_id_;
        std::array<int, 256> byte_to_id_{};
        std::vector<int> byte_value_;
        std::vector<std::pair<std::string, int>> special_pieces_;
        std::unordered_map<std::string, int> bpe_ranks_;
        std::array<uint32_t, 256> byte_to_uni_{};
        std::unordered_map<uint32_t, unsigned char> uni_to_byte_;
    };
}

#endif // DLIB_HF_TOKENIZER_Hh_
