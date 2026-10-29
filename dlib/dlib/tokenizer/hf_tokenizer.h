// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_HF_TOKENIZER_H_
#define DLIB_HF_TOKENIZER_H_

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
#include "hf_unicode.h"
#include "../data_io/gguf_reader.h"
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

        // ------------------------------------------------------------------------------------
        //                                  Pre-tokenizers
        // ------------------------------------------------------------------------------------

        /*  The pre-tokenizers of the byte-level BPE vocabularies, each the hand-written
            equivalent of the regular expression its vocabulary was trained with: the
            alternatives are tried in their order at the current position and the first
            that matches consumes its piece, with the backtracking of the greedy
            quantifiers reproduced where it changes the result.  Each one is checked
            against the expression itself in dlib/test/tokenizer.cpp.  */
        enum class pre_family
        {
            gpt2,         // 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
            gpt2_digits,  // \p{N} isolated first, then gpt2 (StarCoder, SmolLM, Granite 3)
            llama3,       // (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
            qwen2,        // the same with \p{N} one digit at a time (Qwen 2 and 3)
            qwen35,       // qwen2 with the marks \p{M} taken as letters (Qwen 3.5)
            tekken        // [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|
                          // [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|
                          // \p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+  (Mistral)
        };

        /*  The family of a tokenizer.ggml.pre name.  exact is false for the names this
            file does not know, which receive the qwen2 pattern, the closest to most
            modern vocabularies.  */
        inline pre_family pre_family_of(const std::string& pre, bool& exact)
        {
            exact = true;
            if (pre == "qwen2" || pre == "stablelm2" || pre == "hunyuan" || pre == "solar-open" ||
                pre == "grok-2" || pre == "laguna")
                return pre_family::qwen2;
            if (pre == "qwen35")
                return pre_family::qwen35;
            if (pre == "llama3" || pre == "llama-v3" || pre == "llama-bpe" || pre == "falcon3" ||
                pre == "falcon-h1" || pre == "midm-2.0" || pre == "lfm2" || pre == "dbrx" ||
                pre == "smaug-bpe" || pre == "chatglm-bpe" || pre == "jina-v5-nano")
                return pre_family::llama3;
            if (pre == "tekken")
                return pre_family::tekken;
            if (pre == "smollm" || pre == "starcoder" || pre == "command-r" || pre == "codeshell" ||
                pre == "exaone" || pre == "minerva")
                return pre_family::gpt2_digits;
            /* "refact" is the name the conversion tools give to the vocabulary of the
               Granite 3 models, whose reference tokenizer applies the plain gpt2 pattern
               (its vocabulary holds no merge of digits, so isolating them changes nothing
               but the attachment of a space to the other numerals, such as "½"). */
            if (pre == "gpt-2" || pre == "gpt2" || pre == "phi-2" || pre == "olmo" || pre == "mpt" ||
                pre == "jais" || pre == "trillion" || pre == "granite-docling" || pre == "refact")
                return pre_family::gpt2;
            exact = false;
            if (pre.empty() || pre == "default") return pre_family::gpt2;
            return pre_family::qwen2;
        }

        /*  Whether the reference tokenizer of a byte-level vocabulary normalizes its
            input to the canonical composition (NFC): the Qwen family.  */
        inline bool pre_normalizes_nfc(const std::string& pre)
        {
            return pre == "qwen2" || pre == "qwen35";
        }

        /*  The canonical composition of text: a decomposed character, such as a letter
            followed by a combining accent, becomes the precomposed character the
            vocabulary was built from.  The text is returned unchanged when it holds no
            code point at or above U+0300, which is the case of nearly every input.
            Otherwise the combining marks are put in canonical order and composed with
            the character they follow, under the blocking rule of the standard; the
            precomposed characters of the input are not decomposed first, which only
            matters when they are followed by further marks.  Bytes that are not valid
            UTF-8 are kept as they are.  */
        inline std::string nfc_compose(const std::string& text)
        {
            bool needed = false;
            for (unsigned char c : text)
                if (c >= 0xCC) { needed = true; break; }
            if (!needed) return text;

            struct unit { uint32_t cp; unsigned cc; std::string bytes; };
            const uint32_t invalid = 0xFFFFFFFFu;
            std::vector<unit> units;
            for (size_t i = 0; i < text.size();)
            {
                const unsigned char lead = static_cast<unsigned char>(text[i]);
                const size_t len = static_cast<size_t>(utf8_char_len(lead));
                bool valid = (lead < 0x80 || len > 1) && i + len <= text.size();
                for (size_t k = 1; valid && k < len; ++k)
                    valid = (static_cast<unsigned char>(text[i + k]) & 0xC0) == 0x80;
                if (!valid)
                {
                    units.push_back(unit{ invalid, 0, text.substr(i, 1) });
                    ++i;
                    continue;
                }
                size_t dummy = 0;
                const uint32_t c = utf8_decode(text, i, dummy);
                units.push_back(unit{ c, hf_unicode::combining_class(c), text.substr(i, len) });
                i += len;
            }

            // Canonical order: each run of marks sorted by class, stably.
            for (size_t i = 0; i < units.size();)
            {
                if (units[i].cc == 0) { ++i; continue; }
                size_t j = i;
                while (j < units.size() && units[j].cc != 0) ++j;
                std::stable_sort(units.begin() + static_cast<std::ptrdiff_t>(i), units.begin() + static_cast<std::ptrdiff_t>(j),
                    [](const unit& a, const unit& b) { return a.cc < b.cc; });
                i = j;
            }

            // Composition: a character joins the last starter unless a character between
            // them has a class of 0 or one at least its own.
            std::vector<unit> out;
            long starter = -1;
            unsigned last_cc = 0;
            for (unit& u : units)
            {
                if (u.cp == invalid)
                {
                    out.push_back(u);
                    starter = -1;
                    continue;
                }
                if (starter >= 0)
                {
                    const bool adjacent = static_cast<long>(out.size()) == starter + 1;
                    if (adjacent || (last_cc != 0 && last_cc < u.cc))
                    {
                        const uint32_t m = hf_unicode::compose(out[static_cast<size_t>(starter)].cp, u.cp);
                        if (m)
                        {
                            out[static_cast<size_t>(starter)].cp = m;
                            out[static_cast<size_t>(starter)].bytes = utf8_encode(m);
                            continue;
                        }
                    }
                }
                out.push_back(u);
                if (u.cc == 0)
                {
                    starter = static_cast<long>(out.size()) - 1;
                    last_cc = 0;
                }
                else
                {
                    last_cc = u.cc;
                }
            }
            std::string r;
            r.reserve(text.size());
            for (const unit& u : out) r += u.bytes;
            return r;
        }

        /*  Whether a byte-level vocabulary skips the merges of a piece that is itself a
            token (ignore_merges of the reference tokenizers of the llama 3 and Mistral
            families).  */
        inline bool pre_ignores_merges(const std::string& pre)
        {
            return pre == "llama3" || pre == "llama-v3" || pre == "llama-bpe" || pre == "falcon3" ||
                pre == "falcon-h1" || pre == "midm-2.0" || pre == "lfm2" || pre == "jina-v5-nano" ||
                pre == "tekken";
        }

        struct cp_class_cache
        {
            /*  The class of the code points of one text, computed once.  */
            std::vector<uint32_t> cp;
            std::vector<hf_unicode::char_class> cls;
            std::vector<size_t> off;    // byte offset of each code point, then the size

            void assign(const std::string& text)
            {
                cp.clear(); cls.clear(); off.clear();
                for (size_t i = 0; i < text.size();)
                {
                    size_t len = 0;
                    const uint32_t c = utf8_decode(text, i, len);
                    cp.push_back(c);
                    cls.push_back(hf_unicode::classify(c));
                    off.push_back(i);
                    i += len;
                }
                off.push_back(text.size());
            }
        };

        inline bool is_letter(hf_unicode::char_class k) { return k >= hf_unicode::Lu && k <= hf_unicode::Lo; }
        inline bool is_nl(uint32_t c) { return c == '\r' || c == '\n'; }
        inline uint32_t ascii_lower(uint32_t c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }

        // End of the whitespace alternatives \s*[\r\n]+|\s+(?!\S)|\s+ at i, a white space.
        inline size_t match_spaces(const cp_class_cache& t, size_t i)
        {
            const size_t n = t.cp.size();
            size_t j = i;
            size_t last_nl = n;
            while (j < n && hf_unicode::is_white_space(t.cp[j]))
            {
                if (is_nl(t.cp[j])) last_nl = j;
                ++j;
            }
            if (last_nl < n) return last_nl + 1;      // \s*[\r\n]+, up to the last newline
            if (j == n || j - i == 1) return j;       // \s+(?!\S) at the end, or \s+
            return j - 1;                             // \s+(?!\S): the last blank stays
        }

        // (?i:'s|'t|'re|'ve|'m|'ll|'d), or the same in lower case only.
        inline size_t match_contraction(const cp_class_cache& t, size_t i, bool any_case)
        {
            const size_t n = t.cp.size();
            if (t.cp[i] != '\'' || i + 1 >= n) return i;
            const uint32_t a = any_case ? ascii_lower(t.cp[i + 1]) : t.cp[i + 1];
            if (a == 's' || a == 't' || a == 'm' || a == 'd') return i + 2;
            if (i + 2 < n)
            {
                const uint32_t b = any_case ? ascii_lower(t.cp[i + 2]) : t.cp[i + 2];
                if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return i + 3;
            }
            return i;
        }

        // The llama3, qwen2 and qwen35 patterns at i.
        inline size_t match_gpt4_style(const cp_class_cache& t, size_t i, size_t max_digits, bool marks_are_letters)
        {
            using namespace hf_unicode;
            const size_t n = t.cp.size();
            auto letter = [&](size_t k) { return is_letter(t.cls[k]) || (marks_are_letters && t.cls[k] == M); };
            auto number = [&](size_t k) { return t.cls[k] == N; };
            size_t e = match_contraction(t, i, true);
            if (e > i) return e;
            // [^\r\n\p{L}\p{N}]?\p{L}+
            if (letter(i))
            {
                e = i + 1;
                while (e < n && letter(e)) ++e;
                return e;
            }
            if (!is_nl(t.cp[i]) && !is_letter(t.cls[i]) && !number(i) && i + 1 < n && letter(i + 1))
            {
                e = i + 2;
                while (e < n && letter(e)) ++e;
                return e;
            }
            // \p{N}{1,max_digits}
            if (number(i))
            {
                e = i + 1;
                while (e < n && e - i < max_digits && number(e)) ++e;
                return e;
            }
            //  ?[^\s\p{L}\p{N}]+[\r\n]*
            auto punct = [&](size_t k) { return !is_white_space(t.cp[k]) && !letter(k) && !number(k); };
            size_t s = (t.cp[i] == ' ' && i + 1 < n && punct(i + 1)) ? i + 1 : i;
            if (punct(s))
            {
                e = s + 1;
                while (e < n && punct(e)) ++e;
                while (e < n && is_nl(t.cp[e])) ++e;
                return e;
            }
            return match_spaces(t, i);
        }

        // End of \s+(?!\S)|\s+ at i, a white space: the alternatives of gpt2, which has
        // no rule of its own for the newlines.
        inline size_t match_spaces_gpt2(const cp_class_cache& t, size_t i)
        {
            const size_t n = t.cp.size();
            size_t j = i;
            while (j < n && hf_unicode::is_white_space(t.cp[j])) ++j;
            if (j == n || j - i == 1) return j;
            return j - 1;
        }

        // The gpt2 pattern at i.
        inline size_t match_gpt2(const cp_class_cache& t, size_t i)
        {
            using namespace hf_unicode;
            const size_t n = t.cp.size();
            size_t e = match_contraction(t, i, false);
            if (e > i) return e;
            auto letter = [&](size_t k) { return is_letter(t.cls[k]); };
            auto number = [&](size_t k) { return t.cls[k] == N; };
            auto punct = [&](size_t k) { return !is_white_space(t.cp[k]) && !letter(k) && !number(k); };
            const bool space = t.cp[i] == ' ' && i + 1 < n;
            //  ?\p{L}+,  ?\p{N}+ and  ?[^\s\p{L}\p{N}]+, in this order.
            for (int rule = 0; rule < 3; ++rule)
            {
                auto in_class = [&](size_t k) { return rule == 0 ? letter(k) : rule == 1 ? number(k) : punct(k); };
                size_t s = n;
                if (in_class(i)) s = i;
                else if (space && in_class(i + 1)) s = i + 1;
                if (s < n)
                {
                    e = s + 1;
                    while (e < n && in_class(e)) ++e;
                    return e;
                }
            }
            return match_spaces_gpt2(t, i);
        }

        // The tekken pattern at i.
        inline size_t match_tekken(const cp_class_cache& t, size_t i)
        {
            using namespace hf_unicode;
            const size_t n = t.cp.size();
            auto upper = [&](size_t k) { const char_class c = t.cls[k]; return c == Lu || c == Lt || c == Lm || c == Lo || c == M; };
            auto lower = [&](size_t k) { const char_class c = t.cls[k]; return c == Ll || c == Lm || c == Lo || c == M; };
            const bool lead = !is_nl(t.cp[i]) && !is_letter(t.cls[i]) && t.cls[i] != N;

            // [^\r\n\p{L}\p{N}]?U*W+ : the longest run of U that leaves a W after it.
            for (int with_lead = 1; with_lead >= 0; --with_lead)
            {
                if (with_lead && !lead) continue;
                const size_t s = i + static_cast<size_t>(with_lead);
                if (s >= n) continue;
                size_t u = s;
                while (u < n && upper(u)) ++u;
                for (size_t k = u + 1; k-- > s;)
                {
                    if (k < n && lower(k))
                    {
                        size_t e = k + 1;
                        while (e < n && lower(e)) ++e;
                        return e;
                    }
                }
            }
            // [^\r\n\p{L}\p{N}]?U+W*
            for (int with_lead = 1; with_lead >= 0; --with_lead)
            {
                if (with_lead && !lead) continue;
                const size_t s = i + static_cast<size_t>(with_lead);
                if (s < n && upper(s))
                {
                    size_t e = s + 1;
                    while (e < n && upper(e)) ++e;
                    while (e < n && lower(e)) ++e;
                    return e;
                }
            }
            if (t.cls[i] == N) return i + 1;
            //  ?[^\s\p{L}\p{N}]+[\r\n/]*
            auto punct = [&](size_t k) { return !is_white_space(t.cp[k]) && !is_letter(t.cls[k]) && t.cls[k] != N; };
            const size_t s = (t.cp[i] == ' ' && i + 1 < n && punct(i + 1)) ? i + 1 : i;
            if (punct(s))
            {
                size_t e = s + 1;
                while (e < n && punct(e)) ++e;
                while (e < n && (is_nl(t.cp[e]) || t.cp[e] == '/')) ++e;
                return e;
            }
            return match_spaces(t, i);
        }

        /*  Splits text into the pieces BPE merges separately, as byte ranges
            [first, second) of text.  */
        inline void pretokenize(const std::string& text, pre_family family,
            std::vector<std::pair<size_t, size_t>>& pieces)
        {
            pieces.clear();
            cp_class_cache t;
            t.assign(text);
            const size_t n = t.cp.size();
            size_t i = 0;
            while (i < n)
            {
                size_t e = i + 1;
                switch (family)
                {
                case pre_family::gpt2: e = match_gpt2(t, i); break;
                case pre_family::gpt2_digits:
                    if (t.cls[i] == hf_unicode::N)
                    {
                        e = i + 1;
                    }
                    else
                    {
                        // The pattern runs on the text between two digits only.
                        size_t stop = i;
                        while (stop < n && t.cls[stop] != hf_unicode::N) ++stop;
                        cp_class_cache part;
                        part.cp.assign(t.cp.begin() + i, t.cp.begin() + stop);
                        part.cls.assign(t.cls.begin() + i, t.cls.begin() + stop);
                        size_t k = 0;
                        while (k < part.cp.size())
                        {
                            const size_t ke = match_gpt2(part, k);
                            pieces.emplace_back(t.off[i + k], t.off[i + ke]);
                            k = ke;
                        }
                        i = stop;
                        continue;
                    }
                    break;
                case pre_family::llama3: e = match_gpt4_style(t, i, 3, false); break;
                case pre_family::qwen2: e = match_gpt4_style(t, i, 1, false); break;
                case pre_family::qwen35: e = match_gpt4_style(t, i, 1, true); break;
                case pre_family::tekken: e = match_tekken(t, i); break;
                }
                if (e <= i) e = i + 1;
                pieces.emplace_back(t.off[i], t.off[e]);
                i = e;
            }
        }
    }

    // ----------------------------------------------------------------------------------------

    class hf_tokenizer
    {
    public:
        enum class kind { spm, bpe };

        // GGUF token-type values (the common GGUF convention).
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
            /* The declared chat template travels with the tokenizer so any program
               (including one loading a serialized model archive) can identify the
               conversation format without the source container. */
            chat_template_ = g.get_str("tokenizer.chat_template", "");

            load(v);
        }

        size_t size() const { return id_to_token_.size(); }
        kind type() const { return kind_; }
        int bos_id() const { return bos_; }
        int eos_id() const { return eos_; }

        // The chat template declared by the source model (raw Jinja string from
        // tokenizer.chat_template), or an empty string when the model declares none.
        // Persisted with the tokenizer, so it survives model archive round-trips;
        // chat_template_formatter fingerprints it to select the conversation format.
        const std::string& chat_template() const { return chat_template_; }
        int unk_id() const { return unk_; }
        int pad_id() const { return pad_; }
        bool add_bos_default() const { return add_bos_; }
        bool add_eos_default() const { return add_eos_; }
        const std::string& pretokenizer() const { return pre_; }
        const std::string& id_to_token(int id) const { return id_to_token_.at(static_cast<size_t>(id)); }

        bool exact_pretokenizer() const { return kind_ == kind::spm || pre_exact_; }

        int token_type(int id) const
        {
            return (id >= 0 && id < static_cast<int>(token_types_.size())) ? token_types_[static_cast<size_t>(id)] : TT_UNDEFINED;
        }

        int token_id(const std::string& piece) const
        {
            auto it = token_to_id_.find(piece);
            return it == token_to_id_.end() ? -1 : it->second;
        }

        std::string token_bytes(int id) const
        {
            std::string out;
            append_token_bytes(id, out);
            return out;
        }

        std::vector<int> encode(const std::string& text) const
        {
            return encode(text, add_bos_, add_eos_, true);
        }

        std::vector<int> encode(const std::string& text, bool add_bos, bool add_eos,
            bool parse_special = true, bool allow_space_prefix = true) const
        {
            std::vector<int> out;
            if (add_bos && bos_ >= 0) out.push_back(bos_);
            encode_text(text, parse_special, allow_space_prefix, out);
            if (add_eos && eos_ >= 0) out.push_back(eos_);
            return out;
        }

        std::string decode(const std::vector<int>& ids, bool skip_special = true) const
        {
            std::string out;
            for (int id : ids)
            {
                if (id < 0 || id >= static_cast<int>(id_to_token_.size())) continue;
                if (skip_special && is_special(id)) continue;
                append_token_bytes(id, out);
            }
            if (kind_ == kind::spm && add_space_prefix_ && !out.empty() && out[0] == ' ') out.erase(0, 1);
            return out;
        }

        friend void serialize(const hf_tokenizer& t, std::ostream& out)
        {
            dlib::serialize(std::string("hf_tokenizer"), out);
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
            dlib::serialize(t.chat_template_, out);
        }

        friend void deserialize(hf_tokenizer& t, std::istream& in)
        {
            std::string version;
            dlib::deserialize(version, in);
            if (version != "hf_tokenizer")
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
            dlib::deserialize(t.chat_template_, in);
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
            /* Longest pieces first so matching is greedy and unambiguous, grouped by their
               first byte so that a position is only compared with the pieces that can
               start there. */
            std::sort(special_pieces_.begin(), special_pieces_.end(),
                [](const std::pair<std::string, int>& a, const std::pair<std::string, int>& b)
                { return a.first.size() > b.first.size(); });
            for (auto& bucket : special_by_byte_) bucket.clear();
            for (size_t i = 0; i < special_pieces_.size(); ++i)
                special_by_byte_[static_cast<unsigned char>(special_pieces_[i].first[0])].push_back(i);

            pre_family_ = hf_tok_impl::pre_family_of(pre_, pre_exact_);
            ignore_merges_ = hf_tok_impl::pre_ignores_merges(pre_);
            nfc_ = kind_ == kind::bpe && hf_tok_impl::pre_normalizes_nfc(pre_);
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
            for (size_t index : special_by_byte_[static_cast<unsigned char>(text[pos])])
            {
                const auto& sp = special_pieces_[index];
                const std::string& piece = sp.first;
                if (pos + piece.size() <= text.size() && text.compare(pos, piece.size(), piece) == 0)
                {
                    id = sp.second; len = piece.size(); return true;
                }
            }
            return false;
        }

        /* Split the raw text on special pieces, encoding the spans in between. The dummy
           space prefix (SentencePiece) is applied only to the first text fragment, and only
           when the caller allows it; a continuation turn passes allow_space_prefix=false so no
           leading space token is inserted mid-conversation. */
        void encode_text(const std::string& text, bool parse_special, bool allow_space_prefix,
            std::vector<int>& out) const
        {
            if (!parse_special || special_pieces_.empty())
            {
                encode_fragment(text, true, allow_space_prefix, out);
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
                        encode_fragment(text.substr(frag_start, pos - frag_start), first, allow_space_prefix, out);
                        first = false;
                    }
                    out.push_back(sid);
                    pos += slen;
                    frag_start = pos;
                }
                else ++pos;
            }
            if (frag_start < text.size())
                encode_fragment(text.substr(frag_start), first, allow_space_prefix, out);
        }

        void encode_fragment(const std::string& frag, bool first, bool allow_space_prefix,
            std::vector<int>& out) const
        {
            std::vector<int> ids = (kind_ == kind::spm)
                ? encode_spm(frag, first && allow_space_prefix && add_space_prefix_)
                : encode_bpe(nfc_ ? hf_tok_impl::nfc_compose(frag) : frag);
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

        /* The bytes a token stands for, appended to out: the byte of a byte token, the
           text of a special token as written, and otherwise the vocabulary string with
           the meta-space of SentencePiece or the byte remapping of byte-level BPE undone. */
        void append_token_bytes(int id, std::string& out) const
        {
            if (id < 0 || id >= static_cast<int>(id_to_token_.size())) return;
            const size_t i = static_cast<size_t>(id);
            if (byte_value_[i] >= 0)
            {
                out.push_back(static_cast<char>(byte_value_[i]));
                return;
            }
            const std::string& tok = id_to_token_[i];
            const int tt = token_types_[i];
            if (tt == TT_CONTROL || tt == TT_USER_DEFINED)
            {
                out += tok;
                return;
            }
            if (kind_ == kind::spm)
            {
                static const char meta_space[] = "\xe2\x96\x81";
                size_t pos = 0;
                while (pos < tok.size())
                {
                    if (tok.compare(pos, 3, meta_space) == 0) { out.push_back(' '); pos += 3; }
                    else out.push_back(tok[pos++]);
                }
                return;
            }
            for (size_t k = 0; k < tok.size();)
            {
                size_t len = 0;
                const uint32_t cp = hf_tok_impl::utf8_decode(tok, k, len);
                auto it = uni_to_byte_.find(cp);
                if (it != uni_to_byte_.end()) out.push_back(static_cast<char>(it->second));
                k += len;
            }
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
                byte_to_piece_[static_cast<size_t>(b)] = hf_tok_impl::utf8_encode(cp);
                uni_to_byte_[cp] = static_cast<unsigned char>(b);
            }
        }

        struct bpe_symbol { int prev; int next; size_t start; size_t len; };
        struct bpe_bigram { int left; int right; int rank; size_t size; };
        struct bpe_bigram_cmp
        {
            // Lowest rank first, then leftmost, as the reference implementations merge.
            bool operator()(const bpe_bigram& a, const bpe_bigram& b) const
            {
                return a.rank > b.rank || (a.rank == b.rank && a.left > b.left);
            }
        };

        /* Applies the merges to one pre-tokenized piece, already remapped to the
           byte-level alphabet, with a queue of the candidate pairs ordered by rank: the
           cost grows with the length of the piece times the log of it, so that a long run
           of characters without spaces (a hash, a base64 block) stays cheap. */
        void encode_bpe_piece(const std::string& word, std::vector<int>& out) const
        {
            if (word.empty()) return;
            if (ignore_merges_)
            {
                auto it = token_to_id_.find(word);
                if (it != token_to_id_.end()) { out.push_back(it->second); return; }
            }
            std::vector<bpe_symbol> syms;
            for (size_t i = 0; i < word.size();)
            {
                size_t len = hf_tok_impl::utf8_char_len(static_cast<unsigned char>(word[i]));
                if (i + len > word.size()) len = 1;
                bpe_symbol s;
                s.prev = static_cast<int>(syms.size()) - 1;
                s.next = -1;
                s.start = i;
                s.len = len;
                if (!syms.empty()) syms.back().next = static_cast<int>(syms.size());
                syms.push_back(s);
                i += len;
            }

            std::priority_queue<bpe_bigram, std::vector<bpe_bigram>, bpe_bigram_cmp> work;
            std::string key;
            auto try_add = [&](int left, int right)
            {
                if (left < 0 || right < 0) return;
                const bpe_symbol& l = syms[static_cast<size_t>(left)];
                const bpe_symbol& r = syms[static_cast<size_t>(right)];
                key.assign(word, l.start, l.len);
                key.push_back(' ');
                key.append(word, r.start, r.len);
                auto it = bpe_ranks_.find(key);
                if (it == bpe_ranks_.end()) return;
                work.push(bpe_bigram{ left, right, it->second, l.len + r.len });
            };
            for (int i = 1; i < static_cast<int>(syms.size()); ++i) try_add(i - 1, i);

            while (!work.empty())
            {
                const bpe_bigram b = work.top();
                work.pop();
                bpe_symbol& l = syms[static_cast<size_t>(b.left)];
                bpe_symbol& r = syms[static_cast<size_t>(b.right)];
                // Skip the pairs a previous merge made obsolete.
                if (l.len == 0 || r.len == 0 || l.len + r.len != b.size || l.next != b.right) continue;
                l.len += r.len;
                r.len = 0;
                l.next = r.next;
                if (r.next >= 0) syms[static_cast<size_t>(r.next)].prev = b.left;
                try_add(l.prev, b.left);
                try_add(b.left, l.next);
            }

            for (int i = 0; i != -1; i = syms[static_cast<size_t>(i)].next)
            {
                const bpe_symbol& s = syms[static_cast<size_t>(i)];
                if (s.len == 0) continue;
                auto it = token_to_id_.find(word.substr(s.start, s.len));
                if (it != token_to_id_.end()) { out.push_back(it->second); continue; }
                // A symbol missing from the vocabulary: its characters one by one.
                for (size_t k = s.start; k < s.start + s.len;)
                {
                    size_t len = hf_tok_impl::utf8_char_len(static_cast<unsigned char>(word[k]));
                    if (k + len > s.start + s.len) len = 1;
                    auto c = token_to_id_.find(word.substr(k, len));
                    if (c != token_to_id_.end()) out.push_back(c->second);
                    else if (unk_ >= 0) out.push_back(unk_);
                    k += len;
                }
            }
        }

        std::vector<int> encode_bpe(const std::string& text) const
        {
            std::vector<int> out;
            std::vector<std::pair<size_t, size_t>> pieces;
            hf_tok_impl::pretokenize(text, pre_family_, pieces);
            std::string word;
            for (const auto& p : pieces)
            {
                word.clear();
                for (size_t k = p.first; k < p.second; ++k)
                    word += byte_to_piece_[static_cast<unsigned char>(text[k])];
                encode_bpe_piece(word, out);
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
        std::string chat_template_;   // raw declared chat template (may be empty)

        /* Derived, rebuilt by finalize(); not serialized. */
        std::unordered_map<std::string, int> token_to_id_;
        std::array<int, 256> byte_to_id_{};
        std::vector<int> byte_value_;
        std::vector<std::pair<std::string, int>> special_pieces_;
        std::array<std::vector<size_t>, 256> special_by_byte_;
        std::unordered_map<std::string, int> bpe_ranks_;
        std::array<uint32_t, 256> byte_to_uni_{};
        std::array<std::string, 256> byte_to_piece_;
        std::unordered_map<uint32_t, unsigned char> uni_to_byte_;
        hf_tok_impl::pre_family pre_family_ = hf_tok_impl::pre_family::gpt2;
        bool pre_exact_ = true;
        bool ignore_merges_ = false;
        bool nfc_ = false;
    };
}

#endif // DLIB_HF_TOKENIZER_H_
