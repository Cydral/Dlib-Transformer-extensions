// Copyright (C) 2025 Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_BPE_TOKENIZER_ABSTRACT_
#ifdef DLIB_BPE_TOKENIZER_ABSTRACT_

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <iostream>
#include <mutex>
#include <thread>
#include <future>
#include <queue>

namespace dlib
{

    enum class bpe_segmentation
    {
        whitespace_split = 0,
        space_prefixed = 1
    };
    /*!
        WHAT THIS OBJECT REPRESENTS
            How bpe_tokenizer cuts a text into the segments that merges are learned within and
            applied to. A merge never joins two segments.

            - whitespace_split: every whitespace byte (space, newline, tab, carriage return) is a
              segment of its own, and every run of other bytes is a segment. This is the
              historical behaviour and the default.
            - space_prefixed: as whitespace_split, except that a space directly followed by a
              non-whitespace byte opens the segment that follows it. A word and the space before
              it can then become one token, which shortens the token sequence of prose by more
              than a quarter.
    !*/

    class bpe_tokenizer
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This class implements a Byte Pair Encoding (BPE) tokenizer, which is a subword
                tokenization algorithm commonly used in natural language processing (NLP). The
                BPE algorithm iteratively merges the most frequent pairs of bytes or characters
                to form a vocabulary of subword units. This approach is particularly useful for
                handling out-of-vocabulary words and reducing the size of the vocabulary while
                maintaining the ability to represent any text.

                Key Features:
                - Base vocabulary of 256 single-byte tokens (0-255)
                - 28 predefined special tokens (e.g., <text>, <url>, <unk>, etc.)
                - Dynamic vocabulary expansion through merges during training
                - Configurable training with:
                  * Vocabulary size control
                  * Byte-level truncation (via max_bytes)
                  * Progress reporting
                - A choice of segmentation (see bpe_segmentation)
                - Access to the merge table, so that a tokenizer can be stored as data and
                  rebuilt from it

                TRAINING COST
                    The merge table depends on the segments only through their frequencies.
                    Training therefore holds each distinct segment once with its count, derives
                    the pair counts after each merge from those before it by updating only the
                    segments that held the merged pair, and takes the next merge from a priority
                    queue. Its cost grows with the number of distinct segments, which increases
                    far more slowly than the length of the text, and the merge table is the one
                    the plain algorithm produces, with the same pairs, order and tie-breaking.

                ENCODING COST
                    Each distinct segment of a text is encoded once, by applying the merge of
                    lowest rank among its adjacent pairs until none applies, and its tokens are
                    reused wherever it occurs again. The tokens are those of applying every merge
                    in order over the whole text.

                REFERENCES
                    - Sennrich, R., Haddow, B., & Birch, A. (2016). Neural Machine Translation of
                      Rare Words with Subword Units. In Proceedings of the 54th Annual Meeting of
                      the Association for Computational Linguistics (ACL 2016).
        */

    public:
        bpe_tokenizer();
        /*!
            ensures
                - Initializes the tokenizer with a base vocabulary of single-byte tokens (0-255).
                - Pre-defines special tokens and assigns them unique IDs starting from 256.
        !*/

        void train(
            const std::string& text,
            size_t vocab_size,
            size_t max_bytes = 0,
            bool verbose = false
        );
        /*!
            requires
                - vocab_size >= 256
            ensures
                - Trains the tokenizer on the provided text corpus, or up to `max_bytes`.
                - Iteratively merges the most frequent pairs of tokens to form a subword vocabulary
                  of size `vocab_size`. Ties go to the smallest pair (left token first, then
                  right token).
                - Segments the text as get_segmentation() says.
                - If max_bytes==0, uses entire text.
                - If `verbose` is true, progress information is printed to the standard output.
        !*/

        std::vector<int> encode(
            const std::string& text
        ) const;
        /*!
            ensures
                - Encodes the input text into a sequence of subword tokens.
                - Special tokens are automatically added to mark the beginning and end of paragraphs.
                - Returns a vector of token IDs representing the encoded text.
        !*/

        std::string decode(
            const std::vector<int>& ids,
            bool display_special_tokens = true
        ) const;
        /*!
            ensures
                - Decodes a sequence of token IDs back into a human-readable string.
                - If `display_special_tokens` is true, special tokens are included in the output.
                - Returns the decoded text as a UTF-8 encoded string.
        !*/

        std::string decode(
            int id,
            bool display_special_tokens = true
        ) const;
        /*!
            ensures
                - decode a single token back into text.
        !*/

        int get_special_token_id(
            const std::string& token
        ) const;
        /*!
            ensures
                - Returns the ID of the specified special token.
                - Throws an exception if the token is not found in the special tokens map.
        !*/

        size_t get_specials_size() const;
        /*!
            ensures
                - Returns count of special tokens
        */

        size_t get_vocab_size() const;
        /*!
            ensures
                - Returns the total size of the vocabulary, including base tokens and special tokens.
        !*/

        size_t get_vocab_without_specials_size() const;
        /*!
            ensures
                - Returns vocabulary size excluding special tokens
        */

        void set_segmentation(
            bpe_segmentation mode
        );
        /*!
            ensures
                - #get_segmentation() == mode
                - If mode differs from get_segmentation(), resets the tokenizer to its base
                  vocabulary, since merges learned under one segmentation do not apply under
                  another. Call it before train() or set_merges().
        !*/

        bpe_segmentation get_segmentation(
        ) const;
        /*!
            ensures
                - Returns the segmentation used by train() and encode().
                  bpe_segmentation::whitespace_split by default.
        !*/

        std::vector<std::pair<int, int>> get_merges(
        ) const;
        /*!
            ensures
                - Returns the merge table: element i holds the (left, right) pair of tokens that
                  merge i joins into token 256 + i.
        !*/

        void set_merges(
            const std::vector<std::pair<int, int>>& merges
        );
        /*!
            requires
                - for all valid i: 0 <= merges[i].first < 256 + i and 0 <= merges[i].second < 256 + i
            ensures
                - Rebuilds the tokenizer from a merge table, as get_merges() returns it, keeping
                  the current segmentation. The token patterns, the vocabulary size and the
                  special token ids are derived from it.
                - #get_merges() == merges
            throws
                - std::invalid_argument if a merge refers to a token that does not precede it.
        !*/
    };

    void serialize(
        const bpe_tokenizer& tok,
        std::ostream& out
    );
    /*!
        ensures
            - Saves the entire state of tok to out.
            - A tokenizer with the whitespace_split segmentation is written in the historical
              format, byte for byte. Any other segmentation is written under its own tag.
    !*/

    void deserialize(
        bpe_tokenizer& tok,
        std::istream& in
    );
    /*!
        ensures
            - Restores the state of a bpe_tokenizer from a serialized state.
    !*/
}

#endif // DLIB_BPE_TOKENIZER_ABSTRACT_
