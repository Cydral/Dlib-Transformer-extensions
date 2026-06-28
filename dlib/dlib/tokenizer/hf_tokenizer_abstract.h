// Copyright (C) 2025 Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_HF_TOKENIZER_ABSTRACT_
#ifdef DLIB_HF_TOKENIZER_ABSTRACT_

#include <string>
#include <vector>
#include <array>
#include <queue>
#include <unordered_map>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace dlib
{

    class hf_tokenizer
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object is a subword tokenizer that reproduces the tokenization of
                pretrained, open-weight language models distributed in the GGUF container
                (Llama, Mistral, Qwen, Gemma and similar decoder-only models). Unlike
                bpe_tokenizer, which is trained from scratch on a corpus, this object is
                loaded from an already-trained vocabulary and only performs token<->string
                conversion. It is intended to accompany a model imported by the GGUF
                conversion tool, and is serialized next to that model.

                It covers the two tokenizer families used by current open-weight models,
                selected automatically from the vocabulary metadata:

                  - SentencePiece (kind::spm): used by Llama 1/2, Mistral and Gemma. Encoding
                    performs the score-driven bigram merge of SentencePiece, with byte
                    fallback for characters that are not in the vocabulary. encode and decode
                    are exact.

                  - Byte-level BPE (kind::bpe): used by Llama 3, Qwen2 and GPT-2/NeoX-style
                    models. decode is exact. encode applies the BPE merge rules exactly, but
                    the pre-tokenization step is an ASCII-correct approximation of the GPT-2
                    pre-tokenizer. Byte-exact parity for a specific model requires that
                    model's own pre-tokenizer regular expression, which is reported by
                    pretokenizer() but not yet branched on.

                Special tokens (those flagged as control or user-defined in the vocabulary,
                such as begin/end-of-sequence or chat-template markers) are recognized inside
                the input text and emitted as their single token id, rather than being split
                into characters. This behavior can be disabled per call.

                THREAD SAFETY
                    After loading (or deserialization), const member functions (encode,
                    decode and the accessors) are safe to call concurrently from multiple
                    threads. load(), load_from_gguf() and deserialize() are mutating and must
                    not run concurrently with any other use of the same object.

                REFERENCES
                    - Sennrich, R., Haddow, B., & Birch, A. (2016). Neural Machine Translation
                      of Rare Words with Subword Units. ACL 2016.
                    - Kudo, T., & Richardson, J. (2018). SentencePiece: A simple and language
                      independent subword tokenizer and detokenizer for Neural Text Processing.
        !*/

    public:

        enum class kind { spm, bpe };
        /*!
            The tokenizer family:
                - spm : SentencePiece (meta-space and score-driven merges).
                - bpe : byte-level Byte Pair Encoding (GPT-2 style).
        !*/

        enum token_type
        {
            TT_UNDEFINED = 0, TT_NORMAL = 1, TT_UNKNOWN = 2, TT_CONTROL = 3,
            TT_USER_DEFINED = 4, TT_UNUSED = 5, TT_BYTE = 6
        };
        /*!
            Per-token classification, using the same integer values as the GGUF
            tokenizer.ggml.token_type metadata. TT_BYTE marks the 256 byte-fallback tokens;
            TT_CONTROL and TT_USER_DEFINED mark special tokens.
        !*/

        struct vocab_data
        {
            kind family;
            std::vector<std::string> tokens;
            std::vector<float> scores;
            std::vector<int> types;
            std::vector<std::string> merges;
            int bos_id, eos_id, unk_id, pad_id;
            bool add_bos, add_eos, add_space_prefix;
            std::string pretokenizer;
            /*!
                WHAT THIS OBJECT REPRESENTS
                    A source-independent description of a pretrained vocabulary, used to load
                    an hf_tokenizer without depending on any particular file format.

                    - family            : spm or bpe.
                    - tokens            : the vocabulary; tokens[i] is the string of token i.
                    - scores            : SentencePiece merge scores (one per token); may be
                                          empty for bpe, in which case zeros are assumed.
                    - types             : per-token token_type values; may be empty, in which
                                          case every token is treated as TT_NORMAL.
                    - merges            : byte-level BPE merge rules as "left right" strings,
                                          in priority order; empty for spm.
                    - bos_id, eos_id, unk_id, pad_id : special token ids, or -1 if absent.
                    - add_bos, add_eos  : whether encode() adds those tokens by default.
                    - add_space_prefix  : whether SentencePiece adds the leading meta-space.
                    - pretokenizer      : optional name of the byte-level BPE pre-tokenizer.
            !*/
        };

        hf_tokenizer(
        );
        /*!
            ensures
                - constructs an empty tokenizer. #size() == 0. load() or load_from_gguf()
                  (or deserialization) must be called before encode()/decode() are useful.
        !*/

        void load(
            const vocab_data& data
        );
        /*!
            ensures
                - configures this tokenizer from data, copying the vocabulary and rebuilding
                  all internal lookup tables.
                - #type() == data.family
                - #size() == data.tokens.size()
                - #bos_id(), #eos_id(), #unk_id(), #pad_id() reflect the corresponding
                  fields of data.
                - #add_bos_default() == data.add_bos and #add_eos_default() == data.add_eos.
        !*/

        void load_from_gguf(
            const gguf_reader& g
        );
        /*!
            ensures
                - reads the tokenizer.ggml.* metadata from g into a vocab_data and calls
                  load() with it. The family is taken from tokenizer.ggml.model ("gpt2" or
                  "bpe" => kind::bpe, otherwise kind::spm).
            throws
                - std::runtime_error if g does not contain tokenizer.ggml.tokens.
        !*/

        size_t size(
        ) const;
        /*!
            ensures
                - returns the number of tokens in the vocabulary (including byte and special
                  tokens).
        !*/

        kind type(
        ) const;
        /*!
            ensures
                - returns the tokenizer family of the currently loaded vocabulary.
        !*/

        int bos_id() const;  /*!  ensures - returns the begin-of-sequence token id, or -1.  !*/
        int eos_id() const;  /*!  ensures - returns the end-of-sequence token id, or -1.     !*/
        int unk_id() const;  /*!  ensures - returns the unknown token id, or -1.             !*/
        int pad_id() const;  /*!  ensures - returns the padding token id, or -1.             !*/

        bool add_bos_default(
        ) const;
        /*!
            ensures
                - returns whether the single-argument encode() prepends bos_id() by default.
        !*/

        bool add_eos_default(
        ) const;
        /*!
            ensures
                - returns whether the single-argument encode() appends eos_id() by default.
        !*/

        const std::string& pretokenizer(
        ) const;
        /*!
            ensures
                - returns the name of the byte-level BPE pre-tokenizer reported by the source
                  vocabulary, or an empty string if none was provided. Informational only.
        !*/

        const std::string& id_to_token(
            int id
        ) const;
        /*!
            requires
                - 0 <= id < size()
            ensures
                - returns the raw vocabulary string of token id (in the family's own encoding,
                  for example with the SentencePiece meta-space or the byte-level remapping).
        !*/

        std::vector<int> encode(
            const std::string& text
        ) const;
        /*!
            ensures
                - returns the token ids of text, using the model defaults for adding the
                  begin/end-of-sequence tokens (add_bos_default(), add_eos_default()) and
                  recognizing special tokens inside text.
                - equivalent to encode(text, add_bos_default(), add_eos_default(), true).
        !*/

        std::vector<int> encode(
            const std::string& text,
            bool add_bos,
            bool add_eos,
            bool parse_special = true
        ) const;
        /*!
            ensures
                - returns the token ids of text.
                - if add_bos and bos_id() >= 0, bos_id() is prepended.
                - if add_eos and eos_id() >= 0, eos_id() is appended.
                - if parse_special, substrings of text that exactly match a control or
                  user-defined token are emitted as that single token id; the spans between
                  them are tokenized normally. If parse_special is false, the whole text is
                  tokenized as ordinary content.
        !*/

        std::string decode(
            const std::vector<int>& ids,
            bool skip_special = true
        ) const;
        /*!
            ensures
                - returns the text reconstructed from ids.
                - byte and meta-space tokens are converted back to their raw bytes; for
                  SentencePiece a single leading space introduced by the space prefix is
                  removed.
                - if skip_special, control tokens (and bos/eos/pad) are omitted from the
                  output; otherwise they are rendered using their vocabulary strings.
                - ids outside [0, size()) are ignored.
        !*/
    };

    void serialize(
        const hf_tokenizer& tok,
        std::ostream& out
    );
    /*!
        ensures
            - saves the entire state of tok to out.
    !*/

    void deserialize(
        hf_tokenizer& tok,
        std::istream& in
    );
    /*!
        ensures
            - restores the state of an hf_tokenizer from a serialized state.
        throws
            - serialization_error if the stream does not contain a compatible
              hf_tokenizer state.
    !*/
}

#endif // DLIB_HF_TOKENIZER_ABSTRACT_
