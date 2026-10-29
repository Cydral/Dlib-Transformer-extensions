// Copyright (C) 2005  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.


#include <string>
#include <sstream>
#include <regex>
#include <map>
#include <set>
#include <vector>
#include <cctype>

#include <dlib/tokenizer.h>
#include <dlib/tokenizer/hf_tokenizer.h>
#include <dlib/rand.h>
#include "tester.h"

namespace  
{
    using namespace test;
    using namespace std;
    using namespace dlib;
  
    logger dlog("test.tokenizer");

    template <
        typename tok
        >
    void tokenizer_kernel_test (
    )
    /*!
        requires
            - tok is an implementation of tokenizer_kernel_abstract.h
        ensures
            - runs tests on tok for compliance with the specs 
    !*/
    {        

        print_spinner();

        tok test;

        DLIB_TEST(test.numbers() == "0123456789");
        DLIB_TEST(test.uppercase_letters() == "ABCDEFGHIJKLMNOPQRSTUVWXYZ");
        DLIB_TEST(test.lowercase_letters() == "abcdefghijklmnopqrstuvwxyz");

        DLIB_TEST_MSG(test.get_identifier_body() == "_" + test.lowercase_letters() +
                     test.uppercase_letters() + test.numbers(),"");
        DLIB_TEST_MSG(test.get_identifier_head() == "_" + test.lowercase_letters() +
                     test.uppercase_letters(),"");

        DLIB_TEST(test.stream_is_set() == false);
        test.clear();
        DLIB_TEST(test.stream_is_set() == false);

        DLIB_TEST_MSG(test.get_identifier_body() == "_" + test.lowercase_letters() +
                     test.uppercase_letters() + test.numbers(),"");
        DLIB_TEST_MSG(test.get_identifier_head() == "_" + test.lowercase_letters() +
                     test.uppercase_letters(),"");

        tok test2;

        ostringstream sout;
        istringstream sin;
        test2.set_stream(sin);

        DLIB_TEST(test2.stream_is_set());
        DLIB_TEST(&test2.get_stream() == &sin);

        int type;
        string token;

        test2.get_token(type,token);
        DLIB_TEST(type == tok::END_OF_FILE);
        test2.get_token(type,token);
        DLIB_TEST(type == tok::END_OF_FILE);
        test2.get_token(type,token);
        DLIB_TEST(type == tok::END_OF_FILE);            


        sin.clear();
        sin.str("  The cat 123asdf1234 ._ \n test.");

        test2.get_token(type,token);
        DLIB_TEST(type == tok::WHITE_SPACE);
        DLIB_TEST(token == "  ");

        DLIB_TEST(test2.peek_type() == tok::IDENTIFIER);
        DLIB_TEST(test2.peek_token() == "The"); 
        test2.get_token(type,token);
        DLIB_TEST(type == tok::IDENTIFIER);
        DLIB_TEST(token == "The");            

        test2.get_token(type,token);
        DLIB_TEST(type == tok::WHITE_SPACE);
        DLIB_TEST(token == " ");

        test2.get_token(type,token);
        DLIB_TEST(type == tok::IDENTIFIER);
        DLIB_TEST(token == "cat");            

        test2.get_token(type,token);
        DLIB_TEST(type == tok::WHITE_SPACE);
        DLIB_TEST(token == " ");

        test2.get_token(type,token);
        DLIB_TEST(type == tok::NUMBER);
        DLIB_TEST_MSG(token == "123","token: " << token);

        DLIB_TEST(test2.peek_type() == tok::IDENTIFIER);
        DLIB_TEST(test2.peek_token() == "asdf1234");
        DLIB_TEST(test2.peek_type() == tok::IDENTIFIER);
        DLIB_TEST(test2.peek_token() == "asdf1234");
        DLIB_TEST(test2.peek_type() == tok::IDENTIFIER);
        DLIB_TEST(test2.peek_token() == "asdf1234");
        test2.get_token(type,token);
        DLIB_TEST(type == tok::IDENTIFIER);
        DLIB_TEST(token == "asdf1234");

        test2.get_token(type,token);
        DLIB_TEST(type == tok::WHITE_SPACE);
        DLIB_TEST_MSG(token == " ","token: " << token);

        test2.get_token(type,token);
        DLIB_TEST(type == tok::CHAR);
        DLIB_TEST_MSG(token == ".","token: " << token);

        test2.get_token(type,token);
        DLIB_TEST(type == tok::IDENTIFIER);
        DLIB_TEST(token == "_");

        DLIB_TEST(test2.peek_type() == tok::WHITE_SPACE);
        DLIB_TEST_MSG(test2.peek_token() == " ","token: \"" << token << "\"" <<
                     "\ntoken size: " << (unsigned int)token.size());

        swap(test,test2);

        DLIB_TEST(test2.stream_is_set() == false);

        DLIB_TEST(test.peek_type() == tok::WHITE_SPACE);
        DLIB_TEST_MSG(test.peek_token() == " ","token: \"" << token << "\"" <<
                     "\ntoken size: " << (unsigned int)token.size());
        test.get_token(type,token);
        DLIB_TEST(type == tok::WHITE_SPACE);
        DLIB_TEST_MSG(token == " ","token: \"" << token << "\"" <<
                     "\ntoken size: " << (unsigned int)token.size());

        test.get_token(type,token);
        DLIB_TEST_MSG(type == tok::END_OF_LINE,"token: " << token);
        DLIB_TEST_MSG(token == "\n","token: " << token);

        swap(test,test2);
        DLIB_TEST(test.stream_is_set() == false);

        test2.get_token(type,token);
        DLIB_TEST(type == tok::WHITE_SPACE);
        DLIB_TEST_MSG(token == " ","token: " << token);

        test2.get_token(type,token);
        DLIB_TEST(type == tok::IDENTIFIER);
        DLIB_TEST_MSG(token == "test","token: " << token);

        test2.get_token(type,token);
        DLIB_TEST(type == tok::CHAR);
        DLIB_TEST_MSG(token == ".","token: " << token);

        test2.get_token(type,token);
        DLIB_TEST(type == tok::END_OF_FILE);










        test2.set_identifier_token("_" + test.uppercase_letters() +
                                   test.lowercase_letters(),test.numbers() + "_" + test.uppercase_letters()
                                   +test.lowercase_letters());


        sin.clear();
        sin.str("  The cat 123asdf1234 ._ \n\r test.");

        test2.get_token(type,token);
        DLIB_TEST(type == tok::WHITE_SPACE);
        DLIB_TEST(token == "  ");

        test2.get_token(type,token);
        DLIB_TEST(type == tok::IDENTIFIER);
        DLIB_TEST(token == "The");            

        test2.get_token(type,token);
        DLIB_TEST(type == tok::WHITE_SPACE);
        DLIB_TEST(token == " ");

        test2.get_token(type,token);
        DLIB_TEST(type == tok::IDENTIFIER);
        DLIB_TEST(token == "cat");            

        test2.get_token(type,token);
        DLIB_TEST(type == tok::WHITE_SPACE);
        DLIB_TEST(token == " ");

        test2.get_token(type,token);
        DLIB_TEST(type == tok::NUMBER);
        DLIB_TEST_MSG(token == "123","token: " << token);

        test2.get_token(type,token);
        DLIB_TEST(type == tok::IDENTIFIER);
        DLIB_TEST(token == "asdf1234");

        test2.get_token(type,token);
        DLIB_TEST(type == tok::WHITE_SPACE);
        DLIB_TEST_MSG(token == " ","token: " << token);

        test2.get_token(type,token);
        DLIB_TEST(type == tok::CHAR);
        DLIB_TEST_MSG(token == ".","token: " << token);

        test2.get_token(type,token);
        DLIB_TEST(type == tok::IDENTIFIER);
        DLIB_TEST(token == "_");

        swap(test,test2);

        DLIB_TEST(test2.stream_is_set() == false);

        test.get_token(type,token);
        DLIB_TEST(type == tok::WHITE_SPACE);
        DLIB_TEST_MSG(token == " ","token: \"" << token << "\"" <<
                     "\ntoken size: " << (unsigned int)token.size());

        test.get_token(type,token);
        DLIB_TEST_MSG(type == tok::END_OF_LINE,"token: " << token);
        DLIB_TEST_MSG(token == "\n","token: " << token);

        swap(test,test2);
        DLIB_TEST(test.stream_is_set() == false);

        test2.get_token(type,token);
        DLIB_TEST(type == tok::WHITE_SPACE);
        DLIB_TEST_MSG(token == "\r ","token: " << token);

        test2.get_token(type,token);
        DLIB_TEST(type == tok::IDENTIFIER);
        DLIB_TEST_MSG(token == "test","token: " << token);

        test2.get_token(type,token);
        DLIB_TEST(type == tok::CHAR);
        DLIB_TEST_MSG(token == ".","token: " << token);

        test2.get_token(type,token);
        DLIB_TEST(type == tok::END_OF_FILE);













        test2.set_identifier_token(test.uppercase_letters() +
                                   test.lowercase_letters(),test.numbers() + test.uppercase_letters()
                                   +test.lowercase_letters());


        sin.clear();
        sin.str("  The cat 123as_df1234 ._ \n test.");

        test2.get_token(type,token);
        DLIB_TEST(type == tok::WHITE_SPACE);
        DLIB_TEST(token == "  ");

        test2.get_token(type,token);
        DLIB_TEST(type == tok::IDENTIFIER);
        DLIB_TEST(token == "The");            

        test2.get_token(type,token);
        DLIB_TEST(type == tok::WHITE_SPACE);
        DLIB_TEST(token == " ");

        test2.get_token(type,token);
        DLIB_TEST(type == tok::IDENTIFIER);
        DLIB_TEST(token == "cat");            

        test2.get_token(type,token);
        DLIB_TEST(type == tok::WHITE_SPACE);
        DLIB_TEST(token == " ");

        test2.get_token(type,token);
        DLIB_TEST(type == tok::NUMBER);
        DLIB_TEST_MSG(token == "123","token: " << token);

        test2.get_token(type,token);
        DLIB_TEST(type == tok::IDENTIFIER);
        DLIB_TEST(token == "as");

        test2.get_token(type,token);
        DLIB_TEST(type == tok::CHAR);
        DLIB_TEST_MSG(token == "_","token: " << token);

        test2.get_token(type,token);
        DLIB_TEST(type == tok::IDENTIFIER);
        DLIB_TEST(token == "df1234");

        test2.get_token(type,token);
        DLIB_TEST(type == tok::WHITE_SPACE);
        DLIB_TEST_MSG(token == " ","token: " << token);

        test2.get_token(type,token);
        DLIB_TEST(type == tok::CHAR);
        DLIB_TEST_MSG(token == ".","token: " << token);

        test2.get_token(type,token);
        DLIB_TEST(type == tok::CHAR);
        DLIB_TEST(token == "_");

        swap(test,test2);

        DLIB_TEST(test2.stream_is_set() == false);

        test.get_token(type,token);
        DLIB_TEST(type == tok::WHITE_SPACE);
        DLIB_TEST_MSG(token == " ","token: \"" << token << "\"" <<
                     "\ntoken size: " << (unsigned int)token.size());

        test.get_token(type,token);
        DLIB_TEST_MSG(type == tok::END_OF_LINE,"token: " << token);
        DLIB_TEST_MSG(token == "\n","token: " << token);

        swap(test,test2);
        DLIB_TEST(test.stream_is_set() == false);

        test2.get_token(type,token);
        DLIB_TEST(type == tok::WHITE_SPACE);
        DLIB_TEST_MSG(token == " ","token: " << token);

        test2.get_token(type,token);
        DLIB_TEST(type == tok::IDENTIFIER);
        DLIB_TEST_MSG(token == "test","token: " << token);

        test2.get_token(type,token);
        DLIB_TEST(type == tok::CHAR);
        DLIB_TEST_MSG(token == ".","token: " << token);

        test2.get_token(type,token);
        DLIB_TEST(type == tok::END_OF_FILE);


    }

    string postprocess_decoded_text(const string& decoded) {
        string result = decoded;
        result = regex_replace(result, std::regex("<text>"), "");
        result = regex_replace(result, std::regex("</text>"), "\n");
        if (!result.empty() && result.back() == '\n') result.pop_back();
        return result;
    }

    template <
        typename bpe_tok
    >
    void bpe_tokenizer_test(
    )
        /*!
            requires
                - bpe_tok is an implementation of bpe_tokenizer.h
            ensures
                - runs tests on bpe_tok for compliance with the specs
        !*/
    {
        print_spinner();

        bpe_tok test;

        std::string training_text = R"(
        Byte Pair Encoding (BPE) is a subword tokenization algorithm widely used in Natural Language Processing (NLP).
        It iteratively merges the most frequent pairs of bytes or characters to form a vocabulary of subword units.
        This approach is particularly useful for handling out-of-vocabulary words and reducing the size of the vocabulary
        while maintaining the ability to represent any text. BPE was introduced in the paper "Neural Machine Translation
        of Rare Words with Subword Units" by Sennrich et al. in 2016. The algorithm is simple yet effective and has been
        adopted in many state-of-the-art NLP models, including GPT and BERT.
        )";

        test.train(training_text, 300);

        std::ostringstream out_stream;
        serialize(test, out_stream);

        bpe_tok loaded_test;
        std::istringstream in_stream(out_stream.str());
        deserialize(loaded_test, in_stream);

        std::vector<std::string> test_strings = {
            u8"This is a test of the tokenisation process...\nimplemented in the Dlib library!", // English
            u8"Ceci est un test du processus de\ntokenisation implémenté dans\nla bibliothèque Dlib!", // French
            u8"Dette er en test af tokeniseringsprocessen implementeret i Dlib-biblioteket!", // Danish
            u8"这是对Dlib库中实现的标记化过程的测试！" // Chinese
        };

        for (const auto& text : test_strings) {
            std::vector<int> encoded = loaded_test.encode(text);
            std::string decoded = postprocess_decoded_text(loaded_test.decode(encoded));

            DLIB_TEST_MSG(text == decoded, "decoded: " << decoded);
        }
    }

    // ----------------------------------------------------------------------------------------

    /*
        The plain BPE algorithm, written out directly: count every adjacent pair inside every
        segment, merge the most frequent one (ties to the smallest pair) everywhere, repeat. The
        incremental trainer of bpe_tokenizer must reproduce its merge table exactly.
    */
    std::vector<std::pair<int, int>> reference_bpe_merges(
        const std::vector<std::vector<int>>& segments,
        size_t num_merges
    )
    {
        std::vector<std::vector<int>> work = segments;
        std::vector<std::pair<int, int>> merges;
        for (size_t m = 0; m < num_merges; ++m)
        {
            std::map<std::pair<int, int>, long> counts;
            for (const auto& s : work)
                for (size_t i = 0; i + 1 < s.size(); ++i)
                    counts[{ s[i], s[i + 1] }]++;
            std::pair<int, int> best{ 0, 0 };
            long best_count = 0;
            for (const auto& c : counts)
                if (c.second > best_count) { best_count = c.second; best = c.first; }
            if (best_count == 0) break;
            const int id = 256 + static_cast<int>(m);
            merges.push_back(best);
            for (auto& s : work)
            {
                std::vector<int> next;
                for (size_t i = 0; i < s.size(); )
                {
                    if (i + 1 < s.size() && s[i] == best.first && s[i + 1] == best.second) { next.push_back(id); i += 2; }
                    else next.push_back(s[i++]);
                }
                s.swap(next);
            }
        }
        return merges;
    }

    // The segments the training algorithm sees under segmentation mode 0, 1 or 2.
    std::vector<std::vector<int>> reference_segments(
        const std::string& text,
        int mode
    )
    {
        auto delim = [](unsigned char c) { return c == ' ' || c == '\n' || c == '\t' || c == '\r'; };
        auto cls = [&](unsigned char c) {
            if (delim(c)) return 0;
            if (c >= 0x80 || std::isalpha(c)) return 1;
            if (std::isdigit(c)) return 2;
            return 3;
        };
        std::vector<std::vector<int>> segments;
        size_t i = 0;
        while (i < text.size())
        {
            const unsigned char c = text[i];
            size_t j = i + 1;
            if (delim(c) && !(mode > 0 && c == ' ' && j < text.size() && !delim(text[j])))
            {
                segments.push_back({ c });
                i = j;
                continue;
            }
            if (mode == 2)
            {
                // The class of the run is that of its first byte after an opening space.
                const int run = cls(static_cast<unsigned char>(text[delim(c) ? j : i]));
                if (delim(c)) ++j;
                while (j < text.size() && cls(static_cast<unsigned char>(text[j])) == run) ++j;
            }
            else
            {
                while (j < text.size() && !delim(text[j])) ++j;
            }
            std::vector<int> s;
            for (size_t k = i; k < j; ++k) s.push_back(static_cast<unsigned char>(text[k]));
            segments.push_back(s);
            i = j;
        }
        return segments;
    }

    // Applies the merges in order over the whole text.
    std::vector<int> reference_encode(
        const std::string& text,
        const std::vector<std::pair<int, int>>& merges
    )
    {
        std::vector<int> t;
        for (unsigned char c : text) t.push_back(c);
        for (size_t m = 0; m < merges.size(); ++m)
        {
            std::vector<int> next;
            for (size_t i = 0; i < t.size(); )
            {
                if (i + 1 < t.size() && t[i] == merges[m].first && t[i + 1] == merges[m].second)
                {
                    next.push_back(256 + static_cast<int>(m));
                    i += 2;
                }
                else next.push_back(t[i++]);
            }
            t.swap(next);
        }
        return t;
    }

    void bpe_tokenizer_merge_test(
    )
    {
        print_spinner();

        // A text with repeated syllables, so that pairs tie and merges chain, with numbers,
        // markup and UTF-8 letters touching the words.
        dlib::rand rnd(7);
        const char* syllables[] = { "ka", "ri", "to", "ne", "su", "ma", "lo", "pe", "a", "e" };
        const char* spaces[] = { " ", " ", " ", "  ", "\n", "\t", " \n", "\r\n" };
        const char* marks[] = { ",", ".", "[[", "]]", "&quot;", "19", "84", "\xc3\xa9", "''", "|" };
        std::string text;
        for (int w = 0; w < 3000; ++w)
        {
            if (rnd.get_random_32bit_number() % 5 == 0) text += marks[rnd.get_random_32bit_number() % 10];
            const int len = 1 + rnd.get_random_32bit_number() % 4;
            for (int k = 0; k < len; ++k) text += syllables[rnd.get_random_32bit_number() % 10];
            if (rnd.get_random_32bit_number() % 4 == 0) text += marks[rnd.get_random_32bit_number() % 10];
            text += spaces[rnd.get_random_32bit_number() % 8];
        }
        const size_t vocab = 256 + 120 + 28;
        const bpe_segmentation modes[] = {
            bpe_segmentation::whitespace_split, bpe_segmentation::space_prefixed, bpe_segmentation::character_classes
        };
        const std::string probe = text.substr(1000, 4000) + "  ka ri\n\n to [[Kari]] 1984,ka";
        std::vector<std::pair<int, int>> whitespace_merges;

        for (int mode = 0; mode < 3; ++mode)
        {
            bpe_tokenizer tok;
            tok.set_segmentation(modes[mode]);
            tok.train(text, vocab);
            const std::vector<std::pair<int, int>> merges = tok.get_merges();
            if (mode == 0) whitespace_merges = merges;

            const auto expected = reference_bpe_merges(reference_segments(text, mode), 120);
            DLIB_TEST_MSG(merges == expected, "segmentation " << mode << ": merge table differs from the plain algorithm");

            // Encoding segment by segment gives what the whole-text application gives.
            const std::vector<int> ids = tok.encode(probe);
            DLIB_TEST(ids == reference_encode(probe, merges));
            DLIB_TEST(tok.decode(ids, false) == probe);

            // Under character_classes, no token mixes classes past a leading space.
            if (mode == 2)
            {
                for (size_t id = 256; id < 256 + merges.size(); ++id)
                {
                    std::string piece = tok.decode(static_cast<int>(id), false);
                    if (piece.size() > 1 && piece[0] == ' ') piece.erase(0, 1);
                    std::set<int> classes;
                    for (unsigned char c : piece)
                        classes.insert(c >= 0x80 || std::isalpha(c) ? 1 : std::isdigit(c) ? 2 : std::isspace(c) ? 0 : 3);
                    DLIB_TEST_MSG(classes.size() == 1 && classes.count(0) == 0, "token " << id << " mixes classes");
                }
            }

            // The merge table rebuilds the same tokenizer.
            bpe_tokenizer rebuilt;
            rebuilt.set_segmentation(tok.get_segmentation());
            rebuilt.set_merges(merges);
            DLIB_TEST(rebuilt.encode(probe) == ids);
            DLIB_TEST(rebuilt.get_vocab_size() == tok.get_vocab_size());

            // Serialization keeps the segmentation, and the historical one keeps its tag.
            std::ostringstream out;
            serialize(tok, out);
            std::istringstream in(out.str());
            std::string tag;
            deserialize(tag, in);
            DLIB_TEST(tag == (mode == 0 ? "bpe_tokenizer_" : "bpe_tokenizer_segmented_"));
            std::istringstream in2(out.str());
            bpe_tokenizer loaded;
            deserialize(loaded, in2);
            DLIB_TEST(loaded.get_segmentation() == tok.get_segmentation());
            DLIB_TEST(loaded.encode(probe) == ids);
        }

        // Merges learned under another segmentation still encode as the whole-text application.
        bpe_tokenizer foreign;
        foreign.set_segmentation(bpe_segmentation::character_classes);
        foreign.set_merges(whitespace_merges);
        DLIB_TEST(foreign.encode(probe) == reference_encode(probe, whitespace_merges));

        // The space-prefixed segmentation yields fewer tokens on prose.
        bpe_tokenizer plain, prefixed;
        prefixed.set_segmentation(bpe_segmentation::space_prefixed);
        plain.train(text, vocab);
        prefixed.train(text, vocab);
        DLIB_TEST(prefixed.encode(text).size() < plain.encode(text).size());

        // A merge that refers to a later token is refused.
        bpe_tokenizer bad;
        bool refused = false;
        try { bad.set_merges({ { 300, 1 } }); }
        catch (const std::invalid_argument&) { refused = true; }
        DLIB_TEST(refused);
    }

    // ----------------------------------------------------------------------------------------

    std::string utf8(uint32_t cp)
    {
        return hf_tok_impl::utf8_encode(cp);
    }

    /*  A byte-level vocabulary: the 256 byte symbols in the remapping of GPT-2, then the
        given tokens, merges and special tokens.  */
    hf_tokenizer::vocab_data byte_level_vocab(
        const std::vector<std::string>& extra,
        const std::vector<std::string>& merges,
        const std::vector<std::string>& specials,
        const std::string& pre
    )
    {
        hf_tokenizer::vocab_data v;
        v.family = hf_tokenizer::kind::bpe;
        int n = 0;
        for (int b = 0; b < 256; ++b)
        {
            const bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
            v.tokens.push_back(utf8(printable ? static_cast<uint32_t>(b) : static_cast<uint32_t>(256 + n++)));
            v.types.push_back(hf_tokenizer::TT_NORMAL);
        }
        for (const auto& t : extra) { v.tokens.push_back(t); v.types.push_back(hf_tokenizer::TT_NORMAL); }
        for (const auto& t : specials) { v.tokens.push_back(t); v.types.push_back(hf_tokenizer::TT_CONTROL); }
        v.merges = merges;
        v.pretokenizer = pre;
        return v;
    }

    void hf_tokenizer_test(
    )
    /*!
        ensures
            - The pre-tokenizers of the byte-level vocabularies split texts exactly as the
              regular expressions of their reference tokenizers.  The expected pieces were
              produced by those expressions, evaluated by a regular expression engine with
              Unicode properties.
            - Merges follow their ranks, leftmost first; special tokens are recognized only
              when asked; the canonical composition of the Qwen vocabularies is applied.
    !*/
    {
        struct pretok_case
        {
            hf_tok_impl::pre_family family;
            const char* text;
            std::vector<std::string> pieces;
        };
        const std::vector<pretok_case> cases = {
            { hf_tok_impl::pre_family::gpt2, "Hello world! It's DON'T 'LL you'd",
              { "Hello", " world", "!", " It", "'s", " DON", "'", "T", " '", "LL", " you", "'d" } },
            { hf_tok_impl::pre_family::gpt2, "  two spaces, three   then\n\n\nnewlines \r\n tab\there  ",
              { " ", " two", " spaces", ",", " three", "  ", " then", "\n\n", "\n", "newlines", " \r\n", " tab", "\t", "here", "  " } },
            { hf_tok_impl::pre_family::gpt2, "camelCaseName HTTPServer iPhone XMLHttp \xc3\x89""COLE \xc3\x89""cole \xc7\x85""emal",
              { "camelCaseName", " HTTPServer", " iPhone", " XMLHttp", " \xc3\x89""COLE", " \xc3\x89""cole", " \xc7\x85""emal" } },
            { hf_tok_impl::pre_family::gpt2, "Numbers 1234567 and 3.14, x2 \xe2\x85\xab \xc2\xbd \xc2\xb2",
              { "Numbers", " 1234567", " and", " 3", ".", "14", ",", " x", "2", " \xe2\x85\xab", " \xc2\xbd", " \xc2\xb2" } },
            { hf_tok_impl::pre_family::gpt2, "Ponctuation...!!! (x) [y] {z} a/b/c\n// comment\n",
              { "Ponctuation", "...!!!", " (", "x", ")", " [", "y", "]", " {", "z", "}", " a", "/", "b", "/", "c", "\n", "//", " comment", "\n" } },
            { hf_tok_impl::pre_family::gpt2, "\xe4\xb8\xad\xe6\x96\x87\xe6\xa8\xa1\xe5\x9e\x8b\xe6\x8e\xa8\xe7\x90\x86\xef\xbc\x8c\xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf\xe3\x80\x82\xed\x95\x9c\xea\xb5\xad\xec\x96\xb4 \xd9\x85\xd8\xb1\xd8\xad\xd8\xa8\xd8\xa7 \xe0\xa4\xa8\xe0\xa4\xae\xe0\xa4\xb8\xe0\xa5\x8d\xe0\xa4\xa4\xe0\xa5\x87",
              { "\xe4\xb8\xad\xe6\x96\x87\xe6\xa8\xa1\xe5\x9e\x8b\xe6\x8e\xa8\xe7\x90\x86", "\xef\xbc\x8c", "\xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf", "\xe3\x80\x82", "\xed\x95\x9c\xea\xb5\xad\xec\x96\xb4", " \xd9\x85\xd8\xb1\xd8\xad\xd8\xa8\xd8\xa7", " \xe0\xa4\xa8\xe0\xa4\xae\xe0\xa4\xb8", "\xe0\xa5\x8d", "\xe0\xa4\xa4", "\xe0\xa5\x87" } },
            { hf_tok_impl::pre_family::gpt2, "e\xcc\x81 a\xcc\x80\xcc\x88 \xcc\x81x Z\xcc\xb5lgo",
              { "e", "\xcc\x81", " a", "\xcc\x80\xcc\x88", " \xcc\x81", "x", " Z", "\xcc\xb5", "lgo" } },
            { hf_tok_impl::pre_family::gpt2, "emoji \xf0\x9f\x98\x80\xf0\x9f\x9a\x80 ok \xf0\x9f\x91\x8d\xf0\x9f\x8f\xbd",
              { "emoji", " \xf0\x9f\x98\x80\xf0\x9f\x9a\x80", " ok", " \xf0\x9f\x91\x8d\xf0\x9f\x8f\xbd" } },
            { hf_tok_impl::pre_family::gpt2, "\xc2\xa0nbsp\xe3\x80\x80ideo\xe2\x80\x89thin",
              { "\xc2\xa0", "nbsp", "\xe3\x80\x80", "ideo", "\xe2\x80\x89", "thin" } },
            { hf_tok_impl::pre_family::gpt2, "x  \n  y\n\n  \n z",
              { "x", "  \n ", " y", "\n\n  \n", " z" } },
            { hf_tok_impl::pre_family::gpt2_digits, "Hello world! It's DON'T 'LL you'd",
              { "Hello", " world", "!", " It", "'s", " DON", "'", "T", " '", "LL", " you", "'d" } },
            { hf_tok_impl::pre_family::gpt2_digits, "  two spaces, three   then\n\n\nnewlines \r\n tab\there  ",
              { " ", " two", " spaces", ",", " three", "  ", " then", "\n\n", "\n", "newlines", " \r\n", " tab", "\t", "here", "  " } },
            { hf_tok_impl::pre_family::gpt2_digits, "camelCaseName HTTPServer iPhone XMLHttp \xc3\x89""COLE \xc3\x89""cole \xc7\x85""emal",
              { "camelCaseName", " HTTPServer", " iPhone", " XMLHttp", " \xc3\x89""COLE", " \xc3\x89""cole", " \xc7\x85""emal" } },
            { hf_tok_impl::pre_family::gpt2_digits, "Numbers 1234567 and 3.14, x2 \xe2\x85\xab \xc2\xbd \xc2\xb2",
              { "Numbers", " ", "1", "2", "3", "4", "5", "6", "7", " and", " ", "3", ".", "1", "4", ",", " x", "2", " ", "\xe2\x85\xab", " ", "\xc2\xbd", " ", "\xc2\xb2" } },
            { hf_tok_impl::pre_family::gpt2_digits, "Ponctuation...!!! (x) [y] {z} a/b/c\n// comment\n",
              { "Ponctuation", "...!!!", " (", "x", ")", " [", "y", "]", " {", "z", "}", " a", "/", "b", "/", "c", "\n", "//", " comment", "\n" } },
            { hf_tok_impl::pre_family::gpt2_digits, "\xe4\xb8\xad\xe6\x96\x87\xe6\xa8\xa1\xe5\x9e\x8b\xe6\x8e\xa8\xe7\x90\x86\xef\xbc\x8c\xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf\xe3\x80\x82\xed\x95\x9c\xea\xb5\xad\xec\x96\xb4 \xd9\x85\xd8\xb1\xd8\xad\xd8\xa8\xd8\xa7 \xe0\xa4\xa8\xe0\xa4\xae\xe0\xa4\xb8\xe0\xa5\x8d\xe0\xa4\xa4\xe0\xa5\x87",
              { "\xe4\xb8\xad\xe6\x96\x87\xe6\xa8\xa1\xe5\x9e\x8b\xe6\x8e\xa8\xe7\x90\x86", "\xef\xbc\x8c", "\xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf", "\xe3\x80\x82", "\xed\x95\x9c\xea\xb5\xad\xec\x96\xb4", " \xd9\x85\xd8\xb1\xd8\xad\xd8\xa8\xd8\xa7", " \xe0\xa4\xa8\xe0\xa4\xae\xe0\xa4\xb8", "\xe0\xa5\x8d", "\xe0\xa4\xa4", "\xe0\xa5\x87" } },
            { hf_tok_impl::pre_family::gpt2_digits, "e\xcc\x81 a\xcc\x80\xcc\x88 \xcc\x81x Z\xcc\xb5lgo",
              { "e", "\xcc\x81", " a", "\xcc\x80\xcc\x88", " \xcc\x81", "x", " Z", "\xcc\xb5", "lgo" } },
            { hf_tok_impl::pre_family::gpt2_digits, "emoji \xf0\x9f\x98\x80\xf0\x9f\x9a\x80 ok \xf0\x9f\x91\x8d\xf0\x9f\x8f\xbd",
              { "emoji", " \xf0\x9f\x98\x80\xf0\x9f\x9a\x80", " ok", " \xf0\x9f\x91\x8d\xf0\x9f\x8f\xbd" } },
            { hf_tok_impl::pre_family::gpt2_digits, "\xc2\xa0nbsp\xe3\x80\x80ideo\xe2\x80\x89thin",
              { "\xc2\xa0", "nbsp", "\xe3\x80\x80", "ideo", "\xe2\x80\x89", "thin" } },
            { hf_tok_impl::pre_family::gpt2_digits, "x  \n  y\n\n  \n z",
              { "x", "  \n ", " y", "\n\n  \n", " z" } },
            { hf_tok_impl::pre_family::llama3, "Hello world! It's DON'T 'LL you'd",
              { "Hello", " world", "!", " It", "'s", " DON", "'T", " '", "LL", " you", "'d" } },
            { hf_tok_impl::pre_family::llama3, "  two spaces, three   then\n\n\nnewlines \r\n tab\there  ",
              { " ", " two", " spaces", ",", " three", "  ", " then", "\n\n\n", "newlines", " \r\n", " tab", "\there", "  " } },
            { hf_tok_impl::pre_family::llama3, "camelCaseName HTTPServer iPhone XMLHttp \xc3\x89""COLE \xc3\x89""cole \xc7\x85""emal",
              { "camelCaseName", " HTTPServer", " iPhone", " XMLHttp", " \xc3\x89""COLE", " \xc3\x89""cole", " \xc7\x85""emal" } },
            { hf_tok_impl::pre_family::llama3, "Numbers 1234567 and 3.14, x2 \xe2\x85\xab \xc2\xbd \xc2\xb2",
              { "Numbers", " ", "123", "456", "7", " and", " ", "3", ".", "14", ",", " x", "2", " ", "\xe2\x85\xab", " ", "\xc2\xbd", " ", "\xc2\xb2" } },
            { hf_tok_impl::pre_family::llama3, "Ponctuation...!!! (x) [y] {z} a/b/c\n// comment\n",
              { "Ponctuation", "...!!!", " (", "x", ")", " [", "y", "]", " {", "z", "}", " a", "/b", "/c", "\n", "//", " comment", "\n" } },
            { hf_tok_impl::pre_family::llama3, "\xe4\xb8\xad\xe6\x96\x87\xe6\xa8\xa1\xe5\x9e\x8b\xe6\x8e\xa8\xe7\x90\x86\xef\xbc\x8c\xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf\xe3\x80\x82\xed\x95\x9c\xea\xb5\xad\xec\x96\xb4 \xd9\x85\xd8\xb1\xd8\xad\xd8\xa8\xd8\xa7 \xe0\xa4\xa8\xe0\xa4\xae\xe0\xa4\xb8\xe0\xa5\x8d\xe0\xa4\xa4\xe0\xa5\x87",
              { "\xe4\xb8\xad\xe6\x96\x87\xe6\xa8\xa1\xe5\x9e\x8b\xe6\x8e\xa8\xe7\x90\x86", "\xef\xbc\x8c\xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf", "\xe3\x80\x82\xed\x95\x9c\xea\xb5\xad\xec\x96\xb4", " \xd9\x85\xd8\xb1\xd8\xad\xd8\xa8\xd8\xa7", " \xe0\xa4\xa8\xe0\xa4\xae\xe0\xa4\xb8", "\xe0\xa5\x8d\xe0\xa4\xa4", "\xe0\xa5\x87" } },
            { hf_tok_impl::pre_family::llama3, "e\xcc\x81 a\xcc\x80\xcc\x88 \xcc\x81x Z\xcc\xb5lgo",
              { "e", "\xcc\x81", " a", "\xcc\x80\xcc\x88", " \xcc\x81", "x", " Z", "\xcc\xb5lgo" } },
            { hf_tok_impl::pre_family::llama3, "emoji \xf0\x9f\x98\x80\xf0\x9f\x9a\x80 ok \xf0\x9f\x91\x8d\xf0\x9f\x8f\xbd",
              { "emoji", " \xf0\x9f\x98\x80\xf0\x9f\x9a\x80", " ok", " \xf0\x9f\x91\x8d\xf0\x9f\x8f\xbd" } },
            { hf_tok_impl::pre_family::llama3, "\xc2\xa0nbsp\xe3\x80\x80ideo\xe2\x80\x89thin",
              { "\xc2\xa0nbsp", "\xe3\x80\x80ideo", "\xe2\x80\x89thin" } },
            { hf_tok_impl::pre_family::llama3, "x  \n  y\n\n  \n z",
              { "x", "  \n", " ", " y", "\n\n  \n", " z" } },
            { hf_tok_impl::pre_family::qwen2, "Hello world! It's DON'T 'LL you'd",
              { "Hello", " world", "!", " It", "'s", " DON", "'T", " '", "LL", " you", "'d" } },
            { hf_tok_impl::pre_family::qwen2, "  two spaces, three   then\n\n\nnewlines \r\n tab\there  ",
              { " ", " two", " spaces", ",", " three", "  ", " then", "\n\n\n", "newlines", " \r\n", " tab", "\there", "  " } },
            { hf_tok_impl::pre_family::qwen2, "camelCaseName HTTPServer iPhone XMLHttp \xc3\x89""COLE \xc3\x89""cole \xc7\x85""emal",
              { "camelCaseName", " HTTPServer", " iPhone", " XMLHttp", " \xc3\x89""COLE", " \xc3\x89""cole", " \xc7\x85""emal" } },
            { hf_tok_impl::pre_family::qwen2, "Numbers 1234567 and 3.14, x2 \xe2\x85\xab \xc2\xbd \xc2\xb2",
              { "Numbers", " ", "1", "2", "3", "4", "5", "6", "7", " and", " ", "3", ".", "1", "4", ",", " x", "2", " ", "\xe2\x85\xab", " ", "\xc2\xbd", " ", "\xc2\xb2" } },
            { hf_tok_impl::pre_family::qwen2, "Ponctuation...!!! (x) [y] {z} a/b/c\n// comment\n",
              { "Ponctuation", "...!!!", " (", "x", ")", " [", "y", "]", " {", "z", "}", " a", "/b", "/c", "\n", "//", " comment", "\n" } },
            { hf_tok_impl::pre_family::qwen2, "\xe4\xb8\xad\xe6\x96\x87\xe6\xa8\xa1\xe5\x9e\x8b\xe6\x8e\xa8\xe7\x90\x86\xef\xbc\x8c\xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf\xe3\x80\x82\xed\x95\x9c\xea\xb5\xad\xec\x96\xb4 \xd9\x85\xd8\xb1\xd8\xad\xd8\xa8\xd8\xa7 \xe0\xa4\xa8\xe0\xa4\xae\xe0\xa4\xb8\xe0\xa5\x8d\xe0\xa4\xa4\xe0\xa5\x87",
              { "\xe4\xb8\xad\xe6\x96\x87\xe6\xa8\xa1\xe5\x9e\x8b\xe6\x8e\xa8\xe7\x90\x86", "\xef\xbc\x8c\xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf", "\xe3\x80\x82\xed\x95\x9c\xea\xb5\xad\xec\x96\xb4", " \xd9\x85\xd8\xb1\xd8\xad\xd8\xa8\xd8\xa7", " \xe0\xa4\xa8\xe0\xa4\xae\xe0\xa4\xb8", "\xe0\xa5\x8d\xe0\xa4\xa4", "\xe0\xa5\x87" } },
            { hf_tok_impl::pre_family::qwen2, "e\xcc\x81 a\xcc\x80\xcc\x88 \xcc\x81x Z\xcc\xb5lgo",
              { "e", "\xcc\x81", " a", "\xcc\x80\xcc\x88", " \xcc\x81", "x", " Z", "\xcc\xb5lgo" } },
            { hf_tok_impl::pre_family::qwen2, "emoji \xf0\x9f\x98\x80\xf0\x9f\x9a\x80 ok \xf0\x9f\x91\x8d\xf0\x9f\x8f\xbd",
              { "emoji", " \xf0\x9f\x98\x80\xf0\x9f\x9a\x80", " ok", " \xf0\x9f\x91\x8d\xf0\x9f\x8f\xbd" } },
            { hf_tok_impl::pre_family::qwen2, "\xc2\xa0nbsp\xe3\x80\x80ideo\xe2\x80\x89thin",
              { "\xc2\xa0nbsp", "\xe3\x80\x80ideo", "\xe2\x80\x89thin" } },
            { hf_tok_impl::pre_family::qwen2, "x  \n  y\n\n  \n z",
              { "x", "  \n", " ", " y", "\n\n  \n", " z" } },
            { hf_tok_impl::pre_family::qwen35, "Hello world! It's DON'T 'LL you'd",
              { "Hello", " world", "!", " It", "'s", " DON", "'T", " '", "LL", " you", "'d" } },
            { hf_tok_impl::pre_family::qwen35, "  two spaces, three   then\n\n\nnewlines \r\n tab\there  ",
              { " ", " two", " spaces", ",", " three", "  ", " then", "\n\n\n", "newlines", " \r\n", " tab", "\there", "  " } },
            { hf_tok_impl::pre_family::qwen35, "camelCaseName HTTPServer iPhone XMLHttp \xc3\x89""COLE \xc3\x89""cole \xc7\x85""emal",
              { "camelCaseName", " HTTPServer", " iPhone", " XMLHttp", " \xc3\x89""COLE", " \xc3\x89""cole", " \xc7\x85""emal" } },
            { hf_tok_impl::pre_family::qwen35, "Numbers 1234567 and 3.14, x2 \xe2\x85\xab \xc2\xbd \xc2\xb2",
              { "Numbers", " ", "1", "2", "3", "4", "5", "6", "7", " and", " ", "3", ".", "1", "4", ",", " x", "2", " ", "\xe2\x85\xab", " ", "\xc2\xbd", " ", "\xc2\xb2" } },
            { hf_tok_impl::pre_family::qwen35, "Ponctuation...!!! (x) [y] {z} a/b/c\n// comment\n",
              { "Ponctuation", "...!!!", " (", "x", ")", " [", "y", "]", " {", "z", "}", " a", "/b", "/c", "\n", "//", " comment", "\n" } },
            { hf_tok_impl::pre_family::qwen35, "\xe4\xb8\xad\xe6\x96\x87\xe6\xa8\xa1\xe5\x9e\x8b\xe6\x8e\xa8\xe7\x90\x86\xef\xbc\x8c\xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf\xe3\x80\x82\xed\x95\x9c\xea\xb5\xad\xec\x96\xb4 \xd9\x85\xd8\xb1\xd8\xad\xd8\xa8\xd8\xa7 \xe0\xa4\xa8\xe0\xa4\xae\xe0\xa4\xb8\xe0\xa5\x8d\xe0\xa4\xa4\xe0\xa5\x87",
              { "\xe4\xb8\xad\xe6\x96\x87\xe6\xa8\xa1\xe5\x9e\x8b\xe6\x8e\xa8\xe7\x90\x86", "\xef\xbc\x8c\xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf", "\xe3\x80\x82\xed\x95\x9c\xea\xb5\xad\xec\x96\xb4", " \xd9\x85\xd8\xb1\xd8\xad\xd8\xa8\xd8\xa7", " \xe0\xa4\xa8\xe0\xa4\xae\xe0\xa4\xb8\xe0\xa5\x8d\xe0\xa4\xa4\xe0\xa5\x87" } },
            { hf_tok_impl::pre_family::qwen35, "e\xcc\x81 a\xcc\x80\xcc\x88 \xcc\x81x Z\xcc\xb5lgo",
              { "e\xcc\x81", " a\xcc\x80\xcc\x88", " \xcc\x81x", " Z\xcc\xb5lgo" } },
            { hf_tok_impl::pre_family::qwen35, "emoji \xf0\x9f\x98\x80\xf0\x9f\x9a\x80 ok \xf0\x9f\x91\x8d\xf0\x9f\x8f\xbd",
              { "emoji", " \xf0\x9f\x98\x80\xf0\x9f\x9a\x80", " ok", " \xf0\x9f\x91\x8d\xf0\x9f\x8f\xbd" } },
            { hf_tok_impl::pre_family::qwen35, "\xc2\xa0nbsp\xe3\x80\x80ideo\xe2\x80\x89thin",
              { "\xc2\xa0nbsp", "\xe3\x80\x80ideo", "\xe2\x80\x89thin" } },
            { hf_tok_impl::pre_family::qwen35, "x  \n  y\n\n  \n z",
              { "x", "  \n", " ", " y", "\n\n  \n", " z" } },
            { hf_tok_impl::pre_family::tekken, "Hello world! It's DON'T 'LL you'd",
              { "Hello", " world", "!", " It", "'s", " DON", "'T", " '", "LL", " you", "'d" } },
            { hf_tok_impl::pre_family::tekken, "  two spaces, three   then\n\n\nnewlines \r\n tab\there  ",
              { " ", " two", " spaces", ",", " three", "  ", " then", "\n\n\n", "newlines", " \r\n", " tab", "\there", "  " } },
            { hf_tok_impl::pre_family::tekken, "camelCaseName HTTPServer iPhone XMLHttp \xc3\x89""COLE \xc3\x89""cole \xc7\x85""emal",
              { "camel", "Case", "Name", " HTTPServer", " i", "Phone", " XMLHttp", " \xc3\x89""COLE", " \xc3\x89""cole", " \xc7\x85""emal" } },
            { hf_tok_impl::pre_family::tekken, "Numbers 1234567 and 3.14, x2 \xe2\x85\xab \xc2\xbd \xc2\xb2",
              { "Numbers", " ", "1", "2", "3", "4", "5", "6", "7", " and", " ", "3", ".", "1", "4", ",", " x", "2", " ", "\xe2\x85\xab", " ", "\xc2\xbd", " ", "\xc2\xb2" } },
            { hf_tok_impl::pre_family::tekken, "Ponctuation...!!! (x) [y] {z} a/b/c\n// comment\n",
              { "Ponctuation", "...!!!", " (", "x", ")", " [", "y", "]", " {", "z", "}", " a", "/b", "/c", "\n", "//", " comment", "\n" } },
            { hf_tok_impl::pre_family::tekken, "\xe4\xb8\xad\xe6\x96\x87\xe6\xa8\xa1\xe5\x9e\x8b\xe6\x8e\xa8\xe7\x90\x86\xef\xbc\x8c\xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf\xe3\x80\x82\xed\x95\x9c\xea\xb5\xad\xec\x96\xb4 \xd9\x85\xd8\xb1\xd8\xad\xd8\xa8\xd8\xa7 \xe0\xa4\xa8\xe0\xa4\xae\xe0\xa4\xb8\xe0\xa5\x8d\xe0\xa4\xa4\xe0\xa5\x87",
              { "\xe4\xb8\xad\xe6\x96\x87\xe6\xa8\xa1\xe5\x9e\x8b\xe6\x8e\xa8\xe7\x90\x86", "\xef\xbc\x8c\xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf", "\xe3\x80\x82\xed\x95\x9c\xea\xb5\xad\xec\x96\xb4", " \xd9\x85\xd8\xb1\xd8\xad\xd8\xa8\xd8\xa7", " \xe0\xa4\xa8\xe0\xa4\xae\xe0\xa4\xb8\xe0\xa5\x8d\xe0\xa4\xa4\xe0\xa5\x87" } },
            { hf_tok_impl::pre_family::tekken, "e\xcc\x81 a\xcc\x80\xcc\x88 \xcc\x81x Z\xcc\xb5lgo",
              { "e\xcc\x81", " a\xcc\x80\xcc\x88", " \xcc\x81x", " Z\xcc\xb5lgo" } },
            { hf_tok_impl::pre_family::tekken, "emoji \xf0\x9f\x98\x80\xf0\x9f\x9a\x80 ok \xf0\x9f\x91\x8d\xf0\x9f\x8f\xbd",
              { "emoji", " \xf0\x9f\x98\x80\xf0\x9f\x9a\x80", " ok", " \xf0\x9f\x91\x8d\xf0\x9f\x8f\xbd" } },
            { hf_tok_impl::pre_family::tekken, "\xc2\xa0nbsp\xe3\x80\x80ideo\xe2\x80\x89thin",
              { "\xc2\xa0nbsp", "\xe3\x80\x80ideo", "\xe2\x80\x89thin" } },
            { hf_tok_impl::pre_family::tekken, "x  \n  y\n\n  \n z",
              { "x", "  \n", " ", " y", "\n\n  \n", " z" } },
        };
        std::vector<std::pair<size_t, size_t>> spans;
        for (const auto& c : cases)
        {
            const std::string text = c.text;
            hf_tok_impl::pretokenize(text, c.family, spans);
            std::vector<std::string> got;
            for (const auto& s : spans) got.push_back(text.substr(s.first, s.second - s.first));
            DLIB_TEST_MSG(got == c.pieces, "pre-tokenizer " << static_cast<int>(c.family) << " on: " << text);
        }

        // Names of the vocabularies.
        bool exact = false;
        DLIB_TEST(hf_tok_impl::pre_family_of("qwen2", exact) == hf_tok_impl::pre_family::qwen2 && exact);
        DLIB_TEST(hf_tok_impl::pre_family_of("llama-bpe", exact) == hf_tok_impl::pre_family::llama3 && exact);
        DLIB_TEST(hf_tok_impl::pre_family_of("tekken", exact) == hf_tok_impl::pre_family::tekken && exact);
        DLIB_TEST(hf_tok_impl::pre_family_of("smollm", exact) == hf_tok_impl::pre_family::gpt2_digits && exact);
        DLIB_TEST(hf_tok_impl::pre_family_of("unknown-vocab", exact) == hf_tok_impl::pre_family::qwen2 && !exact);

        // Merges by rank, the leftmost pair first among equal ranks.
        {
            hf_tokenizer tok;
            tok.load(byte_level_vocab({ "aa", "ab", "abc", "\xc4\xa0x" }, { "a a", "a b", "ab c", "\xc4\xa0 x" },
                { "<|end|>" }, "qwen2"));
            const int a = tok.token_id("a"), aa = tok.token_id("aa"), abc = tok.token_id("abc");
            DLIB_TEST(a >= 0 && aa >= 0 && abc >= 0);
            DLIB_TEST(tok.encode("aaa", false, false) == std::vector<int>({ aa, a }));
            DLIB_TEST(tok.encode("aaaa", false, false) == std::vector<int>({ aa, aa }));
            DLIB_TEST(tok.encode("abc", false, false) == std::vector<int>({ abc }));
            DLIB_TEST(tok.encode("aabc", false, false) == std::vector<int>({ aa, tok.token_id("b"), tok.token_id("c") }));
            DLIB_TEST(tok.encode(" x", false, false) == std::vector<int>({ tok.token_id("\xc4\xa0x") }));
            DLIB_TEST(tok.exact_pretokenizer());

            // Special tokens: recognized in the text when asked, spelled out otherwise.
            const int end = tok.token_id("<|end|>");
            DLIB_TEST(end >= 0 && tok.token_type(end) == hf_tokenizer::TT_CONTROL);
            DLIB_TEST(tok.encode("ab<|end|>", false, false, true) == std::vector<int>({ tok.token_id("ab"), end }));
            const std::vector<int> spelled = tok.encode("<|end|>", false, false, false);
            DLIB_TEST(spelled.size() > 1);
            DLIB_TEST(tok.decode(spelled) == "<|end|>");
            DLIB_TEST(tok.decode({ tok.token_id("ab"), end }) == "ab");
            DLIB_TEST(tok.decode({ tok.token_id("ab"), end }, false) == "ab<|end|>");
            DLIB_TEST(tok.token_bytes(end) == "<|end|>");
            DLIB_TEST(tok.token_bytes(tok.token_id("\xc4\xa0x")) == " x");

            // A long run without spaces is merged in reasonable time and comes back whole.
            std::string run;
            for (int i = 0; i < 20000; ++i) run += (i % 3 == 0) ? "ab" : "a";
            const std::vector<int> ids = tok.encode(run, false, false);
            DLIB_TEST(tok.decode(ids) == run);

            // Every byte, valid UTF-8 or not, survives a round trip.
            std::string bytes;
            for (int b = 1; b < 256; ++b) bytes.push_back(static_cast<char>(b));
            DLIB_TEST(tok.decode(tok.encode(bytes, false, false, false)) == bytes);

            // Serialization keeps the behavior.
            std::ostringstream sout;
            serialize(tok, sout);
            std::istringstream sin(sout.str());
            hf_tokenizer copy;
            deserialize(copy, sin);
            DLIB_TEST(copy.encode("aaaab<|end|> x", false, false) == tok.encode("aaaab<|end|> x", false, false));
        }

        // ignore_merges: a piece that is itself a token is taken whole.
        {
            hf_tokenizer with, without;
            with.load(byte_level_vocab({ "xy", "xyz" }, { "x y" }, {}, "llama-bpe"));
            without.load(byte_level_vocab({ "xy", "xyz" }, { "x y" }, {}, "qwen2"));
            DLIB_TEST(with.encode("xyz", false, false) == std::vector<int>({ with.token_id("xyz") }));
            DLIB_TEST(without.encode("xyz", false, false) == std::vector<int>({ without.token_id("xy"), without.token_id("z") }));
        }

        // Canonical composition.
        DLIB_TEST(hf_tok_impl::nfc_compose("plain ascii") == "plain ascii");
        DLIB_TEST(hf_tok_impl::nfc_compose("e\xcc\x81t\xc3\xa9") == "\xc3\xa9t\xc3\xa9");            // e + acute
        DLIB_TEST(hf_tok_impl::nfc_compose("a\xcc\xa3\xcc\x82") == "\xe1\xba\xad");                     // a + dot below + circumflex
        DLIB_TEST(hf_tok_impl::nfc_compose("a\xcc\x82\xcc\xa3") == "\xe1\xba\xad");                     // the same marks out of order
        DLIB_TEST(hf_tok_impl::nfc_compose("\xe1\x84\x80\xe1\x85\xa1\xe1\x86\xa8") == "\xea\xb0\x81"); // Hangul jamo
        DLIB_TEST(hf_tok_impl::nfc_compose("\xe3\x81\x8b\xe3\x82\x99") == "\xe3\x81\x8c");            // ka + voiced mark
        DLIB_TEST(hf_tok_impl::nfc_compose("\xff\xcc\x81x\xcc") == "\xff\xcc\x81x\xcc");               // invalid bytes kept
        DLIB_TEST(hf_unicode::classify('A') == hf_unicode::Lu && hf_unicode::classify(0x4E2D) == hf_unicode::Lo);
        DLIB_TEST(hf_unicode::classify(0x0301) == hf_unicode::M && hf_unicode::classify(0x00BD) == hf_unicode::N);
        DLIB_TEST(hf_unicode::is_white_space(0x3000) && !hf_unicode::is_white_space(0x200B));
    }

    class tokenizer_tester : public tester
    {
    public:
        tokenizer_tester (
        ) :
            tester ("test_tokenizer",
                    "Runs tests on the tokenizer component.")
        {}

        void perform_test (
        )
        {
            dlog << LINFO << "testing kernel_1a";
            tokenizer_kernel_test<tokenizer::kernel_1a>  ();
            dlog << LINFO << "testing kernel_1a_c";
            tokenizer_kernel_test<tokenizer::kernel_1a_c>();
            dlog << LINFO << "testing bpe_tokenizer";
            bpe_tokenizer_test<bpe_tokenizer>();
            dlog << LINFO << "testing bpe_tokenizer merge table";
            bpe_tokenizer_merge_test();
            dlog << LINFO << "testing hf_tokenizer";
            hf_tokenizer_test();
        }
    } a;

}
