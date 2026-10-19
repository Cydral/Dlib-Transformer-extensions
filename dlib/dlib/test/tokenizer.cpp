// Copyright (C) 2005  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.


#include <string>
#include <sstream>
#include <regex>
#include <map>
#include <vector>

#include <dlib/tokenizer.h>
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

    // The segments the training algorithm sees, for either segmentation.
    std::vector<std::vector<int>> reference_segments(
        const std::string& text,
        bool space_prefixed
    )
    {
        auto delim = [](unsigned char c) { return c == ' ' || c == '\n' || c == '\t' || c == '\r'; };
        std::vector<std::vector<int>> segments;
        size_t i = 0;
        while (i < text.size())
        {
            const unsigned char c = text[i];
            size_t j = i + 1;
            if (delim(c) && !(space_prefixed && c == ' ' && j < text.size() && !delim(text[j])))
            {
                segments.push_back({ c });
                i = j;
                continue;
            }
            while (j < text.size() && !delim(text[j])) ++j;
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

        // A text with repeated syllables, so that pairs tie and merges chain.
        dlib::rand rnd(7);
        const char* syllables[] = { "ka", "ri", "to", "ne", "su", "ma", "lo", "pe", "a", "e" };
        const char* spaces[] = { " ", " ", " ", "  ", "\n", "\t", " \n", "\r\n" };
        std::string text;
        for (int w = 0; w < 3000; ++w)
        {
            const int len = 1 + rnd.get_random_32bit_number() % 4;
            for (int k = 0; k < len; ++k) text += syllables[rnd.get_random_32bit_number() % 10];
            if (rnd.get_random_32bit_number() % 7 == 0) text += ",";
            text += spaces[rnd.get_random_32bit_number() % 8];
        }
        const size_t vocab = 256 + 120 + 28;

        for (int mode = 0; mode < 2; ++mode)
        {
            bpe_tokenizer tok;
            if (mode == 1) tok.set_segmentation(bpe_segmentation::space_prefixed);
            tok.train(text, vocab);
            const std::vector<std::pair<int, int>> merges = tok.get_merges();

            const auto expected = reference_bpe_merges(reference_segments(text, mode == 1), 120);
            DLIB_TEST_MSG(merges == expected, "segmentation " << mode << ": merge table differs from the plain algorithm");

            // Encoding segment by segment gives what the whole-text application gives.
            const std::string probe = text.substr(1000, 4000) + "  ka ri\n\n to";
            const std::vector<int> ids = tok.encode(probe);
            DLIB_TEST(ids == reference_encode(probe, merges));
            DLIB_TEST(tok.decode(ids, false) == probe);

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
        }
    } a;

}
