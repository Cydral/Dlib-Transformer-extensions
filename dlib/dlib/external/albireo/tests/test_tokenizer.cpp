// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
/*
    Compares the tokenization of dlib::hf_tokenizer, loaded from GGUF files, with the
    token identifiers produced by the reference tokenizers of the same models.

        test_tokenizer texts.json expected.json name=model.gguf [name=model.gguf...]

    texts.json is a list of strings; expected.json maps each name to the list of token
    identifier lists of the texts, as the tokenizers library of the model hub returns
    them without special tokens added.  Every text must also decode back to itself.
*/

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <dlib/tokenizer/hf_tokenizer.h>

#include "../common/json.h"

static std::string read_file(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static std::string show(const std::vector<int>& v)
{
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) s += (i ? " " : "") + std::to_string(v[i]);
    return s;
}

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::printf("usage: test_tokenizer texts.json expected.json name=model.gguf...\n");
        return 2;
    }
    albireo::json texts, expected;
    std::string error;
    const bool parsed = albireo::json::parse(read_file(argv[1]), texts, &error)
        && albireo::json::parse(read_file(argv[2]), expected, &error);
    if (!parsed)
    {
        std::printf("cannot parse the inputs: %s\n", error.c_str());
        return 2;
    }

    int failures = 0;
    for (int a = 3; a < argc; ++a)
    {
        const std::string spec = argv[a];
        const size_t eq = spec.find('=');
        const std::string name = spec.substr(0, eq);
        dlib::gguf_reader g(spec.substr(eq + 1));
        dlib::hf_tokenizer tok;
        tok.load_from_gguf(g);
        const albireo::json& exp = expected[name];
        int ok = 0;
        for (size_t i = 0; i < texts.size(); ++i)
        {
            const std::string& text = texts[i].as_string();
            std::vector<int> want;
            for (const auto& v : exp[i].elements()) want.push_back(static_cast<int>(v.as_int()));
            const std::vector<int> got = tok.encode(text, false, false, true);
            if (got != want)
            {
                ++failures;
                std::printf("%s: text %zu differs\n  want %s\n  got  %s\n", name.c_str(), i, show(want).c_str(),
                    show(got).c_str());
                continue;
            }
            // Decoding gives the text back, special tokens included, in its canonical
            // composition when the vocabulary normalizes it.
            const std::string back = tok.decode(got, false);
            const std::string original = dlib::hf_tok_impl::pre_normalizes_nfc(tok.pretokenizer())
                ? dlib::hf_tok_impl::nfc_compose(text) : text;
            if (back != original && !(tok.type() == dlib::hf_tokenizer::kind::spm && back == original.substr(1)))
            {
                ++failures;
                std::printf("%s: text %zu does not decode back\n", name.c_str(), i);
                continue;
            }
            std::string streamed;
            for (int id : got) streamed += tok.token_bytes(id);
            if (streamed != back)
            {
                ++failures;
                std::printf("%s: text %zu, token bytes differ from decode\n", name.c_str(), i);
                continue;
            }
            ++ok;
        }
        std::printf("%-10s pre-tokenizer '%s'%s: %d of %zu texts identical\n", name.c_str(), tok.pretokenizer().c_str(),
            tok.exact_pretokenizer() ? "" : " (approximated)", ok, texts.size());
    }
    if (failures)
    {
        std::printf("%d texts FAILED\n", failures);
        return 1;
    }
    std::printf("all texts identical\n");
    return 0;
}
