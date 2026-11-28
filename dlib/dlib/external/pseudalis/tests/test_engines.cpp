// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
/*
    Tests of the engine libraries built beside this program, loaded as the service loads them: the
    library of every x64 processor, and the one of the processors with AVX2, whose code no other
    test runs.

        test_engines baseline
        test_engines avx2

    The variant named protects documents under a fixed key, restores them, and prints one line for
    each document, with a digest of what it wrote.  Both variants read the same models and follow
    the same rules: the scripts of the build run the program once for each and compare these
    lines.  On a processor that runs no AVX2 code, the second command says so and runs nothing.
*/
#if defined(_WIN32) && !defined(_WIN32_WINNT)
#define _WIN32_WINNT 0x0A00
#endif

#include <cstdio>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include "../common/engine_loader.h"
#include "../common/json.h"
#include "../common/platform.h"

using pseudalis::json;

namespace
{
    int failures = 0;
#define CHECK(cond) check((cond), #cond, __LINE__)
    void check(bool ok, const char* what, int line)
    {
        if (ok) return;
        ++failures;
        std::cerr << "test_engines.cpp:" << line << ": failed: " << what << std::endl;
    }

    /*
        The end of the program once a library is loaded.  On Windows the process ends at once, as the
        service does (host/main.cpp): the library stays loaded, and its static destructors, run at the
        end of the process, would join the threads of the pool of dlib that Windows ended first, and
        wait forever.
    */
    int finish(int code)
    {
        std::cout.flush();
        std::cerr.flush();
#if defined(_WIN32)
        TerminateProcess(GetCurrentProcess(), static_cast<UINT>(code));
#endif
        return code;
    }

    // The FNV-1a digest of data, in hexadecimal.
    std::string digest(const std::string& data)
    {
        unsigned long long h = 14695981039346656037ULL;
        for (const char c : data)
        {
            h ^= static_cast<unsigned char>(c);
            h *= 1099511628211ULL;
        }
        char text[17];
        std::snprintf(text, sizeof(text), "%016llx", h);
        return text;
    }

    struct sample
    {
        const char* name;       // the name of the file, whose extension chooses its reader
        std::string data;
        const char* hidden;     // a value that the protected file must not hold
        const char* tag;        // the opening of a tag that it must hold; null for a file written without tags
    };

    std::vector<sample> samples()
    {
        std::vector<sample> all;
        // The rules and both recognizers: a person, a phone number, an address of electronic mail.
        all.push_back({ "lettre.txt",
            "Bonjour,\nJe suis Jean Dupont, joignable au 06 12 34 56 78 ou jean.dupont@example.fr.\n"
            "Notre agence de Quimper vous recevra lundi.\n",
            "Dupont", "[[PERSON:" });
        // Names and an organisation that no label announces, which the recognizers read.
        all.push_back({ "minutes.txt",
            "Minutes of the meeting. Sarah Whitfield presented the audit of Northbridge Logistics Ltd, "
            "then Tom Baker asked for the figures of the Leeds warehouse.\n",
            "Whitfield", "[[PERSON:" });
        // A secret that neither a label nor a key announces: the span model alone reads it.
        all.push_back({ "acces.txt",
            "Pour la recette, voici les acces du compte de test : jdupont / Tournesol2024!\n"
            "Merci de ne pas les diffuser.\n",
            "Tournesol2024!", "[[SECRET:" });
        // The settings of a program: a key names the secret.
        all.push_back({ "service.env",
            "DB_HOST=db.internal\nDB_USER=crm_app\nDB_PASSWORD=Vx7!kPq2#Lm9zR\nMAIL_FROM=support@example.org\n",
            "Vx7!kPq2#Lm9zR", "[[SECRET:" });
        // A table, read cell by cell under its header.
        all.push_back({ "clients.csv",
            "Nom;Courriel;Telephone;Ville\n"
            "Claire Martin;claire.martin@example.com;01 23 45 67 89;Lyon\n"
            "Paolo Rossi;p.rossi@example.it;+39 06 1234 5678;Roma\n",
            "Rossi", nullptr });
        // A document of a programming interface.
        all.push_back({ "request.json",
            "{\"customer\": {\"name\": \"Maria Schneider\", \"email\": \"m.schneider@example.de\"}, "
            "\"note\": \"Call her back at +49 30 1234567 before Friday.\"}\n",
            "Schneider", nullptr });
        return all;
    }
}

int main(int argc, char** argv)
{
    const std::string variant = argc > 1 ? argv[1] : "";
    if (variant != "baseline" && variant != "avx2")
    {
        std::cerr << "Usage: test_engines baseline|avx2" << std::endl;
        return 2;
    }
    if (variant == "avx2" && !pseudalis::cpu_has_avx2())
    {
        std::cout << "test_engines: avx2 not run, the processor runs no AVX2 code" << std::endl;
        return 0;
    }
    std::string error;
    std::shared_ptr<pseudalis::engine_library> library = std::make_shared<pseudalis::engine_library>();
    if (!library->load(pseudalis::executable_dir(), variant, error))
    {
        std::cerr << "test_engines: " << error << std::endl;
        return finish(1);
    }
    // The library asked for, never the other one in its place.
    CHECK(library->variant() == variant);
    if (library->variant() != variant)
    {
        std::cerr << "test_engines: " << library->path() << " is the " << library->variant() << " engine" << std::endl;
        return finish(1);
    }
    const std::string key(32, '\x5a');
    json options = json::object();
    options["date_order"] = "dmy";
    pseudalis::engine_client engine(library, key, options, error);
    CHECK(engine.ok());
    if (!engine.ok())
    {
        std::cerr << "test_engines: " << error << std::endl;
        return finish(1);
    }
    CHECK(engine.variant() == variant);
    // The models compiled into the library, the same for both variants.
    std::cout << "models " << digest(engine.models().dump()) << std::endl;

    const std::function<std::string(const pseudalis::picture_request&)> no_pictures;
    for (const sample& s : samples())
    {
        json request = json::object(), result;
        request["operation"] = "anonymize";
        request["name"] = s.name;
        std::string written;
        const bool anonymized = engine.process(request, s.data, result, written, no_pictures, error);
        CHECK(anonymized);
        if (!anonymized)
        {
            std::cerr << "test_engines: " << s.name << ": " << error << std::endl;
            continue;
        }
        CHECK(written.find(s.hidden) == std::string::npos);
        CHECK(result["report"]["total"].as_int() >= 1);
        if (s.tag)
        {
            // The tags carry their values: the key alone gives the file back.
            CHECK(written.find(s.tag) != std::string::npos);
            request["operation"] = "restore";
            json restored_result;
            std::string restored;
            const bool given_back = engine.process(request, written, restored_result, restored, no_pictures, error);
            CHECK(given_back);
            CHECK(restored == s.data);
        }
        std::cout << "sample " << s.name << " values " << result["report"]["total"].as_int() << " bytes "
                  << written.size() << " digest " << digest(written) << std::endl;
    }

    if (failures) std::cout << "test_engines: " << failures << " checks FAILED (" << variant << ")" << std::endl;
    else std::cout << "test_engines: all tests passed (" << variant << ")" << std::endl;
    return finish(failures ? 1 : 0);
}
