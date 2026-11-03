// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef PSEUDALIS_WINDOWS_STRINGS_H_
#define PSEUDALIS_WINDOWS_STRINGS_H_

#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

#include "../common/json.h"

namespace pseudalis
{

    /*
        The texts of the icon, of the commands of the Explorer and of the installer, in
        English, French, German, Italian and Spanish.  A language without a text for a key
        falls back on English; a key without any text is shown as is.
    */

    // The language of the interface of Windows among the five, English otherwise.
    std::string system_language();

    // Selects the language of the texts: a code ("en", "fr", "de", "it", "es") or "auto".
    void set_language(const std::string& code);
    std::string current_language();

    std::string tr_utf8(const char* key);
    std::wstring tr(const char* key);
    // A text with its {name} marks replaced by the values given.
    std::wstring tr(const char* key, std::initializer_list<std::pair<const char*, std::string>> values);
    std::wstring tr(const std::string& key, const std::vector<std::pair<std::string, std::string>>& values);
    // Whether the table holds a text for key.
    bool has_text(const std::string& key);
    // The texts of key in the five languages, UTF-8.
    std::vector<std::string> texts_of(const char* key);
    /*
        A text that counts n things, its {n} replaced by n: the text key_one for one thing, and for
        none in French, when the table holds it, else the text key.
    */
    std::wstring tr_count(const char* key, long long n,
        std::initializer_list<std::pair<const char*, std::string>> values = {});

    /*
        The error of an answer of the service ({"error", "code", "args", "sizes"}, as
        host/user_error.h writes it): the text err_<code> in the language of the texts, with
        its values, or the English text of the service for a code without a text.
    */
    std::wstring error_text(const json& answer);

    /*
        The answer to explain for a request that failed: the error of the connection, with
        the code and the values the service gave, or the code "service" when no answer came.
    */
    json failed_answer(const json& answer, const std::string& error);

}

#endif // PSEUDALIS_WINDOWS_STRINGS_H_
