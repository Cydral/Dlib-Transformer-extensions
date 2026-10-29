// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_WINDOWS_STRINGS_H_
#define ALBIREO_WINDOWS_STRINGS_H_

#include <initializer_list>
#include <string>
#include <utility>

namespace albireo
{

    /*
        The texts of the tray and of the installer, in English, French, German, Italian
        and Spanish.  A language without a text for a key falls back on English; a key
        without any text is shown as is.
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

}

#endif // ALBIREO_WINDOWS_STRINGS_H_
