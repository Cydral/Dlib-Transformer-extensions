// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_HOST_USER_ERROR_H_
#define ALBIREO_HOST_USER_ERROR_H_

#include <string>

#include "../common/json.h"

namespace albireo
{

    struct user_error
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The failure of an operation that a user asked for: its English text, for
                the log and for the clients of the interface, and the code with which the
                windows explain it in the language of the user (the texts err_<code> of
                windows/strings.cpp), with its values and its sizes in bytes, which the
                windows write in the units of that language.
        !*/
        std::string text;
        std::string code;
        json args = json::object();
        json sizes = json::object();

        user_error& set(const std::string& english, const std::string& error_code = std::string())
        {
            text = english;
            code = error_code;
            return *this;
        }

        user_error& arg(const std::string& name, const std::string& value)
        {
            args[name] = value;
            return *this;
        }

        user_error& size(const std::string& name, unsigned long long bytes)
        {
            sizes[name] = bytes;
            return *this;
        }

        // Adds error, and code with its values when there is one, to the answer of a request.
        void add_to(json& answer) const
        {
            answer["error"] = text;
            if (code.empty()) return;
            answer["code"] = code;
            if (args.size() > 0) answer["args"] = args;
            if (sizes.size() > 0) answer["sizes"] = sizes;
        }
    };

}

#endif // ALBIREO_HOST_USER_ERROR_H_
