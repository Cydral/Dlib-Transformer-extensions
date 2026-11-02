// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_COMMON_LOG_H_
#define ALBIREO_COMMON_LOG_H_

#include <cstdint>
#include <deque>
#include <mutex>
#include <sstream>
#include <string>

#include "json.h"

namespace albireo
{

    enum class log_level { debug = 0, info = 1, warning = 2, error = 3 };

    /*!
        ensures
            - Returns text with each word that names the publication space of Albireo replaced
              by "the publication space of Albireo": the addresses and the names of the hosts of
              the service that holds it (a word holding "pcloud" or "publink", whatever their
              case), and the words holding a text given to hide_in_messages().  A word ends at a
              space, a quote, a bracket, a comma or a semicolon, and the punctuation that ends a
              sentence after it stays.
            - The log (log_file), the errors the routes of the service return, the windows of
              the tray and the log of the installer pass their texts through this function, the
              last guard after the messages, which name the files of the publication space and
              never its addresses.
    !*/
    std::string without_publication_space(const std::string& text);

    // Whether without_publication_space() changes text.
    bool names_publication_space(const std::string& text);

    // Adds secret, the code or the password of the publication space, to what without_publication_space()
    // hides; a text of fewer than 6 characters is left out, so that no common word is hidden.
    void hide_in_messages(const std::string& secret);

    // Passes the lines of the text file at path through without_publication_space(), written again when one changes.
    void hide_publication_space_in_file(const std::string& path);

    // Whether a string of document under a member that tells an error ("error", "errors", "args"), at any
    // depth, names the publication space.
    bool errors_name_publication_space(const json& document);

    // Passes the strings of document under the members that tell an error through without_publication_space().
    void hide_publication_space_in_errors(json& document);

    class log_file
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The log of a program: lines with their time and severity, written to a
                file that is renamed with a ".1" suffix when it reaches its size limit, so
                that two files at most are kept.  The last lines also stay in memory, for
                the windows that show them.  Each message passes through
                without_publication_space(), and the files a version before this one wrote
                pass through it when they are opened.
        !*/
    public:
        static log_file& instance();

        void open(const std::string& path, uint64_t max_bytes = 4u << 20);
        void set_console(bool on) { console_ = on; }
        void set_level(log_level l) { level_ = l; }

        void write(log_level level, const std::string& message);
        std::deque<std::string> recent() const;

    private:
        log_file() {}
        mutable std::mutex m_;
        std::string path_;
        uint64_t max_bytes_ = 4u << 20;
        bool console_ = false;
        log_level level_ = log_level::info;
        std::deque<std::string> recent_;
    };

    inline void log_debug(const std::string& m) { log_file::instance().write(log_level::debug, m); }
    inline void log_info(const std::string& m) { log_file::instance().write(log_level::info, m); }
    inline void log_warning(const std::string& m) { log_file::instance().write(log_level::warning, m); }
    inline void log_error(const std::string& m) { log_file::instance().write(log_level::error, m); }

}

#endif // ALBIREO_COMMON_LOG_H_
