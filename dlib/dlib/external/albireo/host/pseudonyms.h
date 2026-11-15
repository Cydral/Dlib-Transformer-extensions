// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_HOST_PSEUDONYMS_H_
#define ALBIREO_HOST_PSEUDONYMS_H_

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "../common/json.h"

namespace httplib { class Client; }

/*
    The pseudonymization of the relay (relay.h): the texts that leave for an external engine
    pass through Pseudalis, the anonymization service of the computer, which replaces each
    personal value by a substitute of the same form, or by a tag that carries it encrypted.
    The table of the substitutes and of their values lives in the memory of the service alone:
    it is never written, and the answers of the external engine get their values back from it
    as they stream.
*/

namespace albireo
{

    class cancellable
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The cancellation of an answer of the relay: its flag, and the client of the
                connection the answer holds, to Pseudalis or to the external engine, which
                cancel() stops from a thread of its own, since httplib waits for a connection
                being opened before it stops it: no caller waits.
        !*/
    public:
        bool cancelled() const { return cancelled_; }
        const std::atomic<bool>& flag() const { return cancelled_; }
        void cancel();
        // The client of the connection that starts, stopped at once when the answer is cancelled already.
        void hold(const std::shared_ptr<httplib::Client>& c);
        void release();

    private:
        std::atomic<bool> cancelled_{false};
        std::mutex m_;
        std::shared_ptr<httplib::Client> client_;
    };

    class pseudalis_client
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                Pseudalis as the relay reaches it, when it is installed on the same computer:
                its data folder (DataDir of HKLM\Software\Pseudalis, which its installer
                records), the port of its settings.json (8444 by default) and its local
                authority (certificates\pseudalis-authority.crt), which checks its
                certificate, reached on 127.0.0.1 under the name localhost.  On Windows, the
                folder, its certificates folder, the authority and the settings must be those
                of SYSTEM or the administrators, so that no other account puts a service of
                its own on the path of the texts.  The state found is kept for a minute,
                checked again by one caller at a time, outside the lock of the state, the
                other callers waiting for its result.
        !*/
    public:
        // {"found", "port", "reason"}
        json status();

        /*!
            ensures
                - Gives in out the text protected by Pseudalis (/v1/text/anonymize): each
                  personal value replaced by its tag, or by its substitute when tags is
                  false.
                - Gives in values each tag or substitute written in out with the value it
                  stands for, which /v1/text/restore gives back, in the order of out.
                - Waits for Pseudalis while it is busy with another work, until cancel, when
                  given, is cancelled, which also stops the request in progress.
                - Returns false with error when Pseudalis is not found, does not answer, or
                  refuses the text: the text then never leaves.
        !*/
        bool protect(const std::string& text, bool tags, cancellable* cancel, std::string& out,
            std::vector<std::pair<std::string, std::string>>& values, std::string& error);

        // The data folder of Pseudalis on this computer, empty when it has none.
        static std::string pseudalis_data_dir();

        /*!
            Reserved for the tests (tests/test_host.cpp).  No setting and no route reaches it.
            ensures
                - Sends the requests to origin ("http://127.0.0.1:port"), where a test server
                  answers as Pseudalis; an empty origin gives them back to Pseudalis.
        !*/
        static void endpoint_for_tests(const std::string& origin);

    private:
        struct endpoint
        {
            std::string origin;     // "https://localhost:port", or the origin of the tests
            std::string ca;         // the authority of Pseudalis
        };
        bool reach(endpoint& e, std::string& error);
        static bool probe(endpoint& e, std::string& reason);
        bool post(const endpoint& e, const std::string& path, const json& body, cancellable* cancel, json& answer,
            std::string& error);

        std::mutex m_;
        std::condition_variable checked_cv_;
        bool checking_ = false;
        int64_t checked_ = 0;
        bool found_ = false;
        endpoint endpoint_;
        std::string reason_;
    };

    class pseudonym_memory
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The memory of the protections of the relay, held by the service alone and never
                written: for each text protected, by the SHA-256 of the text and of its mode,
                its protected form and the substitutes it holds, so that the messages of a
                discussion, sent again at each of its turns, pass through Pseudalis once; for
                each substitute, its value.

                Both tables are bounded, in entries and in bytes, the least recently used
                entries going first when a new one needs room, and forget the entries left
                unused for max_idle seconds.  A text whose substitutes went is protected again.
                The values forgotten are erased from memory before their place is freed.
        !*/
    public:
        struct limits
        {
            size_t texts = 4096;
            size_t text_bytes = size_t(32) << 20;
            size_t values = 16384;
            size_t value_bytes = size_t(4) << 20;
            int64_t max_idle = 7200;        // seconds
        };

        pseudonym_memory();
        explicit pseudonym_memory(const limits& l);
        ~pseudonym_memory();
        pseudonym_memory(const pseudonym_memory&) = delete;
        pseudonym_memory& operator=(const pseudonym_memory&) = delete;

        /*!
            ensures
                - Gives in out the protected form of text, from memory or from client, and adds
                  to values each substitute of out with its value.
                - Returns false with error when client fails.
        !*/
        bool protect(const std::string& text, bool tags, pseudalis_client& client, cancellable* cancel,
            std::string& out, std::map<std::string, std::string>& values, std::string& error);

        // Forgets everything, the values erased.
        void clear();

        // {"texts", "values", "bytes"}
        json stats();

        // Reserved for the tests: the time of the memory, in seconds, 0 giving back the clock.
        void set_time_for_tests(int64_t t);

    private:
        struct text_entry
        {
            std::string out;
            std::vector<std::string> tokens;
            size_t bytes = 0;
            std::list<std::string>::iterator lru;
            int64_t used = 0;
        };
        struct value_entry
        {
            std::string value;
            size_t bytes = 0;
            std::list<std::string>::iterator lru;
            int64_t used = 0;
        };
        int64_t now() const;
        // Forgets the entries left unused for max_idle seconds, then the oldest beyond the bounds; m_ is held.
        void trim();
        void drop_text(std::map<std::string, text_entry>::iterator it);
        void drop_value(std::map<std::string, value_entry>::iterator it);

        limits limits_;
        std::mutex m_;
        std::map<std::string, text_entry> texts_;
        std::list<std::string> text_lru_;           // most recent first
        size_t text_bytes_ = 0;
        std::map<std::string, value_entry> values_;
        std::list<std::string> value_lru_;
        size_t value_bytes_ = 0;
        int64_t time_for_tests_ = 0;
    };

    class stream_restorer
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The values of an answer put back in place of their substitutes as the answer
                streams.  feed() gives back the text of a chunk up to the substitutes that its
                end may begin, which the next chunk completes; finish() gives back the rest.

                A substitute is found whole, the longest first, and only between word
                boundaries when it begins or ends with a letter or a digit, so that a short
                substitute never replaces a part of another word.  The words of a substitute of
                several words are found alone too, each giving the word of its value at the
                same place, when they have four bytes or more and stand for one word of one
                value only: an answer that names a person by the surname alone gets the surname
                back.  A whole substitute keeps its value when it is also a word of another
                one.  The text of a tag (its body, between ':' and '#') is found as the tag is.
        !*/
    public:
        explicit stream_restorer(const std::map<std::string, std::string>& values);
        ~stream_restorer();
        stream_restorer(const stream_restorer&) = delete;
        stream_restorer& operator=(const stream_restorer&) = delete;

        std::string feed(const std::string& chunk);
        std::string finish();

    private:
        std::string run(bool end);
        void add(const std::string& token, const std::string& value, bool whole);

        // The tokens, each with its value; by_byte_ lists them by their first byte, the longest first.
        std::vector<std::pair<std::string, std::string>> tokens_;
        std::vector<std::vector<size_t>> by_byte_;
        std::map<std::string, size_t> index_;
        std::vector<bool> ambiguous_, whole_;
        std::string pending_;
        std::string tail_;                          // the last bytes of the source given back
    };

}

#endif // ALBIREO_HOST_PSEUDONYMS_H_
