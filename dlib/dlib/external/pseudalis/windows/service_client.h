// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef PSEUDALIS_WINDOWS_SERVICE_CLIENT_H_
#define PSEUDALIS_WINDOWS_SERVICE_CLIENT_H_

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>

#include "../common/json.h"

namespace pseudalis
{

    class service_client
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The connection of the tray to the local service: HTTPS to localhost on the
                port of the settings, the server certificate checked against the local
                authority of the data folder, and the administration token of the data
                folder sent with every request.  get(), send(), upload() and post_bytes() open a
                connection of their own at each call, so that calls may run on several
                threads at once.

            KEPT CONNECTION
                poll() sends its requests on one connection kept open between its calls,
                so that the periodic polls pay the TLS handshake once.  The copies of an
                object share that connection, which serves one call at a time: a call of
                another thread waits for the call in progress.  The connection closes
                once it stays unused for 20 seconds, and opens again after a failure, a
                restart of the service or a change of the port.  own_connection() gives
                a copy a kept connection of its own, for a window that polls beside the
                icon.
        !*/
    public:
        service_client();

        // Reads the port and the token again, after a change of the settings.
        void reload();

        int port() const { return port_; }
        // The administration token of the data folder, which the icon gives to the web application.
        const std::string& token() const { return token_; }
        std::string base_url() const;
        std::string data_dir() const { return data_dir_; }

        bool get(const std::string& path, json& out, std::string& error, int timeout_seconds = 10) const;
        // method: "POST", "PUT" or "DELETE".
        bool send(const char* method, const std::string& path, const json& body, json& out, std::string& error,
            int timeout_seconds = 30) const;
        // Sends a file to the service; progress receives the bytes sent and returns false to stop.
        bool upload(const std::string& path, const std::string& file,
            const std::function<bool(uint64_t, uint64_t)>& progress, json& out, std::string& error) const;

        // The answer of a request whose body and answer are bytes.
        struct raw_answer
        {
            int status = 0;
            std::string body;
            std::map<std::string, std::string> headers;   // names in lower case
        };
        /*
            A POST of bytes, the file to protect or to restore, whose answer is the file made.
            Returns false, with error, when no answer comes; an answer of the service, an
            error included, gives true with its status.
        */
        bool post_bytes(const std::string& path, const std::string& bytes, raw_answer& out, std::string& error,
            int timeout_seconds = 600) const;

        // A GET on the kept connection; a failure on a connection kept from an earlier call
        // is tried once more on a new connection.
        bool poll(const std::string& path, json& out, std::string& error, int timeout_seconds = 10) const;
        void own_connection();
        // The connections poll() opened since the creation of the kept connection.
        unsigned long long poll_connections() const;

    private:
        struct kept_connection;

        std::string data_dir_;
        std::string ca_path_;
        std::string token_;
        int port_ = 8444;
        std::shared_ptr<kept_connection> kept_;
    };

}

#endif // PSEUDALIS_WINDOWS_SERVICE_CLIENT_H_
