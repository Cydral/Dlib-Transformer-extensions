// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef PSEUDALIS_HOST_ALBIREO_CLIENT_H_
#define PSEUDALIS_HOST_ALBIREO_CLIENT_H_

#include <cstdint>
#include <mutex>
#include <string>

#include "../common/json.h"

namespace pseudalis
{
    class albireo_client
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The descriptions of the pictures by Albireo, the local inference service,
                when it is installed on the same computer and one of its models reads
                images.  Albireo is found through its interface: its data folder
                (DataDir of HKLM\Software\Albireo, else %ProgramData%\Albireo), the port of
                its settings.json (8443 by default) and its local authority
                (certificates\albireo-authority.crt), which checks its certificate; then
                GET /v1/models, whose "inputs" name the models that read images.  The
                result is kept for a minute.

                describe() sends the picture to POST /v1/chat/completions, one request at a
                time, and asks for one or two factual sentences that name no person and
                copy no name, address or number the picture shows; the engine protects the
                answer as any text before writing it.  Only PNG and JPEG pictures of 15 MB
                at most are sent.
        !*/
    public:
        albireo_client();

        // {"enabled", "found", "port", "model", "reason"}
        json status();
        void set_enabled(bool on);

        // The description, or an empty string with the reason in error.
        std::string describe(const std::string& bytes, const std::string& media_type, const std::string& language,
            std::string& error);

        // The data folder of Albireo on this computer, empty when it has none.
        static std::string albireo_data_dir();

    private:
        bool refresh(std::string& reason);
        std::mutex m_;              // the state
        std::mutex request_m_;      // one description at a time
        bool enabled_ = true;
        int64_t checked_ = 0;
        bool found_ = false;
        int port_ = 8443;
        std::string ca_;
        std::string model_;
        std::string reason_;
    };
}

#endif // PSEUDALIS_HOST_ALBIREO_CLIENT_H_
