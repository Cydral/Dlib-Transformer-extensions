// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

// The speech synthesis as the service uses it: the voices installed and the program that reads aloud.

#include "speech.h"

#include <chrono>
#include <condition_variable>
#include <locale>
#include <mutex>
#include <sstream>

#include "../common/platform.h"

namespace albireo
{

    namespace
    {
        // The programs of the speech synthesis that run at a time.
        std::mutex workers_m;
        std::condition_variable workers_cv;
        int workers = 0;
        const int most_workers = 2;
    }

    std::vector<speech_voice> installed_voices(const std::string& dir)
    {
        std::vector<speech_voice> out;
        for (const auto& v : speech_voices())
            if (file_exists(join_path(dir, v.text_file)) && file_exists(join_path(dir, v.sound_file))) out.push_back(v);
        return out;
    }

    bool speak(const std::string& program, const std::string& dir, const speech_voice& voice, double speed,
        const std::string& text, std::string& wav, std::string& error)
    {
        {
            std::unique_lock<std::mutex> lock(workers_m);
            if (!workers_cv.wait_for(lock, std::chrono::seconds(30), [] { return workers < most_workers; }))
            {
                error = "the speech synthesis is busy";
                return false;
            }
            ++workers;
        }
        struct release
        {
            ~release()
            {
                {
                    std::lock_guard<std::mutex> lock(workers_m);
                    --workers;
                }
                workers_cv.notify_one();
            }
        } released;
        std::ostringstream factor;
        factor.imbue(std::locale::classic());
        factor << speed;
        process_options options;
        options.input = &text;
        options.restricted = true;
        options.memory_limit = size_t(256) << 20;
        options.directory = dir;
        std::string output;
        const int code = run_process(program, { voice.id, factor.str() }, &output, 60000, nullptr, &options);
        if (code == 0 && output.size() >= 44 && output.compare(0, 4, "RIFF") == 0)
        {
            wav.swap(output);
            return true;
        }
        while (!output.empty() && (output.back() == '\n' || output.back() == '\r')) output.pop_back();
        if (output.size() > 300) output.resize(300);
        if (code == -1) error = "the speech program could not start or ran out of time";
        else if (!output.empty() && code != 0) error = output;
        else
        {
            // A code beyond a byte is a status of Windows, an access violation for instance.
            std::ostringstream o;
            o << "the speech program ended with the code " << code;
            if (code < 0 || code > 255) o << " (0x" << std::hex << static_cast<unsigned>(code) << ")";
            error = o.str();
        }
        return false;
    }

}
