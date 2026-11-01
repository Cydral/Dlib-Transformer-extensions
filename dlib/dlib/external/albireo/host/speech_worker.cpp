// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
/*
    albireo-speech: reads a text aloud with the SVOX Pico synthesizer, for the service.

        albireo-speech <voice> <speed>

    The program reads the text in UTF-8 on its standard input (at most 16 KiB), keeps what
    speech_text() keeps of it, and writes a RIFF WAVE file of 16-bit mono samples at 16 kHz
    on its standard output.  The two files of the voice ("fr-FR", "de-DE" ...) are read
    from the current folder, where the service starts the program: the speech folder of the
    installation.  speed goes from 0.5 to 2, 1 the speed of the voice.  A failure is told on
    the standard error, with the exit code 2 (arguments), 4 (synthesis) or 5 (output).

    The program links the synthesizer, host/speech_text.cpp and the runtime of the
    compiler, and nothing of the rest of Albireo: it loads no library of the windows of
    Windows, so that it starts with the token of low integrity and in the job the service
    gives it, without console and without access to a desktop.
*/

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "speech.h"

extern "C"
{
#include "picoapi.h"
}

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#endif

namespace
{
    using albireo::speech_voice;

    // The samples a text gives at most: ten minutes at 16 kHz.
    const size_t most_samples = size_t(16000) * 600;

    // The bytes of the standard input the program reads at most.
    const size_t most_input = albireo::speech_max_bytes * 2;

    void put_u32(std::string& s, uint32_t v)
    {
        for (int b = 0; b < 4; ++b) s += static_cast<char>((v >> (8 * b)) & 0xFF);
    }

    void put_u16(std::string& s, uint16_t v)
    {
        s += static_cast<char>(v & 0xFF);
        s += static_cast<char>((v >> 8) & 0xFF);
    }

    // A RIFF WAVE file of 16-bit mono samples at 16 kHz.
    std::string wave_file(const std::vector<int16_t>& samples)
    {
        const uint32_t data = static_cast<uint32_t>(samples.size() * 2);
        std::string w;
        w.reserve(44 + data);
        w += "RIFF";
        put_u32(w, 36 + data);
        w += "WAVEfmt ";
        put_u32(w, 16);
        put_u16(w, 1);          // PCM
        put_u16(w, 1);          // mono
        put_u32(w, 16000);
        put_u32(w, 32000);      // bytes per second
        put_u16(w, 2);          // bytes per frame
        put_u16(w, 16);         // bits per sample
        w += "data";
        put_u32(w, data);
        for (int16_t v : samples) put_u16(w, static_cast<uint16_t>(v));
        return w;
    }

    std::string status_text(pico_System sys, pico_Status s)
    {
        pico_Retstring message;
        std::memset(message, 0, sizeof(message));
        if (sys) pico_getSystemStatusMessage(sys, s, message);
        std::ostringstream o;
        o << "speech synthesis error " << s;
        if (message[0]) o << " (" << reinterpret_cast<const char*>(message) << ")";
        return o.str();
    }

    // The synthesizer and its voice, released in the reverse order of their making.
    struct pico_session
    {
        std::vector<char> memory;
        pico_System sys = nullptr;
        pico_Resource text = nullptr, sound = nullptr;
        pico_Engine engine = nullptr;
        bool voice = false;

        ~pico_session()
        {
            if (engine) pico_disposeEngine(sys, &engine);
            if (voice) pico_releaseVoiceDefinition(sys, reinterpret_cast<const pico_Char*>("albireo"));
            if (sound) pico_unloadResource(sys, &sound);
            if (text) pico_unloadResource(sys, &text);
            if (sys) pico_terminate(&sys);
        }
    };

    /*
        Synthesizes text, which speech_text() gives, with the files of voice in the current
        folder, at speed, into wav.  Returns false with the reason in error.
    */
    bool synthesize(const speech_voice& voice, const std::string& text, double speed, std::string& wav,
        std::string& error)
    {
        pico_session p;
        p.memory.resize(size_t(4) << 20);
        pico_Status s = pico_initialize(p.memory.data(), static_cast<pico_Uint32>(p.memory.size()), &p.sys);
        if (s != PICO_OK)
        {
            error = status_text(p.sys, s);
            return false;
        }
        s = pico_loadResource(p.sys, reinterpret_cast<const pico_Char*>(voice.text_file.c_str()), &p.text);
        if (s == PICO_OK)
            s = pico_loadResource(p.sys, reinterpret_cast<const pico_Char*>(voice.sound_file.c_str()), &p.sound);
        if (s != PICO_OK)
        {
            error = "the voice " + voice.id + " cannot be read: " + status_text(p.sys, s);
            return false;
        }
        // The names of the resources, in strings of the size the interface writes.
        pico_Retstring text_name, sound_name;
        std::memset(text_name, 0, sizeof(text_name));
        std::memset(sound_name, 0, sizeof(sound_name));
        const pico_Char* name = reinterpret_cast<const pico_Char*>("albireo");
        s = pico_getResourceName(p.sys, p.text, text_name);
        if (s == PICO_OK) s = pico_getResourceName(p.sys, p.sound, sound_name);
        if (s == PICO_OK) s = pico_createVoiceDefinition(p.sys, name);
        if (s == PICO_OK)
        {
            p.voice = true;
            s = pico_addResourceToVoiceDefinition(p.sys, name, reinterpret_cast<const pico_Char*>(text_name));
        }
        if (s == PICO_OK)
            s = pico_addResourceToVoiceDefinition(p.sys, name, reinterpret_cast<const pico_Char*>(sound_name));
        if (s == PICO_OK) s = pico_newEngine(p.sys, name, &p.engine);
        if (s != PICO_OK)
        {
            error = status_text(p.sys, s);
            return false;
        }

        // The speed through the markup of the synthesizer, the only markup of the text it reads.
        const double factor = std::isfinite(speed) ? std::min(2.0, std::max(0.5, speed)) : 1.0;
        std::string input = "<speed level=\"" + std::to_string(static_cast<int>(std::lround(factor * 100.0)))
            + "\">" + text + "</speed>";
        input.push_back('\0');

        std::vector<int16_t> samples;
        size_t at = 0;
        int stalls = 0;
        while (at < input.size())
        {
            const size_t left = input.size() - at;
            pico_Int16 sent = 0;
            const pico_Int16 chunk = static_cast<pico_Int16>(std::min<size_t>(left, 4096));
            s = pico_putTextUtf8(p.engine, reinterpret_cast<const pico_Char*>(input.data() + at), chunk, &sent);
            if (s != PICO_OK)
            {
                error = status_text(p.sys, s);
                return false;
            }
            at += static_cast<size_t>(std::max<pico_Int16>(sent, 0));
            pico_Status step;
            do
            {
                int16_t buffer[64];
                pico_Int16 received = 0, type = 0;
                step = pico_getData(p.engine, buffer, static_cast<pico_Int16>(sizeof(buffer)), &received, &type);
                if (step != PICO_STEP_BUSY && step != PICO_STEP_IDLE)
                {
                    error = status_text(p.sys, step);
                    return false;
                }
                if (received > 0 && type != PICO_DATA_PCM_16BIT)
                {
                    error = "the voice gives samples of another format";
                    return false;
                }
                if (received > 0)
                    samples.insert(samples.end(), buffer, buffer + std::min<long>(received / 2, 64));
                if (samples.size() > most_samples)
                {
                    error = "the text is too long to be read at once";
                    return false;
                }
            } while (step == PICO_STEP_BUSY);
            // The synthesizer takes nothing more only when its input is full and it is idle: an error.
            stalls = sent > 0 ? 0 : stalls + 1;
            if (stalls > 3)
            {
                error = "the speech synthesis does not take the text";
                return false;
            }
        }
        wav = wave_file(samples);
        return true;
    }
}

int main(int argc, char** argv)
{
#if defined(_WIN32)
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    if (argc != 3)
    {
        std::fprintf(stderr, "usage: albireo-speech <voice> <speed>, the text on the standard input\n");
        return 2;
    }
    const speech_voice* voice = albireo::find_voice(argv[1]);
    if (!voice)
    {
        std::fprintf(stderr, "unknown voice %s\n", argv[1]);
        return 2;
    }
    std::string input;
    char buffer[4096];
    for (;;)
    {
        const size_t n = std::fread(buffer, 1, sizeof(buffer), stdin);
        if (n == 0) break;
        input.append(buffer, std::min(n, most_input - std::min(most_input, input.size())));
        if (input.size() >= most_input) break;
    }
    std::string wav, error;
    if (!synthesize(*voice, albireo::speech_text(input, albireo::speech_max_bytes), std::strtod(argv[2], nullptr),
            wav, error))
    {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 4;
    }
    const bool written = std::fwrite(wav.data(), 1, wav.size(), stdout) == wav.size();
    return std::fflush(stdout) == 0 && written ? 0 : 5;
}
