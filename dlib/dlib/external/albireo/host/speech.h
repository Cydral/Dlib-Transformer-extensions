// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_HOST_SPEECH_H_
#define ALBIREO_HOST_SPEECH_H_

#include <string>
#include <vector>

namespace albireo
{

    /*!
        The speech synthesis that comes with Albireo, which reads the answers aloud without a
        network and without the voices of the system: the SVOX Pico synthesizer and its voices
        of six languages (American and British English, French, German, Italian and Spanish),
        each a text analysis file and a signal generation file in the speech folder of the
        installation.  The output is a RIFF WAVE file of 16-bit mono samples at 16 kHz.

        The service never runs the synthesizer itself.  The program albireo-speech of the
        installation (host/speech_worker.cpp) reads the text on its standard input and writes
        the WAVE file on its standard output.  The service starts it in the speech folder,
        without console, with a token of low integrity and without privileges, in a job
        bounded in memory (run_process with process_options::restricted), so that a fault of
        the synthesizer on some text never reaches the service.  That program holds the
        synthesizer, the functions of host/speech_text.cpp and the runtime of the compiler,
        nothing else: it loads none of the libraries of the windows of Windows, which a
        process of low integrity started by a service could not initialize.

        speech_voices(), find_voice() and speech_text() serve both the service and that
        program (host/speech_text.cpp); installed_voices() and speak() serve the service
        (host/speech.cpp).
    !*/

    struct speech_voice
    {
        std::string id;         // "fr-FR"
        std::string language;   // "fr"
        std::string text_file;  // text analysis, "fr-FR_ta.bin"
        std::string sound_file; // signal generation, "fr-FR_nk0_sg.bin"
    };

    // The voices Albireo knows, present or not, the first of each language its default.
    const std::vector<speech_voice>& speech_voices();

    /*
        The voice named by an identifier or a language, whatever its case ("fr-FR", "fr",
        "en-gb"), else nullptr.
    */
    const speech_voice* find_voice(const std::string& name);

    // The longest text a request reads, in bytes of UTF-8.
    const size_t speech_max_bytes = 8192;

    /*
        The text the synthesizer reads: the letters of the Latin scripts, the digits, the
        usual punctuation and the line breaks, the typographic quotes and dashes in their
        plain forms, every other character a space (the synthesizer reads <...> as commands,
        some of which name files); runs of spaces collapsed, words cut every 64 characters,
        at most max_bytes bytes cut at a character.
    */
    std::string speech_text(const std::string& text, size_t max_bytes);

    // The voices whose two files lie in dir.
    std::vector<speech_voice> installed_voices(const std::string& dir);

    /*
        Reads text aloud at speed (0.5 to 2, 1 the speed of the voice) with program, the
        program albireo-speech, started in dir, the speech folder: returns true with the WAVE
        file in wav, or false with the reason in error.  At most two such programs run at a
        time; a request waits up to 30 seconds for its turn.
    */
    bool speak(const std::string& program, const std::string& dir, const speech_voice& voice, double speed,
        const std::string& text, std::string& wav, std::string& error);

}

#endif // ALBIREO_HOST_SPEECH_H_
