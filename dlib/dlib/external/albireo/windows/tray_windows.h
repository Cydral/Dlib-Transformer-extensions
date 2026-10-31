// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_WINDOWS_TRAY_WINDOWS_H_
#define ALBIREO_WINDOWS_TRAY_WINDOWS_H_

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <string>
#include <utility>

#include "service_client.h"

namespace albireo
{

    // The texts the menu of the icon shares with the windows.

    // The error of an answer of the service: the text err_<code> in the language of the
    // tray, with its values, or the English text of the service for a code without a text.
    std::wstring error_text(const json& answer);

    /*
        The answer to explain for a request that failed: the error of the connection, with
        the code and the values the service gave beside its error or inside it, as the
        interface compatible with OpenAI does, or the code "service" when no answer came.
    */
    json failed_answer(const json& answer, const std::string& error);

    // Why an engine of the list of the service (GET /albireo/v1/engines) cannot run here.
    std::wstring reason_text(const json& engine);

    // The runtime of an engine variant, "CUDA" or "OpenVINO", and the name of the variant
    // ("OpenVINO AVX-512"); the variant itself for the processor engines.
    std::string runtime_name(const std::string& variant);
    std::string engine_name(const std::string& variant);

    /*
        A choice of the engines (an element of "options"): its name, with the kind of its
        device, the names of its devices and its runtime ("GPU NVIDIA GeForce RTX 4070
        (CUDA)", "NPU + GPU Intel (OpenVINO)"), and the settings that select it, the
        engine "cuda" with the accelerator "gpu", or "openvino" with the kind of the choice,
        "npu", "gpu" or "both".
    */
    std::wstring choice_label(const json& option);
    std::pair<std::string, std::string> choice_settings(const json& option);

    // The windows of the tray; each one exists once and comes to the front when asked again.

    // The models: catalog, downloads, activation per function, checks, deletion, imports.
    void show_models_window(const service_client& client);

    // This computer: processor, memory, engines, extended graphics memory, models folder,
    // port, language and updates.
    void show_host_window(const service_client& client);

    // About Albireo.
    void show_about_window(const service_client& client);

    // Keyboard navigation in the windows of the tray; true when the message was handled.
    bool dispatch_dialog_message(MSG* msg);

}

#endif // ALBIREO_WINDOWS_TRAY_WINDOWS_H_
