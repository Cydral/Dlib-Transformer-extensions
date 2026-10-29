// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_WINDOWS_TRAY_WINDOWS_H_
#define ALBIREO_WINDOWS_TRAY_WINDOWS_H_

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include "service_client.h"

namespace albireo
{

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
