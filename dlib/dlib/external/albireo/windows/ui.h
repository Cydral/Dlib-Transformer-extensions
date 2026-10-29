// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_WINDOWS_UI_H_
#define ALBIREO_WINDOWS_UI_H_

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <commctrl.h>

#include <functional>
#include <string>

#include "../common/json.h"

/*
    The few pieces of user interface the tray and the installer share: the scaling of
    sizes to the resolution of the monitor of a window, the fonts, the images of the
    resources (PNG decoded and resized by the Windows Imaging Component), the creation of
    the standard controls, and work done on a thread whose result comes back to a window
    as a message.
*/

namespace albireo
{
namespace ui
{

    // Common controls, COM; once per program.
    void init();

    // Resolution of the monitor of a window (96 at 100 %), and a size scaled to it.
    UINT dpi_of(HWND w);
    int px(HWND w, int size96);

    enum class font_kind { normal, bold, note, title, subtitle, mono };
    // A font for the resolution of a window, kept for the life of the program.
    HFONT font(HWND w, font_kind kind);

    // Colors of the windows.
    COLORREF color_background();
    COLORREF color_text();
    COLORREF color_faint();
    COLORREF color_accent();
    HBRUSH background_brush();

    /*
        A PNG resource of the program decoded and resized to width x height pixels, as a
        32-bit premultiplied bitmap, or null.  The caller deletes it with DeleteObject.
    */
    HBITMAP load_png(int resource_id, int width, int height);
    // The size of a PNG resource in pixels.
    SIZE png_size(int resource_id);
    // Draws a premultiplied bitmap at x, y.
    void draw_bitmap(HDC dc, HBITMAP bmp, int x, int y);

    // Controls; the font of the window is set on each.
    HWND label(HWND parent, const std::wstring& text, int id = 0, DWORD style = 0, font_kind f = font_kind::normal);
    HWND button(HWND parent, const std::wstring& text, int id, DWORD style = 0);
    HWND checkbox(HWND parent, const std::wstring& text, int id, bool checked);
    HWND edit(HWND parent, const std::wstring& text, int id, DWORD style = 0);
    HWND combo(HWND parent, int id);
    HWND progress(HWND parent, int id);
    HWND listview(HWND parent, int id, DWORD style = 0);

    void place(HWND control, int x, int y, int w, int h);
    // Width of a button for its text and font, at least min_width pixels.
    int button_width(HWND button, int min_width);
    std::wstring text_of(HWND control);
    void set_text(HWND control, const std::wstring& text);
    bool checked(HWND control);
    void enable(HWND control, bool on);

    // Message box with the title of the program.
    int message(HWND owner, const std::wstring& text, UINT flags);

    // Opens an address or a folder with the program Windows associates with it.
    void open_url(const std::wstring& target);

    // A folder chosen by the user, or an empty string.
    std::wstring choose_folder(HWND owner, const std::wstring& initial);

    /*
        Runs work on a thread of its own and posts its result to target as the message
        msg, with a json* in lParam that the window procedure deletes.
    */
    void run_async(HWND target, UINT msg, WPARAM tag, std::function<json()> work);

    // Formats a size in bytes: "2.4 GB".
    std::string format_size(unsigned long long bytes);

}
}

#endif // ALBIREO_WINDOWS_UI_H_
