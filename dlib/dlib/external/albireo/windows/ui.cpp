// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "ui.h"

#include <algorithm>
#include <exception>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

#include <objbase.h>
#include <shellapi.h>
#include <shlobj.h>
#include <shobjidl.h>
#include <wincodec.h>

#include "../common/platform.h"
#include "strings.h"

namespace albireo
{
namespace ui
{

    namespace
    {
        std::mutex fonts_m;
        std::map<std::pair<UINT, int>, HFONT> fonts;
        std::wstring text_face, display_face;

        int CALLBACK font_found(const LOGFONTW*, const TEXTMETRICW*, DWORD, LPARAM found)
        {
            *reinterpret_cast<bool*>(found) = true;
            return 0;
        }

        bool font_exists(const wchar_t* face)
        {
            HDC dc = GetDC(nullptr);
            LOGFONTW lf = {};
            lf.lfCharSet = DEFAULT_CHARSET;
            wcsncpy(lf.lfFaceName, face, LF_FACESIZE - 1);
            bool found = false;
            EnumFontFamiliesExW(dc, &lf, font_found, reinterpret_cast<LPARAM>(&found), 0);
            ReleaseDC(nullptr, dc);
            return found;
        }

        HFONT create_font(UINT dpi, font_kind kind)
        {
            if (text_face.empty())
            {
                // The variable faces of Windows 11, else those of Windows 10.
                text_face = font_exists(L"Segoe UI Variable Text") ? L"Segoe UI Variable Text" : L"Segoe UI";
                display_face = font_exists(L"Segoe UI Variable Display") ? L"Segoe UI Variable Display" : L"Segoe UI";
            }
            int points = 9, weight = FW_NORMAL;
            const wchar_t* face = text_face.c_str();
            switch (kind)
            {
            case font_kind::bold: weight = FW_SEMIBOLD; break;
            case font_kind::note: points = 8; break;
            case font_kind::title: points = 17; weight = FW_SEMIBOLD; face = display_face.c_str(); break;
            case font_kind::subtitle: points = 11; weight = FW_SEMIBOLD; face = display_face.c_str(); break;
            case font_kind::mono: face = L"Cascadia Mono"; if (!font_exists(face)) face = L"Consolas"; break;
            default: break;
            }
            return CreateFontW(-MulDiv(points, static_cast<int>(dpi), 72), 0, 0, 0, weight, FALSE, FALSE, FALSE,
                DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH, face);
        }

        void set_font(HWND control, HWND parent, font_kind kind = font_kind::normal)
        {
            SendMessageW(control, WM_SETFONT, reinterpret_cast<WPARAM>(font(parent, kind)), TRUE);
        }

        HWND create(HWND parent, const wchar_t* cls, const std::wstring& text, DWORD style, int id, DWORD ex = 0)
        {
            return CreateWindowExW(ex, cls, text.c_str(), WS_CHILD | WS_VISIBLE | style, 0, 0, 10, 10, parent,
                reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)), GetModuleHandleW(nullptr), nullptr);
        }

        struct png_source
        {
            const void* data = nullptr;
            DWORD size = 0;
        };

        png_source resource(int id)
        {
            png_source s;
            HMODULE m = GetModuleHandleW(nullptr);
            HRSRC r = FindResourceW(m, MAKEINTRESOURCEW(id), MAKEINTRESOURCEW(10));
            if (!r) return s;
            HGLOBAL g = LoadResource(m, r);
            if (!g) return s;
            s.data = LockResource(g);
            s.size = SizeofResource(m, r);
            return s;
        }

        template <typename T>
        struct com_ptr
        {
            T* p = nullptr;
            ~com_ptr() { if (p) p->Release(); }
            T** operator&() { return &p; }
            T* operator->() { return p; }
            explicit operator bool() const { return p != nullptr; }
        };

        bool decode(int id, IWICImagingFactory* f, IWICBitmapFrameDecode** frame)
        {
            const png_source src = resource(id);
            if (!src.data) return false;
            com_ptr<IWICStream> stream;
            if (FAILED(f->CreateStream(&stream))) return false;
            BYTE* bytes = static_cast<BYTE*>(const_cast<void*>(src.data));
            if (FAILED(stream->InitializeFromMemory(bytes, src.size))) return false;
            com_ptr<IWICBitmapDecoder> decoder;
            if (FAILED(f->CreateDecoderFromStream(stream.p, nullptr, WICDecodeMetadataCacheOnDemand, &decoder)))
                return false;
            return SUCCEEDED(decoder->GetFrame(0, frame));
        }
    }

    void init()
    {
        INITCOMMONCONTROLSEX icc = { sizeof(icc), ICC_STANDARD_CLASSES | ICC_LISTVIEW_CLASSES | ICC_PROGRESS_CLASS |
            ICC_TAB_CLASSES | ICC_LINK_CLASS | ICC_BAR_CLASSES };
        InitCommonControlsEx(&icc);
        CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
    }

    UINT dpi_of(HWND w)
    {
        typedef UINT(WINAPI * get_dpi_t)(HWND);
        static get_dpi_t get_dpi = reinterpret_cast<get_dpi_t>(
            reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"user32.dll"), "GetDpiForWindow")));
        if (get_dpi && w)
        {
            const UINT d = get_dpi(w);
            if (d) return d;
        }
        HDC dc = GetDC(nullptr);
        const int d = GetDeviceCaps(dc, LOGPIXELSX);
        ReleaseDC(nullptr, dc);
        return d > 0 ? static_cast<UINT>(d) : 96;
    }

    int px(HWND w, int size96)
    {
        return MulDiv(size96, static_cast<int>(dpi_of(w)), 96);
    }

    HFONT font(HWND w, font_kind kind)
    {
        const UINT dpi = dpi_of(w);
        std::lock_guard<std::mutex> lock(fonts_m);
        const auto key = std::make_pair(dpi, static_cast<int>(kind));
        auto it = fonts.find(key);
        if (it != fonts.end()) return it->second;
        HFONT f = create_font(dpi, kind);
        fonts[key] = f;
        return f;
    }

    COLORREF color_background() { return RGB(250, 251, 254); }
    COLORREF color_text() { return RGB(23, 26, 43); }
    COLORREF color_faint() { return RGB(110, 116, 140); }
    COLORREF color_accent() { return RGB(61, 107, 255); }

    HBRUSH background_brush()
    {
        static HBRUSH b = CreateSolidBrush(color_background());
        return b;
    }

    SIZE png_size(int resource_id)
    {
        SIZE s = { 0, 0 };
        com_ptr<IWICImagingFactory> f;
        if (FAILED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_IWICImagingFactory,
                reinterpret_cast<void**>(&f))))
            return s;
        com_ptr<IWICBitmapFrameDecode> frame;
        if (!decode(resource_id, f.p, &frame)) return s;
        UINT w = 0, h = 0;
        frame->GetSize(&w, &h);
        s.cx = static_cast<LONG>(w);
        s.cy = static_cast<LONG>(h);
        return s;
    }

    HBITMAP load_png(int resource_id, int width, int height)
    {
        if (width <= 0 || height <= 0) return nullptr;
        com_ptr<IWICImagingFactory> f;
        if (FAILED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_IWICImagingFactory,
                reinterpret_cast<void**>(&f))))
            return nullptr;
        com_ptr<IWICBitmapFrameDecode> frame;
        if (!decode(resource_id, f.p, &frame)) return nullptr;
        com_ptr<IWICBitmapScaler> scaler;
        if (FAILED(f->CreateBitmapScaler(&scaler)) ||
            FAILED(scaler->Initialize(frame.p, static_cast<UINT>(width), static_cast<UINT>(height),
                WICBitmapInterpolationModeFant)))
            return nullptr;
        com_ptr<IWICFormatConverter> conv;
        if (FAILED(f->CreateFormatConverter(&conv)) ||
            FAILED(conv->Initialize(scaler.p, GUID_WICPixelFormat32bppPBGRA, WICBitmapDitherTypeNone, nullptr, 0.0,
                WICBitmapPaletteTypeCustom)))
            return nullptr;
        BITMAPINFO bi = {};
        bi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        bi.bmiHeader.biWidth = width;
        bi.bmiHeader.biHeight = -height;
        bi.bmiHeader.biPlanes = 1;
        bi.bmiHeader.biBitCount = 32;
        bi.bmiHeader.biCompression = BI_RGB;
        void* bits = nullptr;
        HBITMAP bmp = CreateDIBSection(nullptr, &bi, DIB_RGB_COLORS, &bits, nullptr, 0);
        if (!bmp) return nullptr;
        const UINT stride = static_cast<UINT>(width) * 4;
        if (FAILED(conv->CopyPixels(nullptr, stride, stride * static_cast<UINT>(height), static_cast<BYTE*>(bits))))
        {
            DeleteObject(bmp);
            return nullptr;
        }
        return bmp;
    }

    void draw_bitmap(HDC dc, HBITMAP bmp, int x, int y)
    {
        if (!bmp) return;
        BITMAP b;
        GetObjectW(bmp, sizeof(b), &b);
        HDC mem = CreateCompatibleDC(dc);
        HGDIOBJ old = SelectObject(mem, bmp);
        BLENDFUNCTION bf = { AC_SRC_OVER, 0, 255, AC_SRC_ALPHA };
        AlphaBlend(dc, x, y, b.bmWidth, b.bmHeight, mem, 0, 0, b.bmWidth, b.bmHeight, bf);
        SelectObject(mem, old);
        DeleteDC(mem);
    }

    HWND label(HWND parent, const std::wstring& text, int id, DWORD style, font_kind f)
    {
        HWND c = create(parent, L"STATIC", text, SS_LEFT | SS_NOPREFIX | style, id);
        set_font(c, parent, f);
        return c;
    }

    HWND button(HWND parent, const std::wstring& text, int id, DWORD style)
    {
        HWND c = create(parent, L"BUTTON", text, WS_TABSTOP | BS_PUSHBUTTON | style, id);
        set_font(c, parent);
        return c;
    }

    HWND checkbox(HWND parent, const std::wstring& text, int id, bool on)
    {
        HWND c = create(parent, L"BUTTON", text, WS_TABSTOP | BS_AUTOCHECKBOX | BS_MULTILINE, id);
        set_font(c, parent);
        SendMessageW(c, BM_SETCHECK, on ? BST_CHECKED : BST_UNCHECKED, 0);
        return c;
    }

    HWND edit(HWND parent, const std::wstring& text, int id, DWORD style)
    {
        // A multiline edit wraps its lines; a single line scrolls.
        const DWORD scroll = (style & ES_MULTILINE) ? 0 : ES_AUTOHSCROLL;
        HWND c = create(parent, L"EDIT", text, WS_TABSTOP | scroll | style, id, WS_EX_CLIENTEDGE);
        set_font(c, parent);
        return c;
    }

    HWND combo(HWND parent, int id)
    {
        HWND c = create(parent, L"COMBOBOX", L"", WS_TABSTOP | WS_VSCROLL | CBS_DROPDOWNLIST, id);
        set_font(c, parent);
        return c;
    }

    HWND progress(HWND parent, int id)
    {
        HWND c = create(parent, PROGRESS_CLASSW, L"", PBS_SMOOTH, id);
        SendMessageW(c, PBM_SETRANGE32, 0, 1000);
        return c;
    }

    HWND listview(HWND parent, int id, DWORD style)
    {
        const DWORD list_style = WS_TABSTOP | LVS_REPORT | LVS_SHOWSELALWAYS | LVS_SINGLESEL | style;
        HWND c = create(parent, WC_LISTVIEWW, L"", list_style, id, WS_EX_CLIENTEDGE);
        ListView_SetExtendedListViewStyle(c, LVS_EX_FULLROWSELECT | LVS_EX_DOUBLEBUFFER | LVS_EX_LABELTIP);
        set_font(c, parent);
        return c;
    }

    void place(HWND control, int x, int y, int w, int h)
    {
        SetWindowPos(control, nullptr, x, y, w, h, SWP_NOZORDER | SWP_NOACTIVATE);
    }

    int button_width(HWND button, int min_width)
    {
        const std::wstring text = text_of(button);
        HDC dc = GetDC(button);
        HGDIOBJ old = SelectObject(dc, reinterpret_cast<HFONT>(SendMessageW(button, WM_GETFONT, 0, 0)));
        SIZE s = { 0, 0 };
        GetTextExtentPoint32W(dc, text.c_str(), static_cast<int>(text.size()), &s);
        SelectObject(dc, old);
        ReleaseDC(button, dc);
        return std::max(min_width, static_cast<int>(s.cx) + px(button, 32));
    }

    std::wstring text_of(HWND control)
    {
        const int n = GetWindowTextLengthW(control);
        std::wstring s(static_cast<size_t>(n) + 1, L'\0');
        GetWindowTextW(control, &s[0], n + 1);
        s.resize(static_cast<size_t>(n));
        return s;
    }

    void set_text(HWND control, const std::wstring& text)
    {
        if (text_of(control) != text) SetWindowTextW(control, text.c_str());
    }

    bool checked(HWND control)
    {
        return SendMessageW(control, BM_GETCHECK, 0, 0) == BST_CHECKED;
    }

    void enable(HWND control, bool on)
    {
        EnableWindow(control, on ? TRUE : FALSE);
    }

    int message(HWND owner, const std::wstring& text, UINT flags)
    {
        return MessageBoxW(owner, text.c_str(), L"Albireo", flags);
    }

    void open_url(const std::wstring& target)
    {
        ShellExecuteW(nullptr, L"open", target.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
    }

    std::wstring choose_folder(HWND owner, const std::wstring& initial)
    {
        std::wstring out;
        IFileOpenDialog* dlg = nullptr;
        if (FAILED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER, IID_IFileOpenDialog,
                reinterpret_cast<void**>(&dlg))))
            return out;
        DWORD opts = 0;
        dlg->GetOptions(&opts);
        dlg->SetOptions(opts | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST);
        if (!initial.empty())
        {
            IShellItem* item = nullptr;
            if (SUCCEEDED(SHCreateItemFromParsingName(initial.c_str(), nullptr, IID_IShellItem,
                    reinterpret_cast<void**>(&item))))
            {
                dlg->SetFolder(item);
                item->Release();
            }
        }
        if (SUCCEEDED(dlg->Show(owner)))
        {
            IShellItem* item = nullptr;
            if (SUCCEEDED(dlg->GetResult(&item)))
            {
                PWSTR path = nullptr;
                if (SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &path)))
                {
                    out = path;
                    CoTaskMemFree(path);
                }
                item->Release();
            }
        }
        dlg->Release();
        return out;
    }

    void run_async(HWND target, UINT msg, WPARAM tag, std::function<json()> work)
    {
        std::thread([target, msg, tag, work]() {
            json* result = new json();
            try
            {
                *result = work();
            }
            catch (const std::exception& e)
            {
                *result = json::object();
                (*result)["error"] = e.what();
            }
            if (!PostMessageW(target, msg, tag, reinterpret_cast<LPARAM>(result))) delete result;
        }).detach();
    }

    std::string format_size(unsigned long long n)
    {
        // French writes the units in octets.
        const bool fr = current_language() == "fr";
        const char* units[] = { fr ? "o" : "B", fr ? "Ko" : "KB", fr ? "Mo" : "MB", fr ? "Go" : "GB" };
        char buf[64];
        if (n < 1024ull) snprintf(buf, sizeof(buf), "%llu %s", n, units[0]);
        else if (n < (1ull << 20)) snprintf(buf, sizeof(buf), "%.1f %s", n / 1024.0, units[1]);
        else if (n < (1ull << 30)) snprintf(buf, sizeof(buf), "%.1f %s", n / 1048576.0, units[2]);
        else snprintf(buf, sizeof(buf), "%.2f %s", n / 1073741824.0, units[3]);
        std::string s = buf;
        if (fr)
            for (char& c : s)
                if (c == '.') c = ',';
        return s;
    }

}
}
