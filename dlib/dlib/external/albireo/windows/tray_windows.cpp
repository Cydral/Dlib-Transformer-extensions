// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "tray_windows.h"

#include "ui.h"

#include <shellapi.h>
#include <shobjidl.h>

#include <algorithm>
#include <cstring>
#include <cwchar>
#include <map>
#include <memory>
#include <set>
#include <thread>
#include <vector>

#include "../common/platform.h"
#include "../common/version.h"
#include "resource.h"
#include "strings.h"

namespace albireo
{

    namespace
    {
        const wchar_t window_class[] = L"AlbireoToolWindow";
        const UINT WM_RESULT = WM_APP + 20;
        const UINT WM_PROGRESS = WM_APP + 21;

        /*
            A text that the service gives with a code: the text of prefix + code in the
            language of the windows, with its values and its sizes, or the English text of
            the service for a code the windows do not know.
        */
        std::wstring coded_text(const char* prefix, const json& code, const json& args, const json& sizes,
            const std::string& english)
        {
            const std::string key = prefix + code.as_string();
            if (code.as_string().empty() || !has_text(key)) return widen(english);
            std::vector<std::pair<std::string, std::string>> values;
            for (const auto& kv : args.items())
                values.emplace_back(kv.first, kv.second.is_string() ? kv.second.as_string() : kv.second.dump());
            for (const auto& kv : sizes.items())
                values.emplace_back(kv.first, ui::format_size(static_cast<unsigned long long>(kv.second.as_int(0))));
            return tr(key, values);
        }

        // The key of the text of a kind of device: "gpu", "npu", "both" or "cpu".
        const char* device_key(const std::string& kind)
        {
            return kind == "npu" ? "device_npu" : kind == "both" ? "device_both" : kind == "gpu" ? "device_gpu"
                : "device_cpu";
        }

        // The first word all the names share, without its trademark sign: "Intel" for
        // "Intel(R) AI Boost" and "Intel(R) Graphics".
        std::string common_vendor(const json& names)
        {
            std::string word;
            bool first = true;
            for (const auto& n : names.elements())
            {
                const std::string name = n.as_string();
                std::string w = name.substr(0, name.find(' '));
                for (const char* mark : { "(R)", "(TM)", "\xC2\xAE", "\xE2\x84\xA2" })
                    for (size_t p = w.find(mark); p != std::string::npos; p = w.find(mark))
                        w.erase(p, std::strlen(mark));
                if (first) word = w;
                else if (w != word) return std::string();
                first = false;
            }
            return word;
        }

        // A value of the query of a request: the bytes other than the letters, the digits and
        // "-._~" as %XX, so that a file name with &, # or + reaches the service whole.
        std::string query_value(const std::string& s)
        {
            const char hex[] = "0123456789ABCDEF";
            std::string out;
            for (const char ch : s)
            {
                const unsigned char c = static_cast<unsigned char>(ch);
                const bool plain = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')
                    || c == '-' || c == '.' || c == '_' || c == '~';
                if (plain)
                {
                    out += ch;
                    continue;
                }
                out += '%';
                out += hex[c >> 4];
                out += hex[c & 15];
            }
            return out;
        }
    }

    std::wstring error_text(const json& answer)
    {
        return coded_text("err_", answer["code"], answer["args"], answer["sizes"], answer["error"].as_string());
    }

    json failed_answer(const json& answer, const std::string& error)
    {
        json r = json::object();
        r["error"] = error;
        const json& code = answer["code"].is_string() ? answer["code"] : answer["error"]["code"];
        if (code.is_string())
        {
            r["code"] = code;
            if (answer["args"].is_object()) r["args"] = answer["args"];
            if (answer["sizes"].is_object()) r["sizes"] = answer["sizes"];
        }
        else if (!answer.is_object())
        {
            // The service stopped, or listens on another port.
            r["code"] = "service";
            json args = json::object();
            args["e"] = error;
            r["args"] = args;
        }
        return r;
    }

    std::wstring reason_text(const json& engine)
    {
        return coded_text("why_", engine["reason_code"], engine["reason_args"], json(), engine["reason"].as_string());
    }

    std::string runtime_name(const std::string& variant)
    {
        if (variant == "cuda") return "CUDA";
        if (variant.compare(0, 8, "openvino") == 0) return "OpenVINO";
        return variant;
    }

    std::string engine_name(const std::string& variant)
    {
        if (variant == "openvino-avxvnni") return "OpenVINO AVX-VNNI";
        return runtime_name(variant);
    }

    std::wstring choice_label(const json& option)
    {
        const std::string kind = option["kind"].as_string();
        std::string names;
        if (kind == "both")
            names = common_vendor(option["names"]);
        else
            for (const auto& n : option["names"].elements()) names += (names.empty() ? "" : ", ") + n.as_string();
        const std::string runtime = runtime_name(option["variant"].as_string());
        if (names.empty()) return tr("accel_choice_plain", { { "k", tr_utf8(device_key(kind)) }, { "r", runtime } });
        return tr("accel_choice", { { "k", tr_utf8(device_key(kind)) }, { "n", names }, { "r", runtime } });
    }

    std::pair<std::string, std::string> choice_settings(const json& option)
    {
        const std::string variant = option["variant"].as_string();
        return std::make_pair(variant == "cuda" ? std::string("cuda") : std::string("openvino"),
            option["kind"].as_string());
    }

    namespace
    {
        /*
            The rows of a report list, each one named by a key.  refresh() writes in place the
            cells whose text changed, so that a list the windows refresh every few seconds is
            painted where it changes only, and keeps its selection and its scroll position.  A
            new set of keys fills the list again out of sight, the selected row and the row at
            the top found again by their keys.
        */
        struct list_rows
        {
            std::vector<std::string> keys;
            std::vector<std::vector<std::wstring>> texts;
            // True while the rows are written: the notifications of the list are then ignored.
            bool filling = false;
            // Whether the last refresh() filled the list again, and the columns whose texts it wrote.
            bool rebuilt = false;
            std::set<int> changed;

            // The key of the selected row, or an empty string.
            std::string selected(HWND list) const
            {
                const int row = ListView_GetNextItem(list, -1, LVNI_SELECTED);
                return row >= 0 && row < static_cast<int>(keys.size()) ? keys[static_cast<size_t>(row)] : std::string();
            }

            void refresh(HWND list, const std::vector<std::string>& now,
                const std::vector<std::vector<std::wstring>>& cells)
            {
                filling = true;
                rebuilt = false;
                changed.clear();
                if (now != keys)
                {
                    const std::string chosen = selected(list);
                    const int top = ListView_GetTopIndex(list);
                    const std::string top_key = top >= 0 && top < static_cast<int>(keys.size())
                        ? keys[static_cast<size_t>(top)] : std::string();
                    const int left = GetScrollPos(list, SB_HORZ);
                    SendMessageW(list, WM_SETREDRAW, FALSE, 0);
                    ListView_DeleteAllItems(list);
                    keys = now;
                    texts = cells;
                    int first = -1;
                    for (size_t row = 0; row < keys.size(); ++row)
                    {
                        const int r = static_cast<int>(row);
                        std::wstring label = texts[row].empty() ? std::wstring() : texts[row][0];
                        LVITEMW it = {};
                        it.mask = LVIF_TEXT;
                        it.iItem = r;
                        it.pszText = &label[0];
                        ListView_InsertItem(list, &it);
                        for (size_t c = 1; c < texts[row].size(); ++c)
                        {
                            std::wstring t = texts[row][c];
                            ListView_SetItemText(list, r, static_cast<int>(c), &t[0]);
                        }
                        const UINT focus = LVIS_SELECTED | LVIS_FOCUSED;
                        if (!chosen.empty() && keys[row] == chosen) ListView_SetItemState(list, r, focus, focus);
                        if (!top_key.empty() && keys[row] == top_key) first = r;
                    }
                    if (first < 0) first = std::min(top, static_cast<int>(keys.size()) - 1);
                    /*
                        The scroll position can hold while the rows are replaced out of sight: the
                        list scrolls from where the new rows are, the left edge of the first one
                        giving the horizontal position, to the row and the position kept.
                    */
                    RECT item = {};
                    if (ListView_GetItemRect(list, 0, &item, LVIR_BOUNDS))
                    {
                        const int dx = left + static_cast<int>(item.left);
                        const int dy = (std::max(0, first) - ListView_GetTopIndex(list)) * (item.bottom - item.top);
                        if (dx != 0 || dy != 0) ListView_Scroll(list, dx, dy);
                    }
                    SendMessageW(list, WM_SETREDRAW, TRUE, 0);
                    InvalidateRect(list, nullptr, TRUE);
                    rebuilt = true;
                }
                else
                {
                    for (size_t row = 0; row < keys.size(); ++row)
                    {
                        if (texts[row].size() < cells[row].size()) texts[row].resize(cells[row].size());
                        for (size_t c = 0; c < cells[row].size(); ++c)
                        {
                            if (texts[row][c] == cells[row][c]) continue;
                            texts[row][c] = cells[row][c];
                            std::wstring t = cells[row][c];
                            ListView_SetItemText(list, static_cast<int>(row), static_cast<int>(c), &t[0]);
                            changed.insert(static_cast<int>(c));
                        }
                    }
                }
                filling = false;
            }
        };

        // Sets the width of a column of a list when it differs, which repaints its header.
        void set_column_width(HWND list, int column, int width)
        {
            if (ListView_GetColumnWidth(list, column) != width) ListView_SetColumnWidth(list, column, width);
        }

        /*
            The text of the devices of an engine (an element of "engines"): its accelerators
            with their kind, "NPU Intel(R) AI Boost, GPU Intel(R) Graphics", else the
            processor it runs on.
        */
        std::wstring devices_text(const json& engine)
        {
            std::string accelerators, processors;
            for (const auto& d : engine["devices"].elements())
            {
                const std::string type = d["type"].as_string();
                const std::string name = d["name"].str(d["id"].as_string());
                if (type == "gpu" || type == "npu")
                    accelerators += (accelerators.empty() ? "" : ", ") + tr_utf8(device_key(type)) + " " + name;
                else
                    processors += (processors.empty() ? "" : ", ") + name;
            }
            return widen(accelerators.empty() ? processors : accelerators);
        }

        /*
            The text of a measure of the engines: tokens per second in the language of the
            interface ("7.3 tokens/s", "7,3 jetons/s"), with the kind of the device before it when
            the engine offers several devices.
        */
        std::wstring measures_text(const json& engines, const std::string& variant)
        {
            int count = 0;
            for (const auto& o : engines["options"].elements()) count += o["variant"].as_string() == variant;
            std::wstring text;
            for (const auto& o : engines["options"].elements())
            {
                if (o["variant"].as_string() != variant || !o["measure"]["tokens_per_second"].is_number()) continue;
                const std::wstring rate = tr("tokens_s",
                    { { "n", ui::format_decimal(o["measure"]["tokens_per_second"].as_double(), 1) } });
                const std::string kind = o["kind"].as_string();
                const std::wstring prefix = count > 1 && kind != "cpu" ? widen(tr_utf8(device_key(kind))) + L" " : L"";
                text += (text.empty() ? L"" : L", ") + prefix + rate;
            }
            return text;
        }

        /*
            An argument between quotes for a command line, read back as written: the "\"
            before a quote are doubled, so that the folder D:\ does not end as D:".
        */
        std::wstring quoted_argument(const std::wstring& a)
        {
            std::wstring q = L"\"";
            size_t backslashes = 0;
            for (wchar_t c : a)
            {
                if (c == L'\\')
                {
                    ++backslashes;
                    continue;
                }
                q.append(c == L'"' ? backslashes * 2 + 1 : backslashes, L'\\');
                backslashes = 0;
                q += c;
            }
            q.append(backslashes * 2, L'\\');
            return q + L"\"";
        }

        class tool_window
        {
            /*!
                WHAT THIS OBJECT REPRESENTS
                    A top-level window of the tray: its controls, their fonts at the
                    resolution of the monitor, and the results of the requests it sends to
                    the service, which come back as WM_RESULT messages.
            !*/
        public:
            virtual ~tool_window() {}

            HWND hwnd = nullptr;
            service_client client;
            std::map<HWND, ui::font_kind> fonts;

            bool create(const std::wstring& title, int w96, int h96, bool resizable);
            HWND add(HWND control, ui::font_kind f = ui::font_kind::normal)
            {
                fonts[control] = f;
                SendMessageW(control, WM_SETFONT, reinterpret_cast<WPARAM>(ui::font(hwnd, f)), TRUE);
                return control;
            }
            int px(int v) const { return ui::px(hwnd, v); }

            void request(WPARAM tag, std::function<json()> work)
            {
                ui::run_async(hwnd, WM_RESULT, tag, std::move(work));
            }

            virtual void build() = 0;
            virtual void layout(int w, int h) = 0;
            virtual void on_result(WPARAM tag, const json& r) = 0;
            virtual void on_command(int id, int code) = 0;
            virtual LRESULT on_message(UINT msg, WPARAM wp, LPARAM lp, bool& handled)
            {
                handled = false;
                return 0;
            }
            virtual void on_destroy() {}
            SIZE min_size{ 480, 360 };
        };

        std::set<tool_window*> open_windows;

        LRESULT CALLBACK window_proc(HWND w, UINT msg, WPARAM wp, LPARAM lp)
        {
            tool_window* self = reinterpret_cast<tool_window*>(GetWindowLongPtrW(w, GWLP_USERDATA));
            if (msg == WM_NCCREATE)
            {
                self = static_cast<tool_window*>(reinterpret_cast<CREATESTRUCTW*>(lp)->lpCreateParams);
                self->hwnd = w;
                SetWindowLongPtrW(w, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(self));
            }
            if (!self) return DefWindowProcW(w, msg, wp, lp);
            bool handled = false;
            const LRESULT r = self->on_message(msg, wp, lp, handled);
            if (handled) return r;
            switch (msg)
            {
            case WM_CREATE:
                self->build();
                return 0;
            case WM_SIZE:
                self->layout(LOWORD(lp), HIWORD(lp));
                return 0;
            case WM_GETMINMAXINFO:
            {
                MINMAXINFO* mm = reinterpret_cast<MINMAXINFO*>(lp);
                mm->ptMinTrackSize.x = ui::px(w, self->min_size.cx);
                mm->ptMinTrackSize.y = ui::px(w, self->min_size.cy);
                return 0;
            }
            case WM_DPICHANGED:
            {
                const RECT* r2 = reinterpret_cast<RECT*>(lp);
                for (const auto& kv : self->fonts)
                    SendMessageW(kv.first, WM_SETFONT, reinterpret_cast<WPARAM>(ui::font(w, kv.second)), TRUE);
                SetWindowPos(w, nullptr, r2->left, r2->top, r2->right - r2->left, r2->bottom - r2->top,
                    SWP_NOZORDER | SWP_NOACTIVATE);
                return 0;
            }
            case WM_ERASEBKGND:
            {
                RECT rc;
                GetClientRect(w, &rc);
                FillRect(reinterpret_cast<HDC>(wp), &rc, ui::background_brush());
                return 1;
            }
            case WM_CTLCOLORSTATIC:
            case WM_CTLCOLORBTN:
            {
                HDC dc = reinterpret_cast<HDC>(wp);
                SetBkColor(dc, ui::color_background());
                SetTextColor(dc, ui::color_text());
                return reinterpret_cast<LRESULT>(ui::background_brush());
            }
            case WM_COMMAND:
                self->on_command(LOWORD(wp), HIWORD(wp));
                return 0;
            case WM_RESULT:
            {
                json* r3 = reinterpret_cast<json*>(lp);
                self->on_result(wp, *r3);
                delete r3;
                return 0;
            }
            case WM_CLOSE:
                DestroyWindow(w);
                return 0;
            case WM_DESTROY:
                self->on_destroy();
                return 0;
            case WM_NCDESTROY:
                open_windows.erase(self);
                SetWindowLongPtrW(w, GWLP_USERDATA, 0);
                delete self;
                return 0;
            default:
                return DefWindowProcW(w, msg, wp, lp);
            }
        }

        bool tool_window::create(const std::wstring& title, int w96, int h96, bool resizable)
        {
            static bool registered = false;
            HINSTANCE inst = GetModuleHandleW(nullptr);
            if (!registered)
            {
                WNDCLASSEXW wc = { sizeof(wc) };
                wc.style = CS_HREDRAW | CS_VREDRAW;
                wc.lpfnWndProc = window_proc;
                wc.hInstance = inst;
                wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
                wc.hIcon = LoadIconW(inst, MAKEINTRESOURCEW(IDI_APP));
                wc.hIconSm = static_cast<HICON>(LoadImageW(inst, MAKEINTRESOURCEW(IDI_APP), IMAGE_ICON,
                    GetSystemMetrics(SM_CXSMICON), GetSystemMetrics(SM_CYSMICON), 0));
                wc.lpszClassName = window_class;
                RegisterClassExW(&wc);
                registered = true;
            }
            const DWORD style = resizable ? WS_OVERLAPPEDWINDOW
                : (WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX);
            // Created hidden at its size for the resolution of the primary monitor, then
            // centered on the work area.
            const int dpi = static_cast<int>(ui::dpi_of(nullptr));
            RECT r = { 0, 0, MulDiv(w96, dpi, 96), MulDiv(h96, dpi, 96) };
            AdjustWindowRectEx(&r, style, FALSE, WS_EX_CONTROLPARENT);
            RECT work;
            SystemParametersInfoW(SPI_GETWORKAREA, 0, &work, 0);
            const int ww = r.right - r.left, wh = r.bottom - r.top;
            const int x = work.left + std::max(0L, (work.right - work.left - ww) / 2);
            const int y = work.top + std::max(0L, (work.bottom - work.top - wh) / 2);
            if (!CreateWindowExW(WS_EX_CONTROLPARENT, window_class, title.c_str(), style, x, y, ww, wh,
                    nullptr, nullptr, inst, this))
                return false;
            open_windows.insert(this);
            ShowWindow(hwnd, SW_SHOWNORMAL);
            SetForegroundWindow(hwnd);
            return true;
        }

        template <typename T>
        T* find_window()
        {
            for (auto* w : open_windows)
                if (T* t = dynamic_cast<T*>(w)) return t;
            return nullptr;
        }

        std::wstring role_name(const std::string& role)
        {
            return tr(("role_" + role).c_str());
        }

        /*
            The order of two texts for the user: the collation of the locale of the user, without
            regard to case, the numbers in the texts compared by their values ("Qwen3 8B" before
            "Qwen3 30B A3B").  Negative, zero or positive.
        */
        int compare_text(const std::wstring& a, const std::wstring& b)
        {
            const int r = CompareStringEx(LOCALE_NAME_USER_DEFAULT, LINGUISTIC_IGNORECASE | SORT_DIGITSASNUMBERS,
                a.c_str(), static_cast<int>(a.size()), b.c_str(), static_cast<int>(b.size()), nullptr, nullptr, 0);
            return r ? r - CSTR_EQUAL : a.compare(b);
        }

        // The columns of the list of the models: the key of their title, and the name that keeps the
        // order of the list for the next start.
        enum { COL_MODEL, COL_PUBLISHER, COL_ROLES, COL_INPUTS, COL_SIZE, COL_STATE, COL_ACTIVE, COLUMN_COUNT };
        const char* const column_titles[COLUMN_COUNT] = { "col_model", "col_publisher", "col_roles", "col_inputs",
            "col_size", "col_state", "col_active" };
        const wchar_t* const column_names[COLUMN_COUNT] = { L"model", L"publisher", L"functions", L"inputs", L"size",
            L"state", L"active" };

        /*
            The order of the list of the models: the column the user sorted it by, -1 for the order
            of the catalog, and its direction.  It holds while the tray runs; the values
            ModelsSortColumn and ModelsSortDescending of HKCU\Software\Albireo keep it for the next
            start.
        */
        struct list_order
        {
            int column = -1;
            bool descending = false;
        };

        const wchar_t user_key[] = L"Software\\Albireo";

        list_order& models_order()
        {
            static list_order order;
            static bool read = false;
            if (read) return order;
            read = true;
            wchar_t name[32] = {};
            DWORD size = sizeof(name);
            if (RegGetValueW(HKEY_CURRENT_USER, user_key, L"ModelsSortColumn", RRF_RT_REG_SZ, nullptr, name, &size)
                    != ERROR_SUCCESS)
                return order;
            for (int c = 0; c < COLUMN_COUNT; ++c)
                if (std::wcscmp(name, column_names[c]) == 0) order.column = c;
            DWORD down = 0;
            size = sizeof(down);
            if (RegGetValueW(HKEY_CURRENT_USER, user_key, L"ModelsSortDescending", RRF_RT_REG_DWORD, nullptr, &down,
                    &size) == ERROR_SUCCESS)
                order.descending = order.column >= 0 && down != 0;
            return order;
        }

        void save_models_order(const list_order& order)
        {
            const wchar_t* name = column_names[order.column];
            const DWORD down = order.descending ? 1 : 0;
            RegSetKeyValueW(HKEY_CURRENT_USER, user_key, L"ModelsSortColumn", REG_SZ, name,
                static_cast<DWORD>((std::wcslen(name) + 1) * sizeof(wchar_t)));
            RegSetKeyValueW(HKEY_CURRENT_USER, user_key, L"ModelsSortDescending", REG_DWORD, &down, sizeof(down));
        }

        // The models window.
        enum
        {
            IDC_INTRO = 1000, IDC_SPACE, IDC_LIST, IDC_DETAILS, IDC_PROGRESS, IDC_STATUS, IDC_DOWNLOAD, IDC_ACTIVATE,
            IDC_DEACTIVATE, IDC_VERIFY, IDC_DELETE, IDC_IMPORT, IDC_REFRESH, IDC_CLOSE, IDC_TABS,
            IDM_ROLE_GENERAL = 1100, IDM_ROLE_CODE, IDM_ROLE_SECURITY, IDM_ROLE_EMBEDDING, IDM_IMPORT_FILES,
            IDM_IMPORT_FOLDER,
            IDC_R_INTRO = 1200, IDC_R_ENABLED, IDC_R_PROVIDER_LABEL, IDC_R_PROVIDER, IDC_R_URL_LABEL, IDC_R_URL,
            IDC_R_AUTH_LABEL, IDC_R_AUTH, IDC_R_USER_LABEL, IDC_R_USER, IDC_R_SECRET_LABEL, IDC_R_SECRET,
            IDC_R_MODEL_LABEL, IDC_R_MODEL_LIST, IDC_R_MODEL_EDIT, IDC_R_TRAITS, IDC_R_CONTEXT_LABEL, IDC_R_CONTEXT,
            IDC_R_OUTPUT_LABEL, IDC_R_OUTPUT, IDC_R_IMAGES, IDC_R_REASONING, IDC_R_PRIVACY, IDC_R_PSEUDONYMIZE,
            IDC_R_TAGS, IDC_R_PSEUDALIS, IDC_R_TEST, IDC_R_SAVE, IDC_R_STATUS, IDC_R_KEEP, IDC_R_PSEUDALIS_ICON,
            IDC_R_PSEUDALIS_ABOUT, IDC_R_OFFLINE
        };
        enum
        {
            R_LIST = 1, R_ACTION, R_VERIFY, R_IMPORT, R_UPLOAD_DONE, R_CATALOG, R_RELAY, R_RELAY_TEST, R_RELAY_SAVE,
            R_RELAY_MODELS
        };
        // The providers of the relay, in the order of the service (relay_providers of host/settings.h), and
        // its authentications, each in the order of its list.
        const char* const relay_providers[4] = { "openai", "chatgpt", "claude", "ollama" };
        const char* const relay_auths[3] = { "key", "basic", "none" };

        // The text of an edit field without the spaces around it, in UTF-8.
        std::string field_text(HWND control)
        {
            std::string t = narrow(ui::text_of(control));
            const size_t b = t.find_first_not_of(" \t");
            if (b == std::string::npos) return std::string();
            return t.substr(b, t.find_last_not_of(" \t") - b + 1);
        }

        // The index of the selection of a list, 0 when none is selected.
        int selection(HWND combo)
        {
            const LRESULT i = SendMessageW(combo, CB_GETCURSEL, 0, 0);
            return i == CB_ERR ? 0 : static_cast<int>(i);
        }

        class models_window : public tool_window
        {
        public:
            HWND intro, space, list, details, progress, status, download, activate, deactivate, verify, remove, import,
                refresh, close;
            json overview;
            // The rows of the list, one per model, named by its identifier.
            list_rows rows;
            std::string selected;
            bool busy = false;
            bool uploading = false;
            bool showing_download = false;   // the status line shows the progress of a download
            // Whether the columns took the widths of their texts, and the width the texts of the last column need.
            bool fitted = false;
            int last_fit = 0;
            /*
                The tabs: the models of this computer (page 0) and the relay to an external engine
                (page 1).  The bar of progress shows on the first page, while a download or an
                import wants it.
            */
            HWND tabs = nullptr;
            int page = 0;
            bool progress_wanted = false;
            struct relay_page
            {
                HWND intro = nullptr, enabled = nullptr, provider_label = nullptr, provider = nullptr;
                HWND url_label = nullptr, url = nullptr, auth_label = nullptr, auth = nullptr, user_label = nullptr;
                HWND user = nullptr, secret_label = nullptr, secret = nullptr, model_label = nullptr;
                HWND model_list = nullptr, model_edit = nullptr, traits = nullptr, context_label = nullptr;
                HWND context = nullptr, output_label = nullptr, output = nullptr, images = nullptr;
                HWND reasoning = nullptr, privacy = nullptr, pseudonymize = nullptr, tags = nullptr;
                HWND pseudalis = nullptr, test = nullptr, save = nullptr, status = nullptr;
                HWND keep = nullptr, pseudalis_icon = nullptr, pseudalis_about = nullptr, offline_note = nullptr;
            } relay;
            // The providers and their models, the state of Pseudalis, and the relay of the settings.
            json relay_info, relay_saved;
            /*
                The provider whose profile the form shows, and what the form held for each provider
                it showed since the settings were read: a provider chosen again finds what was typed
                for it, else what the settings keep for it, its key masked.
            */
            std::string relay_shown = relay_providers[0];
            std::map<std::string, json> relay_drafts;
            // The models of the list of the relay, in its order.
            std::vector<std::string> relay_models;
            bool relay_busy = false;
            // The fields whose models the latest listing asked for, and the number of that listing.
            std::string relay_list_key;
            long long relay_list_seq = 0;

            // High enough for the longest page of the relay: offline, a user, an address in clear and the boxes.
            models_window() { min_size = SIZE{ 760, 680 }; }

            // The icon of Pseudalis, which its control does not free.
            void on_destroy() override { ui::set_icon(relay.pseudalis_icon, IDI_PSEUDALIS, 0); }

            void build() override
            {
                tabs = add(ui::tabs(hwnd, IDC_TABS, { tr("tab_models"), tr("tab_relay") }));
                intro = add(ui::label(hwnd, tr("models_intro"), IDC_INTRO, SS_LEFT), ui::font_kind::normal);
                space = add(ui::label(hwnd, L"", IDC_SPACE, SS_RIGHT), ui::font_kind::note);
                list = add(ui::listview(hwnd, IDC_LIST));
                // The widths until the first rows come; the columns then take the widths of their texts
                // (fit_columns), and the last one the width the others leave (layout).
                const int widths[COLUMN_COUNT] = { 170, 120, 130, 90, 75, 130, 140 };
                for (int i = 0; i < COLUMN_COUNT; ++i)
                {
                    std::wstring text = tr(column_titles[i]);
                    LVCOLUMNW c = {};
                    c.mask = LVCF_TEXT | LVCF_WIDTH | LVCF_SUBITEM;
                    c.pszText = &text[0];
                    c.cx = px(widths[i]);
                    c.iSubItem = i;
                    ListView_InsertColumn(list, i, &c);
                }
                last_fit = px(widths[COLUMN_COUNT - 1]);
                show_sort_arrow();
                const DWORD reader = ES_MULTILINE | ES_READONLY | WS_VSCROLL | ES_AUTOVSCROLL;
                details = add(ui::edit(hwnd, L"", IDC_DETAILS, reader));
                progress = add(ui::progress(hwnd, IDC_PROGRESS));
                status = add(ui::label(hwnd, L"", IDC_STATUS), ui::font_kind::note);
                download = add(ui::button(hwnd, tr("btn_download"), IDC_DOWNLOAD));
                activate = add(ui::button(hwnd, tr("btn_activate") + L"…", IDC_ACTIVATE));
                deactivate = add(ui::button(hwnd, tr("btn_deactivate"), IDC_DEACTIVATE));
                verify = add(ui::button(hwnd, tr("btn_verify"), IDC_VERIFY));
                remove = add(ui::button(hwnd, tr("btn_delete"), IDC_DELETE));
                import = add(ui::button(hwnd, tr("btn_import"), IDC_IMPORT));
                refresh = add(ui::button(hwnd, tr("btn_refresh"), IDC_REFRESH));
                close = add(ui::button(hwnd, tr("btn_close"), IDC_CLOSE));
                show_progress(false);
                build_relay();
                SetTimer(hwnd, 1, 1000, nullptr);
                update_buttons();
                load();
                load_relay();
            }

            // The controls of the page of the models, the bar of progress and the Close button apart.
            std::vector<HWND> model_controls() const
            {
                return { intro, space, list, details, status, download, activate, deactivate, verify, remove, import,
                    refresh };
            }

            std::vector<HWND> relay_controls() const
            {
                return { relay.intro, relay.enabled, relay.provider_label, relay.provider, relay.url_label, relay.url,
                    relay.auth_label, relay.auth, relay.user_label, relay.user, relay.secret_label, relay.secret,
                    relay.model_label, relay.model_list, relay.model_edit, relay.traits, relay.context_label,
                    relay.context, relay.output_label, relay.output, relay.images, relay.reasoning, relay.privacy,
                    relay.pseudonymize, relay.tags, relay.keep, relay.pseudalis_icon, relay.pseudalis,
                    relay.pseudalis_about, relay.offline_note, relay.test, relay.save, relay.status };
            }

            void show_progress(bool on)
            {
                progress_wanted = on;
                ShowWindow(progress, on && page == 0 ? SW_SHOW : SW_HIDE);
            }

            void show_page(int p)
            {
                page = p;
                for (HWND c : model_controls()) ShowWindow(c, page == 0 ? SW_SHOW : SW_HIDE);
                ShowWindow(progress, page == 0 && progress_wanted ? SW_SHOW : SW_HIDE);
                // The controls of the relay show by its layout, which hides those its provider does not take.
                if (page != 0) relay_update();
                else
                    for (HWND c : relay_controls()) ShowWindow(c, SW_HIDE);
                RECT rc;
                GetClientRect(hwnd, &rc);
                layout(rc.right, rc.bottom);
            }

            void build_relay()
            {
                relay.intro = add(ui::label(hwnd, tr("relay_intro"), IDC_R_INTRO, SS_LEFT));
                relay.enabled = add(ui::checkbox(hwnd, tr("relay_enabled"), IDC_R_ENABLED, false));
                relay.offline_note = add(ui::label(hwnd, tr("relay_offline_note"), IDC_R_OFFLINE, SS_LEFT),
                    ui::font_kind::note);
                relay.provider_label = add(ui::label(hwnd, tr("relay_provider"), IDC_R_PROVIDER_LABEL));
                relay.provider = add(ui::combo(hwnd, IDC_R_PROVIDER));
                for (const char* id : relay_providers)
                    SendMessageW(relay.provider, CB_ADDSTRING, 0,
                        reinterpret_cast<LPARAM>(tr((std::string("relay_provider_") + id).c_str()).c_str()));
                SendMessageW(relay.provider, CB_SETCURSEL, 0, 0);
                relay.url_label = add(ui::label(hwnd, tr("relay_url"), IDC_R_URL_LABEL));
                relay.url = add(ui::edit(hwnd, L"", IDC_R_URL));
                relay.auth_label = add(ui::label(hwnd, tr("relay_auth"), IDC_R_AUTH_LABEL));
                relay.auth = add(ui::combo(hwnd, IDC_R_AUTH));
                for (const char* k : { "relay_auth_key", "relay_auth_basic", "relay_auth_none" })
                    SendMessageW(relay.auth, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(tr(k).c_str()));
                SendMessageW(relay.auth, CB_SETCURSEL, 0, 0);
                relay.user_label = add(ui::label(hwnd, tr("relay_user"), IDC_R_USER_LABEL));
                relay.user = add(ui::edit(hwnd, L"", IDC_R_USER));
                relay.secret_label = add(ui::label(hwnd, tr("relay_key"), IDC_R_SECRET_LABEL));
                relay.secret = add(ui::edit(hwnd, L"", IDC_R_SECRET, ES_PASSWORD));
                relay.model_label = add(ui::label(hwnd, tr("relay_model"), IDC_R_MODEL_LABEL));
                relay.model_list = add(ui::combo(hwnd, IDC_R_MODEL_LIST));
                relay.model_edit = add(ui::combo(hwnd, IDC_R_MODEL_EDIT, true));
                relay.traits = add(ui::label(hwnd, L"", IDC_R_TRAITS, SS_LEFT), ui::font_kind::note);
                relay.context_label = add(ui::label(hwnd, tr("relay_context"), IDC_R_CONTEXT_LABEL));
                relay.context = add(ui::edit(hwnd, L"131072", IDC_R_CONTEXT, ES_NUMBER));
                relay.output_label = add(ui::label(hwnd, tr("relay_output"), IDC_R_OUTPUT_LABEL));
                relay.output = add(ui::edit(hwnd, L"16384", IDC_R_OUTPUT, ES_NUMBER));
                relay.images = add(ui::checkbox(hwnd, tr("relay_images"), IDC_R_IMAGES, false));
                relay.reasoning = add(ui::checkbox(hwnd, tr("relay_reasoning"), IDC_R_REASONING, false));
                relay.privacy = add(ui::label(hwnd, tr("relay_privacy"), IDC_R_PRIVACY, SS_LEFT));
                relay.pseudonymize = add(ui::checkbox(hwnd, tr("relay_pseudonymize"), IDC_R_PSEUDONYMIZE, false));
                relay.tags = add(ui::checkbox(hwnd, tr("relay_tags"), IDC_R_TAGS, false));
                relay.keep = add(ui::checkbox(hwnd, tr("relay_keep"), IDC_R_KEEP, false));
                relay.pseudalis_icon = ui::icon(hwnd, IDI_PSEUDALIS, px(20), IDC_R_PSEUDALIS_ICON);
                relay.pseudalis = add(ui::label(hwnd, L"", IDC_R_PSEUDALIS, SS_LEFT), ui::font_kind::note);
                relay.pseudalis_about = add(ui::label(hwnd, tr("relay_pseudalis_about"), IDC_R_PSEUDALIS_ABOUT,
                    SS_LEFT), ui::font_kind::note);
                relay.test = add(ui::button(hwnd, tr("btn_test"), IDC_R_TEST));
                relay.save = add(ui::button(hwnd, tr("btn_save"), IDC_R_SAVE));
                relay.status = add(ui::label(hwnd, L"", IDC_R_STATUS), ui::font_kind::note);
                for (HWND c : relay_controls()) ShowWindow(c, SW_HIDE);
            }

            std::string relay_provider() const
            {
                return relay_providers[std::min(selection(relay.provider), 3)];
            }

            // The authentication of a provider: the one chosen for an engine of OpenAI, a key for the others.
            std::string relay_auth(const std::string& provider) const
            {
                return provider == "openai" ? relay_auths[std::min(selection(relay.auth), 2)] : "key";
            }

            std::string relay_auth() const { return relay_auth(relay_provider()); }

            const json* provider_info(const std::string& id) const
            {
                for (const auto& p : relay_info["providers"].elements())
                    if (p["id"].as_string() == id) return &p;
                return nullptr;
            }

            // The model chosen for a provider: in the list of its models, or typed for an engine of OpenAI.
            std::string relay_model(const std::string& provider) const
            {
                if (provider == "openai") return field_text(relay.model_edit);
                const LRESULT i = SendMessageW(relay.model_list, CB_GETCURSEL, 0, 0);
                return i >= 0 && static_cast<size_t>(i) < relay_models.size() ? relay_models[static_cast<size_t>(i)]
                    : std::string();
            }

            std::string relay_model() const { return relay_model(relay_provider()); }

            /*
                The profile of a provider as the form holds it: its address, its model, its
                authentication and its secret, with what its model offers for an engine of OpenAI.
            */
            json relay_profile(const std::string& provider) const
            {
                json p = json::object();
                const std::string auth = relay_auth(provider);
                p["url"] = provider == "openai" ? field_text(relay.url) : std::string();
                p["model"] = relay_model(provider);
                p["auth"] = auth;
                p["user"] = auth == "basic" ? field_text(relay.user) : std::string();
                p["secret"] = auth == "none" ? std::string() : field_text(relay.secret);
                if (provider == "openai")
                {
                    const long long context = std::wcstoll(ui::text_of(relay.context).c_str(), nullptr, 10);
                    const long long output = std::wcstoll(ui::text_of(relay.output).c_str(), nullptr, 10);
                    p["context"] = std::min<long long>(std::max<long long>(context, 1024), 1LL << 24);
                    p["max_output"] = std::min<long long>(std::max<long long>(output, 16), 1LL << 20);
                    p["images"] = ui::checked(relay.images);
                    p["reasoning"] = ui::checked(relay.reasoning);
                }
                return p;
            }

            // The profile that the settings keep for a provider: the one of the relay, or one saved beside it.
            const json& stored_relay_profile(const std::string& provider) const
            {
                return provider == relay_saved["provider"].str(relay_providers[0]) ? relay_saved
                    : relay_saved["saved"][provider];
            }

            // Shows a profile in the form; one that holds nothing gives the fields their defaults.
            void show_relay_profile(const json& p)
            {
                const std::string auth = p["auth"].str("key");
                for (int i = 0; i < 3; ++i)
                    if (auth == relay_auths[i]) SendMessageW(relay.auth, CB_SETCURSEL, i, 0);
                ui::set_text(relay.url, widen(p["url"].as_string()));
                ui::set_text(relay.user, widen(p["user"].as_string()));
                ui::set_text(relay.secret, widen(p["secret"].as_string()));
                ui::set_text(relay.context, std::to_wstring(p["context"].as_int(131072)));
                ui::set_text(relay.output, std::to_wstring(p["max_output"].as_int(16384)));
                const std::pair<HWND, const char*> boxes[] = { { relay.images, "images" },
                    { relay.reasoning, "reasoning" } };
                for (const auto& b : boxes)
                    SendMessageW(b.first, BM_SETCHECK, p[b.second].as_bool(false) ? BST_CHECKED : BST_UNCHECKED, 0);
                fill_relay_models(p["model"].as_string());
            }

            void load_relay()
            {
                service_client c = client;
                request(R_RELAY, [c]() {
                    json r = json::object(), info, set;
                    std::string error;
                    if (!c.get("/albireo/v1/relay", info, error)) return failed_answer(info, error);
                    if (!c.get("/albireo/v1/settings", set, error)) return failed_answer(set, error);
                    r["info"] = info;
                    r["relay"] = set["relay"];
                    return r;
                });
            }

            void fill_relay()
            {
                const json& r = relay_saved;
                relay_drafts.clear();
                SendMessageW(relay.enabled, BM_SETCHECK, r["enabled"].as_bool(false) ? BST_CHECKED : BST_UNCHECKED, 0);
                relay_shown = relay_providers[0];
                for (int i = 0; i < 4; ++i)
                    if (r["provider"].as_string() == relay_providers[i]) relay_shown = relay_providers[i];
                for (int i = 0; i < 4; ++i)
                    if (relay_shown == relay_providers[i]) SendMessageW(relay.provider, CB_SETCURSEL, i, 0);
                const std::pair<HWND, const char*> boxes[] = { { relay.pseudonymize, "pseudonymize" },
                    { relay.tags, "tags" }, { relay.keep, "keep_substitutes" } };
                for (const auto& b : boxes)
                    SendMessageW(b.first, BM_SETCHECK, r[b.second].as_bool(false) ? BST_CHECKED : BST_UNCHECKED, 0);
                show_relay_profile(r);
                list_relay_models();
            }

            // The models of the provider chosen, model selected when its provider offers it.
            void fill_relay_models(const std::string& model)
            {
                SendMessageW(relay.model_list, CB_RESETCONTENT, 0, 0);
                relay_models.clear();
                const json* p = provider_info(relay_provider());
                int chosen = 0;
                if (p)
                    for (const auto& m : (*p)["models"].elements())
                    {
                        if (m["id"].as_string() == model) chosen = static_cast<int>(relay_models.size());
                        relay_models.push_back(m["id"].as_string());
                        SendMessageW(relay.model_list, CB_ADDSTRING, 0,
                            reinterpret_cast<LPARAM>(widen(m["name"].as_string()).c_str()));
                    }
                SendMessageW(relay.model_list, CB_SETCURSEL,
                    relay_models.empty() ? static_cast<WPARAM>(-1) : static_cast<WPARAM>(chosen), 0);
                if (relay_provider() == "openai") ui::set_text(relay.model_edit, widen(model));
                relay_update();
            }

            // What the model chosen offers, as the service knows it.
            std::wstring relay_traits() const
            {
                const json* p = provider_info(relay_provider());
                if (!p) return L"";
                for (const auto& m : (*p)["models"].elements())
                {
                    if (m["id"].as_string() != relay_model()) continue;
                    std::string efforts;
                    for (const auto& e : m["efforts"].elements())
                        efforts += (efforts.empty() ? "" : ", ") + e.as_string();
                    const std::string thinking = m["thinking"].as_string();
                    std::wstring reasoning = thinking == "none" ? tr("relay_no")
                        : efforts.empty() ? tr("relay_yes") : widen(efforts);
                    std::wstring text = tr("relay_traits", { { "c", ui::format_decimal(m["context"].as_double(), 0) },
                        { "o", ui::format_decimal(m["max_output"].as_double(), 0) } })
                        + L" · " + tr("relay_traits_images") + L" " + (m["images"].as_bool(false) ? tr("relay_yes")
                        : tr("relay_no")) + L" · " + tr("relay_traits_reasoning") + L" " + reasoning;
                    // What a request costs with the providers that bill each of them.
                    if (relay_provider() == "ollama") text += L"\n" + tr("relay_ollama_note");
                    if (relay_provider() == "chatgpt") text += L"\n" + tr("relay_chatgpt_note");
                    return text;
                }
                return L"";
            }

            /*
                Whether the key or the password of an engine of OpenAI would cross the network in clear: an
                address in http:// that is not the computer itself.
            */
            bool relay_in_clear() const
            {
                const std::string url = field_text(relay.url);
                if (relay_provider() != "openai" || relay_auth() == "none" || url.compare(0, 7, "http://") != 0)
                    return false;
                const std::string host = url.substr(7, url.find_first_of(":/", 7) - 7);
                return host != "localhost" && host != "127.0.0.1" && url.compare(7, 5, "[::1]") != 0;
            }

            void relay_update()
            {
                const bool openai = relay_provider() == "openai";
                const std::string auth = relay_auth();
                const bool in_clear = relay_in_clear();
                ui::set_text(relay.secret_label, auth == "basic" ? tr("relay_password") : tr("relay_key"));
                ui::set_text(relay.traits, openai ? (in_clear ? tr("relay_http_warning") : L"") : relay_traits());
                ui::enable(relay.tags, ui::checked(relay.pseudonymize));
                ui::enable(relay.keep, ui::checked(relay.pseudonymize));
                // The state of Pseudalis once the service gave it.
                const json& p = relay_info["pseudalis"];
                const std::string code = p["code"].as_string();
                const bool known = code == "not_installed" || code == "no_authority" || code == "unprotected";
                const std::wstring missing = known ? tr(("relay_pseudalis_" + code).c_str())
                    : tr("relay_pseudalis_missing", { { "r", p["reason"].as_string() } });
                ui::set_text(relay.pseudalis, !p.is_object() ? std::wstring()
                    : p["found"].as_bool(false) ? tr("relay_pseudalis_found") : missing);
                for (HWND c : { relay.test, relay.save, relay.enabled, relay.provider }) ui::enable(c, !relay_busy);
                if (page != 1) return;
                const bool basic = openai && auth == "basic", keyed = auth != "none";
                // The offline mode keeps the engines of this computer and of its local network alone.
                const bool offline = relay_info["relay"]["offline"].as_bool(false);
                const std::pair<HWND, bool> shown[] = { { relay.intro, true }, { relay.enabled, true },
                    { relay.offline_note, offline },
                    { relay.provider_label, true }, { relay.provider, true }, { relay.url_label, openai },
                    { relay.url, openai }, { relay.auth_label, openai }, { relay.auth, openai },
                    { relay.user_label, basic }, { relay.user, basic }, { relay.secret_label, keyed },
                    { relay.secret, keyed }, { relay.model_label, true }, { relay.model_list, !openai },
                    { relay.model_edit, openai }, { relay.traits, !openai || in_clear },
                    { relay.context_label, openai },
                    { relay.context, openai }, { relay.output_label, openai }, { relay.output, openai },
                    { relay.images, openai }, { relay.reasoning, openai }, { relay.privacy, true },
                    { relay.pseudonymize, true }, { relay.tags, true }, { relay.keep, true },
                    { relay.pseudalis_icon, true }, { relay.pseudalis, true }, { relay.pseudalis_about, true },
                    { relay.test, true }, { relay.save, true }, { relay.status, true } };
                for (const auto& c : shown) ShowWindow(c.first, c.second ? SW_SHOW : SW_HIDE);
                RECT rc;
                GetClientRect(hwnd, &rc);
                layout(rc.right, rc.bottom);
            }

            // The relay as the form gives it, for its test and its saving: its provider with the profile shown.
            json relay_form() const
            {
                json r = json::object();
                const std::string provider = relay_provider();
                r["enabled"] = ui::checked(relay.enabled);
                r["provider"] = provider;
                const json profile = relay_profile(provider);
                for (const auto& kv : profile.items()) r[kv.first] = kv.second;
                r["pseudonymize"] = ui::checked(relay.pseudonymize);
                r["tags"] = ui::checked(relay.tags);
                r["keep_substitutes"] = ui::checked(relay.keep);
                return r;
            }

            /*
                The fields that the models of an engine of OpenAI depend on, once they let the service
                ask for them: an address of the web and the secret its authentication takes.  Empty
                otherwise.
            */
            std::string relay_listing_key() const
            {
                if (relay_provider() != "openai") return std::string();
                const std::string url = field_text(relay.url);
                const std::string auth = relay_auth();
                const std::string user = field_text(relay.user);
                const std::string secret = field_text(relay.secret);
                const size_t scheme = url.compare(0, 7, "http://") == 0 ? 7
                    : url.compare(0, 8, "https://") == 0 ? 8 : 0;
                if (scheme == 0 || url.size() <= scheme || url.find(' ') != std::string::npos) return std::string();
                if ((auth != "none" && secret.empty()) || (auth == "basic" && user.empty())) return std::string();
                return url + '\n' + auth + '\n' + user + '\n' + secret;
            }

            /*
                Asks the engine of OpenAI of the fields for its models, which fill the proposals of
                the field of the model: at the opening, once the field of its address, of its user
                or of its key is left, and when the list opens, never while a field is typed, so
                that no key goes to an address typed halfway.  A listing that fails says nothing,
                the field staying as typed.
            */
            void list_relay_models()
            {
                const std::string key = relay_listing_key();
                if (key.empty() || key == relay_list_key) return;
                relay_list_key = key;
                const long long seq = ++relay_list_seq;
                service_client c = client;
                const json form = relay_form();
                request(R_RELAY_MODELS, [c, form, seq]() {
                    json r;
                    std::string error;
                    if (!c.send("POST", "/albireo/v1/relay/test", form, r, error, 40) || !r.is_object())
                        r = json::object();
                    r["seq"] = seq;
                    return r;
                });
            }

            // The models of an engine of OpenAI as the proposals of its field, whose text and selection stay.
            void fill_relay_proposals(const json& models)
            {
                const std::wstring typed = ui::text_of(relay.model_edit);
                const LRESULT selected = SendMessageW(relay.model_edit, CB_GETEDITSEL, 0, 0);
                SendMessageW(relay.model_edit, CB_RESETCONTENT, 0, 0);
                for (const auto& m : models.elements())
                    if (m.is_string())
                        SendMessageW(relay.model_edit, CB_ADDSTRING, 0,
                            reinterpret_cast<LPARAM>(widen(m.as_string()).c_str()));
                ui::set_text(relay.model_edit, typed);
                SendMessageW(relay.model_edit, CB_SETEDITSEL, 0, MAKELPARAM(LOWORD(selected), HIWORD(selected)));
            }

            // The failure of a test of the relay in the language of the tray, the message of the engine after it.
            std::wstring relay_failure(const json& r) const
            {
                const std::string code = r["code"].as_string();
                if (code.empty() || !has_text("err_" + code))
                    return tr("relay_test_failed", { { "e", r["error"].as_string() } });
                std::wstring text = error_text(r);
                const std::string detail = r["detail"].as_string();
                if (!detail.empty()) text += L" " + tr("relay_engine_says", { { "d", detail } });
                return text;
            }

            void relay_send(WPARAM tag)
            {
                relay_busy = true;
                ui::set_text(relay.status, tag == R_RELAY_TEST ? tr("relay_testing") : L"");
                relay_update();
                service_client c = client;
                const json form = relay_form();
                request(tag, [c, form, tag]() {
                    json r;
                    std::string error;
                    json changes = json::object();
                    changes["relay"] = form;
                    const bool sent = tag == R_RELAY_TEST ? c.send("POST", "/albireo/v1/relay/test", form, r, error, 60)
                        : c.send("PUT", "/albireo/v1/settings", changes, r, error);
                    return sent ? r : failed_answer(r, error);
                });
            }

            void relay_result(WPARAM tag, const json& r)
            {
                if (tag == R_RELAY)
                {
                    if (r.contains("error") && !r.contains("info")) return;
                    relay_info = r["info"];
                    relay_saved = r["relay"];
                    fill_relay();
                    return;
                }
                relay_busy = false;
                if (r.contains("error") && r["error"].is_string() && !r.contains("ok"))
                    ui::set_text(relay.status, error_text(r));
                else if (tag == R_RELAY_TEST && r["ok"].as_bool(false))
                {
                    std::wstring text = tr("relay_test_ok", { { "n", std::to_string(r["models"].size()) } });
                    // The base where the interface answers, when the service found it beside the address typed.
                    if (r["url"].is_string()) text += L" " + tr("relay_test_base", { { "u", r["url"].as_string() } });
                    // The model chosen, when the engine does not list it: a name mistyped, or a model not reached.
                    const std::string model = relay_model();
                    bool listed = model.empty() || r["models"].size() == 0;
                    for (const auto& m : r["models"].elements()) listed = listed || m.as_string() == model;
                    if (!listed) text += L" " + tr("relay_model_unlisted", { { "m", model } });
                    ui::set_text(relay.status, text);
                    // The models of an engine of OpenAI fill the proposals of its field.
                    if (relay_provider() == "openai") fill_relay_proposals(r["models"]);
                }
                else if (tag == R_RELAY_TEST)
                {
                    ui::set_text(relay.status, relay_failure(r));
                }
                else
                {
                    ui::set_text(relay.status, tr("relay_saved"));
                    load_relay();
                }
                relay_update();
            }

            void layout_relay(int w, int h, int top)
            {
                const int m = px(16), g = px(8), bh = px(30), row = px(34), label_w = px(190);
                const int column = (w - 3 * m) * 3 / 5;
                const int field_x = m + label_w, field_w = column - label_w;
                ui::place(relay.intro, m, top, w - 2 * m, px(52));
                int y = top + px(58);
                ui::place(relay.enabled, m, y, column, px(24));
                y += px(32);
                if (IsWindowVisible(relay.offline_note))
                {
                    ui::place(relay.offline_note, m, y, column, px(40));
                    y += px(46);
                }
                auto field = [&](HWND label, HWND control, int width, int height) {
                    if (!IsWindowVisible(control)) return;
                    ui::place(label, m, y + px(4), label_w - g, px(22));
                    ui::place(control, field_x, y, width, height);
                    y += row;
                };
                field(relay.provider_label, relay.provider, field_w, px(200));
                field(relay.url_label, relay.url, field_w, px(26));
                field(relay.auth_label, relay.auth, std::min(field_w, px(240)), px(120));
                field(relay.user_label, relay.user, field_w, px(26));
                field(relay.secret_label, relay.secret, field_w, px(26));
                field(relay.model_label, IsWindowVisible(relay.model_list) ? relay.model_list : relay.model_edit,
                    field_w, px(240));
                if (IsWindowVisible(relay.traits))
                {
                    // The traits of the model and the note of its provider take the lines their width leaves them.
                    const int lines = std::max(px(18), ui::text_height(relay.traits, field_w));
                    ui::place(relay.traits, field_x, y - px(2), field_w, lines);
                    y += lines + px(6);
                }
                field(relay.context_label, relay.context, px(120), px(26));
                field(relay.output_label, relay.output, px(120), px(26));
                for (HWND c : { relay.images, relay.reasoning })
                    if (IsWindowVisible(c))
                    {
                        ui::place(c, field_x, y, field_w, px(24));
                        y += px(28);
                    }
                // The pseudonymization beside, in the second column.
                const int x2 = m + column + m, w2 = w - x2 - m;
                int y2 = top + px(58);
                ui::place(relay.privacy, x2, y2, w2, px(40));
                y2 += px(46);
                ui::place(relay.pseudonymize, x2, y2, w2, px(40));
                y2 += px(44);
                ui::place(relay.tags, x2 + px(20), y2, w2 - px(20), px(40));
                y2 += px(44);
                ui::place(relay.keep, x2 + px(20), y2, w2 - px(20), px(40));
                y2 += px(52);
                // The state of Pseudalis beside its icon, then what Pseudalis does, a little apart.
                const int mark = px(20);
                ui::place(relay.pseudalis_icon, x2, y2, mark, mark);
                ui::place(relay.pseudalis, x2 + mark + px(8), y2 + px(2), w2 - mark - px(8), px(40));
                y2 += px(54);
                ui::place(relay.pseudalis_about, x2, y2, w2, px(120));
                const int bottom = h - m - bh;
                ui::place(relay.status, m, bottom - g - px(20), w - 2 * m, px(20));
                int x = m;
                for (HWND b : { relay.test, relay.save })
                {
                    const int bw = ui::button_width(b, px(90));
                    ui::place(b, x, bottom, bw, bh);
                    x += bw + g;
                }
            }

            void layout(int w, int h) override
            {
                // The actions on the selected model under the list, the window commands at the bottom.
                const int m = px(16), g = px(8), bh = px(30);
                // The tabs above the pages, the Close button on both.
                ui::place(tabs, m, px(10), w - 2 * m, px(28));
                const int top = px(46);
                {
                    const int bw = ui::button_width(close, px(90));
                    ui::place(close, w - m - bw, h - m - bh, bw, bh);
                }
                if (page == 1)
                {
                    layout_relay(w, h, top);
                    return;
                }
                // The introduction, then the origin and the state of the catalog.
                // Four lines: the introduction and, offline, the origin of the models.
                ui::place(intro, m, top, w - 2 * m - px(260), px(66));
                ui::place(space, w - m - px(250), top, px(250), px(36));
                const int list_top = top + px(72);
                const int details_h = px(100);
                const int bottom = h - m - bh;
                const int status_top = bottom - g - px(20);
                const int details_top = status_top - g - details_h;
                const int actions_top = details_top - g - bh;
                ui::place(list, m, list_top, w - 2 * m, actions_top - g - list_top);
                RECT inside = {};
                GetClientRect(list, &inside);
                int taken = 0;
                for (int i = 0; i < COLUMN_COUNT - 1; ++i) taken += ListView_GetColumnWidth(list, i);
                set_column_width(list, COLUMN_COUNT - 1, std::max<int>(last_fit, inside.right - taken));
                int x = m;
                for (HWND b : { download, activate, deactivate, verify, remove })
                {
                    const int bw = b == download
                        ? ui::button_width(b, px(90), { tr("btn_download"), tr("btn_cancel_download") })
                        : ui::button_width(b, px(90));
                    ui::place(b, x, actions_top, bw, bh);
                    x += bw + g;
                }
                ui::place(details, m, details_top, w - 2 * m, details_h);
                ui::place(progress, m, status_top + px(3), px(200), px(14));
                const int status_left = m + (IsWindowVisible(progress) ? px(210) : 0);
                ui::place(status, status_left, status_top, w - 2 * m - px(210), px(20));
                int r = w - m;
                for (HWND b : { close, refresh, import })
                {
                    const int bw = ui::button_width(b, px(90));
                    r -= bw;
                    ui::place(b, r, bottom, bw, bh);
                    r -= g;
                }
            }

            const json* model(const std::string& id) const
            {
                for (const auto& m : overview["models"].elements())
                    if (m["id"].as_string() == id) return &m;
                return nullptr;
            }

            const json* download_of(const std::string& id) const
            {
                for (const auto& d : overview["downloads"].elements())
                    if (d["id"].as_string() == id) return &d;
                return nullptr;
            }

            bool downloading(const std::string& id) const
            {
                const json* d = download_of(id);
                if (!d) return false;
                const std::string s = (*d)["state"].as_string();
                return s == "downloading" || s == "verifying" || s == "queued";
            }

            // The functions a model serves: the ones chosen for it, and those it serves for
            // want of a model of their own (the model that comes with Albireo).
            std::vector<std::string> active_roles(const std::string& id) const
            {
                std::vector<std::string> out;
                for (const auto& kv : overview["active"].items())
                    if (kv.second.as_string() == id) out.push_back(kv.first);
                return out;
            }

            // The functions chosen for a model, which Deactivate gives back.
            std::vector<std::string> selected_roles(const std::string& id) const
            {
                const json& chosen = overview.contains("selected") ? overview["selected"] : overview["active"];
                std::vector<std::string> out;
                for (const auto& kv : chosen.items())
                    if (kv.second.as_string() == id) out.push_back(kv.first);
                return out;
            }

            void load()
            {
                service_client c = client;
                const std::string lang = current_language();
                request(R_LIST, [c, lang]() {
                    json r;
                    std::string error;
                    if (!c.poll("/albireo/v1/models?lang=" + lang, r, error))
                    {
                        r = json::object();
                        r["error"] = error;
                    }
                    return r;
                });
            }

            // "unloading" or "removing" for a model that waits for its requests before it
            // leaves the engine.
            std::string leaving(const std::string& id) const
            {
                return overview["leaving"][id].as_string();
            }

            std::wstring state_text(const json& m) const
            {
                const std::string id = m["id"].as_string();
                const std::string leave = leaving(id);
                if (leave == "removing") return tr("state_removing");
                if (leave == "unloading") return tr("state_unloading");
                if (const json* d = download_of(id))
                {
                    const std::string s = (*d)["state"].as_string();
                    if (s == "downloading")
                    {
                        const double total = (*d)["total"].as_double(0), got = (*d)["received"].as_double(0);
                        const int percent = total > 0 ? static_cast<int>(100 * got / total) : 0;
                        return tr("state_downloading", { { "p", std::to_string(percent) } });
                    }
                    if (s == "verifying") return tr("state_verifying");
                    if (s == "queued") return tr("state_queued");
                    if (s == "error") return tr("state_error");
                }
                if (m["installed"].as_bool(false))
                {
                    // The default model of the catalog standing for the model that comes with Albireo has versions.
                    const bool outdated = m.contains("up_to_date") && !m["up_to_date"].as_bool(true);
                    if (outdated) return tr("state_update");
                    return m["builtin"].as_bool(false) ? tr("state_builtin") : tr("state_installed");
                }
                return tr("state_available");
            }

            /*
                The rank of a model in the order of the states: the models installed and active first,
                then the models installed, the downloads in progress, the downloads that failed, and the
                models available.
            */
            int state_rank(const json& m) const
            {
                const std::string id = m["id"].as_string();
                if (m["installed"].as_bool(false))
                    return active_roles(id).empty() && id != overview["background"].as_string() ? 1 : 0;
                if (downloading(id)) return 2;
                const json* d = download_of(id);
                return d && (*d)["state"].as_string() == "error" ? 3 : 4;
            }

            // What a model reads, the text first, then the images and the sound; the attachments of
            // the web application follow them.
            std::wstring inputs_text(const json& m) const
            {
                std::vector<std::string> kinds;
                for (const auto& i : m["inputs"].elements()) kinds.push_back(i.as_string());
                if (kinds.empty()) kinds.push_back("text");
                const char* const known[][2] = { { "text", "input_text" }, { "image", "input_image" },
                    { "audio", "input_audio" } };
                std::wstring text;
                for (const auto& k : known)
                    if (std::find(kinds.begin(), kinds.end(), k[0]) != kinds.end())
                        text += (text.empty() ? L"" : L", ") + tr(k[1]);
                for (const auto& kind : kinds)
                    if (kind != "text" && kind != "image" && kind != "audio")
                        text += (text.empty() ? L"" : L", ") + widen(kind);
                return text;
            }

            void fill_list()
            {
                /*
                    One row per model, in the order of the catalog, or sorted by the column the user
                    chose, the models it ranks alike in the order of the catalog.  The same models in
                    the same order keep their rows: only the texts that changed are written, and painted
                    again.
                */
                struct model_row
                {
                    std::string id;
                    std::vector<std::wstring> cells;
                    unsigned long long size;
                    int rank;
                };
                std::vector<model_row> found;
                for (const auto& m : overview["models"].elements())
                {
                    const std::string id = m["id"].as_string();
                    std::wstring roles;
                    const json& listed = m["roles"];
                    if (listed.size() == 3)
                        roles = tr("roles_all");
                    else
                        for (const auto& r : listed.elements())
                            roles += (roles.empty() ? L"" : L", ") + role_name(r.as_string());
                    // The decoder and the projector of a model that reads images.
                    const unsigned long long size = static_cast<unsigned long long>(m["total_size"].as_int(
                        m["size"].as_int(0)));
                    std::wstring act;
                    const std::vector<std::string> active = active_roles(id);
                    if (active.size() == 3)
                        act = tr("roles_all");
                    else
                        for (const auto& r : active) act += (act.empty() ? L"" : L", ") + role_name(r);
                    // The model that comes with Albireo serves the work in the background.
                    if (id == overview["background"].as_string())
                        act += (act.empty() ? L"" : L", ") + tr("role_background");
                    found.push_back({ id, { widen(m["name"].str(id)), widen(m["publisher"].as_string()), roles,
                        inputs_text(m), size ? widen(ui::format_size(size)) : std::wstring(), state_text(m), act },
                        size, state_rank(m) });
                }
                const list_order order = models_order();
                if (order.column >= 0)
                    std::stable_sort(found.begin(), found.end(), [&order](const model_row& a, const model_row& b) {
                        const int c = order.column;
                        int d = 0;
                        if (c == COL_SIZE) d = (a.size > b.size) - (a.size < b.size);
                        else if (c == COL_STATE) d = a.rank - b.rank;
                        else d = compare_text(a.cells[static_cast<size_t>(c)], b.cells[static_cast<size_t>(c)]);
                        return order.descending ? d > 0 : d < 0;
                    });
                std::vector<std::string> now;
                std::vector<std::vector<std::wstring>> cells;
                for (auto& r : found)
                {
                    now.push_back(r.id);
                    cells.push_back(std::move(r.cells));
                }
                rows.refresh(list, now, cells);
                // The selected model keeps its selection when the list is filled again.
                if (rows.rebuilt && !selected.empty() && rows.selected(list) != selected)
                {
                    rows.filling = true;
                    for (size_t i = 0; i < rows.keys.size(); ++i)
                    {
                        const UINT focus = LVIS_SELECTED | LVIS_FOCUSED;
                        if (rows.keys[i] == selected) ListView_SetItemState(list, static_cast<int>(i), focus, focus);
                    }
                    rows.filling = false;
                }
                if (!fitted || rows.rebuilt || !rows.changed.empty()) fit_columns();
            }

            /*
                The widths of the columns: when the first rows come, each one takes the width of its
                title, with the arrow of the sorted column, and of its texts, the name and the developer
                within a limit; afterwards a column widens when its texts need more, so that a width the
                user chose holds.  The last column takes the width the others leave, at least that of its
                texts (layout).
            */
            void fit_columns()
            {
                for (int c = 0; c < COLUMN_COUNT; ++c)
                {
                    const std::wstring title = tr(column_titles[c]);
                    const int arrow = c == models_order().column ? px(20) : 0;
                    int fit = ListView_GetStringWidth(list, title.c_str()) + px(16) + arrow;
                    for (const auto& row : rows.texts)
                    {
                        if (row.size() <= static_cast<size_t>(c)) continue;
                        const int text = ListView_GetStringWidth(list, row[static_cast<size_t>(c)].c_str());
                        fit = std::max(fit, text + px(16));
                    }
                    if (c == COL_MODEL) fit = std::min(fit, px(260));
                    if (c == COL_PUBLISHER) fit = std::min(fit, px(180));
                    if (c == COLUMN_COUNT - 1) last_fit = fit;
                    else if (!fitted || fit > ListView_GetColumnWidth(list, c)) set_column_width(list, c, fit);
                }
                fitted = true;
                RECT rc;
                GetClientRect(hwnd, &rc);
                layout(rc.right, rc.bottom);
            }

            // The arrow of the sorted column in its title: up for the ascending order, down for the
            // descending one.
            void show_sort_arrow()
            {
                const list_order order = models_order();
                HWND header = ListView_GetHeader(list);
                for (int c = 0; c < COLUMN_COUNT; ++c)
                {
                    HDITEMW item = {};
                    item.mask = HDI_FORMAT;
                    if (!Header_GetItem(header, c, &item)) continue;
                    int format = item.fmt & ~(HDF_SORTUP | HDF_SORTDOWN);
                    if (c == order.column) format |= order.descending ? HDF_SORTDOWN : HDF_SORTUP;
                    if (format == item.fmt) continue;
                    item.fmt = format;
                    Header_SetItem(header, c, &item);
                }
            }

            // A click on the title of a column sorts the list by it, ascending, then descending at
            // the next click.
            void sort_by(int column)
            {
                if (column < 0 || column >= COLUMN_COUNT) return;
                list_order& order = models_order();
                order.descending = column == order.column && !order.descending;
                order.column = column;
                save_models_order(order);
                show_sort_arrow();
                fill_list();
                // The title of the sorted column takes the room of its arrow.
                fit_columns();
                // The sorted list shows its first rows, down to the selected model; the horizontal
                // position holds.
                const int row = ListView_GetNextItem(list, -1, LVNI_SELECTED);
                const int page = std::max(1, ListView_GetCountPerPage(list));
                RECT item = {};
                if (!ListView_GetItemRect(list, 0, &item, LVIR_BOUNDS)) return;
                const int top = row >= page ? row - page + 1 : 0;
                SendMessageW(list, WM_SETREDRAW, FALSE, 0);
                ListView_Scroll(list, 0, (top - ListView_GetTopIndex(list)) * (item.bottom - item.top));
                SendMessageW(list, WM_SETREDRAW, TRUE, 0);
                InvalidateRect(list, nullptr, TRUE);
            }

            void show_details()
            {
                const json* m = model(selected);
                if (!m)
                {
                    ui::set_text(details, L"");
                    return;
                }
                std::string text = (*m)["name"].str(selected) + "\r\n";
                const std::string desc = (*m)["description"].as_string();
                if (!desc.empty()) text += desc + "\r\n";
                text += "\r\n";
                auto line = [&](const char* key, const std::string& v) {
                    if (!v.empty()) text += tr_utf8(key) + ": " + v + "\r\n";
                };
                line("lbl_publisher", (*m)["publisher"].as_string());
                if ((*m)["parameters"].as_int(0) > 0)
                {
                    char buf[32];
                    snprintf(buf, sizeof(buf), "%.2f B", (*m)["parameters"].as_double() / 1e9);
                    line("lbl_parameters", buf);
                }
                line("lbl_quantization", (*m)["quantization"].as_string());
                line("lbl_inputs", narrow(inputs_text(*m)));
                if ((*m)["context"].as_int(0) > 0) line("lbl_context", std::to_string((*m)["context"].as_int()));
                line("lbl_license", (*m)["license"].as_string());
                const json& source = (*m)["source"].is_string() ? (*m)["source"] : (*m)["homepage"];
                const json& sha = (*m)["sha256"].is_string() ? (*m)["sha256"] : (*m)["local_sha256"];
                line("lbl_source", source.as_string());
                line("lbl_sha256", sha.as_string());
                std::wstring w = widen(text);
                if (const json* d = download_of(selected))
                    if ((*d)["error"].is_string()) w += L"\r\n" + error_text(*d);
                ui::set_text(details, w);
            }

            void update_buttons()
            {
                const json* m = model(selected);
                const bool installed = m && (*m)["installed"].as_bool(false);
                const bool in_catalog = m && (*m)["files"].is_array();
                const bool dl = m && downloading(selected);
                const bool update = installed && m->contains("up_to_date") && !(*m)["up_to_date"].as_bool(true);
                // A model deleted once its requests are served takes no other action.
                const bool removing = leaving(selected) == "removing";
                // Offline, the models come from files only.
                const bool offline = overview["offline"].as_bool(false);
                ui::set_text(download, dl ? tr("btn_cancel_download") : tr("btn_download"));
                ui::enable(download, !busy && m && (dl || (!offline && in_catalog && (!installed || update))));
                ui::enable(activate, !busy && installed && !dl && !removing);
                // The document search keeps its embedding model, which another one replaces.
                ui::enable(deactivate, !busy && installed && !selected_roles(selected).empty()
                    && !embedding_model(selected) && !removing);
                ui::enable(verify, !busy && installed && !removing);
                // A model that comes with Albireo stays installed.
                ui::enable(remove, !busy && installed && !dl && !removing && !(*m)["builtin"].as_bool(false));
                ui::enable(import, !busy && !uploading);
                ui::enable(refresh, !busy && !offline);
                // The progress of the download of the selected model.
                const json* d = m ? download_of(selected) : nullptr;
                if (d && dl && !uploading)
                {
                    const double total = (*d)["total"].as_double(0), got = (*d)["received"].as_double(0);
                    show_progress(true);
                    SendMessageW(progress, PBM_SETPOS, total > 0 ? static_cast<WPARAM>(1000 * got / total) : 0, 0);
                    const std::string done = ui::format_size(static_cast<unsigned long long>(got));
                    const std::string size = ui::format_size(static_cast<unsigned long long>(total));
                    const unsigned long long rate = static_cast<unsigned long long>((*d)["speed"].as_double(0));
                    const std::string speed = ui::format_size(rate);
                    ui::set_text(status, widen(done + " / " + size + " · " + speed + "/s"));
                    showing_download = true;
                }
                else if (!uploading)
                {
                    show_progress(false);
                    // The progress of a download leaves with the selection of its model.
                    if (showing_download) ui::set_text(status, L"");
                    showing_download = false;
                }
                RECT rc;
                GetClientRect(hwnd, &rc);
                layout(rc.right, rc.bottom);
            }

            void action(const char* method, const std::string& path, const json& body)
            {
                busy = true;
                update_buttons();
                service_client c = client;
                const std::string m = method;
                request(R_ACTION, [c, m, path, body]() {
                    json r;
                    std::string error;
                    if (!c.send(m.c_str(), path, body, r, error, 120)) r = failed_answer(r, error);
                    return r;
                });
            }

            // The origin and the state of the catalog, under the introduction.
            std::wstring catalog_text() const
            {
                const json& c = overview["catalog"];
                const std::string state = c["state"].as_string();
                if (overview["offline"].as_bool(false)) return tr("catalog_offline");
                if (state == "checking") return tr("catalog_checking");
                if (overview["catalog_time"].as_int(0) == 0) return tr("no_catalog");
                if (state == "error") return tr("catalog_error", { { "e", c["error"].as_string() } });
                return std::wstring();
            }

            /*
                What an import installed: the catalog, the models, the models installed already
                (with the file that holds each one for the files sent), the update, the errors.
            */
            std::wstring import_text(const json& r) const
            {
                std::wstring text;
                if (!r["imported"].empty() || r["already"].empty())
                    text = tr("import_done", { { "n", std::to_string(r["imported"].size()) } });
                for (const auto& a : r["already"].elements())
                {
                    const std::string id = a.is_object() ? a["m"].as_string() : a.as_string();
                    const json* m = model(id);
                    const std::string name = m && !(*m)["name"].as_string().empty() ? (*m)["name"].as_string() : id;
                    text += (text.empty() ? L"" : L"\n") + (a.is_object()
                        ? tr("import_already_file", { { "f", a["file"].as_string() }, { "m", name } })
                        : tr("import_already", { { "m", name } }));
                }
                const json& c = r["catalog"];
                if (c["models"].is_number())
                    text += L"\n" + tr("import_catalog", { { "n", std::to_string(c["models"].as_int()) } });
                else if (c["state"].as_string() == "kept") text += L"\n" + tr("import_catalog_kept");
                else if (c["error"].is_string()) text += L"\n" + widen(c["error"].as_string());
                const json& u = r["update"];
                const std::string version = u["version"].as_string();
                if (u["state"].as_string() == "ready") text += L"\n" + tr("import_update_ready", { { "v", version } });
                else if (u["state"].as_string() == "up_to_date")
                    text += L"\n" + tr("import_update_current", { { "v", version } });
                else if (u["error"].is_string()) text += L"\n" + error_text(u);
                for (const auto& e : r["errors"].elements()) text += L"\n" + widen(e.as_string());
                return text;
            }

            void on_result(WPARAM tag, const json& r) override
            {
                if (tag == R_RELAY_MODELS)
                {
                    // The latest listing alone, for the fields it was asked for; a failure leaves no proposal.
                    if (r["seq"].as_int(0) == relay_list_seq && relay_provider() == "openai")
                        fill_relay_proposals(r["ok"].as_bool(false) ? r["models"] : json::array());
                    return;
                }
                if (tag == R_RELAY || tag == R_RELAY_TEST || tag == R_RELAY_SAVE)
                {
                    relay_result(tag, r);
                    return;
                }
                if (tag == R_LIST)
                {
                    if (r.contains("error") && !r.contains("models"))
                    {
                        ui::set_text(status, widen(r["error"].as_string()));
                        return;
                    }
                    overview = r;
                    ui::set_text(intro, tr("models_intro") + L"\n" + catalog_text());
                    const std::string available = ui::format_size(static_cast<unsigned long long>(r["free"].as_int(0)));
                    ui::set_text(space, tr("lbl_free_space", { { "s", available } }));
                    fill_list();
                    show_details();
                    update_buttons();
                    return;
                }
                busy = false;
                if (tag == R_UPLOAD_DONE) uploading = false;
                if (r.contains("error") && r["error"].is_string() && !r["error"].as_string().empty())
                    ui::message(hwnd, error_text(r), MB_OK | MB_ICONWARNING);
                else if (tag == R_VERIFY)
                {
                    std::wstring text = r["local_match"].as_bool(false) ? tr("verify_ok") : tr("verify_bad");
                    if (r.contains("catalog_match"))
                    {
                        const bool same = r["catalog_match"].as_bool(false);
                        text += L"\n" + (same ? tr("verify_catalog_ok") : tr("verify_catalog_bad"));
                    }
                    const UINT icon = r["local_match"].as_bool(false) ? MB_ICONINFORMATION : MB_ICONWARNING;
                    ui::message(hwnd, text, MB_OK | icon);
                }
                else if (tag == R_IMPORT || tag == R_UPLOAD_DONE)
                {
                    const bool failed = !r["errors"].empty() || r["catalog"].contains("error")
                        || r["update"].contains("error");
                    ui::message(hwnd, import_text(r), MB_OK | (failed ? MB_ICONWARNING : MB_ICONINFORMATION));
                    show_progress(false);
                    ui::set_text(status, L"");
                }
                update_buttons();
                load();
            }

            // Whether a model computes the vectors of the document search.
            bool embedding_model(const std::string& id) const
            {
                const json* m = model(id);
                if (!m) return false;
                if ((*m)["embedding"].as_bool(false)) return true;
                for (const auto& r : (*m)["roles"].elements())
                    if (r.as_string() == "embedding") return true;
                return false;
            }

            /*
                The functions of the selected model, those it serves checked.  A function not
                checked is taken from the model that serves it, and a checked function goes back
                to the model that comes with Albireo, which keeps the work in the background and
                whose own functions stay checked and greyed, as the document search of an
                embedding model does.
            */
            void role_menu()
            {
                HMENU m = CreatePopupMenu();
                const auto act = active_roles(selected);
                const bool builtin = selected == overview["background"].as_string();
                // An embedding model serves the document search only; the language models
                // answer for the other functions.
                const char* roles[] = { "general", "code", "security", "embedding" };
                const bool embedding = embedding_model(selected);
                bool on[4] = { false, false, false, false };
                for (int i = 0; i < 4; ++i)
                {
                    if ((i == 3) != embedding) continue;
                    on[i] = std::find(act.begin(), act.end(), roles[i]) != act.end();
                    // The document search keeps its model until another embedding model takes it.
                    const bool kept = on[i] && (builtin || i == 3);
                    const UINT flags = MF_STRING | (on[i] ? MF_CHECKED : 0) | (kept ? MF_GRAYED : 0);
                    AppendMenuW(m, flags, IDM_ROLE_GENERAL + i, role_name(roles[i]).c_str());
                }
                if (builtin)
                {
                    AppendMenuW(m, MF_SEPARATOR, 0, nullptr);
                    AppendMenuW(m, MF_STRING | MF_CHECKED | MF_GRAYED, 0, tr("role_background").c_str());
                }
                RECT r;
                GetWindowRect(activate, &r);
                const UINT flags = TPM_RETURNCMD | TPM_LEFTALIGN | TPM_BOTTOMALIGN;
                const UINT cmd = TrackPopupMenuEx(m, flags, r.left, r.top, hwnd, nullptr);
                DestroyMenu(m);
                if (cmd >= IDM_ROLE_GENERAL && cmd <= IDM_ROLE_EMBEDDING)
                {
                    const int i = static_cast<int>(cmd - IDM_ROLE_GENERAL);
                    json body = json::object();
                    body["role"] = roles[i];
                    action("POST", "/albireo/v1/models/" + selected + (on[i] ? "/deactivate" : "/activate"), body);
                }
            }

            void import_menu()
            {
                HMENU m = CreatePopupMenu();
                AppendMenuW(m, MF_STRING, IDM_IMPORT_FILES, tr("import_files").c_str());
                AppendMenuW(m, MF_STRING, IDM_IMPORT_FOLDER, tr("import_folder_menu").c_str());
                RECT r;
                GetWindowRect(import, &r);
                const UINT flags = TPM_RETURNCMD | TPM_LEFTALIGN | TPM_BOTTOMALIGN;
                const UINT cmd = TrackPopupMenuEx(m, flags, r.left, r.top, hwnd, nullptr);
                DestroyMenu(m);
                if (cmd == IDM_IMPORT_FILES) import_files();
                else if (cmd == IDM_IMPORT_FOLDER) import_folder();
            }

            void import_folder()
            {
                const std::wstring dir = ui::choose_folder(hwnd, L"");
                if (dir.empty()) return;
                busy = true;
                update_buttons();
                service_client c = client;
                const std::string path = narrow(dir);
                request(R_IMPORT, [c, path]() {
                    json r, body = json::object();
                    body["path"] = path;
                    std::string error;
                    // A refusal keeps its code and the path, which its text names.
                    if (!c.send("POST", "/albireo/v1/import", body, r, error, 3600)) r = failed_answer(r, error);
                    return r;
                });
            }

            void import_files()
            {
                // The files are sent to the service, which cannot read the folders of the user.
                std::vector<std::string> files;
                IFileOpenDialog* dlg = nullptr;
                if (FAILED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER, IID_IFileOpenDialog,
                        reinterpret_cast<void**>(&dlg))))
                    return;
                const std::wstring filter_name = tr("import_files");
                COMDLG_FILTERSPEC spec = { filter_name.c_str(), L"*.gguf" };
                dlg->SetFileTypes(1, &spec);
                DWORD opts = 0;
                dlg->GetOptions(&opts);
                dlg->SetOptions(opts | FOS_ALLOWMULTISELECT | FOS_FILEMUSTEXIST | FOS_FORCEFILESYSTEM);
                if (SUCCEEDED(dlg->Show(hwnd)))
                {
                    IShellItemArray* items = nullptr;
                    if (SUCCEEDED(dlg->GetResults(&items)))
                    {
                        DWORD n = 0;
                        items->GetCount(&n);
                        for (DWORD i = 0; i < n; ++i)
                        {
                            IShellItem* it = nullptr;
                            if (FAILED(items->GetItemAt(i, &it))) continue;
                            PWSTR p = nullptr;
                            if (SUCCEEDED(it->GetDisplayName(SIGDN_FILESYSPATH, &p)))
                            {
                                files.push_back(narrow(p));
                                CoTaskMemFree(p);
                            }
                            it->Release();
                        }
                        items->Release();
                    }
                }
                dlg->Release();
                if (files.empty()) return;
                uploading = true;
                busy = true;
                show_progress(true);
                update_buttons();
                service_client c = client;
                HWND target = hwnd;
                ui::run_async(hwnd, WM_RESULT, R_UPLOAD_DONE, [c, files, target]() {
                    json all = json::object();
                    json imported = json::array(), already = json::array(), errors = json::array();
                    for (const auto& f : files)
                    {
                        const std::string name = file_name(f);
                        json r;
                        std::string error;
                        const bool ok = c.upload("/albireo/v1/models/upload?name=" + query_value(name), f,
                            [target, name](uint64_t done, uint64_t total) {
                                std::string* msg = new std::string(name);
                                const WPARAM permille = total ? static_cast<WPARAM>(1000 * done / total) : 0;
                                if (!PostMessageW(target, WM_PROGRESS, permille, reinterpret_cast<LPARAM>(msg)))
                                    delete msg;
                                return IsWindow(target) != FALSE;
                            }, r, error);
                        if (ok)
                        {
                            for (const auto& id : r["imported"].elements()) imported.push_back(id);
                            // A file that holds a model installed already, a success too.
                            for (const auto& id : r["already"].elements())
                            {
                                json a = json::object();
                                a["file"] = name;
                                a["m"] = id;
                                already.push_back(a);
                            }
                        }
                        for (const auto& e : r["errors"].elements()) errors.push_back(e);
                        if (!ok && r["errors"].empty()) errors.push_back(name + ": " + error);
                    }
                    all["imported"] = imported;
                    all["already"] = already;
                    all["errors"] = errors;
                    return all;
                });
            }

            LRESULT on_message(UINT msg, WPARAM wp, LPARAM lp, bool& handled) override
            {
                handled = false;
                if (msg == WM_PROGRESS)
                {
                    std::string* name = reinterpret_cast<std::string*>(lp);
                    SendMessageW(progress, PBM_SETPOS, wp, 0);
                    ui::set_text(status, tr("import_progress", { { "f", *name }, { "p", std::to_string(wp / 10) } }));
                    delete name;
                    handled = true;
                    return 0;
                }
                if (msg == WM_TIMER)
                {
                    // Every second while a download runs, every five seconds otherwise.
                    static int ticks = 0;
                    bool active = false;
                    for (const auto& d : overview["downloads"].elements())
                    {
                        const std::string s = d["state"].as_string();
                        active = active || s == "downloading" || s == "verifying" || s == "queued";
                    }
                    if (!busy && (active || ++ticks % 5 == 0)) load();
                    handled = true;
                    return 0;
                }
                if (msg == WM_NOTIFY)
                {
                    const NMHDR* n = reinterpret_cast<NMHDR*>(lp);
                    if (n->idFrom == IDC_TABS && n->code == TCN_SELCHANGE)
                        show_page(static_cast<int>(SendMessageW(tabs, TCM_GETCURSEL, 0, 0)));
                    else if (n->idFrom == IDC_LIST && n->code == LVN_ITEMCHANGED && !rows.filling)
                    {
                        const NMLISTVIEW* lv = reinterpret_cast<const NMLISTVIEW*>(lp);
                        const bool known = lv->iItem >= 0 && lv->iItem < static_cast<int>(rows.keys.size());
                        if ((lv->uNewState & LVIS_SELECTED) && known)
                        {
                            selected = rows.keys[static_cast<size_t>(lv->iItem)];
                            show_details();
                            update_buttons();
                        }
                    }
                    else if (n->idFrom == IDC_LIST && n->code == LVN_COLUMNCLICK)
                    {
                        sort_by(reinterpret_cast<const NMLISTVIEW*>(lp)->iSubItem);
                    }
                    handled = true;
                    return 0;
                }
                // At another resolution the texts take other widths: the columns fit them again at the
                // next rows.
                if (msg == WM_DPICHANGED)
                {
                    fitted = false;
                    ui::set_icon(relay.pseudalis_icon, IDI_PSEUDALIS, px(20));
                }
                return 0;
            }

            void on_command(int id, int code) override
            {
                switch (id)
                {
                case IDC_R_PROVIDER:
                    if (code != CBN_SELCHANGE) break;
                    {
                        /*
                            The form keeps what it held for the provider left, and shows for the
                            provider taken what it held for it before, else what the settings keep for
                            it: the mask of its key stands for the key stored for that provider.
                        */
                        const std::string taken = relay_provider();
                        if (taken == relay_shown) break;
                        relay_drafts[relay_shown] = relay_profile(relay_shown);
                        relay_shown = taken;
                        const auto draft = relay_drafts.find(taken);
                        show_relay_profile(draft != relay_drafts.end() ? draft->second : stored_relay_profile(taken));
                        list_relay_models();
                    }
                    break;
                case IDC_R_AUTH:
                    if (code != CBN_SELCHANGE) break;
                    relay_update();
                    list_relay_models();
                    break;
                case IDC_R_MODEL_LIST:
                    if (code == CBN_SELCHANGE) relay_update();
                    break;
                case IDC_R_MODEL_EDIT:
                    // A list opened empty asks again for the models of the fields as they are.
                    if (code != CBN_DROPDOWN) break;
                    if (SendMessageW(relay.model_edit, CB_GETCOUNT, 0, 0) <= 0) relay_list_key.clear();
                    list_relay_models();
                    break;
                case IDC_R_URL:
                    if (code == EN_CHANGE) relay_update();
                    if (code == EN_KILLFOCUS) list_relay_models();
                    break;
                case IDC_R_USER:
                case IDC_R_SECRET:
                    if (code == EN_KILLFOCUS) list_relay_models();
                    break;
                case IDC_R_PSEUDONYMIZE:
                    if (code == BN_CLICKED) relay_update();
                    break;
                case IDC_R_TEST: relay_send(R_RELAY_TEST); break;
                case IDC_R_SAVE: relay_send(R_RELAY_SAVE); break;
                case IDC_DOWNLOAD:
                    if (downloading(selected)) action("DELETE", "/albireo/v1/models/" + selected + "/download", json());
                    else action("POST", "/albireo/v1/models/" + selected + "/download", json::object());
                    break;
                case IDC_ACTIVATE: role_menu(); break;
                case IDC_DEACTIVATE:
                    action("POST", "/albireo/v1/models/" + selected + "/deactivate", json::object());
                    break;
                case IDC_VERIFY:
                {
                    busy = true;
                    update_buttons();
                    service_client c = client;
                    const std::string id2 = selected;
                    request(R_VERIFY, [c, id2]() {
                        json r;
                        std::string error;
                        if (!c.send("POST", "/albireo/v1/models/" + id2 + "/verify", json::object(), r, error, 3600))
                            r = failed_answer(r, error);
                        return r;
                    });
                    break;
                }
                case IDC_DELETE:
                {
                    const json* m = model(selected);
                    const std::string name = m ? (*m)["name"].str(selected) : selected;
                    if (ui::message(hwnd, tr("delete_confirm", { { "m", name } }), MB_YESNO | MB_ICONQUESTION) == IDYES)
                        action("DELETE", "/albireo/v1/models/" + selected, json());
                    break;
                }
                case IDC_IMPORT: import_menu(); break;
                case IDC_REFRESH: action("POST", "/albireo/v1/models/refresh", json::object()); break;
                case IDC_CLOSE: DestroyWindow(hwnd); break;
                default: break;
                }
            }
        };

        // The window of this computer.
        enum
        {
            IDC_H_INFO = 2000, IDC_H_ENGINE, IDC_H_ENGINES, IDC_H_V2RAM, IDC_H_V2MAX, IDC_H_BAR, IDC_H_LEGEND,
            IDC_H_FOLDER, IDC_H_OPEN, IDC_H_CHANGE, IDC_H_PORT, IDC_H_URL, IDC_H_LANG, IDC_H_UPDATES, IDC_H_APPLY,
            IDC_H_CLOSE, IDC_H_STATUS, IDC_H_PROXY, IDC_H_OFFLINE, IDC_H_GROUP = 2100
        };
        enum { R_HOST = 1, R_APPLY };

        class host_window : public tool_window
        {
        public:
            HWND groups[6];
            HWND info_keys, info_values, engine, engines, v2ram, v2max_label, v2max, bar, legend, folder;
            HWND open_folder, change_folder, port_label, port, url, lang_label, lang, proxy_label, proxy, updates;
            HWND offline, apply, close, status;
            json st, eng, settings;
            // The settings an Apply in progress changes, as they were, kept again when it fails.
            json previous;
            bool loaded = false;
            // The rows of the engines, one per library, and the memory the bar shows.
            list_rows rows;
            json drawn_memory;

            host_window() { min_size = SIZE{ 720, 756 }; }

            void build() override
            {
                const char* titles[] = { "sec_computer", "sec_engine", "sec_memory", "sec_storage", "sec_server",
                    "sec_preferences" };
                for (int i = 0; i < 6; ++i)
                {
                    HMENU id = reinterpret_cast<HMENU>(static_cast<INT_PTR>(IDC_H_GROUP + i));
                    HWND box = CreateWindowExW(0, L"BUTTON", tr(titles[i]).c_str(), WS_CHILD | WS_VISIBLE | BS_GROUPBOX,
                        0, 0, 10, 10, hwnd, id, nullptr, nullptr);
                    groups[i] = add(box, ui::font_kind::bold);
                }
                info_keys = add(ui::label(hwnd, L"", IDC_H_INFO), ui::font_kind::normal);
                info_values = add(ui::label(hwnd, L"", IDC_H_INFO + 50), ui::font_kind::normal);
                engine = add(ui::label(hwnd, L"", IDC_H_ENGINE), ui::font_kind::normal);
                engines = add(ui::listview(hwnd, IDC_H_ENGINES, LVS_NOSORTHEADER));
                // The state comes last: it gives the reason of an engine that cannot run here,
                // and takes the width of its longest text.
                const char* columns[] = { "col_engine", "col_version", "col_devices", "col_measure", "col_state" };
                const int widths[] = { 115, 55, 260, 100, 160 };
                for (int i = 0; i < 5; ++i)
                {
                    std::wstring text = tr(columns[i]);
                    LVCOLUMNW c = {};
                    c.mask = LVCF_TEXT | LVCF_WIDTH;
                    c.pszText = &text[0];
                    c.cx = px(widths[i]);
                    ListView_InsertColumn(engines, i, &c);
                }
                v2ram = add(ui::checkbox(hwnd, tr("lbl_v2ram"), IDC_H_V2RAM, true));
                v2max_label = add(ui::label(hwnd, tr("lbl_v2ram_max"), 0));
                v2max = add(ui::edit(hwnd, L"0", IDC_H_V2MAX, ES_NUMBER));
                bar = CreateWindowExW(0, L"STATIC", L"", WS_CHILD | WS_VISIBLE | SS_OWNERDRAW, 0, 0, 10, 10, hwnd,
                    reinterpret_cast<HMENU>(static_cast<INT_PTR>(IDC_H_BAR)), nullptr, nullptr);
                legend = add(ui::label(hwnd, L"", IDC_H_LEGEND), ui::font_kind::note);
                folder = add(ui::label(hwnd, L"", IDC_H_FOLDER, SS_PATHELLIPSIS));
                open_folder = add(ui::button(hwnd, tr("btn_open_folder"), IDC_H_OPEN));
                change_folder = add(ui::button(hwnd, tr("btn_change"), IDC_H_CHANGE));
                port_label = add(ui::label(hwnd, tr("lbl_port"), 0));
                port = add(ui::edit(hwnd, L"", IDC_H_PORT, ES_NUMBER));
                url = add(ui::label(hwnd, L"", IDC_H_URL), ui::font_kind::normal);
                lang_label = add(ui::label(hwnd, tr("lbl_language"), 0));
                lang = add(ui::combo(hwnd, IDC_H_LANG));
                const wchar_t* names[] = { nullptr, L"English", L"Français", L"Deutsch", L"Italiano", L"Español" };
                SendMessageW(lang, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(tr("lang_auto").c_str()));
                for (int i = 1; i < 6; ++i) SendMessageW(lang, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(names[i]));
                proxy_label = add(ui::label(hwnd, tr("lbl_proxy"), 0));
                proxy = add(ui::edit(hwnd, L"", IDC_H_PROXY, ES_AUTOHSCROLL));
                SendMessageW(proxy, EM_SETCUEBANNER, TRUE, reinterpret_cast<LPARAM>(tr("proxy_hint").c_str()));
                updates = add(ui::checkbox(hwnd, tr("lbl_auto_updates"), IDC_H_UPDATES, true));
                offline = add(ui::checkbox(hwnd, tr("lbl_offline"), IDC_H_OFFLINE, false));
                status = add(ui::label(hwnd, L"", IDC_H_STATUS), ui::font_kind::note);
                apply = add(ui::button(hwnd, tr("btn_apply"), IDC_H_APPLY));
                close = add(ui::button(hwnd, tr("btn_close"), IDC_H_CLOSE));
                SetTimer(hwnd, 1, 3000, nullptr);
                load();
            }

            void layout(int w, int h) override
            {
                const int m = px(14), g = px(8), lh = px(20), bh = px(30);
                const int inner = w - 2 * m;
                int y = m;
                // Computer.
                ui::place(groups[0], m, y, inner, px(126));
                ui::place(info_keys, m + px(12), y + px(24), px(150), px(96));
                ui::place(info_values, m + px(170), y + px(24), inner - px(182), px(96));
                y += px(126) + g;
                // Engine: the list shows the engines of an installation with every device.
                ui::place(groups[1], m, y, inner, px(194));
                ui::place(engine, m + px(12), y + px(24), inner - px(24), lh);
                ui::place(engines, m + px(12), y + px(48), inner - px(24), px(134));
                y += px(194) + g;
                // Extended memory.
                ui::place(groups[2], m, y, inner, px(118));
                ui::place(v2ram, m + px(12), y + px(22), inner - px(24), lh);
                ui::place(v2max_label, m + px(12), y + px(48), px(260), lh);
                ui::place(v2max, m + px(280), y + px(45), px(100), px(24));
                ui::place(bar, m + px(12), y + px(76), inner - px(24), px(14));
                ui::place(legend, m + px(12), y + px(94), inner - px(24), px(18));
                y += px(118) + g;
                // Models folder and local server side by side.
                const int half = (inner - g) / 2;
                ui::place(groups[3], m, y, half, px(86));
                ui::place(folder, m + px(12), y + px(24), half - px(24), lh);
                ui::place(open_folder, m + px(12), y + px(48), px(90), px(28));
                ui::place(change_folder, m + px(110), y + px(48), px(110), px(28));
                ui::place(groups[4], m + half + g, y, half, px(86));
                ui::place(port_label, m + half + g + px(12), y + px(27), px(60), lh);
                ui::place(port, m + half + g + px(76), y + px(24), px(90), px(24));
                ui::place(url, m + half + g + px(12), y + px(56), half - px(24), lh);
                y += px(86) + g;
                // Preferences.
                ui::place(groups[5], m, y, inner, px(136));
                ui::place(lang_label, m + px(12), y + px(27), px(110), lh);
                ui::place(lang, m + px(126), y + px(23), px(200), px(200));
                ui::place(proxy_label, m + px(12), y + px(56), px(110), lh);
                ui::place(proxy, m + px(126), y + px(53), inner - px(138), px(24));
                ui::place(updates, m + px(12), y + px(82), inner - px(24), lh);
                ui::place(offline, m + px(12), y + px(106), inner - px(24), lh);
                // Buttons.
                ui::place(status, m, h - m - bh + px(6), inner - px(220), lh);
                ui::place(close, w - m - px(100), h - m - bh, px(100), bh);
                ui::place(apply, w - m - px(208), h - m - bh, px(100), bh);
            }

            void load()
            {
                service_client c = client;
                request(R_HOST, [c]() {
                    json r = json::object();
                    std::string error;
                    json s, e, set;
                    if (!c.poll("/albireo/v1/status", s, error))
                    {
                        r["error"] = error;
                        return r;
                    }
                    r["status"] = s;
                    if (c.poll("/albireo/v1/engines", e, error)) r["engines"] = e;
                    if (c.poll("/albireo/v1/settings", set, error)) r["settings"] = set;
                    return r;
                });
            }

            void show()
            {
                const json& host = st["host"];
                std::wstring keys = tr("lbl_name") + L"\n" + tr("lbl_os") + L"\n" + tr("lbl_cpu") + L"\n"
                    + tr("lbl_cores") + L"\n" + tr("lbl_features") + L"\n" + tr("lbl_memory");
                std::string features;
                for (const auto& f : host["cpu"]["features"].elements())
                    features += (features.empty() ? "" : ", ") + f.as_string();
                const std::string values = host["name"].as_string() + "\n" + host["os"].as_string() + "\n" +
                    host["cpu"]["name"].as_string() + "\n" + std::to_string(host["cpu"]["cores"].as_int()) + " / " +
                    std::to_string(host["cpu"]["threads"].as_int()) + "\n" + features + "\n" +
                    ui::format_size(static_cast<unsigned long long>(host["memory"]["total"].as_int())) + " (" +
                    ui::format_size(static_cast<unsigned long long>(host["memory"]["available"].as_int())) + " " +
                    tr_utf8("lbl_free") + ")";
                ui::set_text(info_keys, keys);
                ui::set_text(info_values, widen(values));

                // The engine in use, with the device of its choice.
                const json& e = st["engine"];
                std::wstring current = widen(e["variant"].as_string());
                bool named = false;
                for (const auto& o : eng["options"].elements())
                {
                    const std::string kind = o["kind"].as_string();
                    if (!o["current"].as_bool(false) || kind.empty() || kind == "cpu") continue;
                    current += L"  ·  " + choice_label(o);
                    named = true;
                }
                if (!named && !e["device"].as_string().empty())
                    current += L" (" + widen(e["device"].as_string()) + L")";
                const double tps = e["stats"]["generation_tokens_per_second"].as_double(0);
                std::wstring line = tr("lbl_engine") + L": " + current;
                if (tps > 0.05) line += L"  ·  " + tr("tokens_s", { { "n", ui::format_decimal(tps, 1) } });
                if (!e["error"].as_string().empty()) line += L"  ·  " + widen(e["error"].as_string());
                ui::set_text(engine, line);

                /*
                    The engines, their devices and their state: the reason of an engine that
                    cannot run here, and the devices that the runtime of an engine left out.
                    The rows are written in place, so that the list is painted where it changes.
                */
                std::vector<std::string> paths;
                std::vector<std::vector<std::wstring>> cells;
                for (const auto& x : eng["engines"].elements())
                {
                    const bool usable = x["usable"].as_bool(false);
                    std::wstring state = x["current"].as_bool(false) ? tr("in_use") : usable ? tr("usable")
                        : tr("unavailable");
                    if (!usable && (x["reason"].is_string() || x["reason_code"].is_string()))
                        state += L": " + reason_text(x);
                    if (usable && !x["missing"].as_string().empty())
                        state += L"  ·  " + tr("devices_left_out", { { "m", x["missing"].as_string() } });
                    const std::string variant = x["variant"].as_string();
                    paths.push_back(x["path"].str(variant));
                    cells.push_back({ widen(variant), widen(x["version"].as_string()), devices_text(x),
                        measures_text(eng, variant), state });
                }
                rows.refresh(engines, paths, cells);
                if (rows.rebuilt || rows.changed.count(4)) fit_state_column();

                // V²RAM extends the memory of an NVIDIA card, for the CUDA engine only.
                bool cuda = false;
                for (const auto& x : eng["engines"].elements())
                    cuda = cuda || (x["variant"].as_string() == "cuda" && x["usable"].as_bool(false));
                ui::enable(v2ram, cuda);
                ui::enable(v2max_label, cuda);
                ui::enable(v2max, cuda);
                const json& mem = e["stats"]["memory"];
                if (!cuda)
                {
                    ui::set_text(legend, tr("v2ram_no_cuda"));
                }
                else if (mem.is_object() && mem["vram_budget"].as_int(0) > 0)
                {
                    auto size_of = [&](const char* key) {
                        return ui::format_size(static_cast<unsigned long long>(mem[key].as_int()));
                    };
                    const std::string text = tr_utf8("v2ram_device") + " " + size_of("device_bytes") + " / "
                        + size_of("vram_budget") + "   ·   " + tr_utf8("v2ram_host") + " " + size_of("pinned_bytes")
                        + "   ·   " + tr_utf8("v2ram_disk") + " " + size_of("store_bytes");
                    ui::set_text(legend, widen(text));
                }
                else
                {
                    ui::set_text(legend, tr("v2ram_none"));
                }
                // The bar is painted again when the memory it shows changes; it covers its whole surface.
                if (mem != drawn_memory)
                {
                    drawn_memory = mem;
                    InvalidateRect(bar, nullptr, FALSE);
                }
                ui::set_text(folder, widen(st["models_dir"].as_string()));
                ui::set_text(url, widen(st["url"].as_string()));

                if (!loaded && settings.is_object())
                {
                    // The fields the user edits are filled once.
                    loaded = true;
                    SendMessageW(v2ram, BM_SETCHECK, settings["v2ram"].as_bool(true) ? BST_CHECKED : BST_UNCHECKED, 0);
                    ui::set_text(v2max, std::to_wstring(settings["v2ram_max_mb"].as_int(0)));
                    ui::set_text(port, std::to_wstring(settings["port"].as_int(8443)));
                    const char* codes[] = { "auto", "en", "fr", "de", "it", "es" };
                    int sel = 0;
                    for (int i = 0; i < 6; ++i)
                        if (settings["language"].as_string() == codes[i]) sel = i;
                    SendMessageW(lang, CB_SETCURSEL, static_cast<WPARAM>(sel), 0);
                    const bool check = settings["check_updates"].as_bool(true);
                    SendMessageW(updates, BM_SETCHECK, check ? BST_CHECKED : BST_UNCHECKED, 0);
                    const bool off = settings["offline"].as_bool(false);
                    SendMessageW(offline, BM_SETCHECK, off ? BST_CHECKED : BST_UNCHECKED, 0);
                    // Offline, neither the updates nor the proxy are used.
                    ui::enable(updates, !off);
                    ui::enable(proxy, !off);
                    ui::set_text(proxy, widen(settings["proxy"].as_string()));
                }
            }

            // The state takes the width of its longest text, or the rest of the list.
            void fit_state_column()
            {
                int fit = ListView_GetStringWidth(engines, tr("col_state").c_str());
                for (const auto& row : rows.texts)
                    if (row.size() > 4) fit = std::max(fit, ListView_GetStringWidth(engines, row[4].c_str()));
                RECT inside = {};
                GetClientRect(engines, &inside);
                int taken = 0;
                for (int i = 0; i < 4; ++i) taken += ListView_GetColumnWidth(engines, i);
                set_column_width(engines, 4, std::max(fit + px(16), static_cast<int>(inside.right) - taken));
            }

            void draw_bar(const DRAWITEMSTRUCT* d)
            {
                RECT r = d->rcItem;
                HBRUSH back = CreateSolidBrush(RGB(226, 229, 238));
                FillRect(d->hDC, &r, back);
                DeleteObject(back);
                const json& mem = st["engine"]["stats"]["memory"];
                const double budget = mem["vram_budget"].as_double(0);
                if (budget <= 0) return;
                const double total = budget + mem["pinned_bytes"].as_double(0) + mem["store_bytes"].as_double(0);
                const int width = r.right - r.left;
                int x = r.left;
                auto segment = [&](double bytes, COLORREF color) {
                    const int w = static_cast<int>(width * bytes / total);
                    if (w <= 0) return;
                    RECT s = { x, r.top, std::min(static_cast<int>(r.right), x + w), r.bottom };
                    HBRUSH b = CreateSolidBrush(color);
                    FillRect(d->hDC, &s, b);
                    DeleteObject(b);
                    x += w;
                };
                segment(mem["device_bytes"].as_double(0), RGB(61, 107, 255));
                segment(budget - mem["device_bytes"].as_double(0), RGB(190, 204, 255));
                segment(mem["pinned_bytes"].as_double(0), RGB(242, 181, 68));
                segment(mem["store_bytes"].as_double(0), RGB(124, 77, 255));
            }

            void on_result(WPARAM tag, const json& r) override
            {
                if (tag == R_HOST)
                {
                    if (r.contains("error"))
                    {
                        ui::set_text(status, widen(r["error"].as_string()));
                        return;
                    }
                    st = r["status"];
                    eng = r["engines"];
                    if (r["settings"].is_object()) settings = r["settings"];
                    show();
                    return;
                }
                ui::enable(apply, true);
                if (r.contains("error"))
                {
                    for (const auto& kv : previous.items()) settings[kv.first] = kv.second;
                    ui::message(hwnd, error_text(r), MB_OK | MB_ICONWARNING);
                    return;
                }
                ui::set_text(status, r["restart"].as_bool(false) ? tr("restart_note") : tr("saved"));
                // A new language applies to the tray at once; the window opens again in it.
                const std::string lang_code = r["language"].as_string();
                if (!lang_code.empty() && lang_code != current_language())
                {
                    set_language(lang_code);
                    service_client c = client;
                    DestroyWindow(hwnd);
                    show_host_window(c);
                    return;
                }
                // The tray reads the port again once the service restarted.
                if (r["restart"].as_bool(false)) SetTimer(hwnd, 2, 5000, nullptr);
            }

            LRESULT on_message(UINT msg, WPARAM wp, LPARAM lp, bool& handled) override
            {
                handled = false;
                if (msg == WM_DRAWITEM && wp == IDC_H_BAR)
                {
                    draw_bar(reinterpret_cast<DRAWITEMSTRUCT*>(lp));
                    handled = true;
                    return TRUE;
                }
                if (msg == WM_TIMER)
                {
                    if (wp == 2)
                    {
                        KillTimer(hwnd, 2);
                        client.reload();
                        loaded = false;
                    }
                    load();
                    handled = true;
                    return 0;
                }
                return 0;
            }

            void on_command(int id, int code) override
            {
                switch (id)
                {
                case IDC_H_OPEN: ui::open_url(widen(st["models_dir"].as_string())); break;
                case IDC_H_CHANGE:
                {
                    const std::wstring dir = ui::choose_folder(hwnd, widen(st["models_dir"].as_string()));
                    if (dir.empty()) break;
                    // The installer changes the folder with the rights it needs.
                    const std::wstring setup = widen(join_path(executable_dir(), "albireo-setup.exe"));
                    const std::wstring args = L"/models-dir " + quoted_argument(dir);
                    SHELLEXECUTEINFOW sei = { sizeof(sei) };
                    sei.lpVerb = L"runas";
                    sei.lpFile = setup.c_str();
                    sei.lpParameters = args.c_str();
                    sei.nShow = SW_HIDE;
                    ShellExecuteExW(&sei);
                    SetTimer(hwnd, 2, 8000, nullptr);
                    break;
                }
                case IDC_H_APPLY:
                {
                    json changes = json::object();
                    const long p = std::wcstol(ui::text_of(port).c_str(), nullptr, 10);
                    if (p >= 1024 && p <= 65535 && p != settings["port"].as_int(8443))
                        changes["port"] = static_cast<long long>(p);
                    const bool v = ui::checked(v2ram);
                    if (v != settings["v2ram"].as_bool(true)) changes["v2ram"] = v;
                    const long long vm = std::wcstoll(ui::text_of(v2max).c_str(), nullptr, 10);
                    if (vm != settings["v2ram_max_mb"].as_int(0)) changes["v2ram_max_mb"] = vm;
                    const char* codes[] = { "auto", "en", "fr", "de", "it", "es" };
                    const LRESULT sel = SendMessageW(lang, CB_GETCURSEL, 0, 0);
                    const std::string lang_code = sel >= 0 && sel < 6 ? codes[sel] : "auto";
                    if (lang_code != settings["language"].as_string()) changes["language"] = lang_code;
                    const bool u = ui::checked(updates);
                    if (u != settings["check_updates"].as_bool(true)) changes["check_updates"] = u;
                    const bool off = ui::checked(offline);
                    if (off != settings["offline"].as_bool(false)) changes["offline"] = off;
                    const std::string pr = narrow(ui::text_of(proxy));
                    if (pr != settings["proxy"].as_string()) changes["proxy"] = pr;
                    if (changes.empty())
                    {
                        ui::set_text(status, tr("saved"));
                        break;
                    }
                    ui::enable(apply, false);
                    service_client c = client;
                    previous = json::object();
                    for (const auto& kv : changes.items())
                    {
                        previous[kv.first] = settings[kv.first];
                        settings[kv.first] = kv.second;
                    }
                    request(R_APPLY, [c, changes]() {
                        json r;
                        std::string error;
                        if (!c.send("PUT", "/albireo/v1/settings", changes, r, error))
                        {
                            // The code of the error names its text in the language of the window.
                            const json& answer = r;
                            json failure = json::object();
                            failure["error"] = error;
                            failure["code"] = answer["error"]["code"];
                            json args = json::object();
                            args["port"] = changes["port"];
                            failure["args"] = args;
                            r = failure;
                        }
                        else if (changes.contains("language"))
                        {
                            r["language"] = changes["language"].as_string() == "auto" ? system_language()
                                : changes["language"].as_string();
                        }
                        return r;
                    });
                    break;
                }
                case IDC_H_OFFLINE:
                    ui::enable(updates, !ui::checked(offline));
                    ui::enable(proxy, !ui::checked(offline));
                    break;
                case IDC_H_PROXY:
                    // The service gives the password of the proxy as ********, which keeps it.
                    if (code == EN_CHANGE && ui::text_of(proxy).find(L"********") != std::wstring::npos)
                        ui::set_text(status, tr("proxy_mask_hint"));
                    break;
                case IDC_H_CLOSE: DestroyWindow(hwnd); break;
                default: break;
                }
            }
        };

        // The About window.
        enum { IDC_A_IMAGE = 3000, IDC_A_TITLE, IDC_A_TEXT, IDC_A_PROJECT, IDC_A_DOCS, IDC_A_CLOSE };

        class about_window : public tool_window
        {
        public:
            HWND title, text, project, docs, close;
            HBITMAP art = nullptr;
            int art_w = 0, art_h = 0;

            about_window() { min_size = SIZE{ 560, 470 }; }

            void build() override
            {
                title = add(ui::label(hwnd, widen(ALBIREO_FULL_NAME), IDC_A_TITLE), ui::font_kind::title);
                const std::wstring body = tr("about_version", { { "v", ALBIREO_VERSION }, { "e", ALBIREO_EDITION } })
                    + L"\n\n"
                    + tr("about_built") + L"\n\n"
                    + tr("about_license") + L"\n" + widen(ALBIREO_COPYRIGHT) + L"\n"
                    + widen(ALBIREO_WEBSITE_URL) + L" \u00B7 " + widen(ALBIREO_CONTACT_EMAIL);
                text = add(ui::label(hwnd, body, IDC_A_TEXT));
                project = add(ui::button(hwnd, tr("btn_project"), IDC_A_PROJECT));
                docs = add(ui::button(hwnd, tr("tray_docs"), IDC_A_DOCS));
                close = add(ui::button(hwnd, tr("btn_close"), IDC_A_CLOSE));
            }

            void layout(int w, int h) override
            {
                const int m = px(18);
                const SIZE s = ui::png_size(IDB_ABOUT);
                const int aw = w, ah = s.cx > 0 ? MulDiv(w, s.cy, s.cx) : 0;
                if (aw != art_w || ah != art_h)
                {
                    if (art) DeleteObject(art);
                    art = ui::load_png(IDB_ABOUT, aw, ah);
                    art_w = aw;
                    art_h = ah;
                    InvalidateRect(hwnd, nullptr, TRUE);
                }
                int y = ah + px(14);
                ui::place(title, m, y, w - 2 * m, px(40));
                y += px(44);
                ui::place(text, m, y, w - 2 * m, h - y - px(64));
                const int bh = px(30);
                ui::place(close, w - m - px(100), h - m - bh, px(100), bh);
                ui::place(docs, w - m - px(100) - px(8) - px(210), h - m - bh, px(210), bh);
                ui::place(project, m, h - m - bh, px(150), bh);
            }

            LRESULT on_message(UINT msg, WPARAM, LPARAM, bool& handled) override
            {
                handled = false;
                if (msg == WM_PAINT)
                {
                    PAINTSTRUCT ps;
                    HDC dc = BeginPaint(hwnd, &ps);
                    ui::draw_bitmap(dc, art, 0, 0);
                    EndPaint(hwnd, &ps);
                    handled = true;
                }
                return 0;
            }

            void on_result(WPARAM, const json&) override {}

            void on_command(int id, int) override
            {
                if (id == IDC_A_PROJECT) ui::open_url(widen(ALBIREO_PROJECT_URL));
                else if (id == IDC_A_DOCS) ui::open_url(widen(client.base_url() + "docs/"));
                else if (id == IDC_A_CLOSE) DestroyWindow(hwnd);
            }

            void on_destroy() override
            {
                if (art) DeleteObject(art);
                art = nullptr;
            }
        };

        template <typename T>
        void show(const service_client& client, const std::wstring& title, int w, int h, bool resizable)
        {
            if (T* existing = find_window<T>())
            {
                ShowWindow(existing->hwnd, SW_RESTORE);
                SetForegroundWindow(existing->hwnd);
                return;
            }
            T* t = new T();
            t->client = client;
            // The window polls the service on a kept connection of its own, beside that of the icon.
            t->client.own_connection();
            if (!t->create(title, w, h, resizable)) delete t;
        }
    }

    void show_models_window(const service_client& client)
    {
        // Wide enough for the seven columns of the list without a horizontal scroll bar.
        show<models_window>(client, tr("models_title"), 1040, 680, true);
    }

    void show_host_window(const service_client& client)
    {
        show<host_window>(client, tr("host_title"), 760, 766, false);
    }

    void show_about_window(const service_client& client)
    {
        show<about_window>(client, tr("about_title"), 600, 520, false);
    }

    bool dispatch_dialog_message(MSG* msg)
    {
        for (auto* w : open_windows)
            if (w->hwnd && IsDialogMessageW(w->hwnd, msg)) return true;
        return false;
    }

}
