// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "tray_windows.h"

#include "ui.h"

#include <shellapi.h>
#include <shobjidl.h>

#include <algorithm>
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

        // The error of an answer of the service, or of a download.
        std::wstring error_text(const json& r)
        {
            return coded_text("err_", r["code"], r["args"], r["sizes"], r["error"].as_string());
        }

        // Why an engine cannot run on this computer.
        std::wstring reason_text(const json& engine)
        {
            return coded_text("why_", engine["reason_code"], engine["reason_args"], json(),
                engine["reason"].as_string());
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

        // The models window.
        enum
        {
            IDC_INTRO = 1000, IDC_SPACE, IDC_LIST, IDC_DETAILS, IDC_PROGRESS, IDC_STATUS, IDC_DOWNLOAD, IDC_ACTIVATE,
            IDC_DEACTIVATE, IDC_VERIFY, IDC_DELETE, IDC_IMPORT, IDC_REFRESH, IDC_CLOSE,
            IDM_ROLE_GENERAL = 1100, IDM_ROLE_CODE, IDM_ROLE_SECURITY, IDM_ROLE_EMBEDDING, IDM_IMPORT_FILES,
            IDM_IMPORT_FOLDER
        };
        enum { R_LIST = 1, R_ACTION, R_VERIFY, R_IMPORT, R_UPLOAD_DONE, R_CATALOG };

        class models_window : public tool_window
        {
        public:
            HWND intro, space, list, details, progress, status, download, activate, deactivate, verify, remove, import,
                refresh, close;
            json overview;
            std::vector<std::string> ids;
            std::string selected;
            bool busy = false;
            bool uploading = false;
            bool showing_download = false;   // the status line shows the progress of a download

            models_window() { min_size = SIZE{ 760, 520 }; }

            void build() override
            {
                intro = add(ui::label(hwnd, tr("models_intro"), IDC_INTRO, SS_LEFT), ui::font_kind::normal);
                space = add(ui::label(hwnd, L"", IDC_SPACE, SS_RIGHT), ui::font_kind::note);
                list = add(ui::listview(hwnd, IDC_LIST));
                const char* columns[] = { "col_model", "col_roles", "col_inputs", "col_size", "col_state",
                    "col_active" };
                // The last column takes the width the others leave (layout).
                const int widths[] = { 200, 165, 90, 75, 135, 180 };
                for (int i = 0; i < 6; ++i)
                {
                    std::wstring text = tr(columns[i]);
                    LVCOLUMNW c = {};
                    c.mask = LVCF_TEXT | LVCF_WIDTH | LVCF_SUBITEM;
                    c.pszText = &text[0];
                    c.cx = px(widths[i]);
                    c.iSubItem = i;
                    ListView_InsertColumn(list, i, &c);
                }
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
                ShowWindow(progress, SW_HIDE);
                SetTimer(hwnd, 1, 1000, nullptr);
                update_buttons();
                load();
            }

            void layout(int w, int h) override
            {
                // The actions on the selected model under the list, the window commands at the bottom.
                const int m = px(16), g = px(8), bh = px(30);
                // The introduction, then the origin and the state of the catalog.
                // Four lines: the introduction and, offline, the origin of the models.
                ui::place(intro, m, m, w - 2 * m - px(260), px(66));
                ui::place(space, w - m - px(250), m, px(250), px(36));
                const int list_top = m + px(72);
                const int details_h = px(100);
                const int bottom = h - m - bh;
                const int status_top = bottom - g - px(20);
                const int details_top = status_top - g - details_h;
                const int actions_top = details_top - g - bh;
                ui::place(list, m, list_top, w - 2 * m, actions_top - g - list_top);
                RECT inside = {};
                GetClientRect(list, &inside);
                int taken = 0;
                for (int i = 0; i < 5; ++i) taken += ListView_GetColumnWidth(list, i);
                ListView_SetColumnWidth(list, 5, std::max<int>(px(140), inside.right - taken));
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
                    if (!c.get("/albireo/v1/models?lang=" + lang, r, error))
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
                    if (m["builtin"].as_bool(false)) return tr("state_builtin");
                    const bool outdated = m.contains("up_to_date") && !m["up_to_date"].as_bool(true);
                    return outdated ? tr("state_update") : tr("state_installed");
                }
                return tr("state_available");
            }

            // What a model reads: text, images, sound; the attachments of the web
            // application follow them.
            std::wstring inputs_text(const json& m) const
            {
                std::wstring text;
                for (const auto& i : m["inputs"].elements())
                {
                    const std::string kind = i.as_string();
                    const std::wstring name = kind == "image" ? tr("input_image") : kind == "audio" ? tr("input_audio")
                        : kind == "text" ? tr("input_text") : widen(kind);
                    text += (text.empty() ? L"" : L", ") + name;
                }
                return text.empty() ? tr("input_text") : text;
            }

            void fill_list()
            {
                // The same models in the same order keep their rows, and the list its scroll position:
                // only the texts that changed are written.
                std::vector<std::string> now;
                for (const auto& m : overview["models"].elements()) now.push_back(m["id"].as_string());
                const bool rebuild = now != ids;
                ids = now;
                SendMessageW(list, WM_SETREDRAW, FALSE, 0);
                if (rebuild) ListView_DeleteAllItems(list);
                auto cell = [this](int row, int column, std::wstring text) {
                    wchar_t current[512] = {};
                    ListView_GetItemText(list, row, column, current, 512);
                    if (text != current) ListView_SetItemText(list, row, column, &text[0]);
                };
                int row = 0;
                for (const auto& m : overview["models"].elements())
                {
                    const std::string id = m["id"].as_string();
                    if (rebuild)
                    {
                        LVITEMW it = {};
                        it.mask = LVIF_PARAM;
                        it.iItem = row;
                        it.lParam = row;
                        ListView_InsertItem(list, &it);
                    }
                    cell(row, 0, widen(m["name"].str(id)));
                    std::wstring roles;
                    const json& listed = m["roles"];
                    if (listed.size() == 3)
                        roles = tr("roles_all");
                    else
                        for (const auto& r : listed.elements())
                            roles += (roles.empty() ? L"" : L", ") + role_name(r.as_string());
                    cell(row, 1, roles);
                    cell(row, 2, inputs_text(m));
                    const unsigned long long size = static_cast<unsigned long long>(m["size"].as_int(0));
                    cell(row, 3, size ? widen(ui::format_size(size)) : std::wstring());
                    cell(row, 4, state_text(m));
                    std::wstring act;
                    const std::vector<std::string> active = active_roles(id);
                    if (active.size() == 3)
                        act = tr("roles_all");
                    else
                        for (const auto& r : active) act += (act.empty() ? L"" : L", ") + role_name(r);
                    // The model that comes with Albireo serves the work in the background.
                    if (id == overview["background"].as_string())
                        act += (act.empty() ? L"" : L", ") + tr("role_background");
                    cell(row, 5, act);
                    const UINT focus = LVIS_SELECTED | LVIS_FOCUSED;
                    if (rebuild && id == selected) ListView_SetItemState(list, row, focus, focus);
                    ++row;
                }
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
                    ShowWindow(progress, SW_SHOW);
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
                    ShowWindow(progress, SW_HIDE);
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
                    if (!c.send(m.c_str(), path, body, r, error, 120) && !r.contains("code"))
                    {
                        r = json::object();
                        r["error"] = error;
                        r["code"] = "service";
                        json args = json::object();
                        args["e"] = error;
                        r["args"] = args;
                    }
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

            // What an import installed: the catalog, the models, the update, the errors.
            std::wstring import_text(const json& r) const
            {
                std::wstring text = tr("import_done", { { "n", std::to_string(r["imported"].size()) } });
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
                else if (u["error"].is_string()) text += L"\n" + widen(u["error"].as_string());
                for (const auto& e : r["errors"].elements()) text += L"\n" + widen(e.as_string());
                return text;
            }

            void on_result(WPARAM tag, const json& r) override
            {
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
                    ShowWindow(progress, SW_HIDE);
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

            void role_menu()
            {
                HMENU m = CreatePopupMenu();
                const auto act = active_roles(selected);
                // An embedding model serves the document search only; the language models
                // answer for the other functions.
                const char* roles[] = { "general", "code", "security", "embedding" };
                const bool embedding = embedding_model(selected);
                for (int i = 0; i < 4; ++i)
                {
                    if ((i == 3) != embedding) continue;
                    const bool on = std::find(act.begin(), act.end(), roles[i]) != act.end();
                    const UINT flags = MF_STRING | (on ? MF_CHECKED : 0);
                    AppendMenuW(m, flags, IDM_ROLE_GENERAL + i, role_name(roles[i]).c_str());
                }
                RECT r;
                GetWindowRect(activate, &r);
                const UINT flags = TPM_RETURNCMD | TPM_LEFTALIGN | TPM_BOTTOMALIGN;
                const UINT cmd = TrackPopupMenuEx(m, flags, r.left, r.top, hwnd, nullptr);
                DestroyMenu(m);
                if (cmd >= IDM_ROLE_GENERAL && cmd <= IDM_ROLE_EMBEDDING)
                {
                    json body = json::object();
                    body["role"] = roles[cmd - IDM_ROLE_GENERAL];
                    action("POST", "/albireo/v1/models/" + selected + "/activate", body);
                }
            }

            void import_menu()
            {
                HMENU m = CreatePopupMenu();
                AppendMenuW(m, MF_STRING, IDM_IMPORT_FILES, tr("import_files").c_str());
                AppendMenuW(m, MF_STRING, IDM_IMPORT_FOLDER, tr("btn_import").c_str());
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
                    if (!c.send("POST", "/albireo/v1/import", body, r, error, 3600))
                    {
                        r = json::object();
                        r["error"] = error;
                    }
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
                ShowWindow(progress, SW_SHOW);
                update_buttons();
                service_client c = client;
                HWND target = hwnd;
                ui::run_async(hwnd, WM_RESULT, R_UPLOAD_DONE, [c, files, target]() {
                    json all = json::object();
                    json imported = json::array(), errors = json::array();
                    for (const auto& f : files)
                    {
                        const std::string name = file_name(f);
                        json r;
                        std::string error;
                        const bool ok = c.upload("/albireo/v1/models/upload?name=" + name, f,
                            [target, name](uint64_t done, uint64_t total) {
                                std::string* msg = new std::string(name);
                                const WPARAM permille = total ? static_cast<WPARAM>(1000 * done / total) : 0;
                                if (!PostMessageW(target, WM_PROGRESS, permille, reinterpret_cast<LPARAM>(msg)))
                                    delete msg;
                                return IsWindow(target) != FALSE;
                            }, r, error);
                        if (ok)
                            for (const auto& id : r["imported"].elements()) imported.push_back(id);
                        for (const auto& e : r["errors"].elements()) errors.push_back(e);
                        if (!ok && r["errors"].empty()) errors.push_back(name + ": " + error);
                    }
                    all["imported"] = imported;
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
                    if (n->idFrom == IDC_LIST && n->code == LVN_ITEMCHANGED)
                    {
                        const NMLISTVIEW* lv = reinterpret_cast<const NMLISTVIEW*>(lp);
                        const bool known = lv->iItem >= 0 && lv->iItem < static_cast<int>(ids.size());
                        if ((lv->uNewState & LVIS_SELECTED) && known)
                        {
                            selected = ids[static_cast<size_t>(lv->iItem)];
                            show_details();
                            update_buttons();
                        }
                    }
                    handled = true;
                    return 0;
                }
                return 0;
            }

            void on_command(int id, int) override
            {
                switch (id)
                {
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
                        {
                            r = json::object();
                            r["error"] = error;
                        }
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
                const int widths[] = { 110, 60, 220, 90, 160 };
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
                ui::place(groups[0], m, y, inner, px(150));
                ui::place(info_keys, m + px(12), y + px(24), px(150), px(118));
                ui::place(info_values, m + px(170), y + px(24), inner - px(182), px(118));
                y += px(150) + g;
                // Engine.
                ui::place(groups[1], m, y, inner, px(170));
                ui::place(engine, m + px(12), y + px(24), inner - px(24), lh);
                ui::place(engines, m + px(12), y + px(48), inner - px(24), px(110));
                y += px(170) + g;
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
                    if (!c.get("/albireo/v1/status", s, error))
                    {
                        r["error"] = error;
                        return r;
                    }
                    r["status"] = s;
                    if (c.get("/albireo/v1/engines", e, error)) r["engines"] = e;
                    if (c.get("/albireo/v1/settings", set, error)) r["settings"] = set;
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

                const json& e = st["engine"];
                std::string current = e["variant"].as_string();
                if (!e["device"].as_string().empty()) current += " (" + e["device"].as_string() + ")";
                const double tps = e["stats"]["generation_tokens_per_second"].as_double(0);
                std::wstring line = tr("lbl_engine") + L": " + widen(current);
                if (tps > 0.05)
                {
                    char buf[32];
                    snprintf(buf, sizeof(buf), "%.1f", tps);
                    line += L"  ·  " + tr("tokens_s", { { "n", buf } });
                }
                if (!e["error"].as_string().empty()) line += L"  ·  " + widen(e["error"].as_string());
                ui::set_text(engine, line);

                // The list is rebuilt out of sight, then shown at once.
                SendMessageW(engines, WM_SETREDRAW, FALSE, 0);
                ListView_DeleteAllItems(engines);
                int row = 0;
                for (const auto& x : eng["engines"].elements())
                {
                    std::wstring name = widen(x["variant"].as_string());
                    LVITEMW it = {};
                    it.mask = LVIF_TEXT;
                    it.iItem = row;
                    it.pszText = &name[0];
                    ListView_InsertItem(engines, &it);
                    std::wstring v = widen(x["version"].as_string());
                    ListView_SetItemText(engines, row, 1, &v[0]);
                    std::string devs;
                    for (const auto& d : x["devices"].elements())
                        devs += (devs.empty() ? "" : ", ") + d["name"].as_string();
                    std::wstring dv = widen(devs);
                    ListView_SetItemText(engines, row, 2, &dv[0]);
                    std::wstring state = x["current"].as_bool(false) ? tr("in_use")
                        : x["usable"].as_bool(false) ? tr("usable") : tr("unavailable");
                    if (!x["usable"].as_bool(false) && x["reason"].is_string())
                        state += L": " + reason_text(x);
                    ListView_SetItemText(engines, row, 4, &state[0]);
                    std::string measure;
                    for (const auto& o : eng["options"].elements())
                        if (o["variant"].as_string() == x["variant"].as_string()
                            && o["measure"]["tokens_per_second"].is_number())
                        {
                            char buf[48];
                            snprintf(buf, sizeof(buf), "%.1f tok/s", o["measure"]["tokens_per_second"].as_double());
                            measure += (measure.empty() ? "" : ", ") + std::string(buf);
                        }
                    std::wstring ms = widen(measure);
                    ListView_SetItemText(engines, row, 3, &ms[0]);
                    ++row;
                }
                // The state takes the width of its longest text, or the rest of the list.
                ListView_SetColumnWidth(engines, 4, LVSCW_AUTOSIZE);
                const int fit = ListView_GetColumnWidth(engines, 4);
                ListView_SetColumnWidth(engines, 4, LVSCW_AUTOSIZE_USEHEADER);
                if (ListView_GetColumnWidth(engines, 4) < fit) ListView_SetColumnWidth(engines, 4, fit);
                SendMessageW(engines, WM_SETREDRAW, TRUE, 0);
                InvalidateRect(engines, nullptr, TRUE);

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
                InvalidateRect(bar, nullptr, TRUE);
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
                    + tr("about_built") + L"\n" + tr("about_created", { { "a", ALBIREO_AUTHOR_NAME } }) + L"\n"
                    + tr("about_company", { { "c", ALBIREO_COMPANY_NAME } }) + L"\n\n" + tr("about_license") + L"\n"
                    + widen(ALBIREO_COPYRIGHT);
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
            if (!t->create(title, w, h, resizable)) delete t;
        }
    }

    void show_models_window(const service_client& client)
    {
        show<models_window>(client, tr("models_title"), 900, 620, true);
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
