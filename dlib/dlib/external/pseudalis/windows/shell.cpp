// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

/*
    pseudalis: the commands of the Explorer on the files Pseudalis protects, which the
    service carries out.

        pseudalis protect <file>...          anonymizes each file in its place
        pseudalis protect --copy <file>...   writes the anonymized copy beside each file,
                                             "name (anonymized).ext"
        pseudalis restore <file>...          puts the original back in the place of each file
        pseudalis restore --copy <file>...   writes the original beside each file,
                                             "name (restored).ext"

    A picture becomes the text file of its marker, "photo.jpg.txt", which its restoration
    turns back into the picture.  The original of an anonymized file comes from the vault of
    the service when the file is the one it made (its SHA-256, or its NTFS object identifier,
    which the service records with the place of the file); a file edited since gets its
    values back from its tags.  The files of the drive of Pseudalis are anonymized already:
    their originals are saved where the user chooses, outside the drive, which would
    anonymize them again.

    The results go to the icon of the session, which notifies them together; without the
    icon, a message box tells them, and from a console they are printed there.  The exit
    code is 0 when every file was processed, 1 otherwise.
*/

#include "ui.h"

#include <shellapi.h>
#include <shlobj.h>
#include <shobjidl.h>
#include <winioctl.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cwctype>
#include <string>
#include <vector>

#include "../common/file_kinds.h"
#include "../common/platform.h"
#include "../common/version.h"
#include "../host/crypto.h"
#include "service_client.h"
#include "strings.h"

namespace pseudalis
{

    namespace
    {
        // The mark of the results sent to the icon (WM_COPYDATA), and the largest file read.
        const ULONG_PTR results_mark = 0x50534431;
        const uint64_t max_file_bytes = 1ull << 30;

        struct outcome
        {
            bool ok = false;
            std::wstring text;
            std::string name;
            std::string error;
        };

        outcome success(const std::wstring& text, const std::string& name)
        {
            outcome o;
            o.ok = true;
            o.text = text;
            o.name = name;
            return o;
        }

        outcome failure(const std::string& name, const std::wstring& why)
        {
            outcome o;
            o.name = name;
            o.error = narrow(why);
            o.text = tr("shell_failed", { { "f", name }, { "e", o.error } });
            return o;
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
                const bool plain = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
                    c == '-' || c == '.' || c == '_' || c == '~';
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

        // A header of the service, percent-encoded.
        std::string percent_decoded(const std::string& s)
        {
            std::string out;
            for (size_t i = 0; i < s.size(); ++i)
            {
                if (s[i] == '%' && i + 2 < s.size() && std::isxdigit(static_cast<unsigned char>(s[i + 1])) &&
                    std::isxdigit(static_cast<unsigned char>(s[i + 2])))
                {
                    out += static_cast<char>(std::stoi(s.substr(i + 1, 2), nullptr, 16));
                    i += 2;
                }
                else
                {
                    out += s[i];
                }
            }
            return out;
        }

        std::string header(const service_client::raw_answer& a, const char* name)
        {
            const auto it = a.headers.find(name);
            return it == a.headers.end() ? std::string() : percent_decoded(it->second);
        }

        // The text of an answer of the service that failed.
        std::wstring answer_error(const service_client::raw_answer& a)
        {
            return error_text(failed_answer(json::parse_or_null(a.body), "status " + std::to_string(a.status)));
        }

        bool sharing_error(DWORD e)
        {
            return e == ERROR_SHARING_VIOLATION || e == ERROR_LOCK_VIOLATION;
        }

        std::wstring system_message(DWORD e)
        {
            wchar_t* text = nullptr;
            FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                nullptr, e, 0, reinterpret_cast<LPWSTR>(&text), 0, nullptr);
            std::wstring out = text ? text : L"";
            if (text) LocalFree(text);
            while (!out.empty() && std::wstring(L"\r\n .").find(out.back()) != std::wstring::npos) out.pop_back();
            return out.empty() ? L"error " + std::to_wstring(e) : out;
        }

        // What tells that a file changed: its identity, its size and the time of its last write.
        struct file_stamp
        {
            DWORD volume = 0, index_high = 0, index_low = 0, size_high = 0, size_low = 0;
            FILETIME written = {};

            bool operator==(const file_stamp& o) const
            {
                return volume == o.volume && index_high == o.index_high && index_low == o.index_low &&
                    size_high == o.size_high && size_low == o.size_low &&
                    written.dwLowDateTime == o.written.dwLowDateTime &&
                    written.dwHighDateTime == o.written.dwHighDateTime;
            }
        };

        bool stamp_of(HANDLE h, file_stamp& s)
        {
            BY_HANDLE_FILE_INFORMATION i;
            if (!GetFileInformationByHandle(h, &i)) return false;
            s.volume = i.dwVolumeSerialNumber;
            s.index_high = i.nFileIndexHigh;
            s.index_low = i.nFileIndexLow;
            s.size_high = i.nFileSizeHigh;
            s.size_low = i.nFileSizeLow;
            s.written = i.ftLastWriteTime;
            return true;
        }

        // Whether path is still the file read, as it was read.
        bool unchanged(const std::wstring& path, const file_stamp& read)
        {
            HANDLE h = CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES,
                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, 0, nullptr);
            if (h == INVALID_HANDLE_VALUE) return false;
            file_stamp now;
            const bool same = stamp_of(h, now) && now == read;
            CloseHandle(h);
            return same;
        }

        /*
            Reads a whole file, which no other program may write meanwhile, and its stamp, which
            tells later whether it changed since; error receives the code of Windows.
        */
        bool read_whole(const std::wstring& path, std::string& data, file_stamp& stamp, DWORD& error)
        {
            HANDLE h = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
            if (h == INVALID_HANDLE_VALUE)
            {
                error = GetLastError();
                return false;
            }
            LARGE_INTEGER size = {};
            bool ok = GetFileSizeEx(h, &size) != 0 && stamp_of(h, stamp);
            if (ok && static_cast<uint64_t>(size.QuadPart) > max_file_bytes)
            {
                error = ERROR_FILE_TOO_LARGE;
                ok = false;
            }
            if (ok)
            {
                data.resize(static_cast<size_t>(size.QuadPart));
                size_t done = 0;
                while (ok && done < data.size())
                {
                    DWORD got = 0;
                    const DWORD want = static_cast<DWORD>(std::min<size_t>(data.size() - done, 1u << 24));
                    ok = ReadFile(h, &data[done], want, &got, nullptr) != 0 && got > 0;
                    done += got;
                }
                if (!ok) error = GetLastError();
            }
            CloseHandle(h);
            return ok;
        }

        // Writes a new file, which must not exist, flushed to the disk.
        bool write_new(const std::wstring& path, const std::string& data, DWORD& error,
                       DWORD attributes = FILE_ATTRIBUTE_NORMAL)
        {
            HANDLE h = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, attributes, nullptr);
            if (h == INVALID_HANDLE_VALUE)
            {
                error = GetLastError();
                return false;
            }
            bool ok = true;
            size_t done = 0;
            while (ok && done < data.size())
            {
                DWORD put = 0;
                const DWORD want = static_cast<DWORD>(std::min<size_t>(data.size() - done, 1u << 24));
                ok = WriteFile(h, data.data() + done, want, &put, nullptr) != 0 && put == want;
                done += put;
            }
            if (ok) ok = FlushFileBuffers(h) != 0;
            if (!ok) error = GetLastError();
            CloseHandle(h);
            if (!ok) DeleteFileW(path.c_str());
            return ok;
        }

        // Deletes a file, a few times when another program holds it a moment (an antivirus).
        bool delete_soon(const std::wstring& path)
        {
            for (int attempt = 0; attempt < 10; ++attempt)
            {
                if (DeleteFileW(path.c_str()) || GetLastError() == ERROR_FILE_NOT_FOUND) return true;
                Sleep(200);
            }
            return false;
        }

        /*
            Puts data in the place of the file path: written beside it, hidden, then exchanged
            with it by ReplaceFile, which keeps its attributes, its permissions and its object
            identifier.  The previous content goes through a backup file, deleted at once,
            which also keeps it when ReplaceFile fails halfway: the file then comes back in its
            place, so that no failure ever loses it.
        */
        bool replace_content(const std::wstring& path, const std::string& data, DWORD& error)
        {
            const size_t slash = path.find_last_of(L'\\');
            const std::wstring dir = slash == std::wstring::npos ? L"." : path.substr(0, slash);
            const std::wstring stem = dir + L"\\~pseudalis-" + widen(random_hex(8));
            const std::wstring temp = stem + L".tmp", backup = stem + L".bak";
            if (!write_new(temp, data, error, FILE_ATTRIBUTE_HIDDEN)) return false;
            for (int attempt = 0; attempt < 10; ++attempt)
            {
                if (ReplaceFileW(path.c_str(), temp.c_str(), backup.c_str(), REPLACEFILE_IGNORE_MERGE_ERRORS, nullptr,
                        nullptr))
                {
                    delete_soon(backup);
                    return true;
                }
                error = GetLastError();
                /*
                    1177: the file lies under the name of the backup, the new content under its
                    own name: the new content takes the place, else the file comes back.
                */
                if (error == ERROR_UNABLE_TO_MOVE_REPLACEMENT_2)
                {
                    if (MoveFileExW(temp.c_str(), path.c_str(), MOVEFILE_WRITE_THROUGH))
                    {
                        delete_soon(backup);
                        return true;
                    }
                    error = GetLastError();
                    MoveFileExW(backup.c_str(), path.c_str(), MOVEFILE_WRITE_THROUGH);
                    break;
                }
                // 1175 and 1176 with a backup named: both files keep their names.
                if (!sharing_error(error) && error != ERROR_ACCESS_DENIED) break;
                Sleep(200);
            }
            delete_soon(temp);
            return false;
        }

        // The object identifier of a file of NTFS, made when it has none: 32 hexadecimal digits, or "".
        std::string object_id(const std::wstring& path)
        {
            HANDLE h = CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES | FILE_WRITE_ATTRIBUTES,
                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, 0, nullptr);
            if (h == INVALID_HANDLE_VALUE) return std::string();
            FILE_OBJECTID_BUFFER b = {};
            DWORD got = 0;
            const bool ok = DeviceIoControl(h, FSCTL_CREATE_OR_GET_OBJECT_ID, nullptr, 0, &b, sizeof(b), &got,
                nullptr) != 0;
            CloseHandle(h);
            if (!ok) return std::string();
            static const char hex[] = "0123456789abcdef";
            std::string out;
            for (unsigned char c : b.ObjectId)
            {
                out += hex[c >> 4];
                out += hex[c & 15];
            }
            return out;
        }

        // The name of a copy: "rapport (anonymized).docx", "photo (anonymized).jpg.txt".
        std::string copy_name(const std::string& original, const std::string& made, const std::string& suffix)
        {
            const std::string base = made.compare(0, original.size(), original) == 0 ? original : made;
            const std::string tail = made.substr(base == original ? original.size() : made.size());
            const std::string ext = name_extension(base);
            const std::string stem = base.substr(0, base.size() - ext.size());
            return stem + " (" + suffix + ")" + base.substr(stem.size()) + tail;
        }

        // A name free in dir: name itself, else "name (2).ext", "name (3).ext"...
        std::string free_name(const std::string& dir, const std::string& name)
        {
            if (GetFileAttributesW(widen(join_path(dir, name)).c_str()) == INVALID_FILE_ATTRIBUTES) return name;
            std::string ext = name_extension(name);
            // The text file of a picture keeps the extension of the picture before its own.
            if (ext == ".txt" && kind_of_name(name.substr(0, name.size() - 4)) == name_kind::picture)
                ext = name_extension(name.substr(0, name.size() - 4)) + ".txt";
            const std::string stem = name.substr(0, name.size() - ext.size());
            for (int i = 2; i < 10000; ++i)
            {
                const std::string n = stem + " (" + std::to_string(i) + ")" + ext;
                if (GetFileAttributesW(widen(join_path(dir, n)).c_str()) == INVALID_FILE_ATTRIBUTES) return n;
            }
            return name;
        }

        std::wstring full_path(const std::wstring& path)
        {
            const DWORD n = GetFullPathNameW(path.c_str(), 0, nullptr, nullptr);
            if (n == 0) return path;
            std::wstring out(n, L'\0');
            const DWORD m = GetFullPathNameW(path.c_str(), n, &out[0], nullptr);
            out.resize(m);
            return out;
        }

        // A folder of the user where to save the original of a file of the drive, or "".
        std::wstring choose_destination(const std::string& name)
        {
            std::wstring chosen;
            IFileSaveDialog* dialog = nullptr;
            if (FAILED(CoCreateInstance(CLSID_FileSaveDialog, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&dialog))))
                return chosen;
            dialog->SetTitle(tr("shell_save_title", { { "f", name } }).c_str());
            dialog->SetFileName(widen(name).c_str());
            DWORD options = 0;
            dialog->GetOptions(&options);
            dialog->SetOptions(options | FOS_OVERWRITEPROMPT | FOS_FORCEFILESYSTEM | FOS_NOREADONLYRETURN);
            PWSTR documents = nullptr;
            if (SHGetKnownFolderPath(FOLDERID_Documents, 0, nullptr, &documents) == S_OK)
            {
                IShellItem* folder = nullptr;
                if (SUCCEEDED(SHCreateItemFromParsingName(documents, nullptr, IID_PPV_ARGS(&folder))))
                {
                    dialog->SetFolder(folder);
                    folder->Release();
                }
            }
            if (documents) CoTaskMemFree(documents);
            if (SUCCEEDED(dialog->Show(nullptr)))
            {
                IShellItem* item = nullptr;
                if (SUCCEEDED(dialog->GetResult(&item)))
                {
                    PWSTR p = nullptr;
                    if (SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &p)) && p)
                    {
                        chosen = p;
                        CoTaskMemFree(p);
                    }
                    item->Release();
                }
            }
            dialog->Release();
            return chosen;
        }

        class commands
        {
        public:
            explicit commands(bool copy) : copy_(copy)
            {
                json info;
                std::string error;
                online_ = client_.get("/v1/info", info, error, 10);
                if (!online_) offline_error_ = error;
                const json& d = info["drive"];
                if (d["mounted"].as_bool(false)) drive_ = d["letter"].as_string();
            }

            outcome run(bool protect, const std::wstring& given)
            {
                const std::wstring path = full_path(given);
                const std::string name = narrow(path.substr(path.find_last_of(L'\\') + 1));
                if (!online_) return failure(name, error_text(failed_answer(json(), offline_error_)));
                if (!supported_name(name)) return failure(name, tr("shell_unsupported", { { "f", name } }));
                std::string data;
                DWORD e = 0;
                if (!read_whole(path, data, stamp_, e))
                {
                    if (sharing_error(e)) return failure(name, tr("shell_in_use", { { "f", name } }));
                    return failure(name, tr("shell_cannot_read", { { "f", name },
                        { "e", narrow(system_message(e)) } }));
                }
                const bool on_drive = !drive_.empty() && path.size() > 2 && path[1] == L':' &&
                    std::towupper(path[0]) == static_cast<wchar_t>(drive_[0]);
                if (protect)
                {
                    // Nothing to do: the drive holds anonymized files only.
                    if (on_drive) return success(tr("shell_on_drive", { { "l", drive_ } }), name);
                    return anonymize(path, name, data);
                }
                return restore(path, name, data, on_drive);
            }

        private:
            outcome anonymize(const std::wstring& path, const std::string& name, const std::string& data)
            {
                json found;
                std::string error;
                if (client_.get("/v1/lookup?sha256=" + sha256_hex(data), found, error) && found["exact"].as_bool(false))
                    return failure(name, tr("shell_already", { { "f", name } }));
                service_client::raw_answer a;
                if (!client_.post_bytes("/v1/anonymize?source=shell&name=" + query_value(name), data, a, error))
                    return failure(name, error_text(failed_answer(json(), error)));
                if (a.status != 200) return failure(name, answer_error(a));
                const std::string made = header(a, "x-pseudalis-name");
                const long long record = std::atoll(header(a, "x-pseudalis-record").c_str());
                const json report = json::parse_or_null(header(a, "x-pseudalis-report"));
                const std::string values = std::to_string(report["total"].as_int(0));

                const std::string dir = narrow(path.substr(0, path.find_last_of(L'\\')));
                std::wstring target = path;
                DWORD e = 0;
                if (copy_ || made != name)
                {
                    const std::string wanted = copy_ ? copy_name(name, made, tr_utf8("suffix_anonymized")) : made;
                    target = widen(join_path(dir, free_name(dir, wanted)));
                    if (!copy_ && !unchanged(path, stamp_))
                        return failure(name, tr("shell_changed", { { "f", name } }));
                    if (!write_new(target, a.body, e))
                        return failure(name, tr("shell_cannot_write", { { "f", narrow(file_of(target)) },
                            { "e", narrow(system_message(e)) } }));
                    // In its place, a picture leaves for the text of its marker.
                    if (!copy_ && !DeleteFileW(path.c_str()))
                        return failure(name, tr("shell_cannot_write", { { "f", name },
                            { "e", narrow(system_message(GetLastError())) } }));
                }
                else if (!unchanged(path, stamp_))
                {
                    // Saved meanwhile: its new content is not the one protected, and stays.
                    return failure(name, tr("shell_changed", { { "f", name } }));
                }
                else if (!replace_content(path, a.body, e))
                {
                    if (sharing_error(e)) return failure(name, tr("shell_in_use", { { "f", name } }));
                    return failure(name, tr("shell_cannot_write", { { "f", name },
                        { "e", narrow(system_message(e)) } }));
                }
                if (record > 0)
                {
                    // The place of the file and its object identifier find its original again.
                    json body = json::object(), out;
                    body["object_id"] = object_id(target);
                    body["path"] = narrow(target);
                    client_.send("POST", "/v1/records/" + std::to_string(record) + "/location", body, out, error);
                }
                const std::string shown = narrow(file_of(target));
                return success(tr(copy_ ? "shell_protected_copy" : "shell_protected", { { "f", shown },
                    { "n", values } }), shown);
            }

            outcome restore(const std::wstring& path, const std::string& name, const std::string& data, bool on_drive)
            {
                service_client::raw_answer a;
                std::string error;
                const std::string oid = on_drive ? std::string() : object_id(path);
                if (!client_.post_bytes("/v1/restore?name=" + query_value(name) + "&object_id=" + oid, data, a, error))
                    return failure(name, error_text(failed_answer(json(), error)));
                if (a.status != 200) return failure(name, answer_error(a));
                const std::string how = header(a, "x-pseudalis-restored");
                const std::string original = header(a, "x-pseudalis-name");
                const json report = json::parse_or_null(header(a, "x-pseudalis-report"));
                const std::string values = std::to_string(report["tags_restored"].as_int(0));

                std::wstring target = path;
                DWORD e = 0;
                if (on_drive)
                {
                    // The drive would anonymize the original again: it goes where the user says.
                    target = choose_destination(original.empty() ? name : original);
                    // Cancelled by the user: nothing to tell.
                    if (target.empty()) return success(std::wstring(), name);
                    const std::string shown = narrow(file_of(target));
                    const bool exists = GetFileAttributesW(target.c_str()) != INVALID_FILE_ATTRIBUTES;
                    if (exists ? !replace_content(target, a.body, e) : !write_new(target, a.body, e))
                        return failure(shown, tr("shell_cannot_write", { { "f", shown },
                            { "e", narrow(system_message(e)) } }));
                    return success(tr("shell_restored_copy", { { "f", shown } }), shown);
                }
                const std::string dir = narrow(path.substr(0, path.find_last_of(L'\\')));
                /*
                    The original keeps the name of the file in its place, unless it is of another
                    kind: the text file of a picture gives back the picture under its own name.
                    A copy takes the name of the original.
                */
                const bool same_kind = how == "tags" || original.empty() ||
                    name_extension(original) == name_extension(name);
                const std::string base = (copy_ || !same_kind) && !original.empty() ? original : name;
                if (copy_ || !same_kind)
                {
                    const std::string wanted = copy_ ? copy_name(base, base, tr_utf8("suffix_restored")) : base;
                    target = widen(join_path(dir, free_name(dir, wanted)));
                    if (!copy_ && !unchanged(path, stamp_))
                        return failure(name, tr("shell_changed", { { "f", name } }));
                    if (!write_new(target, a.body, e))
                        return failure(name, tr("shell_cannot_write", { { "f", narrow(file_of(target)) },
                            { "e", narrow(system_message(e)) } }));
                    if (!copy_ && !DeleteFileW(path.c_str()))
                        return failure(name, tr("shell_cannot_write", { { "f", name },
                            { "e", narrow(system_message(GetLastError())) } }));
                }
                else if (!unchanged(path, stamp_))
                {
                    return failure(name, tr("shell_changed", { { "f", name } }));
                }
                else if (!replace_content(path, a.body, e))
                {
                    if (sharing_error(e)) return failure(name, tr("shell_in_use", { { "f", name } }));
                    return failure(name, tr("shell_cannot_write", { { "f", name },
                        { "e", narrow(system_message(e)) } }));
                }
                const std::string shown = narrow(file_of(target));
                if (how == "tags")
                    return success(tr("shell_restored_tags", { { "f", shown }, { "n", values } }), shown);
                return success(tr(copy_ ? "shell_restored_copy" : "shell_restored", { { "f", shown } }), shown);
            }

            static std::wstring file_of(const std::wstring& path)
            {
                const size_t slash = path.find_last_of(L'\\');
                return slash == std::wstring::npos ? path : path.substr(slash + 1);
            }

            bool copy_;
            file_stamp stamp_;      // of the file being processed, as it was read
            service_client client_;
            bool online_ = false;
            std::string offline_error_;
            std::string drive_;
        };

        std::string language_setting()
        {
            std::string text;
            if (read_file(join_path(data_dir(), "settings.json"), text, 1u << 20))
                return json::parse_or_null(text)["language"].str("auto");
            return "auto";
        }

        // Sends a result to the icon of the session; false without icon.
        bool send_to_tray(const outcome& o)
        {
            HWND tray = FindWindowW(L"PseudalisTrayWindow", nullptr);
            if (!tray) return false;
            json j = json::object();
            j["ok"] = o.ok;
            j["text"] = narrow(o.text);
            j["name"] = o.name;
            j["error"] = o.error;
            const std::string body = j.dump();
            COPYDATASTRUCT cd = {};
            cd.dwData = results_mark;
            cd.cbData = static_cast<DWORD>(body.size());
            cd.lpData = const_cast<char*>(body.data());
            DWORD_PTR answer = 0;
            return SendMessageTimeoutW(tray, WM_COPYDATA, 0, reinterpret_cast<LPARAM>(&cd), SMTO_ABORTIFHUNG, 5000,
                &answer) != 0;
        }

        void print(const std::wstring& line)
        {
            HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
            const std::wstring text = line + L"\r\n";
            DWORD written = 0;
            if (!WriteConsoleW(out, text.c_str(), static_cast<DWORD>(text.size()), &written, nullptr))
            {
                const std::string utf8 = narrow(text);
                WriteFile(out, utf8.data(), static_cast<DWORD>(utf8.size()), &written, nullptr);
            }
        }

        const wchar_t usage[] =
            L"Pseudalis " PSEUDALIS_VERSION L"\r\n"
            L"  pseudalis protect [--copy] <file>...   anonymize each file, in its place or as a copy\r\n"
            L"  pseudalis restore [--copy] <file>...   restore the original, in its place or as a copy";
    }

}

int WINAPI wWinMain(HINSTANCE, HINSTANCE, PWSTR, int)
{
    SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_SYSTEM32);
    using namespace pseudalis;
    // Started from a console, the results are printed there.
    const bool console = AttachConsole(ATTACH_PARENT_PROCESS) != 0;
    int argc = 0;
    LPWSTR* argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    std::vector<std::wstring> args;
    for (int i = 1; argv && i < argc; ++i) args.push_back(argv[i]);
    if (argv) LocalFree(argv);

    set_language(language_setting());
    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    ui::init();

    const bool protect = !args.empty() && args[0] == L"protect";
    const bool restore = !args.empty() && args[0] == L"restore";
    bool copy = false;
    std::vector<std::wstring> files;
    for (size_t i = 1; i < args.size(); ++i)
    {
        if (args[i] == L"--copy") copy = true;
        else files.push_back(args[i]);
    }
    if ((!protect && !restore) || files.empty())
    {
        if (console) print(usage);
        else ui::message(nullptr, usage, MB_OK | MB_ICONINFORMATION);
        return 2;
    }

    commands c(copy);
    int code = 0;
    for (const auto& f : files)
    {
        const outcome o = c.run(protect, f);
        if (!o.ok) code = 1;
        if (o.text.empty()) continue;
        if (console) print(o.text);
        else if (!send_to_tray(o)) ui::message(nullptr, o.text, MB_OK | (o.ok ? MB_ICONINFORMATION : MB_ICONWARNING));
    }
    CoUninitialize();
    return code;
}
