// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

/*
    pseudalis: the commands of the Explorer on the files and the folders Pseudalis protects,
    which the service carries out.

        pseudalis protect <file or folder>...          anonymizes each one in its place
        pseudalis protect --copy <file or folder>...   writes the anonymized copy beside each one,
                                                       "name (anonymized).ext"
        pseudalis restore <file or folder>...          puts the original back in the place of each one
        pseudalis restore --copy <file or folder>...   writes the original beside each one,
                                                       "name (restored).ext"

    A picture becomes the text file of its marker, "photo.jpg.txt", which its restoration
    turns back into the picture.  The name of a file is anonymized too, its values written
    without tags, and the original comes back under its own name.  The original of an
    anonymized file comes from the vault of the service when the file is the one it made
    (its SHA-256, or its NTFS object identifier, which the service records with the place
    of the file); a file edited since gets its values back from its tags, and those of its
    name and of its properties from the substitutes the service recorded.  The files of the
    drive of Pseudalis are anonymized already: their originals are saved where the user
    chooses, outside the drive, which would anonymize them again.  The drive is known by its
    letter and by its folder in the data folder.

    A folder is processed file by file, its folders included, each one taking its name
    anonymized, or restored, once its files are done: in its place, or as a copy that
    receives the same tree.  A file of another kind stays as it is in its place and out of an
    anonymized copy, which holds anonymized files only; a restored copy takes it as it is.
    The symbolic links and the junctions are passed, and the processing stops beyond
    max_folder_files files or max_folder_depth levels.

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
        // The bounds of the processing of a folder: the levels of its folders and the files it meets.
        const int max_folder_depth = 64;
        const long long max_folder_files = 100000;

        struct outcome
        {
            bool ok = false;
            std::wstring text;
            std::string name;
            std::string error;
            std::string code;           // the code of the service, for a failure it answered
            bool already = false;       // the file is anonymized already
            long long values = 0;       // the values replaced, or put back from the tags
        };

        outcome success(const std::wstring& text, const std::string& name)
        {
            outcome o;
            o.ok = true;
            o.text = text;
            o.name = name;
            return o;
        }

        outcome failure(const std::string& name, const std::wstring& why, const std::string& code = std::string())
        {
            outcome o;
            o.name = name;
            o.error = narrow(why);
            o.code = code;
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

        // The failure an answer of the service tells, with its code.
        outcome answer_failure(const std::string& name, const service_client::raw_answer& a)
        {
            const json body = json::parse_or_null(a.body);
            return failure(name, error_text(failed_answer(body, "status " + std::to_string(a.status))),
                body["code"].as_string());
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

        /*
            A name without the suffix that Anonymize as a copy gave it, in any of the languages, and
            the number that a name taken added after it: "rapport (anonymized) (2).docx" and
            "Dossier (anonymisé)" give "rapport.docx" and "Dossier".  Its restored copy takes its
            own suffix in its place.
        */
        std::string without_copy_suffix(const std::string& name)
        {
            const std::string ext = name_extension(name);
            const std::string stem = name.substr(0, name.size() - ext.size());
            size_t end = stem.size();
            const size_t open = stem.rfind(" (");
            if (open != std::string::npos && end > open + 3 && stem[end - 1] == ')' &&
                stem.find_first_not_of("0123456789", open + 2) == end - 1)
                end = open;
            for (const std::string& s : texts_of("suffix_anonymized"))
            {
                const std::string tail = " (" + s + ")";
                if (end > tail.size() && stem.compare(end - tail.size(), tail.size(), tail) == 0)
                    return stem.substr(0, end - tail.size()) + ext;
            }
            return name;
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

        /*
            Where the user saves a file that cannot stay on its drive, or "": title, then the name
            proposed; with an empty name, the folder where the originals of a folder go.  The
            dialog opens on the documents of the user.
        */
        std::wstring choose_destination(const std::wstring& title, const std::string& name)
        {
            std::wstring chosen;
            const bool folder = name.empty();
            IFileDialog* dialog = nullptr;
            if (FAILED(CoCreateInstance(folder ? CLSID_FileOpenDialog : CLSID_FileSaveDialog, nullptr,
                    CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&dialog))))
                return chosen;
            dialog->SetTitle(title.c_str());
            if (!folder) dialog->SetFileName(widen(name).c_str());
            DWORD options = 0;
            dialog->GetOptions(&options);
            dialog->SetOptions(options | FOS_FORCEFILESYSTEM | FOS_NOREADONLYRETURN |
                (folder ? FOS_PICKFOLDERS | FOS_PATHMUSTEXIST : FOS_OVERWRITEPROMPT));
            PWSTR documents = nullptr;
            if (SHGetKnownFolderPath(FOLDERID_Documents, 0, nullptr, &documents) == S_OK)
            {
                IShellItem* start = nullptr;
                if (SUCCEEDED(SHCreateItemFromParsingName(documents, nullptr, IID_PPV_ARGS(&start))))
                {
                    dialog->SetFolder(start);
                    start->Release();
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

        // The names a folder holds for the system or for a program at work, which stay as they are.
        bool passed_name(const std::wstring& name, DWORD attributes)
        {
            if ((attributes & FILE_ATTRIBUTE_SYSTEM) && (attributes & FILE_ATTRIBUTE_HIDDEN)) return true;
            std::wstring n = name;
            for (wchar_t& c : n) c = static_cast<wchar_t>(std::towlower(c));
            return n == L"desktop.ini" || n == L"thumbs.db" || n == L"ehthumbs.db" || n == L".ds_store" ||
                n.compare(0, 2, L"~$") == 0 || n.compare(0, 11, L"~pseudalis-") == 0;
        }

        struct folder_entry
        {
            std::wstring name;
            bool folder = false;
        };

        /*
            The files and the folders of dir, by name.  A symbolic link or a junction is passed: it
            may lead out of the folder, or back into it.  A file of a cloud (OneDrive) is read like
            the others, which brings it to the computer.
        */
        bool list_folder(const std::wstring& dir, std::vector<folder_entry>& out, DWORD& error)
        {
            out.clear();
            WIN32_FIND_DATAW d;
            HANDLE h = FindFirstFileExW((dir + L"\\*").c_str(), FindExInfoBasic, &d, FindExSearchNameMatch, nullptr,
                FIND_FIRST_EX_LARGE_FETCH);
            if (h == INVALID_HANDLE_VALUE)
            {
                error = GetLastError();
                return false;
            }
            do
            {
                const std::wstring n = d.cFileName;
                if (n == L"." || n == L"..") continue;
                if ((d.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) &&
                    (d.dwReserved0 == IO_REPARSE_TAG_SYMLINK || d.dwReserved0 == IO_REPARSE_TAG_MOUNT_POINT))
                    continue;
                if (passed_name(n, d.dwFileAttributes)) continue;
                folder_entry e;
                e.name = n;
                e.folder = (d.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
                out.push_back(e);
            } while (FindNextFileW(h, &d));
            error = GetLastError();
            FindClose(h);
            if (error != ERROR_NO_MORE_FILES) return false;
            std::sort(out.begin(), out.end(), [](const folder_entry& a, const folder_entry& b) {
                return _wcsicmp(a.name.c_str(), b.name.c_str()) < 0;
            });
            return true;
        }

        // Whether path lies in the folder dir, or is dir.
        bool within(const std::wstring& path, const std::wstring& dir)
        {
            return path.size() >= dir.size() && _wcsnicmp(path.c_str(), dir.c_str(), dir.size()) == 0 &&
                (path.size() == dir.size() || path[dir.size()] == L'\\');
        }

        // What the processing of a folder did, told at its end.
        struct folder_tally
        {
            long long done = 0;         // the files anonymized, or restored
            long long values = 0;       // the values they had replaced
            long long already = 0;      // the files anonymized already
            long long other = 0;        // the files of another kind
            long long failed = 0;
            long long files = 0;        // the files met
            std::vector<std::string> others;    // the first files of another kind
            std::string failed_name;    // the first failure
            std::string failed_error;
            bool stopped = false;       // more than max_folder_files files
            bool too_deep = false;      // folders deeper than max_folder_depth levels
        };

        // A drive of the service, from its state: its letter, whether Windows shows it, its folder.
        struct drive_place
        {
            std::string letter;     // empty when the service runs no such drive
            bool mounted = false;
            std::wstring folder;

            explicit drive_place(const json& d)
                : letter(d["letter"].as_string()), mounted(d["mounted"].as_bool(false)),
                  folder(widen(d["folder"].as_string()))
            {
            }

            // Whether path lies on the drive: under its letter, or in its folder of the data folder.
            bool holds(const std::wstring& path) const
            {
                if (letter.empty()) return false;
                if (mounted && path.size() > 2 && path[1] == L':' &&
                    std::towupper(path[0]) == static_cast<wchar_t>(std::toupper(static_cast<unsigned char>(letter[0]))))
                    return true;
                return !folder.empty() && path.size() > folder.size() && path[folder.size()] == L'\\' &&
                    _wcsnicmp(path.c_str(), folder.c_str(), folder.size()) == 0;
            }
        };

        class commands
        {
        public:
            explicit commands(bool copy) : copy_(copy)
            {
                json info;
                std::string error;
                online_ = client_.get("/v1/info", info, error, 10);
                if (!online_) offline_error_ = error;
                drive_ = drive_place(info["drive"]);
            }

            outcome run(bool protect, const std::wstring& given)
            {
                const std::wstring path = full_path(given);
                const std::string name = narrow(file_of(path));
                if (!online_) return failure(name, error_text(failed_answer(json(), offline_error_)));
                const DWORD attributes = GetFileAttributesW(path.c_str());
                if (attributes != INVALID_FILE_ATTRIBUTES && (attributes & FILE_ATTRIBUTE_DIRECTORY))
                    return run_folder(protect, path, name);
                // A name without extension: its content decides.
                if (!possible_name(name)) return failure(name, tr("shell_unsupported", { { "f", name } }));
                return run_file(protect, path, name, std::string());
            }

        private:
            // into: the folder that receives the result under the name the service gives, or "".
            outcome run_file(bool protect, const std::wstring& path, const std::string& name, const std::string& into)
            {
                std::string data;
                DWORD e = 0;
                if (!read_whole(path, data, stamp_, e))
                {
                    if (sharing_error(e)) return failure(name, tr("shell_in_use", { { "f", name } }));
                    return failure(name, tr("shell_cannot_read", { { "f", name },
                        { "e", narrow(system_message(e)) } }));
                }
                const bool on_drive = drive_.holds(path);
                if (protect)
                {
                    // Nothing to do: the drive holds anonymized files only.
                    if (on_drive) return success(tr("shell_on_drive", { { "l", drive_.letter } }), name);
                    return anonymize(path, name, data, into);
                }
                return restore(path, name, data, on_drive, into);
            }

            outcome anonymize(const std::wstring& path, const std::string& name, const std::string& data,
                              const std::string& into)
            {
                json found;
                std::string error;
                if (client_.get("/v1/lookup?sha256=" + sha256_hex(data), found, error) && found["exact"].as_bool(false))
                {
                    outcome o = failure(name, tr("shell_already", { { "f", name } }));
                    o.already = true;
                    return o;
                }
                service_client::raw_answer a;
                if (!client_.post_bytes("/v1/anonymize?source=shell&name=" + query_value(name), data, a, error))
                    return failure(name, error_text(failed_answer(json(), error)));
                if (a.status != 200) return answer_failure(name, a);
                const std::string made = header(a, "x-pseudalis-name");
                const long long record = std::atoll(header(a, "x-pseudalis-record").c_str());
                const json report = json::parse_or_null(header(a, "x-pseudalis-report"));
                const long long values = report["total"].as_int(0);

                const std::string dir = into.empty() ? narrow(path.substr(0, path.find_last_of(L'\\'))) : into;
                // In its place, the file leaves for its anonymized form when its name changes.
                const bool moved = into.empty() && !copy_;
                std::wstring target = path;
                DWORD e = 0;
                if (!moved || made != name)
                {
                    const std::string wanted = into.empty() && copy_
                        ? copy_name(name, made, tr_utf8("suffix_anonymized")) : made;
                    target = widen(join_path(dir, free_name(dir, wanted)));
                    if (moved && !unchanged(path, stamp_)) return failure(name, tr("shell_changed", { { "f", name } }));
                    if (!write_new(target, a.body, e))
                        return failure(name, tr("shell_cannot_write", { { "f", narrow(file_of(target)) },
                            { "e", narrow(system_message(e)) } }));
                    // In its place, a picture leaves for the text of its marker.
                    if (moved && !DeleteFileW(path.c_str()))
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
                outcome o = success(tr_count(moved ? "shell_protected" : "shell_protected_copy", values,
                    { { "f", shown } }), shown);
                o.values = values;
                return o;
            }

            outcome restore(const std::wstring& path, const std::string& name, const std::string& data, bool on_drive,
                            const std::string& into)
            {
                service_client::raw_answer a;
                std::string error;
                const std::string oid = on_drive ? std::string() : object_id(path);
                if (!client_.post_bytes("/v1/restore?name=" + query_value(name) + "&object_id=" + oid, data, a, error))
                    return failure(name, error_text(failed_answer(json(), error)));
                if (a.status != 200) return answer_failure(name, a);
                const std::string how = header(a, "x-pseudalis-restored");
                const std::string original = header(a, "x-pseudalis-name");
                const json report = json::parse_or_null(header(a, "x-pseudalis-report"));
                const long long values = report["tags_restored"].as_int(0);
                // The name of an original kept is the one it had; a name restored from its substitutes
                // loses the suffix of an anonymized copy.
                const std::string base = how == "original" && !original.empty()
                    ? original : without_copy_suffix(original.empty() ? name : original);

                std::wstring target = path;
                DWORD e = 0;
                if (on_drive && into.empty())
                {
                    // The drive would anonymize the original again: it goes where the user says.
                    target = choose_destination(tr("shell_save_title", { { "f", base } }), base);
                    // Cancelled by the user: nothing to tell.
                    if (target.empty()) return success(std::wstring(), name);
                    const std::string shown = narrow(file_of(target));
                    const bool exists = GetFileAttributesW(target.c_str()) != INVALID_FILE_ATTRIBUTES;
                    if (exists ? !replace_content(target, a.body, e) : !write_new(target, a.body, e))
                        return failure(shown, tr("shell_cannot_write", { { "f", shown },
                            { "e", narrow(system_message(e)) } }));
                    return success(tr("shell_restored_copy", { { "f", shown } }), shown);
                }
                const std::string dir = into.empty() ? narrow(path.substr(0, path.find_last_of(L'\\'))) : into;
                const bool moved = into.empty() && !copy_;
                /*
                    The original takes its own name back, which the service kept or found again
                    from the substitutes of the name: the text file of a picture gives back the
                    picture under its name too.  A copy takes the name of the original.
                */
                if (!moved || base != name)
                {
                    const std::string wanted = into.empty() && copy_
                        ? copy_name(base, base, tr_utf8("suffix_restored")) : base;
                    target = widen(join_path(dir, free_name(dir, wanted)));
                    if (moved && !unchanged(path, stamp_)) return failure(name, tr("shell_changed", { { "f", name } }));
                    if (!write_new(target, a.body, e))
                        return failure(name, tr("shell_cannot_write", { { "f", narrow(file_of(target)) },
                            { "e", narrow(system_message(e)) } }));
                    if (moved && !DeleteFileW(path.c_str()))
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
                outcome o = how == "tags"
                    ? success(tr_count("shell_restored_tags", values, { { "f", shown } }), shown)
                    : success(tr(moved ? "shell_restored" : "shell_restored_copy", { { "f", shown } }), shown);
                o.values = values;
                return o;
            }

            // The name of a folder anonymized, or restored, by the service; why tells a failure.
            bool name_of(bool protect, const std::string& name, std::string& out, std::string& why)
            {
                json body = json::object(), answer;
                body["name"] = name;
                std::string error;
                out = name;
                if (!client_.send("POST", protect ? "/v1/names/anonymize" : "/v1/names/restore", body, answer, error))
                {
                    why = narrow(error_text(failed_answer(answer, error)));
                    return false;
                }
                const std::string made = answer["name"].as_string();
                if (!made.empty()) out = made;
                return true;
            }

            static void note_failure(folder_tally& t, const std::string& name, const std::string& why)
            {
                if (t.failed++ > 0) return;
                t.failed_name = name;
                t.failed_error = why;
            }

            static void copy_as_is(const std::wstring& path, const std::string& name, const std::string& into,
                                   folder_tally& t)
            {
                const std::wstring target = widen(join_path(into, free_name(into, name)));
                if (!CopyFileW(path.c_str(), target.c_str(), TRUE))
                    note_failure(t, name, narrow(system_message(GetLastError())));
            }

            /*
                A file of another kind: it stays as it is in its place and out of an anonymized
                copy, which holds anonymized files only; a restored copy takes it as it is.
            */
            static void other_kind(bool protect, const std::wstring& path, const std::string& name,
                                   const std::string& into, folder_tally& t)
            {
                if (!protect)
                {
                    if (!into.empty()) copy_as_is(path, name, into, t);
                    return;
                }
                if (t.other++ < 3) t.others.push_back(name);
            }

            /*
                Processes the files of the folder from, then its folders, each one taking its name
                once its files are done: into is "" for a processing in place, else the folder of
                the copy that receives them.
            */
            void walk(bool protect, const std::wstring& from, const std::string& into, int depth, folder_tally& t)
            {
                std::vector<folder_entry> entries;
                DWORD e = 0;
                if (!list_folder(from, entries, e))
                {
                    note_failure(t, narrow(file_of(from)), narrow(system_message(e)));
                    return;
                }
                for (const folder_entry& x : entries)
                {
                    if (x.folder) continue;
                    if (++t.files > max_folder_files)
                    {
                        t.stopped = true;
                        return;
                    }
                    const std::wstring path = from + L"\\" + x.name;
                    const std::string name = narrow(x.name);
                    if (!possible_name(name))
                    {
                        other_kind(protect, path, name, into, t);
                        continue;
                    }
                    const outcome o = run_file(protect, path, name, into);
                    if (o.ok)
                    {
                        ++t.done;
                        t.values += o.values;
                    }
                    else if (o.already)
                    {
                        // Anonymized already: an anonymized copy takes it as it is.
                        ++t.already;
                        if (!into.empty()) copy_as_is(path, name, into, t);
                    }
                    else if (o.code == "unsupported" || o.code == "kind_mismatch")
                    {
                        other_kind(protect, path, name, into, t);
                    }
                    else if (!protect && o.code == "not_protected")
                    {
                        // Nothing to restore: a restored copy takes it as it is.
                        if (!into.empty()) copy_as_is(path, name, into, t);
                    }
                    else
                    {
                        note_failure(t, name, o.error);
                    }
                }
                for (const folder_entry& x : entries)
                {
                    if (!x.folder) continue;
                    if (t.stopped) return;
                    if (depth >= max_folder_depth)
                    {
                        t.too_deep = true;
                        continue;
                    }
                    const std::wstring path = from + L"\\" + x.name;
                    const std::string name = narrow(x.name);
                    std::string named, why;
                    const bool named_ok = name_of(protect, name, named, why);
                    if (into.empty())
                    {
                        walk(protect, path, std::string(), depth + 1, t);
                        if (!named_ok) note_failure(t, name, why);
                        else if (named != name) rename_folder(path, named, t);
                        continue;
                    }
                    // An anonymized copy never takes a name the service could not anonymize.
                    if (!named_ok && protect)
                    {
                        note_failure(t, name, why);
                        continue;
                    }
                    const std::wstring sub = widen(join_path(into, free_name(into, named)));
                    if (!CreateDirectoryW(sub.c_str(), nullptr))
                    {
                        note_failure(t, name, narrow(system_message(GetLastError())));
                        continue;
                    }
                    walk(protect, path, narrow(sub), depth + 1, t);
                }
            }

            // Gives the folder at path the name named, beside it; returns its new path, or path.
            static std::wstring rename_folder(const std::wstring& path, const std::string& named, folder_tally& t)
            {
                const std::string parent = narrow(path.substr(0, path.find_last_of(L'\\')));
                const std::wstring target = widen(join_path(parent, free_name(parent, named)));
                if (MoveFileExW(path.c_str(), target.c_str(), 0)) return target;
                note_failure(t, narrow(file_of(path)), narrow(tr("shell_folder_kept_name",
                    { { "e", narrow(system_message(GetLastError())) } })));
                return path;
            }

            outcome run_folder(bool protect, const std::wstring& path, const std::string& name)
            {
                if (name.empty()) return failure(narrow(path), tr("shell_unsupported", { { "f", narrow(path) } }));
                const bool on_drive = drive_.holds(path);
                // Nothing to do: the drive holds anonymized files only.
                if (protect && on_drive) return success(tr("shell_on_drive", { { "l", drive_.letter } }), name);
                std::string named, why;
                const bool named_ok = name_of(protect, name, named, why);
                // A restored folder loses the suffix of an anonymized copy.
                if (!protect) named = without_copy_suffix(named);
                folder_tally t;
                std::wstring top = path;
                const bool copied = copy_ || on_drive;
                if (!copied)
                {
                    walk(protect, path, std::string(), 0, t);
                    if (!named_ok) note_failure(t, name, why);
                    else if (named != name) top = rename_folder(path, named, t);
                    return folder_outcome(protect, false, top, t);
                }
                // An anonymized copy never takes a name the service could not anonymize.
                if (!named_ok && protect) return failure(name, widen(why));
                std::wstring into = path.substr(0, path.find_last_of(L'\\'));
                if (on_drive)
                {
                    // The drive would anonymize the originals again: they go where the user says.
                    into = choose_destination(tr("shell_save_folder_title", { { "f", name } }), std::string());
                    if (into.empty()) return success(std::wstring(), name);
                }
                const std::string into8 = narrow(into);
                const std::string wanted = copy_ ? named + " (" + tr_utf8(protect ? "suffix_anonymized"
                                                                                  : "suffix_restored") + ")" : named;
                top = widen(join_path(into8, free_name(into8, wanted)));
                // A copy inside the folder it copies would copy itself.
                if (within(top, path))
                    return failure(name, tr("shell_cannot_write", { { "f", narrow(file_of(top)) },
                        { "e", narrow(system_message(ERROR_BAD_PATHNAME)) } }));
                if (!CreateDirectoryW(top.c_str(), nullptr))
                    return failure(name, tr("shell_cannot_write", { { "f", narrow(file_of(top)) },
                        { "e", narrow(system_message(GetLastError())) } }));
                walk(protect, path, narrow(top), 0, t);
                return folder_outcome(protect, true, top, t);
            }

            static outcome folder_outcome(bool protect, bool copied, const std::wstring& top, const folder_tally& t)
            {
                const std::string shown = narrow(file_of(top));
                const char* key = protect ? (copied ? "shell_folder_protected_copy" : "shell_folder_protected")
                                          : (copied ? "shell_folder_restored_copy" : "shell_folder_restored");
                std::wstring text = tr(key, { { "f", shown },
                    { "files", narrow(tr_count(protect ? "count_files_anonymized" : "count_files_restored", t.done)) },
                    { "values", narrow(tr_count("count_values_replaced", t.values)) } });
                if (t.already) text += L" " + tr_count("shell_folder_already", t.already);
                if (t.other)
                {
                    std::string list;
                    for (const std::string& n : t.others) list += (list.empty() ? "" : ", ") + n;
                    if (t.other > static_cast<long long>(t.others.size())) list += "...";
                    text += L" " + tr_count(copied ? "shell_folder_other_out" : "shell_folder_other_kept", t.other,
                        { { "l", list } });
                }
                if (t.failed)
                    text += L" " + tr_count("shell_folder_failed", t.failed, { { "f", t.failed_name },
                        { "e", t.failed_error } });
                if (t.too_deep)
                    text += L" " + tr("shell_folder_too_deep", { { "n", std::to_string(max_folder_depth) } });
                if (t.stopped)
                    text += L" " + tr("shell_folder_stopped", { { "n", std::to_string(max_folder_files) } });
                outcome o;
                o.ok = t.failed == 0 && !t.stopped && !t.too_deep;
                o.text = text;
                o.name = shown;
                o.error = t.failed_error;
                return o;
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
            drive_place drive_{ json() };
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
            L"  pseudalis protect [--copy] <file or folder>...   anonymize each one, in its place or as a copy\r\n"
            L"  pseudalis restore [--copy] <file or folder>...   restore the original, in its place or as a copy";
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
