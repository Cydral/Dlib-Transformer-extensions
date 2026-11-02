// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

/*
    The drive of Pseudalis on WinFsp (drive.h).

    WinFsp - Windows File System Proxy, Copyright (C) Bill Zissimopoulos,
    https://github.com/winfsp/winfsp.  Its library is loaded at run time from its
    installation (FspLoad), and its functions are resolved by name, so that the service
    starts, without drive, on a computer without WinFsp.

    The tree lives in memory, under one lock, and in the database at each change.  A file
    being written holds its data in memory only, within a quarter of the memory of the
    computer for all the files; when its last writing handle closes, the data is protected
    (service::protect()), the original goes to the vault with a record, the protected bytes
    to the vault as the content of the file, and the data in clear is erased.  Only the
    handles that write a file read its data in clear.  FlushAndPurgeOnCleanup makes the cache
    of Windows give the written pages to the file system and forget them at that close, so
    that no page in clear is read back, and FileInfoTimeout 0 makes every size and name come
    from here, as the protection changes them.

    A file under protection is not opened (sharing violation): WinFsp serializes the
    creations, renames and deletions of the drive around an open, which must not wait.  A
    version that cannot be protected leaves the previous one in place.  Deleting or replacing
    a file marks its record removed rather than forgetting it, since a move out of the drive
    is a copy followed by a deletion.  No exception leaves an operation: WinFsp calls them
    from C.
*/

#if !defined(_WIN32_WINNT)
#define _WIN32_WINNT 0x0A00
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <sddl.h>
#if defined(__MINGW32__)
// The headers of WinFsp declare the barrier intrinsic of Visual C++, a macro of MinGW.
#undef _ReadWriteBarrier
extern "C" inline void _ReadWriteBarrier(void) { __asm__ __volatile__("" ::: "memory"); }
#endif
#include <winfsp/winfsp.h>

#include "drive.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <set>
#include <thread>
#include <vector>

#include "../common/file_kinds.h"
#include "../common/log.h"
#include "../common/platform.h"
#include "keys.h"
#include "service.h"

namespace pseudalis
{
    namespace
    {
        // The functions of WinFsp the drive calls, resolved after FspLoad().
        struct fsp_api
        {
            NTSTATUS (*create)(PWSTR, const FSP_FSCTL_VOLUME_PARAMS*, const FSP_FILE_SYSTEM_INTERFACE*,
                               FSP_FILE_SYSTEM**) = nullptr;
            VOID (*destroy)(FSP_FILE_SYSTEM*) = nullptr;
            NTSTATUS (*set_mount_point)(FSP_FILE_SYSTEM*, PWSTR) = nullptr;
            VOID (*remove_mount_point)(FSP_FILE_SYSTEM*) = nullptr;
            NTSTATUS (*start_dispatcher)(FSP_FILE_SYSTEM*, ULONG) = nullptr;
            VOID (*stop_dispatcher)(FSP_FILE_SYSTEM*) = nullptr;
            BOOLEAN (*add_dir_info)(FSP_FSCTL_DIR_INFO*, PVOID, ULONG, PULONG) = nullptr;
            NTSTATUS (*notify_begin)(FSP_FILE_SYSTEM*, ULONG) = nullptr;
            NTSTATUS (*notify)(FSP_FILE_SYSTEM*, FSP_FSCTL_NOTIFY_INFO*, SIZE_T) = nullptr;
            NTSTATUS (*notify_end)(FSP_FILE_SYSTEM*) = nullptr;
        };

        bool load_winfsp(fsp_api& api, std::string& error)
        {
            static std::mutex m;
            static HMODULE module = nullptr;
            std::lock_guard<std::mutex> lock(m);
            if (!module)
            {
                PVOID h = nullptr;
                if (FspLoad(&h) != STATUS_SUCCESS || !h)
                {
                    error = "WinFsp is not installed";
                    return false;
                }
                module = static_cast<HMODULE>(h);
            }
            auto get = [](const char* name) { return reinterpret_cast<void*>(GetProcAddress(module, name)); };
#define PSD_FSP(member, name) api.member = reinterpret_cast<decltype(api.member)>(get(name))
            PSD_FSP(create, "FspFileSystemCreate");
            PSD_FSP(destroy, "FspFileSystemDelete");
            PSD_FSP(set_mount_point, "FspFileSystemSetMountPoint");
            PSD_FSP(remove_mount_point, "FspFileSystemRemoveMountPoint");
            PSD_FSP(start_dispatcher, "FspFileSystemStartDispatcher");
            PSD_FSP(stop_dispatcher, "FspFileSystemStopDispatcher");
            PSD_FSP(add_dir_info, "FspFileSystemAddDirInfo");
            PSD_FSP(notify_begin, "FspFileSystemNotifyBegin");
            PSD_FSP(notify, "FspFileSystemNotify");
            PSD_FSP(notify_end, "FspFileSystemNotifyEnd");
#undef PSD_FSP
            if (!api.create || !api.destroy || !api.set_mount_point || !api.remove_mount_point ||
                !api.start_dispatcher || !api.stop_dispatcher || !api.add_dir_info)
            {
                error = "the installed WinFsp is too old";
                return false;
            }
            return true;
        }

        UINT64 now_filetime()
        {
            FILETIME ft;
            GetSystemTimeAsFileTime(&ft);
            return (static_cast<UINT64>(ft.dwHighDateTime) << 32) | ft.dwLowDateTime;
        }

        // The key of a name in a folder: its upper case, independent of the language, as the file
        // systems of Windows compare names.
        std::wstring name_key(const std::wstring& name)
        {
            std::wstring k = name;
            if (!k.empty() &&
                LCMapStringEx(LOCALE_NAME_INVARIANT, LCMAP_UPPERCASE, name.c_str(), static_cast<int>(name.size()),
                              &k[0], static_cast<int>(k.size()), nullptr, nullptr, 0) == 0)
                CharUpperBuffW(&k[0], static_cast<DWORD>(k.size()));
            return k;
        }

        // The components of a path of the drive: "\a\b.txt" -> {"a", "b.txt"}.
        std::vector<std::wstring> components(PCWSTR path)
        {
            std::vector<std::wstring> out;
            std::wstring cur;
            for (const wchar_t* p = path; *p; ++p)
            {
                if (*p == L'\\' || *p == L'/')
                {
                    if (!cur.empty()) out.push_back(cur);
                    cur.clear();
                }
                else cur.push_back(*p);
            }
            if (!cur.empty()) out.push_back(cur);
            return out;
        }

        const UINT32 kept_attributes = FILE_ATTRIBUTE_READONLY | FILE_ATTRIBUTE_HIDDEN | FILE_ATTRIBUTE_SYSTEM |
                                       FILE_ATTRIBUTE_ARCHIVE | FILE_ATTRIBUTE_NOT_CONTENT_INDEXED;

        // Erases a buffer of data in clear over its whole capacity, the bytes a shrink left there
        // included, then frees it.
        void erase_clear(std::string& b)
        {
            if (b.capacity() > 0)
            {
                b.resize(b.capacity());
                wipe(b);
            }
            std::string().swap(b);
        }

        /*
            The operations of WinFsp run on the threads of its dispatcher, in C: an exception must
            not leave them.  guarded<decltype(f)>::call<f> runs f and turns an exception into a
            status (a lack of memory, a failure of the database or of the vault), logged.
        */
        void log_failure(const char* what)
        {
            try
            {
                log_error(std::string("drive: ") + what);
            }
            catch (...)
            {
            }
        }

        template <typename T> struct guarded;

        template <typename... A> struct guarded<NTSTATUS(A...)>
        {
            template <NTSTATUS (*F)(A...)> static NTSTATUS call(A... a)
            {
                try
                {
                    return F(a...);
                }
                catch (const std::bad_alloc&)
                {
                    log_failure("out of memory");
                    return STATUS_INSUFFICIENT_RESOURCES;
                }
                catch (const std::exception& e)
                {
                    log_failure(e.what());
                    return STATUS_INTERNAL_ERROR;
                }
                catch (...)
                {
                    log_failure("unknown error");
                    return STATUS_INTERNAL_ERROR;
                }
            }
        };

        template <typename... A> struct guarded<VOID(A...)>
        {
            template <VOID (*F)(A...)> static VOID call(A... a)
            {
                try
                {
                    F(a...);
                }
                catch (const std::exception& e)
                {
                    log_failure(e.what());
                }
                catch (...)
                {
                    log_failure("unknown error");
                }
            }
        };
    }

    struct drive_host::impl
    {
        struct state
        {
            drive_node n;
            int handles = 0;                    // handles not closed
            int open = 0;                       // handles not cleaned up
            int writable = 0;                   // handles not cleaned up that may write
            bool dirty = false;                 // written since the last protection
            bool created = false;               // made since, without a protected version yet
            bool writing = false;               // buffer holds the data of the file, in clear
            std::string buffer;
            bool processing = false;
            bool deleted = false;
            std::shared_ptr<const std::string> content;   // the protected bytes, while open
        };

        struct handle
        {
            long long id = 0;
            bool writable = false;              // counted in state::writable until its cleanup
            bool cleaned = false;               // its cleanup came; a handle the kernel refuses gets none
        };

        explicit impl(service& s) : svc(s) {}

        service& svc;
        fsp_api api;
        FSP_FILE_SYSTEM* fs = nullptr;
        FSP_FILE_SYSTEM_INTERFACE iface = {};
        std::string letter;
        std::string error;
        std::string error_code;     // "winfsp_missing" when WinFsp is not installed
        bool mounted = false;
        PSECURITY_DESCRIPTOR sd = nullptr;
        DWORD sd_size = 0;
        UINT64 max_bytes = 256ull << 20;
        // The data in clear the buffers hold, and its limit: a quarter of the memory of the computer.
        size_t clear_bytes = 0;
        size_t clear_budget = static_cast<size_t>(512) << 20;
        // The bytes of the drive, as the volume gives them.
        std::mutex usage_m;
        int64_t usage_time = 0;
        long long usage_bytes = 0;

        mutable std::mutex m;
        std::condition_variable cv;
        std::map<long long, state> nodes;
        std::map<std::pair<long long, std::wstring>, long long> names;   // (parent, key) -> node

        // Changes the Explorer hears of, sent by a thread of their own, outside the operations.
        std::mutex notify_m;
        std::condition_variable notify_cv;
        std::deque<std::pair<UINT32, std::wstring>> notices;   // action, path
        std::thread notifier;
        bool stopping = false;

        // ---------------------------------------------------------------------------
        // The tree.

        void load()
        {
            std::vector<long long> pending = { 1 };
            drive_node root;
            if (svc.db().node(1, root)) nodes[1].n = root;
            while (!pending.empty())
            {
                const long long parent = pending.back();
                pending.pop_back();
                for (const drive_node& c : svc.db().children(parent))
                {
                    state& s = nodes[c.id];
                    s.n = c;
                    names[std::make_pair(parent, name_key(widen(c.name)))] = c.id;
                    if (c.directory) pending.push_back(c.id);
                }
            }
        }

        state* find(long long id)
        {
            const auto it = nodes.find(id);
            return it == nodes.end() || it->second.deleted ? nullptr : &it->second;
        }

        state* child(long long parent, const std::wstring& name)
        {
            const auto it = names.find(std::make_pair(parent, name_key(name)));
            return it == names.end() ? nullptr : find(it->second);
        }

        // The node of a path, or null; parent_missing tells a missing folder on the way.
        state* lookup(PCWSTR path, bool* parent_missing = nullptr)
        {
            state* s = find(1);
            const std::vector<std::wstring> parts = components(path);
            for (size_t i = 0; s && i < parts.size(); ++i)
            {
                if (!s->n.directory)
                {
                    if (parent_missing) *parent_missing = true;
                    return nullptr;
                }
                state* next = child(s->n.id, parts[i]);
                if (!next && parent_missing) *parent_missing = i + 1 < parts.size();
                s = next;
            }
            return s;
        }

        std::wstring path_of(long long id)
        {
            std::wstring p;
            for (state* s = find(id); s && s->n.id != 1; s = find(s->n.parent)) p = L"\\" + widen(s->n.name) + p;
            return p.empty() ? L"\\" : p;
        }

        // The children of a folder: one range of names, ordered by their keys.
        bool has_children(long long id) const
        {
            for (auto it = names.lower_bound(std::make_pair(id, std::wstring()));
                 it != names.end() && it->first.first == id; ++it)
            {
                const auto n = nodes.find(it->second);
                if (n != nodes.end() && !n->second.deleted) return true;
            }
            return false;
        }

        void fill(const state& s, FSP_FSCTL_FILE_INFO* fi) const
        {
            const UINT64 size = s.writing ? s.buffer.size() : static_cast<UINT64>(s.n.size);
            // WinFsp reports a file without attribute as FILE_ATTRIBUTE_NORMAL.
            fi->FileAttributes = s.n.directory ? FILE_ATTRIBUTE_DIRECTORY | (s.n.attributes & kept_attributes)
                                               : s.n.attributes & kept_attributes;
            fi->ReparseTag = 0;
            fi->FileSize = s.n.directory ? 0 : size;
            fi->AllocationSize = (fi->FileSize + 4095) / 4096 * 4096;
            fi->CreationTime = static_cast<UINT64>(s.n.created);
            fi->LastAccessTime = static_cast<UINT64>(s.n.accessed);
            fi->LastWriteTime = static_cast<UINT64>(s.n.written);
            fi->ChangeTime = static_cast<UINT64>(s.n.changed);
            fi->IndexNumber = static_cast<UINT64>(s.n.id);
            fi->HardLinks = 0;
            fi->EaSize = 0;
        }

        // Writes a node into the database; false, logged, when the database refuses it.
        bool save_node(const drive_node& n)
        {
            try
            {
                svc.db().update_node(n);
                return true;
            }
            catch (const std::exception& e)
            {
                log_error("drive: " + std::string(e.what()));
                return false;
            }
        }

        bool save(const state& s)
        {
            return save_node(s.n);
        }

        // An event of the drive, for the icon and the web application.
        void event(const std::string& kind, const std::wstring& path, const std::string& message)
        {
            try
            {
                svc.db().add_event(kind, narrow(path), message);
            }
            catch (const std::exception& e)
            {
                log_error("drive: " + std::string(e.what()));
            }
        }

        // The memory a buffer holds: its capacity, once it left the small buffer of std::string.
        static size_t held(const std::string& b)
        {
            return b.capacity() > 15 ? b.capacity() : 0;
        }

        /*
            Gives the buffer of s the size n within the budget of the data in clear, which counts
            the memory the buffers hold; false when the budget does not hold it.  A block a growth
            leaves is erased before it is freed.
        */
        bool resize_buffer(state& s, size_t n)
        {
            if (n <= s.buffer.capacity())
            {
                s.buffer.resize(n, '\0');
                return true;
            }
            const size_t others = clear_bytes - std::min(clear_bytes, held(s.buffer));
            size_t want = std::max(n, std::min(s.buffer.capacity() * 2, static_cast<size_t>(max_bytes)));
            if (others + want > clear_budget) want = n;
            if (others + want > clear_budget) return false;
            std::string bigger;
            bigger.reserve(want);
            bigger.assign(s.buffer);
            bigger.resize(n, '\0');
            erase_clear(s.buffer);
            s.buffer.swap(bigger);
            clear_bytes = others + held(s.buffer);
            return true;
        }

        void drop_buffer(state& s)
        {
            clear_bytes -= std::min(clear_bytes, held(s.buffer));
            erase_clear(s.buffer);
        }

        /*
            Gives WinFsp, in a Create or an Open, the path of the node with the case of its names, so
            that the kernel keeps them and the notifications match them.  WinFsp refuses a name that
            differs from the one asked for otherwise than by its case: such a name is not given.
        */
        void normalize(long long id, PCWSTR asked, FSP_FSCTL_FILE_INFO* fi)
        {
            try
            {
                FSP_FSCTL_OPEN_FILE_INFO* oi = FspFileSystemGetOpenFileInfo(fi);
                const std::wstring path = path_of(id);
                const size_t bytes = path.size() * sizeof(wchar_t);
                if (!oi->NormalizedName || bytes > FSP_FSCTL_TRANSACT_PATH_SIZEMAX || bytes > oi->NormalizedNameSize)
                    return;
                if (CompareStringOrdinal(path.c_str(), static_cast<int>(path.size()), asked, -1, TRUE) != CSTR_EQUAL)
                    return;
                std::memcpy(oi->NormalizedName, path.data(), bytes);
                oi->NormalizedNameSize = static_cast<UINT16>(bytes);
            }
            catch (const std::exception&)
            {
                // The name is optional: WinFsp then keeps it in upper case.
            }
        }

        void notice(UINT32 action, const std::wstring& path)
        {
            std::lock_guard<std::mutex> lock(notify_m);
            notices.push_back(std::make_pair(action, path));
            notify_cv.notify_one();
        }

        void notify_loop()
        {
            std::unique_lock<std::mutex> lock(notify_m);
            while (!stopping)
            {
                notify_cv.wait(lock, [this]() { return stopping || !notices.empty(); });
                if (stopping) break;
                std::deque<std::pair<UINT32, std::wstring>> batch;
                batch.swap(notices);
                lock.unlock();
                NTSTATUS begun = STATUS_NOT_SUPPORTED;
                if (api.notify_begin && api.notify && api.notify_end && fs)
                {
                    // A rename in progress makes the notifications wait: tried again five times.
                    begun = STATUS_CANT_WAIT;
                    for (int attempt = 0; attempt < 5 && begun == STATUS_CANT_WAIT; ++attempt)
                        begun = api.notify_begin(fs, 1000);
                    if (begun != STATUS_SUCCESS) log_warning("drive: the changes of the drive could not be notified");
                }
                if (begun == STATUS_SUCCESS)
                {
                    for (const auto& n : batch)
                    {
                        std::vector<char> buf(sizeof(FSP_FSCTL_NOTIFY_INFO) + n.second.size() * sizeof(wchar_t));
                        FSP_FSCTL_NOTIFY_INFO* info = reinterpret_cast<FSP_FSCTL_NOTIFY_INFO*>(buf.data());
                        info->Size = static_cast<UINT16>(buf.size());
                        info->Filter = FILE_NOTIFY_CHANGE_FILE_NAME | FILE_NOTIFY_CHANGE_SIZE |
                                       FILE_NOTIFY_CHANGE_LAST_WRITE;
                        info->Action = n.first;
                        std::memcpy(info->FileNameBuf, n.second.data(), n.second.size() * sizeof(wchar_t));
                        api.notify(fs, info, buf.size());
                    }
                    api.notify_end(fs);
                }
                lock.lock();
            }
        }

        /*
            Removes a node of the tree and, unless the caller did it, of the database, with its
            protected content.  The record of its file is marked removed: its original stays
            restorable from the copies of the file, a move out of the drive being a copy followed
            by a deletion, until the service forgets it (service::mark_removed()).
        */
        void remove_node(state& s, bool from_database = true)
        {
            const long long file = s.n.file;
            const std::string blob = s.n.blob;
            names.erase(std::make_pair(s.n.parent, name_key(widen(s.n.name))));
            s.deleted = true;
            drop_buffer(s);
            s.content.reset();
            try
            {
                if (from_database) svc.db().remove_node(s.n.id);
            }
            catch (const std::exception& e)
            {
                log_error("drive: " + std::string(e.what()));
            }
            if (file) svc.mark_removed(file);
            if (!blob.empty()) svc.blobs().remove(blob);
            if (s.handles == 0) nodes.erase(s.n.id);
        }

        /*
            A name not taken in parent, of 255 characters at most, the end of its stem giving way to
            its extension: "photo.png.txt", then "photo.png (2).txt"...
        */
        std::string free_name(long long parent, const std::string& name, long long self)
        {
            const std::wstring wide = widen(name);
            const size_t dot = wide.rfind(L'.');
            const bool extension = dot != std::wstring::npos && dot > 0 && wide.size() - dot <= 16;
            const std::wstring stem = extension ? wide.substr(0, dot) : wide;
            const std::wstring tail = extension ? wide.substr(dot) : std::wstring();
            for (int k = 1;; ++k)
            {
                const std::wstring mark = k == 1 ? std::wstring() : L" (" + std::to_wstring(k) + L")";
                std::wstring s = stem;
                const size_t room = 255 - std::min<size_t>(255, tail.size() + mark.size());
                if (s.size() > room)
                {
                    s.resize(room);
                    // Never half of a surrogate pair.
                    if (!s.empty() && s.back() >= 0xD800 && s.back() <= 0xDBFF) s.pop_back();
                }
                const std::wstring candidate = s + mark + tail;
                const state* other = child(parent, candidate);
                if (!other || other->n.id == self) return narrow(candidate);
            }
        }

        bool load_content(state& s)
        {
            if (s.content) return true;
            auto c = std::make_shared<std::string>();
            if (!s.n.blob.empty() && !svc.blobs().get(s.n.blob, *c)) return false;
            s.content = c;
            return true;
        }

        /*
            Protects the data written to s, with m held through lock, released while the engine
            works.  When the new version cannot be protected or recorded, a file that existed keeps
            its previous version, empty or not, and a new file is removed; the record of a version
            replaced is marked removed, as a deleted file's.
        */
        void protect(std::unique_lock<std::mutex>& lock, state& s)
        {
            const long long id = s.n.id;
            // Whatever happens, m is held again, the file is no longer processed and its waiters wake.
            struct finish
            {
                impl& p;
                std::unique_lock<std::mutex>& lock;
                long long id;
                ~finish()
                {
                    if (!lock.owns_lock()) lock.lock();
                    if (state* x = p.find(id)) x->processing = false;
                    p.cv.notify_all();
                }
            } done{ *this, lock, id };

            s.processing = true;
            s.writing = false;
            s.dirty = false;
            std::string data;
            data.swap(s.buffer);
            clear_bytes -= std::min(clear_bytes, held(data));
            const bool fresh = s.created;
            const long long old_file = s.n.file;
            const std::string old_blob = s.n.blob;
            const std::string name = s.n.name;
            const std::wstring path = path_of(id);
            // An empty file, as the Explorer makes one before its name is typed, stays empty.
            if (data.empty())
            {
                erase_clear(data);
                drive_node next = s.n;
                next.blob.clear();
                next.size = 0;
                next.file = 0;
                if (!save_node(next)) return;
                s.n = next;
                s.created = false;
                s.content = std::make_shared<std::string>();
                if (old_file) svc.mark_removed(old_file);
                if (!old_blob.empty()) svc.blobs().remove(old_blob);
                return;
            }
            lock.unlock();

            protect_result r;
            user_error e;
            bool ok = false;
            std::string blob;
            try
            {
                ok = svc.protect(data, name, "drive", true, r, e);
                if (ok) blob = svc.blobs().put(r.bytes);
            }
            catch (const std::exception& x)
            {
                if (!ok) e.set(x.what());
                else blob.clear();
            }
            erase_clear(data);

            lock.lock();
            state* now = find(id);
            if (!now)
            {
                // Deleted meanwhile.
                if (ok && r.record) svc.forget(r.record);
                if (!blob.empty()) svc.blobs().remove(blob);
                return;
            }
            std::string why = ok ? std::string("the vault cannot be written") : e.text;
            if (why.empty()) why = "the file cannot be protected";
            if (ok && !blob.empty())
            {
                drive_node next = now->n;
                next.blob = blob;
                next.size = static_cast<long long>(r.bytes.size());
                next.file = r.record;
                // The time of the last write stays the one of the writer, which a copy keeps.
                next.changed = static_cast<long long>(now_filetime());
                if (r.name != name) next.name = free_name(next.parent, r.name, id);
                if (save_node(next))
                {
                    if (next.name != now->n.name)
                    {
                        names.erase(std::make_pair(now->n.parent, name_key(widen(now->n.name))));
                        names[std::make_pair(next.parent, name_key(widen(next.name)))] = id;
                    }
                    now->n = next;
                    now->created = false;
                    now->content = std::make_shared<std::string>(r.bytes);
                    if (old_file) svc.mark_removed(old_file);
                    if (!old_blob.empty()) svc.blobs().remove(old_blob);
                    const std::wstring new_path = path_of(id);
                    event("protected", new_path, std::to_string(r.report["total"].as_int(0)) + " values, " +
                                                 std::to_string(r.report["dropped"].size()) + " entries left out");
                    if (new_path != path)
                    {
                        notice(FILE_ACTION_RENAMED_OLD_NAME, path);
                        notice(FILE_ACTION_RENAMED_NEW_NAME, new_path);
                    }
                    else notice(FILE_ACTION_MODIFIED, path);
                    return;
                }
                why = "the database cannot be written";
                svc.blobs().remove(blob);
            }
            if (ok && r.record) svc.forget(r.record);
            log_warning("drive: " + narrow(path) + ": " + why);
            if (!fresh)
            {
                event("refused", path, why + "; the previous version stays");
                notice(FILE_ACTION_MODIFIED, path);
            }
            else
            {
                event("refused", path, why);
                remove_node(*now);
                notice(FILE_ACTION_REMOVED, path);
            }
        }

        // Waits, m held, until the protection of s ends; false when it lasts beyond ten minutes.
        bool wait_idle(std::unique_lock<std::mutex>& lock, long long id)
        {
            return cv.wait_for(lock, std::chrono::minutes(10), [&]() {
                state* s = find(id);
                return !s || !s->processing;
            });
        }

        // ---------------------------------------------------------------------------
        // The operations of WinFsp.

        static impl* of(FSP_FILE_SYSTEM* f) { return static_cast<impl*>(f->UserContext); }

        static NTSTATUS get_volume_info(FSP_FILE_SYSTEM* f, FSP_FSCTL_VOLUME_INFO* vi)
        {
            impl* p = of(f);
            const uint64_t free = free_disk_space(p->svc.data_dir());
            long long bytes = 0;
            {
                // The Explorer asks often: the database is read once every five seconds at most.
                std::lock_guard<std::mutex> lock(p->usage_m);
                const int64_t now = now_unix();
                if (now - p->usage_time >= 5)
                {
                    long long files = 0;
                    p->svc.db().drive_usage(files, p->usage_bytes);
                    p->usage_time = now;
                }
                bytes = p->usage_bytes;
            }
            vi->TotalSize = free + static_cast<uint64_t>(bytes);
            vi->FreeSize = std::min<uint64_t>(free, p->max_bytes * 64);
            const wchar_t label[] = L"Pseudalis";
            vi->VolumeLabelLength = static_cast<UINT16>((sizeof(label) / sizeof(wchar_t) - 1) * sizeof(wchar_t));
            std::memcpy(vi->VolumeLabel, label, vi->VolumeLabelLength);
            return STATUS_SUCCESS;
        }

        static NTSTATUS set_volume_label(FSP_FILE_SYSTEM* f, PWSTR, FSP_FSCTL_VOLUME_INFO* vi)
        {
            return get_volume_info(f, vi);
        }

        NTSTATUS security(PSECURITY_DESCRIPTOR out, SIZE_T* size) const
        {
            if (!size) return STATUS_SUCCESS;
            if (out && *size >= sd_size) std::memcpy(out, sd, sd_size);
            const bool fits = *size >= sd_size;
            *size = sd_size;
            return out && !fits ? STATUS_BUFFER_OVERFLOW : STATUS_SUCCESS;
        }

        static NTSTATUS get_security_by_name(FSP_FILE_SYSTEM* f, PWSTR name, PUINT32 attributes,
                                             PSECURITY_DESCRIPTOR out, SIZE_T* size)
        {
            impl* p = of(f);
            std::lock_guard<std::mutex> lock(p->m);
            bool parent_missing = false;
            state* s = p->lookup(name, &parent_missing);
            if (!s) return parent_missing ? STATUS_OBJECT_PATH_NOT_FOUND : STATUS_OBJECT_NAME_NOT_FOUND;
            if (attributes)
            {
                FSP_FSCTL_FILE_INFO fi;
                p->fill(*s, &fi);
                *attributes = fi.FileAttributes;
            }
            return p->security(out, size);
        }

        static NTSTATUS create(FSP_FILE_SYSTEM* f, PWSTR name, UINT32 options, UINT32 access, UINT32 attributes,
                               PSECURITY_DESCRIPTOR, UINT64, PVOID* context, FSP_FSCTL_FILE_INFO* fi)
        {
            impl* p = of(f);
            std::unique_lock<std::mutex> lock(p->m);
            std::vector<std::wstring> parts = components(name);
            if (parts.empty()) return STATUS_OBJECT_NAME_COLLISION;
            const std::wstring leaf = parts.back();
            parts.pop_back();
            state* parent = p->find(1);
            for (const std::wstring& c : parts)
            {
                parent = parent && parent->n.directory ? p->child(parent->n.id, c) : nullptr;
                if (!parent) return STATUS_OBJECT_PATH_NOT_FOUND;
            }
            if (!parent->n.directory) return STATUS_OBJECT_PATH_NOT_FOUND;
            if (p->child(parent->n.id, leaf)) return STATUS_OBJECT_NAME_COLLISION;
            if (leaf.size() > 255) return STATUS_OBJECT_NAME_INVALID;
            const bool directory = (options & FILE_DIRECTORY_FILE) != 0;
            const std::string utf8 = narrow(leaf);
            if (!directory && !supported_name(utf8))
            {
                const std::wstring folder = p->path_of(parent->n.id);
                p->event("refused", folder + (folder == L"\\" ? L"" : L"\\") + leaf,
                         "this kind of file is not protected by Pseudalis");
                return STATUS_ACCESS_DENIED;
            }
            drive_node n;
            n.parent = parent->n.id;
            n.name = utf8;
            n.directory = directory;
            n.attributes = (attributes & kept_attributes) | (directory ? 0 : FILE_ATTRIBUTE_ARCHIVE);
            n.created = n.accessed = n.written = n.changed = static_cast<long long>(now_filetime());
            std::unique_ptr<handle> h(new handle);
            try
            {
                n.id = p->svc.db().add_node(n);
            }
            catch (const std::exception& e)
            {
                log_error("drive: " + std::string(e.what()));
                return STATUS_INTERNAL_ERROR;
            }
            state& s = p->nodes[n.id];
            s.n = n;
            s.writing = !directory;
            s.dirty = !directory;
            s.created = !directory;
            p->names[std::make_pair(n.parent, name_key(leaf))] = n.id;
            h->id = n.id;
            h->writable = !directory && (access & (FILE_WRITE_DATA | FILE_APPEND_DATA | GENERIC_WRITE)) != 0;
            p->normalize(n.id, name, fi);
            p->fill(s, fi);
            s.handles++;
            s.open++;
            if (h->writable) s.writable++;
            *context = h.release();
            return STATUS_SUCCESS;
        }

        static NTSTATUS open(FSP_FILE_SYSTEM* f, PWSTR name, UINT32, UINT32 access, PVOID* context,
                             FSP_FSCTL_FILE_INFO* fi)
        {
            impl* p = of(f);
            std::lock_guard<std::mutex> lock(p->m);
            bool parent_missing = false;
            state* s = p->lookup(name, &parent_missing);
            if (!s) return parent_missing ? STATUS_OBJECT_PATH_NOT_FOUND : STATUS_OBJECT_NAME_NOT_FOUND;
            // A file under protection opens for its attributes only, at once: WinFsp holds the
            // creations, the renames and the deletions of the whole drive while an open waits.
            const UINT32 content_access = FILE_READ_DATA | FILE_WRITE_DATA | FILE_APPEND_DATA | FILE_EXECUTE | DELETE;
            if (s->processing && (access & content_access)) return STATUS_SHARING_VIOLATION;
            const bool writable = !s->n.directory && (access & (FILE_WRITE_DATA | FILE_APPEND_DATA)) != 0;
            // The data of a file being written is in clear: it stays with the handles that write it,
            // and a new handle reaches it only once it is protected.
            const UINT32 data_access = FILE_READ_DATA | FILE_WRITE_DATA | FILE_APPEND_DATA | FILE_EXECUTE;
            if (s->writing && (s->writable > 0 || s->dirty) && (access & data_access))
                return STATUS_SHARING_VIOLATION;
            std::unique_ptr<handle> h(new handle);
            h->id = s->n.id;
            h->writable = writable;
            p->normalize(s->n.id, name, fi);
            p->fill(*s, fi);
            s->handles++;
            s->open++;
            if (writable) s->writable++;
            *context = h.release();
            return STATUS_SUCCESS;
        }

        static NTSTATUS overwrite(FSP_FILE_SYSTEM* f, PVOID context, UINT32 attributes, BOOLEAN replace, UINT64,
                                  FSP_FSCTL_FILE_INFO* fi)
        {
            impl* p = of(f);
            std::lock_guard<std::mutex> lock(p->m);
            handle* h = static_cast<handle*>(context);
            state* s = p->find(h->id);
            if (!s) return STATUS_FILE_DELETED;
            p->drop_buffer(*s);
            s->writing = true;
            s->dirty = true;
            s->n.attributes = (replace ? (attributes & kept_attributes)
                                       : (s->n.attributes | (attributes & kept_attributes))) | FILE_ATTRIBUTE_ARCHIVE;
            s->n.written = s->n.changed = static_cast<long long>(now_filetime());
            p->fill(*s, fi);
            return STATUS_SUCCESS;
        }

        static VOID cleanup(FSP_FILE_SYSTEM* f, PVOID context, PWSTR, ULONG flags)
        {
            impl* p = of(f);
            std::unique_lock<std::mutex> lock(p->m);
            handle* h = static_cast<handle*>(context);
            if (h->cleaned) return;
            h->cleaned = true;
            state* s = p->find(h->id);
            if (!s) return;
            s->open = std::max(0, s->open - 1);
            if (h->writable)
            {
                s->writable = std::max(0, s->writable - 1);
                h->writable = false;
            }
            if (flags & FspCleanupDelete)
            {
                if (s->n.id != 1 && !(s->n.directory && p->has_children(s->n.id))) p->remove_node(*s);
                return;
            }
            const long long now = static_cast<long long>(now_filetime());
            if (flags & FspCleanupSetLastWriteTime) s->n.written = now;
            if (flags & FspCleanupSetChangeTime) s->n.changed = now;
            if ((flags & FspCleanupSetArchiveBit) && !s->n.directory) s->n.attributes |= FILE_ATTRIBUTE_ARCHIVE;
            if (s->writable == 0 && s->dirty && !s->n.directory) p->protect(lock, *s);
            else if (!s->writing && (flags & (FspCleanupSetLastWriteTime | FspCleanupSetChangeTime |
                                              FspCleanupSetArchiveBit)))
                p->save(*s);
        }

        static VOID close(FSP_FILE_SYSTEM* f, PVOID context)
        {
            impl* p = of(f);
            std::unique_lock<std::mutex> lock(p->m);
            std::unique_ptr<handle> h(static_cast<handle*>(context));
            const auto it = p->nodes.find(h->id);
            if (it == p->nodes.end()) return;
            state& s = it->second;
            // A handle the kernel refused once it was opened gets a close without cleanup.
            if (!h->cleaned)
            {
                s.open = std::max(0, s.open - 1);
                if (h->writable) s.writable = std::max(0, s.writable - 1);
            }
            s.handles = std::max(0, s.handles - 1);
            if (s.handles > 0) return;
            if (s.deleted)
            {
                p->drop_buffer(s);
                p->nodes.erase(it);
                return;
            }
            if (s.writing && s.writable == 0)
            {
                if (s.dirty && !s.n.directory && !s.processing)
                {
                    // Written after the last cleanup, through a mapping of the file: protected now.
                    const long long id = h->id;
                    p->protect(lock, s);
                    state* after = p->find(id);
                    if (after && after->handles == 0 && !after->writing) after->content.reset();
                    return;
                }
                p->drop_buffer(s);
                s.writing = false;
                s.dirty = false;
            }
            if (!s.writing) s.content.reset();
        }

        static NTSTATUS read(FSP_FILE_SYSTEM* f, PVOID context, PVOID buffer, UINT64 offset, ULONG length,
                             PULONG transferred)
        {
            impl* p = of(f);
            std::unique_lock<std::mutex> lock(p->m);
            handle* h = static_cast<handle*>(context);
            state* s = p->find(h->id);
            if (!s) return STATUS_FILE_DELETED;
            if (s->processing && !p->wait_idle(lock, h->id)) return STATUS_IO_TIMEOUT;
            s = p->find(h->id);
            if (!s) return STATUS_FILE_DELETED;
            std::shared_ptr<const std::string> data;
            if (s->writing)
            {
                // The data in clear goes to the handles that write it, only.
                if (!h->writable) return STATUS_SHARING_VIOLATION;
                if (offset >= s->buffer.size()) return STATUS_END_OF_FILE;
                const size_t n = static_cast<size_t>(std::min<UINT64>(length, s->buffer.size() - offset));
                std::memcpy(buffer, s->buffer.data() + offset, n);
                *transferred = static_cast<ULONG>(n);
                return STATUS_SUCCESS;
            }
            if (!p->load_content(*s)) return STATUS_FILE_CORRUPT_ERROR;
            data = s->content;
            lock.unlock();
            if (offset >= data->size()) return STATUS_END_OF_FILE;
            const size_t n = static_cast<size_t>(std::min<UINT64>(length, data->size() - offset));
            std::memcpy(buffer, data->data() + offset, n);
            *transferred = static_cast<ULONG>(n);
            return STATUS_SUCCESS;
        }

        // Makes s hold its data in its buffer, from its protected content: the start of an edit.
        NTSTATUS start_writing(state& s)
        {
            if (s.writing) return STATUS_SUCCESS;
            if (!load_content(s)) return STATUS_FILE_CORRUPT_ERROR;
            if (!resize_buffer(s, s.content->size())) return STATUS_DISK_FULL;
            if (!s.content->empty()) std::memcpy(&s.buffer[0], s.content->data(), s.content->size());
            s.writing = true;
            return STATUS_SUCCESS;
        }

        static NTSTATUS write(FSP_FILE_SYSTEM* f, PVOID context, PVOID buffer, UINT64 offset, ULONG length,
                              BOOLEAN to_end, BOOLEAN constrained, PULONG transferred, FSP_FSCTL_FILE_INFO* fi)
        {
            impl* p = of(f);
            std::unique_lock<std::mutex> lock(p->m);
            handle* h = static_cast<handle*>(context);
            state* s = p->find(h->id);
            if (!s) return STATUS_FILE_DELETED;
            if (s->processing && !p->wait_idle(lock, h->id)) return STATUS_IO_TIMEOUT;
            s = p->find(h->id);
            if (!s) return STATUS_FILE_DELETED;
            if (s->n.directory) return STATUS_INVALID_DEVICE_REQUEST;
            const NTSTATUS started = p->start_writing(*s);
            if (started != STATUS_SUCCESS) return started;
            UINT64 at = to_end ? s->buffer.size() : offset;
            UINT64 n = length;
            if (constrained)
            {
                if (at >= s->buffer.size())
                {
                    *transferred = 0;
                    p->fill(*s, fi);
                    return STATUS_SUCCESS;
                }
                n = std::min<UINT64>(n, s->buffer.size() - at);
            }
            if (at + n > p->max_bytes) return STATUS_DISK_FULL;
            if (at + n > s->buffer.size() && !p->resize_buffer(*s, static_cast<size_t>(at + n)))
                return STATUS_DISK_FULL;
            if (n) std::memcpy(&s->buffer[static_cast<size_t>(at)], buffer, static_cast<size_t>(n));
            s->dirty = true;
            s->n.written = s->n.changed = static_cast<long long>(now_filetime());
            *transferred = static_cast<ULONG>(n);
            p->fill(*s, fi);
            return STATUS_SUCCESS;
        }

        static NTSTATUS flush(FSP_FILE_SYSTEM* f, PVOID context, FSP_FSCTL_FILE_INFO* fi)
        {
            impl* p = of(f);
            std::lock_guard<std::mutex> lock(p->m);
            if (!context) return STATUS_SUCCESS;
            state* s = p->find(static_cast<handle*>(context)->id);
            if (!s) return STATUS_FILE_DELETED;
            p->fill(*s, fi);
            return STATUS_SUCCESS;
        }

        static NTSTATUS get_file_info(FSP_FILE_SYSTEM* f, PVOID context, FSP_FSCTL_FILE_INFO* fi)
        {
            impl* p = of(f);
            std::lock_guard<std::mutex> lock(p->m);
            state* s = p->find(static_cast<handle*>(context)->id);
            if (!s) return STATUS_FILE_DELETED;
            p->fill(*s, fi);
            return STATUS_SUCCESS;
        }

        static NTSTATUS set_basic_info(FSP_FILE_SYSTEM* f, PVOID context, UINT32 attributes, UINT64 created,
                                       UINT64 accessed, UINT64 written, UINT64 changed, FSP_FSCTL_FILE_INFO* fi)
        {
            impl* p = of(f);
            std::lock_guard<std::mutex> lock(p->m);
            state* s = p->find(static_cast<handle*>(context)->id);
            if (!s) return STATUS_FILE_DELETED;
            if (attributes != INVALID_FILE_ATTRIBUTES) s->n.attributes = attributes & kept_attributes;
            if (created) s->n.created = static_cast<long long>(created);
            if (accessed) s->n.accessed = static_cast<long long>(accessed);
            if (written) s->n.written = static_cast<long long>(written);
            if (changed) s->n.changed = static_cast<long long>(changed);
            if (!s->writing) p->save(*s);
            p->fill(*s, fi);
            return STATUS_SUCCESS;
        }

        static NTSTATUS set_file_size(FSP_FILE_SYSTEM* f, PVOID context, UINT64 size, BOOLEAN allocation,
                                      FSP_FSCTL_FILE_INFO* fi)
        {
            impl* p = of(f);
            std::unique_lock<std::mutex> lock(p->m);
            handle* h = static_cast<handle*>(context);
            state* s = p->find(h->id);
            if (!s) return STATUS_FILE_DELETED;
            if (s->processing && !p->wait_idle(lock, h->id)) return STATUS_IO_TIMEOUT;
            s = p->find(h->id);
            if (!s) return STATUS_FILE_DELETED;
            const UINT64 current = s->writing ? s->buffer.size() : static_cast<UINT64>(s->n.size);
            if (!allocation || size < current)
            {
                if (size > p->max_bytes) return STATUS_DISK_FULL;
                const NTSTATUS started = p->start_writing(*s);
                if (started != STATUS_SUCCESS) return started;
                if (!p->resize_buffer(*s, static_cast<size_t>(size))) return STATUS_DISK_FULL;
                s->dirty = true;
            }
            p->fill(*s, fi);
            return STATUS_SUCCESS;
        }

        static NTSTATUS can_delete(FSP_FILE_SYSTEM* f, PVOID context, PWSTR)
        {
            impl* p = of(f);
            std::lock_guard<std::mutex> lock(p->m);
            state* s = p->find(static_cast<handle*>(context)->id);
            if (!s) return STATUS_FILE_DELETED;
            if (s->n.id == 1) return STATUS_ACCESS_DENIED;
            if (s->n.directory && p->has_children(s->n.id)) return STATUS_DIRECTORY_NOT_EMPTY;
            return STATUS_SUCCESS;
        }

        static NTSTATUS rename(FSP_FILE_SYSTEM* f, PVOID context, PWSTR, PWSTR new_name, BOOLEAN replace)
        {
            impl* p = of(f);
            std::lock_guard<std::mutex> lock(p->m);
            state* s = p->find(static_cast<handle*>(context)->id);
            if (!s || s->n.id == 1 || s->processing) return STATUS_ACCESS_DENIED;
            std::vector<std::wstring> parts = components(new_name);
            if (parts.empty()) return STATUS_OBJECT_NAME_INVALID;
            const std::wstring leaf = parts.back();
            parts.pop_back();
            if (leaf.size() > 255) return STATUS_OBJECT_NAME_INVALID;
            state* parent = p->find(1);
            for (const std::wstring& c : parts)
            {
                parent = parent && parent->n.directory ? p->child(parent->n.id, c) : nullptr;
                if (!parent) return STATUS_OBJECT_PATH_NOT_FOUND;
            }
            if (!parent->n.directory) return STATUS_OBJECT_PATH_NOT_FOUND;
            // A folder never goes inside itself.
            for (state* a = parent; a; a = a->n.id == 1 ? nullptr : p->find(a->n.parent))
                if (a->n.id == s->n.id) return STATUS_INVALID_PARAMETER;
            state* target = p->child(parent->n.id, leaf);
            if (target && target->n.id == s->n.id) target = nullptr;
            if (target)
            {
                if (!replace) return STATUS_OBJECT_NAME_COLLISION;
                if (target->n.directory || target->open > 0 || target->processing) return STATUS_ACCESS_DENIED;
            }
            drive_node next = s->n;
            next.parent = parent->n.id;
            next.name = narrow(leaf);
            next.changed = static_cast<long long>(now_filetime());
            // The target goes and the file takes its name in one transaction, or nothing changes.
            if (target)
            {
                try
                {
                    p->svc.db().replace_node(next, target->n.id);
                }
                catch (const std::exception& e)
                {
                    log_error("drive: " + std::string(e.what()));
                    return STATUS_INTERNAL_ERROR;
                }
                p->remove_node(*target, false);
            }
            else if (!p->save_node(next)) return STATUS_INTERNAL_ERROR;
            p->names.erase(std::make_pair(s->n.parent, name_key(widen(s->n.name))));
            s->n = next;
            p->names[std::make_pair(s->n.parent, name_key(leaf))] = s->n.id;
            return STATUS_SUCCESS;
        }

        static NTSTATUS get_security(FSP_FILE_SYSTEM* f, PVOID, PSECURITY_DESCRIPTOR out, SIZE_T* size)
        {
            return of(f)->security(out, size);
        }

        static NTSTATUS set_security(FSP_FILE_SYSTEM*, PVOID, SECURITY_INFORMATION, PSECURITY_DESCRIPTOR)
        {
            // The permissions of the drive are those of every file: a change is accepted and ignored.
            return STATUS_SUCCESS;
        }

        static NTSTATUS read_directory(FSP_FILE_SYSTEM* f, PVOID context, PWSTR, PWSTR marker, PVOID buffer,
                                       ULONG length, PULONG transferred)
        {
            impl* p = of(f);
            struct entry { std::wstring name; FSP_FSCTL_FILE_INFO info; };
            std::vector<entry> entries;
            {
                std::lock_guard<std::mutex> lock(p->m);
                state* d = p->find(static_cast<handle*>(context)->id);
                if (!d || !d->n.directory) return STATUS_NOT_A_DIRECTORY;
                if (d->n.id != 1)
                {
                    entry dot;
                    dot.name = L".";
                    p->fill(*d, &dot.info);
                    entries.push_back(dot);
                    entry up;
                    up.name = L"..";
                    state* parent = p->find(d->n.parent);
                    p->fill(parent ? *parent : *d, &up.info);
                    entries.push_back(up);
                }
                // In the order of their keys, which the marker of a listing follows.
                for (auto it = p->names.lower_bound(std::make_pair(d->n.id, std::wstring()));
                     it != p->names.end() && it->first.first == d->n.id; ++it)
                {
                    state* c = p->find(it->second);
                    if (!c) continue;
                    entry e;
                    e.name = widen(c->n.name);
                    p->fill(*c, &e.info);
                    entries.push_back(e);
                }
            }
            size_t start = 0;
            if (marker)
            {
                const std::wstring m = name_key(marker);
                start = entries.size();
                for (size_t i = 0; i < entries.size(); ++i)
                    if (name_key(entries[i].name) == m)
                    {
                        start = i + 1;
                        break;
                    }
                if (start == entries.size())
                    for (size_t i = 0; i < entries.size(); ++i)
                        if (entries[i].name != L"." && entries[i].name != L".." && name_key(entries[i].name) > m)
                        {
                            start = i;
                            break;
                        }
            }
            std::vector<char> buf;
            for (size_t i = start; i < entries.size(); ++i)
            {
                const entry& e = entries[i];
                buf.assign(sizeof(FSP_FSCTL_DIR_INFO) + e.name.size() * sizeof(wchar_t), 0);
                FSP_FSCTL_DIR_INFO* di = reinterpret_cast<FSP_FSCTL_DIR_INFO*>(buf.data());
                di->Size = static_cast<UINT16>(buf.size());
                di->FileInfo = e.info;
                std::memcpy(di->FileNameBuf, e.name.data(), e.name.size() * sizeof(wchar_t));
                if (!p->api.add_dir_info(di, buffer, length, transferred)) return STATUS_SUCCESS;
            }
            p->api.add_dir_info(nullptr, buffer, length, transferred);
            return STATUS_SUCCESS;
        }
    };

    drive_host::drive_host(service& svc) : p_(new impl(svc)) {}

    drive_host::~drive_host()
    {
        stop();
        if (p_->sd) LocalFree(p_->sd);
    }

    bool drive_host::start(const std::string& letter, std::string& error)
    {
        impl& p = *p_;
        p.letter = letter;
        if (!load_winfsp(p.api, error))
        {
            p.error = error;
            p.error_code = "winfsp_missing";
            return false;
        }
        if (!p.sd)
        {
            // SYSTEM, the administrators and the users of the console: full control.
            ULONG size = 0;
            if (!ConvertStringSecurityDescriptorToSecurityDescriptorW(
                    L"O:BAG:BAD:P(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)(A;OICI;FA;;;IU)", SDDL_REVISION_1, &p.sd, &size))
            {
                error = p.error = "cannot build the permissions of the drive";
                return false;
            }
            p.sd_size = size;
        }
        p.max_bytes = static_cast<UINT64>(p.svc.settings().get().max_file_mb) << 20;
        MEMORYSTATUSEX memory;
        memory.dwLength = sizeof(memory);
        if (GlobalMemoryStatusEx(&memory))
            p.clear_budget = static_cast<size_t>(std::max<DWORDLONG>(memory.ullTotalPhys / 4, 512ull << 20));
        {
            std::lock_guard<std::mutex> lock(p.m);
            p.nodes.clear();
            p.names.clear();
            p.load();
        }
        FSP_FSCTL_VOLUME_PARAMS vp;
        std::memset(&vp, 0, sizeof(vp));
        vp.Version = sizeof(FSP_FSCTL_VOLUME_PARAMS);
        vp.SectorSize = 4096;
        vp.SectorsPerAllocationUnit = 1;
        vp.MaxComponentLength = 255;
        vp.VolumeCreationTime = now_filetime();
        // The serial number of the volume stays from a mount to the next, as the programs that
        // remember a volume by it expect.
        std::string serial;
        try
        {
            serial = p.svc.db().value("drive_serial");
            if (serial.size() != 8)
            {
                serial = random_hex(8);
                p.svc.db().set_value("drive_serial", serial);
            }
        }
        catch (const std::exception& e)
        {
            log_error("drive: " + std::string(e.what()));
            serial = random_hex(8);
        }
        vp.VolumeSerialNumber = static_cast<UINT32>(std::strtoul(serial.c_str(), nullptr, 16));
        vp.FileInfoTimeout = 0;
        vp.CaseSensitiveSearch = 0;
        vp.CasePreservedNames = 1;
        vp.UnicodeOnDisk = 1;
        vp.PersistentAcls = 1;
        vp.PostCleanupWhenModifiedOnly = 0;
        vp.FlushAndPurgeOnCleanup = 1;
        vp.UmFileContextIsUserContext2 = 1;
        const wchar_t fsname[] = L"Pseudalis";
        std::memcpy(vp.FileSystemName, fsname, sizeof(fsname));

        p.iface = FSP_FILE_SYSTEM_INTERFACE();
#define PSD_OPERATION(member, function) p.iface.member = guarded<decltype(impl::function)>::call<impl::function>
        PSD_OPERATION(GetVolumeInfo, get_volume_info);
        PSD_OPERATION(SetVolumeLabel, set_volume_label);
        PSD_OPERATION(GetSecurityByName, get_security_by_name);
        PSD_OPERATION(Create, create);
        PSD_OPERATION(Open, open);
        PSD_OPERATION(Overwrite, overwrite);
        PSD_OPERATION(Cleanup, cleanup);
        PSD_OPERATION(Close, close);
        PSD_OPERATION(Read, read);
        PSD_OPERATION(Write, write);
        PSD_OPERATION(Flush, flush);
        PSD_OPERATION(GetFileInfo, get_file_info);
        PSD_OPERATION(SetBasicInfo, set_basic_info);
        PSD_OPERATION(SetFileSize, set_file_size);
        PSD_OPERATION(CanDelete, can_delete);
        PSD_OPERATION(Rename, rename);
        PSD_OPERATION(GetSecurity, get_security);
        PSD_OPERATION(SetSecurity, set_security);
        PSD_OPERATION(ReadDirectory, read_directory);
#undef PSD_OPERATION

        wchar_t device[] = L"" FSP_FSCTL_DISK_DEVICE_NAME;
        NTSTATUS st = p.api.create(device, &vp, &p.iface, &p.fs);
        if (st != STATUS_SUCCESS)
        {
            error = p.error = "WinFsp cannot create the file system (status " + std::to_string(st) + ")";
            p.fs = nullptr;
            return false;
        }
        p.fs->UserContext = &p;
        std::wstring mount = widen(letter + ":");
        st = p.api.set_mount_point(p.fs, &mount[0]);
        if (st != STATUS_SUCCESS)
        {
            error = p.error = "the letter " + letter + ": is in use or cannot be mounted (status " +
                              std::to_string(st) + ")";
            p.api.destroy(p.fs);
            p.fs = nullptr;
            return false;
        }
        p.stopping = false;
        p.notifier = std::thread([&p]() { p.notify_loop(); });
        // A protection holds its thread of the dispatcher while the engine works: twice as many
        // threads as processors, 8 to 32, keep the drive answering meanwhile.
        const unsigned processors = std::max(1u, std::thread::hardware_concurrency());
        st = p.api.start_dispatcher(p.fs, std::max(8u, std::min(32u, 2 * processors)));
        if (st != STATUS_SUCCESS)
        {
            error = p.error = "WinFsp cannot start the drive (status " + std::to_string(st) + ")";
            stop();
            return false;
        }
        p.mounted = true;
        p.error.clear();
        p.error_code.clear();
        return true;
    }

    void drive_host::stop()
    {
        impl& p = *p_;
        if (p.fs) p.api.stop_dispatcher(p.fs);
        // The notifier stops before the file system it notifies goes.
        {
            std::lock_guard<std::mutex> lock(p.notify_m);
            p.stopping = true;
            p.notify_cv.notify_all();
        }
        if (p.notifier.joinable()) p.notifier.join();
        if (p.fs)
        {
            p.api.remove_mount_point(p.fs);
            p.api.destroy(p.fs);
            p.fs = nullptr;
        }
        std::lock_guard<std::mutex> lock(p.m);
        for (auto& kv : p.nodes) erase_clear(kv.second.buffer);
        p.clear_bytes = 0;
        p.mounted = false;
    }

    json drive_host::status() const
    {
        json j = json::object();
        j["letter"] = p_->letter;
        j["mounted"] = p_->mounted;
        long long files = 0, bytes = 0;
        p_->svc.db().drive_usage(files, bytes);
        j["files"] = files;
        j["bytes"] = bytes;
        j["error"] = p_->error;
        if (!p_->error_code.empty()) j["error_code"] = p_->error_code;
        return j;
    }
}
