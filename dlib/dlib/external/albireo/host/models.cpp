// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "models.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstring>
#include <exception>
#include <fstream>
#include <map>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

#include <zlib.h>

#include <dlib/data_io/gguf_reader.h>

#include "../common/log.h"
#include "../common/platform.h"
#include "crypto.h"
#include "http_fetch.h"

namespace albireo
{

    namespace
    {
        // A file written at given offsets by several threads.
        class random_access_file
        {
        public:
            bool open(const std::string& path, uint64_t size)
            {
#if defined(_WIN32)
                h_ = CreateFileW(widen(path).c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ, nullptr,
                    OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
                if (h_ == INVALID_HANDLE_VALUE) return false;
                LARGE_INTEGER s;
                s.QuadPart = static_cast<LONGLONG>(size);
                LARGE_INTEGER cur;
                GetFileSizeEx(h_, &cur);
                if (static_cast<uint64_t>(cur.QuadPart) != size)
                {
                    if (!SetFilePointerEx(h_, s, nullptr, FILE_BEGIN) || !SetEndOfFile(h_)) return false;
                }
                return true;
#else
                fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
                if (fd_ < 0) return false;
                return ftruncate(fd_, static_cast<off_t>(size)) == 0;
#endif
            }

            bool write_at(uint64_t offset, const char* data, size_t n)
            {
#if defined(_WIN32)
                while (n > 0)
                {
                    OVERLAPPED ov = {};
                    ov.Offset = static_cast<DWORD>(offset & 0xFFFFFFFFu);
                    ov.OffsetHigh = static_cast<DWORD>(offset >> 32);
                    DWORD w = 0;
                    const DWORD chunk = static_cast<DWORD>(std::min<size_t>(n, 1u << 30));
                    if (!WriteFile(h_, data, chunk, &w, &ov) || w == 0) return false;
                    offset += w;
                    data += w;
                    n -= w;
                }
                return true;
#else
                while (n > 0)
                {
                    const ssize_t w = pwrite(fd_, data, n, static_cast<off_t>(offset));
                    if (w <= 0) return false;
                    offset += static_cast<uint64_t>(w);
                    data += w;
                    n -= static_cast<size_t>(w);
                }
                return true;
#endif
            }

            void close()
            {
#if defined(_WIN32)
                if (h_ != INVALID_HANDLE_VALUE)
                {
                    FlushFileBuffers(h_);
                    CloseHandle(h_);
                }
                h_ = INVALID_HANDLE_VALUE;
#else
                if (fd_ >= 0)
                {
                    fsync(fd_);
                    ::close(fd_);
                }
                fd_ = -1;
#endif
            }

            ~random_access_file() { close(); }

        private:
#if defined(_WIN32)
            HANDLE h_ = INVALID_HANDLE_VALUE;
#else
            int fd_ = -1;
#endif
        };

        std::string localized(const json& text, const std::string& language)
        {
            if (text.is_string()) return text.as_string();
            if (text[language].is_string()) return text[language].as_string();
            if (text["en"].is_string()) return text["en"].as_string();
            for (const auto& kv : text.items())
                if (kv.second.is_string()) return kv.second.as_string();
            return std::string();
        }

        bool safe_id(const std::string& id)
        {
            if (id.empty() || id.size() > 128) return false;
            for (char c : id)
                if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_' || c == '.')) return false;
            return id[0] != '.';
        }
    }

    json describe_gguf(const std::string& path)
    {
        json d = json::object();
        try
        {
            dlib::gguf_reader g(path);
            const std::string arch = g.get_str("general.architecture", "");
            d["name"] = g.get_str("general.name", file_name(path));
            d["arch"] = arch;
            d["context"] = static_cast<long long>(g.get_int(arch + ".context_length", 0));
            d["experts"] = static_cast<long long>(g.get_int(arch + ".expert_count", 0));
            uint64_t params = 0;
            std::map<std::string, uint64_t> by_type;
            for (const auto& t : g.tensors())
            {
                params += t.n_elements();
                by_type[std::to_string(static_cast<unsigned>(t.type))] += t.nbytes;
            }
            d["parameters"] = static_cast<unsigned long long>(params);
            static const std::map<std::string, std::string> names = { { "0", "F32" }, { "1", "F16" }, { "2", "Q4_0" },
                { "3", "Q4_1" }, { "6", "Q5_0" }, { "7", "Q5_1" }, { "8", "Q8_0" }, { "10", "Q2_K" }, { "11", "Q3_K" },
                { "12", "Q4_K" }, { "13", "Q5_K" }, { "14", "Q6_K" }, { "20", "IQ4_NL" }, { "30", "BF16" } };
            uint64_t most = 0;
            std::string q;
            for (const auto& kv : by_type)
                if (kv.second > most) { most = kv.second; q = kv.first; }
            d["quantization"] = names.count(q) ? names.at(q) : q;
            const std::string tmpl = g.get_str("tokenizer.chat_template", "");
            // An embedding model declares how it pools its states; it serves the document
            // search and nothing else.
            const bool embedding = g.get_int(arch + ".pooling_type", 0) != 0;
            d["embedding"] = embedding;
            if (embedding)
            {
                json roles = json::array();
                roles.push_back("embedding");
                d["roles"] = roles;
            }
            d["reasoning"] = !embedding && (tmpl.find("think") != std::string::npos
                || tmpl.find("[THINK]") != std::string::npos);
            d["tools"] = !embedding && tmpl.find("tool") != std::string::npos;
        }
        catch (const std::exception& e)
        {
            d["error"] = e.what();
        }
        return d;
    }

    struct model_manager::job
    {
        struct chunk
        {
            std::string url;
            uint64_t from = 0;        // byte range in the resource
            uint64_t length = 0;      // bytes of the resource (compressed size for gzip)
            uint64_t offset = 0;      // where the bytes go in the file
            std::atomic<uint64_t> done{0};   // bytes of the resource received
            bool gzip = false;
            uint64_t output = 0;      // bytes the part produces (gzip: decompressed size)
            size_t part = 0;
        };

        std::string id;
        json entry;
        std::string target, partial;
        uint64_t total = 0;
        std::vector<std::unique_ptr<chunk>> chunks;
        std::atomic<uint64_t> received{0};
        std::atomic<bool> cancel{false};
        std::string state = "queued";
        std::string error;
        double speed = 0;
        int64_t started = 0;
        std::thread worker;
    };

    model_manager::model_manager(store& db, const std::string& models_dir, const std::string& catalog_url)
        : db_(db), dir_(models_dir), catalog_url_(catalog_url)
    {
        make_dirs(dir_);
        catalog_ = json::parse_or_null(db_.value("catalog", "{}"));
        if (!catalog_.is_object()) catalog_ = json::object();
    }

    model_manager::~model_manager()
    {
        stop();
        if (refresher_.joinable()) refresher_.join();
        std::map<std::string, std::shared_ptr<job>> jobs;
        {
            std::lock_guard<std::mutex> lock(m_);
            jobs = jobs_;
        }
        for (auto& kv : jobs)
            if (kv.second->worker.joinable()) kv.second->worker.join();
    }

    void model_manager::stop()
    {
        stopped_ = true;
        std::lock_guard<std::mutex> lock(m_);
        for (auto& kv : jobs_) kv.second->cancel = true;
    }

    bool model_manager::install_catalog(const std::string& body, const std::string& sig, bool keep_newer,
        json& result, std::string& error)
    {
        if (!verify_signature(body, sig, feed_public_key_pem))
        {
            error = "the signature of the catalog is not valid";
            return false;
        }
        json c;
        if (!json::parse(body, c, &error) || !c["models"].is_array())
        {
            error = "the catalog is not valid: " + error;
            return false;
        }
        {
            std::lock_guard<std::mutex> lock(m_);
            catalog_state_ = "ok";
            catalog_error_.clear();
            // The times of publication, "YYYY-MM-DDTHH:MM:SSZ" (the day alone in the first
            // catalogs), compare as strings: an older catalog, replayed or brought on an old
            // medium, never replaces a newer one.
            const json& current = catalog_;
            if (keep_newer && current["updated"].as_string() > c["updated"].as_string())
            {
                result["state"] = "kept";
                return true;
            }
            catalog_ = c;
            // Written under the lock, so that of two catalogs installed at once, from the
            // network and from a medium, the database keeps the one kept in memory.
            db_.set_value("catalog", body);
            db_.set_value("catalog_time", std::to_string(now_unix()));
        }
        ++generation_;
        result["models"] = static_cast<long long>(c["models"].size());
        log_info("models: catalog of " + std::to_string(c["models"].size()) + " models, published "
            + c["updated"].as_string());
        return true;
    }

    bool model_manager::refresh_catalog(std::string& error)
    {
        std::string body, sig;
        if (!http_get(catalog_url_, body, error)) return false;
        if (!http_get(catalog_url_ + ".sig", sig, error, 4096))
        {
            error = "the catalog has no signature: " + error;
            return false;
        }
        json result = json::object();
        return install_catalog(body, sig, true, result, error);
    }

    void model_manager::refresh_catalog_async()
    {
        std::lock_guard<std::mutex> start(refresher_m_);
        if (stopped_ || refreshing_.exchange(true)) return;
        if (refresher_.joinable()) refresher_.join();
        std::string before;
        {
            std::lock_guard<std::mutex> lock(m_);
            before = catalog_state_;
            catalog_state_ = "checking";
        }
        refresher_ = std::thread([this, before]() {
            std::string error;
            bool ok = false;
            try
            {
                ok = refresh_catalog(error);
            }
            catch (const std::exception& e)
            {
                // A database that cannot be written, for instance a full disk.
                error = e.what();
            }
            // A refresh ended by the service, which stops or goes offline, is no error.
            const bool ended = stopped_ || http_cancelled();
            if (!ok)
            {
                std::lock_guard<std::mutex> lock(m_);
                catalog_state_ = ended ? before : std::string("error");
                if (!ended) catalog_error_ = error;
            }
            if (!ok && !ended) log_warning("models: catalog not refreshed: " + error);
            refreshing_ = false;
        });
    }

    json model_manager::import_catalog(const std::string& folder)
    {
        json result = json::object();
        std::string body, sig, error;
        if (!read_file(join_path(folder, "catalog.json"), body, 16u << 20)
            || !read_file(join_path(folder, "catalog.json.sig"), sig, 4096))
        {
            result["error"] = "no signed catalog (catalog.json and catalog.json.sig) in " + folder;
            return result;
        }
        if (!install_catalog(body, sig, true, result, error)) result["error"] = error;
        return result;
    }

    json model_manager::catalog_status() const
    {
        json s = json::object();
        {
            std::lock_guard<std::mutex> lock(m_);
            s["state"] = catalog_state_;
            if (!catalog_error_.empty()) s["error"] = catalog_error_;
            s["updated"] = catalog_["updated"].as_string();
            s["models"] = static_cast<long long>(catalog_["models"].size());
        }
        s["time"] = static_cast<long long>(std::atoll(db_.value("catalog_time", "0").c_str()));
        return s;
    }

    json model_manager::catalog() const
    {
        std::lock_guard<std::mutex> lock(m_);
        return catalog_;
    }

    bool model_manager::catalog_entry(const std::string& id, json& entry) const
    {
        std::lock_guard<std::mutex> lock(m_);
        for (const auto& e : catalog_["models"].elements())
            if (e["id"].as_string() == id)
            {
                entry = e;
                return true;
            }
        return false;
    }

    json model_manager::overview(const std::string& language) const
    {
        const std::vector<installed_model> installed = db_.models();
        json out = json::array();
        std::map<std::string, bool> seen;
        json cat = catalog();
        for (const auto& e : cat["models"].elements())
        {
            json m = e;
            m["description"] = localized(e["description"], language);
            m["installed"] = false;
            const std::string id = e["id"].as_string();
            for (const auto& im : installed)
                if (im.id == id)
                {
                    m["installed"] = true;
                    m["file"] = im.file;
                    m["local_sha256"] = im.sha256;
                    m["verified_at"] = static_cast<long long>(im.verified_at);
                    m["up_to_date"] = im.sha256 == e["sha256"].as_string();
                }
            seen[id] = true;
            out.push_back(m);
        }
        for (const auto& im : installed)
        {
            if (seen.count(im.id)) continue;
            json m = im.descriptor.is_object() ? im.descriptor : json::object();
            m["id"] = im.id;
            if (!m.contains("name")) m["name"] = im.name;
            m["description"] = localized(m["description"], language);
            m["installed"] = true;
            m["file"] = im.file;
            m["size"] = static_cast<unsigned long long>(im.size);
            m["local_sha256"] = im.sha256;
            m["source"] = im.source;
            out.push_back(m);
        }
        json result = json::object();
        result["models"] = out;
        result["downloads"] = downloads();
        const json cs = catalog_status();
        result["catalog_time"] = cs["time"];
        result["catalog"] = cs;
        return result;
    }

    json model_manager::downloads() const
    {
        std::lock_guard<std::mutex> lock(m_);
        json out = json::array();
        for (const auto& kv : jobs_)
        {
            const job& j = *kv.second;
            json d = json::object();
            d["id"] = j.id;
            d["state"] = j.state;
            d["received"] = static_cast<unsigned long long>(j.received.load());
            d["total"] = static_cast<unsigned long long>(j.total);
            d["speed"] = j.speed;
            if (!j.error.empty()) d["error"] = j.error;
            out.push_back(d);
        }
        return out;
    }

    bool model_manager::start_download(const std::string& id, std::string& error)
    {
        if (stopped_)
        {
            error = "the service stops";
            return false;
        }
        json entry;
        if (!catalog_entry(id, entry))
        {
            error = "the model '" + id + "' is not in the catalog";
            return false;
        }
        if (!safe_id(id))
        {
            error = "invalid model identifier";
            return false;
        }
        std::shared_ptr<job> j(new job());
        j->id = id;
        j->entry = entry;
        j->target = join_path(dir_, id + ".gguf");
        j->partial = j->target + ".partial";

        // Parts, then segments of the uncompressed parts.
        const uint64_t segment = 64ull << 20;
        uint64_t offset = 0;
        size_t part = 0;
        for (const auto& f : entry["files"].elements())
        {
            const std::string url = f["url"].as_string();
            const uint64_t size = static_cast<uint64_t>(f["size"].as_int(0));
            const bool gz = f["compression"].as_string() == "gzip";
            if (url.empty() || size == 0)
            {
                error = "the catalog entry of '" + id + "' lists an invalid file";
                return false;
            }
            if (gz)
            {
                std::unique_ptr<job::chunk> c(new job::chunk());
                c->url = url;
                c->from = 0;
                c->length = static_cast<uint64_t>(f["compressed_size"].as_int(0));
                c->offset = offset;
                c->gzip = true;
                c->output = size;
                c->part = part;
                if (c->length == 0)
                {
                    error = "the catalog entry of '" + id + "' gives no compressed size";
                    return false;
                }
                j->chunks.push_back(std::move(c));
            }
            else
            {
                for (uint64_t s = 0; s < size; s += segment)
                {
                    std::unique_ptr<job::chunk> c(new job::chunk());
                    c->url = url;
                    c->from = s;
                    c->length = std::min(segment, size - s);
                    c->offset = offset + s;
                    c->output = c->length;
                    c->part = part;
                    j->chunks.push_back(std::move(c));
                }
            }
            offset += size;
            ++part;
        }
        j->total = offset;
        if (static_cast<uint64_t>(entry["size"].as_int(static_cast<long long>(offset))) != offset)
        {
            error = "the sizes of the parts of '" + id + "' do not add up to the size of the model";
            return false;
        }
        const uint64_t room = free_disk_space(dir_);
        if (room < offset + (offset / 20) + (256ull << 20) && !file_exists(j->partial))
        {
            error = "not enough disk space: " + std::to_string(offset >> 20) + " MB needed, "
                + std::to_string(room >> 20) + " MB available";
            return false;
        }

        // Resume what a previous run received.
        const json saved = db_.download_state(id);
        if (saved.is_object() && saved["sha256"].as_string() == entry["sha256"].as_string() && file_exists(j->partial)
            && saved["done"].size() == j->chunks.size())
        {
            for (size_t i = 0; i < j->chunks.size(); ++i)
            {
                job::chunk& c = *j->chunks[i];
                uint64_t d = static_cast<uint64_t>(saved["done"][i].as_int(0));
                if (c.gzip && d < c.length) d = 0;   // a compressed stream restarts from its beginning
                c.done = std::min(d, c.length);
                j->received += c.gzip ? (c.done == c.length ? c.output : 0) : c.done.load();
            }
        }

        std::lock_guard<std::mutex> lock(m_);
        auto it = jobs_.find(id);
        if (it != jobs_.end())
        {
            if (it->second->state == "downloading" || it->second->state == "verifying" || it->second->state == "queued")
                return true;
            if (it->second->worker.joinable()) it->second->worker.join();
        }
        jobs_[id] = j;
        j->started = now_unix();
        j->worker = std::thread([this, j]() {
            try
            {
                run_job(j);
            }
            catch (const std::exception& e)
            {
                // A database or a disk that fails under the download.
                {
                    std::lock_guard<std::mutex> lock(m_);
                    j->state = "error";
                    j->error = e.what();
                }
                log_error("models: download of " + j->id + ": " + e.what());
            }
        });
        return true;
    }

    void model_manager::cancel_download(const std::string& id)
    {
        std::lock_guard<std::mutex> lock(m_);
        auto it = jobs_.find(id);
        if (it != jobs_.end()) it->second->cancel = true;
        db_.remove_download_state(id);
    }

    void model_manager::resume_pending()
    {
        for (const auto& id : db_.pending_downloads())
        {
            std::string error;
            if (!start_download(id, error)) log_warning("models: cannot resume " + id + ": " + error);
        }
    }

    namespace
    {
        // SHA-256 of the bytes [offset, offset + length) of a file.
        std::string sha256_range(const std::string& path, uint64_t offset, uint64_t length)
        {
#if defined(_WIN32)
            std::ifstream in(widen(path).c_str(), std::ios::binary);
#else
            std::ifstream in(path, std::ios::binary);
#endif
            if (!in) return std::string();
            in.seekg(static_cast<std::streamoff>(offset));
            sha256_stream s;
            std::vector<char> buf(1 << 20);
            while (length > 0 && in)
            {
                const size_t n = static_cast<size_t>(std::min<uint64_t>(length, buf.size()));
                in.read(buf.data(), static_cast<std::streamsize>(n));
                if (in.gcount() != static_cast<std::streamsize>(n)) return std::string();
                s.update(buf.data(), n);
                length -= n;
            }
            return s.hex();
        }
    }

    void model_manager::run_job(std::shared_ptr<job> j)
    {
        auto set_state = [&](const std::string& s, const std::string& e = std::string()) {
            std::lock_guard<std::mutex> lock(m_);
            j->state = s;
            j->error = e;
        };
        auto save = [&]() {
            json st = json::object();
            st["sha256"] = j->entry["sha256"];
            json done = json::array();
            for (const auto& c : j->chunks) done.push_back(static_cast<unsigned long long>(c->done.load()));
            st["done"] = done;
            db_.set_download_state(j->id, st);
        };

        log_info("models: downloading " + j->id + " (" + std::to_string(j->total >> 20) + " MB, "
            + std::to_string(j->chunks.size()) + " segments)");
        for (int round = 0; round < 2; ++round)
        {
            set_state("downloading");
            random_access_file file;
            if (!file.open(j->partial, j->total))
            {
                set_state("error", "cannot create " + j->partial);
                return;
            }
            save();

            std::mutex chunk_m;
            size_t next = 0;
            std::atomic<bool> failed{false};
            std::string failure;

            auto worker = [&]() {
                for (;;)
                {
                    job::chunk* cp = nullptr;
                    {
                        std::lock_guard<std::mutex> lock(chunk_m);
                        while (next < j->chunks.size() && j->chunks[next]->done >= j->chunks[next]->length) ++next;
                        if (next >= j->chunks.size() || failed || j->cancel) return;
                        cp = j->chunks[next++].get();
                    }
                    job::chunk& c = *cp;
                    std::string error;
                    bool ok = false;
                    for (int attempt = 0; attempt < 4 && !ok && !j->cancel; ++attempt)
                    {
                        // A pause before each new attempt, which a cancellation ends.
                        for (int t = 0; t < 20 * attempt && !j->cancel && !http_cancelled(); ++t)
                            std::this_thread::sleep_for(std::chrono::milliseconds(100));
                        if (j->cancel) break;
                        if (c.gzip)
                        {
                            // Inflated as it arrives, written at the offsets of the part.
                            z_stream zs;
                            std::memset(&zs, 0, sizeof(zs));
                            if (inflateInit2(&zs, 15 + 32) != Z_OK) break;
                            uint64_t produced = 0;
                            std::vector<char> out(1 << 20);
                            bool bad = false;
                            uint64_t got_in = 0;
                            ok = http_get_range(c.url, 0, -1, [&](const char* data, size_t n) {
                                if (j->cancel) return false;
                                zs.next_in = reinterpret_cast<Bytef*>(const_cast<char*>(data));
                                zs.avail_in = static_cast<uInt>(n);
                                while (zs.avail_in > 0)
                                {
                                    zs.next_out = reinterpret_cast<Bytef*>(out.data());
                                    zs.avail_out = static_cast<uInt>(out.size());
                                    const int r = inflate(&zs, Z_NO_FLUSH);
                                    if (r != Z_OK && r != Z_STREAM_END) { bad = true; return false; }
                                    const size_t got = out.size() - zs.avail_out;
                                    const bool fits = produced + got <= c.output;
                                    if (!fits || !file.write_at(c.offset + produced, out.data(), got))
                                    {
                                        bad = true;
                                        return false;
                                    }
                                    produced += got;
                                    j->received += got;
                                    if (r == Z_STREAM_END) break;
                                }
                                got_in += n;
                                return true;
                            }, error);
                            inflateEnd(&zs);
                            if (ok && (bad || produced != c.output))
                            {
                                ok = false;
                                error = "the compressed part " + std::to_string(c.part) + " is corrupt";
                            }
                            if (ok) c.done = c.length;
                            else j->received -= produced;
                        }
                        else
                        {
                            ok = http_get_range(c.url, c.from + c.done, static_cast<int64_t>(c.from + c.length - 1),
                                [&](const char* data, size_t n) {
                                    if (j->cancel) return false;
                                    const uint64_t done = c.done;
                                    const size_t take = static_cast<size_t>(std::min<uint64_t>(n, c.length - done));
                                    if (!file.write_at(c.offset + done, data, take)) return false;
                                    c.done = done + take;
                                    j->received += take;
                                    return true;
                                }, error);
                            ok = ok && c.done == c.length;
                        }
                        // The downloads ended for the whole service (it stops, or goes
                        // offline) are not attempted again; they resume later.
                        if (!ok && http_cancelled()) break;
                    }
                    if (!ok && !j->cancel)
                    {
                        std::lock_guard<std::mutex> lock(chunk_m);
                        failed = true;
                        failure = error.empty() ? "download failed" : error;
                        return;
                    }
                }
            };

            std::vector<std::thread> threads;
            for (int t = 0; t < 4; ++t) threads.emplace_back(worker);
            auto last_save = std::chrono::steady_clock::now();
            auto speed_t = std::chrono::steady_clock::now();
            uint64_t speed_bytes = j->received;
            // Progress, speed and periodic saving while the workers run.
            for (;;)
            {
                bool running = false;
                for (const auto& c : j->chunks) running = running || c->done < c->length;
                if (!running || failed || j->cancel) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                const auto now = std::chrono::steady_clock::now();
                const double dt = std::chrono::duration<double>(now - speed_t).count();
                if (dt >= 2.0)
                {
                    std::lock_guard<std::mutex> lock(m_);
                    const uint64_t r = j->received;
                    j->speed = r > speed_bytes ? (r - speed_bytes) / dt : 0.0;
                    speed_bytes = r;
                    speed_t = now;
                }
                if (std::chrono::duration<double>(now - last_save).count() >= 5.0)
                {
                    save();
                    last_save = now;
                }
            }
            for (auto& t : threads) t.join();
            file.close();

            if (j->cancel)
            {
                set_state("cancelled");
                if (db_.download_state(j->id).is_null()) remove_file(j->partial);
                log_info("models: download of " + j->id + " cancelled");
                return;
            }
            save();
            if (failed)
            {
                set_state("error", failure);
                log_error("models: download of " + j->id + " failed: " + failure);
                return;
            }

            set_state("verifying");
            const std::string sha = sha256_file(j->partial, nullptr, &j->cancel);
            if (j->cancel)
            {
                // Stopped during the check: the file is kept and checked again at the
                // next start, unless the download itself was cancelled.
                set_state("cancelled");
                if (db_.download_state(j->id).is_null()) remove_file(j->partial);
                return;
            }
            if (sha == j->entry["sha256"].as_string())
            {
                std::string error;
                if (!rename_file(j->partial, j->target) ||
                    !install_file(j->id, j->target, j->entry, sha, "catalog", error))
                {
                    set_state("error", error.empty() ? "cannot install the file" : error);
                    return;
                }
                db_.remove_download_state(j->id);
                set_state("done");
                log_info("models: " + j->id + " installed");
                return;
            }

            // The parts whose own hash fails are fetched again, once.
            bool any = false;
            uint64_t offset = 0;
            size_t part = 0;
            for (const auto& f : j->entry["files"].elements())
            {
                const uint64_t size = static_cast<uint64_t>(f["size"].as_int(0));
                const std::string expected = f["sha256"].as_string();
                if (round == 0 && !expected.empty() && sha256_range(j->partial, offset, size) != expected)
                {
                    for (auto& c : j->chunks)
                        if (c->part == part)
                        {
                            j->received -= c->gzip ? c->output : c->done.load();
                            c->done = 0;
                        }
                    any = true;
                    log_warning("models: part " + std::to_string(part) + " of " + j->id + " is corrupt, fetched again");
                }
                offset += size;
                ++part;
            }
            if (!any) break;
        }
        set_state("error", "the downloaded file does not match the hash of the catalog");
        log_error("models: " + j->id + " does not match its hash");
        db_.remove_download_state(j->id);
        remove_file(j->partial);
    }

    bool model_manager::install_file(const std::string& id, const std::string& file, const json& descriptor,
        const std::string& sha, const std::string& source, std::string& error)
    {
        installed_model m;
        m.id = id;
        m.name = descriptor["name"].str(id);
        m.file = file;
        m.size = file_size(file);
        m.sha256 = sha;
        m.source = source;
        m.installed_at = now_unix();
        m.verified_at = now_unix();
        m.descriptor = descriptor;
        if (m.size == 0)
        {
            error = file + " is empty";
            return false;
        }
        db_.put_model(m);
        if (installed_cb_) installed_cb_(m);
        return true;
    }

    bool model_manager::remove(const std::string& id, std::string& error)
    {
        installed_model m;
        if (!db_.find_model(id, m))
        {
            error = "the model '" + id + "' is not installed";
            return false;
        }
        if (file_exists(m.file) && !remove_file(m.file))
        {
            error = "cannot delete " + m.file + " (in use?)";
            return false;
        }
        db_.remove_model(id);
        log_info("models: " + id + " removed");
        return true;
    }

    json model_manager::verify(const std::string& id)
    {
        json r = json::object();
        r["id"] = id;
        installed_model m;
        if (!db_.find_model(id, m))
        {
            r["error"] = "not installed";
            return r;
        }
        const std::string sha = sha256_file(m.file);
        r["sha256"] = sha;
        r["local_match"] = !sha.empty() && sha == m.sha256;
        json entry;
        if (catalog_entry(id, entry))
        {
            r["catalog_sha256"] = entry["sha256"];
            r["catalog_match"] = sha == entry["sha256"].as_string();
        }
        if (!sha.empty() && sha == m.sha256) db_.set_verified(id, now_unix());
        return r;
    }

    bool model_manager::import_file(const std::string& f, const std::string& original_name,
        const std::string& known_sha, bool move, std::string& id, std::string& error)
    {
        const std::string sha = known_sha.empty() ? sha256_file(f, nullptr, &stopped_) : known_sha;
        if (sha.empty())
        {
            error = stopped_ ? "the service stops" : "unreadable";
            return false;
        }
        // A catalog entry with this hash gives the identifier and the description.
        json entry;
        bool known = false;
        {
            std::lock_guard<std::mutex> lock(m_);
            const json& catalog = catalog_;
            for (const auto& e : catalog["models"].elements())
                if (e["sha256"].as_string() == sha)
                {
                    entry = e;
                    known = true;
                }
        }
        if (known)
        {
            id = entry["id"].as_string();
        }
        else
        {
            entry = describe_gguf(f);
            if (entry.contains("error"))
            {
                error = entry["error"].as_string();
                return false;
            }
            std::string base = file_name(original_name);
            if (base.size() > 5 && base.substr(base.size() - 5) == ".gguf") base = base.substr(0, base.size() - 5);
            for (char& c : base)
                if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_' || c == '.')) c = '-';
            id = base;
            entry["id"] = id;
            // A sidecar description written next to the file, if any.
            std::string side;
            if (f.size() > 5 && f.substr(f.size() - 5) == ".gguf" &&
                read_file(f.substr(0, f.size() - 5) + ".json", side, 1u << 20))
            {
                const json s = json::parse_or_null(side);
                for (const auto& kv : s.items())
                    if (kv.first != "id" && kv.first != "files" && kv.first != "sha256") entry[kv.first] = kv.second;
            }
        }
        if (!safe_id(id))
        {
            error = "invalid identifier";
            return false;
        }
        const std::string target = join_path(dir_, id + ".gguf");
        installed_model existing;
        if (db_.find_model(id, existing) && existing.sha256 == sha && file_exists(existing.file))
        {
            if (move && f != existing.file) remove_file(f);
            return true;
        }
        // Moved when asked and when the models folder is on the same volume; copied
        // otherwise, through a temporary name, then checked again.
        if (target != f && !(move && rename_file(f, target)))
        {
            const std::string tmp = target + ".partial";
            const bool copied = copy_file(f, tmp, &stopped_);
            if (!copied || sha256_file(tmp, nullptr, &stopped_) != sha || !rename_file(tmp, target))
            {
                remove_file(tmp);
                error = stopped_ ? "the service stops" : "copy failed";
                return false;
            }
            if (move) remove_file(f);
        }
        return install_file(id, target, entry, sha, known ? "catalog" : "import", error);
    }

    json model_manager::import_path(const std::string& path, bool move)
    {
        json result = json::object();
        json imported = json::array();
        json errors = json::array();
        std::vector<std::string> files;
        if (dir_exists(path))
        {
            // The folder and its models subfolder, the layout of an offline bundle.
            for (const std::string& dir : { path, join_path(path, "models") })
                for (const auto& n : list_dir(dir))
                    if (n.size() > 5 && n.substr(n.size() - 5) == ".gguf") files.push_back(join_path(dir, n));
        }
        else if (file_exists(path))
        {
            files.push_back(path);
        }
        else
        {
            errors.push_back(path + ": not found, or not readable by the service");
        }
        for (const auto& f : files)
        {
            if (stopped_) break;
            std::string id, error;
            if (!import_file(f, f, std::string(), move, id, error))
            {
                errors.push_back(f + ": " + error);
                continue;
            }
            imported.push_back(id);
            log_info("models: imported " + id + " from " + f);
        }
        result["imported"] = imported;
        result["errors"] = errors;
        return result;
    }

    json model_manager::import_received(const std::string& file, const std::string& original_name,
        const std::string& sha)
    {
        json result = json::object();
        json imported = json::array();
        json errors = json::array();
        std::string id, error;
        if (import_file(file, original_name, sha, true, id, error))
        {
            imported.push_back(id);
            log_info("models: imported " + id + " from a transfer of " + original_name);
        }
        else
        {
            remove_file(file);
            errors.push_back(original_name + ": " + error);
        }
        result["imported"] = imported;
        result["errors"] = errors;
        return result;
    }

}
