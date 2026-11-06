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
#include <set>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winioctl.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <zlib.h>

#include <dlib/data_io/gguf_reader.h>

#include "../common/builtin_models.h"
#include "../common/log.h"
#include "../common/platform.h"
#include "crypto.h"
#include "http_fetch.h"
#include "public_link.h"

namespace albireo
{

    namespace
    {
        /*
            A file written at given offsets by several threads: the partial file of a download,
            created or opened again at its own name, never through a link there, with a single
            name and lying at path reached through no link, so that no write goes elsewhere.
        */
        class random_access_file
        {
        public:
            bool open(const std::string& path, uint64_t size)
            {
#if defined(_WIN32)
                // A reparse point at path is opened as itself, and refused below.
                h_ = CreateFileW(widen(path).c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ, nullptr,
                    OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
                if (h_ == INVALID_HANDLE_VALUE) return false;
                FILE_ATTRIBUTE_TAG_INFO tag = {};
                BY_HANDLE_FILE_INFORMATION info = {};
                if (!GetFileInformationByHandleEx(h_, FileAttributeTagInfo, &tag, sizeof(tag))
                    || (tag.FileAttributes & (FILE_ATTRIBUTE_REPARSE_POINT | FILE_ATTRIBUTE_DIRECTORY)) != 0
                    || !GetFileInformationByHandle(h_, &info) || info.nNumberOfLinks != 1 || !handle_at(h_, path))
                {
                    CloseHandle(h_);
                    h_ = INVALID_HANDLE_VALUE;
                    return false;
                }
                /*
                    A sparse file: the segments written ahead of the others leave ranges not yet
                    written, which the system would otherwise fill with zeros before each write
                    beyond them.  A volume without sparse files refuses it, and the file stays plain.
                */
                DWORD unused = 0;
                DeviceIoControl(h_, FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0, &unused, nullptr);
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
                fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0644);
                if (fd_ < 0) return false;
                struct stat st;
                if (fstat(fd_, &st) != 0 || !S_ISREG(st.st_mode) || st.st_nlink != 1 || !handle_at(fd_, path))
                {
                    ::close(fd_);
                    fd_ = -1;
                    return false;
                }
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

        // A name of a file within a folder: no separator, stream, wildcard or name "." and "..".
        bool plain_name(const std::string& name)
        {
            return !name.empty() && name.size() <= 255 && name != "." && name != ".."
                && name.find_first_of("/\\:*?\"<>|") == std::string::npos;
        }

        // Why a file could not be opened through no link (link_free_file::open()), in words.
        std::string unreadable_reason(const std::string& why)
        {
            if (why == "link") return "reached through a link, which imports never follow";
            if (why == "missing") return "not found";
            return "not a regular file";
        }

        bool ends_with(const std::string& s, const std::string& tail)
        {
            return s.size() >= tail.size() && s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
        }

        std::string lower(std::string s)
        {
            for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            return s;
        }

        // The name of a projector of the models folder: <identifier>.mmproj.gguf.
        const char* const projector_suffix = ".mmproj.gguf";

        // Whether a part of a file name names a quantization: Q4_K_M, IQ4_NL, Q8_0, F16, BF16...
        bool quantization_name(const std::string& s)
        {
            if (s == "f16" || s == "f32" || s == "bf16" || s == "f64") return true;
            const size_t at = s.compare(0, 2, "iq") == 0 ? 2 : s.compare(0, 1, "q") == 0 ? 1 : 0;
            if (at == 0 || at >= s.size() || !std::isdigit(static_cast<unsigned char>(s[at]))) return false;
            for (size_t i = at; i < s.size(); ++i)
                if (!std::isalnum(static_cast<unsigned char>(s[i])) && s[i] != '_') return false;
            return true;
        }

        /*
            The name a projector and its decoder share: the file name in lower case without its
            extension, the prefix or the suffix "mmproj" and the quantization, its letters and
            digits alone.  "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf" and
            "Qwen3VL-2B-Instruct-Q4_K_M.gguf" both give "qwen3vl2binstruct".
        */
        std::string name_stem(const std::string& file)
        {
            std::string s = lower(file_name(file));
            if (ends_with(s, ".gguf")) s.resize(s.size() - 5);
            for (const char* tail : { ".mmproj", "-mmproj", "_mmproj" })
                if (ends_with(s, tail)) s.resize(s.size() - std::strlen(tail));
            if (s.compare(0, 6, "mmproj") == 0 && s.size() > 6 && (s[6] == '-' || s[6] == '_' || s[6] == '.'))
                s = s.substr(7);
            for (size_t i = s.size(); i-- > 0;)
                if ((s[i] == '-' || s[i] == '_' || s[i] == '.') && quantization_name(s.substr(i + 1)))
                {
                    s.resize(i);
                    break;
                }
            std::string out;
            for (char c : s)
                if (std::isalnum(static_cast<unsigned char>(c))) out += c;
            return out;
        }

        // The letters and digits of a name in lower case, which general.name compares by.
        std::string plain_name_of(const std::string& name)
        {
            std::string out;
            for (char c : lower(name))
                if (std::isalnum(static_cast<unsigned char>(c))) out += c;
            return out;
        }

        /*
            How close the names of a projector and of a decoder are: the same general.name in both
            headers, then the same stem of their file names, then the length of the common
            beginning of these stems.
        */
        long name_closeness(const pairing_file& projector, const pairing_file& decoder)
        {
            long score = 0;
            const std::string a = plain_name_of(projector.description["gguf_name"].as_string());
            const std::string b = plain_name_of(decoder.description["gguf_name"].as_string());
            if (!a.empty() && a == b) score += 1 << 20;
            const std::string x = name_stem(projector.name), y = name_stem(decoder.name);
            if (!x.empty() && x == y) return score + (1 << 16);
            size_t common = 0;
            while (common < x.size() && common < y.size() && x[common] == y[common]) ++common;
            return score + static_cast<long>(std::min<size_t>(common, 0xFFFF));
        }

        json text_inputs()
        {
            json inputs = json::array();
            inputs.push_back("text");
            return inputs;
        }

        /*
            The inputs named by list, "text" first, with "image" when image is true and without it
            otherwise; the other inputs (audio) as listed.
        */
        json inputs_with(const json& list, bool image)
        {
            json out = text_inputs();
            if (image) out.push_back("image");
            for (const auto& s : list.elements())
            {
                const std::string v = s.as_string();
                if (v.empty() || v == "text" || v == "image") continue;
                bool seen = false;
                for (const auto& o : out.elements()) seen = seen || o.as_string() == v;
                if (!seen) out.push_back(v);
            }
            return out;
        }

        /*
            The model that comes with Albireo whose file is that of the catalog entry, made from it by
            the build of Albireo under another identifier: its manifest gives the hash of the file it
            was made from (source_sha256).  Null when there is none.
        */
        const installed_model* builtin_carrier(const std::vector<installed_model>& installed, const json& entry)
        {
            const std::string sha = entry["sha256"].as_string();
            if (sha.empty()) return nullptr;
            for (const auto& m : installed)
                if (m.source == "builtin" && m.descriptor["source_sha256"].as_string() == sha) return &m;
            return nullptr;
        }
    }

    bool reads_images(const std::string& arch)
    {
        return arch == "qwen3vl" || arch == "qwen3vlmoe";
    }

    json model_inputs(const installed_model& m)
    {
        return inputs_with(m.descriptor["inputs"], !m.projector.empty());
    }

    json entry_inputs(const json& entry)
    {
        bool image = false;
        for (const auto& s : entry["inputs"].elements()) image = image || s.as_string() == "image";
        return inputs_with(entry["inputs"], image && entry["projector"].is_object());
    }

    std::vector<int> pair_projectors(const std::vector<pairing_file>& decoders,
        const std::vector<pairing_file>& projectors)
    {
        std::vector<int> pair(projectors.size(), -1);
        std::vector<bool> taken(decoders.size(), false);
        // The hashes of the catalog first.
        for (size_t p = 0; p < projectors.size(); ++p)
            for (size_t d = 0; d < decoders.size() && pair[p] < 0; ++d)
                if (!taken[d] && !projectors[p].sha256.empty()
                    && decoders[d].expected_projector == projectors[p].sha256)
                {
                    pair[p] = static_cast<int>(d);
                    taken[d] = true;
                }
        // Then the widths of the headers, and the names among several decoders.
        for (size_t p = 0; p < projectors.size(); ++p)
        {
            if (pair[p] >= 0) continue;
            const long long width = projectors[p].description["projection_width"].as_int(0);
            if (width <= 0) continue;
            long best = -1;
            int chosen = -1;
            bool tie = false;
            for (size_t d = 0; d < decoders.size(); ++d)
            {
                const json& dd = decoders[d].description;
                const bool images = dd["reads_images"].as_bool(reads_images(dd["arch"].as_string()));
                if (taken[d] || !images || dd["projection_width"].as_int(0) != width) continue;
                const long score = name_closeness(projectors[p], decoders[d]);
                if (score > best)
                {
                    best = score;
                    chosen = static_cast<int>(d);
                    tie = false;
                }
                else if (score == best)
                {
                    tie = true;
                }
            }
            if (chosen >= 0 && !tie)
            {
                pair[p] = chosen;
                taken[static_cast<size_t>(chosen)] = true;
            }
        }
        return pair;
    }

    json describe_gguf(const std::string& path)
    {
        json d = json::object();
        try
        {
            dlib::gguf_reader g(path);
            const std::string arch = g.get_str("general.architecture", "");
            d["name"] = g.get_str("general.name", file_name(path));
            d["gguf_name"] = g.get_str("general.name", "");
            d["arch"] = arch;
            // The company that made the model, when its header names it.
            const std::string publisher = !g.get_str("general.organization", "").empty()
                ? g.get_str("general.organization") : g.get_str("general.author", "");
            if (!publisher.empty()) d["publisher"] = publisher;
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
            // The widths of the vision tokens, bounded so that a damaged header gives no width.
            auto width_of = [](int64_t width, int64_t layers) -> long long {
                if (width <= 0 || width > (int64_t(1) << 20) || layers < 0 || layers > 256) return 0;
                return static_cast<long long>(width * (1 + layers));
            };
            if (arch == "clip")
            {
                // A projector: the vision encoder of a model that reads images, and the
                // projection of its output into the embeddings of the decoder.
                d["projector"] = true;
                d["projector_type"] = g.get_str("clip.projector_type", "");
                d["vision"] = g.get_int("clip.has_vision_encoder", 1) != 0;
                int64_t deepstack = 0;
                if (g.has("clip.vision.is_deepstack_layers"))
                    for (int64_t v : g.at("clip.vision.is_deepstack_layers").arr_int) deepstack += v != 0 ? 1 : 0;
                // A LLaVA projector names no projection width, and that of a Pixtral projector is
                // another one: their last layer gives it.
                const std::string type = d["projector_type"].as_string();
                long long width = type == "pixtral" ? 0
                    : width_of(g.get_int("clip.vision.projection_dim", 0), deepstack);
                for (const char* last : { "mm.2.weight", "mm.0.weight" })
                {
                    const dlib::gguf_tensor_info* t = width > 0 ? nullptr : g.find_tensor(last);
                    if (t && t->dims.size() >= 2) width = width_of(static_cast<int64_t>(t->dims[1]), 0);
                }
                d["projection_width"] = width;
                d["image_size"] = static_cast<long long>(g.get_int("clip.vision.image_size", 0));
                return d;
            }
            d["context"] = static_cast<long long>(g.get_int(arch + ".context_length", 0));
            d["experts"] = static_cast<long long>(g.get_int(arch + ".expert_count", 0));
            /*
                A decoder that reads images: Qwen3-VL, a LLaVA decoder, whose vocabulary holds
                <image>, or a Mistral decoder, whose vocabulary holds [IMG] and [IMG_END].
            */
            bool images = reads_images(arch);
            if (!images && g.has("tokenizer.ggml.tokens"))
            {
                const std::vector<std::string>& tokens = g.at("tokenizer.ggml.tokens").arr_str;
                auto holds = [&](const char* t) { return std::find(tokens.begin(), tokens.end(), t) != tokens.end(); };
                images = holds("<image>") || (holds("[IMG]") && holds("[IMG_END]"));
            }
            d["reads_images"] = images;
            if (images)
                d["projection_width"] = width_of(g.get_int(arch + ".embedding_length", 0),
                    g.get_int(arch + ".n_deepstack_layers", 0));
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
            /*
                The capabilities the formats of the engine give a model of this template, read from
                the template alone (chat_format::detect() reads the vocabulary as well):
                    - tools: the formats of Mistral, Llama 3, Granite 3 and Granite 4.2, and the
                      ChatML templates that describe tools; Phi-4-mini, Foundation-Sec and the
                      other formats call none;
                    - reasoning: the ChatML templates that open the reasoning or switch it, Granite
                      4.2, Foundation-Sec-8B-Reasoning and the templates with [THINK];
                    - reasoning always: the thinking variants of Qwen3 and Qwen3-VL, whose template
                      opens the reasoning without a switch, and Foundation-Sec-8B-Reasoning, whose
                      files carry a template without <|end|> that takes the reasoning out of the
                      earlier answers.
            */
            const auto holds = [&](const char* s) { return tmpl.find(s) != std::string::npos; };
            const bool chatml = holds("<|im_start|>");
            const bool granite4 = chatml && holds("<function=") && holds("<think></think>");
            const bool opens = chatml && holds("<|im_start|>assistant\\n<think>\\n'");
            const bool switchable = chatml && holds("enable_thinking");
            const bool foundation_sec = !chatml && holds("<|assistant|>") && holds("'</think>' in content")
                && !holds("<|end|>");
            d["reasoning"] = !embedding && (granite4 || opens || switchable || foundation_sec || holds("[THINK]"));
            d["reasoning_always"] = !embedding && ((opens && !switchable) || foundation_sec);
            d["tools"] = !embedding && (holds("[INST]") || holds("<|start_header_id|>") || holds("<|start_of_role|>")
                || granite4 || (chatml && (holds("<tool_call>") || holds("tools"))));
            // The inputs the engine reads from the model, and the identity a file prepared
            // by the build of Albireo carries (tools/albireo_builtin.cpp).
            json inputs = json::array();
            if (g.has("albireo.inputs"))
                for (const auto& s : g.at("albireo.inputs").arr_str) inputs.push_back(s);
            if (inputs.size() == 0) inputs.push_back("text");
            d["inputs"] = inputs;
            if (g.has("albireo.id"))
            {
                d["albireo_id"] = g.get_str("albireo.id");
                if (g.has("albireo.name")) d["name"] = g.get_str("albireo.name");
                if (g.has("albireo.roles"))
                {
                    json roles = json::array();
                    for (const auto& s : g.at("albireo.roles").arr_str) roles.push_back(s);
                    d["roles"] = roles;
                }
            }
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
            std::string url;          // the official address of the resource, read when the space does not serve it
            std::string link_path;    // the copy of the resource in the publication space (public_link.h)
            uint64_t resource = 0;    // bytes of the resource
            uint64_t from = 0;        // byte range in the resource
            uint64_t length = 0;      // bytes of the resource (compressed size for gzip)
            uint64_t offset = 0;      // where the bytes go in the file
            std::atomic<uint64_t> done{0};   // bytes of the resource received
            bool gzip = false;
            uint64_t output = 0;      // bytes the part produces (gzip: decompressed size)
            size_t part = 0;          // the part of its file
            size_t file = 0;          // 0: the decoder, 1: the projector
        };

        // A file of the model: the decoder, and the projector of a model that reads images.
        struct output_file
        {
            std::string target, partial;
            uint64_t size = 0;
            std::string sha256;
            json parts;               // the "files" of the catalog: url, size, sha256
        };

        std::string id;
        json entry;
        std::vector<output_file> files;
        uint64_t total = 0;
        std::vector<std::unique_ptr<chunk>> chunks;
        std::atomic<uint64_t> received{0};
        std::atomic<bool> cancel{false};
        // Cancelled by the user, set under m_: the progress and the partial file are dropped,
        // where a stop of the service keeps them for the next start.
        std::atomic<bool> abandoned{false};
        std::string state = "queued";
        user_error error;
        double speed = 0;
        int64_t started = 0;
        std::thread worker;
    };

    model_manager::model_manager(store& db, const std::string& models_dir)
        : db_(db), dir_(models_dir)
    {
        /*
            The service writes, renames and deletes the model files with its rights: only
            SYSTEM and the administrators may change the folder (secure_folder()).  A folder
            that cannot be made so leaves the models to the folder of the data.
        */
        std::string error;
        if (!secure_folder(dir_, error))
        {
            const std::string fallback = join_path(data_dir(), "models");
            log_error("models: " + error + "; the models go to " + fallback);
            dir_ = fallback;
            if (!secure_folder(dir_, error))
            {
                log_error("models: " + error);
                make_dirs(dir_);
            }
        }
        // The folder by its own path: a link placed later on its way shows (direct_path()).
        const std::string resolved = final_path(dir_);
        if (!resolved.empty()) dir_ = resolved;
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
            /*
                Written under the lock, so that of two catalogs installed at once, from the
                network and from a medium, the database keeps the one kept in memory, and
                written first: a database that refuses it, on a full disk say, leaves the
                catalog in use as it was.
            */
            db_.set_value("catalog", body);
            db_.set_value("catalog_time", std::to_string(now_unix()));
            catalog_ = c;
        }
        ++generation_;
        result["models"] = static_cast<long long>(c["models"].size());
        log_info("models: catalog of " + std::to_string(c["models"].size()) + " models, published "
            + c["updated"].as_string());
        return true;
    }

    namespace
    {
        /*
            The official address of the catalog: the repository of Albireo, which keeps the signed catalog the
            publisher places in the publication space.  Its signature lies at the same address with ".sig".
        */
        const char* const official_catalog = "https://raw.githubusercontent.com/Cydral/Dlib-Transformer-extensions/"
            "main/dist/albireo/catalog.json";

        // The address official_catalog_for_tests() gives in place of the official one, empty otherwise.
        std::mutex catalog_address_m;
        std::string catalog_address_of_tests;

        // The address the catalog is read from when the publication space does not serve it.
        std::string catalog_address()
        {
            std::lock_guard<std::mutex> lock(catalog_address_m);
            return catalog_address_of_tests.empty() ? std::string(official_catalog) : catalog_address_of_tests;
        }
    }

    void official_catalog_for_tests(const std::string& url)
    {
        std::lock_guard<std::mutex> lock(catalog_address_m);
        catalog_address_of_tests = url;
    }

    bool model_manager::refresh_catalog(std::string& error)
    {
        /*
            The catalog and its signature come from the folder dist of the publication space, and from their
            official address when the space does not serve them: a space that is not set, does not hold them,
            does not answer, or gives a catalog whose signature fails.  The signature checks the catalog of either.
        */
        std::string body, sig, space;
        json result = json::object();
        if (public_link_get("dist/catalog.json", body, space, 16u << 20)
            && public_link_get("dist/catalog.json.sig", sig, space, 4096))
        {
            std::string refused;
            if (install_catalog(body, sig, true, result, refused))
            {
                log_info("models: catalog read from the publication space of Albireo");
                return true;
            }
            space = "the catalog of the publication space of Albireo is refused: " + refused;
        }
        // A service that stops or goes offline asks no other source.
        if (http_cancelled())
        {
            error = space;
            return false;
        }
        const std::string url = catalog_address();
        std::string e;
        body.clear();
        sig.clear();
        if (!http_get(url, body, e, 16u << 20))
        {
            error = space + "; the catalog cannot be downloaded from its official address " + url + " either: " + e;
            return false;
        }
        if (!http_get(url + ".sig", sig, e, 4096))
        {
            error = space + "; the signature of the catalog cannot be downloaded from its official address " + url
                + ".sig either: " + e;
            return false;
        }
        if (!install_catalog(body, sig, true, result, e))
        {
            error = space + "; the catalog of its official address " + url + " is refused: " + e;
            return false;
        }
        log_warning("models: catalog read from its official address " + url + ", the publication space of Albireo "
            "did not serve it (" + space + ")");
        return true;
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
            catch (...)
            {
                error = "the catalog cannot be refreshed";
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

    json model_manager::import_catalog(const std::string& folder, const std::string& base)
    {
        json result = json::object();
        std::string body, sig, error;
        if (!read_file_link_free(join_path(folder, "catalog.json"), base, body, 16u << 20)
            || !read_file_link_free(join_path(folder, "catalog.json.sig"), base, sig, 4096))
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
        auto embedding = [](const json& d) {
            if (d["embedding"].as_bool(false)) return true;
            for (const auto& r : d["roles"].elements())
                if (r.as_string() == "embedding") return true;
            return false;
        };
        /*
            What the installation of a model adds to its entry: its file, its projector, its
            hashes, the inputs it reads (text, and image with its projector) and the bytes of
            its files.
        */
        auto installed_fields = [](json& m, const installed_model& im) {
            const json& d = im.descriptor;
            m["installed"] = true;
            m["file"] = im.file;
            m["local_sha256"] = im.sha256;
            m["verified_at"] = static_cast<long long>(im.verified_at);
            m["inputs"] = model_inputs(im);
            if (im.projector.empty())
            {
                m.erase("projector");
            }
            else
            {
                json p = d["projector"].is_object() ? d["projector"] : json::object();
                p.erase("files");
                p["file"] = im.projector;
                p["size"] = static_cast<unsigned long long>(im.projector_size);
                p["sha256"] = im.projector_sha256;
                m["projector"] = p;
            }
            m["total_size"] = static_cast<unsigned long long>(im.size + im.projector_size);
        };
        // The models that come with Albireo first, in the order of their manifest.
        for (const auto& im : installed)
        {
            if (im.source != "builtin" || embedding(im.descriptor)) continue;
            json m = im.descriptor;
            m["id"] = im.id;
            m["name"] = im.name;
            m["description"] = localized(m["description"], language);
            m["size"] = static_cast<unsigned long long>(im.size);
            installed_fields(m, im);
            seen[im.id] = true;
            out.push_back(m);
        }
        for (const auto& e : cat["models"].elements())
        {
            if (embedding(e)) continue;
            // The entry a model that comes with Albireo stands for, unless installed under its own identifier.
            const std::string eid = e["id"].as_string();
            bool own = false;
            for (const auto& im : installed) own = own || im.id == eid;
            if (!own && builtin_carrier(installed, e)) continue;
            json m = e;
            m["description"] = localized(e["description"], language);
            m["installed"] = false;
            m["inputs"] = entry_inputs(e);
            m["total_size"] = static_cast<unsigned long long>(e["size"].as_int(0) + e["projector"]["size"].as_int(0));
            const std::string id = e["id"].as_string();
            for (const auto& im : installed)
                if (im.id == id)
                {
                    installed_fields(m, im);
                    // Up to date: the files of the entry, the projector it names included.
                    const std::string wanted = e["projector"]["sha256"].as_string();
                    m["up_to_date"] = im.sha256 == e["sha256"].as_string()
                        && (wanted.empty() || im.projector_sha256 == wanted);
                }
            seen[id] = true;
            out.push_back(m);
        }
        for (const auto& im : installed)
        {
            if (seen.count(im.id) || embedding(im.descriptor)) continue;
            json m = im.descriptor.is_object() ? im.descriptor : json::object();
            m["id"] = im.id;
            if (!m.contains("name")) m["name"] = im.name;
            m["description"] = localized(m["description"], language);
            m["size"] = static_cast<unsigned long long>(im.size);
            m["source"] = im.source;
            installed_fields(m, im);
            out.push_back(m);
        }
        // The company that made each model, empty when neither its entry nor its header names it.
        for (auto& m : out.elements())
            if (!m["publisher"].is_string()) m["publisher"] = "";
        json embedders = json::array();
        for (const auto& im : installed)
        {
            if (!embedding(im.descriptor)) continue;
            json e = json::object();
            e["id"] = im.id;
            e["name"] = im.name;
            e["source"] = im.source;
            e["size"] = static_cast<unsigned long long>(im.size);
            e["inputs"] = text_inputs();
            // The license and the publisher of its entry in the catalog or in the manifest, as for
            // the language models.
            e["publisher"] = "";
            for (const char* k : {"license", "publisher"})
                if (im.descriptor.contains(k)) e[k] = im.descriptor[k];
            embedders.push_back(e);
        }
        // The projectors waiting for their decoders.
        json pending = json::array();
        for (const auto& p : db_.projectors())
        {
            json j = json::object();
            j["name"] = p.name;
            j["size"] = static_cast<unsigned long long>(p.size);
            j["sha256"] = p.sha256;
            j["projector_type"] = p.descriptor["projector_type"];
            j["projection_width"] = p.descriptor["projection_width"];
            pending.push_back(j);
        }
        json result = json::object();
        result["models"] = out;
        result["embedding_models"] = embedders;
        result["pending_projectors"] = pending;
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
            if (!j.error.text.empty()) j.error.add_to(d);
            out.push_back(d);
        }
        return out;
    }

    bool model_manager::start_download(const std::string& id, user_error& error)
    {
        if (stopped_)
        {
            error.set("the service stops", "stopping");
            return false;
        }
        json entry;
        if (!catalog_entry(id, entry))
        {
            error.set("the model '" + id + "' is not in the catalog", "not_in_catalog").arg("m", id);
            return false;
        }
        const std::string name = entry["name"].str(id);
        if (!safe_id(id))
        {
            error.set("invalid model identifier", "catalog_entry").arg("m", name);
            return false;
        }
        // A model that comes with Albireo keeps its identifier: a download of it would replace its entry.
        installed_model holder;
        if (db_.find_model(id, holder) && holder.source == "builtin")
        {
            db_.remove_download_state(id);
            error.set("the model '" + id + "' comes with Albireo and is installed already", "builtin_installed")
                .arg("m", name);
            return false;
        }
        // The file of the entry is the one a model that comes with Albireo carries under another name.
        const std::vector<installed_model> installed = db_.models();
        const installed_model* carrier = builtin_carrier(installed, entry);
        if (carrier)
        {
            db_.remove_download_state(id);
            error.set("the model '" + id + "' comes with Albireo as " + carrier->name, "builtin_installed")
                .arg("m", carrier->name);
            return false;
        }
        std::shared_ptr<job> j(new job());
        j->id = id;
        j->entry = entry;
        // The decoder, and the projector of a model that reads images, each in a partial file of its own.
        const json& e = j->entry;
        job::output_file decoder;
        decoder.target = join_path(dir_, id + ".gguf");
        decoder.partial = decoder.target + ".partial";
        decoder.sha256 = e["sha256"].as_string();
        decoder.parts = e["files"];
        j->files.push_back(decoder);
        if (e["projector"].is_object())
        {
            job::output_file projector;
            projector.target = join_path(dir_, id + projector_suffix);
            projector.partial = projector.target + ".partial";
            projector.sha256 = e["projector"]["sha256"].as_string();
            projector.parts = e["projector"]["files"];
            if (projector.sha256.size() != 64 || projector.parts.size() == 0)
            {
                error.set("the catalog entry of '" + id + "' lists an invalid projector", "catalog_entry")
                    .arg("m", name);
                return false;
            }
            j->files.push_back(projector);
        }

        // Parts, then segments of the uncompressed parts.
        const uint64_t segment = 64ull << 20;
        for (size_t fi = 0; fi < j->files.size(); ++fi)
        {
            job::output_file& out = j->files[fi];
            uint64_t offset = 0;
            size_t part = 0;
            for (const auto& f : static_cast<const json&>(out.parts).elements())
            {
                const std::string url = f["url"].as_string();
                // A size of a part above 1 TiB, or negative, would make billions of segments: the entry is invalid.
                const long long given = f["size"].as_int(0);
                const uint64_t size = given > 0 && given <= (1ll << 40) ? static_cast<uint64_t>(given) : 0;
                const bool gz = f["compression"].as_string() == "gzip";
                // The copy of the part in the publication space, named after its official address, serves it first.
                const std::string link_path = public_link_model_path(id, url);
                if (url.empty() || size == 0 || link_path.empty())
                {
                    error.set("the catalog entry of '" + id + "' lists an invalid file", "catalog_entry")
                        .arg("m", name);
                    return false;
                }
                if (gz)
                {
                    std::unique_ptr<job::chunk> c(new job::chunk());
                    c->url = url;
                    c->link_path = link_path;
                    c->from = 0;
                    c->length = static_cast<uint64_t>(std::max<long long>(0, f["compressed_size"].as_int(0)));
                    c->resource = c->length;
                    c->offset = offset;
                    c->gzip = true;
                    c->output = size;
                    c->part = part;
                    c->file = fi;
                    if (c->length == 0)
                    {
                        error.set("the catalog entry of '" + id + "' gives no compressed size", "catalog_entry")
                            .arg("m", name);
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
                        c->link_path = link_path;
                        c->resource = size;
                        c->from = s;
                        c->length = std::min(segment, size - s);
                        c->offset = offset + s;
                        c->output = c->length;
                        c->part = part;
                        c->file = fi;
                        j->chunks.push_back(std::move(c));
                    }
                }
                offset += size;
                ++part;
            }
            // The size of the file, when the entry gives it, is that of its parts.
            const long long declared = fi == 0 ? e["size"].as_int(-1) : e["projector"]["size"].as_int(-1);
            if ((declared >= 0 && static_cast<uint64_t>(declared) != offset) || offset == 0)
            {
                error.set("the sizes of the parts of '" + id + "' do not add up to the size of the model",
                    "catalog_entry").arg("m", name);
                return false;
            }
            out.size = offset;
            j->total += offset;
        }
        // The files of the model, and room kept free for the system and the other programs.
        const uint64_t total = j->total;
        const uint64_t room = free_disk_space(dir_);
        const uint64_t reserve = total / 20 + (256ull << 20);
        if (room < total + reserve && !file_exists(j->files[0].partial))
        {
            error.set("not enough disk space for " + id + ": " + std::to_string((total + reserve) >> 20)
                + " MB needed (" + std::to_string(total >> 20) + " MB for the model, " + std::to_string(reserve >> 20)
                + " MB kept free), " + std::to_string(room >> 20) + " MB available", "disk_space")
                .arg("m", name).size("needed", total + reserve).size("model", total).size("reserve", reserve)
                .size("available", room);
            return false;
        }

        // Resume what a previous run received, when the files of the catalog are the same.
        const json saved = db_.download_state(id);
        bool partials = true;
        for (const auto& f : j->files) partials = partials && file_exists(f.partial);
        const std::string saved_projector = j->files.size() > 1 ? j->files[1].sha256 : std::string();
        if (saved.is_object() && saved["sha256"].as_string() == e["sha256"].as_string()
            && saved["projector_sha256"].as_string() == saved_projector && partials
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
        // The worker of an earlier download of the model, which may still take m_ in its last
        // steps, is joined by the new worker before it starts.
        std::thread previous;
        auto it = jobs_.find(id);
        if (it != jobs_.end())
        {
            if (it->second->state == "downloading" || it->second->state == "verifying" || it->second->state == "queued")
                return true;
            previous = std::move(it->second->worker);
        }
        jobs_[id] = j;
        j->started = now_unix();
        auto failed = [this, j](const std::string& why) {
            {
                std::lock_guard<std::mutex> lock(m_);
                j->state = "error";
                j->error.set(why, "download_failed").arg("e", why);
            }
            log_error("models: download of " + j->id + ": " + why);
        };
        // The earlier worker, kept by the job when no thread can take it, so that it is joined later all the same.
        const auto before = std::make_shared<std::thread>(std::move(previous));
        try
        {
            j->worker = std::thread([this, j, failed, before]() {
                if (before->joinable()) before->join();
                // No exception leaves the thread: a database or a disk that fails under the download ends it in error.
                try
                {
                    run_job(j);
                }
                catch (const std::exception& e)
                {
                    failed(e.what());
                }
                catch (...)
                {
                    failed("the download fails");
                }
            });
        }
        catch (const std::exception& e)
        {
            // A thread the system refuses leaves the download in error, never queued.
            j->worker = std::move(*before);
            const std::string why = std::string("the download cannot start: ") + e.what();
            j->state = "error";
            j->error.set(why, "download_failed").arg("e", why);
        }
        return true;
    }

    void model_manager::note_failure(const std::string& id, const user_error& error)
    {
        std::shared_ptr<job> j(new job());
        j->id = id;
        j->state = "error";
        j->error = error;
        std::thread previous;
        {
            std::lock_guard<std::mutex> lock(m_);
            auto it = jobs_.find(id);
            if (it != jobs_.end())
            {
                const std::string& s = it->second->state;
                if (s == "downloading" || s == "verifying" || s == "queued") return;
                previous = std::move(it->second->worker);
            }
            jobs_[id] = j;
        }
        // The worker of the download that ended, joined outside m_, which its last steps may take.
        if (previous.joinable()) previous.join();
    }

    void model_manager::cancel_download(const std::string& id)
    {
        std::lock_guard<std::mutex> lock(m_);
        auto it = jobs_.find(id);
        const std::string state = it == jobs_.end() ? std::string() : it->second->state;
        if (it != jobs_.end())
        {
            it->second->abandoned = true;
            it->second->cancel = true;
        }
        db_.remove_download_state(id);
        if (state == "queued" || state == "downloading" || state == "verifying") return;
        /*
            A download that ended with an error, or that a restart left, has no worker to drop
            its partial file: the file goes here, and the entry of the download then tells the
            cancellation.
        */
        if (state == "error")
        {
            it->second->state = "cancelled";
            it->second->error = user_error();
        }
        // The partial files of the decoder and of the projector.
        if (safe_id(id) && direct_path(dir_))
        {
            remove_file(join_path(dir_, id + ".gguf.partial"));
            remove_file(join_path(dir_, id + projector_suffix + std::string(".partial")));
        }
    }

    void model_manager::resume_pending()
    {
        for (const auto& id : db_.pending_downloads())
        {
            user_error error;
            if (!start_download(id, error)) log_warning("models: cannot resume " + id + ": " + error.text);
        }
    }

    namespace
    {
        // SHA-256 of the bytes [offset, offset + length) of a file.
        std::string sha256_range(link_free_file& in, uint64_t offset, uint64_t length)
        {
            if (offset + length > in.size() || !in.seek(offset)) return std::string();
            sha256_stream s;
            std::vector<char> buf(1 << 20);
            while (length > 0)
            {
                const size_t n = static_cast<size_t>(std::min<uint64_t>(length, buf.size()));
                if (in.read(buf.data(), n) != static_cast<long long>(n)) return std::string();
                s.update(buf.data(), n);
                length -= n;
            }
            return s.hex();
        }

        // How a transfer of a segment of a download ends (run_job()).
        enum class transfer_end
        {
            complete,
            failed,       // the request fails
            corrupt,      // a compressed part that does not inflate to its size
            other_size,   // the server announces the resource at another size than the catalog
            local         // the partial file refuses the bytes, or the inflation cannot start
        };
    }

    bool model_manager::place(const std::string& from, const std::string& target)
    {
        const std::string before = file_identity(from);
        if (before.empty() || parent_path(target) != dir_ || !direct_path(dir_) || !rename_file(from, target))
            return false;
        // The folder replaced by a link between the check and the move shows here: target names another file.
        return file_identity(target) == before;
    }

    void model_manager::run_job(std::shared_ptr<job> j)
    {
        auto set_state = [&](const std::string& s, const user_error& e = user_error()) {
            std::lock_guard<std::mutex> lock(m_);
            j->state = s;
            j->error = e;
        };
        // The partial files go only from the models folder reached through no link.
        auto drop_partials = [&]() {
            if (!direct_path(dir_)) return;
            for (const auto& f : j->files) remove_file(f.partial);
        };
        // The progress is written under m_, where cancel_download() removes it, so that a
        // download cancelled by the user never comes back at the next start.
        auto save = [&]() {
            const json& e = j->entry;
            json st = json::object();
            st["sha256"] = e["sha256"];
            st["projector_sha256"] = j->files.size() > 1 ? j->files[1].sha256 : std::string();
            json done = json::array();
            for (const auto& c : j->chunks) done.push_back(static_cast<unsigned long long>(c->done.load()));
            st["done"] = done;
            std::lock_guard<std::mutex> lock(m_);
            if (!j->abandoned) db_.set_download_state(j->id, st);
        };
        const bool with_projector = j->files.size() > 1;
        /*
            The sources of the parts, which the transfers share under source_m.  The publication space serves
            first.  space_down tells why it failed for the download, with the file of the space it names
            (down_path), once it is not set, does not answer or refuses the requests for a segment: the next
            segments go to their official addresses at once.  lacking tells why it does not serve a part, a part
            it does not hold, holds at another size or whose bytes from it fail their hash, which goes to its
            official address with all its segments.  The log tells each of these changes once.
        */
        typedef std::pair<size_t, size_t> part_key;   // the file of the model, and the part in it
        const int by_space = 1, by_official = 2;
        std::mutex source_m;
        std::string space_down, down_path;
        std::map<part_key, std::string> lacking;
        std::map<part_key, int> sources;              // the sources that gave bytes of each part in this run
        // Records that the source gave bytes of the part.
        auto gave = [&](const part_key& part, int source) {
            std::lock_guard<std::mutex> lock(source_m);
            sources[part] |= source;
        };
        // Tells that the download continues from the official address url, the space not serving path.
        auto tell = [&](const std::string& url, const std::string& path, const std::string& why) {
            log_warning("models: download of " + j->id + " continues from its official address " + without_queries(url)
                + ", the publication space of Albireo did not serve " + path + " (" + why + ")");
        };

        log_info("models: downloading " + j->id + " (" + std::to_string(j->total >> 20) + " MB, "
            + std::to_string(j->chunks.size()) + " segments" + (with_projector ? ", with its projector)" : ")"));
        for (int round = 0; round < 2; ++round)
        {
            set_state("downloading");
            // A partial file per file of the model, the models folder checked again right before each
            // opens, which then checks its own path.
            std::vector<std::unique_ptr<random_access_file>> outputs;
            for (const auto& f : j->files)
            {
                outputs.emplace_back(new random_access_file());
                if (!direct_path(dir_) || !outputs.back()->open(f.partial, f.size))
                {
                    set_state("error", user_error().set("cannot create " + f.partial, "download_file")
                        .arg("file", f.partial));
                    log_error("models: download of " + j->id + ": " + f.partial + " cannot be created in the models "
                        "folder reached through no link");
                    return;
                }
            }
            save();

            std::mutex chunk_m;
            size_t next = 0;
            std::atomic<bool> failed{false};
            std::string failure;
            // The first failure of the download; chunk_m is held.
            auto fail_with = [&](const std::string& why) {
                if (!failed) failure = why;
                failed = true;
            };
            // The transfers end: the download is cancelled, or a segment failed it.
            auto ended = [&]() { return j->cancel || failed; };
            // The pause before the attempt number attempt, 2 s for each attempt made, which an end of the
            // transfers or of the downloads of the service ends.
            auto pause = [&](int attempt) {
                for (int t = 0; t < 20 * attempt && !ended() && !http_cancelled(); ++t)
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
            };

            /*
                One transfer of the segment c from address into its partial file: its range from where it
                stopped, or the whole resource of a compressed part, inflated as it arrives.  With sized, a
                resource the server announces at another size than the catalog stops before any of its bytes is
                written.  wrote receives the bytes written; a compressed part that fails takes them back from the
                progress, and restarts from its beginning.  reason tells why a transfer fails, or the local
                failure (a partial file that refuses the bytes).
            */
            auto fetch = [&](job::chunk& c, const std::string& address, bool sized, std::string& reason,
                uint64_t& wrote) {
                random_access_file& file = *outputs[c.file];
                reason.clear();
                wrote = 0;
                uint64_t total = 0;
                bool ok = false, unwritten = false, corrupt = false, other_size = false;
                // The size of the resource, which the server announces before its first bytes.
                auto resized = [&]() {
                    other_size = other_size || (sized && total != 0 && total != c.resource);
                    return other_size;
                };
                if (c.gzip)
                {
                    // Inflated as it arrives, written at the offsets of the part.
                    z_stream zs;
                    std::memset(&zs, 0, sizeof(zs));
                    if (inflateInit2(&zs, 15 + 32) != Z_OK)
                    {
                        reason = "the compressed part " + std::to_string(c.part) + " cannot be inflated";
                        return transfer_end::local;
                    }
                    uint64_t produced = 0;
                    std::vector<char> out(1 << 20);
                    ok = http_get_range(address, 0, -1, [&](const char* data, size_t n) {
                        if (j->cancel || resized()) return false;
                        zs.next_in = reinterpret_cast<Bytef*>(const_cast<char*>(data));
                        zs.avail_in = static_cast<uInt>(n);
                        while (zs.avail_in > 0)
                        {
                            zs.next_out = reinterpret_cast<Bytef*>(out.data());
                            zs.avail_out = static_cast<uInt>(out.size());
                            const int r = inflate(&zs, Z_NO_FLUSH);
                            const size_t got = out.size() - zs.avail_out;
                            if ((r != Z_OK && r != Z_STREAM_END) || produced + got > c.output)
                            {
                                corrupt = true;
                                return false;
                            }
                            if (!file.write_at(c.offset + produced, out.data(), got))
                            {
                                unwritten = true;
                                return false;
                            }
                            produced += got;
                            j->received += got;
                            if (r == Z_STREAM_END) break;
                        }
                        return true;
                    }, reason, &total);
                    inflateEnd(&zs);
                    corrupt = corrupt || (ok && produced != c.output);
                    ok = ok && !corrupt;
                    if (ok) c.done = c.length;
                    else j->received -= produced;
                    wrote = produced;
                }
                else
                {
                    const uint64_t before = c.done;
                    ok = http_get_range(address, c.from + c.done, static_cast<int64_t>(c.from + c.length - 1),
                        [&](const char* data, size_t n) {
                            if (j->cancel || resized()) return false;
                            const uint64_t done = c.done;
                            const size_t take = static_cast<size_t>(std::min<uint64_t>(n, c.length - done));
                            if (!file.write_at(c.offset + done, data, take))
                            {
                                unwritten = true;
                                return false;
                            }
                            c.done = done + take;
                            j->received += take;
                            return true;
                        }, reason, &total);
                    ok = ok && c.done == c.length;
                    wrote = c.done - before;
                }
                if (ok) return transfer_end::complete;
                if (unwritten)
                {
                    reason = "cannot write " + j->files[c.file].partial;
                    return transfer_end::local;
                }
                if (corrupt) return transfer_end::corrupt;
                if (other_size) return transfer_end::other_size;
                if (reason.empty()) reason = "the answer ends too soon";
                return transfer_end::failed;
            };

            auto transfer = [&]() {
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
                    const part_key part(c.file, c.part);
                    std::string space_error, reason, error;
                    uint64_t wrote = 0;
                    transfer_end end = transfer_end::failed;
                    // Whether space_error names the file of this segment, which a failure of another part may not.
                    bool here = true;
                    // Whether the space failed for the download or does not serve the part; space_error tells why.
                    auto refused = [&]() {
                        std::lock_guard<std::mutex> lock(source_m);
                        const auto it = lacking.find(part);
                        if (it != lacking.end())
                        {
                            space_error = it->second;
                            here = true;
                            return true;
                        }
                        if (space_down.empty()) return false;
                        space_error = space_down;
                        here = down_path == c.link_path;
                        return true;
                    };
                    bool skipped = refused();
                    // Whether the failure of the space concerns the whole download, or this part alone.
                    bool whole = true;
                    /*
                        The copy of the resource in the publication space, its address asked again after a failure.
                        A space that does not hold the resource at its size ends the attempts, and so does another
                        segment that finds the space failing.  The messages name the file in the space, never its
                        address.
                    */
                    for (int attempt = 0; attempt < 4 && !skipped && !ended(); ++attempt)
                    {
                        // A pause before each new attempt, which a cancellation ends.
                        pause(attempt);
                        if (ended() || (attempt > 0 && (skipped = refused()))) break;
                        std::string address;
                        uint64_t listed = 0;
                        link_status status = link_status::ok;
                        if (!public_link_file(c.link_path, address, listed, space_error, attempt > 0, &status))
                        {
                            whole = status != link_status::missing;
                            // A file the space lacks is looked for once more in a new listing.
                            if (status == link_status::unset || (!whole && attempt > 0) || http_cancelled()) break;
                            continue;
                        }
                        if (listed != c.resource)
                        {
                            space_error = "the copy of " + c.link_path + " in the publication space of Albireo does "
                                "not have the size the catalog gives";
                            whole = false;
                            if (attempt > 0) break;
                            continue;
                        }
                        end = fetch(c, address, false, reason, wrote);
                        if (wrote > 0) gave(part, by_space);
                        if (end == transfer_end::complete || end == transfer_end::local) break;
                        whole = end != transfer_end::corrupt;
                        space_error = end == transfer_end::corrupt ? "the compressed copy of " + c.link_path
                            + " in the publication space of Albireo is corrupt" : "the copy of " + c.link_path
                            + " in the publication space of Albireo cannot be downloaded: " + reason;
                        // The downloads ended for the whole service (it stops, or goes offline) are not
                        // attempted again; they resume later.
                        if (http_cancelled()) break;
                    }
                    bool ok = end == transfer_end::complete;
                    error = end == transfer_end::local ? reason : space_error;
                    /*
                        The official address of the part, the "url" of the catalog, when the space does not serve
                        the segment: the same ranges, in three attempts.  The failure of the space is kept for the
                        next segments, of the download or of the part, which go there at once; the log tells it
                        when it changes the source of the download or of the part.
                    */
                    if (!ok && end != transfer_end::local && !ended() && !http_cancelled())
                    {
                        if (space_error.empty()) space_error = "the publication space of Albireo does not serve "
                            + c.link_path;
                        const std::string official = without_queries(c.url);
                        bool changed = false;
                        {
                            std::lock_guard<std::mutex> lock(source_m);
                            if (!skipped && whole && space_down.empty())
                            {
                                space_down = space_error;
                                down_path = c.link_path;
                                changed = true;
                            }
                            if (!skipped && !whole) changed = lacking.emplace(part, space_error).second;
                        }
                        if (changed) tell(c.url, c.link_path, space_error);
                        bool tried = false;
                        for (int attempt = 0; attempt < 3 && !ok && !ended(); ++attempt)
                        {
                            pause(attempt);
                            if (ended()) break;
                            tried = true;
                            end = fetch(c, c.url, true, reason, wrote);
                            if (wrote > 0) gave(part, by_official);
                            ok = end == transfer_end::complete;
                            if (end == transfer_end::local || (!ok && http_cancelled())) break;
                        }
                        if (end == transfer_end::local)
                        {
                            error = reason;
                        }
                        else if (tried && !ok)
                        {
                            // Both sources failed: the file in the space, its official address, the failure of each.
                            const std::string why = end == transfer_end::corrupt
                                ? "the compressed file it gives is corrupt" : end == transfer_end::other_size
                                ? "the file it gives does not have the size the catalog gives" : reason;
                            error = space_error + (here ? "; its official address " + official
                                + " cannot be downloaded either: " : "; " + c.link_path
                                + " cannot be downloaded from its official address " + official + " either: ") + why;
                        }
                    }
                    if (!ok && !j->cancel)
                    {
                        std::lock_guard<std::mutex> lock(chunk_m);
                        fail_with(error.empty() ? "download failed" : error);
                        return;
                    }
                }
            };
            // An exception of a transfer, memory exhausted say, fails the download and never leaves its thread.
            auto worker = [&]() {
                try
                {
                    transfer();
                }
                catch (const std::exception& e)
                {
                    std::lock_guard<std::mutex> lock(chunk_m);
                    fail_with(std::string("the download fails: ") + e.what());
                }
                catch (...)
                {
                    std::lock_guard<std::mutex> lock(chunk_m);
                    fail_with("the download fails");
                }
            };

            std::vector<std::thread> threads;
            try
            {
                for (int t = 0; t < 4; ++t) threads.emplace_back(worker);
            }
            catch (const std::exception& e)
            {
                // A thread the system refuses: those started end with their chunk, joined below.
                std::lock_guard<std::mutex> lock(chunk_m);
                fail_with(std::string("the download cannot start its transfers: ") + e.what());
            }
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
            for (auto& o : outputs) o->close();

            if (j->cancel)
            {
                set_state("cancelled");
                if (j->abandoned) drop_partials();
                log_info("models: download of " + j->id + " cancelled");
                return;
            }
            save();
            if (failed)
            {
                set_state("error", user_error().set(failure, "download_failed").arg("e", failure));
                log_error("models: download of " + j->id + " failed: " + failure);
                // A cancellation that came while the download failed drops its partial files all the same.
                if (j->abandoned) drop_partials();
                return;
            }

            set_state("verifying");
            /*
                Each file written, read through no link from the models folder, is checked against
                its own hash; it is closed before it is renamed or deleted, which its handle would
                refuse on Windows.  The parts whose own hash fails are fetched again, once, from
                their official addresses when the publication space gave bytes of them.
            */
            std::vector<std::string> shas(j->files.size());
            bool all_match = true;
            bool any = false;
            for (size_t fi = 0; fi < j->files.size(); ++fi)
            {
                const job::output_file& f = j->files[fi];
                link_free_file written;
                std::string why;
                const bool readable = direct_path(dir_) && written.open(f.partial, dir_, why) && written.links() == 1;
                shas[fi] = readable ? sha256_open_file(written, &j->cancel) : std::string();
                if (j->cancel)
                {
                    // Stopped during the check: the files are kept and checked again at the
                    // next start, unless the download itself was cancelled.
                    written.close();
                    set_state("cancelled");
                    if (j->abandoned) drop_partials();
                    log_info("models: download of " + j->id + " cancelled during its check");
                    return;
                }
                if (!readable)
                {
                    set_state("error", user_error().set("cannot read " + f.partial, "download_file")
                        .arg("file", f.partial));
                    log_error("models: download of " + j->id + ": " + f.partial + " is no longer the file written");
                    return;
                }
                if (shas[fi] == f.sha256) continue;
                all_match = false;
                uint64_t offset = 0;
                size_t part = 0;
                for (const auto& p : static_cast<const json&>(f.parts).elements())
                {
                    const uint64_t size = static_cast<uint64_t>(p["size"].as_int(0));
                    const std::string expected = p["sha256"].as_string();
                    if (round == 0 && !expected.empty() && sha256_range(written, offset, size) != expected)
                    {
                        std::string path, url;
                        for (auto& c : j->chunks)
                            if (c->file == fi && c->part == part)
                            {
                                j->received -= c->gzip ? c->output : c->done.load();
                                c->done = 0;
                                path = c->link_path;
                                url = c->url;
                            }
                        any = true;
                        log_warning("models: part " + std::to_string(part) + " of " + (fi == 0 ? std::string()
                            : std::string("the projector of ")) + j->id + " is corrupt, fetched again");
                        // Bytes the space gave fail their hash: the part comes from its official address.
                        const std::string why = "the copy of " + path + " in the publication space of Albireo does "
                            "not match the hash of the catalog";
                        bool changed = false;
                        {
                            std::lock_guard<std::mutex> lock(source_m);
                            const part_key key(fi, part);
                            if ((sources[key] & by_space) && lacking.emplace(key, why).second)
                                changed = space_down.empty();
                            sources[key] = 0;
                        }
                        if (changed) tell(url, path, why);
                    }
                    offset += size;
                    ++part;
                }
            }
            if (all_match)
            {
                /*
                    The files take their names together, the decoder first.  When one cannot, or
                    when the database cannot record the model, those placed take their partial
                    names again: the next start finds the progress complete, checks the files and
                    installs them.
                */
                std::string error;
                size_t placed = 0;
                // A newer version of an installed model takes its names while the engine holds no file of it.
                replacing(j->id, [&]() {
                    placed = 0;
                    while (placed < j->files.size() && place(j->files[placed].partial, j->files[placed].target))
                        ++placed;
                    return placed == j->files.size();
                });
                if (placed < j->files.size() || !install_file(j->id, j->files[0].target, j->entry, shas[0], "catalog",
                        with_projector ? j->files[1].target : std::string(), with_projector ? shas[1] : std::string(),
                        error))
                {
                    for (size_t i = 0; i < placed; ++i) place(j->files[i].target, j->files[i].partial);
                    if (error.empty())
                        error = "cannot install " + j->files[std::min(placed, j->files.size() - 1)].target;
                    set_state("error", user_error().set(error, "download_install").arg("e", error));
                    log_error("models: download of " + j->id + ": " + error);
                    return;
                }
                db_.remove_download_state(j->id);
                if (with_projector) drop_pending_copy(shas[1]);
                set_state("done");
                // The sources of the bytes of this run; a download resumed complete has none.
                int all = 0;
                for (const auto& s : sources) all |= s.second;
                const char* const from = all == (by_space | by_official) ? ", from the publication space of Albireo "
                    "and its official addresses" : all == by_official ? ", from its official addresses"
                    : all == by_space ? ", from the publication space of Albireo" : "";
                log_info("models: " + j->id + " installed" + (with_projector ? ", with its projector" : "") + from);
                return;
            }
            if (!any) break;
        }
        set_state("error", user_error().set("the downloaded file does not match the hash of the catalog",
            "download_hash"));
        log_error("models: " + j->id + " does not match its hash");
        db_.remove_download_state(j->id);
        drop_partials();
    }

    bool model_manager::install_file(const std::string& id, const std::string& file, const json& descriptor,
        const std::string& sha, const std::string& source, const std::string& projector,
        const std::string& projector_sha, std::string& error)
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
        m.projector = projector;
        m.projector_size = projector.empty() ? 0 : file_size(projector);
        m.projector_sha256 = projector.empty() ? std::string() : projector_sha;
        if (m.size == 0)
        {
            error = file + " is empty";
            return false;
        }
        if (!projector.empty() && m.projector_size == 0)
        {
            error = projector + " is empty";
            return false;
        }
        try
        {
            // The entry of a model that comes with Albireo is never replaced by another file.
            installed_model existing;
            if (source != "builtin" && db_.find_model(id, existing) && existing.source == "builtin")
            {
                error = "the model " + id + " comes with Albireo";
                return false;
            }
            db_.put_model(m);
        }
        catch (const std::exception& e)
        {
            // A database that cannot be written, on a full disk for instance.
            error = "the model cannot be recorded: " + std::string(e.what());
            return false;
        }
        // The model is installed whatever the functions it receives.
        try
        {
            if (installed_cb_) installed_cb_(m);
        }
        catch (const std::exception& e)
        {
            log_error("models: " + id + " is installed, its functions are not given: " + e.what());
        }
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
        // A file of the models folder is deleted only through the folder reached without a link.
        for (const std::string& f : { m.file, m.projector })
            if (!f.empty() && parent_path(f) == dir_ && !direct_path(dir_))
            {
                error = "the models folder " + dir_ + " now leads elsewhere through a link";
                return false;
            }
        if (file_exists(m.file) && !remove_file(m.file))
        {
            error = "cannot delete " + m.file + " (in use?)";
            return false;
        }
        // The projector goes with its decoder.
        if (!m.projector.empty() && file_exists(m.projector) && !remove_file(m.projector))
            log_warning("models: the projector " + m.projector + " of " + id + " cannot be deleted");
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
        // The check ends at once when the service stops.
        const std::string sha = sha256_file(m.file, nullptr, &stopped_);
        const std::string projector_sha = m.projector.empty() ? std::string()
            : sha256_file(m.projector, nullptr, &stopped_);
        if (stopped_)
        {
            r["error"] = "the service stops";
            return r;
        }
        const bool decoder_match = !sha.empty() && sha == m.sha256;
        const bool projector_match = m.projector.empty()
            || (!projector_sha.empty() && projector_sha == m.projector_sha256);
        r["sha256"] = sha;
        if (!m.projector.empty())
        {
            r["projector_sha256"] = projector_sha;
            r["projector_match"] = projector_match;
        }
        r["local_match"] = decoder_match && projector_match;
        json found;
        if (catalog_entry(id, found))
        {
            const json& entry = found;
            r["catalog_sha256"] = entry["sha256"];
            r["catalog_match"] = sha == entry["sha256"].as_string();
            if (entry["projector"].is_object())
            {
                r["catalog_projector_sha256"] = entry["projector"]["sha256"];
                r["catalog_projector_match"] = projector_sha == entry["projector"]["sha256"].as_string();
            }
        }
        if (decoder_match && projector_match) db_.set_verified(id, now_unix());
        return r;
    }

    bool model_manager::installed_with_hash(const std::string& sha, installed_model& m)
    {
        bool found = false;
        for (const auto& x : db_.models())
        {
            if (x.sha256 != sha || !file_exists(x.file)) continue;
            if (!found || x.source == "builtin") m = x;
            found = true;
        }
        return found;
    }

    bool model_manager::projector_with_hash(const std::string& sha, std::string& holder, std::string& path)
    {
        holder.clear();
        path.clear();
        for (const auto& x : db_.models())
            if (!x.projector.empty() && x.projector_sha256 == sha && file_exists(x.projector))
            {
                holder = x.id;
                path = x.projector;
                return true;
            }
        for (const auto& p : db_.projectors())
            if (p.sha256 == sha && file_exists(p.file))
            {
                path = p.file;
                return true;
            }
        return false;
    }

    std::string model_manager::catalog_projector(const std::string& decoder_sha) const
    {
        std::lock_guard<std::mutex> lock(m_);
        const json& catalog = catalog_;
        for (const auto& e : catalog["models"].elements())
            if (e["sha256"].as_string() == decoder_sha) return e["projector"]["sha256"].as_string();
        return std::string();
    }

    void model_manager::drop_pending_copy(const std::string& sha)
    {
        try
        {
            for (const auto& p : db_.projectors())
            {
                if (p.sha256 != sha) continue;
                if (parent_path(p.file) == dir_ && direct_path(dir_)) remove_file(p.file);
                db_.remove_projector(sha);
                log_info("models: the pending projector " + p.name + " is installed with its model");
            }
        }
        catch (const std::exception& e)
        {
            log_warning(std::string("models: the pending projectors cannot be read: ") + e.what());
        }
    }

    bool model_manager::take_pending(const pending_projector& pending, const std::string& target)
    {
        if (parent_path(pending.file) != dir_ || file_size(pending.file) != pending.size
            || !place(pending.file, target))
        {
            log_warning("models: the pending projector " + pending.file + " cannot be taken");
            return false;
        }
        try
        {
            db_.remove_projector(pending.sha256);
        }
        catch (const std::exception& e)
        {
            place(target, pending.file);
            log_warning("models: the pending projector " + pending.file + " cannot be taken: " + e.what());
            return false;
        }
        return true;
    }

    void model_manager::return_pending(const pending_projector& pending, const std::string& target)
    {
        if (!place(target, pending.file)) return;
        try
        {
            db_.put_projector(pending);
        }
        catch (const std::exception& e)
        {
            log_warning("models: the pending projector " + pending.file + " cannot be recorded again: " + e.what());
        }
    }

    std::string model_manager::free_identifier(const std::string& id, bool update)
    {
        installed_model holder;
        if (!db_.find_model(id, holder) || (update && holder.source == "catalog")) return id;
        for (int n = 1; n < 100; ++n)
        {
            const std::string candidate = id + "-import" + (n > 1 ? "-" + std::to_string(n) : std::string());
            if (!db_.find_model(candidate, holder)) return candidate;
        }
        return id + "-import-" + random_hex(8);
    }

    struct model_manager::import_item
    {
        std::string path;           // where the file lies
        std::string base;           // the trusted folder it is read from, empty for the root of its drive
        std::string name;           // the name it came under
        std::string sha256;
        json description;           // describe_gguf()
        bool projector = false;
        std::unique_ptr<link_free_file> file;   // open from inspect() until the file is placed
        std::string identity;
        // The outcome: "imported", "already", "pending" or "error"; the model concerned; the error.
        std::string result;
        std::string id;
        std::string error;
    };

    namespace
    {
        bool lacks_projector(const installed_model& m)
        {
            return m.projector.empty() || !file_exists(m.projector);
        }

        bool embedding_entry(const json& d)
        {
            if (d["embedding"].as_bool(false)) return true;
            for (const auto& r : d["roles"].elements())
                if (r.as_string() == "embedding") return true;
            return false;
        }
    }

    bool model_manager::inspect(import_item& item, const std::string& known_sha)
    {
        /*
            The file itself, reached through no link from base and with a single name: a link
            or a second name would lead the service, which reads with the rights of the
            system, to the file of another account.
        */
        item.file.reset(new link_free_file());
        std::string why;
        if (!item.file->open(item.path, item.base, why))
        {
            item.error = unreadable_reason(why);
            return false;
        }
        if (item.file->links() != 1)
        {
            item.file->close();
            item.error = "the file has several names (hard links), which imports never read";
            return false;
        }
        item.identity = item.file->identity();
        item.sha256 = known_sha.empty() ? sha256_open_file(*item.file, &stopped_) : known_sha;
        if (item.sha256.empty())
        {
            item.file->close();
            item.error = stopped_ ? "the service stops" : "unreadable";
            return false;
        }
        item.description = describe_gguf(item.path);
        // A projector by its header, or by the hash a catalog entry gives its projector.
        bool listed = false;
        {
            std::lock_guard<std::mutex> lock(m_);
            const json& catalog = catalog_;
            for (const auto& e : catalog["models"].elements())
                listed = listed || e["projector"]["sha256"].as_string() == item.sha256;
        }
        const json& description = item.description;
        item.projector = description["projector"].as_bool(false) || listed;
        return true;
    }

    namespace
    {
        pairing_file pairing_of_pending(const pending_projector& p)
        {
            pairing_file f;
            f.name = p.name;
            f.sha256 = p.sha256;
            f.description = p.descriptor;
            return f;
        }
    }

    void model_manager::import_items(std::vector<import_item>& items, bool move)
    {
        auto fail = [](import_item& it, const std::string& error) {
            it.result = "error";
            it.error = error;
            if (it.file) it.file->close();
        };
        auto pairing_of = [this](const import_item& it) {
            pairing_file f;
            f.name = file_name(it.name);
            f.sha256 = it.sha256;
            f.description = it.description;
            if (!it.projector) f.expected_projector = catalog_projector(it.sha256);
            return f;
        };
        // The files that could not be read, the decoders and the projectors.
        std::vector<size_t> decoders, projectors;
        for (size_t i = 0; i < items.size(); ++i)
        {
            import_item& it = items[i];
            if (!it.error.empty())
                fail(it, it.error);
            else if (it.projector && !static_cast<const json&>(it.description)["vision"].as_bool(true))
                fail(it, "the projector holds no vision encoder, and Albireo reads images only");
            else if (it.projector)
                projectors.push_back(i);
            else
                decoders.push_back(i);
        }
        // A projector installed already, or pending, is not copied again; with move, its file leaves the folder.
        std::vector<size_t> free_projectors;
        for (size_t i : projectors)
        {
            import_item& p = items[i];
            std::string holder, path;
            if (!projector_with_hash(p.sha256, holder, path))
            {
                free_projectors.push_back(i);
                continue;
            }
            p.result = "already";
            p.id = holder;
            p.file->close();
            if (move && p.identity != file_identity(path)) remove_file(p.path);
            log_info("models: " + p.name + " holds the projector " + (holder.empty() ? std::string("waiting for its "
                "decoder") : "of " + holder) + ", installed already: not copied again");
        }
        // The projectors of the files go with their decoders.
        std::vector<pairing_file> dp, pp;
        for (size_t i : decoders) dp.push_back(pairing_of(items[i]));
        for (size_t i : free_projectors) pp.push_back(pairing_of(items[i]));
        const std::vector<int> pairs = pair_projectors(dp, pp);
        std::vector<int> projector_of(items.size(), -1);
        for (size_t k = 0; k < pairs.size(); ++k)
            if (pairs[k] >= 0)
                projector_of[decoders[static_cast<size_t>(pairs[k])]] = static_cast<int>(free_projectors[k]);

        for (size_t i : decoders)
        {
            if (stopped_) break;
            import_item& d = items[i];
            import_item* p = projector_of[i] >= 0 ? &items[static_cast<size_t>(projector_of[i])] : nullptr;
            std::string error;
            // A file that repeats an installed model, one that comes with Albireo included, is not copied again.
            installed_model same;
            if (installed_with_hash(d.sha256, same))
            {
                d.result = "already";
                d.id = same.id;
                d.file->close();
                if (move && d.identity != file_identity(same.file)) remove_file(d.path);
                log_info("models: " + d.name + " holds " + same.id + ", installed already: not copied again"
                    + (move ? ", removed from its folder" : ""));
                // Its projector comes from the folder when the installed model lacks one.
                if (p && lacks_projector(same) && same.source != "builtin" && parent_path(same.file) == dir_)
                {
                    if (attach_projector(*p, same, move, error))
                    {
                        p->result = "imported";
                        p->id = same.id;
                    }
                    else
                    {
                        fail(*p, error);
                    }
                }
                continue;
            }
            // Without a projector among the files, a pending projector it goes with.
            pending_projector pending;
            bool with_pending = false;
            if (!p) with_pending = pending_for(d, pending);
            if (import_decoder(d, p, with_pending ? &pending : nullptr, move, error))
            {
                d.result = "imported";
                if (p)
                {
                    p->result = "imported";
                    p->id = d.id;
                }
                log_info("models: imported " + d.id + " from " + d.name + (p ? ", with the projector " + p->name
                    : with_pending ? ", with the projector " + pending.name : std::string()));
            }
            else
            {
                fail(d, error);
                if (p) fail(*p, "its decoder " + file_name(d.name) + " cannot be installed: " + error);
            }
        }
        // The other projectors go to an installed decoder that lacks one, or wait for theirs.
        for (size_t i : free_projectors)
        {
            if (stopped_) break;
            import_item& p = items[i];
            if (!p.result.empty()) continue;
            std::string error;
            installed_model target;
            if (installed_for(p, target))
            {
                if (attach_projector(p, target, move, error))
                {
                    p.result = "imported";
                    p.id = target.id;
                }
                else
                {
                    fail(p, error);
                }
            }
            else if (keep_pending(p, move, error))
            {
                p.result = "pending";
            }
            else
            {
                fail(p, error);
            }
        }
        for (auto& it : items)
            if (it.file) it.file->close();
    }

    json model_manager::import_result(const std::vector<import_item>& items)
    {
        json imported = json::array();
        json already = json::array();
        json pending = json::array();
        json errors = json::array();
        json files = json::array();
        auto add = [](json& list, const std::string& v) {
            for (const auto& x : static_cast<const json&>(list).elements())
                if (x.as_string() == v) return;
            list.push_back(v);
        };
        for (const auto& it : items)
        {
            json f = json::object();
            f["file"] = it.name;
            f["result"] = it.result;
            if (!it.id.empty()) f["id"] = it.id;
            files.push_back(f);
            if (it.result == "imported") add(imported, it.id);
            else if (it.result == "already" && !it.id.empty()) add(already, it.id);
            // A projector that waits, received again.
            else if (it.result == "already" || it.result == "pending") add(pending, file_name(it.name));
            else if (it.result == "error") errors.push_back(it.name + ": " + it.error);
        }
        json result = json::object();
        result["imported"] = imported;
        result["already"] = already;
        result["pending"] = pending;
        result["errors"] = errors;
        result["files"] = files;
        return result;
    }

    bool model_manager::replacing(const std::string& id, const std::function<bool()>& work)
    {
        installed_model m;
        if (id.empty() || !replace_cb_ || !db_.find_model(id, m)) return work();
        return replace_cb_(id, work);
    }

    bool model_manager::put_file(link_free_file& source, const std::string& path, const std::string& base,
        const std::string& sha, bool move, const std::string& target, placement& how, std::string& error,
        const std::string& id)
    {
        how = placement();
        // A file of the models folder at its own name, recorded again where it lies (a database started anew).
        how.in_place = !source.identity().empty() && file_identity(target) == source.identity();
        // The renaming that gives the file its name, over the file of an installed model through replacing().
        auto rename_to_target = [&](const std::string& from) {
            return replacing(id, [&]() { return place(from, target); });
        };
        /*
            Moved when asked and when the models folder is on the same volume, as the file
            checked above; copied otherwise from the file opened above, through a new
            temporary name, and checked against the hash as it is written.
        */
        if (move && !how.in_place)
        {
            const std::string identity = source.identity();
            source.close();
            how.moved = file_identity(path) == identity && rename_to_target(path);
            std::string why;
            if (!how.moved && !source.open(path, base, why))
            {
                error = unreadable_reason(why);
                return false;
            }
        }
        how.copied = !how.moved && !how.in_place;
        if (how.copied)
        {
            const std::string tmp = join_path(dir_, ".import-" + random_hex(16) + ".partial");
            if (copy_open_file(source, tmp, sha, &stopped_).empty() || !rename_to_target(tmp))
            {
                remove_file(tmp);
                source.close();
                how.copied = false;
                error = stopped_ ? "the service stops" : "copy failed";
                return false;
            }
        }
        source.close();
        return true;
    }

    void model_manager::undo_put(const std::string& path, const std::string& target, const placement& how)
    {
        if (how.moved) rename_file(target, path);
        else if (how.copied) remove_file(target);
    }

    bool model_manager::pending_for(import_item& d, pending_projector& found)
    {
        std::vector<pending_projector> list;
        for (const auto& p : db_.projectors())
            if (parent_path(p.file) == dir_ && file_size(p.file) == p.size) list.push_back(p);
        if (list.empty()) return false;
        // The header of a file of a bundle, read here only, once pending projectors exist.
        if (!d.description.is_object()) d.description = describe_gguf(d.path);
        pairing_file f;
        f.name = file_name(d.name);
        f.sha256 = d.sha256;
        f.description = d.description;
        f.expected_projector = catalog_projector(d.sha256);
        std::vector<pairing_file> pp;
        for (const auto& p : list) pp.push_back(pairing_of_pending(p));
        const std::vector<int> pairs = pair_projectors(std::vector<pairing_file>(1, f), pp);
        for (size_t k = 0; k < pairs.size(); ++k)
            if (pairs[k] == 0)
            {
                found = list[k];
                return true;
            }
        return false;
    }

    bool model_manager::installed_for(const import_item& p, installed_model& found)
    {
        // The installed decoders of the models folder that lack a projector and may read images.
        std::vector<installed_model> candidates;
        std::vector<pairing_file> dp;
        for (const auto& m : db_.models())
        {
            if (m.source == "builtin" || parent_path(m.file) != dir_ || !lacks_projector(m) || !file_exists(m.file)
                || embedding_entry(m.descriptor))
                continue;
            pairing_file f;
            f.name = file_name(m.file);
            f.sha256 = m.sha256;
            const json& d = m.descriptor;
            f.expected_projector = d["projector"]["sha256"].as_string();
            if (f.expected_projector.empty()) f.expected_projector = catalog_projector(m.sha256);
            // An entry that names neither a projector nor an architecture that reads images is left out unread.
            if (f.expected_projector.empty() && d["arch"].is_string()
                && !d["reads_images"].as_bool(reads_images(d["arch"].as_string())))
                continue;
            f.description = describe_gguf(m.file);
            candidates.push_back(m);
            dp.push_back(f);
        }
        if (candidates.empty()) return false;
        pairing_file f;
        f.name = file_name(p.name);
        f.sha256 = p.sha256;
        f.description = p.description;
        const std::vector<int> pairs = pair_projectors(dp, std::vector<pairing_file>(1, f));
        if (pairs[0] < 0) return false;
        found = candidates[static_cast<size_t>(pairs[0])];
        return true;
    }

    bool model_manager::import_decoder(import_item& d, import_item* p, const pending_projector* pending, bool move,
        std::string& error)
    {
        // A catalog entry with this hash gives the identifier and the description.
        json entry;
        bool known = false;
        {
            std::lock_guard<std::mutex> lock(m_);
            const json& catalog = catalog_;
            for (const auto& e : catalog["models"].elements())
                if (e["sha256"].as_string() == d.sha256)
                {
                    entry = e;
                    known = true;
                }
        }
        std::string id;
        if (known)
        {
            id = static_cast<const json&>(entry)["id"].as_string();
        }
        else
        {
            if (d.description.contains("error"))
            {
                error = d.description["error"].as_string();
                return false;
            }
            entry = d.description;
            std::string name = file_name(d.name);
            if (name.size() > 5 && name.substr(name.size() - 5) == ".gguf") name = name.substr(0, name.size() - 5);
            for (char& c : name)
                if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_' || c == '.')) c = '-';
            // A model of Albireo keeps its identifier.
            const std::string own = static_cast<const json&>(entry)["albireo_id"].as_string();
            id = own.empty() ? name : own;
            entry.erase("albireo_id");
            // The fields of the header that only the pairing reads.
            entry.erase("gguf_name");
            entry.erase("projection_width");
            /*
                A sidecar description written next to the file, read in the import folder
                only, where the administrators alone place files, and through no link.
            */
            std::string side;
            const std::string& f = d.path;
            if (f.size() > 5 && f.substr(f.size() - 5) == ".gguf" && d.base == join_path(data_dir(), "import")
                && read_file_link_free(f.substr(0, f.size() - 5) + ".json", d.base, side, 1u << 20))
            {
                const json s = json::parse_or_null(side);
                for (const auto& kv : s.items())
                    if (kv.first != "id" && kv.first != "files" && kv.first != "sha256" && kv.first != "projector")
                        entry[kv.first] = kv.second;
            }
        }
        /*
            A file never takes the identifier of another installed model, a model that comes with
            Albireo included; a newer version from the catalog of a model of the catalog replaces it.
        */
        id = free_identifier(id, known);
        entry["id"] = id;
        if (!safe_id(id))
        {
            error = "invalid identifier";
            return false;
        }
        // The models folder must still be reached without a link before the service writes in it.
        if (!direct_path(dir_))
        {
            error = "the models folder " + dir_ + " now leads elsewhere through a link";
            return false;
        }
        const std::string target = join_path(dir_, id + ".gguf");
        placement dh;
        if (!put_file(*d.file, d.path, d.base, d.sha256, move, target, dh, error, id)) return false;
        // The projector takes the name of the model: from the files, or from the pending projectors.
        std::string projector, projector_sha;
        placement ph;
        if (p)
        {
            projector = join_path(dir_, id + projector_suffix);
            projector_sha = p->sha256;
            if (!put_file(*p->file, p->path, p->base, p->sha256, move, projector, ph, error, id))
            {
                undo_put(d.path, target, dh);
                return false;
            }
        }
        else if (pending)
        {
            projector = join_path(dir_, id + projector_suffix);
            projector_sha = pending->sha256;
            // A pending projector that cannot be taken stays pending: the model reads text only.
            if (!take_pending(*pending, projector))
            {
                projector.clear();
                projector_sha.clear();
                pending = nullptr;
            }
        }
        if (!install_file(id, target, entry, d.sha256, known ? "catalog" : "import", projector, projector_sha, error))
        {
            // A model the database cannot record leaves the models folder: the files moved go
            // back where they came from, the copies are deleted, a pending projector waits again.
            undo_put(d.path, target, dh);
            if (p) undo_put(p->path, projector, ph);
            if (pending) return_pending(*pending, projector);
            return false;
        }
        d.id = id;
        // The sources of the copies leave once the model is recorded.
        if (move && dh.copied) remove_file(d.path);
        if (p && move && ph.copied) remove_file(p->path);
        return true;
    }

    bool model_manager::attach_projector(import_item& p, const installed_model& m, bool move, std::string& error)
    {
        if (!direct_path(dir_))
        {
            error = "the models folder " + dir_ + " now leads elsewhere through a link";
            return false;
        }
        const std::string target = join_path(dir_, m.id + projector_suffix);
        placement how;
        // The model lacks a projector: no program holds a file at that name, and the model stays in the engine.
        if (!put_file(*p.file, p.path, p.base, p.sha256, move, target, how, error, std::string())) return false;
        installed_model updated = m;
        updated.projector = target;
        updated.projector_size = file_size(target);
        updated.projector_sha256 = p.sha256;
        try
        {
            db_.put_model(updated);
        }
        catch (const std::exception& e)
        {
            undo_put(p.path, target, how);
            error = "the projector cannot be recorded: " + std::string(e.what());
            return false;
        }
        if (move && how.copied) remove_file(p.path);
        log_info("models: " + m.id + " receives the projector " + file_name(p.name) + " and reads images");
        // The model, loaded already, loads again with its projector.
        try
        {
            if (installed_cb_) installed_cb_(updated);
        }
        catch (const std::exception& e)
        {
            log_error("models: " + m.id + " has its projector, the service is not told: " + e.what());
        }
        return true;
    }

    bool model_manager::keep_pending(import_item& p, bool move, std::string& error)
    {
        if (p.description.contains("error"))
        {
            error = p.description["error"].as_string();
            return false;
        }
        if (!direct_path(dir_))
        {
            error = "the models folder " + dir_ + " now leads elsewhere through a link";
            return false;
        }
        const std::string target = join_path(dir_, ".pending-" + p.sha256.substr(0, 16) + projector_suffix);
        placement how;
        if (!put_file(*p.file, p.path, p.base, p.sha256, move, target, how, error, std::string())) return false;
        pending_projector pp;
        pp.sha256 = p.sha256;
        pp.file = target;
        pp.size = file_size(target);
        pp.name = file_name(p.name);
        pp.descriptor = p.description;
        pp.received_at = now_unix();
        try
        {
            db_.put_projector(pp);
        }
        catch (const std::exception& e)
        {
            undo_put(p.path, target, how);
            error = "the projector cannot be recorded: " + std::string(e.what());
            return false;
        }
        if (move && how.copied) remove_file(p.path);
        log_info("models: the projector " + pp.name + " waits for its decoder, in " + target);
        return true;
    }

    std::vector<installed_model> model_manager::register_builtin(const std::string& dir,
        std::vector<installed_model>& removed)
    {
        std::vector<installed_model> added;
        std::string text;
        const json manifest = read_file(join_path(dir, builtin_manifest_name), text, 16u << 20)
            ? json::parse_or_null(text) : json();
        auto plain = [](const std::string& name) {
            return !name.empty() && name.find_first_of("/\\:") == std::string::npos;
        };
        std::set<std::string> listed;
        for (const auto& e : manifest["models"].elements())
        {
            const std::string id = e["id"].as_string();
            const std::string name = e["file"].as_string();
            const std::string file = join_path(dir, name);
            if (!safe_id(id) || !plain(name) || !file_exists(file))
            {
                log_warning("models: the model " + id + " of " + builtin_manifest_name + " is missing from " + dir);
                continue;
            }
            listed.insert(id);
            json d = e;
            d.erase("file");
            d.erase("offset");
            d["builtin"] = true;
            // The projector of a model that reads images lies beside it; without it, the model reads text only.
            std::string projector, projector_sha;
            const json& p = e["projector"];
            if (p.is_object())
            {
                json pd = p;
                pd.erase("file");
                pd.erase("offset");
                d["projector"] = pd;
                const std::string pname = p["file"].as_string();
                if (plain(pname) && file_exists(join_path(dir, pname)))
                {
                    projector = join_path(dir, pname);
                    projector_sha = p["sha256"].as_string();
                }
                else
                {
                    log_warning("models: the projector of " + id + " is missing from " + dir + ": it reads text only");
                }
            }
            installed_model existing;
            const bool known = db_.find_model(id, existing);
            if (known && existing.source == "builtin" && existing.file == file
                && existing.sha256 == e["sha256"].as_string() && existing.descriptor == d
                && existing.projector == projector && existing.projector_sha256 == projector_sha)
                continue;
            installed_model m;
            m.id = id;
            m.name = e["name"].str(id);
            m.file = file;
            m.size = file_size(file);
            m.sha256 = e["sha256"].as_string();
            m.source = "builtin";
            m.installed_at = known ? existing.installed_at : now_unix();
            m.projector = projector;
            m.projector_size = projector.empty() ? 0 : file_size(projector);
            m.projector_sha256 = projector_sha;
            // The installer checked the hashes of the files it wrote.
            m.verified_at = known && existing.sha256 == m.sha256 && existing.projector_sha256 == m.projector_sha256
                ? existing.verified_at : now_unix();
            m.descriptor = d;
            db_.put_model(m);
            if (!known) added.push_back(m);
            log_info("models: " + id + " comes with Albireo, in " + file + (projector.empty() ? std::string()
                : ", with its projector " + projector));
        }
        for (const auto& m : db_.models())
        {
            if (m.source != "builtin" || listed.count(m.id)) continue;
            db_.remove_model(m.id);
            removed.push_back(m);
            log_info("models: " + m.id + " no longer comes with Albireo");
        }
        return added;
    }

    json model_manager::import_path(const std::string& path, const std::string& base, bool move)
    {
        std::vector<std::string> files;
        // The import folder itself, or a folder under it reached through no link.
        if (path == base ? dir_exists(path) : link_free_folder(path, base))
        {
            // The folder and its models subfolder, the layout of an offline bundle; inspect()
            // refuses the links among them.
            for (const std::string& dir : { path, join_path(path, "models") })
            {
                if (dir != path && !link_free_folder(dir, base)) continue;
                for (const auto& n : list_dir(dir))
                    if (n.size() > 5 && n.substr(n.size() - 5) == ".gguf") files.push_back(join_path(dir, n));
            }
        }
        else
        {
            files.push_back(path);
        }
        std::vector<import_item> items;
        for (const auto& f : files)
        {
            if (stopped_) break;
            import_item it;
            it.path = f;
            it.base = base;
            it.name = f;
            inspect(it, std::string());
            items.push_back(std::move(it));
        }
        import_items(items, move);
        return import_result(items);
    }

    bool model_manager::bundle_file(const std::string& folder, const json& parts, const std::string& fallback,
        const std::string& sha, uint64_t size, import_item& out)
    {
        // The names under which tools/make_offline_bundle.py places a file: no other file is read.
        std::vector<std::string> names;
        if (parts.size() == 1) names.push_back(file_name(parts[0]["url"].as_string()));
        names.push_back(fallback);
        out.file.reset(new link_free_file());
        std::string why, refused;
        for (const std::string& dir : { join_path(folder, "models"), folder })
            for (const auto& name : names)
            {
                if (out.file->is_open() || !plain_name(name)) continue;
                const std::string p = join_path(dir, name);
                if (out.file->open(p, std::string(), why)) out.path = p;
                else if (why == "link" && refused.empty()) refused = p;
            }
        if (!out.file->is_open())
        {
            if (!refused.empty())
            {
                out.name = refused;
                out.error = unreadable_reason("link");
            }
            return false;
        }
        out.name = out.path;
        out.base = std::string();
        out.identity = out.file->identity();
        // Its size first, then its hash, before anything is copied.
        if (out.file->size() != size || out.file->links() != 1 || sha256_open_file(*out.file, &stopped_) != sha)
        {
            out.file->close();
            out.error = stopped_ ? std::string("the service stops") : "does not match the catalog of the bundle";
            return false;
        }
        out.sha256 = sha;
        return true;
    }

    json model_manager::import_bundle(const std::string& folder, const json& catalog)
    {
        std::vector<import_item> items;
        for (const auto& e : catalog["models"].elements())
        {
            if (stopped_) break;
            const std::string listed = e["id"].as_string();
            const std::string sha = e["sha256"].as_string();
            const uint64_t size = static_cast<uint64_t>(e["size"].as_int(0));
            if (!safe_id(listed) || sha.size() != 64 || size == 0) continue;
            import_item d;
            if (!bundle_file(folder, e["files"], listed + ".gguf", sha, size, d))
            {
                if (!d.error.empty())
                {
                    d.result = "error";
                    if (d.error != unreadable_reason("link")) d.error += " (the model " + listed + ")";
                    items.push_back(std::move(d));
                }
                continue;
            }
            // The projector the entry names, when the bundle holds it.
            import_item p;
            bool with_projector = false;
            const json& pe = e["projector"];
            const std::string projector_sha = pe["sha256"].as_string();
            const uint64_t projector_size = static_cast<uint64_t>(pe["size"].as_int(0));
            if (pe.is_object() && projector_sha.size() == 64 && projector_size > 0)
            {
                with_projector = bundle_file(folder, pe["files"], listed + projector_suffix, projector_sha,
                    projector_size, p);
                std::string holder, path;
                if (with_projector && projector_with_hash(projector_sha, holder, path))
                {
                    // Installed already, or pending.
                    p.file->close();
                    p.result = "already";
                    p.id = holder;
                    with_projector = false;
                }
                else if (!with_projector && !p.error.empty())
                {
                    p.result = "error";
                }
            }
            std::string error;
            installed_model same;
            if (installed_with_hash(sha, same))
            {
                d.file->close();
                d.result = "already";
                d.id = same.id;
                log_info("models: " + d.path + " holds " + same.id + ", installed already: not copied again");
                // The projector of the bundle goes to the installed decoder when it lacks one.
                if (with_projector && lacks_projector(same) && same.source != "builtin"
                    && parent_path(same.file) == dir_)
                {
                    if (attach_projector(p, same, false, error))
                    {
                        p.result = "imported";
                        p.id = same.id;
                    }
                    else
                    {
                        p.result = "error";
                        p.error = error;
                    }
                }
            }
            else
            {
                // The entry of the catalog in use when it describes this file, else that of the bundle.
                json found;
                const bool current = catalog_entry(listed, found) && static_cast<const json&>(found)["sha256"]
                    .as_string() == sha;
                json entry = current ? found : e;
                // A newer version of a model of the catalog takes its place; another model keeps its own.
                const std::string id = free_identifier(listed, true);
                entry["id"] = id;
                const std::string target = join_path(dir_, id + ".gguf");
                const std::string projector = join_path(dir_, id + projector_suffix);
                placement dh, ph;
                pending_projector pending;
                const bool with_pending = !with_projector && pending_for(d, pending);
                bool ok = direct_path(dir_)
                    && put_file(*d.file, d.path, std::string(), sha, false, target, dh, error, id);
                if (ok && with_projector
                    && !put_file(*p.file, p.path, std::string(), projector_sha, false, projector, ph, error, id))
                {
                    undo_put(d.path, target, dh);
                    ok = false;
                }
                const bool taken = ok && with_pending && take_pending(pending, projector);
                const std::string kept = with_projector || taken ? projector : std::string();
                const std::string kept_sha = with_projector ? projector_sha : taken ? pending.sha256 : std::string();
                if (ok && !install_file(id, target, entry, sha, "catalog", kept, kept_sha, error))
                {
                    undo_put(d.path, target, dh);
                    if (with_projector) undo_put(p.path, projector, ph);
                    if (taken) return_pending(pending, projector);
                    ok = false;
                }
                if (ok)
                {
                    d.result = "imported";
                    d.id = id;
                    if (with_projector)
                    {
                        p.result = "imported";
                        p.id = id;
                    }
                    log_info("models: imported " + id + " from the offline bundle " + d.path
                        + (with_projector ? ", with its projector " + p.path : std::string()));
                }
                else
                {
                    d.result = "error";
                    d.error = stopped_ ? std::string("the service stops") : error.empty() ? std::string("copy failed")
                        : error;
                    if (with_projector)
                    {
                        p.result = "error";
                        p.error = d.error;
                    }
                }
            }
            if (d.file) d.file->close();
            if (p.file) p.file->close();
            items.push_back(std::move(d));
            if (!p.result.empty()) items.push_back(std::move(p));
        }
        json result = import_result(items);
        result.erase("files");
        return result;
    }

    json model_manager::import_received(const std::string& file, const std::string& original_name,
        const std::string& sha)
    {
        std::vector<import_item> items(1);
        import_item& it = items[0];
        it.path = file;
        it.base = dir_;
        it.name = original_name;
        inspect(it, sha);
        import_items(items, true);
        if (it.result.empty())
        {
            it.result = "error";
            it.error = "the service stops";
        }
        // A file that cannot be installed leaves the models folder.
        if (it.result == "error") remove_file(file);
        else if (it.result == "imported") log_info("models: the transfer of " + original_name + " installed " + it.id);
        json result = import_result(items);
        result.erase("files");
        return result;
    }

}
