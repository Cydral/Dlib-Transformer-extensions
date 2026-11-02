// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#if defined(_WIN32) && !defined(_WIN32_WINNT)
#define _WIN32_WINNT 0x0A00
#endif
#include <dlib/external/httplib/httplib.h>

#include "albireo_client.h"

#include <cstdlib>

#if defined(_WIN32)
#include <windows.h>
#endif

#include "../common/log.h"
#include "../common/platform.h"
#include "crypto.h"

namespace pseudalis
{
    namespace
    {
        const char* language_name(const std::string& code)
        {
            if (code == "fr") return "French";
            if (code == "de") return "German";
            if (code == "it") return "Italian";
            if (code == "es") return "Spanish";
            return "English";
        }

        // The answer without the reasoning that some models write before it.
        std::string without_reasoning(std::string text)
        {
            for (size_t b = text.find("<think>"); b != std::string::npos; b = text.find("<think>"))
            {
                const size_t e = text.find("</think>", b);
                text.erase(b, e == std::string::npos ? std::string::npos : e + 8 - b);
            }
            const size_t first = text.find_first_not_of(" \t\r\n");
            if (first == std::string::npos) return std::string();
            return text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
        }

        std::unique_ptr<httplib::SSLClient> client(int port, const std::string& ca, int timeout)
        {
            std::unique_ptr<httplib::SSLClient> c(new httplib::SSLClient("localhost", port));
            c->set_ca_cert_path(ca.c_str());
            c->enable_server_certificate_verification(true);
            c->set_connection_timeout(3);
            c->set_read_timeout(timeout);
            c->set_write_timeout(timeout);
            c->set_keep_alive(false);
            return c;
        }
    }

    albireo_client::albireo_client() {}

    std::string albireo_client::albireo_data_dir()
    {
#if defined(_WIN32)
        HKEY key;
        if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, L"Software\\Albireo", 0, KEY_READ | KEY_WOW64_64KEY, &key)
            == ERROR_SUCCESS)
        {
            wchar_t buf[1024];
            DWORD size = sizeof(buf), type = 0;
            const LONG rc = RegQueryValueExW(key, L"DataDir", nullptr, &type, reinterpret_cast<BYTE*>(buf), &size);
            RegCloseKey(key);
            if (rc == ERROR_SUCCESS && type == REG_SZ && size >= sizeof(wchar_t))
            {
                const std::string dir = narrow(std::wstring(buf, size / sizeof(wchar_t) - 1));
                if (dir_exists(dir)) return dir;
            }
        }
        /*
            Only the folder that the installer of Albireo recorded, which only an administrator
            can record: any user can create a folder Albireo in %ProgramData%.
        */
        return std::string();
#else
        const char* home = std::getenv("ALBIREO_HOME");
        if (home && *home) return dir_exists(home) ? std::string(home) : std::string();
        const char* user = std::getenv("HOME");
        const std::string dir = user ? join_path(user, ".local/share/albireo") : std::string();
        return dir_exists(dir) ? dir : std::string();
#endif
    }

    void albireo_client::set_enabled(bool on)
    {
        std::lock_guard<std::mutex> lock(m_);
        enabled_ = on;
        checked_ = 0;
    }

    bool albireo_client::probe(int& port, std::string& ca, std::string& model, std::string& reason)
    {
        port = 8443;
        ca.clear();
        model.clear();
        const std::string dir = albireo_data_dir();
        if (dir.empty())
        {
            reason = "Albireo is not installed";
            return false;
        }
        const std::string settings_path = join_path(dir, "settings.json");
        ca = join_path(join_path(dir, "certificates"), "albireo-authority.crt");
        if (!file_exists(ca))
        {
            reason = "the local authority of Albireo is missing";
            return false;
        }
        /*
            A folder, an authority or settings that another account than SYSTEM or the
            administrators could have written are not those of Albireo: the pictures would
            reach whoever placed them.
        */
        if (!folder_admin_only(dir) || !folder_admin_only(join_path(dir, "certificates")) || !owned_by_system(ca)
            || (file_exists(settings_path) && !owned_by_system(settings_path)))
        {
            reason = "the data folder of Albireo is not protected: its pictures are not described";
            return false;
        }
        std::string text;
        json settings;
        if (read_file(settings_path, text, 1u << 20) && json::parse(text, settings))
        {
            const long long p = settings["port"].as_int(8443);
            port = p >= 1 && p <= 65535 ? static_cast<int>(p) : 8443;
        }
        auto c = client(port, ca, 10);
        auto r = c->Get("/v1/models");
        json models;
        if (!r || r->status != 200 || !json::parse(r->body, models))
        {
            reason = "Albireo does not answer on port " + std::to_string(port);
            return false;
        }
        std::string any;
        for (const json& m : models["data"].elements())
        {
            bool image = false;
            for (const json& i : m["inputs"].elements()) image = image || i.str() == "image";
            if (!image) continue;
            if (m["id"].str() == "albireo-general") model = m["id"].str();
            if (any.empty()) any = m["id"].str();
        }
        if (model.empty()) model = any;
        if (model.empty())
        {
            reason = "no model of Albireo reads images";
            return false;
        }
        reason.clear();
        return true;
    }

    bool albireo_client::refresh(std::string& reason)
    {
        /*
            The state is checked again once a minute, by one caller at a time and outside the
            lock, so that a slow Albireo never holds the callers of status(): the others get
            the state in place meanwhile.
        */
        {
            std::lock_guard<std::mutex> lock(m_);
            if (!enabled_)
            {
                reason = "turned off in the settings";
                return false;
            }
            if (checking_ || (checked_ && now_unix() - checked_ < 60))
            {
                reason = checked_ ? reason_ : std::string("Albireo is being looked for");
                return found_;
            }
            checking_ = true;
        }
        int port = 8443;
        std::string ca, model, why;
        bool found = false;
        try
        {
            found = probe(port, ca, model, why);
        }
        catch (const std::exception& x)
        {
            why = x.what();
        }
        std::lock_guard<std::mutex> lock(m_);
        checking_ = false;
        checked_ = now_unix();
        found_ = found;
        port_ = port;
        ca_ = ca;
        model_ = model;
        reason_ = why;
        reason = why;
        return found;
    }

    json albireo_client::status()
    {
        std::string reason;
        const bool ok = refresh(reason);
        std::lock_guard<std::mutex> lock(m_);
        json j = json::object();
        j["enabled"] = enabled_;
        j["found"] = ok;
        j["port"] = port_;
        j["model"] = ok ? model_ : std::string();
        j["reason"] = reason;
        return j;
    }

    std::string albireo_client::describe(const std::string& bytes, const std::string& media_type,
                                         const std::string& language, std::string& error)
    {
        if (media_type != "image/png" && media_type != "image/jpeg")
        {
            error = "Albireo reads PNG and JPEG pictures only";
            return std::string();
        }
        if (bytes.size() > (15u << 20))
        {
            error = "the picture is larger than 15 MB";
            return std::string();
        }
        int port = 0;
        std::string ca, model;
        if (!refresh(error)) return std::string();
        {
            std::lock_guard<std::mutex> lock(m_);
            if (!found_)
            {
                error = reason_;
                return std::string();
            }
            port = port_;
            ca = ca_;
            model = model_;
        }
        json body = json::object();
        body["model"] = model;
        body["stream"] = false;
        body["temperature"] = 0.2;
        body["max_tokens"] = 1024;
        json messages = json::array();
        json system = json::object();
        system["role"] = "system";
        system["content"] = std::string("You describe pictures taken from documents whose personal data is protected. "
            "Write one or two factual sentences in ") + language_name(language) + ": what the picture shows and its "
            "kind (photograph, diagram, chart, logo, map, screenshot, scanned page, signature...). Never name a person "
            "and never copy a name, an address, a number, a plate or an identifier the picture shows: write \"a "
            "person\", \"a name\", \"a number\" instead. Write the description alone.";
        messages.push_back(system);
        json user = json::object();
        user["role"] = "user";
        json content = json::array();
        json text = json::object();
        text["type"] = "text";
        text["text"] = "Describe this picture.";
        content.push_back(text);
        json image = json::object();
        image["type"] = "image_url";
        json url = json::object();
        url["url"] = "data:" + media_type + ";base64," + base64_encode(bytes);
        image["image_url"] = url;
        content.push_back(image);
        user["content"] = content;
        messages.push_back(user);
        body["messages"] = messages;

        std::lock_guard<std::mutex> one(request_m_);
        auto c = client(port, ca, 300);
        auto r = c->Post("/v1/chat/completions", body.dump(), "application/json");
        json answer;
        if (!r)
        {
            error = "Albireo does not answer";
            std::lock_guard<std::mutex> lock(m_);
            checked_ = 0;
            return std::string();
        }
        if (r->status != 200 || !json::parse(r->body, answer))
        {
            error = "Albireo answered " + std::to_string(r->status);
            return std::string();
        }
        const json& choices = answer["choices"];
        const std::string description =
            choices.size() > 0 ? without_reasoning(choices.elements()[0]["message"]["content"].str()) : std::string();
        if (description.empty()) error = "Albireo gave no description";
        return description;
    }
}
