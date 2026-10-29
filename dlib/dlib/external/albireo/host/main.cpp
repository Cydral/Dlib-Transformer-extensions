// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

/*
    albireo-service: the Albireo service.

        albireo-service                  as a Windows service, started by the service manager;
                                         in the console elsewhere, or when started by hand
        albireo-service --console        in the console, stopped by Ctrl+C
        albireo-service --version

    The service starts itself again in two cases: the arguments below, used for the
    work that must never take the service down with it.

        --probe-engine <library>         prints the description of an engine library
        --bench-engine <library> <model> <configuration>
                                         measures an engine and prints the result

    A restart normally happens within the process.  Once the engine of a device has run,
    whose library stays loaded, the service restarts in a new process instead: as a
    Windows service, through a copy of the program started with --restart-service, which
    waits for the service to stop and starts it again.
*/

#if defined(_WIN32) && !defined(_WIN32_WINNT)
#define _WIN32_WINNT 0x0A00
#endif

#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#else
#include <limits.h>
#include <pthread.h>
#include <signal.h>
#include <unistd.h>
#endif

#include "../common/log.h"
#include "../common/platform.h"
#include "../common/version.h"
#include "engine_host.h"
#include "service.h"

namespace
{
    std::mutex current_m;
    albireo::service* current = nullptr;
    std::atomic<bool> stop_requested{false};

    // Returned by serve() when the restart needs a new process.
    const int restart_process = 75;

    void request_stop()
    {
        stop_requested = true;
        std::lock_guard<std::mutex> lock(current_m);
        if (current) current->stop();
    }

    // Runs the service, again after each restart it asks for; returns the exit code.
    int serve(bool console)
    {
        int failures = 0;
        while (!stop_requested)
        {
            albireo::service svc;
            {
                std::lock_guard<std::mutex> lock(current_m);
                current = &svc;
            }
            std::string error;
            const bool ok = !stop_requested && svc.run(error);
            {
                std::lock_guard<std::mutex> lock(current_m);
                current = nullptr;
            }
            if (stop_requested) break;
            if (!ok)
            {
                albireo::log_error("service: " + error);
                if (console) std::cerr << "albireo-service: " << error << std::endl;
                // A start that fails is tried again a few times, for a port released late
                // by the previous run or a disk that mounts after the service.
                if (++failures >= 3) return 1;
                for (int i = 0; i < 100 && !stop_requested; ++i)
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            failures = 0;
            if (svc.update_started() || !svc.restart_requested()) break;
            if (albireo::engine_host::pinned_engine_started())
            {
                albireo::log_info("service: restart in a new process");
                return restart_process;
            }
            albireo::log_info("service: restart");
        }
        return 0;
    }

#if defined(_WIN32)
    const wchar_t service_name[] = L"Albireo";
    SERVICE_STATUS_HANDLE status_handle = nullptr;
    SERVICE_STATUS status = {};
    std::mutex status_m;

    void report(DWORD state, DWORD exit_code = NO_ERROR, DWORD wait_hint = 0)
    {
        std::lock_guard<std::mutex> lock(status_m);
        static DWORD checkpoint = 1;
        status.dwServiceType = SERVICE_WIN32_OWN_PROCESS;
        status.dwCurrentState = state;
        status.dwWaitHint = wait_hint;
        status.dwControlsAccepted = state == SERVICE_START_PENDING ? 0
            : SERVICE_ACCEPT_STOP | SERVICE_ACCEPT_PRESHUTDOWN;
        if (exit_code != NO_ERROR)
        {
            status.dwWin32ExitCode = ERROR_SERVICE_SPECIFIC_ERROR;
            status.dwServiceSpecificExitCode = exit_code;
        }
        else
        {
            status.dwWin32ExitCode = NO_ERROR;
            status.dwServiceSpecificExitCode = 0;
        }
        status.dwCheckPoint = state == SERVICE_RUNNING || state == SERVICE_STOPPED ? 0 : checkpoint++;
        SetServiceStatus(status_handle, &status);
    }

    DWORD WINAPI control_handler(DWORD control, DWORD, LPVOID, LPVOID)
    {
        switch (control)
        {
        case SERVICE_CONTROL_STOP:
        case SERVICE_CONTROL_PRESHUTDOWN:
        case SERVICE_CONTROL_SHUTDOWN:
            report(SERVICE_STOP_PENDING, NO_ERROR, 30000);
            request_stop();
            return NO_ERROR;
        case SERVICE_CONTROL_INTERROGATE:
            return NO_ERROR;
        default:
            return ERROR_CALL_NOT_IMPLEMENTED;
        }
    }

    std::wstring program_path()
    {
        std::wstring path(32768, L'\0');
        const DWORD n = GetModuleFileNameW(nullptr, &path[0], static_cast<DWORD>(path.size()));
        path.resize(n);
        return path;
    }

    // Starts a copy of the program that starts the service again once it stopped.
    void start_restarter()
    {
        const std::wstring exe = program_path();
        std::wstring command = L"\"" + exe + L"\" --restart-service";
        STARTUPINFOW si = {};
        si.cb = sizeof(si);
        PROCESS_INFORMATION pi = {};
        if (CreateProcessW(exe.c_str(), &command[0], nullptr, nullptr, FALSE,
            CREATE_NO_WINDOW | DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP, nullptr, nullptr, &si, &pi))
        {
            CloseHandle(pi.hThread);
            CloseHandle(pi.hProcess);
        }
        else
        {
            albireo::log_error("service: cannot start the restart of the service (error "
                + std::to_string(GetLastError()) + ")");
        }
    }

    // --restart-service: waits up to a minute for the service to stop, then starts it.
    int restart_service()
    {
        SC_HANDLE scm = OpenSCManagerW(nullptr, nullptr, SC_MANAGER_CONNECT);
        if (!scm) return 1;
        SC_HANDLE svc = OpenServiceW(scm, service_name, SERVICE_QUERY_STATUS | SERVICE_START);
        if (!svc)
        {
            CloseServiceHandle(scm);
            return 1;
        }
        SERVICE_STATUS st = {};
        for (int i = 0; i < 600; ++i)
        {
            if (!QueryServiceStatus(svc, &st) || st.dwCurrentState == SERVICE_STOPPED) break;
            Sleep(100);
        }
        const BOOL ok = StartServiceW(svc, 0, nullptr);
        CloseServiceHandle(svc);
        CloseServiceHandle(scm);
        return ok ? 0 : 1;
    }

    void WINAPI service_main(DWORD, LPWSTR*)
    {
        status_handle = RegisterServiceCtrlHandlerExW(service_name, control_handler, nullptr);
        if (!status_handle) return;
        report(SERVICE_START_PENDING, NO_ERROR, 10000);
        // The service answers the manager at once; the engines and the models load in the
        // background while the server already runs.
        report(SERVICE_RUNNING);
        const int code = serve(false);
        if (code == restart_process)
        {
            start_restarter();
            report(SERVICE_STOPPED);
            return;
        }
        report(SERVICE_STOPPED, code == 0 ? NO_ERROR : static_cast<DWORD>(code));
    }

    BOOL WINAPI console_handler(DWORD)
    {
        request_stop();
        return TRUE;
    }

    int run_console(const std::vector<std::string>&)
    {
        SetConsoleCtrlHandler(console_handler, TRUE);
        albireo::log_file::instance().set_console(true);
        const int code = serve(true);
        if (code != restart_process) return code;
        // The same command line in a new process, in the same console.
        const std::wstring exe = program_path();
        std::wstring command = GetCommandLineW();
        STARTUPINFOW si = {};
        si.cb = sizeof(si);
        PROCESS_INFORMATION pi = {};
        if (!CreateProcessW(exe.c_str(), &command[0], nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi))
            return 1;
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
        return 0;
    }
#else
    int run_console(const std::vector<std::string>& args)
    {
        // The signals are handled by a thread of their own, outside any signal handler.
        sigset_t set;
        sigemptyset(&set);
        sigaddset(&set, SIGINT);
        sigaddset(&set, SIGTERM);
        sigaddset(&set, SIGHUP);
        pthread_sigmask(SIG_BLOCK, &set, nullptr);
        signal(SIGPIPE, SIG_IGN);
        std::thread([set]() {
            int sig = 0;
            if (sigwait(&set, &sig) == 0) request_stop();
        }).detach();
        albireo::log_file::instance().set_console(true);
        const int code = serve(true);
        if (code != restart_process) return code;
        // The same command line in a new process.
        std::vector<char*> argv;
        for (const auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
        argv.push_back(nullptr);
        char self[PATH_MAX];
        const ssize_t n = readlink("/proc/self/exe", self, sizeof(self) - 1);
        if (n > 0)
        {
            self[n] = '\0';
            execv(self, argv.data());
        }
        execvp(argv[0], argv.data());
        return 1;
    }
#endif
}

int main(int argc, char** argv)
{
    std::vector<std::string> args;
#if defined(_WIN32)
    // The arguments in UTF-8, whatever the code page of the console.
    int wargc = 0;
    LPWSTR* wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
    for (int i = 0; wargv && i < wargc; ++i) args.push_back(albireo::narrow(wargv[i]));
    if (wargv) LocalFree(wargv);
    (void)argc;
    (void)argv;
#else
    for (int i = 0; i < argc; ++i) args.push_back(argv[i]);
#endif

    if (args.size() >= 3 && args[1] == "--probe-engine") return albireo::service::probe_engine(args[2]);
    if (args.size() >= 5 && args[1] == "--bench-engine")
        return albireo::service::benchmark_engine(args[2], args[3], args[4]);
    if (args.size() >= 2 && args[1] == "--version")
    {
        std::cout << ALBIREO_PRODUCT_NAME << " " << ALBIREO_VERSION << std::endl;
        return 0;
    }
    const bool console = args.size() >= 2 && args[1] == "--console";
#if defined(_WIN32)
    if (args.size() >= 2 && args[1] == "--restart-service") return restart_service();
    if (!console)
    {
        SERVICE_TABLE_ENTRYW table[] = { { const_cast<LPWSTR>(service_name), service_main }, { nullptr, nullptr } };
        if (StartServiceCtrlDispatcherW(table)) return 0;
        if (GetLastError() != ERROR_FAILED_SERVICE_CONTROLLER_CONNECT) return 1;
    }
#else
    (void)console;
#endif
    return run_console(args);
}
