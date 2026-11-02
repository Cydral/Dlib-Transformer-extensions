// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef PSEUDALIS_HOST_DRIVE_H_
#define PSEUDALIS_HOST_DRIVE_H_

#include <memory>
#include <string>

#include "../common/json.h"

namespace pseudalis
{
    class service;

    class drive_host
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The drive of Pseudalis: a file system of WinFsp, served by the service and
                mounted on the letter of the settings for every session of the computer.
                It works as any drive (the Explorer, cmd: dir, copy, move, del, ren, md,
                rd /s, type, edit), with one difference: a file written there is
                protected when its last handle closes, and only the protected file is ever
                read back.

                    - a document keeps its name, protected, its original kept in the vault;
                    - a picture becomes <name>.txt, its marker "[Image@<id> : <description>]";
                    - a ZIP, TAR or TAR.GZ archive keeps its name, its documents protected,
                      its pictures as text files, its other entries left out;
                    - a file of another kind is refused at its creation (access denied); a
                      file whose content cannot be read keeps its previous version, or is
                      removed when it had none, with an event;
                    - while a file is written and protected, the other programs cannot open
                      its content (sharing violation).

                The tree of the drive lies in the database, the protected content in the
                vault.  The originals come back through the "Restore" command of the
                Explorer, the web application or the interface, from any copy of a
                protected file, since a file system cannot tell a move out of the drive
                from a copy followed by a deletion: deleting or replacing a file marks its
                record (service::mark_removed()), whose original stays for the days of the
                setting removed_days, then goes with the pictures no other file shows.

                On a system without WinFsp, and outside Windows, start() fails and the
                service runs without drive.
        !*/
    public:
        explicit drive_host(service& svc);
        ~drive_host();
        drive_host(const drive_host&) = delete;
        drive_host& operator=(const drive_host&) = delete;

        bool start(const std::string& letter, std::string& error);
        void stop();
        // {"letter", "mounted", "files", "bytes", "error"}
        json status() const;

    private:
        struct impl;
        std::unique_ptr<impl> p_;
    };
}

#endif // PSEUDALIS_HOST_DRIVE_H_
