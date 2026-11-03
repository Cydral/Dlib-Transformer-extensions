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
                The drive of Pseudalis: the folder drive of the data folder, which the
                service watches and shows under the letter of the settings for every session
                of the computer, as the command subst would.  It works as any folder, from
                the Explorer and from the prompt, with one difference: a file written to it
                is anonymized once its writer closed it.  Its original goes to the encrypted
                vault with a record, the anonymized file takes its place, under its name with
                its values replaced (pictures: <name>.txt, holding their marker), and the file
                written is deleted.  A folder that comes in takes a name without personal data
                once the files under it rest.  A file of another kind, or that cannot be
                anonymized, goes to the folder of the files not processed of the user who
                wrote it (refused\<user> in the data folder), at the place it had in the tree
                of the drive, with an event.

                No driver: the file written lies in its folder, in clear, from its last write
                to its processing, a few seconds for most documents, during which the service
                holds it alone (sharing violation).  Deleting a file of the drive marks its
                record (service::mark_removed()), whose original stays for the days of the
                setting removed_days; a file found again in the drive, renamed or moved there,
                keeps it.  The files already processed are recorded in the database, so that a
                restart processes only what changed.

                Outside Windows, the folder works without letter.
        !*/
    public:
        explicit drive_host(service& svc);
        ~drive_host();
        drive_host(const drive_host&) = delete;
        drive_host& operator=(const drive_host&) = delete;

        // letter: "A".."Z", or empty for no drive.
        bool start(const std::string& letter, std::string& error);
        void stop();
        /*!
            ensures
                - Returns {"letter", "mounted", "folder", "files", "bytes", "error",
                  "refused_folder"}.
        !*/
        json status() const;

    private:
        struct impl;
        std::unique_ptr<impl> p_;
    };
}

#endif // PSEUDALIS_HOST_DRIVE_H_
