// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef PSEUDALIS_WINDOWS_RESOURCE_H_
#define PSEUDALIS_WINDOWS_RESOURCE_H_

// Identifiers of the resources of the icon, of pseudalis.exe and of the installer.

#define IDI_APP 1
#define IDI_TRAY 101
#define IDI_TRAY_OFF 103
#define IDI_TRAY_UPDATE 104

#define IDB_ABOUT 201
#define IDB_SIDE 202
#define IDB_MARK 203

#define IDR_LICENSE 301
#define IDR_PAYLOAD 401

/*
    The texts of the commands of the Explorer, in each language of pseudalis.exe.  The Explorer
    keeps the text it read for an indirect string ("@pseudalis.exe,-511"): new identifiers make it
    read these tables again.
*/
#define IDS_VERB_PROTECT 511
#define IDS_VERB_PROTECT_COPY 512
#define IDS_VERB_RESTORE 513
#define IDS_VERB_RESTORE_COPY 514

#endif // PSEUDALIS_WINDOWS_RESOURCE_H_
