# Pseudalis

Pseudalis protects the personal data of documents on a Windows computer, on that computer alone.
A service anonymizes the files and the folders it receives through a drive, through commands of the
Explorer, through a web application and through a local interface, and keeps their originals encrypted,
so that each one comes back on demand from any copy of its anonymized file, through the same commands,
application and interface. It is built on [`dlib/privacy`](../../privacy.h) and on the recognizers
trained for the example [`slm_document_anonymizer_ex`](../../../examples/slm_document_anonymizer_ex.cpp),
which its engines carry.

Pseudalis Community, the edition published here, is written by Aldric Pierrain, Copyright (C) 2026
Cydral Technology, and distributed under the Boost Software License 1.0
([`resources/license.txt`](resources/license.txt)). It installs no driver: its drive is a folder of its
data folder, which the service watches and Windows shows under its letter. The installer of each version
lies in [`dist/pseudalis`](../../../../dist/pseudalis), where anyone downloads it.

The architecture of the product, its four detection devices, its format-preserving encryption and the
measures of its recognizers are described in
[Technical application architecture of Pseudalis Community](../../../../docs/pseudalis-technical-architecture.pdf),
a research report kept in [`docs/`](../../../../docs).

## What it does

Each personal value of a document (names, addresses, telephone numbers, e-mail addresses, identifiers,
bank details, dates and the rest of the types of `dlib/privacy`) is replaced by a tag such as
`[[PERSON:Xoydoeh Nuggwdfsorn#EKR]]`, which carries the value encrypted with FF1 over AES-256 and a
check. The key of the computer alone restores the document, or any text derived from it. Word, Excel,
PowerPoint, PDF, text, CSV, Markdown, HTML, XML, JSON and YAML files keep their layout; a PDF keeps its
pages, its text drawn again in place.

The places that a tag would spoil take the value in its protected form without the tag, its
substitute: `Xoydoeh Nuggwdfsorn` for `Jean Dupont`, a date shifted in the form of the original. They
are the name of the file (`Contrat Jean Dupont.docx` becomes `Contrat Xoydoeh Nuggwdfsorn.docx`), the
names of the entries and folders of an archive, and the properties that the windows of properties show:
title, subject, authors, keywords, description, category, company, manager and custom properties of
Word, Excel and PowerPoint, title, author, subject, keywords and creator of a PDF. A substitute holds no
character that Windows refuses in a name. The database records each substitute with its tag, never with
its value in clear; a restoration gives back the original name and the original properties, those of a
copy edited since included, as long as the substitutes stay in place.

### Without tags

The values can also be written without their tags, their substitutes alone, as the names already are:
the file then reads as an ordinary document, for a prompt sent to an assistant, test or training data,
or a file sent outside, but nothing in it tells a value from its substitute any more, and its record
alone gives its values back. The setting Write the tags by default, at the top of the Settings card
of the web application (`tags`, on by default, kept with the other settings when the service starts
again), decides for the drive and for the requests of the interface that do not say it (`tags=false`
or `true`); the switches of the cards Protect files and Protect a text start from it, and each applies to
its files or to its text alone, which tries the interface live; the commands of the Explorer offer
both.

- The record of a file protected without tags keeps its original, as any record, and the substitutes of
  its text with the tags they stand for, never its values in clear: its copies give back the original,
  and a copy edited since, known by its NTFS object identifier, its values from those substitutes,
  without its pictures. The list of the originals kept says which files were protected without tags.
- Once its original is forgotten, at the request of a user, at the end of its time or to make room, its
  values cannot come back: Restore refuses the file (HTTP 410, code `tagless_forgotten`) and tells the
  date of the forgetting, and Forget says so before it forgets. A file protected without tags and
  without record (Keep the originals unchecked) never comes back, which its report says
  (`"restorable": false`).
- A text protected without tags in the web application keeps the list of its substitutes in the page,
  which gives back the text, or the answer of an assistant that quotes them, while the page stays open;
  the interface gives that list in `substitutions`, which `/v1/text/restore` takes back.
- A protected file protected again, by a newer version that finds more values in it for instance, keeps
  the values protected before: with tags, as they are, the substitutes that its record keeps included,
  which its new record inherits; without tags, its tags give way to the substitutes of their values,
  which its new record keeps. A file of the drive protected without tags and edited there is protected
  again the same way. Its restoration gives back the values of both protections, with or without tags:
  the original of each record of the chain that still keeps it, the values of the tags and of the
  inherited substitutes of the others, without any warning.

### The kinds of files

A file is processed when its extension names one of the kinds of Pseudalis and its content is of that
kind, or, of any other extension or without one, when its content is text through and through:

| Kind | Extensions |
|---|---|
| Word, Excel, PowerPoint | `.docx` `.docm` `.dotx` `.dotm` `.xlsx` `.xlsm` `.xltx` `.xltm` `.pptx` `.pptm` `.potx` `.potm` `.ppsx` `.ppsm` |
| PDF | `.pdf` |
| Texts | `.txt` `.text` `.log` `.md` `.markdown` `.rst` `.csv` `.tsv` `.tab` `.json` `.jsonl` `.ndjson` `.xml` `.html` `.htm` `.xhtml` `.svg` `.yaml` `.yml` `.toml` `.ini` `.cfg` `.conf` `.properties` `.sql` `.tex` `.srt` `.vtt` |
| Pictures | `.png` `.jpg` `.jpeg` `.jpe` `.jfif` `.gif` `.bmp` `.tif` `.tiff` `.webp` |
| Archives | `.zip` `.tar` `.tgz` `.tar.gz` |

A file of any other extension, or without extension, `journal.log2` or `README` for instance, is
processed as a text when its content is text through and through, which the engine checks with the
detection of dlib (`dlib/data_io`): no binary signature (pictures, sounds, videos, programs, archives,
PDF), the proportion of printable characters and the entropy of a text over its start and over its whole
content, every character decoded, in UTF-8, UTF-16 or Windows-1252, with no control character but the
tabulation, the ends of lines and the form feed, and no encoded part that would keep values from the
detector. An encoded part is a block of base64, base64url or hexadecimal digits (1024 characters in a
row, or 16 runs in a row of 60 or more, the lines of a block, in a string of JSON or of a program too,
which a line of dashes or a path does not make), data or JSON in base64 (the tokens of OpenID Connect,
`eyJ...`), bytes written in hexadecimal, a dump or an array (64 in a row, or 4 lines in a row of 8 or
more, with letters or marks among them, `4a 65`, `0x4a`, `\x4a`, the groups of 4 or 8 digits of xxd or
od counting in the lines that open with an offset), or a format that writes its values encoded: a
message with parts in base64, quoted-printable or uuencode, or with encoded words in its headers
(`=?UTF-8?B?...?=`), an RTF document, a vCard or an iCalendar, LDIF with values in base64 (`cn:: ...`),
a block armored by PEM or OpenPGP (`-----BEGIN ...-----` and its body), in a string too, a block of
uuencode or of ASCII85, and, by its first line, a PostScript program or an export of the registry of
Windows. The interface refuses such a text with the code `encoded_content`. Without extension, a
picture, an Office document, an archive or a PDF is also taken by its signature. A file of an extension
of the table keeps its rules: its content must be of the kind of its extension, and a text is read as it
lies. For a file of an extension of no kind, the drive and the commands read the first 64 KB first, and
leave a file whose start is not a text without reading the rest, unless the drive holds a protected file
of that size, which may be one renamed.

Any other file is of another kind, and so is a file whose content is not the kind its name gives: an
OpenDocument, a document of an older Office format, a program, a text named `.docx`. Its content would
come out half protected, compressed or encoded as it lies, so Pseudalis never writes it in an anonymized
place: the drive moves it out, the commands leave it as it is, an anonymized archive or copy of a folder
leaves it out, and the interface refuses it with the code `unsupported` or `kind_mismatch`. A ZIP
archive named `.zip` is an archive whatever it holds, an Office document renamed included.

A folder goes through whole, its subfolders and their files included, on the drive, through the
commands of the Explorer, and through the web application, which sends it as a ZIP archive and gives
back the anonymized archive. The archives and the folders keep their tree: their documents are
anonymized, their pictures become text files, the names of their files and of their folders take
substitutes, the same in all the entries that share a folder, and their files of another kind are left
out and listed. A restoration keeps every file it cannot read as it is, under its name restored, and
reads the tags of a text under any name.

### The drive

The installer gives the drive a letter chosen among the free ones, `A:` when it is free. It is the
folder `drive` of the data folder, which the service shows under its letter for every session of the
computer, as the command `subst` would, and watches: it works as any folder, from the Explorer and from
the prompt (`dir`, `copy`, `xcopy`, `move`, `del`, `ren`, `md`, `rd /s`, `type`, any editor), with one
difference: a file is anonymized once its writer closed it and it rested two seconds, its name
included. The anonymized file takes its place, the original goes to the encrypted vault, and the file
written is deleted.

- A document keeps its kind and its layout, its values replaced by tags in its content, or by their
  substitutes alone when the setting Write the tags by default is off, and by substitutes in its name and
  its properties.
- A picture becomes `<name>.txt`, its marker `[Image@<id> : <description>]`, or `[Image@<id>]` when no
  description is available, and the picture goes to the vault.
- A ZIP, TAR or TAR.GZ archive keeps its kind, the values of its name replaced; its documents are
  anonymized, its pictures become text files, the names of its entries and folders take substitutes,
  and its other entries are left out, which its report says.
- A folder that comes in takes its name, its values replaced, once nothing changed under it for six
  seconds: a copy in progress keeps its folders.
- A file of another kind, or whose content cannot be read, a damaged document for instance, leaves the
  drive for the folder of the files not processed of the user who wrote it (`refused\<user>` in the
  data folder, which only that user, SYSTEM and the administrators read), at the place it had in the
  tree of the drive, and the icon says it; its menu opens that folder.
- A file that a program still holds, a document open in its editor for instance, waits until the
  program closes it: a document edited on the drive is anonymized again when its editor closes it, and
  its version before leaves the list of the files of the drive. While the service processes a file, no
  other program opens it (sharing violation), a few seconds for most documents. The lock and temporary
  files of the editors, whose names start with `~` or end with `.tmp`, are left as they are.
- Deleting a file, a folder or the whole tree removes them from the drive. The original of a file
  deleted or replaced by a new version stays restorable from the copies of the file for 45 days (the
  setting `removed_days`, 0 for no limit of time) once the last copy of the drive left it, then leaves
  the vault with the pictures no other file shows. A file moved, renamed or copied within the drive keeps
  its original, which all its copies share.
- Whatever their time, the originals kept, those of the drive and of every other source, stay within
  10 GB (the setting `history_mb`, 10240 MB, never below `max_file_mb`): the originals first kept are
  forgotten to make room, the oldest first, before a new one is written, every hour and at once when the
  settings lower the days or the space. A new original also leaves 256 MB free on the disk of the data
  folder: the oldest make room for it when forgetting them gives that room, and stay when other files
  fill the disk. An installation updated from version 1.0.1 keeps every original: its settings, without
  `history_mb`, take the largest space (1048576 MB) and the 30 days of its default become 45, until an
  administrator sets them; a damaged `settings.json` keeps them the same way.
- A file extracted on the drive from an anonymized archive that the service recorded stays as it is, and
  the list of the originals kept shows it under its original name, its original being the entry of the
  original archive, which the vault keeps apart: the archive and each file extracted from it are
  restored and forgotten alone. A file extracted elsewhere is restored from the original of the archive
  while the service keeps it.
- An anonymized file whose original was forgotten, or left the vault once its time was over, stays as it
  is when it comes back to the drive, unrecorded: it is not anonymized again, which would keep the
  anonymized file itself as a new original. Its restoration puts back the values of its tags, as for
  the files extracted from such an archive. A document edited on the drive and anonymized again, or
  written elsewhere from anonymized values and copied to the drive, keeps its name in clear in the list
  of the originals, and its original comes back with the values of its tags, without the markers of
  its pictures.

The drive never gives back an original: the originals come back through Restore in the Explorer, the
web application and the interface, from any copy of the anonymized file.

### The commands of the Explorer

A right click on a file of one of the extensions above, or on a folder, shows, under Show more options
on Windows 11, a menu Pseudalis with six commands, which `pseudalis.exe` carries out through the
service; a text of another extension goes through the commands of its folder, the drive, the web
application or the interface, the menu staying off the programs, sounds and videos:

| Command | On a file | On a folder |
|---|---|---|
| Anonymize with tags, Anonymize without tags | the file is replaced by its anonymized version; a picture by its text file | each file, then each folder under its anonymized name |
| Anonymize as a copy, with tags or without tags | `name (anonymized).ext` is written beside the file | `name (anonymized)` receives the tree anonymized |
| Restore the original | the original takes the place of the file, under its original name | each file and each folder takes back its original |
| Restore as a copy | `<original name> (restored).ext` is written beside the file | `<original name> (restored)` receives the tree restored |

The commands without tags write the values as their substitutes alone (see Without tags above). A
selection of up to 100 files or folders takes the commands, which the Explorer starts once for each:
they take their turn, two at a time, and the icon notifies together the results that come within a
second and a half.

The suffixes follow the language of the installation, and ` (2)` is added to a name already taken. In a
folder, a file of another kind stays as it is in its place, out of an anonymized copy, and as it is in a
restored copy, which also takes the files that hold nothing to restore; a file anonymized already is
counted as such. The symbolic links and the junctions are passed, the hidden files of the system and the
lock files of the editors (`~$...`) stay as they are, and a folder stops at 100 000 files or 64 levels.
One notification tells the files processed, those anonymized already, those of another kind, named, and
the first failure. The original of an anonymized file is found by the SHA-256 of its content, and by the
NTFS object identifier of the file, which the command records with its place; a file edited since its
anonymization gets its values back from its tags, and its name and its properties from the substitutes
the database recorded. A file whose original was forgotten, from the web application, at the end of
its time once its file left the drive or to make room for newer originals, and a file written elsewhere
from anonymized values, a document that a language model wrote from an anonymized prompt for instance,
are restored from their tags the same way, as long as the key stays: Restore tells the date of the
forgetting. Such a restoration cannot give a picture back: the marker of a picture leaves the document,
its runs in Word, its gray text box on a slide, its text over the gray box of a PDF, which stays, and
the text file of a picture leaves an archive; Restore counts them. The text file of a picture alone
gives back its picture while the vault keeps it, and is refused once it was forgotten (HTTP 410, code
`picture_forgotten`). A file saved by another
program while a command processes it stays as it is, and the command says so. A file of the drive is
anonymized already: Restore asks where to save its original, outside the drive, which would anonymize it
again, and the folder where the originals of a folder of the drive go. The same program runs from a
prompt, `pseudalis protect [--copy] [--no-tags] <file or folder>...` and
`pseudalis restore [--copy] <file or folder>...`, its results printed there (`start /wait pseudalis ...`
waits for them).

### The web application and the interface

The icon opens the web application of the service, `https://localhost:8444`: files dropped there are
anonymized or restored and downloaded back, a folder dropped or chosen there as the ZIP archive of its
tree, texts are anonymized or restored in place, their tags highlighted and their values restored
underlined in the color of their kind, or analyzed, the values found listed below them, and, opened from
the icon, it lists the originals kept (download, forget), exports and imports the key, and changes the
settings, the imposed terms and the imposed columns. Its options are switches. Its files and its texts take their tags or not
as the switch of their card says, which starts from the setting Write the tags by default, and it asks
and tells in a window of its own, over the page, Escape or a click beside it cancelling. The interface it
uses is described with Swagger UI at `https://localhost:8444/docs/` (Interface documentation in the menu
of the icon), without account: the operations of administration take the token of the data folder in the
header `X-Pseudalis-Admin`, which the pages opened by the icon receive in the fragment of their address
and use by themselves.

| Route | Purpose |
|---|---|
| `GET /v1/health`, `GET /v1/info` | the state of the service, of its engine, of the drive and of Albireo |
| `POST /v1/anonymize?name=&record=&tags=&format=` | anonymizes the file of the body, without tags with `tags=false`; the answer is the file, or JSON with `format=json` |
| `POST /v1/restore?name=&object_id=&format=` | restores the file of the body: its original, its picture or its tags |
| `GET /v1/lookup?sha256=&object_id=` | whether a file is an anonymized file of this computer |
| `POST /v1/records/{id}/location` | records where an anonymized file was written (token) |
| `POST /v1/text/anonymize`, `/restore`, `/detect` | the same protection on a text, `{"text": ...}`, `"tags": false` without tags, whose `substitutions` `/restore` takes back; the places of the tags written (`tags`) or of the values restored (`restored`) come back with it |
| `POST /v1/names/anonymize`, `/restore` | a name alone, the one of a folder, `{"name": ...}`, its substitutes recorded |
| `GET /v1/drive`, `GET /v1/events?since=` | the drive, the files it protected or refused (token for the events) |
| `GET /v1/albireo` | whether Albireo describes the pictures |
| `GET /v1/records`, `GET /v1/records/{id}/original`, `DELETE /v1/records/{id}` | the originals kept (token) |
| `GET /v1/settings`, `PUT /v1/settings`, `POST /v1/restart` | the settings and the restart (token) |
| `GET /v1/terms`, `PUT /v1/terms` | the imposed terms (token) |
| `GET /v1/columns`, `PUT /v1/columns` | the imposed columns of the CSV files (token) |
| `GET /v1/updates`, `POST /v1/updates/check`, `POST /v1/updates/apply` | the updates |
| `POST /v1/key/export`, `POST /v1/key/import` | the key, protected by a passphrase (token) |
| `GET /v1/logs` | the last lines of the log (token) |

### The icon

The icon in the notification area follows the service: a click opens the web application, once for a
double click, its tooltip gives the version, the drive and the number of originals kept, and it notifies
the files the drive anonymized or did not process, the results of the commands of the Explorer, the
updates and a service that stops answering. It tells the Explorer of the drive that comes and goes. Its
menu opens Pseudalis (the web application), the drive and the folder of the files not processed of its
user, checks for updates and installs them, manages the service, opens the documentation of the interface
and the About window. Service management restarts the service, and stops it while it runs or starts it
while it is stopped: a service stopped there stays stopped, without the warning of the icon, until the
computer starts again, its start with the system unchanged, or until a user starts it.

### Imposed terms

The administrators give up to 100 terms that Pseudalis replaces in every file, text and name before
anything else is read: the name of a client, of a project, of a contract or of an organisation that the
detector does not know, each with the tag of its type (ORG, PERSON, LOCATION, ADDRESS, ID, EMAIL, PHONE
or URL) and, when asked, its case. A term is found as whole words, its words and numbers in the same
order, separated by at most two characters of one line ("Projet Alpha", "PROJET-ALPHA"); without case
sensitivity, its case and its accents are ignored. Its occurrences outrank every other value found: a
name or an organisation read over a term keeps its parts outside it, while a term inside an e-mail
address, a web address, a date or a number stays protected with that value. The web application edits
the list in its Imposed terms card, the interface through `GET` and `PUT /v1/terms`; the service keeps it
in its database, which only SYSTEM and the administrators read, and the engine applies a new list at
once, to the files processed afterwards.

### Imposed columns

A CSV file with a header holds its values in columns, the names of the clients or their e-mail addresses
in an export for instance. The administrators name up to 100 such columns, each with the type of all its
values (PERSON, ORG, LOCATION, ADDRESS, EMAIL, PHONE, ID, DATE, IBAN, CARD, NIR, SIRET, URL, IP, MAC,
PLATE, GEO, TITLE, GENDER or NATIONALITY). Every value below the name of such a column takes its
substitute, without detection and without tag, whatever the tags of the rest of the file; an empty cell
stays empty, and the spaces around a value stay in place. Such files are typically the exports of a
production database (PostgreSQL, Oracle, MySQL), protected before they are imported into a program under
development or under acceptance tests: a value takes the same substitute wherever an imposed column of
its type holds it, in the file and in the other exports, so that the joins between the tables hold, and
the record gives the originals back.

- The header is found by these names: it is the first of the first 16 rows that names an imposed column
  in a cell, compared without regard to case and accents, the spaces at its ends left out (`Prénom` names
  the columns `PRENOM` and ` prénom `). A title may stand above it, and a first line `sep=;` written by
  Excel gives the delimiter. A file of one column is read too. A text of another extension, or without
  extension, is read as a CSV file when its header holds the delimiter and the two records that follow it
  hold it as many times; a text that names a column in a sentence stays a text.
- An IP column takes the IPv4 and the IPv6 addresses alike, mixed in its rows as the source wrote them:
  each address takes the substitute of its own form, and what surrounds it stays, a prefix length
  (`10.0.0.0/8`), the brackets and the port (`[2001:db8::1]:443`, `192.0.2.1:8080`) or a zone
  (`fe80::1%eth0`); the restoration gives each value back as it was written.
- The values of these columns are learned as values of their types: a client named in a comment of the
  same file takes the tag of the value of its column, whose body is that substitute. The other columns
  are read as before, their names typing them when they are labels (`ville`, `telephone`).
- The record of the file keeps each substitute with the tag it stands for, as for a file protected
  without tags: its copies give back the original, a copy edited on the drive its values, while the
  record lasts. Without record, the report says that they cannot come back (`"restorable": false`).
- A file in Windows-1252 whose accented letters all stand in imposed columns is plain ASCII once
  protected. Its record keeps its encoding, so that its restoration writes it in Windows-1252 again, from
  a copy and from a copy edited on the drive too.
- The web application edits the list in its Imposed columns card, the interface through `GET` and
  `PUT /v1/columns`; the service keeps it in its database, which only SYSTEM and the administrators read,
  and the engine applies a new list at once, to the files processed afterwards. The sheets of the
  workbooks, and the other kinds of files, are read as before.

### Pictures and Albireo

When Albireo, the local inference service, is installed on the same computer and one of its models reads
images, Pseudalis finds it through its interface (its data folder, its port, its local authority) and
asks it to describe each picture, in the language of the installation, naming no person and copying no
name, address, number or identifier the picture shows. A description may still name what the picture
shows: the engine reads it with the detector before it writes it, with what the text of the document
taught the detector besides, a name that the description repeats bare for instance, its values replaced
by tags and counted in the report with those of the file, so that it brings no personal data back into
the anonymized file; only then is it cut at 1000 bytes, outside its tags. A picture alone becomes
`<name>.txt` holding `[Image@<id> : <description>]`, and a picture inside a document gives the same text
in its place (a run of Word, a text box of PowerPoint at the place of the picture, the caption of a gray
box in a PDF). The identifier is a digest of the picture keyed by the key of the installation, which
tells nothing of the picture to whoever does not hold the key and links no picture from one installation
to another; it is recorded in the database with the picture and its description in the vault, both
encrypted, so that restoring the text file gives the picture back and a picture met again keeps its
description, protected again each time it is written. The setting `albireo` turns the descriptions off,
those kept included.

### Languages

The installer, the icon, the commands of the Explorer and the web application speak English, French,
German, Italian and Spanish, and English for any other language. They follow the language of the system:
the icon and the commands of each user take the language of that user's Windows, the web application the
first language of the browser that it speaks, without a menu of its own, and the installer the language
of Windows, which its first page changes. The documentation of the interface stays in English. The
setting `language`, `auto` by default, sets one language for the icon, the commands and the descriptions
of pictures of every user instead; a language chosen in the installer other than the one of Windows
becomes that setting. The web application keeps the language of the browser. The errors of the service
reach the web application and the windows as codes, written in their language.

## Programs and data

| File | Role |
|---|---|
| `pseudalis-service.exe` | the service (LocalSystem): the engine, the drive, the vault, the database, the interface on `127.0.0.1` |
| `pseudalis-engine.dll`, `pseudalis-engine-avx2.dll` | the engine, for every x64 processor and for those with AVX2, chosen at the start of the service |
| `pseudalis-tray.exe` | the icon, started at the sign in of each user |
| `pseudalis.exe` | the commands of the Explorer |
| `pseudalis-setup.exe` | the installer, kept in the folder of the programs as the uninstaller |
| `web\` | the web application and the documentation of the interface |
| `licenses\` | the licenses of the components |

The data lie in `%ProgramData%\Pseudalis`, or in the folder chosen at the installation, recorded in
`HKLM\Software\Pseudalis`: `settings.json`, `admin.token`, `keys\master.key`, `vault\`, `db\pseudalis.db`,
`certificates\`, `logs\`, `updates\`, the folder of the drive, `drive\`, and the folder of the files not
processed, `refused\`.

## Security

- The master key, 32 random bytes, is encrypted with DPAPI for the account of the service (LocalSystem)
  in `keys\master.key`, whose permissions admit SYSTEM and the administrators alone. HKDF-SHA-256
  derives from it the key of the pseudonymization and the two keys of the vault.
- Each original, each picture and each description of a picture by Albireo is kept in the vault
  encrypted with AES-256-CTR and authenticated with HMAC-SHA-256; the database holds the names, the
  sizes, the hashes and the places of the files, never their content nor a description: those that a
  database of an earlier build kept in clear are erased when the service opens it.
- The programs and the engines link the runtime of the compiler statically and import the libraries of
  Windows alone, which the build checks. Built with Visual C++, the service, the icon, the commands and
  the installer take the libraries they import from System32 alone (`/DEPENDENTLOADFLAG`), never from
  the folder of the program, a folder of downloads for the installer for instance, and their later loads
  from System32, and from its own folder for the service (`SetDefaultDllDirectories()`). The service
  loads only the engine of its own build, by its full path, in the folder of the programs, whose
  permissions admit SYSTEM and the administrators alone for any change: a library of the same name that
  another installation placed elsewhere is never loaded.
- The key can be exported, protected by a passphrase of 12 characters or more (PBKDF2-SHA-256 with
  600 000 iterations, AES-256-CTR and HMAC-SHA-256), and imported into a new installation that keeps no
  original yet. Without its key, an anonymized document cannot be restored anywhere: the export is the
  backup of the installation.
- The interface listens on the loopback address only, over HTTPS with a certificate of the local
  authority "Pseudalis Local Authority", restricted by name constraints to the local names and
  addresses, whose key is never written and whose server key is encrypted with DPAPI. Requests whose
  `Host` is not local, and requests of another origin, are refused, so that no web page reads or changes
  anything through the browser.
- The service serves several sessions and users at once: each request is processed on a thread of its
  own, the files of the drive on up to four threads of their own, the engine runs its recognizers for
  several files at once, and the locks of the service are taken in one order, without wait on the
  network or on a client while held.
  The files and texts of the interface processed at once are bounded by the processors and the memory
  (half the processors, 2 to 8): a request that finds no free slot within a minute gets 503 and
  `Retry-After`. A body larger than its route takes is refused before it is read (413).
- An archive or an Office document may decompress 64 times its size at most, from 16 MB to 2 GB, all
  levels together, and each part of an Office document 32 times the document at most, from 32 MB: a
  small file that would hold gigabytes of text is refused. A text is analyzed in pieces of 64 KB, so
  that its memory stays bounded whatever its length.
- The trust model is the computer: every user of the computer shares its key, its originals and its
  administration. Every program of the users reaches the interface, and the token of administration
  lies in the data folder, readable by the interactive users: a user of the computer can restore any
  anonymized file, read the records and export the key. People who must not see each other's documents
  need an installation each, on computers of their own.
- The folder of the drive gives its files to the interactive users, SYSTEM and the administrators; the
  users read, write and delete there, and change no permission, the owners of the files included. The
  folder of the files not processed of a user, and each file moved there, admit that user, SYSTEM and
  the administrators alone.
- The service opens every file of the drive alone, through no link, checks where its handle lies, and
  deletes or moves it through that handle; it neither follows nor processes a link, a junction or a
  mount point placed in the drive, so that no user leads it elsewhere with its rights.
- The icon and the commands reach the service on 127.0.0.1 alone, and open its pages at
  `https://127.0.0.1:8444`: `localhost` may resolve to `::1` first, where another account could
  listen.
- Albireo describes the pictures only when its data folder is the one its installer recorded in
  `HKLM\Software\Albireo`, and when that folder, its certificates, its authority and its settings
  belong to SYSTEM or the administrators: any user can create a folder in `%ProgramData%`, which would
  otherwise receive the pictures of every user.
- The messages, the answers and the logs of the updates never name the places of the publication; the
  settings name them only when they differ from those of the build.

## Choices made from the specification

- The drive needs no driver. A drive whose reads and writes reach a program needs a driver of the
  kernel of Windows, which Windows 10 and 11 load only once Microsoft signed it (an EV certificate and
  an attestation for each version). Without driver, the drive is a folder that the service watches: the
  file a program writes lies in the folder of the drive in clear, from its last write to its
  processing, a few seconds for most documents, under the permissions of the drive, before its original
  goes encrypted to the vault and the file is deleted, its sectors freed but not overwritten; BitLocker,
  where it is on, keeps them encrypted on the disk. A file of another kind cannot be refused at its
  creation: it leaves the drive for the folder of the files not processed of its user.
- Windows copies a file out of a drive, then deletes it: a move out of the drive cannot be told from a
  copy followed by a deletion. The drive therefore never gives back an original, which would otherwise
  reach any program that reads it; the originals come back through Restore, the web application or the
  interface, from any copy of the anonymized file, and no second drive restores them, since it would
  give them to any program as well. For the same reason, the original of a file that leaves the drive
  stays in the vault for `removed_days` days, 45 by default, which the records of the web application
  show, then goes, or at once when it is forgotten there; with 0 days it stays without limit of time.
  Gone with its file, it would leave the copy restorable from its tags alone, without its pictures and
  the parts the protection removed. The space of the originals is bounded all the same, `history_mb`
  and the free space of the disk: when it is full, the oldest originals go first, whatever their time.
  Once an original is forgotten, Restore gives its protected file back from its tags alone and tells the
  date of the forgetting: the database keeps the SHA-256 and the object identifier of the file, never
  its content.
- The names and the properties take substitutes rather than tags, which a name cannot hold and a window
  of properties would show as they are; the database keeps their tags. A substitute is found again in a
  name or a property edited since as long as it keeps its form, a run of up to twelve words; one of four
  characters or fewer, a year for instance, is too common to be read back there, and comes back only
  with the original kept in the vault.
- The record of a file protected without tags keeps the substitutes of its text with their tags rather
  than their places in the file: the places of an Office document or of a PDF lie in its compressed
  parts, which give back nothing more than the original that its exact copies receive, while the
  substitutes, each a distinct code with its length, are found again in a copy edited since, and in the
  original of a file protected again, whose new record inherits them.
- A picture becomes a text file of the same name with `.txt` added, which the Explorer and the prompt
  show as text; its picture comes back through Restore, the web application and the interface.
- A file of another kind is never written in an anonymized place, since its content would come out
  half protected: the drive moves it out, keeping its place in the tree, and an anonymized archive or
  copy of a folder leaves it out and names it. The kinds are those whose whole content Pseudalis reads.
- On Windows 11, the commands of the Explorer lie under Show more options: the first menu of Windows 11
  only takes the commands of packaged applications.
- The service listens on port 8444, beside Albireo on 8443, with the same stack: dlib, cpp-httplib and
  Mbed TLS.
- A PDF always keeps its pages: the text is drawn again in place, the option of a PDF rebuilt as text
  alone is gone.

## Installing

`pseudalis-setup.exe` installs the programs, the service, the icon, the commands of the Explorer, the
shortcuts and the entry of the installed applications of Windows; started again over an installation,
it updates it and keeps the data, the key, the drive and the options. The wizard asks for the
language, the license, the folders, the letter of the drive and the options. Without window:

```
pseudalis-setup /silent [/dir <folder>] [/data-dir <folder>] [/drive <letter>|none] [/port <port>] [/no-context-menu]
pseudalis-setup /uninstall [/silent] [/purge]
pseudalis-setup /service start|stop|restart
```

The uninstallation removes the programs and keeps the key, the originals, the files of the drive and
the settings for a later installation, unless the option of the wizard is unticked or `/purge` is given:
the files of the drive go with the data folder, and the documents anonymized with that key can then no
longer be restored. A tool of deployment that uninstalls before it installs again
(`QuietUninstallString`) therefore never takes them. The uninstaller left in the folder of the
programs is a copy of the very program that installed them, held from its start so that no other
account replaces it meanwhile.

## Building

### Windows

With Visual Studio 2022 or its Build Tools (C++ workload, CMake 3.21 or later), from the root of the
copy of the repository of the production chain, `%USERPROFILE%\work\Dlib-Transformer-extensions`, the
only one the script builds and the publication reads (`-AnyFolder` builds another copy, which is not
published):

```
powershell -ExecutionPolicy Bypass -File dlib\dlib\external\pseudalis\tools\build_windows.ps1
```

The script builds every program into `build-pseudalis\bin`; the installer,
`build-pseudalis\bin\pseudalis-setup.exe`, carries them all, with the runtime linked statically. The
engines compile the models of the header
[`examples/slm_document_anonymizer_ex.h`](../../../examples/slm_document_anonymizer_ex.h)
(`PSEUDALIS_MODELS_HEADER`). `-Tests` also builds and runs the tests.

The build checks, before it packs the installer and once it is linked, that each program and each engine
imports the libraries of Windows alone, found in System32, or its API sets
([`tools/check_imports.cpp`](tools/check_imports.cpp)): a runtime of the compiler (`vcruntime`, `msvcp`,
`ucrtbase`, the sets `api-ms-win-crt`) or any other library stops the build, so that the installer
carries no library of a third party and the programs run on Windows as it is installed. Each
configuration of the build tree gives the programs and the engines a build stamp of their own
(`psd_build()` of the engines): the service loads its engine by its full path, beside it in the folder
of the programs, and refuses an engine of another version or of another build.

### Linux

The service builds on Linux with its engine inside and its drive as a folder without letter, for the
tests of the engine, of the host, of the interface and of the drive:

```
bash dlib/dlib/external/pseudalis/tools/test_linux.sh
```

### Cross compilation

```
cmake -S dlib/dlib/external/pseudalis -B build-pseudalis-win -DCMAKE_TOOLCHAIN_FILE=dlib/dlib/external/pseudalis/cmake/mingw-w64.cmake
cmake --build build-pseudalis-win
```

A cross build checks the libraries of the programs only when it can run a program of Windows:
`-DCMAKE_CROSSCOMPILING_EMULATOR=<wine>` gives it one; without it, the configuration says that the check
is left out.

## Publishing a version

A version is published on GitHub alone, in [`dist/pseudalis`](../../../../dist/pseudalis): its
installer, `pseudalis-setup-<version>.exe`, which anyone downloads there, and the update feed that names
it, with its signature (ECDSA P-256), which the installed copies read there. The programs check the
signature with the public key of `resources/feed-public-key.pem`, built into them. The publisher makes
the pair once, on Linux, the private key in a folder of its own outside the repository, readable by its
owner alone, and commits the public key before the installer is built; the copies installed before trust
the former key alone, and are installed again by hand. A copy of the private key kept apart is the only
way to publish an update that the installed copies take.

```
python3 dlib/dlib/external/pseudalis/tools/publish_release.py --create-key --key ~/keys/pseudalis-feed-signing.key
```

Each version goes through three steps, each on the system that holds what it needs, each stopping at its
first failure:

1. Linux: the sources change and are tested there (`tools/test_linux.sh`), the key is checked, then the
   sources are committed and pushed. `PSEUDALIS_VERSION` of [`common/version.h`](common/version.h)
   increases for each version that the installed copies are to receive: a feed that announces the
   version a copy holds installs nothing.
2. Windows, in the copy of the production chain, `%USERPROFILE%\work\Dlib-Transformer-extensions`:
   the installer is built from the sources pushed, tested and installed in place
   (`tools/build_windows.ps1 -Tests -Install`), which prints its size, its SHA-256 and its path under WSL.
3. Linux, where the private key lies: the installer of that copy, which Linux reads under
   `/mnt/c/Users/<user>/work`, goes into `dist/pseudalis` with the feed written and signed for it, then
   both are committed and pushed, and the publication ends once GitHub serves them.

```
bash dlib/dlib/external/pseudalis/tools/test_linux.sh &&
python3 dlib/dlib/external/pseudalis/tools/publish_release.py --check-key --key ~/keys/pseudalis-feed-signing.key &&
git add dlib/dlib/external/pseudalis && git commit -m "<message>" -- dlib/dlib/external/pseudalis && git push
```

```
Set-Location $env:USERPROFILE\work\Dlib-Transformer-extensions; git pull --ff-only; if ($LASTEXITCODE -eq 0) { powershell -ExecutionPolicy Bypass -File .\dlib\dlib\external\pseudalis\tools\build_windows.ps1 -Tests -Install }
```

```
git pull --ff-only &&
python3 dlib/dlib/external/pseudalis/tools/publish_release.py --key ~/keys/pseudalis-feed-signing.key --notes notes.json --push
```

Without an installer named, `publish_release.py` takes the one of the copy of the production chain,
`build-pseudalis\bin\pseudalis-setup.exe` under `%USERPROFILE%\work\Dlib-Transformer-extensions`, and it
refuses an installer of another folder unless `--any-folder` is given.

`publish_release.py` checks that the installer carries the version of
[`common/version.h`](common/version.h), that the feed announces no later one, that the installer stays
under the limits of GitHub (it refuses 95 MB and more, and warns from 50 MB) and that the key is the one
of the public key of the sources, copies the installer as `pseudalis-setup-<version>.exe`, writes the
feed with its name, its size, its SHA-256 and the notes of each language
(`{"en": "...", "fr": "...", "de": "...", "it": "...", "es": "..."}`), signs it, checks the signature
with the public key of the sources, then puts the installer, the signature and the feed in place of the
published ones; the installers of the earlier versions leave the folder and stay in the history of the
repository. With `--push`, it first checks that the branch lags behind its remote one in nothing, then
commits and pushes the folder, and ends once GitHub serves the feed and its signature as they were
written and the installer by its size and its SHA-256, as the installed copies will read them; a step
that fails prints the commands that end the publication, and `--check` makes the last check again. A
failure before the writing leaves the published files as they are, and `--dry-run` checks everything
without writing. The services read the feed and its signature on GitHub five minutes after their start,
then every day, check the signature, download the installer beside the feed in the background, check its
size and its SHA-256, and install it at the next start of the service, or at once from the menu of the
icon. Their messages and logs never name the places they read.

## Editions and licenses

- Pseudalis Community is free and open source software under the Boost Software License 1.0, as dlib
  and Dlib-Transformer-extensions are; its components keep their licenses: Mbed TLS (Apache 2.0),
  cpp-httplib (MIT), SQLite (public domain), zlib (zlib License), Swagger UI (Apache 2.0).
- No component of Pseudalis is under a copyleft license, the drive needing no driver: an edition under a
  proprietary license, an edition for companies sold under its own terms for instance, may take every
  component as it is.

## Limitations

- The recognizers read French, English, Italian, Spanish and German; a document in another language
  keeps more values in clear. A protected document should be read again before it is shared.
- A scanned page without text layer is kept as a gray box: no OCR is performed.
- A description of Albireo is read as the text of its document is, with the same limits: a value that
  the detector misses stays in clear in it. The setting `albireo` turns the descriptions off.
- The drive takes files of 150 MB at most by default (`max_file_mb`): a larger file goes to the folder
  of the files not processed. A folder sent by the web application is one archive, under the same limit.
- A file written to the drive lies there in clear until its processing, a few seconds for most
  documents (see the choices above).
- A value written encoded stays as it is: in a text of an extension of the table, whose content is
  read as it lies (a message saved as `.txt`, a key in a `.md`), and in a text of another extension
  when its encoding escapes the rules of the kinds of files (a block of base64 shorter than their
  thresholds, bytes in hexadecimal written with digits alone, the percent-encoding of an address). A
  text that may hold such values should be read again before it is shared.
- A restored archive gives back its pictures as their text files, unless the archive is the one the
  service made: its original then comes back whole from the vault.
- The values of the imposed columns come back from the record of their file alone. A copy of the
  protected CSV file reworked elsewhere, by another program for instance, gives back the values of its
  tags alone, and comes back in UTF-8 when its original was written in Windows-1252 and its protection in
  plain ASCII. A value written in clear in such a column of a file of the drive, which equals a
  substitute of that file, a shifted date for instance, is taken for that substitute.
- A value of an imposed column takes the form that its substitute can keep. In an ID column, a number of
  fewer than six digits and a code of fewer than four letters and digits, in an IP column, a value of no
  IP form and an IPv6 address written otherwise than in the canonical form of RFC 5952 (leading zeros,
  capitals), with its port or its prefix length, take an opaque code of letters; a network keeps its
  prefix length after the substitute of its address, whose bits beyond the prefix are seldom zero. A
  program that checks these values when the protected file is imported, a column of integers or the type
  `cidr` of PostgreSQL for instance, may refuse them; all of them come back.
- The analysis reads some tens of kilobytes per second on each processor when nearly every word is
  personal data: a text file of tens of megabytes takes minutes.
- The protection is a pseudonymization in the sense of the GDPR: the key restores the data, which stay
  personal data for its holder.
