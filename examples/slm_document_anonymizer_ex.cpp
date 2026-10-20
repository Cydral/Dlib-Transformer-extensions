// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This program protects the personal data of documents before they are shared, sent
    to a language model or used outside the organisation that owns them, and restores
    them afterwards. It handles Word, Excel and PowerPoint files (DOCX, XLSX, PPTX and
    their variants), PDF files and text files of every kind (plain text, CSV, XML, HTML,
    JSON, logs), in French and English, and is built entirely on dlib: no external
    library is needed.

    Every personal value found in a document is replaced by a tag such as

        [[PERSON:Xoydoeh Nuggwdfsorn#EKR]]   [[EMAIL:xoydoeh.nuggwdfsorn@qbuavps.fr#TBY]]
        [[PHONE:06 18 94 58 33#XJW]]         [[IBAN:FR36 5430 1462 7126 5447 9327 661#QG3]]
        [[DATE:2027-03-29~0050#584]]         [[IPV4:192.174.112.109#FMM]]

    The keyword gives the nature of the value, so that a reader, a search engine or a
    language model still understands the document. The body is the value itself,
    encrypted with format-preserving encryption (NIST SP 800-38G FF1 over AES-256), with a
    code chosen per type so that the useful operations keep working on protected data:
    the same name always gets the same code, in every document and every format; the
    surname code is shared by "Jean Dupont", "M. Dupont" and jean.dupont@acme.fr; a
    protected IBAN or card number still passes its checksum; dates are all moved into the
    past by the same secret number of weeks, so intervals, ages and weekdays are preserved
    and no date shows up in the future, while a month with its year ("mars 2022") or a year
    that dates a fact ("en 2019") moves by whole months or years; IP addresses are
    encrypted prefix by prefix (Crypto-PAn), so subnets are preserved; GPS coordinates and
    UUIDs keep their layout. The last three characters authenticate the tag with an HMAC.

    No correspondence table is written. The original values are inside the tags, and
    the key file is the only secret needed to restore them, in the protected document or
    in any text derived from it, such as a summary written by a language model. The
    source document is never modified: outputs are new files.

    Pictures, charts, embedded objects and macros are removed from Office documents, and
    PDF files are rebuilt as text only, since none of those can be checked for personal
    data here.

    FINDING THE PERSONAL DATA

    Four channels work together. Validated scanners find what has a structure: e-mail
    and IP addresses, phone numbers, IBAN, card, social security, SIRET, passport and
    the national numbers of about forty countries, IMEI, VIN, UUID, GPS coordinates,
    plates and dates, each accepted only when its check digits verify or a cue announces
    it. Linguistic rules find names, places, addresses and organisations, public bodies
    included, from titles, field labels, legal forms, nickname cues, the postal codes of
    other countries in their own layout and lexicons (given names of every region of the
    world, nicknames and stage names, organisations of all countries known by their own
    name), and the civility, sex and nationality of a person from their labels and cues.
    Regular expressions given with --patterns add what is specific to an organisation.
    And two learned recognizers, trained here on synthetic data, find the names, places,
    addresses and organisations that the rules miss:

        - a structural SVM sequence labeler (dlib::sequence_labeler) over word tokens and
          their lexical, shape and context features;
        - a small byte level transformer (pii_span_model), which reads the UTF-8 bytes of
          the text, so that spelling variants and OCR damage reach it unchanged, and
          reports, for any string, whether it holds personal data, where it starts and
          ends, and what it is (given name, family name, city...).

    Either one can be loaded, or both: the detector then follows the span model and adds,
    where the span model kept nothing, the confident spans of the tagger and the name
    parts that complete a name already found; where the span model reads an organisation
    with little confidence and the tagger reads its words as the parts of a name, the
    name is kept.

    The recognizers can be compiled into the program, so that the executable travels
    alone. The embed command writes them, compressed and encoded as file_to_code_ex does,
    into slm_document_anonymizer_ex.h; with that header next to this file, the build
    carries them and uses them by default. The repository provides the header with trained
    recognizers, and embed rewrites it after a new training. --tagger and --model each
    replace the compiled model of their kind by a model file, independently of each other,
    and --no-tagger and --no-model leave one out.

    TRYING IT ON A WORD FILE

        ./slm_document_anonymizer_ex keygen --key demo.key
            creates a key of your own; without --key, the first anonymize run creates
            ~/.config/dlib/privacy.key and every command uses it.
        ./slm_document_anonymizer_ex scan contract.docx
            lists the personal data found, by type, and writes nothing.
        ./slm_document_anonymizer_ex anonymize --key demo.key --report contract.docx
            writes contract.anon.docx with the recognizers compiled into the program,
            and contract.anon.docx.report.json: the counts by type and the removed
            content, never a personal value.
        ./slm_document_anonymizer_ex restore --key demo.key contract.anon.docx
            writes contract.restored.docx, every tag replaced by its value again.

    Other uses:

        ./slm_document_anonymizer_ex anonymize payroll.xlsx slides.pptx scan.pdf folder
            protects several files, and every file of a folder, in one run.
        ./slm_document_anonymizer_ex text < prompt.txt > protected.txt
        ./slm_document_anonymizer_ex text --restore < answer.txt
            protect text on its way to a language model, and restore the answer.
        ./slm_document_anonymizer_ex detect < sample.txt
            prints whether the text holds personal data and the spans found: offsets,
            type and, for each learned recognizer, the fine label.

    USING MODELS OF YOUR OWN

        ./slm_document_anonymizer_ex anonymize --model pii_span.dat contract.docx
            the span model of the file with the tagger compiled into the program.
        ./slm_document_anonymizer_ex anonymize --tagger pii_tagger.dat
            --model pii_span.dat contract.docx
            both recognizers read from files.
        ./slm_document_anonymizer_ex anonymize --no-tagger --model pii_span.dat
            contract.docx
            the span model of the file alone; --no-tagger --no-model leaves the rules
            alone.
        ./slm_document_anonymizer_ex embed --tagger pii_tagger.dat --model pii_span.dat
            --out slm_document_anonymizer_ex.h
            writes the header that compiles both recognizers into the program at the
            next build.

    TRAINING AND EVALUATING THE RECOGNIZERS

    The data comes from slm_tools/build_pii_dataset.py, one example per line, read by
    load_pii_dataset():

        python3 slm_tools/build_pii_dataset.py --out-dir pii-data --train 100000
            writes the splits; --abroad-share sets the share of places, organisations,
            phone numbers and IBANs of other countries.
        ./slm_document_anonymizer_ex train-tagger pii-data/train.tsv --dev pii-data/dev.tsv
            --examples 70000 --threads 4 --out pii_tagger.dat
            trains the tagger afresh, in minutes; --cache 5 trains about three times
            faster with about six times the memory.
        ./slm_document_anonymizer_ex train-model pii-data/train.tsv --dev pii-data/dev.tsv
            --init embedded --lr 0.0005 --steps 10000 --out pii_span.dat
            continues the training of the span model compiled into the program; mix
            the new examples with data like the one it learned from, so that it keeps
            what it knows.
        ./slm_document_anonymizer_ex train-model pii-data/train.tsv --dev pii-data/dev.tsv
            --size small --steps 30000 --out pii_span_small.dat
            trains a small span model from scratch, best on a GPU.
        ./slm_document_anonymizer_ex evaluate pii-data/test.tsv --tagger pii_tagger.dat
            --model pii_span.dat
            scores the rules alone, each recognizer alone and the whole detector.

    BUILDING THE PROTECTION COMMANDS ONLY

    Built with -DANONYMIZER_ESSENTIALS_ONLY=1, the program keeps the round trip of
    protection, keygen, anonymize, restore, scan and text, with the recognizers compiled
    into it, and leaves out detect, train-tagger, train-model, evaluate and embed with
    their options: the build given to the people who only protect and restore documents,
    such as a standalone executable for Windows. The default, 0, builds every command.

        g++ -std=c++14 -O2 -DNDEBUG -DDLIB_NO_GUI_SUPPORT -DANONYMIZER_ESSENTIALS_ONLY=1
            -I.. slm_document_anonymizer_ex.cpp ../dlib/all/source.cpp -lpthread
            -o slm_document_anonymizer
        x86_64-w64-mingw32-g++-posix -std=c++14 -O2 -DNDEBUG -DDLIB_NO_GUI_SUPPORT
            -DANONYMIZER_ESSENTIALS_ONLY=1 -I.. slm_document_anonymizer_ex.cpp
            ../dlib/all/source.cpp -static -lws2_32 -lwinmm -o slm_document_anonymizer.exe
            cross-compiles it for 64-bit Windows as one executable that needs no library
            beyond those of the system. Linked with a manifest that sets UTF-8 as the code
            page of the process (activeCodePage), it also opens the files whose names the
            ANSI code page cannot write.

    In the terms of the GDPR, the output is pseudonymized data: whoever holds the key can
    restore it, and it remains personal data for that party. With --irreversible the key
    is random and never stored, which removes the restoration path; whether the result
    qualifies as anonymous data then depends on what the remaining text reveals.

    The exit status is 0 when every file was processed, 2 when a file was skipped or
    failed, or when restoration rejected tags (another key, another context or altered
    tags), and 1 for a usage error. On Windows, paths go to the C runtime as narrow
    strings: file names must be representable in the active code page.
*/

#include <dlib/privacy.h>
#include <dlib/privacy/pii_sequence_tagger.h>
#include <dlib/privacy/pii_span_model.h>
#include <dlib/base64.h>
#include <dlib/cmd_line_parser.h>
#include <dlib/dir_nav.h>
#include <dlib/zip.h>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <dlib/windows_magic.h>
#include <windows.h>
#include <direct.h>
#include <fcntl.h>
#include <io.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <limits.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

/* The trained recognizers can be compiled into this program from the header the embed
   command writes. The header is detected rather than declared, as slm_distill_ex does for
   its student model: without it the program compiles and loads its models from files only;
   with it next to this file, the next build carries the models inside the executable. A
   header created after a previous build is not a dependency of the old object file, so
   rebuild the target after writing it. An external definition of either macro keeps
   priority over the detection. */
#ifndef ANONYMIZER_MODELS_HEADER
#  define ANONYMIZER_MODELS_HEADER "slm_document_anonymizer_ex.h"
#endif
#if !defined(WITH_ANONYMIZER_MODELS) && defined(__has_include)
#  if __has_include(ANONYMIZER_MODELS_HEADER)
#    define WITH_ANONYMIZER_MODELS 1
#  endif
#endif
#ifdef WITH_ANONYMIZER_MODELS
#  include ANONYMIZER_MODELS_HEADER
#endif

/*  ANONYMIZER_ESSENTIALS_ONLY, 0 by default, builds every command. Set to 1
    (-DANONYMIZER_ESSENTIALS_ONLY=1), it builds a program limited to the round trip of
    protection, for the people who only protect and restore documents: keygen, anonymize,
    restore, scan and text, with the recognizers compiled into it. detect, train-tagger,
    train-model, evaluate and embed are not compiled, nor the training options and the
    options that load, leave out or tune a recognizer. */
#ifndef ANONYMIZER_ESSENTIALS_ONLY
#  define ANONYMIZER_ESSENTIALS_ONLY 0
#endif

using namespace std;
using namespace dlib;

// ----------------------------------------------------------------------------------------

namespace
{
    enum class command { anonymize, restore, scan, text, keygen, detect, train_tagger, train_model, evaluate, embed };

    const char separator =
#if defined(_WIN32)
        '\\';
#else
        '/';
#endif

    string default_key_path()
    {
#if defined(_WIN32)
        const char* base = getenv("APPDATA");
        return string(base ? base : ".") + "\\dlib\\privacy.key";
#else
        const char* xdg = getenv("XDG_CONFIG_HOME");
        const char* home = getenv("HOME");
        const string base = (xdg && *xdg) ? string(xdg) : (home ? string(home) + "/.config" : string("."));
        return base + "/dlib/privacy.key";
#endif
    }

#if defined(_WIN32)
    /* Standard input and output carry bytes unchanged, as on other systems, and the console
       shows UTF-8 while the program runs; it gets its code pages back at the end. */
    struct windows_console
    {
        const UINT input = GetConsoleCP();
        const UINT output = GetConsoleOutputCP();

        windows_console()
        {
            _setmode(_fileno(stdin), _O_BINARY);
            _setmode(_fileno(stdout), _O_BINARY);
            if (input) SetConsoleCP(CP_UTF8);
            if (output) SetConsoleOutputCP(CP_UTF8);
        }

        ~windows_console()
        {
            if (input) SetConsoleCP(input);
            if (output) SetConsoleOutputCP(output);
        }
    };
#endif

    /* The name the program was started under, without its folder and its ".exe". */
    string program_name(const char* argv0)
    {
        string name = argv0 ? argv0 : "";
        const size_t p = name.find_last_of("/\\");
        if (p != string::npos) name = name.substr(p + 1);
        string tail = name.size() > 4 ? name.substr(name.size() - 4) : string();
        for (char& c : tail) c = static_cast<char>(tolower(static_cast<unsigned char>(c)));
        if (tail == ".exe") name.resize(name.size() - 4);
        return name.empty() ? string("slm_document_anonymizer_ex") : name;
    }

    string parent_of(const string& path)
    {
        const size_t p = path.find_last_of("/\\");
        return p == string::npos ? string() : path.substr(0, p);
    }

    /* Creates a directory and its missing parents; the last one is private on POSIX
       systems when it holds a key. */
    void make_directories(const string& dir, bool private_dir)
    {
        if (dir.empty() || file_exists(dir)) return;
        make_directories(parent_of(dir), false);
#if defined(_WIN32)
        // The user profile that holds %APPDATA% is private already.
        (void)private_dir;
        _mkdir(dir.c_str());
#else
        mkdir(dir.c_str(), private_dir ? 0700 : 0777);
#endif
    }

    /* The absolute form of a path, links resolved. For a file that does not exist yet,
       its folder is resolved, or the current folder for a relative path, so that two
       spellings of the same output compare equal. */
    string absolute_path(const string& path)
    {
#if defined(_WIN32)
        char buf[_MAX_PATH];
        if (_fullpath(buf, path.c_str(), _MAX_PATH)) return buf;
        return path;
#else
        char buf[PATH_MAX];
        if (realpath(path.c_str(), buf)) return buf;
        const size_t slash = path.find_last_of('/');
        const string dir = slash == string::npos ? string(".") : (slash == 0 ? string("/") : path.substr(0, slash));
        if (realpath(dir.c_str(), buf)) return string(buf) + "/" + path.substr(slash == string::npos ? 0 : slash + 1);
        if (!path.empty() && path[0] != '/' && getcwd(buf, sizeof(buf))) return string(buf) + "/" + path;
        return path;
#endif
    }

    /* The lines of a list file, without a byte order mark, blank lines and lines starting
       with '#'. With trim, the spaces and tabs around each line go too. */
    std::vector<string> read_list(const string& filename, bool trim = true)
    {
        ifstream in(filename, ios::binary);
        if (!in) throw error("cannot read " + filename);
        std::vector<string> lines;
        string line;
        bool first = true;
        while (getline(in, line))
        {
            if (first && line.compare(0, 3, "\xEF\xBB\xBF") == 0) line.erase(0, 3);
            first = false;
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (trim)
            {
                while (!line.empty() && (line.back() == ' ' || line.back() == '\t')) line.pop_back();
                size_t b = 0;
                while (b < line.size() && (line[b] == ' ' || line[b] == '\t')) ++b;
                line = line.substr(b);
            }
            if (line.find_first_not_of(" \t") == string::npos || line[0] == '#') continue;
            lines.push_back(line);
        }
        return lines;
    }

    string file_name_of(const string& path)
    {
        const size_t p = path.find_last_of("/\\");
        return p == string::npos ? path : path.substr(p + 1);
    }

    bool ends_with(const string& s, const string& suffix)
    {
        return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
    }

    /* contract.docx -> contract.anon.docx -> contract.restored.docx, README -> README.anon
       -> README.restored. Only the last component of a path carries an extension. */
    string output_name(const string& path, command cmd)
    {
        const size_t slash = path.find_last_of("/\\");
        const string dir = slash == string::npos ? string() : path.substr(0, slash + 1);
        const string name = slash == string::npos ? path : path.substr(slash + 1);
        string stem = name, ext;
        if (cmd != command::anonymize && ends_with(name, ".anon"))
        {
            stem = name.substr(0, name.size() - 5);
        }
        else
        {
            const size_t dot = name.find_last_of('.');
            if (dot != string::npos && dot > 0)
            {
                stem = name.substr(0, dot);
                ext = name.substr(dot);
            }
            if (cmd != command::anonymize && ends_with(stem, ".anon")) stem.resize(stem.size() - 5);
        }
        return dir + stem + (cmd == command::anonymize ? ".anon" : ".restored") + ext;
    }

    bool is_report(const string& name)
    {
        return ends_with(name, ".report.json");
    }

    bool is_protected_output(const string& name)
    {
        return (name.find(".anon.") != string::npos || ends_with(name, ".anon")) && !is_report(name);
    }

    bool is_previous_output(const string& name)
    {
        return is_protected_output(name) || name.find(".restored.") != string::npos || ends_with(name, ".restored") ||
               is_report(name);
    }

    struct job
    {
        string input;
        string relative;        // path below the folder given on the command line
    };

    bool is_symbolic_link(const string& path)
    {
#if defined(_WIN32)
        (void)path;
        return false;
#else
        struct stat st;
        return lstat(path.c_str(), &st) == 0 && S_ISLNK(st.st_mode);
#endif
    }

    /* The files of a folder and of its subfolders. A link to a folder is not followed, so
       that a link back to a parent cannot make the walk endless. */
    void collect(const directory& dir, const string& prefix, std::vector<job>& jobs)
    {
        for (const file& f : dir.get_files())
            jobs.push_back(job{ f.full_name(), prefix + f.name() });
        for (const directory& d : dir.get_dirs())
        {
            if (is_symbolic_link(d.full_name()))
            {
                cerr << d.full_name() << ": link to a folder, not followed\n";
                continue;
            }
            collect(d, prefix + d.name() + separator, jobs);
        }
    }

    string summary(const document_report& r)
    {
        ostringstream out;
        if (r.operation == "restore")
        {
            out << r.tags_restored << " values restored";
            if (r.tags_rejected) out << ", " << r.tags_rejected << " tags rejected (altered or from another key)";
            return out.str();
        }
        out << r.total_entities() << " values";
        if (!r.entities.empty())
        {
            out << " (";
            bool first = true;
            for (const auto& e : r.entities)
            {
                out << (first ? "" : ", ") << e.first << " " << e.second;
                first = false;
            }
            out << ")";
        }
        return out.str();
    }

    /* Writes bytes to a new file next to path, then renames it to path, so that path is
       never left half written. The temporary file is created exclusively under a random
       name and without following links, so that nothing planted there is written
       through. */
    void write_atomically(const string& path, const string& bytes)
    {
        const string tmp = path + "." + to_hex(generate_secure_random_bytes(6)) + ".tmp";
#if defined(_WIN32)
        const int fd = _open(tmp.c_str(), _O_WRONLY | _O_CREAT | _O_EXCL | _O_BINARY, _S_IREAD | _S_IWRITE);
#else
        const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0666);
#endif
        if (fd < 0) throw error("cannot write " + tmp);
        size_t done = 0;
        bool ok = true;
        while (ok && done < bytes.size())
        {
            const size_t chunk = std::min<size_t>(bytes.size() - done, 1 << 20);
#if defined(_WIN32)
            const int n = _write(fd, bytes.data() + done, static_cast<unsigned int>(chunk));
#else
            const ssize_t n = ::write(fd, bytes.data() + done, chunk);
#endif
            if (n <= 0) ok = false;
            else done += static_cast<size_t>(n);
        }
#if defined(_WIN32)
        ok = _close(fd) == 0 && ok;
        // rename() does not replace an existing file on Windows; MoveFileEx does.
        ok = ok && MoveFileExA(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
        ok = ::close(fd) == 0 && ok && std::rename(tmp.c_str(), path.c_str()) == 0;
#endif
        if (!ok)
        {
            std::remove(tmp.c_str());
            throw error("cannot write " + path);
        }
    }

#if !ANONYMIZER_ESSENTIALS_ONLY
    string printable(const string& value)
    {
        string out;
        for (char c : value) out += (c == '\n') ? string("\\n") : (c == '\t') ? string("\\t") : string(1, c);
        return out;
    }
#endif

// ----------------------------------------------------------------------------------------
// The regular expression channel

    /* One pattern per line: TYPE <TAB> regular expression [<TAB> options], the options
       being a comma separated list among luhn, iban (a check the match must pass), icase,
       group=N (report capture group N rather than the whole match) and confidence=X. */
    void read_patterns(const string& filename, pii_detector& detector)
    {
        // The lines are not trimmed: a regular expression may end with a space.
        for (const string& line : read_list(filename, false))
        {
            const size_t tab = line.find('\t');
            if (tab == string::npos)
                throw error(filename + ": expected \"TYPE<tab>regex[<tab>options]\": " + line);
            pii_pattern p;
            p.type = pii_type_from_keyword(line.substr(0, tab));
            if (p.type == pii_type::none)
                throw error(filename + ": unknown type in: " + line);
            const size_t tab2 = line.find('\t', tab + 1);
            p.regex = line.substr(tab + 1, tab2 == string::npos ? string::npos : tab2 - tab - 1);
            if (tab2 != string::npos)
            {
                istringstream options(line.substr(tab2 + 1));
                string option;
                while (getline(options, option, ','))
                {
                    try
                    {
                        size_t used = 0;
                        if (option == "luhn" || option == "iban") p.check = option;
                        else if (option == "icase") p.ignore_case = true;
                        else if (option.compare(0, 6, "group=") == 0)
                        {
                            p.group = std::stoul(option.substr(6), &used);
                            if (used != option.size() - 6) throw std::invalid_argument(option);
                        }
                        else if (option.compare(0, 11, "confidence=") == 0)
                        {
                            p.confidence = std::stod(option.substr(11), &used);
                            if (used != option.size() - 11 || p.confidence < 0 || p.confidence > 1)
                                throw std::invalid_argument(option);
                        }
                        else throw error(filename + ": unknown pattern option \"" + option + "\"");
                    }
                    catch (std::logic_error&)
                    {
                        throw error(filename + ": invalid value in the pattern option \"" + option + "\"");
                    }
                }
            }
            detector.add_pattern(p);
        }
    }

// ----------------------------------------------------------------------------------------
// The learned recognizers

    /* The span model sizes this program can load; the size is read from the model file.
       pii_span_model_base is added the same way. */
    struct span_model
    {
        string size;
        std::shared_ptr<pii_span_model_tiny> tiny_net;
        std::shared_ptr<pii_span_model_small> small_net;

        std::vector<pii_span> find(const string& text) const
        {
            return tiny_net ? tiny_net->find(text) : small_net->find(text);
        }
    };

    /* The size of the span model serialized at the start of in: "tiny" or "small". The
       stream is left where it was. */
    string span_model_size_in(std::istream& in, const string& source)
    {
        const std::streampos at = in.tellg();
        string name;
        long d = 0, l = 0, h = 0;
        try
        {
            deserialize(name, in);
            if (name == "pii_span_model")
            {
                deserialize(d, in);
                deserialize(l, in);
                deserialize(h, in);
            }
        }
        catch (serialization_error&) {}
        in.clear();
        in.seekg(at);
        if (name != "pii_span_model") throw error(source + " is not a pii_span_model");
        if (d == 64 && l == 2 && h == 2) return "tiny";
        if (d == 128 && l == 4 && h == 4) return "small";
        throw error(source + ": unsupported model size (" + to_string(d) + ", " + to_string(l) + ", " +
                    to_string(h) + ")");
    }

    span_model read_span_model(std::istream& in, const string& source)
    {
        span_model m;
        m.size = span_model_size_in(in, source);
        if (m.size == "tiny")
        {
            m.tiny_net = std::make_shared<pii_span_model_tiny>();
            deserialize(*m.tiny_net, in);
        }
        else
        {
            m.small_net = std::make_shared<pii_span_model_small>();
            deserialize(*m.small_net, in);
        }
        return m;
    }

#if !ANONYMIZER_ESSENTIALS_ONLY
    span_model load_span_model(const string& filename)
    {
        ifstream in(filename, ios::binary);
        if (!in) throw error("cannot read " + filename);
        return read_span_model(in, filename);
    }
#endif

// ----------------------------------------------------------------------------------------
// The models compiled into the program

    /* The embed command writes the serialized models compressed and encoded in base64, as
       file_to_code_ex does, cut into string literals short enough for every compiler. The
       compression is zlib (dlib/zip) rather than compress_stream: a little larger, and
       decoded about fifty times faster, which is what a program reading its models at
       every start needs. */
#if !ANONYMIZER_ESSENTIALS_ONLY
    string encode_model(const string& bytes)
    {
        base64 coder;
        istringstream packed(zlib_compress(bytes, 9));
        ostringstream text;
        coder.encode(packed, text);
        string out;
        for (char c : text.str())
            if (c != '\n' && c != '\r') out.push_back(c);
        return out;
    }
#endif

    string decode_model(const string& text)
    {
        if (text.empty()) return string();
        base64 coder;
        istringstream encoded(text);
        ostringstream packed;
        coder.decode(encoded, packed);
        return zlib_decompress(packed.str());
    }

    string embedded_tagger()
    {
#ifdef WITH_ANONYMIZER_MODELS
        return anonymizer_models::tagger();
#else
        return string();
#endif
    }

    string embedded_span_model()
    {
#ifdef WITH_ANONYMIZER_MODELS
        return anonymizer_models::span_model();
#else
        return string();
#endif
    }

    // What the program carries, for the help message and the embed command.
    string embedded_models_description()
    {
#ifdef WITH_ANONYMIZER_MODELS
        string d;
        if (!embedded_tagger().empty()) d += "tagger (" + string(anonymizer_models::tagger_source()) + ")";
        if (!embedded_span_model().empty())
            d += string(d.empty() ? "" : ", ") + "span model " + anonymizer_models::span_model_size() + " (" +
                 anonymizer_models::span_model_source() + ")";
        return d.empty() ? string("none") : d;
#else
        return "none";
#endif
    }

#if !ANONYMIZER_ESSENTIALS_ONLY
    // text written inside a C++ string literal.
    string literal_text(const string& text)
    {
        string out;
        for (char c : text)
        {
            if (c == '\\' || c == '"') out.push_back('\\');
            out.push_back(static_cast<unsigned char>(c) < 0x20 ? ' ' : c);
        }
        return out;
    }

    // One function returning the base64 text of a model, from literals of 4000 characters.
    void write_model_function(std::ostream& out, const string& name, const string& encoded)
    {
        out << "    inline std::string " << name << "()\n    {\n";
        if (encoded.empty())
        {
            out << "        return std::string();\n    }\n\n";
            return;
        }
        out << "        static const char* const parts[] = {\n";
        for (size_t p = 0; p < encoded.size(); p += 4000)
            out << "            \"" << encoded.substr(p, 4000) << "\",\n";
        out << "        };\n"
            << "        std::string text;\n"
            << "        text.reserve(" << encoded.size() << ");\n"
            << "        for (const char* part : parts) text += part;\n"
            << "        return text;\n    }\n\n";
    }

    /* embed: writes the header that compiles the recognizers into the program. A model not
       given on the command line is taken from those the program already carries, unless
       --no-tagger or --no-model leaves it out. */
    int embed(const command_line_parser& parser)
    {
        if (parser.number_of_arguments() != 1 || !parser.option("out"))
        {
            cerr << "Usage: embed --out slm_document_anonymizer_ex.h [--tagger tagger.dat] [--model model.dat] "
                 << "[--no-tagger] [--no-model]\n";
            return 1;
        }
        auto read_file = [](const string& filename) {
            ifstream in(filename, ios::binary);
            if (!in) throw error("cannot read " + filename);
            return string((istreambuf_iterator<char>(in)), istreambuf_iterator<char>());
        };
        string tagger_text, tagger_source, model_text, model_source, model_size;
        if (parser.option("tagger"))
        {
            const string bytes = read_file(parser.option("tagger").argument());
            pii_sequence_tagger check;
            istringstream in(bytes);
            deserialize(check, in);
            tagger_text = encode_model(bytes);
            tagger_source = file_name_of(parser.option("tagger").argument());
        }
        else if (!parser.option("no-tagger") && !embedded_tagger().empty())
        {
            tagger_text = embedded_tagger();
#ifdef WITH_ANONYMIZER_MODELS
            tagger_source = anonymizer_models::tagger_source();
#endif
        }
        if (parser.option("model"))
        {
            const string bytes = read_file(parser.option("model").argument());
            istringstream in(bytes);
            model_size = read_span_model(in, parser.option("model").argument()).size;
            model_text = encode_model(bytes);
            model_source = file_name_of(parser.option("model").argument());
        }
        else if (!parser.option("no-model") && !embedded_span_model().empty())
        {
            model_text = embedded_span_model();
#ifdef WITH_ANONYMIZER_MODELS
            model_source = anonymizer_models::span_model_source();
            model_size = anonymizer_models::span_model_size();
#endif
        }
        if (tagger_text.empty() && model_text.empty())
        {
            cerr << "embed: no model to write; give --tagger, --model or both.\n";
            return 1;
        }
        ostringstream out;
        out << "// Generated by slm_document_anonymizer_ex embed. Do not edit; run embed again after a new training.\n"
            << "// The trained recognizers of the program, compressed with dlib::zlib_compress(), in base64.\n"
            << "#ifndef SLM_DOCUMENT_ANONYMIZER_EX_H_\n#define SLM_DOCUMENT_ANONYMIZER_EX_H_\n\n"
            << "#include <string>\n\nnamespace anonymizer_models\n{\n"
            << "    inline const char* tagger_source() { return \"" << literal_text(tagger_source) << "\"; }\n"
            << "    inline const char* span_model_source() { return \"" << literal_text(model_source)
            << "\"; }\n"
            << "    inline const char* span_model_size() { return \"" << model_size << "\"; }\n\n";
        write_model_function(out, "tagger", tagger_text);
        write_model_function(out, "span_model", model_text);
        out << "}\n\n#endif // SLM_DOCUMENT_ANONYMIZER_EX_H_\n";
        write_atomically(parser.option("out").argument(), out.str());
        cout << "Wrote " << parser.option("out").argument() << ": "
             << (tagger_text.empty() ? string("no tagger") : "tagger " + tagger_source) << ", "
             << (model_text.empty() ? string("no span model") : "span model " + model_size + " " + model_source)
             << ".\nPlace it next to slm_document_anonymizer_ex.cpp and rebuild the target.\n";
        return 0;
    }
#endif

    struct recognizers
    {
        std::shared_ptr<pii_sequence_tagger> tagger;
        std::shared_ptr<span_model> model;
    };

    /* The models of the detector: those given with --tagger and --model, each read from its
       file independently of the other, and otherwise those compiled into the program;
       --no-tagger and --no-model leave one out. */
    recognizers load_recognizers(const command_line_parser& parser)
    {
        recognizers r;
#if ANONYMIZER_ESSENTIALS_ONLY
        (void)parser;
        const bool use_tagger = true, use_model = true;
#else
        if (parser.option("tagger"))
        {
            r.tagger = std::make_shared<pii_sequence_tagger>();
            deserialize(parser.option("tagger").argument()) >> *r.tagger;
        }
        if (parser.option("model"))
            r.model = std::make_shared<span_model>(load_span_model(parser.option("model").argument()));
        const bool use_tagger = !parser.option("no-tagger"), use_model = !parser.option("no-model");
#endif
        if (!r.tagger && use_tagger && !embedded_tagger().empty())
        {
            r.tagger = std::make_shared<pii_sequence_tagger>();
            istringstream in(decode_model(embedded_tagger()));
            deserialize(*r.tagger, in);
        }
        if (!r.model && use_model && !embedded_span_model().empty())
        {
            istringstream in(decode_model(embedded_span_model()));
            r.model = std::make_shared<span_model>(read_span_model(in, "the embedded span model"));
        }
        return r;
    }

    bool name_part(const pii_span& s)
    {
        return s.label == "GIVEN_NAME" || s.label == "FAMILY_NAME";
    }

    // True when s is a name part separated from a name part of spans by spaces or a hyphen only.
    bool next_to_name(const string& text, const pii_span& s, const std::vector<pii_span>& spans)
    {
        for (const pii_span& o : spans)
        {
            if (!name_part(o) || (o.begin < s.end && s.begin < o.end)) continue;
            const size_t from = o.end <= s.begin ? o.end : s.end, to = o.end <= s.begin ? s.begin : o.begin;
            if (to - from > 2) continue;
            bool gap = true;
            for (size_t k = from; k < to; ++k)
                if (text[k] != ' ' && text[k] != '-') gap = false;
            if (gap) return true;
        }
        return false;
    }

    /* The spans of the model, where a span read as an organisation with a confidence below
       doubt is replaced by the name parts the tagger reads over its words with more confidence,
       when they cover all of it: "Lucie Rey" alone on its line. */
    std::vector<pii_span> prefer_names(const string& text, const std::vector<pii_span>& model_spans,
                                       const std::vector<pii_span>& tagger_spans, double doubt)
    {
        std::vector<pii_span> out;
        for (const pii_span& m : model_spans)
        {
            std::vector<pii_span> parts;
            if (m.label == "ORGANIZATION" && m.confidence < doubt)
                for (const pii_span& s : tagger_spans)
                    if (name_part(s) && s.begin < m.end && m.begin < s.end && s.confidence > m.confidence)
                        parts.push_back(s);
            bool covered = !parts.empty();
            for (const pii_span& s : parts)
                if (s.begin < m.begin || s.end > m.end) covered = false;
            for (size_t k = m.begin; k < m.end && covered; ++k)
            {
                if (text[k] == ' ' || text[k] == '-') continue;
                bool inside = false;
                for (const pii_span& s : parts)
                    if (s.begin <= k && k < s.end) inside = true;
                covered = inside;
            }
            if (covered) out.insert(out.end(), parts.begin(), parts.end());
            else out.push_back(m);
        }
        return out;
    }

    /* Alone, either recognizer joins the detector with min_confidence. With both, the span
       model leads: the tagger, whose word features also fire on capitalized words that are not
       personal data, only fills its gaps, with spans of at least tagger_min_confidence where the
       span model kept nothing, and with the less confident name parts that complete a name
       already found, so that the parts of a name are replaced together. A doubtful
       organisation of the span model gives way to the name the tagger reads there. */
    void attach(const recognizers& r, pii_detector& detector, double min_confidence, double tagger_min_confidence)
    {
        const std::shared_ptr<pii_sequence_tagger> tagger = r.tagger;
        const std::shared_ptr<span_model> model = r.model;
        if (model && tagger)
        {
            detector.add_recognizer([model, tagger, min_confidence, tagger_min_confidence](const string& t) {
                std::vector<pii_span> kept, spans, pending;
                for (const pii_span& s : model->find(t))
                    if (s.confidence >= min_confidence) kept.push_back(s);
                const std::vector<pii_span> tagged = tagger->find(t);
                kept = prefer_names(t, kept, tagged, 0.9);
                spans = kept;
                for (const pii_span& s : tagged)
                {
                    bool free = true;
                    for (const pii_span& m : kept)
                        if (s.begin < m.end && m.begin < s.end) free = false;
                    if (!free) continue;
                    if (s.confidence >= tagger_min_confidence) spans.push_back(s);
                    else if (s.confidence >= min_confidence && name_part(s)) pending.push_back(s);
                }
                for (bool grown = true; grown;)
                {
                    grown = false;
                    for (size_t k = 0; k < pending.size(); ++k)
                    {
                        if (!next_to_name(t, pending[k], spans)) continue;
                        spans.push_back(pending[k]);
                        pending.erase(pending.begin() + static_cast<long>(k));
                        grown = true;
                        break;
                    }
                }
                std::sort(spans.begin(), spans.end(),
                          [](const pii_span& a, const pii_span& b) { return a.begin < b.begin; });
                return spans;
            }, 0);
        }
        else if (model)
            detector.add_recognizer([model](const string& t) { return model->find(t); }, min_confidence);
        else if (tagger)
            detector.add_recognizer([tagger](const string& t) { return tagger->find(t); }, min_confidence);
    }

#if !ANONYMIZER_ESSENTIALS_ONLY
    std::vector<pii_example> load_examples(const string& filename, unsigned long limit, unsigned long seed)
    {
        std::vector<pii_example> examples = load_pii_dataset(filename);
        if (limit && limit < examples.size())
        {
            std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
            std::shuffle(examples.begin(), examples.end(), rng);
            examples.resize(limit);
        }
        return examples;
    }

    std::vector<pii_span> detector_spans(pii_detector& detector, const string& text)
    {
        detector.begin_document();
        detector.learn(text);
        std::vector<pii_span> spans;
        for (const pii_entity& e : detector.detect(text))
        {
            pii_span s;
            s.begin = e.begin;
            s.end = e.end;
            s.type = e.type;
            s.label = pii_type_keyword(e.type);
            s.confidence = e.confidence;
            spans.push_back(s);
        }
        return spans;
    }

    void print_evaluation(const string& title, const pii_evaluation& exact, const pii_evaluation& overlap)
    {
        cout << "\n" << title << "\n" << exact.to_string();
        cout << "overlapping spans: precision " << overlap.overall.precision() << ", recall "
             << overlap.overall.recall() << ", f1 " << overlap.overall.f1() << "\n";
    }

    int train_tagger(const command_line_parser& parser)
    {
        if (parser.number_of_arguments() != 2 || !parser.option("out"))
        {
            cerr << "Usage: train-tagger train.tsv --out tagger.dat [--dev dev.tsv] [--examples N] [--c C] "
                 << "[--iterations N] [--threads N] [--cache N]\n";
            return 1;
        }
        const unsigned long seed = get_option(parser, "seed", 1UL);
        pii_tagger_training_options options;
        options.c = get_option(parser, "c", 10.0);
        options.max_iterations = get_option(parser, "iterations", 400UL);
        options.num_threads = get_option(parser, "threads", 2UL);
        options.max_cache_size = get_option(parser, "cache", 0UL);
        options.verbose = parser.option("verbose");
        if (options.c <= 0 || options.max_iterations == 0 || options.num_threads == 0)
        {
            cerr << "--c must be positive, --iterations and --threads at least 1.\n";
            return 1;
        }
        const std::vector<pii_example> train = load_examples(parser[1], get_option(parser, "examples", 20000UL), seed);
        cout << "Training the sequence tagger on " << train.size() << " examples (C = " << options.c << ")\n";
        const auto start = chrono::steady_clock::now();
        pii_sequence_tagger tagger;
        tagger.train(train, options);
        cout << "Trained in " << chrono::duration<double>(chrono::steady_clock::now() - start).count() << " s, "
             << tagger.misaligned_spans() << " spans not on token boundaries\n";
        serialize(parser.option("out").argument()) << tagger;
        if (parser.option("dev"))
        {
            const std::vector<pii_example> dev = load_examples(parser.option("dev").argument(), 2000, seed);
            std::vector<std::vector<pii_span>> truth, predicted;
            for (const pii_example& ex : dev)
            {
                truth.push_back(ex.spans);
                predicted.push_back(tagger.find(ex.text));
            }
            print_evaluation("Development set, exact spans",
                             evaluate_pii_spans(truth, predicted, span_match::exact, pii_fine_labels()),
                             evaluate_pii_spans(truth, predicted, span_match::overlap, pii_fine_labels()));
        }
        cout << "Saved " << parser.option("out").argument() << "\n";
        return 0;
    }

    /* init_bytes, when not empty, is the serialized model whose training goes on: the
       complementary training of --init. */
    template <typename model_type>
    int train_span_model(const command_line_parser& parser, const string& size, const string& init_bytes)
    {
        const unsigned long seed = get_option(parser, "seed", 1UL);
        const std::vector<pii_example> train = load_examples(parser[1], get_option(parser, "examples", 0UL), seed);
        std::vector<pii_example> dev;
        if (parser.option("dev")) dev = load_examples(parser.option("dev").argument(), 2000, seed);
        pii_span_model_training_options options;
        options.steps = get_option(parser, "steps", 20000UL);
        options.batch_size = get_option(parser, "batch", 32UL);
        options.learning_rate = get_option(parser, "lr", 1e-3);
        options.warmup_steps = std::min<unsigned long>(options.steps / 10, 1000);
        options.evaluate_every = get_option(parser, "evaluate-every", 2000UL);
        options.optimizer = get_option(parser, "optimizer", "adamw") == "adam-atan2" ? pii_span_optimizer::adam_atan2
                                                                                     : pii_span_optimizer::adamw;
        options.seed = seed;
        if (options.steps == 0 || options.batch_size == 0 || options.learning_rate <= 0)
        {
            cerr << "--steps and --batch must be at least 1, --lr positive.\n";
            return 1;
        }
        // The tiny model reads windows of up to 128 bytes, the window it is saved with.
        if (size == "tiny")
        {
            options.window_lengths = { 32, 64, 128 };
            options.window_weights = { 1, 2, 3 };
        }
        model_type model;
        if (!init_bytes.empty())
        {
            istringstream in(init_bytes);
            deserialize(model, in);
            options.continue_training = true;
            cout << "Continuing the training of the " << size << " span model from "
                 << parser.option("init").argument() << "\n";
        }
        cout << "Training the " << size << " span model on " << train.size() << " examples, " << options.steps
             << " steps of " << options.batch_size << " windows\n";
        model.train(train, dev, options, &cout);
        cout << model.parameter_count() << " parameters\n";
        serialize(parser.option("out").argument()) << model;
        cout << "Saved " << parser.option("out").argument() << "\n";
        if (!dev.empty())
        {
            std::vector<std::vector<pii_span>> truth, predicted;
            for (const pii_example& ex : dev)
            {
                truth.push_back(ex.spans);
                predicted.push_back(model.find(ex.text));
            }
            print_evaluation("Development set, exact spans",
                             evaluate_pii_spans(truth, predicted, span_match::exact, pii_fine_labels()),
                             evaluate_pii_spans(truth, predicted, span_match::overlap, pii_fine_labels()));
        }
        return 0;
    }

    int train_model(const command_line_parser& parser)
    {
        if (parser.number_of_arguments() != 2 || !parser.option("out"))
        {
            cerr << "Usage: train-model train.tsv --out model.dat [--dev dev.tsv] [--size tiny|small] [--steps N] "
                 << "[--batch N] [--lr X] [--optimizer adamw|adam-atan2] [--init model.dat|embedded]\n";
            return 1;
        }
        string size = get_option(parser, "size", "small");
        // --init: the model whose training goes on, from a file or from the program itself.
        string init_bytes;
        if (parser.option("init"))
        {
            const string from = parser.option("init").argument();
            if (from == "embedded")
            {
                if (embedded_span_model().empty())
                {
                    cerr << "--init embedded: this program carries no span model.\n";
                    return 1;
                }
                init_bytes = decode_model(embedded_span_model());
            }
            else
            {
                ifstream in(from, ios::binary);
                if (!in) throw error("cannot read " + from);
                init_bytes.assign(istreambuf_iterator<char>(in), istreambuf_iterator<char>());
            }
            istringstream in(init_bytes);
            const string init_size = span_model_size_in(in, from);
            if (parser.option("size") && init_size != size)
            {
                cerr << "--size " << size << " differs from the size of the --init model, " << init_size << ".\n";
                return 1;
            }
            size = init_size;
        }
        if (size == "tiny") return train_span_model<pii_span_model_tiny>(parser, size, init_bytes);
        return train_span_model<pii_span_model_small>(parser, size, init_bytes);
    }
#endif

    // The dictionaries and the regular expressions given on the command line.
    void configure(const command_line_parser& parser, pii_detector& detector)
    {
        if (parser.option("names"))
            for (const string& name : read_list(parser.option("names").argument()))
                detector.add_person_name(name);
        if (parser.option("terms"))
        {
            const string filename = parser.option("terms").argument();
            for (const string& line : read_list(filename))
            {
                const size_t tab = line.find('\t');
                const pii_type t = tab == string::npos ? pii_type::none : pii_type_from_keyword(line.substr(0, tab));
                if (t == pii_type::none)
                    throw error(filename + ": expected \"TYPE<tab>term\": " + line);
                detector.add_term(t, line.substr(tab + 1));
            }
        }
        if (parser.option("keep"))
            for (const string& term : read_list(parser.option("keep").argument()))
                detector.add_allowed_term(term);
        if (parser.option("patterns"))
            read_patterns(parser.option("patterns").argument(), detector);
    }

    /* Reads a comma separated list of type keywords; spaces around them are ignored. */
    std::vector<pii_type> read_types(const string& list, const string& option)
    {
        std::vector<pii_type> types;
        istringstream in(list);
        string keyword;
        while (getline(in, keyword, ','))
        {
            const size_t b = keyword.find_first_not_of(" \t"), e = keyword.find_last_not_of(" \t");
            if (b == string::npos) continue;
            const pii_type t = pii_type_from_keyword(keyword.substr(b, e - b + 1));
            if (t == pii_type::none) throw error("--" + option + ": unknown type \"" + keyword + "\"");
            types.push_back(t);
        }
        if (types.empty()) throw error("--" + option + ": no type given");
        return types;
    }

#if !ANONYMIZER_ESSENTIALS_ONLY
    /* Scores the rules alone, each recognizer alone and the detector with the recognizers,
       on a test set annotated by build_pii_dataset.py. Models are scored on the fine labels
       they learn; every system is also scored at the level at which values are replaced,
       where a given name and a family name side by side form one PERSON. */
    int evaluate(const command_line_parser& parser, const recognizers& r, const pii_detector_options& dopt,
                 double min_confidence, double tagger_min_confidence)
    {
        if (parser.number_of_arguments() != 2)
        {
            cerr << "Usage: evaluate test.tsv [--tagger tagger.dat] [--model model.dat] [--examples N] [--seed N]\n";
            return 1;
        }
        const std::vector<pii_example> test =
            load_examples(parser[1], get_option(parser, "examples", 0UL), get_option(parser, "seed", 1UL));
        std::vector<std::vector<pii_span>> fine_truth, coarse_truth;
        for (const pii_example& ex : test)
        {
            fine_truth.push_back(ex.spans);
            coarse_truth.push_back(coarsen_pii_spans(ex.text, ex.spans));
        }
        cout << test.size() << " examples\n";

        auto score_coarse = [&](const string& title, const std::vector<std::vector<pii_span>>& predicted,
                                const std::vector<string>& labels) {
            print_evaluation(title, evaluate_pii_spans(coarse_truth, predicted, span_match::exact, labels),
                             evaluate_pii_spans(coarse_truth, predicted, span_match::overlap, labels));
        };
        const std::vector<string> learned = { "PERSON", "LOCATION", "ADDRESS", "ORG" };

        pii_detector rules(dopt);
        configure(parser, rules);
        std::vector<std::vector<pii_span>> predicted;
        auto start = chrono::steady_clock::now();
        for (const pii_example& ex : test) predicted.push_back(detector_spans(rules, ex.text));
        cout << "rules: " << chrono::duration<double>(chrono::steady_clock::now() - start).count() << " s\n";
        score_coarse("Rules alone, categories the models learn (replacement level)", predicted, learned);
        score_coarse("Rules alone, every category", predicted, std::vector<string>());

        auto score_recognizer = [&](const string& name, const std::function<std::vector<pii_span>(const string&)>& f) {
            std::vector<std::vector<pii_span>> fine, coarse;
            const auto t0 = chrono::steady_clock::now();
            for (const pii_example& ex : test)
            {
                fine.push_back(f(ex.text));
                coarse.push_back(coarsen_pii_spans(ex.text, fine.back()));
            }
            const double seconds = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
            cout << "\n" << name << ": " << seconds << " s\n";
            print_evaluation(name + " alone, fine labels",
                             evaluate_pii_spans(fine_truth, fine, span_match::exact, pii_fine_labels()),
                             evaluate_pii_spans(fine_truth, fine, span_match::overlap, pii_fine_labels()));
            score_coarse(name + " alone (replacement level)", coarse, learned);
        };
        if (r.tagger) score_recognizer("Sequence tagger", [&](const string& t) { return r.tagger->find(t); });
        if (r.model)
            score_recognizer("Span model (" + r.model->size + ")", [&](const string& t) { return r.model->find(t); });

        if (r.tagger || r.model)
        {
            pii_detector combined(dopt);
            configure(parser, combined);
            attach(r, combined, min_confidence, tagger_min_confidence);
            predicted.clear();
            start = chrono::steady_clock::now();
            for (const pii_example& ex : test) predicted.push_back(detector_spans(combined, ex.text));
            const double seconds = chrono::duration<double>(chrono::steady_clock::now() - start).count();
            cout << "\ndetector with recognizers: " << seconds << " s\n";
            score_coarse("Detector with recognizers, categories the models learn", predicted, learned);
            score_coarse("Detector with recognizers, every category", predicted, std::vector<string>());
        }
        return 0;
    }

    /* For any text: is there personal data, where, and what. Each line gives the begin and
       end byte offsets, the type keyword (the fine label for the recognizers), the
       confidence and the value. */
    int detect(pii_detector& detector, const recognizers& r)
    {
#if defined(_WIN32)
        _setmode(_fileno(stdin), _O_BINARY);
#endif
        const string text((istreambuf_iterator<char>(cin)), istreambuf_iterator<char>());
        const std::vector<pii_span> spans = detector_spans(detector, text);
        cout << "found: " << (spans.empty() ? "no" : "yes") << "\n";
        for (const pii_span& s : spans)
            cout << "  " << s.begin << "\t" << s.end << "\t" << s.label << "\t" << s.confidence << "\t"
                 << printable(text.substr(s.begin, s.end - s.begin)) << "\n";
        auto show = [&](const string& name, const std::vector<pii_span>& found) {
            cout << name << ": " << (found.empty() ? "no" : "yes") << "\n";
            for (const pii_span& s : found)
                cout << "  " << s.begin << "\t" << s.end << "\t" << s.label << "\t" << s.confidence << "\t"
                     << printable(text.substr(s.begin, s.end - s.begin)) << "\n";
        };
        if (r.tagger) show("tagger", r.tagger->find(text));
        if (r.model) show("model", r.model->find(text));
        return 0;
    }
#endif
}

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{
#if defined(_WIN32)
    const windows_console console;
#endif
    try
    {
        command_line_parser parser;
        parser.set_group_name("Key options");
        parser.add_option("key", "Key file. Default: " + default_key_path() + ", created by the first anonymize run.",
                          1);
        parser.add_option("passphrase-env", "Derive the key from the passphrase held in environment variable <arg> "
                          "(PBKDF2-HMAC-SHA-256, 600000 iterations); no key file is used.", 1);
        parser.add_option("context",
                          "Domain separation label, \"default\" if absent. Documents protected under different "
                          "contexts do not share pseudonyms; restoration needs the same context.",
                          1);
        parser.add_option("irreversible", "Use a random key that is never stored: the outputs cannot be restored.");
        parser.set_group_name("Detection options");
        parser.add_option("types",
                          "Comma separated keywords of the types to replace, among PERSON ORG LOCATION ADDRESS "
                          "EMAIL PHONE IBAN CARD NIR SIRET ID IPV4 IPV6 MAC URL DATE PLATE GEO TITLE GENDER "
                          "NATIONALITY. Default: all.",
                          1);
        parser.add_option("except", "Comma separated keywords of types left readable, such as TITLE,GENDER.", 1);
        parser.add_option("no-public-bodies",
                          "Leave administrations, courts and public agencies readable; they are replaced as ORG "
                          "by default.");
        parser.add_option("names", "File of person names always replaced, one per line.", 1);
        parser.add_option("terms", "File of terms always replaced, one \"TYPE<tab>term\" per line.", 1);
        parser.add_option(
            "keep", "File of terms never replaced (the organisation's own name, public figures), one per line.", 1);
        parser.add_option("patterns", "File of regular expressions, one \"TYPE<tab>regex[<tab>options]\" per line; "
                          "options: luhn, iban, icase, group=N, confidence=X.", 1);
#if !ANONYMIZER_ESSENTIALS_ONLY
        parser.add_option("tagger", "Sequence tagger model (train-tagger) used with the rules, in place of the one "
                          "compiled into the program.", 1);
        parser.add_option("model", "Span model (train-model) used with the rules, in place of the one compiled into "
                          "the program.", 1);
        parser.add_option("no-tagger", "Use no sequence tagger.");
        parser.add_option("no-model", "Use no span model.");
        parser.add_option("min-confidence", "Smallest confidence of a span kept from the recognizers. Default: 0.5.",
                          1);
        parser.add_option("tagger-min-confidence", "With --model: smallest confidence of a tagger span kept where the "
                          "span model found nothing; a name part that completes a name found needs only "
                          "--min-confidence. Default: 0.9.", 1);
#endif
        parser.add_option("dates", "Date policy: shift (default) moves every date into the past by the same secret "
                          "number of weeks, a month with its year or a year alone by whole months or years; "
                          "encrypt makes dates opaque, keep leaves them.", 1);
        parser.add_option("date-order", "Order of numeric dates such as 03/04/2026: dmy (default) or mdy.", 1);
        parser.add_option("all-urls",
                          "Replace every URL; the default replaces links to personal profiles, links holding an "
                          "e-mail address, credentials, an IP address or a known name, and keeps the others "
                          "readable and whole.");
        parser.set_group_name("Output options");
        parser.add_option("out-dir", "Write the outputs under <arg>; by default they go next to their source.", 1);
        parser.add_option("report", "Write a JSON report next to each output: counts by type, removed content, notes. "
                          "Reports never contain personal values.");
        parser.add_option("force", "Overwrite existing outputs.");
        parser.add_option("restore", "With the text command: restore standard input.");
#if !ANONYMIZER_ESSENTIALS_ONLY
        parser.set_group_name("Training options");
        parser.add_option("out", "File the trained model is written to.", 1);
        parser.add_option("dev", "Development set evaluated after training.", 1);
        parser.add_option("examples", "Number of examples drawn at random from the training or test set (0: all). "
                          "Default: 20000 for train-tagger, all otherwise.", 1);
        parser.add_option("seed", "Random seed of the example draw and of the span model training. Default: 1.", 1);
        parser.add_option("c", "Tagger: SVM C. Default: 10.", 1);
        parser.add_option("iterations", "Tagger: cutting plane iterations. Default: 400.", 1);
        parser.add_option("threads", "Tagger: threads. Default: 2.", 1);
        parser.add_option("cache", "Tagger: labelings the solver keeps per example; 5 trains about three times "
                          "faster with about six times the memory. Default: 0.", 1);
        parser.add_option("size", "Span model size: tiny or small. Default: small.", 1);
        parser.add_option("steps", "Span model: optimizer steps. Default: 20000.", 1);
        parser.add_option("batch", "Span model: windows per step. Default: 32.", 1);
        parser.add_option("lr", "Span model: peak learning rate. Default: 0.001.", 1);
        parser.add_option("optimizer", "Span model: adamw or adam-atan2 (dlib::adam_atan2). Default: adamw.", 1);
        parser.add_option("init", "Span model: continue the training of <arg>, a model file or \"embedded\" for the "
                          "model compiled into the program; its size is kept.", 1);
        parser.add_option("evaluate-every", "Span model: steps between two evaluations on the dev set. Default: 2000.",
                          1);
        parser.add_option("verbose", "Print the progress of the solver.");
#endif
        parser.set_group_name("Miscellaneous options");
        parser.add_option("h", "Display this help message.");
        parser.add_option("help", "Display this help message.");

        parser.parse(argc, argv);
#if ANONYMIZER_ESSENTIALS_ONLY
        const char* single[] = { "key", "passphrase-env", "context", "types", "except", "names", "terms", "keep",
                                 "patterns", "dates", "date-order", "out-dir" };
#else
        const char* single[] = { "key", "passphrase-env", "context", "types", "except", "names", "terms", "keep",
                                 "patterns",
                                 "tagger", "model", "min-confidence", "tagger-min-confidence", "dates", "date-order",
                                 "out-dir", "out", "dev",
                                 "examples", "seed", "c", "iterations", "threads", "cache", "size", "steps", "batch",
                                 "lr", "optimizer", "init", "evaluate-every" };
#endif
        parser.check_one_time_options(single);
        const char* key_sources[] = { "key", "passphrase-env", "irreversible" };
        parser.check_incompatible_options(key_sources);
#if !ANONYMIZER_ESSENTIALS_ONLY
        parser.check_incompatible_options("tagger", "no-tagger");
        parser.check_incompatible_options("model", "no-model");
#endif
        const char* dates[] = { "shift", "encrypt", "keep" };
        parser.check_option_arg_range("dates", dates);
        const char* orders[] = { "dmy", "mdy" };
        parser.check_option_arg_range("date-order", orders);
#if !ANONYMIZER_ESSENTIALS_ONLY
        const char* sizes[] = { "tiny", "small" };
        parser.check_option_arg_range("size", sizes);
        const char* optimizers[] = { "adamw", "adam-atan2" };
        parser.check_option_arg_range("optimizer", optimizers);
#endif

        if (parser.option("h") || parser.option("help") || parser.number_of_arguments() == 0)
        {
            cout << "Usage: " << program_name(argc > 0 ? argv[0] : nullptr)
                 << " <command> [options] [files or folders]\n\n"
                 << "Commands:\n"
                 << "  anonymize     Protect files and folders: x.docx gives x.anon.docx.\n"
                 << "  restore       Restore protected files: x.anon.docx gives x.restored.docx.\n"
                 << "  scan          Report the personal data of files and folders, without writing anything.\n"
                 << "  text          Protect standard input to standard output; restore it with --restore.\n"
#if !ANONYMIZER_ESSENTIALS_ONLY
                 << "  detect        Print the personal data found in standard input: offsets, type, label.\n"
#endif
                 << "  keygen        Create a new key file.\n"
#if !ANONYMIZER_ESSENTIALS_ONLY
                 << "  train-tagger  Train the sequence tagger on a dataset (slm_tools/build_pii_dataset.py).\n"
                 << "  train-model   Train the span model on such a dataset.\n"
                 << "  evaluate      Score the rules, the recognizers and the whole detector on a test set.\n"
                 << "  embed         Write the header that compiles the recognizers into the program.\n"
#endif
                 << "\nRecognizers compiled into this program: " << embedded_models_description() << ".\n";
            parser.print_options();
            return parser.number_of_arguments() == 0 && !parser.option("h") && !parser.option("help") ? 1 : 0;
        }

        const string verb = parser[0];
        command cmd;
        if (verb == "anonymize") cmd = command::anonymize;
        else if (verb == "restore") cmd = command::restore;
        else if (verb == "scan") cmd = command::scan;
        else if (verb == "text") cmd = command::text;
        else if (verb == "keygen") cmd = command::keygen;
#if !ANONYMIZER_ESSENTIALS_ONLY
        else if (verb == "detect") cmd = command::detect;
        else if (verb == "train-tagger") cmd = command::train_tagger;
        else if (verb == "train-model") cmd = command::train_model;
        else if (verb == "evaluate") cmd = command::evaluate;
        else if (verb == "embed") cmd = command::embed;
#endif
        else
        {
            cerr << "Unknown command: " << verb << " (try --help)\n";
            return 1;
        }
#if !ANONYMIZER_ESSENTIALS_ONLY
        if (cmd == command::train_tagger) return train_tagger(parser);
        if (cmd == command::train_model) return train_model(parser);
        if (cmd == command::embed) return embed(parser);
#endif

        // The detector, shared by every remaining command.
        const numeric_date_order order = get_option(parser, "date-order", "dmy") == "mdy"
            ? numeric_date_order::month_day_year : numeric_date_order::day_month_year;
        const string date_policy_name = get_option(parser, "dates", "shift");
        pii_detector_options dopt;
        dopt.date_order = order;
        dopt.profile_urls_only = !parser.option("all-urls");
        dopt.public_bodies = !parser.option("no-public-bodies");
        try
        {
            if (parser.option("types")) dopt.types = read_types(parser.option("types").argument(), "types");
            if (parser.option("except"))
                for (pii_type t : read_types(parser.option("except").argument(), "except"))
                    dopt.types.erase(std::remove(dopt.types.begin(), dopt.types.end(), t), dopt.types.end());
        }
        catch (error& e)
        {
            cerr << e.what() << "\n";
            return 1;
        }
        if (date_policy_name == "keep")
            dopt.types.erase(std::remove(dopt.types.begin(), dopt.types.end(), pii_type::date), dopt.types.end());
#if ANONYMIZER_ESSENTIALS_ONLY
        const double min_confidence = 0.5, tagger_min_confidence = 0.9;
#else
        const double min_confidence = get_option(parser, "min-confidence", 0.5);
        const double tagger_min_confidence = get_option(parser, "tagger-min-confidence", 0.9);
#endif
        // Restoring and creating a key detect nothing, and load no model.
        const bool restoring_only = cmd == command::restore || cmd == command::keygen ||
                                    (cmd == command::text && parser.option("restore"));
        const recognizers learned = restoring_only ? recognizers() : load_recognizers(parser);
#if !ANONYMIZER_ESSENTIALS_ONLY
        if (cmd == command::evaluate) return evaluate(parser, learned, dopt, min_confidence, tagger_min_confidence);
#endif

        pii_detector detector(dopt);
        configure(parser, detector);
        attach(learned, detector, min_confidence, tagger_min_confidence);
        if ((cmd == command::detect || cmd == command::text || cmd == command::keygen) &&
            parser.number_of_arguments() > 1)
        {
            cerr << verb << " takes no file argument" << (cmd == command::keygen ? "" : ": it reads standard input")
                 << ".\n";
            return 1;
        }
#if !ANONYMIZER_ESSENTIALS_ONLY
        if (cmd == command::detect) return detect(detector, learned);
#endif

        const bool restoring = cmd == command::restore || (cmd == command::text && parser.option("restore"));
        if (parser.option("irreversible") && (restoring || cmd == command::keygen))
        {
            cerr << "--irreversible keeps no key: there is nothing to restore with.\n";
            return 1;
        }

        // The key.
        const string key_path = get_option(parser, "key", default_key_path());
        pseudonymization_key key;
        if (cmd == command::keygen)
        {
            make_directories(parent_of(key_path), true);
            pseudonymization_key::generate().save(key_path);
            cout << "Key created in " << key_path << ". Keep it secret and keep a copy: without it, protected "
                 << "documents cannot be restored.\n";
            return 0;
        }
        if (parser.option("irreversible"))
        {
            key = pseudonymization_key::generate();
        }
        else if (parser.option("passphrase-env"))
        {
            const char* pass = getenv(parser.option("passphrase-env").argument().c_str());
            if (!pass || string(pass).size() < 12)
            {
                cerr << "The environment variable " << parser.option("passphrase-env").argument()
                     << " must hold a passphrase of at least 12 characters.\n";
                return 1;
            }
            key = pseudonymization_key::from_passphrase(pass);
        }
        else if (file_exists(key_path))
        {
            key = pseudonymization_key::load(key_path);
        }
        else if (cmd == command::anonymize || (cmd == command::text && !restoring))
        {
            make_directories(parent_of(key_path), true);
            key = pseudonymization_key::generate();
            key.save(key_path);
            cerr << "Key created in " << key_path << ". Keep it secret and keep a copy: it is the only way to "
                 << "restore the documents protected with it.\n";
        }
        else if (cmd == command::scan)
        {
            key = pseudonymization_key::generate();
        }
        else
        {
            cerr << "No key file in " << key_path << " (use --key or --passphrase-env).\n";
            return 1;
        }

        pseudonymizer_options popt;
        popt.context = get_option(parser, "context", "default");
        popt.dates = date_policy_name == "encrypt" ? date_policy::encrypt :
                     date_policy_name == "keep" ? date_policy::keep : date_policy::shift;
        popt.date_order = order;
        const pseudonymizer p(key, popt);
        document_anonymizer anonymizer(p, detector);

        // Standard input to standard output.
        if (cmd == command::text)
        {
#if defined(_WIN32)
            _setmode(_fileno(stdin), _O_BINARY);
            _setmode(_fileno(stdout), _O_BINARY);
#endif
            const string input((istreambuf_iterator<char>(cin)), istreambuf_iterator<char>());
            document_report report;
            const string output = restoring ? anonymizer.restore(input, document_format::text, report)
                                            : anonymizer.anonymize(input, document_format::text, report);
            cout << output << flush;
            cerr << summary(report) << "\n";
            return report.tags_rejected > 0 ? 2 : 0;
        }

        // Files and folders.
        std::vector<job> jobs;
        for (unsigned long i = 1; i < parser.number_of_arguments(); ++i)
        {
            const string arg = parser[i];
            if (!file_exists(arg))
            {
                try
                {
                    const directory dir(arg);
                    std::vector<job> found;
                    collect(dir, "", found);
                    for (const job& j : found)
                    {
                        const string name = file_name_of(j.input);
                        // In a folder, only sources are protected and only protected files restored.
                        if (cmd == command::restore ? is_protected_output(name) : !is_previous_output(name))
                            jobs.push_back(j);
                    }
                    continue;
                }
                catch (directory::dir_not_found&)
                {
                    cerr << "Not found: " << arg << "\n";
                    return 1;
                }
            }
            jobs.push_back(job{ arg, file_name_of(arg) });
        }
        if (jobs.empty())
        {
            cerr << "No file to process.\n";
            return 1;
        }

        const string out_dir = get_option(parser, "out-dir", "");
        // No output may replace a source or another output of the same run, even with --force.
        std::set<string> sources, targets;
        for (const job& j : jobs) sources.insert(absolute_path(j.input));
        int failures = 0;
        for (const job& j : jobs)
        {
            try
            {
                const document_format format = detect_document_file_format(j.input);
                if (format == document_format::unknown || format == document_format::legacy_office)
                {
                    cerr << j.input << ": skipped, "
                         << (format == document_format::unknown
                                 ? "unsupported format"
                                 : "binary or password protected Office file, save it as "
                                   "DOCX, XLSX or PPTX without password first")
                         << "\n";
                    ++failures;
                    continue;
                }
                const string bytes = read_file_bytes(j.input);
                document_report report;
                const string output = restoring ? anonymizer.restore(bytes, format, report)
                                                : anonymizer.anonymize(bytes, format, report);
                if (cmd == command::scan)
                {
                    cout << j.input << " [" << document_format_name(format) << "]: " << summary(report) << "\n";
                    for (const string& r : report.removed) cout << "  would remove: " << r << "\n";
                    for (const string& n : report.notes) cout << "  note: " << n << "\n";
                    continue;
                }

                const string target = out_dir.empty() ? output_name(j.input, cmd)
                                                      : out_dir + separator + output_name(j.relative, cmd);
                const string where = absolute_path(target);
                if (sources.count(where)) throw error("the output " + target + " would replace a source");
                if (!targets.insert(where).second)
                    throw error("the output " + target + " is also the output of another file");
                if (file_exists(target) && !parser.option("force"))
                    throw error(target + " exists (use --force to overwrite)");
                const string report_file = target + ".report.json";
                if (parser.option("report") && file_exists(report_file) && !parser.option("force"))
                    throw error(report_file + " exists (use --force to overwrite)");
                make_directories(parent_of(target), false);
                write_atomically(target, output);
                if (parser.option("report"))
                {
                    report.key_fingerprint = parser.option("irreversible") ? string("none (irreversible)")
                                                                          : report.key_fingerprint;
                    write_atomically(report_file, report.to_json());
                }
                cout << j.input << " -> " << target << ": " << summary(report) << "\n";
                for (const string& r : report.removed) cout << "  removed: " << r << "\n";
                for (const string& n : report.notes) cout << "  note: " << n << "\n";
                // Tags another key or context wrote, or altered ones, stay as they are.
                if (report.tags_rejected > 0) ++failures;
            }
            catch (std::exception& e)
            {
                cerr << j.input << ": " << e.what() << "\n";
                ++failures;
            }
        }
        if (cmd != command::scan && !parser.option("irreversible"))
            cout << "Key fingerprint " << p.key_fingerprint() << " (context \"" << popt.context << "\")\n";
        return failures == 0 ? 0 : 2;
    }
    catch (std::exception& e)
    {
        cerr << e.what() << endl;
        return 1;
    }
}

// ----------------------------------------------------------------------------------------
