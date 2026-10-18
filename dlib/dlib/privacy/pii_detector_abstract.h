// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_PII_DETECTOR_ABSTRACT_H_
#ifdef DLIB_PII_DETECTOR_ABSTRACT_H_

#include <functional>
#include <string>
#include <vector>
#include "pii_types_abstract.h"
#include "date_formats_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    struct pii_detector_options
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The configuration of a pii_detector.

                types             - the categories reported; a value of another type is
                                    left in place
                date_order        - how numeric dates such as 03/04/2026 are read
                profile_urls_only - when true, a web address is reported only if it is
                                    personal data: a profile on a social network, an
                                    address holding an e-mail address, credentials or an
                                    IP address, or a word of a name or of an e-mail
                                    address seen in the document. The other addresses
                                    are protected: nothing inside them is reported, so
                                    that a link to a public page keeps working. When
                                    false, every web address is reported.
                public_bodies     - when true, administrations, courts, social security
                                    bodies and regulators of France and of many other
                                    countries are reported as organisations: acronyms ("la
                                    CAF", "HMRC", "DGFiP", "BAMF"), names of their own
                                    ("France Travail", "Home Office", "Agenzia delle
                                    Entrate", "Bundesagentur für Arbeit"), a head noun with
                                    what the body does or where it sits ("Tribunal
                                    judiciaire de Rennes", "Sous-préfecture de Brest",
                                    "Leeds Crown Court", "Town of Vancouver"), and the
                                    branch of a national body with its place ("Agence
                                    France Travail d'Arras", "the Leeds Jobcentre Plus").
                                    A head noun alone ("la préfecture"), a branch named by
                                    its place only ("l'agence de Lyon") and the councils of
                                    an organisation ("la Direction financière", "the
                                    Steering Group") stay readable.
        !*/
        std::vector<pii_type> types = all_pii_types();
        numeric_date_order date_order = numeric_date_order::day_month_year;
        bool profile_urls_only = true;
        bool public_bodies = true;
    };

    struct detection_context
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                What the document structure says about a text unit before its words are
                read.

                forced - the unit is a value of this type as a whole: a document author
                         in the metadata, a comment author, a cell of a column whose
                         header is "Nom" or "E-mail". A unit forced to title, gender or
                         nationality is reported only when its value belongs to the
                         vocabulary of that type, so that a column headed "Titre" holding
                         "Rapport annuel" stays readable.
                label  - the unit follows a field label of this type, typically the
                         neighbouring table cell ("Nom" | "DUPONT").
        !*/
        pii_type forced = pii_type::none;
        pii_type label = pii_type::none;
    };

    struct pii_pattern
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                A regular expression added to a detector with add_pattern(), for the
                identifiers specific to an organisation: employee numbers, file
                references, internal account codes.

                type        - the type reported for a match
                regex       - an ECMAScript regular expression (std::regex), matched on
                              the UTF-8 bytes of each line
                confidence  - the confidence reported for a match
                ignore_case - match without regard to case
                group       - the capture group reported, 0 for the whole match, so that
                              "badge\s*:\s*([A-Z0-9-]+)" reports the code alone
                check       - "" or a check the match must pass: "luhn" (on its digits) or
                              "iban" (ISO 13616 mod 97 on its letters, in capitals, and
                              its digits)
        !*/
        pii_type type = pii_type::identifier;
        std::string regex;
        double confidence = 0.9;
        bool ignore_case = false;
        unsigned long group = 0;
        std::string check;
    };

    typedef std::function<std::vector<pii_span>(const std::string& text)> pii_recognizer;
    /*!
        A learned recognizer, such as pii_sequence_tagger::find() or pii_span_model::find():
        given a UTF-8 text, it returns the spans it finds, with byte offsets, a label or a
        type, and a confidence.
    !*/

// ----------------------------------------------------------------------------------------

    class pii_detector
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                Finds personal data in UTF-8 text, French and English first.

                Five channels contribute candidates. Overlaps are resolved by priority,
                then length, then position: the validated scanners first, then the user
                patterns and dictionaries, the confident learned spans, the labelled
                values and the linguistic rules. Typographic spaces read as spaces and
                invisible marks (soft hyphens, zero width spaces) are skipped, so that
                they hide no value; offsets always point into the original bytes.

                1. Validated scanners. E-mail addresses (internationalized ones
                   included), web addresses, IPv4 and IPv6 addresses (IPv4-mapped
                   forms included), MAC addresses, phone numbers, IBAN (country length
                   and mod 97), payment cards (Luhn, issuer lengths and prefixes), NIR
                   (control key), SIREN and SIRET (Luhn), license plates, dates, GPS
                   coordinates (decimal or degrees, minutes and seconds), IMEI, ICCID
                   and IMSI numbers, UUIDs, vehicle identification numbers, French bank
                   details (RIB key), tax and passport numbers, and the national numbers
                   of about forty countries: Belgium, Spain (DNI, NIE), Italy (codice
                   fiscale), the United Kingdom (NINO, NHS), the United States (SSN,
                   ITIN), Switzerland (AVS), Luxembourg, the Netherlands (BSN), Germany
                   (Steuer-IdNr), Austria, Poland (PESEL), Portugal (NIF), Czechia,
                   Denmark, Finland, Sweden, Norway, Iceland, Ireland (PPSN), Romania,
                   Bulgaria, Croatia, Estonia and Lithuania, Slovenia, Greece, Canada
                   (SIN), Quebec (RAMQ), Brazil (CPF), Argentina, Chile, Mexico (CURP),
                   India (Aadhaar), China, Japan, Korea, Turkey, Israel, South Africa,
                   Australia (TFN) and the Maghreb (CIN, NIN). Each value is accepted
                   only when its check digits verify, or after the label that announces
                   it for the numbers that have none (RAMQ, CIN, NIN); a number whose
                   check leaves one chance in ten or eleven also needs its label ("BSN",
                   "PESEL", "CPR-nr."), as do phone numbers outside the French and
                   international layouts, social handles, pre-2009 French plates
                   and the references of files, customers, contracts, claims, invoices,
                   requests, applications, permits or asset tags ("dossier n° 2024-00123",
                   reference "B4694342", "Ref: 2476486"), a file name excepted. A cue is a
                   whole word: "SIM" is not found in "Simon", nor "tel" in "hotel"; a cue
                   written with a final '*' in this implementation is a prefix ("tél." and
                   "téléphone"). These scanners work in time linear in the length of the
                   text.

                2. Linguistic evidence. Person names are scored from civilities and
                   titles (M., Mme, Dr), field labels ("Nom :", "Signé par"), the
                   first-name lexicon (given names from all regions of the world, their
                   short forms, nicknames and stage names), shape (Jean DUPONT, Amadou BA),
                   surrounding cues ("né le", "épouse", "je soussigné") and the document
                   lexicon; particles ("M. Le Goff", "M. de La Guen") and apostrophe forms
                   ("N'Diaye", "O'Connor") belong to the name, a professional title (Dr,
                   Maître) and a greeting ("Salut Murat") do not. A nickname, a stage name
                   or a pen name announced by its cue is a person too ("Jean Dupont, dit
                   « Jojo »", "surnommée Mimi", "a.k.a. 'Smithy'", "sous le pseudonyme de
                   Vernon Sullivan"), while reported speech stays readable ("Il a dit
                   « bonjour »"). Organisations come from legal forms before or after the
                   name ("SAS Iroise", "Dupont Conseil SAS"), head nouns ("Cabinet Martin",
                   "Universidad de Sevilla"), sector words after a name ("Zorvane Finance",
                   "Ulmeo Seguros"), partners ("Martin et Fils"), public body names, and
                   about two thousand organisations of all countries known by their own
                   name ("Airbus", "Johnson & Johnson", "La Banque postale"), with a legal
                   form, a trade or a country after them ("AXA France"); such a name
                   followed by another capitalized word is a product or a longer name and
                   is left to the other channels ("Microsoft Teams", "Airbus A320").
                   Products, applications and the intergovernmental organisations of the
                   United Nations system are left out of that table. "La société"
                   introduces a name and stays outside it. Addresses come
                   from street types, places from postal codes ("CEDEX" stays readable),
                   from "City, ST 90012" and British postcodes, from the postal codes of
                   other countries written before or after their town, after a street,
                   before a country or alone on their line, as in a paragraph of a letter
                   ("8004 Zurich", "00-238 Warszawa", "1015 CJ Amsterdam", "Tel Aviv
                   6688116", "Tokyo 153-0063", "Dublin D02 X285"),
                   from "né à", "domicilié à", "originaire de", and from the place before
                   the date of a letter ("Rennes, le 12 mars 2024"). User names come from
                   logs and chats ("user=jdupont", "[10:42] jdupont:", "opened by
                   jdupont").

                   A civility before a name is reported as title, since it tells the sex
                   of the person: "M. Dupont", "M. et Mme Martin", "Titulaire : Mme
                   VAILLANT", "MmeLamia Chaoui" when a space was lost; after another
                   title, "M." is an initial ("Monsieur M. Denis"). A civility without a
                   name ("Madame, Monsieur,", "Monsieur le Procureur") stays readable.
                   Sexes and nationalities are reported as gender and nationality in
                   labelled fields, after a cue ("de nationalité française", "passeport
                   algérien", "a British citizen", "an 82 year old American", "française
                   de 73 ans"), in apposition to a name ("Louis Patel, American,") and in
                   the cells of a row that holds a name ("PEREIRA Awa – F – belge").

                   Tables written as text are read like spreadsheets: a header line of
                   field labels ("| Nom | Sexe | Nationalité |", "Civilité;Nom;Prénom",
                   tab separated columns) types the cells of the lines below it.

                3. Document-level propagation. Detection runs in two passes over a
                   document: learn() records the names established with strong evidence,
                   the places and organisations found in context, and the name fragments
                   of e-mail addresses; detect() then also flags bare occurrences of
                   those names ("Dupont a signé"), so that every mention of a person is
                   replaced, bare surnames included.

                4. User knowledge. Regular expressions added with add_pattern() and
                   terms added with add_term() describe what is specific to an
                   organisation; they take precedence over the scanners and the rules.
                   Existing pseudonymization tags and forced values keep precedence over
                   them.

                5. Learned recognizers added with add_recognizer(): pii_sequence_tagger,
                   pii_span_model or any function with the same signature. Their spans
                   join the candidates above the linguistic rules and below the
                   validated scanners; a given name and a family name side by side
                   become one person, a postal code and its city one place, and a
                   confident full name feeds the propagation of channel 3. A piece they
                   read inside an organisation known by its own name, or inside a postal
                   code of another country with its town, leaves that value whole ("Bahn"
                   in "Deutsche Bahn", "Tokyo 153" in "Tokyo 153-0063"); a span that goes
                   beyond it keeps its own boundaries.

                The detector is a recall-oriented filter. Its output is meant to be
                pseudonymized, and a residual-risk audit of the protected document is
                still advisable before release.

            THREAD SAFETY
                detect() is const and may run concurrently; learn(), begin_document()
                and the dictionary functions must not run concurrently with anything
                else.
        !*/
    public:
        explicit pii_detector(const pii_detector_options& options = pii_detector_options());
        /*!
            ensures
                - #pattern_count() == 0
                - #recognizer_count() == 0
                - #learned_name_count() == 0
        !*/

        void add_person_name(const std::string& name);
        /*!
            ensures
                - name and its words become known person names for every document
                  processed afterwards. Used for user dictionaries and for the authors
                  found in document metadata.
        !*/

        void add_term(pii_type type, const std::string& term);
        /*!
            ensures
                - Every occurrence of term is reported with the given type: its words and
                  numbers in the same order, case and accent insensitive, separated by at
                  most two characters ("EMP-4521", "Dupont Conseil"). A number is matched
                  as a whole: the term "12345" is not found in "A12345".
                - Does nothing if type is pii_type::none or term holds no word or number.
        !*/

        void add_allowed_term(const std::string& term);
        /*!
            ensures
                - Entities whose text equals term (case and accent insensitive) are never
                  reported: the organisation's own name, a public figure quoted in the
                  document, a product name.
        !*/

        void add_pattern(const pii_pattern& pattern);
        /*!
            ensures
                - Every match of pattern.regex (the capture group pattern.group of it),
                  passing pattern.check, is reported with pattern.type, above the
                  validated scanners.
                - Matching is done line by line, in windows of 2048 bytes that overlap by
                  512 bytes, which bounds the recursion and the backtracking of std::regex
                  on long paragraphs; '^', '$' and '\b' see the text around a window. A
                  match is reported once; a match longer than 512 bytes may be cut at the
                  end of a window.
                - #pattern_count() == pattern_count() + 1
            throws
                - error if pattern.type is none, the expression is invalid, the group does
                  not exist or the check is unknown. detect() and learn() throw error
                  naming the expression if std::regex fails while matching.
        !*/

        std::size_t pattern_count() const;
        /*!
            ensures
                - Returns the number of patterns added with add_pattern().
        !*/

        void add_recognizer(
            const pii_recognizer& recognizer,
            double min_confidence = 0.5
        );
        /*!
            ensures
                - recognizer is called on every text analysed afterwards, and its spans
                  whose confidence is at least min_confidence are added to the candidates.
                  A span whose offsets fall inside a UTF-8 character covers that character.
                - The pieces of one word, one name or one login that recognizer reports
                  side by side are merged first ("E" and "sch-sur-Alzette"; "Inès", "de"
                  and "Brunet"; "v" and "wojcikski"), and the at sign before a login joins
                  it ("@hpetit38"). A span that names nothing is then ignored: a piece of a
                  word that no other piece completes ("Rem" in "Remplacez"), a login inside
                  a longer token, a file name ("rapport_final.docx"), a single character or
                  a common word in lower case ("month") and, for persons,
                  places and organisations, function words alone ("Votre"), a greeting, a
                  closing or a field label alone ("Bonsoir", "Cordialement", "Nom"), a
                  common word in lower case ("commande"), an organisation opened by an
                  article and a word in lower case ("Le logiciel Excel"), a person or a
                  place that joins capitalized words to words in lower case other than
                  particles ("Le projet", while "Jean de La Fontaine" is kept), a word in
                  lower case, other than a first name, in a text that capitalizes its
                  words, a country alone read as an organisation ("Pologne"), the civility
                  "M." before a name ("avec M. Martin", while "Monsieur M. Denis" keeps
                  its initial), and a participle that opens a line before "à" ("Fait à
                  Lyon").
                - The calls are serialized and their results cached, keyed by the
                  SHA-256 digest of the text, until the next begin_document(): learn() and
                  detect() see the same texts and models are seldom reentrant, so
                  recognizer need not be thread safe.
                - #recognizer_count() == recognizer_count() + 1
        !*/

        std::size_t recognizer_count() const;
        /*!
            ensures
                - Returns the number of recognizers added with add_recognizer().
        !*/

        void begin_document();
        /*!
            ensures
                - Forgets what learn() recorded and the cached results of the
                  recognizers; dictionaries, patterns and recognizers are kept.
                - #learned_name_count() == 0
        !*/

        void learn(const std::string& text, const detection_context& context = detection_context());
        /*!
            ensures
                - First pass: records the names established with strong evidence in text,
                  the places and organisations found in context, and the fragments of the
                  e-mail addresses it contains.
            throws
                - error if a pattern fails while matching.
        !*/

        std::vector<pii_entity> detect(
            const std::string& text,
            const detection_context& context = detection_context()
        ) const;
        /*!
            ensures
                - Returns the entities of text, sorted by position, non-overlapping, with
                  byte offsets into text.
                - Spans that are pseudonymization tags ("[[KEYWORD:body#CCC]]" with a known
                  keyword and a check of three Crockford characters) are never reported,
                  nor anything inside them; brackets that only look like a tag are read
                  like any text.
            throws
                - error if a pattern fails while matching.
        !*/

        pii_type label_type(const std::string& text) const;
        /*!
            ensures
                - If text is a field label on its own ("Nom", "Date de naissance :",
                  "E-mail", "N° allocataire", "Phone No.", "first_name"), returns the type
                  of the values it introduces, otherwise pii_type::none. Document handlers
                  use it on table cells and column headers.
        !*/

        std::size_t learned_name_count() const;
        /*!
            ensures
                - Returns the number of full names learn() recorded since the last
                  begin_document().
        !*/
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_PII_DETECTOR_ABSTRACT_H_

