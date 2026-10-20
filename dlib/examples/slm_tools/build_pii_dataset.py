#!/usr/bin/env python3
#
# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
#
# Synthetic training data for the learned personal data recognizers of dlib/privacy.
#
# Writes annotated French and English texts in the format read by load_pii_dataset():
# one example per line, the text with \\, \t, \n and \r escaped, a tab, then the spans
# as begin:end:LABEL items, byte offsets into the UTF-8 text. The two languages come in
# equal numbers unless --fr-share sets the share of French.
#
# WHY SYNTHETIC DATA
#
# Annotated personal data cannot be collected without collecting personal data, which the
# recognizers exist to avoid. Every value written here is drawn from
# lexicons of common names and places, or made up by the procedural generators below,
# and combined at random: no line describes a real person. The generator also controls
# what no annotated corpus offers, an exact and consistent annotation of every span and
# a measured share of the difficulties met in real documents.
#
# WHAT IS ANNOTATED
#
# Two kinds of labels are written. The fine labels are those the models learn:
#
#   GIVEN_NAME      given names, compound (Jean-Pierre) or initial (J.), several given
#                   names in a row forming one span
#   FAMILY_NAME     family names with their particles (de La Fontaine, Le Goff), birth
#                   names after "née", double-barrelled names
#   USERNAME        logins, handles and account names (jdupont, @jdupont35)
#   STREET_ADDRESS  the street line (12 bis rue Victor Hugo) and each address complement
#                   (Bâtiment C, BP 1234, Lieu-dit Kerbrat) as a span of its own
#   POSTAL_CODE     the postal code of an address
#   CITY            every town or village, wherever it appears: like the Safe Harbor rule
#                   of HIPAA, any place smaller than a region is replaced
#   ORGANIZATION    named organisations: companies, practices, associations, schools,
#                   clinics, local authorities, and public bodies named by their name or
#                   acronym (la CAF, l'URSSAF, France Travail, le ministère de la Justice,
#                   the NHS, HMRC). A generic reference (la préfecture, le tribunal, the
#                   council) is not labelled.
#
# Professional titles (Dr, Maître), job titles, regions, departments and countries are
# never part of a span. The other categories, which the validated scanners and the rules
# already find, carry their tag keyword: EMAIL, PHONE, DATE, IBAN, CARD, NIR, SIRET, ID
# (references, accounts, devices and the national numbers of other countries), IPV4,
# IPV6, MAC, URL, PLATE, GEO, TITLE (a civility before a name or after its label: M.,
# Mme, Mrs), GENDER (the value of a sex field) and NATIONALITY. A DATE is a calendar date,
# a month with its year ("mars 2022") or a year that dates a fact of a person's life ("a
# rejoint Kerlys en 2019"), not the year of a report, a plan or a file name. Models ignore
# these labels when they train; they make the annotation complete, so that the whole
# detection system, rules and models together, can be scored.
#
# HOW TEXTS ARE MADE
#
# A template is a line of text with slots. {GIVEN#1} and {FAMILY#1} name person 1,
# {CITY#2} and {STREET#2} describe place 2, {ORG#1} names organisation 1: an identifier
# shared by several slots keeps the same entity, so that a document mentions "Jean
# Dupont" and then "M. Dupont". {g#1:né|née} agrees with the gender of person 1,
# {GENDER:f#2} sets the gender of person 2 when the text implies one ("Madame
# {FAMILY#2}"), and {?a|b|} picks one alternative, possibly empty. An unknown slot stops
# the generator with the file and line of its template. The words written before a slot
# agree with the value drawn: "de {CITY#1}" gives "d'Orléans" or "du Mans", the span
# then covering "Mans", and "a {JOB}" gives "an engineer". The templates are grouped by
# genre (forms, letters, e-mails, contracts, medical reports, chats, logs, tables...) and
# the genres weighted to resemble the documents an organisation holds.
#
# {TITLE#1} writes the civility of person 1 (M., Mme, Mr, Mrs), {TITLE:short#1} and
# {TITLE:long#1} its abbreviated and full forms (Monsieur, Madam); a civility keeps its
# TITLE label before a name, beside another civility ("M. et Mme") and after a field
# label, where the detector replaces it. {SEX#1} writes the value of a sex field (F,
# Masculin, Female), {NATIONALITY#1} the nationality of person 1 agreeing with its
# gender, and {NATID#1} the label and the national number of its country with a valid
# check digit; {NATID:de#1} sets that country. {ORG:kind#2} draws organisation 2 of a
# kind (company, law, public, council, prefecture, court, police), and {ART#2}, or
# {Art#2} capitalized, writes the article of a public body: "{ART#2}{ORG:council#2}"
# gives "la Mairie de Brest". {INSTITUTION} draws a reference to a public body from its
# lexicon, the name between brackets being labelled: "la [CAF]", "the [NHS]". {NICK#1}
# writes the nickname of person 1 ("Jojo", "Smithy") and {NICK:stage#1} a stage name ("DJ
# Kerlys"), both labelled GIVEN_NAME; the variant abroad of the place slots, {CITY:abroad#2}
# or {ADDRESS:abroad#2}, puts place 2 in another country. {YEAR} writes a year left
# unlabelled (a report, a file name, a plan); {YEAR:date} a year that dates a fact of a
# person's life ("a rejoint {ORG#1} en {YEAR:date}"), labelled DATE together with the
# month name written just before it ("Bulletin de paie – mars 2022").
#
# Names are drawn from lexicons by origin and gender, or synthesized from the
# morphology of the family names of about twenty naming traditions (French, Breton,
# Alsatian, Occitan, Italian, Iberian, Polish, English, Gaelic, German, Dutch, Nordic,
# East Slavic, Greek, Turkish, Arabic, Persian, Armenian, South Asian, Japanese and
# Nigerian), so that a model learns what a name looks like and where it appears rather
# than memorizing a list. Places and organisation names follow the same two routes;
# streets come from their lexicon. Some organisations are known by their own name: a
# share of the companies, banks, universities, hospitals and associations is drawn from
# known_orgs.txt ("BNP Paribas", "Siemens Healthineers"), while the names of products and
# applications stay unlabelled (products.txt).
#
# PLACES ABROAD
#
# A share of the places (--abroad-share) lies in another country, about fifty of them, with
# the street line, postal code and city line of that country: "Hauptstraße 12, 10115
# Berlin", "ul. Długa 5, 00-238 Warszawa", "Toronto, ON M5H 2N2", "2-11-3 Meguro, Tokyo
# 153-0063". The country written after the city stays unlabelled, like a region. The
# organisations of such a place take the legal forms and the names of that country
# ("Kerlys GmbH", "Ayuntamiento de Sevilla", "Novexa Pvt. Ltd."), and the same share of
# telephone numbers and IBANs belongs to other countries. A person whose name comes from
# another tradition often has an e-mail address at a provider of that country.
#
# NOISE
#
# Real inputs come out of PDF and Office extraction, old databases and chats. Each
# example draws a style: family names in capitals, a whole text in capitals or in lower
# case, accents stripped, no-break spaces before French colons, doubled spaces, and the
# damage of optical character recognition (a lost space, rn read as m, l as 1). The
# spans follow every transformation exactly, since the text is assembled from labelled
# fragments and the offsets computed at the end.
#
# SPLITS
#
# Every lexicon entry belongs either to the seen part or, for a share given by
# --unseen-percent, to the held-out part, decided by a hash of its value. The values of
# the procedural generators are split by the same hash. train and dev draw from the seen
# part only. test draws from both; test_unseen from the held-out part only, which
# measures how the models read names, places and organisations they never saw during
# training. test_noisy applies heavy noise, test_negatives holds texts full of
# capitalized words that are not personal data, test_documents longer texts of several
# paragraphs, and test_abroad texts whose places, organisations, phone numbers and IBANs
# all belong to other countries (--abroad-share 1 for that split). No evaluation split
# repeats a text of train: an example drawn again is replaced by another.
#
# CONTEXT
#
# Real paragraphs open with "Le dossier", "La réunion" or "Bonjour," far more often than
# with a name. Plain sentences without personal data (sentences.txt) are written around a
# share of the examples (--context-rate), before or after them, and form examples of their
# own (--prose-share), so that the models do not read every capitalized word that opens a
# text as a name. The sentences may use the filler slots and the choices of templates.
#
# RESOURCES
#
# A small resource set is built into this script so that it runs on its own. A larger
# set, lexicons and templates in plain text files, can be given with --resources; files
# found there replace the built-in ones of the same name:
#
#   <lang>/given_female.txt, <lang>/given_male.txt   name <TAB> origin [<TAB> weight]
#   <lang>/family.txt                                name <TAB> origin [<TAB> weight]
#   <lang>/cities.txt                                city <TAB> postal code [<TAB> weight]
#   <lang>/streets.txt, <lang>/street_types.txt      one per line [<TAB> weight]
#   <lang>/org_words.txt, <lang>/jobs.txt, <lang>/regions.txt,
#   <lang>/countries.txt, <lang>/products.txt        one per line [<TAB> weight]
#   <lang>/institutions.txt                          one per line [<TAB> weight], the name of a
#                                                    named body between brackets: "la [CAF]"
#   <lang>/negatives.txt                             phrase <TAB> category [<TAB> weight]
#   <lang>/sentences.txt                             sentence [<TAB> weight], no personal data
#   <lang>/templates.txt                             genre <TAB> weight <TAB> template
#   <lang>/world_cities.txt                          city <TAB> postal code <TAB> country code
#                                                    [<TAB> weight], the state or province first
#                                                    in Canada, Australia, Brazil, Mexico and the US
#   <lang>/world_streets.txt                         street <TAB> country code [<TAB> weight]
#   <lang>/known_orgs.txt                            name <TAB> category [<TAB> weight]: company,
#                                                    retail, bank, insurer, media, airline,
#                                                    transport, telecom, energy, university,
#                                                    hospital, ngo, club
#   <lang>/nicknames.txt                             nickname <TAB> f, m or x [<TAB> weight]
#
# Usage:
#   build_pii_dataset.py --out-dir pii-data
#   build_pii_dataset.py --out-dir pii-data --resources path/to/resources \
#       --train 60000 --seed 7
#   build_pii_dataset.py --sample 5            print five examples with their spans

import argparse
import bisect
import hashlib
import json
import os
import random
import re
import sys
import unicodedata
from typing import Dict, List, Optional, Tuple

MODEL_LABELS = ["GIVEN_NAME", "FAMILY_NAME", "USERNAME", "STREET_ADDRESS", "POSTAL_CODE", "CITY", "ORGANIZATION"]
RULE_LABELS = ["EMAIL", "PHONE", "DATE", "IBAN", "CARD", "NIR", "SIRET", "ID", "IPV4", "IPV6", "MAC", "URL",
               "PLATE", "GEO", "TITLE", "GENDER", "NATIONALITY"]

LABEL_DESCRIPTIONS = {
    "GIVEN_NAME": "given names of a natural person",
    "FAMILY_NAME": "family names, birth names, particles included",
    "USERNAME": "logins, handles and account names",
    "STREET_ADDRESS": "street line or address complement",
    "POSTAL_CODE": "postal code of an address",
    "CITY": "town, village or locality",
    "ORGANIZATION": "named organisation",
    "EMAIL": "e-mail address",
    "PHONE": "telephone or fax number",
    "DATE": "calendar date, month with its year, or year that dates a fact of a person's life",
    "IBAN": "international bank account number",
    "CARD": "payment card number",
    "NIR": "French social security number",
    "SIRET": "French company identifier SIREN or SIRET",
    "ID": "reference, account, file, device or foreign national identifier",
    "IPV4": "IPv4 address",
    "IPV6": "IPv6 address",
    "MAC": "MAC address",
    "URL": "link to a personal profile",
    "PLATE": "vehicle registration plate",
    "GEO": "GPS coordinates",
    "TITLE": "civility before a name or after its label",
    "GENDER": "value of a sex or gender field",
    "NATIONALITY": "nationality of a person",
}


# ----------------------------------------------------------------------------------
# Built-in resources
#
# A small set, enough to exercise every slot and every genre. The values are common first
# names, family names and places; the organisation words are made up.

# The built-in places abroad: a few cities and streets of each country of WORLD, with the
# layout of their postal codes.
WORLD_CITIES_BUILTIN = """
Berlin	10115	de
München	80331	de
Hamburg	20095	de
Wien	1010	at
Graz	8010	at
Zürich	8001	ch
Genève	1204	ch
Luxembourg	L-1611	lu
Bruxelles	1000	be
Liège	4000	be
Gent	9000	be
Amsterdam	1015 CJ	nl
Rotterdam	3011 AD	nl
Madrid	28013	es
Sevilla	41004	es
Barcelona	08002	es
Lisboa	1100-053	pt
Porto	4000-322	pt
Roma	00184	it
Milano	20121	it
Napoli	80134	it
Warszawa	00-238	pl
Kraków	31-008	pl
Praha	110 00	cz
Brno	602 00	cz
Stockholm	111 51	se
Göteborg	411 06	se
København	1050	dk
Oslo	0154	no
Helsinki	00100	fi
Dublin	D02 X285	ie
Cork	T12 W8KV	ie
Athina	105 57	gr
Thessaloniki	546 24	gr
București	010061	ro
Cluj-Napoca	400001	ro
Budapest	1061	hu
Casablanca	20000	ma
Rabat	10000	ma
Marrakech	40000	ma
Alger	16000	dz
Oran	31000	dz
Tunis	1000	tn
Sfax	3000	tn
Dakar		sn
Abidjan		ci
Douala		cm
Montréal	QC H2X 1Z5	ca
Toronto	ON M5H 2N2	ca
Vancouver	BC V6B 1A1	ca
New York	NY 10001	us
San Francisco	CA 94103	us
London	SW1A 1AA	gb
Manchester	M1 1AE	gb
Sydney	NSW 2000	au
Melbourne	VIC 3000	au
Auckland	1010	nz
Mumbai	400001	in
Bengaluru	560001	in
Tokyo	150-0002	jp
Osaka	530-0001	jp
Shanghai	200001	cn
Beijing	100006	cn
Seoul	04524	kr
Singapore	049315	sg
São Paulo	SP 01001-000	br
Rio de Janeiro	RJ 20040-002	br
Ciudad de México	CDMX 06600	mx
Guadalajara	JAL 44100	mx
Buenos Aires	C1043	ar
İstanbul	34430	tr
Ankara	06420	tr
Tel Aviv	6473921	il
Dubai		ae
Beyrouth		lb
Cairo	11511	eg
Lagos	101233	ng
Nairobi	00100	ke
Johannesburg	2196	za
Cape Town	8001	za
Moskva	125009	ru
Kyiv	01001	ua
"""

WORLD_STREETS_BUILTIN = """
Hauptstraße	de
Bahnhofstraße	de
Mariahilfer Straße	at
Bahnhofstrasse	ch
Rue du Rhône	ch
Avenue de la Liberté	lu
Rue de la Loi	be
Kerkstraat	be
Keizersgracht	nl
Calle Mayor	es
Gran Vía	es
Rua Augusta	pt
Via Roma	it
Via Garibaldi	it
ul. Marszałkowska	pl
Václavské náměstí	cz
Drottninggatan	se
Vesterbrogade	dk
Karl Johans gate	no
Mannerheimintie	fi
O'Connell Street	ie
Ermou	gr
Calea Victoriei	ro
Andrássy út	hu
boulevard Zerktouni	ma
rue Mohammed V	ma
rue Didouche Mourad	dz
avenue Habib Bourguiba	tn
avenue Cheikh Anta Diop	sn
boulevard Latrille	ci
boulevard de la Liberté	cm
rue Sainte-Catherine	ca
Yonge Street	ca
Main Street	us
Broadway	us
High Street	gb
George Street	au
Queen Street	nz
MG Road	in
Shibuya	jp
Nanjing Road	cn
Teheran-ro	kr
Orchard Road	sg
Avenida Paulista	br
Paseo de la Reforma	mx
Avenida Corrientes	ar
İstiklal Caddesi	tr
Dizengoff Street	il
Sheikh Zayed Road	ae
rue Hamra	lb
Tahrir Street	eg
Awolowo Road	ng
Kenyatta Avenue	ke
Long Street	za
Tverskaya ulitsa	ru
vulytsia Khreshchatyk	ua
"""

KNOWN_ORGS_BUILTIN = """
BNP Paribas	bank
Crédit Agricole	bank
Deutsche Bank	bank
AXA	insurer
Allianz	insurer
Airbus	company
Siemens Healthineers	company
Capgemini	company
Air France	airline
Lufthansa	airline
SNCF	transport
Vodafone	telecom
EDF	energy
Reuters	media
Carrefour Market	retail
Universidad de Salamanca	university
Imperial College London	university
Institut Curie	hospital
Médecins sans frontières	ngo
Croix-Rouge française	ngo
Real Madrid	club
"""

NICKNAMES_BUILTIN = {"fr": """
Jojo	x
Mimi	f
Lulu	x
Titi	x
Pierrot	m
Dédé	m
Fanfan	x
Nono	m
Zizou	m
Loulou	x
""", "en": """
Smithy	m
Jonno	m
Ginger	x
Lofty	m
Dusty	x
Sully	m
Bunny	f
Trixie	f
Buddy	m
Red	x
"""}


BUILTIN = {
    "fr/given_female.txt": """
Marie fr\nNathalie fr\nIsabelle fr\nSylvie fr\nCatherine fr\nChristine fr\nSophie fr\nValérie fr
Sandrine fr\nCéline fr\nJulie fr\nCamille fr\nLéa fr\nManon fr\nChloé fr\nEmma fr\nInès fr\nJade fr
Louise fr\nAnne-Sophie fr\nMarie-Claire fr\nHélène fr\nBrigitte fr\nFrançoise fr\nMonique fr\nOcéane fr
Maëlle fr\nLaëtitia fr\nAgnès fr\nRose fr\nFatima maghreb\nYasmina maghreb\nAïcha west_africa\nAna iberia
Giulia italy\nAgnieszka poland\nLinh vietnam\nMei china\nElif turkey
""",
    "fr/given_male.txt": """
Jean fr\nPierre fr\nMichel fr\nPhilippe fr\nAlain fr\nNicolas fr\nChristophe fr\nLaurent fr\nFrédéric fr
Thomas fr\nJulien fr\nLucas fr\nHugo fr\nLouis fr\nGabriel fr\nArthur fr\nJean-Pierre fr\nJean-Luc fr
François fr\nGérard fr\nBernard fr\nYves fr\nGaël fr\nLoïc fr\nMathéo fr\nClaude fr\nMohamed maghreb
Karim maghreb\nMamadou west_africa\nJoão iberia\nMarco italy\nPiotr poland\nMinh vietnam\nWei china
Mehmet turkey
""",
    "fr/family.txt": """
Martin fr\nBernard fr\nDubois fr\nThomas fr\nRobert fr\nRichard fr\nPetit fr\nDurand fr\nLeroy fr
Moreau fr\nSimon fr\nLaurent fr\nLefebvre fr\nMichel fr\nGarcia iberia\nDavid fr\nBertrand fr\nRoux fr
Vincent fr\nFournier fr\nMorel fr\nGirard fr\nAndré fr\nLefèvre fr\nMercier fr\nDupont fr\nLambert fr
Bonnet fr\nFrançois fr\nMartinez iberia\nLe Goff breton\nLe Gall breton\nKerjean breton\nGuillou breton
Schmitt alsace\nMuller alsace\nde La Fontaine fr\nBoulanger fr\nChevalier fr\nBenali maghreb
Diallo west_africa\nPereira iberia\nRossi italy\nKowalski poland\nNguyen vietnam\nWang china
Yılmaz turkey
""",
    "fr/cities.txt": """
Paris\t75001\nMarseille\t13001\nLyon\t69001\nToulouse\t31000\nNice\t06000\nNantes\t44000
Strasbourg\t67000\nMontpellier\t34000\nBordeaux\t33000\nLille\t59000\nRennes\t35000\nBrest\t29200
Quimper\t29000\nVannes\t56000\nLorient\t56100\nSaint-Malo\t35400\nSaint-Brieuc\t22000\nLe Mans\t72000
Angers\t49000\nTours\t37000\nDijon\t21000\nGrenoble\t38000\nLimoges\t87000\nClermont-Ferrand\t63000
La Roche-sur-Yon\t85000\nAix-en-Provence\t13100\nSaint-Étienne\t42000\nPerpignan\t66000
Fougères\t35300\nVitré\t35500\nLannion\t22300\nPloërmel\t56800\nConcarneau\t29900\nDinan\t22100
Châteaugiron\t35410\nBruz\t35170\nCesson-Sévigné\t35510\nBruxelles\t1000\nGenève\t1201
""",
    "fr/streets.txt": """
Victor Hugo\nJean Jaurès\ndu Général de Gaulle\nPasteur\nde la République\nde la Gare\nde l'Église
du Moulin\ndes Lilas\ndes Chênes\nSaint-Michel\nde Paris\ndu 8 Mai 1945\nde la Paix\ndes Tanneurs
Gambetta\nde Brest\ndu Port\ndes Écoles\nJules Ferry\nde la Liberté\ndu Château\ndes Acacias
""",
    "fr/street_types.txt": """
rue\t10\navenue\t3\nboulevard\t2\nimpasse\t2\nallée\t2\nchemin\t1\nplace\t1\nquai\t1\nroute\t1
""",
    "fr/org_words.txt": """
Altrys\nNovexa\nBriovia\nKerlys\nZentiq\nArmoris\nVelatis\nOptimea\nSolvane\nTerranis
""",
    "fr/institutions.txt": """
la [CAF]\nl'[URSSAF]\nla [Sécurité sociale]\nl'[Assurance maladie]\n[France Travail]\nla [CPAM]
le [Trésor public]\nl'administration fiscale\nle [ministère de la Justice]\nla préfecture\nle tribunal judiciaire
la [Banque de France]
""",
    "fr/jobs.txt": """
comptable\ninfirmière\ndéveloppeur\nresponsable des ressources humaines\ndirecteur commercial
assistante de direction\ntechnicien de maintenance\nchef de projet\nmédecin du travail\njuriste
""",
    "fr/regions.txt": """
Bretagne\nNormandie\nOccitanie\nÎle-de-France\nIlle-et-Vilaine\nFinistère\nMorbihan\nCôtes-d'Armor
""",
    "fr/countries.txt": """
France\nBelgique\nSuisse\nEspagne\nItalie\nAllemagne\nPortugal\nMaroc\nSénégal\nCanada
""",
    "fr/negatives.txt": """
la Pierre philosophale\nle Conseil d'administration\nl'Assemblée générale\nle Code du travail
le Code civil\nla Direction générale\nle Comité social et économique\nla Toussaint\nNoël
le Président\nle Journal officiel\nla Saint-Valentin
""",
    "fr/products.txt": """
Excel\nWord\nWindows\nOutlook\nTeams\nSAP\nSalesforce\nLinux\nAndroid\niPhone
""",
    "fr/sentences.txt": r"""
Le dossier est complet et sera transmis au service concerné.
La réunion de suivi est reportée à {?jeudi|vendredi|la semaine prochaine}.
Merci de compléter le formulaire avant la date limite.
Veuillez trouver ci-joint le compte rendu de la réunion.
Les congés sont validés par le responsable du service.
Votre commande a été expédiée et sera livrée sous 48 heures.
Il a été convenu de revoir le planning en fin de mois.
Nous vous remercions pour votre retour rapide.
Le logiciel {PRODUCT} doit être mis à jour sur tous les postes.
Le projet « {?Horizon|Atlas|Phénix|Boréal} » entre dans sa deuxième phase.
Remplacez <Nom> et <Prénom> par vos informations avant l'envoi.
Le champ <adresse> est obligatoire.
Le fichier {?rapport_final|compte_rendu|planning_v2|devis_2024}.{?docx|xlsx|pdf} est joint à ce message.
Bonjour, je reviens vers vous au sujet de votre demande.
Bonsoir, merci pour ces précisions.
Cordialement,
Bien à vous,
Pour toute question, contactez le support.
La procédure est décrite dans le guide « Bonnes pratiques ».
Selon la note de service, l'accès au bâtiment est limité.
Ce document est strictement confidentiel.
Point d'étape : les tests sont terminés, la recette commence lundi.
À noter : la mise en production est prévue en {?mars|juin|octobre}.
Il faut prévoir une marge sur le budget.
Les résultats seront présentés en comité.
""",
    "fr/templates.txt": r"""
form	4	Nom : {FAMILY#1}\nPrénom : {GIVEN#1}\nDate de naissance : {BIRTHDATE#1}\nLieu de naissance : {CITY#2}\nAdresse : {STREET#3}\n{CITYLINE#3}\nTéléphone : {PHONE#1}\nCourriel : {EMAIL#1}
form	2	{TITLE#1} {PERSON#1}, {g#1:né|née} le {BIRTHDATE#1} à {CITY#2}, {g#1:domicilié|domiciliée} {ADDRESS#3}.
form	1	Civilité : {TITLE#1}\nNom : {FAMILY#1}\nPrénom : {GIVEN#1}\nSexe : {SEX#1}\nNationalité : {NATIONALITY#1}\nPièce d'identité : {NATID#1}
admin	1	{TITLE#1} {PERSON#1}, de nationalité {NATIONALITY#1}, a déposé sa demande auprès de {INSTITUTION} et de {ART#2}{ORG:public#2}.
letter	3	{PERSON#1}\n{STREET#2}\n{CITYLINE#2}\n\n{ORG#1}\nService clients\n{STREET#3}\n{CITYLINE#3}\n\nObjet : résiliation du contrat n° {REF}\n\n{CITY#2}, le {DATE}\n\nMadame, Monsieur,\nJe vous prie de bien vouloir résilier mon abonnement.\n\n{PERSON:signature#1}
email	3	De : {PERSON:signature#1} <{EMAIL#1@1}>\nÀ : {PERSON:signature#2} <{EMAIL#2@1}>\nObjet : {?point d'avancement|réunion de lundi|dossier en cours}\n\nBonjour {GIVEN#2},\n\nPeux-tu transmettre le document à {TITLE#3} {FAMILY#3} avant {?jeudi|vendredi|la fin de semaine} ?\n\nCordialement,\n{PERSON:signature#1}\n{JOB}\n{ORG#1}\nTél. : {PHONE#1}
hr	2	Entre les soussignés :\nLa société {ORG#1}, SIRET {SIRET#1}, dont le siège est situé {ADDRESS#1}, représentée par {TITLE#1} {PERSON#1} en qualité de {JOB},\nEt {TITLE#2} {PERSON#2}, {g#2:né|née} le {BIRTHDATE#2} à {CITY#2}, demeurant {ADDRESS#3}, numéro de sécurité sociale {NIR#2},\nci-après dénommé{g#2:|e} « le salarié ».
medical	2	Patient{g#1:|e} : {TITLE#1} {PERSON:formal#1}, {g#1:né|née} le {BIRTHDATE#1}\nMédecin traitant : Dr {PERSON#2}, {CITY#3}\n{TITLE#1} {FAMILY#1} est {g#1:adressé|adressée} pour un bilan. Antécédents sans particularité.
crm	2	Appel de {TITLE#1} {FAMILY#1} le {DATE} : souhaite modifier son RIB ({IBAN}). Rappeler au {PHONE#1}. Dossier client {REF}.
chat	2	[{TIME}] {USERNAME#1}: salut {GIVEN#2}, tu as vu le mail de {GIVEN#3} {FAMILY#3} ?\n[{TIME}] {USERNAME#2}: oui, je l'appelle cet après-midi
log	2	{DATE:iso} {TIME} INFO login ok user={USERNAME#1} ip={IP}\n{DATE:iso} {TIME} WARN password expired user={USERNAME#1}
table	2	NOM;PRÉNOM;DATE DE NAISSANCE;VILLE;TÉLÉPHONE\n{FAMILY:upper#1};{GIVEN#1};{BIRTHDATE#1};{CITY#1};{PHONE#1}\n{FAMILY:upper#2};{GIVEN#2};{BIRTHDATE#2};{CITY#2};{PHONE#2}
narrative	2	{PERSON#1}, {AGE} ans, habitant de {CITY#1}, a rejoint {ORG#1} en {YEAR:date}. {Il#1} travaille avec {PERSON#2} sur le projet {PRODUCT}.
legal	1	Par devant Maître {PERSON#1}, notaire à {CITY#1}, a comparu {TITLE#2} {PERSON:formal#2}, {g#2:né|née} à {CITY#3} le {BIRTHDATE#2}, demeurant {ADDRESS#4}.
negative	2	{NEGATIVE} a validé la proposition le {DATE}. La réunion avec {INSTITUTION} aura lieu en {REGION}.
negative	1	Le {JOB} doit compléter le formulaire dans {PRODUCT} avant la fin du mois, selon {NEGATIVE}.
""",
    "en/given_female.txt": """
Mary anglo\nPatricia anglo\nJennifer anglo\nLinda anglo\nElizabeth anglo\nSarah anglo\nEmily anglo
Olivia anglo\nSophie anglo\nCharlotte anglo\nGrace anglo\nPriya south_asia\nSiobhan irish
""",
    "en/given_male.txt": """
James anglo\nJohn anglo\nRobert anglo\nMichael anglo\nWilliam anglo\nDavid anglo\nRichard anglo
Thomas anglo\nOliver anglo\nHarry anglo\nGeorge anglo\nRajesh south_asia\nSean irish
""",
    "en/family.txt": """
Smith anglo\nJohnson anglo\nWilliams anglo\nBrown anglo\nJones anglo\nTaylor anglo\nWilson anglo
Evans anglo\nThompson anglo\nWright anglo\nWalker anglo\nO'Brien irish\nMacDonald scottish\nPatel south_asia
""",
    "en/cities.txt": """
London\tSW1A 1AA\nManchester\tM1 1AE\nBirmingham\tB1 1AA\nLeeds\tLS1 4AP\nBristol\tBS1 4DJ
Edinburgh\tEH1 1YZ\nGlasgow\tG1 1XQ\nCardiff\tCF10 1EP\nBoston\tMA 02108\nChicago\tIL 60601\nSeattle\tWA 98101
Austin\tTX 78701\nDenver\tCO 80202\nDublin\tD02 X285
""",
    "en/streets.txt": """
High Street\nStation Road\nChurch Lane\nVictoria Road\nMain Street\nPark Avenue\nOak Drive
Mill Lane\nKing Street\nElm Street\nMaple Avenue\nWashington Street
""",
    "en/street_types.txt": "\n",
    "en/org_words.txt": """
Brightwell\nNorthgate\nKestrel\nHollis\nOakmere\nVantora\nCresthaven\nLumora
""",
    "en/institutions.txt": """
the [NHS]\n[HMRC]\nthe [IRS]\nthe [Social Security Administration]\nthe [DVLA]\nthe [Home Office]
the county court\nthe [Bank of England]\nthe [Supreme Court]
""",
    "en/jobs.txt": """
accountant\nnurse\nsoftware engineer\nHR manager\nsales director\nproject manager\nparalegal
""",
    "en/regions.txt": """
Yorkshire\nCalifornia\nTexas\nScotland\nWales\nOntario\nNew England
""",
    "en/countries.txt": """
United Kingdom\nIreland\nUnited States\nCanada\nAustralia\nIndia\nFrance
""",
    "en/negatives.txt": """
the Board of Directors\nthe Annual General Meeting\nthe Data Protection Act\nChristmas\nthe Chairman
Independence Day
""",
    "en/products.txt": """
Excel\nWord\nWindows\nOutlook\nSlack\nJira\nSalesforce\nAndroid
""",
    "en/sentences.txt": r"""
The file is complete and will be sent to the relevant team.
The follow-up meeting has been moved to {?Thursday|Friday|next week}.
Please fill in the form before the deadline.
Please find attached the minutes of the meeting.
Your order has been shipped and should arrive within two days.
We agreed to review the schedule at the end of the month.
Thank you for your quick reply.
The {PRODUCT} software must be updated on every workstation.
The "{?Phoenix|Atlas|Horizon|Northstar}" project enters its second phase.
Replace <Name> and <Address> with your details before sending.
The file {?final_report|meeting_notes|budget_v2}.{?docx|xlsx|pdf} is attached.
Hi, just following up on your request.
Hello, thanks for the update.
Best regards,
Kind regards,
For any question, please contact support.
This document is strictly confidential.
Status: testing is complete and acceptance starts on Monday.
The results will be presented to the committee.
""",
    "en/templates.txt": r"""
email	3	From: {PERSON:signature#1} <{EMAIL#1@1}>\nTo: {PERSON:signature#2} <{EMAIL#2@1}>\nSubject: {?project update|meeting notes|your request}\n\nHi {GIVEN#2},\n\nCould you send the file to {TITLE#3} {FAMILY#3} by {?Thursday|Friday|the end of the week}?\n\nBest regards,\n{PERSON:signature#1}\n{JOB}, {ORG#1}\nPhone: {PHONE#1}
form	3	Name: {PERSON#1}\nDate of birth: {BIRTHDATE#1}\nAddress: {ADDRESS#2}\nPhone: {PHONE#1}\nEmail: {EMAIL#1}
form	1	Title: {TITLE#1}\nSurname: {FAMILY#1}\nFirst name: {GIVEN#1}\nSex: {SEX#1}\nNationality: {NATIONALITY#1}\nIdentity document: {NATID#1}
admin	1	{TITLE#1} {PERSON#1}, a {NATIONALITY#1} citizen, wrote to {INSTITUTION} and to {ART#2}{ORG:public#2} about {his#1} claim.
letter	2	{PERSON#1}\n{STREET#2}\n{CITYLINE#2}\n\n{DATE}\n\nDear {TITLE#3} {FAMILY#3},\n\nThank you for your letter regarding account {REF}.\n\nYours sincerely,\n{PERSON:signature#1}
hr	2	This agreement is made between {ORG#1}, whose registered office is at {ADDRESS#1}, and {PERSON#2}, of {ADDRESS#2} (the "Employee"), born on {BIRTHDATE#2}.
medical	2	Patient: {PERSON:formal#1}, DOB {BIRTHDATE#1}. Referred by Dr {PERSON#2} of {ORG#1}. {He#1} reports chest pain since {DATE}.
chat	2	[{TIME}] {USERNAME#1}: hey {GIVEN#2}, did {GIVEN#3} reply?\n[{TIME}] {USERNAME#2}: not yet, I'll ping {HANDLE#3}
log	2	{DATE:iso} {TIME} INFO session started user={USERNAME#1} ip={IP}
narrative	2	{PERSON#1}, {AGE}, of {CITY#1}, joined {ORG#1} in {YEAR:date} and now works with {PERSON#2}.
negative	2	{NEGATIVE} approved the budget on {DATE}. The next review with {INSTITUTION} is planned in {REGION}.
""",
    "fr/world_cities.txt": WORLD_CITIES_BUILTIN,
    "en/world_cities.txt": WORLD_CITIES_BUILTIN,
    "fr/world_streets.txt": WORLD_STREETS_BUILTIN,
    "en/world_streets.txt": WORLD_STREETS_BUILTIN,
    "fr/known_orgs.txt": KNOWN_ORGS_BUILTIN,
    "en/known_orgs.txt": KNOWN_ORGS_BUILTIN,
    "fr/nicknames.txt": NICKNAMES_BUILTIN["fr"],
    "en/nicknames.txt": NICKNAMES_BUILTIN["en"],
}


# ----------------------------------------------------------------------------------
# Small utilities


def strip_accents(s: str) -> str:
    out = []
    for ch in unicodedata.normalize("NFD", s):
        if unicodedata.category(ch) != "Mn":
            out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def ascii_fold(s: str) -> str:
    s = strip_accents(s).replace("œ", "oe").replace("Œ", "OE").replace("æ", "ae").replace("ß", "ss")
    s = s.replace("ı", "i").replace("ł", "l").replace("Ł", "L").replace("ø", "o")
    return s


def held_out(value: str, percent: int) -> bool:
    h = hashlib.sha1(("pii-split|" + value).encode("utf-8")).hexdigest()
    return int(h[:8], 16) % 100 < percent


def luhn_digit(payload: str) -> int:
    total, double = 0, True
    for ch in reversed(payload):
        v = int(ch)
        if double:
            v *= 2
            if v > 9:
                v -= 9
        total += v
        double = not double
    return (10 - total % 10) % 10


def mod97(s: str) -> int:
    r = 0
    for ch in s:
        r = (r * 10 + int(ch)) % 97 if ch.isdigit() else (r * 100 + ord(ch.upper()) - 55) % 97
    return r


class Lexicon:
    """Weighted entries with a seen and a held-out part.

    Each entry is a tuple of fields, the value first. pick() draws from the part the
    current split may use. A condition that no entry of that part meets is dropped
    rather than the part, so that a held-out entry never reaches the training splits;
    only a part left empty by a very small lexicon falls back on the whole lexicon.
    """

    def __init__(self, entries: List[Tuple[Tuple[str, ...], float]], unseen_percent: int):
        self.entries = entries
        self.seen = [e for e in entries if not held_out(e[0][0], unseen_percent)]
        self.unseen = [e for e in entries if held_out(e[0][0], unseen_percent)]
        # The cumulative weights of the parts, and of the parts filtered under a key.
        self.cache = {}

    def __len__(self):
        return len(self.entries)

    def pick(self, rng: random.Random, pool: str, where=None, key=None) -> Optional[Tuple[str, ...]]:
        """An entry drawn by weight. A condition given with a key keeps its filtered pool, so
        that drawing again under the same key costs a binary search."""
        cached = self.cache.get((pool, key)) if where is None or key is not None else None
        if cached is None:
            part = self.entries if pool == "all" else (self.unseen if pool == "unseen" else self.seen)
            if not part:
                part = self.entries
            source = part
            if where is not None:
                source = [e for e in part if where(e[0])] or part
            if not source:
                return None
            cumulative, total = [], 0.0
            for _, w in source:
                total += w
                cumulative.append(total)
            cached = (source, cumulative)
            if where is None or key is not None:
                self.cache[(pool, key)] = cached
        source, cumulative = cached
        if not source or cumulative[-1] <= 0:
            return source[-1][0] if source else None
        k = bisect.bisect_left(cumulative, rng.random() * cumulative[-1])
        return source[min(k, len(source) - 1)][0]


def parse_lexicon(text: str, data_fields: int = 1) -> List[Tuple[Tuple[str, ...], float]]:
    """One entry per line, fields separated by tabs, or by a single space when the
    line holds a value and a one word origin tag ("Marie fr"). The field that follows
    the data_fields first ones, if any, is a weight."""
    entries = []
    for raw in text.splitlines():
        line = raw.strip("\r")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if "\t" in line:
            fields = [f.strip() for f in line.split("\t")]
        else:
            parts = line.strip().rsplit(" ", 1)
            tags = {"fr", "maghreb", "west_africa", "iberia", "italy", "poland", "vietnam", "china", "turkey",
                    "breton", "alsace", "occitan", "anglo", "irish", "scottish", "south_asia", "armenia",
                    "germanic", "portugal", "corsica", "basque", "east_europe", "middle_east", "caribbean"}
            fields = parts if len(parts) == 2 and parts[1] in tags else [line.strip()]
        weight = 1.0
        if len(fields) > data_fields:
            try:
                weight = float(fields[data_fields])
            except ValueError:
                raise ValueError("bad weight in lexicon line: %r" % line)
            if weight < 0:
                raise ValueError("negative weight in lexicon line: %r" % line)
            fields = fields[:data_fields]
        entries.append((tuple(fields), weight))
    return entries


class Resources:
    """The lexicons and templates of the languages being generated."""

    NAMES = ["given_female.txt", "given_male.txt", "family.txt", "cities.txt", "streets.txt", "street_types.txt",
             "org_words.txt", "institutions.txt", "jobs.txt", "regions.txt", "countries.txt", "negatives.txt",
             "products.txt", "sentences.txt", "templates.txt", "world_cities.txt", "world_streets.txt",
             "known_orgs.txt", "nicknames.txt"]
    # The number of data fields of each lexicon, the fields after them being a weight.
    FIELDS = {"given_female.txt": 2, "given_male.txt": 2, "family.txt": 2, "cities.txt": 2, "negatives.txt": 2,
              "world_cities.txt": 3, "world_streets.txt": 2, "known_orgs.txt": 2, "nicknames.txt": 2}

    def __init__(self, directory: Optional[str], languages: List[str], unseen_percent: int):
        self.lex: Dict[str, Dict[str, Lexicon]] = {}
        self.templates: Dict[str, List[Tuple[str, float, list]]] = {}
        self.sentences: Dict[str, List[Tuple[float, list]]] = {}
        self.origin = {}
        self.unseen_percent = unseen_percent
        if directory and not os.path.isdir(directory):
            raise ValueError("resource directory not found: %s" % directory)
        for lang in languages:
            self.lex[lang] = {}
            for name in self.NAMES:
                text = None
                if directory:
                    path = os.path.join(directory, lang, name)
                    if os.path.isfile(path):
                        with open(path, "r", encoding="utf-8") as f:
                            text = f.read()
                        # Relative to the resource directory: the card says which files were
                        # used, not where they live on the machine that built it.
                        self.origin[lang + "/" + name] = os.path.join(
                            os.path.basename(os.path.normpath(directory)), lang, name)
                if text is None:
                    text = BUILTIN.get(lang + "/" + name, "")
                    self.origin[lang + "/" + name] = "built-in"
                if name == "templates.txt":
                    self.templates[lang] = parse_templates(text, lang + "/" + name)
                    if not self.templates[lang]:
                        raise ValueError("%s/%s holds no template" % (lang, name))
                elif name == "sentences.txt":
                    self.sentences[lang] = parse_sentences(text, lang + "/" + name)
                else:
                    self.lex[lang][name[:-4]] = Lexicon(parse_lexicon(text, self.FIELDS.get(name, 1)), unseen_percent)


# ----------------------------------------------------------------------------------
# Templates
#
# A template is parsed once into a list of nodes: ("text", s), ("slot", name, variant,
# ident, extra) and ("choice", [alternatives]). Rendering a node yields fragments, pairs
# of a string and a label (None outside personal data).


def parse_templates(text: str, origin: str) -> List[Tuple[str, float, list]]:
    out = []
    for number, raw in enumerate(text.splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        parts = raw.split("\t", 2)
        if len(parts) != 3:
            raise ValueError("%s:%d: expected genre<TAB>weight<TAB>template" % (origin, number))
        genre, weight, body = parts
        try:
            weight = float(weight)
        except ValueError:
            raise ValueError("%s:%d: bad weight %r" % (origin, number, weight))
        if weight < 0:
            raise ValueError("%s:%d: negative weight" % (origin, number))
        body = body.replace("\\n", "\n").replace("\\t", "\t")
        try:
            nodes, end = parse_nodes(body, 0, "")
        except (ValueError, IndexError) as e:
            raise ValueError("%s:%d: %s" % (origin, number, e))
        if end != len(body):
            raise ValueError("%s:%d: unbalanced braces" % (origin, number))
        if re.search(r"(?<![^\W\d_])[ldLD]'\{(INSTITUTION|ART|Art)\b", body):
            raise ValueError("%s:%d: {INSTITUTION} and {ART} write their own article, \"l'\" or \"d'\" before "
                             "them doubles it" % (origin, number))
        kinds = {}
        for spec in org_kinds(nodes):
            ident, kind = spec
            if kinds.setdefault(ident, kind) != kind:
                raise ValueError("%s:%d: organisation %s is given the kinds %s and %s" % (origin, number, ident,
                                                                                       kinds[ident], kind))
        out.append((genre.strip(), weight, nodes))
    return out


def parse_sentences(text: str, origin: str) -> List[Tuple[float, list]]:
    """One sentence per line, an optional weight after a tab. A sentence holds no value
    of the learned categories; it may use the filler slots and the choices of templates."""
    out = []
    for number, raw in enumerate(text.splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        body, _, weight = raw.partition("\t")
        try:
            weight = float(weight) if weight.strip() else 1.0
        except ValueError:
            raise ValueError("%s:%d: bad weight %r" % (origin, number, weight))
        if weight < 0:
            raise ValueError("%s:%d: negative weight" % (origin, number))
        try:
            nodes, end = parse_nodes(body.strip().replace("\\n", "\n"), 0, "")
        except (ValueError, IndexError) as e:
            raise ValueError("%s:%d: %s" % (origin, number, e))
        if end != len(body.strip().replace("\\n", "\n")):
            raise ValueError("%s:%d: unbalanced braces" % (origin, number))
        if any(n[0] == "slot" and n[1] not in FILLER_SLOTS for n in flatten_nodes(nodes)):
            raise ValueError("%s:%d: a sentence uses filler slots only (%s)" % (origin, number,
                                                                                ", ".join(sorted(FILLER_SLOTS))))
        out.append((weight, nodes))
    return out


def flatten_nodes(nodes):
    for node in nodes:
        if node[0] == "choice":
            for alternative in node[1]:
                for n in flatten_nodes(alternative):
                    yield n
        else:
            yield node


def org_kinds(nodes):
    """The (identifier, kind) pairs of the {ORG:kind#n} slots of a template."""
    for node in nodes:
        if node[0] == "slot" and node[1] == "ORG" and node[2] and node[3]:
            yield node[3], node[2]
        elif node[0] == "choice":
            for alternative in node[1]:
                for spec in org_kinds(alternative):
                    yield spec


def parse_nodes(s: str, i: int, stops: str):
    nodes, buf = [], []

    def flush():
        if buf:
            nodes.append(("text", "".join(buf)))
            buf.clear()

    while i < len(s):
        c = s[i]
        if c in stops:
            break
        if c == "{":
            if s.startswith("{{", i):
                buf.append("{")
                i += 2
                continue
            flush()
            if s.startswith("{?", i):
                i += 2
                alternatives = []
                while True:
                    alt, i = parse_nodes(s, i, "|}")
                    alternatives.append(alt)
                    if i >= len(s):
                        raise ValueError("unterminated choice")
                    if s[i] == "|":
                        i += 1
                        continue
                    i += 1
                    break
                nodes.append(("choice", alternatives))
                continue
            j = s.find("}", i)
            if j < 0:
                raise ValueError("unclosed slot %r" % s[i:i + 30])
            nodes.append(parse_slot(s[i + 1:j]))
            i = j + 1
            continue
        if c == "}" and s.startswith("}}", i):
            buf.append("}")
            i += 2
            continue
        if c == "}" and not stops:
            raise ValueError("unbalanced braces: a lone '}' (write '}}' for the character)")
        buf.append(c)
        i += 1
    flush()
    return nodes, i


# Every slot a template may use. parse_slot() rejects the others, so that a typing error
# in a template stops the generator with its line number.
SLOT_NAMES = {
    "g", "e", "il", "Il", "le", "Le", "un", "he", "He", "his", "His", "him", "GENDER", "PERSON", "GIVEN",
    "FAMILY", "BIRTHNAME", "TITLE", "EMAIL", "USERNAME", "HANDLE", "BIRTHDATE", "NIR", "PHONE", "URL", "CITY",
    "POSTAL", "CITYLINE", "STREET", "COMPLEMENT", "ADDRESS", "ADDRESS_BLOCK", "ORG", "DOMAIN", "SIRET", "SIREN",
    "DATE", "TIME", "YEAR", "AGE", "AMOUNT", "IBAN", "CARD", "IP", "IPV6", "MAC", "PLATE", "GEO", "REF", "NINO",
    "SSN", "PASSPORT", "IMEI", "UUID", "VIN", "NEGATIVE", "JOB", "INSTITUTION", "REGION", "COUNTRY", "PRODUCT",
    "SEX", "NATIONALITY", "NATID", "ART", "Art", "NICK",
}


# The slots that write no personal data, or only data that the rules find: the ones a
# context sentence may use.
FILLER_SLOTS = {"DATE", "TIME", "YEAR", "AGE", "AMOUNT", "JOB", "INSTITUTION", "REGION", "COUNTRY", "PRODUCT",
                "NEGATIVE", "REF"}


def parse_slot(spec: str):
    """NAME[:variant][#ident][@org] or g#ident:masculine|feminine."""
    if spec.startswith("g#"):
        ident, _, forms = spec[2:].partition(":")
        masc, _, fem = forms.partition("|")
        return ("slot", "g", None, ident, (masc, fem))
    extra = None
    if "@" in spec:
        spec, extra = spec.split("@", 1)
    ident = None
    if "#" in spec:
        spec, ident = spec.split("#", 1)
    variant = None
    if ":" in spec:
        spec, variant = spec.split(":", 1)
    if spec not in SLOT_NAMES:
        raise ValueError("unknown slot {%s}" % spec)
    if spec == "GENDER" and (variant not in ("f", "m") or not ident):
        raise ValueError("{GENDER:f#1} or {GENDER:m#1} expected")
    return ("slot", spec, variant, ident, extra)


# ----------------------------------------------------------------------------------
# Entities


class Person:
    def __init__(self):
        self.gender = "m"
        self.given = ""
        self.family = ""
        self.birth_name = ""
        self.birth = None
        self.phone = None
        self.email = {}
        self.username = None
        self.handle = None
        self.nir = None
        self.country = None
        self.natid = None
        self.origin = None      # the naming tradition of the given name
        self.nick = None
        self.stage = None


class Place:
    def __init__(self):
        self.city = ""
        self.postal = ""
        self.street = None
        self.complement = None
        self.flat_first = False
        self.cedex = False
        self.country = None     # a code of WORLD for a place abroad, None at home


class Org:
    def __init__(self):
        self.name = ""
        self.article = ""       # "le ", "la ", "l'" or "the " before the name, "" when none is used
        self.domain = ""
        self.siren = None
        self.siret = None


FR_MONTHS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre",
             "novembre", "décembre"]
EN_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
             "November", "December"]

FREE_MAIL = {"fr": ["gmail.com", "orange.fr", "free.fr", "laposte.net", "outlook.fr", "yahoo.fr", "hotmail.fr",
                    "sfr.fr", "wanadoo.fr", "icloud.com", "proton.me"],
             "en": ["gmail.com", "outlook.com", "yahoo.co.uk", "hotmail.com", "icloud.com", "btinternet.com",
                    "aol.com", "proton.me"]}

# Morphemes of the procedural family names, by naming tradition.
SURNAME_PARTS = {
    "fr": (["Ber", "Gau", "Ri", "Lema", "Gui", "Mo", "Four", "Lefe", "Che", "Rou", "Gar", "Bou", "Fau", "Mi", "Pe",
            "Tai", "Mai", "Brun", "Cha", "Del", "Mar", "Mou", "Bla", "Cor", "Duc", "Fer", "Gil", "Jou", "Lan", "Ro",
            "Sau", "Ver", "Bes", "Cla", "Dam", "Gre", "Ler", "Mal", "Pon", "Tes"],
           ["rard", "tier", "ault", "deau", "lot", "chin", "net", "zon", "mand", "lois", "bert", "chet", "ret", "vel",
            "dier", "mont", "ville", "court", "lin", "teau", "guet", "nier", "rat", "card", "quet", "rey", "lard"]),
    "breton": (["Le Bi", "Le Ga", "Le Flo", "Le Mo", "Le Cor", "Ker", "Tré", "Le Ro", "Le Pen", "Gui", "Le Du",
                "Mor", "Plou", "Le Bri"],
               ["han", "llec", "ch", "al", "re", "vella", "guen", "marrec", "gall", "nnec", "rvé", "zoc", "dré",
                "ffret"]),
    "alsace": (["Schm", "Mül", "Hof", "Wag", "Bau", "Kel", "Stoe", "Ries", "Hub", "Zim", "Fisch", "Wolf"],
               ["itt", "ler", "mann", "berger", "heim", "inger", "lin", "hardt", "stein", "acker", "wald"]),
    "occitan": (["Rou", "Ca", "Fa", "Mas", "Puj", "Esta", "Cam", "Del", "Bou", "Laf"],
                ["gnac", "rgues", "bre", "sac", "ol", "dieu", "ès", "pas", "rie", "fon"]),
    "italy": (["Ros", "Bian", "Fer", "Esposi", "Ric", "Mari", "Gal", "Con", "Col", "Mor", "Bru", "Fon"],
              ["sini", "chelli", "rari", "to", "ci", "ani", "lucci", "tini", "elli", "etti", "ozzi"]),
    "iberia": (["Fern", "Gonz", "Rodr", "Mart", "Sanch", "Perei", "Olive", "Carv", "Alme", "Teix", "Ferr"],
               ["andes", "alez", "iguez", "inez", "ez", "ra", "ira", "alho", "ida", "eira", "inho"]),
    "poland": (["Kowal", "Nowa", "Wiśnie", "Wójcik", "Kamiń", "Lewan", "Zieliń", "Szyma", "Woźnia", "Dąbro"],
               ["ski", "k", "wski", "owicz", "czyk", "ska", "dowski", "ński", "ewicz"]),
    "anglo": (["Ash", "Brad", "Hart", "Whit", "Craw", "Pres", "Ather", "Black", "Wood", "Ever", "Hamp", "Thorn",
               "Ald", "Fair", "Lang", "Mor"],
              ["ford", "ley", "ton", "well", "ridge", "wood", "by", "son", "field", "worth", "more", "dale"]),
    "gaelic": (["O'", "Mac", "Mc", "Fitz", "O'", "Mac"],
               ["Brien", "Connor", "Sullivan", "Neill", "Carthy", "Donnell", "Kenna", "Gerald", "Mahon", "Grath",
                "Bride", "Gowan", "Laughlin", "Kinnon", "Leod", "Farlane", "Intyre", "Rory"]),
    "germanic": (["Schm", "Müll", "Schn", "Fisch", "Web", "Mey", "Wag", "Beck", "Hoff", "Kön", "Rich", "Klein",
                  "Neum", "Zimm", "Hart", "Lang", "Brau", "Krü", "Wern", "Stein"],
                 ["idt", "er", "eider", "mann", "inger", "berg", "hardt", "ner", "hold", "bach", "feld", "ke"]),
    "dutch": (["van ", "de ", "van der ", "van den ", "", "", ""],
              ["Berg", "Vries", "Dijk", "Bakker", "Jansen", "Visser", "Smit", "Meijer", "Mulder", "Bos", "Vos",
               "Hendriks", "Dekker", "Brouwer", "Wit", "Leeuwen", "Graaf", "Heuvel", "Linden", "Velde"]),
    "nordic": (["Lars", "Ander", "Eriks", "Johans", "Nils", "Svens", "Berg", "Lind", "Sjö", "Holm", "Dahl", "Lund",
                "Strand", "Ny", "Fors"],
               ["son", "sen", "ström", "gren", "qvist", "lund", "blad", "berg", "man"]),
    "slavic_east": (["Ivan", "Petr", "Smirn", "Kuzn", "Sokol", "Lebed", "Kozl", "Novik", "Moroz", "Shevch",
                     "Kovalen", "Bondar", "Tkach", "Kravch", "Melnych", "Volk"],
                    ["ov", "ova", "enko", "uk", "chuk", "sky", "in", "ina", "ich", "ets"]),
    "greek": (["Papad", "Georgi", "Kon", "Nikol", "Dimitr", "Ioann", "Vasil", "Christ", "Pappa", "Karam", "Alex",
               "Stavr", "Mich"],
              ["opoulos", "ou", "akis", "idis", "iou", "atos", "anis", "elis"]),
    "turkish": (["Yıl", "Kay", "Dem", "Çel", "Şah", "Yıld", "Yüc", "Aks", "Öz", "Kar", "Ars", "Doğ", "Kılı",
                 "Ay"],
                ["maz", "a", "ir", "ik", "in", "ız", "el", "oy", "türk", "oğlu", "can", "kaya", "dın"]),
    "arabic": (["Ben ", "Bou", "El ", "Ait ", "Abdel", "Ben ", "Kh", "Mans", "Ham", "Ch", "Belk", "Mez", "Al-"],
               ["Ali", "aziz", "kader", "ouri", "idi", "zaoui", "hadji", "rif", "souri", "laoui", "Kacem", "medi",
                "Amrani", "Idrissi", "Haddad"]),
    "persian": (["Moham", "Ahm", "Hoss", "Rez", "Kar", "Jaf", "Rah", "Sad", "Naz", "Far", "Tehr", "Shir", "Esf"],
                ["adi", "ani", "eini", "aei", "imi", "ari", "zadeh", "pour", "i", "ian", "nejad", "far"]),
    "armenian": (["Hov", "Sarg", "Grig", "Pet", "Avet", "Khach", "Hakob", "Mart", "Bagd", "Gev", "Ter-Ab"],
                 ["hannisyan", "sian", "oryan", "rosyan", "isyan", "atryan", "yan", "asaryan", "orgian", "ian"]),
    "south_asia": (["Shar", "Ver", "Gup", "Pat", "Red", "Nai", "Iye", "Kum", "Sin", "Chau", "Desh", "Agar", "Mehr",
                    "Kap", "Bhat", "Josh", "Kul", "Ran"],
                   ["ma", "ta", "el", "dy", "r", "ar", "gh", "han", "pande", "wal", "ra", "oor", "t", "i", "karni"]),
    "japan": (["Ta", "Yama", "Naka", "Ko", "Wata", "Ishi", "Ka", "Sa", "Matsu", "Ino", "Kimu", "Hayashi", "Shimi",
               "Yoshi", "Mori", "Ike", "Hashi", "Fuji", "Oka", "Kiku"],
              ["naka", "moto", "mura", "bayashi", "nabe", "kawa", "ta", "da", "ra", "zu", "no", "uchi", "ki", "hara",
               "shita", "saki"]),
    "nigeria": (["Ok", "Ad", "Ol", "Ob", "Ez", "Nw", "Ib", "Ak", "Ogun", "Chukw", "On", "Uch", "Bab", "Fol"],
                ["afor", "eyemi", "uwole", "i", "eh", "abueze", "onkwo", "eke", "ade", "uma", "emi", "ike", "ola",
                 "ayo"]),
}

PROCEDURAL_CITY = {
    "fr_saint": (["Saint-Aubin", "Saint-Martin", "Saint-Germain", "Saint-Hilaire", "Saint-Méen", "Saint-Gildas",
                  "Saint-Brice", "Saint-Marc", "Saint-Ouen", "Saint-Denis", "Saint-Jean", "Saint-Pierre",
                  "Saint-Rémy", "Saint-Vigor", "Saint-Avé", "Saint-Laurent", "Saint-Cyr", "Saint-Didier",
                  "Saint-Gervais", "Sainte-Anne", "Sainte-Marie", "Sainte-Croix", "Sainte-Hélène", "Saint-Léger"],
                 ["-sur-Mer", "-sur-Loire", "-en-Bresse", "-des-Bois", "-le-Château", "-la-Forêt", "-du-Rocher",
                  "-de-Bretagne", "-sur-Vilaine", "-en-Coglès", "-les-Bains", "-d'Aunis", "-du-Désert",
                  "-sous-Bois", "-en-Retz", "-le-Vieux", "-la-Plaine", "-sur-Oust", "-de-Lézat"]),
    "fr_article": ["Le Mesnil", "Le Plessis", "Le Chesnay", "Le Pin", "Le Rheu", "Le Faouët", "Le Hézo", "La Haye",
                   "La Ferté", "La Roche", "La Motte", "La Chapelle", "La Croix", "La Trinité", "La Selle",
                   "La Bosse", "La Barre", "La Garde", "Le Bourg", "Les Sables", "Les Herbiers", "Les Essarts",
                   "Les Granges", "L'Isle", "L'Hermitage", "L'Aiguillon", "L'Oie"],
    "fr_root": (["Plou", "Ker", "Tré", "Lan", "Gui", "Pont-", "Mont", "Ville", "Bour", "Chât", "Beau", "Roche",
                 "Mar", "Val", "Bel", "Font", "Mais", "Brou", "Neu", "Ar", "Loc", "Lo", "Saint-Ger", "Mor"],
                ["gastel", "vern", "bœuf", "nilis", "lan", "ville", "court", "fort", "pierre", "mont", "lieu",
                 "gny", "ac", "argues", "heim", "willer", "ange", "ennes", "ières", "elle", "maria", "mené",
                 "quidy", "dern", "ivy", "nay"]),
    "en": (["", "", "North ", "South ", "East ", "Upper ", "Little ", "Great "],
           ["Ash", "Brook", "Thorn", "Wex", "Hal", "Brad", "Kings", "Oak", "Ell", "Harrow", "Mel", "Stan"],
           ["ford", "ton", "ley", "bury", "ham", "field", "wick", "wood", "minster", "by", "stead", "dale"]),
}

ORG_FORMS = {"fr": ["SAS", "SARL", "SA", "EURL", "SCI", "SASU", "SELARL", "SNC"],
             "en": ["Ltd", "Limited", "Inc.", "LLC", "plc", "LLP", "Corp."]}

ORG_SECTORS = {"fr": ["Conseil", "Services", "Immobilier", "Transports", "Industrie", "Distribution", "Bâtiment",
                      "Informatique", "Ingénierie", "Logistique", "Assurances", "Courtage", "Formation", "Santé",
                      "Énergie", "Environnement", "Patrimoine", "Expertise", "Finance", "Médical"],
               "en": ["Consulting", "Services", "Logistics", "Engineering", "Solutions", "Systems", "Holdings",
                      "Partners", "Analytics", "Healthcare", "Properties", "Capital", "Technologies", "Media"]}



# ----------------------------------------------------------------------------------
# Places abroad
#
# For each country: its names in French and in English, its weights in French and in
# English texts, the order of the street line ("before": "12 rue X", "after": "X 12",
# "comma": "X, 12", "no": "X No: 12", "nr": "X nr. 12", "block": "2-11-3 X"), the city line
# ("pc": postal code first, "cp": city first, "state": the state or province held at the
# start of the postal field), the telephone prefix and layouts ('#' a digit), the layout
# of the IBAN ("n" digits, "a" capitals, with counts: "4a14n"), the top-level domain, the
# naming tradition of its people and the names of its organisations by kind ({w} a made up
# word, {f} and {f2} family names, {g} a given name, {c} the city, {s} a trade, {F} a legal
# form).

WORLD = {
    "de": dict(fr="Allemagne", en="Germany", w=(6, 5), street="after", line="pc",
               phone=("+49", ["30 ########", "151 ########", "89 #######", "40 ########"]), iban="18n", tld="de",
               origin="germanic", forms=["GmbH", "AG", "GmbH & Co. KG", "KG"],
               sectors=["Immobilien", "Logistik", "Bau", "Beratung", "Versicherungen", "Elektrotechnik"],
               orgs={"company": ["{w} {F}", "{f} {s} {F}", "{f} & Söhne", "{w} {s}", "{f} & {f2} {F}"],
                     "public": ["Stadt {c}", "Landratsamt {c}", "Amtsgericht {c}", "Jobcenter {c}"],
                     "school": ["Universität {c}", "Technische Universität {c}", "Hochschule {c}"],
                     "health": ["Klinikum {c}", "Universitätsklinikum {c}", "Praxis Dr. {f}"],
                     "bank": ["Sparkasse {c}", "Volksbank {c}", "{w} Bank"],
                     "association": ["{w} e.V.", "Sportverein {c}"]}),
    "at": dict(fr="Autriche", en="Austria", w=(1.5, 1), street="after", line="pc",
               phone=("+43", ["664 #######", "1 ### ####"]), iban="16n", tld="at", origin="germanic",
               forms=["GmbH", "AG", "KG"], sectors=["Immobilien", "Bau", "Beratung"],
               orgs={"company": ["{w} {F}", "{f} {s} {F}"], "public": ["Magistrat der Stadt {c}", "Stadt {c}"],
                     "school": ["Universität {c}"], "health": ["Landesklinikum {c}"], "bank": ["Raiffeisenbank {c}"]}),
    "ch": dict(fr="Suisse", en="Switzerland", w=(6, 2), street="after", line="pc",
               phone=("+41", ["79 ### ## ##", "22 ### ## ##", "44 ### ## ##"]), iban="17n", tld="ch",
               origin="germanic", forms=["AG", "SA", "GmbH", "Sàrl"], sectors=["Immobilien", "Conseil", "Services"],
               orgs={"company": ["{w} {F}", "{f} {s} {F}", "{f} & Cie"], "public": ["Commune de {c}", "Gemeinde {c}"],
                     "school": ["Université de {c}", "Universität {c}"], "health": ["Hôpital de {c}"],
                     "bank": ["Banque {w}", "{w} Privatbank"]}),
    "lu": dict(fr="Luxembourg", en="Luxembourg", w=(2, 0.5), street="before", line="pc",
               phone=("+352", ["621 ### ###", "26 ## ## ##"]), iban="16n", tld="lu", origin=None,
               forms=["S.A.", "S.à r.l.", "SARL"], sectors=["Services", "Conseil", "Finance"],
               orgs={"company": ["{w} {F}", "{f} {s} {F}"], "public": ["Commune de {c}"], "bank": ["Banque {w}"]}),
    "be": dict(fr="Belgique", en="Belgium", w=(7, 2), street="after", line="pc",
               phone=("+32", ["470 ## ## ##", "2 ### ## ##", "4## ## ## ##"]), iban="12n", tld="be", origin=None,
               forms=["SRL", "SA", "BV", "NV", "ASBL"], sectors=["Services", "Conseil", "Bouw", "Logistiek"],
               orgs={"company": ["{w} {F}", "{f} {s} {F}", "{f} & Fils"], "public": ["Commune de {c}", "Ville de {c}",
                                                                                    "Gemeente {c}"],
                     "school": ["Université de {c}", "Haute École de {c}"], "health": ["CHU de {c}"],
                     "bank": ["Banque {w}"], "association": ["{w} ASBL"]}),
    "nl": dict(fr="Pays-Bas", en="Netherlands", w=(3, 3), street="after", line="pc",
               phone=("+31", ["6 ########", "20 ### ####"]), iban="4a10n", tld="nl", origin="dutch",
               forms=["B.V.", "N.V.", "V.O.F."], sectors=["Bouw", "Logistiek", "Advies", "Vastgoed", "Diensten"],
               orgs={"company": ["{w} {F}", "{f} {s} {F}", "{w} {s}"], "public": ["Gemeente {c}"],
                     "school": ["Hogeschool {c}"], "health": ["Ziekenhuis {w}"]}),
    "es": dict(fr="Espagne", en="Spain", w=(6, 4), street="comma", line="pc",
               phone=("+34", ["6## ## ## ##", "91 ### ## ##", "93 ### ## ##"]), iban="20n", tld="es",
               origin="iberia", forms=["S.L.", "S.A.", "S.L.U."],
               sectors=["Servicios", "Inmobiliaria", "Construcciones", "Consultores", "Seguros", "Transportes"],
               orgs={"company": ["{w} {F}", "{s} {f} {F}", "{f} y Asociados", "{w} {s} {F}"],
                     "public": ["Ayuntamiento de {c}", "Juzgado de Primera Instancia de {c}"],
                     "school": ["Universidad de {c}", "Instituto {g} {f}"],
                     "health": ["Hospital Universitario de {c}", "Clínica {w}"], "bank": ["Banco {w}", "Caja {w}"],
                     "association": ["Asociación {w}", "Club Deportivo {c}"]}),
    "pt": dict(fr="Portugal", en="Portugal", w=(5, 2), street="comma", line="pc",
               phone=("+351", ["9## ### ###", "21 ### ####"]), iban="21n", tld="pt", origin="portugal",
               forms=["Lda.", "S.A.", "Unipessoal Lda."],
               sectors=["Construções", "Serviços", "Imobiliária", "Transportes"],
               orgs={"company": ["{w} {F}", "{f} & Filhos, Lda.", "{s} {f} {F}"],
                     "public": ["Câmara Municipal de {c}", "Junta de Freguesia de {c}"],
                     "school": ["Universidade de {c}"], "health": ["Hospital de {c}"], "bank": ["Banco {w}"]}),
    "it": dict(fr="Italie", en="Italy", w=(5, 4), street="after", line="pc",
               phone=("+39", ["3## ### ####", "06 #### ####", "02 #### ####"]), iban="1a22n", tld="it",
               origin="italy", forms=["S.r.l.", "S.p.A.", "S.n.c.", "S.a.s."],
               sectors=["Costruzioni", "Servizi", "Trasporti", "Immobiliare", "Assicurazioni"],
               orgs={"company": ["{w} {F}", "{f} {s} {F}", "{f} & C. S.n.c.", "{s} {f} {F}"],
                     "public": ["Comune di {c}", "Tribunale di {c}"],
                     "school": ["Università degli Studi di {c}", "Liceo {g} {f}"],
                     "health": ["Azienda Ospedaliera di {c}", "Policlinico {w}"], "bank": ["Banca {w}"],
                     "association": ["Associazione {w}"]}),
    "pl": dict(fr="Pologne", en="Poland", w=(2, 3), street="after", line="pc",
               phone=("+48", ["5## ### ###", "22 ### ## ##"]), iban="24n", tld="pl", origin="poland",
               forms=["Sp. z o.o.", "S.A.", "sp.j."], sectors=["Budownictwo", "Logistyka", "Usługi"],
               orgs={"company": ["{w} {F}", "{f} i Wspólnicy {F}", "{w} {s} {F}"], "school": ["Politechnika {w}"],
                     "bank": ["Bank {w}"]}),
    "cz": dict(fr="République tchèque", en="Czech Republic", w=(1, 1), street="after", line="pc",
               phone=("+420", ["6## ### ###", "2## ### ###"]), iban="20n", tld="cz", origin="poland",
               forms=["s.r.o.", "a.s."], sectors=["Stavby", "Služby"],
               orgs={"company": ["{w} {F}", "{f} {s} {F}"]}),
    "se": dict(fr="Suède", en="Sweden", w=(1, 1.5), street="after", line="pc",
               phone=("+46", ["7# ### ## ##", "8 ### ### ##"]), iban="20n", tld="se", origin="nordic",
               forms=["AB", "HB"], sectors=["Bygg", "Konsult", "Fastigheter"],
               orgs={"company": ["{w} {F}", "{f} {s} {F}"], "public": ["{c} kommun"]}),
    "dk": dict(fr="Danemark", en="Denmark", w=(1, 1), street="after", line="pc",
               phone=("+45", ["## ## ## ##"]), iban="14n", tld="dk", origin="nordic", forms=["A/S", "ApS"],
               sectors=["Byg", "Rådgivning"], orgs={"company": ["{w} {F}", "{f} {s} {F}"], "public": ["{c} Kommune"]}),
    "no": dict(fr="Norvège", en="Norway", w=(0.8, 1), street="after", line="pc",
               phone=("+47", ["4## ## ###", "9## ## ###"]), iban="11n", tld="no", origin="nordic", forms=["AS", "ASA"],
               sectors=["Bygg", "Eiendom"], orgs={"company": ["{w} {F}", "{f} {s} {F}"], "public": ["{c} kommune"]}),
    "fi": dict(fr="Finlande", en="Finland", w=(0.6, 0.8), street="after", line="pc",
               phone=("+358", ["4# ### ####", "9 #### ####"]), iban="14n", tld="fi", origin="nordic",
               forms=["Oy", "Oyj"], sectors=["Rakennus", "Palvelut"], orgs={"company": ["{w} {F}", "{f} {s} {F}"]}),
    "ie": dict(fr="Irlande", en="Ireland", w=(1, 5), street="before", line="cp",
               phone=("+353", ["8# ### ####", "1 ### ####"]), iban="4a14n", tld="ie", origin="irish",
               forms=["Ltd", "DAC"], sectors=["Consulting", "Services", "Construction"],
               orgs={"company": ["{w} {F}", "{f} & Sons {F}", "{f} {s}"], "public": ["{c} City Council"],
                     "school": ["University of {c}"], "health": ["{c} University Hospital"]}),
    "gr": dict(fr="Grèce", en="Greece", w=(1, 1), street="after", line="pc",
               phone=("+30", ["69# ### ####", "21# ### ####"]), iban="7n16n", tld="gr", origin="balkan",
               forms=["A.E.", "E.P.E.", "I.K.E."], sectors=["Techniki", "Emporiki"], orgs={"company": ["{w} {F}"]}),
    "ro": dict(fr="Roumanie", en="Romania", w=(1.5, 1.5), street="nr", line="pc",
               phone=("+40", ["7## ### ###", "21 ### ####"]), iban="4a16n", tld="ro", origin="balkan",
               forms=["S.R.L.", "S.A."], sectors=["Construcții", "Servicii"],
               orgs={"company": ["SC {w} {F}", "{w} {s} {F}"], "public": ["Primăria {c}"]}),
    "hu": dict(fr="Hongrie", en="Hungary", w=(0.8, 0.8), street="after", line="pc",
               phone=("+36", ["20 ### ####", "1 ### ####"]), iban="24n", tld="hu", origin="balkan",
               forms=["Kft.", "Zrt.", "Bt."], sectors=["Építő", "Szolgáltató"],
               orgs={"company": ["{w} {F}", "{w} {s} {F}"]}),
    "ma": dict(fr="Maroc", en="Morocco", w=(6, 1), street="before", line="pc",
               phone=("+212", ["6 ## ## ## ##", "5 ## ## ## ##"]), iban="24n", tld="ma", origin="maghreb",
               forms=["SARL", "SA", "SARL AU"], sectors=["Services", "Immobilier", "Transport", "Négoce"],
               orgs={"company": ["{w} {F}", "Société {w} {F}", "Groupe {w}", "{w} {s}"], "public": ["Commune de {c}"],
                     "school": ["Université de {c}", "École {w}"], "health": ["Clinique {w}", "CHU de {c}"],
                     "bank": ["Banque {w}"]}),
    "dz": dict(fr="Algérie", en="Algeria", w=(4, 0.5), street="before", line="pc",
               phone=("+213", ["5## ## ## ##", "6## ## ## ##"]), iban=None, tld="dz", origin="maghreb",
               forms=["SARL", "EURL", "SPA"], sectors=["Services", "Travaux", "Distribution"],
               orgs={"company": ["{w} {F}", "Entreprise {f}", "{w} {s} {F}"], "public": ["APC de {c}"],
                     "school": ["Université de {c}"], "health": ["Clinique {w}"]}),
    "tn": dict(fr="Tunisie", en="Tunisia", w=(3, 0.5), street="before", line="pc",
               phone=("+216", ["2# ### ###", "9# ### ###"]), iban="20n", tld="tn", origin="maghreb",
               forms=["SARL", "SA", "SUARL"], sectors=["Services", "Industrie", "Commerce"],
               orgs={"company": ["{w} {F}", "Société {w} {F}"], "public": ["Municipalité de {c}"],
                     "school": ["Université de {c}"]}),
    "sn": dict(fr="Sénégal", en="Senegal", w=(2, 0.3), street="before", line="pc",
               phone=("+221", ["77 ### ## ##", "78 ### ## ##"]), iban=None, tld="sn", origin="west_africa",
               forms=["SARL", "SA", "SUARL"], sectors=["Services", "Négoce", "BTP"],
               orgs={"company": ["{w} {F}", "Société {w} {F}"], "public": ["Mairie de {c}"]}),
    "ci": dict(fr="Côte d'Ivoire", en="Ivory Coast", w=(2, 0.3), street="before", line="pc",
               phone=("+225", ["07 ## ## ## ##", "05 ## ## ## ##"]), iban=None, tld="ci", origin="west_africa",
               forms=["SARL", "SA"], sectors=["Services", "Négoce", "BTP"],
               orgs={"company": ["{w} {F}", "Société {w} {F}"]}),
    "cm": dict(fr="Cameroun", en="Cameroon", w=(1.5, 0.5), street="before", line="pc",
               phone=("+237", ["6 ## ## ## ##"]), iban=None, tld="cm", origin="west_africa",
               forms=["SARL", "SA"], sectors=["Services", "Négoce"], orgs={"company": ["{w} {F}"]}),
    "ca": dict(fr="Canada", en="Canada", w=(4, 6), street="before", line="state",
               phone=("+1", ["514 ### ####", "416 ### ####", "604 ### ####", "418 ### ####"]), iban=None, tld="ca",
               origin=None, forms=["inc.", "ltée", "Inc.", "Ltd."], sectors=["Services", "Conseil", "Solutions"],
               orgs={"company": ["{w} {F}", "{f} {s} {F}"], "public": ["Ville de {c}", "City of {c}"],
                     "school": ["Université de {c}", "University of {c}"]}),
    "us": dict(fr="États-Unis", en="United States", w=(3, 0), street="before", line="state",
               phone=("+1", ["212 ### ####", "415 ### ####", "617 ### ####"]), iban=None, tld="com", origin=None,
               forms=["Inc.", "LLC", "Corp."], sectors=["Consulting", "Holdings", "Technologies"],
               orgs={"company": ["{w} {F}", "{f} {s} {F}"], "public": ["City of {c}"]}),
    "gb": dict(fr="Royaume-Uni", en="United Kingdom", w=(3, 0), street="before", line="cp",
               phone=("+44", ["20 #### ####", "7### ######"]), iban="4a14n", tld="co.uk", origin=None,
               forms=["Ltd", "plc", "LLP"], sectors=["Consulting", "Services", "Properties"],
               orgs={"company": ["{w} {F}", "{f} & Sons {F}"], "public": ["{c} City Council"]}),
    "au": dict(fr="Australie", en="Australia", w=(0.8, 3), street="before", line="state",
               phone=("+61", ["4## ### ###", "2 #### ####"]), iban=None, tld="com.au", origin=None,
               forms=["Pty Ltd", "Ltd"], sectors=["Consulting", "Services", "Mining"],
               orgs={"company": ["{w} {F}", "{f} {s} {F}"], "public": ["City of {c} Council"],
                     "school": ["University of {c}"]}),
    "nz": dict(fr="Nouvelle-Zélande", en="New Zealand", w=(0.3, 1.5), street="before", line="cp",
               phone=("+64", ["21 ### ####", "9 ### ####"]), iban=None, tld="co.nz", origin=None,
               forms=["Limited", "Ltd"], sectors=["Services", "Farming"], orgs={"company": ["{w} {F}"],
                                                                                "public": ["{c} City Council"]}),
    "in": dict(fr="Inde", en="India", w=(1, 5), street="before", line="cp",
               phone=("+91", ["9#### #####", "22 #### ####", "80 #### ####"]), iban=None, tld="in",
               origin="south_asia", forms=["Pvt. Ltd.", "Private Limited", "LLP", "Ltd."],
               sectors=["Technologies", "Exports", "Infotech", "Pharma"],
               orgs={"company": ["{w} {F}", "{f} & Sons {F}", "{w} {s} {F}"],
                     "public": ["{c} Municipal Corporation"], "school": ["{w} Institute of Technology"],
                     "health": ["{w} Hospital"], "bank": ["{w} Bank"]}),
    "jp": dict(fr="Japon", en="Japan", w=(1, 2), street="block", line="cp",
               phone=("+81", ["90-####-####", "3-####-####"]), iban=None, tld="co.jp", origin="japan",
               forms=["K.K.", "Co., Ltd.", "Inc."], sectors=["Trading", "Electronics", "Industries"],
               orgs={"company": ["{w} {F}", "{f} {s} {F}"], "school": ["{c} University"]}),
    "cn": dict(fr="Chine", en="China", w=(1, 2), street="before", line="cp",
               phone=("+86", ["13# #### ####", "10 #### ####"]), iban=None, tld="cn", origin="china",
               forms=["Co., Ltd."], sectors=["Trading", "Technology", "Manufacturing"],
               orgs={"company": ["{c} {w} {s} {F}", "{w} {F}"], "school": ["{c} University"]}),
    "kr": dict(fr="Corée du Sud", en="South Korea", w=(0.5, 1), street="before", line="cp",
               phone=("+82", ["10-####-####", "2-###-####"]), iban=None, tld="co.kr", origin="korea",
               forms=["Co., Ltd.", "Corp."], sectors=["Electronics", "Industries"], orgs={"company": ["{w} {F}"]}),
    "sg": dict(fr="Singapour", en="Singapore", w=(0.3, 1.5), street="before", line="cp",
               phone=("+65", ["9### ####", "6### ####"]), iban=None, tld="com.sg", origin="china",
               forms=["Pte. Ltd."], sectors=["Holdings", "Trading", "Logistics"], orgs={"company": ["{w} {s} {F}",
                                                                                                    "{w} {F}"]}),
    "br": dict(fr="Brésil", en="Brazil", w=(1.5, 2), street="comma", line="state",
               phone=("+55", ["11 9####-####", "21 9####-####"]), iban="23n1a1n", tld="com.br", origin="portugal",
               forms=["Ltda.", "S.A."], sectors=["Serviços", "Comércio", "Engenharia"],
               orgs={"company": ["{w} {F}", "{s} {f} {F}"], "public": ["Prefeitura de {c}"],
                     "school": ["Universidade Federal de {c}"]}),
    "mx": dict(fr="Mexique", en="Mexico", w=(1, 2), street="after", line="state",
               phone=("+52", ["55 #### ####", "33 #### ####"]), iban=None, tld="com.mx", origin="latam",
               forms=["S.A. de C.V.", "S. de R.L. de C.V."], sectors=["Servicios", "Construcciones", "Comercial"],
               orgs={"company": ["{w} {F}", "{s} {f} {F}"], "public": ["Municipio de {c}"]}),
    "ar": dict(fr="Argentine", en="Argentina", w=(0.6, 0.8), street="after", line="pc",
               phone=("+54", ["9 11 ####-####"]), iban=None, tld="com.ar", origin="latam", forms=["S.A.", "S.R.L."],
               sectors=["Servicios", "Agropecuaria"], orgs={"company": ["{w} {F}", "{s} {f} {F}"]}),
    "tr": dict(fr="Turquie", en="Turkey", w=(1.5, 1.5), street="no", line="pc",
               phone=("+90", ["5## ### ## ##", "212 ### ## ##"]), iban="22n", tld="com.tr", origin="turkey",
               forms=["A.Ş.", "Ltd. Şti."], sectors=["İnşaat", "Ticaret", "Lojistik"],
               orgs={"company": ["{w} {s} {F}", "{w} {F}"], "public": ["{c} Belediyesi"]}),
    "il": dict(fr="Israël", en="Israel", w=(0.5, 1), street="before", line="cp",
               phone=("+972", ["5#-###-####", "3-###-####"]), iban="19n", tld="co.il", origin="hebrew",
               forms=["Ltd."], sectors=["Technologies", "Systems"], orgs={"company": ["{w} {s} {F}", "{w} {F}"]}),
    "ae": dict(fr="Émirats arabes unis", en="United Arab Emirates", w=(0.8, 2), street="before", line="cp",
               phone=("+971", ["5# ### ####", "4 ### ####"]), iban="19n", tld="ae", origin="arabic",
               forms=["LLC", "FZE", "FZ-LLC"], sectors=["Trading", "General Trading", "Contracting"],
               orgs={"company": ["{w} {s} {F}", "{w} {F}"]}),
    "lb": dict(fr="Liban", en="Lebanon", w=(1, 0.5), street="before", line="cp",
               phone=("+961", ["3 ### ###", "1 ### ###"]), iban="4n20n", tld="com.lb", origin="arabic",
               forms=["SAL", "SARL"], sectors=["Trading", "Services"], orgs={"company": ["{w} {F}"]}),
    "eg": dict(fr="Égypte", en="Egypt", w=(0.6, 1), street="before", line="cp",
               phone=("+20", ["10 #### ####", "2 #### ####"]), iban="25n", tld="com.eg", origin="arabic",
               forms=["S.A.E.", "LLC"], sectors=["Trading", "Contracting"], orgs={"company": ["{w} {F}"]}),
    "ng": dict(fr="Nigeria", en="Nigeria", w=(0.5, 3), street="before", line="cp",
               phone=("+234", ["80# ### ####", "70# ### ####"]), iban=None, tld="com.ng", origin="africa",
               forms=["Ltd", "Plc", "Nigeria Limited"], sectors=["Ventures", "Oil and Gas", "Logistics"],
               orgs={"company": ["{w} {s} {F}", "{w} {F}", "{f} and Sons {F}"],
                     "school": ["University of {c}"]}),
    "ke": dict(fr="Kenya", en="Kenya", w=(0.3, 1.5), street="before", line="cp",
               phone=("+254", ["7## ### ###"]), iban=None, tld="co.ke", origin="africa",
               forms=["Ltd", "Kenya Limited"], sectors=["Enterprises", "Logistics"], orgs={"company": ["{w} {F}"]}),
    "za": dict(fr="Afrique du Sud", en="South Africa", w=(0.5, 2), street="before", line="cp",
               phone=("+27", ["8# ### ####", "11 ### ####"]), iban=None, tld="co.za", origin=None,
               forms=["(Pty) Ltd", "Ltd"], sectors=["Holdings", "Mining", "Services"],
               orgs={"company": ["{w} {F}", "{w} {s} {F}"], "public": ["City of {c}"]}),
    "ru": dict(fr="Russie", en="Russia", w=(0.5, 0.8), street="comma", line="pc",
               phone=("+7", ["9## ###-##-##", "495 ###-##-##"]), iban=None, tld="ru", origin="east_slavic",
               forms=["OOO", "AO"], sectors=["Trade", "Stroy"], orgs={"company": ["{F} {w}"]}),
    "ua": dict(fr="Ukraine", en="Ukraine", w=(1, 1.5), street="comma", line="pc",
               phone=("+380", ["67 ### ## ##", "44 ### ## ##"]), iban="6n19n", tld="ua", origin="east_slavic",
               forms=["TOV", "PrAT"], sectors=["Trade", "Agro"], orgs={"company": ["{F} {w}"]}),
}

# The provinces of Canada in French, for "Montréal (Québec) H2X 1Z5".
CANADA_PROVINCES_FR = {"QC": "Québec", "ON": "Ontario", "BC": "Colombie-Britannique", "AB": "Alberta",
                       "MB": "Manitoba", "NB": "Nouveau-Brunswick", "NS": "Nouvelle-Écosse", "SK": "Saskatchewan"}

# E-mail providers by naming tradition, for the persons whose name comes from it.
ORIGIN_MAIL = {
    "germanic": ["gmx.de", "web.de", "t-online.de", "gmx.at", "bluewin.ch"],
    "dutch": ["ziggo.nl", "kpnmail.nl", "telenet.be"],
    "nordic": ["telia.com", "online.no", "jubii.dk"],
    "italy": ["libero.it", "virgilio.it", "tiscali.it"],
    "iberia": ["yahoo.es", "telefonica.net", "hotmail.es"],
    "portugal": ["sapo.pt", "uol.com.br", "terra.com.br"],
    "latam": ["hotmail.com", "yahoo.com.mx", "prodigy.net.mx"],
    "poland": ["wp.pl", "onet.pl", "o2.pl", "seznam.cz"],
    "east_slavic": ["mail.ru", "yandex.ru", "ukr.net"],
    "balkan": ["yahoo.gr", "freemail.hu", "yahoo.ro"],
    "turkey": ["mynet.com", "hotmail.com.tr"],
    "maghreb": ["menara.ma", "yahoo.fr", "topnet.tn"],
    "arabic": ["hotmail.com", "yahoo.com"],
    "persian": ["yahoo.com", "chmail.ir"],
    "hebrew": ["walla.co.il", "gmail.com"],
    "south_asia": ["rediffmail.com", "yahoo.co.in"],
    "china": ["qq.com", "163.com", "126.com"],
    "japan": ["yahoo.co.jp", "docomo.ne.jp"],
    "korea": ["naver.com", "daum.net"],
    "vietnam": ["yahoo.com.vn", "gmail.com"],
    "africa": ["yahoo.com", "gmail.com"],
    "west_africa": ["yahoo.fr", "orange.sn"],
}


# ----------------------------------------------------------------------------------
# Nationalities and the national numbers of other countries
#
# A country gives the nationality of a person in French (masculine and feminine) and in
# English, and the national number written in its documents, with the labels that
# announce it. Every number carries a valid check digit, so that the validated scanners
# of dlib/privacy are scored on the numbers they accept.

def _digits(rng: random.Random, n: int) -> str:
    return "".join(str(rng.randint(0, 9)) for _ in range(n))


def _date_digits(rng: random.Random, first: int, last: int) -> Tuple[int, int, int]:
    return rng.randint(first, last), rng.randint(1, 12), rng.randint(1, 28)


def _verhoeff_digit(payload: str) -> int:
    d = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
         [3, 4, 0, 1, 2, 8, 9, 5, 6, 7], [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
         [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3], [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
         [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
    p = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
         [8, 9, 1, 6, 0, 4, 3, 5, 2, 7], [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
         [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]]
    inv = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]
    c = 0
    for i, ch in enumerate(reversed(payload)):
        c = d[c][p[(i + 1) % 8][int(ch)]]
    return inv[c]


def _mod11_10(payload: str) -> int:
    product = 10
    for ch in payload:
        total = (int(ch) + product) % 10
        product = (2 * (total or 10)) % 11
    return (11 - product) % 10


def _weighted(digits: str, weights) -> int:
    return sum(int(c) * w for c, w in zip(digits, weights))


def _id_be(r):
    y, m, d = _date_digits(r, 1940, 2008)
    n = "%02d%02d%02d%03d" % (y % 100, m, d, r.randint(1, 997))
    check = 97 - (int(("2" if y >= 2000 else "") + n) % 97)
    v = n + "%02d" % check
    return v[:2] + "." + v[2:4] + "." + v[4:6] + "-" + v[6:9] + "." + v[9:] if r.random() < 0.6 else v


def _id_es(r):
    letters = "TRWAGMYFPDXBNJZSQVHLCKE"
    if r.random() < 0.3:
        prefix = r.choice("XYZ")
        body = _digits(r, 7)
        return prefix + body + letters[int(str("XYZ".index(prefix)) + body) % 23]
    body = "%08d" % r.randint(1000000, 99999999)
    return body + letters[int(body) % 23]


def _id_it(r):
    consonants = "BCDFGHLMNPRSTVZ"
    out = "".join(r.choice(consonants) for _ in range(6))
    y, m, d = _date_digits(r, 1940, 2006)
    out += "%02d" % (y % 100) + "ABCDEHLMPRST"[m - 1] + "%02d" % (d + (40 if r.random() < 0.5 else 0))
    out += r.choice("ABCDEFGHILMZ") + "%03d" % r.randint(1, 999)
    odd = [1, 0, 5, 7, 9, 13, 15, 17, 19, 21, 2, 4, 18, 20, 11, 3, 6, 8, 12, 14, 16, 10, 22, 25, 24, 23]
    total = 0
    for i, c in enumerate(out):
        v = int(c) if c.isdigit() else ord(c) - 65
        total += odd[v] if i % 2 == 0 else v
    return out + chr(65 + total % 26)


def _id_de(r):
    payload = str(r.randint(1, 9)) + _digits(r, 9)
    return payload + str(_mod11_10(payload))


def _id_nl(r):
    while True:
        body = str(r.randint(1, 9)) + _digits(r, 7)
        rest = _weighted(body, range(9, 1, -1)) % 11
        if rest <= 9:
            return body + str(rest)


def _id_pt(r):
    body = r.choice("1235689") + _digits(r, 7)
    rest = _weighted(body, range(9, 1, -1)) % 11
    return body + str(0 if rest < 2 else 11 - rest)


def _id_pl(r):
    y, m, d = _date_digits(r, 1940, 2008)
    body = "%02d%02d%02d" % (y % 100, m + (20 if y >= 2000 else 0), d) + _digits(r, 4)
    return body + str((10 - _weighted(body, [1, 3, 7, 9, 1, 3, 7, 9, 1, 3]) % 10) % 10)


def _id_ch(r):
    body = "756" + _digits(r, 9)
    check = (10 - sum(int(c) * (1 if i % 2 == 0 else 3) for i, c in enumerate(body)) % 10) % 10
    v = body + str(check)
    return v[:3] + "." + v[3:7] + "." + v[7:11] + "." + v[11:]


def _id_lu(r):
    y, m, d = _date_digits(r, 1940, 2008)
    payload = "%04d%02d%02d" % (y, m, d) + _digits(r, 3)
    return payload + str(luhn_digit(payload)) + str(_verhoeff_digit(payload))


def _id_at(r):
    while True:
        serial = str(r.randint(1, 9)) + _digits(r, 2)
        y, m, d = _date_digits(r, 1940, 2006)
        date = "%02d%02d%02d" % (d, m, y % 100)
        check = _weighted(serial + "0" + date, [3, 7, 9, 0, 5, 8, 4, 2, 1, 6]) % 11
        if check < 10:
            return serial + str(check) + date


def _id_se(r):
    y, m, d = _date_digits(r, 1940, 2006)
    body = "%02d%02d%02d" % (y % 100, m, d) + _digits(r, 3)
    v = body + str(luhn_digit(body))
    return v[:6] + ("-" if r.random() < 0.8 else "") + v[6:] if r.random() < 0.8 else "%02d" % (y // 100) + v


def _id_dk(r):
    y, m, d = _date_digits(r, 1940, 1999)
    v = "%02d%02d%02d" % (d, m, y % 100) + str(r.randint(0, 3)) + _digits(r, 3)
    return v[:6] + "-" + v[6:] if r.random() < 0.8 else v


def _id_fi(r):
    y, m, d = _date_digits(r, 1940, 2008)
    individual = "%03d" % r.randint(2, 899)
    date = "%02d%02d%02d" % (d, m, y % 100)
    control = "0123456789ABCDEFHJKLMNPRSTUVWXY"[int(date + individual) % 31]
    return date + ("A" if y >= 2000 else "-") + individual + control


def _id_no(r):
    while True:
        y, m, d = _date_digits(r, 1940, 2006)
        body = "%02d%02d%02d" % (d, m, y % 100) + _digits(r, 3)
        k1 = (11 - _weighted(body, [3, 7, 6, 1, 8, 9, 4, 5, 2]) % 11) % 11
        if k1 == 10:
            continue
        k2 = (11 - _weighted(body + str(k1), [5, 4, 3, 2, 7, 6, 5, 4, 3, 2]) % 11) % 11
        if k2 != 10:
            return body + str(k1) + str(k2)


def _id_ie(r):
    body = _digits(r, 7)
    return body + "WABCDEFGHIJKLMNOPQRSTUV"[sum(int(c) * (8 - i) for i, c in enumerate(body)) % 23]


def _id_ro(r):
    y, m, d = _date_digits(r, 1940, 1999)
    body = r.choice("12") + "%02d%02d%02d" % (y % 100, m, d) + "%02d" % r.randint(1, 46) + _digits(r, 3)
    check = _weighted(body, [2, 7, 9, 1, 4, 6, 3, 5, 8, 2, 7, 9]) % 11
    return body + str(1 if check == 10 else check)


def _id_bg(r):
    y, m, d = _date_digits(r, 1940, 1999)
    body = "%02d%02d%02d" % (y % 100, m, d) + _digits(r, 3)
    return body + str(_weighted(body, [2, 4, 8, 5, 10, 9, 7, 3, 6]) % 11 % 10)


def _id_hr(r):
    payload = _digits(r, 10)
    return payload + str(_mod11_10(payload))


def _id_cz(r):
    y, m, d = _date_digits(r, 1954, 1999)
    head = "%02d%02d%02d" % (y % 100, m + (50 if r.random() < 0.5 else 0), d) + _digits(r, 3)
    return head[:6] + "/" + head[6:] + str(int(head) % 11 % 10)


def _id_nhs(r):
    while True:
        body = str(r.randint(4, 9)) + _digits(r, 8)
        check = 11 - _weighted(body, range(10, 1, -1)) % 11
        if check == 11:
            check = 0
        if check != 10:
            v = body + str(check)
            return v[:3] + " " + v[3:6] + " " + v[6:] if r.random() < 0.7 else v


def _id_ca(r):
    body = str(r.choice([1, 2, 3, 4, 5, 6, 7, 9])) + _digits(r, 7)
    v = body + str(luhn_digit(body))
    sep = r.choice([" ", "-"])
    return v[:3] + sep + v[3:6] + sep + v[6:] if r.random() < 0.8 else v


def _id_br(r):
    body = _digits(r, 9)
    s1 = sum(int(c) * (10 - i) for i, c in enumerate(body))
    d1 = (11 - s1 % 11) % 11 % 10
    s2 = sum(int(c) * (11 - i) for i, c in enumerate(body)) + 2 * d1
    d2 = (11 - s2 % 11) % 11 % 10
    v = body + str(d1) + str(d2)
    return v[:3] + "." + v[3:6] + "." + v[6:9] + "-" + v[9:] if r.random() < 0.8 else v


def _id_in(r):
    while True:
        payload = str(r.randint(2, 9)) + _digits(r, 10)
        v = payload + str(_verhoeff_digit(payload))
        if v != v[::-1]:
            return v[:4] + " " + v[4:8] + " " + v[8:] if r.random() < 0.8 else v


def _id_tr(r):
    body = str(r.randint(1, 9)) + _digits(r, 8)
    weighted = sum(int(c) * (3 if i % 2 == 0 else 1) for i, c in enumerate(body))
    c1 = (10 - weighted % 10) % 10
    c2 = (c1 + sum(int(c) for c in body)) % 10
    return body + str(c1) + str(c2)


def _id_ma(r):
    return r.choice(["A", "B", "BE", "BK", "C", "D", "EE", "G", "J", "K", "AB", "BH"]) + "%06d" % r.randint(1000, 999999)


def _id_dz(r):
    return "1" + _digits(r, 17)


def _id_tn(r):
    return "%08d" % r.randint(1000000, 99999999)


def _id_fr(r):
    letters = "ABCDEFGHJKLMNPRSTUVWXYZ"
    return "%02d%s%s%05d" % (r.randint(1, 99), r.choice(letters), r.choice(letters), r.randint(0, 99999))


def _id_us(r):
    return "%03d-%02d-%04d" % (r.choice([r.randint(1, 665), r.randint(667, 899)]), r.randint(1, 99), r.randint(1, 9999))


def _id_au(r):
    while True:
        body = str(r.randint(1, 9)) + _digits(r, 7)
        last = _weighted(body, [1, 4, 3, 7, 5, 8, 6, 9]) % 11
        if last < 10:
            v = body + str(last)
            return v[:3] + " " + v[3:6] + " " + v[6:] if r.random() < 0.7 else v


def _id_cn(r):
    y, m, d = _date_digits(r, 1950, 2005)
    body = str(r.randint(11, 65)) + _digits(r, 4) + "%04d%02d%02d" % (y, m, d) + _digits(r, 3)
    total = sum(int(c) * (2 ** (17 - i) % 11) for i, c in enumerate(body))
    check = (12 - total % 11) % 11
    return body + ("X" if check == 10 else str(check))


# code: (French masculine, French feminine, English, weight in French texts, weight in English
# texts, number generator or None, labels announcing the number)
COUNTRIES = {
    "fr": ("français", "française", "French", 60, 4, _id_fr, ("passeport", "passeport n°", "numéro de passeport")),
    "be": ("belge", "belge", "Belgian", 4, 1, _id_be, ("registre national", "numéro national", "NISS")),
    "ch": ("suisse", "suisse", "Swiss", 3, 1, _id_ch, ("AVS", "AHV", "n° AVS")),
    "lu": ("luxembourgeois", "luxembourgeoise", "Luxembourgish", 1, 0.2, _id_lu,
           ("numéro d'identification national", "matricule national")),
    "es": ("espagnol", "espagnole", "Spanish", 3, 2, _id_es, ("DNI", "NIE", "documento nacional de identidad")),
    "pt": ("portugais", "portugaise", "Portuguese", 4, 1, _id_pt, ("NIF", "número de contribuinte")),
    "it": ("italien", "italienne", "Italian", 3, 2, _id_it, ("codice fiscale", "code fiscal italien")),
    "de": ("allemand", "allemande", "German", 3, 2, _id_de, ("Steuer-ID", "Steuer-IdNr.", "IdNr")),
    "nl": ("néerlandais", "néerlandaise", "Dutch", 1, 1, _id_nl, ("BSN", "burgerservicenummer")),
    "pl": ("polonais", "polonaise", "Polish", 2, 3, _id_pl, ("PESEL", "numéro PESEL")),
    "at": ("autrichien", "autrichienne", "Austrian", 0.5, 0.5, _id_at, ("SVNR", "Sozialversicherungsnummer")),
    "se": ("suédois", "suédoise", "Swedish", 0.5, 1, _id_se, ("personnummer",)),
    "dk": ("danois", "danoise", "Danish", 0.5, 0.5, _id_dk, ("CPR-nr.", "CPR")),
    "fi": ("finlandais", "finlandaise", "Finnish", 0.5, 0.5, _id_fi, ("HETU", "henkilötunnus")),
    "no": ("norvégien", "norvégienne", "Norwegian", 0.5, 0.5, _id_no, ("fødselsnummer", "fnr")),
    "ie": ("irlandais", "irlandaise", "Irish", 0.5, 4, _id_ie, ("PPSN", "PPS number")),
    "ro": ("roumain", "roumaine", "Romanian", 2, 2, _id_ro, ("CNP", "cod numeric personal")),
    "bg": ("bulgare", "bulgare", "Bulgarian", 0.5, 0.5, _id_bg, ("EGN",)),
    "hr": ("croate", "croate", "Croatian", 0.5, 0.5, _id_hr, ("OIB",)),
    "cz": ("tchèque", "tchèque", "Czech", 0.5, 0.5, _id_cz, ("rodné číslo", "birth number")),
    "gb": ("britannique", "britannique", "British", 2, 40, _id_nhs, ("NHS number", "NHS no.")),
    "us": ("américain", "américaine", "American", 2, 15, _id_us, ("SSN", "Social Security number")),
    "ca": ("canadien", "canadienne", "Canadian", 2, 4, _id_ca, ("SIN", "NAS", "numéro d'assurance sociale")),
    "br": ("brésilien", "brésilienne", "Brazilian", 1, 1, _id_br, ("CPF",)),
    "in": ("indien", "indienne", "Indian", 0.5, 4, _id_in, ("Aadhaar", "Aadhaar number")),
    "tr": ("turc", "turque", "Turkish", 2, 1, _id_tr, ("T.C. Kimlik No", "kimlik numarası")),
    "ma": ("marocain", "marocaine", "Moroccan", 6, 0.5, _id_ma, ("CIN", "CNIE", "carte d'identité nationale")),
    "dz": ("algérien", "algérienne", "Algerian", 5, 0.5, _id_dz, ("NIN", "numéro d'identification national")),
    "tn": ("tunisien", "tunisienne", "Tunisian", 3, 0.3, _id_tn, ("CIN",)),
    "cn": ("chinois", "chinoise", "Chinese", 1, 1, _id_cn, ("carte d'identité chinoise", "resident identity card")),
    "au": ("australien", "australienne", "Australian", 0.5, 3, _id_au, ("TFN", "tax file number")),
}

# Values of a sex field, short and long.
SEX_VALUES = {"fr": {"f": ["F", "F", "Féminin", "Femme"], "m": ["M", "M", "Masculin", "Homme"]},
              "en": {"f": ["F", "F", "Female", "Woman"], "m": ["M", "M", "Male", "Man"]}}


# ----------------------------------------------------------------------------------
# The generator


class Generator:
    def __init__(self, res: Resources, lang: str, rng: random.Random, pool: str, procedural: float,
                 noise: float, abroad: float = 0.0):
        self.res = res
        self.abroad = abroad
        self.lang = lang
        self.lex = res.lex[lang]
        self.rng = rng
        self.pool = pool
        self.procedural = procedural
        self.noise = noise
        self.people: Dict[str, Person] = {}
        self.places: Dict[str, Place] = {}
        self.orgs: Dict[str, Org] = {}
        self.genders: Dict[str, str] = {}
        self.org_kinds: Dict[str, str] = {}
        self.abroad_places = set()
        self.style = {}

    # -- document style ---------------------------------------------------------------

    def new_document(self, genre: str):
        r = self.rng
        self.people, self.places, self.orgs, self.genders, self.org_kinds = {}, {}, {}, {}, {}
        self.abroad_places = set()
        formal = genre in ("form", "letter", "hr", "legal", "medical", "table", "banking", "real_estate", "admin")
        self.style = {
            "upper_family": r.random() < (0.45 if formal else 0.12) and self.lang == "fr",
            "upper_city": r.random() < (0.25 if formal else 0.05),
            "all_upper": r.random() < 0.02 * self.noise,
            "all_lower": r.random() < (0.35 if genre == "chat" else 0.02) * max(self.noise, 0.3),
            "no_accents": r.random() < 0.07 * self.noise,
            "nbsp_colon": self.lang == "fr" and r.random() < 0.3,
            "ocr": r.random() < 0.06 * self.noise,
            "double_space": r.random() < 0.05 * self.noise,
            "name_order": "family_first" if r.random() < (0.25 if self.lang == "fr" else 0.04) else "given_first",
            "date_format": r.choice(["slash", "slash", "long", "dash", "dot", "iso"]),
            "phone_format": r.choice(["spaced", "spaced", "spaced", "dotted", "compact", "intl"]),
        }

    def constrain(self, nodes):
        """Records the {GENDER:f#1} slots of a template before it is rendered, so that they
        hold wherever they appear in it; the first one met wins."""
        for node in nodes:
            if node[0] == "slot" and node[1] == "GENDER":
                self.genders.setdefault(node[3], node[2])
            elif node[0] == "slot" and node[1] == "ORG" and node[2] and node[3]:
                self.org_kinds.setdefault(node[3], node[2])
            elif node[0] == "slot" and node[2] == "abroad" and node[3]:
                self.abroad_places.add(node[3])
            elif node[0] == "choice":
                for alternative in node[1]:
                    self.constrain(alternative)

    # -- lexicon access ---------------------------------------------------------------

    def pick(self, name: str, where=None, key=None) -> str:
        fields = self.lex[name].pick(self.rng, self.pool, where, key)
        return fields[0] if fields else ""

    def pick_fields(self, name: str, where=None, key=None) -> Tuple[str, ...]:
        return self.lex[name].pick(self.rng, self.pool, where, key) or ("",)

    def in_split(self, value: str) -> bool:
        """Whether a procedural value belongs to the part of the current split: the hash
        that splits the lexicons splits the made up values too."""
        if self.pool == "all":
            return True
        return held_out(value, self.res.unseen_percent) == (self.pool == "unseen")

    def split_value(self, make):
        """A value of make() in the part of the current split, drawn again a few times."""
        value = make()
        for _ in range(40):
            if self.in_split(value):
                break
            value = make()
        return value

    # -- persons ---------------------------------------------------------------------

    def procedural_family(self, tradition: str) -> str:
        parts = SURNAME_PARTS.get(tradition) or SURNAME_PARTS["fr"]

        def make():
            prefix, suffix = self.rng.choice(parts[0]), self.rng.choice(parts[1])
            # "Ben Kacem", "O'Brien", "Al-Haddad": after a particle the name takes a capital; "Bouzaoui"
            # and "Le Bihan" are written in one word, "MacDonald" keeps both capitals.
            if prefix[-1:] in (" ", "'", "-"):
                suffix = suffix[:1].upper() + suffix[1:]
            elif prefix and tradition != "gaelic":
                suffix = suffix[:1].lower() + suffix[1:]
            return prefix + suffix
        return self.split_value(make)

    # Origins whose family names the procedural generator can imitate.
    TRADITIONS = {"fr": ["fr", "fr", "breton", "alsace", "occitan"], "breton": ["breton"], "alsace": ["alsace"],
                  "occitan": ["occitan"], "italy": ["italy"], "iberia": ["iberia"], "portugal": ["iberia"],
                  "latam": ["iberia"], "hispanic": ["iberia"], "poland": ["poland"], "east_europe": ["poland"],
                  "anglo": ["anglo"], "irish": ["gaelic", "anglo"], "scottish": ["gaelic", "anglo"],
                  "germanic": ["germanic"], "dutch": ["dutch"], "nordic": ["nordic"], "east_slavic": ["slavic_east"],
                  "balkan": ["greek"], "turkey": ["turkish"], "maghreb": ["arabic"], "arabic": ["arabic"],
                  "middle_east": ["arabic"], "persian": ["persian"], "armenia": ["armenian"],
                  "caucasus": ["armenian"], "south_asia": ["south_asia"], "japan": ["japan"],
                  "africa": ["nigeria"]}

    def make_family(self, origin: Optional[str] = None) -> str:
        """A family name, of the same origin as the given name most of the time: mixed
        couples and adoptions exist, but "Wei Duong" should not be the rule."""
        r = self.rng
        if origin and r.random() > 0.75:
            origin = None
        if r.random() < self.procedural and (origin is None or origin in self.TRADITIONS):
            if origin:
                tradition = r.choice(self.TRADITIONS[origin])
            elif self.lang == "fr":
                tradition = r.choices(["fr", "breton", "alsace", "occitan", "italy", "iberia", "poland", "anglo",
                                       "arabic", "germanic", "turkish", "slavic_east", "greek", "nigeria"],
                                      [36, 13, 7, 7, 7, 7, 4, 3, 6, 3, 2, 2, 2, 1])[0]
            else:
                tradition = r.choices(["anglo", "gaelic", "fr", "italy", "iberia", "poland", "germanic", "nordic",
                                       "south_asia", "nigeria", "slavic_east", "japan", "dutch"],
                                      [55, 6, 5, 6, 6, 4, 3, 2, 5, 3, 2, 1, 2])[0]
            name = self.procedural_family(tradition)
        else:
            where = (lambda f: len(f) > 1 and f[1] == origin) if origin else None
            name = self.pick("family", where, ("origin", origin) if origin else None)
        x = r.random()
        if x < 0.03:
            other = self.pick("family")
            if other and other != name and " " not in other:
                name = name + "-" + other
        elif x < 0.06 and self.lang == "fr" and origin in (None, "fr") and " " not in name and "'" not in name and \
                name[:2].lower() not in ("le", "la", "de", "du"):
            if name[0] in "AEIOUYÉ":
                name = "d'" + name
            else:
                name = r.choice(["de ", "du ", "de La ", "Le "]) + name
        return name

    def make_given(self, gender: str, with_origin: bool = False):
        r = self.rng
        lex = "given_female" if gender == "f" else "given_male"
        fields = self.pick_fields(lex)
        name, origin = fields[0], (fields[1] if len(fields) > 1 else None)
        if self.lang == "fr" and origin == "fr" and r.random() < 0.05 and "-" not in name:
            other = self.pick(lex, lambda f: len(f) > 1 and f[1] == "fr", ("origin", "fr"))
            if other and other != name and "-" not in other:
                name = name + "-" + other
        return (name, origin) if with_origin else name

    def person(self, ident: Optional[str]) -> Person:
        key = ident or "_%d" % len(self.people)
        if key not in self.people:
            p = Person()
            p.gender = "f" if self.rng.random() < 0.5 else "m"
            p.gender = self.genders.get(key, p.gender)
            p.given, origin = self.make_given(p.gender, True)
            p.origin = origin
            p.family = self.make_family(origin)
            if p.gender == "f" and self.rng.random() < 0.3:
                p.birth_name = self.make_family()
            self.people[key] = p
        return self.people[key]

    def family_text(self, p: Person, variant: Optional[str] = None) -> str:
        if variant == "upper" or (variant != "plain" and self.style["upper_family"]):
            return p.family.upper()
        return p.family

    def person_fragments(self, p: Person, variant: Optional[str]) -> list:
        r = self.rng
        g = (p.given, "GIVEN_NAME")
        sp = (" ", None)
        if variant == "signature":
            return [g, sp, (p.family, "FAMILY_NAME")]
        if variant == "formal":
            return [(p.family.upper(), "FAMILY_NAME"), sp, g]
        if variant == "initial":
            return [(initials(p.given), "GIVEN_NAME"), sp, (self.family_text(p), "FAMILY_NAME")]
        if variant == "family":
            return [(self.family_text(p), "FAMILY_NAME")]
        if variant == "given":
            return [g]
        f = (self.family_text(p), "FAMILY_NAME")
        x = r.random()
        if x < 0.08:
            return [(initials(p.given), "GIVEN_NAME"), sp, f]
        if x < 0.12:
            return [(p.family, "FAMILY_NAME"), (", ", None), g]
        if self.style["name_order"] == "family_first" or (self.style["upper_family"] and x < 0.4):
            return [f, sp, g]
        return [g, sp, f]

    def username(self, p: Person) -> str:
        if p.username is None:
            r = self.rng
            gf = ascii_fold(p.given).lower().replace("-", "").replace("'", "").replace(" ", "")
            ff = ascii_fold(p.family).lower().replace("-", "").replace("'", "").replace(" ", "")
            forms = [gf[:1] + ff, gf + "." + ff, ff + gf[:1], gf + ff[:1], gf + "_" + ff, ff,
                     gf[:1] + ff + str(r.randint(1, 99)), gf + str(r.randint(10, 99))]
            p.username = r.choice(forms)[:24]
        return p.username

    def email(self, p: Person, org: Optional[Org]) -> str:
        key = org.domain if org else None
        if key not in p.email:
            r = self.rng
            gf = ascii_fold(p.given).lower().replace(" ", "-").replace("'", "")
            ff = ascii_fold(p.family).lower().replace(" ", "").replace("'", "")
            local = r.choice([gf + "." + ff, gf[:1] + "." + ff, gf[:1] + ff, ff + "." + gf, gf + ff,
                              gf + "." + ff + str(r.randint(1, 99)), gf + "_" + ff])
            if org:
                domain = org.domain
            elif p.origin in ORIGIN_MAIL and r.random() < 0.45:
                domain = r.choice(ORIGIN_MAIL[p.origin])
            else:
                domain = r.choice(FREE_MAIL[self.lang])
            p.email[key] = local + "@" + domain
        return p.email[key]

    def nickname(self, p: Person) -> str:
        """A nickname, drawn once: from the lexicon, or made from the name: the first
        syllable doubled ("Juju", "Lulu"), a short form ("Nico", "Fred"), the family name
        with -y ("Smithy")."""
        if p.nick is None:
            r = self.rng
            given = p.given.split("-")[0].split(" ")[0]
            family = p.family.split(" ")[-1].split("-")[0]
            x = r.random()
            value = ""
            if x < 0.65 and len(self.lex["nicknames"]):
                value = self.pick("nicknames", lambda f: len(f) < 2 or f[1] in (p.gender, "x"), ("gender", p.gender))
            elif x < 0.8 and self.lang == "fr":
                vowels = "aeiouyàâäéèêëîïôöùûü"
                m = re.match(r"([^%s]{1,2}[%s])" % (vowels, vowels), given.lower())
                if m and len(given) > len(m.group(1)) + 1:
                    value = m.group(1).capitalize() + m.group(1)
            elif x < 0.9 and len(given) >= 6:
                value = given[:4] if given[3].lower() not in "aeiouy" else given[:3]
            elif self.lang == "en" and family[-1:].lower() not in "aeiouy" and " " not in family:
                value = family + "y"
            p.nick = value or self.pick("nicknames") or given
        return p.nick

    def stage_name(self, p: Person) -> str:
        """A stage name, drawn once: "DJ Kerlys", "Lady Nova", "MC Zentiq", "Lil Momo"."""
        if p.stage is None:
            r = self.rng
            base = self.invented_word() if r.random() < 0.6 else self.nickname(p)
            p.stage = r.choice(["DJ ", "MC ", "Lil ", "Big ", "Lady " if p.gender == "f" else "King ",
                                "Miss " if p.gender == "f" else "Young ", "", "", ""]) + base
        return p.stage

    def country(self, p: Person) -> str:
        """The country of a person, drawn once with the weights of the language of the text."""
        if p.country is None:
            codes = list(COUNTRIES)
            weights = [COUNTRIES[c][3 if self.lang == "fr" else 4] for c in codes]
            p.country = self.rng.choices(codes, weights)[0]
        return p.country

    def nationality(self, p: Person, feminine: Optional[bool] = None) -> str:
        masc, fem, english = COUNTRIES[self.country(p)][:3]
        if self.lang == "en":
            return english
        return fem if (p.gender == "f" if feminine is None else feminine) else masc

    def national_number(self, p: Person) -> list:
        """The label and the value of the national number of a person of another country:
        "BSN 123456782", "PESEL : 85010112345"."""
        if p.natid is None:
            code = self.country(p)
            make, labels = COUNTRIES[code][5], COUNTRIES[code][6]
            p.natid = (self.rng.choice(labels), make(self.rng))
        label, value = p.natid
        return [(label + self.rng.choice([" ", " : ", ": ", " n° " if self.lang == "fr" else " no. "]), None),
                (value, "ID")]

    def birth(self, p: Person):
        if p.birth is None:
            p.birth = (self.rng.randint(1940, 2006), self.rng.randint(1, 12), self.rng.randint(1, 28))
        return p.birth

    def nir(self, p: Person) -> str:
        if p.nir is None:
            r = self.rng
            y, m, _ = self.birth(p)
            dept = r.choice(["%02d" % r.randint(1, 95), "%02d" % r.randint(1, 95), "2A", "2B"])
            body = ("1" if p.gender == "m" else "2") + "%02d" % (y % 100) + "%02d" % m + dept + \
                "%03d" % r.randint(1, 990) + "%03d" % r.randint(1, 999)
            numeric = body.replace("2A", "19").replace("2B", "18")
            key = 97 - int(numeric) % 97
            full = body + "%02d" % key
            if r.random() < 0.7:
                full = " ".join([full[0], full[1:3], full[3:5], full[5:7], full[7:10], full[10:13], full[13:]])
            p.nir = full
        return p.nir

    # -- places -----------------------------------------------------------------------

    def procedural_city(self) -> str:
        return self.split_value(self.make_city)

    def make_city(self) -> str:
        r = self.rng
        if self.lang == "en":
            a, b, c = PROCEDURAL_CITY["en"]
            return r.choice(a) + r.choice(b) + r.choice(c)
        x = r.random()
        if x < 0.35:
            saints, suffixes = PROCEDURAL_CITY["fr_saint"]
            name = r.choice(saints)
            return name + (r.choice(suffixes) if r.random() < 0.6 else "")
        if x < 0.55:
            name = r.choice(PROCEDURAL_CITY["fr_article"])
            return name + (r.choice(PROCEDURAL_CITY["fr_saint"][1]) if r.random() < 0.5 else "")
        roots, ends = PROCEDURAL_CITY["fr_root"]
        name = r.choice(roots) + r.choice(ends)
        if r.random() < 0.2:
            name += r.choice(PROCEDURAL_CITY["fr_saint"][1])
        return name

    def world_country(self) -> str:
        """A country of WORLD other than the home country of the language, by weight."""
        codes = [c for c in WORLD if WORLD[c]["w"][0 if self.lang == "fr" else 1] > 0]
        weights = [WORLD[c]["w"][0 if self.lang == "fr" else 1] for c in codes]
        return self.rng.choices(codes, weights)[0]

    def place(self, ident: Optional[str], abroad: Optional[bool] = None) -> Place:
        """Place ident, drawn once: abroad when a slot asked for it or for the share of places
        abroad, at home when abroad is False."""
        key = ident or "_%d" % len(self.places)
        if key not in self.places:
            pl = Place()
            if abroad is None:
                abroad = key in self.abroad_places or (self.rng.random() < self.abroad and
                                                       len(self.lex["world_cities"]) > 0)
            if abroad and len(self.lex["world_cities"]) > 0:
                country = self.world_country()
                fields = self.pick_fields("world_cities", lambda f: len(f) > 2 and f[2] == country,
                                          ("country", country))
                pl.city = fields[0]
                pl.postal = fields[1] if len(fields) > 1 else ""
                pl.country = fields[2] if len(fields) > 2 and fields[2] in WORLD else country
                self.places[key] = pl
                return pl
            if self.rng.random() < self.procedural:
                pl.city = self.procedural_city()
                if self.lang == "fr":
                    pl.postal = "%02d%03d" % (self.rng.randint(1, 95), self.rng.randint(0, 99) * 10)
                else:
                    pl.postal = self.rng.choice(["%05d" % self.rng.randint(1000, 99950),
                                                 "%s%d %d%s" % (self.rng.choice(["AB", "BS", "CF", "EX", "LS", "NG"]),
                                                                self.rng.randint(1, 20), self.rng.randint(1, 9),
                                                                self.rng.choice(["AA", "BD", "HQ", "NW"]))])
            else:
                fields = self.pick_fields("cities")
                pl.city = fields[0]
                pl.postal = fields[1] if len(fields) > 1 else ""
            pl.cedex = self.lang == "fr" and self.rng.random() < 0.08
            self.places[key] = pl
        return self.places[key]

    def city_text(self, pl: Place) -> str:
        return pl.city.upper() if self.style["upper_city"] else pl.city

    def country_name(self, pl: Place) -> str:
        return WORLD[pl.country]["fr" if self.lang == "fr" else "en"]

    def world_city_line(self, pl: Place, country_separator: Optional[str]) -> list:
        """The city line of a place abroad, in the layout of its country, and the name of the
        country after country_separator, unlabelled; no country when it is None."""
        w = WORLD[pl.country]
        city = (self.city_text(pl), "CITY")
        postal = pl.postal
        if not postal:
            out = [city]
        elif w["line"] == "pc":
            out = [(postal, "POSTAL_CODE"), (" ", None), city]
        elif w["line"] == "cp":
            out = [city, (" ", None), (postal, "POSTAL_CODE")]
        else:
            # The state or province first: "QC H2X 1Z5", "NSW 2000", "SP 01001-000", "CDMX 06600".
            head, _, tail = postal.partition(" ")
            if not tail:
                out = [city, (" ", None), (postal, "POSTAL_CODE")]
            elif pl.country == "au":
                out = [city, (" " + head + " ", None), (tail, "POSTAL_CODE")]
            elif pl.country == "br":
                out = [city, (" - " + head + ", ", None), (tail, "POSTAL_CODE")]
            elif pl.country == "mx":
                out = [(tail, "POSTAL_CODE"), (" ", None), city, (", " + head, None)]
            elif pl.country == "ca" and self.lang == "fr":
                out = [city, (" (" + CANADA_PROVINCES_FR.get(head, head) + ") ", None), (tail, "POSTAL_CODE")]
            else:
                out = [city, (", " + head + " ", None), (tail, "POSTAL_CODE")]
        if country_separator is not None:
            out.append((country_separator + self.country_name(pl), None))
        return out

    def world_street(self, pl: Place) -> str:
        """The street line of a place abroad: "Hauptstraße 12", "Calle Mayor, 5", "12 rue
        Mohammed V", "2-11-3 Meguro"."""
        r = self.rng
        name = self.pick("world_streets", lambda f: len(f) > 1 and f[1] == pl.country,
                         ("country", pl.country)) or "Main Street"
        order = WORLD[pl.country]["street"]
        number = str(r.choice([r.randint(1, 40), r.randint(1, 180)]))
        if order == "block":
            return "%d-%d-%d %s" % (r.randint(1, 5), r.randint(1, 30), r.randint(1, 20), name)
        if order == "after":
            return name + " " + number
        if order == "comma":
            return name + ", " + number
        if order == "no":
            return name + r.choice([" No: ", " No:", " No. "]) + number
        if order == "nr":
            return name + r.choice([" nr. ", " nr "]) + number
        # "12, rue Mohammed V": the comma of the French-speaking countries.
        comma = pl.country in ("lu", "ma", "dz", "tn", "sn", "ci", "cm") or (pl.country == "ca" and self.lang == "fr")
        return number + (", " if comma and r.random() < 0.3 else " ") + name

    def english_city_line(self, pl: Place, separator: str) -> list:
        """Boston, MA 02108 for a ZIP code; London SW1A 1AA or the postcode on its own
        line for a British one."""
        city = (self.city_text(pl), "CITY")
        if not pl.postal:
            return [city]
        head, _, tail = pl.postal.partition(" ")
        if tail and head.isalpha() and head.isupper() and 2 <= len(head) <= 3:
            # "MA 02108", "ON M5H 2N2", "NSW 2000": the state stays readable.
            return [city, (", " + head + " ", None), (tail, "POSTAL_CODE")]
        if pl.postal[0].isdigit():
            state = self.rng.choice(["MA", "IL", "WA", "TX", "CO", "NY", "CA", "OR"])
            return [city, (", " + state + " ", None), (pl.postal, "POSTAL_CODE")]
        return [city, (separator if separator != ", " else " ", None), (pl.postal, "POSTAL_CODE")]

    def street(self, pl: Place, numbered: bool = False) -> str:
        """The street line of a place, drawn once; numbered guarantees a house number, for
        the templates that write "au {STREET:numbered#1}"."""
        if pl.street is None and pl.country:
            pl.street = self.world_street(pl)
        if pl.street is None:
            r = self.rng
            name = self.pick("streets")
            if self.lang == "fr":
                number = str(r.choice([r.randint(1, 40), r.randint(1, 250)]))
                if r.random() < 0.08:
                    number += r.choice([" bis", " ter", "bis", " B"])
                stype = self.pick("street_types") or "rue"
                if r.random() < 0.15:
                    stype = stype[:1].upper() + stype[1:]
                sep = r.choice([" ", " ", ", "]) if r.random() < 0.1 else " "
                pl.street = number + sep + stype + " " + name
                if r.random() < 0.06 and not numbered:
                    pl.street = stype[:1].upper() + stype[1:] + " " + name
            else:
                number = str(r.randint(1, 450))
                if r.random() < 0.05:
                    number += "B"
                pl.street = number + " " + name
                if r.random() < 0.08 and pl.complement is None:
                    # "Flat 3, 12 High Street": the flat is a complement, a span of its own.
                    pl.complement = r.choice(["Flat %d" % r.randint(1, 30), "Apt %d" % r.randint(1, 30),
                                              "Unit %d" % r.randint(1, 12)])
                    pl.flat_first = True
        elif numbered and self.lang == "fr" and not pl.country and not pl.street[:1].isdigit():
            pl.street = str(self.rng.randint(1, 40)) + " " + pl.street[:1].lower() + pl.street[1:]
        return pl.street

    def street_fragments(self, pl: Place, numbered: bool = False) -> list:
        """The street line, after the flat it opens with in English ("Flat 3, 12 High
        Street"), each a span of its own."""
        street = self.street(pl, numbered)
        if pl.flat_first:
            return [(pl.complement, "STREET_ADDRESS"), (", ", None), (street, "STREET_ADDRESS")]
        return [(street, "STREET_ADDRESS")]

    def complement(self) -> str:
        r = self.rng
        if self.lang == "fr":
            return r.choice(["Bâtiment %s" % r.choice("ABCDE"), "Bât. %s" % r.choice("ABCDE"),
                             "Appartement %d" % r.randint(1, 60), "Apt %d" % r.randint(1, 60),
                             "Résidence Les %s" % r.choice(["Mimosas", "Tilleuls", "Hortensias", "Glycines"]),
                             "BP %d" % r.randint(10, 9999), "CS %d" % r.randint(10000, 99999),
                             "Lieu-dit %s" % self.procedural_city(), "Escalier %d" % r.randint(1, 6)])
        return r.choice(["Suite %d" % r.randint(100, 999), "PO Box %d" % r.randint(10, 9999),
                         "Building %s" % r.choice("ABCD"), "Floor %d" % r.randint(1, 20)])

    # -- organisations -----------------------------------------------------------------

    def invented_word(self) -> str:
        r = self.rng
        word = self.pick("org_words")
        if not word or r.random() < 0.5:
            onsets = ["Al", "Bri", "Cal", "Dex", "El", "Fen", "Gal", "Hel", "Ior", "Kal", "Lum", "Mer", "Nov",
                      "Op", "Pra", "Qua", "Ral", "Sol", "Tal", "Ver", "Xan", "Zen", "Ker", "Arm", "Vel"]
            codas = ["ys", "exa", "ovia", "iq", "oris", "atis", "ea", "ane", "anis", "ix", "eo", "ion", "ance",
                     "ora", "ium", "elle", "is", "ova"]
            word = self.split_value(lambda: r.choice(onsets) + r.choice(codas))
        return word

    # The categories of known_orgs.txt drawn for each kind of organisation, and how often.
    KNOWN_KINDS = {"company": (("company", "retail", "media", "airline", "transport", "telecom", "energy",
                                "insurer"), 0.1),
                   "shop": (("retail",), 0.15), "bank": (("bank", "insurer"), 0.3), "school": (("university",), 0.12),
                   "health": (("hospital",), 0.08), "association": (("ngo", "club"), 0.12)}

    def known_org(self, kind: str) -> Optional[Tuple[str, str]]:
        """An organisation known by its own name, for a share of the organisations of a
        kind: the name and the word of its domain."""
        r = self.rng
        categories, share = self.KNOWN_KINDS.get(kind, ((), 0))
        if not categories or not len(self.lex["known_orgs"]) or r.random() >= share:
            return None
        name = self.pick("known_orgs", lambda f: len(f) > 1 and f[1] in categories, ("categories", categories))
        if not name:
            return None
        return name, re.sub(r"[^a-z0-9]", "", ascii_fold(name).lower())

    def world_org_name(self, kind: str, pl: Place) -> Optional[str]:
        """The name of an organisation of a place abroad, in the forms of its country."""
        w = WORLD[pl.country]
        group = {"court": "public", "police": "public", "council": "public", "prefecture": "public",
                 "law": "company", "shop": "company"}.get(kind, kind)
        patterns = w["orgs"].get(group) or (w["orgs"]["company"] if group not in ("public",) else None)
        if not patterns:
            return None
        r = self.rng
        origin = w["origin"]
        return r.choice(patterns).format(
            w=self.invented_word(), f=self.make_family(origin), f2=self.make_family(origin), c=pl.city,
            g=self.make_given(r.choice("fm")), s=r.choice(w["sectors"]), F=r.choice(w["forms"]))

    def org_name(self, kind: str, key: str) -> Tuple[str, str]:
        """The name of an organisation of the given kind, and the word its domain is
        made from."""
        r = self.rng
        known = self.known_org(kind)
        if known:
            return known
        # French bodies (prefectures, courts, police) stay at home; the others may lie abroad.
        pl = self.place("_org" + key, False if kind in ("court", "police", "council", "prefecture") else None)
        if pl.country:
            name = self.world_org_name(kind, pl)
            if name:
                return name, re.sub(r"[^a-z0-9]", "", ascii_fold(name.split(" ")[0]).lower()) or "org"
            pl = self.place("_org_home" + key, False)
        fam = self.make_family()
        fam2 = self.make_family()
        word = self.invented_word()
        city = pl.city
        of_city = french_de(city)
        forms, sectors = ORG_FORMS[self.lang], ORG_SECTORS[self.lang]
        if self.lang == "fr":
            patterns = {
                "company": [word + " " + r.choice(forms), word + " " + r.choice(sectors), r.choice(forms) + " " + word,
                            fam + " " + r.choice(sectors), fam + " et Fils", fam + " & Associés", "Groupe " + word,
                            word + " " + r.choice(sectors) + " " + r.choice(forms), word.upper(),
                            "Transports " + fam, "Établissements " + fam, "Ets " + fam, fam + " " + r.choice(forms),
                            word + " " + r.choice(["France", "Ouest", "Atlantique", "Grand Ouest", "Europe"])],
                "shop": ["Boulangerie " + fam, "Garage " + fam, "Boucherie " + fam, "Plomberie " + fam,
                         "Pharmacie de " + r.choice(["la Gare", "l'Église", "la Place", "la Mairie", "la Plage"]),
                         "Restaurant Le " + r.choice(["Galion", "Grand Large", "Relais", "Cormoran", "Vieux Port"]),
                         "Coiffure " + word, "Taxi " + fam, "Fleurs " + word, "Auto-école " + word],
                "health": ["Clinique Saint-" + r.choice(["Joseph", "Vincent", "Laurent", "Grégoire", "Yves", "Michel"]),
                           "Centre hospitalier " + of_city, "CHU " + of_city, "Laboratoire " + word,
                           "Cabinet médical " + fam, "Centre dentaire " + word, "EHPAD Les " +
                           r.choice(["Tilleuls", "Hortensias", "Glycines", "Jardins", "Mouettes"]),
                           "Pôle santé " + word],
                "school": ["Lycée " + self.make_given("m") + " " + fam, "Collège " + self.make_given("f") + " " + fam,
                           "Université " + of_city, "IUT " + of_city, "École " + r.choice(["Sainte-Marie", "Jules Verne",
                                                                                          "des Tilleuls", "du Centre"]),
                           "Institut " + word, "Groupe scolaire " + fam],
                "public": ["Mairie " + of_city, "Ville " + of_city, "Commune " + of_city, city + " Métropole",
                           "Communauté de communes " + of_city, "Syndicat des eaux " + of_city,
                           "Tribunal judiciaire " + of_city, "Tribunal administratif " + of_city,
                           "Conseil de prud'hommes " + of_city, "Cour d'appel " + of_city,
                           "Tribunal de commerce " + of_city, "CPAM " + of_city, "CAF " + of_city,
                           "Centre des finances publiques " + of_city, "Commissariat " + of_city,
                           "Agence France Travail " + of_city, "Préfecture " + of_city,
                           "Centre communal d'action sociale " + of_city],
                "court": ["Tribunal judiciaire " + of_city, "Tribunal administratif " + of_city,
                          "Tribunal de commerce " + of_city, "Conseil de prud'hommes " + of_city,
                          "Cour d'appel " + of_city, "Tribunal de proximité " + of_city],
                "police": ["Commissariat " + of_city, "Commissariat de police " + of_city,
                           "Brigade de gendarmerie " + of_city, "Police municipale " + of_city],
                "council": ["Mairie " + of_city, "Conseil municipal " + of_city, "Ville " + of_city,
                            "Commune " + of_city],
                "prefecture": ["Préfecture " + of_city, "Sous-préfecture " + of_city],
                "association": ["Association " + r.choice(["Les Amis ", "Sportive ", "Culturelle "]) + of_city,
                                "Amicale des " + r.choice(["Pompiers", "Anciens", "Locataires"]) + " " + of_city,
                                "Club " + word, "Comité des fêtes " + of_city, "Union sportive " + of_city],
                "law": ["Cabinet " + fam, "Cabinet " + fam + " Avocats", fam + " & Associés", "SCP " + fam + "-" + fam2,
                        "Étude " + fam + " notaires", "SELARL " + fam],
                "bank": ["Banque " + word, "Crédit " + word + " de l'Ouest", word + " Assurances", "Mutuelle " + word,
                         "Caisse " + word, word + " Finance"],
            }
        else:
            patterns = {
                "company": [word + " " + r.choice(forms), word + " " + r.choice(sectors), fam + " & Sons",
                            fam + " " + r.choice(sectors) + " " + r.choice(forms), word + " Group", word.upper(),
                            word + " " + r.choice(sectors) + " " + r.choice(forms)],
                "shop": [fam + "'s Bakery", fam + " Motors", "The " + word + " Café", fam + " Plumbing"],
                "health": ["St " + r.choice(["Mary's", "John's", "Luke's", "Thomas'"]) + " Hospital",
                           city + " General Hospital", fam + " Medical Practice", word + " Clinic"],
                "school": ["University of " + city, city + " College", "St " + r.choice(["Anne's", "Paul's"]) +
                           " School", word + " Academy"],
                "public": [city + " City Council", city + " Borough Council", "Town of " + city,
                           city + " Crown Court", city + " Magistrates' Court", city + " County Court",
                           city + " Jobcentre Plus", city + " Police Department", "City of " + city,
                           city + " County Council", city + " Housing Authority"],
                "court": [city + " Crown Court", city + " County Court", city + " Magistrates' Court",
                          city + " Employment Tribunal", city + " Family Court"],
                "police": [city + " Police Department", city + " Police Station", city + " Police"],
                "council": [city + " City Council", city + " Borough Council", city + " County Council",
                            city + " District Council"],
                "association": [city + " Rugby Club", "Friends of " + city + " Library", word + " Society"],
                "law": [fam + " Solicitors", fam + " & " + fam2 + " LLP", fam + " Law"],
                "bank": [word + " Bank", word + " Building Society", word + " Insurance"],
            }
        if kind == "prefecture" and kind not in patterns:
            kind = "council"
        kind = kind if kind in patterns else "company"
        return r.choice(patterns[kind]), (word if r.random() < 0.6 else fam)

    # The gender of the head nouns of public body names, for the article written before them;
    # a name that starts with its city ("Rennes Métropole") takes none.
    FEMININE_HEADS = {"mairie", "ville", "commune", "communauté", "cour", "cpam", "caf", "agence", "préfecture",
                      "sous-préfecture", "caisse", "direction", "maison", "brigade", "police"}
    MASCULINE_HEADS = {"syndicat", "tribunal", "conseil", "centre", "commissariat"}

    def article(self, name: str) -> str:
        if self.lang == "en":
            return "the "
        head = name.split(" ", 1)[0].lower()
        if head not in self.FEMININE_HEADS and head not in self.MASCULINE_HEADS:
            return ""
        if french_elides_before(name):
            return "l'"
        return "la " if head in self.FEMININE_HEADS else "le "

    def org(self, ident: Optional[str], kind: Optional[str] = None) -> Org:
        key = ident or "_%d" % len(self.orgs)
        if key not in self.orgs:
            r = self.rng
            kind = kind or self.org_kinds.get(key)
            o = Org()
            o.name, base = self.org_name(kind or "company", key)
            o.article = self.article(o.name) if kind in ("public", "court", "police", "council", "prefecture") else ""
            base = ascii_fold(base).lower().replace(" ", "").replace("'", "")
            pl = self.places.get("_org" + key)
            if pl is not None and pl.country and r.random() < 0.7:
                o.domain = base + "." + WORLD[pl.country]["tld"]
            else:
                o.domain = base + "." + r.choice(["fr", "fr", "com", "eu", "net"] if self.lang == "fr"
                                                  else ["com", "co.uk", "com", "io", "net"])
            siren = "".join(str(r.randint(0, 9)) for _ in range(8))
            siren += str(luhn_digit(siren))
            nic = "".join(str(r.randint(0, 9)) for _ in range(4))
            nic += str(luhn_digit(siren + nic))
            o.siren, o.siret = siren, siren + nic
            self.orgs[key] = o
        return self.orgs[key]

    # -- values found by the scanners ------------------------------------------------

    def date_text(self, y: int, m: int, d: int, variant: Optional[str]) -> str:
        r = self.rng
        fmt = variant or self.style["date_format"]
        if fmt == "iso":
            return "%04d-%02d-%02d" % (y, m, d)
        if self.lang == "fr":
            if fmt == "long":
                return ("1er" if d == 1 else str(d)) + " " + FR_MONTHS[m - 1] + " " + str(y)
            sep = {"slash": "/", "dash": "-", "dot": "."}.get(fmt, "/")
            return "%02d%s%02d%s%04d" % (d, sep, m, sep, y)
        if fmt == "long" or fmt in ("dash", "dot"):
            return r.choice(["%s %d, %d" % (EN_MONTHS[m - 1], d, y), "%d %s %d" % (d, EN_MONTHS[m - 1], y)])
        return "%d %s %d" % (d, EN_MONTHS[m - 1], y)

    def world_phone(self) -> str:
        """A telephone number of another country in its international form: "+49 30 12345678"."""
        r = self.rng
        prefix, layouts = WORLD[self.world_country()]["phone"]
        number = "".join(str(r.randint(0, 9)) if ch == "#" else ch for ch in r.choice(layouts))
        return prefix + " " + number

    def world_iban(self) -> str:
        """The IBAN of another country, with valid check digits."""
        r = self.rng
        codes = [c for c in WORLD if WORLD[c]["iban"] and WORLD[c]["w"][0 if self.lang == "fr" else 1] > 0]
        code = r.choices(codes, [WORLD[c]["w"][0 if self.lang == "fr" else 1] for c in codes])[0]
        bban = ""
        for count, kind in re.findall(r"(\d+)([an])", WORLD[code]["iban"]):
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if kind == "a" else "0123456789"
            bban += "".join(r.choice(alphabet) for _ in range(int(count)))
        cc = code.upper()
        iban = cc + "%02d" % (98 - mod97(bban + cc + "00")) + bban
        return " ".join(iban[i:i + 4] for i in range(0, len(iban), 4)) if r.random() < 0.7 else iban

    def phone_text(self) -> str:
        r = self.rng
        if self.abroad and r.random() < self.abroad * 0.6:
            return self.world_phone()
        if self.lang == "fr":
            first = r.choice(["06", "07", "06", "02", "01", "03", "04", "05", "09"])
            digits = first + "".join("%02d" % r.randint(0, 99) for _ in range(4))
            fmt = self.style["phone_format"]
            if fmt == "intl":
                return "+33 " + digits[1] + " " + " ".join(digits[i:i + 2] for i in range(2, 10, 2))
            if fmt == "compact":
                return digits
            sep = "." if fmt == "dotted" else " "
            return sep.join(digits[i:i + 2] for i in range(0, 10, 2))
        kind = r.random()
        if kind < 0.5:
            return r.choice(["+44 20 %04d %04d" % (r.randint(1000, 9999), r.randint(0, 9999)),
                             "+44 7700 %06d" % r.randint(0, 999999)])
        if kind < 0.8:
            return "(%03d) %03d-%04d" % (r.randint(201, 989), r.randint(200, 999), r.randint(0, 9999))
        return "+1 %03d %03d %04d" % (r.randint(201, 989), r.randint(200, 999), r.randint(0, 9999))

    def iban_text(self) -> str:
        r = self.rng
        if self.abroad and r.random() < self.abroad * 0.5:
            return self.world_iban()
        if self.lang == "en" and r.random() < 0.8:
            bban = "".join(r.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(4)) + "%06d%08d" % (
                r.randint(0, 999999), r.randint(0, 99999999))
            iban = "GB%02d" % (98 - mod97(bban + "GB00")) + bban
            return " ".join(iban[i:i + 4] for i in range(0, len(iban), 4)) if r.random() < 0.7 else iban
        bank = "%05d" % r.randint(10000, 99999)
        branch = "%05d" % r.randint(0, 99999)
        account = "%011d" % r.randint(0, 99999999999)
        key = 97 - (89 * int(bank) + 15 * int(branch) + 3 * int(account)) % 97
        bban = bank + branch + account + "%02d" % key
        check = 98 - mod97(bban + "FR00")
        iban = "FR%02d" % check + bban
        return " ".join(iban[i:i + 4] for i in range(0, len(iban), 4)) if r.random() < 0.7 else iban

    def card_text(self) -> str:
        r = self.rng
        payload = r.choice(["4", "5"]) + "".join(str(r.randint(0, 9)) for _ in range(14))
        card = payload + str(luhn_digit(payload))
        return " ".join(card[i:i + 4] for i in range(0, 16, 4)) if r.random() < 0.8 else card

    def ref_text(self) -> str:
        r = self.rng
        return r.choice(["%d-%05d" % (r.randint(2019, 2026), r.randint(1, 99999)),
                         "%s-%06d" % (r.choice(["CL", "DOS", "CMD", "FAC", "SIN", "RES", "TCK"]),
                                      r.randint(1, 999999)),
                         "%07d" % r.randint(1000000, 9999999),
                         "%s%d" % (r.choice(["A", "B", "C", "K", "X"]), r.randint(100000, 9999999))])

    def national_id(self, kind: str) -> str:
        """Identifiers with a fixed layout, valid where they carry a check: a British
        National Insurance number, a US Social Security number, a French passport
        number, an IMEI, a UUID and a vehicle identification number."""
        r = self.rng
        if kind == "NINO":
            first = r.choice("ABCEGHJKLMNOPRSTWXYZ")
            second = r.choice("ABCEGHJKLMNPRSTWXYZ")
            digits = "%06d" % r.randint(0, 999999)
            suffix = r.choice("ABCD")
            if first + second in ("BG", "GB", "KN", "NK", "NT", "TN", "ZZ"):
                first = "A"
            return first + second + " " + digits[:2] + " " + digits[2:4] + " " + digits[4:] + " " + suffix \
                if r.random() < 0.6 else first + second + digits + suffix
        if kind == "SSN":
            return "%03d-%02d-%04d" % (r.choice([r.randint(1, 665), r.randint(667, 899)]), r.randint(1, 99),
                                       r.randint(1, 9999))
        if kind == "PASSPORT":
            letters = "ABCDEFGHJKLMNPRSTUVWXYZ"
            return "%02d%s%s%05d" % (r.randint(1, 99), r.choice(letters), r.choice(letters), r.randint(0, 99999))
        if kind == "IMEI":
            payload = "35" + "".join(str(r.randint(0, 9)) for _ in range(12))
            imei = payload + str(luhn_digit(payload))
            return imei if r.random() < 0.5 else imei[:2] + "-" + imei[2:8] + "-" + imei[8:14] + "-" + imei[14]
        if kind == "UUID":
            h = "%032x" % r.getrandbits(128)
            return h[:8] + "-" + h[8:12] + "-" + h[12:16] + "-" + h[16:20] + "-" + h[20:]
        # A vehicle identification number with the check digit of its position 9.
        alphabet = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"
        vin = list(r.choice(["VF1", "VF3", "VF7", "WVW", "WBA", "ZFA", "VSS", "1HG", "JHM"]) + "".join(
            r.choice(alphabet) for _ in range(14)))
        values = {c: v for c, v in zip("ABCDEFGHJKLMNPRSTUVWXYZ", [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 7, 9, 2, 3,
                                                                      4, 5, 6, 7, 8, 9])}
        weights = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2]
        total = sum((int(c) if c.isdigit() else values[c]) * w for c, w in zip(vin, weights))
        vin[8] = "X" if total % 11 == 10 else str(total % 11)
        return "".join(vin)

    # -- rendering --------------------------------------------------------------------

    def render(self, nodes) -> list:
        out = []
        for node in nodes:
            if node[0] == "text":
                out.append((node[1], None))
            elif node[0] == "choice":
                out.extend(self.render(self.rng.choice(node[1])))
            else:
                out.extend(self.slot(node))
        return out

    def slot(self, node) -> list:
        _, name, variant, ident, extra = node
        r = self.rng
        fr = self.lang == "fr"
        if name == "g":
            p = self.person(ident)
            return [(extra[1] if p.gender == "f" else extra[0], None)]
        if name == "GENDER":
            return []
        if name in ("e", "il", "Il", "le", "Le", "un", "he", "He", "his", "His", "him"):
            p = self.person(ident)
            fem = p.gender == "f"
            forms = {"e": ("", "e"), "il": ("il", "elle"), "Il": ("Il", "Elle"), "le": ("le", "la"),
                     "Le": ("Le", "La"), "un": ("un", "une"), "he": ("he", "she"), "He": ("He", "She"),
                     "his": ("his", "her"), "His": ("His", "Her"), "him": ("him", "her")}[name]
            return [(forms[1] if fem else forms[0], None)]
        if name == "PERSON":
            return self.person_fragments(self.person(ident), variant)
        if name == "GIVEN":
            p = self.person(ident)
            return [(initials(p.given) if variant == "initial" else p.given, "GIVEN_NAME")]
        if name == "FAMILY":
            return [(self.family_text(self.person(ident), variant), "FAMILY_NAME")]
        if name == "BIRTHNAME":
            p = self.person(ident)
            if not p.birth_name:
                p.birth_name = self.make_family()
            return [(p.birth_name.upper() if self.style["upper_family"] else p.birth_name, "FAMILY_NAME")]
        if name == "TITLE":
            # A civility tells the sex of the person: it is replaced, as TITLE.
            p = self.person(ident)
            if fr:
                if variant == "long":
                    return [("Madame" if p.gender == "f" else "Monsieur", "TITLE")]
                if variant == "short":
                    return [("Mme" if p.gender == "f" else "M.", "TITLE")]
                return [(r.choice(["Mme", "Mme", "Madame"]) if p.gender == "f" else r.choice(["M.", "M.", "Monsieur"]),
                         "TITLE")]
            if variant == "long":
                return [("Madam" if p.gender == "f" else "Sir", "TITLE")]
            return [(r.choice(["Mrs", "Ms", "Ms.", "Mrs."]) if p.gender == "f" else r.choice(["Mr", "Mr."]), "TITLE")]
        if name == "SEX":
            p = self.person(ident)
            values = SEX_VALUES[self.lang][p.gender]
            value = values[0] if variant == "short" else (values[2] if variant == "long" else r.choice(values))
            return [(value.lower() if variant == "long" else value, "GENDER")]
        if name == "NATIONALITY":
            p = self.person(ident)
            return [(self.nationality(p, True if variant == "f" else None), "NATIONALITY")]
        if name == "NATID":
            p = self.person(ident)
            if variant and variant in COUNTRIES and COUNTRIES[variant][5] is not None:
                p.country = variant
            return self.national_number(p)
        if name == "EMAIL":
            p = self.person(ident)
            o = self.org(extra) if extra else None
            return [(self.email(p, o), "EMAIL")]
        if name == "USERNAME":
            return [(self.username(self.person(ident)), "USERNAME")]
        if name == "NICK":
            p = self.person(ident)
            return [(self.stage_name(p) if variant == "stage" else self.nickname(p), "GIVEN_NAME")]
        if name == "HANDLE":
            p = self.person(ident)
            if p.handle is None:
                p.handle = "@" + self.username(p).replace(".", "_")
            return [(p.handle, "USERNAME")]
        if name == "BIRTHDATE":
            y, m, d = self.birth(self.person(ident))
            return [(self.date_text(y, m, d, variant), "DATE")]
        if name == "NIR":
            return [(self.nir(self.person(ident)), "NIR")]
        if name == "PHONE":
            if ident is not None:
                p = self.person(ident)
                if p.phone is None:
                    p.phone = self.phone_text()
                return [(p.phone, "PHONE")]
            return [(self.phone_text(), "PHONE")]
        if name == "URL":
            p = self.person(ident)
            slug = ascii_fold(p.given + "-" + p.family).lower().replace(" ", "-").replace("'", "")
            return [(r.choice(["https://www.linkedin.com/in/%s-%x" % (slug, r.randint(4096, 65535)),
                               "https://github.com/" + self.username(p).replace(".", "-")]), "URL")]
        if name == "CITY":
            pl = self.place(ident)
            return [(pl.city.upper() if variant == "upper" else self.city_text(pl), "CITY")]
        if name == "POSTAL":
            pl = self.place(ident)
            postal = pl.postal
            if pl.country and WORLD[pl.country]["line"] == "state" and " " in postal:
                postal = postal.partition(" ")[2]
            return [(postal, "POSTAL_CODE")]
        if name == "CITYLINE":
            pl = self.place(ident)
            if pl.country:
                return self.world_city_line(pl, r.choice(["\n", ", ", None]))
            if fr:
                out = [(pl.postal, "POSTAL_CODE"), (" ", None)] if pl.postal else []
                out.append((pl.city.upper() if r.random() < 0.6 else self.city_text(pl), "CITY"))
                if pl.cedex:
                    out.append((" CEDEX" + (" %d" % r.randint(1, 9) if r.random() < 0.4 else ""), None))
                return out
            return self.english_city_line(pl, "\n" if r.random() < 0.5 else " ")
        if name == "STREET":
            return self.street_fragments(self.place(ident), variant == "numbered")
        if name == "COMPLEMENT":
            pl = self.place(ident)
            if pl.complement is None:
                pl.complement = self.complement()
            return [(pl.complement, "STREET_ADDRESS")]
        if name == "ADDRESS":
            pl = self.place(ident)
            out = self.street_fragments(pl) + [(", ", None)]
            if pl.country:
                return out + self.world_city_line(pl, ", " if r.random() < 0.6 else None)
            if fr:
                if pl.postal:
                    out += [(pl.postal, "POSTAL_CODE"), (" ", None)]
                return out + [(self.city_text(pl), "CITY")]
            return out + self.english_city_line(pl, ", " if r.random() < 0.5 else " ")
        if name == "ADDRESS_BLOCK":
            pl = self.place(ident)
            out = self.street_fragments(pl) + [("\n", None)]
            if r.random() < 0.3 and not pl.flat_first:
                if pl.complement is None:
                    pl.complement = self.complement()
                out += [(pl.complement, "STREET_ADDRESS"), ("\n", None)]
            if pl.country:
                return out + self.world_city_line(pl, "\n" if r.random() < 0.7 else None)
            return out + self.slot(("slot", "CITYLINE", None, ident, None))
        if name == "ORG":
            return [(self.org(ident, variant).name, "ORGANIZATION")]
        if name == "DOMAIN":
            return [(self.org(ident).domain, None)]
        if name in ("ART", "Art"):
            # The article of an organisation: "{ART#2}{ORG:public#2}" gives "la Mairie de Brest".
            a = self.org(ident).article
            return [(a[:1].upper() + a[1:] if name == "Art" else a, None)] if a else []
        if name == "SIRET":
            s = self.org(ident).siret
            return [((s[:3] + " " + s[3:6] + " " + s[6:9] + " " + s[9:]) if r.random() < 0.6 else s, "SIRET")]
        if name == "SIREN":
            s = self.org(ident).siren
            return [((s[:3] + " " + s[3:6] + " " + s[6:]) if r.random() < 0.6 else s, "SIRET")]
        if name == "DATE":
            y = r.randint(2015, 2027)
            return [(self.date_text(y, r.randint(1, 12), r.randint(1, 28), variant), "DATE")]
        if name == "TIME":
            return [("%02d:%02d" % (r.randint(7, 19), r.randint(0, 59)) +
                     (":%02d" % r.randint(0, 59) if r.random() < 0.3 else ""), None)]
        if name == "YEAR":
            year = str(r.randint(1995, 2026))
            return [(DATE_OPEN + year + DATE_CLOSE if variant == "date" else year, None)]
        if name == "AGE":
            return [(str(r.randint(19, 88)), None)]
        if name == "AMOUNT":
            v = r.randint(10, 99999) + r.randint(0, 99) / 100
            x = r.random()
            if fr:
                unit = " €" if x < 0.88 else r.choice([" CHF", " $ CA", " MAD", " DA", " FCFA", " $"])
                return [(("%.2f" % v).replace(".", ",") + unit, None)]
            if x < 0.55:
                return [("£%.2f" % v, None)]
            if x < 0.8:
                return [("$%.2f" % v, None)]
            if x < 0.92:
                return [("€%.2f" % v, None)]
            return [(r.choice(["CA$", "A$", "₹", "R", "¥"]) + "%.2f" % v, None)]
        if name == "IBAN":
            return [(self.iban_text(), "IBAN")]
        if name == "CARD":
            return [(self.card_text(), "CARD")]
        if name == "IP":
            return [("%d.%d.%d.%d" % (r.choice([10, 172, 192, 81, 90, 37, 109]), r.randint(0, 255),
                                      r.randint(0, 255), r.randint(1, 254)), "IPV4")]
        if name == "IPV6":
            return [("2001:db8:%x:%x::%x" % (r.randint(0, 65535), r.randint(0, 65535), r.randint(1, 65535)),
                     "IPV6")]
        if name == "MAC":
            return [(":".join("%02X" % r.randint(0, 255) for _ in range(6)), "MAC")]
        if name == "PLATE":
            letters = "ABCDEFGHJKLMNPQRSTVWXYZ"
            return [("%s%s-%03d-%s%s" % (r.choice(letters), r.choice(letters), r.randint(1, 999), r.choice(letters),
                                         r.choice(letters)), "PLATE")]
        if name == "GEO":
            return [("%.5f, %.5f" % (r.uniform(42.0, 51.0), r.uniform(-4.8, 7.9)), "GEO")]
        if name == "REF":
            return [(self.ref_text(), "ID")]
        if name in ("NINO", "SSN", "PASSPORT", "IMEI", "UUID", "VIN"):
            return [(self.national_id(name), "ID")]
        if name == "NEGATIVE":
            # Capitalized words that are not personal data: {NEGATIVE:body}, {NEGATIVE:event},
            # {NEGATIVE:law}, {NEGATIVE:title}, {NEGATIVE:misc}; any category when none is given.
            category = (variant or "").replace("cap", "").strip(",") or None
            value = self.pick("negatives", (lambda f: len(f) > 1 and f[1] == category) if category else None)
            if variant and "cap" in variant and value:
                value = value[:1].upper() + value[1:]
            return [(value, None)]
        if name == "INSTITUTION":
            # "la [CAF]": the name between brackets is an organisation, a generic reference is not.
            value = self.pick("institutions")
            if variant == "cap" and value:
                value = value[:1].upper() + value[1:] if value[0] != "[" else "[" + value[1:2].upper() + value[2:]
            b, e = value.find("["), value.find("]")
            if b < 0 or e < b:
                return [(value.replace("[", "").replace("]", ""), None)]
            out = [(value[:b], None), (value[b + 1:e], "ORGANIZATION"), (value[e + 1:], None)]
            return [f for f in out if f[0]]
        if name in ("JOB", "REGION", "COUNTRY", "PRODUCT"):
            lex = {"JOB": "jobs", "REGION": "regions", "COUNTRY": "countries", "PRODUCT": "products"}[name]
            value = self.pick(lex)
            if variant == "cap" and value:
                value = value[:1].upper() + value[1:]
            return [(value, None)]
        raise ValueError("unknown slot {%s}" % name)


def settle_titles(fragments: list) -> list:
    """A civility is replaced as TITLE where the detector replaces it: before a name,
    after a field label ("Civilité : Mme"), and in a cell of its own in a row that holds a
    name ("Mme;CARRÉ;Stéphanie"; three cells at least, four when separated by commas). A
    salutation alone ("Madame,") stays unlabelled. In "M. et Mme Martin" both civilities
    go with the name."""
    out = [list(f) for f in fragments]
    visible = [k for k in range(len(out)) if out[k][0]]
    starts, pos = [], 0
    for f in out:
        starts.append(pos)
        pos += len(f[0])
    full = "".join(f[0] for f in out)

    def after(k, step):
        i = visible.index(k) + step
        return out[visible[i]] if 0 <= i < len(visible) else None

    def in_cell(k):
        b, e = starts[k], starts[k] + len(out[k][0])
        line_b = full.rfind("\n", 0, b) + 1
        line_e = full.find("\n", e)
        line_e = len(full) if line_e < 0 else line_e
        named = any(out[j][1] in ("GIVEN_NAME", "FAMILY_NAME") and line_b <= starts[j] < line_e
                    for j in range(len(out)))
        if not named:
            return False
        line = full[line_b:line_e]
        before, rest = full[line_b:b].rstrip(" \u00a0"), full[e:line_e].lstrip(" \u00a0")
        for sep, cells in (("|", 3), ("\t", 3), (";", 3), (" – ", 3), (",", 4)):
            if len(line.split(sep)) >= cells and (not before or before.endswith(sep.strip(" ") or sep)) and \
                    (not rest or rest.startswith(sep.strip(" ") or sep)):
                return True
        return False

    for k in visible:
        if out[k][1] != "TITLE":
            continue
        nxt, nxt2, prev = after(k, 1), after(k, 2), after(k, -1)
        before_name = nxt is not None and nxt[1] is None and nxt[0] in (" ", "\u00a0") and nxt2 is not None and \
            nxt2[1] in ("GIVEN_NAME", "FAMILY_NAME")
        joined = nxt is not None and nxt[1] is None and nxt[0] in (" et ", " and ") and nxt2 is not None and \
            nxt2[1] == "TITLE"
        labelled = prev is not None and prev[1] is None and prev[0].rstrip(" \u00a0").endswith(":")
        if not (before_name or joined or labelled or in_cell(k)):
            out[k][1] = None
    return [tuple(f) for f in out]


def initials(given: str) -> str:
    parts = given.replace("-", " - ").split()
    out = []
    for part in parts:
        out.append("-" if part == "-" else part[0] + ".")
    return "".join(out)


# ----------------------------------------------------------------------------------
# Articles
#
# A template writes "originaire de {CITY#1}" or "a {JOB}" before knowing the value; the
# words before a slot are made to agree with it once the fragments are rendered.

FR_VOWELS = "aàâäeéèêëiîïoôöuùûüœæ"
FR_ELIDED = {"de": "d'", "que": "qu'", "le": "l'", "la": "l'", "je": "j'"}
FR_CONTRACTED = {("à", "le"): "au", ("à", "les"): "aux", ("de", "le"): "du", ("de", "les"): "des"}
LAST_WORD = re.compile(r"(?:^|(?<=[\s(«\"']))(\w+) $")


def french_elides_before(value: str) -> bool:
    """True when a French article or preposition is elided before value: d'Orléans,
    d'Yves, but de Yann and de Hugo, h being left alone."""
    if not value:
        return False
    c = value[0].lower()
    if c == "y":
        return len(value) > 1 and value[1].lower() not in FR_VOWELS
    return c in FR_VOWELS


def english_takes_an(value: str) -> bool:
    """True when the English article before value is "an": an engineer, an IT consultant,
    an 80 year old, but a university and a user."""
    head = value.split(" ", 1)[0]
    if not head:
        return False
    if head[0].isdigit():
        digits = re.match(r"[0-9]+", head).group(0)
        return digits[0] == "8" or digits in ("11", "18")
    if len(head) > 1 and head.isupper() and head.isalpha():
        return head[0] in "AEFHILMNORSX"
    w = head.lower()
    if w.startswith(("uni", "use", "usu", "ura", "ure", "uro", "uti", "eu", "one", "once")):
        return False
    if w.startswith(("hour", "honest", "honour", "honor", "heir")):
        return True
    return w[0] in "aeiou"


def french_de(name: str) -> str:
    """The preposition "de" and a name, elided or contracted: d'Angers, du Mans, de Rennes."""
    first, _, rest = name.partition(" ")
    if rest and first.lower() in ("le", "les"):
        return ("du " if first.lower() == "le" else "des ") + rest
    return ("d'" if french_elides_before(name) else "de ") + name


def apply_articles(fragments: list, lang: str) -> list:
    """French elision and contraction, English a and an, at the end of each unlabelled
    fragment followed by a value. A contraction also removes the article that starts the
    value: "à" and "Le Mans" give "au" and "Mans", and the label stays on what remains,
    as in a real text where "au Mans" holds the city name "Mans"."""
    out = [list(f) for f in fragments]
    for k in range(len(out) - 1):
        text, label = out[k]
        if label is not None or not text or not text.endswith(" "):
            continue
        n = k + 1
        while n < len(out) and not out[n][0]:
            n += 1
        if n == len(out):
            break
        m = LAST_WORD.search(text)
        if not m:
            continue
        word, value = m.group(1), out[n][0]
        lower = word.lower()
        if lang == "fr" and out[n][1] is None and value in ("le ", "les ") and (lower, value[:-1]) in FR_CONTRACTED:
            # "à" and the article of "le [ministère de la Justice]" give "au".
            fused = FR_CONTRACTED[(lower, value[:-1])]
            out[k][0] = text[:m.start(1)] + (fused.capitalize() if word[0].isupper() else fused) + " "
            out[n][0] = ""
            continue
        if lang == "fr":
            first, _, rest = value.partition(" ")
            key = (lower, first.lower())
            if key in FR_CONTRACTED and rest and out[n][1] in (None, "CITY") and \
                    (first in ("le", "les") or (out[n][1] == "CITY" and first.lower() in ("le", "les"))):
                fused = FR_CONTRACTED[key]
                out[k][0] = text[:m.start(1)] + (fused.capitalize() if word[0].isupper() else fused) + " "
                out[n][0] = rest
            elif lower in FR_ELIDED and french_elides_before(value):
                elided = FR_ELIDED[lower]
                out[k][0] = text[:m.start(1)] + (elided.capitalize() if word[0].isupper() else elided)
        elif lang == "en" and lower == "a" and english_takes_an(value):
            out[k][0] = text[:m.start(1)] + word + "n "
    return [tuple(f) for f in out]


# ----------------------------------------------------------------------------------
# Noise
#
# Applied to the fragments, so that the spans follow: a transformation of a labelled
# fragment changes the value but keeps its label, and the offsets are computed after.

OCR_PAIRS = [("rn", "m"), ("m", "rn"), ("l", "1"), ("I", "l"), ("O", "0"), ("é", "e"), ("e", "c"), ("ii", "ü")]


def apply_style(fragments: list, style: dict, rng: random.Random) -> list:
    out = []
    for text, label in fragments:
        if style.get("no_accents"):
            text = strip_accents(text)
        if style.get("all_upper"):
            text = text.upper()
        elif style.get("all_lower"):
            text = text.lower()
        if label is None:
            if style.get("nbsp_colon"):
                text = text.replace(" :", "\u00a0:")
            if style.get("double_space") and " " in text and rng.random() < 0.3:
                k = text.index(" ")
                text = text[:k] + "  " + text[k + 1:]
        elif style.get("ocr") and rng.random() < 0.25 and len(text) > 3:
            a, b = rng.choice(OCR_PAIRS)
            if a in text:
                k = text.index(a)
                text = text[:k] + b + text[k + len(a):]
        out.append((text, label))
    if style.get("ocr"):
        # A lost space between two labelled fragments, as when a PDF is read too greedily.
        for k in range(1, len(out) - 1):
            if out[k] == (" ", None) and out[k - 1][1] and out[k + 1][1] and rng.random() < 0.3:
                out[k] = ("", None)
    return out


# A year that dates a fact of a person's life ({YEAR:date}) travels between these marks,
# as unlabelled text, until assemble() labels it: the noise and the style then draw the
# same random numbers as for any other year, and the texts stay those of a plain {YEAR}.
DATE_OPEN, DATE_CLOSE = "\ue000", "\ue001"
MONTH_BEFORE = re.compile(r"(?i)(?<![^\W\d_])(%s),? {1,2}$" % "|".join(
    sorted({m for m in FR_MONTHS + EN_MONTHS} | {strip_accents(m) for m in FR_MONTHS}, key=len, reverse=True)))


def assemble(fragments: list) -> Tuple[str, list]:
    """Joins the fragments and returns the text and its spans in byte offsets. Leading
    and trailing spaces of a labelled fragment are left outside its span. A year between
    date marks is labelled DATE, with the month name written just before it."""
    text_parts, spans, dates, pos = [], [], [], 0
    for text, label in fragments:
        if not text:
            continue
        if label:
            data = text.encode("utf-8")
            stripped = text.strip()
            if stripped:
                lead = len(text[:len(text) - len(text.lstrip())].encode("utf-8"))
                b = pos + lead
                e = b + len(stripped.encode("utf-8"))
                spans.append((b, e, label))
            text_parts.append(text)
            pos += len(data)
            continue
        for piece in re.split("([%s%s])" % (DATE_OPEN, DATE_CLOSE), text):
            if piece == DATE_OPEN:
                opened = pos
            elif piece == DATE_CLOSE:
                dates.append((opened, pos))
            elif piece:
                text_parts.append(piece)
                pos += len(piece.encode("utf-8"))
    joined = "".join(text_parts)
    data = joined.encode("utf-8")
    for b, e in dates:
        m = MONTH_BEFORE.search(data[:b].decode("utf-8"))
        if m:
            month = len(data[:b].decode("utf-8")[:m.start()].encode("utf-8"))
            if not any(sb < e and month < se for sb, se, _ in spans):
                b = month
        spans.append((b, e, "DATE"))
    spans.sort()
    return joined, spans


def escape(text: str) -> str:
    out = text.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")
    # A line starting with '#' is a comment: a text starting with it is escaped.
    return "\\" + out if out.startswith("#") else out


def format_example(text: str, spans: list) -> str:
    return escape(text) + "\t" + ",".join("%d:%d:%s" % s for s in spans)


def unescape(text: str) -> str:
    """The inverse of escape()."""
    out, i = [], 0
    while i < len(text):
        if text[i] == "\\" and i + 1 < len(text):
            out.append({"n": "\n", "t": "\t", "r": "\r", "\\": "\\", "#": "#"}[text[i + 1]])
            i += 2
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def check_example(text: str, spans: list):
    data = text.encode("utf-8")
    last = 0
    for b, e, label in spans:
        if not (last <= b < e <= len(data)):
            raise AssertionError("bad span %r in %r" % ((b, e, label), text))
        value = data[b:e].decode("utf-8")
        if value != value.strip():
            raise AssertionError("span with outer space %r in %r" % (value, text))
        if any(c in value for c in "\t\n\r"):
            raise AssertionError("span across a line or a cell %r in %r" % (value, text))
        if label not in MODEL_LABELS and label not in RULE_LABELS:
            raise AssertionError("unknown label %r in %r" % (label, text))
        last = e


# ----------------------------------------------------------------------------------
# Splits


SPLITS = [
    # name, default count, pool, procedural share, noise, genres
    ("train", 40000, "seen", 0.25, 1.0, None),
    ("dev", 4000, "seen", 0.25, 1.0, None),
    ("test", 4000, "all", 0.25, 1.0, None),
    ("test_unseen", 3000, "unseen", 0.5, 1.0, None),
    ("test_noisy", 2000, "all", 0.25, 6.0, None),
    ("test_negatives", 2000, "all", 0.25, 1.0, {"negative", "prose"}),
    ("test_documents", 1000, "all", 0.25, 1.0, "documents"),
    ("test_abroad", 4000, "all", 0.25, 1.0, None),
]
# The splits whose places all lie abroad, whatever --abroad-share says.
ABROAD_SPLITS = {"test_abroad"}


def choose_template(res: Resources, lang: str, rng: random.Random, genres) -> Tuple[str, list]:
    pool = res.templates[lang]
    if genres and genres != "documents":
        pool = [t for t in pool if t[0] in genres] or pool
    total = sum(w for _, w, _ in pool)
    x = rng.random() * total
    for genre, w, nodes in pool:
        x -= w
        if x <= 0:
            return genre, nodes
    return pool[-1][0], pool[-1][2]


def choose_sentence(res: Resources, lang: str, rng: random.Random) -> Optional[list]:
    pool = res.sentences.get(lang) or []
    if not pool:
        return None
    total = sum(w for w, _ in pool)
    x = rng.random() * total
    for w, nodes in pool:
        x -= w
        if x <= 0:
            return nodes
    return pool[-1][1]


def prose_fragments(gen, res: Resources, lang: str, rng: random.Random, low: int, high: int) -> list:
    """low to high context sentences, joined by a space or a line break."""
    fragments = []
    for k in range(rng.randint(low, high)):
        nodes = choose_sentence(res, lang, rng)
        if nodes is None:
            break
        if fragments:
            fragments.append((rng.choice([" ", " ", "\n"]), None))
        fragments.extend(gen.render(nodes))
    return fragments


def with_context(gen, res: Resources, lang: str, rng: random.Random, fragments: list) -> list:
    """The example with a context sentence before it, after it, or both."""
    before, after = rng.random() < 0.6, rng.random() < 0.6
    if not before and not after:
        before = True
    out = []
    if before:
        lead = prose_fragments(gen, res, lang, rng, 1, 2)
        if lead:
            out = lead + [(rng.choice([" ", "\n", "\n\n"]), None)]
    out.extend(fragments)
    if after:
        tail = prose_fragments(gen, res, lang, rng, 1, 2)
        if tail:
            out += [(rng.choice([" ", "\n", "\n\n"]), None)] + tail
    return out


def build_split(res: Resources, name: str, count: int, pool: str, procedural: float, noise: float, genres,
                seed: int, languages: List[str], fr_share: float, stats: dict, context_rate: float = 0.0,
                prose_share: float = 0.0, exclude: Optional[set] = None, abroad: float = 0.0) -> List[str]:
    """count examples; with exclude, the texts it holds are drawn again, a few times at
    most, so that an evaluation split repeats no training text."""
    rng = random.Random("%s|%d" % (name, seed))
    lines = []
    split_stats = {"examples": 0, "spans": {}, "genres": {}, "languages": {}, "redrawn": 0}
    for _ in range(count):
        for attempt in range(30):
            lang = languages[0] if len(languages) == 1 else ("fr" if rng.random() < fr_share else "en")
            gen = Generator(res, lang, rng, pool, procedural, noise, abroad)
            prose_genre = genres is not None and genres != "documents" and "prose" in genres
            if genres == "documents":
                genre, parts = "documents", []
                for k in range(rng.randint(3, 6)):
                    g, nodes = choose_template(res, lang, rng, None)
                    parts.append(nodes)
                gen.new_document("letter")
                for nodes in parts:
                    gen.constrain(nodes)
                fragments = []
                for k, nodes in enumerate(parts):
                    if k:
                        fragments.append(("\n\n", None))
                        # A paragraph without personal data between two parts, now and then.
                        if rng.random() < context_rate:
                            fragments.extend(prose_fragments(gen, res, lang, rng, 1, 3))
                            fragments.append(("\n\n", None))
                    fragments.extend(gen.render(nodes))
            elif res.sentences.get(lang) and ((genres is None and rng.random() < prose_share) or
                                              (prose_genre and rng.random() < 0.5)):
                genre = "prose"
                gen.new_document("prose")
                fragments = prose_fragments(gen, res, lang, rng, 1, 4)
            else:
                genre, nodes = choose_template(res, lang, rng, genres)
                gen.new_document(genre)
                gen.constrain(nodes)
                fragments = gen.render(nodes)
                if rng.random() < context_rate:
                    fragments = with_context(gen, res, lang, rng, fragments)
            fragments = apply_style(apply_articles(settle_titles(fragments), lang), gen.style, rng)
            text, spans = assemble(fragments)
            if exclude is None or text not in exclude:
                break
            split_stats["redrawn"] += 1
        else:
            continue
        check_example(text, spans)
        lines.append(format_example(text, spans))
        split_stats["examples"] += 1
        split_stats["genres"][genre] = split_stats["genres"].get(genre, 0) + 1
        split_stats["languages"][lang] = split_stats["languages"].get(lang, 0) + 1
        for _, _, label in spans:
            split_stats["spans"][label] = split_stats["spans"].get(label, 0) + 1
    stats[name] = split_stats
    return lines


def write_card(out_dir: str, args, res: Resources, stats: dict):
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write("# Synthetic personal data dataset\n\n")
        f.write("Generated by build_pii_dataset.py with seed %d, languages %s.\n\n" % (args.seed, args.languages))
        f.write("Format: one example per line, `text<TAB>begin:end:LABEL,...`, byte offsets into the UTF-8 "
                "text, with `\\\\`, `\\t`, `\\n` and `\\r` escaped. Read it with `dlib::load_pii_dataset()`.\n\n")
        f.write("## Labels\n\n")
        for label in MODEL_LABELS + RULE_LABELS:
            f.write("- `%s`: %s%s\n" % (label, LABEL_DESCRIPTIONS[label],
                                         " (learned)" if label in MODEL_LABELS else ""))
        f.write("\n## Splits\n\n| split | examples | spans |\n|---|---:|---:|\n")
        for name, s in stats.items():
            f.write("| %s | %d | %d |\n" % (name, s["examples"], sum(s["spans"].values())))
        f.write("\n## Resources\n\n")
        for key in sorted(res.origin):
            f.write("- %s: %s\n" % (key, res.origin[key]))


def main():
    p = argparse.ArgumentParser(description="Builds synthetic French and English texts annotated with the spans "
                                            "of personal data, for the recognizers of dlib/privacy.")
    p.add_argument("--out-dir", default="pii-data", help="output directory")
    p.add_argument("--resources", default=None, help="directory of lexicons and templates replacing the built-in "
                                                       "ones (<lang>/<file>.txt)")
    p.add_argument("--languages", default="fr,en", help="comma separated languages among fr and en")
    p.add_argument("--fr-share", type=float, default=0.5, help="share of French examples when both languages "
                                                                "are generated")
    p.add_argument("--unseen-percent", type=int, default=15, help="percent of each lexicon held out for "
                                                                   "test_unseen")
    p.add_argument("--seed", type=int, default=1, help="random seed")
    p.add_argument("--context-rate", type=float, default=0.3, help="share of the examples written between "
                                                                   "plain sentences without personal data")
    p.add_argument("--prose-share", type=float, default=0.06, help="share of the examples made of such "
                                                                   "sentences alone")
    p.add_argument("--abroad-share", type=float, default=0.15, help="share of the places, organisations, "
                                                                     "telephone numbers and IBANs of other countries")
    p.add_argument("--sample", type=int, default=0, help="print this many examples with their spans and exit")
    for name, count, _, _, _, _ in SPLITS:
        p.add_argument("--" + name.replace("_", "-"), type=int, default=count,
                       help="examples in %s (default %d)" % (name, count))
    args = p.parse_args()

    languages = []
    for lang in (l.strip() for l in args.languages.split(",")):
        if lang and lang not in languages:
            languages.append(lang)
    if not languages:
        sys.exit("error: no language given")
    for lang in languages:
        if lang not in ("fr", "en"):
            sys.exit("error: unsupported language: %s (fr and en are supported)" % lang)
    if not 0 <= args.fr_share <= 1:
        sys.exit("error: --fr-share lies between 0 and 1")
    if not 0 <= args.context_rate <= 1 or not 0 <= args.prose_share <= 1 or not 0 <= args.abroad_share <= 1:
        sys.exit("error: --context-rate, --prose-share and --abroad-share lie between 0 and 1")
    if not 0 <= args.unseen_percent <= 100:
        sys.exit("error: --unseen-percent lies between 0 and 100")
    for name, _, _, _, _, _ in SPLITS:
        if getattr(args, name) < 0:
            sys.exit("error: --%s cannot be negative" % name.replace("_", "-"))
    res = Resources(args.resources, languages, args.unseen_percent)

    if args.sample:
        stats = {}
        lines = build_split(res, "sample", args.sample, "all", 0.25, 1.0, None, args.seed, languages,
                            args.fr_share, stats, args.context_rate, args.prose_share, None, args.abroad_share)
        for line in lines:
            text, _, spans = line.rpartition("\t")
            raw = unescape(text)
            print(raw)
            data = raw.encode("utf-8")
            for item in filter(None, spans.split(",")):
                b, e, label = item.split(":", 2)
                print("    %-15s %s" % (label, data[int(b):int(e)].decode("utf-8")))
            print()
        return

    os.makedirs(args.out_dir, exist_ok=True)
    stats = {}
    # The texts of train, which no evaluation split repeats.
    train_texts = set()
    for name, _, pool, procedural, noise, genres in SPLITS:
        count = getattr(args, name)
        if count <= 0:
            continue
        lines = build_split(res, name, count, pool, procedural, noise, genres, args.seed, languages,
                            args.fr_share, stats, args.context_rate, args.prose_share,
                            None if name == "train" else train_texts,
                            1.0 if name in ABROAD_SPLITS else args.abroad_share)
        if name == "train":
            train_texts = set(unescape(line.rpartition("\t")[0]) for line in lines)
        with open(os.path.join(args.out_dir, name + ".tsv"), "w", encoding="utf-8", newline="\n") as f:
            f.write("# dlib pii dataset: text<TAB>begin:end:LABEL,... (byte offsets)\n")
            for line in lines:
                f.write(line + "\n")
        print("%-15s %6d examples, %6d spans" % (name, len(lines), sum(stats[name]["spans"].values())))
    with open(os.path.join(args.out_dir, "labels.txt"), "w", encoding="utf-8") as f:
        for label in MODEL_LABELS + RULE_LABELS:
            f.write("%s\t%s\t%s\n" % (label, "learned" if label in MODEL_LABELS else "scanner",
                                      LABEL_DESCRIPTIONS[label]))
    with open(os.path.join(args.out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False, sort_keys=True)
    write_card(args.out_dir, args, res, stats)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as e:
        sys.exit("error: %s" % e)
