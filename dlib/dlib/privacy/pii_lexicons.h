// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_PII_LEXICONS_H_
#define DLIB_PII_LEXICONS_H_

/*
    Built-in word lists used by pii_detector and pseudonymizer, French and English.

    Every entry is a lookup key in the form produced by privacy_impl::fold_key(): lower
    case, without Latin accents, apostrophes unified. The lists hold common words of both
    languages; organisations extend them at run time with their own dictionaries through
    pii_detector::add_person_name(), add_term() and add_allowed_term().

    The first-name list is a recognition aid: a first name raises the score of a
    candidate span, which still needs a capitalized shape and is discarded when it
    collides with a common word, a role or a heading.
*/

#include <map>
#include <sstream>
#include <string>
#include <unordered_set>
#include "pii_types.h"

namespace dlib
{
    namespace privacy_impl
    {

    // ------------------------------------------------------------------------------------

        inline std::unordered_set<std::string> make_word_set(const char* words)
        {
            std::unordered_set<std::string> s;
            std::istringstream in(words);
            std::string w;
            while (in >> w)
                s.insert(w);
            return s;
        }

        inline std::unordered_set<std::string> make_phrase_set(const char* phrases)
        {
            // Entries separated by '|', each a folded word or phrase.
            std::unordered_set<std::string> s;
            std::string p;
            for (const char* c = phrases; ; ++c)
            {
                if (*c == '|' || *c == '\0')
                {
                    if (!p.empty()) s.insert(p);
                    p.clear();
                    if (*c == '\0') break;
                }
                else
                {
                    p.push_back(*c);
                }
            }
            return s;
        }

        inline const std::unordered_set<std::string>& first_names()
        {
            static const std::unordered_set<std::string> s = make_word_set(
                // French, given names of all generations
                "jean pierre michel andre philippe alain jacques bernard marcel rene louis daniel roger robert "
                "paul claude christian henri georges nicolas francois patrick gerard christophe joseph maurice "
                "raymond lucien marc frederic laurent julien david stephane pascal eric thierry olivier "
                "sebastien thomas alexandre antoine vincent guillaume mathieu matthieu maxime romain kevin "
                "jerome didier dominique yves serge gilles herve bruno hugo lucas theo leo gabriel raphael "
                "arthur jules adam nathan ethan tom noah mathis enzo baptiste clement quentin florian benjamin "
                "valentin adrien arnaud cedric fabien franck jeremy loic ludovic mickael michael remi sylvain "
                "xavier yann yannick damien fabrice emmanuel benoit bertrand denis etienne fernand gaston guy "
                "jacky joel lionel marius martin mathias noel patrice regis samuel simon tristan victor "
                "william aurelien axel bastien cyril dylan erwan erwann gael gauthier gregory gregoire jordan "
                "kylian malo mael marin nolan robin sacha timothee tanguy titouan yanis gwenael ronan goulven "
                "corentin brendan gurvan morgan killian riwal tugdual briac alan kilian evan elouan maxence "
                "matteo lenny louka liam timeo aaron nael isaac eden sohan ayden marceau augustin anatole "
                "hippolyte joachim leon leonard lazare edouard gustave albert alfred emile eugene felix "
                "firmin gilbert hubert jeremie josue leopold lucas marcellin max octave odilon oscar "
                "philibert prosper roland sylvestre thibault thibaut ulysse vivien wilfried yvon yvan "
                "jean-pierre jean-claude jean-marie jean-paul jean-luc jean-jacques jean-francois jean-michel "
                "jean-louis jean-marc jean-baptiste jean-christophe jean-yves jean-philippe "
                "marie jeanne francoise monique catherine nathalie isabelle jacqueline anne sylvie martine "
                "madeleine nicole suzanne helene christine marguerite denise louise christiane yvonne "
                "valerie sophie sandrine stephanie celine julie aurelie emilie laetitia camille lea manon "
                "chloe emma ines alice lina anna mila julia zoe lola lucie clara sarah laura pauline "
                "mathilde charlotte juliette margaux oceane elodie audrey melanie virginie caroline delphine "
                "veronique corinne patricia brigitte chantal danielle evelyne florence annie josiane "
                "michele colette odette simone germaine renee paulette ginette jeannine janine bernadette "
                "agnes beatrice elisabeth fabienne frederique geraldine karine laurence ludivine magali "
                "muriel nadine nadia noemie pascale sabrina sabine severine solene sonia vanessa gwenaelle "
                "maelys anais agathe adele albane alexia amandine angelique berenice capucine cecile "
                "coralie diane eleonore elsa estelle eva fanny gaelle heloise ingrid jessica jennifer "
                "josephine justine lise lisa louna lucile maelle margot marion maud morgane nina ophelie "
                "perrine romane salome sofia victoria yasmine fatima leila samira nora amina aicha myriam "
                "rachel rebecca solange therese yolande huguette arlette lucienne gisele raymonde marcelle "
                "paule mireille liliane eliane jocelyne francine annick armelle soizic nolwenn gwen enora "
                "maiwenn rozenn klervi anaelle oriane clemence apolline ambrine lou louane romy rosalie "
                "anne-marie anne-sophie marie-claude marie-christine marie-france marie-therese marie-helene "
                "marie-laure marie-jose marie-paule marie-pierre marie-noelle "
                // English
                "john james robert william richard joseph charles christopher matthew anthony mark donald "
                "steven andrew joshua kenneth brian george timothy ronald edward jason jeffrey ryan jacob "
                "gary nicholas jonathan stephen larry justin scott brandon gregory alexander frank raymond "
                "jack dennis jerry tyler aaron jose henry douglas zachary peter kyle jeremy walter keith "
                "terry austin sean gerald carl harold arthur lawrence jesse bryan billy bruce joe logan "
                "juan willie elijah wayne randy mason roy ralph bobby russell bradley philip mary jennifer "
                "linda elizabeth barbara susan karen nancy betty sandra margaret ashley kimberly emily "
                "donna michelle carol amanda melissa deborah dorothy sharon cynthia amy kathleen angela "
                "shirley brenda pamela samantha katherine debra carolyn janet maria heather olivia joyce "
                "ruth lauren kelly christina joan judith andrea hannah megan cheryl martha madison teresa "
                "gloria sara janice ann kathryn abigail sophia frances judy isabella amber marilyn beverly "
                "natalie theresa brittany doris kayla alexis lori emma oliver harry charlie jacob freddie "
                "archie alfie isla ava mia amelia poppy ella evie "
                // Spanish, Portuguese, Italian, German, Nordic, Slavic
                "carlos luis miguel javier manuel francisco pedro alejandro diego pablo sergio jorge "
                "fernando rafael raul alberto roberto ricardo andres mario marco luca giovanni giuseppe "
                "francesco alessandro matteo lorenzo stefano paolo hans klaus jurgen wolfgang helmut dieter "
                "stefan andreas uwe karl heinz werner gerhard gunter ursula monika petra gabriele heike "
                "birgit katarina anja lars sven erik nils bjorn olaf ivan dmitri sergei vladimir nikolai "
                "alexei olga tatiana natalia svetlana elena irina anastasia piotr tomasz krzysztof wojciech "
                "agnieszka katarzyna magdalena ana lucia carmen isabel pilar dolores teresa cristina marta "
                "paula valentina giulia francesca chiara federica silvia alessia joao tiago rui nuno "
                "goncalo beatriz mariana joana rita "
                // Arabic, African, Asian names frequent in Europe
                "mohamed mohammed muhammad ahmed omar youssef ibrahim hassan hussein khalid abdel abdelkader "
                "abdallah abdellah mustapha moustapha said samir nabil sofiane walid hicham karim rachid ali "
                "hamid farid nordine bilal amine mehdi ilyes rayan sami yacine yassine ayoub anis idriss "
                "karima malika khadija zineb salma hanane souad naima latifa rachida houria djamila "
                "mamadou moussa ousmane abdoulaye ibrahima fatou aminata mariama awa kofi kwame "
                "wei ming jing hui hiroshi takashi yuki kenji akira minh thanh huong "
                "alix ambroise anselme aristide armand aurele barnabe bertille blandine brice cassandre "
                "clelia daphne domitille elie eloise faustine gaspard gautier ghislain hector hortense "
                "ignace joris laurine leonie lilian lisandre lorene lubin maeva mahaut melissandre milo "
                "nathanael ninon ombeline pacome paulin philippine roxane sixtine swann tancrede tiphaine "
                "violaine wandrille aidan callum cerys ciara connor declan eilidh eoin ffion finley freya "
                "gemma hamish imogen keira kirsty lachlan maisie niamh orla owain phoebe rhys roisin rory "
                "saoirse seren sinead tadhg aarav aditya ananya arjun deepak divya harpreet imran kavita "
                "meera nikhil pooja priya rahul ravi sanjay sunita vikram yusuf zainab amara chidi emeka "
                "femi kemi ngozi tunde haruto hiroko seoyeon minjun aoife cillian oisin ruaridh angharad "
                "bethan dafydd gethin rhiannon");
            return s;
        }

        inline const std::unordered_set<std::string>& ambiguous_first_names()
        {
            /* Also common words, places or months: they count as first names only with
               further evidence. */
            static const std::unordered_set<std::string> s = make_word_set(
                "pierre rose marine france victoire constance prudence aime desire fidele honore "
                "patience esperance blanche ange angele celeste claire grace joy hope faith may june "
                "april august summer olive lily iris violette jade ambre max tom guy rene martin "
                "simon roy mark bill jean paris florence lou lea eden noel pascal dominique laurence "
                "marion morgan justice ruth victor william ravi philippine gemma");
            return s;
        }

        inline const std::unordered_set<std::string>& person_titles()
        {
            static const std::unordered_set<std::string> s = make_word_set(
                "m mr mrs ms mme mmes mlle mlles madame monsieur mademoiselle messieurs mesdames me "
                "maitre dr docteur pr professeur prof sir dame lord lady mgr monseigneur pere soeur "
                "colonel commandant capitaine lieutenant adjudant sergent juge president "
                "presidente ministre senateur depute maire chairman chairwoman chairperson speaker mayor governor "
                "senator congressman congresswoman "
                "abbe rabbin commissaire inspecteur brigadier amiral reverend captain sergeant inspector "
                "detective");
            return s;
        }

        inline const std::unordered_set<std::string>& person_particles()
        {
            static const std::unordered_set<std::string> s = make_word_set(
                "de du des d la le les l van von der den ter ten da das do dos di del della dei y e bin "
                "ben ibn al el abu bint jr sr ii iii iv");
            return s;
        }

        inline const std::unordered_set<std::string>& person_cues_before()
        {
            // Words that announce a person name right after them.
            static const std::unordered_set<std::string> s = make_word_set(
                "nomme nommee denomme denommee prenomme prenommee epouse epoux veuve veuf signe signee "
                "patient patiente salarie salariee employe employee candidat candidate temoin dear hello "
                "hi bonjour cher chere chers cheres stagiaire collaborateur collaboratrice "
                "beneficiaire assure assuree locataire proprietaire conjoint conjointe defunt defunte");
            return s;
        }

        inline const std::unordered_set<std::string>& person_cues_after()
        {
            // Words that typically follow a person name: "Jean Dupont, ne le".
            static const std::unordered_set<std::string> s = make_word_set(
                "ne nee nes nees epouse epoux veuve veuf demeurant domicilie domiciliee residant "
                "residante born aged age agee");
            return s;
        }

        inline const std::map<std::string, pii_type>& field_labels()
        {
            /* Labels that type the value written after them ("Nom : DUPONT") or in the next
               table cell. Multi-word labels are matched on their folded form. */
            static const std::map<std::string, pii_type> m = {
                {"nom", pii_type::person}, {"prenom", pii_type::person}, {"prenoms", pii_type::person},
                {"nom de naissance", pii_type::person}, {"nom d'usage", pii_type::person},
                {"nom marital", pii_type::person}, {"nom de jeune fille", pii_type::person},
                {"nom complet", pii_type::person}, {"nom et prenom", pii_type::person},
                {"nom prenom", pii_type::person}, {"prenom nom", pii_type::person},
                {"nom, prenom", pii_type::person}, {"name", pii_type::person}, {"full name", pii_type::person},
                {"first name", pii_type::person}, {"last name", pii_type::person}, {"surname", pii_type::person},
                {"given name", pii_type::person}, {"family name", pii_type::person},
                {"given names", pii_type::person}, {"first names", pii_type::person}, {"forename", pii_type::person},
                {"forenames", pii_type::person}, {"middle name", pii_type::person},
                {"nom de famille", pii_type::person}, {"nom usuel", pii_type::person},
                {"auteur", pii_type::person}, {"author", pii_type::person}, {"redacteur", pii_type::person},
                {"redige par", pii_type::person}, {"etabli par", pii_type::person}, {"valide par", pii_type::person},
                {"approuve par", pii_type::person}, {"verifie par", pii_type::person}, {"signe par", pii_type::person},
                {"prepared by", pii_type::person}, {"approved by", pii_type::person}, {"reviewed by", pii_type::person},
                {"written by", pii_type::person}, {"signed by", pii_type::person}, {"signataire", pii_type::person},
                {"titulaire", pii_type::person}, {"representant", pii_type::person},
                {"representant legal", pii_type::person}, {"interlocuteur", pii_type::person},
                {"destinataire", pii_type::person}, {"expediteur", pii_type::person}, {"de", pii_type::person},
                {"a", pii_type::person}, {"from", pii_type::person}, {"to", pii_type::person}, {"cc", pii_type::person},
                {"patient", pii_type::person}, {"patiente", pii_type::person}, {"medecin", pii_type::person},
                {"beneficiaire", pii_type::person}, {"assure", pii_type::person}, {"souscripteur", pii_type::person},
                {"conjoint", pii_type::person}, {"proprietaire", pii_type::person}, {"locataire", pii_type::person},
                {"salarie", pii_type::person}, {"employe", pii_type::person}, {"candidat", pii_type::person},
                {"contact", pii_type::person}, {"responsable", pii_type::person}, {"owner", pii_type::person},
                {"adresse", pii_type::address}, {"adresse postale", pii_type::address}, {"domicile", pii_type::address},
                {"adresse du domicile", pii_type::address}, {"address", pii_type::address},
                {"home address", pii_type::address}, {"postal address", pii_type::address},
                {"ville", pii_type::location}, {"commune", pii_type::location}, {"city", pii_type::location},
                {"lieu de naissance", pii_type::location}, {"place of birth", pii_type::location},
                {"birthplace", pii_type::location}, {"ne a", pii_type::location}, {"nee a", pii_type::location},
                {"domicilie a", pii_type::location}, {"domiciliee a", pii_type::location},
                {"demeurant a", pii_type::location}, {"residant a", pii_type::location},
                {"residant", pii_type::location}, {"habitant a", pii_type::location},
                {"date de naissance", pii_type::date}, {"ne le", pii_type::date}, {"nee le", pii_type::date},
                {"date of birth", pii_type::date}, {"dob", pii_type::date}, {"birth date", pii_type::date},
                {"telephone", pii_type::phone}, {"tel", pii_type::phone}, {"mobile", pii_type::phone},
                {"portable", pii_type::phone}, {"fax", pii_type::phone}, {"phone", pii_type::phone},
                {"cell", pii_type::phone}, {"numero de telephone", pii_type::phone}, {"phone number", pii_type::phone},
                {"email", pii_type::email}, {"e-mail", pii_type::email}, {"mail", pii_type::email},
                {"courriel", pii_type::email}, {"adresse mail", pii_type::email}, {"adresse email", pii_type::email},
                {"adresse e-mail", pii_type::email}, {"iban", pii_type::iban}, {"rib", pii_type::iban},
                {"numero de securite sociale", pii_type::nir}, {"securite sociale", pii_type::nir},
                {"nir", pii_type::nir}, {"numero ss", pii_type::nir}, {"ssn", pii_type::nir},
                {"passeport", pii_type::identifier}, {"passport", pii_type::identifier},
                {"numero de passeport", pii_type::identifier}, {"passport number", pii_type::identifier},
                {"cni", pii_type::identifier}, {"carte d'identite", pii_type::identifier},
                {"carte nationale d'identite", pii_type::identifier}, {"id card", pii_type::identifier},
                {"permis de conduire", pii_type::identifier}, {"numero de permis", pii_type::identifier},
                {"driving licence", pii_type::identifier}, {"driver's license", pii_type::identifier},
                {"matricule", pii_type::identifier}, {"numero matricule", pii_type::identifier},
                {"employee id", pii_type::identifier}, {"identifiant", pii_type::identifier},
                {"login", pii_type::identifier}, {"username", pii_type::identifier},
                {"nom d'utilisateur", pii_type::identifier}, {"user id", pii_type::identifier},
                {"numero client", pii_type::identifier}, {"reference client", pii_type::identifier},
                {"customer id", pii_type::identifier}, {"customer number", pii_type::identifier},
                {"numero de dossier", pii_type::identifier}, {"numero adherent", pii_type::identifier},
                {"numero fiscal", pii_type::identifier}, {"numero d'allocataire", pii_type::identifier},
                {"numero allocataire", pii_type::identifier}, {"numero de contrat", pii_type::identifier},
                {"policy number", pii_type::identifier}, {"account number", pii_type::identifier},
                {"numero de compte", pii_type::identifier}, {"employee number", pii_type::identifier},
                {"membership number", pii_type::identifier}, {"member number", pii_type::identifier},
                {"case number", pii_type::identifier}, {"reference number", pii_type::identifier},
                {"id number", pii_type::identifier}, {"licence number", pii_type::identifier},
                {"license number", pii_type::identifier}, {"tax number", pii_type::identifier},
                {"national insurance number", pii_type::identifier}, {"nhs number", pii_type::identifier},
                {"social security number", pii_type::identifier},
                {"siren", pii_type::siret}, {"siret", pii_type::siret}, {"rcs", pii_type::siret},
                {"immatriculation", pii_type::license_plate}, {"plaque", pii_type::license_plate},
                {"adresse ip", pii_type::ipv4}, {"ip address", pii_type::ipv4},
                {"raison sociale", pii_type::organization}, {"denomination", pii_type::organization},
                {"denomination sociale", pii_type::organization}, {"societe", pii_type::organization},
                {"entreprise", pii_type::organization}, {"employeur", pii_type::organization},
                {"employeur actuel", pii_type::organization}, {"nom de l'employeur", pii_type::organization},
                {"nom de l'entreprise", pii_type::organization}, {"nom de la societe", pii_type::organization},
                {"etablissement", pii_type::organization}, {"organisme", pii_type::organization},
                {"civilite", pii_type::title}, {"titre de civilite", pii_type::title}, {"titre", pii_type::title},
                {"title", pii_type::title}, {"salutation", pii_type::title},
                {"sexe", pii_type::gender}, {"genre", pii_type::gender}, {"sex", pii_type::gender},
                {"gender", pii_type::gender},
                {"nationalite", pii_type::nationality}, {"nationalites", pii_type::nationality},
                {"nationality", pii_type::nationality}, {"nationalities", pii_type::nationality},
                {"citoyennete", pii_type::nationality}, {"citizenship", pii_type::nationality},
                {"pays de nationalite", pii_type::nationality}, {"country of citizenship", pii_type::nationality},
                {"company", pii_type::organization}, {"company name", pii_type::organization},
                {"employer", pii_type::organization}, {"current employer", pii_type::organization},
                {"organisation", pii_type::organization}, {"organization", pii_type::organization},
                {"coordonnees gps", pii_type::geolocation}, {"gps", pii_type::geolocation},
                {"imei", pii_type::identifier}, {"vin", pii_type::identifier},
                {"numero de serie", pii_type::identifier}, {"serial number", pii_type::identifier},
                {"identifiant client", pii_type::identifier}, {"matricule fiscal", pii_type::identifier},
                {"id", pii_type::identifier}, {"identifier", pii_type::identifier},
                {"titre de sejour", pii_type::identifier}, {"numero de titre de sejour", pii_type::identifier},
                {"carte de sejour", pii_type::identifier}, {"residence permit", pii_type::identifier},
                {"mrn", pii_type::identifier}, {"medical record number", pii_type::identifier},
                {"hospital number", pii_type::identifier}, {"patient id", pii_type::identifier},
                {"ipp", pii_type::identifier}, {"ni number", pii_type::identifier},
                {"nino", pii_type::identifier}, {"vous etes", pii_type::gender},
                {"code postal", pii_type::location}, {"postal code", pii_type::location},
                {"zip code", pii_type::location}, {"postcode", pii_type::location},
                {"nationalite d'origine", pii_type::nationality},
                {"nom du patient", pii_type::person}, {"nom de la patiente", pii_type::person},
                {"nom de l'assure", pii_type::person}, {"nom de l'assuree", pii_type::person},
                {"nom du titulaire", pii_type::person}, {"nom du conjoint", pii_type::person},
                {"nom de l'epoux", pii_type::person}, {"nom de l'epouse", pii_type::person},
                {"nom du pere", pii_type::person}, {"nom de la mere", pii_type::person},
                {"prenom usuel", pii_type::person}, {"nom et prenoms", pii_type::person},
                {"personne a contacter", pii_type::person}, {"personne a prevenir", pii_type::person},
                {"personne de confiance", pii_type::person}, {"medecin traitant", pii_type::person},
                {"medecin referent", pii_type::person}, {"a l'attention de", pii_type::person},
                {"suivi par", pii_type::person}, {"dossier suivi par", pii_type::person},
                {"affaire suivie par", pii_type::person}, {"tuteur", pii_type::person},
                {"tutrice", pii_type::person}, {"epoux", pii_type::person}, {"epouse", pii_type::person},
                {"temoin", pii_type::person}, {"stagiaire", pii_type::person}, {"eleve", pii_type::person},
                {"etudiant", pii_type::person}, {"etudiante", pii_type::person}, {"enfant", pii_type::person},
                {"contact person", pii_type::person}, {"contact name", pii_type::person},
                {"emergency contact", pii_type::person}, {"next of kin", pii_type::person},
                {"spouse", pii_type::person}, {"spouse's name", pii_type::person},
                {"father's name", pii_type::person}, {"mother's name", pii_type::person},
                {"maiden name", pii_type::person}, {"account holder", pii_type::person},
                {"account name", pii_type::person}, {"cardholder", pii_type::person},
                {"policyholder", pii_type::person}, {"policy holder", pii_type::person},
                {"employee", pii_type::person}, {"employee name", pii_type::person},
                {"applicant name", pii_type::person}, {"candidate", pii_type::person},
                {"candidate name", pii_type::person}, {"student", pii_type::person},
                {"student name", pii_type::person}, {"pupil", pii_type::person},
                {"patient name", pii_type::person}, {"patient's name", pii_type::person}, {"gp", pii_type::person},
                {"referred by", pii_type::person}, {"requested by", pii_type::person},
                {"submitted by", pii_type::person}, {"completed by", pii_type::person}, {"attn", pii_type::person},
                {"witness", pii_type::person},
                {"adresse de facturation", pii_type::address}, {"adresse de livraison", pii_type::address},
                {"adresse de correspondance", pii_type::address}, {"adresse personnelle", pii_type::address},
                {"adresse actuelle", pii_type::address}, {"nouvelle adresse", pii_type::address},
                {"ancienne adresse", pii_type::address}, {"billing address", pii_type::address},
                {"shipping address", pii_type::address}, {"mailing address", pii_type::address},
                {"delivery address", pii_type::address}, {"residential address", pii_type::address},
                {"correspondence address", pii_type::address}, {"current address", pii_type::address},
                {"previous address", pii_type::address}, {"permanent address", pii_type::address},
                {"street address", pii_type::address}, {"adresse electronique", pii_type::email},
                {"adresse de messagerie", pii_type::email}, {"adresse courriel", pii_type::email},
                {"email address", pii_type::email}, {"e-mail address", pii_type::email},
                {"work email", pii_type::email}, {"personal email", pii_type::email},
                {"courriel professionnel", pii_type::email}, {"courriel personnel", pii_type::email},
                {"email professionnel", pii_type::email}, {"email personnel", pii_type::email},
                {"telephone portable", pii_type::phone}, {"telephone mobile", pii_type::phone},
                {"telephone fixe", pii_type::phone}, {"telephone professionnel", pii_type::phone},
                {"telephone personnel", pii_type::phone}, {"telephone domicile", pii_type::phone},
                {"tel portable", pii_type::phone}, {"tel fixe", pii_type::phone}, {"tel mobile", pii_type::phone},
                {"tel domicile", pii_type::phone}, {"home phone", pii_type::phone},
                {"work phone", pii_type::phone}, {"mobile phone", pii_type::phone},
                {"mobile number", pii_type::phone}, {"telephone number", pii_type::phone},
                {"contact number", pii_type::phone}, {"cell phone", pii_type::phone},
                {"cell number", pii_type::phone}, {"daytime phone", pii_type::phone},
                {"landline", pii_type::phone}, {"lieu de residence", pii_type::location},
                {"ville de residence", pii_type::location}, {"ville de naissance", pii_type::location},
                {"commune de naissance", pii_type::location}, {"commune de residence", pii_type::location},
                {"place of residence", pii_type::location}, {"town of birth", pii_type::location},
                {"city of birth", pii_type::location}, {"town", pii_type::location},
                {"numero de badge", pii_type::identifier}, {"badge number", pii_type::identifier},
                {"numero d'employe", pii_type::identifier}, {"numero employe", pii_type::identifier},
                {"staff number", pii_type::identifier}, {"staff id", pii_type::identifier},
                {"student id", pii_type::identifier}, {"student number", pii_type::identifier},
                {"numero etudiant", pii_type::identifier}, {"numero d'etudiant", pii_type::identifier},
                {"numero ine", pii_type::identifier}, {"numero de sinistre", pii_type::identifier},
                {"claim number", pii_type::identifier}, {"numero de police", pii_type::identifier},
                {"numero d'adherent", pii_type::identifier}, {"patient number", pii_type::identifier},
                {"nom de l'organisme", pii_type::organization}, {"nom de l'etablissement", pii_type::organization},
                {"etablissement scolaire", pii_type::organization}, {"employer name", pii_type::organization},
                {"organisation name", pii_type::organization}, {"organization name", pii_type::organization},
                {"business name", pii_type::organization}, {"trading name", pii_type::organization},
                {"nom commercial", pii_type::organization}, {"enseigne", pii_type::organization},
                {"assureur", pii_type::organization}, {"insurer", pii_type::organization},
                {"mutuelle", pii_type::organization}, {"banque", pii_type::organization},
                {"bank", pii_type::organization}, {"bank name", pii_type::organization},
                {"ecole", pii_type::organization}, {"school", pii_type::organization},
                {"universite", pii_type::organization}, {"university", pii_type::organization},
                {"employeur precedent", pii_type::organization}, {"previous employer", pii_type::organization},
                {"former employer", pii_type::organization}
            };
            return m;
        }

        inline const std::unordered_set<std::string>& street_types()
        {
            static const std::unordered_set<std::string> s = make_word_set(
                "rue avenue av bd boulevard bld bvd allee impasse imp chemin che place pl quai route rte "
                "square sq cours passage voie sentier lieu-dit lotissement residence hameau cite villa "
                "faubourg fbg esplanade parvis rond-point ruelle venelle montee traverse promenade mail "
                "street st road rd lane ln drive dr blvd court ct way terrace highway hwy ave");
            return s;
        }

        inline const std::unordered_set<std::string>& english_street_types()
        {
            // Types written after the street name in English: "Station Road", "Oak Dr".
            static const std::unordered_set<std::string> s = make_word_set(
                "street st road rd lane ln drive dr blvd court ct way terrace highway hwy ave avenue close "
                "crescent grove gardens parade row walk mews hill park circle plaza parkway trail square place "
                "boulevard");
            return s;
        }

        inline const std::unordered_set<std::string>& address_kept_words()
        {
            static const std::unordered_set<std::string> s = make_word_set(
                "rue avenue av bd boulevard bld bvd allee impasse imp chemin che place pl quai route rte "
                "square sq cours passage voie sentier lieu dit lieu-dit lotissement residence hameau cite "
                "villa faubourg fbg esplanade parvis rond point ruelle venelle montee traverse promenade "
                "mail street st road rd lane ln drive dr blvd court ct way terrace highway hwy ave "
                "de du des d la le les l au aux et sur sous en saint sainte bis ter quater bp cs cedex "
                "bat batiment appartement apt appt etage escalier esc porte n no");
            return s;
        }

        inline const std::unordered_set<std::string>& organization_forms()
        {
            static const std::unordered_set<std::string> s = make_word_set(
                "sa sas sasu sarl eurl sci snc scs gie scop scp selarl selas selafa sem spl ei eirl "
                "inc incorporated ltd limited llc llp plc corp corporation co gmbh ag kg ohg ug bv nv spa "
                "srl sl pty ltee asbl sprl scrl bvba vzw kgaa oyj ltda pte pllc sca");
            return s;
        }

        inline const std::unordered_set<std::string>& organization_heads()
        {
            static const std::unordered_set<std::string> s = make_word_set(
                "societe ste cabinet groupe group association fondation foundation banque bank clinique "
                "hopital centre center institut institute universite university ecole school lycee "
                "college laboratoire laboratoires mairie ministere ministry agence agency entreprise "
                "entreprises etablissements ets federation syndicat cooperative mutuelle caisse "
                "tribunal etude pharmacie ehpad club compagnie imprimerie boulangerie garage atelier conservatoire "
                "polyclinique");
            return s;
        }

        inline const std::unordered_set<std::string>& organization_kept_words()
        {
            static const std::unordered_set<std::string> s = make_word_set(
                "sa sas sasu sarl eurl sci snc scs gie scop scp selarl selas selafa sem spl ei eirl inc "
                "incorporated ltd limited llc llp plc corp corporation co gmbh ag kg ohg ug bv nv spa srl "
                "sl pty societe ste cabinet groupe group association fondation foundation banque bank "
                "clinique hopital centre center institut institute universite university ecole school "
                "lycee college laboratoire laboratoires mairie ministere ministry agence agency entreprise "
                "entreprises etablissements ets federation syndicat cooperative mutuelle caisse office "
                "tribunal etude pharmacie ehpad club de du des d la le les l et and of the for pour");
            return s;
        }

        inline const std::unordered_set<std::string>& location_kept_words()
        {
            static const std::unordered_set<std::string> s = make_word_set(
                "saint sainte st ste sur sous en les le la l de du des d lez cedex aux au");
            return s;
        }

        inline const std::unordered_set<std::string>& url_kept_words()
        {
            static const std::unordered_set<std::string> s = make_word_set(
                "http https ftp mailto www com org net fr eu de uk be ch io co html htm php asp aspx jsp "
                "index in pub profile profiles user users people u p linkedin facebook twitter x "
                "instagram github gitlab youtube tiktok viadeo medium blog");
            return s;
        }

        inline const std::unordered_set<std::string>& location_triggers()
        {
            // "ne a", "domicilie a": the place follows the preposition.
            static const std::unordered_set<std::string> s = make_word_set(
                "ne nee domicilie domiciliee demeurant residant residante habitant habitante originaire fait "
                "installe installee lives living born based residing resides");
            return s;
        }

        inline const std::unordered_set<std::string>& common_words()
        {
            /* Capitalized at the start of sentences, in titles and headings: never a name on
               their own, and a name span stops when it reaches one. */
            static const std::unordered_set<std::string> s = make_word_set(
                "le la les un une des du de d l et ou mais donc or ni car ce cet cette ces mon ma mes "
                "ton ta tes son sa ses notre nos votre vos leur leurs je tu il elle on nous vous ils "
                "elles qui que quoi dont ou quand comment pourquoi si ne pas plus moins tres bien aussi "
                "alors ainsi apres avant avec sans sous sur dans par pour entre vers chez contre depuis "
                "pendant selon malgre lors des ici la voici voila ceci cela ca tout tous toute toutes "
                "chaque aucun aucune plusieurs certains certaines autre autres meme memes tel telle quel "
                "quelle quels quelles enfin puis ensuite cependant toutefois neanmoins pourtant egalement "
                "notamment concernant objet cordialement salutations merci bonjour bonsoir madame monsieur "
                "article articles contrat contrats annexe annexes chapitre section page pages total date "
                "reference ref numero nom prenom adresse ville pays telephone email courriel societe "
                "entreprise client clients fournisseur projet projets rapport compte rendu reunion ordre "
                "jour mois annee semaine direction service services departement equipe responsable "
                "directeur directrice president presidente conseil comite assemblee generale general "
                "france europe paris lyon marseille rennes nantes bordeaux lille toulouse nice brest "
                "introduction conclusion resume synthese sommaire table figure tableau note notes "
                "version document documents fichier dossier partie titre sujet message courrier lettre "
                "facture devis commande livraison paiement montant prix tarif offre proposition "
                "conditions condition generales particulieres signature fait signe lu approuve "
                "the a an and or but if then of in on at to for from by with without about as is are "
                "was were be been this that these those it its he she they we you i my your his her "
                "will would shall should can could must might has have had do does did over under into upon "
                "holiday holidays "
                "their our who what which when where why how all any some each every no not yes dear "
                "hello hi thanks thank regards best sincerely yours agreement contract appendix exhibit "
                "name address city country phone company customer report meeting director manager "
                "president board committee department team chapter summary overview introduction "
                "monday tuesday wednesday thursday friday saturday sunday lundi mardi mercredi jeudi "
                "vendredi samedi dimanche january february march april may june july august september "
                "october november december janvier fevrier mars avril mai juin juillet aout septembre "
                "octobre novembre decembre nord sud est ouest north south east west rue avenue "
                "boulevard place chemin route impasse allee quai "
                "monde logiciel logiciels demande demandes reunions conge conges formulaire formulaires "
                "procedure procedures planning budget budgets agenda calendrier programme objectif objectifs "
                "resultat resultats priorite urgent urgence important attention avis remarque remarques question "
                "questions reponse reponses suivi statut identifiant lien piece jointe jointes brouillon "
                "confidentiel "
                "world form forms request requests meetings leave schedule order orders invoice invoices quote "
                "quotes delivery calendar program goal goals result results priority answer answers status "
                "update number password link attachment attachments file files folder project projects draft "
                "week weeks month months year years "
                "confidential");
            return s;
        }

        inline const std::unordered_set<std::string>& function_words()
        {
            /* Articles, pronouns, possessives, prepositions and conjunctions: a span made of
               them alone names nobody. */
            static const std::unordered_set<std::string> s = make_word_set(
                "le la les l un une des du de d au aux a et ou mais donc or ni car ce c cet cette ces "
                "mon ma mes ton ta tes son sa ses notre nos votre vos leur leurs je j tu il elle on nous "
                "vous ils elles me m te t se s moi toi lui eux y en qui que qu quoi dont ou ceci cela ca "
                "the an and nor but if of in on at to for from by with as is are was were be been this "
                "that these those it its he she they we you i my your his her their our him us them who "
                "whom what which when where why how not no yes so than then there here");
            return s;
        }

        inline const std::unordered_set<std::string>& file_extensions()
        {
            // The extensions of the files a document names: "rapport_final.docx".
            static const std::unordered_set<std::string> s = make_word_set(
                "doc docx docm dot dotx xls xlsx xlsm xlsb csv tsv ppt pptx pptm odt ods odp rtf txt md pdf "
                "xml json yaml yml html htm css js ts py cpp c h hpp java cs go rs rb php sh bat ps1 sql log "
                "ini cfg conf zip tar gz tgz 7z rar png jpg jpeg gif bmp tif tiff svg webp mp3 mp4 wav avi "
                "mov mkv eml msg ics vcf exe dll so bin iso dat bak tmp");
            return s;
        }

        inline const std::unordered_set<std::string>& greeting_words()
        {
            /* Greetings, closings and forms of address that open or close a message, often
               capitalized before a comma at the start of a text ("Bonsoir, ...", "Hello, ..."):
               never personal data on their own, whatever a learned recognizer reads there. */
            static const std::unordered_set<std::string> s = make_word_set(
                "bonjour bonsoir salut coucou allo merci cordialement amicalement bien bises bisous "
                "salutations bienvenue bravo felicitations pardon desole desolee cher chere chers cheres "
                "madame monsieur mademoiselle mesdames messieurs hello hi hey greetings dear thanks "
                "thank cheers regards best sincerely welcome congratulations sorry sir madam");
            return s;
        }

        inline const std::unordered_set<std::string>& civility_words()
        {
            /* Forms of address that tell the sex of the person they precede, replaced as
               TITLE. Professional titles (Dr, Me, Pr) stay readable. */
            static const std::unordered_set<std::string> s = make_word_set(
                "m mr mrs ms mx miss mme mmes mlle mlles madame monsieur mademoiselle messieurs mesdames "
                "sir madam");
            return s;
        }

        inline const std::unordered_set<std::string>& gender_values()
        {
            // Values of a sex or gender field, folded with spaces and hyphens removed.
            static const std::unordered_set<std::string> s = make_word_set(
                "m f h x masculin feminin homme femme male female man woman garcon fille boy girl "
                "nonbinaire nonbinary neutre neutral autre other");
            return s;
        }

        inline const std::unordered_set<std::string>& nationality_words()
        {
            /* Nationality adjectives, French in their masculine and feminine forms, and
               English demonyms. */
            static const std::unordered_set<std::string> s = make_phrase_set(
                "afghan|afghane|albanais|albanaise|algerien|algerienne|allemand|allemande|americain|americaine|"
                "andorran|andorrane|angolais|angolaise|argentin|argentine|armenien|armenienne|australien|"
                "australienne|autrichien|autrichienne|azerbaidjanais|azerbaidjanaise|bangladais|bangladaise|belge|"
                "beninois|beninoise|bielorusse|birman|birmane|bolivien|bolivienne|bosnien|bosnienne|bresilien|"
                "bresilienne|britannique|bulgare|burkinabe|burundais|burundaise|cambodgien|cambodgienne|"
                "camerounais|camerounaise|canadien|canadienne|cap-verdien|cap-verdienne|centrafricain|"
                "centrafricaine|chilien|chilienne|chinois|chinoise|chypriote|colombien|colombienne|comorien|"
                "comorienne|congolais|congolaise|coreen|coreenne|costaricien|costaricienne|croate|cubain|cubaine|"
                "danois|danoise|djiboutien|djiboutienne|dominicain|dominicaine|egyptien|egyptienne|emirien|"
                "emirienne|equatorien|equatorienne|erythreen|erythreenne|espagnol|espagnole|estonien|estonienne|"
                "ethiopien|ethiopienne|finlandais|finlandaise|francais|francaise|gabonais|gabonaise|gambien|"
                "gambienne|georgien|georgienne|ghaneen|ghaneenne|grec|grecque|guatemalteque|guineen|guineenne|"
                "haitien|haitienne|hondurien|hondurienne|hongrois|hongroise|indien|indienne|indonesien|"
                "indonesienne|irakien|irakienne|iranien|iranienne|irlandais|irlandaise|islandais|islandaise|"
                "israelien|israelienne|italien|italienne|ivoirien|ivoirienne|jamaicain|jamaicaine|japonais|"
                "japonaise|jordanien|jordanienne|kazakh|kazakhe|kenyan|kenyane|kosovar|kosovare|koweitien|"
                "koweitienne|laotien|laotienne|letton|lettone|libanais|libanaise|liberien|liberienne|libyen|"
                "libyenne|lituanien|lituanienne|luxembourgeois|luxembourgeoise|macedonien|macedonienne|malgache|"
                "malaisien|malaisienne|malien|malienne|maltais|maltaise|marocain|marocaine|mauricien|mauricienne|"
                "mauritanien|mauritanienne|mexicain|mexicaine|moldave|monegasque|mongol|mongole|montenegrin|"
                "montenegrine|mozambicain|mozambicaine|namibien|namibienne|neerlandais|neerlandaise|nepalais|"
                "nepalaise|nicaraguayen|nicaraguayenne|nigerien|nigerienne|nigerian|nigeriane|norvegien|"
                "norvegienne|neo-zelandais|neo-zelandaise|omanais|omanaise|ougandais|ougandaise|ouzbek|ouzbeke|"
                "pakistanais|pakistanaise|palestinien|palestinienne|panameen|panameenne|paraguayen|paraguayenne|"
                "peruvien|peruvienne|philippin|philippine|polonais|polonaise|portugais|portugaise|qatarien|"
                "qatarienne|roumain|roumaine|russe|rwandais|rwandaise|salvadorien|salvadorienne|saoudien|"
                "saoudienne|senegalais|senegalaise|serbe|singapourien|singapourienne|slovaque|slovene|somalien|"
                "somalienne|soudanais|soudanaise|sri-lankais|sri-lankaise|suedois|suedoise|suisse|syrien|"
                "syrienne|tadjik|tadjike|tanzanien|tanzanienne|tchadien|tchadienne|tcheque|thailandais|"
                "thailandaise|togolais|togolaise|tunisien|tunisienne|turc|turque|ukrainien|ukrainienne|"
                "uruguayen|uruguayenne|venezuelien|venezuelienne|vietnamien|vietnamienne|yemenite|zambien|"
                "zambienne|zimbabween|zimbabweenne|"
                "albanian|algerian|american|angolan|argentinian|armenian|australian|austrian|azerbaijani|"
                "bangladeshi|belarusian|belgian|beninese|bolivian|bosnian|brazilian|british|bulgarian|burmese|"
                "burundian|cambodian|cameroonian|canadian|cape verdean|chadian|chilean|chinese|colombian|"
                "comorian|congolese|costa rican|croatian|cuban|cypriot|czech|danish|dominican|dutch|ecuadorian|"
                "egyptian|emirati|english|eritrean|estonian|ethiopian|filipino|finnish|french|gabonese|gambian|"
                "georgian|german|ghanaian|greek|guatemalan|guinean|haitian|honduran|hungarian|icelandic|indian|"
                "indonesian|iranian|iraqi|irish|israeli|italian|ivorian|jamaican|japanese|jordanian|kenyan|"
                "korean|south korean|north korean|kuwaiti|latvian|lebanese|liberian|libyan|lithuanian|"
                "luxembourgish|macedonian|malagasy|malaysian|malian|maltese|mauritanian|mauritian|mexican|"
                "moldovan|mongolian|moroccan|mozambican|namibian|nepalese|new zealander|nicaraguan|nigerian|"
                "norwegian|northern irish|omani|pakistani|palestinian|panamanian|paraguayan|peruvian|polish|"
                "portuguese|qatari|romanian|russian|rwandan|salvadoran|saudi|scottish|senegalese|serbian|"
                "sierra leonean|singaporean|slovak|slovenian|somali|south african|spanish|sri lankan|sudanese|"
                "swedish|swiss|syrian|taiwanese|tajik|tanzanian|thai|togolese|tunisian|turkish|ugandan|"
                "ukrainian|uruguayan|uzbek|venezuelan|vietnamese|welsh|yemeni|zambian|zimbabwean|"
                "bahreini|bahreinie|barbadien|barbadienne|bhoutanais|bhoutanaise|botswanais|botswanaise|"
                "bruneien|bruneienne|fidjien|fidjienne|grenadien|grenadienne|guyanien|guyanienne|kirghize|"
                "kirghiz|liechtensteinois|liechtensteinoise|malawite|maldivien|maldivienne|nauruan|nauruane|"
                "papouan|papouane|samoan|samoane|seychellois|seychelloise|sierra-leonais|sierra-leonaise|"
                "surinamais|surinamaise|timorais|timoraise|tongien|tongienne|trinidadien|trinidadienne|"
                "turkmene|vanuatais|vanuataise|bahraini|barbadian|bhutanese|motswana|bruneian|fijian|"
                "grenadian|guyanese|kazakhstani|kyrgyz|laotian|liechtensteiner|malawian|maldivian|"
                "micronesian|sammarinese|surinamese|swazi|timorese|tongan|"
                "trinidadian|turkmen|tuvaluan|ni-vanuatu|kosovan|azeri");
            return s;
        }

        inline const std::unordered_set<std::string>& country_names()
        {
            // Countries, French and English, accepted as the value of a nationality field.
            static const std::unordered_set<std::string> s = make_phrase_set(
                "afghanistan|albanie|algerie|allemagne|andorre|angola|argentine|armenie|australie|autriche|"
                "azerbaidjan|bangladesh|belgique|benin|bielorussie|birmanie|bolivie|bosnie-herzegovine|bresil|"
                "bulgarie|burkina faso|burundi|cambodge|cameroun|canada|cap-vert|centrafrique|chili|chine|"
                "chypre|colombie|comores|congo|coree du sud|coree du nord|costa rica|cote d'ivoire|croatie|cuba|"
                "danemark|djibouti|egypte|emirats arabes unis|equateur|erythree|espagne|estonie|etats-unis|"
                "ethiopie|finlande|france|gabon|gambie|georgie|ghana|grece|guatemala|guinee|haiti|honduras|"
                "hongrie|inde|indonesie|irak|iran|irlande|islande|israel|italie|jamaique|japon|jordanie|"
                "kazakhstan|kenya|kosovo|koweit|laos|lettonie|liban|liberia|libye|lituanie|luxembourg|"
                "macedoine du nord|madagascar|malaisie|mali|malte|maroc|maurice|mauritanie|mexique|moldavie|"
                "monaco|mongolie|montenegro|mozambique|namibie|nepal|nicaragua|niger|nigeria|norvege|"
                "nouvelle-zelande|oman|ouganda|ouzbekistan|pakistan|palestine|panama|paraguay|pays-bas|perou|"
                "philippines|pologne|portugal|qatar|roumanie|royaume-uni|russie|rwanda|salvador|arabie saoudite|"
                "senegal|serbie|singapour|slovaquie|slovenie|somalie|soudan|sri lanka|suede|suisse|syrie|"
                "tadjikistan|tanzanie|tchad|tchequie|republique tcheque|thailande|togo|tunisie|turquie|"
                "ukraine|uruguay|venezuela|vietnam|yemen|zambie|zimbabwe|"
                "albania|algeria|germany|argentina|armenia|australia|austria|belgium|brazil|bulgaria|cameroon|"
                "china|croatia|cyprus|czech republic|czechia|denmark|egypt|england|estonia|finland|greece|"
                "hungary|india|indonesia|iraq|ireland|iceland|italy|ivory coast|japan|jordan|korea|"
                "south korea|latvia|lebanon|lithuania|malaysia|malta|mauritius|mexico|morocco|netherlands|"
                "new zealand|norway|poland|romania|russia|saudi arabia|scotland|senegal|serbia|slovakia|"
                "slovenia|south africa|spain|sweden|switzerland|syria|tunisia|turkey|united kingdom|uk|"
                "great britain|united states|united states of america|usa|us|wales|northern ireland|"
                "ukraine|vietnam");
            return s;
        }

        inline const std::unordered_set<std::string>& public_body_acronyms()
        {
            /* Acronyms of public bodies, matched when written in capitals: national and
               local administrations, social security bodies, regulators and courts, French
               and British or American. */
            static const std::unordered_set<std::string> s = make_word_set(
                "caf cpam cnam cnaf cnav carsat urssaf msa dgfip ddfip drfip insee cnil anssi ars ants ofii "
                "ofpra cnda crous cnous ansm ademe cnes cnrs inserm inrae ifremer ign onf dgccrf dgsi dgse dga "
                "sgdsn ugap amf acpr arcep arcom hceres onisep cned unedic apec ircantec cnracl aphp dreal dreets "
                "ddets ddpp dgac dgddi anah dila bnf inpi oniam fiva cnous ofb anct dinum "
                "nhs hmrc dvla dvsa dwp hmcts hmpps fcdo dhsc dfe fca cma ofsted ofcom ofgem ofwat cqc mhra ukvi "
                "irs ssa fbi cia nsa dhs uscis tsa fema fda doj hhs ftc fcc dmv nih nasa usda faa nhtsa "
                "ons nca fsa ico moj hmpo dbs ofqual ukhsa ehrc cbp atf osha anses ineris irsn cnsa agefiph "
                "fiphfp pjj spip dgpn dggn dgt dares dgefp");
            return s;
        }

        inline const std::unordered_set<std::string>& public_body_names()
        {
            // Public bodies known by a name of their own, folded.
            static const std::unordered_set<std::string> s = make_phrase_set(
                "pole emploi|france travail|banque de france|ap-hp|assistance publique-hopitaux de paris|"
                "caisse des depots|caisse des depots et consignations|"
                "conseil d'etat|cour de cassation|cour des comptes|conseil constitutionnel|assemblee nationale|"
                "senat|education nationale|gendarmerie nationale|police nationale|police municipale|"
                "tresor public|haute autorite de sante|defenseur des droits|assurance maladie|assurance retraite|"
                "agirc-arrco|france competences|bpifrance|caisse nationale d'assurance vieillesse|"
                "caisse nationale des allocations familiales|agence nationale de la securite des systemes "
                "d'information|"
                "home office|companies house|hm treasury|hm revenue and customs|hm revenue & customs|"
                "metropolitan police|crown prosecution service|land registry|hm land registry|"
                "environment agency|internal revenue service|social security administration|"
                "department of homeland security|department of justice|state department|national health service|"
                "border force|national crime agency|serious fraud office|food standards agency|"
                "student loans company|jobcentre plus|job centre plus|uk visas and immigration|passport office|"
                "hm passport office|citizens advice|european commission|european parliament|"
                "direction generale des finances publiques|office francais de l'immigration et de l'integration|"
                "office national des forets|office francais de la biodiversite|commission europeenne|"
                "parlement europeen|conseil de l'europe|"
                "cour de justice de l'union europeenne|cour europeenne des droits de l'homme|"
                "banque centrale europeenne|autorite des marches financiers|"
                "autorite de controle prudentiel et de resolution|"
                "commission nationale de l'informatique et des libertes|agence nationale de l'habitat|"
                "caisse d'allocations familiales|caisse primaire d'assurance maladie|"
                "mutualite sociale agricole|office national des anciens combattants|"
                "agence nationale des titres securises|"
                "institut national de la statistique et des etudes economiques|"
                "conseil superieur de la magistrature|department for work and pensions|"
                "driver and vehicle licensing agency|ministry of justice|ministry of defence|"
                "department for education|department of health and social care|cabinet office|"
                "financial conduct authority|information commissioner's office|care quality commission|"
                "office for national statistics|health and safety executive|disclosure and barring service|"
                "valuation office agency|federal bureau of investigation|department of veterans affairs|"
                "census bureau|department of labor|federal trade commission|environmental protection agency|"
                "canada revenue agency|service canada|australian taxation office|revenue commissioners|"
                "health service executive|an garda siochana|european central bank|"
                "european court of human rights|court of justice of the european union");
            return s;
        }

        inline const std::unordered_set<std::string>& public_body_heads()
        {
            /* Head nouns of public bodies named by what they do and where: "Tribunal
               judiciaire de Rennes", "Ministere de l'Interieur", "Department for Education". */
            static const std::unordered_set<std::string> s = make_word_set(
                "ministere ministry prefecture sous-prefecture mairie tribunal cour court conseil direction "
                "directorate rectorat academie agence office caisse department departement commissariat "
                "gendarmerie inspection autorite authority commission ambassade consulat embassy consulate "
                "parquet greffe ville commune brigade");
            return s;
        }

        inline const std::unordered_set<std::string>& public_body_words()
        {
            /* Lower case words that a public body name may hold between its head and its
               complement, or at its end. */
            static const std::unordered_set<std::string> s = make_word_set(
                "judiciaire administratif administrative correctionnel correctionnelle commerce prud'hommes "
                "prudhommes police appel assises cassation instance grande departemental departementale regional "
                "regionale municipal municipale general generale interdepartemental interdepartementale primaire "
                "nationale national nationaux superieur superieure interministeriel interministerielle "
                "territorial territoriale territoriaux publique publiques public publics sociale sociales social "
                "economique economiques financiere financieres finances sante education armees interieur justice "
                "culture travail emploi famille familles allocations familiales assurance maladie vieillesse "
                "retraite retraites solidarite solidarites agriculture ecologie transports logement sports "
                "affaires etrangeres europe recherche enseignement jeunesse securite civile douanes impots "
                "statistique protection populations enfance handicap");
            return s;
        }

        inline const std::unordered_set<std::string>& acronym_words()
        {
            static const std::unordered_set<std::string> s = make_word_set(
                "rgpd cnil gdpr pdf url api rh it ue eu usa onu oms tva ttc ht cdi cdd ceo cto cfo coo "
                "dsi rssi dpo pme tpe eti sncf edf ratp cpam caf urssaf insee anssi iso iec nist ia ai "
                "sla kpi roi crm erp sap aws gcp cdr pki vpn ssl tls http https ok nb ps pj ref fyi asap");
            return s;
        }

    // ------------------------------------------------------------------------------------

    }
}

#endif // DLIB_PII_LEXICONS_H_
