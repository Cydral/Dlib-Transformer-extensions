// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_PRIVACY_FICTIVE_LEXICONS_H_
#define DLIB_PRIVACY_FICTIVE_LEXICONS_H_

/*
    The lexicons of the fictive values (fictive_values.h), in French, English, German, Spanish
    and Italian: given names by sex, the parts of family names, the words of street names and
    the cities of each country with their postal codes and their size classes.

    The given names are common names of each language; each list holds names for each of the
    initials that the fictive initials use (fictive_initials()). The family names are made of a
    stem and an ending of their language, so that most of them are rare and a fictive name
    seldom equals a name of the corpus. The cities are real cities: a fictive address names a
    real city, never a real street of it.

    Accented letters are written '%' followed by the letter and its mark, so that the file
    stays in ASCII: ' acute, ` grave, ^ circumflex, : diaeresis, , cedilla, ~ tilde, $ for the
    sharp s ("%E'mile" reads "Emile" with its acute accent).
*/

#include <cstddef>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>
#include "unicode_text.h"

namespace dlib
{
    namespace privacy_impl
    {

    // ------------------------------------------------------------------------------------

        // The text of a lexicon, its accented letters written in ASCII, in UTF-8.
        inline std::string fictive_lexicon_text(const char* s)
        {
            std::string out;
            for (const char* c = s; *c; ++c)
            {
                if (*c != '%' || !c[1] || !c[2])
                {
                    out.push_back(*c);
                    continue;
                }
                const char base = c[1], mark = c[2];
                char32_t cp = 0;
                if (mark == '$') cp = 0xDF;
                else
                {
                    static const char* const letters = "AEIOUaeiou";
                    static const char32_t acute[] = { 0xC1, 0xC9, 0xCD, 0xD3, 0xDA, 0xE1, 0xE9, 0xED, 0xF3, 0xFA };
                    static const char32_t grave[] = { 0xC0, 0xC8, 0xCC, 0xD2, 0xD9, 0xE0, 0xE8, 0xEC, 0xF2, 0xF9 };
                    static const char32_t circ[] = { 0xC2, 0xCA, 0xCE, 0xD4, 0xDB, 0xE2, 0xEA, 0xEE, 0xF4, 0xFB };
                    static const char32_t diaer[] = { 0xC4, 0xCB, 0xCF, 0xD6, 0xDC, 0xE4, 0xEB, 0xEF, 0xF6, 0xFC };
                    for (int k = 0; k < 10; ++k)
                    {
                        if (letters[k] != base) continue;
                        if (mark == '\'') cp = acute[k];
                        else if (mark == '`') cp = grave[k];
                        else if (mark == '^') cp = circ[k];
                        else if (mark == ':') cp = diaer[k];
                    }
                    if (mark == ',' && base == 'c') cp = 0xE7;
                    if (mark == ',' && base == 'C') cp = 0xC7;
                    if (mark == '~' && base == 'n') cp = 0xF1;
                    if (mark == '~' && base == 'N') cp = 0xD1;
                }
                if (cp == 0)
                {
                    out.push_back(*c);
                    continue;
                }
                utf8_append(out, cp);
                c += 2;
            }
            return out;
        }

        inline std::vector<std::string> fictive_lexicon_words(const char* s)
        {
            std::vector<std::string> out;
            const std::string text = fictive_lexicon_text(s);
            std::string w;
            for (char c : text)
            {
                if (c == ' ')
                {
                    if (!w.empty()) out.push_back(w);
                    w.clear();
                }
                else w.push_back(c);
            }
            if (!w.empty()) out.push_back(w);
            return out;
        }

    // ------------------------------------------------------------------------------------

        // The initials of the fictive given names: each list below has names for each of them.
        inline const std::string& fictive_initials()
        {
            static const std::string s = "ABCDEFGLMNPRSTV";
            return s;
        }

        struct fictive_language
        {
            std::string code;                       // "fr", "en", "de", "es", "it"
            std::string country;                    // the country of its places and numbers
            std::vector<std::string> male, female;
            std::vector<std::string> stems, endings;    // the parts of the family names
            /*
                The words of street names, by the word before them: "plural" after a plural
                article (des, de los, dei), "masculine" after a masculine one (du, del), "feminine"
                after a feminine one (de la, della), "vowel" after an elided one (de l'), "plain"
                without article, and the stems of the German compounds ("Linden" of
                "Lindenstrasse").
            */
            std::vector<std::string> plural, plural_feminine, masculine, feminine, vowel, plain;
        };

        inline const std::vector<fictive_language>& fictive_languages()
        {
            static const std::vector<fictive_language> all = []() {
                std::vector<fictive_language> v(5);
                fictive_language& fr = v[0];
                fr.code = "fr";
                fr.country = "fr";
                fr.male = fictive_lexicon_words(
                    "Alain Andr%e' Antoine Arnaud Beno%i^t Bernard Bruno Bastien Christophe Claude Cl%e'ment "
                    "C%e'dric Denis Didier Damien Daniel %E'mile %E'tienne %E'ric Edgar Fran%c,ois Fr%e'd%e'ric "
                    "Fabien F%e'lix Gilles Guillaume G%e'rard Gaston Lucien Laurent Louis Lionel Marcel Mathieu "
                    "Maxime Michel Nicolas No%e:l Norbert Nathan Pascal Patrice Philippe Paul Raymond R%e'mi "
                    "Romain Roland Serge S%e'bastien Simon Sylvain Thierry Th%e'o Thibault Tristan Vincent Victor "
                    "Valentin Vivien");
                fr.female = fictive_lexicon_words(
                    "Agn%e`s Alice Am%e'lie Anne Brigitte Bernadette B%e'atrice Blanche Claire Catherine C%e'cile "
                    "Charlotte Danielle Delphine Denise Diane %E'lise %E'milie Estelle Eva Florence Fran%c,oise "
                    "Fabienne Fanny Genevi%e`ve Gis%e`le Ga%e:lle Germaine Laure Lucie Louise L%e'a Marie Martine "
                    "Mathilde Monique Nathalie Nicole No%e'mie Nadine Pauline Patricia Pascale Perrine Ren%e'e Rose "
                    "Rapha%e:lle Romane Sophie Sylvie Sol%e`ne Suzanne Th%e'r%e`se Tiphaine Tatiana Val%e'rie "
                    "V%e'ronique Virginie Victoire");
                fr.stems = fictive_lexicon_words(
                    "Bar Bel Ber Bes Bon Bou Bru Cam Car Cha Cor Dal Dan Dav Del Des Dor Dum Fav Fer Gal Gar Gil "
                    "Gir Gra Gui Lab Lac Lam Lan Lar Lau Lec Ler Mal Mar Mas Mor Pel Per Pic Pin Pot Rab Ren Rig "
                    "Riv Rol Rou Sau Tes Thi Val Ver Vil");
                fr.endings = fictive_lexicon_words("and ard aud el elle ier in on ot et eau ault ais ois");
                fr.plural = fictive_lexicon_words(
                    "Tilleuls Lilas Acacias Peupliers Ch%e^nes %E'rables Saules Bleuets Coquelicots M%e'sanges "
                    "Hirondelles Marronniers Platanes Ormes Rosiers Glycines Jonquilles Violettes Gen%e^ts "
                    "Bruy%e`res Noisetiers Cerisiers Pommiers Vergers Pr%e's Champs Jardins Vignes Moulins Pins "
                    "C%e`dres Lauriers Mimosas Cam%e'lias Hortensias Pivoines Primev%e`res");
                fr.plural_feminine = fr.plural;
                fr.masculine = fictive_lexicon_words(
                    "Moulin Lavoir Stade Ch%a^teau Parc Port Pont Bois Verger Calvaire March%e' Prieur%e' Manoir "
                    "Puits Pressoir Clos Vallon Coteau Bocage Ruisseau Four Gu%e' Bourg Menhir");
                fr.feminine = fictive_lexicon_words(
                    "Gare Fontaine Source Mairie Poste Forge Chapelle Croix Vall%e'e Prairie Rivi%e`re Garenne "
                    "Lande Colline Halle Tannerie Corderie Libert%e' R%e'publique Paix R%e'sistance Lib%e'ration "
                    "Victoire Madeleine");
                fr.vowel = fictive_lexicon_words(
                    "%E'glise %E'tang Abbaye Orme Aqueduc Arsenal H%o^pital Octroi Ermitage Industrie Avenir Europe "
                    "Oc%e'an Abreuvoir Orangerie");

                fictive_language& en = v[1];
                en.code = "en";
                en.country = "gb";
                en.male = fictive_lexicon_words(
                    "Andrew Alan Arthur Adam Brian Benjamin Bruce Bradley Charles Christopher Colin Craig David "
                    "Daniel Dennis Douglas Edward Eric Ethan Edwin Frank Frederick Francis Felix George Gary Gordon "
                    "Graham Lawrence Liam Leonard Luke Michael Matthew Martin Mark Nicholas Nathan Neil Nigel Peter "
                    "Patrick Paul Philip Robert Richard Ronald Ryan Stephen Samuel Simon Stuart Thomas Timothy "
                    "Trevor Terence Victor Vincent Vernon Vaughan");
                en.female = fictive_lexicon_words(
                    "Alice Amanda Anna Abigail Barbara Bethany Beatrice Brenda Catherine Charlotte Claire Carol "
                    "Diana Deborah Dorothy Daisy Emma Elizabeth Eleanor Emily Fiona Frances Felicity Freya Grace "
                    "Gillian Georgia Gemma Laura Lucy Linda Louise Mary Margaret Megan Melissa Nancy Natalie Nicola "
                    "Nora Patricia Pamela Penelope Phoebe Rachel Rebecca Rosemary Ruth Sarah Susan Sophie Stephanie "
                    "Teresa Tracy Tessa Tamsin Victoria Vanessa Valerie Vivian");
                en.stems = fictive_lexicon_words(
                    "Ash Brad Brook Cal Card Clay Craw Dal Dar Dun Ell Fair Fen Gar Glad Hal Har Hart Hol Kend Kin "
                    "Lang Mar Mill Mor New Nor Pen Rad Red Rid Rus Sal Shel Stan Sut Thorn Wal War Wes Whit Wil Win "
                    "Wood");
                en.endings = fictive_lexicon_words("ford ley ton well wood worth son by ham field more er ing ock");
                en.plain = fictive_lexicon_words(
                    "Oak Elm Ash Birch Cedar Maple Willow Beech Hazel Chestnut Holly Rowan Larch Pine Poplar Laurel "
                    "Church Mill Station Park Bridge School Chapel Market King Queen Victoria Albert George Manor "
                    "Grange Orchard Meadow Brook Hill Green Field Spring Abbey Castle Harbour North South West East "
                    "High New Old Long");

                fictive_language& de = v[2];
                de.code = "de";
                de.country = "de";
                de.male = fictive_lexicon_words(
                    "Andreas Alexander Armin Axel Bernd Benedikt Bj%o:rn Burkhard Christian Carsten Christoph "
                    "Clemens Dieter Dirk Daniel Dominik Erik Elias Ernst Egon Frank Florian Friedrich Felix Georg "
                    "Gerhard G%u:nter Gregor Lukas Lars Ludwig Leon Markus Matthias Martin Moritz Niklas Norbert "
                    "Nils Noah Peter Paul Philipp Patrick Ralf Rainer Robert R%u:diger Stefan Sebastian Simon Sven "
                    "Thomas Tobias Torsten Tim Volker Valentin Viktor Vincent");
                de.female = fictive_lexicon_words(
                    "Anna Andrea Angelika Anke Birgit Barbara Brigitte Bettina Claudia Christina Carina Corinna "
                    "Doris Daniela Dagmar Dorothea Elke Eva Emma Elisabeth Franziska Frieda Friederike Fiona "
                    "Gabriele Gisela Greta Gudrun Laura Lena Lisa Luise Monika Maria Martina Melanie Nina Nicole "
                    "Natalie Nadine Petra Paula Pia Patricia Renate Ruth Rebecca Regina Sabine Susanne Sophie Silke "
                    "Tanja Theresa Tina Tamara Vanessa Verena Vera Viktoria");
                de.stems = fictive_lexicon_words(
                    "Alt Bach Berg Brand Br%u:ck Eich Feld Fried Graf Gr%u:n Hart Heid Hof Holz Kirch Klein Lang "
                    "Lind Mohr Neu Rein Ross Sch%o:n Stein Wald Wei%s$ Wein Wies Wolf Zimm");
                de.endings = fictive_lexicon_words("mann er ler hof berg bach hardt mayer inger huber rich el");
                de.plain = fictive_lexicon_words(
                    "Linden Birken Eichen Ahorn Rosen Garten M%u:hlen Schul Kirch Bahnhof Wald Feld Wiesen Berg Tal "
                    "Sonnen Kastanien Buchen Tannen Fichten Blumen Brunnen Markt Schloss Ulmen Erlen Holunder "
                    "Flieder Lerchen Amsel");

                fictive_language& es = v[3];
                es.code = "es";
                es.country = "es";
                es.male = fictive_lexicon_words(
                    "Alejandro Antonio %A'lvaro Andr%e's Benito Bernardo Bruno Borja Carlos C%e'sar Cristian "
                    "Claudio David Daniel Diego Domingo Enrique Eduardo Emilio Esteban Fernando Francisco F%e'lix "
                    "Fabi%a'n Gonzalo Guillermo Gabriel Gerardo Luis Lorenzo Lucas Leandro Manuel Miguel Mario "
                    "Marcos Nicol%a's N%e'stor Norberto No%e' Pablo Pedro Patricio Pascual Rafael Ram%o'n Roberto "
                    "Rub%e'n Sergio Santiago Samuel Salvador Tom%a's Teodoro Tob%i'as Timoteo V%i'ctor Vicente "
                    "Valent%i'n Virgilio");
                es.female = fictive_lexicon_words(
                    "Ana Alicia Andrea Adela Beatriz Blanca Bel%e'n B%a'rbara Carmen Cristina Clara Celia Dolores "
                    "Diana Daniela D%e'bora Elena Eva Esther Estela Francisca Fernanda F%a'tima Florencia Gloria "
                    "Gabriela Guadalupe Gema Luc%i'a Laura Lorena Leticia Mar%i'a Marta Mercedes M%o'nica Nuria "
                    "Natalia Noelia Nerea Paula Pilar Patricia Paloma Rosa Raquel Roc%i'o Rebeca Sof%i'a Silvia "
                    "Sara Susana Teresa Tamara Tatiana Trinidad Ver%o'nica Victoria Vanesa Virginia");
                es.stems = fictive_lexicon_words(
                    "Al Bel Cal Car Cas Cor Dom Es Fer Gal Gar Gil Her Ib Jim Lor Mar Mol Mor Nav Ort Pas Per Quin "
                    "Ram Sal San Ser Tor Val Vaz Vel Vid");
                es.endings = fictive_lexicon_words("ero ado ales anco illa ino ez era uera ana edo ueta");
                es.plural = fictive_lexicon_words(
                    "Olivos Pinos %A'lamos Almendros Naranjos Casta%n~os Robles Laureles Jazmines Claveles Rosales");
                es.plural_feminine = fictive_lexicon_words(
                    "Violetas Rosas Flores Huertas Fuentes Eras Vi%n~as Acacias Palmeras Moreras");
                es.masculine = fictive_lexicon_words(
                    "Molino Sol Pilar Carmen Puerto Prado Castillo Mercado Pozo Olivar Romero Monte");
                es.feminine = fictive_lexicon_words(
                    "Fuente Iglesia Estaci%o'n Paz Cruz Libertad Constituci%o'n Ermita Ribera Huerta Alameda Torre");
                es.plain = fictive_lexicon_words("Mayor Real Nueva Ancha Larga Alta Baja Hornos Herreros");

                fictive_language& it = v[4];
                it.code = "it";
                it.country = "it";
                it.male = fictive_lexicon_words(
                    "Alessandro Andrea Antonio Alberto Bruno Battista Biagio Bernardo Carlo Claudio Cesare "
                    "Cristiano Davide Daniele Dario Domenico Enrico Emanuele Edoardo Elia Francesco Fabio Federico "
                    "Filippo Giuseppe Giovanni Giorgio Gabriele Luca Lorenzo Leonardo Luigi Marco Matteo Massimo "
                    "Mario Nicola Nicol%o` Nino Natale Paolo Pietro Pasquale Piero Roberto Riccardo Raffaele Renato "
                    "Stefano Simone Salvatore Sergio Tommaso Tiziano Teodoro Tullio Vincenzo Valerio Vittorio "
                    "Valentino");
                it.female = fictive_lexicon_words(
                    "Alessandra Anna Alice Angela Beatrice Barbara Bianca Benedetta Chiara Caterina Claudia "
                    "Cristina Daniela Debora Donatella Diana Elena Elisa Emma Eleonora Francesca Federica Fabiola "
                    "Filomena Giulia Giorgia Gabriella Gaia Laura Lucia Ludovica Lisa Maria Martina Marta Michela "
                    "Nicoletta Noemi Nadia Natalia Paola Patrizia Pamela Piera Roberta Rita Rosa Raffaella Sara "
                    "Silvia Sofia Serena Teresa Tiziana Tamara Tecla Valentina Vittoria Veronica Viola");
                it.stems = fictive_lexicon_words(
                    "Bar Bel Ber Bon Cal Cap Car Cas Cor Fer Gal Gar Gior Lan Lom Mar Mor Pel Pir Ric Ros Sal San "
                    "Ser Tor Val Ver Vit");
                it.endings = fictive_lexicon_words("elli ini etti oni ucci ari esi one ato ino otti uzzi");
                it.plural = fictive_lexicon_words("Tigli Pini Platani Castagni Gelsi Mulini Giardini Orti Cipressi");
                it.plural_feminine = fictive_lexicon_words(
                    "Rose Viole Acacie Querce Mimose Magnolie Fontane Vigne Camelie Ortensie");
                it.masculine = fictive_lexicon_words(
                    "Mulino Castello Porto Ponte Mercato Borgo Duomo Pozzo Colle Bosco Sole Lago");
                it.feminine = fictive_lexicon_words(
                    "Fontana Chiesa Stazione Pace Libert%a` Repubblica Vittoria Vigna Torre Rocca Pieve Quercia");
                it.plain = fictive_lexicon_words(
                    "Roma Milano Venezia Torino Firenze Napoli Trieste Trento Udine Verona");
                return v;
            }();
            return all;
        }

        inline const fictive_language* fictive_language_of(const std::string& code)
        {
            for (const fictive_language& l : fictive_languages())
                if (l.code == code) return &l;
            return nullptr;
        }

        /*
            Common given names of each language and of each sex, folded, that tell the language and
            the sex of a name of the text; the names of both lists of a language are given to men
            and to women alike ("Camille", "Dominique"). The lists of the fictive names add theirs.
        */
        inline const char* const* fictive_given_reading(std::size_t language)
        {
            static const char* const fr[2] = {
                "jean pierre michel andre philippe alain jacques bernard marcel rene louis daniel roger robert paul "
                "claude christian henri georges nicolas francois patrick gerard christophe joseph maurice raymond "
                "lucien marc frederic laurent julien david stephane pascal eric thierry olivier sebastien thomas "
                "alexandre antoine vincent guillaume mathieu matthieu maxime romain kevin jerome didier yves serge "
                "gilles herve bruno hugo lucas theo leo gabriel raphael arthur jules adam nathan ethan tom noah "
                "mathis enzo baptiste clement quentin florian benjamin valentin adrien arnaud cedric fabien franck "
                "jeremy loic ludovic mickael michael remi sylvain xavier yann yannick damien fabrice emmanuel benoit "
                "bertrand denis etienne fernand gaston guy jacky joel lionel marius martin mathias noel patrice "
                "regis samuel simon tristan victor william aurelien axel bastien cyril dylan erwan gael gauthier "
                "gregory gregoire jordan kylian malo mael marin nolan robin tanguy titouan yanis gwenael ronan "
                "goulven corentin brendan gurvan killian tugdual briac alan kilian evan elouan maxence matteo lenny "
                "louka liam timeo aaron nael isaac augustin anatole hippolyte joachim leon leonard edouard gustave "
                "albert alfred emile eugene felix firmin gilbert hubert jeremie leopold marcellin max octave oscar "
                "roland thibault thibaut vivien wilfried yvon yvan norbert edgar jean-pierre jean-claude jean-marie "
                "jean-paul jean-luc jean-jacques jean-francois jean-michel jean-louis jean-marc jean-baptiste "
                "jean-christophe jean-yves jean-philippe dominique camille sacha morgan",
                "marie jeanne francoise monique catherine nathalie isabelle jacqueline anne sylvie martine madeleine "
                "nicole suzanne helene christine marguerite denise louise christiane yvonne valerie sophie sandrine "
                "stephanie celine julie aurelie emilie laetitia lea manon chloe emma ines alice lina anna mila julia "
                "zoe lola lucie clara sarah laura pauline mathilde charlotte juliette margaux oceane elodie audrey "
                "melanie virginie caroline delphine veronique corinne patricia brigitte chantal danielle evelyne "
                "florence annie josiane michele colette odette simone germaine renee paulette ginette jeannine "
                "janine bernadette agnes beatrice elisabeth fabienne frederique geraldine karine laurence ludivine "
                "magali muriel nadine nadia noemie pascale sabrina sabine severine solene sonia vanessa gwenaelle "
                "maelys anais agathe adele albane alexia amandine angelique berenice capucine cecile coralie diane "
                "eleonore elsa estelle eva fanny gaelle heloise ingrid jessica jennifer josephine justine lise lisa "
                "louna lucile maelle margot marion maud morgane nina ophelie perrine romane salome sofia victoria "
                "yasmine fatima leila samira nora amina aicha myriam rachel rebecca solange therese yolande huguette "
                "arlette lucienne gisele raymonde marcelle paule mireille liliane eliane jocelyne francine annick "
                "armelle soizic nolwenn enora maiwenn rozenn klervi anaelle oriane clemence apolline lou louane romy "
                "rosalie claire genevieve blanche tiphaine raphaelle victoire anne-marie anne-sophie marie-claude "
                "marie-christine marie-france marie-therese marie-helene marie-laure marie-jose marie-paule "
                "marie-pierre marie-noelle dominique camille sacha morgan claude" };
            static const char* const en[2] = {
                "john james robert william richard joseph charles christopher matthew anthony mark donald steven "
                "andrew joshua kenneth brian george timothy ronald edward jason jeffrey ryan jacob gary nicholas "
                "jonathan stephen larry justin scott brandon gregory alexander frank raymond jack dennis jerry tyler "
                "aaron henry douglas zachary peter kyle walter keith terry austin sean gerald carl harold arthur "
                "lawrence jesse bryan billy bruce joe logan willie elijah wayne randy mason roy ralph bobby russell "
                "bradley philip oliver harry charlie freddie archie alfie thomas daniel david michael paul simon "
                "stuart trevor terence victor vincent vernon colin craig neil nigel graham gordon liam luke ethan "
                "edwin eric francis frederick felix martin adam alan benjamin samuel nathan patrick ian kevin",
                "mary jennifer linda elizabeth barbara susan karen nancy betty sandra margaret ashley kimberly emily "
                "donna michelle carol amanda melissa deborah dorothy sharon cynthia amy kathleen angela shirley "
                "brenda pamela samantha katherine debra carolyn janet maria heather olivia joyce ruth lauren kelly "
                "christina joan judith andrea hannah megan cheryl martha madison teresa gloria sara janice ann "
                "kathryn abigail sophia frances judy isabella amber marilyn beverly natalie theresa brittany doris "
                "kayla alexis lori emma isla ava mia amelia poppy ella evie alice anna catherine charlotte claire "
                "diana daisy eleanor fiona felicity freya grace gillian georgia gemma laura lucy louise nicola nora "
                "patricia penelope phoebe rachel rebecca rosemary sarah sophie stephanie tracy tessa tamsin victoria "
                "vanessa valerie vivian helen jane jessica rose" };
            static const char* const de[2] = {
                "hans klaus jurgen wolfgang helmut dieter stefan andreas uwe karl heinz werner gerhard gunter peter "
                "michael thomas frank markus martin matthias christian alexander sebastian tobias florian daniel jan "
                "lukas leon finn jonas paul felix maximilian elias noah ben luis moritz niklas philipp tim julian "
                "david simon jakob johannes fabian dominik kevin marcel sven lars nils bjorn olaf ralf rainer bernd "
                "jens jorg torsten carsten dirk volker manfred horst gunther friedrich georg hermann otto ernst kurt "
                "walter heinrich wilhelm ludwig rudolf robert erich fritz armin axel benedikt burkhard christoph "
                "clemens egon erik gregor norbert patrick rudiger viktor valentin vincent oskar emil anton",
                "ursula monika petra gabriele heike birgit anja sabine susanne claudia andrea angelika anke barbara "
                "brigitte bettina christina carina corinna doris daniela dagmar dorothea elke eva emma elisabeth "
                "franziska frieda friederike fiona gisela greta gudrun laura lena lisa luise maria martina melanie "
                "nina nicole natalie nadine paula pia patricia renate ruth rebecca regina sophie silke tanja theresa "
                "tina tamara vanessa verena vera viktoria karin ingrid helga gertrud hildegard erika ilse inge "
                "christa marianne waltraud elfriede annette kerstin katrin julia sarah hannah leonie lea marie "
                "johanna clara charlotte mia emilia amelie sofia lara jana stefanie sandra" };
            static const char* const es[2] = {
                "carlos luis miguel javier manuel francisco pedro alejandro diego pablo sergio jorge fernando rafael "
                "raul alberto roberto ricardo andres mario antonio jose juan david daniel enrique eduardo emilio "
                "esteban felix fabian gonzalo guillermo gabriel gerardo lorenzo lucas leandro marcos nicolas nestor "
                "norberto noe patricio pascual ramon ruben santiago samuel salvador tomas teodoro tobias timoteo "
                "victor vicente valentin virgilio alvaro benito bernardo bruno borja cesar cristian claudio domingo "
                "hugo ivan adrian oscar jaime ignacio joaquin agustin julio alfonso arturo",
                "ana lucia carmen isabel pilar dolores teresa cristina marta paula maria laura sara elena eva alicia "
                "andrea adela beatriz blanca belen barbara clara celia diana daniela debora esther estela francisca "
                "fernanda fatima florencia gloria gabriela guadalupe gema lorena leticia mercedes monica nuria "
                "natalia noelia nerea patricia paloma rosa raquel rocio rebeca sofia silvia susana tamara tatiana "
                "trinidad veronica victoria vanesa virginia irene julia ines rosario concepcion josefa antonia "
                "manuela montserrat encarnacion inmaculada consuelo marina alba claudia carla" };
            static const char* const it[2] = {
                "marco luca giovanni giuseppe francesco alessandro matteo lorenzo stefano paolo andrea antonio "
                "alberto bruno battista biagio bernardo carlo claudio cesare cristiano davide daniele dario domenico "
                "enrico emanuele edoardo elia fabio federico filippo giorgio gabriele leonardo luigi massimo mario "
                "nicola nicolo nino natale pietro pasquale piero roberto riccardo raffaele renato simone salvatore "
                "sergio tommaso tiziano teodoro tullio vincenzo valerio vittorio valentino angelo michele gianni "
                "franco sandro fabrizio maurizio marcello giacomo guido umberto ugo",
                "valentina giulia francesca chiara federica silvia alessia alessandra anna alice angela beatrice "
                "barbara bianca benedetta caterina claudia cristina daniela debora donatella diana elena elisa emma "
                "eleonora fabiola filomena giorgia gabriella gaia laura lucia ludovica lisa maria martina marta "
                "michela nicoletta noemi nadia natalia paola patrizia pamela piera roberta rita rosa raffaella sara "
                "sofia serena teresa tiziana tamara tecla vittoria veronica viola giovanna antonella simona monica "
                "manuela stefania carla rossella" };
            static const char* const* const all[] = { fr, en, de, es, it };
            return all[language < 5 ? language : 0];
        }

        /*
            The languages and the sexes of the given names, by folded name: bit 2*k for a male name
            of language k, bit 2*k+1 for a female one.
        */
        inline const std::unordered_map<std::string, unsigned>& fictive_given_names()
        {
            static const std::unordered_map<std::string, unsigned> m = []() {
                std::unordered_map<std::string, unsigned> r;
                const std::vector<fictive_language>& all = fictive_languages();
                for (std::size_t k = 0; k < all.size(); ++k)
                {
                    for (const std::string& n : all[k].male) r[fold_key(n)] |= 1u << (2 * k);
                    for (const std::string& n : all[k].female) r[fold_key(n)] |= 1u << (2 * k + 1);
                    for (int sex = 0; sex < 2; ++sex)
                        for (const std::string& n : fictive_lexicon_words(fictive_given_reading(k)[sex]))
                            r[n] |= 1u << (2 * k + static_cast<std::size_t>(sex));
                }
                return r;
            }();
            return m;
        }

    // ------------------------------------------------------------------------------------

        struct fictive_city
        {
            std::string name;
            std::string postal;     // the postal code of its centre: the outward code in Great Britain
            std::string state;      // the two letters of its state in the United States
            std::string country;    // "fr", "gb", "us", "de", "es", "it"
            char size = 'S';        // 'L' 200,000 inhabitants and more, 'M' 50,000 and more, 'S' fewer
        };

        inline const std::vector<fictive_city>& fictive_cities()
        {
            static const std::vector<fictive_city> all = []() {
                // country, then "size|name|postal|state" separated by ';'
                static const char* const tables[][2] = {
                    { "fr",
                      "L|Paris|75001;L|Marseille|13001;L|Lyon|69001;L|Toulouse|31000;L|Nice|06000;L|Nantes|44000;"
                      "L|Montpellier|34000;L|Strasbourg|67000;L|Bordeaux|33000;L|Lille|59000;L|Rennes|35000;"
                      "M|Toulon|83000;M|Reims|51100;M|Saint-%E'tienne|42000;M|Le Havre|76600;M|Grenoble|38000;"
                      "M|Dijon|21000;M|Angers|49000;M|N%i^mes|30000;M|Villeurbanne|69100;M|Clermont-Ferrand|63000;"
                      "M|Le Mans|72000;M|Aix-en-Provence|13100;M|Brest|29200;M|Tours|37000;M|Amiens|80000;"
                      "M|Limoges|87000;M|Annecy|74000;M|Perpignan|66000;M|Metz|57000;M|Besan%c,on|25000;"
                      "M|Orl%e'ans|45000;M|Rouen|76000;M|Caen|14000;M|Mulhouse|68100;M|Nancy|54000;M|Avignon|84000;"
                      "M|Poitiers|86000;M|Pau|64000;M|La Rochelle|17000;M|Calais|62100;M|Colmar|68000;"
                      "M|Quimper|29000;M|Valence|26000;M|Troyes|10000;M|Chamb%e'ry|73000;M|Niort|79000;"
                      "M|Lorient|56100;M|Vannes|56000;M|Bayonne|64100;S|Saint-Malo|35400;S|Laval|53000;"
                      "S|Vitr%e'|35500;S|Foug%e`res|35300;S|Dinan|22100;S|Lannion|22300;S|Morlaix|29600;"
                      "S|Concarneau|29900;S|Douarnenez|29100;S|Saumur|49400;S|Cognac|16100;S|Honfleur|14600;"
                      "S|Dole|39100;S|Millau|12100;S|Riom|63200;S|Autun|71400;S|Lun%e'ville|54300;"
                      "S|Sarlat-la-Can%e'da|24200;S|Pontivy|56300;S|Redon|35600;S|Mayenne|53100;S|Lisieux|14100;"
                      "S|Bayeux|14400;S|Granville|50400;S|Coutances|50200;S|Amboise|37400;S|Chinon|37500;"
                      "S|Montargis|45200;S|Beaune|21200;S|Annonay|07100;S|Manosque|04100;S|Carpentras|84200;"
                      "S|Uz%e`s|30700;S|Mende|48000;S|Figeac|46100;S|Auch|32000;S|Dax|40100;S|Lourdes|65100;"
                      "S|Foix|09000;S|Limoux|11300" },
                    { "gb",
                      "L|London|SW1A;L|Birmingham|B1;L|Manchester|M1;L|Leeds|LS1;L|Glasgow|G1;L|Liverpool|L1;"
                      "L|Sheffield|S1;L|Bristol|BS1;L|Edinburgh|EH1;L|Leicester|LE1;L|Cardiff|CF10;"
                      "L|Nottingham|NG1;L|Newcastle upon Tyne|NE1;L|Brighton|BN1;M|Oxford|OX1;M|Cambridge|CB2;"
                      "M|York|YO1;M|Exeter|EX1;M|Norwich|NR2;M|Bath|BA1;M|Reading|RG1;M|Swansea|SA1;"
                      "M|Aberdeen|AB10;M|Dundee|DD1;M|Plymouth|PL1;M|Derby|DE1;M|Coventry|CV1;M|Southampton|SO14;"
                      "M|Portsmouth|PO1;M|Ipswich|IP1;M|Gloucester|GL1;M|Chester|CH1;M|Lincoln|LN1;M|Carlisle|CA3;"
                      "S|Inverness|IV1;S|Ludlow|SY8;S|Whitby|YO21;S|Kendal|LA9;S|Penrith|CA11;S|Totnes|TQ9;"
                      "S|Stamford|PE9;S|Hexham|NE46;S|Thirsk|YO7;S|Keswick|CA12;S|Dorchester|DT1;S|Truro|TR1;"
                      "S|Ely|CB7;S|Wells|BA5;S|Ripon|HG4;S|Alnwick|NE66;S|Bakewell|DE45;S|Tewkesbury|GL20;"
                      "S|Lewes|BN7;S|Rye|TN31;S|Sherborne|DT9;S|Oban|PA34;S|Pitlochry|PH16;S|Brecon|LD3;"
                      "S|Llandudno|LL30" },
                    { "us",
                      "L|New York|10001|NY;L|Los Angeles|90012|CA;L|Chicago|60601|IL;L|Houston|77002|TX;"
                      "L|Phoenix|85003|AZ;L|Philadelphia|19107|PA;L|San Antonio|78205|TX;L|San Diego|92101|CA;"
                      "L|Dallas|75201|TX;L|Austin|78701|TX;L|Seattle|98101|WA;L|Denver|80202|CO;L|Boston|02108|MA;"
                      "M|Ann Arbor|48104|MI;M|Boulder|80302|CO;M|Savannah|31401|GA;M|Eugene|97401|OR;"
                      "M|Portland|04101|ME;S|Burlington|05401|VT;S|Montpelier|05602|VT;S|Bar Harbor|04609|ME;"
                      "S|Sedona|86336|AZ;S|Taos|87571|NM;S|Galena|61036|IL;S|Key West|33040|FL" },
                    { "de",
                      "L|Berlin|10115;L|Hamburg|20095;L|M%u:nchen|80331;L|K%o:ln|50667;L|Frankfurt am Main|60311;"
                      "L|Stuttgart|70173;L|D%u:sseldorf|40213;L|Leipzig|04109;L|Dortmund|44135;L|Essen|45127;"
                      "L|Bremen|28195;L|Dresden|01067;L|Hannover|30159;L|N%u:rnberg|90402;L|Duisburg|47051;"
                      "L|Bochum|44787;L|Wuppertal|42103;L|Bielefeld|33602;L|Bonn|53111;L|M%u:nster|48143;"
                      "L|Mannheim|68159;L|Karlsruhe|76133;L|Augsburg|86150;L|Wiesbaden|65183;L|Kiel|24103;"
                      "L|L%u:beck|23552;L|Freiburg im Breisgau|79098;M|Heidelberg|69117;M|Regensburg|93047;"
                      "M|W%u:rzburg|97070;M|Ulm|89073;M|G%o:ttingen|37073;M|Erlangen|91052;M|Trier|54290;"
                      "M|Jena|07743;M|Potsdam|14467;M|Rostock|18055;M|Oldenburg|26122;M|Osnabr%u:ck|49074;"
                      "M|Koblenz|56068;M|Bamberg|96047;M|Konstanz|78462;M|Weimar|99423;M|Marburg|35037;"
                      "M|Celle|29221;S|Lindau|88131;S|Meersburg|88709;S|Rothenburg ob der Tauber|91541;"
                      "S|Wernigerode|38855;S|Husum|25813;S|Goslar|38640;S|Landsberg am Lech|86899;S|F%u:ssen|87629;"
                      "S|Quedlinburg|06484;S|Wismar|23966;S|Eisenach|99817;S|Bad T%o:lz|83646;S|Traunstein|83278;"
                      "S|Mosbach|74821;S|Schw%a:bisch Hall|74523;S|Limburg an der Lahn|65549;S|Cochem|56812;"
                      "S|Bernkastel-Kues|54470" },
                    { "es",
                      "L|Madrid|28013;L|Barcelona|08002;L|Valencia|46002;L|Sevilla|41004;L|Zaragoza|50001;"
                      "L|M%a'laga|29015;L|Murcia|30001;L|Palma|07001;L|Las Palmas de Gran Canaria|35002;"
                      "L|Bilbao|48005;L|Alicante|03002;L|C%o'rdoba|14003;L|Valladolid|47001;L|Vigo|36202;"
                      "L|Gij%o'n|33201;L|A Coru%n~a|15001;L|Granada|18009;L|Vitoria-Gasteiz|01001;L|Oviedo|33003;"
                      "M|Santander|39003;M|Pamplona|31001;M|Salamanca|37002;M|Burgos|09003;M|C%a'diz|11005;"
                      "M|Le%o'n|24003;M|Toledo|45001;M|Tarragona|43003;M|Girona|17001;M|Lleida|25007;"
                      "M|Logro%n~o|26001;M|Huelva|21001;M|Badajoz|06002;M|C%a'ceres|10003;M|San Sebasti%a'n|20003;"
                      "M|Almer%i'a|04001;M|Ja%e'n|23001;M|Zamora|49001;M|Segovia|40001;M|%A'vila|05001;"
                      "M|Huesca|22001;S|Ronda|29400;S|Teruel|44001;S|Soria|42001;S|Jaca|22700;S|Trujillo|10200;"
                      "S|Sig%u:enza|19250;S|%U'beda|23400;S|Baeza|23440;S|Olot|17800;S|Figueres|17600;S|Llanes|33500;"
                      "S|Cangas de On%i's|33550;S|Zafra|06300;S|Plasencia|10600;S|Calatayud|50300;S|Tudela|31500;"
                      "S|Haro|26200" },
                    { "it",
                      "L|Roma|00184;L|Milano|20121;L|Napoli|80133;L|Torino|10121;L|Palermo|90133;L|Genova|16121;"
                      "L|Bologna|40121;L|Firenze|50122;L|Bari|70121;L|Catania|95121;L|Venezia|30124;L|Verona|37121;"
                      "L|Messina|98122;L|Padova|35122;L|Trieste|34121;M|Brescia|25121;M|Parma|43121;M|Modena|41121;"
                      "M|Reggio Calabria|89125;M|Perugia|06121;M|Livorno|57123;M|Cagliari|09124;M|Pisa|56125;"
                      "M|Siena|53100;M|Lucca|55100;M|Bergamo|24121;M|Trento|38122;M|Bolzano|39100;M|Udine|33100;"
                      "M|Ancona|60121;M|Pescara|65121;M|Lecce|73100;M|Sassari|07100;M|Salerno|84121;M|Novara|28100;"
                      "M|Matera|75100;S|Assisi|06081;S|Orvieto|05018;S|Alba|12051;S|Bassano del Grappa|36061;"
                      "S|Chioggia|30015;S|Cortona|52044;S|Spoleto|06049;S|Urbino|61029;S|Montepulciano|53045;"
                      "S|Volterra|56048;S|San Gimignano|53037;S|Gubbio|06024;S|Todi|06059;S|Bra|12042;"
                      "S|Saluzzo|12037;S|Chiavari|16043;S|Tropea|89861;S|Ostuni|72017;S|Noto|96017;S|Cefal%u`|90015" }
                };
                std::vector<fictive_city> r;
                for (const auto& t : tables)
                {
                    const std::string text = fictive_lexicon_text(t[1]);
                    std::size_t pos = 0;
                    while (pos < text.size())
                    {
                        std::size_t end = text.find(';', pos);
                        if (end == std::string::npos) end = text.size();
                        const std::string entry = text.substr(pos, end - pos);
                        pos = end + 1;
                        std::vector<std::string> f;
                        std::size_t q = 0;
                        while (q <= entry.size())
                        {
                            std::size_t bar = entry.find('|', q);
                            if (bar == std::string::npos) bar = entry.size();
                            f.push_back(entry.substr(q, bar - q));
                            q = bar + 1;
                        }
                        if (f.size() < 3 || f[0].empty()) continue;
                        fictive_city c;
                        c.size = f[0][0];
                        c.name = f[1];
                        c.postal = f[2];
                        c.state = f.size() > 3 ? f[3] : std::string();
                        c.country = t[0];
                        r.push_back(c);
                    }
                }
                return r;
            }();
            return all;
        }

        inline const fictive_city* fictive_city_named(const std::string& name)
        {
            static const std::unordered_map<std::string, std::size_t> index = []() {
                std::unordered_map<std::string, std::size_t> m;
                const std::vector<fictive_city>& all = fictive_cities();
                for (std::size_t k = 0; k < all.size(); ++k)
                    m.insert(std::make_pair(fold_key(all[k].name), k));
                return m;
            }();
            const auto it = index.find(fold_key(name));
            return it == index.end() ? nullptr : &fictive_cities()[it->second];
        }

    // ------------------------------------------------------------------------------------

    }
}

#endif // DLIB_PRIVACY_FICTIVE_LEXICONS_H_
