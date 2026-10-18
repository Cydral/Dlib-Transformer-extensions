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
                "bethan dafydd gethin rhiannon "
                // Given names from all regions of the world, short forms, nicknames and stage names
                "aamir aart aarti abby abdelaziz abdelhak abdelilah abdelmajid abdelrahman abderrahmane abdeslam "
                "abdi abdou abdoul abdul abdulaziz abdullah abdulrahman abe abebe abena abhay abhinav abhishek "
                "abilio abla aboubakr abril achara achille achim adaeze adalbert adama adao adarsh adebayo adel "
                "adelheid adeola adewale adi adil aditi adjoa adnan adriaan adrian adriana adriano adwoa afaf afonso "
                "afua agata agneta agnetha agron agus agustin agustina ahmad ahmet aikaterini aiko ailen ailsa "
                "ainhoa aino aisha aishwarya aisling aissa aissatou aitana aitor ajay ajit akane akash akihiro akiko "
                "akon akos akosua akram akshay akua akwasi alasdair alassane alba albana alcides aldo aled alejandra "
                "alejandrina aleksandar aleksander aleksandr aleksandra aleksei aleksi alemayehu alena "
                "alessandra alex alexandra alexandros alexandru alexey alf alfonso alfredo alia alicja alighiero "
                "alina aline alizee alla almaz almudena alois alok alonso alparslan alper alun alvaro ama amadeu "
                "amadou amaia amaka amal amalia amandeep amani amare amarjit amedeo amel amir amira amit amitabh "
                "amjad amos amparo amrita anahit analyn anamaria anand anas anastasios anatoly anbu anders anderson "
                "andi andoni andreea andrei andreia andrey andrzej aneta anfisa angel angele angelica "
                "angeliki angelo angus anh ani anibal aniko anil anita anjali anke ankit ankita annalisa annegret "
                "anneke annelie anneliese annemarie annette annika anong anouar anouk anselmo antje anton antonela "
                "antonella antoni antonia antonina antonio antonios antti anuj anupam anusha anushka anwar aodhan "
                "aoibhe aparecida apinya apostolos araceli aram arantxa arash arben archana archil arda ardalan ari "
                "arianna arie arief ariel arif aristotelis arjen arkadiusz arkady arman armando armen armin arnaldo "
                "arnav arne arnel arno aron arsen arsenio arseny arshad artem artur arturo artyom arul arun aruna "
                "arvind arya asaf asbjorn asha ashish ashok ashot ashwin asier asif aslan asli asma assia assunta "
                "aster astghik astrid asuka asuncion athanasios attila atul augusto aurelio aurora avi avraham aya "
                "ayaka ayanda ayano ayelet aylin ayman ayodele ayoube ayse aysegul aysel ayumi azad azadeh aziz "
                "azucena babak babatunde babette bachir badr bahar bahram bakary bala balazs baldev baldomero "
                "baltasar baltazar balwinder bambang banksy banu bao barak baraka baran barasa baris bart bartolome "
                "bartosz basma bassam bastian battista beata beate bebel bebert becky begona behnam behrouz beka "
                "bekele bela belen belkacem bella bence benedetta benedikt benedita benedito beniamino benito benno "
                "benny beppe berhanu berivan berk berkay berna bernardo bernd bernhard bert berthold berwyn besa "
                "besnik beth beto betsy bettina beyonce beyza bharat bhavani bhavna bheki bia bianca bibi bice bijan "
                "biljana binh binta birger birgitta birkir bjarne bjork blanca blanka blathnaid blerta boaz bob "
                "bodil bogdan bogumila boguslaw bohdan bohumil bohuslav bojan bolaji bongani booba boris borislav "
                "borja boubacar bouchra bozena brad brahim bram branko brayan bregje brisa britt bronislaw bruna "
                "brunella bryn bucky budi buhle bujar bulent burak burcu burkhard bushra busisiwe busra byungho "
                "caetano cagatay caio cal calogero camelia camila camilla camilo canan candela candelaria caner "
                "cansu caoimhe caradog carla carlo carmela carmine caro carola carolina carsten carys casimiro "
                "cassie cassio catalin catalina catarina caterina cathal cathy catriona cayetana cecilia cees "
                "celestino celia cem cemal cengiz cesar cesare cezary chaewon chafik chaim chaima chakib chakir "
                "chandra changmin charalambos charo chau chawki chaz cheikh chelo chema chengwen chet chetan chiaki "
                "chiamaka chico chidinma chihiro chinedu chinonso chinwe chioma chirag chitra chris christa "
                "christoph christos chrysa chuck chukwuemeka chunhua chus chuy ciaran ciccio cicero cindy cinzia "
                "ciro claudia claudio claudiu claus cleber clodagh coco colm conall conceicao concepcion concetta "
                "concha conrad consolacion constantin consuelo corina cormac cornelia corrado corrie cosimo cosmin "
                "coumba cristhian cristian cristiano cristobal cristovao csaba cuong cyrus czeslaw daan dadju dafna "
                "dag dagmar dahye daiana daiki daisuke dalia dalibor damian damiao damla damso dan dana daniela "
                "daniele daniil danijela danila danilo danny dante danuta dany dao dara dario darius dariush dariusz "
                "darko darragh darshan darya dasha dat dave davey davi davide davit davood dawid dawit dayana "
                "debbie debora dede deepa deepika deirdre dejan demba demi denes denisa deniz dereje dervla derya "
                "despina detlef devendra dewi dhruv dian diana diarmuid didem didi diederik diep dietmar dietrich "
                "dilan dilek dima dimitra dimitrios dinesh dinos diogo dionisio dionysios dirk djamel djibril dmitry "
                "dogan dolf domenico domi dominik dominika donal donatella donghyun dongmei dongsheng "
                "dongwook donia dora dorin dorota dorothea dottie douaa dougal dounia dov doyun dragan driss drita "
                "dritan duc duda dudu dudule duong durga dusan duygu eamon ebba ebere eberhard ebru ebuka ece "
                "eckhard eda eddie edeltraud edgar edit edmilson edmond edmund edmundo edoardo edson eduard eduarda "
                "eduardo edwin edyta eelco eero effie efia efrain efren efua egil egon egor ehsan eimear einar eino "
                "eirik eirlys eitan eithne ekaterina ekin eko ekow elad elaine eleni eleonora elfi elfriede elham "
                "eli eliana elias eliezer elif elin elina eline elio elisa elisangela elise eliseo eliska elizaveta "
                "elke elkin ellen ellie ellinor elmar eloy els elvira elzbieta emanuel emanuele emelie emerson emi "
                "emil emilia emiliano emilio emine eminem emir emre emrys encarnacion endang engin eniko enkeleda "
                "ennio enrico enrique enya eoghan eran erdem eren erez erich erika eriko erion erkan ermanno erminio "
                "ernesto ernst ertugrul erwin erzsebet esfandiar esi eskil espen esperanza esra esteban estefania "
                "ester esther ettore eugenio eugeniusz eunbi eunji eunwoo eusebio evangelia evangelos evaristo "
                "evdokia evelio evert everton evgenia evgeny evi ewa ewald ewan ewelina ezra fabiana fabio fabiola "
                "fabricio fabrizio facundo fadi fadwa fahd faina faisal fanfan fangfang farah faraji faramarz farhad "
                "farhana fariha faris farnaz farouk farzaneh fatih fatiha fatma fatmir fatos fatoumata fausto fawad "
                "fawzi fayza federico fedor feliciano felipe femke fenna ferdinand ferdinando ferenc fereshteh "
                "fergal ferhat fermin fernanda fiachra fidel fifi fiifi fikru filip filippo filiz finbar finn fiona "
                "fionn fiorella fito fitri flavia flavio flo florencia florencio florentina florin floris folake "
                "fotini fouad fran franca franci francisca franciszek franco frankie frantisek franz franziska "
                "fraser fred fredy freek frida frieda frieder friederike friedrich frits fritz frode froukje "
                "fulgencio fulvio fumiko funmilayo gabby gabi gabor gabriela gabriella gaby gaetano gagik gal galina "
                "galit galya gamze ganesh gareth gaspar gaurav gayane gayatri gebhard geert geertje geeta gege geir "
                "gema genaro genet gennady gennaro genowefa gentian georg georgia georgiana georgios georgy geraint "
                "geraldo gerardo gerben gerd gerda gergely gerrit gerson gert gertrud gesine getachew ghada ghasem "
                "ghassan gheorghe ghizlane giacomo giada giancarlo gianfranco giang gianluca gianmarco gianni gideon "
                "gigi gijs gilberto gims gines ginevra gino giorgi giorgia giorgio giovana giovanna girish girma "
                "girolamo gisela gisle gita giuliana giulio giuseppina giusi gizem gjergj glauco gleb glenda goga "
                "gokhan golnar gonul gonzalo gopal goran gordana gordon gorka gottfried govind goyo graca graciela "
                "gradur grainne graziella grazyna greg gregor gregorio greig grigory grisha grzegorz guadalupe "
                "guangming gudrun guido guilherme guillermina guillermo guiying gul gulalai gulsen gunilla "
                "gunnar gunther guoqiang gurdeep gurjit gurkan gurpreet gus gustav gustavo guto gwilym gwyneth "
                "gyorgy gyula haakon habib hachim hadas hadi hafid hafida hafiz hafsa hagen hai haile haim haitao "
                "hajar hajime hajnalka hajun hakan hakim halil halima halina hallvard halvard halyna hamdi hamed "
                "hamisi hamza hana hanako hande haneul hanh hani hank hanna hanneke hannelore hannes hannu "
                "hansi hansjorg hao haoran haoyu harald hari harish harjit harm haroon harpal harsh hartmut hartwig "
                "haruka haruki harvinder hasan hasmik hasna hatem hatice hattie haukur hayat hayato hayoon haziq "
                "hedda hedwig heidi heikki heiko heini heinrich heitor helena helga helge helio heloisa hema hemant "
                "hendra hendrik hendrika henk henning henriette henrik henrique henryk herbert heriberto hermann "
                "herminia hernan herwig hevin hiba hichem hidde hideki hien hiep hieu higinio hikari hikmet hila "
                "hilal hilario hilde hildegard hillel hina hinata hind hinrich hiroki hiroto hirut hitesh hitomi hoa "
                "hoai hoang holger hong hongmei hongwei hooman horacio horst hossein houda houssam hovhannes hrafn "
                "hue huifang hulya hung huseyin huw huy hyejin hyuna hyunjung hyunsoo hyunwoo iain ichiro ida ido "
                "idrissa ieuan ifeanyi ifeoma ignacio ignacy ignaz ignazio igor ijeoma ikenna iker ikram ilaria "
                "ileana ilham ilhan ilias ilir ilker ilkka ilona ilse ilya imad imane imen immacolata imre inaki "
                "inas indah inderjit indila indira ineke inge ingeborg inger ingo inmaculada inna innokenty "
                "inocencio intan ioan ioana ioanna ioannis iolo ion ionut ipek iracema irakli irem irena irene irfan "
                "irit irmgard isa isabela isadora isamu isha ishaan isidro iskandar ismael ismail issa issam istvan "
                "itai italo itamar itzel itziar iulia ivana ivar ivica ivo ivone iwona izabela izumi izzy jaana jaap "
                "jabari jabulani jacek jacinto jackie jacobo jacopo jacquot jadwiga jaeho jaehyun jagdish jaime jair "
                "jake jakob jakub jalal jamal jamie jamila jamshid jan jana janaki janina janneke janos janusz "
                "jaqueline jari jarle jaroslav jaroslaw jasmine jasper jaspreet jasvir jatin javed jawad jaya jayant "
                "jazmin jeannot jedrzej jefferson jelena jelle jen jennalyn jenny jens jericho jeroen jeronimo jerzy "
                "jess jetmir jhon jhonatan jiahao jiahui jianguo jianhua jianjun jianming jianping jiayi jieun "
                "jihane jiho jihoon jim jimena jimin jimmy jindrich jingjing jinlong jinwoo jiri jiro jisoo jitka "
                "jiwoo jiyeon jiyoo jiyoung joanna joaquim joaquin jocelyn jochem jochen jody joep joey johan johann "
                "johanna johannes johnny jojo joko jolanda jolanta jomar jon jonas jongsu joost jorg jorrit josef "
                "josefa josefin joselito josh josu jovan jozef jozsef juana juca judit juha juju jukka julian "
                "juliana juliano julieta julio juma junaid jungwoo junichi junjie junko junseo junwei jupp justyna "
                "jutta jyoti kaan kabir kacper kader kadir kaede kaio kaisa kaito kajal kalle kalliopi kalpana kamal "
                "kamala kamau kambiz kamel kamil kamila kamini kamran kana kangmin kanya kaori kapil karan karel "
                "kari karin karina kariuki karla karlijn karol karolina karsten karthik kartika katalin kate "
                "katerina kateryna katharina kathy katie katja katya kaveh kavya kawa kazimiera kazimierz kazuki "
                "kazuko kazuo kebede kees kehinde keiko keila kelechi kemal kenan kendji kenichi kenny kenta kentaro "
                "kenza kerem keren kerstin ketevan keyvan khaled khalil khang khanh khaoula khoa khurram kian kidane "
                "kike kiki kim kimmo kinga kipchoge kiprono kipruto kira kiran kirill kishore kittisak kitty kjell "
                "klaas klara klavdiya kleber knut kobi kobina koen koji kojo kolya konrad konstantin konstantinos "
                "koray kostas kostya kourosh krishna kristine kristjan kristof kristoffer kristyna krisztina "
                "krystyna ksenia kubra kujtim kuldeep kulwant kumar kumiko kunal kunle kuno kurt kwabena kwadwo "
                "kwaku kwasi kweku kwesi kyoko kyungsoo lacrim ladan lado lahcen laila lais lakshmanan lakshmi lalit "
                "lalo lam lamberto lamine lanying larbi larisa larissa lartiste larysa lasha laszlo lata "
                "laureano lauri lauro lavinia laylow leandro lech leena leif leire lela lele lello lena leni lenka "
                "leocadio leonardo leone leonid leonidas lerato lestari leszek leticia letizia lev levan levent "
                "levente levi lexi leyla lianjie liat libby libor lidia lieke liesel lieselotte lihua lili lilit "
                "limor lindita lindiwe linh linnea lino linus lior lisandro lisanne liudmyla liv liviu lixin liz "
                "lizbeth lizzie ljiljana llewellyn loan loes lolo lomepal lorcan loredana lorella lorena lorenz "
                "loris lorna lotfi lothar lotte lotti lottie loubna loulou lourdes lourenco lovisa lowri luana lubna "
                "lubomir lubos lucho lucian luciana luciano lucila lucio lucyna ludmila ludwig luigi luisa luise "
                "luiza luka lukas lukasz luljeta lulu lunga lupe lutz luuk lwazi lyes lyosha lyuba lyubov lyudmila "
                "maame maarten mabrouk macarena macario maciej madalena madalina maddalena maddie madhav madhu mado "
                "mads maes maeve maggie magnus maha mahdi mahesh mahmoud mahmuda mahmut mahnaz mahnoor mahsa maia "
                "mairead maite maja majid makar makoto malak malalai malee malgorzata malin malini malu mami mamou "
                "mamta mamuka manal mandla mandy manfred manish manlio manoel manoj manolis manolo manpreet mansour "
                "manu manuela marcela marcelino marcello marcelo marcia marcial marcin marcio marcos marek margarete "
                "margarida margarita margherita mariam marian marianne mariano maribel maricel maricela marieke "
                "marija marijke mariko marina marisa marisol marit marites mariusz marivi marja marketa markku marko "
                "markus marli marnix marouane martijn martina marwa maryam marzena masaki masako masao masaru masha "
                "masoud massimiliano massimo mate matej mateo mateus mateusz matheus matias matilda matilde mats "
                "matt matthias matti mattia matvey mauricio maurizio maxim maximilian maximiliano maximino maya "
                "mayank mayra mayu mayumi mechthild meena mees meg megumi mehmet mehrdad mei meihua meiling meinhard "
                "meir meirion mekdes melchor melek melike meltem menachem menno mentor merav mercedes merche merel "
                "meriem mert merve meryem meseret mesut metin mette mhairi micaela michaela michal michalis michela "
                "michiel michiko mick midori mieke mihaela mihai mihaly mika mikael mike mikel mikhail miki mikko "
                "miklos milagros milan milana milena milica milind millie milos miloslav miloud mimi mimile "
                "mimoza mina minah mingyu minho minji minna minori minoru minseo mircea mireia mirella miriam mirjam "
                "mirko miroslav miroslaw misaki misha mitch mitra mitsuko miyu mladen moa moacir moby modou "
                "mohan mohsen moises mokhtar molly momo momoka mona moncef monica montse montserrat morag moritz "
                "morten morteza moshe mostafa mouloud mounia mounir mourad mpho mstislav muireann mukesh mulugeta "
                "murat murdo murilo murugan musa mustafa muthu mwajuma mwangi mykhailo mykola mylene nabila nacer "
                "nacho nada nadav nadeem nadezhda nadir nagarajan nahid nahuel najat najib najwa nalini nam nan "
                "nanami nanda nandi nandini nando nanou naoki naoko naomi narciso narek naresh narong nasrin nasser "
                "nastya nataliia natalya natan natasha nate natsuki nattapong naveed naveen navid navjot nawal "
                "nayeli nayla naza nazanin nazar nazli nazzareno ndeye nebojsa necati ned neema neeraj negar neha "
                "nekfeu nektarios nele nell nello nelson nemanja nemesio nena nerea nesrin nesrine neta neusa nga "
                "ngan ngoc nguyet nhan nhi nhung niall nicky nico nicola nicolae nicoleta nicoletta niels nienke "
                "nieves nihat niina nikita nikki niklas nikola nikolaos nikolaus nikolay nilesh niloufar nima "
                "ninette ninho ninni nino nir nirmala nisha nishant niska nisrine nitin nizar njeri njoroge nkechi "
                "nkosinathi nnamdi noa noam noboru nodar noelia noemi nokuthula nomvula nonna nono noor norbert "
                "norberto noriko nounou nour noura noureddine nuala nune nur nurcan nuria nurul nyokabi oana "
                "obiageli obinna octavio odair oded ofer ognjen oguz okan okonkwo oksana ola olafur olamide ole oleg "
                "oleksandr oleksii olena oliwia olli ollie olumide oluwafemi oluwakemi oluwaseun olya omer omid "
                "omondi ondrej onur onyango orazio orelsan oren oreste orhan ori orlaith orlando orly ornella "
                "orsolya ortrud osama osamu oskar osman osnat osvaldo oswald otar otavio othmane otieno ottavio otto "
                "ouafa oumar oussama ovidiu oya ozan ozge ozgur ozlem paco padma padraig paivi palesa pallavi paloma "
                "pam panagiotis pancha pancho pankaj paola paramjit paraskevi pari parisa parth parvati parviz "
                "pascual pasha pasquale pat patricio patrizia patrycja patryk patty paulina pavel pavla pavlos "
                "pawel payam peadar peder peggy pekka penda penny pepa pepe peppe pernille pertti petar pete petr "
                "petri petro petru petya phat phil philipp phong phuc phuong pierina pierluigi piero pierrette "
                "pierrot pieter pietro pili pim pinar pino pippo piroska piyush placido polina polly porfirio poupou "
                "pouya prabhu pradeep prakash pranav prasert prashant prateek pratik predrag preeti premysl primo "
                "priscila priyanka prudencio przemyslaw purificacion putri qiang qiuyue quan quang quim quinten "
                "quique quirin quirino quoc quyen rabah rabia rachele radek radha radim radka radoslaw "
                "radovan radu rafa rafaela rafal raffaele raffaella rafi rafik raghu ragnar ragnhild rahim rahma "
                "rahmat raimundo rainer raisa raj rajae rajat rajesh rajinder rajiv raju rakesh ralf raluca ram "
                "ramesh ramin ramiro ramon ramya ramzi randa rani rania ranjit raquel rashed rashid rashmi rasmus "
                "ratna razvan rebeca recep reda redouane reggie regina reiko reinaldo reinhard reinhold reinier reka "
                "rekha remedios remigio remigiusz ren renata renate renato rens renske resi revathi revaz "
                "reyes reza rhea rhodri riad riccardo ricky rico ridha rigoberto rihanna riikka riitta rik riku rim "
                "rima rina rinaldo riri risa ritesh riton ritu riza rizwan roald robbie roberta rocco rocio rodel "
                "roderick rodion rodolfo rodrigo roel roelof rogelio rogerio rohff rohit rojda rojin rokhaya "
                "rolando rolf roman romana romano romina rommel ron ronaldo ronit ronnie roos rosa rosalba rosalia "
                "rosana rosangela rosemarie rosendo rosi rosie rossella rostam rostislav roxana roya ruairi "
                "ruben rudi rudiger rudolf rumana rune ruoxi rupa rupert ruslan rut rutger ruya ryosuke ryota "
                "ryszard saad saba sabatino sabri sachiko sachin sadik saeed safia sagrario sahar saida saima sajid "
                "sakura salah salaheddine salif salim sally salman salvatore salwa saman sameer samia samina "
                "sammy samuele sana sanaa sanaz sancho sandeep sander sandor sandro sandy sanghoon sangwoo sanja "
                "sanjana sanket sanna sanne santiago santino santosh saravanan sargis sari sarita sarka sascha sasha "
                "saskia satish satomi satoshi satu saturnino saurabh saverio savita sawsan sayaka seamus seb "
                "sebastian sebastiao seda segun segundo seif selam selim selin selma semra semyon sena senthil "
                "seojun seokjin seonaid seoyun sepp seppl seppo serap sercan serdar serena sergey serhii serkan "
                "seryozha setiawan seulgi seun seunghee seungho sevda severin severino sevgi seydou shabnam shah "
                "shahid shahram shahzad shai shakira shalini shamim shanti shashank shaul shayan shekhar shengli "
                "sherif shigeru shilpa shinji shiori shira shirin shiv shiyu shkelzen shlomo shmuel shobha shoshana "
                "shota shreya shruti shuang shufen shun shweta shyam siamak sian sibel sibongile sibusiso sid "
                "siddharth siebe siegfried sietske sigal sigrid sigrun sigurdur siham silke silvana silvestre silvio "
                "simge simona simran sinan sindre siobhan sipho siriporn sirpa sita siti siv sivakumar siwan siwoo "
                "siyabonga sjoerd slamet slava slavica slawomir slimane smail smita sneha snezana snezhana snorre "
                "sobia socorro sofie sohee soheil sohrab soledad solomon solveig somchai sompong somsak somying "
                "songul soojin soolking sopho soprano soraya sorcha sorin sotiris soufiane souhila souleymane "
                "soumaya souta soyeon spyridon srdjan srinivas stanislav stanislaw stanislawa stavros stavroula "
                "stefania stefanie steffen steffi steinar stelios sten stepan steph stig stijn stina stromae struan "
                "suat subramanian suda sudha sueli sujata sujin sukhdev sukhwinder sultan sultana sumaira suman "
                "sumiko sumit sunanda sune sunee sungho sungmin sunil supachai suresh susana susanne susanti sushila "
                "susi susie suzy svatopluk svea sverre sveta svyatoslav swapnil swati syafiq sybille sylwester "
                "szabolcs szymon tad tadashi tadeo tadeu tadeusz taehyung taeyang taha tahar tahir tahmineh taiwo "
                "takeshi takumi takuya tal tali tam tamaki tamar tamara tamari tamas tamer tamilselvi tan taner "
                "tania tanja tanvi tanya taoufik tapio taraneh taras tarek tarik tariq tarja taro tarun tasnim "
                "tatsuya tatyana tayc tayeb tayfun teddy teemu teguh tejinder telmo teodora teodoro teona tereza "
                "terje tesfaye tess tetiana tetsuya teun tewodros thabo thais thanasis thanawat thandeka thandiwe "
                "thao themba theodoros thiago thijs thinh thom thor thordis thorsten thu thuy tiakola tianyu tiao "
                "tibor tiburcio tidiane tien tigist tigran tiina tijmen tim timea timmy timo timofei timur tin tina "
                "tineke tingting titi titilayo titou titti tiziana toan tobi tobias toby toinette tolga tolya tomas "
                "tomer tommaso tommy tomoko tomomi tomos toni tonino tony torbjorn tornike toru tova tove trang tri "
                "trinh trinidad trish trixie trond trude trudy trung tshepo tsion tsubasa tuan tudor tugba tulay "
                "tullio tuncay tung tuomas turgut tushar tuula tuva tuyet tzvi uchenna uday udi udo ufuk ugo ugur "
                "ulf ulises ulisses ulla ulrich ulrika ulrike ulyana uma umberto umesh umut una unai urbano uri "
                "urszula uschi usha usman ute uyen uzma vaclav vadim vahe vahid vaibhav vakhtang val valbona valeria "
                "valeriano valerio valeriya valery valon valter vandana vanesa vanya vardan varun varvara vasco "
                "vasile vasileios vasiliki vasily vega vegard vegedream veikko veli vendula veniamin venkat vera "
                "verena veronica veronika vesna vianney vicente vicky vidya viet viggo vijay vikas vikrant viktor "
                "viktoria viktoriia vilhelm vilma vinay vince vincenzo vinh vinicius vinod vinzenz viola violeta "
                "viorel virginia vishal vitaa vitaly vitor vitoria vittoria vittorio vitya vivek vladislav vlasta "
                "vojtech volkan volker volodya volodymyr vova vsevolod vusi vyacheslav wael wafa wafula wagner wahyu "
                "wairimu wakana waldemar wali walt waltraud wanda wanderley wanida wanjiku wanjiru waqas warda "
                "wassim wati weimin weiwei wellington wenbin wenceslao wendelin wendy weronika wiam wichai wiebke "
                "wieslaw wigald wiktor wiktoria wilhelm willem willi wilma wilmer wilson wim wioletta witold "
                "wladyslaw wooyoung wouter xabier xaver xiaobo xiaohong xiaojun xiaoli xiaolong xiaomei xiaoming "
                "xiaoqing xiaowei xiaoxia xiaoyan xiaoyu xico ximena xinyi xiufen xochitl xuan yaa yaakov yacouba "
                "yael yaffa yahia yair yaiza yakov yamato yamila yamini yana yannis yanping yaroslav yaroslava "
                "yasaman yasemin yash yasin yasmin yasmina yasser yasuko yavuz yaw yaya yazid yehuda yeison yejin "
                "yejun yekaterina yelena yelle yen yeonhee yerson yesenia yetunde yevgeny yevhen yiannis yichen "
                "yigal yildiz yilmaz yingying yitzhak yoav yogesh yohannes yoko yolanda yonas yonatan yongjun yonit "
                "yosef yoshiaki yoshiko yoshinori yoshio younes younghee youngjae youngmi youri yousef yousra "
                "youssoupha yseult yudi yuhang yui yujie yuka yukiko yulia yuliana yumi yumiko yuna yuqing yura yuri "
                "yurii yusuke yuta yuto yutong yuval yuxin yuxuan yuying zacarias zach zafar zafer zaher zahra zaida "
                "zakaria zakhar zana zanele zawadi zayd zaz zaza zazie zbigniew zdenek zdzislaw zdzislawa zeca "
                "zeeshan zeferino zehra zelia zeljko zenon zeynep zezette zezinho zhanna zhenya zhenyu zhihua "
                "zhiqiang zhiwei zhiyuan zhongwei zihan zinaida zine ziv zixuan zizou zofia zohar zohra zohreh zoi "
                "zola zoltan zoran zorica zoubir zouhair zoya zsofia zsolt zsuzsanna zubin zurab zuri zuzana zuzanna "
                "zygmunt");
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
                "marion morgan justice ruth victor william ravi philippine gemma "
                "alba alexandra alun amadou amir anders angel angelica aparecida ariel aster asuncion "
                "aurora austin baraka berk bibi bob brad bucky cal camelia candela caner carmine carolina cesar "
                "charlotte cheikh chuck cicero cleber coco concepcion concha corrie dag dan darshan davit "
                "demi dirk dorin duc edit emir engin ester eugene evert fatma florencia florin franca franco "
                "fritz gabby gaby gal gayatri genet gentian georgia gesine gus hakim hank harm harsh helio "
                "hila hind hong horst hue hung ido inge ion italo jacquot jan jasmine jasper jenny jess jimmy joey "
                "johnny jojo josh kamala kana kawa kike kiki kitty lam lamine lawrence lech leon lev lino loan lolo "
                "loris lotte loulou lourdes lulu lutz madison mahdi maia malin manu margarita marina marli mate matt "
                "maxim maya meg mentor mercedes mick mike milan mimi molly nacho nadir nan nancy "
                "nasser ned nelson nono nounou ole ollie orlando pari pasha pat patty peggy penny pepe phat "
                "pierrette pierrot pieter primo rainer ram ramon rani regina rhea rim roman "
                "romano rune rut sally sana sander sandy santiago sari satoshi shah sherif shiv shun "
                "soprano sultan sultana tad tali tam tamer tan taro teddy tien tin titi tomer tommy tony tri trixie "
                "tyler val veronica victoria viola virginia vitoria wellington yaw yen zine");
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
                "beneficiaire assure assuree locataire proprietaire conjoint conjointe defunt defunte "
                "soussigne soussignee soussignes soussignees");
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
                "srl sl pty ltee asbl sprl scrl bvba vzw kgaa oyj ltda pte pllc sca "
                "mbh gbr aktiengesellschaft kommanditgesellschaft sau slu sll sapa sicav eireli ehf aps lllp bhd sdn "
                "pvt pjsc jsc oao zao tov nyrt");
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
                "polyclinique "
                "universidad universidade universita universitat universiteit uniwersytet hochschule fachhochschule "
                "escuela escola scuola schule colegio liceo instituto istituto instytut stiftung fundacion fundacao "
                "fondazione verein asociacion associacao associazione vereniging banco banca sparkasse volksbank "
                "ospedale klinik clinica klinikum krankenhaus policlinico museo museum musee "
                "teatro theater theatre orquesta orchestra orchestre conservatorio akademie accademia "
                "compania compagnia gesellschaft grupo gruppo gruppe verlag");
            return s;
        }

        inline const std::unordered_set<std::string>& organization_sectors()
        {
            /* Words naming the trade of a company, written after its name: "Zorvane Finance",
               "Kerlys Logistik", "Ulmeo Seguros", "Yarrow Building Society". */
            static const std::unordered_set<std::string> s = make_word_set(
                "conseil services immobilier transports industrie industries distribution batiment informatique "
                "ingenierie logistique assurances assurance courtage formation sante energie environnement "
                "patrimoine expertise finance finances medical consulting logistics engineering solutions systems "
                "holdings partners analytics healthcare properties capital technologies media group insurance "
                "bank law academy clinic motors plumbing bakery practice solicitors society surveyors "
                "accountants metropole agglomeration agglo "
                "airlines airways pharma pharmaceuticals laboratories telecom telecommunications shipping "
                "immobilien versicherung versicherungen logistik beratung bau seguros servicios inmobiliaria "
                "construcciones consultores asesores ingenieria transportes costruzioni servizi assicurazioni "
                "immobiliare trasporti consulenza imobiliaria seguradora verzekeringen building");
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
               their own, and a name span stops when it reaches one. The words are held in
               several literals, each shorter than the 65535 bytes some compilers accept. */
            static const std::unordered_set<std::string> s = []() {
                std::unordered_set<std::string> words = make_word_set(
                    "le la les un une des du de d l en et ou mais donc or ni car ce cet cette ces mon ma mes "
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
                    "confidential ");
                // Frequent words of French and English that are neither given names nor family names
                const char* const frequent[] = {
                    "one like time get also only after see make think back want because most very need really work being "
                    "too going before off made still take got many never life through last while such look something "
                    "same used both part three around between always better find help since another own things during "
                    "game thing give again against found show feel sure family keep please put money someone away days "
                    "night play until doing few let real called different having thought done however getting government "
                    "looking women business start system already anything nothing today change enough everything making "
                    "told yet bad hard mean once support including music seen stop believe call took far though try went "
                    "actually later party run says open season shit children everyone trying using following makes "
                    "together war whole face kind maybe per working course games health least means within able book "
                    "friends information local ago others talk fact guys hand level mind often single become body coming "
                    "control food hours pay problem almost fuck history known research room several started taking wrong "
                    "along anyone else matter pretty remember bit hit needs playing probably saying understand yeah "
                    "class comes idea international looks past possible wanted cause due human members move series wait "
                    "ask data late saw special watch either fucking future light million morning stay taken buy reason "
                    "soon turn whether among check development further heart minutes myself yourself although asked fire "
                    "fun players behind easy gonna non political quite talking works according available final former "
                    "kids list ready sometimes bring current example experience meet type baby process study word across "
                    "action gave gets himself outside self students words cost cut held instead moment mother seems "
                    "thinking wants energy fight fine force hear issue played points running shows space term wife "
                    "beautiful goes killed sex shot site account especially eyes include parents period position record "
                    "similar above died happened knew likely military perfect personal security share won couple dead "
                    "happen hold industry inside issues online player problems return rights sense test break companies "
                    "event member needed present takes training wish design finally girls gone guess interest learn "
                    "policy added alone average brought certain hands longer original performance press received role "
                    "sent themselves tried worked areas became books cool exactly giving provide relationship usually "
                    "value evidence follow official production rate save stuff tax whatever amount countries eat feeling "
                    "felt management match model picture size step trust changes forward groups mom paid range review "
                    "science trade various brother cannot character cup football hate looked natural property quality "
                    "style vote amazing blood complete dog economic hell involved itself language serious stage terms "
                    "title attack damn decided decision enjoy entire kill met perhaps poor release situation technology "
                    "turned website written choice code considered continue cover currently election events financial "
                    "hair increase legal lose pick race seem sign simple simply staff began built career changed crazy "
                    "daily daughter difficult knows loss modern ones paper parts popular published safe starting voice "
                    "whose writing army forget huge listen okay rules success towards waiting ways access below created "
                    "deep followed missing offer professional released risk schools sleep box build dark mine minister "
                    "percent products recent seeing straight visit wanna wrote allowed boys etc fans gives growth "
                    "included married officer pain places respect response speak specific standard tonight write album "
                    "century charge create effect except funny moving network provided recently required sales spent "
                    "store student tomorrow track via watching weight addition ahead allow anti beat conference "
                    "difference double expect gas moved plans population potential pressure radio text treatment ass "
                    "beginning campaign certainly completely content credit described despite focus husband individual "
                    "interesting join kept leading loved nearly particular previous quickly reported sort speed travel "
                    "consider contact drop feet kid throughout absolutely additional beyond earlier forces immediately "
                    "leaving minute nature quick sell significant studies unless winning agree clean construction "
                    "episode favorite income movement photo posted scene sold sounds spend statement teams ability "
                    "announced asking calling coach collection continued costs definitely designed expected gun happens "
                    "heavy includes knowledge particularly search subject train wide wow author claim dad developed fear "
                    "fit generally global gotta hotel interested leader letter lines material named nobody opportunity "
                    "product regular sister stories unit workers annual anymore degree families features finished floor "
                    "growing hurt image majority meant opening opinion physical rule seriously stupid successful active "
                    "administration approach biggest civil dance defense none reasons weekend worst awesome band clearly "
                    "compared effort ended fighting imagine impact lack latest learning multiple older operation "
                    "organization passed pictures protect secret telling activities analysis anyway bought calls choose "
                    "color competition details direct dream easily finish increased literally luck marriage names "
                    "necessary patients resources skin supposed thus yesterday caught closed damage directly disease "
                    "doubt drink driving established feels glad grow largest machine overall professor programs records "
                    "reports shown sit trip associated carry cars crime effective effects explain fully highly holding "
                    "laws parties reality spot worse advice award block caused challenge characters equipment eventually "
                    "helped lived lots otherwise prices primary purpose rates responsible shop showing sick teacher "
                    "theory uses avoid camera catch cell comments drug economy environment executive foot meaning "
                    "officers operations politics produced ran therefore trial truly weather activity app application "
                    "claims complex division evening flight freedom heat highest interview library located location "
                    "murder offered putting seconds showed sitting standing stars walking accept actual appear attempt "
                    "broke channel distance eating finding learned losing opened placed powerful reached receive "
                    "religious screen serve signed slow species speech traffic types wearing wonderful agreed airport "
                    "animals appears benefits cities demand engine everybody famous ideas investment keeping partner "
                    "plays raised runs sad solution songs stopped structure traditional twice wind worry appeared "
                    "becomes bus cent count covered critical digital forced fresh mental mentioned missed mostly mouth "
                    "owner photos previously realize remain scale score separate starts surface throw totally views "
                    "wedding acting actions arms benefit click estate failed fashion feature fund generation hearing "
                    "larger metal pull push returned seemed sexual understanding agent animal apply basis becoming dude "
                    "employees enter follows gain individuals leaders memory prime ring served soul spread supply waste "
                    "weird adult apparently artist edition happening healthy method obviously option prison provides "
                    "remains smaller somebody strength wild window winner arrived bet cast continues correct dangerous "
                    "extremely firm greatest handle improve indeed leaves movies prevent removed spirit television till "
                    "trouble videos advantage apart aware cat customers decide dinner dollars function gift helping "
                    "herself impossible influence items marketing materials produce progress proud require shooting shut "
                    "standards tells thinks background birth carried classes completed concept copy dogs drugs efforts "
                    "housing journal labor leadership length lucky neither onto possibly prove rare setting skills "
                    "software thousands units alive balance birthday bitch boss cards changing connection dress easier "
                    "fellow horse knowing liked magic managed map owned stick turns vehicle volume beauty believed "
                    "billion busy buying cells concerned conversation criminal cultural develop driver ends existing fix "
                    "fly guide images investigation operating paying presented raise responsibility roll slightly "
                    "suggest technical thoughts treat unique variety violence weapons youth appreciate bigger breaking "
                    "discovered evil excited forever funds helps injury lovely magazine models offers ordered prepared "
                    "religion sites somewhere stated strategy teachers wine accounts audience closer core description "
                    "dropped excellent exist figures guard honest issued joined lies likes mention nuclear presence "
                    "reaction reduce shoot sides solid sport taste tea victory afternoon assistant citizens classic "
                    "clothes decisions electric emergency entered entirely facts failure festival flat fuel houses "
                    "initial introduced kick massive matters pair picked pieces plane plenty proper providing session "
                    "shape sky teaching transfer useful watched willing zone accident advanced alternative anywhere "
                    "awards bringing capacity climate communities discussion drinking duty fantastic feelings flying "
                    "hundred industrial joint mix options path plants policies promise proposed purchase rain remove "
                    "signs spending supporting terrible tired treated turning warm afraid arts command crew crowd dating "
                    "elements enemy ensure environmental filled fixed intelligence intended limit profit proof soldiers "
                    "wins appearance attorney behavior bodies brothers buildings chair creating debt domestic expensive "
                    "grew historical homes honestly honor jump launch listed minimum native noted originally planned "
                    "sets suddenly survey tech trees younger ancient attacks charges combined communication connected "
                    "contains download ending exercise express flow formed girlfriend hero illegal increasing joke "
                    "methods officials performed planet relationships restaurant selected shared shopping soft stuck "
                    "suggested supported surprised taught transport accepted adding affairs allows appeal applied "
                    "appropriate artists confirmed device drama entry feed grown hoping keeps lawyer legs lying measures "
                    "mistake organizations platform pulled regarding relations requires saved scientific shoes smoke "
                    "squad teach testing tests values walked abuse angry businesses comfortable concern developing "
                    "discuss elections emotional everywhere facilities falling guns interests internal laugh leg letters "
                    "listening lunch pack payment perform recorded relatively sector sharing streets strike studio weak "
                    "actor advance apartment chain committed confidence cute equal fake focused hits identity journey "
                    "leads maintain measure numerous owners posts quiet revealed specifically task taxes twenty affected "
                    "aircraft applications approved approximately argument arrested claimed conflict considering "
                    "corporate debate determined escape extended factors faster fill films friendly lay lights millions "
                    "mixed phase properly pure reduced requirements residents revenue secure smile strange talent "
                    "temperature troops truck votes authorities basically besides blame bowl causes chicken collected "
                    "context coverage determine display dying elected examples experienced false fired forgot funding "
                    "identified incredible inspired launched meat mode noticed novel obvious passing positions remaining "
                    "scored shots slowly stadium stores surgery trading vision whenever worried zero allowing begins "
                    "champion charged cream crisis delivered estimated giant jail literature moments opposite ourselves "
                    "remained selection serving struggle suicide talked theme tiny typically unfortunately usual "
                    "vehicles voted voting alcohol breakfast brings capable carrying chosen combination cutting "
                    "destroyed drunk essential fail familiar finds granted guilty humans hundreds improved largely "
                    "laughing markets opportunities papers perfectly recommend referred relevant seek sending solo spoke "
                    "stands talks ticket unable upset birds creative cycle dealing directed educational entertainment "
                    "facility goods hang holds info mainly maximum newspaper offering painting reserve returns scared "
                    "shares statistics territory threat tickets adults affect appointed armed aside assistance blow "
                    "boyfriend careful circumstances communications concerns controlled cry danger deals deserve devices "
                    "dollar dreams empty enjoyed explained faces folks fucked gender kinda matches motion moves prize "
                    "realized reasonable receiving register resolution saving sees singing tools typical universe "
                    "warning wars admit attitude conducted decades dedicated definition drawing favor flag frame guest "
                    "heaven institutions kiss plot possibility recovery rent replace represent reviews scenes seeking "
                    "sentence teeth tips trained understood academic accurate achieve afford assume bottle bunch "
                    "category cheese chemical competitive detail diet fruit harder item mess normally occurred permanent "
                    "personally pleasure prefer scheme shift stood storage tank tend tight transportation ultimately "
                    "unlike weekly anybody assets basketball button candidates combat consumer counter crying defined "
                    "depending depression describe drivers employment exclusive excuse expert frequently golf hopefully "
                    "identify importance laid latter manufacturing mining object pattern performing personnel "
                    "perspective pregnant premier promote revolution rooms severe sleeping suppose tool tournament "
                    "victim victims agents arrest attend brilliant chose concert crash declared deliver depth deputy "
                    "dirty doctors earned electronic error existence experiences expression factory headed interior "
                    "legislation maintenance manner nearby noise origin panel personality practices prepare relief "
                    "replaced retail roads shame ships somewhat staying stronger surely tip updated writers absolute "
                    "advertising agencies baseball bathroom cable calm championship checked dates degrees democrats "
                    "doors driven dumb exciting expansion heavily incident linked manage messages multi politicians "
                    "print quit refused reporting sight significantly sing weapon wet widely worldwide ages anniversary "
                    "attractive bike cake causing closely constantly contest deaths depends drawn fees hardly height "
                    "hidden invited letting loud motor officially portion pounds princess protein puts raw reform "
                    "regions represented respond retirement sample seats secondary solar somehow stayed suffering tries "
                    "ultimate unknown wondering attached attacked automatically balls bills blind breath brief chest "
                    "conduct debut decade destroy differences engaged experts expressed external fantasy grab immediate "
                    "license paint pilot pink principal recognize recognized registered regularly representatives "
                    "seasons smoking steam suffered survive tall therapy witness adopted aim campus chances childhood "
                    "clinical clubs comedy comparison covers defeat democracy detailed entitled exact exposed fee "
                    "injured kinds lets loans lock musical nose objects opposed organized plastic protected purposes "
                    "recording semi statements suspect swear techniques tie trend valuable wealth yards approval aspects "
                    "attempts burning champions contain convention dancing eggs engineer equivalent facing fairly "
                    "fingers founded functions gang graduate lift marked memories monthly mountains neighborhood operate "
                    "outstanding permission porn racing recommended regulations reply republicans rid scientists "
                    "shoulder shower sons stations tradition visited visual wheel achieved admitted appointment authors "
                    "barely celebrate challenges chocolate contemporary criticism effectively extensive faced filed "
                    "fought gained gallery historic improvement initially jury monster obtained philosophy pride "
                    "promised repeat returning smell sought studied suggests surrounding topic vast visitors wanting "
                    "auto consistent continuing earn exists finger guitar heading ignore involving meal meanwhile "
                    "naturally necessarily offices pants partnership payments percentage pocket practical primarily "
                    "proved rape regardless relative represents rescue resulting sessions soccer stable structures "
                    "supplies symptoms temporary tested trick attended bone bullshit chamber circuit clothing "
                    "complicated confused consequences defend divided everyday extent fishing format gotten household "
                    "immigration impressive joining killer lesson limits managers membership mirror nights occur parking ",
                    "proposal province purchased recognition reputation shortly situations strongly tears technique thin "
                    "tied accused adventure argue assessment atmosphere awful bedroom belief breaks carefully cats "
                    "choices closing colors contrast courses courts element elsewhere establish extension founder gear "
                    "hitting increases infrastructure locations machines offensive package pointed poverty processes "
                    "qualified reaching ridiculous sensitive server shock silence supporters thick threw tons transition "
                    "violent voters acid actress alongside anxiety babies bars bonus compare contained cooking covering "
                    "curious directors discovery discussed encourage enforcement featuring finals flash formal formula "
                    "governments horses hungry informed losses math minds mistakes mystery networks passes penalty "
                    "phones photography producing protest publication rating refer respectively scheduled select silent "
                    "spoken successfully suffer tracks uncle unusual arrival asks assault awareness badly captured "
                    "components concrete deeply expectations explanation exposure featured fiction guarantee happiness "
                    "hearts horrible ideal injuries legend mood picking pleased producer pushing rank replacement "
                    "retired roles savings settled shadow singles tag tape thread visiting wage wings bags beating "
                    "believes blocks boring checking clock commissioner commitment confident containing copies crimes "
                    "custom denied desk drinks ear electricity episodes grounds gym helpful horror label locked naked "
                    "opens output persons pitch pizza pushed raising rear reveal romantic scores sisters speaks stages "
                    "strategic swimming welfare winners wire worker afterwards alright architecture assist attempted "
                    "behalf belt capture centers ceremony comic cops cuts designer disappointed dressed economics "
                    "efficient electrical employed enjoying entering essentially expecting explains ghost guests handed "
                    "hockey hunting languages maps nervous newly ordinary participate prayer principles racist rarely "
                    "references sexy skill soil solve stomach struck studying suck supports trash ugly virus whoever "
                    "amounts aspect banned boost comfort controls crack deck demands dies dragon dramatic dust engineers "
                    "evolution foods hired illness inspiration institution knife lately lowest minority mum opinions "
                    "patterns presents promotion rail readers remote repair steal stolen titles trans vol whereas "
                    "abandoned acquired actors annoying bid buried butter cares confirm contracts convinced crap decent "
                    "decline delay describes downtown elite enemies forgotten gods hire hop hopes insane installed "
                    "landing managing marry nowhere nurse obtain organic ownership participants poetry pot pray printed "
                    "recall sake signing smooth spiritual stops sudden throwing thrown vacation abroad assigned "
                    "associate assumed bench bother broadcast bye citizen cleaning compete consists consumers "
                    "contributed critics damaged disaster discover entrance equally fallen figured fitness friendship "
                    "handling idiot intense lawyers lifetime liquid makeup medal mortgage narrative narrow observed "
                    "occasionally physics posting potentially reduction reflect refuse researchers resource sciences "
                    "serves silly subsequent translation visible yep adds amendment arrive belong channels commonly "
                    "connect defensive designs efficiency experiment females findings firms gifts hence increasingly "
                    "incredibly journalist kicked lessons lists maintained occasion passenger races regulation resident "
                    "rocks sixth spin styles subjects sucks suitable thirty valid vital whilst alleged bands collect cop "
                    "currency emotions exhibition fraud funeral genuine immigrants improving instructions introduce "
                    "lands legacy merely prisoners programming publishing ratio regret rejected remind resort resulted "
                    "reverse routine scary seed spell survival sword waves achievement argued asleep automatic begun "
                    "cents comprehensive consent daddy destruction diseases divorce ears engage extraordinary fate "
                    "frequency gaming glory headquarters heritage initiative interviews juice landscape logic meets "
                    "objective privacy profits reducing regard representing roughly salary scoring script searching "
                    "sections strip surrounded threatened transferred tube universities writes ambassador apps awarded "
                    "banking breast cant chemistry concluded consumption corruption crossed discount dozen engines epic "
                    "exception exit expand fancy gorgeous grateful heroes holes impression inches indicate input leather "
                    "lips luxury lyrics manufacturers movements operated ought outcome painted poll preferred pulling "
                    "ranked referring removal reporter risks screaming sequence stretch tear tennis terrorist ties "
                    "twelve versions voices wishes absence agricultural asshole ate athletes bears blues boxes bull "
                    "cameras contribute contribution contributions couples deny deserves ease extend fame flood "
                    "generated genetic glasses impressed indicated instant investors involves kills ministers monitoring "
                    "occurs passengers photographs principle producers progressive punishment rally rapidly reader "
                    "representation restaurants reveals roots samples shops sum swing texts upcoming veterans alert "
                    "arena arguments boom boots brave claiming column commit compensation composition computers "
                    "conservation constitutional crossing defending density difficulty dropping drops elementary ethnic "
                    "expenses fundamental genius greatly guidance hospitals infection intention jokes knee mechanical "
                    "participation periods precious pregnancy premium preparing pretend prominent proven radical "
                    "remembered requested residential reward rings shake spots stats struggling substantial teen "
                    "temperatures transmission trap uniform wildlife wooden ads aggressive answered apparent bang blast "
                    "bones brands centuries complaint component connections courage cure desperate diversity duties "
                    "encouraged faculty feedback fighter frozen guards hiding humanity innovation instruments invest "
                    "jacket legislative listing manual mothers murdered nursing occupied ongoing operator painful "
                    "preparation purple railroad registration releases romance submitted sufficient survived suspended "
                    "tissue trailer trends trials underground versus virtual walks wounded announcement arranged arsenal "
                    "attending attracted biological bite blocked boards burned categories checks chip concerning "
                    "database define discrimination disorder distributed districts documentary domain dynamic edited "
                    "engagement explore fewer footage giants grave implementation investigate jazz laboratory literary "
                    "mask midnight mouse packed piano praise presentation psychology relation restrictions rocket ruin "
                    "secrets slave stability steady stones symbol terminal toilet treaty triple unlikely updates viewed "
                    "affair collective conversations cooperation darkness deeper enable equity estimates failing "
                    "finishing goodbye hardware hurts intellectual invite involvement nuts partly petition phrase "
                    "physically protecting racial regime rounds ruled sauce separated shield similarly slide stem "
                    "talented throat tiger touched toy visits warriors wisdom accounting alien attacking awkward beast "
                    "beef celebration celebrity certificate cited coaching colleagues constructed dated default derived "
                    "dialogue disabled distinct drag educated eligible estimate execution existed fifty followers fool "
                    "framework franchise funded furniture generations guaranteed integrated intelligent interaction jet "
                    "journalists lifestyle lighting loop mall overseas performances recommendations recover regarded "
                    "relax reliable rely remarkable responses ruling sacrifice sole stopping strategies succeed tables "
                    "tale targets timing volunteers witnesses wore worship acted alarm bloody breathing butt "
                    "characteristics collaboration consideration counts creates crucial daughters dependent discussions "
                    "drives equipped expanded experimental feeding filter globe grades highlights hoped intent involve "
                    "judgment mature odds peaceful photographer prevention printing promoting publicly pump repeated "
                    "replied revenge satisfied seeds signals spaces spare specialist stranger submit surprising tap "
                    "threats tourism volunteer acceptable allies attempting auction challenging chaos churches composed "
                    "concentration copper counting credits dispute earnings editing executed firing fits frequent "
                    "gathered hilarious huh ignored improvements investments margin mechanism moderate opera overcome "
                    "parallel psychological publications quest radiation shocked sized stroke stunning tanks topics "
                    "trains traveling treating tune utility vessel weed wherever acquisition addressed anime announce "
                    "backed bold borders breathe choosing classical classified clip coaches coins concepts conspiracy "
                    "controversy convince disappeared encounter equality exam examination fails fiscal guardian homeless "
                    "instrument intervention lover mainstream menu mounted mutual nope occasions offense oral panic "
                    "peoples pursue refugees removing requirement responded ruined scope segment spectrum stays terror "
                    "virtually waited worn yea accompanied aimed approaches arguing arrangement beliefs boats boundaries "
                    "colleges considerable conventional designated emperor employers enormous errors focusing forgive "
                    "gains gathering guidelines handled hosted inquiry jumped lion loaded lonely maintaining measured "
                    "mercy nevertheless newspapers outer oxygen pissed poem powered promises racism ratings reads "
                    "recovered refers screw seventh shelter sooner strikes suggesting suits toys tracking tribute "
                    "trigger vary venue wages wheels abortion accuracy applying artificial belongs beneath bullet "
                    "celebrated consistently conversion copyright counties deposit destination dirt diverse emails "
                    "exclusively export fastest formerly functional gather grandfather habit indicates isolated jealous "
                    "knocked landed laughed lazy mama modified naval oldest picks popularity professionals pussy "
                    "reactions relate robot sacred securities shoe speakers spy suggestions supplied suspension "
                    "terrorism toxic tunnel unions upgrade warrant wider wound actively applies arrangements asset "
                    "assuming backing blessed brush burden carries casual certified chef civilian coalition cock "
                    "complain complaints controversial describing differently directions discipline discussing "
                    "disgusting dominant earning essay expense explaining furthermore graphic hiring hosts implemented "
                    "instantly invasion jumping laptop legendary opponents outdoor photograph quarters rangers ranks "
                    "reception recipe regulatory reviewed rubber secured serial settings shed sponsored stealing strict "
                    "subsequently substance suggestion syndrome tasks trips ultra unexpected usage worlds accidentally "
                    "affordable amateur appeals bearing beats biology briefly canal cheaper climb competing completion "
                    "custody delete demonstrated departure developers developments dig eagles employer explosion fever "
                    "fluid folk generate handsome hotels imagination integration integrity interpretation leaf "
                    "legitimate lightning loads longest magical motivation nasty outfit pension permit plates portrait "
                    "productive reminds reserves safely shirts shorter slight streaming targeted tension theories "
                    "touching transactions twist ugh unemployment useless viewers winds woke abilities advocate aims "
                    "backup beaten bitter blown branches campaigns chips clever closest collections continuous converted "
                    "correctly creatures criteria declined difficulties disability dish duck evaluation excess farming "
                    "fence fighters flights forcing forming gradually gravity habits highlight identical investigations "
                    "legally lied listened males manufacturer meters nail negotiations nonsense operational owns "
                    "playoffs quoted relating repeatedly rolled scientist sink skip slavery sorts stole swim transaction "
                    "transformation veteran vulnerable wealthy additionally attract blowing bored bug caring catching "
                    "cheating chronic cleared communicate convicted cultures dealt delayed demonstrate departments "
                    "depend developer diagnosis dismissed distinguished dose eighth experiments flesh flip forty "
                    "generous hated implement influenced kidding laser loyal marijuana mentally missions occupation "
                    "opponent paintings patch pic pointing pollution precisely prisoner privilege proposals protests "
                    "punk radar relatives resist solely stepped striking terrorists tourist transit trucks trusted "
                    "vessels volumes websites wireless wondered wrap yoga adopt albums anytime bacteria beings beside "
                    "blade boot bottles bulk cargo coin colored commentary confusion cried customs dealer deemed distant "
                    "electronics emerging emotion emphasis ethics excitement exploration fights filling filming graphics "
                    "humor invested meals nerve nightmare operators overnight partially participating platforms poster "
                    "practically preserve produces qualify raid ranging ranking receives respective restricted routes "
                    "scenario situated slaves spotted spreading sustainable sustained taxi themes threatening tobacco "
                    "trace trapped uncomfortable wasted weakness widespread accepting accessible acknowledge advised "
                    "advisory animation assignment balanced bare basement bases battles bias bits cancel carpet ceiling "
                    "chill classification clue codes collapse collecting compound conscious consecutive contents costume "
                    "deleted devoted displayed dominated endless escaped examine floating garbage grid grows heating "
                    "identification knees lap lions liver mixture nominated parliamentary patent perception physician "
                    "proceed proceedings pupils reserved restore rifle rival runner sadly shoulders significance sits "
                    "sizes slept soap spray stored stressed structural suite tropical unnecessary verse vintage warned "
                    "acres adapted adoption anonymous approaching artistic attendance aviation barrel beds beloved bless "
                    "boxing celebrating charging chemicals cinema comics compliance contrary controlling corporations "
                    "couch crush decrease defeated diabetes dressing expanding fears fires genre gentle illustrated "
                    "invented jam layers lease licensed loyalty magnetic monsters mysterious notion partial piss placing "
                    "propaganda rat reflection reminded resolve revolutionary scandal shine simultaneously substitute "
                    "surveillance tactics testimony treasure trophy tweet underlying unfair acceptance accidents affects "
                    "annually apologize appreciated approached arriving aunt bubble buyers casino charts clouds "
                    "connecting counsel creature deadly decides desired determination embrace emerged flew gentleman "
                    "hammer hosting icon imposed indigenous infinite installation inter interactions introducing kicking "
                    "laying legislature liability makers moreover perceived pics planes politician preliminary premiere "
                    "presidency reaches react realistic remarks retain satisfaction scratch sheets shy sometime spirits "
                    "sporting strictly teens thou tier vocal warrior worries yield accomplished admission adventures aka "
                    "appearing barrier believing blacks bombs burst caps casting cattle classroom compromise convenient "
                    "criminals crop earthquake elderly eliminate embarrassing finest grants hates incidents inform "
                    "lovers mathematics medication minded podcast portfolio productivity promoted protocol quietly "
                    "replacing responsibilities salad scholarship screening sends smiling soup stake stating strain "
                    "suspected tackle tigers timeline torture traded translated tricks twins vegetables vertical "
                    "violation wallet workshop wrapped aboard abstract accent addiction associates awake binding blank "
                    "conservatives contacts conviction corrupt curve depressed deserved dining disorders duration "
                    "encouraging farms fifteen flows graduated grandmother hurry immune inflation ingredients install "
                    "instruction intensity inventory investigated invitation judicial justify lecture libraries logical "
                    "meaningful migration missile motivated muscles nurses organ patrol pepper pig pile plug provision "
                    "releasing requiring revised scream stairs staring statistical sticks strangers succeeded sweat "
                    "switched tattoo teenage tragedy trauma wrestling zoo accordance acquire activist activists "
                    "addresses alike applicable availability breach cabin circles closet combine companion comparing "
                    "consciousness consultant controller corresponding courtesy damages demanding dishes dozens eaten "
                    "engaging fascinating financing fitted flexible gaining gentlemen goodness guilt helicopter homework "
                    "households iconic infected keen lesser liberals lip mandatory manufactured mechanics miracle mud ",
                    "observation operates owe permitted phenomenon playoff precise profession protective providers "
                    "publisher reportedly retreat rookie seeks sentences separation sexually ski skilled surgeon theft "
                    "understands valve washing adjacent agreements appreciation athletic authorized banner blew blocking "
                    "charm chasing climbing complaining cookies cruel curriculum deadline demanded dive divide electoral "
                    "entity excessive exercises feminist governing heal interface jewelry journalism jungle linear "
                    "occasional oriented pilots prayers predicted pressed preventing provisions pursuit rap reflected "
                    "reminder restored scholars sealed sounded streams strongest tends tribe unfortunate variable "
                    "worrying zones ace adjusted alternate arrives artwork athlete attraction babe bankruptcy "
                    "capabilities cared chains cognitive competitors convert cooked cups deciding defender dental "
                    "diplomatic divisions drum editorial enabled entertaining establishing eternal freeze generic "
                    "grandma grip handful happily hmm hurting hybrid intentions investing keyboard lasting locally loses "
                    "mild minimal mixing molecular nearest neighbor noon nowadays openly pairs pathetic poems "
                    "possibilities potato preference sont comme etre faire etait ete peut suis avoir encore avait temps "
                    "peu rien dire fois faut toujours voir jamais deja soit trop gens oui vraiment beaucoup quelques "
                    "sera parce doit fin mieux ceux jours histoire sais annees fais prendre veut peux politique reste "
                    "veux celui personnes seul vais vrai droit coup mettre serait pense quelque tant avais maintenant "
                    "tete etaient aurait avons enfants parler assez besoin savoir trouve passe pouvoir raison ailleurs "
                    "avez devant jeu soir souvent effet partir surtout choses dernier nombre parle seulement ensemble "
                    "femmes passer vois loi parti pris peuvent trouver autant fut gouvernement niveau pourrait quatre "
                    "seule accord ligne probleme vient crois exemple jeune mis systeme facon heures loin derniere idee "
                    "nuit vont etais argent vue etes aura dessus meilleur forme plutot avaient tard ayant matin mise "
                    "saison sait afin aide choix retour sommes vite yeux journee devrait prend etant cinq maniere semble "
                    "truc agit parfois compris gauche livre musique jeunes longtemps permet voit chambre deuxieme "
                    "importe lorsque peur plein serie ancien anglais grands jouer membres merde parmi peine tellement "
                    "epoque allez doute lire seront voix ecrit autour developpement faites hier nombreux propre rendre "
                    "vivre droite environ mec mesure ouais abord affaire envie gars laquelle nouveaux pouvez siecle "
                    "sortir terme viens voiture droits frere jeux venir difficile existe laisse milieu produit propos "
                    "doivent feu mots periode armee confiance croire laisser manque petits produits rencontre vas aurais "
                    "classe commence comprendre demain demander liste bout espere interet mouvement origine presente "
                    "regarde sortie auteur langue meilleure perdu politiques qualite risque sinon travaux appelle cadre "
                    "changer grandes libre premiers semaines connais facile pire rester sauf troisieme aider coupe "
                    "faisait partout plupart problemes seconde valeur etudes chacun devient donnees hors lequel montre "
                    "particulier penser reseau texte veulent celles controle dois succes activite carte chercher connu "
                    "devenir euros filles fonction putain simplement annonce arreter bientot differents espace mariage "
                    "organisation sein economie entree pied presse prise realite sorte utiliser acces differentes depart "
                    "mauvais puisque regarder soient taux travailler acheter attendre finalement fini manger moyens "
                    "perdre pratique verite cherche communaute idees nombreuses payer peuple publie quant recu repondre "
                    "suivre aupres entendu faisant gagner honneur matiere permis rapidement retrouver soiree arriver "
                    "arrivee devoir moitie particulierement pourra prochaine pret rend voyage eviter arrete bord combien "
                    "contraire courant derniers devait entendre faits faute fera finale heureux joueur necessaire "
                    "pouvait serieux terrain villes comprends creer desormais frais livres majorite modele obtenir outre "
                    "puisse rappelle tandis actuellement bande comprend faux groupes morts suffit tot voulu bons "
                    "critique dehors dernieres devenu directement discours dur explique forte fete informations joueurs "
                    "petites premieres preuve propose apprendre attaque avance endroit essayer exactement huit meilleurs "
                    "medias retrouve activites arriere bonnes commencer completement faite habitants montrer physique "
                    "quartier rendez territoire vente vis appele changement connaitre cul europeenne lieux parole pieds "
                    "prochain regles vaut analyse assurer carriere commun couleur disent duree erreur espece faible "
                    "importante mauvaise membre probablement secteur soutien tomber utilise vacances allait concours fou "
                    "gestion mesures metres poids reussi sert sociaux suivant taille viennent visite voulait voulez "
                    "absolument bref ciel continuer dirait fallait memoire ouvert revenir tenir tient trucs bah crise "
                    "expliquer garder generalement historique militaire mondiale medecin permettre prends soi tiens "
                    "valeurs vouloir ecoute animaux chanson pose raisons seigneur sorti traitement visage veritable "
                    "ecrire anciens avion calme enquete envoyer ligue limite parfait seuls vitesse etranger americains "
                    "cheveux hiver lutte ouvrir pression puissance volonte vraie eleves adore drole debat oublier pauvre "
                    "propres represente revient reve souhaite touche travaille utilisation voulais etudiants "
                    "connaissance coups dela habitude poser tombe totalement emission auraient beaux chaine classique "
                    "comptes concerne divers effets envers facilement fil freres futur gaz lignes malade nouvel pareil "
                    "prive proces rapide reseaux tendance vingt vit atteint bebe choisi fasse finir gagne gout humain "
                    "lettres oublie parlement peau tels uniquement vendre venu victime equipes evidemment aimerais "
                    "appris arret avocat bonheur caractere certaine choisir compter connait consommation cru dedans "
                    "decouvrir lendemain magnifique populaire rentrer ressources revenu secretaire vent acte bouche "
                    "capacite centrale chaud contexte dessous disait exterieur faudra gueule hauteur naissance ouvre "
                    "processus recevoir recherches rencontrer rire realise secondes servir victimes vin actuel approche "
                    "croit europeen faveur merite noms papier portes presenter prevu regardez societes dites entrer "
                    "envoye journaliste nul perte possibilite riche seraient serais termes tuer unite etablissement "
                    "anniversaire belles cafe chiffres couleurs faudrait identite interessant largement mets moindre "
                    "parcours parfaitement partage partis proteger prefere remettre responsabilite scientifique sujets "
                    "tirer titres voire eleve appeler attends beaute buts cuisine foi interets milliards mourir objets "
                    "participer personnage planete retard savent sexe situe tenu theorie verre actuelle artiste artistes "
                    "avions certainement citoyens clairement conscience croissance celebre davantage entier exposition "
                    "maman matchs moderne pensee privee propriete quitter quotidien risques soldats triste vide ame "
                    "echange elu enorme evenements actes atteindre bizarre categorie communes conseils contenu dommage "
                    "decouverte entierement espoir exploitation fonctions malheureusement offrir prennent religieux "
                    "signifie spectacle allons appareil auront disant decouvert empecher entretien furent immediatement "
                    "incroyable liens militaires methode naturel revue regle sympa traite univers venez ecran acteurs "
                    "augmentation auteurs bruit classement conseiller entend formes franchement heros mecs personnages "
                    "pourraient raconte rappeler rejoindre ressemble regulierement textes voitures ecouter etape ajouter "
                    "avantage bataille boite certes dangereux danse defendre exercice francaises locaux logique milliers "
                    "monter ouverte palais passant principalement proches reprendre revoir realiser recemment savez "
                    "sentiment soin statistiques surement tente vivant etrangers baisse besoins candidats construire "
                    "domicile employes examen humaine moteur officiel ouvrage partager patron permettant possede "
                    "pourrais principaux preparer rentre revenus reel reforme souvenir tenter utile achat cerveau "
                    "commentaires disposition deputes envoie fortement humains independance justement lancer materiel "
                    "parait parles quatrieme quitte reduire repond sentir sourire tourner agir alcool arme autrement "
                    "battre cartes cout disponible fleurs indique jardin mondial oblige pensez soins sois seance vecu "
                    "accepter alle ameliorer attendant championnat chasse complet complexe declaration developper essaie "
                    "forcement gerer heureusement inscrit joie kilometres lesquels metier obtenu occupe prises profiter "
                    "revanche sauver savais scolaire totale troupes trouvent evenement accueil aimer aurai autorites "
                    "biens bordel cadeau chefs colere conduit diverses dormir especes ferait foutre hasard importants "
                    "longueur morte musulmans numerique necessaires permettent professionnel secours sommet systemes "
                    "electrique achete acteur ajoute arabe augmenter blague devraient difficultes depenses efficace "
                    "essaye europeens fiche histoires interdit journalistes juifs paroles pensais perd pouvons pratiques "
                    "repas soutenir suivante termine vetements echelle aimes appartient aeroport chier chiffre constitue "
                    "contrairement defaut estime glace hommage humanite interesse meilleures piste policiers pouvoirs "
                    "produire prets restent strategie telles terres tourne voyez elus boire chomage cle comportement "
                    "congres critiques demeure importantes langues mandat medecine noirs nourriture parlent phenomene "
                    "portant portee precedent prete rencontres reprise representant repondu tenue amoureux bateau bilan "
                    "blancs considere devenue disparu donnent douleur deces democratie emplois faim formule lait ordres "
                    "peinture printemps programmes retourner rues scientifiques vehicule allemands apporter arrivent "
                    "balle bibliotheque climat conflit difficulte essentiel extremement frontiere gratuit genie honte "
                    "locale menace mettent mouvements occuper pauvres presidentielle quantite rares reconnu reellement "
                    "usine aient ambiance aventure blesse contient dessin declare defaite empeche erreurs finit fous "
                    "huile inquiete litterature medecins penses rapports relativement robe realisation salaire toucher "
                    "traduction tranquille trente verifier ecriture epreuve auparavant chaleur commentaire concurrence "
                    "conduite discuter doigts fruits haine heureuse humour interne lesquelles legende magasin organise "
                    "parfaite pensent plait postes rang recours reprises reflexion sentiments souvenirs travailleurs "
                    "veille verra vive aiment apparemment arbre attendu auquel compose conception ennemi euh fonctionne "
                    "fonctionnement impot individus inutile investissement maintenir modeles naturelle pourront "
                    "professionnels profil proposer remplacer respecter savait serai souviens serieusement unites uns "
                    "electricite apparait avoue centaines changements circulation conduire correspond coucher devrais "
                    "entends entraineur essai financement infos jugement limites marcher monnaie murs neige nucleaire "
                    "pluie pote promis reconnaissance riches rythme velo echec amitie associations construit consequence "
                    "dents devons dirais enceinte laissez langage legerement mange perso philosophie pouvant principales "
                    "professionnelle quasi remise russes recuperer reduit salaries semblent situee tourisme traiter "
                    "venait vendu vrais apprend archives arretez chansons couche couverture dispose domaines deteste "
                    "empereur fier grosses methodes plantes policier precis reglement sacre signes socialiste soyez "
                    "speciale tas technologie tort tournee trou voies voter vehicules etudiant affiche apporte arbres "
                    "casse centres collectif complique condamne consiste couper effectivement essentiellement eut "
                    "exprimer fermer hausse journaux ministres officielle organisme passent passee personnalite "
                    "princesse prouver puissant quartiers representants responsables retirer retourne recoit reflechir "
                    "suffisamment tire vaste voudrais equilibre accepte attente cercle cesse chiens connue courte dette "
                    "doigt douce douze democratique exemples fenetre fournir frappe greve imaginer jusque meurtre "
                    "melange normalement ombre personnelle plat repris servi surpris etablir actualite agreable avancer "
                    "avantages bete catholique chaussures chretiens cible devra donnee decider decrit endroits genial "
                    "installer jambes locales marches nulle partenaire particuliers plainte portable prenez preuves "
                    "rouges reussir reves sors superbe tournoi ventes viande vierge etrange balles batterie chacune "
                    "commis contente convaincre copine devez diffusion differente dirigeants entiere essence impose "
                    "imposer normale necessite officiellement ouvrages proprietaires reconnaitre regler simples "
                    "specialiste vivent evident acquis actif allais automne bloc chocolat collegues confirme extrait "
                    "fixe foule immense immeuble inverse longues maladies ordinateur outils pouvais principes "
                    "precisement peche rarement recettes rentree souci soumis sejour taxe transfert vigueur voient voile "
                    "allant aurez banques billets calcul caracteristiques chevaux choc cles connaissent disque decret "
                    "desir determiner enlever entrainement fabrication garcons lourd mener micro morale niveaux "
                    "obligation officier poche prenant reelle solide tentative egard emissions agricole appliquer "
                    "attaquer autorisation courir debats dejeuner delai efficacite ennemis feuilles individu mettant "
                    "offert organiser partenaires pause portent propositions quart remis ridicule secondaire seules "
                    "soucis senat trafic utilisateurs venant violences echanges electeurs etudier affirme appui attaques "
                    "auras billet biere batiments cacher competences coupable culturel ceremonie debout disais drogue "
                    "decouvre detruire enfer europeennes frontieres gardien geste juger majeur meurt participe possibles "
                    "poursuivre pousse raconter releve regne recit salles sommeil traces tache etabli accueillir associe "
                    "automobile charte cinquieme circonstances communautes connaissances conserver deviennent donnant "
                    "definitivement demarche gouverneur gratuitement incendie journees juges juridique luxe "
                    "manifestation manifestations multiples noter parlant particuliere positif procureur petrole repose "
                    "siecles suivantes utilises vague abus aise anciennes appareils citoyen compagnies convient degre "
                    "disponibles dite drapeau detruit essais franc fuite gamme invites lunettes obligatoire paradis "
                    "potentiel pousser repos reprend semblait sondage trouvait tresor electronique alerte baiser "
                    "chambres comparaison conneries connus coute dossiers douche explication falloir fausse "
                    "fonctionnaires fortes fetes inscription libres levres montrent offres oiseaux oreilles "
                    "organisations outil parisien parlait poissons profite profondeur precedente publicite publiee "
                    "quinze rejoint religieuse representent resoudre reussite souris tonnes traduit temoins utilisee "
                    "echapper adultes alimentation apparition arrondissement artistique ballon chercheurs combats "
                    "controler couronne curieux dispositif decouvrez excellente facteurs fameux feront financier habite "
                    "historiques humaines ingenieur laissant lancement larmes liquide lutter legere matieres mauvaises "
                    "medicaments olympique pensees prevenir puissent pur remplir rupture sombre suivants suivent "
                    "supporter supprimer territoires tetes voyons amelioration augmente bleus bouger bouteille "
                    "considerer cou criteres croyez destine directe disons dizaines developpe engager favorable feuille "
                    "folle forts horreur jure liees logements magasins montage morceau morceaux navire piscine poursuit "
                    "pouvaient presidence rappel rapporte restera reviens saisons symbole troubles trouvez utilisant "
                    "ventre voler ecrivain abri accompagne accuse adopte appels bureaux cellules combattre commencent "
                    "couvert culte culturelle difficiles disparition depot faisaient industriel internationales lis "
                    "lache leve medaille menage odeur peuples pitie plante poesie probable ramener refus rendent "
                    "richesse revele sachant sensible souffle soutient totalite vise editions actrice aides aimez "
                    "attendent blesses classiques croient creme degres demandent dispositions fins folie garantie "
                    "honnete hypothese industrielle independant licence manquer naturellement normes parallele "
                    "profondement provinces precision pretre regrette realisee salaires sexuelle sportif specialement "
                    "specifique toit ecart actifs adulte anglaise applique assis battu chemins chronique cons copie ",
                    "dessins devaient diplome enregistrement envoi espaces esperer existent extraordinaire faibles "
                    "financiers fonde foyer humeur inconnu jouent litteraire longs magique maintien malades marrant "
                    "maitrise mecanique pars paru permettra populaires publier pute remarquer remonter renseignements "
                    "restauration sel supplementaires serieuse terminer toile temoignage vainqueur vues enormement "
                    "etoile abandonne adaptation adjoint agissait appelee consacre convaincu concu drame durable "
                    "dynamique deplacement fallu fermeture fondateur galerie indice interdiction necessairement "
                    "organismes organisee paie papiers physiques pierres pis plaque prives profond qualites refaire "
                    "realisateur redaction secteurs seuil signer supplementaire vins visiter voyages voyant abandonner "
                    "annoncer attirer bassin boutique carrement casser chapeau creee diversite dizaine dechets depasse "
                    "fout hyper intermediaire metiers oncle parlementaire personnellement poete preciser rends refugies "
                    "travaillent utilisent visant voulons elevee equipement acceder aille amoureuse apprentissage "
                    "attentat bombe cadeaux cadres capables charme commerciale comporte copain couts dimension dingue "
                    "defi effectuer facteur gagnant gouvernements identifier justifier lecon limiter liee majeure marre "
                    "mystere mechant nationales oppose orientation originale origines paquet paysage pertes pistes "
                    "pleurer potes prisonniers prouve quasiment recette reconnait remonte representer retrait soldat "
                    "visiteurs etions amende amener amenagement americaines atteinte autonomie autoroute campagnes "
                    "capacites chanter chanteur commandes communique constater destinee diable dirige dise disparaitre "
                    "departements faisons ferais francs fusion juif lecteur marins materiaux milieux militants modernes "
                    "motif naturelles obligations obtient oubliez ouvriers pauvrete pratiquement provenant prevue "
                    "refuser remplacement remporte retenir repartition sache scandale taper terrains terrorisme "
                    "terroristes titulaire verts vivement equipage etrangere etres absolue adversaire associes canadiens "
                    "cense citer composee comprennent conflits consequent contenant correspondant documentaire "
                    "enregistre fac faculte faisais fiscale fondee formations guerres generaux indispensable "
                    "interessante investir joyeux magie modifier medicale nettement nuits ordinaire paire personnels "
                    "pleins poursuite prit promesse registre rempli serez servent sortant souffrir soutenu suivez "
                    "suppression tournage transformer trompe voyageurs ecrits etapes accords alimentaire annuel "
                    "apparence autrefois benefice chaines communiste connaissez couteau dieux doucement derange ferai "
                    "fleuve gras inscrire internationaux intervenir integrer lecteurs liaison migrants miroir officiers "
                    "paraitre partenariat plateforme procede proprietes prefet presentent renforcer respecte referendum "
                    "sensation syndicats vend victoires voudrait ecrite electriques etonnant africains agricoles "
                    "amateurs apprecier attendais avocats bouge chargee cherchent choisis collegue constate couvre "
                    "croyais dommages decu enjeux enseigne esprits feux frapper gardes gloire implique industriels "
                    "inferieur jus limitee marquer mettez moto oiseau olympiques onze parisienne parlez plastique "
                    "privees priere producteurs profonde pseudo recrutement religieuses remplace reponds reunis salope "
                    "sauter sieges sortent souffre tableaux toilettes unies universites volant ere editeur ambassadeur "
                    "autorise carbone cellule chantier civils comparer connard conseillers conservateur constituent "
                    "descendre devoirs dira duo effectue entraine exclusivement familiale flics federale gamin hesite "
                    "inspire intitule negociations parisiens passait pensait prochaines precieux quelconque rayon saisir "
                    "scolaires semblant similaire subir terroriste tester temoignages universitaire vertu ecris "
                    "equipements evoque accompagner adopter apprecie assister aveugle baisser beurre biais blessure "
                    "boucle bulletin candidature catalogue chant consommateurs devais diriger diner essayez excuses "
                    "favoriser financer flotte forcer froide fromage guere incapable installations investissements issus "
                    "jette laissent masque massif mentionne moteurs musulman navires nomination observer ose partisans "
                    "peintre personnelles presentee racisme remercie recente repete savons sortes souffrance tarifs "
                    "tempete voulaient epreuves acier adapte amene appartenant appellent automatique cardinal chante "
                    "chaude communiquer consulter definir enseignants exemplaires fantastique fenetres geant "
                    "independante islamique libertes legitime mensonge personnalites producteur precedemment periodes "
                    "reportage restes seras solaire sorties subi tapis traites trouves viol accueille adapter affronter "
                    "africain africaine ailes arbitre attendait bandes banlieue blessures bleue bouffe benefices camion "
                    "ceinture charbon chats competence conferences conseille consideree couvrir donnez dons decede "
                    "deplacer deposer entrainer environs exprime flux frequence lumieres legislatives legumes maitres "
                    "mineurs motifs nationalite nourrir prendra progressivement prevoit relais relever ronde seins "
                    "succession tiennent vaisseau visiblement volontaire economies eglises etudie admis analyses "
                    "ateliers avouer basee bateaux bijoux britanniques beneficier casque chaise commandement concerts "
                    "conducteur confirmer couilles dames devenus dimensions douceur debuts decennies emplacement "
                    "executif forment fourni foutu francophone haha ingenieurs inscrits liberaux livrer mariee menaces "
                    "noires pacifique possibilites publies quarante quete restant riz recompense reformes scrutin "
                    "specialistes suivie tabac tendances terminee tissu trone uniforme variete energetique ethique "
                    "annuelle attendez attentats basket carton catholiques chauffeur chimie cinquante continuent "
                    "correctement diminuer distinction degats depasser excellence exceptionnel exercer fautes fideles "
                    "forets habitudes habituellement lentement litteralement legale municipalite meres navigation norme "
                    "observe opportunite partant parvenir passagers passera photographe plaindre poitrine poudre "
                    "professeurs pretexte qualifie raisonnable rassemblement recueil remarquable remercier "
                    "respectivement ressort retrouvent retrouvez rois reagir stupide symptomes tenant tombee touristes "
                    "traverser tromper utilisateur utilisees electorale acide adversaires agression allaient appelait "
                    "appeles attache aussitot auxquels belges caracteres chretienne cites conformement createur diffuse "
                    "donnera debit defenseur effectif entourage explications expedition faciliter fatigue forfait "
                    "formidable fraude fumer gratuite inferieure italiens lecons listes marier mentir menee merveilleux "
                    "mince misere naturels negatif ouverts permanente portait programmation promesses provoquer recul "
                    "regardant remet responsabilites restait reveille sacs satisfait secondaires serveur signaler "
                    "sixieme suffisant survie traditionnelle trait tribune veuillez vieilles eventuellement abandon "
                    "abonnes accorde ambition appuie aventures blanches bouton catastrophe cherchez clan collectivites "
                    "compromis comprehension conclusions diametre denonce exister facons maternelle modification numeros "
                    "offrent poil prenne prie precedentes precedents presentes quebecoise rage remporter repeter "
                    "souverainete seances universel utiles utilite vocation voisine egal accorder achats actuelles "
                    "admettre affirmer apparaitre assumer banc bosser chanteuse chimiques civilisation colonne "
                    "commerciaux constamment constante craint crier criminel danser devront domination declarations "
                    "existait expose fesses genres graphique gateau independants infrastructures intellectuelle joindre "
                    "liberer manifestants marquee mien modifications optique ouvertes photographie plafond pomme porno "
                    "proceder progression proportion presentant recommande rendus rever talents trajet trouvera ultime "
                    "veste viendra vingtaine vols ecologique eliminer epaule evoluer actuels agriculteurs appuyer baise "
                    "bancaire centrales climatique colle commerciales complexes conclure conserve contemporain "
                    "dictionnaire decor dedie delegation demission denoncer enleve entoure externe faiblesse fierte "
                    "fournit globale informer modes musicale ordonnance parlais permanence pilotes pommes ponts poule "
                    "pourrez promouvoir propagande protege preserver puissante quotidienne raciste retenu retire roue "
                    "reguliere socialistes sondages soudain strategique tirage visites vole voyait epargne abonnement "
                    "accusation aile aimait ampleur apparaissent arrivera attire blonde bloque boule calmer camarades "
                    "choisit clientele conclu conquete consultation curiosite celibataire descente distingue duquel "
                    "depit employeur exces failli fausses fuir fumee feminine genoux guitare habitation hopitaux "
                    "impressionnant interdire internes lacher manifeste mathematiques monuments moyennes mecanisme "
                    "parfum passages phrases polemique prevoir qualification rendement reduite republicains "
                    "revolutionnaire serre sexuel soigner soirees specifiques stabilite statue suggere traitements "
                    "usines vapeur vendeur epee aie architecte bol centaine combinaison concentrer connaissais contacter "
                    "couches cris culturelles diagnostic dirai doutes decouvertes decrire demontrer esclaves fabrique "
                    "familial fermee futurs gestes grandi introduit invente issu jambe jete lourde manquent ment mesurer "
                    "monument musiciens placee prenait prestation profs promet prononce provisoire prevenu psychologique "
                    "purement realises reveiller sauve similaires smartphone souligne spectacles speciales surveiller "
                    "separer trous unes urbaine volontaires vraies evaluer alimentaires altitude amuser anonyme "
                    "arrestation arretes assise attentes automatiquement bloquer cardiaque citation confort connaissait "
                    "contraintes cuir destines dirigeant depense depose encourager exige exigences fonctionner gosses "
                    "herbe intervient interesser lancee malheureux mobiles modifie monstre pattes pleinement pompiers "
                    "productions racines romantique senti sexuelles soupe specialise survivre symbolique tournant "
                    "traditionnel vagues vivants acquerir apercu autonome barrage brillant beton canard cathedrale "
                    "changent cirque compagnon comportements comprenant contribuer contribue copains dignite diminution "
                    "delit designe encontre enquetes esthetique extraits francophones garantir identique indiens "
                    "individuelle levee libere louer mentale meubles mienne montrant mediatique mepris panne panneaux "
                    "philosophe prestations prononcer provoque ramene relatif religions rendue reproche reproduction "
                    "rigueur regionales reparer republicain satisfaire sauvages sensibles sols sonne specialisee "
                    "speciaux superieurs tenait tendre troupe universelle etend agences annonces asile assaut attaquant "
                    "avancee chimique communs condamnation connexion connues consacree courants crainte demandant "
                    "dramatique declarer delire demonstration deroule espagnols espoirs esperons exterieure facilite "
                    "finis foie ignorer imaginaire intensite integre majeste musulmane memoires observations panique "
                    "passees physiquement pique plats proposent protocole rangs renseignement restee roule rouler "
                    "reussit soixante susceptibles surete tentatives touristique traditions vendus vengeance vif vus "
                    "edifice etablie etablit evite absolu afficher aimais appart bouteilles bebes canape chiant comptait "
                    "consideres constituer constructeur dangereuse demandez devint defauts deficit democrate entrees "
                    "exceptionnelle exercices fauteuil fragile gravite geographique helas illustre immeubles inclus "
                    "interventions largeur latine miel mobilisation moche modeste musiques negocier operateur organes "
                    "panneau parlementaires partent passez piege plaire plaques pratiquer prisonnier prevues psychologie "
                    "quiconque rappele reconnue regardes remboursement romains rumeurs repression reveil saura saut "
                    "septieme souhaitent spectateurs sportive supporte tante temporaire transforme trimestre taches "
                    "valable vents elevage epaules etendue etonne amies aerienne beneficie censure cesser chasseurs "
                    "chaussee chemise chirurgie circonscription colis communautaire connerie contenus correspondance "
                    "coupes dettes donnait emploie entrepreneur enveloppe explorer exploser fixer fiere fievre grille "
                    "horaires imaginez indiquer informe insulte interprete librairie loyer mobilite municipales noblesse "
                    "nuages occupee officiels parlons peint pleure porc portail pourcentage pourras poeme prealable "
                    "presidents puissants puits quoique reglementation resister revolte sacree sortis souvient sportifs "
                    "strictement seche taire tranquillement valoir elever enormes aliments assassinat assiste biologique "
                    "blagues boites brevet caisses concernes console constitutionnel contemporaine continu continuez "
                    "contrainte croyait difficilement debarrasser delegue emporter entretenir envisager esclavage "
                    "exemplaire expres fameuse frequemment harcelement inconnue indices initiale insectes maires "
                    "manieres massacre matinee maitresse mineur musees nerveux opposer pantalon parcs peines perds "
                    "planche plume poignee popularite portefeuille posee pouce primaires priori provenance refuge "
                    "reposer representations recentes repondent revision schema sentent sincere sincerement soyons "
                    "superieures separe tirs traits tranche trio tristesse violente voulant echo etages etudiante adieu "
                    "alentours amuse analyser antenne apprends artillerie bal ble brise charger commerces completer "
                    "conscient consentement constat critiquer deviendra disposer douter degage deplacements effectifs "
                    "enseignant enthousiasme esperant examens gouts genereux geographie individuel insiste intime jaloux "
                    "joues kilos librement manteau messe mixte ordonne plomb poches possedent produite provient "
                    "rapproche rayons rebelles rendant renoncer retenue recoivent remuneration reunir saisi souverain "
                    "suisses surprenant susceptible tragedie transmettre transparence travaillant tribunaux trophee "
                    "universitaires energies etendre aborder accueilli accusations actualites administrateur auxquelles "
                    "ayez bords briser bulle clavier cloture coffre collecte colonisation complot conforme conservateurs "
                    "contraint controles corde criminels densite disques douleurs debile delais detention emporte "
                    "engagements enjeu faillite favorables gentille grecs genetique habitat hectares historien identifie "
                    "immediat inscriptions manches marchandises mensonges milliard mythe menages nettoyer negociation "
                    "payes perdent poussiere problematique qualifier quarts ranger rechercher recommandations reviennent "
                    "reparation reunit solidaire subit tensions tombent temoigne verras echantillon echappe egale "
                    "actionnaires adhesion aire antique arrets bruler betes caracteristique cheque cigarette colonies "
                    "considerablement craindre celebres demontre dependance entrepreneurs envoyes facade fermes fusil "
                    "gamins gouter grandeur grossesse incluant inutiles jouant lourds nucleaires nue occidentaux "
                    "occupent ouvrier oxygene pacte perspectives phenomenes poils prochains pretendre rapprocher "
                    "rembourser renouvellement repartir sensibilite signification situes sovietique traduire telephones "
                    "verres virer ecoutez absent adresser aimerai annuler apport arrivait asseoir aurons avere chouette "
                    "communistes comptent conclut confie considerant consommateur corriger courbe destinees dits dors "
                    "dort defini degager embrasser entente examiner expositions fidelite filiere fonctionnaire gaffe "
                    "globalement hebdomadaire hygiene initialement interesses invention inviter jaunes laicite lyonnais "
                    "manga manifester manquait maths militant monstres monetaire meritent nombres numeriques obliges "
                    "observatoire pareille partiellement parvient patrons perdus poussee proposee ressenti realisees "
                    "recolte regulier renovation repertoire reside semblable solides soumettre sympathique terrasse "
                    "terrestre tissus variations varie absurde adoptee alarme ancetres barriere chauffage chercheur "
                    "collaborateurs colonnes commissions condamnes confier consacrer correspondent courante criminelle "
                    "crises ceder cereales demandait descend distinguer decoration definitive engagee enregistrer exerce "
                    "flammes foire honnetement horaire huitieme hum injuste inquietude inventaire invisible jouait "
                    "juridiques linguistique magnifiques majeurs majoritairement mosquee medailles medicament nommer "
                    "offrant particules pavillon paysages pensant permettrait professionnelles preter qualifies refusent "
                    "recents reels regiment regimes sagesse soirs solde solitude stockage subventions surfaces suspendu ",
                    "testament tienne traditionnels transporter usages violents echanger ecrivains accompagnee accomplir "
                    "administratives assiette cahier calculer cannabis carburant certificat chroniques circulaire "
                    "confirmation construite continuite croise cuivre dessert diplomatique dirigee discret definit defis "
                    "effacer efficaces embrasse engages enseigner excuser exploiter filiale feter golfe generique "
                    "individuelles interdite interroger integralite irai lave nerfs nette neveu"
                };
                for (const char* part : frequent)
                {
                    const std::unordered_set<std::string> more = make_word_set(part);
                    words.insert(more.begin(), more.end());
                }
                return words;
            }();
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
            /* Acronyms of public bodies, matched when written in capitals: national, local and
               European administrations, social security bodies, regulators, courts and police
               forces, from France, the United Kingdom, the United States and many other countries. */
            static const std::unordered_set<std::string> s = make_word_set(
                "caf cpam cnam cnaf cnav carsat urssaf msa dgfip ddfip drfip insee cnil anssi ars ants ofii "
                "ofpra cnda crous cnous ansm ademe cnes cnrs inserm inrae ifremer ign onf dgccrf dgsi dgse dga "
                "sgdsn ugap amf acpr arcep arcom hceres onisep cned unedic apec ircantec cnracl aphp dreal dreets "
                "ddets ddpp dgac dgddi anah dila bnf inpi oniam fiva cnous ofb anct dinum "
                "nhs hmrc dvla dvsa dwp hmcts hmpps fcdo dhsc dfe fca cma ofsted ofcom ofgem ofwat cqc mhra ukvi "
                "irs ssa fbi cia nsa dhs uscis tsa fema fda doj hhs ftc fcc dmv nih nasa usda faa nhtsa "
                "ons nca fsa ico moj hmpo dbs ofqual ukhsa ehrc cbp atf osha anses ineris irsn cnsa agefiph "
                "fiphfp pjj spip dgpn dggn dgt dares dgefp "
                "bka bnd bsi bfv bafin bamf destatis dguv aok dak kfw dfg ptb thw bpol lka suva seco fedpol ahv "
                "onss inami riziv onem rva cpas ocmw onva forem actiris vdab inasti rsz uwv svb rdw cjib kvk sepe "
                "aeat inss tgss cnmv cnmc inps inail istat consob agcm agcom anac enac enav aima iefp gnr inem "
                "cnesst saaq ramq ircc cbsa asfc rcmp csis scrs esdc edsc spvm cnss cnops anapec dgsn onee cnas "
                "casnos anem cnrps steg sonelgaz ipres cnps senelec dgid fdic cfpb nlrb eeoc usps noaa "
                "usgs usaid hmlr gchq ncsc defra dcms dluhc oisc hiqa tusla accc apra ndis vcat ncat uidai epfo cbdt "
                "cbi rbi sebi irdai trai isro drdo upsc anvisa ibge bacen cvm detran imss issste inegi condusef "
                "profeco eba esma eiopa ecb bce eib bei echa efsa edpb edps enisa euipo eurostat eppo cedefop inao "
                "anru sgami gign ocrtis oclctic sdis samu afpa hadopi ird cirad inrap dgos dgcs dgesco dgesip dsden "
                "drees cnamts cea ccas cias");
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
                "european court of human rights|court of justice of the european union|"
                "bundesagentur fur arbeit|agentur fur arbeit|deutsche rentenversicherung|bundeskriminalamt|"
                "bundesamt fur migration und fluchtlinge|bundesministerium der finanzen|finanzamt|bundestag|"
                "bundesrat|bundesverfassungsgericht|bundespolizei|bundeszentralamt fur steuern|agencia tributaria|"
                "seguridad social|tesoreria general de la seguridad social|guardia civil|policia nacional|"
                "servicio publico de empleo estatal|agencia espanola de proteccion de datos|agenzia delle entrate|"
                "guardia di finanza|carabinieri|polizia di stato|agenzia delle dogane|"
                "garante per la protezione dei dati personali|seguranca social|autoridade tributaria|"
                "policia judiciaria|instituto do emprego e formacao profissional|service public federal finances|"
                "office national de l'emploi|office national de securite sociale|"
                "institut national d'assurance maladie-invalidite|autorite de protection des donnees|"
                "administration federale des contributions|secretariat d'etat a l'economie|"
                "office federal de la police|belastingdienst|dienst uitvoering onderwijs|sociale verzekeringsbank|"
                "kamer van koophandel|autoriteit persoonsgegevens|agence du revenu du canada|revenu quebec|"
                "immigration, refugies et citoyennete canada|gendarmerie royale du canada|"
                "royal canadian mounted police|surete du quebec|regie de l'assurance maladie du quebec|"
                "societe de l'assurance automobile du quebec|caisse nationale de securite sociale|"
                "direction generale des impots|caisse nationale des assurances sociales|"
                "office national de l'electricite et de l'eau potable|us citizenship and immigration services|"
                "federal emergency management agency|centers for disease control and prevention|"
                "securities and exchange commission|federal reserve|department of defense|department of education|"
                "department of health and human services|department of housing and urban development|"
                "department of transportation|department of energy|department of the treasury|"
                "united states postal service|national institutes of health|food and drug administration|"
                "drug enforcement administration|national security agency|central intelligence agency|"
                "federal aviation administration|consumer financial protection bureau|"
                "equal employment opportunity commission|her majesty's revenue and customs|"
                "hm courts & tribunals service|foreign, commonwealth & development office|"
                "metropolitan police service|police scotland|department for environment, food & rural affairs|"
                "office of the public guardian|child maintenance service|pension service|"
                "department of social protection|road safety authority|intreo|centrelink|services australia|"
                "australian federal police|department of home affairs|unique identification authority of india|"
                "reserve bank of india|employees' provident fund organisation|central board of direct taxes|"
                "income tax department|receita federal|instituto nacional do seguro social|policia federal|"
                "servicio de administracion tributaria|instituto mexicano del seguro social|"
                "european data protection board|european data protection supervisor|european banking authority|"
                "european securities and markets authority|european medicines agency|"
                "european union agency for cybersecurity|european ombudsman|european court of auditors|"
                "cour des comptes europeenne|european anti-fraud office|office europeen de lutte antifraude|"
                "european public prosecutor's office|parquet europeen|"
                "caisse nationale de solidarite pour l'autonomie|agence nationale pour la renovation urbaine|"
                "institut national de l'origine et de la qualite|service d'aide medicale urgente|"
                "service departemental d'incendie et de secours|europol|eurojust|frontex|"
                "european investment bank|banque europeenne d'investissement|"
                "commissariat a l'energie atomique|commissariat a l'energie atomique et aux energies alternatives|"
                "ifp energies nouvelles|inria|onera|centre communal d'action sociale|"
                "centre intercommunal d'action sociale");
            return s;
        }

        inline const std::unordered_set<std::string>& known_organizations()
        {
            /* Companies, banks, insurers, media, carriers, universities, sports clubs and
               charities known by their own name, folded, from all regions of the world. Names
               that are also common words, given or family names, places, products or services
               (an application, a payment card, a car model) stay out, and so do the
               intergovernmental organisations of the United Nations system: the rules and the
               learned recognizers read those from their context. */
            static const std::unordered_set<std::string> s = make_phrase_set(
                "a1 telekom austria|ab inbev|abanca|abb|abbott laboratories|abbvie|abc news|abn amro|absa|ac milan|"
                "accenture|access bank|acciona|accor|accorhotels|acer|achmea|actes sud|action contre la faim|"
                "activision blizzard|adani group|adecco|adidas|admiral group|adnoc|advanced micro devices|adyen|"
                "aecom|aegean airlines|aegon|aena|aer lingus|aeroflot|aerolineas argentinas|aeromexico|"
                "aeroports de paris|afc ajax|afm-telethon|afnor|ag2r|ag2r la mondiale|ageas|agence france-presse|"
                "agricultural bank of china|ahold delhaize|aia group|aib group|aig|air algerie|air austral|"
                "air canada|air caraibes|air china|air cote d'ivoire|air france|air france-klm|air india|air liquide|"
                "air madagascar|air mauritius|air new zealand|air senegal|air tahiti nui|air transat|airbus|"
                "airbus group|airtel|airtel africa|aisin|ajax amsterdam|akamai|aker solutions|akkodis|akzonobel|"
                "al arabiya|al jazeera|al rajhi bank|alaska airlines|albert heijn|alcoa|alcon|aldi|aldi nord|"
                "aldi sud|alfa laval|alfa romeo|alfa-bank|alibaba|alibaba group|alimentation couche-tard|"
                "all nippon airways|allen & overy|alliance francaise|allianz|allianz trade|allied irish banks|"
                "allstate|alpha bank|alphabet inc|alrosa|alstom|alten|altice|altran|amadeus it group|"
                "amaury sport organisation|ambev|america movil|american airlines|american express company|"
                "american red cross|american university of beirut|amgen|amnesty international|amtrak|amundi|"
                "anheuser-busch|anheuser-busch inbev|ansys|ant group|anticor|anz|apache software foundation|apave|"
                "aperam|apf france handicap|apicil|apollo global management|apollo hospitals|applied materials|"
                "apprentis d'auteuil|aptiv|aramark|aramco|arcadis|arcelik|arcelormittal|arcor|ardian|areva|"
                "arianegroup|arianespace|arkema|arm holdings|armee du salut|arriva|arsenal fc|artelia|as monaco|"
                "as roma|as saint-etienne|asahi group|asahi shimbun|asda|aselsan|asiana airlines|asics|asky airlines|"
                "asm clermont auvergne|asml|asos|associated press|assystem|astellas|aston villa|astra international|"
                "astrazeneca|asus|at&t|athletic bilbao|atkinsrealis|atlas copco|atlassian|atletico de madrid|"
                "atletico madrid|atos|atradius|attac|attijariwafa bank|auchan|audencia|audi|aurobindo pharma|"
                "australian national university|austrian airlines|autodesk|autoliv|avianca|avic|aviva|avtovaz|axa|"
                "axa banque|axa investment managers|axel springer|axiata|axios|axis bank|ayala corporation|"
                "azul linhas aereas|b&b hotels|b&q|back market|bae systems|baidu|bain & company|bajaj auto|"
                "baker hughes|baker mckenzie|baker tilly|banca monte dei paschi di siena|banco bpm|banco do brasil|"
                "banco sabadell|banco santander|bancolombia|bandai namco|bank hapoalim|bank leumi|bank of africa|"
                "bank of america|bank of china|bank of communications|bank of ireland|bank of montreal|bank pekao|"
                "bankinter|banorte|banque nationale du canada|banque populaire|banque postale|baosteel|barclays|"
                "barrick gold|basf|bausch & lomb|baxter international|bayard presse|bayer leverkusen|bayern munchen|"
                "bayern munich|bbc|bbva|bdo|bechtel|becton dickinson|beiersdorf|belfius|bell canada|benfica|"
                "bentley motors|berkshire hathaway|bertelsmann|bertrandt|besiktas|bfm tv|bfmtv|bharti airtel|"
                "bilfinger|bill & melinda gates foundation|biogaran|biogen|biomerieux|biontech|blackrock|blackstone|"
                "blue origin|bmo|bmw|bmw group|bnp paribas|bnp paribas fortis|boca juniors|bocconi|boeing|"
                "bollore logistics|booking holdings|booz allen hamilton|borgwarner|borussia dortmund|bosch rexroth|"
                "boston celtics|boston consulting group|boston red sox|boston scientific|bourse de paris|boursobank|"
                "boursorama|bouygues construction|bouygues group|bouygues immobilier|bouygues telecom|"
                "boy scouts of america|bpce|bpost|bradesco|braskem|bredin prat|brembo|brico depot|bricomarche|"
                "bridgestone|bristol myers squibb|bristol-myers squibb|british airways|british council|british gas|"
                "british red cross|british telecom|brittany ferries|britvic|broadcom|brussels airlines|bt group|"
                "buffalo grill|bumrungrad|bunq|bupa|burberry|bureau veritas|burger king|buzzfeed|byd|bytedance|"
                "cadence design systems|cairo university|caisse d'epargne|caisse de depot et placement du quebec|"
                "caixa geral de depositos|caixabank|caltech|cambridge university|cambridge university press|cameco|"
                "canada life|canada post|canadian tire|capcom|capgemini|capgemini invent|capital one|care france|"
                "care international|cargolux|caritas|carlyle group|carnegie mellon university|carnival corporation|"
                "carrier global|casino guichard-perrachon|castorama|cathay pacific|catl|cbs news|cd projekt|"
                "cdiscount|celtic fc|cemex|cencosud|center parcs|centralesupelec|centrica|cerba healthcare|cern|"
                "ceva logistics|cevital|cez|cez group|cff|cgi inc|channel 4|charles schwab|charoen pokphand|"
                "charter communications|check point software|check point software technologies|chelsea fc|"
                "chicago bulls|chicago tribune|china construction bank|china daily|china eastern airlines|"
                "china life insurance|china southern airlines|chronopost|chugai|chunghwa telecom|cibc|cigna|cih bank|"
                "cimade|cimb|cipla|cirque du soleil|cisco|citibank|citigroup|clariane|clarin|cleveland clinic|"
                "clifford chance|cloudflare|club med|cma cgm|cme group|cms francis lefebvre|cnbc|cnews|cnn|cnooc|"
                "cnp assurances|coca-cola company|codelco|coface|cogeco|coles group|colgate-palmolive|colis prive|"
                "collins aerospace|coloplast|colruyt|columbia university|comac|comcast|"
                "comite international olympique|commerzbank|commonwealth bank|compagnie des alpes|compal|"
                "compass group|conad|concentrix|conde nast|conforama|confucius institute|conocophillips|"
                "consolidated edison|constellation brands|constellium|contemporary amperex technology|contentsquare|"
                "continental ag|convatec|copa airlines|cornell university|correos|corriere della sera|"
                "corsair international|corsica ferries|cosco shipping|costco|coty|couche-tard|courrier international|"
                "covea|covestro|credit agricole|credit agricole cib|credit lyonnais|credit mutuel|"
                "credit mutuel arkea|credit suisse|creditsafe|criteo|croissant-rouge|croix-rouge|"
                "croix-rouge francaise|crowdstrike|csl limited|ctv|curevac|currys|cvs health|dachser|daf trucks|"
                "dai-ichi life|daiichi sankyo|daikin|daily mail|daily telegraph|daimler truck|dallas cowboys|dalloz|"
                "danfoss|dangote group|danone|danske bank|dassault aviation|dassault group|dassault systemes|"
                "databricks|dataiku|db schenker|dbs bank|de beers|de lijn|de telegraaf|de volkskrant|deepmind|"
                "deere & company|dekra|delft university of technology|delhaize|dell technologies|deloitte|"
                "delta air lines|delta airlines|denner|denso|dentsply sirona|dentsu|der spiegel|deutsche bahn|"
                "deutsche bank|deutsche borse|deutsche post|deutsche telekom|deutsche welle|devialet|dfds|dhl|diageo|"
                "die welt|die zeit|disneyland paris|dji|dla piper|dnv|doctors without borders|dominion energy|"
                "domino's pizza|domusvi|dongfeng|dow jones|dp world|dr reddy's laboratories|dsm-firmenich|dsv|ducati|"
                "duke energy|duke university|dupont de nemours|dz bank|easyjet|eaton corporation|ecobank|"
                "ecole des ponts paristech|ecole normale superieure|ecole polytechnique|ecopetrol|edeka|edf|"
                "edf group|edf renouvelables|edhec|editis|edp|edwards lifesciences|egis|egon zehnder|egyptair|"
                "eiffage|eisai|ekwateur|el corte ingles|el pais|el universal|elbit systems|electrabel|electrolux|"
                "electronic arts|electronic frontier foundation|elengy|elevance health|eli lilly|elior|elsan|"
                "elsevier|em lyon business school|emaar|emaar properties|embraer|emeis|emerson electric|"
                "emirates airline|emirates group|emirates nbd|emlyon|emmaus|enbw|endesa|enedis|enel|engie|"
                "engie electrabel|enterprise rent-a-car|epfl|epic games|epson|equifax|equinor|eramet|ergo group|"
                "ernst & young|erste bank|erste group|esade|escp|escp business school|eset|eskom|essec|esselunga|"
                "essilor|essilorluxottica|estee lauder|eth zurich|ethio telecom|ethiopian airlines|etihad|"
                "etihad airways|etisalat|euler hermes|eurazeo|euro disney|eurobank|eurofins|euronext|europacorp|"
                "europcar|europe 1|eurostar|eurowings|eutelsat|evergreen marine|everton fc|eviden|evonik|evri|exelon|"
                "expedia group|experian|expleo|extinction rebellion|exxon|exxonmobil|falabella|familles rurales|"
                "familymart|fanuc|fast retailing|faurecia|fc barcelona|fc bayern munchen|fc nantes|fc porto|fdj|"
                "fdj united|federation francaise de football|federation francaise de rugby|fedex|femsa|fenerbahce|"
                "ferring|ferrovial|ferrovie dello stato|festo|feyenoord|fff|fidal|fidelity investments|fifa|"
                "financial times|finnair|first abu dhabi bank|first quantum minerals|firstgroup|firstrand|fiserv|"
                "fitch ratings|flamengo|flixbus|flunch|fluor corporation|fnac|fnac darty|fondation abbe pierre|"
                "fondation de france|fondation pour la recherche medicale|ford foundation|ford motor company|"
                "forrester research|fortinet|fortis healthcare|fortum|fortuneo|forvia|forvis mazars|fox news|foxconn|"
                "framatome|france 24|france alzheimer|france inter|france medias monde|france televisions|"
                "france terre d'asile|franceinfo|frankfurter allgemeine|frankfurter allgemeine zeitung|franprix|"
                "fraport|frasers group|fraunhofer|fraunhofer-gesellschaft|freeport-mcmoran|freie universitat berlin|"
                "french bee|fresenius|fresenius medical care|freshfields|freshfields bruckhaus deringer|"
                "friends of the earth|fudan university|fujitsu|g4s|galatasaray|galeries lafayette|galp|gameloft|"
                "garanti bbva|garmin|garuda indonesia|gates foundation|gaumont|gavi|gazprom|ge aerospace|"
                "ge healthcare|ge vernova|gedeon richter|geely|gefco|geico|general dynamics|general electric|"
                "general mills|general motors|generali|genting group|geodis|georgetown university|gerdau|gfk|"
                "gide loyrette nouel|gilead sciences|girondins de bordeaux|givaudan|glaxosmithkline|glencore|"
                "globe telecom|gmf|go-ahead group|godrej group|goethe-institut|gol linhas aereas|"
                "golden state warriors|goldman sachs|goodyear|google|grab holdings|grant thornton|grdf|"
                "great wall motor|great-west lifeco|gree electric|greenpeace|grenoble ecole de management|grifols|"
                "groupama|groupe adp|groupe bel|groupe bpce|groupe casino|groupe la poste|groupe ocp|groupe psa|"
                "groupe renault|grtgaz|grundfos|grupo aval|grupo bimbo|grupo mexico|grupo planeta|grupo televisa|gsk|"
                "gustave roussy|h&m|habitat for humanity|hachette livre|haier|haleon|halliburton|handelsbanken|"
                "handelsblatt|handicap international|hankook|hannover re|hapag-lloyd|harley-davidson|"
                "harmonie mutuelle|harpercollins|harris interactive|harrods|harvard|harvard medical school|"
                "harvard university|hasbro|havas|hays plc|hca healthcare|hcl technologies|hcltech|hdfc bank|"
                "hearst communications|hebrew university of jerusalem|hec paris|heidelberg materials|"
                "heidelberg university|heidrick & struggles|hennes & mauritz|hensoldt|hewlett packard|"
                "hewlett packard enterprise|hewlett-packard|hilton worldwide|hindustan aeronautics|hindustan times|"
                "hindustan unilever|hino motors|hiscox|hisense|hitachi|hitachi rail|hochtief|hoffmann-la roche|"
                "holcim|holland & barrett|hon hai|honda motor|honeywell|hornbach|hp inc|hsbc|huawei|hubspot|"
                "huffington post|huffpost|human rights watch|humana|humanity & inclusion|"
                "humboldt-universitat zu berlin|hyatt hotels|hydro one|hydro-quebec|hyundai|hyundai motor|iberdrola|"
                "ibm|icbc|icici bank|ie business school|iese|ifop|ihi corporation|iit bombay|iit delhi|ikea|"
                "illinois tool works|imd|imerys|imperial college|imperial college london|indian institute of science|"
                "indian institute of technology|inditex|inetum|infineon|infosys|ing group|ingenico|ingerop|insa lyon|"
                "insead|institut curie|institut francais|institut pasteur|instituto cervantes|intact financial|"
                "inter milan|intercontinental hotels group|intermarche|international airlines group|"
                "international olympic committee|international rescue committee|internazionale|interpublic group|"
                "intersport|intertek|intesa sanpaolo|intuit|intuitive surgical|ipsos|isbank|islamic relief|"
                "israel aerospace industries|isuzu|itau|itau unibanco|itv|iveco|jacobs engineering|jaguar land rover|"
                "japan airlines|japan post|jbs|jcb|jd sports|jetblue|jio platforms|john lewis partnership|"
                "john wiley & sons|johns hopkins hospital|johns hopkins university|johnson & johnson|"
                "johnson controls|jollibee foods|jp morgan|jpmorgan|jpmorgan chase|jsw steel|julius baer|jumia|"
                "juventus|kaiser permanente|kaist|kantar|kao corporation|karolinska institutet|kaufland|"
                "kawasaki heavy industries|kazatomprom|kbc|kddi|kedge|kellanova|kellogg's|kelly services|"
                "kenya airways|keolis|kering|keurig dr pepper|keyence|kfc|kghm|kia|kimberly-clark|"
                "king's college london|kirin holdings|kkr|kla corporation|klarna|klesia|klm|knds|knorr-bremse|"
                "koc holding|komatsu|konami|kongsberg gruppen|korean air|korian|korn ferry|kotak mahindra bank|kpmg|"
                "kpn|kraft heinz|kroger|kt corporation|ktm|ku leuven|kuaishou|kuka|kuwait petroleum corporation|"
                "kyocera|kyodo news|kyoto university|l'equipe|l'oreal|l3harris|la banque postale|la cimade|"
                "la depeche du midi|la francaise des jeux|la libre belgique|la ligue contre le cancer|la poste|"
                "la poste suisse|la quadrature du net|la repubblica|la stampa|la vanguardia|la voix du nord|lactalis|"
                "lam research|lancia|land rover|lanxess|larsen & toubro|latam airlines|latam airlines group|"
                "latham & watkins|laurentian bank|lci|lcl|le canard enchaine|le dauphine libere|le figaro|"
                "le monde diplomatique|le nouvel obs|le telegramme|lear corporation|lefebvre dalloz|legal & general|"
                "lego group|leiden university|lenovo|lenovo group|leo pharma|leroy merlin|les amis de la terre|"
                "les echos|les inrockuptibles|les petits freres des pauvres|lexisnexis|lg electronics|liberty global|"
                "liberty mutual|lidl|ligue contre le cancer|ligue des droits de l'homme|lindt & sprungli|linklaters|"
                "linux foundation|lions club|lions clubs international|liverpool fc|lloyd's|lloyd's of london|"
                "lloyd's register|lloyds|lloyds bank|lloyds banking group|lmu munich|lna sante|loblaw|"
                "loblaw companies|lockheed martin|logitech|lombard odier|london business school|"
                "london school of economics|london stock exchange|lonza|los angeles lakers|los angeles times|losc|"
                "losc lille|lot polish airlines|loterie nationale|loto-quebec|lotte group|louvre hotels group|"
                "ltimindtree|lucid motors|ludwig-maximilians-universitat|lufthansa|lufthansa group|lukoil|lululemon|"
                "luminus|lundbeck|luxair|luxottica|lvmh|m6 group|maaf|macif|macmillan publishers|macquarie group|"
                "macy's|maersk|magazine luiza|magna international|magyar telekom|mahindra & mahindra|mahindra group|"
                "mahle|maif|maisons du monde|majorel|malakoff humanis|malaysia airlines|man truck|manchester city|"
                "manchester united|manpowergroup|manulife|mapfre|marimekko|marks & spencer|maroc telecom|"
                "marriott international|maruti suzuki|maserati|massachusetts institute of technology|mastercard inc|"
                "matmut|mattel|max planck society|max-planck-gesellschaft|maybank|mayo clinic|mazars|mazda|mbda|"
                "mcdonald's|mcgill university|mckinsey & company|medecins du monde|medecins sans frontieres|"
                "mediapart|mediaset|mediatek|mediclinic|mediobanca|mediterranean shipping company|medtronic|megafon|"
                "meituan|mercadona|mercedes-benz|mercedes-benz group|merck & co|mercy corps|meta platforms|metlife|"
                "metro ag|metro inc|mgen|michelin group|micron technology|microsoft|midea|migros|millennium bcp|"
                "mines paris|mines paristech|mirakl|mistral ai|mitsubishi|mitsubishi heavy industries|"
                "mitsubishi motors|mitsubishi ufj financial group|mizuho|mizuho financial group|mlb|moderna|"
                "moet hennessy|mol group|molson coors|mondadori|mondelez|mondial relay|moneygram|monoprix|"
                "monte dei paschi|monzo|moody's|morgan stanley|mouvement desjardins|movistar|mozilla|"
                "mozilla foundation|mphasis|mr bricolage|msf|msnbc|mtn|mtn group|mufg|munich re|murata manufacturing|"
                "mutua madrilena|mylan|n26|naftogaz|nanyang technological university|nasdaq|national australia bank|"
                "national bank of canada|national bank of greece|national express|national university of singapore|"
                "nationwide building society|natixis|natura &co|naturgy|natwest|natwest group|naval group|naver|"
                "navistar|nba|nbc news|ndtv|nec corporation|nedbank|neoma|neste|nestle|netcare|netease|network rail|"
                "neue zurcher zeitung|new england patriots|new york university|new york yankees|newcastle united|"
                "newmont|newsweek|nexans|nexi|nexter|nextera energy|nfl|nhk|nhl|nice-matin|nikkei|nikon|nintendo|"
                "nippon life|nippon steel|nissan|nissan motor|nmbs|nn group|nnpc|nocibe|nokia|nokian tyres|"
                "nomura holdings|nordea|nordex|nordstrom|norilsk nickel|norinco|nornickel|norsk hydro|"
                "northrop grumman|northwestern university|norwegian air shuttle|norwegian refugee council|novartis|"
                "novatek|novo banco|novo nordisk|novonor|noyb|npr|nrj|ntt|ntt data|ntt docomo|nubank|nucor|"
                "nuffield health|numericable|nvidia|nxp|nyse|nyu|nzz|o globo|ocbc bank|occidental petroleum|"
                "ocp group|octopus energy|odebrecht|office cherifien des phosphates|ogc nice|olympiacos|"
                "olympique de marseille|olympique lyonnais|omnicom|omv|oneplus|ontario power generation|ooredoo|opel|"
                "open society foundations|openai|opmobility|optus|orange business|orano|orkla|orlen|orpea|orsted|"
                "otp bank|otto group|ouest-france|ouigo|outokumpu|ovh|ovhcloud|ovo energy|oxfam|oxford university|"
                "oxford university press|p&o ferries|paccar|pacific gas and electric|page group|pagegroup|palantir|"
                "palo alto networks|panasonic|panathinaikos|paris match|paris saint-germain|parker hannifin|pathe|"
                "paypal holdings|pbs|pearson education|pegasus airlines|pegatron|peking university|pemex|"
                "penguin random house|pepsico|pernod ricard|pertamina|petits freres des pauvres|petrobras|petrochina|"
                "petrom|petronas|peugeot sa|pfizer|philippine airlines|piaggio|pick n pay|pictet|pierre & vacances|"
                "pinduoduo|ping an|ping an insurance|piraeus bank|pizza hut|pkn orlen|pko bank polski|pko bp|"
                "plan international|plastic omnium|pldt|pmu|pnc financial|politecnico di milano|porsche ag|"
                "porter airlines|posco|post luxembourg|poste italiane|postes canada|postfinance|postnl|postnord|"
                "power corporation|pratt & whitney|pricewaterhousecoopers|primark|princeton university|prisa|"
                "prisma media|privacy international|privatbank|pro btp|procter & gamble|progressive corporation|"
                "proximus|prysmian|psa group|psg|psv eindhoven|publicis|puig|pwc|python software foundation|pzu|"
                "qantas|qatar airways|qatarenergy|qnb|qonto|qualcomm|quanta computer|quebecor|rabobank|racing 92|"
                "radio france|radio-canada|rafael advanced defense systems|rai|raiffeisen|"
                "raiffeisen bank international|rakuten|ramsay health care|ramsay sante|randstad|rangers fc|ratp|"
                "ratp dev|raytheon|rb leipzig|rc lens|rc strasbourg|rc toulon|real madrid|realtek|reckitt|"
                "reckitt benckiser|red crescent|red cross|rede globo|reebok|regeneron|relais colis|"
                "reliance industries|reliance jio|relx|remy cointreau|renault group|renesas|renfe|"
                "reporters sans frontieres|reporters without borders|repsol|restaurant brands international|"
                "restos du coeur|reuters|revolut|rewe|rexel|rheinmetall|ricoh|rio tinto|riot games|river plate|"
                "rivian|rmc|roche holding|rockefeller foundation|rocket lab|rockwell automation|"
                "rogers communications|rolls-royce|rolls-royce holdings|rosatom|rosneft|rostec|rostelecom|"
                "rotary club|rotary international|rovio|royal air maroc|royal bank of canada|royal bank of scotland|"
                "royal caribbean|royal jordanian|royal mail|rtbf|rtl group|rtve|rtx corporation|rusal|rwe|ryanair|"
                "s&p global|saab|sabanci holding|sabic|safaricom|safran group|saic motor|sainsbury's|saint-gobain|"
                "saipem|salvation army|samsung|samsung electronics|samsung life|samusocial|san miguel corporation|"
                "sandoz|sandvik|sanofi|santander uk|santos fc|sap se|sapienza university of rome|saputo|"
                "sas scandinavian airlines|sasol|saudi aramco|saudi national bank|saudia|saur|save the children|"
                "savencia|sbb|sberbank|scaleway|scandinavian airlines|scania|schaeffler|schibsted|schiphol group|"
                "schlumberger|schneider electric|schroders|schwarz gruppe|sciences po|scor|scotiabank|scottish power|"
                "scouts et guides de france|screwfix|sea shepherd|seagate|secours catholique|"
                "secours islamique france|secours populaire|securitas|sega|segula technologies|seiko epson|"
                "selfridges|seoul national university|sephora|serco|setec|seven & i holdings|sevilla fc|sfr|"
                "shaw communications|shein|shiseido|shoprite holdings|showroomprive|siam cement group|sidaction|"
                "siemens energy|siemens gamesa|siemens healthineers|sigfox|sika|sime darby|simon & schuster|"
                "singapore airlines|singtel|sinopec|sinopharm|sinovac|sixt|sk hynix|sk telecom|skadden|skanska|skema|"
                "skf|skoda|skoda auto|sky news|sl benfica|smbc|smith & nephew|snc-lavalin|sncb|sncf|sncf reseau|"
                "sncf voyageurs|sobeys|societe generale|societe generale group|socotec|sodexo|sodiaal|softbank|"
                "softbank group|software ag|solvay|sompo|sonatel|sonatrach|sonepar|sonova|sony|sony music|"
                "sony pictures|sophos|sopra banking software|sopra steria|sorbonne|sorbonne universite|"
                "sos mediterranee|sos racisme|south african airways|south china morning post|southwest airlines|"
                "spacex|spie|spire healthcare|splunk|sporting cp|sports direct|springer nature|square enix|srf|"
                "ss lazio|ssab|ssc napoli|stade francais|stade rennais|stade toulousain|stadler rail|stallergenes|"
                "standard & poor's|standard bank|standard chartered|stanford university|starbucks|starling bank|"
                "state bank of india|state farm|state street corporation|statkraft|statoil|stelia aerospace|"
                "stellantis|stellantis nv|stena line|stib|stmicroelectronics|stora enso|storengy|strabag|straumann|"
                "stryker corporation|studiocanal|subaru|subsea 7|suddeutsche zeitung|suez group|"
                "sumitomo mitsui banking corporation|sumitomo rubber|sun life|sun life financial|sun pharma|"
                "sun pharmaceutical|suntory|superdrug|swatch group|swedbank|swiss international air lines|swiss life|"
                "swiss post|swiss re|swisscom|syensqo|symrise|synlab|synopsys|systeme u|systra|t-mobile|"
                "take-two interactive|takealot|talanx|talgo|talktalk|tap air portugal|tap portugal|"
                "target corporation|tata consultancy services|tata group|tata motors|tata steel|tcl technology|"
                "td bank|tdk corporation|teads|tech mahindra|techint|technion|technip energies|technipfmc|"
                "technische universitat munchen|teck resources|tele2|telecom italia|telecom paris|telefonica|telenet|"
                "telenor|teleperformance|telerama|televisa|telia|telkom indonesia|telmex|telstra|telus|temenos|"
                "tenaris|tencent|tenneco|terega|tesco|tesla inc|teva|texas instruments|tf1|thai airways|thales group|"
                "thalys|the coca-cola company|the economist|the football association|the guardian|the hindu|"
                "the new york times|the new yorker|the sydney morning herald|the telegraph|the wall street journal|"
                "the walt disney company|the washington post|thomson reuters|thyssenkrupp|time magazine|"
                "times of india|tnt express|tokio marine|toronto-dominion bank|toshiba|totalenergies|"
                "tottenham hotspur|toyota|toyota motor|trane technologies|transavia|transdev|transferwise|transneft|"
                "transparency international|transunion|trenitalia|tribune de geneve|trinity college dublin|"
                "tsinghua university|tsingtao brewery|tsmc|tu delft|tu munchen|tui group|tunisair|tunisie telecom|"
                "turk telekom|turkcell|turkish aerospace|turkish airlines|tuv rheinland|tuv sud|tv globo|twilio|"
                "uber technologies|ubisoft|ubs|uc berkeley|ucl|ucla|uefa|ufc-que choisir|ukrzaliznytsia|ulb|umicore|"
                "unaf|unam|unapei|under armour|unicredit|unilabs|unilever|uniper|unipol|uniqlo|united airlines|"
                "united parcel service|unitedhealth group|unitedhealthcare|universal music group|universal pictures|"
                "universidad autonoma de madrid|universidad complutense de madrid|universidad de buenos aires|"
                "universidad nacional autonoma de mexico|universidade de lisboa|universidade de sao paulo|"
                "universita bocconi|universita di bologna|universitat de barcelona|universitat heidelberg|"
                "universite catholique de louvain|universite cheikh anta diop|universite de geneve|"
                "universite de lausanne|universite de liege|universite de montreal|universite du quebec|"
                "universite laval|universite libre de bruxelles|universite mohammed v|universite paris cite|"
                "universite paris-saclay|universite psl|universiteit leiden|universiteit van amsterdam|"
                "university college london|university of amsterdam|university of auckland|"
                "university of british columbia|university of california|university of cambridge|"
                "university of cape town|university of chicago|university of copenhagen|university of edinburgh|"
                "university of geneva|university of helsinki|university of manchester|university of melbourne|"
                "university of michigan|university of oxford|university of pennsylvania|university of sydney|"
                "university of the witwatersrand|university of tokyo|university of toronto|uob|upenn|upm-kymmene|"
                "uppsala university|uqam|us bancorp|usa today|vale sa|valencia cf|valeo|vallourec|valve corporation|"
                "vanguard group|vattenfall|vaudoise assurances|veepee|veolia|veolia environnement|verizon|"
                "vertex pharmaceuticals|vestas|via rail|viatris|vice media|videotron|vietnam airlines|viettel|"
                "vinamilk|vinci airports|vinci autoroutes|vinci construction|vinci energies|vingroup|virgin atlantic|"
                "virgin australia|virgin media|virgin money|visa inc|vivalto sante|vivendi|vmware|vodacom|vodafone|"
                "vodafone idea|vodafoneziggo|voestalpine|voice of america|voith|volkswagen|volkswagen group|volotea|"
                "volvo|volvo cars|volvo group|vrt|vtb|vueling|wabtec|waitrose|walgreens|walgreens boots alliance|"
                "wall street journal|walmart|walmex|walt disney company|warner bros|warner music group|wartsila|"
                "wavestone|webhelp|weizmann institute of science|wellcome trust|wells fargo|wesfarmers|"
                "west ham united|western digital|western union|westinghouse electric|westjet|westpac|wharton school|"
                "white & case|wickes|wikimedia foundation|wipro|wistron|withings|wizz air|wolters kluwer|"
                "woolworths group|world rugby|world vision|world wide fund for nature|worldline|wpp|wsp global|"
                "wuxi apptec|wwf|xerox|xiaomi|xinhua|xpeng|xpo logistics|yale|yale university|yamaha motor|yandex|"
                "yara international|ymca|yomiuri shimbun|yougov|ypf|zalando|zdf|zendesk|zenith bank|"
                "zf friedrichshafen|zimmer biomet|zscaler|zte|zurcher kantonalbank|zurich insurance|"
                "zurich insurance group|zydus");
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
                "parquet greffe ville commune brigade "
                "ministerio ministero ministerium bundesministerium bundesamt bundesagentur agencia agenzia agentur "
                "prefettura questura ayuntamiento municipio prefeitura gemeinde landgericht amtsgericht tribunale");
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
