// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "strings.h"

#include <cstring>
#include <map>
#include <string>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include "../common/platform.h"

namespace pseudalis
{

    namespace
    {
        struct entry
        {
            const char* key;
            const char* text[5];   // en, fr, de, it, es
        };

        // The texts of the icon, of the commands of the Explorer and of the installer.  {name}
        // marks a value given at use.
        const entry entries[] = {
            { "tray_updates", { "Check for updates", "Rechercher des mises à jour", "Nach Updates suchen",
                "Cerca aggiornamenti", "Buscar actualizaciones" } },
            { "tray_install_update", { "Install version {v} and restart", "Installer la version {v} et redémarrer",
                "Version {v} installieren und neu starten", "Installa la versione {v} e riavvia",
                "Instalar la versión {v} y reiniciar" } },
            { "tray_restart", { "Restart Pseudalis", "Redémarrer Pseudalis", "Pseudalis neu starten",
                "Riavvia Pseudalis", "Reiniciar Pseudalis" } },
            { "tray_start_service", { "Start the service", "Démarrer le service", "Dienst starten",
                "Avvia il servizio", "Iniciar el servicio" } },
            { "tray_service", { "Service management", "Gestion du service", "Dienstverwaltung",
                "Gestione del servizio", "Gestión del servicio" } },
            { "tray_stop_service", { "Stop the service", "Arrêter le service", "Dienst beenden",
                "Arresta il servizio", "Detener el servicio" } },
            { "notify_service_stopping", { "The Pseudalis service stops. It stays stopped until the computer starts "
                "again, unless you start it from the menu of this icon.",
                "Le service Pseudalis s'arrête. Il reste arrêté jusqu'au prochain démarrage de l'ordinateur, sauf "
                "si vous le démarrez depuis le menu de cette icône.",
                "Der Pseudalis-Dienst wird beendet. Er bleibt bis zum nächsten Start des Computers beendet, sofern "
                "Sie ihn nicht über das Menü dieses Symbols starten.",
                "Il servizio Pseudalis si arresta. Resta arrestato fino al prossimo avvio del computer, a meno che "
                "non lo avviate dal menu di questa icona.",
                "El servicio Pseudalis se detiene. Permanece detenido hasta el próximo inicio del equipo, salvo que "
                "lo inicie desde el menú de este icono." } },
            { "notify_service_starting", { "The Pseudalis service starts.", "Le service Pseudalis démarre.",
                "Der Pseudalis-Dienst startet.", "Il servizio Pseudalis si avvia.",
                "El servicio Pseudalis se inicia." } },
            { "tray_docs", { "Interface documentation", "Documentation de l'interface",
                "Dokumentation der Schnittstelle", "Documentazione dell'interfaccia",
                "Documentación de la interfaz" } },
            { "tray_quit", { "Close the notification icon", "Fermer l'icône de notification",
                "Infobereich-Symbol schließen", "Chiudi l'icona di notifica", "Cerrar el icono de notificación" } },
            { "tip_off", { "Pseudalis: the service does not answer", "Pseudalis : le service ne répond pas",
                "Pseudalis: Der Dienst antwortet nicht", "Pseudalis: il servizio non risponde",
                "Pseudalis: el servicio no responde" } },
            { "tip_starting", { "Pseudalis: the service is starting", "Pseudalis : démarrage du service",
                "Pseudalis: Der Dienst startet", "Pseudalis: avvio del servizio",
                "Pseudalis: el servicio se está iniciando" } },
            { "tip_stopping", { "Pseudalis: the service is stopping", "Pseudalis : arrêt du service",
                "Pseudalis: Der Dienst wird beendet", "Pseudalis: arresto del servizio",
                "Pseudalis: el servicio se está deteniendo" } },
            { "tip_stopped", { "Pseudalis: the service is stopped", "Pseudalis : le service est arrêté",
                "Pseudalis: Der Dienst ist beendet", "Pseudalis: il servizio è arrestato",
                "Pseudalis: el servicio está detenido" } },
            { "notify_update_title", { "Update ready", "Mise à jour prête", "Update bereit", "Aggiornamento pronto",
                "Actualización lista" } },
            { "notify_update_text", { "Pseudalis {v} is downloaded. It will be installed at the next restart.",
                "Pseudalis {v} est téléchargé. Il sera installé au prochain redémarrage.",
                "Pseudalis {v} ist heruntergeladen. Es wird beim nächsten Neustart installiert.",
                "Pseudalis {v} è stato scaricato. Verrà installato al prossimo riavvio.",
                "Pseudalis {v} se ha descargado. Se instalará en el próximo reinicio." } },
            { "notify_uptodate", { "Pseudalis is up to date.", "Pseudalis est à jour.",
                "Pseudalis ist auf dem neuesten Stand.", "Pseudalis è aggiornato.", "Pseudalis está actualizado." } },
            { "notify_update_error", { "The update check failed: {e}", "La recherche de mise à jour a échoué : {e}",
                "Die Update-Suche ist fehlgeschlagen: {e}", "La ricerca di aggiornamenti non è riuscita: {e}",
                "La búsqueda de actualizaciones falló: {e}" } },
            { "notify_update_failed_title", { "Update failed", "Échec de la mise à jour",
                "Aktualisierung fehlgeschlagen", "Aggiornamento non riuscito", "Actualización fallida" } },
            { "notify_update_rolled_back", { "The update to version {v} failed and Pseudalis {o} was put back in "
                "place: {e}",
                "La mise à jour vers la version {v} a échoué et Pseudalis {o} a été remis en place : {e}",
                "Die Aktualisierung auf Version {v} ist fehlgeschlagen, Pseudalis {o} wurde wiederhergestellt: {e}",
                "L'aggiornamento alla versione {v} non è riuscito ed è stato ripristinato Pseudalis {o}: {e}",
                "La actualización a la versión {v} falló y se restableció Pseudalis {o}: {e}" } },
            { "notify_update_failed", { "The update to version {v} failed: {e}",
                "La mise à jour vers la version {v} a échoué : {e}",
                "Die Aktualisierung auf Version {v} ist fehlgeschlagen: {e}",
                "L'aggiornamento alla versione {v} non è riuscito: {e}",
                "La actualización a la versión {v} falló: {e}" } },
            { "notify_service_down", { "The Pseudalis service does not answer.", "Le service Pseudalis ne répond pas.",
                "Der Pseudalis-Dienst antwortet nicht.", "Il servizio Pseudalis non risponde.",
                "El servicio Pseudalis no responde." } },
            { "notify_service_stopped", { "The Pseudalis service is stopped: start it from the menu of this icon.",
                "Le service Pseudalis est arrêté : démarrez-le depuis le menu de cette icône.",
                "Der Pseudalis-Dienst ist beendet: Starten Sie ihn über das Menü dieses Symbols.",
                "Il servizio Pseudalis è arrestato: avviatelo dal menu di questa icona.",
                "El servicio Pseudalis está detenido: inícielo desde el menú de este icono." } },
            { "notify_restarting", { "Pseudalis restarts.", "Pseudalis redémarre.", "Pseudalis startet neu.",
                "Pseudalis si riavvia.", "Pseudalis se reinicia." } },
            { "err_stopping", { "The service is stopping.", "Le service s'arrête.", "Der Dienst wird beendet.",
                "Il servizio si sta arrestando.", "El servicio se está deteniendo." } },
            { "err_service", { "The Pseudalis service does not answer ({e}).",
                "Le service Pseudalis ne répond pas ({e}).", "Der Pseudalis-Dienst antwortet nicht ({e}).",
                "Il servizio Pseudalis non risponde ({e}).", "El servicio Pseudalis no responde ({e})." } },
            { "err_port_unavailable", { "The port {port} is in use or cannot be opened: choose another one.",
                "Le port {port} est utilisé ou ne peut pas être ouvert : choisissez-en un autre.",
                "Der Port {port} wird verwendet oder kann nicht geöffnet werden: Wählen Sie einen anderen.",
                "La porta {port} è in uso o non può essere aperta: sceglietene un'altra.",
                "El puerto {port} está en uso o no se puede abrir: elija otro." } },
            { "err_proxy_password", { "The masked password of the proxy applies to its saved address and user only: "
                "type the password again.",
                "Le mot de passe masqué du proxy ne vaut que pour l'adresse et l'utilisateur enregistrés : "
                "saisissez à nouveau le mot de passe.",
                "Das maskierte Kennwort des Proxys gilt nur für die gespeicherte Adresse und den gespeicherten "
                "Benutzer: Geben Sie das Kennwort erneut ein.",
                "La password mascherata del proxy vale solo per l'indirizzo e l'utente salvati: digitate di nuovo "
                "la password.",
                "La contraseña oculta del proxy solo sirve para la dirección y el usuario guardados: escriba de "
                "nuevo la contraseña." } },
            { "err_update_check", { "The update check failed: {e}", "La recherche de mise à jour a échoué : {e}",
                "Die Update-Suche ist fehlgeschlagen: {e}", "La ricerca di aggiornamenti non è riuscita: {e}",
                "La búsqueda de actualizaciones falló: {e}" } },
            { "err_update_record", { "The update to version {version} cannot be recorded in the updates folder.",
                "La mise à jour vers la version {version} ne peut pas être enregistrée dans le dossier des mises à "
                "jour.", "Die Aktualisierung auf Version {version} kann nicht im Update-Ordner gespeichert werden.",
                "L'aggiornamento alla versione {version} non può essere registrato nella cartella degli "
                "aggiornamenti.",
                "La actualización a la versión {version} no se puede registrar en la carpeta de actualizaciones." } },
            { "err_update_feed_missing", { "No signed update feed lies beside the installer of version {version}.",
                "Aucun flux de mise à jour signé n'accompagne le programme d'installation de la version {version}.",
                "Neben dem Installationsprogramm der Version {version} liegt kein signierter Update-Feed.",
                "Accanto al programma di installazione della versione {version} non c'è alcun feed di aggiornamento "
                "firmato.",
                "Junto al programa de instalación de la versión {version} no hay ningún canal de actualización "
                "firmado." } },
            { "err_update_feed_missing_folder", { "{folder} holds no signed update feed (feed.json and "
                "feed.json.sig).",
                "{folder} ne contient aucun flux de mise à jour signé (feed.json et feed.json.sig).",
                "{folder} enthält keinen signierten Update-Feed (feed.json und feed.json.sig).",
                "{folder} non contiene alcun feed di aggiornamento firmato (feed.json e feed.json.sig).",
                "{folder} no contiene ningún canal de actualización firmado (feed.json y feed.json.sig)." } },
            { "err_update_signature", { "The signature of {file} is not valid.",
                "La signature de {file} n'est pas valide.", "Die Signatur von {file} ist ungültig.",
                "La firma di {file} non è valida.", "La firma de {file} no es válida." } },
            { "err_update_feed_mismatch", { "{file} does not describe the installer of version {version}.",
                "{file} ne décrit pas le programme d'installation de la version {version}.",
                "{file} beschreibt nicht das Installationsprogramm der Version {version}.",
                "{file} non descrive il programma di installazione della versione {version}.",
                "{file} no describe el programa de instalación de la versión {version}." } },
            { "err_update_outside", { "The installer of the update {file} lies outside the updates folder.",
                "Le programme d'installation de la mise à jour {file} se trouve hors du dossier des mises à jour.",
                "Das Installationsprogramm der Aktualisierung {file} liegt außerhalb des Update-Ordners.",
                "Il programma di installazione dell'aggiornamento {file} si trova fuori dalla cartella degli "
                "aggiornamenti.",
                "El programa de instalación de la actualización {file} está fuera de la carpeta de "
                "actualizaciones." } },
            { "err_update_installer_mismatch", { "The installer of version {version} does not match its signed feed.",
                "Le programme d'installation de la version {version} ne correspond pas à son flux signé.",
                "Das Installationsprogramm der Version {version} entspricht nicht seinem signierten Feed.",
                "Il programma di installazione della versione {version} non corrisponde al suo feed firmato.",
                "El programa de instalación de la versión {version} no coincide con su canal firmado." } },
            { "err_update_feed_invalid", { "The update feed is incomplete or names no version.",
                "Le flux des mises à jour est incomplet ou n'indique aucune version.",
                "Der Update-Feed ist unvollständig oder nennt keine Version.",
                "Il feed degli aggiornamenti è incompleto o non indica alcuna versione.",
                "El canal de actualizaciones está incompleto o no indica ninguna versión." } },
            { "err_update_download", { "The download of the update failed: {e}",
                "Le téléchargement de la mise à jour a échoué : {e}",
                "Der Download der Aktualisierung ist fehlgeschlagen: {e}",
                "Il download dell'aggiornamento non è riuscito: {e}", "La descarga de la actualización falló: {e}" } },
            { "err_update_not_ready", { "No update is ready.", "Aucune mise à jour n'est prête.",
                "Keine Aktualisierung ist bereit.", "Nessun aggiornamento è pronto.",
                "No hay ninguna actualización lista." } },
            { "err_update_invalid", { "The version {version} of the update is not valid.",
                "La version {version} de la mise à jour n'est pas valide.",
                "Die Version {version} der Aktualisierung ist ungültig.",
                "La versione {version} dell'aggiornamento non è valida.",
                "La versión {version} de la actualización no es válida." } },
            { "err_update_start", { "The installer of the update cannot be started (error {code}).",
                "Le programme d'installation de la mise à jour ne peut pas être démarré (erreur {code}).",
                "Das Installationsprogramm der Aktualisierung kann nicht gestartet werden (Fehler {code}).",
                "Il programma di installazione dell'aggiornamento non può essere avviato (errore {code}).",
                "El programa de instalación de la actualización no se puede iniciar (error {code})." } },
            { "err_update_package_manager", { "On this system, the package manager applies the updates.",
                "Sur ce système, le gestionnaire de paquets applique les mises à jour.",
                "Auf diesem System wendet die Paketverwaltung die Aktualisierungen an.",
                "Su questo sistema, il gestore dei pacchetti applica gli aggiornamenti.",
                "En este sistema, el gestor de paquetes aplica las actualizaciones." } },
            { "btn_project", { "Project page", "Page du projet", "Projektseite", "Pagina del progetto",
                "Página del proyecto" } },
            { "btn_close", { "Close", "Fermer", "Schließen", "Chiudi", "Cerrar" } },
            { "about_built", { "Based on the open source Dlib-Transformer-extensions library",
                "Reposant sur la bibliothèque open source Dlib-Transformer-extensions",
                "Basiert auf der Open-Source-Bibliothek Dlib-Transformer-extensions",
                "Basato sulla libreria open source Dlib-Transformer-extensions",
                "Basado en la biblioteca de código abierto Dlib-Transformer-extensions" } },
            { "setup_title", { "Pseudalis setup", "Installation de Pseudalis", "Pseudalis-Installation",
                "Installazione di Pseudalis", "Instalación de Pseudalis" } },
            { "welcome_title", { "Welcome to Pseudalis", "Bienvenue dans Pseudalis", "Willkommen bei Pseudalis",
                "Benvenuto in Pseudalis", "Bienvenido a Pseudalis" } },
            { "update_title", { "Pseudalis update", "Mise à jour de Pseudalis", "Pseudalis-Aktualisierung",
                "Aggiornamento di Pseudalis", "Actualización de Pseudalis" } },
            { "update_newer", { "Pseudalis {o}, newer than version {v} of this assistant, is installed in {d}: it "
                "stays in place.",
                "Pseudalis {o}, plus récent que la version {v} de cet assistant, est installé dans {d} : il reste "
                "en place.",
                "Pseudalis {o}, neuer als Version {v} dieses Assistenten, ist in {d} installiert: Es bleibt "
                "erhalten.",
                "Pseudalis {o}, più recente della versione {v} di questa procedura, è installato in {d}: resta al "
                "suo posto.",
                "Pseudalis {o}, más reciente que la versión {v} de este asistente, está instalado en {d}: se "
                "mantiene." } },
            { "btn_update", { "Update", "Mettre à jour", "Aktualisieren", "Aggiorna", "Actualizar" } },
            { "update_progress", { "Update", "Mise à jour", "Aktualisierung", "Aggiornamento", "Actualización" } },
            { "update_done_title", { "Pseudalis is up to date", "Pseudalis est à jour", "Pseudalis ist aktuell",
                "Pseudalis è aggiornato", "Pseudalis está actualizado" } },
            { "update_failed", { "The update failed: {e}", "La mise à jour a échoué : {e}",
                "Die Aktualisierung ist fehlgeschlagen: {e}", "L'aggiornamento non è riuscito: {e}",
                "La actualización falló: {e}" } },
            { "update_rolled_back", { "The programs of Pseudalis {o} were put back in place, and its service "
                "started again.", "Les programmes de Pseudalis {o} ont été remis en place et son service a redémarré.",
                "Die Programme von Pseudalis {o} wurden wiederhergestellt, und sein Dienst wurde neu gestartet.",
                "I programmi di Pseudalis {o} sono stati ripristinati e il suo servizio è stato riavviato.",
                "Los programas de Pseudalis {o} se restablecieron y su servicio se reinició." } },
            { "install_rolled_back", { "The files copied were removed: this computer is as it was before the "
                "installation.",
                "Les fichiers copiés ont été supprimés : cet ordinateur est tel qu'avant l'installation.",
                "Die kopierten Dateien wurden entfernt: Dieser Computer ist wieder wie vor der Installation.",
                "I file copiati sono stati rimossi: questo computer è tornato com'era prima dell'installazione.",
                "Los archivos copiados se eliminaron: este equipo está como antes de la instalación." } },
            { "license_title", { "License", "Licence", "Lizenz", "Licenza", "Licencia" } },
            { "license_accept", { "I accept the terms of the license", "J'accepte les termes de la licence",
                "Ich akzeptiere die Lizenzbedingungen", "Accetto i termini della licenza",
                "Acepto los términos de la licencia" } },
            { "folder_title", { "Installation folder", "Dossier d'installation", "Installationsordner",
                "Cartella di installazione", "Carpeta de instalación" } },
            { "folder_programs", { "Programs", "Programmes", "Programme", "Programmi", "Programas" } },
            { "browse", { "Browse…", "Parcourir…", "Durchsuchen…", "Sfoglia…", "Examinar…" } },
            { "folder_overlap", { "{d} lies in the folder of the programs, or holds it: choose a folder apart.",
                "{d} se trouve dans le dossier des programmes, ou le contient : choisissez un dossier distinct.",
                "{d} liegt im Ordner der Programme oder enthält ihn: Wählen Sie einen getrennten Ordner.",
                "{d} si trova nella cartella dei programmi, o la contiene: scegliete una cartella separata.",
                "{d} está en la carpeta de los programas, o la contiene: elija una carpeta aparte." } },
            { "folder_not_local", { "{d} is not a folder of a drive of this computer.",
                "{d} n'est pas un dossier d'un lecteur de cet ordinateur.",
                "{d} ist kein Ordner eines Laufwerks dieses Computers.",
                "{d} non è una cartella di un'unità di questo computer.",
                "{d} no es una carpeta de una unidad de este equipo." } },
            { "folder_not_fixed", { "{d} lies on a removable or network drive: the data of Pseudalis need a "
                "fixed drive of this computer.",
                "{d} se trouve sur un lecteur amovible ou réseau : les données de Pseudalis demandent un lecteur "
                "fixe de cet ordinateur.",
                "{d} liegt auf einem Wechsel- oder Netzlaufwerk: Die Daten von Pseudalis brauchen ein festes "
                "Laufwerk dieses Computers.",
                "{d} si trova su un'unità rimovibile o di rete: i dati di Pseudalis richiedono un'unità fissa di "
                "questo computer.",
                "{d} está en una unidad extraíble o de red: los datos de Pseudalis necesitan una unidad fija de "
                "este equipo." } },
            { "folder_system", { "{d} is the root of a drive, or a folder of Windows or of the users: choose a "
                "folder of its own for Pseudalis.",
                "{d} est la racine d'un lecteur, ou un dossier de Windows ou des utilisateurs : choisissez un "
                "dossier propre à Pseudalis.",
                "{d} ist das Stammverzeichnis eines Laufwerks oder ein Ordner von Windows oder der Benutzer: Wählen "
                "Sie einen eigenen Ordner für Pseudalis.",
                "{d} è la radice di un'unità, o una cartella di Windows o degli utenti: scegliete una cartella "
                "dedicata ad Pseudalis.",
                "{d} es la raíz de una unidad, o una carpeta de Windows o de los usuarios: elija una carpeta "
                "propia para Pseudalis." } },
            { "folder_not_empty", { "{d} already holds other files: choose a new or empty folder.",
                "{d} contient déjà d'autres fichiers : choisissez un dossier nouveau ou vide.",
                "{d} enthält bereits andere Dateien: Wählen Sie einen neuen oder leeren Ordner.",
                "{d} contiene già altri file: scegliete una cartella nuova o vuota.",
                "{d} ya contiene otros archivos: elija una carpeta nueva o vacía." } },
            { "folder_foreign", { "{d} belongs to another user, or a link made by another user leads to it: choose "
                "another folder.",
                "{d} appartient à un autre utilisateur, ou un lien créé par un autre utilisateur y mène : "
                "choisissez un autre dossier.",
                "{d} gehört einem anderen Benutzer, oder ein symbolischer Link eines anderen Benutzers führt "
                "dorthin: Wählen Sie einen anderen Ordner.",
                "{d} appartiene a un altro utente, oppure vi conduce un collegamento simbolico creato da un altro "
                "utente: scegliete un'altra cartella.",
                "{d} pertenece a otro usuario, o un enlace creado por otro usuario lleva a ella: elija otra "
                "carpeta." } },
            { "folder_parent_unsafe", { "Accounts other than the administrators may rename a folder that holds {d} "
                "and put a folder of their own in its place: choose a folder in Program Files or in "
                "ProgramData, or a new folder at the root of a drive.",
                "Des comptes autres que les administrateurs peuvent renommer un dossier qui contient {d} et mettre "
                "un dossier à eux à sa place : choisissez un dossier dans Program Files ou dans ProgramData, "
                "ou un nouveau dossier à la racine d'un lecteur.",
                "Andere Konten als die Administratoren können einen Ordner umbenennen, der {d} enthält, und einen "
                "eigenen Ordner an seine Stelle setzen: Wählen Sie einen Ordner in Program Files oder in "
                "ProgramData oder einen neuen Ordner im Stammverzeichnis eines Laufwerks.",
                "Account diversi dagli amministratori possono rinominare una cartella che contiene {d} e mettere al "
                "suo posto una cartella propria: scegliete una cartella in Program Files o in ProgramData, "
                "oppure una nuova cartella nella radice di un'unità.",
                "Cuentas distintas de los administradores pueden cambiar el nombre de una carpeta que contiene "
                "{d} y poner una carpeta propia en su lugar: elija una carpeta en Program Files o en "
                "ProgramData, o una carpeta nueva en la raíz de una unidad." } },
            { "space_required", { "Space required by the programs: {s}", "Espace nécessaire aux programmes : {s}",
                "Speicherbedarf der Programme: {s}", "Spazio richiesto dai programmi: {s}",
                "Espacio necesario para los programas: {s}" } },
            { "space_available", { "Space available: {s}", "Espace disponible : {s}", "Verfügbarer Speicher: {s}",
                "Spazio disponibile: {s}", "Espacio disponible: {s}" } },
            { "space_insufficient", { "Not enough free space on this drive.",
                "Espace libre insuffisant sur ce lecteur.", "Nicht genügend freier Speicher auf diesem Laufwerk.",
                "Spazio libero insufficiente su questa unità.", "No hay suficiente espacio libre en esta unidad." } },
            { "options_title", { "Options", "Options", "Optionen", "Opzioni", "Opciones" } },
            { "opt_autostart", { "Show the Pseudalis icon when a user signs in",
                "Afficher l'icône Pseudalis à l'ouverture de session", "Pseudalis-Symbol bei der Anmeldung anzeigen",
                "Mostra l'icona di Pseudalis all'accesso", "Mostrar el icono de Pseudalis al iniciar sesión" } },
            { "opt_desktop", { "Create a shortcut on the desktop", "Créer un raccourci sur le bureau",
                "Verknüpfung auf dem Desktop erstellen", "Crea un collegamento sul desktop",
                "Crear un acceso directo en el escritorio" } },
            { "opt_trust", { "Trust the local certificate of Pseudalis in the browsers",
                "Approuver le certificat local de Pseudalis dans les navigateurs",
                "Dem lokalen Zertifikat von Pseudalis in den Browsern vertrauen",
                "Considerare attendibile nei browser il certificato locale di Pseudalis",
                "Confiar en el certificado local de Pseudalis en los navegadores" } },
            { "opt_port", { "Port of the local server", "Port du serveur local", "Port des lokalen Servers",
                "Porta del server locale", "Puerto del servidor local" } },
            { "progress_title", { "Installation", "Installation", "Installation", "Installazione", "Instalación" } },
            { "step_stop", { "Stopping the running programs", "Arrêt des programmes en cours",
                "Laufende Programme werden beendet", "Arresto dei programmi in esecuzione",
                "Deteniendo los programas en ejecución" } },
            { "step_files", { "Copying {f}", "Copie de {f}", "{f} wird kopiert", "Copia di {f}", "Copiando {f}" } },
            { "step_service", { "Registering the service", "Enregistrement du service", "Dienst wird registriert",
                "Registrazione del servizio", "Registrando el servicio" } },
            { "step_certificates", { "Creating the certificates of the local server",
                "Création des certificats du serveur local", "Zertifikate des lokalen Servers werden erstellt",
                "Creazione dei certificati del server locale", "Creando los certificados del servidor local" } },
            { "step_shortcuts", { "Creating the shortcuts", "Création des raccourcis", "Verknüpfungen werden erstellt",
                "Creazione dei collegamenti", "Creando los accesos directos" } },
            { "step_start", { "Starting the service", "Démarrage du service", "Dienst wird gestartet",
                "Avvio del servizio", "Iniciando el servicio" } },
            { "step_done", { "Done", "Terminé", "Fertig", "Completato", "Terminado" } },
            { "finish_title", { "Pseudalis is installed", "Pseudalis est installé", "Pseudalis ist installiert",
                "Pseudalis è installato", "Pseudalis está instalado" } },
            { "opt_launch", { "Open Pseudalis now", "Ouvrir Pseudalis maintenant", "Pseudalis jetzt öffnen",
                "Apri Pseudalis ora", "Abrir Pseudalis ahora" } },
            { "btn_back", { "< Back", "< Précédent", "< Zurück", "< Indietro", "< Atrás" } },
            { "btn_next", { "Next >", "Suivant >", "Weiter >", "Avanti >", "Siguiente >" } },
            { "btn_install", { "Install", "Installer", "Installieren", "Installa", "Instalar" } },
            { "btn_finish", { "Finish", "Terminer", "Fertig stellen", "Fine", "Finalizar" } },
            { "btn_cancel", { "Cancel", "Annuler", "Abbrechen", "Annulla", "Cancelar" } },
            { "btn_uninstall", { "Uninstall", "Désinstaller", "Deinstallieren", "Disinstalla", "Desinstalar" } },
            { "cancel_confirm", { "Stop the installation of Pseudalis?", "Interrompre l'installation de Pseudalis ?",
                "Installation von Pseudalis abbrechen?", "Interrompere l'installazione di Pseudalis?",
                "¿Detener la instalación de Pseudalis?" } },
            { "error_title", { "Pseudalis setup", "Installation de Pseudalis", "Pseudalis-Installation",
                "Installazione di Pseudalis", "Instalación de Pseudalis" } },
            { "error_failed", { "The installation failed: {e}", "L'installation a échoué : {e}",
                "Die Installation ist fehlgeschlagen: {e}", "L'installazione non è riuscita: {e}",
                "La instalación falló: {e}" } },
            { "uninstall_title", { "Uninstall Pseudalis", "Désinstaller Pseudalis", "Pseudalis deinstallieren",
                "Disinstalla Pseudalis", "Desinstalar Pseudalis" } },
            { "uninstall_done", { "Pseudalis was removed from this computer.",
                "Pseudalis a été retiré de cet ordinateur.", "Pseudalis wurde von diesem Computer entfernt.",
                "Pseudalis è stato rimosso da questo computer.", "Pseudalis se eliminó de este equipo." } },
            { "uninstall_progress", { "Removing Pseudalis", "Suppression de Pseudalis", "Pseudalis wird entfernt",
                "Rimozione di Pseudalis", "Eliminando Pseudalis" } },
            { "not_installed", { "Pseudalis is not installed on this computer.",
                "Pseudalis n'est pas installé sur cet ordinateur.",
                "Pseudalis ist auf diesem Computer nicht installiert.",
                "Pseudalis non è installato su questo computer.", "Pseudalis no está instalado en este equipo." } },
            { "no_payload", { "This copy of the installer, kept in the folder of Pseudalis, only uninstalls it. To "
                "install or update Pseudalis, start the installer you downloaded.",
                "Cette copie du programme d'installation, conservée dans le dossier de Pseudalis, sert uniquement à "
                "le désinstaller. Pour installer ou mettre à jour Pseudalis, lancez le programme "
                "d'installation téléchargé.",
                "Diese Kopie des Installationsprogramms im Ordner von Pseudalis dient nur zur Deinstallation. "
                "Starten Sie zum Installieren oder Aktualisieren von Pseudalis das heruntergeladene "
                "Installationsprogramm.",
                "Questa copia del programma di installazione, conservata nella cartella di Pseudalis, serve solo a "
                "disinstallarlo. Per installare o aggiornare Pseudalis, avviate il programma di installazione "
                "scaricato.",
                "Esta copia del programa de instalación, guardada en la carpeta de Pseudalis, solo sirve para "
                "desinstalarlo. Para instalar o actualizar Pseudalis, inicie el programa de instalación que "
                "descargó." } },
            { "lbl_language", { "Language", "Langue", "Sprache", "Lingua", "Idioma" } },
            { "tray_open", { "Open Pseudalis", "Ouvrir Pseudalis", "Pseudalis öffnen", "Apri Pseudalis",
                "Abrir Pseudalis" } },
            { "tray_drive", { "Open the drive {l}:",
                "Ouvrir le lecteur {l}:",
                "Laufwerk {l}: öffnen",
                "Apri l'unità {l}:",
                "Abrir la unidad {l}:" } },
            { "tray_refused", { "Files not processed by the drive", "Fichiers non traités par le lecteur",
                "Vom Laufwerk nicht verarbeitete Dateien", "File non elaborati dall'unità",
                "Archivos no procesados por la unidad" } },
            { "tray_about", { "About Pseudalis", "À propos de Pseudalis", "Über Pseudalis",
                "Informazioni su Pseudalis", "Acerca de Pseudalis" } },
            { "tip_drive", { "Drive {l}:", "Lecteur {l}:", "Laufwerk {l}:", "Unità {l}:", "Unidad {l}:" } },
            { "tip_no_drive", { "No drive", "Pas de lecteur", "Kein Laufwerk", "Nessuna unità", "Sin unidad" } },
            { "tip_drive_error", { "Drive {l}: not available", "Lecteur {l}: indisponible",
                "Laufwerk {l}: nicht verfügbar", "Unità {l}: non disponibile", "Unidad {l}: no disponible" } },
            { "tip_records", { "{n} originals kept", "{n} originaux conservés", "{n} Originale aufbewahrt",
                "{n} originali conservati", "{n} originales guardados" } },
            { "tip_records_one", { "{n} original kept",
                "{n} original conservé",
                "{n} Original aufbewahrt",
                "{n} originale conservato",
                "{n} original guardado" } },
            { "notify_drive_title", { "Drive {l}:", "Lecteur {l}:", "Laufwerk {l}:", "Unità {l}:", "Unidad {l}:" } },
            { "notify_protected_file", { "{f}: {n} values replaced",
                "{f} : {n} valeurs remplacées",
                "{f}: {n} Werte ersetzt",
                "{f}: {n} valori sostituiti",
                "{f}: {n} valores sustituidos" } },
            { "notify_protected_file_one", { "{f}: {n} value replaced",
                "{f} : {n} valeur remplacée",
                "{f}: {n} Wert ersetzt",
                "{f}: {n} valore sostituito",
                "{f}: {n} valor sustituido" } },
            { "notify_protected_many", { "{n} files anonymized", "{n} fichiers anonymisés", "{n} Dateien anonymisiert",
                "{n} file anonimizzati", "{n} archivos anonimizados" } },
            { "notify_refused", { "{f} was not processed: {e}", "{f} n'a pas été traité : {e}",
                "{f} wurde nicht verarbeitet: {e}", "{f} non è stato elaborato: {e}", "{f} no se procesó: {e}" } },
            { "notify_refused_many", { "{n} files were not processed, {f} among them: {e}",
                "{n} fichiers n'ont pas été traités, dont {f} : {e}",
                "{n} Dateien wurden nicht verarbeitet, darunter {f}: {e}",
                "{n} file non sono stati elaborati, tra cui {f}: {e}",
                "{n} archivos no se procesaron, entre ellos {f}: {e}" } },
            { "notify_drive_error", { "The drive {l}: is not available: {e}",
                "Le lecteur {l}: n'est pas disponible : {e}", "Das Laufwerk {l}: ist nicht verfügbar: {e}",
                "L'unità {l}: non è disponibile: {e}", "La unidad {l}: no está disponible: {e}" } },
            { "shell_protected", { "{f} is anonymized: {n} values replaced.",
                "{f} est anonymisé : {n} valeurs remplacées.", "{f} ist anonymisiert: {n} Werte ersetzt.",
                "{f} è anonimizzato: {n} valori sostituiti.", "{f} está anonimizado: {n} valores sustituidos." } },
            { "shell_protected_one", { "{f} is anonymized: {n} value replaced.",
                "{f} est anonymisé : {n} valeur remplacée.",
                "{f} ist anonymisiert: {n} Wert ersetzt.",
                "{f} è anonimizzato: {n} valore sostituito.",
                "{f} está anonimizado: {n} valor sustituido." } },
            { "shell_protected_copy", { "{f} is written, anonymized: {n} values replaced.",
                "{f} est écrit, anonymisé : {n} valeurs remplacées.",
                "{f} ist anonymisiert geschrieben: {n} Werte ersetzt.",
                "{f} è stato scritto, anonimizzato: {n} valori sostituiti.",
                "{f} se ha escrito, anonimizado: {n} valores sustituidos." } },
            { "shell_protected_copy_one", { "{f} is written, anonymized: {n} value replaced.",
                "{f} est écrit, anonymisé : {n} valeur remplacée.",
                "{f} ist anonymisiert geschrieben: {n} Wert ersetzt.",
                "{f} è stato scritto, anonimizzato: {n} valore sostituito.",
                "{f} se ha escrito, anonimizado: {n} valor sustituido." } },
            { "shell_restored", { "{f} is restored.", "{f} est restauré.", "{f} ist wiederhergestellt.",
                "{f} è ripristinato.", "{f} está restaurado." } },
            { "shell_restored_copy", { "{f} is written, restored.", "{f} est écrit, restauré.",
                "{f} ist wiederhergestellt geschrieben.", "{f} è stato scritto, ripristinato.",
                "{f} se ha escrito, restaurado." } },
            { "shell_restored_tags", { "{f} is restored from its tags, {n} values put back: the file changed after "
                "its anonymization.",
                "{f} est restauré à partir de ses balises, {n} valeurs remises : le fichier a changé après son "
                "anonymisation.",
                "{f} ist aus seinen Markierungen wiederhergestellt, {n} Werte zurückgesetzt: Die Datei wurde nach "
                "ihrer Anonymisierung geändert.",
                "{f} è ripristinato dai suoi tag, {n} valori rimessi: il file è cambiato dopo la sua "
                "anonimizzazione.",
                "{f} se restauró a partir de sus etiquetas, {n} valores repuestos: el archivo cambió después de "
                "su anonimización." } },
            { "shell_restored_tags_one", { "{f} is restored from its tags, {n} value put back: the file changed after "
                "its anonymization.",
                "{f} est restauré à partir de ses balises, {n} valeur remise : le fichier a changé après son "
                "anonymisation.",
                "{f} ist aus seinen Markierungen wiederhergestellt, {n} Wert zurückgesetzt: Die Datei wurde nach "
                "ihrer Anonymisierung geändert.",
                "{f} è ripristinato dai suoi tag, {n} valore rimesso: il file è cambiato dopo la sua anonimizzazione.",
                "{f} se restauró a partir de sus etiquetas, {n} valor repuesto: el archivo cambió después de su "
                "anonimización." } },
            { "count_files_anonymized", { "{n} files anonymized",
                "{n} fichiers anonymisés",
                "{n} Dateien anonymisiert",
                "{n} file anonimizzati",
                "{n} archivos anonimizados" } },
            { "count_files_anonymized_one", { "{n} file anonymized",
                "{n} fichier anonymisé",
                "{n} Datei anonymisiert",
                "{n} file anonimizzato",
                "{n} archivo anonimizado" } },
            { "count_values_replaced", { "{n} values replaced",
                "{n} valeurs remplacées",
                "{n} Werte ersetzt",
                "{n} valori sostituiti",
                "{n} valores sustituidos" } },
            { "count_values_replaced_one", { "{n} value replaced",
                "{n} valeur remplacée",
                "{n} Wert ersetzt",
                "{n} valore sostituito",
                "{n} valor sustituido" } },
            { "count_files_restored", { "{n} files restored",
                "{n} fichiers restaurés",
                "{n} Dateien wiederhergestellt",
                "{n} file ripristinati",
                "{n} archivos restaurados" } },
            { "count_files_restored_one", { "{n} file restored",
                "{n} fichier restauré",
                "{n} Datei wiederhergestellt",
                "{n} file ripristinato",
                "{n} archivo restaurado" } },
            { "shell_folder_protected", { "{f}: {files}, {values}.",
                "{f} : {files}, {values}.",
                "{f}: {files}, {values}.",
                "{f}: {files}, {values}.",
                "{f}: {files}, {values}." } },
            { "shell_folder_protected_copy", { "{f} is written: {files}, {values}.",
                "{f} est écrit : {files}, {values}.",
                "{f} ist geschrieben: {files}, {values}.",
                "{f} è stato scritto: {files}, {values}.",
                "{f} se ha escrito: {files}, {values}." } },
            { "shell_folder_restored", { "{f}: {files}.",
                "{f} : {files}.",
                "{f}: {files}.",
                "{f}: {files}.",
                "{f}: {files}." } },
            { "shell_folder_restored_copy", { "{f} is written: {files}.",
                "{f} est écrit : {files}.",
                "{f} ist geschrieben: {files}.",
                "{f} è stato scritto: {files}.",
                "{f} se ha escrito: {files}." } },
            { "shell_folder_already", { "{n} already anonymized.",
                "{n} déjà anonymisés.",
                "{n} bereits anonymisiert.",
                "{n} già anonimizzati.",
                "{n} ya anonimizados." } },
            { "shell_folder_already_one", { "{n} already anonymized.",
                "{n} déjà anonymisé.",
                "{n} bereits anonymisiert.",
                "{n} già anonimizzato.",
                "{n} ya anonimizado." } },
            { "shell_folder_other_kept", { "{n} of another kind left as they are: {l}.",
                "{n} d'un autre type laissés tels quels : {l}.",
                "{n} anderer Art unverändert gelassen: {l}.",
                "{n} di altro tipo lasciati invariati: {l}.",
                "{n} de otro tipo dejados como están: {l}." } },
            { "shell_folder_other_kept_one", { "{n} of another kind left as it is: {l}.",
                "{n} d'un autre type laissé tel quel : {l}.",
                "{n} anderer Art unverändert gelassen: {l}.",
                "{n} di altro tipo lasciato invariato: {l}.",
                "{n} de otro tipo dejado como está: {l}." } },
            { "shell_folder_other_out", { "{n} of another kind left out: {l}.",
                "{n} d'un autre type écartés : {l}.",
                "{n} anderer Art ausgelassen: {l}.",
                "{n} di altro tipo esclusi: {l}.",
                "{n} de otro tipo descartados: {l}." } },
            { "shell_folder_other_out_one", { "{n} of another kind left out: {l}.",
                "{n} d'un autre type écarté : {l}.",
                "{n} anderer Art ausgelassen: {l}.",
                "{n} di altro tipo escluso: {l}.",
                "{n} de otro tipo descartado: {l}." } },
            { "shell_folder_failed", { "{n} not processed, {f} among them: {e}",
                "{n} non traités, dont {f} : {e}",
                "{n} nicht verarbeitet, darunter {f}: {e}",
                "{n} non elaborati, tra cui {f}: {e}",
                "{n} no procesados, entre ellos {f}: {e}" } },
            { "shell_folder_failed_one", { "{f} not processed: {e}",
                "{f} non traité : {e}",
                "{f} nicht verarbeitet: {e}",
                "{f} non elaborato: {e}",
                "{f} no procesado: {e}" } },
            { "shell_folder_stopped", { "Stopped after {n} files: the folder holds more.",
                "Arrêt après {n} fichiers : le dossier en contient davantage.",
                "Abbruch nach {n} Dateien: Der Ordner enthält mehr.",
                "Interrotto dopo {n} file: la cartella ne contiene di più.",
                "Detenido tras {n} archivos: la carpeta contiene más." } },
            { "shell_folder_too_deep", { "The folders lie more than {n} levels deep.",
                "Les dossiers descendent à plus de {n} niveaux.",
                "Die Ordner liegen mehr als {n} Ebenen tief.",
                "Le cartelle scendono oltre {n} livelli.",
                "Las carpetas bajan más de {n} niveles." } },
            { "shell_folder_kept_name", { "it keeps its name: {e}",
                "il garde son nom : {e}",
                "er behält seinen Namen: {e}",
                "mantiene il suo nome: {e}",
                "conserva su nombre: {e}" } },
            { "shell_many_done", { "{n} files processed.", "{n} fichiers traités.", "{n} Dateien verarbeitet.",
                "{n} file elaborati.", "{n} archivos procesados." } },
            { "shell_many_failed", { "{n} files could not be processed, {f} among them: {e}",
                "{n} fichiers n'ont pas pu être traités, dont {f} : {e}",
                "{n} Dateien konnten nicht verarbeitet werden, darunter {f}: {e}",
                "{n} file non hanno potuto essere elaborati, tra cui {f}: {e}",
                "{n} archivos no pudieron procesarse, entre ellos {f}: {e}" } },
            { "shell_many_failed_one", { "{f} could not be processed: {e}",
                "{f} n'a pas pu être traité : {e}",
                "{f} konnte nicht verarbeitet werden: {e}",
                "{f} non ha potuto essere elaborato: {e}",
                "{f} no pudo procesarse: {e}" } },
            { "shell_failed", { "{f}: {e}", "{f} : {e}", "{f}: {e}", "{f}: {e}", "{f}: {e}" } },
            { "shell_already", { "{f} is already anonymized.", "{f} est déjà anonymisé.",
                "{f} ist bereits anonymisiert.", "{f} è già anonimizzato.", "{f} ya está anonimizado." } },
            { "shell_unsupported", { "Pseudalis does not process the files of this kind: {f}",
                "Pseudalis ne traite pas les fichiers de ce type : {f}",
                "Pseudalis verarbeitet Dateien dieser Art nicht: {f}",
                "Pseudalis non elabora i file di questo tipo: {f}",
                "Pseudalis no procesa los archivos de este tipo: {f}" } },
            { "shell_on_drive", { "The files of the drive {l}: are anonymized already.",
                "Les fichiers du lecteur {l}: sont déjà anonymisés.",
                "Die Dateien des Laufwerks {l}: sind bereits anonymisiert.",
                "I file dell'unità {l}: sono già anonimizzati.",
                "Los archivos de la unidad {l}: ya están anonimizados." } },
            { "shell_save_title", { "Save the original of {f}", "Enregistrer l'original de {f}",
                "Original von {f} speichern", "Salva l'originale di {f}", "Guardar el original de {f}" } },
            { "shell_save_folder_title", { "Choose the folder where the originals of {f} go",
                "Choisir le dossier où vont les originaux de {f}",
                "Ordner für die Originale von {f} wählen",
                "Scegli la cartella in cui vanno gli originali di {f}",
                "Elegir la carpeta a la que van los originales de {f}" } },
            { "shell_cannot_read", { "{f} cannot be read: {e}", "{f} ne peut pas être lu : {e}",
                "{f} kann nicht gelesen werden: {e}", "{f} non può essere letto: {e}", "{f} no se puede leer: {e}" } },
            { "shell_cannot_write", { "{f} cannot be written: {e}", "{f} ne peut pas être écrit : {e}",
                "{f} kann nicht geschrieben werden: {e}", "{f} non può essere scritto: {e}",
                "{f} no se puede escribir: {e}" } },
            { "shell_changed", { "{f} changed while it was processed: it stays as it is. Try again.",
                "{f} a changé pendant son traitement : il reste tel quel. Réessayez.",
                "{f} wurde während der Verarbeitung geändert: Die Datei bleibt unverändert. Versuchen Sie es erneut.",
                "{f} è cambiato durante l'elaborazione: resta com'è. Riprovate.",
                "{f} cambió durante su procesamiento: se queda como está. Vuelva a intentarlo." } },
            { "shell_in_use", { "{f} is open in another program: close it, then try again.",
                "{f} est ouvert dans un autre programme : fermez-le, puis réessayez.",
                "{f} ist in einem anderen Programm geöffnet: Schließen Sie es und versuchen Sie es erneut.",
                "{f} è aperto in un altro programma: chiudetelo, poi riprovate.",
                "{f} está abierto en otro programa: ciérrelo y vuelva a intentarlo." } },
            { "suffix_anonymized", { "anonymized", "anonymisé", "anonymisiert", "anonimizzato", "anonimizado" } },
            { "suffix_restored", { "restored", "restauré", "wiederhergestellt", "ripristinato", "restaurado" } },
            { "err_engine", { "The engine of Pseudalis is not ready.", "Le moteur de Pseudalis n'est pas prêt.",
                "Die Engine von Pseudalis ist nicht bereit.", "Il motore di Pseudalis non è pronto.",
                "El motor de Pseudalis no está listo." } },
            { "err_file_too_large", { "The file is larger than {mb} MB, the limit of the settings.",
                "Le fichier dépasse {mb} Mo, la limite des réglages.",
                "Die Datei ist größer als {mb} MB, die Grenze der Einstellungen.",
                "Il file supera {mb} MB, il limite delle impostazioni.",
                "El archivo supera {mb} MB, el límite de la configuración." } },
            { "err_protect_failed", { "The file cannot be anonymized: {error}",
                "Le fichier ne peut pas être anonymisé : {error}", "Die Datei kann nicht anonymisiert werden: {error}",
                "Il file non può essere anonimizzato: {error}", "El archivo no se puede anonimizar: {error}" } },
            { "err_restore_failed", { "The file cannot be restored: {error}",
                "Le fichier ne peut pas être restauré : {error}",
                "Die Datei kann nicht wiederhergestellt werden: {error}",
                "Il file non può essere ripristinato: {error}", "El archivo no se puede restaurar: {error}" } },
            { "err_unsupported", { "Pseudalis does not process the files of this kind.",
                "Pseudalis ne traite pas les fichiers de ce type.",
                "Pseudalis verarbeitet Dateien dieser Art nicht.",
                "Pseudalis non elabora i file di questo tipo.",
                "Pseudalis no procesa los archivos de este tipo." } },
            { "err_kind_mismatch", { "The content of this file does not match its name.",
                "Le contenu de ce fichier ne correspond pas à son nom.",
                "Der Inhalt dieser Datei entspricht nicht ihrem Namen.",
                "Il contenuto di questo file non corrisponde al suo nome.",
                "El contenido de este archivo no corresponde a su nombre." } },
            { "err_encoded_content", { "This file holds encoded parts, attachments, base64 or RTF for instance, that "
                "Pseudalis cannot read: it is not processed.",
                "Ce fichier contient des parties encodées, pièces jointes, base64 ou RTF par exemple, que Pseudalis "
                "ne sait pas lire : il n'est pas traité.",
                "Diese Datei enthält kodierte Teile, etwa Anhänge, Base64 oder RTF, die Pseudalis nicht lesen kann: "
                "Sie wird nicht verarbeitet.",
                "Questo file contiene parti codificate, allegati, base64 o RTF per esempio, che Pseudalis non sa "
                "leggere: non viene elaborato.",
                "Este archivo contiene partes codificadas, adjuntos, base64 o RTF por ejemplo, que Pseudalis no "
                "sabe leer: no se procesa." } },
            { "err_not_protected", { "This file holds no value that Pseudalis protected with the key of this "
                "computer.",
                "Ce fichier ne contient aucune valeur protégée par Pseudalis avec la clé de cet ordinateur.",
                "Diese Datei enthält keinen Wert, den Pseudalis mit dem Schlüssel dieses Computers geschützt hat.",
                "Questo file non contiene alcun valore protetto da Pseudalis con la chiave di questo computer.",
                "Este archivo no contiene ningún valor protegido por Pseudalis con la clave de este equipo." } },
            { "err_original_missing", { "The original of this file can no longer be read in the vault.",
                "L'original de ce fichier ne peut plus être lu dans le coffre.",
                "Das Original dieser Datei kann im Tresor nicht mehr gelesen werden.",
                "L'originale di questo file non può più essere letto nella cassaforte.",
                "El original de este archivo ya no se puede leer en la caja fuerte." } },
            { "err_busy", { "The service is busy with other files: try again in a moment.",
                "Le service traite d'autres fichiers : réessayez dans un instant.",
                "Der Dienst verarbeitet andere Dateien: Versuchen Sie es gleich noch einmal.",
                "Il servizio sta elaborando altri file: riprovate tra un momento.",
                "El servicio está procesando otros archivos: vuelva a intentarlo en un momento." } },
            { "err_original_forgotten", { "The original of this file was forgotten on {date}: it can no longer be "
                "restored.",
                "L'original de ce fichier a été oublié le {date} : il ne peut plus être restauré.",
                "Das Original dieser Datei wurde am {date} vergessen: Es kann nicht mehr wiederhergestellt werden.",
                "L'originale di questo file è stato dimenticato il {date}: non può più essere ripristinato.",
                "El original de este archivo se olvidó el {date}: ya no se puede restaurar." } },
            { "err_original_space", { "The original of this file was forgotten on {date} to make room for newer "
                "originals: it can no longer be restored.",
                "L'original de ce fichier a été oublié le {date} pour faire place à des originaux plus récents : il "
                "ne peut plus être restauré.",
                "Das Original dieser Datei wurde am {date} vergessen, um Platz für neuere Originale zu schaffen: Es "
                "kann nicht mehr wiederhergestellt werden.",
                "L'originale di questo file è stato dimenticato il {date} per fare spazio a originali più recenti: "
                "non può più essere ripristinato.",
                "El original de este archivo se olvidó el {date} para dejar sitio a originales más recientes: ya no "
                "se puede restaurar." } },
            { "err_original_expired", { "The original of this file was kept until {date} after the file left the "
                "drive: it can no longer be restored.",
                "L'original de ce fichier a été conservé jusqu'au {date} après la sortie du fichier du lecteur : il "
                "ne peut plus être restauré.",
                "Das Original dieser Datei wurde bis zum {date} aufbewahrt, nachdem die Datei das Laufwerk verlassen "
                "hatte: Es kann nicht mehr wiederhergestellt werden.",
                "L'originale di questo file è stato conservato fino al {date} dopo l'uscita del file dall'unità: non "
                "può più essere ripristinato.",
                "El original de este archivo se conservó hasta el {date} después de que el archivo saliera de la "
                "unidad: ya no se puede restaurar." } },
            { "err_vault", { "The original cannot be kept in the vault.",
                "L'original ne peut pas être conservé dans le coffre.",
                "Das Original kann nicht im Tresor aufbewahrt werden.",
                "L'originale non può essere conservato nella cassaforte.",
                "El original no se puede guardar en la caja fuerte." } },
            { "err_database", { "The database of Pseudalis cannot be written: {e}",
                "La base de données de Pseudalis ne peut pas être écrite : {e}",
                "Die Datenbank von Pseudalis kann nicht geschrieben werden: {e}",
                "Il database di Pseudalis non può essere scritto: {e}",
                "La base de datos de Pseudalis no se puede escribir: {e}" } },
            { "err_authentication", { "The administration token of Pseudalis is missing or wrong: restart the icon.",
                "Le jeton d'administration de Pseudalis manque ou est incorrect : redémarrez l'icône.",
                "Das Verwaltungstoken von Pseudalis fehlt oder ist falsch: Starten Sie das Symbol neu.",
                "Il token di amministrazione di Pseudalis manca o è errato: riavviate l'icona.",
                "El token de administración de Pseudalis falta o es incorrecto: reinicie el icono." } },
            { "about_title", { "About Pseudalis", "À propos de Pseudalis", "Über Pseudalis",
                "Informazioni su Pseudalis", "Acerca de Pseudalis" } },
            { "about_version", { "{p}, version {v}", "{p}, version {v}", "{p}, Version {v}", "{p}, versione {v}",
                "{p}, versión {v}" } },
            { "about_purpose", { "Anonymization of the personal data of documents, on this computer: nothing leaves "
                "it.", "Anonymisation des données personnelles des documents, sur cet ordinateur : rien n'en sort.",
                "Anonymisierung der personenbezogenen Daten von Dokumenten auf diesem Computer: Nichts verlässt ihn.",
                "Anonimizzazione dei dati personali dei documenti, su questo computer: nulla ne esce.",
                "Anonimización de los datos personales de los documentos, en este equipo: nada sale de él." } },
            { "about_written", { "Written by {a}", "Écrit par {a}", "Geschrieben von {a}", "Scritto da {a}",
                "Escrito por {a}" } },
            { "about_license", { "The Community edition is free and open source software, distributed under the "
                "Boost Software License 1.0.",
                "L'édition Community est un logiciel libre et open source, distribué sous la licence Boost Software "
                "License 1.0.",
                "Die Community-Edition ist freie Open-Source-Software, vertrieben unter der Boost Software License "
                "1.0.",
                "L'edizione Community è software libero e open source, distribuito con la licenza Boost Software "
                "License 1.0.",
                "La edición Community es software libre y de código abierto, distribuido bajo la licencia Boost "
                "Software License 1.0." } },
            { "welcome_text", { "This assistant installs Pseudalis Community {v}: the service that anonymizes the "
                "personal data of documents on this computer, its drive, its commands in the Explorer, its icon in "
                "the notification area and its web application. Nothing leaves this computer.",
                "Cet assistant installe Pseudalis Community {v} : le service qui anonymise les données personnelles "
                "des documents sur cet ordinateur, son lecteur, ses commandes dans l'Explorateur, son icône dans la "
                "zone de notification et son application web. Rien ne quitte cet ordinateur.",
                "Dieser Assistent installiert Pseudalis Community {v}: den Dienst, der die personenbezogenen Daten "
                "von Dokumenten auf diesem Computer anonymisiert, sein Laufwerk, seine Befehle im Explorer, sein "
                "Symbol im Infobereich und seine Webanwendung. Nichts verlässt diesen Computer.",
                "Questo assistente installa Pseudalis Community {v}: il servizio che anonimizza i dati personali dei "
                "documenti su questo computer, la sua unità, i suoi comandi in Esplora file, la sua icona nell'area "
                "di notifica e la sua applicazione web. Nulla lascia questo computer.",
                "Este asistente instala Pseudalis Community {v}: el servicio que anonimiza los datos personales "
                "de los documentos en este equipo, su unidad, sus comandos en el Explorador, su icono en el área "
                "de notificación y su aplicación web. Nada sale de este equipo." } },
            { "update_text", { "Pseudalis {o} is installed in {d}. This assistant updates it to version {v} and "
                "keeps its key, the originals it keeps, its settings, its certificates, its drive, its "
                "commands in the Explorer, its shortcuts and the start of its icon.",
                "Pseudalis {o} est installé dans {d}. Cet assistant le met à jour vers la version {v} et conserve "
                "sa clé, les originaux qu'il garde, ses réglages, ses certificats, son lecteur, ses commandes "
                "dans l'Explorateur, ses raccourcis et le démarrage de son icône.",
                "Pseudalis {o} ist in {d} installiert. Dieser Assistent aktualisiert es auf Version {v} und behält "
                "seinen Schlüssel, die aufbewahrten Originale, seine Einstellungen, Zertifikate, sein "
                "Laufwerk, seine Befehle im Explorer, seine Verknüpfungen und den Start seines Symbols bei.",
                "Pseudalis {o} è installato in {d}. Questa procedura lo aggiorna alla versione {v} e conserva la "
                "sua chiave, gli originali che custodisce, le impostazioni, i certificati, la sua unità, i "
                "suoi comandi in Esplora file, i collegamenti e l'avvio della sua icona.",
                "Pseudalis {o} está instalado en {d}. Este asistente lo actualiza a la versión {v} y conserva su "
                "clave, los originales que guarda, sus ajustes, sus certificados, su unidad, sus comandos en "
                "el Explorador, sus accesos directos y el inicio de su icono." } },
            { "update_done_text", { "Pseudalis {v} replaced version {o}. The service started again with its key and "
                "the originals it keeps.",
                "Pseudalis {v} a remplacé la version {o}. Le service a redémarré avec sa clé et les originaux qu'il "
                "garde.",
                "Pseudalis {v} hat Version {o} ersetzt. Der Dienst wurde mit seinem Schlüssel und den aufbewahrten "
                "Originalen neu gestartet.",
                "Pseudalis {v} ha sostituito la versione {o}. Il servizio è ripartito con la sua chiave e gli "
                "originali che custodisce.",
                "Pseudalis {v} reemplazó la versión {o}. El servicio se reinició con su clave y los originales "
                "que guarda." } },
            { "folder_data", { "Data: key, originals, settings, logs and certificates",
                "Données : clé, originaux, réglages, journaux et certificats",
                "Daten: Schlüssel, Originale, Einstellungen, Protokolle und Zertifikate",
                "Dati: chiave, originali, impostazioni, log e certificati",
                "Datos: clave, originales, ajustes, registros y certificados" } },
            { "finish_text", { "Pseudalis runs as a service. To anonymize a document or a folder, copy it onto its "
                "drive, or right-click it in the Explorer (Show more options, Pseudalis), which restores it too. Its "
                "icon in the notification area opens the web application.",
                "Pseudalis fonctionne comme un service. Pour anonymiser un document ou un dossier, copiez-le sur son "
                "lecteur, ou faites un clic droit dessus dans l'Explorateur (Afficher d'autres options, Pseudalis), "
                "qui le restaure aussi. Son icône dans la zone de notification ouvre l'application web.",
                "Pseudalis läuft als Dienst. Um ein Dokument oder einen Ordner zu anonymisieren, kopieren Sie es auf "
                "sein Laufwerk oder klicken Sie es im Explorer mit der rechten Maustaste an (Weitere Optionen "
                "anzeigen, Pseudalis), wo es auch wiederhergestellt wird. Sein Symbol im Infobereich öffnet die "
                "Webanwendung.",
                "Pseudalis funziona come servizio. Per anonimizzare un documento o una cartella, copiatelo sulla sua "
                "unità, o fate clic con il tasto destro su di esso in Esplora file (Mostra altre opzioni, Pseudalis), "
                "che lo ripristina anche. La sua icona nell'area di notifica apre l'applicazione web.",
                "Pseudalis funciona como un servicio. Para anonimizar un documento o una carpeta, cópielo en su "
                "unidad o haga clic con el botón derecho sobre él en el Explorador (Mostrar más opciones, "
                "Pseudalis), que también lo restaura. Su icono en el área de notificación abre la aplicación web." } },
            { "uninstall_text", { "Pseudalis, its service, its drive and the files it holds, its commands in the "
                "Explorer, its notification icon, its key, the originals it keeps, its settings and its logs will be "
                "removed from this computer: the documents it anonymized can then no longer be restored.",
                "Pseudalis, son service, son lecteur et les fichiers qu'il contient, ses commandes dans "
                "l'Explorateur, son icône de notification, sa clé, les originaux qu'il garde, ses réglages et ses "
                "journaux seront retirés de cet ordinateur : les documents qu'il a anonymisés ne pourront plus être "
                "restaurés.",
                "Pseudalis, sein Dienst, sein Laufwerk mit den darin enthaltenen Dateien, seine Befehle im Explorer, "
                "sein Benachrichtigungssymbol, sein Schlüssel, die aufbewahrten Originale, seine Einstellungen und "
                "Protokolle werden von diesem Computer entfernt: Die damit anonymisierten Dokumente können dann nicht "
                "mehr wiederhergestellt werden.",
                "Pseudalis, il suo servizio, la sua unità e i file che contiene, i suoi comandi in Esplora file, la "
                "sua icona di notifica, la sua chiave, gli originali che custodisce, le impostazioni e i registri "
                "saranno rimossi da questo computer: i documenti che ha anonimizzato non potranno più essere "
                "ripristinati.",
                "Pseudalis, su servicio, su unidad y los archivos que contiene, sus comandos en el Explorador, su "
                "icono de notificación, su clave, los originales que guarda, sus ajustes y sus registros se "
                "eliminarán de este equipo: los documentos que anonimizó ya no podrán restaurarse." } },
            { "uninstall_text_keep", { "Pseudalis, its service, the letter of its drive, its commands in the Explorer "
                "and its notification icon will be removed from this computer. Its key, the originals it keeps, the "
                "files of its drive and its settings stay in its data folder for a later installation.",
                "Pseudalis, son service, la lettre de son lecteur, ses commandes dans l'Explorateur et son icône de "
                "notification seront retirés de cet ordinateur. Sa clé, les originaux qu'il garde, les fichiers de "
                "son lecteur et ses réglages restent dans son dossier de données pour une installation ultérieure.",
                "Pseudalis, sein Dienst, der Buchstabe seines Laufwerks, seine Befehle im Explorer und sein "
                "Benachrichtigungssymbol werden von diesem Computer entfernt. Sein Schlüssel, die aufbewahrten "
                "Originale, die Dateien seines Laufwerks und seine Einstellungen bleiben in seinem Datenordner für "
                "eine spätere Installation erhalten.",
                "Pseudalis, il suo servizio, la lettera della sua unità, i suoi comandi in Esplora file e la sua "
                "icona di notifica saranno rimossi da questo computer. La sua chiave, gli originali che custodisce, i "
                "file della sua unità e le impostazioni restano nella sua cartella dei dati per un'installazione "
                "successiva.",
                "Pseudalis, su servicio, la letra de su unidad, sus comandos en el Explorador y su icono de "
                "notificación se eliminarán de este equipo. Su clave, los originales que guarda, los archivos de "
                "su unidad y sus ajustes se conservan en su carpeta de datos para una instalación posterior." } },
            { "uninstall_keep_data", { "Keep the key, the originals and the settings for a later installation",
                "Conserver la clé, les originaux et les réglages pour une installation ultérieure",
                "Schlüssel, Originale und Einstellungen für eine spätere Installation behalten",
                "Conserva la chiave, gli originali e le impostazioni per un'installazione successiva",
                "Conservar la clave, los originales y los ajustes para una instalación posterior" } },
            { "uninstall_kept", { "The folder {d} stays, with the data you keep in it.",
                "Le dossier {d} reste en place, avec les données que vous y conservez.",
                "Der Ordner {d} bleibt bestehen, mit den Daten, die Sie darin behalten.",
                "La cartella {d} resta al suo posto, con i dati che vi conservate.",
                "La carpeta {d} se mantiene, con los datos que conserva en ella." } },
            { "opt_drive", { "Letter of the drive:",
                "Lettre du lecteur :",
                "Laufwerksbuchstabe:",
                "Lettera dell'unità:",
                "Letra de la unidad:" } },
            { "opt_drive_none", { "No drive", "Pas de lecteur", "Kein Laufwerk", "Nessuna unità", "Sin unidad" } },
            { "opt_drive_note", { "A document or a folder copied onto this drive is anonymized once written, its name "
                "included; its original comes back through Restore. No driver is installed.",
                "Un document ou un dossier copié sur ce lecteur est anonymisé une fois écrit, nom compris ; son "
                "original revient par Restaurer. Aucun pilote n'est installé.",
                "Ein auf dieses Laufwerk kopiertes Dokument oder ein Ordner wird nach dem Schreiben anonymisiert, "
                "sein Name inbegriffen; das Original kommt über Wiederherstellen zurück. Kein Treiber wird "
                "installiert.",
                "Un documento o una cartella copiati su questa unità vengono anonimizzati una volta scritti, nome "
                "compreso; l'originale torna con Ripristina. Nessun driver viene installato.",
                "Un documento o una carpeta copiados en esta unidad se anonimizan una vez escritos, nombre "
                "incluido; el original vuelve con Restaurar. No se instala ningún controlador." } },
            { "opt_context_menu", { "Add Anonymize and Restore to the menu of the files and folders in the Explorer",
                "Ajouter Anonymiser et Restaurer au menu des fichiers et des dossiers dans l'Explorateur",
                "Anonymisieren und Wiederherstellen zum Menü der Dateien und Ordner im Explorer hinzufügen",
                "Aggiungi Anonimizza e Ripristina al menu dei file e delle cartelle in Esplora file",
                "Añadir Anonimizar y Restaurar al menú de los archivos y las carpetas en el Explorador" } },
            { "verb_menu", { "Pseudalis", "Pseudalis", "Pseudalis", "Pseudalis", "Pseudalis" } },
            { "verb_protect", { "Anonymize", "Anonymiser", "Anonymisieren", "Anonimizza", "Anonimizar" } },
            { "verb_protect_copy", { "Anonymize as a copy", "Anonymiser dans une copie", "Als Kopie anonymisieren",
                "Anonimizza in una copia", "Anonimizar en una copia" } },
            { "verb_restore", { "Restore the original", "Restaurer l'original", "Original wiederherstellen",
                "Ripristina l'originale", "Restaurar el original" } },
            { "verb_restore_copy", { "Restore as a copy", "Restaurer dans une copie", "Als Kopie wiederherstellen",
                "Ripristina in una copia", "Restaurar en una copia" } },
        };

        int lang_index = 0;

        int index_of(const std::string& code)
        {
            static const char* const codes[] = { "en", "fr", "de", "it", "es" };
            for (int i = 0; i < 5; ++i)
                if (code == codes[i]) return i;
            return -1;
        }

        const std::map<std::string, const entry*>& table()
        {
            static std::map<std::string, const entry*> t;
            if (t.empty())
                for (const auto& e : entries) t[e.key] = &e;
            return t;
        }

        void replace_marks(std::string& s, const std::string& name, const std::string& value)
        {
            const std::string mark = "{" + name + "}";
            for (size_t p = s.find(mark); p != std::string::npos; p = s.find(mark, p + value.size()))
                s.replace(p, mark.size(), value);
        }
    }

    std::string system_language()
    {
        const LANGID id = GetUserDefaultUILanguage();
        switch (PRIMARYLANGID(id))
        {
        case LANG_FRENCH: return "fr";
        case LANG_GERMAN: return "de";
        case LANG_ITALIAN: return "it";
        case LANG_SPANISH: return "es";
        default: return "en";
        }
    }

    void set_language(const std::string& code)
    {
        int i = index_of(code);
        if (i < 0) i = index_of(system_language());
        lang_index = i < 0 ? 0 : i;
    }

    std::string current_language()
    {
        static const char* const codes[] = { "en", "fr", "de", "it", "es" };
        return codes[lang_index];
    }

    std::string tr_utf8(const char* key)
    {
        const auto& t = table();
        auto it = t.find(key);
        if (it == t.end()) return key;
        const char* s = it->second->text[lang_index];
        return s && *s ? s : it->second->text[0];
    }

    std::wstring tr(const char* key)
    {
        return widen(tr_utf8(key));
    }

    std::wstring tr(const char* key, std::initializer_list<std::pair<const char*, std::string>> values)
    {
        std::string s = tr_utf8(key);
        for (const auto& v : values) replace_marks(s, v.first, v.second);
        return widen(s);
    }

    std::wstring tr(const std::string& key, const std::vector<std::pair<std::string, std::string>>& values)
    {
        std::string s = tr_utf8(key.c_str());
        for (const auto& v : values) replace_marks(s, v.first, v.second);
        return widen(s);
    }

    bool has_text(const std::string& key)
    {
        return table().count(key) != 0;
    }

    std::vector<std::string> texts_of(const char* key)
    {
        std::vector<std::string> out;
        const auto& t = table();
        auto it = t.find(key);
        if (it == t.end()) return out;
        for (const char* s : it->second->text)
            if (s && *s) out.push_back(s);
        return out;
    }

    std::wstring tr_count(const char* key, long long n,
        std::initializer_list<std::pair<const char*, std::string>> values)
    {
        // One thing takes the singular, and none too in French.
        const bool one = n == 1 || (n == 0 && current_language() == "fr");
        const std::string single = std::string(key) + "_one";
        std::vector<std::pair<std::string, std::string>> all(1, std::make_pair(std::string("n"), std::to_string(n)));
        for (const auto& v : values) all.push_back(std::make_pair(std::string(v.first), v.second));
        return tr(one && has_text(single) ? single : std::string(key), all);
    }

    std::wstring error_text(const json& answer)
    {
        const std::string code = answer["code"].as_string();
        const std::string english = answer["error"].as_string();
        if (code.empty() || !has_text("err_" + code)) return widen(english);
        std::string s = tr_utf8(("err_" + code).c_str());
        for (const auto& kv : answer["args"].items())
            replace_marks(s, kv.first, kv.second.is_string() ? kv.second.as_string() : kv.second.dump());
        // The sizes, in megabytes.
        for (const auto& kv : answer["sizes"].items())
            replace_marks(s, kv.first, std::to_string((kv.second.as_int(0) + (1 << 20) - 1) >> 20));
        // A mark without a value keeps the English text of the service in its place.
        replace_marks(s, "e", english);
        return widen(s);
    }

    json failed_answer(const json& answer, const std::string& error)
    {
        json r = json::object();
        r["error"] = error;
        const json& code = answer["code"].is_string() ? answer["code"] : answer["error"]["code"];
        if (code.is_string())
        {
            r["code"] = code;
            if (answer["args"].is_object()) r["args"] = answer["args"];
            if (answer["sizes"].is_object()) r["sizes"] = answer["sizes"];
        }
        else if (!answer.is_object())
        {
            // The service stopped, or listens on another port.
            r["code"] = "service";
            json args = json::object();
            args["e"] = error;
            r["args"] = args;
        }
        return r;
    }

}
