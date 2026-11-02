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

namespace albireo
{

    namespace
    {
        struct entry
        {
            const char* key;
            const char* text[5];   // en, fr, de, it, es
        };

        // The texts of the tray and of the installer.  {name} marks a value given at use.
        const entry entries[] = {
            { "tray_open", { "Open Albireo", "Ouvrir Albireo", "Albireo öffnen", "Apri Albireo", "Abrir Albireo" } },
            { "tray_models", { "Models…", "Modèles…", "Modelle…", "Modelli…", "Modelos…" } },
            { "tray_host", { "This computer…", "Cet ordinateur…", "Dieser Computer…", "Questo computer…",
                "Este equipo…" } },
            { "tray_accel", { "Accelerator", "Accélérateur", "Beschleuniger", "Acceleratore", "Acelerador" } },
            { "accel_auto", { "Automatic", "Automatique", "Automatisch", "Automatico", "Automático" } },
            { "accel_none", { "Processor only", "Processeur uniquement", "Nur Prozessor", "Solo processore",
                "Solo procesador" } },
            // A device of an engine: {k} its kind, {n} the names of its devices, {r} its runtime.
            { "accel_choice", { "{k} {n} ({r})", "{k} {n} ({r})", "{k} {n} ({r})", "{k} {n} ({r})",
                "{k} {n} ({r})" } },
            { "accel_choice_plain", { "{k} ({r})", "{k} ({r})", "{k} ({r})", "{k} ({r})", "{k} ({r})" } },
            { "device_gpu", { "GPU", "GPU", "GPU", "GPU", "GPU" } },
            { "device_npu", { "NPU", "NPU", "NPU", "NPU", "NPU" } },
            { "device_both", { "NPU + GPU", "NPU + GPU", "NPU + GPU", "NPU + GPU", "NPU + GPU" } },
            { "device_cpu", { "Processor", "Processeur", "Prozessor", "Processore", "Procesador" } },
            { "accel_cannot_run", { "{r} cannot run here: {why}", "{r} ne peut pas fonctionner ici : {why}",
                "{r} kann hier nicht laufen: {why}", "{r} non può funzionare qui: {why}",
                "{r} no puede funcionar aquí: {why}" } },
            { "accel_left_out", { "{r} left devices out: {m}", "{r} a écarté des périphériques : {m}",
                "{r} hat Geräte ausgelassen: {m}", "{r} ha escluso dei dispositivi: {m}",
                "{r} dejó fuera dispositivos: {m}" } },
            { "accel_measure", { "Measure the engines", "Mesurer les moteurs", "Engines messen", "Misura i motori",
                "Medir los motores" } },
            { "accel_not_installed", { "No accelerator engine installed: OpenVINO (Intel NPU and graphics) or CUDA "
                "(NVIDIA)", "Aucun moteur d'accélération installé : OpenVINO (NPU et graphique Intel) ou CUDA (NVIDIA)",
                "Keine Beschleuniger-Engine installiert: OpenVINO (Intel-NPU und -Grafik) oder CUDA (NVIDIA)",
                "Nessun motore di accelerazione installato: OpenVINO (NPU e grafica Intel) o CUDA (NVIDIA)",
                "Ningún motor de aceleración instalado: OpenVINO (NPU y gráficos Intel) o CUDA (NVIDIA)" } },
            { "tray_updates", { "Check for updates", "Rechercher des mises à jour", "Nach Updates suchen",
                "Cerca aggiornamenti", "Buscar actualizaciones" } },
            { "tray_install_update", { "Install version {v} and restart", "Installer la version {v} et redémarrer",
                "Version {v} installieren und neu starten", "Installa la versione {v} e riavvia",
                "Instalar la versión {v} y reiniciar" } },
            { "tray_restart", { "Restart Albireo", "Redémarrer Albireo", "Albireo neu starten", "Riavvia Albireo",
                "Reiniciar Albireo" } },
            { "tray_start_service", { "Start the service", "Démarrer le service", "Dienst starten", "Avvia il servizio",
                "Iniciar el servicio" } },
            { "tray_docs", { "Interface documentation", "Documentation de l'interface",
                "Dokumentation der Schnittstelle", "Documentazione dell'interfaccia",
                "Documentación de la interfaz" } },
            { "tray_about", { "About Albireo", "À propos d'Albireo", "Über Albireo", "Informazioni su Albireo",
                "Acerca de Albireo" } },
            { "tray_quit", { "Close the notification icon", "Fermer l'icône de notification",
                "Infobereich-Symbol schließen", "Chiudi l'icona di notifica", "Cerrar el icono de notificación" } },
            { "tip_off", { "Albireo: the service does not answer", "Albireo : le service ne répond pas",
                "Albireo: Der Dienst antwortet nicht", "Albireo: il servizio non risponde",
                "Albireo: el servicio no responde" } },
            { "tip_starting", { "Albireo: the service is starting", "Albireo : démarrage du service",
                "Albireo: Der Dienst startet", "Albireo: avvio del servizio",
                "Albireo: el servicio se está iniciando" } },
            { "tip_stopping", { "Albireo: the service is stopping", "Albireo : arrêt du service",
                "Albireo: Der Dienst wird beendet", "Albireo: arresto del servizio",
                "Albireo: el servicio se está deteniendo" } },
            { "tip_stopped", { "Albireo: the service is stopped", "Albireo : le service est arrêté",
                "Albireo: Der Dienst ist beendet", "Albireo: il servizio è arrestato",
                "Albireo: el servicio está detenido" } },
            { "tip_no_model", { "no active model", "aucun modèle actif", "kein aktives Modell", "nessun modello attivo",
                "ningún modelo activo" } },
            { "tip_loading", { "loading the model", "chargement du modèle", "Modell wird geladen",
                "caricamento del modello", "cargando el modelo" } },
            { "tip_active", { "{n} active requests", "{n} requêtes actives", "{n} aktive Anfragen",
                "{n} richieste attive", "{n} solicitudes activas" } },
            { "tip_idle", { "ready", "prêt", "bereit", "pronto", "listo" } },
            { "tip_offline", { "offline", "hors ligne", "offline", "offline", "sin conexión" } },
            { "tokens_s", { "{n} tokens/s", "{n} jetons/s", "{n} Token/s", "{n} token/s", "{n} tokens/s" } },
            { "notify_update_title", { "Update ready", "Mise à jour prête", "Update bereit", "Aggiornamento pronto",
                "Actualización lista" } },
            { "notify_update_text", { "Albireo {v} is downloaded. It will be installed at the next restart.",
                "Albireo {v} est téléchargé. Il sera installé au prochain redémarrage.",
                "Albireo {v} ist heruntergeladen. Es wird beim nächsten Neustart installiert.",
                "Albireo {v} è stato scaricato. Verrà installato al prossimo riavvio.",
                "Albireo {v} se ha descargado. Se instalará en el próximo reinicio." } },
            { "notify_uptodate", { "Albireo is up to date.", "Albireo est à jour.",
                "Albireo ist auf dem neuesten Stand.", "Albireo è aggiornato.", "Albireo está actualizado." } },
            { "notify_update_error", { "The update check failed: {e}", "La recherche de mise à jour a échoué : {e}",
                "Die Update-Suche ist fehlgeschlagen: {e}", "La ricerca di aggiornamenti non è riuscita: {e}",
                "La búsqueda de actualizaciones falló: {e}" } },
            // An update made by the installer out of sight that failed ({v} the version it brought,
            // {o} the version in place, {e} the error).
            { "notify_update_failed_title", { "Update failed", "Échec de la mise à jour",
                "Aktualisierung fehlgeschlagen", "Aggiornamento non riuscito", "Actualización fallida" } },
            { "notify_update_rolled_back", { "The update to version {v} failed and Albireo {o} was put back in "
                "place: {e}", "La mise à jour vers la version {v} a échoué et Albireo {o} a été remis en place : {e}",
                "Die Aktualisierung auf Version {v} ist fehlgeschlagen, Albireo {o} wurde wiederhergestellt: {e}",
                "L'aggiornamento alla versione {v} non è riuscito ed è stato ripristinato Albireo {o}: {e}",
                "La actualización a la versión {v} falló y se restableció Albireo {o}: {e}" } },
            { "notify_update_failed", { "The update to version {v} failed: {e}",
                "La mise à jour vers la version {v} a échoué : {e}",
                "Die Aktualisierung auf Version {v} ist fehlgeschlagen: {e}",
                "L'aggiornamento alla versione {v} non è riuscito: {e}",
                "La actualización a la versión {v} falló: {e}" } },
            { "notify_installed", { "{m} is installed.", "{m} est installé.", "{m} ist installiert.",
                "{m} è installato.", "{m} está instalado." } },
            { "notify_failed", { "The download of {m} failed: {e}", "Le téléchargement de {m} a échoué : {e}",
                "Der Download von {m} ist fehlgeschlagen: {e}", "Il download di {m} non è riuscito: {e}",
                "La descarga de {m} falló: {e}" } },
            { "notify_service_down", { "The Albireo service does not answer.", "Le service Albireo ne répond pas.",
                "Der Albireo-Dienst antwortet nicht.", "Il servizio Albireo non risponde.",
                "El servicio Albireo no responde." } },
            { "notify_service_stopped", { "The Albireo service is stopped: start it from the menu of this icon.",
                "Le service Albireo est arrêté : démarrez-le depuis le menu de cette icône.",
                "Der Albireo-Dienst ist beendet: Starten Sie ihn über das Menü dieses Symbols.",
                "Il servizio Albireo è arrestato: avviatelo dal menu di questa icona.",
                "El servicio Albireo está detenido: inícielo desde el menú de este icono." } },
            { "notify_measures", { "The engines are measured: {r}", "Les moteurs sont mesurés : {r}",
                "Die Engines sind gemessen: {r}", "I motori sono stati misurati: {r}",
                "Los motores están medidos: {r}" } },
            { "notify_restarting", { "Albireo restarts.", "Albireo redémarre.", "Albireo startet neu.",
                "Albireo si riavvia.", "Albireo se reinicia." } },

            { "models_title", { "Albireo models", "Modèles d'Albireo", "Albireo-Modelle", "Modelli di Albireo",
                "Modelos de Albireo" } },
            { "models_intro", {
                "Up to three models answer, one per function; a function without a model uses the general model, "
                "else the model that comes with Albireo. The inputs of a model decide the attachments of the web "
                "application; the document search has its own embedding model.",
                "Jusqu'à trois modèles répondent, un par fonction ; une fonction sans modèle utilise le modèle "
                "général, sinon le modèle livré avec Albireo. Les entrées d'un modèle décident des pièces jointes "
                "de l'application web ; la recherche documentaire a son propre modèle d'embeddings.",
                "Bis zu drei Modelle antworten, eines pro Funktion; eine Funktion ohne Modell verwendet das "
                "allgemeine Modell, sonst das mit Albireo gelieferte Modell. Die Eingaben eines Modells bestimmen "
                "die Anhänge der Webanwendung; die Dokumentensuche hat ihr eigenes Embedding-Modell.",
                "Fino a tre modelli rispondono, uno per funzione; una funzione senza modello usa il modello generale, "
                "altrimenti il modello fornito con Albireo. Gli ingressi di un modello decidono gli allegati "
                "dell'applicazione web; la ricerca nei documenti ha il proprio modello di embedding.",
                "Hasta tres modelos responden, uno por función; una función sin modelo usa el modelo general, si no "
                "el modelo incluido con Albireo. Las entradas de un modelo deciden los adjuntos de la aplicación "
                "web; la búsqueda en documentos tiene su propio modelo de embeddings." } },
            { "col_model", { "Model", "Modèle", "Modell", "Modello", "Modelo" } },
            // The company or the organisation that produced a model.
            { "col_publisher", { "Developer", "Producteur", "Hersteller", "Produttore", "Productor" } },
            { "col_roles", { "Functions", "Fonctions", "Funktionen", "Funzioni", "Funciones" } },
            { "col_size", { "Size", "Taille", "Größe", "Dimensione", "Tamaño" } },
            { "col_state", { "State", "État", "Status", "Stato", "Estado" } },
            { "col_active", { "Active for", "Actif pour", "Aktiv für", "Attivo per", "Activo para" } },
            { "col_inputs", { "Inputs", "Entrées", "Eingaben", "Ingressi", "Entradas" } },
            { "input_text", { "Text", "Texte", "Text", "Testo", "Texto" } },
            { "input_image", { "Image", "Image", "Bild", "Immagine", "Imagen" } },
            { "input_audio", { "Sound", "Son", "Ton", "Audio", "Sonido" } },
            { "state_builtin", { "Comes with Albireo", "Livré avec Albireo", "Mit Albireo geliefert",
                "Fornito con Albireo", "Incluido con Albireo" } },
            { "lbl_inputs", { "Inputs", "Entrées", "Eingaben", "Ingressi", "Entradas" } },
            { "lbl_publisher", { "Developer", "Producteur", "Hersteller", "Produttore", "Productor" } },
            { "role_general", { "General", "Général", "Allgemein", "Generale", "General" } },
            { "role_code", { "Code", "Code", "Code", "Codice", "Código" } },
            { "role_embedding", { "Document search", "Recherche documentaire", "Dokumentensuche",
                "Ricerca nei documenti", "Búsqueda en documentos" } },
            { "role_security", { "Cybersecurity", "Cybersécurité", "Cybersicherheit", "Cybersicurezza",
                "Ciberseguridad" } },
            { "role_background", { "Background work", "Tâches de fond", "Hintergrundaufgaben",
                "Attività in background", "Tareas en segundo plano" } },
            { "roles_all", { "All functions", "Toutes les fonctions", "Alle Funktionen", "Tutte le funzioni",
                "Todas las funciones" } },
            { "state_installed", { "Installed", "Installé", "Installiert", "Installato", "Instalado" } },
            { "state_available", { "Available", "Disponible", "Verfügbar", "Disponibile", "Disponible" } },
            { "state_downloading", { "Downloading {p} %", "Téléchargement {p} %", "Download {p} %", "Download {p} %",
                "Descargando {p} %" } },
            { "state_verifying", { "Checking", "Vérification", "Prüfung", "Verifica", "Comprobando" } },
            { "state_queued", { "Waiting", "En attente", "Wartet", "In attesa", "En espera" } },
            { "state_error", { "Error", "Erreur", "Fehler", "Errore", "Error" } },
            { "state_cancelled", { "Cancelled", "Annulé", "Abgebrochen", "Annullato", "Cancelado" } },
            { "state_unloading", { "Unloading", "Déchargement", "Wird entladen", "Rilascio in corso",
                "Liberando memoria" } },
            { "state_removing", { "Deleting", "Suppression", "Wird gelöscht", "Eliminazione", "Eliminando" } },
            // The errors of the service, by their code ({m}: the model).
            { "err_stopping", { "The service is stopping.", "Le service s'arrête.", "Der Dienst wird beendet.",
                "Il servizio si sta arrestando.", "El servicio se está deteniendo." } },
            { "err_offline", { "Offline mode: the catalog, the models and the updates come from imports.",
                "Mode hors ligne : le catalogue, les modèles et les mises à jour viennent d'imports.",
                "Offline-Modus: Katalog, Modelle und Updates kommen aus Importen.",
                "Modalità offline: il catalogo, i modelli e gli aggiornamenti provengono da importazioni.",
                "Modo sin conexión: el catálogo, los modelos y las actualizaciones proceden de importaciones." } },
            { "err_service", { "The Albireo service does not answer ({e}).", "Le service Albireo ne répond pas ({e}).",
                "Der Albireo-Dienst antwortet nicht ({e}).", "Il servizio Albireo non risponde ({e}).",
                "El servicio Albireo no responde ({e})." } },
            { "err_port_unavailable", { "The port {port} is in use or cannot be opened: choose another one.",
                "Le port {port} est utilisé ou ne peut pas être ouvert : choisissez-en un autre.",
                "Der Port {port} wird verwendet oder kann nicht geöffnet werden: Wählen Sie einen anderen.",
                "La porta {port} è in uso o non può essere aperta: sceglietene un'altra.",
                "El puerto {port} está en uso o no se puede abrir: elija otro." } },
            { "err_proxy_password", { "The masked password of the proxy applies to its saved address and user only: "
                "type the password again.", "Le mot de passe masqué du proxy ne vaut que pour l'adresse et "
                "l'utilisateur enregistrés : saisissez à nouveau le mot de passe.", "Das maskierte Kennwort des "
                "Proxys gilt nur für die gespeicherte Adresse und den gespeicherten Benutzer: Geben Sie das Kennwort "
                "erneut ein.", "La password mascherata del proxy vale solo per l'indirizzo e l'utente salvati: "
                "digitate di nuovo la password.", "La contraseña oculta del proxy solo sirve para la dirección y el "
                "usuario guardados: escriba de nuevo la contraseña." } },
            { "err_not_in_catalog", { "{m} is not in the catalog.", "{m} ne figure pas dans le catalogue.",
                "{m} ist nicht im Katalog.", "{m} non è nel catalogo.", "{m} no está en el catálogo." } },
            { "err_catalog_entry", { "The catalog entry of {m} is not valid; update the catalog.",
                "L'entrée du catalogue pour {m} n'est pas valide ; actualisez le catalogue.",
                "Der Katalogeintrag von {m} ist ungültig; aktualisieren Sie den Katalog.",
                "La voce del catalogo di {m} non è valida; aggiornate il catalogo.",
                "La entrada del catálogo de {m} no es válida; actualice el catálogo." } },
            { "err_disk_space", { "Not enough space in the models folder for {m}: {needed} are needed, {model} for "
                "the model and {reserve} kept free for the system, and {available} are free. Free some space, or "
                "choose a models folder on another drive in This computer.",
                "Espace insuffisant dans le dossier des modèles pour {m} : il faut {needed}, soit {model} pour le "
                "modèle et {reserve} laissés libres pour le système, et {available} sont libres. Libérez de la "
                "place, ou choisissez un dossier des modèles sur un autre disque dans Cet ordinateur.",
                "Nicht genug Platz im Modellordner für {m}: Benötigt werden {needed}, {model} für das Modell und "
                "{reserve}, die für das System frei bleiben, und {available} sind frei. Geben Sie Speicherplatz "
                "frei oder wählen Sie unter Dieser Computer einen Modellordner auf einem anderen Laufwerk.",
                "Spazio insufficiente nella cartella dei modelli per {m}: servono {needed}, {model} per il modello "
                "e {reserve} lasciati liberi per il sistema, e sono liberi {available}. Liberate spazio o scegliete "
                "una cartella dei modelli su un altro disco in Questo computer.",
                "No hay espacio suficiente en la carpeta de modelos para {m}: se necesitan {needed}, {model} para "
                "el modelo y {reserve} que quedan libres para el sistema, y hay {available} libres. Libere espacio "
                "o elija una carpeta de modelos en otra unidad en Este equipo." } },
            { "err_unknown_role", { "Unknown function: {role}.", "Fonction inconnue : {role}.",
                "Unbekannte Funktion: {role}.", "Funzione sconosciuta: {role}.", "Función desconocida: {role}." } },
            { "err_not_installed", { "{m} is not installed.", "{m} n'est pas installé.", "{m} ist nicht installiert.",
                "{m} non è installato.", "{m} no está instalado." } },
            { "err_being_deleted", { "{m} is being deleted, once its requests in progress are served.",
                "{m} est en cours de suppression, une fois servies ses requêtes en cours.",
                "{m} wird gelöscht, sobald seine laufenden Anfragen beantwortet sind.",
                "{m} è in corso di eliminazione, appena servite le sue richieste in corso.",
                "{m} se está eliminando, en cuanto se atiendan sus solicitudes en curso." } },
            { "err_needs_embedding_model", { "{m} writes text: the document search needs an embedding model.",
                "{m} rédige du texte : la recherche documentaire demande un modèle d'embeddings.",
                "{m} schreibt Text: Die Dokumentensuche braucht ein Embedding-Modell.",
                "{m} scrive testo: la ricerca nei documenti richiede un modello di embedding.",
                "{m} escribe texto: la búsqueda en documentos necesita un modelo de embeddings." } },
            { "err_embedding_model_only", { "{m} computes the vectors of the document search and answers no "
                "conversation.", "{m} calcule les vecteurs de la recherche documentaire et ne répond à aucune "
                "conversation.", "{m} berechnet die Vektoren der Dokumentensuche und beantwortet keine Unterhaltung.",
                "{m} calcola i vettori della ricerca nei documenti e non risponde ad alcuna conversazione.",
                "{m} calcula los vectores de la búsqueda en documentos y no responde a ninguna conversación." } },
            { "err_load_failed", { "{m} could not be loaded: {e}", "{m} n'a pas pu être chargé : {e}",
                "{m} konnte nicht geladen werden: {e}", "{m} non è stato caricato: {e}",
                "{m} no se pudo cargar: {e}" } },
            { "err_embedding_needed", { "The document search needs an embedding model: activate another one for this "
                "function to replace {m}.", "La recherche documentaire a besoin d'un modèle d'embeddings : activez-en "
                "un autre pour cette fonction afin de remplacer {m}.", "Die Dokumentensuche braucht ein "
                "Embedding-Modell: Aktivieren Sie ein anderes für diese Funktion, um {m} zu ersetzen.",
                "La ricerca nei documenti ha bisogno di un modello di embedding: attivatene un altro per questa "
                "funzione per sostituire {m}.", "La búsqueda en documentos necesita un modelo de embeddings: active "
                "otro para esta función para sustituir a {m}." } },
            { "err_builtin_model", { "{m} comes with Albireo: it can leave its functions to other models, it "
                "cannot be deleted.", "{m} est livré avec Albireo : il peut laisser ses fonctions à d'autres modèles, "
                "il ne peut pas être supprimé.", "{m} wird mit Albireo ausgeliefert: Es kann seine Funktionen anderen "
                "Modellen überlassen, aber nicht gelöscht werden.", "{m} è fornito con Albireo: può lasciare le sue "
                "funzioni ad altri modelli, non può essere eliminato.", "{m} se incluye con Albireo: puede dejar sus "
                "funciones a otros modelos, no se puede eliminar." } },
            { "err_builtin_installed", { "{m} comes with Albireo and is installed already.",
                "{m} est livré avec Albireo et déjà installé.",
                "{m} wird mit Albireo ausgeliefert und ist bereits installiert.",
                "{m} è fornito con Albireo ed è già installato.", "{m} se incluye con Albireo y ya está instalado." } },
            { "err_delete_failed", { "{m} could not be deleted: its file is in use or protected ({file}).",
                "{m} n'a pas pu être supprimé : son fichier est utilisé ou protégé ({file}).",
                "{m} konnte nicht gelöscht werden: Seine Datei ist in Gebrauch oder geschützt ({file}).",
                "{m} non è stato eliminato: il suo file è in uso o protetto ({file}).",
                "{m} no se pudo eliminar: su archivo está en uso o protegido ({file})." } },
            { "err_download_failed", { "The download failed: {e}", "Le téléchargement a échoué : {e}",
                "Der Download ist fehlgeschlagen: {e}", "Il download non è riuscito: {e}",
                "La descarga ha fallado: {e}" } },
            { "err_download_file", { "The file of the download cannot be created: {file}",
                "Le fichier du téléchargement ne peut pas être créé : {file}",
                "Die Datei des Downloads kann nicht angelegt werden: {file}",
                "Il file del download non può essere creato: {file}",
                "No se puede crear el archivo de la descarga: {file}" } },
            { "err_download_hash", { "The downloaded file does not match the fingerprint of the catalog; it was "
                "deleted.", "Le fichier téléchargé ne correspond pas à l'empreinte du catalogue ; il a été supprimé.",
                "Die heruntergeladene Datei entspricht nicht dem Fingerabdruck des Katalogs; sie wurde gelöscht.",
                "Il file scaricato non corrisponde all'impronta del catalogo; è stato eliminato.",
                "El archivo descargado no coincide con la huella del catálogo; se ha eliminado." } },
            { "err_download_not_published", { "The publication space of Albireo does not hold {file} yet: the "
                "model can be downloaded once it is published there.", "L'espace de publication d'Albireo ne "
                "contient pas encore {file} : le modèle pourra être téléchargé une fois publié.", "Der "
                "Veröffentlichungsbereich von Albireo enthält {file} noch nicht: Das Modell kann heruntergeladen "
                "werden, sobald es dort veröffentlicht ist.", "Lo spazio di pubblicazione di Albireo non contiene "
                "ancora {file}: il modello potrà essere scaricato una volta pubblicato.", "El espacio de publicación "
                "de Albireo aún no contiene {file}: el modelo se podrá descargar una vez publicado." } },
            { "err_download_install", { "The downloaded model could not be installed: {e}",
                "Le modèle téléchargé n'a pas pu être installé : {e}",
                "Das heruntergeladene Modell konnte nicht installiert werden: {e}",
                "Il modello scaricato non è stato installato: {e}", "El modelo descargado no se pudo instalar: {e}" } },
            // The refusals of the imports by path, by their code ({path}: the path asked for).
            { "err_import_path_relative", { "The path {path} is relative: an import names a folder by its full path.",
                "Le chemin {path} est relatif : un import désigne un dossier par son chemin complet.",
                "Der Pfad {path} ist relativ: Ein Import nennt einen Ordner mit seinem vollständigen Pfad.",
                "Il percorso {path} è relativo: un'importazione indica una cartella con il suo percorso completo.",
                "La ruta {path} es relativa: una importación indica una carpeta con su ruta completa." } },
            { "err_import_path_network", { "{path} is on a network drive or share: the imports read the local drives "
                "only.", "{path} se trouve sur un lecteur ou un partage réseau : les imports ne lisent que les disques "
                "locaux.", "{path} liegt auf einem Netzlaufwerk oder einer Netzwerkfreigabe: Importe lesen nur lokale "
                "Laufwerke.", "{path} si trova su un'unità o una condivisione di rete: le importazioni leggono solo i "
                "dischi locali.", "{path} está en una unidad o un recurso compartido de red: las importaciones solo "
                "leen las unidades locales." } },
            { "err_import_path_device", { "{path} names a device or a namespace of the system: the imports read "
                "folders of the local drives only.", "{path} désigne un périphérique ou un espace de noms du "
                "système : les imports ne lisent que des dossiers des disques locaux.", "{path} bezeichnet ein Gerät "
                "oder einen Namensraum des Systems: Importe lesen nur Ordner lokaler Laufwerke.", "{path} indica un "
                "dispositivo o uno spazio dei nomi del sistema: le importazioni leggono solo cartelle dei dischi "
                "locali.", "{path} designa un dispositivo o un espacio de nombres del sistema: las importaciones solo "
                "leen carpetas de las unidades locales." } },
            { "err_import_path_invalid", { "The path {path} is not valid for an import.",
                "Le chemin {path} n'est pas valide pour un import.",
                "Der Pfad {path} ist für einen Import nicht gültig.",
                "Il percorso {path} non è valido per un'importazione.",
                "La ruta {path} no es válida para una importación." } },
            { "err_import_use_upload", { "{path} is a model file: send it with Import…, then Model files (*.gguf).",
                "{path} est un fichier de modèle : envoyez-le avec Importer…, puis Fichiers de modèle (*.gguf).",
                "{path} ist eine Modelldatei: Senden Sie sie mit Importieren… und dann Modelldateien (*.gguf).",
                "{path} è un file di modello: inviatelo con Importa…, poi File di modello (*.gguf).",
                "{path} es un archivo de modelo: envíelo con Importar… y luego Archivos de modelo (*.gguf)." } },
            { "err_import_not_found", { "{path} is not found in the import folder.",
                "{path} est introuvable dans le dossier d'import.", "{path} wurde im Importordner nicht gefunden.",
                "{path} non si trova nella cartella di importazione.",
                "{path} no se encuentra en la carpeta de importación." } },
            { "err_import_path_link", { "{path} is reached through a link, which the imports never follow.",
                "{path} est atteint par un lien, que les imports ne suivent jamais.",
                "{path} wird über eine Verknüpfung erreicht, der Importe nie folgen.",
                "{path} è raggiunto tramite un collegamento, che le importazioni non seguono mai.",
                "{path} se alcanza mediante un enlace, que las importaciones nunca siguen." } },
            { "err_import_folder_insecure", { "The import folder lets other accounts than the administrators write in "
                "it and cannot be secured: nothing is imported from it.", "Le dossier d'import permet à d'autres "
                "comptes que les administrateurs d'y écrire et ne peut pas être sécurisé : rien n'en est importé.",
                "In den Importordner können andere Konten als die Administratoren schreiben, und er lässt sich nicht "
                "absichern: Aus ihm wird nichts importiert.", "La cartella di importazione consente di scrivere ad "
                "account diversi dagli amministratori e non può essere protetta: non se ne importa nulla.", "La "
                "carpeta de importación permite escribir a cuentas distintas de los administradores y no se puede "
                "proteger: no se importa nada de ella." } },
            { "err_import_not_bundle", { "{path} is no signed offline bundle of Albireo on a local drive (catalog.json "
                "and catalog.json.sig, or feed.json and feed.json.sig). Model files are sent with Import…, then Model "
                "files (*.gguf).", "{path} n'est pas un paquet hors ligne signé d'Albireo sur un disque local "
                "(catalog.json et catalog.json.sig, ou feed.json et feed.json.sig). Les fichiers de modèle s'envoient "
                "avec Importer…, puis Fichiers de modèle (*.gguf).", "{path} ist kein signiertes Offline-Paket von "
                "Albireo auf einem lokalen Laufwerk (catalog.json und catalog.json.sig oder feed.json und "
                "feed.json.sig). Modelldateien werden mit Importieren… und dann Modelldateien (*.gguf) gesendet.",
                "{path} non è un pacchetto offline firmato di Albireo su un disco locale (catalog.json e "
                "catalog.json.sig, oppure feed.json e feed.json.sig). I file di modello si inviano con Importa…, poi "
                "File di modello (*.gguf).", "{path} no es un paquete sin conexión firmado de Albireo en una unidad "
                "local (catalog.json y catalog.json.sig, o feed.json y feed.json.sig). Los archivos de modelo se "
                "envían con Importar… y luego Archivos de modelo (*.gguf)." } },
            // The errors of the updates, by their code.
            { "err_update_check", { "The update check failed: {e}", "La recherche de mise à jour a échoué : {e}",
                "Die Update-Suche ist fehlgeschlagen: {e}", "La ricerca di aggiornamenti non è riuscita: {e}",
                "La búsqueda de actualizaciones falló: {e}" } },
            { "err_update_record", { "The update to version {version} cannot be recorded in the updates folder.",
                "La mise à jour vers la version {version} ne peut pas être enregistrée dans le dossier des mises à "
                "jour.", "Die Aktualisierung auf Version {version} kann nicht im Update-Ordner gespeichert werden.",
                "L'aggiornamento alla versione {version} non può essere registrato nella cartella degli "
                "aggiornamenti.", "La actualización a la versión {version} no se puede registrar en la carpeta de "
                "actualizaciones." } },
            { "err_update_feed_missing", { "No signed update feed lies beside the installer of version {version}.",
                "Aucun flux de mise à jour signé n'accompagne le programme d'installation de la version {version}.",
                "Neben dem Installationsprogramm der Version {version} liegt kein signierter Update-Feed.",
                "Accanto al programma di installazione della versione {version} non c'è alcun feed di aggiornamento "
                "firmato.", "Junto al programa de instalación de la versión {version} no hay ningún canal de "
                "actualización firmado." } },
            { "err_update_feed_missing_folder", { "{folder} holds no signed update feed (feed.json and feed.json.sig).",
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
                "aggiornamenti.", "El programa de instalación de la actualización {file} está fuera de la carpeta de "
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
            { "err_update_not_published", { "The publication space of Albireo does not hold {file} yet: the "
                "update comes once it is published there.", "L'espace de publication d'Albireo ne contient pas "
                "encore {file} : la mise à jour viendra une fois ce fichier publié.", "Der "
                "Veröffentlichungsbereich von Albireo enthält {file} noch nicht: Die Aktualisierung folgt, sobald die "
                "Datei dort veröffentlicht ist.", "Lo spazio di pubblicazione di Albireo non contiene ancora {file}: "
                "l'aggiornamento arriverà una volta pubblicato il file.", "El espacio de publicación de Albireo aún "
                "no contiene {file}: la actualización llegará una vez publicado el archivo." } },
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
            // Why an engine cannot run on this computer, by its code.
            { "why_isa_avx2", { "the processor lacks AVX2, FMA or F16C", "le processeur ne dispose pas d'AVX2, FMA "
                "ou F16C", "dem Prozessor fehlen AVX2, FMA oder F16C", "il processore non dispone di AVX2, FMA o F16C",
                "el procesador no dispone de AVX2, FMA o F16C" } },
            { "why_isa_avx_vnni", { "the processor lacks AVX-VNNI (Intel Core from the 12th generation on, AMD Zen 5)",
                "le processeur ne dispose pas d'AVX-VNNI (Intel Core à partir de la 12e génération, AMD Zen 5)",
                "dem Prozessor fehlt AVX-VNNI (Intel Core ab der 12. Generation, AMD Zen 5)",
                "il processore non dispone di AVX-VNNI (Intel Core dalla 12ª generazione, AMD Zen 5)",
                "el procesador no dispone de AVX-VNNI (Intel Core desde la 12.ª generación, AMD Zen 5)" } },
            { "why_isa_avx512", { "the processor lacks AVX-512 with VNNI (Intel Xeon from Cascade Lake on, Intel Core "
                "Ice Lake, Tiger Lake and Rocket Lake, AMD Zen 4 and later)", "le processeur ne dispose pas "
                "d'AVX-512 avec VNNI (Intel Xeon depuis Cascade Lake, Intel Core Ice Lake, Tiger Lake et Rocket Lake, "
                "AMD Zen 4 et suivants)", "dem Prozessor fehlt AVX-512 mit VNNI (Intel Xeon ab Cascade Lake, Intel "
                "Core Ice Lake, Tiger Lake und Rocket Lake, AMD Zen 4 und neuer)", "il processore non dispone di "
                "AVX-512 con VNNI (Intel Xeon da Cascade Lake, Intel Core Ice Lake, Tiger Lake e Rocket Lake, AMD "
                "Zen 4 e successivi)", "el procesador no dispone de AVX-512 con VNNI (Intel Xeon desde Cascade Lake, "
                "Intel Core Ice Lake, Tiger Lake y Rocket Lake, AMD Zen 4 y posteriores)" } },
            { "why_no_nvidia_driver", { "no NVIDIA driver is installed", "aucun pilote NVIDIA n'est installé",
                "kein NVIDIA-Treiber installiert", "nessun driver NVIDIA installato",
                "no hay ningún controlador NVIDIA instalado" } },
            { "why_old_nvidia_driver", { "the NVIDIA driver supports {driver}, this engine needs {runtime}: update "
                "the driver", "le pilote NVIDIA prend en charge {driver}, ce moteur demande {runtime} : mettez à jour "
                "le pilote", "der NVIDIA-Treiber unterstützt {driver}, diese Engine braucht {runtime}: Treiber "
                "aktualisieren", "il driver NVIDIA supporta {driver}, questo motore richiede {runtime}: aggiornate "
                "il driver", "el controlador NVIDIA admite {driver}, este motor necesita {runtime}: actualice el "
                "controlador" } },
            { "why_no_nvidia_card", { "no NVIDIA graphics card was found", "aucune carte graphique NVIDIA n'a été "
                "trouvée", "keine NVIDIA-Grafikkarte gefunden", "nessuna scheda grafica NVIDIA trovata",
                "no se ha encontrado ninguna tarjeta gráfica NVIDIA" } },
            { "why_old_nvidia_card", { "{card} predates the cards of this engine (compute capability {cc} and later)",
                "{card} est antérieure aux cartes de ce moteur (capacité de calcul {cc} et suivantes)",
                "{card} ist älter als die Karten dieser Engine (Compute Capability {cc} und höher)",
                "{card} è precedente alle schede di questo motore (capacità di calcolo {cc} e successive)",
                "{card} es anterior a las tarjetas de este motor (capacidad de cómputo {cc} y posteriores)" } },
            { "why_no_intel_device", { "no NPU or Intel graphics processor was found", "aucun NPU ni processeur "
                "graphique Intel n'a été trouvé", "keine NPU und kein Intel-Grafikprozessor gefunden",
                "nessuna NPU né processore grafico Intel trovato",
                "no se ha encontrado ninguna NPU ni procesador gráfico Intel" } },
            { "why_intel_device_unusable", { "no NPU or Intel graphics processor could be used: {detail}",
                "aucun NPU ni processeur graphique Intel n'a pu être utilisé : {detail}",
                "keine NPU und kein Intel-Grafikprozessor nutzbar: {detail}",
                "nessuna NPU né processore grafico Intel utilizzabile: {detail}",
                "no se ha podido usar ninguna NPU ni procesador gráfico Intel: {detail}" } },
            { "why_library_failed", { "the library could not be loaded (code {code})", "la bibliothèque n'a pas pu "
                "être chargée (code {code})", "die Bibliothek konnte nicht geladen werden (Code {code})",
                "la libreria non è stata caricata (codice {code})",
                "no se pudo cargar la biblioteca (código {code})" } },
            { "why_probe_timeout", { "the engine did not describe itself within {seconds} seconds",
                "le moteur ne s'est pas décrit en {seconds} secondes",
                "die Engine hat sich nicht innerhalb von {seconds} Sekunden beschrieben",
                "il motore non si è descritto entro {seconds} secondi",
                "el motor no se describió en {seconds} segundos" } },
            { "state_update", { "Update available", "Mise à jour disponible", "Update verfügbar",
                "Aggiornamento disponibile", "Actualización disponible" } },
            { "btn_download", { "Download", "Télécharger", "Herunterladen", "Scarica", "Descargar" } },
            { "btn_cancel_download", { "Cancel the download", "Annuler le téléchargement", "Download abbrechen",
                "Annulla il download", "Cancelar la descarga" } },
            { "btn_activate", { "Activate for", "Activer pour", "Aktivieren für", "Attiva per", "Activar para" } },
            { "btn_deactivate", { "Deactivate", "Désactiver", "Deaktivieren", "Disattiva", "Desactivar" } },
            { "btn_verify", { "Check the hash", "Vérifier l'empreinte", "Prüfsumme prüfen", "Verifica l'impronta",
                "Comprobar la huella" } },
            { "btn_delete", { "Delete", "Supprimer", "Löschen", "Elimina", "Eliminar" } },
            { "btn_import", { "Import…", "Importer…", "Importieren…", "Importa…", "Importar…" } },
            { "btn_refresh", { "Refresh the catalog", "Actualiser le catalogue", "Katalog aktualisieren",
                "Aggiorna il catalogo", "Actualizar el catálogo" } },
            { "import_files", { "Model files (*.gguf)", "Fichiers de modèle (*.gguf)", "Modelldateien (*.gguf)",
                "File di modello (*.gguf)", "Archivos de modelo (*.gguf)" } },
            { "import_folder_menu", { "Offline bundle or import folder…", "Paquet hors ligne ou dossier d'import…",
                "Offline-Paket oder Importordner…", "Pacchetto offline o cartella di importazione…",
                "Paquete sin conexión o carpeta de importación…" } },
            { "import_done", { "{n} models imported.", "{n} modèles importés.", "{n} Modelle importiert.",
                "{n} modelli importati.", "{n} modelos importados." } },
            { "import_already", { "{m} is installed already: it is not copied again.",
                "{m} est déjà installé : il n'est pas copié à nouveau.",
                "{m} ist bereits installiert: Es wird nicht erneut kopiert.",
                "{m} è già installato: non viene copiato di nuovo.", "{m} ya está instalado: no se copia de nuevo." } },
            { "import_already_file", { "{f} holds {m}, installed already: it is not copied again.",
                "{f} contient {m}, déjà installé : il n'est pas copié à nouveau.",
                "{f} enthält {m}, das bereits installiert ist: Es wird nicht erneut kopiert.",
                "{f} contiene {m}, già installato: non viene copiato di nuovo.",
                "{f} contiene {m}, ya instalado: no se copia de nuevo." } },
            { "import_progress", { "Importing {f}… {p} %", "Import de {f}… {p} %", "{f} wird importiert… {p} %",
                "Importazione di {f}… {p} %", "Importando {f}… {p} %" } },
            { "import_catalog", { "Catalog installed: {n} models.", "Catalogue installé : {n} modèles.",
                "Katalog installiert: {n} Modelle.", "Catalogo installato: {n} modelli.",
                "Catálogo instalado: {n} modelos." } },
            { "import_catalog_kept", { "The catalog in use was published later and stays in use.",
                "Le catalogue en service a été publié plus tard et reste en service.",
                "Der verwendete Katalog wurde später veröffentlicht und bleibt in Gebrauch.",
                "Il catalogo in uso è stato pubblicato più tardi e resta in uso.",
                "El catálogo en uso se publicó más tarde y sigue en uso." } },
            { "import_update_ready", { "Version {v} is ready: it is installed at the next restart of Albireo.",
                "La version {v} est prête : elle s'installe au prochain redémarrage d'Albireo.",
                "Version {v} ist bereit: Sie wird beim nächsten Neustart von Albireo installiert.",
                "La versione {v} è pronta: viene installata al prossimo riavvio di Albireo.",
                "La versión {v} está lista: se instala en el próximo reinicio de Albireo." } },
            { "import_update_current", { "This installation is up to date (the folder holds version {v}).",
                "Cette installation est à jour (le dossier contient la version {v}).",
                "Diese Installation ist aktuell (der Ordner enthält Version {v}).",
                "Questa installazione è aggiornata (la cartella contiene la versione {v}).",
                "Esta instalación está actualizada (la carpeta contiene la versión {v})." } },
            { "verify_ok", { "The local file matches its recorded hash.",
                "Le fichier local correspond à son empreinte enregistrée.",
                "Die lokale Datei entspricht ihrer gespeicherten Prüfsumme.",
                "Il file locale corrisponde alla sua impronta registrata.",
                "El archivo local coincide con su huella registrada." } },
            { "verify_catalog_ok", { "It matches the catalog too.", "Il correspond aussi au catalogue.",
                "Sie entspricht auch dem Katalog.", "Corrisponde anche al catalogo.",
                "También coincide con el catálogo." } },
            { "verify_catalog_bad", { "It differs from the catalog: an update is available.",
                "Il diffère du catalogue : une mise à jour est disponible.",
                "Sie weicht vom Katalog ab: Ein Update ist verfügbar.",
                "Differisce dal catalogo: è disponibile un aggiornamento.",
                "Difiere del catálogo: hay una actualización disponible." } },
            { "verify_bad", { "The local file does not match its hash. Delete it and download it again.",
                "Le fichier local ne correspond pas à son empreinte. Supprimez-le et téléchargez-le à nouveau.",
                "Die lokale Datei entspricht nicht ihrer Prüfsumme. Löschen Sie sie und laden Sie sie erneut herunter.",
                "Il file locale non corrisponde alla sua impronta. Eliminalo e scaricalo di nuovo.",
                "El archivo local no coincide con su huella. Elimínelo y descárguelo de nuevo." } },
            { "delete_confirm", { "Delete {m} from this computer?", "Supprimer {m} de cet ordinateur ?",
                "{m} von diesem Computer löschen?", "Eliminare {m} da questo computer?",
                "¿Eliminar {m} de este equipo?" } },
            { "lbl_license", { "License", "Licence", "Lizenz", "Licenza", "Licencia" } },
            { "lbl_parameters", { "Parameters", "Paramètres", "Parameter", "Parametri", "Parámetros" } },
            { "lbl_quantization", { "Quantization", "Quantification", "Quantisierung", "Quantizzazione",
                "Cuantización" } },
            { "lbl_context", { "Context", "Contexte", "Kontext", "Contesto", "Contexto" } },
            { "lbl_source", { "Source", "Source", "Quelle", "Fonte", "Fuente" } },
            { "lbl_sha256", { "SHA-256", "SHA-256", "SHA-256", "SHA-256", "SHA-256" } },
            { "lbl_free_space", { "Free space in the models folder: {s}",
                "Espace libre dans le dossier des modèles : {s}", "Freier Speicher im Modellordner: {s}",
                "Spazio libero nella cartella dei modelli: {s}", "Espacio libre en la carpeta de modelos: {s}" } },
            { "no_catalog", { "The catalog could not be fetched; the installed models are listed.",
                "Le catalogue n'a pas pu être récupéré ; les modèles installés sont listés.",
                "Der Katalog konnte nicht abgerufen werden; die installierten Modelle werden angezeigt.",
                "Non è stato possibile scaricare il catalogo; sono elencati i modelli installati.",
                "No se pudo obtener el catálogo; se muestran los modelos instalados." } },
            { "catalog_checking", { "Fetching the catalog…", "Récupération du catalogue…", "Katalog wird abgerufen…",
                "Recupero del catalogo…", "Obteniendo el catálogo…" } },
            { "catalog_error", { "The catalog could not be refreshed: {e}",
                "Le catalogue n'a pas pu être actualisé : {e}", "Der Katalog konnte nicht aktualisiert werden: {e}",
                "Non è stato possibile aggiornare il catalogo: {e}", "No se pudo actualizar el catálogo: {e}" } },
            { "catalog_offline", {
                "Offline mode: the catalog, the models and the updates come from a folder (Import…).",
                "Mode hors ligne : le catalogue, les modèles et les mises à jour viennent d'un dossier (Importer…).",
                "Offlinemodus: Katalog, Modelle und Updates kommen aus einem Ordner (Importieren…).",
                "Modalità offline: catalogo, modelli e aggiornamenti provengono da una cartella (Importa…).",
                "Modo sin conexión: el catálogo, los modelos y las actualizaciones vienen de una carpeta "
                "(Importar…)." } },

            { "host_title", { "This computer", "Cet ordinateur", "Dieser Computer", "Questo computer",
                "Este equipo" } },
            { "sec_computer", { "Computer", "Ordinateur", "Computer", "Computer", "Equipo" } },
            { "sec_engine", { "Inference engine", "Moteur d'inférence", "Inferenz-Engine", "Motore di inferenza",
                "Motor de inferencia" } },
            { "sec_memory", { "Graphics memory (V²RAM)", "Mémoire graphique (V²RAM)", "Grafikspeicher (V²RAM)",
                "Memoria grafica (V²RAM)", "Memoria gráfica (V²RAM)" } },
            { "sec_storage", { "Models folder", "Dossier des modèles", "Modellordner", "Cartella dei modelli",
                "Carpeta de modelos" } },
            { "sec_server", { "Local server", "Serveur local", "Lokaler Server", "Server locale", "Servidor local" } },
            { "sec_preferences", { "Preferences", "Préférences", "Einstellungen", "Preferenze", "Preferencias" } },
            { "lbl_name", { "Name", "Nom", "Name", "Nome", "Nombre" } },
            { "lbl_os", { "System", "Système", "System", "Sistema", "Sistema" } },
            { "lbl_cpu", { "Processor", "Processeur", "Prozessor", "Processore", "Procesador" } },
            { "lbl_cores", { "Cores / threads", "Cœurs / fils", "Kerne / Threads", "Core / thread",
                "Núcleos / hilos" } },
            { "lbl_features", { "Instructions", "Instructions", "Befehlssätze", "Istruzioni", "Instrucciones" } },
            { "lbl_memory", { "Memory", "Mémoire", "Arbeitsspeicher", "Memoria", "Memoria" } },
            { "lbl_free", { "free", "libre", "frei", "libera", "libre" } },
            { "lbl_engine", { "Engine in use", "Moteur utilisé", "Verwendete Engine", "Motore in uso",
                "Motor en uso" } },
            { "lbl_speed", { "Speed", "Vitesse", "Geschwindigkeit", "Velocità", "Velocidad" } },
            { "col_engine", { "Engine", "Moteur", "Engine", "Motore", "Motor" } },
            { "col_version", { "Version", "Version", "Version", "Versione", "Versión" } },
            { "col_devices", { "Devices", "Périphériques", "Geräte", "Dispositivi", "Dispositivos" } },
            { "col_measure", { "Measure", "Mesure", "Messung", "Misura", "Medida" } },
            { "usable", { "usable", "utilisable", "verwendbar", "utilizzabile", "utilizable" } },
            { "in_use", { "in use", "utilisé", "in Verwendung", "in uso", "en uso" } },
            { "unavailable", { "unavailable", "indisponible", "nicht verfügbar", "non disponibile", "no disponible" } },
            { "devices_left_out", { "devices left out: {m}", "périphériques écartés : {m}",
                "ausgelassene Geräte: {m}", "dispositivi esclusi: {m}", "dispositivos descartados: {m}" } },
            { "lbl_v2ram", { "Extend the graphics memory with the system memory and the disk (V²RAM)",
                "Étendre la mémoire graphique avec la mémoire système et le disque (V²RAM)",
                "Grafikspeicher mit Arbeitsspeicher und Datenträger erweitern (V²RAM)",
                "Estendi la memoria grafica con la memoria di sistema e il disco (V²RAM)",
                "Ampliar la memoria gráfica con la memoria del sistema y el disco (V²RAM)" } },
            { "lbl_v2ram_max", { "Maximum size (MB, 0 = automatic)", "Taille maximale (Mo, 0 = automatique)",
                "Maximale Größe (MB, 0 = automatisch)", "Dimensione massima (MB, 0 = automatica)",
                "Tamaño máximo (MB, 0 = automático)" } },
            { "v2ram_device", { "on the device", "sur le périphérique", "auf dem Gerät", "sul dispositivo",
                "en el dispositivo" } },
            { "v2ram_host", { "in system memory", "en mémoire système", "im Arbeitsspeicher",
                "nella memoria di sistema", "en la memoria del sistema" } },
            { "v2ram_disk", { "on disk", "sur disque", "auf dem Datenträger", "su disco", "en disco" } },
            { "v2ram_no_cuda", { "V²RAM extends the memory of an NVIDIA graphics card for the CUDA engine, which is "
                "not usable on this computer.", "V²RAM étend la mémoire d'une carte graphique NVIDIA pour le moteur "
                "CUDA, qui n'est pas utilisable sur cet ordinateur.", "V²RAM erweitert den Speicher einer "
                "NVIDIA-Grafikkarte für die CUDA-Engine, die auf diesem Computer nicht verwendbar ist.", "V²RAM "
                "estende la memoria di una scheda grafica NVIDIA per il motore CUDA, che non è utilizzabile su "
                "questo computer.", "V²RAM amplía la memoria de una tarjeta gráfica NVIDIA para el motor CUDA, que "
                "no se puede usar en este equipo." } },
            { "v2ram_none", { "The engine in use keeps its data in system memory.",
                "Le moteur utilisé garde ses données en mémoire système.",
                "Die verwendete Engine hält ihre Daten im Arbeitsspeicher.",
                "Il motore in uso tiene i suoi dati nella memoria di sistema.",
                "El motor en uso mantiene sus datos en la memoria del sistema." } },
            { "lbl_port", { "Port", "Port", "Port", "Porta", "Puerto" } },
            { "lbl_url", { "Address", "Adresse", "Adresse", "Indirizzo", "Dirección" } },
            { "lbl_language", { "Language", "Langue", "Sprache", "Lingua", "Idioma" } },
            { "lbl_proxy", { "Proxy server", "Serveur proxy", "Proxyserver", "Server proxy", "Servidor proxy" } },
            { "proxy_hint", { "Empty: settings of the system; none: direct connection; or http://host:port",
                "Vide : réglages du système ; none : connexion directe ; ou http://hôte:port",
                "Leer: Einstellungen des Systems; none: direkte Verbindung; oder http://Host:Port",
                "Vuoto: impostazioni del sistema; none: connessione diretta; oppure http://host:porta",
                "Vacío: ajustes del sistema; none: conexión directa; o http://host:puerto" } },
            { "proxy_mask_hint", { "******** keeps the saved password while the address and the user stay the same.",
                "******** conserve le mot de passe enregistré tant que l'adresse et l'utilisateur restent les mêmes.",
                "******** behält das gespeicherte Kennwort, solange Adresse und Benutzer gleich bleiben.",
                "******** conserva la password salvata finché indirizzo e utente restano gli stessi.",
                "******** conserva la contraseña guardada mientras la dirección y el usuario no cambien." } },
            { "lang_auto", { "Same as Windows", "Comme Windows", "Wie Windows", "Come Windows", "Como Windows" } },
            { "lbl_auto_updates", { "Check for updates when the service starts",
                "Rechercher les mises à jour au démarrage du service", "Beim Start des Dienstes nach Updates suchen",
                "Cerca aggiornamenti all'avvio del servizio", "Buscar actualizaciones al iniciar el servicio" } },
            { "lbl_offline", { "Offline mode: this computer has no Internet access",
                "Mode hors ligne : cet ordinateur n'a pas accès à Internet",
                "Offlinemodus: Dieser Computer hat keinen Internetzugang",
                "Modalità offline: questo computer non ha accesso a Internet",
                "Modo sin conexión: este equipo no tiene acceso a Internet" } },
            { "btn_open_folder", { "Open", "Ouvrir", "Öffnen", "Apri", "Abrir" } },
            { "btn_change", { "Change…", "Modifier…", "Ändern…", "Modifica…", "Cambiar…" } },
            { "btn_apply", { "Apply", "Appliquer", "Übernehmen", "Applica", "Aplicar" } },
            { "btn_close", { "Close", "Fermer", "Schließen", "Chiudi", "Cerrar" } },
            { "restart_note", { "The service restarts to apply the change.",
                "Le service redémarre pour appliquer la modification.",
                "Der Dienst startet neu, um die Änderung zu übernehmen.",
                "Il servizio si riavvia per applicare la modifica.",
                "El servicio se reinicia para aplicar el cambio." } },
            { "saved", { "Saved.", "Enregistré.", "Gespeichert.", "Salvato.", "Guardado." } },

            { "about_title", { "About Albireo", "À propos d'Albireo", "Über Albireo", "Informazioni su Albireo",
                "Acerca de Albireo" } },
            { "about_version", { "{e}, version {v}", "{e}, version {v}", "{e}, Version {v}", "{e}, versione {v}",
                "{e}, versión {v}" } },
            { "about_built", { "Based on the open source Dlib-Transformer-extensions library",
                "Reposant sur la bibliothèque open source Dlib-Transformer-extensions",
                "Basiert auf der Open-Source-Bibliothek Dlib-Transformer-extensions",
                "Basato sulla libreria open source Dlib-Transformer-extensions",
                "Basado en la biblioteca de código abierto Dlib-Transformer-extensions" } },
            { "about_created", { "Created by {a}", "Créé par {a}", "Erstellt von {a}", "Creato da {a}",
                "Creado por {a}" } },
            { "about_license", { "Distributed under the Boost Software License 1.0. Each model keeps its own license.",
                "Distribué sous la licence Boost Software License 1.0. Chaque modèle conserve sa propre licence.",
                "Vertrieben unter der Boost Software License 1.0. Jedes Modell behält seine eigene Lizenz.",
                "Distribuito con la Boost Software License 1.0. Ogni modello mantiene la propria licenza.",
                "Distribuido con la Boost Software License 1.0. Cada modelo conserva su propia licencia." } },
            { "btn_project", { "Project page", "Page du projet", "Projektseite", "Pagina del progetto",
                "Página del proyecto" } },

            { "setup_title", { "Albireo setup", "Installation d'Albireo", "Albireo-Installation",
                "Installazione di Albireo", "Instalación de Albireo" } },
            { "welcome_title", { "Welcome to Albireo", "Bienvenue dans Albireo", "Willkommen bei Albireo",
                "Benvenuto in Albireo", "Bienvenido a Albireo" } },
            { "welcome_text", {
                "This assistant installs Albireo Community Edition {v}: the service that runs language models on this "
                "computer, its icon in the notification area and its web application. Nothing leaves this computer.",
                "Cet assistant installe Albireo Community Edition {v} : le service qui exécute des modèles de langage "
                "sur cet ordinateur, son icône dans la zone de notification et son application web. Rien ne quitte "
                "cet ordinateur.",
                "Dieser Assistent installiert Albireo Community Edition {v}: den Dienst, der Sprachmodelle auf diesem "
                "Computer ausführt, sein Symbol im Infobereich und seine Webanwendung. Nichts verlässt diesen "
                "Computer.",
                "Questa procedura installa Albireo Community Edition {v}: il servizio che esegue i modelli linguistici "
                "su questo computer, la sua icona nell'area di notifica e la sua applicazione web. Nulla lascia "
                "questo computer.",
                "Este asistente instala Albireo Community Edition {v}: el servicio que ejecuta modelos de lenguaje en "
                "este equipo, su icono en el área de notificación y su aplicación web. Nada sale de este equipo." } },
            { "update_title", { "Albireo update", "Mise à jour d'Albireo", "Albireo-Aktualisierung",
                "Aggiornamento di Albireo", "Actualización de Albireo" } },
            { "update_text", {
                "Albireo {o} is installed in {d}. This assistant updates it to version {v} of Albireo Community "
                "Edition and keeps its settings, its models, its certificates, its shortcuts and the start of its "
                "icon.",
                "Albireo {o} est installé dans {d}. Cet assistant le met à jour vers la version {v} d'Albireo "
                "Community Edition et conserve ses réglages, ses modèles, ses certificats, ses raccourcis et le "
                "démarrage de son icône.",
                "Albireo {o} ist in {d} installiert. Dieser Assistent aktualisiert es auf Version {v} von Albireo "
                "Community Edition und behält seine Einstellungen, Modelle, Zertifikate, Verknüpfungen und den Start "
                "seines Symbols bei.",
                "Albireo {o} è installato in {d}. Questa procedura lo aggiorna alla versione {v} di Albireo Community "
                "Edition e conserva le impostazioni, i modelli, i certificati, i collegamenti e l'avvio della sua "
                "icona.",
                "Albireo {o} está instalado en {d}. Este asistente lo actualiza a la versión {v} de Albireo Community "
                "Edition y conserva sus ajustes, sus modelos, sus certificados, sus accesos directos y el inicio de "
                "su icono." } },
            { "update_newer", {
                "Albireo {o}, newer than version {v} of this assistant, is installed in {d}: it stays in place.",
                "Albireo {o}, plus récent que la version {v} de cet assistant, est installé dans {d} : il reste en "
                "place.",
                "Albireo {o}, neuer als Version {v} dieses Assistenten, ist in {d} installiert: Es bleibt erhalten.",
                "Albireo {o}, più recente della versione {v} di questa procedura, è installato in {d}: resta al suo "
                "posto.",
                "Albireo {o}, más reciente que la versión {v} de este asistente, está instalado en {d}: se "
                "mantiene." } },
            { "btn_update", { "Update", "Mettre à jour", "Aktualisieren", "Aggiorna", "Actualizar" } },
            { "update_progress", { "Update", "Mise à jour", "Aktualisierung", "Aggiornamento", "Actualización" } },
            { "update_done_title", { "Albireo is up to date", "Albireo est à jour", "Albireo ist aktuell",
                "Albireo è aggiornato", "Albireo está actualizado" } },
            { "update_done_text", { "Albireo {v} replaced version {o}. The service started again with its models.",
                "Albireo {v} a remplacé la version {o}. Le service a redémarré avec ses modèles.",
                "Albireo {v} hat Version {o} ersetzt. Der Dienst wurde mit seinen Modellen neu gestartet.",
                "Albireo {v} ha sostituito la versione {o}. Il servizio è ripartito con i suoi modelli.",
                "Albireo {v} reemplazó la versión {o}. El servicio se reinició con sus modelos." } },
            { "update_failed", { "The update failed: {e}", "La mise à jour a échoué : {e}",
                "Die Aktualisierung ist fehlgeschlagen: {e}", "L'aggiornamento non è riuscito: {e}",
                "La actualización falló: {e}" } },
            { "update_rolled_back", { "The programs of Albireo {o} were put back in place, and its service started "
                "again.", "Les programmes d'Albireo {o} ont été remis en place et son service a redémarré.",
                "Die Programme von Albireo {o} wurden wiederhergestellt, und sein Dienst wurde neu gestartet.",
                "I programmi di Albireo {o} sono stati ripristinati e il suo servizio è stato riavviato.",
                "Los programas de Albireo {o} se restablecieron y su servicio se reinició." } },
            { "install_rolled_back", { "The files copied were removed: this computer is as it was before the "
                "installation.", "Les fichiers copiés ont été supprimés : cet ordinateur est tel qu'avant "
                "l'installation.", "Die kopierten Dateien wurden entfernt: Dieser Computer ist wieder wie vor der "
                "Installation.", "I file copiati sono stati rimossi: questo computer è tornato com'era prima "
                "dell'installazione.", "Los archivos copiados se eliminaron: este equipo está como antes de la "
                "instalación." } },
            { "license_title", { "License", "Licence", "Lizenz", "Licenza", "Licencia" } },
            { "license_accept", { "I accept the terms of the license", "J'accepte les termes de la licence",
                "Ich akzeptiere die Lizenzbedingungen", "Accetto i termini della licenza",
                "Acepto los términos de la licencia" } },
            { "folder_title", { "Installation folder", "Dossier d'installation", "Installationsordner",
                "Cartella di installazione", "Carpeta de instalación" } },
            { "folder_programs", { "Programs", "Programmes", "Programme", "Programmi", "Programas" } },
            { "folder_data", { "Data: settings, logs, certificates and, by default, the models",
                "Données : réglages, journaux, certificats et, par défaut, les modèles",
                "Daten: Einstellungen, Protokolle, Zertifikate und standardmäßig die Modelle",
                "Dati: impostazioni, log, certificati e, per impostazione predefinita, i modelli",
                "Datos: ajustes, registros, certificados y, por defecto, los modelos" } },
            { "folder_models", { "Models", "Modèles", "Modelle", "Modelli", "Modelos" } },
            { "browse", { "Browse…", "Parcourir…", "Durchsuchen…", "Sfoglia…", "Examinar…" } },
            { "folder_not_fixed", { "{d} lies on a removable drive or on a network drive: choose a folder of a fixed "
                "drive of this computer.", "{d} se trouve sur un lecteur amovible ou réseau : choisissez un dossier "
                "d'un disque fixe de cet ordinateur.", "{d} liegt auf einem Wechsel- oder Netzlaufwerk: Wählen Sie "
                "einen Ordner auf einem festen Laufwerk dieses Computers.", "{d} si trova su un'unità rimovibile o di "
                "rete: scegliete una cartella di un disco fisso di questo computer.", "{d} está en una unidad "
                "extraíble o de red: elija una carpeta de un disco fijo de este equipo." } },
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
            { "folder_system", { "{d} is the root of a drive, or a folder of Windows or of the users: choose a folder "
                "of its own for Albireo.", "{d} est la racine d'un lecteur, ou un dossier de Windows ou des "
                "utilisateurs : choisissez un dossier propre à Albireo.", "{d} ist das Stammverzeichnis eines "
                "Laufwerks oder ein Ordner von Windows oder der Benutzer: Wählen Sie einen eigenen Ordner für "
                "Albireo.", "{d} è la radice di un'unità, o una cartella di Windows o degli utenti: scegliete una "
                "cartella dedicata ad Albireo.", "{d} es la raíz de una unidad, o una carpeta de Windows o de los "
                "usuarios: elija una carpeta propia para Albireo." } },
            { "folder_not_empty", { "{d} already holds other files: choose a new or empty folder.",
                "{d} contient déjà d'autres fichiers : choisissez un dossier nouveau ou vide.",
                "{d} enthält bereits andere Dateien: Wählen Sie einen neuen oder leeren Ordner.",
                "{d} contiene già altri file: scegliete una cartella nuova o vuota.",
                "{d} ya contiene otros archivos: elija una carpeta nueva o vacía." } },
            { "folder_foreign", { "{d} belongs to another user, or a link made by another user leads to it: choose "
                "another folder.", "{d} appartient à un autre utilisateur, ou un lien créé par un autre utilisateur "
                "y mène : choisissez un autre dossier.", "{d} gehört einem anderen Benutzer, oder ein symbolischer "
                "Link eines anderen Benutzers führt dorthin: Wählen Sie einen anderen Ordner.", "{d} appartiene a un "
                "altro utente, oppure vi conduce un collegamento simbolico creato da un altro utente: scegliete "
                "un'altra cartella.", "{d} pertenece a otro usuario, o un enlace creado por otro usuario lleva a "
                "ella: elija otra carpeta." } },
            { "folder_parent_unsafe", { "Accounts other than the administrators may rename a folder that holds {d} "
                "and put a folder of their own in its place: choose a folder in Program Files or in ProgramData, or "
                "a new folder at the root of a drive.", "Des comptes autres que les administrateurs peuvent renommer "
                "un dossier qui contient {d} et mettre un dossier à eux à sa place : choisissez un dossier dans "
                "Program Files ou dans ProgramData, ou un nouveau dossier à la racine d'un lecteur.", "Andere Konten "
                "als die Administratoren können einen Ordner umbenennen, der {d} enthält, und einen eigenen Ordner "
                "an seine Stelle setzen: Wählen Sie einen Ordner in Program Files oder in ProgramData oder einen "
                "neuen Ordner im Stammverzeichnis eines Laufwerks.", "Account diversi dagli amministratori possono "
                "rinominare una cartella che contiene {d} e mettere al suo posto una cartella propria: scegliete una "
                "cartella in Program Files o in ProgramData, oppure una nuova cartella nella radice di un'unità.",
                "Cuentas distintas de los administradores pueden cambiar el nombre de una carpeta que contiene {d} y "
                "poner una carpeta propia en su lugar: elija una carpeta en Program Files o en ProgramData, o una "
                "carpeta nueva en la raíz de una unidad." } },
            { "space_required", { "Space required by the programs: {s}", "Espace nécessaire aux programmes : {s}",
                "Speicherbedarf der Programme: {s}", "Spazio richiesto dai programmi: {s}",
                "Espacio necesario para los programas: {s}" } },
            { "space_available", { "Space available: {s}", "Espace disponible : {s}", "Verfügbarer Speicher: {s}",
                "Spazio disponibile: {s}", "Espacio disponible: {s}" } },
            { "welcome_models", { "This installer brings the models that come with Albireo ({s}): Albireo "
                "answers as soon as it is installed, without network.", "Ce programme d'installation apporte les "
                "modèles livrés avec Albireo ({s}) : Albireo répond dès son installation, sans réseau.",
                "Dieses Installationsprogramm bringt die mit Albireo gelieferten Modelle mit ({s}): Albireo "
                "antwortet gleich nach der Installation, ohne Netzwerk.", "Questo programma di installazione porta i "
                "modelli forniti con Albireo ({s}): Albireo risponde appena installato, senza rete.", "Este programa "
                "de instalación trae los modelos incluidos con Albireo ({s}): Albireo responde en cuanto se instala, "
                "sin red." } },
            { "space_models", { "A model needs 1 to 20 GB in the models folder.",
                "Un modèle occupe de 1 à 20 Go dans le dossier des modèles.",
                "Ein Modell benötigt 1 bis 20 GB im Modellordner.",
                "Un modello occupa da 1 a 20 GB nella cartella dei modelli.",
                "Un modelo ocupa de 1 a 20 GB en la carpeta de modelos." } },
            { "space_insufficient", { "Not enough free space on this drive.",
                "Espace libre insuffisant sur ce lecteur.", "Nicht genügend freier Speicher auf diesem Laufwerk.",
                "Spazio libero insufficiente su questa unità.", "No hay suficiente espacio libre en esta unidad." } },
            { "options_title", { "Options", "Options", "Optionen", "Opzioni", "Opciones" } },
            { "opt_autostart", { "Show the Albireo icon when a user signs in",
                "Afficher l'icône Albireo à l'ouverture de session", "Albireo-Symbol bei der Anmeldung anzeigen",
                "Mostra l'icona di Albireo all'accesso", "Mostrar el icono de Albireo al iniciar sesión" } },
            { "opt_desktop", { "Create a shortcut on the desktop", "Créer un raccourci sur le bureau",
                "Verknüpfung auf dem Desktop erstellen", "Crea un collegamento sul desktop",
                "Crear un acceso directo en el escritorio" } },
            { "opt_trust", { "Trust the local certificate of Albireo in the browsers",
                "Approuver le certificat local d'Albireo dans les navigateurs",
                "Dem lokalen Zertifikat von Albireo in den Browsern vertrauen",
                "Considerare attendibile nei browser il certificato locale di Albireo",
                "Confiar en el certificado local de Albireo en los navegadores" } },
            { "opt_default_model", { "Download the default general model at the first start (about 2.5 GB)",
                "Télécharger le modèle général par défaut au premier démarrage (environ 2,5 Go)",
                "Das allgemeine Standardmodell beim ersten Start herunterladen (etwa 2,5 GB)",
                "Scarica il modello generale predefinito al primo avvio (circa 2,5 GB)",
                "Descargar el modelo general predeterminado en el primer inicio (unos 2,5 GB)" } },
            { "opt_offline", { "This computer has no Internet access (offline mode)",
                "Cet ordinateur n'a pas accès à Internet (mode hors ligne)",
                "Dieser Computer hat keinen Internetzugang (Offlinemodus)",
                "Questo computer non ha accesso a Internet (modalità offline)",
                "Este equipo no tiene acceso a Internet (modo sin conexión)" } },
            { "opt_port", { "Port of the local server", "Port du serveur local", "Port des lokalen Servers",
                "Porta del server locale", "Puerto del servidor local" } },
            { "progress_title", { "Installation", "Installation", "Installation", "Installazione", "Instalación" } },
            { "step_stop", { "Stopping the running programs", "Arrêt des programmes en cours",
                "Laufende Programme werden beendet", "Arresto dei programmi in esecuzione",
                "Deteniendo los programas en ejecución" } },
            { "step_files", { "Copying {f}", "Copie de {f}", "{f} wird kopiert", "Copia di {f}", "Copiando {f}" } },
            { "step_models", { "Installing the model {f}, which comes with Albireo",
                "Installation du modèle {f}, livré avec Albireo",
                "Das mit Albireo gelieferte Modell {f} wird installiert",
                "Installazione del modello {f}, fornito con Albireo",
                "Instalando el modelo {f}, incluido con Albireo" } },
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
            { "finish_title", { "Albireo is installed", "Albireo est installé", "Albireo ist installiert",
                "Albireo è installato", "Albireo está instalado" } },
            { "finish_text", {
                "Albireo runs as a service. Its icon in the notification area gives access to the models, to this "
                "computer's settings and to the web application.",
                "Albireo fonctionne comme un service. Son icône dans la zone de notification donne accès aux modèles, "
                "aux réglages de cet ordinateur et à l'application web.",
                "Albireo läuft als Dienst. Sein Symbol im Infobereich bietet Zugriff auf die Modelle, die "
                "Einstellungen dieses Computers und die Webanwendung.",
                "Albireo funziona come servizio. La sua icona nell'area di notifica dà accesso ai modelli, alle "
                "impostazioni di questo computer e all'applicazione web.",
                "Albireo funciona como servicio. Su icono en el área de notificación da acceso a los modelos, a los "
                "ajustes de este equipo y a la aplicación web." } },
            { "bundle_notice", { "The offline bundle next to the installer was not copied whole: {e}",
                "Le paquet hors ligne placé à côté du programme d'installation n'a pas été copié entièrement : {e}",
                "Das Offline-Paket neben dem Installationsprogramm wurde nicht vollständig kopiert: {e}",
                "Il pacchetto offline accanto al programma di installazione non è stato copiato per intero: {e}",
                "El paquete sin conexión junto al programa de instalación no se copió por completo: {e}" } },
            { "opt_launch", { "Open Albireo now", "Ouvrir Albireo maintenant", "Albireo jetzt öffnen",
                "Apri Albireo ora", "Abrir Albireo ahora" } },
            { "btn_back", { "< Back", "< Précédent", "< Zurück", "< Indietro", "< Atrás" } },
            { "btn_next", { "Next >", "Suivant >", "Weiter >", "Avanti >", "Siguiente >" } },
            { "btn_install", { "Install", "Installer", "Installieren", "Installa", "Instalar" } },
            { "btn_finish", { "Finish", "Terminer", "Fertig stellen", "Fine", "Finalizar" } },
            { "btn_cancel", { "Cancel", "Annuler", "Abbrechen", "Annulla", "Cancelar" } },
            { "btn_uninstall", { "Uninstall", "Désinstaller", "Deinstallieren", "Disinstalla", "Desinstalar" } },
            { "cancel_confirm", { "Stop the installation of Albireo?", "Interrompre l'installation d'Albireo ?",
                "Installation von Albireo abbrechen?", "Interrompere l'installazione di Albireo?",
                "¿Detener la instalación de Albireo?" } },
            { "error_title", { "Albireo setup", "Installation d'Albireo", "Albireo-Installation",
                "Installazione di Albireo", "Instalación de Albireo" } },
            { "error_failed", { "The installation failed: {e}", "L'installation a échoué : {e}",
                "Die Installation ist fehlgeschlagen: {e}", "L'installazione non è riuscita: {e}",
                "La instalación falló: {e}" } },
            { "uninstall_title", { "Uninstall Albireo", "Désinstaller Albireo", "Albireo deinstallieren",
                "Disinstalla Albireo", "Desinstalar Albireo" } },
            { "uninstall_text", { "Albireo, its service, its notification icon, its models, its settings and its "
                "logs will be removed from this computer.", "Albireo, son service, son icône de notification, ses "
                "modèles, ses réglages et ses journaux seront retirés de cet ordinateur.", "Albireo, sein Dienst, "
                "sein Infobereich-Symbol, seine Modelle, seine Einstellungen und seine Protokolle werden von diesem "
                "Computer entfernt.", "Albireo, il suo servizio, la sua icona di notifica, i suoi modelli, le sue "
                "impostazioni e i suoi log verranno rimossi da questo computer.", "Albireo, su servicio, su icono de "
                "notificación, sus modelos, sus ajustes y sus registros se eliminarán de este equipo." } },
            { "uninstall_text_keep", { "Albireo, its service and its notification icon will be removed from this "
                "computer. Its models and its settings stay for a later installation.", "Albireo, son service et son "
                "icône de notification seront retirés de cet ordinateur. Ses modèles et ses réglages restent pour une "
                "installation ultérieure.", "Albireo, sein Dienst und sein Infobereich-Symbol werden von diesem "
                "Computer entfernt. Seine Modelle und seine Einstellungen bleiben für eine spätere Installation "
                "erhalten.", "Albireo, il suo servizio e la sua icona di notifica verranno rimossi da questo "
                "computer. I suoi modelli e le sue impostazioni restano per un'installazione successiva.", "Albireo, "
                "su servicio y su icono de notificación se eliminarán de este equipo. Sus modelos y sus ajustes se "
                "conservan para una instalación posterior." } },
            { "uninstall_keep_data", { "Keep the models and the settings for a later installation",
                "Conserver les modèles et les réglages pour une installation ultérieure",
                "Modelle und Einstellungen für eine spätere Installation behalten",
                "Conserva i modelli e le impostazioni per un'installazione successiva",
                "Conservar los modelos y los ajustes para una instalación posterior" } },
            { "uninstall_browser", { "The discussions are kept by each browser; Albireo does not store them.",
                "Les discussions sont conservées par chaque navigateur ; Albireo ne les stocke pas.",
                "Die Unterhaltungen werden von jedem Browser gespeichert; Albireo speichert sie nicht.",
                "Le conversazioni sono conservate da ciascun browser; Albireo non le memorizza.",
                "Las conversaciones las conserva cada navegador; Albireo no las almacena." } },
            { "uninstall_done", { "Albireo was removed from this computer.", "Albireo a été retiré de cet ordinateur.",
                "Albireo wurde von diesem Computer entfernt.", "Albireo è stato rimosso da questo computer.",
                "Albireo se eliminó de este equipo." } },
            { "uninstall_kept", { "The folder {d} stays, with the models and the data you keep in it.",
                "Le dossier {d} reste en place, avec les modèles et les données que vous y conservez.",
                "Der Ordner {d} bleibt bestehen, mit den Modellen und Daten, die Sie darin behalten.",
                "La cartella {d} resta al suo posto, con i modelli e i dati che vi conservate.",
                "La carpeta {d} se mantiene, con los modelos y los datos que conserva en ella." } },
            { "uninstall_progress", { "Removing Albireo", "Suppression d'Albireo", "Albireo wird entfernt",
                "Rimozione di Albireo", "Eliminando Albireo" } },
            { "admin_required", { "Administrator rights are required.", "Les droits d'administrateur sont nécessaires.",
                "Administratorrechte sind erforderlich.", "Sono necessari i diritti di amministratore.",
                "Se necesitan derechos de administrador." } },
            { "not_installed", { "Albireo is not installed on this computer.",
                "Albireo n'est pas installé sur cet ordinateur.", "Albireo ist auf diesem Computer nicht installiert.",
                "Albireo non è installato su questo computer.", "Albireo no está instalado en este equipo." } },
            { "no_payload", { "This copy of the installer, kept in the folder of Albireo, only uninstalls it. To "
                "install or update Albireo, start the installer you downloaded.", "Cette copie du programme "
                "d'installation, conservée dans le dossier d'Albireo, sert uniquement à le désinstaller. Pour "
                "installer ou mettre à jour Albireo, lancez le programme d'installation téléchargé.", "Diese Kopie "
                "des Installationsprogramms im Ordner von Albireo dient nur zur Deinstallation. Starten Sie zum "
                "Installieren oder Aktualisieren von Albireo das heruntergeladene Installationsprogramm.", "Questa "
                "copia del programma di installazione, conservata nella cartella di Albireo, serve solo a "
                "disinstallarlo. Per installare o aggiornare Albireo, avviate il programma di installazione "
                "scaricato.", "Esta copia del programa de instalación, guardada en la carpeta de Albireo, solo sirve "
                "para desinstalarlo. Para instalar o actualizar Albireo, inicie el programa de instalación que "
                "descargó." } },
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

    namespace
    {
        void replace_marks(std::string& s, const std::string& name, const std::string& value)
        {
            const std::string mark = "{" + name + "}";
            for (size_t p = s.find(mark); p != std::string::npos; p = s.find(mark, p + value.size()))
                s.replace(p, mark.size(), value);
        }
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

}
