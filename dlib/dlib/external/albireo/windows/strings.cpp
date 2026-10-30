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
            { "accel_npu", { "Neural processor (NPU)", "Processeur neuronal (NPU)", "Neuronaler Prozessor (NPU)",
                "Processore neurale (NPU)", "Procesador neuronal (NPU)" } },
            { "accel_gpu", { "Graphics processor (GPU)", "Processeur graphique (GPU)", "Grafikprozessor (GPU)",
                "Processore grafico (GPU)", "Procesador gráfico (GPU)" } },
            { "accel_both", { "NPU or GPU, the fastest measured", "NPU ou GPU, le plus rapide mesuré",
                "NPU oder GPU, der schnellste gemessene", "NPU o GPU, il più veloce misurato",
                "NPU o GPU, el más rápido medido" } },
            { "accel_measure", { "Measure the engines", "Mesurer les moteurs", "Engines messen", "Misura i motori",
                "Medir los motores" } },
            { "accel_not_installed", { "No accelerator engine installed: OpenVINO (Intel NPU and graphics) or CUDA "
                "(NVIDIA)", "Aucun moteur d'accélération installé : OpenVINO (NPU et graphique Intel) ou CUDA (NVIDIA)",
                "Keine Beschleuniger-Engine installiert: OpenVINO (Intel-NPU und -Grafik) oder CUDA (NVIDIA)",
                "Nessun motore di accelerazione installato: OpenVINO (NPU e grafica Intel) o CUDA (NVIDIA)",
                "Ningún motor de aceleración instalado: OpenVINO (NPU y gráficos Intel) o CUDA (NVIDIA)" } },
            { "accel_unusable", { "The accelerator engines of this installation cannot run here (see This computer)",
                "Les moteurs d'accélération de cette installation ne peuvent pas fonctionner ici (voir Cet "
                "ordinateur)", "Die Beschleuniger-Engines dieser Installation können hier nicht laufen (siehe "
                "Dieser Computer)", "I motori di accelerazione di questa installazione non possono funzionare qui "
                "(vedere Questo computer)", "Los motores de aceleración de esta instalación no pueden funcionar aquí "
                "(véase Este equipo)" } },
            { "accel_cpu_only", { "Available while the processor engine runs",
                "Disponible lorsque le moteur processeur est actif", "Verfügbar, solange die Prozessor-Engine läuft",
                "Disponibile quando è attivo il motore del processore",
                "Disponible mientras funciona el motor del procesador" } },
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
                "Up to three models answer, one per function; a function without a model uses the general model. "
                "The inputs of a model decide the attachments of the web application; the document search has its "
                "own embedding model.",
                "Jusqu'à trois modèles répondent, un par fonction ; une fonction sans modèle utilise le modèle "
                "général. Les entrées d'un modèle décident des pièces jointes de l'application web ; la recherche "
                "documentaire a son propre modèle d'embeddings.",
                "Bis zu drei Modelle antworten, eines pro Funktion; eine Funktion ohne Modell verwendet das "
                "allgemeine Modell. Die Eingaben eines Modells bestimmen die Anhänge der Webanwendung; die "
                "Dokumentensuche hat ihr eigenes Embedding-Modell.",
                "Fino a tre modelli rispondono, uno per funzione; una funzione senza modello usa il modello generale. "
                "Gli ingressi di un modello decidono gli allegati dell'applicazione web; la ricerca nei documenti ha "
                "il proprio modello di embedding.",
                "Hasta tres modelos responden, uno por función; una función sin modelo usa el modelo general. "
                "Las entradas de un modelo deciden los adjuntos de la aplicación web; la búsqueda en documentos "
                "tiene su propio modelo de embeddings." } },
            { "col_model", { "Model", "Modèle", "Modell", "Modello", "Modelo" } },
            { "col_roles", { "Functions", "Fonctions", "Funktionen", "Funzioni", "Funciones" } },
            { "col_size", { "Size", "Taille", "Größe", "Dimensione", "Tamaño" } },
            { "col_state", { "State", "État", "Status", "Stato", "Estado" } },
            { "col_active", { "Active for", "Actif pour", "Aktiv für", "Attivo per", "Activo para" } },
            { "col_inputs", { "Inputs", "Entrées", "Eingaben", "Ingressi", "Entradas" } },
            { "input_text", { "Text", "Texte", "Text", "Testo", "Texto" } },
            { "input_image", { "images", "images", "Bilder", "immagini", "imágenes" } },
            { "input_audio", { "sound", "son", "Ton", "audio", "sonido" } },
            { "state_builtin", { "Comes with Albireo", "Livré avec Albireo", "Mit Albireo geliefert",
                "Fornito con Albireo", "Incluido con Albireo" } },
            { "lbl_inputs", { "Inputs", "Entrées", "Eingaben", "Ingressi", "Entradas" } },
            { "role_general", { "General", "Général", "Allgemein", "Generale", "General" } },
            { "role_code", { "Code", "Code", "Code", "Codice", "Código" } },
            { "role_embedding", { "Document search", "Recherche documentaire", "Dokumentensuche",
                "Ricerca nei documenti", "Búsqueda en documentos" } },
            { "role_security", { "Cybersecurity", "Cybersécurité", "Cybersicherheit", "Cybersicurezza",
                "Ciberseguridad" } },
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
            { "err_download_install", { "The downloaded model could not be installed: {e}",
                "Le modèle téléchargé n'a pas pu être installé : {e}",
                "Das heruntergeladene Modell konnte nicht installiert werden: {e}",
                "Il modello scaricato non è stato installato: {e}", "El modelo descargado no se pudo instalar: {e}" } },
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
            { "why_library_failed", { "the library could not be loaded (code {code})", "la bibliothèque n'a pas pu "
                "être chargée (code {code})", "die Bibliothek konnte nicht geladen werden (Code {code})",
                "la libreria non è stata caricata (codice {code})",
                "no se pudo cargar la biblioteca (código {code})" } },
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
            { "import_done", { "{n} models imported.", "{n} modèles importés.", "{n} Modelle importiert.",
                "{n} modelli importati.", "{n} modelos importados." } },
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
            { "about_built", { "Built on the Dlib-Transformer-extensions library",
                "Construit sur la bibliothèque Dlib-Transformer-extensions",
                "Entwickelt auf der Bibliothek Dlib-Transformer-extensions",
                "Costruito sulla libreria Dlib-Transformer-extensions",
                "Construido sobre la biblioteca Dlib-Transformer-extensions" } },
            { "about_created", { "Created by {a}", "Créé par {a}", "Erstellt von {a}", "Creato da {a}",
                "Creado por {a}" } },
            { "about_company", { "Published by {c}", "Édité par {c}", "Herausgegeben von {c}", "Pubblicato da {c}",
                "Publicado por {c}" } },
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
            { "license_title", { "License", "Licence", "Lizenz", "Licenza", "Licencia" } },
            { "license_accept", { "I accept the terms of the license", "J'accepte les termes de la licence",
                "Ich akzeptiere die Lizenzbedingungen", "Accetto i termini della licenza",
                "Acepto los términos de la licencia" } },
            { "folder_title", { "Installation folder", "Dossier d'installation", "Installationsordner",
                "Cartella di installazione", "Carpeta de instalación" } },
            { "folder_programs", { "Programs", "Programmes", "Programme", "Programmi", "Programas" } },
            { "folder_models", { "Models", "Modèles", "Modelle", "Modelli", "Modelos" } },
            { "browse", { "Browse…", "Parcourir…", "Durchsuchen…", "Sfoglia…", "Examinar…" } },
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
            { "uninstall_text", { "Albireo, its service and its notification icon will be removed from this computer.",
                "Albireo, son service et son icône de notification seront retirés de cet ordinateur.",
                "Albireo, sein Dienst und sein Infobereich-Symbol werden von diesem Computer entfernt.",
                "Albireo, il suo servizio e la sua icona di notifica verranno rimossi da questo computer.",
                "Albireo, su servicio y su icono de notificación se eliminarán de este equipo." } },
            { "uninstall_data", { "Also delete the models, the settings and the logs",
                "Supprimer aussi les modèles, les réglages et les journaux",
                "Auch die Modelle, die Einstellungen und die Protokolle löschen",
                "Elimina anche i modelli, le impostazioni e i log",
                "Eliminar también los modelos, los ajustes y los registros" } },
            { "uninstall_browser", { "The discussions are kept by each browser; Albireo does not store them.",
                "Les discussions sont conservées par chaque navigateur ; Albireo ne les stocke pas.",
                "Die Unterhaltungen werden von jedem Browser gespeichert; Albireo speichert sie nicht.",
                "Le conversazioni sono conservate da ciascun browser; Albireo non le memorizza.",
                "Las conversaciones las conserva cada navegador; Albireo no las almacena." } },
            { "uninstall_done", { "Albireo was removed from this computer.", "Albireo a été retiré de cet ordinateur.",
                "Albireo wurde von diesem Computer entfernt.", "Albireo è stato rimosso da questo computer.",
                "Albireo se eliminó de este equipo." } },
            { "uninstall_progress", { "Removing Albireo", "Suppression d'Albireo", "Albireo wird entfernt",
                "Rimozione di Albireo", "Eliminando Albireo" } },
            { "admin_required", { "Administrator rights are required.", "Les droits d'administrateur sont nécessaires.",
                "Administratorrechte sind erforderlich.", "Sono necessari i diritti di amministratore.",
                "Se necesitan derechos de administrador." } },
            { "reboot_note", { "Some files were in use; they will be replaced at the next restart of Windows.",
                "Certains fichiers étaient utilisés ; ils seront remplacés au prochain redémarrage de Windows.",
                "Einige Dateien waren in Verwendung; sie werden beim nächsten Windows-Neustart ersetzt.",
                "Alcuni file erano in uso; verranno sostituiti al prossimo riavvio di Windows.",
                "Algunos archivos estaban en uso; se reemplazarán en el próximo reinicio de Windows." } },
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
