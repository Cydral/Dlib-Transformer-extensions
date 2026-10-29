# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
"""
Writes the model catalog of Albireo from the list below and from the Hugging Face hub.

    python3 make_catalog.py <output catalog.json>

For each model, the hub gives the commit the files are pinned to, the size and the SHA-256
of the file, the number of parameters and the trained context.  The download addresses
name the commit, so that the file behind an address never changes after the catalog is
signed.  A model published elsewhere, such as the parts made by publish_model.py on a
GitHub release, is listed with its "files" given in full instead of "hub".

The catalog names one default language model, which the service downloads when no language
model is installed, and one default embedding model, which it downloads in every case: the
search in the documents of the conversations and of the projects relies on it.  An embedding
model holds the "embedding" role alone.

The catalog is then signed with albireo-sign (sign <private key> catalog.json), and both
files are published in dist/albireo of the repository.
"""

import json
import sys
import urllib.request
from datetime import datetime, timezone

ROLES = ("general", "code", "security", "embedding")

MODELS = [
    {
        "id": "qwen3-4b",
        "name": "Qwen3 4B",
        "publisher": "Qwen",
        "hub": ("Qwen/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf"),
        "quantization": "Q4_K_M",
        "roles": ["general"],
        "default": True,
        "tools": True,
        "reasoning": True,
        "license": "Apache-2.0",
        "description": {
            "en": "General assistant for conversation, writing, analysis and tool use. It can reason step by step "
                  "before answering. It runs on most computers with 8 GB of memory.",
            "fr": "Assistant généraliste pour la conversation, la rédaction, l'analyse et l'usage d'outils. Il peut "
                  "raisonner étape par étape avant de répondre. Il fonctionne sur la plupart des ordinateurs dotés "
                  "de 8 Go de mémoire.",
            "de": "Allgemeiner Assistent für Gespräche, Texte, Analysen und Werkzeugnutzung. Er kann vor der Antwort "
                  "Schritt für Schritt nachdenken. Er läuft auf den meisten Computern mit 8 GB Arbeitsspeicher.",
            "it": "Assistente generale per conversazione, scrittura, analisi e uso di strumenti. Può ragionare passo "
                  "dopo passo prima di rispondere. Funziona sulla maggior parte dei computer con 8 GB di memoria.",
            "es": "Asistente general para conversación, redacción, análisis y uso de herramientas. Puede razonar paso "
                  "a paso antes de responder. Funciona en la mayoría de los ordenadores con 8 GB de memoria.",
        },
    },
    {
        "id": "qwen3-embedding-0.6b",
        "name": "Qwen3 Embedding 0.6B",
        "publisher": "Qwen",
        "hub": ("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
        "quantization": "Q8_0",
        "roles": ["embedding"],
        "default": True,
        "embedding": True,
        "dimensions": 1024,
        "tools": False,
        "reasoning": False,
        "license": "Apache-2.0",
        "description": {
            "en": "Embedding model of the document search. It turns the passages of the attached documents and the "
                  "questions into vectors, in more than 100 languages, so that the passages answering a question are "
                  "found by their meaning. Albireo installs it in every case, the search relies on it.",
            "fr": "Modèle d'embeddings de la recherche documentaire. Il convertit en vecteurs les passages des "
                  "documents joints et les questions, dans plus de 100 langues, pour retrouver par leur sens les "
                  "passages qui répondent à une question. Albireo l'installe dans tous les cas, la recherche en "
                  "dépend.",
            "de": "Embedding-Modell der Dokumentsuche. Es wandelt die Abschnitte der angehängten Dokumente und die "
                  "Fragen in mehr als 100 Sprachen in Vektoren um, damit die Abschnitte, die eine Frage beantworten, "
                  "über ihre Bedeutung gefunden werden. Albireo installiert es in jedem Fall, die Suche beruht darauf.",
            "it": "Modello di embedding della ricerca nei documenti. Converte in vettori i passaggi dei documenti "
                  "allegati e le domande, in più di 100 lingue, per ritrovare in base al significato i passaggi che "
                  "rispondono a una domanda. Albireo lo installa in ogni caso, la ricerca si basa su di esso.",
            "es": "Modelo de embeddings de la búsqueda en documentos. Convierte en vectores los pasajes de los "
                  "documentos adjuntos y las preguntas, en más de 100 idiomas, para encontrar por su significado los "
                  "pasajes que responden a una pregunta. Albireo lo instala en todos los casos, la búsqueda depende "
                  "de él.",
        },
    },
    {
        "id": "qwen3-1.7b",
        "name": "Qwen3 1.7B",
        "publisher": "Qwen",
        "hub": ("Qwen/Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q8_0.gguf"),
        "quantization": "Q8_0",
        "roles": ["general"],
        "tools": True,
        "reasoning": True,
        "license": "Apache-2.0",
        "description": {
            "en": "Compact general assistant with step by step reasoning, for computers with little memory and no "
                  "accelerator. It answers faster than the larger models, in less depth.",
            "fr": "Assistant généraliste compact, avec raisonnement étape par étape, pour les ordinateurs disposant de "
                  "peu de mémoire et sans accélérateur. Il répond plus vite que les modèles plus grands, avec moins "
                  "de profondeur.",
            "de": "Kompakter allgemeiner Assistent mit schrittweisem Denken, für Computer mit wenig Arbeitsspeicher "
                  "und ohne Beschleuniger. Er antwortet schneller als die größeren Modelle, mit weniger Tiefe.",
            "it": "Assistente generale compatto, con ragionamento passo dopo passo, per computer con poca memoria e "
                  "senza acceleratore. Risponde più rapidamente dei modelli più grandi, con meno profondità.",
            "es": "Asistente general compacto, con razonamiento paso a paso, para ordenadores con poca memoria y sin "
                  "acelerador. Responde más rápido que los modelos más grandes, con menos profundidad.",
        },
    },
    {
        "id": "ministral-3-3b",
        "name": "Ministral 3 3B",
        "publisher": "Mistral AI",
        "hub": ("mistralai/Ministral-3-3B-Instruct-2512-GGUF", "Ministral-3-3B-Instruct-2512-Q4_K_M.gguf"),
        "quantization": "Q4_K_M",
        "roles": ["general"],
        "tools": True,
        "reasoning": False,
        "license": "Apache-2.0",
        "description": {
            "en": "General assistant from Mistral AI, trained on many European languages. It uses tools and answers "
                  "directly, without a reasoning phase.",
            "fr": "Assistant généraliste de Mistral AI, entraîné sur de nombreuses langues européennes, dont le "
                  "français. Il utilise des outils et répond directement, sans phase de raisonnement.",
            "de": "Allgemeiner Assistent von Mistral AI, trainiert auf vielen europäischen Sprachen, darunter "
                  "Deutsch. Er nutzt Werkzeuge und antwortet direkt, ohne Denkphase.",
            "it": "Assistente generale di Mistral AI, addestrato su molte lingue europee, tra cui l'italiano. Usa "
                  "strumenti e risponde direttamente, senza fase di ragionamento.",
            "es": "Asistente general de Mistral AI, entrenado en muchas lenguas europeas, entre ellas el español. Usa "
                  "herramientas y responde directamente, sin fase de razonamiento.",
        },
    },
    {
        "id": "llama-3.2-3b",
        "name": "Llama 3.2 3B",
        "publisher": "Meta",
        "hub": ("bartowski/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
        "quantization": "Q4_K_M",
        "roles": ["general"],
        "tools": True,
        "reasoning": False,
        "license": "Llama 3.2 Community License, Built with Llama",
        "description": {
            "en": "General assistant from Meta, trained for English, French, German, Italian, Spanish, Portuguese, "
                  "Hindi and Thai, with tool use and a context of 128K tokens. It answers directly, without a "
                  "reasoning phase, and runs on computers with 8 GB of memory.",
            "fr": "Assistant généraliste de Meta, entraîné pour l'anglais, le français, l'allemand, l'italien, "
                  "l'espagnol, le portugais, l'hindi et le thaï, avec usage d'outils et un contexte de 128K jetons. "
                  "Il répond directement, sans phase de raisonnement, et fonctionne sur les ordinateurs dotés de "
                  "8 Go de mémoire.",
            "de": "Allgemeiner Assistent von Meta, trainiert für Englisch, Französisch, Deutsch, Italienisch, "
                  "Spanisch, Portugiesisch, Hindi und Thai, mit Werkzeugnutzung und einem Kontext von 128K Token. Er "
                  "antwortet direkt, ohne Denkphase, und läuft auf Computern mit 8 GB Arbeitsspeicher.",
            "it": "Assistente generale di Meta, addestrato per inglese, francese, tedesco, italiano, spagnolo, "
                  "portoghese, hindi e thailandese, con uso di strumenti e un contesto di 128K token. Risponde "
                  "direttamente, senza fase di ragionamento, e funziona su computer con 8 GB di memoria.",
            "es": "Asistente general de Meta, entrenado para inglés, francés, alemán, italiano, español, portugués, "
                  "hindi y tailandés, con uso de herramientas y un contexto de 128K tokens. Responde directamente, "
                  "sin fase de razonamiento, y funciona en ordenadores con 8 GB de memoria.",
        },
    },
    {
        "id": "qwen3-8b",
        "name": "Qwen3 8B",
        "publisher": "Qwen",
        "hub": ("Qwen/Qwen3-8B-GGUF", "Qwen3-8B-Q4_K_M.gguf"),
        "quantization": "Q4_K_M",
        "roles": ["general"],
        "tools": True,
        "reasoning": True,
        "license": "Apache-2.0",
        "description": {
            "en": "Larger general assistant with step by step reasoning and tool use, for computers with 16 GB of "
                  "memory or a graphics processor.",
            "fr": "Assistant généraliste plus grand, avec raisonnement étape par étape et usage d'outils, pour les "
                  "ordinateurs dotés de 16 Go de mémoire ou d'un processeur graphique.",
            "de": "Größerer allgemeiner Assistent mit schrittweisem Denken und Werkzeugnutzung, für Computer mit "
                  "16 GB Arbeitsspeicher oder einem Grafikprozessor.",
            "it": "Assistente generale più grande, con ragionamento passo dopo passo e uso di strumenti, per computer "
                  "con 16 GB di memoria o un processore grafico.",
            "es": "Asistente general más grande, con razonamiento paso a paso y uso de herramientas, para ordenadores "
                  "con 16 GB de memoria o un procesador gráfico.",
        },
    },
    {
        "id": "qwen3-30b-a3b",
        "name": "Qwen3 30B A3B",
        "publisher": "Qwen",
        "hub": ("Qwen/Qwen3-30B-A3B-GGUF", "Qwen3-30B-A3B-Q4_K_M.gguf"),
        "quantization": "Q4_K_M",
        "roles": ["general"],
        "tools": True,
        "reasoning": True,
        "license": "Apache-2.0",
        "description": {
            "en": "Mixture of experts of 30 billion parameters, 3 billion of which compute each token: it generates "
                  "at the pace of a model of 3 billion parameters. It needs 24 GB of memory.",
            "fr": "Mélange d'experts de 30 milliards de paramètres, dont 3 milliards calculent chaque jeton : il "
                  "génère au rythme d'un modèle de 3 milliards de paramètres. Il demande 24 Go de mémoire.",
            "de": "Mixture of Experts mit 30 Milliarden Parametern, von denen 3 Milliarden jedes Token berechnen: "
                  "es erzeugt Text im Tempo eines Modells mit 3 Milliarden Parametern. Es benötigt 24 GB "
                  "Arbeitsspeicher.",
            "it": "Mixture of experts da 30 miliardi di parametri, di cui 3 miliardi calcolano ogni token: genera al "
                  "ritmo di un modello da 3 miliardi di parametri. Richiede 24 GB di memoria.",
            "es": "Mezcla de expertos de 30 000 millones de parámetros, de los que 3000 millones calculan cada token: "
                  "genera al ritmo de un modelo de 3000 millones de parámetros. Necesita 24 GB de memoria.",
        },
    },
    {
        "id": "qwen2.5-coder-1.5b",
        "name": "Qwen2.5 Coder 1.5B",
        "publisher": "Qwen",
        "hub": ("Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"),
        "quantization": "Q4_K_M",
        "roles": ["code"],
        "tools": True,
        "reasoning": False,
        "license": "Apache-2.0",
        "description": {
            "en": "Programming assistant: writing, explaining and fixing code in the common languages. Its size lets "
                  "it answer quickly on any computer.",
            "fr": "Assistant de programmation : écriture, explication et correction de code dans les langages "
                  "courants. Sa taille lui permet de répondre rapidement sur tout ordinateur.",
            "de": "Programmierassistent: Schreiben, Erklären und Korrigieren von Code in den gängigen Sprachen. "
                  "Seine Größe erlaubt schnelle Antworten auf jedem Computer.",
            "it": "Assistente di programmazione: scrittura, spiegazione e correzione di codice nei linguaggi comuni. "
                  "Le sue dimensioni gli permettono di rispondere rapidamente su qualsiasi computer.",
            "es": "Asistente de programación: escritura, explicación y corrección de código en los lenguajes "
                  "habituales. Su tamaño le permite responder rápidamente en cualquier ordenador.",
        },
    },
    {
        "id": "qwen2.5-coder-7b",
        "name": "Qwen2.5 Coder 7B",
        "publisher": "Qwen",
        "hub": ("Qwen/Qwen2.5-Coder-7B-Instruct-GGUF", "qwen2.5-coder-7b-instruct-q4_k_m.gguf"),
        "quantization": "Q4_K_M",
        "roles": ["code"],
        "tools": True,
        "reasoning": False,
        "license": "Apache-2.0",
        "description": {
            "en": "Programming assistant for writing, reviewing and fixing code in several dozen languages, with a "
                  "context of 128K tokens for large files.",
            "fr": "Assistant de programmation pour écrire, relire et corriger du code dans plusieurs dizaines de "
                  "langages, avec un contexte de 128K jetons pour les fichiers volumineux.",
            "de": "Programmierassistent zum Schreiben, Prüfen und Korrigieren von Code in mehreren Dutzend Sprachen, "
                  "mit einem Kontext von 128K Token für große Dateien.",
            "it": "Assistente di programmazione per scrivere, rivedere e correggere codice in diverse decine di "
                  "linguaggi, con un contesto di 128K token per i file voluminosi.",
            "es": "Asistente de programación para escribir, revisar y corregir código en varias decenas de lenguajes, "
                  "con un contexto de 128K tokens para los archivos grandes.",
        },
    },
    {
        "id": "granite-3.3-2b",
        "name": "Granite 3.3 2B",
        "publisher": "IBM",
        "hub": ("ibm-granite/granite-3.3-2b-instruct-GGUF", "granite-3.3-2b-instruct-Q4_K_M.gguf"),
        "quantization": "Q4_K_M",
        "roles": ["general", "code"],
        "tools": True,
        "reasoning": False,
        "license": "Apache-2.0",
        "description": {
            "en": "IBM model for business tasks: summaries, extraction, classification, code and tool use, with a "
                  "context of 128K tokens.",
            "fr": "Modèle d'IBM pour les tâches d'entreprise : résumés, extraction, classification, code et usage "
                  "d'outils, avec un contexte de 128K jetons.",
            "de": "IBM-Modell für Unternehmensaufgaben: Zusammenfassungen, Extraktion, Klassifizierung, Code und "
                  "Werkzeugnutzung, mit einem Kontext von 128K Token.",
            "it": "Modello di IBM per le attività aziendali: riassunti, estrazione, classificazione, codice e uso di "
                  "strumenti, con un contesto di 128K token.",
            "es": "Modelo de IBM para tareas empresariales: resúmenes, extracción, clasificación, código y uso de "
                  "herramientas, con un contexto de 128K tokens.",
        },
    },
    {
        "id": "granite-3.3-8b",
        "name": "Granite 3.3 8B",
        "publisher": "IBM",
        "hub": ("ibm-granite/granite-3.3-8b-instruct-GGUF", "granite-3.3-8b-instruct-Q4_K_M.gguf"),
        "quantization": "Q4_K_M",
        "roles": ["general", "code"],
        "tools": True,
        "reasoning": False,
        "license": "Apache-2.0",
        "description": {
            "en": "Larger IBM model for business tasks, documents and code, with tool use and a context of 128K "
                  "tokens, for computers with 16 GB of memory.",
            "fr": "Modèle d'IBM plus grand pour les tâches d'entreprise, les documents et le code, avec usage "
                  "d'outils et un contexte de 128K jetons, pour les ordinateurs dotés de 16 Go de mémoire.",
            "de": "Größeres IBM-Modell für Unternehmensaufgaben, Dokumente und Code, mit Werkzeugnutzung und einem "
                  "Kontext von 128K Token, für Computer mit 16 GB Arbeitsspeicher.",
            "it": "Modello di IBM più grande per attività aziendali, documenti e codice, con uso di strumenti e un "
                  "contesto di 128K token, per computer con 16 GB di memoria.",
            "es": "Modelo de IBM más grande para tareas empresariales, documentos y código, con uso de herramientas y "
                  "un contexto de 128K tokens, para ordenadores con 16 GB de memoria.",
        },
    },
    {
        "id": "foundation-sec-8b",
        "name": "Foundation-Sec 1.1 8B",
        "publisher": "Cisco Foundation AI",
        "hub": ("fdtn-ai/Foundation-Sec-1.1-8B-Instruct-Q4_K_M-GGUF", "foundation-sec-1.1-8b-instruct-q4_k_m.gguf"),
        "quantization": "Q4_K_M",
        "roles": ["security"],
        "tools": False,
        "reasoning": False,
        "license": "Llama 3.1 Community License; Apache-2.0 for the changes of Cisco",
        "description": {
            "en": "Cybersecurity assistant trained on threat intelligence, vulnerabilities, attack techniques and "
                  "security operations: analysis of alerts and logs, explanation of attacks and of their "
                  "mitigations. It answers in English.",
            "fr": "Assistant de cybersécurité entraîné sur le renseignement sur les menaces, les vulnérabilités, "
                  "les techniques d'attaque et les opérations de sécurité : analyse d'alertes et de journaux, "
                  "explication des attaques et de leurs parades. Il répond en anglais.",
            "de": "Cybersicherheitsassistent, trainiert auf Bedrohungsinformationen, Schwachstellen, Angriffstechniken "
                  "und Sicherheitsbetrieb: Analyse von Alarmen und Protokollen, Erklärung von Angriffen und ihren "
                  "Gegenmaßnahmen. Er antwortet auf Englisch.",
            "it": "Assistente di cybersicurezza addestrato su threat intelligence, vulnerabilità, tecniche di attacco "
                  "e operazioni di sicurezza: analisi di allarmi e log, spiegazione degli attacchi e delle loro "
                  "contromisure. Risponde in inglese.",
            "es": "Asistente de ciberseguridad entrenado en inteligencia de amenazas, vulnerabilidades, técnicas de "
                  "ataque y operaciones de seguridad: análisis de alertas y registros, explicación de los ataques y "
                  "de sus contramedidas. Responde en inglés.",
        },
    },
]


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def from_hub(model):
    repo, name = model["hub"]
    info = get_json("https://huggingface.co/api/models/" + repo)
    tree = get_json("https://huggingface.co/api/models/%s/tree/%s" % (repo, info["sha"]))
    entry = next(t for t in tree if t.get("path") == name)
    size = entry["lfs"]["size"]
    sha = entry["lfs"]["oid"]
    gguf = info.get("gguf") or {}
    out = {k: v for k, v in model.items() if k != "hub"}
    out["arch"] = gguf.get("architecture", "")
    out["parameters"] = gguf.get("total", 0)
    out["context"] = gguf.get("context_length", 0)
    out["source"] = "https://huggingface.co/" + repo
    out["size"] = size
    out["sha256"] = sha
    out["files"] = [{
        "url": "https://huggingface.co/%s/resolve/%s/%s" % (repo, info["sha"], name),
        "size": size,
        "sha256": sha,
    }]
    return out


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    models = []
    for m in MODELS:
        assert all(r in ROLES for r in m["roles"]), m["id"]
        assert "embedding" not in m["roles"] or (m["roles"] == ["embedding"] and m.get("embedding")), m["id"]
        assert set(m["description"]) == {"en", "fr", "de", "it", "es"}, m["id"]
        models.append(from_hub(m) if "hub" in m else m)
        print("%-22s %6.2f GB  %s" % (models[-1]["id"], models[-1]["size"] / 1e9, models[-1]["sha256"]))
    for embedding in (False, True):
        defaults = [m["id"] for m in models if m.get("default") and ("embedding" in m["roles"]) == embedding]
        assert len(defaults) == 1, defaults
    # The time of publication, which the services compare as a string to keep the newer of
    # two catalogs: the earlier catalogs gave the day alone, which sorts before any time of it.
    updated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    catalog = {"format": 1, "updated": updated, "models": models}
    with open(sys.argv[1], "w", encoding="utf-8", newline="\n") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
