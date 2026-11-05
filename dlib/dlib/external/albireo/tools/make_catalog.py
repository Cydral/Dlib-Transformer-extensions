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

A model that reads images names its projector, the image encoder published beside the
decoder ("projector": repository, file and quantization of the mmproj GGUF file); its entry
then carries the files, the size and the SHA-256 of the projector, which the service
downloads with the decoder.  "inputs" gives what a model reads, text and images.

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
INPUTS = ("text", "image")

MODELS = [
    {
        "id": "qwen3-vl-2b-thinking",
        "name": "Qwen3-VL 2B Thinking",
        "publisher": "Alibaba",
        "hub": ("Qwen/Qwen3-VL-2B-Thinking-GGUF", "Qwen3VL-2B-Thinking-Q4_K_M.gguf"),
        "projector": ("Qwen/Qwen3-VL-2B-Thinking-GGUF", "mmproj-Qwen3VL-2B-Thinking-Q8_0.gguf", "Q8_0"),
        "quantization": "Q4_K_M",
        "roles": ["general", "code", "security"],
        "inputs": ["text", "image"],
        "default": True,
        "tools": True,
        "reasoning": True,
        "license": "Apache-2.0",
        "description": {
            "en": "Multimodal assistant that reads images and text and reasons step by step before answering: "
                  "conversation, documents, screenshots, diagrams, code and security. Albireo installs it by "
                  "default: half the size of the 4B version, it answers faster, from 8 GB of memory on.",
            "fr": "Assistant multimodal qui lit les images et le texte et raisonne étape par étape avant de "
                  "répondre : discussion, documents, captures d'écran, schémas, code et sécurité. Albireo "
                  "l'installe par défaut : deux fois plus petit que la version 4B, il répond plus vite, dès 8 Go "
                  "de mémoire.",
            "de": "Multimodaler Assistent, der Bilder und Text liest und vor der Antwort Schritt für Schritt "
                  "nachdenkt: Gespräche, Dokumente, Bildschirmfotos, Diagramme, Code und Sicherheit. Albireo "
                  "installiert ihn standardmäßig: halb so groß wie die 4B-Fassung, antwortet er schneller, ab 8 GB "
                  "Arbeitsspeicher.",
            "it": "Assistente multimodale che legge immagini e testo e ragiona passo dopo passo prima di "
                  "rispondere: conversazione, documenti, schermate, schemi, codice e sicurezza. Albireo lo installa "
                  "per impostazione predefinita: grande la metà della versione 4B, risponde più rapidamente, a "
                  "partire da 8 GB di memoria.",
            "es": "Asistente multimodal que lee imágenes y texto y razona paso a paso antes de responder: "
                  "conversación, documentos, capturas de pantalla, diagramas, código y seguridad. Albireo lo "
                  "instala por defecto: con la mitad del tamaño de la versión 4B, responde más rápido, a partir de "
                  "8 GB de memoria.",
        },
    },
    {
        "id": "qwen3-vl-4b-thinking",
        "name": "Qwen3-VL 4B Thinking",
        "publisher": "Alibaba",
        "hub": ("Qwen/Qwen3-VL-4B-Thinking-GGUF", "Qwen3VL-4B-Thinking-Q4_K_M.gguf"),
        "projector": ("Qwen/Qwen3-VL-4B-Thinking-GGUF", "mmproj-Qwen3VL-4B-Thinking-Q8_0.gguf", "Q8_0"),
        "quantization": "Q4_K_M",
        "roles": ["general", "code", "security"],
        "inputs": ["text", "image"],
        "tools": True,
        "reasoning": True,
        "license": "Apache-2.0",
        "description": {
            "en": "Multimodal assistant that reads images and text and reasons step by step before answering: "
                  "conversation, documents, screenshots, diagrams, code and security. Twice the size of the 2B "
                  "version installed by default, it reasons in more depth; for computers with 8 GB of memory.",
            "fr": "Assistant multimodal qui lit les images et le texte et raisonne étape par étape avant de "
                  "répondre : discussion, documents, captures d'écran, schémas, code et sécurité. Deux fois plus "
                  "grand que la version 2B installée par défaut, il raisonne avec plus de profondeur ; pour les "
                  "ordinateurs dotés de 8 Go de mémoire.",
            "de": "Multimodaler Assistent, der Bilder und Text liest und vor der Antwort Schritt für Schritt "
                  "nachdenkt: Gespräche, Dokumente, Bildschirmfotos, Diagramme, Code und Sicherheit. Doppelt so "
                  "groß wie die standardmäßig installierte 2B-Fassung, denkt er gründlicher nach; für Computer mit "
                  "8 GB Arbeitsspeicher.",
            "it": "Assistente multimodale che legge immagini e testo e ragiona passo dopo passo prima di "
                  "rispondere: conversazione, documenti, schermate, schemi, codice e sicurezza. Grande il doppio "
                  "della versione 2B installata per impostazione predefinita, ragiona con maggiore profondità; per "
                  "computer con 8 GB di memoria.",
            "es": "Asistente multimodal que lee imágenes y texto y razona paso a paso antes de responder: "
                  "conversación, documentos, capturas de pantalla, diagramas, código y seguridad. Con el doble del "
                  "tamaño de la versión 2B instalada por defecto, razona con más profundidad; para ordenadores con "
                  "8 GB de memoria.",
        },
    },
    {
        "id": "qwen3-embedding-0.6b",
        "name": "Qwen3 Embedding 0.6B",
        "publisher": "Alibaba",
        "hub": ("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
        "quantization": "Q8_0",
        "roles": ["embedding"],
        "inputs": ["text"],
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
        "id": "qwen3-vl-2b-instruct",
        "name": "Qwen3-VL 2B Instruct",
        "publisher": "Alibaba",
        "hub": ("Qwen/Qwen3-VL-2B-Instruct-GGUF", "Qwen3VL-2B-Instruct-Q4_K_M.gguf"),
        "projector": ("Qwen/Qwen3-VL-2B-Instruct-GGUF", "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf", "Q8_0"),
        "quantization": "Q4_K_M",
        "roles": ["general"],
        "inputs": ["text", "image"],
        "tools": True,
        "reasoning": False,
        "license": "Apache-2.0",
        "description": {
            "en": "Compact multimodal model for the text of images and scanned documents, screenshots and short "
                  "answers. Its size lets it answer quickly on any computer; its answers on text are less deep than "
                  "those of the larger models.",
            "fr": "Modèle multimodal compact pour le texte des images et des documents numérisés, les captures "
                  "d'écran et les réponses courtes. Sa taille lui permet de répondre rapidement sur tout "
                  "ordinateur ; ses réponses sur le texte sont moins approfondies que celles des modèles plus grands.",
            "de": "Kompaktes multimodales Modell für den Text von Bildern und gescannten Dokumenten, Bildschirmfotos "
                  "und kurze Antworten. Seine Größe erlaubt schnelle Antworten auf jedem Computer; seine Antworten "
                  "zu Texten sind weniger tief als die der größeren Modelle.",
            "it": "Modello multimodale compatto per il testo di immagini e documenti scansionati, schermate e "
                  "risposte brevi. Le sue dimensioni gli permettono di rispondere rapidamente su qualsiasi computer; "
                  "le sue risposte sui testi sono meno approfondite di quelle dei modelli più grandi.",
            "es": "Modelo multimodal compacto para el texto de imágenes y documentos escaneados, capturas de pantalla "
                  "y respuestas breves. Su tamaño le permite responder rápidamente en cualquier ordenador; sus "
                  "respuestas sobre textos son menos profundas que las de los modelos más grandes.",
        },
    },
    {
        "id": "qwen3-vl-4b-instruct",
        "name": "Qwen3-VL 4B Instruct",
        "publisher": "Alibaba",
        "hub": ("Qwen/Qwen3-VL-4B-Instruct-GGUF", "Qwen3VL-4B-Instruct-Q4_K_M.gguf"),
        "projector": ("Qwen/Qwen3-VL-4B-Instruct-GGUF", "mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf", "Q8_0"),
        "quantization": "Q4_K_M",
        "roles": ["general", "code"],
        "inputs": ["text", "image"],
        "tools": True,
        "reasoning": False,
        "license": "Apache-2.0",
        "description": {
            "en": "Multimodal assistant that reads images and text and answers directly, without a reasoning phase: "
                  "description and reading of images, scanned documents and screenshots, conversation, writing and "
                  "code. It runs on computers with 8 GB of memory.",
            "fr": "Assistant multimodal qui lit les images et le texte et répond directement, sans phase de "
                  "raisonnement : description et lecture d'images, de documents numérisés et de captures d'écran, "
                  "discussion, rédaction et code. Il fonctionne sur les ordinateurs dotés de 8 Go de mémoire.",
            "de": "Multimodaler Assistent, der Bilder und Text liest und direkt antwortet, ohne Denkphase: "
                  "Beschreibung und Lesen von Bildern, gescannten Dokumenten und Bildschirmfotos, Gespräche, Texte "
                  "und Code. Er läuft auf Computern mit 8 GB Arbeitsspeicher.",
            "it": "Assistente multimodale che legge immagini e testo e risponde direttamente, senza fase di "
                  "ragionamento: descrizione e lettura di immagini, documenti scansionati e schermate, conversazione, "
                  "scrittura e codice. Funziona su computer con 8 GB di memoria.",
            "es": "Asistente multimodal que lee imágenes y texto y responde directamente, sin fase de razonamiento: "
                  "descripción y lectura de imágenes, documentos escaneados y capturas de pantalla, conversación, "
                  "redacción y código. Funciona en ordenadores con 8 GB de memoria.",
        },
    },
    {
        "id": "qwen3-vl-8b-instruct",
        "name": "Qwen3-VL 8B Instruct",
        "publisher": "Alibaba",
        "hub": ("Qwen/Qwen3-VL-8B-Instruct-GGUF", "Qwen3VL-8B-Instruct-Q4_K_M.gguf"),
        "projector": ("Qwen/Qwen3-VL-8B-Instruct-GGUF", "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf", "Q8_0"),
        "quantization": "Q4_K_M",
        "roles": ["general", "code", "security"],
        "inputs": ["text", "image"],
        "tools": True,
        "reasoning": False,
        "license": "Apache-2.0",
        "description": {
            "en": "Larger multimodal assistant that answers directly: reading of documents, tables and screenshots, "
                  "analysis of images and diagrams, questions of code and security. For computers with 16 GB of "
                  "memory or a graphics processor.",
            "fr": "Assistant multimodal plus grand qui répond directement : lecture de documents, de tableaux et de "
                  "captures d'écran, analyse d'images et de schémas, questions de code et de sécurité. Pour les "
                  "ordinateurs dotés de 16 Go de mémoire ou d'un processeur graphique.",
            "de": "Größerer multimodaler Assistent, der direkt antwortet: Lesen von Dokumenten, Tabellen und "
                  "Bildschirmfotos, Analyse von Bildern und Diagrammen, Fragen zu Code und Sicherheit. Für Computer "
                  "mit 16 GB Arbeitsspeicher oder einem Grafikprozessor.",
            "it": "Assistente multimodale più grande che risponde direttamente: lettura di documenti, tabelle e "
                  "schermate, analisi di immagini e schemi, domande di codice e sicurezza. Per computer con 16 GB di "
                  "memoria o un processore grafico.",
            "es": "Asistente multimodal más grande que responde directamente: lectura de documentos, tablas y "
                  "capturas de pantalla, análisis de imágenes y diagramas, preguntas de código y seguridad. Para "
                  "ordenadores con 16 GB de memoria o un procesador gráfico.",
        },
    },
    {
        "id": "qwen3-vl-8b-thinking",
        "name": "Qwen3-VL 8B Thinking",
        "publisher": "Alibaba",
        "hub": ("Qwen/Qwen3-VL-8B-Thinking-GGUF", "Qwen3VL-8B-Thinking-Q4_K_M.gguf"),
        "projector": ("Qwen/Qwen3-VL-8B-Thinking-GGUF", "mmproj-Qwen3VL-8B-Thinking-Q8_0.gguf", "Q8_0"),
        "quantization": "Q4_K_M",
        "roles": ["general", "code", "security"],
        "inputs": ["text", "image"],
        "tools": True,
        "reasoning": True,
        "license": "Apache-2.0",
        "description": {
            "en": "Larger multimodal assistant that reasons step by step before answering, for the hardest questions "
                  "on images, code and security: logs and diagrams in screenshots, architectures, mathematics. For "
                  "computers with 16 GB of memory or a graphics processor.",
            "fr": "Assistant multimodal plus grand qui raisonne étape par étape avant de répondre, pour les questions "
                  "les plus difficiles sur les images, le code et la sécurité : journaux et schémas en captures "
                  "d'écran, architectures, mathématiques. Pour les ordinateurs dotés de 16 Go de mémoire ou d'un "
                  "processeur graphique.",
            "de": "Größerer multimodaler Assistent, der vor der Antwort Schritt für Schritt nachdenkt, für die "
                  "schwierigsten Fragen zu Bildern, Code und Sicherheit: Protokolle und Diagramme in "
                  "Bildschirmfotos, Architekturen, Mathematik. Für Computer mit 16 GB Arbeitsspeicher oder einem "
                  "Grafikprozessor.",
            "it": "Assistente multimodale più grande che ragiona passo dopo passo prima di rispondere, per le domande "
                  "più difficili su immagini, codice e sicurezza: log e schemi in schermate, architetture, "
                  "matematica. Per computer con 16 GB di memoria o un processore grafico.",
            "es": "Asistente multimodal más grande que razona paso a paso antes de responder, para las preguntas más "
                  "difíciles sobre imágenes, código y seguridad: registros y diagramas en capturas de pantalla, "
                  "arquitecturas, matemáticas. Para ordenadores con 16 GB de memoria o un procesador gráfico.",
        },
    },
    {
        "id": "eurovlm-9b-preview",
        "name": "EuroVLM 9B Preview",
        "publisher": "EuroLLM consortium",
        "hub": ("nyuuzyou/EuroVLM-9B-Preview-GGUF", "EuroVLM-9B-Preview-Q4_K_M.gguf"),
        "projector": ("nyuuzyou/EuroVLM-9B-Preview-GGUF", "mmproj-model-f16.gguf", "F16"),
        "quantization": "Q4_K_M",
        "roles": ["general"],
        "inputs": ["text", "image"],
        "tools": False,
        "reasoning": False,
        "license": "Apache-2.0",
        "description": {
            "en": "Preview of the European vision-language model built on EuroLLM 9B: description of images and "
                  "documents, and answers about them, in the 24 official languages of the European Union and 11 other "
                  "languages. Its publishers give it as a version still under development. It answers directly, for "
                  "computers with 16 GB of memory.",
            "fr": "Préversion du modèle vision-langage européen construit sur EuroLLM 9B : description d'images et de "
                  "documents, et réponses à leur sujet, dans les 24 langues officielles de l'Union européenne et 11 "
                  "autres langues. Ses éditeurs la présentent comme une version encore en développement. Il répond "
                  "directement, pour les ordinateurs dotés de 16 Go de mémoire.",
            "de": "Vorabversion des europäischen Bild-Sprach-Modells auf der Grundlage von EuroLLM 9B: Beschreibung "
                  "von Bildern und Dokumenten und Antworten dazu in den 24 Amtssprachen der Europäischen Union und 11 "
                  "weiteren Sprachen. Seine Herausgeber stellen es als Version in Entwicklung vor. Es antwortet "
                  "direkt, für Computer mit 16 GB Arbeitsspeicher.",
            "it": "Anteprima del modello visione-linguaggio europeo basato su EuroLLM 9B: descrizione di immagini e "
                  "documenti, e risposte su di essi, nelle 24 lingue ufficiali dell'Unione europea e in altre 11 "
                  "lingue. I suoi autori la presentano come una versione ancora in sviluppo. Risponde direttamente, "
                  "per computer con 16 GB di memoria.",
            "es": "Versión preliminar del modelo de visión y lenguaje europeo construido sobre EuroLLM 9B: descripción "
                  "de imágenes y documentos, y respuestas sobre ellos, en las 24 lenguas oficiales de la Unión Europea "
                  "y en otras 11 lenguas. Sus autores la presentan como una versión aún en desarrollo. Responde "
                  "directamente, para ordenadores con 16 GB de memoria.",
        },
    },
    {
        "id": "phi-4-mini",
        "name": "Phi-4-mini",
        "publisher": "Microsoft",
        "hub": ("bartowski/microsoft_Phi-4-mini-instruct-GGUF", "microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"),
        "quantization": "Q4_K_M",
        "roles": ["general"],
        "inputs": ["text"],
        "tools": False,
        "reasoning": False,
        "license": "MIT",
        "description": {
            "en": "Compact general assistant from Microsoft for conversation, writing and mathematics, in English and "
                  "some twenty other languages, with a context of 128K tokens. It answers directly, without a "
                  "reasoning phase, and runs on computers with 8 GB of memory.",
            "fr": "Assistant généraliste compact de Microsoft pour la discussion, la rédaction et les mathématiques, "
                  "en anglais et dans une vingtaine d'autres langues, avec un contexte de 128K jetons. Il répond "
                  "directement, sans phase de raisonnement, et fonctionne sur les ordinateurs dotés de 8 Go de "
                  "mémoire.",
            "de": "Kompakter allgemeiner Assistent von Microsoft für Gespräche, Texte und Mathematik, auf Englisch "
                  "und in rund zwanzig weiteren Sprachen, mit einem Kontext von 128K Token. Er antwortet direkt, ohne "
                  "Denkphase, und läuft auf Computern mit 8 GB Arbeitsspeicher.",
            "it": "Assistente generale compatto di Microsoft per conversazione, scrittura e matematica, in inglese e "
                  "in una ventina di altre lingue, con un contesto di 128K token. Risponde direttamente, senza fase "
                  "di ragionamento, e funziona su computer con 8 GB di memoria.",
            "es": "Asistente general compacto de Microsoft para conversación, redacción y matemáticas, en inglés y en "
                  "una veintena de otras lenguas, con un contexto de 128K tokens. Responde directamente, sin fase de "
                  "razonamiento, y funciona en ordenadores con 8 GB de memoria.",
        },
    },
    {
        "id": "ministral-3-3b",
        "name": "Ministral 3 3B",
        "publisher": "Mistral AI",
        "hub": ("mistralai/Ministral-3-3B-Instruct-2512-GGUF", "Ministral-3-3B-Instruct-2512-Q4_K_M.gguf"),
        "projector": ("mistralai/Ministral-3-3B-Instruct-2512-GGUF", "Ministral-3-3B-Instruct-2512-BF16-mmproj.gguf",
                      "BF16"),
        "quantization": "Q4_K_M",
        "roles": ["general"],
        "inputs": ["text", "image"],
        "tools": True,
        "reasoning": False,
        "license": "Apache-2.0",
        "description": {
            "en": "General assistant from Mistral AI, developed in partnership with NVIDIA and trained on many "
                  "European languages, with a context of 256K tokens. It reads images, uses tools and answers "
                  "directly, without a reasoning phase, on computers with 8 GB of memory.",
            "fr": "Assistant généraliste de Mistral AI, développé en partenariat avec NVIDIA et entraîné sur de "
                  "nombreuses langues européennes, avec un contexte de 256K jetons. Il lit les images, utilise des "
                  "outils et répond directement, sans phase de raisonnement, sur les ordinateurs dotés de 8 Go de "
                  "mémoire.",
            "de": "Allgemeiner Assistent von Mistral AI, in Partnerschaft mit NVIDIA entwickelt und auf vielen "
                  "europäischen Sprachen trainiert, mit einem Kontext von 256K Token. Er liest Bilder, nutzt "
                  "Werkzeuge und antwortet direkt, ohne Denkphase, auf Computern mit 8 GB Arbeitsspeicher.",
            "it": "Assistente generale di Mistral AI, sviluppato in collaborazione con NVIDIA e addestrato su molte "
                  "lingue europee, con un contesto di 256K token. Legge le immagini, usa strumenti e risponde "
                  "direttamente, senza fase di ragionamento, su computer con 8 GB di memoria.",
            "es": "Asistente general de Mistral AI, desarrollado en colaboración con NVIDIA y entrenado en muchas "
                  "lenguas europeas, con un contexto de 256K tokens. Lee imágenes, usa herramientas y responde "
                  "directamente, sin fase de razonamiento, en ordenadores con 8 GB de memoria.",
        },
    },
    {
        "id": "ministral-3-8b",
        "name": "Ministral 3 8B",
        "publisher": "Mistral AI",
        "hub": ("mistralai/Ministral-3-8B-Instruct-2512-GGUF", "Ministral-3-8B-Instruct-2512-Q4_K_M.gguf"),
        "projector": ("mistralai/Ministral-3-8B-Instruct-2512-GGUF", "Ministral-3-8B-Instruct-2512-BF16-mmproj.gguf",
                      "BF16"),
        "quantization": "Q4_K_M",
        "roles": ["general"],
        "inputs": ["text", "image"],
        "tools": True,
        "reasoning": False,
        "license": "Apache-2.0",
        "description": {
            "en": "Larger general assistant from Mistral AI, developed in partnership with NVIDIA and trained on many "
                  "European languages, with a context of 256K tokens. It reads images, uses tools and answers "
                  "directly, for computers with 16 GB of memory or a graphics processor.",
            "fr": "Assistant généraliste plus grand de Mistral AI, développé en partenariat avec NVIDIA et entraîné "
                  "sur de nombreuses langues européennes, avec un contexte de 256K jetons. Il lit les images, utilise "
                  "des outils et répond directement, pour les ordinateurs dotés de 16 Go de mémoire ou d'un "
                  "processeur graphique.",
            "de": "Größerer allgemeiner Assistent von Mistral AI, in Partnerschaft mit NVIDIA entwickelt und auf "
                  "vielen europäischen Sprachen trainiert, mit einem Kontext von 256K Token. Er liest Bilder, nutzt "
                  "Werkzeuge und antwortet direkt, für Computer mit 16 GB Arbeitsspeicher oder einem Grafikprozessor.",
            "it": "Assistente generale più grande di Mistral AI, sviluppato in collaborazione con NVIDIA e addestrato "
                  "su molte lingue europee, con un contesto di 256K token. Legge le immagini, usa strumenti e risponde "
                  "direttamente, per computer con 16 GB di memoria o un processore grafico.",
            "es": "Asistente general más grande de Mistral AI, desarrollado en colaboración con NVIDIA y entrenado en "
                  "muchas lenguas europeas, con un contexto de 256K tokens. Lee imágenes, usa herramientas y responde "
                  "directamente, para ordenadores con 16 GB de memoria o un procesador gráfico.",
        },
    },
    {
        "id": "eurollm-9b",
        "name": "EuroLLM 9B Instruct",
        "publisher": "EuroLLM consortium",
        "hub": ("mradermacher/EuroLLM-9B-Instruct-2512-GGUF", "EuroLLM-9B-Instruct-2512.Q4_K_M.gguf"),
        "quantization": "Q4_K_M",
        "roles": ["general"],
        "inputs": ["text"],
        "tools": False,
        "reasoning": False,
        "license": "Apache-2.0",
        "description": {
            "en": "European assistant trained on the 24 official languages of the European Union and 11 other "
                  "languages, for writing, translation and conversation in these languages, with a context of 32K "
                  "tokens. It answers directly, without a reasoning phase, and suits code and mathematics less. For "
                  "computers with 16 GB of memory.",
            "fr": "Assistant européen entraîné sur les 24 langues officielles de l'Union européenne et 11 autres "
                  "langues, pour la rédaction, la traduction et la discussion dans ces langues, avec un contexte de "
                  "32K jetons. Il répond directement, sans phase de raisonnement, et convient moins au code et aux "
                  "mathématiques. Pour les ordinateurs dotés de 16 Go de mémoire.",
            "de": "Europäischer Assistent, trainiert auf den 24 Amtssprachen der Europäischen Union und 11 weiteren "
                  "Sprachen, für Texte, Übersetzungen und Gespräche in diesen Sprachen, mit einem Kontext von 32K "
                  "Token. Er antwortet direkt, ohne Denkphase, und eignet sich weniger für Code und Mathematik. Für "
                  "Computer mit 16 GB Arbeitsspeicher.",
            "it": "Assistente europeo addestrato sulle 24 lingue ufficiali dell'Unione europea e su altre 11 lingue, "
                  "per scrittura, traduzione e conversazione in queste lingue, con un contesto di 32K token. Risponde "
                  "direttamente, senza fase di ragionamento, ed è meno adatto al codice e alla matematica. Per "
                  "computer con 16 GB di memoria.",
            "es": "Asistente europeo entrenado en las 24 lenguas oficiales de la Unión Europea y en otras 11 lenguas, "
                  "para redacción, traducción y conversación en estas lenguas, con un contexto de 32K tokens. "
                  "Responde directamente, sin fase de razonamiento, y es menos adecuado para el código y las "
                  "matemáticas. Para ordenadores con 16 GB de memoria.",
        },
    },
    {
        "id": "granite-4.2-3b",
        "name": "Granite 4.2 3B",
        "publisher": "IBM",
        "hub": ("ibm-granite/granite-4.2-3b-GGUF", "granite-4.2-3b-Q4_K_M.gguf"),
        "quantization": "Q4_K_M",
        "roles": ["general", "code"],
        "inputs": ["text"],
        "tools": True,
        "reasoning": True,
        "license": "Apache-2.0",
        "description": {
            "en": "IBM model for business tasks, code and tool use, which can reason step by step before answering, "
                  "with a context of 128K tokens. It runs on computers with 8 GB of memory.",
            "fr": "Modèle d'IBM pour les tâches d'entreprise, le code et l'usage d'outils, qui peut raisonner étape "
                  "par étape avant de répondre, avec un contexte de 128K jetons. Il fonctionne sur les ordinateurs "
                  "dotés de 8 Go de mémoire.",
            "de": "IBM-Modell für Unternehmensaufgaben, Code und Werkzeugnutzung, das vor der Antwort Schritt für "
                  "Schritt nachdenken kann, mit einem Kontext von 128K Token. Es läuft auf Computern mit 8 GB "
                  "Arbeitsspeicher.",
            "it": "Modello di IBM per attività aziendali, codice e uso di strumenti, che può ragionare passo dopo "
                  "passo prima di rispondere, con un contesto di 128K token. Funziona su computer con 8 GB di memoria.",
            "es": "Modelo de IBM para tareas empresariales, código y uso de herramientas, que puede razonar paso a "
                  "paso antes de responder, con un contexto de 128K tokens. Funciona en ordenadores con 8 GB de "
                  "memoria.",
        },
    },
    {
        "id": "granite-4.2-8b",
        "name": "Granite 4.2 8B",
        "publisher": "IBM",
        "hub": ("ibm-granite/granite-4.2-8b-GGUF", "granite-4.2-8b-Q4_K_M.gguf"),
        "quantization": "Q4_K_M",
        "roles": ["general", "code"],
        "inputs": ["text"],
        "tools": True,
        "reasoning": True,
        "license": "Apache-2.0",
        "description": {
            "en": "Larger IBM model for business tasks, documents, code and tool use, which can reason step by step "
                  "before answering, with a context of 128K tokens, for computers with 16 GB of memory or a graphics "
                  "processor.",
            "fr": "Modèle d'IBM plus grand pour les tâches d'entreprise, les documents, le code et l'usage d'outils, "
                  "qui peut raisonner étape par étape avant de répondre, avec un contexte de 128K jetons, pour les "
                  "ordinateurs dotés de 16 Go de mémoire ou d'un processeur graphique.",
            "de": "Größeres IBM-Modell für Unternehmensaufgaben, Dokumente, Code und Werkzeugnutzung, das vor der "
                  "Antwort Schritt für Schritt nachdenken kann, mit einem Kontext von 128K Token, für Computer mit "
                  "16 GB Arbeitsspeicher oder einem Grafikprozessor.",
            "it": "Modello di IBM più grande per attività aziendali, documenti, codice e uso di strumenti, che può "
                  "ragionare passo dopo passo prima di rispondere, con un contesto di 128K token, per computer con "
                  "16 GB di memoria o un processore grafico.",
            "es": "Modelo de IBM más grande para tareas empresariales, documentos, código y uso de herramientas, que "
                  "puede razonar paso a paso antes de responder, con un contexto de 128K tokens, para ordenadores con "
                  "16 GB de memoria o un procesador gráfico.",
        },
    },
    {
        "id": "qwen3-8b",
        "name": "Qwen3 8B",
        "publisher": "Alibaba",
        "hub": ("Qwen/Qwen3-8B-GGUF", "Qwen3-8B-Q4_K_M.gguf"),
        "quantization": "Q4_K_M",
        "roles": ["general", "code", "security"],
        "inputs": ["text"],
        "tools": True,
        "reasoning": True,
        "license": "Apache-2.0",
        "description": {
            "en": "Larger general assistant with step by step reasoning and tool use, at ease with code and with the "
                  "knowledge of cybersecurity, for computers with 16 GB of memory or a graphics processor.",
            "fr": "Assistant généraliste plus grand, avec raisonnement étape par étape et usage d'outils, à l'aise "
                  "avec le code et les connaissances en cybersécurité, pour les ordinateurs dotés de 16 Go de mémoire "
                  "ou d'un processeur graphique.",
            "de": "Größerer allgemeiner Assistent mit schrittweisem Denken und Werkzeugnutzung, sicher im Code und im "
                  "Wissen der Cybersicherheit, für Computer mit 16 GB Arbeitsspeicher oder einem Grafikprozessor.",
            "it": "Assistente generale più grande, con ragionamento passo dopo passo e uso di strumenti, a suo agio "
                  "con il codice e le conoscenze di cybersicurezza, per computer con 16 GB di memoria o un processore "
                  "grafico.",
            "es": "Asistente general más grande, con razonamiento paso a paso y uso de herramientas, cómodo con el "
                  "código y los conocimientos de ciberseguridad, para ordenadores con 16 GB de memoria o un "
                  "procesador gráfico.",
        },
    },
    {
        "id": "qwen3-30b-a3b",
        "name": "Qwen3 30B A3B",
        "publisher": "Alibaba",
        "hub": ("Qwen/Qwen3-30B-A3B-GGUF", "Qwen3-30B-A3B-Q4_K_M.gguf"),
        "quantization": "Q4_K_M",
        "roles": ["general", "code", "security"],
        "inputs": ["text"],
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
        "publisher": "Alibaba",
        "hub": ("Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"),
        "quantization": "Q4_K_M",
        "roles": ["code"],
        "inputs": ["text"],
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
        "publisher": "Alibaba",
        "hub": ("Qwen/Qwen2.5-Coder-7B-Instruct-GGUF", "qwen2.5-coder-7b-instruct-q4_k_m.gguf"),
        "quantization": "Q4_K_M",
        "roles": ["code"],
        "inputs": ["text"],
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
        "id": "devstral-small-2",
        "name": "Devstral Small 2",
        "publisher": "Mistral AI",
        "hub": ("lmstudio-community/Devstral-Small-2-24B-Instruct-2512-GGUF",
                "Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf"),
        "projector": ("lmstudio-community/Devstral-Small-2-24B-Instruct-2512-GGUF",
                      "mmproj-Devstral-Small-2-24B-Instruct-2512-F16.gguf", "F16"),
        "quantization": "Q4_K_M",
        "roles": ["code"],
        "inputs": ["text", "image"],
        "tools": True,
        "reasoning": False,
        "license": "Apache-2.0",
        "description": {
            "en": "Programming agent from Mistral AI with 24 billion parameters, for the tasks that span a whole "
                  "codebase, the use of tools and the reading of screenshots and diagrams, with a context of 256K "
                  "tokens. It answers directly, without a reasoning phase, for computers with 32 GB of memory or a "
                  "graphics processor of 16 GB.",
            "fr": "Agent de programmation de Mistral AI de 24 milliards de paramètres, pour les tâches qui portent sur "
                  "tout un code source, l'usage d'outils et la lecture de captures d'écran et de schémas, avec un "
                  "contexte de 256K jetons. Il répond directement, sans phase de raisonnement, pour les ordinateurs "
                  "dotés de 32 Go de mémoire ou d'un processeur graphique de 16 Go.",
            "de": "Programmieragent von Mistral AI mit 24 Milliarden Parametern, für Aufgaben über eine ganze "
                  "Codebasis, die Nutzung von Werkzeugen und das Lesen von Bildschirmfotos und Diagrammen, mit einem "
                  "Kontext von 256K Token. Er antwortet direkt, ohne Denkphase, für Computer mit 32 GB "
                  "Arbeitsspeicher oder einem Grafikprozessor mit 16 GB.",
            "it": "Agente di programmazione di Mistral AI da 24 miliardi di parametri, per i compiti che riguardano "
                  "un intero codice sorgente, l'uso di strumenti e la lettura di schermate e diagrammi, con un "
                  "contesto di 256K token. Risponde direttamente, senza fase di ragionamento, per computer con 32 GB "
                  "di memoria o un processore grafico da 16 GB.",
            "es": "Agente de programación de Mistral AI de 24.000 millones de parámetros, para las tareas que abarcan "
                  "todo un código fuente, el uso de herramientas y la lectura de capturas de pantalla y diagramas, "
                  "con un contexto de 256K tokens. Responde directamente, sin fase de razonamiento, para ordenadores "
                  "con 32 GB de memoria o un procesador gráfico de 16 GB.",
        },
    },
    {
        "id": "foundation-sec-8b",
        "name": "Foundation-Sec 1.1 8B",
        "publisher": "Cisco",
        "hub": ("fdtn-ai/Foundation-Sec-1.1-8B-Instruct-Q4_K_M-GGUF", "foundation-sec-1.1-8b-instruct-q4_k_m.gguf"),
        "quantization": "Q4_K_M",
        "roles": ["security"],
        "inputs": ["text"],
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
    {
        "id": "foundation-sec-8b-reasoning",
        "name": "Foundation-Sec 8B Reasoning",
        "publisher": "Cisco",
        "hub": ("fdtn-ai/Foundation-Sec-8B-Reasoning-Q4_K_M-GGUF", "foundation-sec-8b-reasoning-q4_k_m.gguf"),
        "quantization": "Q4_K_M",
        "roles": ["security"],
        "inputs": ["text"],
        "tools": False,
        "reasoning": True,
        "license": "Llama 3.1 Community License; Apache-2.0 for the changes of Cisco",
        "description": {
            "en": "Cybersecurity assistant that reasons step by step before answering: threat intelligence, mapping "
                  "of vulnerabilities to weaknesses (CVE, CWE) and to attack techniques, analysis of alerts and "
                  "configurations. It always reasons and answers in English, for computers with 16 GB of memory or a "
                  "graphics processor.",
            "fr": "Assistant de cybersécurité qui raisonne étape par étape avant de répondre : renseignement sur les "
                  "menaces, rattachement des vulnérabilités aux faiblesses (CVE, CWE) et aux techniques d'attaque, "
                  "analyse d'alertes et de configurations. Il raisonne toujours et répond en anglais, pour les "
                  "ordinateurs dotés de 16 Go de mémoire ou d'un processeur graphique.",
            "de": "Cybersicherheitsassistent, der vor der Antwort Schritt für Schritt nachdenkt: "
                  "Bedrohungsinformationen, Zuordnung von Schwachstellen zu Schwächen (CVE, CWE) und zu "
                  "Angriffstechniken, Analyse von Alarmen und Konfigurationen. Er denkt immer nach und antwortet auf "
                  "Englisch, für Computer mit 16 GB Arbeitsspeicher oder einem Grafikprozessor.",
            "it": "Assistente di cybersicurezza che ragiona passo dopo passo prima di rispondere: threat intelligence, "
                  "associazione delle vulnerabilità alle debolezze (CVE, CWE) e alle tecniche di attacco, analisi di "
                  "allarmi e configurazioni. Ragiona sempre e risponde in inglese, per computer con 16 GB di memoria "
                  "o un processore grafico.",
            "es": "Asistente de ciberseguridad que razona paso a paso antes de responder: inteligencia de amenazas, "
                  "asociación de las vulnerabilidades con las debilidades (CVE, CWE) y las técnicas de ataque, "
                  "análisis de alertas y configuraciones. Siempre razona y responde en inglés, para ordenadores con "
                  "16 GB de memoria o un procesador gráfico.",
        },
    },
]


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def hub_file(repo, name, cache):
    """The commit of a repository and the size and SHA-256 of one of its files at that commit."""
    if repo not in cache:
        info = get_json("https://huggingface.co/api/models/" + repo)
        tree = get_json("https://huggingface.co/api/models/%s/tree/%s" % (repo, info["sha"]))
        cache[repo] = (info, tree)
    info, tree = cache[repo]
    entry = next(t for t in tree if t.get("path") == name)
    return info, entry["lfs"]["size"], entry["lfs"]["oid"]


def from_hub(model, cache):
    repo, name = model["hub"]
    info, size, sha = hub_file(repo, name, cache)
    gguf = info.get("gguf") or {}
    out = {k: v for k, v in model.items() if k not in ("hub", "projector")}
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
    if "projector" in model:
        p_repo, p_name, p_quantization = model["projector"]
        p_info, p_size, p_sha = hub_file(p_repo, p_name, cache)
        out["projector"] = {
            "quantization": p_quantization,
            "size": p_size,
            "sha256": p_sha,
            "files": [{
                "url": "https://huggingface.co/%s/resolve/%s/%s" % (p_repo, p_info["sha"], p_name),
                "size": p_size,
                "sha256": p_sha,
            }],
        }
    return out


def check(condition, what):
    """Raises ValueError with what when condition is false: unlike assert, which python -O
    removes, the check holds however the program runs."""
    if not condition:
        raise ValueError(what)


def check_models(models):
    """Checks the list of the models before anything is read from the hub."""
    for m in models:
        check(all(r in ROLES for r in m["roles"]), "%s: a role is not one of %s" % (m["id"], ", ".join(ROLES)))
        check("embedding" not in m["roles"] or (m["roles"] == ["embedding"] and m.get("embedding")),
              "%s: an embedding model has the embedding role alone and \"embedding\": true" % m["id"])
        check(set(m["description"]) == {"en", "fr", "de", "it", "es"},
              "%s: the description has the languages en, fr, de, it and es, and no other" % m["id"])
        check(m["inputs"][0] == "text" and all(i in INPUTS for i in m["inputs"]),
              "%s: the inputs start with text and are among %s" % (m["id"], ", ".join(INPUTS)))
        check(("image" in m["inputs"]) == ("projector" in m),
              "%s: a model reads images when it names a projector, and only then" % m["id"])
    ids = [m["id"] for m in models]
    check(len(ids) == len(set(ids)), "identifiers listed twice: %s"
          % ", ".join(sorted({i for i in ids if ids.count(i) > 1})))
    for embedding in (False, True):
        defaults = [m["id"] for m in models if m.get("default") and ("embedding" in m["roles"]) == embedding]
        check(len(defaults) == 1, "one default %s model is needed, not %d: %s"
              % ("embedding" if embedding else "language", len(defaults), ", ".join(defaults) or "none"))


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    try:
        check_models(MODELS)
    except (ValueError, KeyError, IndexError) as e:
        print("error: %s" % e, file=sys.stderr)
        return 1
    models = []
    cache = {}
    for m in MODELS:
        models.append(from_hub(m, cache) if "hub" in m else m)
        e = models[-1]
        total = e["size"] + e.get("projector", {}).get("size", 0)
        print("%-28s %-18s %6.2f GB  %s" % (e["id"], e["publisher"], total / 1e9, e["sha256"]))
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
