# Dlib Transformer Extensions

> [!IMPORTANT]
> **Dlib Transformer Extensions** brings modern **Transformer-oriented modeling** into the **Dlib** ecosystem through reusable architectural components, language-model utilities, training helpers, progressive examples, and reusable checkpoints — all designed for practical use in standard **C++14**.
>
> The project now covers a complete lifecycle: **load an open-weight model from a GGUF container**, run it, **convert it into a native Dlib network**, **continue training it**, **specialize it by SFT and LoRA**, or **distill it into a smaller student of a different architecture** — without leaving C++ and without a Python runtime in the loop.

> [!NOTE]
> This repository is documented through two complementary entry points:
>
> - [`examples/`](examples) for the progressive workflows and runnable demonstrations
> - [`models/`](models) for reusable artifacts and pre-trained checkpoints

---

## At a glance

This repository exists to make **modern Transformer development more accessible, inspectable, and reusable inside Dlib**.

It is not just a collection of example programs, and it is not just a low-level patch set either. Instead, it provides a **full practical bridge** between:

- **core Transformer building blocks**
- **training and inference utilities**
- **reference usage programs**
- **reusable trained artifacts**

In concrete terms, the project gives Dlib users a way to explore, implement, test, and progressively industrialize Transformer-based workflows in a codebase that remains close to the strengths of the Dlib philosophy: **clarity, portability, composability, and C++ integration**.

---

## Project overview

Transformers were introduced in the 2017 paper **“Attention Is All You Need”**, which proposed an architecture based on attention mechanisms rather than recurrence or convolution.[^vaswani2017] The paper presented a full **encoder-decoder** architecture and emphasized improved parallelization and stronger modeling of sequence relationships.[^vaswani2017]

Since then, Transformers have become a dominant architecture across many machine learning applications. They are now widely used for a broad range of language-modeling and sequence-processing tasks, with encoder/decoder variants and self-attention at the core of their operation.[^google-transformers]

This repository brings that evolution into the Dlib world by extending Dlib with a practical base for:

- Transformer-oriented architectural components
- language-model data and utility layers
- training and scheduling support
- progressive example programs
- reusable trained model artifacts

The project is publicly exposed on GitHub as an open-source repository and is licensed under **BSL-1.0**, reinforcing its role as a reusable and inspectable codebase rather than a closed demonstration artifact.

---

## Why this project matters

### 1. It expands Dlib into a strategically important AI space
The repository positions Dlib within a domain that has become central to modern AI: Transformer-based modeling. In practice, this means giving Dlib users access to a family of methods that now underpins a wide range of state-of-the-art sequence and representation workflows.[^google-transformers][^islam2023]

### 2. It helps keep advanced model development open and inspectable
Because the repository is public and open source, it offers something many teams actively look for: a codebase that can be **read, modified, audited, adapted, and self-hosted**. That matters especially for organizations that want to reduce dependence on opaque hosted services and instead build **specialized, controlled, or sovereign AI workflows** around code they can inspect directly.

### 3. It makes specialization more realistic than generic model consumption
Large general-purpose models are powerful, but many real-world needs are domain-specific. This repository is particularly interesting because it does not stop at generic theory: it provides **example programs**, **training paths**, and **artifact organization** that make specialization more concrete.

### 4. It supports a full learning-to-reuse continuum
Many repositories are either too low-level to use easily or too high-level to teach anything. Here, the structure is more valuable because it spans:

- reusable library components
- practical workflows in `examples/`
- reusable artifacts in `models/`

That makes the project useful not only for experimentation, but also for reproducibility, adaptation, and eventual deployment.

### 5. It closes the loop between consuming a model and owning one
Most tooling around open-weight models stops at inference. This project deliberately goes further: a container read from disk becomes a **native Dlib network whose weights are ordinary trainable parameters**. From there the usual Dlib machinery applies — optimizers, visitors, serialization, fine-tuning — which turns a downloaded artifact into a starting point rather than an endpoint.

### 6. It turns the vocabulary into a decision rather than an inheritance

A published small model carries a tokenizer sized for the largest member of its family, and spends a third of its parameters on entries it barely needs. A project that builds its own tokenizer does not: it can size the vocabulary to the model, hold the embedding table to a sixth of the parameter count, and put the difference back into the blocks where it does work.

That freedom has to be used carefully, because a vocabulary is a fixed budget shared between writing systems that share nothing with each other, and one spent badly leaves a language segmented almost byte by byte — a defect no training curve reveals. The library measures it before a run starts rather than after.

### 7. It creates a foundation for sovereign and specialized model stacks in C++
One of the strongest strategic dimensions of the project is that it supports an approach where organizations can build **specialized**, **inspectable**, and potentially **sovereign** model stacks without leaving the C++ / Dlib ecosystem. That is particularly compelling when teams need stronger control over model behavior, runtime integration, training artifacts, or deployment boundaries.

---

## What are Transformers?

Transformers are a family of neural network architectures built around **attention**, especially **self-attention**, which allows each token in a sequence to directly weigh the importance of other tokens in the same sequence.[^google-transformers]

The original Transformer introduced in 2017 used an **encoder-decoder** design. In broad terms:

- the **encoder** transforms the input sequence into an internal representation
- the **decoder** consumes that representation to produce an output sequence

This architecture was first applied to machine translation, but the same underlying principles generalized well beyond that initial use case.[^vaswani2017]

### Why Transformers changed the field

Transformers became so influential because they solved several practical limitations of earlier sequence models:

- they reduced the sequential bottleneck of recurrent processing
- they improved the handling of long-range dependencies
- they scaled better under parallel hardware training
- they generalized across many task families

This is one of the key reasons the architecture went on to underpin major modern model families and expanded far beyond NLP.[^google-transformers][^islam2023]

### Main Transformer families today

Modern practice commonly distinguishes between:

- **encoder-only models** for representation and understanding tasks
- **decoder-only models** for autoregressive generation
- **encoder-decoder models** for conditional transformation and structured transduction

This distinction is important for reading this repository because the project already covers some Transformer-derived workflows more directly than others, while still leaving room for future expansion.

---

## Why Dlib for Transformers?

Dlib has long been appreciated for its emphasis on **clean APIs**, **strong engineering discipline**, and **practical C++ use**. This repository matters because it extends those strengths into a domain that is usually dominated by Python-first ecosystems.

That creates a valuable alternative for users who need one or more of the following:

- close integration with existing C++ systems
- inspectable and portable model code
- direct control over data paths and training logic
- reusable abstractions rather than opaque wrappers
- an environment where modern Transformer experimentation can remain aligned with a systems-engineering mindset

In other words, the project is not only about adding Transformers to Dlib. It is about doing so in a way that preserves the practical identity of the Dlib ecosystem.

---

## What the project currently covers

The project has grown from a set of Transformer building blocks into a **model lifecycle**, from an existing open-weight artifact to a specialized model of your own design.

### Interoperability with the open-weight ecosystem

**GGUF containers are read directly.** The reader parses the metadata, detects the architecture family, and reports what it found before touching a single weight. Detection currently spans the **llama, mistral, mixtral, gemma, gemma2, qwen2, qwen3 and granitemoe** families, which together account for the large majority of open-weight decoder models published today.

**Quantized weights are dequantized on the fly.** The legacy block formats (**Q4_0, Q4_1, Q5_0, Q5_1, Q8_0**), the whole **k-quant** range (**Q2_K** through **Q6_K**) and **IQ4_NL** are supported. The grid-based i-quants (**IQ1, IQ2, IQ3**) are not, and the loader says so rather than producing plausible noise.

**Compatibility is reported, not assumed.** Before any import, a compatibility pass lists what is supported, what is merely noted, and what blocks. Mixture-of-experts containers, for instance, are served by the runtime engine but explicitly refused by the static path, because the compiled network has no expert packing. An honest refusal is worth more than a silent approximation.

### Two execution paths, one behaviour

The project deliberately maintains **two ways of running a model**, and holds them to bit-level agreement:

- a **shape-dynamic runtime engine**, compiled once, that adapts to whatever container it is given;
- a **statically compiled network**, generated as a C++ header from the model's own geometry, where every dimension is a compile-time constant.

The dynamic path is what you use to explore a new model. The static path is what you use to **train** it, since it is an ordinary Dlib network. Both are driven through a shared chat endpoint, so a discrepancy between them is a bug rather than a matter of interpretation.

### Vision and multimodality

**Vision Transformers are first-class.** A `vision_transformer_config` assembles the patch embedding, a learned position table, **bidirectional** attention blocks — image patches have no reading order — and the pooling that turns a bag of patch vectors into one image vector. The same configuration serves as a **standalone image backbone** for classification or self-supervised pretraining, or as the **encoder of a multimodal model**, by naming an image input instead of a tensor input.

**Multimodal fusion is a layer.** A fusion layer holds the vision tower as a subnetwork and writes its output over the positions a chat template reserved for an image. Because the tower lives inside the network rather than beside it, **a gradient reaches it**: the vision path is trainable, not merely pluggable.

### An interactive interface for the models you build

A built-in **web chat interface** and an **OpenAI-compatible endpoint** (`/v1/chat/completions`) let you exercise a model as soon as it exists, with streaming, sampling controls, multi-model selection, and **inline image attachments** when the model carries a vision tower. Serving is shared code between the two execution paths, so what you observe in the browser is what the library computes.

### Training, specialization, and distillation

**Continued training from a converted model.** A GGUF container converted to a native archive holds the decoder, the vision tower when there is one, the tokenizer, and the pixel normalization. Nothing external is needed afterwards.

**Supervised fine-tuning with prompt masking.** Question-and-answer records are rendered through the model's own chat template, and only the answer positions are scored — the standard SFT objective, with the padding hidden from attention rather than merely ignored by the loss.

**Parameter-efficient adaptation.** **LoRA** and **DoRA** adapters attach to the attention projections and the feed-forward path, with the base weights frozen and the adapters merged back into the weights when training ends. Host-resident parameters (`--offload-params`) already provide half of what **QLoRA** needs; the quantized base is the remaining step.

**Knowledge distillation, teacher to student.** A teacher is recorded once over a corpus — its top-k logits at every position — into a reusable trace file. Students are then raised on that recording without the teacher ever running again. Because the transfer happens **at the logits**, the student's internals are unconstrained: a **dense** teacher can raise a **mixture-of-experts** student, wider, deeper, or narrower. The only thing both models must share is the tokenizer, and a fingerprint in the trace file makes any disagreement impossible to miss.

**Depth pruning, as the other way to make a model smaller.** Rather than raising a student from nothing, whole blocks are removed from a published model and the survivors are distilled back into shape. Which blocks go is measured rather than guessed: a few forward passes record how far each one moves what passes through it, and those whose output stays closest to their input are the cheapest to lose. Since the survivors keep the teacher's weights, the run repairs what the missing blocks did instead of learning the language again — and what comes out is an ordinary model of its family with fewer blocks, which the standard converter turns into a container this library reads.

The two methods trade the same thing in opposite directions. Pruning starts from the teacher's weights and cannot change the architecture; logit distillation starts from nothing and can change everything. At an equal budget the first wins, which is exactly why the second exists.

### Retrieval, for answering from documents

An embedding model is the decoder already covered here **with its last step removed**: instead of projecting the final hidden state onto the vocabulary, it returns that state. Containers say so themselves — one such model declares itself a `qwen3`, carries the usual blocks and final normalization, and has no output projection at all — so the same loader reads them with no special case.

From there the library indexes a directory of text and answers a question with the passage that best matches it. The parts that decide whether this works are the unglamorous ones: cutting passages on sentence boundaries and starting them on a word, encoding questions and documents differently because retrieval models are trained on pairs of the two, and shortlisting by angular sketch once an index outgrows an exact scan. An index also records **what the model that built it makes of a fixed sentence**, because comparing tokenizers is not enough: an embedding model built from a decoder inherits its vocabulary, and an index searched with the wrong one would return a plausible ranking of the wrong thing.

This closes a loop a generative model cannot close alone — answering from documents it never saw during training, with the passage shown rather than paraphrased.

### Python utilities, used as witnesses

A small set of Python tools lives beside the examples, and their role is verification rather than production:

- **reference evaluation through `llama-cpp-python`**, so that the C++ import can be checked against the implementation everyone else uses;
- **reference vision encoding**, for the same reason on the image side;
- **reference loss measurement** against Hugging Face weights, which is how a training pipeline is proven to measure what it claims;
- **corpus preparation** for knowledge-alignment and task-alignment datasets;
- **corpus collection for a tokenizer**, drawing on encyclopaedic prose, news archives, filtered and multilingual web text, and instruction data — the last of these being the one people leave out, and the one that cannot be recovered afterwards;
- **benchmarking against published models**, each task run at the few-shot count under which it is normally reported, since a table built without that detail compares nothing.

A pipeline that nothing contradicts is a pipeline nobody has verified.

### Current project emphasis

Today, the project is especially strong as a platform for:

- importing, inspecting and running open-weight decoder models in C++
- converting them into trainable Dlib networks
- specializing them by supervised fine-tuning and low-rank adaptation
- raising smaller models by distillation, across architecture boundaries
- shrinking published models by depth pruning, into containers the ecosystem still reads
- indexing and searching a document collection with an embedding model
- text, image, and text-with-image workflows in a single codebase

---

---

## Repository structure

The repository is easiest to understand if you read it as three complementary zones.

### 1. Core implementation: [`dlib/`](dlib)
This is where the reusable library-side components live.

### 2. Progressive workflows: [`examples/`](examples)
This directory contains the practical entry points: training programs, inference pipelines, and specialized demonstrations.

### 3. Reusable artifacts: [`models/`](models)
This directory acts as the checkpoint and artifact layer.

The examples appear twice on purpose. [`examples/`](examples) is what a visitor reads: this project's files and the guide that explains them, with none of the upstream library's own examples in the way. `dlib/examples/` is what CMake builds: upstream's tree with this project's files added to it. Nothing connects the two, so `sync_examples.sh` at the repository root reports any drift between them and copies one over the other on request — the divergence being silent by construction, since both copies compile and both look complete.

---

## How to read the repository

A simple strategy is:

- start with the **main README** if you want the global picture
- move to [`examples/`](examples) if you want the workflow details
- move to [`models/`](models) if you want reusable trained artifacts

A practical rule of thumb is:

- if you want to **understand the architecture**, start here
- if you want to **run or study a workflow**, go to `examples/`
- if you want to **reuse a trained artifact**, go to `models/`

---

## Future directions

Several of the directions listed in earlier revisions of this page — image-oriented workflows and multimodal modeling — have since been implemented. The horizon has moved accordingly.

### Completing QLoRA
Host-resident parameters and per-layer materialization already exist. What remains is keeping the frozen base **quantized** in memory and dequantizing it per layer inside the forward and backward passes. The dequantization routines are already present for the import path; the work is to call them on the fly, and to split the parameter block so that the adapters stay in floating point while the base does not.

### Preference optimization
**DPO** replaces the heavy RLHF loop with a comparison between a preferred and a rejected answer. It normally requires a second frozen reference model, which doubles the memory. With LoRA that reference is free: the same network with its adapters neutralized. That makes preference optimization markedly cheaper here than in a general setting.

### Mixture-of-experts on the static path
The runtime engine already serves MoE containers. Giving the statically compiled network an expert-packing path would make those models trainable and convertible like any other, and would let a distilled MoE student be served by the same tooling as a dense one.

### Tied embeddings
On a large vocabulary, the embedding table and the output projection dominate a small model: a 40-million-parameter student built on a 152k-token vocabulary spends four fifths of its parameters there. Sharing the two matrices, as several open-weight models do, would change what a small student can be.

### A model of this library's own
Everything above serves one end: a small language model built here rather than borrowed, on a topology sized for a vocabulary this project chooses, with a mixture of experts that pays its capacity once in memory and adaptive computation that spends depth where the text is difficult. The tooling is in place — corpus collection, tokenizer training and measurement, pre-training, instruction tuning, benchmarking. What separates that from a finished model is compute, and the honest figure is billions of tokens rather than millions.

### Broader reusable ecosystem
Additional future progress can also happen around:

- richer loaders and preprocessing paths
- broader checkpoint availability
- stronger encoder-decoder coverage, which remains essential for translation and conditional generation[^vaswani2017][^google-transformers]
- clearer bridges between core layers, examples, and reusable artifacts

---

## Citing this project

If you wish to reference this project in a publication, report, article, technical note, or any other document, you can cite the repository in a software-oriented form such as the BibTeX entry below.

```bibtex
@misc{cydral_dlib_transformer_extensions,
  author       = {Cydral Technology, Aldric PIERRAIN},
  title        = {Dlib Transformer Extensions},
  howpublished = {GitHub repository},
  url          = {https://github.com/Cydral/Dlib-Transformer-extensions},
  note         = {Open-source repository for transformer architectures, language-model utilities, examples, and reusable artifacts for Dlib}
}
```

If you are discussing the broader methodological foundations rather than the repository itself, it is also appropriate to cite the key academic references listed below.

---

## Final note

This main page is intended to remain the **stable architectural front door** of the repository:

- broad enough to explain the project at a glance
- strong enough to explain why the project matters
- concise enough not to duplicate the detailed example and model pages
- structured enough to support future growth of the repository

---

## Academic references

[^vaswani2017]: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. **Attention Is All You Need**. *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.

[^google-transformers]: Google Developers. **LLMs: What's a large language model? / What's a Transformer?** Machine Learning Crash Course.

[^xu2023]: Peng Xu, Xiatian Zhu, and David A. Clifton. **Multimodal Learning with Transformers: A Survey**. *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*, 2023.

[^islam2023]: Saidul Islam, Hanae Elmekki, Ahmed Elsebai, Jamal Bentahar, Najat Drawel, Gaith Rjoub, and Witold Pedrycz. **A Comprehensive Survey on Applications of Transformers for Deep Learning Tasks**. arXiv, 2023.
