# Articles

Longer-form writing about what this project does and why, one subject at a time.

The [examples guide](../examples/Readme.md) explains what each program does. These articles answer the questions that sit above the code: why a design was chosen, what it costs, where it breaks, and how to reproduce the result. Each is written to stand alone, so a reader arriving from a search engine with one question does not have to read the others first.

Every article is published in two places. The Markdown version here is the reference and travels with the code it describes. A companion version is posted on Medium, which is where most readers will find it.

---

## On this page

- [Published](#published)
- [Planned](#planned)
- [How these articles are written](#how-these-articles-are-written)

---

## Published

| Article | Subject |
|---|---|
| [Building a Small Language Model Stack in C++](01-introduction.md) | What the project is, what it covers today, and why a C++ library for this exists at all |

---

## Planned

The order below is the order in which they are being written, which follows the order in which the underlying work was done rather than any pedagogical ideal.

**A small language model from end to end.** The full chain with nothing skipped: collecting a corpus, training a byte-level BPE tokenizer and measuring what it costs per writing system, pre-training a foundation model, then turning it into one that answers questions rather than continuing text. The article that most people will want, and the longest.

**Deriving a model from another.** Distillation on logits, distillation on sequences, and depth pruning, treated as three answers to one question. Which to use depends on whether the architecture is allowed to change and on how much compute is available, and the article gives the measurements behind that choice rather than a recommendation.

**Running open-weight models from GGUF.** The container format, the two paths this library offers for it, and the reason there are two. A shape-dynamic engine that reads any supported container at runtime, and a code generator that emits a statically typed network which trains and serializes like any other Dlib model.

**Retrieval, and answering from documents.** What an embedding model is, why it is the same decoder with its last step removed, and what actually decides whether retrieval works: chunk boundaries, the asymmetry between a question and a passage, and knowing which model built an index.

**Compression as a use for prediction.** A language model is a probability distribution over what comes next, which is exactly what an arithmetic coder needs. The article covers what this buys, what it costs, and the determinism problem that appears the moment a GPU is involved.

**Adaptive depth, and the traps around it.** Hierarchical reasoning as implemented in one of the examples: how a network learns to spend more computation on the positions that need it, why the halting signal is difficult to train, and the failure modes worth recognising early.

**Transformers for images, and models that read both.** Vision transformers for classification and for self-supervised training, then the vision tower that lets a decoder take an image alongside text.

Other subjects will be added as the work reaches them.

---

## How these articles are written

Three rules, stated here so that a reader can hold the articles to them.

**Numbers come from runs, not from memory.** Every measurement quoted was produced by the code in this repository, and the command that produced it is given so the reader can obtain their own. Where a figure comes from published work, it is cited.

**Failures are reported.** Several of these articles exist because something did not work, and the diagnosis was more instructive than the fix. A guide that only shows the path that worked teaches very little about the terrain.

**Nothing is claimed to be settled that is not.** Where a design is a bet rather than a conclusion, the article says so and names what would change the answer.
