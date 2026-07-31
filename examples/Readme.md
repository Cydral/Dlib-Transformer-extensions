# Examples

> [!IMPORTANT]
> This directory provides a **progressive, pragmatic, and implementation-oriented** tour of the most recent **Transformer-related capabilities** added around **Dlib** in this project.
>
> The objective is not only to provide runnable programs, but also to document the **design patterns**, **training utilities**, **specialized losses**, **tokenization strategies**, and **inference helpers** that make advanced Transformer systems easier to build in modern **C++**.

> [!NOTE]
> The examples are intentionally ordered from the most accessible foundations to more specialized and research-oriented pipelines. Taken together, they form a coherent path from **minimal next-token prediction** to **compact modern Transformer training**, **runtime-selectable model topologies**, **instruction tuning**, **large-text memorization**, **structured reasoning**, and **predictive compression**.
>
> A second family of examples then opens the door outwards: **importing open-weight GGUF models**, **serving them through a built-in chat interface**, **converting them into trainable Dlib networks**, **specializing them by SFT and LoRA**, **training Vision Transformers**, and **raising a new model by distillation from a larger teacher**.

---

## On this page

- [What this directory demonstrates](#what-this-directory-demonstrates)
- [Suggested reading order](#suggested-reading-order)
- [Pedagogical backbone](#pedagogical-backbone)
- [Example-by-example guide](#example-by-example-guide)
  - [`slm_basic_train_ex.cpp`](#slm_basic_train_excpp)
  - [`slm_advanced_train_ex.cpp`](#slm_advanced_train_excpp)
  - [`slm_advanced_gqa_train_ex.cpp`](#slm_advanced_gqa_train_excpp)
  - [`slm_transformer_configs_ex.cpp`](#slm_transformer_configs_excpp)
  - [`slm_enwiki_train_ex.cpp`](#slm_enwiki_train_excpp)
  - [`slm_hrm_arc_agi_ex.cpp`](#slm_hrm_arc_agi_excpp)
  - [`slm_predictive_compressor_ex.cpp`](#slm_predictive_compressor_excpp)
  - [`slm_advanced_gqa_kvc_train_ex.cpp`](#slm_advanced_gqa_kvc_train_excpp)
  - [`slm_cygnus_foundation_ex.cpp`](#slm_cygnus_foundation_excpp)
  - [`slm_cygnus_instruct_ex.cpp`](#slm_cygnus_instruct_excpp)
  - [`slm_lora_adapter_check_ex.cpp`](#slm_lora_adapter_check_excpp)
  - [`slm_tok_predictive_compressor_ex.cpp`](#slm_tok_predictive_compressor_excpp)
  - [`slm_gguf_runtime_ex.cpp`](#slm_gguf_runtime_excpp)
  - [`slm_gguf_import_ex.cpp`](#slm_gguf_import_excpp)
  - [`slm_lora_finetune_ex.cpp`](#slm_lora_finetune_excpp)
  - [`slm_vision_tower_ex.cpp`](#slm_vision_tower_excpp)
  - [`slm_vit_classify_ex.cpp` and `slm_vit_ssl_ex.cpp`](#slm_vit_classify_excpp-and-slm_vit_ssl_excpp)
  - [`slm_distill_ex.cpp`](#slm_distill_excpp)
  - [`slm_tools/` Python witnesses](#slm_tools-python-witnesses)
  - [`slm_data.h`](#slm_datah-shared-data-layer)
- [Cross-cutting concepts worth noticing](#cross-cutting-concepts-worth-noticing)
- [Which example should I start with?](#which-example-should-i-start-with)
- [Final perspective](#final-perspective)

---

## What this directory demonstrates

Across the full example suite, the repository now covers:

- **character-level and subword-level language modeling**
- **BPE tokenization** and tokenizer persistence
- **compact Transformer configurations** suitable for small and medium experiments
- **Grouped Query Attention (GQA)** and more advanced attention-efficient designs
- **specialized sequence losses** that avoid awkward tensor flattening stages
- **dataset construction utilities** for next-token prediction
- **training helpers** such as shuffling, augmentation, checkpointing, optimizer propagation, and padding-aware execution
- **multi-stage fine-tuning** for chatbot-style instruction following
- **runtime selection of pre-configured architectures** such as **MoE** and **HRM**
- **structured autoregressive generation** for text, grid reasoning, and byte-level prediction
- **non-standard Transformer applications**, including ARC-like symbolic reasoning and predictive compression
- **GGUF interoperability**: architecture detection, on-the-fly dequantization of legacy and k-quant formats, and a compatibility report that refuses rather than approximates
- **two execution paths held to bit-level agreement**: a shape-dynamic runtime engine and a statically compiled network generated from the model's own geometry
- **a built-in web chat interface** and an OpenAI-compatible endpoint, with streaming, sampling controls and inline images when the model carries a vision tower
- **Vision Transformers** as standalone backbones, for supervised classification and self-supervised pretraining
- **multimodal fusion** where the vision tower lives inside the network and receives a gradient
- **supervised fine-tuning and low-rank adaptation** (LoRA, DoRA) over an imported model
- **knowledge distillation** from a teacher container into a student of a freely chosen architecture, dense or mixture-of-experts

> [!TIP]
> Read this directory not as a loose collection of demos, but as a **progressive design reference** for building Transformer-based applications with Dlib.

---

## Suggested reading order

1. [`slm_basic_train_ex.cpp`](#slm_basic_train_excpp)  
   Minimal entry point: **character-level next-token prediction**.
2. [`slm_advanced_train_ex.cpp`](#slm_advanced_train_excpp)  
   Compact practical Transformer with **BPE** and **sequence-native loss**.
3. [`slm_advanced_gqa_train_ex.cpp`](#slm_advanced_gqa_train_excpp)  
   Efficient attention with **GQA** and adaptive FFN computation.
4. [`slm_transformer_configs_ex.cpp`](#slm_transformer_configs_excpp)  
   Unified pipeline over pre-configured architectures (**MoE / HRM**).
5. [`slm_cygnus_foundation_ex.cpp`](#slm_cygnus_foundation_excpp) and [`slm_cygnus_instruct_ex.cpp`](#slm_cygnus_instruct_excpp)  
   A full **foundation then instruct** pipeline for a compact GQA + MoE model family.
6. [`slm_enwiki_train_ex.cpp`](#slm_enwiki_train_excpp)  
   Longer-corpus training and context-window management.
7. [`slm_hrm_arc_agi_ex.cpp`](#slm_hrm_arc_agi_excpp)  
   Structured reasoning over ARC-style grid outputs.
8. [`slm_predictive_compressor_ex.cpp`](#slm_predictive_compressor_excpp)  
   Transformer as a **byte-level predictive model** for compression.

Then the interoperability and specialization track, which can be read independently:

9. `slm_gguf_runtime_ex.cpp`  
   Run **any supported GGUF container** through the shape-dynamic engine: chat, serve, describe an image.
10. `slm_gguf_import_ex.cpp`  
   Turn a container into a **statically compiled Dlib network**, then convert it into a native archive.
11. `slm_lora_finetune_ex.cpp`  
   **SFT and LoRA** over an imported model, in two chainable stages.
12. `slm_vision_tower_ex.cpp`  
   Validate a **Vision Transformer** against the reference encoder, weight by weight.
13. `slm_vit_classify_ex.cpp` and `slm_vit_ssl_ex.cpp`  
   Train a Vision Transformer **from scratch**, with labels and without.
14. `slm_distill_ex.cpp`  
   Raise a new model by **distillation**, in three steps.

A shared support layer, [`slm_data.h`](#slm_datah-shared-data-layer), provides embedded datasets and utilities used throughout the examples, and [`slm_tools/`](slm_tools) holds the Python witnesses used to verify the C++ paths against reference implementations.

---

## Pedagogical backbone

At a high level, almost all examples instantiate the same autoregressive idea:

$$
P(x_1,\ldots,x_T)=\prod_{t=1}^{T} P(x_t \mid x_1,\ldots,x_{t-1})
$$

and optimize a next-token objective of the form:

$$
\mathcal{L} = -\sum_t \log P_\theta(x_t \mid x_1,\ldots,x_{t-1})
$$

The key differences between examples come from:

- **how text is tokenized**: character-level, BPE, byte-level, or structured symbolic tokens
- **which Transformer block is used**: canonical, GQA, MoE, or HRM
- **how training data is built**: plain sliding window, padded prompt/answer formatting, grid contexts, or file bytes
- **how inference is constrained**: plain argmax, stochastic decoding, early-stop structural validation, or compression bitstream logic

For attention itself, the usual scaled dot-product mechanism remains central:

```math
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
```

Several examples show how this familiar building block can be packaged in Dlib with increasingly high-level abstractions.

---

## Example-by-example guide

<a id="slm_basic_train_excpp"></a>
## `slm_basic_train_ex.cpp`

### Purpose
A **minimal educational Transformer** for **character-level language modeling**. This is the best entry point if you want to understand the mechanics of next-token prediction with the smallest conceptual overhead.

### What the example teaches
- direct conversion of each character into a token ID
- construction of a sliding-window next-token dataset
- training a small Transformer with Dlib's standard training loop
- autoregressive generation with a moving inference context
- the difference between **memorization capacity** and **generalization**

### Main technical choices
- **tokenization**: one character = one token
- **vocabulary**: 257 symbols (256 byte-range characters + 1 padding token)
- **architecture**: 3 Transformer layers, 4 attention heads, 64-dimensional embeddings
- **context length**: 50 tokens
- **dataset source**: Shakespeare extract embedded in `slm_data.h`
- **generation**: deterministic next-token prediction over a rolling context window

### Why this example matters
This example keeps everything intentionally simple:
- there is no BPE training,
- no instruction tuning,
- no sparse experts,
- no runtime architecture dispatch,
- no advanced sampling pipeline.

That simplicity is its strength: it isolates the **core language-model loop** and shows that even a small Transformer can learn local sequential dependencies very effectively on a limited corpus.

### Useful reminders
Character-level models are pedagogically excellent because they avoid hidden preprocessing complexity. The tradeoff is that the model must learn:
- character composition into words,
- punctuation patterns,
- long-range dependencies,
- and formatting structure,
all from the raw sequence itself.

### What to inspect in the code
- `char_based_tokenize()`
- `build_single_token_prediction_dataset()`
- `shuffle_samples_and_labels()`
- `inference_context`

[⬆ Back to top](#on-this-page)

---

<a id="slm_advanced_train_excpp"></a>
## `slm_advanced_train_ex.cpp`

### Purpose
A **compact modern Transformer pipeline** showing how to move from toy character-level modeling to a more realistic **subword-based** workflow while keeping the implementation concise.

### What this example adds compared to the basic one
- **BPE tokenization** learned from corpus data
- a more compact but stronger practical configuration
- **token cache persistence** to avoid repeated preprocessing
- a **byte-accurate reconstruction / verification** workflow
- a more direct sequence modeling setup using a specialized loss adapted to Transformer logits

### Main technical choices
- **tokenization**: BPE
- **vocabulary size**: 2000
- **architecture**: 4 layers, 6 heads, 228-dimensional embeddings
- **context length**: 100 tokens
- **model objective**: sequence prediction with explicit reconstruction verification

### Pedagogical interest
This example is especially valuable because it shows the transition from a didactic Transformer to a **practical compact language model**:
- BPE reduces sequence length compared to pure character-level modeling,
- vocabulary growth remains controlled,
- unknown strings can still be decomposed into subword pieces,
- training remains manageable in memory and storage.

### Why BPE changes the game
Byte Pair Encoding sits at an interesting midpoint:
- **characters only** are simple but long and inefficient,
- **whole words** explode the vocabulary,
- **subwords** provide a strong compromise.

In practice, frequent character patterns become reusable units, which improves both compression of the input sequence and statistical reuse across related words.

### Why the specialized loss is interesting
In modern sequence models, it is often preferable to keep the tensor layout naturally aligned with sequence positions rather than flattening everything through an extra awkward reshaping step. This example highlights that design philosophy and makes the training graph easier to read and maintain.

### What to inspect in the code
- tokenizer training / loading logic
- token serialization helpers: `save_tokens_to_file`, `load_tokens_from_file`
- `verify_match()`
- compact `canonical_transformer_config` usage

> [!TIP]
> This example is the practical reference point for users who want a **small but serious baseline** before exploring more specialized variants.

[⬆ Back to top](#on-this-page)

---

<a id="slm_advanced_gqa_train_excpp"></a>
## `slm_advanced_gqa_train_ex.cpp`

### Purpose
This example extends the compact training pipeline with **Grouped Query Attention (GQA)** and an **Adaptive Computation Time-like FFN mechanism**, illustrating how more advanced Transformer internals can be exposed without making the surrounding training code significantly more complex.

### What the example demonstrates
- a GQA-based Transformer configuration
- separation between the number of **query heads** and the number of **key/value heads**
- reduced K/V projection cost while preserving a rich multi-query attention structure
- adaptive FFN computation with a bounded number of internal steps
- the same practical workflow as the advanced compact example: tokenization, training, persistence, generation, verification

### Main technical choices
- **tokenization**: BPE
- **vocabulary size**: 2000
- **architecture**: 4 layers
- **attention layout**: 6 query heads, 2 key/value heads
- **embedding dimension**: 228
- **context length**: 100
- **adaptive FFN cap**: 4 internal steps

### Why GQA is useful
In standard multi-head attention, each head often maintains its own Q/K/V projections. GQA relaxes that symmetry:
- multiple query heads can **share fewer K/V heads**,
- memory and compute for K/V projection can be reduced,
- inference becomes more efficient, especially when scaling context and generation.

A simple way to read the design is:
- queries remain finely split,
- keys/values are shared more aggressively.

This often preserves quality better than naively shrinking the model everywhere.

### Why adaptive FFN computation matters
Not all positions need the same depth of processing. An adaptive FFN design allows the model to spend more computation where needed, up to a bounded maximum. Conceptually, this is close to the broader literature on **conditional / adaptive computation**, where the network learns when additional internal refinement is useful.

### What to inspect in the code
- the GQA configuration type definition
- the explicit distinction between total heads and K/V heads
- the comments documenting the compute savings rationale
- the preservation of the exact same end-user workflow despite more advanced internals

[⬆ Back to top](#on-this-page)

---

<a id="slm_transformer_configs_excpp"></a>
## `slm_transformer_configs_ex.cpp`

### Purpose
A **unified high-level training / generation pipeline** for **pre-configured Transformer architectures** provided by Dlib. This is one of the most instructive examples in the repository because it shows how to industrialize experimentation without duplicating the whole application pipeline.

### What makes this example particularly important
This file is less about “one more model” and more about **software architecture for model experimentation**. It demonstrates how the same surrounding pipeline can drive several Transformer families selected at runtime.

### Supported architectures
- `--arch moe` : **Grouped Query Attention + Mixture-of-Experts FFN**
- `--arch hrm` : **Hierarchical Recurrent Model** for multi-scale sequence processing

### Main features of the pipeline
- runtime architecture selection with compile-time instantiation per configuration
- BPE tokenizer training / reuse
- support for **internal datasets** and **external text files / directories**
- recursive file collection for external corpora
- delimiter normalization and segment parsing
- token persistence to disk
- sliding-window dataset construction
- **dataset augmentation** with controlled noise injection
- checkpoint support
- optimizer parameter propagation to nested sub-networks
- padding-aware training and inference
- prompt/reference split validation during generation
- MoE parameter and expert-usage reporting when relevant

### Why this example is pedagogically strong
It teaches an essential point often missed in toy projects:

> building a modern model is not just about the block definition; it is also about the **training pipeline**, the **data path**, the **evaluation path**, the **checkpoint strategy**, and the **ability to switch architectures without rewriting the application**.

### Focus on the MoE branch
For `moe`, the file additionally exposes:
- per-expert parameter accounting
- distinction between training-time total parameters and inference-time active parameters
- expert usage statistics
- a simple balance diagnostic through usage variance / coefficient of variation

This is extremely useful for understanding one of the main MoE failure modes: **expert collapse**, where only a few experts dominate routing.

### Focus on the HRM branch
For `hrm`, the interest is different: the example illustrates how a sequence model can be configured to capture information at **multiple temporal scales**, which is particularly relevant when local token-level dependencies and broader context organization must coexist.

### Useful reminders
Mixture-of-Experts models aim to increase capacity without activating the whole network for every token. At a high level:

```math
\mathrm{FFN}(x) \approx \sum_{e \in \mathcal{S}(x)} g_e(x)\,E_e(x)
```

where only a small selected set of experts $\mathcal{S}(x)$ is active for a given input.

### What to inspect in the code
- `run_pipeline<TRANSFORMER_CONFIG>()`
- `load_external_data()` and `parse_delimited_segments()`
- `augment_training_dataset()` and `shuffle_training_dataset()`
- `network_context::set_optimizer_params(...)`
- `try_print_moe_info(...)`
- the prompt/reference split used in generation-time validation

> [!NOTE]
> This example is one of the best templates if you plan to build your own high-level executable on top of the library.

[⬆ Back to top](#on-this-page)

---

<a id="slm_enwiki_train_excpp"></a>
## `slm_enwiki_train_ex.cpp`

### Purpose
A text training / generation example designed for a **larger and more realistic corpus**, with utilities that make longer-text workflows easier to manage.

### What this example adds
- explicit file-size handling
- partial reading of a large corpus (`max_bytes` logic)
- token cache persistence bound to the input file
- a dedicated `context_manager` abstraction
- prompt-length and context-length management for long-sequence generation
- explicit byte-for-byte verification support

### Why it is useful
A lot of “minimal language model” code works only because the corpus is tiny and fully embedded. This example bridges the gap toward workflows where:
- the corpus is stored externally,
- reading the whole file may not always be desirable,
- prompt sizing and context truncation become first-class concerns.

### Pedagogical contribution
The `context_manager` abstraction is especially helpful because it makes explicit something that is often left implicit in simpler examples: **the model can only see a bounded context**, so the application must manage what remains visible and what falls out of the attention window.

### What to inspect in the code
- `get_file_size()` and `read_enwiki()`
- generation of token-cache filenames from the corpus path
- `context_manager`
- exact verification utilities for reconstruction analysis

[⬆ Back to top](#on-this-page)

---

<a id="slm_hrm_arc_agi_excpp"></a>
## `slm_hrm_arc_agi_ex.cpp`

### Purpose
A specialized example showing how Transformer-like machinery can be applied to **structured reasoning tasks** inspired by **ARC-style grid transformations**.

### What makes this example stand out
This is not ordinary text generation. The model autoregressively predicts a **structured output grid**, while the program actively validates the generated structure during decoding.

### Main ideas demonstrated
- conversion of ARC-style input context into tokens
- bounded grid constraints (rows, columns, output length)
- generation-state tracking during decoding
- early stopping when invalid patterns are detected
- row-consistency monitoring
- explicit failure handling when the generated output becomes structurally invalid

### Why this is interesting pedagogically
This file highlights a key strength of autoregressive modeling:

> a Transformer can be used on much more than plain prose, provided the task can be encoded as a sequence and the decoding process is constrained appropriately.

The code also shows that for structured outputs, **post-token validation** is often just as important as the neural model itself.

### Useful reminders
Many symbolic reasoning tasks can be recast as sequence generation, but raw generation alone is rarely sufficient. Constraints, termination conditions, and structural validation frequently play a decisive role.

### What to inspect in the code
- `generation_state`
- `generate_output_for_test_pair_with_info(...)`
- context-size computation from ARC input/task pairs
- the stop criteria based on end-of-output token, invalid row structure, and output-length limits

[⬆ Back to top](#on-this-page)

---

<a id="slm_predictive_compressor_excpp"></a>
## `slm_predictive_compressor_ex.cpp`

### Purpose
A particularly original example using a Transformer as a **byte-level predictive model** inside a compression / decompression workflow.

### Why this example is remarkable
It shows that the library is not restricted to text generation or chatbot scenarios. A next-token model over bytes can become a **probabilistic predictor** for a compression scheme.

### What the example demonstrates
- byte-level vocabulary (`0..255`) without extra text tokenization
- compact GQA-based network for prediction
- bitstream input/output helpers
- separation between training, compression, and decompression modes
- file-level integrity checking via **CRC32**
- custom container format with a magic number

### Main technical choices
- **window size**: fixed prediction window of 10 bytes
- **vocabulary**: 256 exact byte values
- **architecture**: 2 layers, 4 heads, 16-dimensional embeddings
- **network family**: GQA-based Transformer configuration

### Pedagogical interest
This example is a very clean illustration of a deep idea:

> compression quality depends on prediction quality.

If the model assigns high probability to the next byte, an entropy-coding stage can encode that byte more efficiently. In that sense, predictive compression is a direct operational use of sequence modeling.

### What to inspect in the code
- `out_bit_stream` / `in_bit_stream`
- `compute_crc32(...)`
- `train_predictor_model(...)`
- `compress_file(...)`
- `decompress_file(...)`
- the compact GQA configuration used uniformly for training and inference

[⬆ Back to top](#on-this-page)

---

## Interoperability and specialization track

The examples above build models from nothing. The ones below start from **models that already exist** and take them somewhere: run them, convert them, specialize them, or use them to raise smaller ones.

---

## `slm_gguf_runtime_ex.cpp`

### Purpose
Run **any supported GGUF container** without recompiling anything.

### What the example teaches
The engine reads the container's metadata, detects the architecture, allocates its own structures at run time, and dequantizes weights as it loads them. Nothing about the model is a compile-time constant, which is what makes this the right tool for **exploring** a model you have never seen.

### Main technical choices
- **Architecture detection** across the llama, mistral, mixtral, gemma, gemma2, qwen2, qwen3 and granitemoe families.
- **On-the-fly dequantization** of the legacy block formats (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0), the k-quant range (Q2_K to Q6_K) and IQ4_NL. Grid-based i-quants are refused explicitly.
- **Mixture-of-experts routing** — softmax over router logits, top-k selection, renormalization of the selected weights — which the static path does not implement.
- A **KV cache** with a pinned prefix, so a system prompt survives eviction.

### Why this example matters
It is the **reference behaviour** the rest of the project is measured against. When the statically compiled network and this engine disagree, one of them is wrong, and the disagreement is a bug rather than a matter of interpretation.

### What to inspect in the code
- `--probe-logits`, which prints the five most probable continuations and a per-position argmax: the cheapest possible check on a set of weights
- the serving path, shared with the compiled network through `chat_service`

[⬆ Back to top](#on-this-page)

---

## `slm_gguf_import_ex.cpp`

### Purpose
Turn a GGUF container into a **native Dlib network** whose weights are ordinary trainable parameters.

### What the example teaches
The program runs in **two phases**, and the reason is instructive: a Dlib network is a C++ type, so a model's geometry has to be known at compile time. Phase one reads the container and **generates a header** describing it. After a rebuild, phase two repacks the weights into that exact network and serializes the result.

### Main technical choices
- A **compatibility report** before any import, separating notes from blockers.
- **Weight repacking** rather than reinterpretation: matrices are transposed where the conventions differ, the attention prescale is folded into the stored query weights, and the result is checked against the reference implementation.
- A **self-contained archive**: decoder, vision tower when present, tokenizer, and the pixel normalization the tower was trained with. Nothing external is needed afterwards.
- **Chat, serve, describe, and probe** commands over either a container or an archive.

### Why this example matters
This is where a downloaded artifact stops being a black box. Once the weights sit in a Dlib network, the whole library applies: optimizers, visitors, serialization, fine-tuning.

### Useful reminders
A program compiled for one geometry reads archives of that geometry only. Serving a different model — a distilled student, for instance — means rebuilding against **its** header. That is a property of static compilation, not a limitation of the format.

[⬆ Back to top](#on-this-page)

---

## `slm_lora_finetune_ex.cpp`

### Purpose
Specialize an imported model by **supervised fine-tuning** and **low-rank adaptation**.

### What the example teaches
Two chainable stages: **knowledge alignment** over a plain corpus, and **task alignment** over question-and-answer records. The second is the standard SFT objective — the prompt is masked, only the answer is scored — with the chat template rendered by the model's own formatter.

### Main technical choices
- **LoRA and DoRA** adapters on the query, value and feed-forward projections, with a configurable rank, scale and width ceiling that keeps the vocabulary head out.
- The base weights are **frozen**, and a vision tower, when present, is frozen and reported as such.
- **Padding is hidden from attention**, not merely ignored by the loss: a window is padded to a fixed length, and without a mask every real position would attend to filler.
- A **loss measured before the first step**, so that a run improving a pretrained model can be told from one that first breaks it and then rebuilds it.
- Adapters are **merged into the weights** at the end, producing an archive indistinguishable from an imported one.

### Why this example matters
It is the difference between consuming a model and owning one. It also carries a hard-won instrument: without the loss measured before training, a falling curve says nothing about whether the model is getting better or merely recovering.

[⬆ Back to top](#on-this-page)

---

## `slm_vision_tower_ex.cpp`

### Purpose
Validate a **Vision Transformer** built as a Dlib network against the shape-dynamic encoder, weight by weight.

### What the example teaches
Two implementations of the same tower, fed the same image, must agree to the bit. Anything less means one of them has a convention wrong — a transposed matrix, a normalization on the wrong axis, a position table off by one.

### Main technical choices
- A **patch-per-sample layout**, which is what makes `layer_norm` and `fc` exact on a patch sequence without rewriting them.
- **Bidirectional** attention: an image has no reading order, and a causal mask would halve the receptive field of every patch.
- A **pixel shuffle** that folds neighbouring patches together before the projector, shortening the visual sequence a decoder has to read.

[⬆ Back to top](#on-this-page)

---

## `slm_vit_classify_ex.cpp` and `slm_vit_ssl_ex.cpp`

### Purpose
Train a Vision Transformer **from scratch**, with labels and without.

### What these examples teach
The tower built for multimodal work is a backbone in its own right. Naming an image input instead of a tensor input turns it into a standalone network, and the whole model then fits in two lines:

```cpp
using tower = dlib::vision_transformer_config<
    32, 4, 128, 6, 4, 512, 1, 128, dlib::input_rgb_image_sized<32>>;
using net_type = dlib::loss_multiclass_log<tower::classifier<10>>;
```

**`slm_vit_classify_ex`** trains that on CIFAR-10 and explains the three settings that are easy to get wrong on small images: patches of 4 rather than 16, the pixel shuffle disabled, and a projector as wide as the tower.

**`slm_vit_ssl_ex`** trains the same tower **without labels**, by Barlow Twins: two augmented views of one image go through the tower, a projector maps both to a wider space, and the loss drives the cross-correlation matrix towards the identity. Agreement teaches invariance, decorrelation prevents collapse. The representation is then measured by nearest neighbours — the labels appear only in the measurement.

### Why they matter
A tower pretrained this way can be handed to a fusion layer as an **already-trained encoder**, which is the first stage of the usual multimodal recipe, learned without a single caption.

[⬆ Back to top](#on-this-page)

---

## `slm_distill_ex.cpp`

### Purpose
Raise a **new model** by distillation: a student of a shape you choose, trained on what a larger teacher predicts over a corpus of your own.

### What the example teaches
A hard label says the next token is X. A teacher's distribution says X, but Y was nearly as good, Z was plausible, and the remaining fifty thousand were not. **That ordering over the vocabulary is not present in the corpus**, and no student could derive it from the text alone — which is why a student learns from a teacher far faster than from the same corpus on its own.

The transfer is bounded by the corpus, and the example says so plainly: the teacher is only ever asked what it would predict at the positions the corpus contains. What survives is the intersection of what the teacher knows and what the corpus probes.

### Three steps, and why there are three
1. **Emit the student's header.** Its geometry is yours; its vocabulary is the teacher's, which is not a choice.
2. **Record the teacher.** Its top-k logits at every position go into a trace file. This pass is the expensive one, and it is paid **once per corpus rather than once per student**: several students of different shapes can be raised on the same recording.
3. **Train the student.** Neither the teacher nor the corpus is needed again.

### Main technical choices
- The loss mixes both terms, $\mathcal{L} = \alpha T^2 \cdot \mathrm{soft} + (1-\alpha) \cdot \mathrm{hard}$. The temperature flattens both distributions so that what the teacher thinks of unlikely tokens becomes visible; the $T^2$ keeps the two terms comparable as $T$ varies.
- **Raw logits** are stored rather than probabilities, so a temperature can still be chosen at training time.
- A **tokenizer fingerprint** travels in the trace header. A student built on another tokenizer would read the recorded ids as perfectly valid integers standing for other words, and would train on nonsense without a single error being raised.
- **Several recordings are interleaved** within an epoch rather than chained, because a student raised on prose and then on conversation forgets the prose while it learns the conversation.
- The student may be **dense or a mixture of experts**, independently of the teacher: distilling on logits constrains the vocabulary and nothing else.

### Why this example matters
It is the model factory. The architecture becomes a parameter, the teacher and the corpus become inputs, and the recording becomes a reusable asset.

[⬆ Back to top](#on-this-page)

---

## `slm_advanced_gqa_kvc_train_ex.cpp`

### Purpose
The same model, the same data and the same command line as `slm_advanced_gqa_train_ex.cpp`, with **one** difference: the fused attention layer carries a **KV cache**.

### What the example teaches
Autoregressive generation recomputes the whole prefix at every step unless the keys and values of past positions are kept. The cache is held **inside the attention layer** rather than managed by the generation loop, which therefore has nothing to do about it: the same code generates with or without a cache.

### Why this example matters
It exists to prove that the cached path is **byte-accurate** against the uncached one. A cache is exactly the kind of optimization that produces almost-right output, and almost-right is indistinguishable from right until someone compares.

[⬆ Back to top](#on-this-page)

---

<a id="slm_cygnus_foundation_excpp"></a>
## `slm_cygnus_foundation_ex.cpp`

### Purpose
The **foundation pre-training** stage of the Cygnus family: compact small language models built on this library's target architecture, **Grouped Query Attention with a Mixture-of-Experts feed-forward sublayer**.

### What the example teaches
Sparse routing means only a fraction of the parameters activate per token, so a model can hold capacity it does not pay for at every step. Combined with GQA, which shares key and value heads across query heads, this is the shape most recent open-weight models converge towards.

### Main technical choices
- **Top-k routing** over independent SwiGLU experts, with the gate trained jointly.
- A **KV cache** in the fused attention layer, for generation that stays fast as the context grows.
- Chunked corpus handling, so that pre-training is not bounded by memory.

[⬆ Back to top](#on-this-page)

---

<a id="slm_cygnus_instruct_excpp"></a>
## `slm_cygnus_instruct_ex.cpp`

### Purpose
The **instruct** stage that follows the foundation model: specialization on conversational pairs, then an interactive chat mode for immediate testing.

### What the example teaches
The two stages of the usual recipe, made explicit. A foundation model completes text; an instruct model recognizes a turn, answers it, and stops. The difference is entirely in the data and the formatting, not in the architecture.

### Useful reminders
The architecture constants **must match the foundation model exactly**. They are compile-time constants on both sides, and a mismatch is a deserialization error rather than a silent degradation — which is the desired behaviour.

[⬆ Back to top](#on-this-page)

---

## `slm_lora_adapter_check_ex.cpp`

### Purpose
Numerical validation of the **low-rank adaptation core**, against a double-precision reference written straight from the definition.

### What the example teaches
`low_rank_adapter` is the one piece of the fine-tuning path whose correctness cannot be read off a training run. A wrong gradient does not crash: it produces a loss curve that goes down slowly and a model that is merely disappointing. This program settles the question **before** the core is wired into the attention layer.

### Why this example matters
It is the pattern worth copying more than the code. Any component whose failure mode is "slightly worse results" deserves a test that compares it to a reference implementation, because no amount of watching the loss will ever reveal the problem.

[⬆ Back to top](#on-this-page)

---

## `slm_tok_predictive_compressor_ex.cpp`

### Purpose
Lossless compression driven by a Transformer, over **BPE tokens** rather than raw bytes.

### What the example teaches
A language model is a probability distribution over what comes next, and a good distribution is a good compressor. Working at the token level shortens the sequence the model has to predict, which improves both speed and ratio compared to the byte-level variant.

### Main technical choices
The pipeline is **argmax-based** rather than arithmetic-coded, and the header says why: GPU floating-point results are not reproducible between runs, so an arithmetic coder that depends on exact probabilities would decompress to something else on another machine. Comparing predictions instead of encoding them makes the scheme robust to that non-determinism.

[⬆ Back to top](#on-this-page)

---

## `slm_tools/` Python witnesses

### Purpose
Verify the C++ paths against the implementations everyone else uses.

### What lives here
- **`slm_reference_chat.py`** and **`slm_reference_vision.py`** evaluate a GGUF model through `llama-cpp-python`, so that an import can be checked against the reference rather than against itself.
- **`slm_eval_loss.py`** measures the loss of a source model on a prepared dataset through Hugging Face weights, with the same masking, windowing and label smoothing as the C++ run. It is how a training pipeline is proven to measure what it claims.
- **`nist_corpus_prepare.py`** and **`cve_qa_prepare.py`** build knowledge-alignment and task-alignment corpora in the sentinel format the C++ side reads.
- **`slm_lora_finetune.py`** performs the same fine-tuning in PyTorch, as a comparison point for the training curve.

### Why this matters
An import pipeline that nothing contradicts is a pipeline nobody has verified. These tools exist so that a number produced in C++ can be put next to a number produced elsewhere, on the same data, with the same conventions.

[⬆ Back to top](#on-this-page)

---

<a id="slm_datah-shared-data-layer"></a>
## `slm_data.h` shared data layer

### Purpose
This header is not merely a convenience include: it is a **small data access layer** embedded directly into the example suite.

### What it provides
- embedded compressed datasets
- a central `dataset_id` enumeration
- decompression utilities
- accessors returning datasets as:
  - raw text
  - segmented text
  - paired text

### Why it matters
Having a unified data layer makes the examples easier to compare because they share the same access conventions. It also removes boilerplate around dataset loading and keeps the pedagogical focus on the model pipeline itself.

### What to inspect in the code
- `dataset_id`
- `get_dataset_as_text(...)`
- `get_dataset_as_segments(...)`
- `get_dataset_as_pairs(...)`
- the embedded compression / decompression path

[⬆ Back to top](#on-this-page)

---

## Cross-cutting concepts worth noticing

### 1. Padding-aware execution
Several examples explicitly propagate padding information through `network_context`. This is important because many practical sequence pipelines require correct masking or padding semantics when batches contain variable effective lengths.

### 2. Token persistence
Multiple files save tokenized corpora to disk. This is a very practical optimization: tokenization can be expensive, and caching improves iteration speed when experimenting on model definitions or training parameters.

### 3. Exact verification mindset
A recurring pattern in the examples is not merely “generate something plausible”, but also **measure exact reconstruction fidelity**. This makes the examples particularly useful for debugging and benchmarking.

### 4. Architectural abstraction without hiding the mechanics
The repository strikes a useful balance:
- the APIs are high-level enough to simplify experimentation,
- but the examples still expose the important moving parts: tokenization, windows, padding, optimizer configuration, generation loop, and validation.

---

## Which example should I start with?

- Start with **`slm_basic_train_ex.cpp`** if you want the clearest conceptual introduction.
- Move to **`slm_advanced_train_ex.cpp`** if you want a compact practical baseline.
- Use **`slm_advanced_gqa_train_ex.cpp`** if attention efficiency matters.
- Use **`slm_transformer_configs_ex.cpp`** if you want a strong reusable application template.
- Use **`slm_cygnus_foundation_ex.cpp`** then **`slm_cygnus_instruct_ex.cpp`** if your target is a compact model trained end to end, from pre-training to instruction following.
- Use **`slm_enwiki_train_ex.cpp`** if you work with larger external corpora.
- Explore **`slm_hrm_arc_agi_ex.cpp`** if you are interested in structured reasoning beyond plain text.
- Explore **`slm_predictive_compressor_ex.cpp`** or **`slm_tok_predictive_compressor_ex.cpp`** if you want to see Transformers used as generic sequence predictors outside classical NLP.
- Start with **`slm_gguf_runtime_ex.cpp`** if you already have a model and want to run it today.
- Use **`slm_gguf_import_ex.cpp`** then **`slm_lora_finetune_ex.cpp`** if you want to specialize an existing open-weight model.
- Use **`slm_vit_classify_ex.cpp`** or **`slm_vit_ssl_ex.cpp`** if your data is images rather than text.
- Use **`slm_distill_ex.cpp`** if you want to build a smaller model of your own design from a larger one.

---

## Final perspective

Taken together, these examples document a clear evolution of the library toward a **high-level yet explicit Transformer toolkit for Dlib**:

- simple enough to learn from
- modular enough to extend
- rich enough to cover modern practical needs such as compact training, GQA, architecture dispatch, instruction tuning, long-context preparation, structured generation, and non-standard predictive tasks

In that sense, this directory is not just a collection of demos: it is a **progressive design reference** for building Transformer-based applications in **C++ with Dlib**.
