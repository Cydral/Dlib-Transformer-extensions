# Examples

> [!IMPORTANT]
> This directory provides a **progressive, pragmatic, and implementation-oriented** tour of the most recent **Transformer-related capabilities** added around **Dlib** in this project.
>
> The objective is not only to provide runnable programs, but also to document the **design patterns**, **training utilities**, **specialized losses**, **tokenization strategies**, and **inference helpers** that make advanced Transformer systems easier to build in modern **C++**.

> [!NOTE]
> The examples are intentionally ordered from the most accessible foundations to more specialized and research-oriented pipelines. Taken together, they form a coherent path from **minimal next-token prediction** to **compact modern Transformer training**, **runtime-selectable model topologies**, **instruction tuning**, **large-text memorization**, **structured reasoning**, and even **predictive compression**.

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
  - [`slm_chatbot_ex.cpp`](#slm_chatbot_excpp)
  - [`slm_enwiki_train_ex.cpp`](#slm_enwiki_train_excpp)
  - [`slm_hrm_arc_agi_ex.cpp`](#slm_hrm_arc_agi_excpp)
  - [`slm_predictive_compressor_ex.cpp`](#slm_predictive_compressor_excpp)
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
5. [`slm_chatbot_ex.cpp`](#slm_chatbot_excpp)  
   Two-stage specialization toward conversational generation.
6. [`slm_enwiki_train_ex.cpp`](#slm_enwiki_train_excpp)  
   Longer-corpus training and context-window management.
7. [`slm_hrm_arc_agi_ex.cpp`](#slm_hrm_arc_agi_excpp)  
   Structured reasoning over ARC-style grid outputs.
8. [`slm_predictive_compressor_ex.cpp`](#slm_predictive_compressor_excpp)  
   Transformer as a **byte-level predictive model** for compression.

A shared support layer, [`slm_data.h`](#slm_datah-shared-data-layer), provides embedded datasets and utilities used throughout the examples.

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

<a id="slm_chatbot_excpp"></a>
## `slm_chatbot_ex.cpp`

### Purpose
A full **instruction-tuning / chatbot specialization** example built on top of a pre-trained Transformer backbone. It demonstrates that the project is not limited to raw language modeling: it also supports **domain adaptation** and **interactive prompting**.

### What the example demonstrates
- two-stage workflow: base model reuse + conversational specialization
- formatting of Q/A pairs with explicit structural markers
- partial freezing through **layer-wise learning-rate multipliers**
- fine-tuning on domain-specific Q/A datasets
- interactive prompting mode
- deterministic and stochastic generation modes
- advanced decoding controls

### Main technical ideas

#### 1. Structured conversational formatting
The model is not trained on naked question/answer strings. Instead, data is wrapped with explicit tags such as:

```xml
<question><text>...</text><answer><text>...</text>
```

This teaches the model not only the content, but also the **role structure** of the exchange.

#### 2. Partial freezing for stable fine-tuning
All layers are not updated equally. The example uses **learning-rate multipliers** so that some parts of the network adapt faster than others. This is an important practical pattern when the specialization dataset is much smaller than the original pretraining corpus.

#### 3. Proper sequence-level sampling
The inference path uses a **per-row softmax** and then applies several generation controls:
- temperature
- top-k
- top-p / nucleus sampling
- repetition penalty
- min-p filtering
- deterministic argmax mode when needed

### Why this example matters
This is a concrete answer to a very practical question:

> how do I move from a generic compact Transformer to a usable specialized assistant?

The example shows that the answer is not merely “train more”, but rather:
- structure the supervision,
- preserve part of the pretrained representation,
- adapt learning rates,
- and control decoding carefully at inference time.

### Useful reminders
Fine-tuning is a tradeoff between **specialization** and **catastrophic forgetting**. Partial freezing and differentiated learning rates are classical ways to preserve useful generic knowledge while allowing the model to shift toward a target behavior.

### What to inspect in the code
- `append_special_or_text()`
- Q/A dataset loading and tokenization
- `set_all_learning_rate_multipliers(...)`
- the custom sampling routine combining top-k, top-p, min-p, and repetition penalty
- `inference_context` used as a rolling multi-turn conversation buffer

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
- Use **`slm_chatbot_ex.cpp`** if your target is instruction tuning or interactive Q/A.
- Use **`slm_enwiki_train_ex.cpp`** if you work with larger external corpora.
- Explore **`slm_hrm_arc_agi_ex.cpp`** if you are interested in structured reasoning beyond plain text.
- Explore **`slm_predictive_compressor_ex.cpp`** if you want to see Transformers used as generic sequence predictors outside classical NLP.

---

## Final perspective

Taken together, these examples document a clear evolution of the library toward a **high-level yet explicit Transformer toolkit for Dlib**:

- simple enough to learn from
- modular enough to extend
- rich enough to cover modern practical needs such as compact training, GQA, architecture dispatch, instruction tuning, long-context preparation, structured generation, and non-standard predictive tasks

In that sense, this directory is not just a collection of demos: it is a **progressive design reference** for building Transformer-based applications in **C++ with Dlib**.
