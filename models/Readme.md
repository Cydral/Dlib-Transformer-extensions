# Models

> [!IMPORTANT]
> This directory gathers the **pre-trained checkpoints and related model artifacts** made available for the **Dlib Transformer extensions** project.
>
> These files are intended to complement the programs documented in [`../examples`](../examples), so that users can move more quickly from **example code** to **practical loading, generation, validation, and fine-tuning workflows**.

---

## Purpose of this directory

This part of the repository is intentionally kept **more concise** than the examples documentation.

The goal here is not to restate the full training logic of each program, nor to repeat detailed architectural breakdowns for every checkpoint. Instead, this page is meant to answer a simpler question:

> **Which ready-to-use models are available here, and how should they be approached in relation to the example programs?**

In practice, the models stored in this directory are meant to serve one or more of the following purposes:

- **starting points for inference and experimentation**
- **reference checkpoints** associated with the example training pipelines
- **reproducibility anchors** for users who want to compare their own runs against repository-provided artifacts
- **convenient baselines** before retraining on custom corpora or tasks

---

## How this directory relates to `examples/`

The repository documentation is easier to understand if you view the two directories as complementary:

- [`examples/`](../examples) explains **how the models are trained, configured, and used**
- [`models/`](./) provides the **resulting artifacts** that can be reused directly when available

As a result, this page stays deliberately focused on the **availability and intended role of the checkpoints**, while the implementation details remain documented in the example programs themselves.

---

## Typical model families covered here

Depending on the files currently present in this directory, the available checkpoints may correspond to one or more of the repository's main example families, such as:

- **minimal character-level language models**
- **compact BPE-based Transformer language models**
- **Grouped Query Attention (GQA) variants**
- **runtime-selectable architecture experiments** such as MoE or HRM-based configurations
- **chatbot / instruction-tuned checkpoints**
- **larger-corpus text checkpoints**
- **specialized structured-generation models**
- **predictive byte-level models** for compression-oriented experiments

The important point is not to memorize every architectural detail from this page, but rather to identify the **matching example program** whenever you want to understand:

- the original training setup
- the expected tokenizer or preprocessing path
- the inference mode
- the evaluation or validation strategy

---

## Available checkpoints

Each entry names the program that produced the file, the shape of the network, and what the artifact actually does when loaded. That last column matters more than the others: a checkpoint trained on a small corpus behaves in ways a size and a loss value do not convey.

### `dlib_lm_chars_model.dat`

| | |
|---|---|
| Produced by | [`slm_basic_train_ex`](../examples/slm_basic_train_ex.cpp) |
| Architecture | Fused transformer, 3 layers, 4 heads, width 64, window 50 |
| Vocabulary | 257 entries, one per byte value plus padding. No tokenizer file needed |
| Parameters | 4,798,593 |
| Training | Built-in Shakespeare extract, 14,590 sequences, around ninety epochs |
| Final state | Loss below 0.05, accuracy 0.999 on the training sample |

**What it does.** It reproduces its corpus. At an accuracy of 0.999 on five million parameters against a text of that size, the model has memorized rather than generalized, and a prompt drawn from the extract continues with the lines that actually follow it in the play.

That is the correct outcome for this example and worth stating plainly, because the output looks better than the model is. Someone seeing fluent Shakespeare might conclude the network learned English; it learned this text. The point of the checkpoint is to demonstrate that the machinery works end to end, on a corpus small enough to train in minutes and a vocabulary that needs no preparation.

Load it, generate from it, and treat the fluency as a property of the corpus rather than of the model.

**On the figures above.** The epoch count and the final loss are given loosely on purpose. Reproducing this run gives the same loss to five decimals for the first few epochs, and then diverges: the training curve crosses several sharp rises, up to 0.44 late in the run, and where it happens to stop depends on the order in which floating-point reductions land on a given device. Two runs of the same command reached 0.999109 after 93 epochs and 0.998835 after 95, a difference of four sequences out of 14,590. Quoting an exact epoch count would suggest a precision the process does not have.

```
./slm_basic_train_ex --generate
```

The program reads and writes the checkpoint in the current directory, and in training mode it continues from an existing file rather than starting over. Move or rename it first if a fresh run is what you want.

---

### `dlib_lm_tokens_model.dat` and `dlib_lm_tokenizer.vocab`

| | |
|---|---|
| Produced by | [`slm_advanced_train_ex`](../examples/slm_advanced_train_ex.cpp) |
| Architecture | Canonical transformer, 4 layers, 6 heads, width 228, window 100 |
| Vocabulary | 1,400 BPE entries, trained by the same program on the same corpus |
| Parameters | 2,822,444 |
| Training | Built-in article, 8,066 bytes, 3,076 sequences, 400 epochs |
| Final state | Loss 0.066, byte-for-byte reconstruction exact |

**What it does.** It reproduces its corpus exactly. Running the program with `--generate --verify` regenerates all 8,066 bytes and confirms they are identical to the original.

Two things are worth understanding about this checkpoint. It carries **its own tokenizer**, trained alongside it on the same text, so the pair must be used together: the vocabulary is specific to this corpus and means nothing elsewhere. And its reconstruction is memorization, as with the character-level model, but here demonstrated rather than inferred, since the program verifies it byte by byte.

```
./slm_advanced_train_ex --generate --verify
```

**On the accuracy figure.** The program reports a next-token accuracy that a reader should not confuse with the reconstruction. Generation feeds the model its own output and asks only for the token that follows, so a window whose earlier positions are wrong still reproduces the text as long as the last one is right. The verification pass is the authoritative check; the accuracy figure is an indication.

---

### `dlib_lm_tokens_gqa_model.dat`

| | |
|---|---|
| Produced by | [`slm_advanced_gqa_train_ex`](../examples/slm_advanced_gqa_train_ex.cpp) |
| Architecture | Grouped query attention, 3 layers, 6 query heads over 2 KV heads, width 228, head dim 38, window 200 |
| Vocabulary | Shares `dlib_lm_tokenizer.vocab` with the model above |
| Parameters | 738,755 |
| Training | Same 8,066-byte article, 2,976 sequences, 600 epochs |
| Final state | Loss 0.047, next-token accuracy 100%, reconstruction exact |

**What it does.** The same thing as the model above, on a quarter of the parameters and twice the window. It reproduces the corpus byte for byte, verified.

That comparison is the reason this checkpoint is worth keeping beside the other. **3.8 times fewer parameters, a window twice as long, and the same exact reconstruction.** Some of that comes from having one layer fewer, but the shape of the attention is what makes the trade possible: six query heads share two key-value heads, so the projections that dominate a small attention block are cut without reducing the number of ways the model can attend.

The saving that matters at inference is elsewhere again. A key-value cache holds one entry per KV head per position, so three query heads sharing one KV head divide that cache by three. On a long context, that is usually what limits how much can be served at once.

```
./slm_advanced_gqa_train_ex --generate --verify
```

**Use it with the tokenizer beside it.** This model and the one above share `dlib_lm_tokenizer.vocab`, trained on a composite corpus rather than on the article alone. Pairing either model with a different vocabulary produces confident nonsense and raises nothing.

---

### `dlib_lm_tokens_gqa_kvc_model.dat`

| | |
|---|---|
| Produced by | [`slm_advanced_gqa_kvc_train_ex`](../examples/slm_advanced_gqa_kvc_train_ex.cpp) |
| Architecture | Same as above, with a key-value cache in the attention layer |
| Vocabulary | Shares `dlib_lm_tokenizer.vocab` |
| Parameters | 739,895 |
| Training | Same corpus and settings as the model above |
| Final state | Loss 0.046, next-token accuracy 100%, reconstruction exact |

**What it does.** The same thing again, and that is the point. This checkpoint exists to isolate one variable: the cache changes nothing about what the model learns and everything about how fast it generates.

Trained identically to the model above, it reaches the same loss and reproduces the corpus byte for byte. Generating the same 2,976 tokens takes **9 seconds instead of 14, at 322 tokens per second against 241** — a third faster, and the gap widens with sequence length, since generation without a cache recomputes every previous position at every step while generation with one recomputes nothing.

The parameter count differs by about a thousand, which is not the cache: a cache stores activations rather than weights and adds nothing to learn. The difference comes from the unified attention implementation this variant uses.

```
./slm_advanced_gqa_kvc_train_ex --generate --verify
```

---

### `dlib_lm_enwiki_model.dat`

| | |
|---|---|
| Produced by | [`slm_enwiki_train_ex`](../examples/slm_enwiki_train_ex.cpp) |
| Architecture | 2 layers, 6 heads, width 180, head dim 30, window 100, with a two-expert mixture built from elementary layers |
| Vocabulary | 800 BPE entries, in `enwiki_tokenizer.vocab` |
| Parameters | 15,239,868 |
| Training | 23,373 bytes of encyclopaedic text, 10,292 sequences, 400 epochs |
| Final state | Loss 0.218, next-token accuracy 100%, reconstruction exact |

**What it does.** It reproduces its corpus byte for byte, verified over all 23,373 bytes.

What distinguishes this checkpoint from the others is how its mixture of experts is built. Elsewhere in this project the routing goes through the optimized `moe_` layer, which activates only the experts a token is routed to. Here the two experts are ordinary subnetworks, both traversed on every token, and their outputs combined by a learned router into a weighted sum. Adaptive computation wraps each of them, applying the same expert up to six times depending on how hard the position is.

That construction is the point: it shows a mixture assembled from elementary layers, with nothing hidden inside a fused implementation. It also explains why this model carries fifteen million parameters where the grouped-query checkpoints above hold three quarters of a million. Two experts held whole, each wrapped in up to six adaptive steps, cost what a routed mixture is designed to avoid — and every token pays for both of them, where the `moe_` layer would pay for one.

That is a deliberate trade, not a defect. The example exists to show the construction rather than to be efficient at it.

```
./slm_enwiki_train_ex --generate --verify
```

**On the size.** The width was narrowed from 228 to 180 and the vocabulary from 1,000 to 800, bringing the serialized model from 207 MB to 61 MB. The earlier file had to be split across two archives to be published; this one travels whole. Nothing the example demonstrates depended on the larger size.

---

### `dlib_lm_moe_model.dat`

| | |
|---|---|
| Produced by | [`slm_transformer_configs_ex --arch moe`](../examples/slm_transformer_configs_ex.cpp) |
| Architecture | Grouped query attention, 3 layers, 6 query heads over 2 KV heads, width 228, window 200, with 3 experts routed one at a time |
| Vocabulary | Shares `dlib_lm_tokenizer.vocab` |
| Parameters | 4,495,421 stored, 1,991,525 active |
| Training | 828 segments, 93,085 tokens, 97,529 sequences |

**What it demonstrates.** That routed sparsity works, and what it costs. This checkpoint holds 4.5 million parameters and runs 2.0 million of them per token: the remaining 56% sit in memory and are never touched unless a token is routed to them. That is the whole argument for a mixture of experts, and it is the reason this checkpoint sits beside the dense ones rather than replacing them.

The number that says whether the arrangement actually works is not the loss. It is the balance of the router, which the program reports at the end of training. Three experts under top-1 routing received 0.3296, 0.3346 and 0.3358 of the traffic against an ideal third each, a coefficient of variation of 0.008. **Expert collapse did not occur** — the failure where two experts absorb everything while a third stays idle, which the training loss never reveals because the model converges perfectly well while it happens.

```
./slm_transformer_configs_ex --generate --arch moe
```

**On the training budget.** Reproduction is partial: some segments come back exactly, others diverge. The run would benefit from more epochs, and the internal sizing is very likely not optimal either — this checkpoint is here to show the architecture rather than to be the best model the architecture allows. Compare it with the dense and grouped-query checkpoints above, which share the same corpus and tokenizer: that comparison is what it is for.

---

## Recommended usage workflow

If you are downloading a model from this directory, the most robust workflow is usually:

1. **Identify the corresponding example program** in [`../examples`](../examples).
2. **Use the same preprocessing path** as the one expected by that example (character-level, BPE, byte-level, structured tokens, etc.).
3. **Keep tokenizer or auxiliary files together** with the checkpoint whenever the model depends on them.
4. **Reuse the same inference conventions** as in the associated example (prompt formatting, rolling context, sampling strategy, structural validation, or decompression path).

This is especially important for checkpoints whose behavior depends not only on the neural weights themselves, but also on:

- tokenizer vocabulary and merges
- special tokens or structural markers
- context-window conventions
- padding behavior
- decoding constraints

---

## Practical reading guide

A simple rule of thumb is:

- if you want to **understand the method**, start with [`../examples`](../examples)
- if you want to **reuse a trained artifact**, start here
- if you want to **adapt or fine-tune a checkpoint**, use both together

In other words, the examples explain the **why and how**, while this directory provides the **ready-to-use outputs**.

---

## Compatibility and reproducibility notes

When working with pre-trained artifacts, remember that reproducibility depends on more than a single checkpoint file.

For best results, keep aligned:

- the **corresponding example source file**
- the **expected tokenizer / vocabulary artifacts**, when relevant
- the **same model configuration family**

Two hazards deserve naming, because both fail silently.

**A checkpoint and its tokenizer belong together.** Several programs here train their own vocabulary alongside the model, and a checkpoint paired with a different one produces confident nonsense rather than an error: the token identifiers are all valid, they simply mean something else. Where an entry above names a tokenizer, take both files or neither.

**Caches survive a change of configuration.** Programs that pre-tokenize a corpus write the token stream to disk and reload it on the next run to save time. Changing the vocabulary size does not invalidate that file, so a run can train a new architecture on tokens produced by a vocabulary that no longer exists. Deleting the cached tokens along with the tokenizer is the safe habit whenever a configuration changes.
- the **same prompt or input formatting conventions**

If you plan to publish additional checkpoints later, it is often enough to keep the naming consistent with the associated example so that users can immediately infer the intended workflow.
