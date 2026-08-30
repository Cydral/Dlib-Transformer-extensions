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
| Training | Built-in Shakespeare extract, 14,590 sequences, 93 epochs |
| Final state | Loss 0.047, accuracy 0.999 on the training sample |

**What it does.** It reproduces its corpus. At an accuracy of 0.999 on five million parameters against a text of that size, the model has memorized rather than generalized, and a prompt drawn from the extract continues with the lines that actually follow it in the play.

That is the correct outcome for this example and worth stating plainly, because the output looks better than the model is. Someone seeing fluent Shakespeare might conclude the network learned English; it learned this text. The point of the checkpoint is to demonstrate that the machinery works end to end, on a corpus small enough to train in minutes and a vocabulary that needs no preparation.

Load it, generate from it, and treat the fluency as a property of the corpus rather than of the model.

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
- the **same prompt or input formatting conventions**

If you plan to publish additional checkpoints later, it is often enough to keep the naming consistent with the associated example so that users can immediately infer the intended workflow.
