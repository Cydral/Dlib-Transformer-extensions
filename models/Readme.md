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
