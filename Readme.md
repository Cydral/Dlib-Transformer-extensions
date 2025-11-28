# Dlib Transformer extensions

Advanced language modeling capabilities for the Dlib C++ library, enabling modern transformer architectures, efficient training pipelines, and production-ready inference.

**Keywords:** transformer, language-model, LLM, attention-mechanism, deep-learning, neural-networks

---

## Overview

This repository extends [Dlib](http://dlib.net/) with comprehensive support for transformer-based architectures and language modeling. The implementation maintains Dlib's philosophy of providing high-level abstractions while ensuring performance and simplicity. All components are written in standard C++14, ensuring cross-platform compatibility and integration with existing Dlib workflows.

The extensions introduce three major capability areas:

1. **Core architectural components**: attention mechanisms, positional encodings, layer normalization variants, and specialized tensor operations for sequence processing.

2. **Language modeling utilities**: dataset preparation, tokenization interfaces, training data augmentation, and inference context management.

3. **Complete transformer implementations**: both canonical and fused variants, with examples demonstrating training pipelines from basic character-level models to advanced mixture-of-experts architectures.

---

## Technical foundations

### Matrix plane processing paradigm

Traditional Dlib layers operate channel-wise on 4D tensors `(batch, channels, rows, cols)`, processing each channel independently. Language models require plane-wise operations where the `(rows, cols)` dimensions form semantic units representing sequence positions and token embeddings.

The extensions introduce plane-wise processing modes for operations like matrix multiplication, normalization, and attention computation. This architectural shift enables:

- direct representation of sequence data: `(batch, 1, sequence_length, embedding_dim)`
- efficient attention computation over the `(rows, cols)` plane
- natural integration with Dlib's computational graph

Dimension flow in attention illustrates the transformation from input sequences through multi-head processing to output representations, with reshaping operations enabling parallel head computation.

### Attention mechanisms

The scaled dot-product attention follows Vaswani et al. (2017):

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

The implementation provides two architectural variants:

**Canonical transformer**: explicit Q, K, V projections using separate linear layers, followed by reshape operations. This approach offers modularity and clarity, making it suitable for research and fine-grained control.

**Fused transformer**: combined QKV projection with extraction-based separation. This variant optimizes memory usage and computational efficiency, particularly beneficial for deployment scenarios.

Both variants support:
- causal masking for autoregressive generation via triangular mask layer
- multi-head attention with configurable head count
- RMS normalization (Zhang & Sennrich, 2019) as an efficient alternative to LayerNorm
- residual connections with skip connection mechanisms

### Positional encoding strategies

Sequence models require position information since attention is permutation-invariant. The implementation provides multiple encoding strategies:

**Absolute positional encodings**: learnable embeddings added to token embeddings, providing explicit position signals. Implemented with automatic dimension matching to the model's embedding space.

**Rotary positional embeddings (RoPE)**: rotation-based encoding (Su et al., 2021) that naturally captures relative positions through rotation matrices applied to query and key vectors. RoPE offers:
- efficient computation via complex number representation
- natural decay of attention with distance
- extrapolation to longer sequences than seen during training

The implementation supports configurable base frequency and dimension-wise rotation application.

### Normalization techniques

Standard LayerNorm computes mean and variance, which can be computationally expensive. RMS normalization simplifies this:

```
RMSNorm(x) = x / RMS(x) * γ
where RMS(x) = √(1/n ∑ x_i²)
```

RMS normalization provides:
- single pass computation without mean calculation
- learnable scale parameter
- numerical stability through epsilon addition
- plane-wise operation mode for sequence tensors

---

## Extended layer implementations

### Linear transformations

The `linear` layer extends Dlib's fully connected layer with plane-wise processing, enabling matrix multiplication along the last dimension. This transformation maps `(*, *, *, d_in)` to `(*, *, *, d_out)` with weight sharing across spatial dimensions and bias term broadcasting.

### Tensor reshaping

The `reshape_to` layer enables dimension manipulation without data copying, critical for multi-head attention where tensors are reshaped from single-head format to multi-head format and back.

### Dropout variants

Standard dropout applies per-element with fixed probability. The `dropout_rate` layer provides configurable rates at layer definition, enabling per-layer dropout schedules common in transformer architectures.

### Causal masking

The triangular mask layer generates lower-triangular masks for autoregressive attention, ensuring position *i* only attends to positions ≤ *i*, preventing information leakage from future tokens during training.

### Matrix operations

**Matrix multiplication with skip connection**: plane-wise matrix multiplication combining current layer output with skip connection for attention score computation.

**Transpose operation**: permutes the last two dimensions, necessary for computing K^T in the attention mechanism.

### Token embeddings

The token embeddings layer combines embedding lookup with positional encoding, converting discrete token IDs to continuous vectors with position information in a single operation.

### Rotary positional embeddings

Implements rotation-based positional encoding through complex number multiplication, applying frequency-dependent rotations to query and key vectors before attention computation.

---

## Language modeling utilities

### Dataset preparation

The language modeling data utilities provide functions for converting raw token sequences into training samples:

**Single-token prediction**: sliding window approach for autoregressive training, creating input-output pairs where inputs contain context windows and outputs contain the next token. Supports optional left-padding for sequences shorter than the window.

**Sequence-to-sequence prediction**: for translation or transformation tasks, aligning source and target windows with synchronized sliding, supporting encoder-decoder architectures.

### Training data augmentation

**Shuffling**: random permutation while maintaining sample-label correspondence, using Fisher-Yates algorithm for uniform random permutation.

**Noise injection**: replaces random tokens with unknown token to improve robustness. Creates noisy copies of existing samples, capping noise at 30% of non-padding tokens to maintain sample quality. Default augmentation ratio of 20% follows common practices in language model training literature.

### Inference context management

The `inference_context` class manages token history for autoregressive generation with:
- FIFO buffer with configurable capacity
- sliding window extraction
- left-padding when context is not full
- serialization support for checkpoint and resume

Features include dynamic resizing without data loss and context multiplier for extended history beyond the model's window.

### Text similarity metrics

Comprehensive evaluation functions for generated text quality assessment:

**Edit distance**: Levenshtein distance measuring minimum single-token edits required to transform one sequence into another. Normalized score returns values in [0, 1] where 1 indicates identical sequences.

**Token overlap**: order-independent precision, recall, and F1-score treating sequences as bags of tokens. Useful for vocabulary coverage assessment, handling duplicates correctly through multiset matching.

**N-gram overlap**: BLEU-like metric evaluating matching n-grams for structural similarity. Computes precision for n-grams of size 1 through 4, returning average n-gram precision across all n values.

### File type detection

Automatic content classification for preprocessing pipelines, supporting over 30 formats via magic number signatures and entropy analysis. Multi-stage detection process:

1. magic number detection for binary formats (images, documents, compressed archives, executables, audio, video)
2. XML and HTML detection via declaration markers
3. Shannon entropy analysis to distinguish text from compressed content
4. character distribution heuristics with printable character counting

Enables text extraction pipelines to filter binary files automatically based on content rather than filename extensions.

---

## Transformer architectures

### Multi-head attention

The core attention mechanism with configurable heads implementing scaled dot-product attention with normalization, causal masking for autoregressive models, residual connections, and configurable activation and dropout policies.

Both canonical and fused variants support flexible head count configuration and maintain compatibility with Dlib's training infrastructure.

### Feed-forward networks

Position-wise feed-forward networks following the standard transformer design, expanding to 4× the model dimension, applying activation, then projecting back. Includes skip connection for gradient flow.

**SwiGLU variant**: gated activation (Shazeer, 2020) for improved performance, splitting the expanded dimension and applying gate mechanism through element-wise multiplication.

### Complete transformer blocks

Standard transformer blocks combine attention and feed-forward networks with normalization and residual connections. Stack composition allows building deep architectures through layer repetition.

### Hierarchical reasoning model

Advanced architecture with dual recurrent modules (Hu et al., 2024) implementing iterative refinement:
- high-level reasoning network produces abstract representations
- low-level processing network refines representations through multiple iterations
- process repeats for deep reasoning over multiple cycles

The total reasoning steps equal the product of high-level cycles and low-level iterations, enabling adaptive computation depth.

### Mixture of experts

Sparse conditional computation with per-sample routing, where each input independently selects a subset of expert networks based on learned gating probabilities.

Architecture components:
1. gate network produces expert probabilities via softmax over learned logits
2. top-k expert selection per sample
3. sample routing through selected experts
4. output combination with normalized weights

Features include:
- per-sample routing enabling different experts for different inputs
- exploration noise during training via Gaussian perturbation of gate logits
- load balancing loss to prevent expert collapse
- usage tracking via exponential moving average

Load balancing loss formulation:
```
L_aux = α · N · Σ(f_e · P_e)
```
where f_e represents routing fraction, P_e represents average gate probability for expert e, α is the balancing weight coefficient, and N is the number of experts.

The MoE feed-forward layer serves as drop-in replacement for standard feed-forward networks, combining gate network, expert routing, normalization, and skip connections. This architecture increases model capacity without proportional compute increase, as only top-k experts are active per sample.

### Adaptive computation time

Dynamic computation allocation based on input complexity (Graves, 2016), learning to continue processing until confidence threshold is reached. Includes ponder cost penalty to encourage efficiency and maximum step limit to prevent runaway computation.

---

## Loss functions

### Cross-entropy per logit

Specialized loss for sequence models working directly with linear layer output, computing cross-entropy only at the last position of each sequence. This matches autoregressive next-token prediction and avoids dimension flattening required by standard classification loss, preserving sequence structure.

Implements numerically stable softmax computation via log-sum-exp trick to prevent overflow in exponential calculations.

### Standard classification loss

For smaller models, standard multiclass logarithmic loss with final fully connected layer remains effective, providing dimensionality reduction and additional parameter capacity before classification.

---

## Example programs

Three progressive examples demonstrate transformer-based language modeling:

### Basic character-level model

Minimal transformer training on Shakespeare text, demonstrating fundamental concepts:
- character-level tokenization (256 characters plus padding token)
- small transformer architecture with 3 layers and 4 attention heads
- 64-dimensional embeddings with 50-token context window
- training on approximately 14,600 sequences

The basic example illustrates core mechanisms of attention and demonstrates perfect memorization capability on character sequences, serving as educational tool for understanding transformer mechanics.

### Advanced tokenization and training

BPE-based training pipeline with specialized loss function:
- byte-pair encoding tokenization with 3,500 vocabulary entries
- compact architecture with 4 layers and 6 attention heads
- 228-dimensional embeddings with 100-token context window
- demonstrates byte-for-byte reproduction of training data

The advanced example introduces subword tokenization and modern architectural patterns, showing how BPE enables efficient vocabulary management while maintaining model compactness. The specialized loss function works directly with sequence outputs, avoiding dimension flattening.

### Mixture of experts with data augmentation

Sparse conditional computation with enhanced training techniques:
- mixture-of-experts architecture with 4 experts per layer
- automatic top-n expert selection (20% active per sample)
- data augmentation through shuffle_training_dataset() and augment_training_dataset()
- noise injection for improved robustness and generalization
- load balancing mechanism to prevent expert collapse
- text similarity metrics for generation quality assessment

The MoE example demonstrates production-grade training techniques including randomization and noise injection, showing how sparse activation enables efficient scaling. The augmentation utilities improve model robustness when training on large volumes of information.

---

## Theoretical background

### Attention mechanism foundations

The attention mechanism, introduced by Bahdanau et al. (2015) for neural machine translation and generalized by Vaswani et al. (2017) in the Transformer architecture, computes weighted combinations of values based on query-key compatibility.

The scaled dot-product formulation uses scaling factor 1/√d_k to prevent softmax saturation as dimension increases. Multi-head attention applies this mechanism in parallel across different representation subspaces, enabling the model to attend to information from different positions and representation aspects simultaneously.

### Positional encoding rationale

Attention operations are permutation-equivariant, processing all positions in parallel without inherent position information. Positional encodings inject position signals, either through:

- additive encoding: adding position-dependent vectors to input embeddings
- relative encoding: modifying attention computation to incorporate position differences
- rotary encoding: applying rotation transformations maintaining relative position relationships

The choice of encoding strategy affects the model's ability to extrapolate to sequence lengths beyond training data.

### Normalization in deep networks

Normalization layers address training stability in deep networks by controlling activation statistics. LayerNorm normalizes across the feature dimension for each sample independently, while RMS normalization simplifies this by removing the mean centering step.

The learnable affine transformation following normalization allows the network to recover the original distribution if beneficial, balancing stability with representational capacity.

### Mixture of experts motivation

Mixture of experts architectures enable conditional computation, where different network components activate for different inputs. This approach increases model capacity while maintaining computational efficiency, as the total computation per sample remains bounded by the number of active experts.

The gating mechanism learns to route inputs to appropriate experts, potentially enabling specialization where different experts handle different input types or subtasks. Load balancing mechanisms prevent collapse scenarios where the gating network learns to route all inputs to a small subset of experts.

---

## Citation

If you use these extensions in your research, please cite:

```bibtex
@software{dlib_transformer_extensions,
  title = {Dlib Transformer extensions},
  author = {[Cydral Technology, Aldric Pierrain]},
  year = {2025},
  url = {https://github.com/Cydral/Dlib-Transformer-extensions}
}
```

Core references:
- Vaswani et al. (2017). "Attention Is All You Need." NeurIPS.
- Bahdanau et al. (2015). "Neural Machine Translation by Jointly Learning to Align and Translate." ICLR.
- Su et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding." arXiv:2104.09864.
- Zhang & Sennrich (2019). "Root Mean Square Layer Normalization." NeurIPS.
- Shazeer (2020). "GLU Variants Improve Transformer." arXiv:2002.05202.
- Graves (2016). "Adaptive Computation Time for Recurrent Neural Networks." arXiv:1603.08983.
- Hu et al. (2024). "Hierarchical Reasoning Model for Complex Problem Solving." arXiv:2506.21734.

---

## License

This extension maintains Dlib's Boost Software License. See LICENSE for details.

## Acknowledgments

Built upon the [Dlib](http://dlib.net/) library by Davis E. King. The extensions follow Dlib's design philosophy: simple APIs, comprehensive documentation, and production-ready implementations.

