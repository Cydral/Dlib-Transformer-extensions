# Examples

This directory contains progressive demonstrations of transformer-based language modeling with Dlib, from basic character-level training to advanced mixture-of-experts architectures.

## Available examples

### slm_basic_train_ex.cpp

Minimal transformer implementation for character-level text generation.

**Purpose**: educational introduction to transformer training and autoregressive generation.

**Key characteristics**:
- character-based tokenization treating each character as a discrete token
- small transformer architecture designed for simplicity and efficiency
- training on Shakespeare text extract
- demonstrates attention mechanics and memorization capability

The basic example serves as foundation for understanding transformer principles, showing how the model can perfectly memorize and reproduce character sequences from training data. This illustrates the attention mechanism's ability to capture sequential dependencies.

**Architecture notes**: as documented in the source file, the model uses 3 transformer layers with 4 attention heads, 64-dimensional embeddings, and a 50-token context window. The vocabulary consists of 257 tokens (256 characters plus padding). Total parameter count is approximately 5.2 million, resulting in a model file of about 20MB.

**Training characteristics**: the program demonstrates training on approximately 14,600 sequences extracted from the Shakespeare text. After training, the model achieves near-perfect prediction accuracy (approximately 99.99%) on the training data.

**Generation behavior**: when generating text, the model produces sequences very similar or identical to the original training data, as documented in the source comments showing example output in Shakespearean style.

---

### slm_advanced_train_ex.cpp

Production-ready training pipeline with tokenization, augmentation, and validation monitoring.

**Purpose**: demonstrates complete training workflow for practical language modeling applications.

**Key enhancements**:
- byte-pair encoding (BPE) tokenization with vocabulary learning
- multi-document training capability
- data augmentation through noise injection
- validation set monitoring with model checkpointing
- learning rate scheduling with decay
- early stopping based on patience threshold

The advanced example introduces techniques essential for training models on diverse text sources, handling vocabulary construction from corpus, and preventing overfitting through validation monitoring.

**Tokenization approach**: uses BPE algorithm to learn subword units from the training corpus, balancing vocabulary size with coverage. The tokenizer learns merge operations that combine frequent character sequences, enabling the model to handle words outside the training vocabulary through subword decomposition.

**Data augmentation strategy**: implements noise injection by randomly replacing tokens with unknown token symbols, improving model robustness to variations and errors. The augmentation ratio controls the proportion of noisy copies added to the training set.

**Validation methodology**: splits data into training and validation sets, monitoring validation accuracy during training. Saves checkpoints when validation performance improves, enabling recovery of the best model state even if training continues past the optimal point.

**Architecture scale**: larger than the basic example, with 4 transformer layers, 6 attention heads, 228-dimensional embeddings, and 100-token context window. Parameter count reaches approximately 4 million.

---

### slm_mixture_of_experts_ex.cpp

Sparse conditional computation with per-sample expert routing.

**Purpose**: advanced architecture demonstrating mixture-of-experts for efficient scaling.

**Key innovations**:
- MoE feed-forward layers replacing standard networks
- per-sample dynamic routing to different expert subnetworks
- load balancing mechanism preventing expert collapse
- expert usage statistics for monitoring specialization
- text similarity metrics for generation quality assessment

The MoE example shows how sparse activation enables scaling model capacity without proportional compute increase. Each input selects only a subset of available experts, reducing inference computation while maintaining large total parameter count.

**Routing mechanism**: a gating network learns to produce probability distributions over experts for each input. The top-k experts with highest probabilities are selected and activated, with their outputs combined using the normalized gate probabilities as weights.

**Load balancing**: an auxiliary loss encourages uniform expert utilization across the training set. Without this mechanism, the gating network might learn to route all inputs to a few experts, leaving others unused. The balancing loss adds a penalty proportional to the imbalance in expert usage.

**Expert specialization**: through training, different experts may specialize for different input patterns or subtasks. The usage statistics tracked during training provide visibility into whether experts are balanced or if certain experts dominate.

**Quality assessment**: the example includes similarity metrics comparing generated text with reference sequences. Multiple metrics (edit distance, token overlap, n-gram matching) provide complementary views of generation quality.

---

## Example progression

The three examples form a pedagogical sequence:

**Stage 1 - basic**: establishes fundamental concepts of transformer architecture, attention mechanisms, and autoregressive generation. Uses simplest tokenization (character-level) and smallest model to clearly demonstrate core principles.

**Stage 2 - advanced**: introduces production techniques including subword tokenization, data augmentation, validation monitoring, and larger model capacity. Shows complete training workflow applicable to real applications.

**Stage 3 - MoE**: demonstrates advanced architectural pattern enabling efficient scaling through conditional computation. Introduces concepts of expert routing, load balancing, and sparse activation.

Each stage builds upon the previous, progressively adding sophistication while maintaining clarity in demonstrating specific concepts.

---

## Training considerations

**Data preparation**: all examples use internal datasets from `slm_data.h`, providing consistent training sources. The dataset preparation functions create sliding windows over token sequences, with each window serving as input and the following token as target.

**Augmentation benefits**: noise injection in the advanced example improves robustness by exposing the model to imperfect inputs during training. This prevents overfitting to exact token sequences and helps the model handle variations.

**Validation strategy**: monitoring validation performance provides early signal of overfitting. When training accuracy continues improving but validation accuracy plateaus or degrades, the model is memorizing training data rather than learning generalizable patterns.

**MoE stability**: mixture-of-experts training can be less stable than standard transformers due to the discrete routing decisions and balancing constraints. The load balancing loss coefficient and exploration noise level require tuning for optimal results.

---

## Extending the examples

The examples provide templates for experimentation with different configurations:

**Architecture variations**: modify layer counts, attention heads, embedding dimensions, or context window length to explore capacity-performance tradeoffs.

**Tokenization alternatives**: experiment with different vocabulary sizes for BPE tokenization, or implement alternative tokenization schemes like WordPiece or SentencePiece.

**Training techniques**: add gradient clipping, implement learning rate warmup, or explore alternative optimization algorithms.

**MoE configurations**: vary expert count, top-k selection, or balancing loss weight to study effects on specialization and performance.

**Evaluation metrics**: implement additional metrics like perplexity, specific domain accuracy, or human evaluation protocols.

The modular structure of the examples facilitates such modifications while maintaining compatibility with Dlib's training infrastructure.
