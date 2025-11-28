# Pre-trained models

This directory contains trained models ready for immediate inference and experimentation. All models were trained on internal datasets and are provided for demonstration purposes.

## Available models

### dlib_lm_chars_model.dat (refer to slm_basic_train_ex.cpp)

Character-level transformer trained on Shakespeare text.

**Architecture specifications**:
- 3-layer transformer with 4 attention heads
- 64-dimensional embeddings
- 50-token context window
- vocabulary of 257 tokens (256 characters plus padding)
- parameter count: 5,185,864
- file size: approximately 20 MB

**Training context**:
the model was trained on Shakespeare text extract containing approximately 14,600 character sequences. Training utilized the Adam optimizer with standard hyperparameters and achieved near-perfect accuracy on the training corpus (approximately 99.99%).

**Characteristics**: this model demonstrates the transformer's capability for perfect memorization of character sequences. When generating text, it produces output with high fidelity to the training distribution, reproducing Shakespearean style and structure.

---

### dlib_lm_tokens_model.dat (refer to slm_advanced_train_ex.cpp)

BPE tokenizer model with compact architecture and specialized loss function.

**Architecture specifications**:
- 4-layer transformer with 6 attention heads
- 228-dimensional embeddings
- 100-token context window
- vocabulary of 3,500 BPE tokens
- parameter count: 4,002,458
- file size: approximately 18.5 MB

**Training context**:
trained on internal datasets using byte-pair encoding tokenization learned from the training corpus. The model employs a compact architecture optimized for efficiency while maintaining strong memorization capability. User can change to use the specialized loss function (loss_cross_entropy_per_logit) that operates directly on sequence outputs without dimension flattening.

Training achieved perfect memorization of the dataset, enabling byte-for-byte reproduction of the original text during generation. The compact architecture demonstrates that smaller models with appropriate tokenization can achieve excellent results.

**Associated files**: requires the vocabulary file `dlib_lm_tokenizer.vocab` containing the learned BPE merge operations and token mappings. Both model and vocabulary must be present for proper operation.

**Characteristics**: this model demonstrates efficient architecture design with BPE tokenization enabling subword decomposition for handling diverse text. The specialized loss function can also simplify network definition while maintaining sequence structure throughout processing.

---

### dlib_lm_moe_model.dat (refer to slm_mixture_of_experts_ex.cpp)

Mixture-of-experts model with sparse activation and data augmentation.

**Architecture specifications**:
- 4-layer transformer with MoE feed-forward networks
- 4 experts per layer with automatic top-n selection (20% = 1 expert active)
- 6 attention heads
- 228-dimensional embeddings
- 100-token context window
- vocabulary of 3,500 BPE tokens
- parameter count: 5,970,554 (training) / 5,432,738 (inference)

**Training context**:
Model trained using production-grade dataset preparation utilities including shuffle_training_dataset() for randomization and augment_training_dataset() for noise injection. These techniques improve model robustness and generalization when training on large volumes of information.

Each input sample routes to a single expert per layer (automatic top-n = 20% of 4 experts), enabling efficient inference through sparse activation. The training incorporated exploration noise on gate logits to encourage diverse expert utilization, and load balancing loss to prevent expert collapse. Expert usage statistics showed balanced utilization, indicating successful load distribution across experts.

**Associated files**: requires vocabulary file `dlib_lm_tokenizer.vocab` matching the BPE tokenization scheme.

**Efficiency characteristics**: during inference, only one expert per layer is active due to sparse expert selection. This enables efficient inference while maintaining large total capacity. The parameter count difference between training (all experts) and inference (active experts) demonstrates the efficiency gain from sparse activation.

**Expert specialization**: through training with balanced load distribution, the four experts in each layer developed distinct activation patterns across different input types. The specialization emerges from the gating mechanism learning to route different patterns to appropriate experts.

---

## Model characteristics

**Basic model**: optimized for demonstrating transformer fundamentals with minimal resource requirements. The character-level tokenization and small architecture make it suitable for educational purposes and quick experimentation. Achieves perfect memorization of training sequences.

**Advanced model**: demonstrates efficient architecture design with compact dimensions (4 layers, 6 heads, 228-dim embeddings). The BPE vocabulary enables handling diverse text sources through subword decomposition. Uses specialized loss function that operates directly on sequences. Achieves byte-for-byte reproduction of training data.

**MoE model**: demonstrates efficient scaling through conditional computation with sparse activation. The automatic top-n selection (20%) reduces inference cost while maintaining large total capacity. Trained with production-grade augmentation utilities (shuffle and noise injection) for improved robustness. Expert usage statistics show balanced load distribution.

---

## Tokenizer vocabularies

The vocabulary files contain:
- BPE merge operations learned from training corpus
- token-to-ID mappings for encoding
- special token definitions (padding, unknown, end-of-text, etc.)
- statistics used for vocabulary construction

The vocabulary learning process balances coverage (handling diverse text) with efficiency (limiting vocabulary size to 3,500 entries). Merge operations combine frequent character sequences, creating subword units that decompose unknown words into known components.

---

## File format details

Dlib's serialization format provides:
- binary encoding for compact storage
- version strings for compatibility verification
- nested structure support for complex architectures
- automatic endianness handling for portability

The serialization captures the complete network state, enabling exact reconstruction of the trained model. Layer parameters are stored hierarchically, preserving the network's compositional structure. MoE models include additional expert network weights and gating parameters.

During deserialization, Dlib automatically reconstructs the network layers and populates weights from the stored values. The architecture definition in the loading code must match the architecture used during training for successful reconstruction.

---

## Future model releases

Planned additions to this collection:

**Larger MoE variants**: configurations with 8 experts and top-2 selection, increasing capacity while maintaining sparse activation benefits.

**Hierarchical reasoning models**: implementations of the HRM architecture demonstrating iterative refinement over multiple reasoning cycles.

**Vision transformers**: image classification models when vision transformer components become available, demonstrating the versatility of the architectural patterns.

