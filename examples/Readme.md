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

BPE-based training with specialized loss function and compact architecture.

**Purpose**: demonstrates modern tokenization and efficient architectural patterns for practical language modeling.

**Key enhancements**:
- byte-pair encoding (BPE) tokenization with vocabulary learning from corpus
- compact transformer architecture optimized for efficiency
- specialized loss function (loss_cross_entropy_per_logit) for sequence modeling
- direct sequence-to-sequence processing without dimension flattening
- verification mode for byte-for-byte validation against original text

The advanced example introduces subword tokenization that balances vocabulary size with coverage, enabling the model to handle words outside the training vocabulary through subword decomposition. The specialized loss function works directly with sequence outputs from linear layers, avoiding the need for flattening through fully connected layers.

**Tokenization approach**: uses BPE algorithm to learn 3,500 subword units from the training corpus. The tokenizer learns merge operations that combine frequent character sequences, creating efficient representations that decompose unknown words into known components.

**Architecture design**: as documented in the source file, the model employs a compact configuration with 4 transformer layers, 6 attention heads, and 228-dimensional embeddings with a 100-token context window. Total parameter count is approximately 4 million, producing a model file of about 18.5MB.

**Loss function innovation**: the loss_cross_entropy_per_logit layer computes loss directly on sequence outputs, operating on tensors of shape (batch, 1, seq_len, vocab_size) without requiring dimension flattening. This preserves sequence structure and simplifies the network definition.

**Verification capability**: after generation, the program can perform byte-for-byte comparison between generated and original text, demonstrating perfect memorization when training converges.

---

### slm_mixture_of_experts_ex.cpp

Sparse conditional computation with production-grade training utilities.

**Purpose**: demonstrates mixture-of-experts architecture with advanced dataset preparation techniques.

**Key innovations**:
- mixture-of-experts feed-forward layers replacing standard networks
- 4 experts per layer with automatic top-n selection (20% = 1 expert active)
- per-sample dynamic routing to different expert subnetworks
- production-grade dataset utilities: shuffle_training_dataset() and augment_training_dataset()
- noise injection for improved robustness and generalization
- load balancing mechanism preventing expert collapse
- expert usage statistics for monitoring specialization
- text similarity metrics for generation quality assessment

The MoE example demonstrates how sparse activation enables scaling model capacity without proportional compute increase. As documented in the source file, this program showcases Dlib's advanced utilities for dataset preparation, using shuffling for randomization and noise injection to improve model robustness when training on large volumes of information.

**Routing mechanism**: a gating network learns to produce probability distributions over experts for each input. With automatic top-n selection set to 0, the system selects 20% of available experts (1 out of 4), reducing inference computation while maintaining large total capacity.

**Data augmentation strategy**: the program demonstrates production-grade training techniques through two key utilities. The shuffle_training_dataset() function performs random permutation while maintaining sample-label correspondence. The augment_training_dataset() function replaces random tokens with unknown token symbols, creating noisy copies that expose the model to imperfect inputs during training. This prevents overfitting to exact token sequences and improves the model's ability to handle variations.

**Load balancing**: an auxiliary loss encourages uniform expert utilization across the training set. Without this mechanism, the gating network might learn to route all inputs to a few experts, leaving others unused. The balancing loss adds a penalty proportional to the imbalance in expert usage.

**Expert specialization**: through training, different experts may specialize for different input patterns or subtasks. The usage statistics tracked during training provide visibility into whether experts are balanced or if certain experts dominate.

**Architecture specifications**: as documented in the source file, the model uses 4 transformer layers with 6 attention heads, 228-dimensional embeddings, and a 100-token context window. The vocabulary contains 3,500 BPE tokens. Total parameter count is approximately 6.0 million during training (all experts) but only 5.4 million active during inference (top-1 expert), representing sparse activation.

**Quality assessment**: the example includes comprehensive similarity metrics comparing generated text with reference sequences. Multiple metrics (edit distance, token overlap, n-gram matching) provide complementary views of generation quality, enabling detailed validation of model performance.

---

### slm_chatbot_ex.cpp

Conversational AI with two-stage training and advanced inference techniques.

**Purpose**: demonstrates production-grade chatbot development from base language model training through specialized fine-tuning to interactive generation with sophisticated sampling strategies.

**Key characteristics**:
- two-stage training pipeline: base language model on pure text, then conversational fine-tuning on Q&A pairs
- layer-wise learning rate strategy for efficient fine-tuning with small datasets
- checkpoint-based fine-tuning loading best validation loss from base training
- structured conversational format with role markers: `<question>`, `<answer>`, `<text>`
- stochastic text generation with multiple sampling techniques
- proper per-row softmax for sequence probability distributions
- conversational context management with sliding window

The chatbot example demonstrates the complete pipeline for creating specialized conversational models, showing how a general language model can be efficiently adapted for question-answering tasks with minimal additional training data.

**Training pipeline**: the program implements industry-standard two-stage training. Stage 1 trains a base language model on domain-specific text corpora using simple `<text>content</text>` format, learning language structure and knowledge without conversational patterns. Stage 2 fine-tunes this base model on 100-300 Q&A pairs using structured format `<question><text>Q</text><answer><text>A</text>`, specializing for conversational interaction. This separation enables efficient knowledge acquisition in stage 1 and behavioral specialization in stage 2.

**Fine-tuning strategy**: as documented in the source file, fine-tuning employs differentiated learning rates across network layers. All layers receive 10% of the base learning rate, while specific components receive higher rates: the linear classification head (100%), normalization layers (50%), and embeddings (50%). This strategy preserves pre-trained representations in intermediate layers while allowing adaptation of input/output mappings, preventing catastrophic forgetting while enabling effective specialization.

**Fine-tuning hyperparameters**: the training parameters differ significantly from base training to reflect the small dataset size and specialization objective. Learning rate is set to 1e-5 (compared to 3e-4 for base training), batch size to 16 (versus 64), maximum epochs to 10 (versus 150), and patience to 300 steps (versus 8000). The reduced batch size provides more frequent gradient updates critical when training on small question-answer datasets, while the low learning rate prevents disruption of pre-trained representations.

**Inference innovations**: the program demonstrates proper sequence model inference with per-row softmax (`softmaxm` layer) that computes independent probability distributions at each sequence position, contrasting with global `softmax` that would normalize across all positions. Temperature scaling via `multiply` layer controls output diversity by scaling logits before normalization. The inference pipeline correctly extracts predictions from the last sequence position only, matching the autoregressive training objective.

**Stochastic generation**: the program implements a comprehensive sampling pipeline combining multiple techniques for quality control. Repetition penalty discourages token repetition by reducing probabilities of recently generated tokens (most recent 20% of context). Min-p filtering sets adaptive thresholds relative to maximum probability, automatically adjusting to distribution confidence. Top-k filtering limits consideration to the k most probable tokens. Nucleus sampling (top-p) dynamically selects the smallest token set whose cumulative probability exceeds threshold p. These techniques are applied sequentially with proper renormalization between steps, enabling fine-grained control over generation quality and diversity.

**Context management**: the `inference_context` class maintains conversation history with FIFO buffer behavior, supporting multi-turn conversations with proper handling of role markers. The sliding window extraction aligns to the model's context length, with automatic padding for incomplete contexts. This enables long-form conversations exceeding the model's training window while preserving conversational structure.

**Architecture specifications**: the model uses the same compact configuration as the MoE example with 4 transformer layers, 6 attention heads, 228-dimensional embeddings, and a 100-token context window. The vocabulary contains 3,500 BPE tokens. This demonstrates that effective conversational models can be built with modest architectures when proper training strategies and inference techniques are applied.

**Generation modes**: the program supports both deterministic (argmax) and stochastic generation, with command-line parameters controlling sampling behavior. Deterministic mode selects the highest probability token at each step, producing consistent but potentially repetitive output. Stochastic mode applies the full sampling pipeline, producing diverse responses with controllable randomness. Parameters include temperature (default 0.8), top-k (50), top-p (0.9), repetition penalty (1.2), and min-p (0.05).

**Dataset structure**: the fine-tuning stage expects Q&A pairs formatted with explicit role markers, enabling the model to learn conversational structure. Questions are wrapped in `<question><text>question_text</text>` and answers in `<answer><text>answer_text</text>`. This structured format allows the model to distinguish between user queries and assistant responses, essential for proper conversational behavior. The program includes utilities to display random training examples, facilitating dataset validation and quality assessment.

---

## Example progression

The three examples form a pedagogical sequence:

**Stage 1 - basic**: establishes fundamental concepts of transformer architecture, attention mechanisms, and autoregressive generation. Uses simplest tokenization (character-level) and smallest model to clearly demonstrate core principles.

**Stage 2 - advanced**: introduces modern tokenization (BPE) and specialized loss functions optimized for sequence modeling. Shows efficient architectural patterns with compact dimensions while maintaining strong performance. Demonstrates how specialized components simplify network definition.

**Stage 3 - MoE**: demonstrates advanced architectural pattern enabling efficient scaling through conditional computation. Introduces production-grade dataset preparation utilities including shuffling and noise injection for robust training. Shows concepts of expert routing, load balancing, and sparse activation.

Each stage builds upon the previous, progressively adding sophistication while maintaining clarity in demonstrating specific concepts.

---

## Training considerations

**Data preparation**: all examples use internal datasets from `slm_data.h`, providing consistent training sources. The dataset preparation functions create sliding windows over token sequences, with each window serving as input and the following token as target.

**Tokenization strategies**: character-level tokenization in the basic example provides simplicity but requires the model to learn character combinations. BPE tokenization in advanced and MoE examples creates subword units that balance vocabulary size with coverage, enabling efficient handling of diverse text.

**Augmentation benefits**: noise injection in the MoE example improves robustness by exposing the model to imperfect inputs during training. The augmentation utilities (shuffle_training_dataset and augment_training_dataset) demonstrate production-grade techniques for improving generalization when training on large volumes of information.

**MoE stability**: mixture-of-experts training can be less stable than standard transformers due to the discrete routing decisions and balancing constraints. The load balancing loss coefficient and exploration noise level require tuning for optimal results.

---

## Extending the examples

The examples provide templates for experimentation with different configurations:

**Architecture variations**: modify layer counts, attention heads, embedding dimensions, or context window length to explore capacity-performance tradeoffs. The compact dimensions in advanced and MoE examples demonstrate that smaller architectures can achieve excellent results with appropriate tokenization.

**Tokenization alternatives**: experiment with different vocabulary sizes for BPE tokenization, or implement alternative tokenization schemes like WordPiece or SentencePiece. The vocabulary size affects both model efficiency and generalization capability.

**Training techniques**: the MoE example demonstrates augmentation utilities that can be applied to other architectures. Experiment with different augmentation ratios or implement additional techniques like gradient clipping or learning rate warmup.

**MoE configurations**: vary expert count, top-k selection, or balancing loss weight to study effects on specialization and performance. The automatic top-n selection provides adaptive routing based on the number of experts.

**Loss function choices**: the advanced example can easily be updated to show loss_cross_entropy_per_logit for direct sequence modeling, while basic and MoE examples use standard classification loss. Compare these approaches for different model sizes and tasks.

The modular structure of the examples facilitates such modifications while maintaining compatibility with Dlib's training infrastructure.
