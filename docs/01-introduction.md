# A Small Language Model Stack, Built in C++ on One Machine

*Notes on assembling a full pipeline from corpus to served model, without a cluster, and on the judgement that decides how small is small enough.*

The interesting question about small language models is not how to make one. It is how small a model can get before it stops being worth running, and that boundary moves depending on what you ask of it. A model that answers general knowledge questions badly at 500M parameters may summarise documents perfectly well at the same size. Finding where the line sits for a given task is an exercise in judgement, and judgement needs a pipeline you can turn quickly.

For the past several months I have been building that pipeline as Dlib-Transformer, a set of extensions to Dlib, the C++ machine learning library. The repository imports open-weight models from `GGUF` containers, runs them two different ways, fine-tunes them, distils them into smaller ones, prunes them by depth, and searches document collections with them. Everything in it runs on one laptop, on a CPU or a single consumer GPU, because that constraint is the point rather than a limitation to apologise for.

This article is a map of that work and of the trade it explores.

## The constraint is the subject

There is a version of this field where progress means more compute, and a version where it means getting acceptable behaviour out of hardware that already exists. The second version has become considerably more interesting over the past two years, for reasons that are as much about control as about cost.

An organisation running a model on its own machines knows where its data goes. It is not subject to a provider's deprecation schedule, its rate limits, or its terms changing. That argument gets made most often about regulated sectors, and it applies there, but it applies just as well to anyone who wants a system they can still run in five years.

The price is quality. A model small enough to run on a workstation is worse than one running in a datacentre, and pretending otherwise helps nobody. The useful question is how much worse, at what size, for which task. That is an empirical question, and answering it requires being able to build several models and compare them, which is what this pipeline is for.

## Why Dlib, and what that does and does not mean

The framing that C++ is unusual for this work is out of date. `llama.cpp` is written in C and C++, and it is where a large share of local inference happens; `ggml`, the tensor library beneath it, is C. What is true is that most people reach these libraries through Python bindings, so the C++ underneath is a runtime rather than something they write against.

Dlib is a different proposition, and its history explains why. Davis King has maintained it since 2002 [1], and it began as a general-purpose collection of independent components: networking, threads, linear algebra, data structures, numerical optimization, Bayesian networks. Image processing and computer vision came to dominate its visible surface over the years, and for many people Dlib means face detection and shape prediction. That reputation understates what is underneath.

The deep learning framework added later kept the library's original discipline. A network is a type rather than a runtime graph, so a layer mismatch is a compilation error and not a shape error at step 400. Serialization is generated from the type. The compiler sees the whole network and inlines across layer boundaries. Crucially for what follows, the forward and backward passes are explicit and inspectable at every layer: adding a new layer means writing both, and the framework holds you to the contract between them.

That last point is what made this project possible at all. Extending Dlib towards transformers was not a matter of wrapping an external kernel library; it meant writing the layers, the loss functions and the optimizations against a framework that already knew how to compose and differentiate them. Grouped query attention [2], rotary position embedding [3], RMS normalization, gated feed-forward blocks [4], mixture-of-experts routing, a key-value cache: each is a component in the library's own idiom rather than a special case bolted on.

Dlib-Transformer, the set of extensions this article describes, follows the same two-level pattern Dlib uses elsewhere. Elementary layers exist and can be assembled by hand, which is how the introductory examples are written and how someone learning the material should read them. Above them sit high-level layers that encapsulate the implementation complexity, so that a working decoder is a few lines rather than a few hundred. Making prototyping fluid without hiding what is underneath is one of Dlib's long-standing design drivers, and it applies here unchanged.

The design has costs. Compilation time grows with network depth, template error messages are famously difficult, and an architecture cannot change without recompiling. The trade is deliberate: correctness moves earlier, flexibility moves later. It suits a pipeline where the architecture is decided once and then trained many times.

Dlib also targets CPU and CUDA, and has since long before transformers existed. Every program discussed here builds and runs both ways. That is not a compromise; it is the original contract, and it is what makes the same code usable on a workstation without a GPU and on one with.

## Reading models other people trained

A library that can only train from scratch is a teaching exercise. The first substantial capability was reading `GGUF`, the container format `llama.cpp` established, which is where most open-weight models now circulate.

The loader currently recognises the `llama`, `qwen2`, `qwen3`, `mistral`, `gemma` and `gemma2` architecture identifiers, which between them cover a large share of what is published in small sizes. On the quantization side it dequantizes the legacy formats, the k-quants from `Q2_K` to `Q6_K`, `IQ4_NL`, and the floating-point ones. The grid-based i-quants are not supported.

Models I have imported and run include, for example, `TinyLlama`, `SmolLM2`, `SmolVLM-256M-Instruct` with its vision tower, `Qwen2.5-1.5B-Instruct`, `Qwen3-0.6B`, and `jina-embeddings-v5-text-small-retrieval`. The list is not closed; each new family mostly needs its metadata conventions checked against what the loader expects.

Importing a model remains a somewhat demanding operation even with the mechanisms that automate most of it, and that is a barrier to anyone who wants to use Dlib-Transformer rather than extend it. I am considering a repository of already-imported models, downloadable in the native archive format and usable immediately, so that the first thing a newcomer does is run something rather than debug a conversion.

Reading happens two ways, and the reason there are two is the clearest illustration of what the template design costs and buys.

The **runtime engine** reads any supported container and shapes itself to it. Layer counts, widths, head counts and quantization formats come from the file at load time. It serves a model without recompilation, which is what you want when the model is an input to your program rather than a part of it.

The **static path** does the opposite. A code generator reads the container and emits a C++ header declaring the network as a type. That header is compiled in, and from then on the model is an ordinary Dlib network: it trains with Dlib's standard trainer, serializes with its standard mechanism, and accepts adapters like anything else. The price is a compilation step per architecture. The benefit is that the rest of Dlib applies to it without special cases.

Both paths must agree, and checking that they do catches an entire class of import bugs that otherwise produce plausible-looking output.

## Dense and sparse, both supported

Two architecture families dominate what gets published at small sizes, and Dlib-Transformer handles both.

A **dense** model runs every parameter for every token. It is simple, predictable, and its memory footprint equals its compute footprint. Almost everything under a billion parameters is dense.

A **mixture of experts** replaces the feed-forward block with several independent ones and a small gate that routes each token to a few of them. The argument, made at scale by Fedus and colleagues [5], is that capacity and computation stop being the same quantity: a model can hold parameters it does not run at every step.

That distinction matters more on one machine than in a datacentre, and in a specific way. Sparse routing trades memory for compute, and on a workstation memory is usually the cheaper of the two. A configuration I have been working with holds four experts routed two at a time: 269M parameters stored, 151M activated. The storage is paid once at load; the arithmetic per token is that of a much smaller model.

Dlib-Transformer implements both. The runtime engine serves mixture containers, and the training path builds them with the gate trained jointly.

## Spending depth where the question is hard

One element of the target architecture has no equivalent in the open models I am aware of, and it is the piece I find most interesting.

A transformer spends the same computation on every token. The word "the" traverses all 28 blocks exactly as a token requiring inference does. That is obviously wasteful, and the idea of making it conditional is old: Graves proposed adaptive computation time for recurrent networks in 2016 [6], Dehghani and colleagues carried it into transformers with Universal Transformers in 2018 [7], and Banino and colleagues reformulated the halting decision probabilistically as PonderNet in 2021 [8].

The idea has not made it into mainstream open models, and the reason is honest: the halting signal is hard to train. A network given the option to stop early will take it, because stopping is cheap and the penalty is deferred. Keeping the mechanism from collapsing into "always halt immediately" requires a regularization term whose weight is itself a hyperparameter, and getting that wrong wastes a training run.

More recent work approaches the same goal from different angles. Mixture-of-Depths [9] lets a router skip blocks for some tokens under a fixed compute budget, which sidesteps the halting problem by making the budget a constraint rather than a learned quantity. Confident Adaptive Language Modeling [10] exits early at inference based on confidence, without changing training.

`dlib` carries an ACT layer, and the architecture I am building uses it. The bet is that a small model benefits from this more than a large one does: when total capacity is scarce, spending it selectively matters more. That is a bet and not a result. I have not yet trained the model that would settle it.

## Four ways to make a model smaller

Size is what stands between an open-weight model and a machine that can run it, and the repository implements four approaches to it.

**Low-rank adaptation** is the least ambitious and the most used. LoRA [11] attaches small matrices to the attention projections, freezes the base weights, and merges the adapters back when training ends. Dlib-Transformer implements LoRA and DoRA, with base parameters optionally kept in host memory and materialized per layer.

**Distillation on logits** raises a new model from a larger one's predictions. The idea is Hinton's [12]: a hard label says the next token is X, while a teacher's distribution says X, but Y was nearly as good and Z was plausible. That ordering over the vocabulary is not present in the corpus, and no student could derive it from the text alone.

The implementation records the teacher once over a corpus, storing its top-k logits at every position into a trace file, then trains students from that recording without the teacher running again. Because the transfer happens at the output distribution, the student's internals are unconstrained: a dense teacher can raise a mixture-of-experts student of a different width and depth. The only thing both must share is the tokenizer.

**Depth pruning** takes the opposite approach: keep the teacher's weights, remove whole blocks, and distil the survivors back into shape. Which blocks go is measured rather than assumed, by recording how far each block moves what passes through it.

Take one model as an illustration. Measured this way, `Qwen3-0.6B` does most of its work at the beginning and the end of its 28 blocks, and several blocks in its middle third barely change what passes through them. That pattern is not specific to it, and the same measurement on another model would give a different set of block indices, which is the point of measuring rather than assuming.

The two distillation methods trade the same thing in opposite directions. Pruning starts from the teacher's weights and cannot change the architecture; logit distillation starts from nothing and can change everything. At an equal budget the first wins, which is precisely why the second exists.

**Vocabulary sizing** is the fourth, and it is the one nobody mentions. A published small model inherits a tokenizer dimensioned for the largest member of its family: `Qwen3-0.6B` spends 32% of its parameters on a table of 151,936 entries. A project that trains its own tokenizer does not have to. Sizing the vocabulary to the model holds the embedding table to a sixth of the parameter count and puts the difference into the blocks, where it does work.

That freedom has to be used carefully. A vocabulary is a fixed budget shared between writing systems that share nothing with each other. Latin scripts pool their subwords, so four European languages cost little more than one; every other script starts from the raw bytes, and UTF-8 spells a Han character in three of them. A budget spread too thin leaves a language segmented almost byte by byte, which consumes vocabulary and delivers nothing.

## Where the judgement actually happens

Everything above exists to support one decision, made repeatedly: how much can be removed before the model stops doing what it is needed for.

The measurements are unambiguous about one thing. Pruning `Qwen3-0.6B` from 28 blocks to 24 and distilling produced a model that still answers "The capital of France is" with "Paris" as its first prediction. Pruning to 20 blocks and distilling on five times as much data did not. **Four more blocks beat five times more data**, by a wide margin, for a fifth of the compute.

The damage profile is consistent across sizes. Syntactic competence survives pruning easily: the reduced models predict function words with confidence matching or exceeding the teacher's. What degrades is factual recall, which appears to live in the middle blocks that the influence measurement marks as idle. The criterion measures how far a block moves a vector, not how much is stored in it, and a block can move little while holding a great deal.

That is the trade in its clearest form. A pruned model keeps the language and loses the encyclopaedia. Whether that is acceptable depends entirely on whether the encyclopaedia is what you needed, and for a model that answers from retrieved documents rather than from memory, it may not be.

## Retrieval, which changes the calculation

Adding document search took less work than expected, for a reason obvious in hindsight. An embedding model is the decoder already implemented with its last step removed: instead of projecting the final hidden state onto the vocabulary, it returns that state. Containers say so themselves. `jina-embeddings-v5-text-small-retrieval` declares itself a `qwen3`, carries the usual blocks and final normalization, and has no output projection at all, so the same loader reads it with no special case.

This matters for the size question directly. If a model answers from passages it is handed rather than from what it memorised, the factual recall that pruning damages is less of what it needed in the first place. A smaller model paired with a good index can outperform a larger one working from memory, and it can cite where the answer came from.

The parts that decide whether it works are unglamorous: cutting passages on sentence boundaries and starting them on a word, encoding questions and documents differently because retrieval models are trained on pairs of the two, and shortlisting by angular sketch once the index outgrows an exact scan.

## Where Dlib-Transformer stands

Dlib-Transformer covers import, both execution paths, fine-tuning, both distillation methods, retrieval, and a vision tower for models that read images alongside text. The tooling for building a model from nothing is in place as well: corpus collection, byte-level BPE training and measurement, pre-training, instruction tuning, benchmarking.

That completeness is what I would point to rather than any single feature. The components needed to assemble a model, and to assemble one whose structure departs from what is currently deployed, are present and work together. The library has moved far enough that the question is no longer whether the pieces exist, but what to build with them.

I am currently evaluating one answer to that, a family I call Cygnus: a vocabulary sized to the model rather than inherited, a width of 640 across 18 blocks, four experts routed two at a time, and adaptive computation. Whether that particular experiment reaches completion is open, since the compute it needs runs to weeks on one laptop GPU.

The point is not that particular design. It is that the components are there for anyone who wants to try their own, which is a different situation from the one this project started in.

Some of what is already built is easy to overlook because it sits at the far end of the pipeline. One example program imports a model, starts a local HTTP server, and opens a working chat interface in a browser, alongside an API that speaks the shape most clients already expect. Dlib-Transformer carries a complete web front end, which means a model you have just converted or trained can be tried immediately and comfortably rather than through a command line, and it is where local retrieval will surface as well.

The project is active, and more is coming.

## What I hope this is useful for

The pipeline exists because I wanted to explore the size-versus-usefulness trade myself rather than read about it. If it helps anyone else do the same, in C++ or by reading what the C++ does, that is the better outcome.

The general direction the field is moving, towards models that run on ordinary hardware under their owner's control, seems to me both correct and under-tooled. There is a great deal of writing about what these models can do and comparatively little about how to build one on a machine you already have. That gap is where this work sits.

## References

1. King, D. E. (2009). Dlib-ml: A Machine Learning Toolkit. *Journal of Machine Learning Research*, 10, 1755-1758. http://dlib.net
2. Ainslie, J., et al. (2023). *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*. arXiv:2305.13245. https://arxiv.org/abs/2305.13245
3. Su, J., et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding*. arXiv:2104.09864. https://arxiv.org/abs/2104.09864
4. Shazeer, N. (2020). *GLU Variants Improve Transformer*. arXiv:2002.05202. https://arxiv.org/abs/2002.05202
5. Fedus, W., Zoph, B., & Shazeer, N. (2021). *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*. arXiv:2101.03961. https://arxiv.org/abs/2101.03961
6. Graves, A. (2016). *Adaptive Computation Time for Recurrent Neural Networks*. arXiv:1603.08983. https://arxiv.org/abs/1603.08983
7. Dehghani, M., et al. (2018). *Universal Transformers*. arXiv:1807.03819. https://arxiv.org/abs/1807.03819
8. Banino, A., Balaguer, J., & Blundell, C. (2021). *PonderNet: Learning to Ponder*. arXiv:2107.05407. https://arxiv.org/abs/2107.05407
9. Raposo, D., et al. (2024). *Mixture-of-Depths: Dynamically allocating compute in transformer-based language models*. arXiv:2404.02258. https://arxiv.org/abs/2404.02258
10. Schuster, T., et al. (2022). *Confident Adaptive Language Modeling*. arXiv:2207.07061. https://arxiv.org/abs/2207.07061
11. Hu, E. J., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685. https://arxiv.org/abs/2106.09685
12. Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the Knowledge in a Neural Network*. arXiv:1503.02531. https://arxiv.org/abs/1503.02531
13. Cydral. Dlib-Transformer-extensions. GitHub. https://github.com/Cydral/Dlib-Transformer-extensions

---

If you have pushed a model further down in size than seemed reasonable and it still did the job, or found the point where it stopped, I would like to hear where that line fell and what the task was. That boundary is the whole subject, and it is not one anybody can map alone. The repository is open, and the articles that follow will take the parts of this pipeline one at a time.

---

[Back to the article index](Readme.md)
