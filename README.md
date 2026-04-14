# Exploring Neural Network Approaches to ARC with Regularized Task Embeddings

This repository is an exploration of different deep learning techniques applied to the [**ARC-AGI tasks**](https://arcprize.org/arc), specifically the [**MINI-ARC**](https://github.com/KSB21ST/MINI-ARC) variant (5×5 grids, 149 tasks). The central question driving this work: can regularizing the task embedding space improve a neural network's ability to generalize across ARC tasks?

## Motivations and vision

My approach in this repository is motivated by the following thinking:

1. **Inspired by the Tiny Recursive Model (TRM)**
   The recent success of the [Tiny Recursive Model](https://arxiv.org/abs/2510.04871) on ARC-AGI suggested that relatively small transformer-based models with learned per-task embeddings are a promising direction. The core idea is that a model can learn a shared input-output transformation function, conditioned on a compact task-specific latent vector that gets optimized at test time.

2. **The Regularization Hypothesis**
   My intuition was that regularizing the task embedding space — through vector quantization or LeJEPA-style compression to an isotropic Gaussian — might help the model learn more structured, generalizable representations of ARC tasks. Techniques explored include:
   - **Random Projection Quantizer (RQ)**: Random projection into a discrete codebook with Gumbel softmax quantization, inspired by BEST-RQ
   - **wav2vec 2.0 style quantization**: Use gumbel softmaxes to quantize, similar to wav2vec 2.0
   - **Finite Scalar Quantization (FSQ)**: Deterministic quantization to a bounded integer lattice
   - **JEPA (Joint Embedding Predictive Architecture)**: Latent space compression inspired by LeJEPA, which regularizes embeddings toward an isotropic Gaussian

4. **MINI-ARC for Efficiency**
   Rather than training on full ARC grids (up to 30×30), all experiments use MINI-ARC (fixed 5×5 grids). Combined with aggressive data augmentation — random grid rotations, reflections, and color permutations — this yields ~50,000 tasks while still making rapid iteration practical on a single GPU.

## Details of the approach

### Data augmentation

Each of the 149 MINI-ARC tasks is augmented using all 8 symmetries of the square (rotations and reflections) combined with random color permutations, producing ~50,000 tasks in total. Tasks are split 80/20 into train (120 tasks) and test (29 tasks) sets at the task level before augmenting, so the test set contains entirely unseen task types.

### Shared iterative denoising approach

Nearly all modules share the same core inference procedure, inspired by masked diffusion. The output grid is initially fully masked (all 25 tokens set to a special mask value). At each of 25 iterations, the model takes the input grid, the current (partially unmasked) output grid, and the task embedding as inputs, then predicts a probability distribution over all output positions. The mask token at the most confident position is replaced with the predicted token, and the process repeats until the output is fully revealed. What differs between modules is how the **task embedding** is represented and regularized during training. In some approaches (e.g. JEPA) I also experimented with a GPT-style autoregressive decoder — predicting output tokens sequentially left-to-right rather than via iterative denoising — and observed similar accuracy plateaus, further suggesting the bottleneck is not the decoding strategy.

### Module overview

| Module | Approach | Key idea |
|---|---|---|
| [`nb_rq.py`](https://github.com/jblackwood/mini_arc_neural_networks/blob/main/mini_arc_rq/nb_rq.py) | Random Projection Quantizer | Task token projected into a frozen random codebook |
| [`nb_fsq.py`](https://github.com/jblackwood/mini_arc_neural_networks/blob/main/mini_arc_fsq/nb_fsq.py) | Finite Scalar Quantization | Task token quantized to a discrete bounded integer lattice |
| [`nb_2vec.py`](https://github.com/jblackwood/mini_arc_neural_networks/blob/main/mini_arc_2vec/nb_2vec.py) | wav2vec 2.0 style | Gumbel softmax over quantized task token categories |
| [`nb_jepa.py`](https://github.com/jblackwood/mini_arc_neural_networks/blob/main/mini_arc_jepa/nb_jepa.py) | JEPA | A LeJEPA-style encoder builds an isotropic Gaussian embedding of each task and a decoder takes the embedding and an input grid to produce an output grid  |


### Standalone scripts

Each `nb_*.py` script is intentionally self-contained — all model definitions, dataset creation, training loop, and evaluation code lives in a single file with no local package dependencies. This makes it easy to copy-paste the entire file into a [Google Colab](https://colab.research.google.com/) notebook cell and run it without any additional setup. Each script will automatically download the MINI-ARC dataset on first run and save checkpoints and TensorBoard logs to the configured output directory (or Google Drive when running in Colab).

Each module also contains:
- `create_tensor_board_events.py` — utilities for replaying training metrics into TensorBoard
- `results/` — saved checkpoint metadata and training logs

### Evaluation

The primary metric is **task accuracy**: the fraction of test tasks for which the model predicts the correct output grid for the held-out test example (exact match).

## Learnings

Despite the variety of regularization techniques, **all approaches plateau at approximately 25% task accuracy** on the MINI-ARC test set with ~5M parameter transformer encoder models. This plateau is remarkably consistent across quantization (FSQ, RQ, wav2vec 2.0 style Gumbel softmax) and LeJEPA, suggesting that the bottleneck is not the structure of the task embedding space but something more fundamental.

This convergence of results has convinced me that **pure neural network approaches are insufficient** for robust ARC generalization, and that further progress requires additional methods (e.g. neuro-symbolic).

## Future directions

These findings motivate me to explore hybrid neuro-symbolic approaches. In particular, I am interested in combining learned neural representations (like the task embeddings here) with structured program search. Specifically, I am inspired by:

- **Search-based RL** algorithms such as AlphaGo/AlphaZero and [EfficientZero](https://arxiv.org/abs/2111.00210) applied to program synthesis — using a world model to plan over a DSL action space.
- **[clingoARC](https://github.com/jblackwood/clingoArc)** (my repository from Answer Set Programming explorations): encoding ARC program search as first-order logic metaprogramming and solving with Answer Set Programming. The grounding bottleneck encountered there motivates using neural heuristics to drastically prune the search space.

