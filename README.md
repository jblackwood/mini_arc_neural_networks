# Exploring Neural Network Approaches to ARC with Regularized Task Embeddings

This repository is an exploration of different deep learning techniques applied to the [**ARC-AGI tasks**](https://arcprize.org/arc), specifically the [**MINI-ARC**](https://github.com/KSB21ST/MINI-ARC) variant (5×5 grids, 149 tasks). The central question driving this work: can regularizing the task embedding space improve a neural network's ability to generalize across ARC tasks?

## Motivations and vision

My approach in this repository is motivated by the following thinking:

1. **Inspired by the Tiny Recursive Model (TRM)**
   The recent success of the [Tiny Recursive Model](https://arcprize.org/) on ARC-AGI suggested that relatively small transformer-based models with learned per-task embeddings are a promising direction. The core idea is that a model can learn a shared input-output transformation function, conditioned on a compact task-specific latent vector that gets optimized at test time.

2. **The Regularization Hypothesis**
   My intuition was that regularizing the task embedding space — through noise injection (VAE-style), vector quantization, or LeJEPA-style compression to an isotropic gaussian — might help the model learn more structured, generalizable representations of ARC tasks. Techniques explored include:
   - **VAE**: Gaussian noise injection via the reparameterization trick
   - **JEPA (Joint Embedding Predictive Architecture)**: Latent space compression inspired by LeJEPA, which regularizes embeddings toward an isotropic Gaussian
   - **wave2vec 2.0 / Random Projection Quantizer (RQ)**: Random projection into a discrete codebook with Gumbel softmax quantization, inspired by wav2vec 2.0
   - **Finite Scalar Quantization (FSQ)**: Deterministic quantization to a bounded integer lattice
   - **Equilibrium Matching (EQM)**: Iterative denoising / equilibrium-finding in the embedding space

3. **MINI-ARC for Efficiency**
   Rather than training on full ARC grids (up to 30×30), all experiments use MINI-ARC (fixed 5×5 grids). Combined with aggressive data augmentation — random grid rotations, reflections, and color permutations — this yields ~50,000 training tasks from the original 120, making rapid iteration practical on a single GPU.

## Details of the approach

### Data augmentation

Each of the 149 MINI-ARC tasks is augmented using all 8 symmetries of the square (rotations and reflections) combined with random color permutations, producing ~50,000 tasks in total. Tasks are split 80/20 into train (120 tasks) and test (29 tasks) sets at the task level before augmenting, so the test set contains entirely unseen task types.

### Shared iterative denoising approach

Nearly all modules share the same core inference procedure, inspired by masked diffusion. The output grid is initially fully masked (all 25 tokens set to a special mask value). At each of 25 iterations, the model takes the input grid, the current (partially unmasked) output grid, and the task embedding as inputs, then predicts a probability distribution over all output positions. The mask token at the most confident position is replaced with the predicted token, and the process repeats until the output is fully revealed. What differs between modules is how the **task embedding** is represented and regularized during training. In some approaches (e.g. JEPA) I also experimented with a GPT-style autoregressive decoder — predicting output tokens sequentially left-to-right rather than via iterative denoising — and observed similar accuracy plateaus, further suggesting the bottleneck is not the decoding strategy.

### Module overview

| Module | Approach | Key idea |
|---|---|---|
| `mini_arc_eqm/` | Equilibrium Matching | Basic iterative denoise task embedding and input/output grids similar to equilibrium matching and diffusion|
| `mini_arc_vae/` | VAE | Iteratively denoise but with a latent VAE bottleneck|
| `mini_arc_jepa/` | JEPA | A LeJEPA-style encoder builds an isotropic gaussian embedding of each task and decoder takes the embedding and an input to produce an output grid  |
| `mini_arc_rq/` | Random Projection Quantizer | Task token projected into a frozen random codebook |
| `mini_arc_fsq/` | Finite Scalar Quantization | Task token quantized to a discrete bounded integer lattice |
| `mini_arc_2vec/` | wave2vec 2.0 style | Gumbel softmax over quantized task token categories |

Each module contains:
- `nb_*.py` — the main training script (designed to run locally or in a Colab notebook)
- `create_tensor_board_events.py` — utilities for visualizing training runs in TensorBoard
- `results/` — saved checkpoints and training logs

### Evaluation

The primary metric is **task accuracy**: the fraction of test tasks for which the model predicts the correct output grid for the held-out test example (exact match).

## Learnings

Despite the variety of regularization techniques, **all approaches plateau at approximately 25% task accuracy** on the MINI-ARC test set with ~5M parameter transformer encoder models. This plateau is remarkably consistent across VAE, quantization (FSQ, RQ, wave2vec 2.0 style gumbel softmax), iterative denoising (equilibrium matching) and LeJEPA, suggesting that the bottleneck is not the structure of the task embedding space but something more fundamental.

This convergence of results has convinced me that **pure neural network approaches are insufficient** for robust ARC generalization, and that further progress requires additional methods (e.g. neuro-symbolic).

## Future directions

These findings motivate me to explore hybrid neuro-symbolic approaches. In particular, I am interested in combining learned neural representations (like the task embeddings here) with structured program search. Specifically, I am inspired by:

- **Search-based RL** algorithms such as AlphaGo/AlphaZero and [EfficientZero](https://arxiv.org/abs/2111.00210) applied to program synthesis — using a world model to plan over a DSL action space.
- **clingoARC** (my  repository): encoding ARC program search as first-order logic metaprogramming and solving with Answer Set Programming. The grounding bottleneck encountered there motivates using neural heuristics to drastically prune the search space.

```
