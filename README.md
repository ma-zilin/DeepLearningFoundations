# Deep Learning Foundations

A learning portfolio for building the neural-network foundations needed for robot learning and embodied AI. The repository combines textbook exercises, small neural networks built for understanding, and a fully traceable two-dimensional diffusion experiment.

The emphasis is not on production-ready frameworks. Each implementation is kept small enough to connect mathematical ideas, tensor operations, code, and experimental evidence.

## Learning Path

The repository follows a progression from basic tensor operations to generative modeling and robot-learning concepts:

```text
deep learning fundamentals
→ neural networks from first principles
→ generative modeling with DDPM
→ conceptual bridge to Diffusion Policy
```

| Directory | Focus | Representative work |
| --- | --- | --- |
| [`d2l/`](d2l/) | Core deep learning with *Dive into Deep Learning* | Linear regression, multilayer perceptrons, CNNs, image classification, and Kaggle house-price regression |
| [`karpathy-nn/`](karpathy-nn/) | Neural networks from first principles | MLP and Makemore exercises for understanding backpropagation and autoregressive modeling |
| [`diffusion/`](diffusion/) | Generative modeling through a two-dimensional DDPM | Forward diffusion, timestep-conditioned noise prediction, reverse sampling, distribution evaluation, and ablation |

## Featured Study: A Two-Dimensional DDPM

The diffusion track builds a DDPM for a bimodal two-dimensional Gaussian distribution without relying on a high-level diffusion library. Its purpose is to make the complete data flow observable:

```text
bimodal samples
→ forward noising
→ timestep-conditioned noise prediction
→ iterative reverse sampling
→ quantitative distribution evaluation
```

The implementation covers:

- closed-form sampling at arbitrary diffusion timesteps;
- sinusoidal timestep embeddings and epsilon-prediction training;
- checkpointed training and iterative reverse generation;
- evaluation across three independent sampling seeds;
- a controlled ablation that removes timestep conditioning.

### Verified Results

The generated samples are evaluated using mode balance, mode-center error, vertical bias, and within-mode standard deviation. All three sampling seeds passed the predefined criteria.

| Experiment | Validation loss | Distribution evaluation |
| --- | ---: | --- |
| With timestep conditioning | 0.194 | Passed |
| Without timestep conditioning | 0.384 | Failed |

The ablation changes only whether the model uses its timestep features. It shows that timestep conditioning is necessary for this model to distinguish noise levels and recover the target distribution; the result is evidence for this experiment, not a general performance benchmark.

The DDPM variables and sampling process have also been mapped conceptually to action-sequence generation in Diffusion Policy. This repository does **not** contain a complete Diffusion Policy or robot-control implementation.

See the [gate-based diffusion learning plan](diffusion/LEARNING_PLAN.md) for the detailed questions, experiments, and acceptance criteria.

## What This Work Demonstrates

- connecting mathematical definitions to tensor shapes, data flow, and executable code;
- implementing the central DDPM mechanisms instead of treating diffusion as a black-box library;
- evaluating a generative model with distribution-level metrics rather than relying only on visual inspection;
- using a controlled ablation to test whether timestep conditioning provides meaningful information;
- transferring the DDPM data flow from two-dimensional samples to conditional action-sequence generation.

## Current Boundary

This is a personal learning portfolio rather than a reusable framework or production system. The current diffusion work covers an unconditional two-dimensional DDPM and a conceptual mapping to Diffusion Policy; it does not yet include conditional diffusion training, visuomotor observations, action execution, or robot deployment.

## References

- [Dive into Deep Learning — PyTorch edition](https://d2l.ai/)
- [Karpathy's Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)
