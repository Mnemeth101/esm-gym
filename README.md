# ESMGym

The goal of this repo is to set up metrics and strategies to evaluate how to optimally sample ESM3 for protein design.

### Objectives
- [x] Choose proteins for testing - CASP15, single domains only. T1137 thrown out since they're really long and only make sense in oligomeric context.
- [ ] Set up metrics/eval methods
- [ ] Set up sampling methods


### Metrics
1. Foldability: RMSD of (ESMFold predicted structure) --8x-->  ESM-IF designed sequence --> (ESMFolded Structure)
2. pLDDT (ESM3)
4. pAE (ESM3)
3. pTM (ESMFold)
6. (Sequence) Entropy
7. (Sequence) Diversity



### Important Sources
1. [ESM: Evolutionary Scale Modeling](https://github.com/facebookresearch/esm)
2. [MeMDLM: De Novo Membrane Protein Design with Masked Discrete Diffusion Protein Language Models](http://arxiv.org/abs/2410.16735)
3. [Path Planning for Masked Diffusion Model Sampling](http://arxiv.org/abs/2502.03540)
4. [Simulating 500 Million Years of Evolution with a Language Model](https://doi.org/10.1126/science.ads0018)
5. [Think While You Generate: Discrete Diffusion with Planned Denoising](https://doi.org/10.48550/arXiv.2410.06264)
6. [Unlocking the Capabilities of Masked Generative Models for Image Synthesis via Self-Guidance](https://proceedings.neurips.cc/paper_files/paper/2024/file/ecd92623ac899357312aaa8915853699-Paper-Conference.pdf)
7. [script for decoding using ESM3](https://github.com/evolutionaryscale/esm/blob/main/cookbook/snippets/esm3.py#L43)
8. [script for guided generation with ESM3](https://github.com/evolutionaryscale/esm/blob/main/cookbook/tutorials/5_guided_generation.ipynb)