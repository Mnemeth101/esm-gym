# ESMGym

The goal of this repo is to set up metrics and strategies to evaluate how to optimally sample ESM3 for protein design.

### Objectives
- [x] Choose proteins for testing - CASP15, 27 single domains only. T1137 thrown out since they're really long and only make sense in oligomeric context.
- [ ] Set up metrics/eval methods
- [ ] Set up sampling methods


### Metrics
1. Designability: RMSD of (ESMFold predicted structure) --8x-->  ESM-IF designed sequence --> (ESMFolded Structure) - run the ESMFolds on GCP ❌
2. pLDDT (ESM3) ✅
3. pTM (ESMFold) ✅
4. Foldability: % structures generated from a single sequence that achieve (pLDDT > 80 & pTM > 0.7) ✅
6. (Sequence) Entropy ✅
7. (Sequence) Diversity ✅
8. Computational cost of sampling ✅
9. Cosine similarity to ESM-C embedding of original sequence ✅
10. Unmask aggregated categorical cross entropy (instead of pseudo-likelihood which takes O(L) runs)) ✅
11. (?) Difference in function track tokens/embedding

### Algorithmic Strategies ()
1. (Baseline) Structure given, one-shot sequence.
2. (Baseline) Structure given, in t steps choose L/t residues to denoise. Choose lowest entropy.
3. (Baseline) Structure given, in t steps choose L/t residues to denoise. Choose highest probability difference. (lowest confusion)
4. Reward-guided diffusion using similarity score to ESM-C embedding as score, with high sequence temperature. Starting with structure
5. Reward-guided diffusion using similarity score to ESM-C embedding as score, with high sequence temperature. Fully denoising without structure input.
6. (inspired by Link 9) t steps/budget. n = L/t. Iterate: denoise n*2 lowest entropy residues (can choose from top 3 residues at each position)x    . Then noise n/2 least confidence positions.
7. (? This one is important and I think can be the novelty here. The idea is the similar to that emphasized in Uehara et al. 2025 When you try to predict the soft reward value function at smaller t (less noisy), it becomes more accurate. There can be an exploration-exploitation situation in terms of % of the sequence that is masked. For example, we do more exploration in embedding space when we keep 80% of the sequence masked than if we keep 20% of the seqeuence masked. We can frame this as a simulated annealing task, the temperature determines % of the sequence that we unmask in the next decoding step. While diffusion models have noise schedulers, ESM3 doesn't inherently have one, so exploring these are relevant for understanding how to optimize the generative process using ESM3.) Simulated annealing. 


### Important Sources
1. [ESM: Evolutionary Scale Modeling](https://github.com/facebookresearch/esm)
2. [MeMDLM: De Novo Membrane Protein Design with Masked Discrete Diffusion Protein Language Models](http://arxiv.org/abs/2410.16735)
3. [Path Planning for Masked Diffusion Model Sampling](http://arxiv.org/abs/2502.03540)
4. [Simulating 500 Million Years of Evolution with a Language Model](https://doi.org/10.1126/science.ads0018)
5. [Think While You Generate: Discrete Diffusion with Planned Denoising](https://doi.org/10.48550/arXiv.2410.06264)
6. [Unlocking the Capabilities of Masked Generative Models for Image Synthesis via Self-Guidance](https://proceedings.neurips.cc/paper_files/paper/2024/file/ecd92623ac899357312aaa8915853699-Paper-Conference.pdf)
7. [script for decoding using ESM3](https://github.com/evolutionaryscale/esm/blob/main/cookbook/snippets/esm3.py#L43)
8. [script for guided generation with ESM3](https://github.com/evolutionaryscale/esm/blob/main/cookbook/tutorials/5_guided_generation.ipynb)
9. [Reward-Guided Iterative Refinement in Diffusion Models at Test-Time with Applications to Protein and DNA Design.](https://doi.org/10.48550/ARXIV.2502.14944)