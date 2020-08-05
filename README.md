# DARTS: Differentiable Architecture Search

This repository is a non-official implementation.

# Introduction of DARTS
https://speakerdeck.com/udonda/darts-differentiable-architecture-search

# Usage
1. Run `bash scripts/1_search.sh`
2. Set the output of training script to `genotype` parameters in `scripts/2_augment.sh`
2. Run `bash scripts/2_augment.sh`

# Results
| CIFAR-10 | Final validation acc | Best validation acc |
| ------- | -------------------- | ------------------- |
| [Original](https://github.com/quark0/darts)         | 97.17% | 97.23% |
| Ours | 96.39% | 96.87% |

# Causion
+ My implementation can not handle multiple-gpu training.

# Acknowledgements
```
@article{liu2018darts,
  title={DARTS: Differentiable Architecture Search},
  author={Liu, Hanxiao and Simonyan, Karen and Yang, Yiming},
  journal={arXiv preprint arXiv:1806.09055},
  year={2018}
}
```