# DARTS: Differentiable Architecture Search

This repository is a non-official implementation.

# Usage
1. Run `bash scripts/1_search.sh`
2. Set the output of training script to `genotype` parameters in `scripts/2_augment.sh`
2. Run `bash scripts/2_augment.sh`


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