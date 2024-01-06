# Propagate & Disill (P&D)

Implementation of the paper "Propagate & Distill: Towards Effective Graph Learners Using Propagation-Embracing MLPs" (LoG 2023) [(link)](https://arxiv.org/abs/2311.17781).

Tested in the following versions:
- Pytorch: 1.12.0
- Pytorch Geometric: 2.2.0
- Deep Graph Library: 0.9.1
- Numpy: 1.21.2

Main files:
- `train_teacher.py`: Train teacher GNNs
- `train_student.py`: Train student MLPs
  - Set `--prop_iteration` to a non-zero value for P&D
  - Include `--fix_train` flag for P&D_fix

Other notes:
- The majority of the codebase is from the implementation of GLNN (Zhang et al., "Graph-less Neural Networks: Teaching Old MLPs New Tricks via Distillation", ICLR 2022)!
- The data files need to be downloaded from the GLNN repo [(link)](https://github.com/snap-research/graphless-neural-networks).
