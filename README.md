# CEMGD

Code for [CEM-GD: Cross-Entropy Method with Gradient Descent Planner for Model-Based Reinforcement Learning](https://arxiv.org/abs/2112.07746).

Implementation for the CEM-GD planner can be found in `mbrl/planning/gradient_optimizer.py`.

`pets-experiments.py` is a notebook for running the end-to-end MBRL algorithm based on PETS + CEM-GD and the experiments in the paper. 

Portions of this code, including the implementation of PETS, is adapted from [mbrl-lib](https://github.com/facebookresearch/mbrl-lib) under the MIT license. 
