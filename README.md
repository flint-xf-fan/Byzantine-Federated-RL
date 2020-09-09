# Roadmap

- [x] Settings
- [ ] Formulation
  - [ ] what objective to optimize?
    - [ ] direct PG or Q? or AC-style?
  - [ ] what became invalid using martingale, for the case of RL objective?
    - [ ] convexity, lipschitzness?
    - [ ] IID samples?
    - [ ] what is the input of the objective function to be optimized?
- [ ] Algorithm
- [ ] Theoretical analysis
  - [ ] Sample complexity bound
  - [ ] Regret bound
- [ ] Experiments


# Reference paper

## Byzantine Problem
* [Byzantine SGD](https://arxiv.org/abs/1803.08917), NeurIPS2018: Distributed SGD optimization under Byzantine settings, for convex optimization
* [Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates](https://arxiv.org/pdf/1803.01498.pdf), ICML2018: Distributed GD, convex optimization.
* [Byzantine Resilient Non-Convex SVRG with Distributed Batch Gradient Computations](https://arxiv.org/pdf/1912.04531.pdf), NeurIPS2019 workshop: Byzantine SGD for non-convex optimization.

