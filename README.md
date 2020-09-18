# Roadmap

- [x] Problem Settings
- [ ] Formulation
  - [ ] what objective to optimize?
    - [ ] direct PG or Q? or AC-style?
    - [ ] how about approximating the trajectory distribution using a simpler & easier-to-analysis distribution?
    - [ ] vanilla PG is known to have high variance (solved by introducing baselines). what happen to its variance in dsitributed setting?
  - [ ] what became invalid using martingale, for the case of RL objective?
    - [ ] convexity, lipschitzness?
    - [ ] IID samples?
  - [ ] RL settings
    - [ ] on-policy?
    - [ ] deterministic or stochastic?
- [ ] Algorithm
- [ ] Theoretical analysis
  - [ ] Sample complexity bound
  - [ ] Regret bound
- [ ] Experiments


# Reference paper

## Byzantine Problem (the papers are listed in recommended order)
* [Generalized Byzantine-tolerant SGD](https://arxiv.org/pdf/1802.10116.pdf), arxiv 2018: easier to follow. 
* [Byzantine SGD](https://arxiv.org/abs/1803.08917), NeurIPS2018: Distributed SGD optimization under Byzantine settings, for convex optimization
* [Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates](https://arxiv.org/pdf/1803.01498.pdf), ICML2018: Distributed GD, convex optimization.
* [Byzantine Resilient Non-Convex SVRG with Distributed Batch Gradient Computations](https://arxiv.org/pdf/1912.04531.pdf), NeurIPS2019 workshop: Byzantine SGD for non-convex optimization.



## RL settings
* [Policy gradient in Lipschitz Markov Decision Processes](https://link.springer.com/article/10.1007/s10994-015-5484-1), Springer2015: proof for that the objective (expected return) and its gradient are Lipschitz continuous w.r.t policy parameters, given the assumption of Lipschitz continuity of the transition, reward function and the policies.
* [SVRG for Policy Evaluation with Fewer Gradient Evaluations](https://www.ijcai.org/Proceedings/2020/0374.pdf), IJCAI2020

## Analysis of RL
* [Unifying PAC and Regret: Uniform PAC Bounds for Episodic Reinforcement Learning](https://arxiv.org/abs/1703.07710) NIPS2017
* [An Off-policy Policy Gradient Theorem Using Emphatic Weightings](https://arxiv.org/pdf/1811.09013.pdf), NeurIPS2018.
* [Private Reinforcement Learning with PAC and Regret Guarantees](https://proceedings.icml.cc/static/paper_files/icml/2020/2453-Paper.pdf), ICML2020
* [Provably Efficient Reinforcement Learning with Linear Function Approximation](https://arxiv.org/abs/1907.05388), COLT2020