# Federated Policy Gradient with Byzantine Resilience (FedPG-BR)
This is the code for the FedPG-BR framework  presented in the paper: 

Flint Xiaofeng Fan, Yining Ma, Zhongxiang Dai, Wei Jing, Cheston Tan and Kian Hsiang Low. "[Fault-Tolerant Federated Reinforcement Learning with Theoretical Guarantee](https://arxiv.org/pdf/2110.14074.pdf)." *In 35th Conference on Neural Information Processing Systems (NeurIPS-21)*, Dec 6-14, 2021.


The experiments in the paper was conducted on Ubuntu 18.04 with a 14 cores (28 threads) Intel(R) Core(TM) i9-10940X CPU @ 3.30GHz and 64G memory.

The code following this instruction is also tested on Ubuntu 20.04 with ... note, for his setup, due to the CPU, you may run into the following issue like we had.. we solved this issue by following...


Ubuntu 20.04

# Dependencies
- Python 3.7 
- Pytorch 1.5.0 
- numpy 
- tensorboard 
- tqdm 
- sklearn 
- matplotlib 
- OpenAI Gym 
- Box2d [for running experiments on LunarLander environment] 
- mujoco150
- mujoco-py 1.50.1.68 [for running experiments on HalfCheetah environment]



# Installation 


```
$ conda create -n FedPG-BR pytorch=1.5.0

$ conda activate FedPG-BR

```


Please then follow the instructions [here](https://github.com/openai/mujoco-py) to setup mujoco and install mujoco-py. Please download the legacy version of mujoco (mujoco150) as we did not test using the latest version of mujoco.


Once you have ensured that mujoco_py has been successfully installed, proceed with

```

$ pip install -r requirements.txt

```

If you run into issue of `Intel MKL FATAL ERROR: Cannot load libmkl_avx512.so or libmkl_def.so`, then follow this [solution](https://stackoverflow.com/questions/36659453/intel-mkl-fatal-error-cannot-load-libmkl-avx2-so-or-libmkl-def-so). It will reinstall certain packages for certain intel chips and also removes `pytorch` and `mujoco-py.` So you will want to reinstall `pytorch (1.5.0)` and `mujoco-py (1.50.1.68)`


---

To check your installation, run
```
$ python run.py --env_name HalfCheetah-v2 --FedPG-BR --num_worker 10 --num_Byzantine 0 --log_dir ./logs_HalfCheetah --multiple_run 10 --run_name HalfCheetah_FedPGBR_W10B0

```

If terminal returns messages similar to those shown below, then your installation is all good.

![log](training-log-sample.png)


# Usage
To reproduce the results of FedPG-BR (K= 10) in Figure 1 for the HalfCheetah task, run the following command:
```
$ python run.py --env_name HalfCheetah-v2 --FedPG-BR --num_worker 10 --num_Byzantine 0 --log_dir ./logs_HalfCheetah --multiple_run 10 --run_name HalfCheetah_FedPGBR_W10B0

```
If terminal returns messages similar to those shown below, then your installation is all good.

![log](training-log-sample.png)


To reproduce the results of FedPG-BR (K= 10B= 3) in Figure 2 where 3 Byzantine agents are Random Noise in theHalfCheetah task environment, run the following command:

# Visualization
XXX


# Examples
XXX

# Acknowledgements
XXX



Errors:
Intel MKL FATAL ERROR: Cannot load libmkl_avx512.so or libmkl_def.so.

https://stackoverflow.com/questions/36659453/intel-mkl-fatal-error-cannot-load-libmkl-avx2-so-or-libmkl-def-so