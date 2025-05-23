# BenchNPIN: Benchmarking Non-prehensile Interactive Navigation
BenchNPIN is a comprehensive suite of benchmarking tools for mobile robot non-prehensile interactive navigation. The goal of BenchNPIN is to provide researchers a standarized platform for training and evaluating algorithms in non-prehensile interactive navigation. BenchNPIN provides simulated environments for a wide range of non-prehensile interactive navigation tasks, metrics specifically capturing both task efficiency and interaction effort, policy templates with reference implementations. 

Checkout our paper "[Bench-NPIN: Benchmarking Non-prehensile Interactive Navigation](https://arxiv.org/abs/2505.12084)", currently under review for IROS 2025. [[Project Page](https://sites.google.com/view/bench-npin/home)]


## The Environments

### Navigating Maze with Movable Obstacles

```python
env = gym.make('maze-NAMO-v0')
```

This environment features a static maze structure with randomly initialized obstacles. The robot's task is to navigate from a starting position to a goal location while minimizing path length and obstacle collisions. 


<p align="center">
    <img src="./media/maze-demo.gif"><br/>
    <em>The Maze environment.</em>
</p>


### Autonomous Ship Navigation in Icy Waters

```python
env = gym.make('ship-ice-v0')
```

In this task, an autonomous surface vehicle must reach a horizontal goal line ahead while minimizing collisions with broken ice floes in the channel. 

<p align="center">
    <img src="./media/ship-ice-demo.gif"><br/>
    <em>The Ship-Ice environment.</em>
</p>


### Box Delivery

```python
env = gym.make('box-pushing-v0')
```

The _Box-Delivery_ environment consists of a set of movable boxes to be delivered to a designated _receptacle_. The robot is tasked to delivery all boxes using its front bumper.

<p align="center">
    <img src="./media/box-delivery-demo.gif"><br/>
    <em>The Box-Delivery environment.</em>
</p>


### Area Clearing

```python
env = gym.make('area-clearing-v0')
```

This envronment consists of a set of movable boxes and a _clearance_ area. The task of the robot is to remove all boxes from this _clearance_ area. 


<p align="center">
    <img src="./media/area-clearing-demo.gif"><br/>
    <em>The Area-Clearing environment.</em>
</p>


## Installation

### Install from `pip`

```bash
pip install benchnpin
```

The `pip install` above is sufficient to run _Ship-Ice_ and _Maze_ environments. To run _Box-Delivery_ and _Area-Clearing_, please install shortest path module as follows

```bash
git clone https://github.com/IvanIZ/spfa.git
cd spfa
pip install -e .
```


### Build from source

1. Clone the project
```bash
git clone https://github.com/IvanIZ/BenchNPIN.git
```

2. Clone the submodules (spfa and GLNS)
```bash
cd BenchNPIN
git submodule update --init --recursive
```

3. Install dependencies.
```bash
pip install -r requirements.txt
```

4. Install Gym environment
```bash
pip install -e .
```

5. Install shortest path module
```bash
cd deps/spfa
pip install -e .
```

## Usage


### Running an interactive navigation environment

```python
import benchnpin.environments
import gymnasium as gym

env = gym.make('ship-ice-v0')
observation, info = env.reset()

terminated = truncated = False
while not (terminated or truncated):

    action = your_policy(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
```

To configure the parameters for each environment, please refer to the configuration examples for [Maze](./scripts/configure_maze.py),  [Ship-Ice](./scripts/configure_ship_ice.py), [Box-Delivery](./scripts/configure_box_delivery.py), and [Area-Clearing](./scripts/configure_area_clearing.py).


### Creating a custom policy from the policy template
```python
from benchnpin.baselines.base_class import BasePolicy

class CustomPolicy(BasePolicy):

    def __init__(self) -> None:
        super().__init__()

        # initialize costum policy here
        ...

    
    def train(self):
        # train the custom policy here, if needed
        ...


    def act(self, observation, **kwargs):
        # define how custom policy acts in the environment
        ...

    
    def evaluate(self, num_eps: int, model_eps: str ='latest'):
        # define how custom policy is evaluated here
        ...
```


### Running benchmarks on policies
```python
from benchnpin.common.metrics.base_metric import BaseMetric
import CustomPolicy1      # some custom policies
import CustomPolicy2
import CustomPolicy3

# initialize policies to be evaluated
policy1 = CustomPolicy1()
policy2 = CustomPolicy2()
policy3 = CustomPolicy3()

# run evaluations
num_eps = 200    # number of episodes to evaluate each policy
benchmark_results = []
benchmark_results.append(policy1.evaluate(num_eps=num_eps))
benchmark_results.append(policy2.evaluate(num_eps=num_eps))
benchmark_results.append(policy3.evaluate(num_eps=num_eps))

# plot efficiency and effort scores
BaseMetric.plot_algs_scores(benchmark_results, save_fig_dir='./')
```


## Implemented Baselines


| **Tasks**                | **Baselines** |
| --------------------------- | ----------------------|
| _Maze_ | SAC<sup>[1](#f1)</sup>, PPO<sup>[1](#f1)</sup> |
| _Ship-Ice_               | SAC<sup>[1](#f1)</sup>, PPO<sup>[1](#f1)</sup>, SAV Planning<sup>[2](#f1), [3](#f1)</sup> |
| _Box-Delivery_         | SAC<sup>[1](#f1)</sup>, PPO<sup>[1](#f1)</sup>, SAM<sup>[4](#f1)</sup> |
| _Area_Clearing_             | SAC<sup>[1](#f1)</sup>, PPO<sup>[1](#f1)</sup>, SAM<sup>[4](#f1)</sup>, GTSP<sup>[5](#f1)</sup> |

<b id="f1">1</b>: Reinforcement Learning policies Integrated with [Stable Baselines 3](https://stable-baselines3.readthedocs.io/en/master/).

<b id="f1">2</b>: Planning-based policy using an [ASV ice navigation lattice planner](https://ieeexplore.ieee.org/abstract/document/10161044).

<b id="f1">3</b>: Planning-based policy using a [predictive ASV ice navigation planner](https://arxiv.org/abs/2409.11326).

<b id="f1">4</b>: [Spatial Action Maps](https://www.roboticsproceedings.org/rss16/p035.pdf) policy.

<b id="f1">5</b>: A Generalized Traveling Salesman Problem (GTSP) policy. Please see Appendix I of our paper for details. 

### Trained Models

You may download the our trained model weights from [here](https://drive.google.com/drive/folders/1jBeFHgArBXuH7eQCzlNSVhZjhJIFlQVY?usp=sharing).

### GTSP policy requirements

To run the GTSP policy, we use a fork of the [GLNS](https://github.com/stephenlsmith/GLNS.jl) solver. To run this solver, the [Julia](https://julialang.org/) programming language must be installed. 

The path to the GLNS solver can be configured as a parameter. Please refer to the cofiguration example for [Area-Clearing](./scripts/configure_area_clearing.py).