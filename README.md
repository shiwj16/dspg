# DSPG
# Soft Policy Gradient Methods for Maximum Entropy Deep Reinforcement Learning

Tensorflow implementation of Deterministic Soft Policy Gradients (DSPG).

Method is tested on [MuJoCo](http://www.mujoco.org/) continuous control tasks in [OpenAI gym](https://github.com/openai/gym). 
Networks are trained using [Tensorflow 1.6.0](https://github.com/tensorflow/tensorflow) and Python 3.6. 

### Usage
Partial results of this paper can be reproduced exactly by running:
```
./run_Walker2d.sh
```

Hyper-parameters can be modified with different arguments to main.py.
