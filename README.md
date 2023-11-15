# EMG-Control Dynamical System Learning 
Regression of Dynamical Systems (DS) with EMG control input.

## Changelog
- Introduced `torchdiffeq`, PyTorch-based ODE solver offering various integrators and better parallelization for multi-trajectory sampling
- Moved to fully differentiable PyTorch-based dynamics for GPU parallelization support and compatibility with `torchdiffeq`
- High performance generation of sliding window dataset via PyTorch `unfold` API
- Added first-order dynamics, **Spiral** and **Lorenz**, for testing on easier case
- Temporary removed **LSTM** model and substituted it with simpler **RNN** model for understanding better the mechanism
- Separated workflow in three scripts: `simulate.py` generates the dataset and saves it under `data/` in efficient numpy format preserving trajectory separation; `train.py` generates sliding window dataset and train the model that is saved under `models/` folder; `test.py` loads the trained model and test against synthetically generated trajectories
- All parameters handled via configuration files under `configs`
- All figures saved under `media/`

## ToDo
- define how to add controllers to dynamics without breaking `torchdiffeq`
- make rnn regression work for the very the Spiral dynamics (in progress)
- implement **NODE** and **Transformer** models
- test `proj_size` option for **LSTM**
- add initial padding to the training set

## Useful links
- PyTorch ODE solver: https://github.com/rtqichen/torchdiffeq/tree/master
- NeuralODE example: https://github.com/rtqichen/torchdiffeq/blob/master/examples/ode_demo.py
- DS learning via vector fields: https://towardsdatascience.com/deep-learning-for-dynamics-the-intuitions-20a67942dfbc
- control input variables?: https://github.com/rtqichen/torchdiffeq/issues/128
- Transformers paper: https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html (https://arxiv.org/abs/1706.03762)
- PyTorch RNN doc: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
- PyTorch sliding window data: https://stackoverflow.com/questions/60157188/how-can-i-resize-a-pytorch-tensor-with-a-sliding-window
- PyTorch LSTM doc: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html (https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)
- PyTorch Transformers doc: https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
- Interesting paper to check: https://arxiv.org/abs/2010.03957
- DS learning via LSTM: https://towardsdatascience.com/a-long-short-term-memory-network-to-model-nonlinear-dynamic-systems-72f703885818
- Nice explanation of LSTM: https://colah.github.io/posts/2015-08-Understanding-LSTMs/

## Dynamics
- Spiral
  - First Order
  - 2D
  - Autonomous
- Lorenz
  - First Order
  - 3D
  - Autonomous
  - Chaotic
- Pendulum
  - Second Order
  - 1D
  - Autonomous / Controlled
- Spherical Pendulum
  - Second Order
  - 2D
  - Autonomous / Controlled

## Examples
Generate the training trajectories
```sh
python3 examples/simulate.py
```

Train model
```sh
python3 examples/train.py
```

Test model
```sh
python3 examples/test.py
```

## To Check
- https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/
- https://cnvrg.io/pytorch-lstm/
- https://machinelearningmastery.com/an-introduction-to-recurrent-neural-networks-and-the-math-that-powers-them/