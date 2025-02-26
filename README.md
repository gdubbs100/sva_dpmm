# sva_dpmm
Implement and apply Stochastic Variational Approximation to DPMM estimation

Eventually I would like to use this for task inference and/or continual learning in a reinforcement learning context.
Project ideas may include:
- Use learnt DPMM to identify tasks and apply EWC or another method
- Learn a DPMM policy (why?)

## TODO:
[ ] Refactor `run` function to be separate from the `SVA` class
[ ] Enable user to specify base distribution(s) and cluster distributions in `SVA` class. 
- [ ] Specify which parameters to learn and which to hold constant (e.g. gaussian with fixed $\sigma$)

## Optional
[ ] Add tensorboard logging??
[ ] Train and test validation??
[ ] Create `DataStream` class or something like it
