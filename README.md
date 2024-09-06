### Control drone in gym-pybullet-drones using ppo
Hovering a quacopter with some predefined position using [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones/) env with PPO algorithm from [PPO-PyTorch](https://github.com/nikhilbarhate99/PPO-PyTorch)
#### 06/09/2024 Update fly through the gate
* I test FlyThruGateAvitary environment with PPO with some modify in reward function. I created a gate model with figuro.io to and add pybullet.
* To train: `python train_thrugate.py`, to test: `python test_thrugate.py`
#### 20/02/2024 Note about ppo implementation
* Recently, I figure out the frustration of drone at hover position may come from fixed `action_std` of this PPO implementation, they setting `action_std_init = 0.6` and decay this value during training time. ~~In inference mode, there is no mechanism to reduce or remove this variance, so control output this vary all the time.~~ I look at some other implementation of Soft Actor Critic, they use one more layer to learn action std beside action mean.
#### 13/01/2024 Update hovering with some constrains
* Add some [contrains](https://github.com/phuongboi/drone-control-using-reinforcement-learning/blob/da52ed17e0bc1923a1f0eb7d7d2cecdf01aec4f9/gym_pybullet_drones/envs/HoverAviary.py#L88) to naive reward, drone look more stable at hover position, reference from [paper](https://web.stanford.edu/class/aa228/reports/2019/final62.pdf)
#### 30/12/2023 Update training result
#### 28/12/2023 Init commit
* Change reward function, compute terminate
##### Fly through the gate

![alt text](https://github.com/phuongboi/drone-control-using-reinforcement-learning/blob/main/results/fly_gate.gif)

##### Hover at (0, 0, 1) position

![alt text](https://github.com/phuongboi/drone-control-using-reinforcement-learning/blob/main/results/202312301540.gif)
##### Hover at (0, 1, 1) position

![alt text](https://github.com/phuongboi/drone-control-using-reinforcement-learning/blob/main/results/202312301513.gif)

#### How to use
* Follow author's guide to install gym-pybullet-drones environment
* Training `python train_hover.py`
* Test pretrained model ` python test_hover.py`

### References
* https://github.com/utiasDSL/gym-pybullet-drones/
* https://github.com/nikhilbarhate99/PPO-PyTorch
* Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
* https://web.stanford.edu/class/aa228/reports/2019/final62.pdf
