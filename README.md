### Control drone in gym-pybullet-drones using ppo
Hovering a quacopter with some predefined position using [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones/) env with PPO algorithm from [PPO-PyTorch](https://github.com/nikhilbarhate99/PPO-PyTorch)
#### 13/01/2023 Update hovering with some constrains
* Add some [contrains](https://github.com/phuongboi/drone-control-using-reinforcement-learning/blob/da52ed17e0bc1923a1f0eb7d7d2cecdf01aec4f9/gym_pybullet_drones/envs/HoverAviary.py#L88) to naive reward, drone look more stable at hover position, reference from [paper](https://web.stanford.edu/class/aa228/reports/2019/final62.pdf)
#### 30/12/2023 Update training result
#### 28/12/2023 Init commit
* Change reward function, compute terminate

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
