### Control drone in gym-pybullet-drones using ppo
Hovering a quacopter with some predefined position using [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones/) env with PPO algorithm from [PPO-PyTorch](https://github.com/nikhilbarhate99/PPO-PyTorch)
#### 30/12/2023 Update training result
#### 28/12/2023 Init commit
* Change reward function, compute terminate

##### Hover at (0, 0, 1) position

![alt text](https://github.com/phuongboi/drone-control-using-reinforcement-learning/blob/main/results/202312301540.gif)
##### Hover at (0, 1, 1) position

![alt text](https://github.com/phuongboi/drone-control-using-reinforcement-learning/blob/main/results/202312301513.gif)

#### How to use
* Follow author's guide to install gym-pybullet-drones environment
* Training `python train_ppo.py`
* Test pretrained model ` python test_ppo.py`

### References
* https://github.com/utiasDSL/gym-pybullet-drones/
* https://github.com/nikhilbarhate99/PPO-PyTorch
* Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
