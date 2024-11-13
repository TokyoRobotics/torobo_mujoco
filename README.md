# torobo_mujoco

## Installation

### Install MuJoCo
Python 3.8 or higher is required.

For more information, please refer to https://github.com/google-deepmind/mujoco .

```
pip install mujoco
```

### Install torobo_mujoco

```
git clone https://github.com/TokyoRobotics/torobo_mujoco.git
cd torobo_mujoco
```

## Test
```
cd example
python example_torobo2.py
```

<img src="./doc/torobo2.png" width="600">

Move joints by dragging in Control tab.

<img src="./doc/torobo2_move_leftarm.gif" width="600">

## Play with the trained policy in Isaac Sim (Reinforcement Learning Sim2Sim)

### Install related python packages

```
pip install -r example_rl_play/requirements.txt
```

### Bipedal walk
※ leg_v1 model is now under research and development and is not currently scheduled for sale.

```
cd example_rl_play
python bipedal_walk.py
```

<img src="./doc/leg_v1_bipedal_walk.gif" width="600">