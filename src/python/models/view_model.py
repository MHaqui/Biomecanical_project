import sys
from pathlib import Path

sys.path.append(Path(sys.path[0]).parent.as_posix())

from environments.custom_acrobot import CustomAcrobot
import numpy as np
import matplotlib.pyplot as plt
# from CACLA import CACLA

# env = CustomAcrobot(render_mode="human", simplified_state=True, actions_array=[-1. - .1, 0, .1, 1], reward_func=lambda x, y: np.linalg.norm(x))
env = CustomAcrobot(render_mode="human",
                    continuous_actions=True,
                    continuous_actions_boundary=.5,
                    terminal_func=lambda state: np.abs(state[0]) > 1)
options = {"low": -np.pi / 2, "high": np.pi / 2}
observation, info = env.reset(options=options)

# model = CACLA(env.observation_space.shape,
#               [1 / np.pi, 1 / np.pi, 1 / env.MAX_VEL_1, 1 / env.MAX_VEL_2], 12)
# model.load_weights('CACLA_weights')

# def policy(state):
#     return model.predict(state[np.newaxis], verbose=0)[0][-1][-1]

actions = []
rewards = []

for _ in range(300):
    action = env.action_space.sample()
    actions.append(action)

    observation, reward, terminated, truncated, info = env.step(action)

    rewards.append(reward)

    if terminated or truncated:
        observation, info = env.reset(options=options)

env.close()

fig, axs = plt.subplots(1, 2)
axs[0].plot(actions)
axs[1].plot(rewards)
axs[0].set_title('Actions')
axs[1].set_title('Rewards')
fig.savefig('plot.png')
