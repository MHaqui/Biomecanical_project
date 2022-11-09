import gym
from our_env import OurEnv
from pprint import pprint

env = OurEnv((-.5, -5), render_mode="human")
env.action_space.seed(42)

observation, info = env.reset(seed=42)
q_table = {}

for _ in range(300):
    s = env.state
    a = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(a)

    q_table[(tuple(s), a)] = (env.state, reward)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
