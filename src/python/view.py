from our_env import OurEnv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

env = OurEnv((-.5, -5), render_mode="human")
reset_args = {
    "seed": 42,
    "options": {
        "low": -np.pi / 2,
        "high": np.pi / 2
    },
}
observation, info = env.reset(**reset_args)

input_shape = env.observation_space.shape
n_outputs = env.action_space.n
model = tf.keras.models.load_model('models/model1.h5')
print(model.summary())


def policy(state):
    pred = model.predict(state[np.newaxis], verbose=0)[0]
    print(pred)
    return np.argmax(pred)


actions = []
rewards = []

for _ in range(300):
    action = policy(observation)
    actions.append(action)

    observation, reward, terminated, truncated, info = env.step(action)

    rewards.append(reward)

    if terminated or truncated:
        observation, info = env.reset(**reset_args)

env.close()

fig, axs = plt.subplots(1, 2)
axs[0].plot(np.arange(len(rewards)), rewards)
axs[1].scatter(np.arange(len(actions)), actions)
fig.savefig('plot.png')
