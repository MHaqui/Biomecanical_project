import tensorflow as tf
import numpy as np
from typing import Iterable, Tuple


class CACLA(tf.keras.Model):

    def __init__(self, input_shape: Tuple[int], state_scale: Iterable[float],
                 num_hidden_units: int):
        super().__init__()

        self.common = tf.keras.models.Sequential([
            tf.keras.Input(shape=input_shape),
            RescalingAcrobot(state_scale, name='rescale_state'),
            tf.keras.layers.Dense(num_hidden_units,
                                  activation='sigmoid',
                                  name='hidden')
        ])
        self.actor = tf.keras.layers.Dense(1,
                                           activation='sigmoid',
                                           name='Action')
        self.critic = tf.keras.layers.Dense(1, name='Value')

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.common(inputs)
        return self.actor(x), self.critic(x)


class RescalingAcrobot(tf.keras.layers.Rescaling):

    def __init__(self, scales: np.ndarray, **kwargs):
        super().__init__(scales[0], **kwargs)
        self.scales = scales

    def call(self, inputs):
        dtype = self.compute_dtype
        scales = tf.cast(self.scales, dtype)
        offset = tf.cast(self.offset, dtype)
        return tf.cast(inputs, dtype) * scales + offset

    def get_config(self):
        config = {
            "scale": self.scale,
            "scales": self.scales,
            "offset": self.offset,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


def gaussian_exploration(predicted_action, stddev):
    return np.random.normal(predicted_action, stddev)


def step(env, predicted_action, stddev):
    action = gaussian_exploration(predicted_action, stddev)
    next_state, reward, done, *_ = env.step(action)
    return action, next_state, reward, done


def training_step(env, model, stddev, discount_factor, optimizer, loss_fn):
    state = env.state
    with tf.GradientTape() as tape:
        predicted_action, predicted_value = model(state[np.newaxis])
        target_action, next_state, reward, done = step(env, predicted_action,
                                                       stddev)
        next_value = model(next_state[np.newaxis])[1]
        target_value = reward + discount_factor * next_value
        TD_error = target_value - predicted_value
        if TD_error > 0:
            loss = loss_fn([(target_action, target_value)],
                           [(predicted_action, predicted_value)])
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return reward, done
