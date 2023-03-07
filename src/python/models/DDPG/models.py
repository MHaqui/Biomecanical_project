#########                                       ############
#########   CRÉATION DE L'ACTEUR ET DU CRITIQUE ############
#########                                       ############
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input


def get_actor(num_states: int) -> tf.keras.Model:
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = Input(shape=(num_states,))
    out = Dense(256, activation="relu")(inputs)
    out = Dense(256, activation="relu")(out)
    outputs = Dense(1, activation="tanh", kernel_initializer=last_init)(out)

    # Our upper bound is 2.0 for Pendulum and 1.0 for Acrobot
    outputs = outputs
    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic(num_states: int, num_actions: int) -> tf.keras.Model:
    # State as input
    state_input = Input(shape=(num_states))
    state_out = Dense(16, activation="relu")(state_input)
    state_out = Dense(32, activation="relu")(state_out)

    # Action as input
    action_input = Input(shape=(num_actions))
    action_out = Dense(32, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = tf.keras.layers.Concatenate()([state_out, action_out])

    out = Dense(256, activation="relu")(concat)
    out = Dense(256, activation="relu")(out)
    outputs = Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))
