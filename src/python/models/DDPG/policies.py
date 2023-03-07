import numpy as np
import tensorflow as tf
import random

### POLITIQUE POUR UNE EXPLORATION BROWNIENNE


def policy_noise(state, noise_object, model, lower_bound, upper_bound):
    #print('??????????????????')
    #print(model(state))
    #print('??????????????????')
    sampled_actions = tf.squeeze(model(state))
    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return [np.squeeze(legal_action)]


#### POLITIQUE EPSILON GREEDY ( POSE PROBLEME POUR L'INSTANT, NOUS N'ARRIVONS PAS À RESOUDRE LES ERREURS)


def policy_epsilon_greedy(state, epsilon, model):
    number = random.random()
    if number < epsilon:
        #print("EPSILON")
        #print(state)
        action1 = random.uniform(-1., 1.)
        action_finale = tf.constant([[action1]], tf.float32)
        return [np.squeeze(action_finale.numpy())]
    else:
        #print("Normal")
        #print(state)
        #print(model(state))
        action2 = tf.squeeze(model(state))
        #print('ACTION2')
        #print(action2)
        return [np.squeeze(action2)]
