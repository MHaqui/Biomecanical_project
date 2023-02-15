import numpy as np
from numpy import cos, pi, sin

from gymnasium import spaces
from gymnasium.envs.classic_control.acrobot import AcrobotEnv, bound, rk4, wrap

import pygame
from pygame import gfxdraw

from typing import Optional


class AcrobotContinuous(AcrobotEnv):

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__(render_mode)
        high = np.array([pi, pi, self.MAX_VEL_1, self.MAX_VEL_2],
                        dtype=np.float32)
        self.observation_space = spaces.Box(low=-high,
                                            high=high,
                                            dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, dtype=np.float32)

    def _get_ob(self):
        # state is array of size 4: [theta1, theta2, omega1, omega2]
        s = self.state
        assert s is not None, "Call reset before using this environment"
        return s.astype(np.float32)

    def step(self, action):
        s = self._get_ob()
        torque = action

        # Add noise to the force action
        if self.torque_noise_max > 0:
            torque += self.np_random.uniform(-self.torque_noise_max,
                                             self.torque_noise_max)

        # Now, augment the state with our force action so it can be passed to
        # _dsdt
        s_augmented = np.append(s, torque)

        ns = rk4(self._dsdt, s_augmented, [0, self.dt])

        ns[0] = wrap(ns[0], -pi, pi)
        ns[1] = wrap(ns[1], -pi, pi)
        ns[2] = bound(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[3] = bound(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
        self.state = ns
        terminated = self._terminal()
        reward = -1.0 if not terminated else 0.0

        if self.render_mode == "human":
            self.render()
        return (self._get_ob(), reward, terminated, False, {})
