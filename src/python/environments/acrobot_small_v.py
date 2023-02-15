import numpy as np
from numpy import cos, pi, sin

from gymnasium import spaces
from gymnasium.envs.classic_control.acrobot import AcrobotEnv, bound, rk4, wrap

import pygame
from pygame import gfxdraw

from typing import Optional


class AcrobotSmallV(AcrobotEnv):

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__(render_mode)
        high = np.array([pi, pi, self.MAX_VEL_1, self.MAX_VEL_2],
                        dtype=np.float32)
        self.observation_space = spaces.Box(low=-high,
                                            high=high,
                                            dtype=np.float32)

    def _get_ob(self):
        # state is array of size 4: [theta1, theta2, omega1, omega2]
        s = self.state
        assert s is not None, "Call reset before using this environment"
        return s.astype(np.float32)

    def _terminal(self):
        s = self.state
        assert s is not None, "Call reset before using AcrobotEnv object."
        return bool(-cos(s[0]) - cos(s[1] + s[0]) > 1.0) and np.abs(
            s[2]) < 1
