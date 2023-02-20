import pickle
from pathlib import Path

import numpy as np
from numpy import cos, pi, sin

from gymnasium import spaces
from gymnasium.envs.classic_control.acrobot import AcrobotEnv, bound, rk4, wrap

from typing import Callable, Optional, Union
from numpy.typing import ArrayLike


def acrobot_reward_func(state: ArrayLike, terminated: bool) -> float:
    return -1.0 if not terminated else 0.0


def acrobot_terminal_func(state: ArrayLike) -> bool:
    return bool(-cos(state[0]) - cos(state[1] + state[0]) > 1.0)


class CustomAcrobot(AcrobotEnv):
    """Custom environment that allows for modifications on top of Acrobot
       environment.
    """

    def __init__(self,
                 render_mode: Optional[str] = None,
                 simplified_state: bool = False,
                 continuous_actions: bool = False,
                 continuous_actions_boundary: float = 1,
                 actions_array: Optional[ArrayLike] = None,
                 reward_func: Callable[[ArrayLike, bool],
                                       float] = acrobot_reward_func,
                 terminal_func: Callable[[ArrayLike],
                                         bool] = acrobot_terminal_func):
        """Custom Acrobot initializer.

        Args:
            render_mode: How to render environment steps. Defaults to None.
            simplified_state:
                If true, observation returned is:
                    [`theta1`,
                     `theta2`,
                     Angular velocity of `theta1`,
                     Angular velocity of `theta2`].
                Else, observation returned is:
                    [cos(`theta1`),
                     sin(`theta1`),
                     cos(`theta2`),
                     sin(`theta2`),
                     Angular velocity of `theta1`,
                     Angular velocity of `theta2`].
                Defaults to False.
            continuous_actions: If true, environment action space will be
                continuous between `-continuous_actions_boundary` and
                `continuous_actions_boundary`. Defaults to False.
            continuous_actions_boundary: Defaults to 1.
            actions_array: To be used in discrete actions modes. List of
                torques used as environment actions. Defaults to None.
            reward_func: Function that gives a reward based on environment
                state and terminated status. Defaults to Acrobot rewards.
            terminal_func: Function that checks if environment has terminated
                based on its state. Defaults to Acrobot termination.

        Raises:
            ValueError: When parameters passed don't match.
        """
        super().__init__(render_mode)
        self.simplified_state = simplified_state
        if self.simplified_state:
            high = np.array([pi, pi, self.MAX_VEL_1, self.MAX_VEL_2],
                            dtype=np.float32)
            self.observation_space = spaces.Box(low=-high,
                                                high=high,
                                                dtype=np.float32)

        self.continuous_actions = continuous_actions
        if self.continuous_actions:
            if actions_array is not None:
                raise ValueError(
                    "`actions_array` must only be used in discrete actions mode."
                )
            self.action_space = spaces.Box(low=-continuous_actions_boundary,
                                           high=continuous_actions_boundary,
                                           dtype=np.float32)
        else:
            if actions_array is not None:
                self.actions_array = actions_array
                self.action_space = spaces.Discrete(len(actions_array))
            else:
                self.actions_array = self.AVAIL_TORQUE

        # inspect functions for better error messages
        self.reward_func = reward_func
        self.terminal_func = terminal_func

    @classmethod
    def from_file(cls, filename) -> 'CustomAcrobot':
        """Create environment from saved file.

        Examples:
            >>> env = CustomAcrobot.from_file('test.pickle')
        """
        path = Path(filename).resolve()
        if not path.exists():
            print(f'{path} not found.')
            return
        with path.open('rb') as file:
            obj = pickle.load(file)
        return obj

    def _apply_torque(self, torque: float) -> ArrayLike:
        # Add noise to the force action
        if self.torque_noise_max > 0:
            torque += self.np_random.uniform(-self.torque_noise_max,
                                             self.torque_noise_max)

        # Now, augment the state with our force action so it can be passed to
        # _dsdt
        s_augmented = np.append(self.state, torque)

        ns = rk4(self._dsdt, s_augmented, [0, self.dt])

        ns[0] = wrap(ns[0], -pi, pi)
        ns[1] = wrap(ns[1], -pi, pi)
        ns[2] = bound(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[3] = bound(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
        return ns

    def step(
            self,
            action: Union[int,
                          float]) -> tuple[ArrayLike, float, bool, bool, dict]:
        s = self.state
        assert s is not None, "Call reset before using CustomAcrobot object."
        if self.continuous_actions:
            torque = action
        else:
            torque = self.actions_array[action]

        self.state = self._apply_torque(torque)
        terminated = self.terminal_func(self.state)
        reward = self.reward_func(self.state, terminated)

        if self.render_mode == "human":
            self.render()
        return (self._get_ob(), reward, terminated, False, {})

    def _get_ob(self) -> ArrayLike:
        s = self.state
        assert s is not None, "Call reset before using CustomAcrobot object."
        if self.simplified_state:
            return np.array(s, dtype=np.float32)
        else:
            return np.array(
                [cos(s[0]),
                 sin(s[0]),
                 cos(s[1]),
                 sin(s[1]), s[2], s[3]],
                dtype=np.float32)

    def save(self, filename, overwrite=False, verbose=True):
        """Save environment to file.

        Examples:
            >>> env.save('env.pickle')
            >>> env.save('env.pickle', overwrite=True)
            >>> env.save('env.pickle', verbose=False)
        """
        path = Path(filename).resolve()
        if path.exists() and not overwrite:
            print(
                f'{path} already exists. If you want to overwrite it, call env.save(filename, overwrite=True)'
            )
            return
        with path.open('wb') as file:
            pickle.dump(self, file)
        if verbose:
            print(f'Saved environment to {path}')
