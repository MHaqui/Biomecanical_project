import numpy as np
from numpy import cos, pi, sin

import gym
from gym import core, logger, spaces
from gym.envs.classic_control.acrobot import AcrobotEnv, wrap
from gym.error import DependencyNotInstalled


class OurEnv(AcrobotEnv):

    # it's trivial to add more granulation to possible torques
    AVAIL_TORQUE = [-1.0, 0.0, +1]

    # with noise environment becomes stochastisc rather than deterministic
    torque_noise_max = 0.0

    proximity_threshold = 2

    def __init__(self,
                 final_state: tuple[float, float],
                 render_mode: str | None = None):
        self.final_state = np.array(
            [
                wrap(final_state[0], -np.pi, np.pi),  # theta1
                wrap(final_state[1], -np.pi, np.pi),  # theta1
                0,  # omega1
                0  # omega2
            ],)
        super().__init__(render_mode)

    def _terminal(self):
        # state is array of size 4: [theta1, theta2, omega1, omega2]
        assert self.state is not None, "Call reset before using AcrobotEnv object."
        return np.allclose(self.state, self.final_state)

    def step(self, action: int):
        # override reward to have intermediate rewards
        observation, reward, terminated, truncated, info = super().step(action)
        distance = np.linalg.norm(self.final_state - self.state)
        if distance < self.proximity_threshold:
            reward = 100 / np.linalg.norm(self.final_state - self.state)**3
        # IDEA: have position and velocity distances and evaluate different
        # rewards based on each
        # pos_distance = np.linalg.norm(self.final_state[:2] - self.state[:2])
        # vel_distance = np.linalg.norm(self.final_state[2:] - self.state[2:])
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")')
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.SCREEN_DIM, self.SCREEN_DIM))
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.SCREEN_DIM, self.SCREEN_DIM))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        surf = pygame.Surface((self.SCREEN_DIM, self.SCREEN_DIM))
        surf.fill((255, 255, 255))
        s = self.state

        bound = self.LINK_LENGTH_1 + self.LINK_LENGTH_2 + 0.2  # 2.2 for default
        scale = self.SCREEN_DIM / (bound * 2)
        offset = self.SCREEN_DIM / 2

        if s is None:
            return None

        p1 = [
            -self.LINK_LENGTH_1 * cos(s[0]) * scale,
            self.LINK_LENGTH_1 * sin(s[0]) * scale,
        ]

        p2 = [
            p1[0] - self.LINK_LENGTH_2 * cos(s[0] + s[1]) * scale,
            p1[1] + self.LINK_LENGTH_2 * sin(s[0] + s[1]) * scale,
        ]

        xys = np.array([[0, 0], p1, p2])[:, ::-1]
        thetas = [s[0] - pi / 2, s[0] + s[1] - pi / 2]
        link_lengths = [self.LINK_LENGTH_1 * scale, self.LINK_LENGTH_2 * scale]

        pygame.draw.line(
            surf,
            start_pos=(-2.2 * scale + offset, 1 * scale + offset),
            end_pos=(2.2 * scale + offset, 1 * scale + offset),
            color=(0, 0, 0),
        )

        self.render_final_state(scale, offset, surf)

        for i, ((x, y), th, llen) in enumerate(zip(xys, thetas, link_lengths)):
            x = x + offset
            y = y + offset
            l, r, t, b = 0, llen, 0.1 * scale, -0.1 * scale
            coords = [(l, b), (l, t), (r, t), (r, b)]
            transformed_coords = []
            for coord in coords:
                coord = pygame.math.Vector2(coord).rotate_rad(th)
                coord = (coord[0] + x, coord[1] + y)
                transformed_coords.append(coord)
            gfxdraw.aapolygon(surf, transformed_coords, (0, 204, 204))
            gfxdraw.filled_polygon(surf, transformed_coords, (0, 204, 204))

            gfxdraw.aacircle(surf, int(x), int(y), int(0.1 * scale),
                             (204, 204, 0))
            gfxdraw.filled_circle(surf, int(x), int(y), int(0.1 * scale),
                                  (204, 204, 0))

            font = pygame.font.SysFont(None, 24)
            img = pygame.transform.flip(
                font.render(f'θ = {s[i]:.3f}, ω = {s[i + len(thetas)]:.3f}',
                            True, (0, 0, 0)), False, True)
            surf.blit(img, (int(x), int(y)))

        surf = pygame.transform.flip(surf, False, True)
        self.screen.blit(surf, (0, 0))

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(
                self.screen)),
                                axes=(1, 0, 2))

    def render_final_state(self, scale, offset, surf):
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )
        s = self.final_state
        p1 = [
            -self.LINK_LENGTH_1 * cos(s[0]) * scale,
            self.LINK_LENGTH_1 * sin(s[0]) * scale,
        ]

        p2 = [
            p1[0] - self.LINK_LENGTH_2 * cos(s[0] + s[1]) * scale,
            p1[1] + self.LINK_LENGTH_2 * sin(s[0] + s[1]) * scale,
        ]

        xys = np.array([[0, 0], p1, p2])[:, ::-1]
        thetas = [s[0] - pi / 2, s[0] + s[1] - pi / 2]
        link_lengths = [self.LINK_LENGTH_1 * scale, self.LINK_LENGTH_2 * scale]

        pygame.draw.line(
            surf,
            start_pos=(-2.2 * scale + offset, 1 * scale + offset),
            end_pos=(2.2 * scale + offset, 1 * scale + offset),
            color=(0, 0, 0),
        )

        for i, ((x, y), th, llen) in enumerate(zip(xys, thetas, link_lengths)):
            x = x + offset
            y = y + offset
            l, r, t, b = 0, llen, 0.1 * scale, -0.1 * scale
            coords = [(l, b), (l, t), (r, t), (r, b)]
            transformed_coords = []
            for coord in coords:
                coord = pygame.math.Vector2(coord).rotate_rad(th)
                coord = (coord[0] + x, coord[1] + y)
                transformed_coords.append(coord)
            gfxdraw.aapolygon(surf, transformed_coords, (200, 200, 200))
            gfxdraw.filled_polygon(surf, transformed_coords, (200, 200, 200))

            gfxdraw.aacircle(surf, int(x), int(y), int(0.1 * scale),
                             (120, 120, 120))
            gfxdraw.filled_circle(surf, int(x), int(y), int(0.1 * scale),
                                  (120, 120, 120))
