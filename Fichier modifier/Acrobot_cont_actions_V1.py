import numpy as np
from numpy import cos, pi, sin
from gym import spaces
from gym.envs.classic_control.acrobot import AcrobotEnv, bound, rk4, wrap
import pygame
from pygame import gfxdraw
from typing import Optional


class AcrobotContActions(AcrobotEnv):

    TORQUE_SCALE = 2

    torque_noise_max = 0.0

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__(render_mode)
        high = np.array([pi, pi, self.MAX_VEL_1, self.MAX_VEL_2],
                        dtype=np.float32)
        self.observation_space = spaces.Box(low=-high,
                                            high=high,
                                            dtype=np.float32)
        self.action_space = spaces.Box(low=-1,
                                       high=1,
                                       dtype=np.float32)

    def _terminal(self):
        """choisis la position final d'équilibre
        avec theta l'angle du haut et theta 2 l'angle du bas
        """        
        theta1, theta2, omega1, omega2 = self._get_ob()
        return cos(theta1) <= -.95 and cos(theta2) >= .95 and np.abs(
            omega1) <= .5 and np.abs(omega2) <= .5

    def _get_ob(self):
        # state is array of size 4: [theta1, theta2, omega1, omega2]
        s = self.state
        assert s is not None, "Call reset before using this environment"
        return s.astype(np.float32)

    def step(self, action):
        s = self._get_ob()

        torque = action * self.TORQUE_SCALE

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

    def render(self):
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
        s = [pi, 0]
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
