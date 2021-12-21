"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math

import gym
import numpy as np
from gym import logger, spaces
from gym.utils import seeding


class DroneLinear(gym.Env):
    """
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self):
        self.dt = 0.1
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        act_high = np.array((1,), dtype=np.float32)
        self.action_space = spaces.Box(-act_high, act_high, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        self.goal = self.x_threshold / 2
        self.goal_epsilon = 0.1

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action = action.squeeze()
        x, v = self.state

        x = x + self.dt * v
        v = v + action
        self.state = (x, v)
        
        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
        )

        position_reward = np.exp(-(x - self.goal)**2)

        if not done:
            reward = position_reward
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = position_reward
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(2,))
        self.state[0] = self.state[0] - self.x_threshold / 2
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode="human"):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 75  # TOP OF CART
        cartwidth = 25.0
        cartheight = 30.0

        goaly = 50

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            l, r, t, b = (
                -10 / 2,
                10 / 2,
                10 / 2,
                -10 / 2,
            )
            goal = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            goal.set_color(0.1, 1.0, 0.1)
            self.goaltrans = rendering.Transform()
            goal.add_attr(self.goaltrans)
            self.viewer.add_geom(goal)

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        goalx = self.goal * scale + screen_width / 2.0
        self.goaltrans.set_translation(goalx, goaly)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
