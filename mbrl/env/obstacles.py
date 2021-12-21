"""
Inspired by and adapted from the toy environment from 

Homanga Bharadhwaj et al. Model-Predictive Control via 
Cross-Entropy and Gradient-Based Optimization 2020.

https://github.com/homangab/gradcem/blob/master/mpc/test_energy.py
"""

import math

import gym
import torch
import numpy as np
from gym import logger, spaces
from gym.utils import seeding

class BatchRepulseCircle:
    def __init__(self, origins, radius, batch_dims=[0,1], k=1.0):
        self.B = origins.shape[0]
        self.origins = origins
        self.radius = radius
        self.k = k
        self.batch_dims = batch_dims

    def force(self, x, y):
        x = np.array([[x, y]])
        contact_vector = (x-self.origins)[..., np.arange(self.B).reshape(-1,1), self.batch_dims]
        dist = np.linalg.norm(contact_vector, axis=-1, keepdims=True)
        penetration = np.clip((self.radius - dist), 0,self.radius)

        rebound = np.sum(self.k / 10 * penetration * contact_vector, axis=0)

        return rebound


class Obstacles2Env(gym.Env):
    """
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self, num_obs=3):
        self.mass = 1.0
        self.dt = 0.05
        num_obs = num_obs
        self.num_obs = num_obs

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                np.inf,
                np.inf
            ],
            dtype=np.float32,
        )

        self.max_x = 1.5
        self.min_x = 0.5

        act_high = np.array((0.05, 0.05), dtype=np.float32)
        self.action_space = spaces.Box(-act_high, act_high, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.goal = (1, 0)

        self.steps_beyond_done = None

        radius = 2 / ((num_obs + 1)*4)
        self.radius = radius

        x_list = np.linspace(-1, 1, num_obs+2)[1:-1]
        y_list = np.random.randint(0, 2, num_obs)
        factor = 1
        y_list[0] = 1
        y_list[y_list == 1] = radius / factor
        y_list[y_list == 0] = -radius / factor 

        origin_list = list(zip(x_list, y_list))
        self.origin_list = origin_list
        origin_list = np.array(origin_list)
        self.obstacle = BatchRepulseCircle(origin_list, radius, 
            batch_dims=(torch.tensor([0,1], dtype=torch.long).view(1,2).expand(origin_list.shape[0],2)).numpy(), k=100.0)

        self.set_future_locations = False

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        actx, acty = action
        x, y = self.state

        x_re, y_re = self.obstacle.force(x, y)

        x += actx + x_re
        y += acty + y_re

        self.state = (x, y)

        done = False
        gx, gy = self.goal

        quadratic_rew = -(x - gx)**2 + -5*(y - gy)**2

        penalty = 0
        failure_reward = 0

        if not done:
            reward = quadratic_rew + penalty
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward = quadratic_rew + penalty
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = failure_reward

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = np.zeros((2,))
        self.state[0] = -1
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, future_locations=None, mode="human"):
        screen_width = 600
        screen_height = 600

        world_width = 3
        scale = screen_width / world_width

        if self.viewer is None or not self.set_future_locations:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -5, 5, 5, -5
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)

            l, r, t, b = -5, 5, 5, -5
            goal = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.goalttrans = rendering.Transform()
            goal.add_attr(self.goalttrans)
            self.viewer.add_geom(goal)

            if future_locations is not None:
                l, r, t, b = -2, 2, 2, -2

                orig_traj = future_locations[0]
                modified_traj = future_locations[1]
                sample_traj = future_locations[2]

                for i in range(len(orig_traj)):
                    pred = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                    pred._color.vec4 = (0.1, 0.1, 1.0, (len(modified_traj) - i) / len(orig_traj))
                    setattr(self, f'orig_predtrans_{i}', rendering.Transform())
                    pred.add_attr(getattr(self, f'orig_predtrans_{i}'))
                    self.viewer.add_geom(pred)

                for i in range(len(modified_traj)):
                    pred = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                    pred._color.vec4 = (1.0, 0.1, 0.1, (len(modified_traj) - i) / len(modified_traj))
                    setattr(self, f'mod_predtrans_{i}', rendering.Transform())
                    pred.add_attr(getattr(self, f'mod_predtrans_{i}'))
                    self.viewer.add_geom(pred)

                for i in range(len(sample_traj)):
                    pred = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                    pred._color.vec4 = (0.1, 1.0, 0.1, (len(modified_traj) - i) / len(modified_traj))
                    setattr(self, f'sample_predtrans_{i}', rendering.Transform())
                    pred.add_attr(getattr(self, f'sample_predtrans_{i}'))
                    self.viewer.add_geom(pred)
                self.set_future_locations = True
            

            for i in range(len(self.origin_list)):
                obs = rendering.make_circle(radius=np.floor(scale*self.radius), filled=False)
                obs.set_color(0.1, 1.0, 0.1)
                setattr(self, f'obstrans_{i}', rendering.Transform())
                obs.add_attr(getattr(self, f'obstrans_{i}'))
                self.viewer.add_geom(obs)

        if self.state is None:
            return None

        x = self.state
        for i, (x_pos, y_pos) in enumerate(self.origin_list):
            obstrans = getattr(self, f'obstrans_{i}')
            xx = x_pos * scale + screen_width / 2.0
            yy = y_pos * scale + screen_height / 2.0
            obstrans.set_translation(xx, yy)

        if future_locations is not None:
            orig_traj = future_locations[0]
            modified_traj = future_locations[1]
            sample_traj = future_locations[2]

            for i, (x_pos, y_pos) in enumerate(orig_traj):
                x_pos = x_pos.cpu().numpy().astype('float64')
                y_pos = y_pos.cpu().numpy().astype('float64')
                orig_predtrans = getattr(self, f'orig_predtrans_{i}')
                xx = x_pos * scale + screen_width / 2.0
                yy = y_pos * scale + screen_height / 2.0
                orig_predtrans.set_translation(xx, yy)

            for i, (x_pos, y_pos) in enumerate(modified_traj):
                x_pos = x_pos.cpu().numpy().astype('float64')
                y_pos = y_pos.cpu().numpy().astype('float64')
                mod_predtrans = getattr(self, f'mod_predtrans_{i}')
                xx = x_pos * scale + screen_width / 2.0
                yy = y_pos * scale + screen_height / 2.0
                mod_predtrans.set_translation(xx, yy)

            for i, (x_pos, y_pos) in enumerate(sample_traj):
                x_pos = x_pos.cpu().numpy().astype('float64')
                y_pos = y_pos.cpu().numpy().astype('float64')
                sample_predtrans = getattr(self, f'sample_predtrans_{i}')
                xx = x_pos * scale + screen_width / 2.0
                yy = y_pos * scale + screen_height / 2.0
                sample_predtrans.set_translation(xx, yy)


        cartx = x[0] * scale + screen_width / 2.0
        carty = x[1] * scale + screen_height / 2.0
        self.carttrans.set_translation(cartx, carty)

        goalx = 1 * scale + screen_width / 2.0  
        goaly = 0 * scale + screen_height / 2.0
        self.goalttrans.set_translation(goalx, goaly)


        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
