# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, Optional, Tuple
import time
import timeit
from copy import deepcopy

import gym
import numpy as np
import torch
# import cyipopt

import mbrl.types

from .adam_projected import Adam
# from ..sqp.ftocp import FTOCP



class ModelEnv:
    """Wraps a dynamics model into a gym-like environment.

    This class can wrap a dynamics model to be used as an environment. The only requirement
    to use this class is for the model to use this wrapper is to have a method called
    ``predict()``
    with signature `next_observs, rewards = model.predict(obs,actions, sample=, rng=)`

    Args:
        env (gym.Env): the original gym environment for which the model was trained.
        model (:class:`mbrl.models.Model`): the model to wrap.
        termination_fn (callable): a function that receives actions and observations, and
            returns a boolean flag indicating whether the episode should end or not.
        reward_fn (callable, optional): a function that receives actions and observations
            and returns the value of the resulting reward in the environment.
            Defaults to ``None``, in which case predicted rewards will be used.
        generator (torch.Generator, optional): a torch random number generator (must be in the
            same device as the given model). If None (default value), a new generator will be
            created using the default torch seed.
    """

    def __init__(
        self,
        env: gym.Env,
        model,
        termination_fn: mbrl.types.TermFnType,
        reward_fn: Optional[mbrl.types.RewardFnType] = None,
        generator: Optional[torch.Generator] = None,
        n_delay = 0
    ):
        self.dynamics_model = model
        self.termination_fn = termination_fn
        self.reward_fn = reward_fn
        self.device = model.device

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.n_delay = n_delay
        self.reset_prev_actions()

        self._current_obs: torch.Tensor = None
        self._propagation_method: Optional[str] = None
        self._model_indices = None
        if generator:
            self._rng = generator
        else:
            self._rng = torch.Generator(device=self.device)
        self._return_as_np = True

    def reset(
        self,
        initial_obs_batch: np.ndarray,
        return_as_np: bool = True,
        shuffle_indices = True
    ) -> mbrl.types.TensorType:
        """Resets the model environment.

        Args:
            initial_obs_batch (np.ndarray): a batch of initial observations. One episode for
                each observation will be run in parallel. Shape must be ``B x D``, where
                ``B`` is batch size, and ``D`` is the observation dimension.
            return_as_np (bool): if ``True``, this method and :meth:`step` will return
                numpy arrays, otherwise it returns torch tensors in the same device as the
                model. Defaults to ``True``.

        Returns:
            (torch.Tensor or np.ndarray): the initial observation in the type indicated
            by ``return_as_np``.
        """
        assert len(initial_obs_batch.shape) == 2  # batch, obs_dim
        batch = mbrl.types.TransitionBatch(
            initial_obs_batch.astype(np.float32), None, None, None, None
        )
        self._current_obs = self.dynamics_model.reset(batch, rng=self._rng,
            shuffle_indices=shuffle_indices)
        self._return_as_np = return_as_np
        if self._return_as_np:
            return self._current_obs.cpu().numpy()
        return self._current_obs

    def reset_prev_actions(self):
        self.previous_actions = torch.zeros(0, self.action_space.shape[0]).to(self.device)

    def update_previous_actions(self, action):
        if not isinstance(action, torch.Tensor):
            action = np.array(action)
            action = torch.tensor(action)
        action = action.to(self.device)
        action = action.reshape(1, -1)
        self.previous_actions = torch.cat([self.previous_actions, action], dim=0)

    def get_previous_actions(self, t):
        if self.previous_actions.shape[0] >= t:
            return self.previous_actions[-t:-1].cpu().numpy()
        else:
            default_action = self.get_default_action()
            return default_action

    def get_default_action(self):
        return np.zeros(self.action_space.shape[0])

    def step(
        self, actions: mbrl.types.TensorType, sample: bool = False, num_particles=None
    ) -> Tuple[mbrl.types.TensorType, mbrl.types.TensorType, np.ndarray, Dict]:
        """Steps the model environment with the given batch of actions.

        Args:
            actions (torch.Tensor or np.ndarray): the actions for each "episode" to rollout.
                Shape must be ``B x A``, where ``B`` is the batch size (i.e., number of episodes),
                and ``A`` is the action dimension. Note that ``B`` must correspond to the
                batch size used when calling :meth:`reset`. If a np.ndarray is given, it's
                converted to a torch.Tensor and sent to the model device.
            sample (bool): if ``True`` model predictions are stochastic. Defaults to ``False``.

        Returns:
            (tuple): contains the predicted next observation, reward, done flag and metadata.
            The done flag is computed using the termination_fn passed in the constructor.
        """
        assert len(actions.shape) == 2  # batch, action_dim
        with torch.no_grad():
            # if actions is tensor, code assumes it's already on self.device
            if isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions).to(self.device)
            model_in = mbrl.types.TransitionBatch(
                self._current_obs, actions, None, None, None
            )

            next_observs, pred_rewards = self.dynamics_model.sample(
                model_in,
                deterministic=not sample,
                rng=self._rng,
            )
            obs_dim = next_observs.shape[1]
            if not sample:
                actions = actions[::num_particles]
                next_observs_mean = next_observs.reshape(-1, num_particles, obs_dim).mean(dim=1)
            else:
                next_observs_mean = next_observs
            rewards = (
                pred_rewards
                if self.reward_fn is None
                else self.reward_fn(actions, next_observs_mean)
            )
            dones = self.termination_fn(actions, next_observs)
            self._current_obs = next_observs
            if self._return_as_np:
                next_observs = next_observs.cpu().numpy()
                rewards = rewards.cpu().numpy()
                dones = dones.cpu().numpy()
            return next_observs, rewards, dones, {}

    def step_grad(
        self, actions: mbrl.types.TensorType, sample: bool = False, num_particles=None
    ) -> Tuple[mbrl.types.TensorType, mbrl.types.TensorType, np.ndarray, Dict]:
        """Steps the model environment with the given batch of actions.

        Args:
            actions (torch.Tensor or np.ndarray): the actions for each "episode" to rollout.
                Shape must be ``B x A``, where ``B`` is the batch size (i.e., number of episodes),
                and ``A`` is the action dimension. Note that ``B`` must correspond to the
                batch size used when calling :meth:`reset`. If a np.ndarray is given, it's
                converted to a torch.Tensor and sent to the model device.
            sample (bool): if ``True`` model predictions are stochastic. Defaults to ``False``.

        Returns:
            (tuple): contains the predicted next observation, reward, done flag and metadata.
            The done flag is computed using the termination_fn passed in the constructor.
        """
        
        assert len(actions.shape) == 2  # batch, action_dim
        # if actions is tensor, code assumes it's already on self.device
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(self.device)
        model_in = mbrl.types.TransitionBatch(
            self._current_obs, actions, None, None, None
        )
        
        next_observs, pred_rewards = self.dynamics_model.sample(
            model_in,
            deterministic=True,
            use_grad=True
        )

        obs_dim = next_observs.shape[1]
        if not sample:
            actions = actions[::num_particles]
            # batch size // population size
            next_observs_mean = next_observs.reshape(-1, num_particles, obs_dim).mean(dim=1)
        else:
            next_observs_mean = next_observs
        rewards = (
            pred_rewards
            if self.reward_fn is None
            else self.reward_fn(actions, next_observs_mean)
        )
        dones = self.termination_fn(actions, next_observs)
        self._current_obs = next_observs
        if self._return_as_np:
            next_observs = next_observs.cpu().numpy()
            rewards = rewards.cpu().numpy()
            dones = dones.cpu().numpy()
        
        return next_observs, rewards, dones, {}

    def render(self, mode="human"):
        pass

    def evaluate_action_sequences(
        self,
        action_sequences: torch.Tensor,
        initial_state: np.ndarray,
        num_particles: int,
        sample=True,
    ) -> torch.Tensor:
        """Evaluates a batch of action sequences on the model.

        Args:
            action_sequences (torch.Tensor): a batch of action sequences to evaluate.  Shape must
                be ``B x H x A``, where ``B``, ``H``, and ``A`` represent batch size, horizon,
                and action dimension, respectively.
            initial_state (np.ndarray): the initial state for the trajectories.
            num_particles (int): number of times each action sequence is replicated. The final
                value of the sequence will be the average over its particles values.

        Returns:
            (torch.Tensor): the accumulated reward for each action sequence, averaged over its
            particles.
        """
        assert (
            len(action_sequences.shape) == 3
        )  # population_size, horizon, action_shape
        action_sequences = action_sequences.to(self.device)
        shuffle_indices = True if sample else False
        _, total_rewards = self.run_trajectory(initial_state, action_sequences,
            num_particles, use_grad=False, sample=sample,
            shuffle_indices=shuffle_indices)

        try:
            total_rewards = total_rewards.reshape(-1, num_particles)
        except:
            return total_rewards.flatten()
        return total_rewards.mean(dim=1)

    def compute_objective(self, initial_state, action_sequences, num_particles, use_grad):
        _, total_rewards = self.run_trajectory(initial_state, action_sequences, 
            num_particles, use_grad=use_grad, sample=False,
            shuffle_indices=False)
        
        total_rewards = -total_rewards
        avg_reward = total_rewards
        objective = avg_reward.flatten()
        return objective

    def fix_action_sequences(self, action_sequences):
        action_sequences = np.expand_dims(action_sequences, axis=0)
        action_sequences = torch.from_numpy(action_sequences).to(self.device)
        return action_sequences

    def run_trajectory(self, initial_state, action_sequences, num_particles=5, use_grad=False, sample=False,
            shuffle_indices=True):
        if isinstance(action_sequences, np.ndarray):
            action_sequences = self.fix_action_sequences(action_sequences)
        state_dim = initial_state.shape[0]
        population_size, horizon, action_dim = action_sequences.shape
        initial_obs_batch = np.tile(
            initial_state, (num_particles * population_size, 1)
        ).astype(np.float32)
        self.reset(initial_obs_batch, return_as_np=False, shuffle_indices=shuffle_indices)
        batch_size = initial_obs_batch.shape[0]
        if not sample:
            total_rewards = torch.zeros(population_size, 1).to(self.device)
        else:
            total_rewards = torch.zeros(batch_size, 1).to(self.device)
        all_obs = torch.zeros(horizon, state_dim).to(self.device)
        for time_step in range(horizon):
            n_prev = time_step - self.n_delay
            if n_prev < 0:
                prev_exist = self.previous_actions.shape[0]
                if -n_prev - prev_exist > 0:
                    default_actions = torch.zeros(-n_prev - prev_exist, action_dim).to(self.device)
                    extra_actions = torch.cat([default_actions, self.previous_actions], dim=0)
                else:
                    extra_actions = self.previous_actions[n_prev:]

                extra_actions = torch.repeat_interleave(extra_actions[None, :, :], population_size, dim=0)

                actions_for_step = torch.cat([extra_actions, action_sequences[:, :(time_step + 1), :]], dim=1)
            else:
                actions_for_step = action_sequences[:, (time_step - self.n_delay):(time_step + 1), :]
            actions_for_step = actions_for_step.reshape(actions_for_step.shape[0], -1)
            action_batch = torch.repeat_interleave(
                actions_for_step, num_particles, dim=0
            )
            if use_grad:
                obs, rewards, dones, _ = self.step_grad(action_batch, num_particles=num_particles)
            else:
                obs, rewards, dones, _ = self.step(action_batch, sample=sample, num_particles=num_particles)
            obs = torch.mean(obs, axis=0)
            all_obs[time_step, :] = obs
            total_rewards += rewards
        return all_obs, total_rewards