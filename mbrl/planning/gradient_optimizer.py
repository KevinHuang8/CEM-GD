from typing import Callable, List, Optional, Sequence, cast
from copy import deepcopy

import hydra
import numpy as np
import omegaconf
import torch
import torch.distributions

import mbrl.models
import mbrl.types
import mbrl.util.math
from torch.serialization import save

from .core import Agent, complete_agent_cfg

from .trajectory_opt import Optimizer
from .adam_projected import Adam

class GradientOptimizer(Optimizer):

    def __init__(
        self,
        num_iterations: int,
        elite_ratio: float,
        population_size: int,
        lower_bound: Sequence[float],
        upper_bound: Sequence[float],
        alpha: float,
        device: torch.device,
        return_mean_elites: bool = False,
        num_top: int = 3,
        resample_amount = 20
    ):
        assert num_top <= population_size
        assert num_top <= resample_amount and resample_amount <= population_size

        super().__init__()
        self.num_iterations = num_iterations
        self.elite_ratio = elite_ratio
        self.population_size = population_size
        self.resample_amount = resample_amount
        self.elite_num = np.ceil(self.population_size * self.elite_ratio).astype(
            np.int32
        )
        self.lower_bound = torch.tensor(lower_bound, device=device, dtype=torch.float32)
        self.upper_bound = torch.tensor(upper_bound, device=device, dtype=torch.float32)

        self.initial_var = ((self.upper_bound - self.lower_bound) ** 2) / 16
        self.alpha = alpha
        self.return_mean_elites = return_mean_elites
        self.device = device
        assert num_top <= self.elite_num
        self.num_top = num_top

    def get_top_trajectories(
        self,
        reward_fun: Callable[[torch.Tensor], torch.Tensor],
        x0: Optional[torch.Tensor] = None,
        default_x0 = None,
        callback: Optional[Callable[[torch.Tensor, torch.Tensor, int], None]] = None,
        is_resampling=True,
        use_opt=False,
        flag=False,
        init=False,
        **kwargs,
    ) -> List[torch.Tensor]:
        """
        
        """
        mu = x0.clone()
        var = self.initial_var.clone()

        top_rewards = np.array([-1e10 for i in range(self.num_top)])
        top_sols = [torch.empty_like(mu) for i in range(self.num_top)]

        amount = self.population_size if is_resampling else self.resample_amount
        n_iter = self.num_iterations
        if use_opt:
            n_iter = 5

        elite_num = self.elite_num
        if init and use_opt:
            amount = 3000
            elite_num = 300

        population = torch.zeros((amount,) + x0.shape).to(
            device=self.device
        )
        for i in range(n_iter):
            lb_dist = mu - self.lower_bound
            ub_dist = self.upper_bound - mu
            mv = torch.min(torch.square(lb_dist / 2), torch.square(ub_dist / 2))
            constrained_var = torch.min(mv, var)

            population = mbrl.util.math.truncated_normal_(population)
            population = population * torch.sqrt(constrained_var) + mu

            values = reward_fun(population, sample=True)

            if callback is not None:
                callback(population, values, i)

            # filter out NaN values
            values[values.isnan()] = -1e-10
            elite_amount = elite_num if is_resampling else self.resample_amount
            if is_resampling and flag:
                elite_amount = self.resample_amount
            best_values, elite_idx = values.topk(elite_amount)
            best_values = best_values.cpu().numpy()
            elite = population[elite_idx]

            new_mu = torch.mean(elite, dim=0)
            new_var = torch.var(elite, unbiased=False, dim=0)
            mu = self.alpha * mu + (1 - self.alpha) * new_mu
            var = self.alpha * var + (1 - self.alpha) * new_var

            for i in range(self.num_top):
                # keep track of self.num_top trajectories
                s = best_values[i] > top_rewards
                if np.any(s):
                    mask = np.ma.masked_array(top_rewards, mask=~s)
                    highest_idx = np.argmax(mask)
                    top_rewards[highest_idx] = best_values[i]
                    top_sols[highest_idx] = population[elite_idx[i]].detach().clone()

        if use_opt or not self.return_mean_elites:
            return top_sols
        else:
            return [mu]

    def optimize_trajectory_batch(self, obj_fun, action_sequences_list,
        start_lr=0.01, factor_shrink=1.5, max_tries=7, max_iterations=15):
        for action_sequences in action_sequences_list:
            action_sequences.requires_grad = True

        n = len(action_sequences_list)

        optimizer = Adam(
            [{
                'params': act_seq, 
                'factor': 1,
                'action_bounds': (self.lower_bound, self.upper_bound)
                } for act_seq in action_sequences_list],
            lr=start_lr
        )
        optimizer.zero_grad()

        saved_parameters = [None for i in range(n)]
        saved_opt_states = [None for i in range(n)]
        current_iteration = np.array([0 for i in range(n)])
        done = np.array([False for i in range(n)])

        action_sequences_batch = torch.stack(action_sequences_list)
        objective_all = obj_fun(action_sequences_batch, True)

        current_objective = [objective_all[i] for i in range(n)]

        for i in range(n):
            action_sequences = action_sequences_list[i]
            saved_parameters[i] = action_sequences.detach().clone()
            saved_opt_states[i] = deepcopy(optimizer.state[action_sequences])
            objective_all[i].backward(retain_graph=(i != n - 1))

        while not np.all(done):
            optimizer.step()

            # Compute objectives of all trajectories after stepping
            action_sequences_batch = torch.stack(action_sequences_list)
            objective_all = obj_fun(action_sequences_batch, True)

            backwards_pass = []

            for i in range(n):
                if done[i]:
                    continue
                action_sequences = action_sequences_list[i]
                if objective_all[i] > current_objective[i]:
                    # If after the step, the cost is higher, then undo
                    action_sequences.data = saved_parameters[i].data.clone()
                    optimizer.state[action_sequences] = deepcopy(saved_opt_states[i])
                    optimizer.param_groups[i]['factor'] *= factor_shrink

                    if optimizer.param_groups[i]['factor'] > factor_shrink**max_tries:
                        # line search failed, mark action sequence as done
                        action_sequences.grad = None
                        done[i] = True
                else:
                    # successfully completed step.
                    # Save current state, and compute gradients
                    saved_parameters[i] = action_sequences.detach().clone()
                    saved_opt_states[i] = deepcopy(optimizer.state[action_sequences])
                    current_objective[i] = objective_all[i]
                    optimizer.param_groups[i]['factor'] = 1
                    action_sequences.grad = None
                    backwards_pass.append(i)
                                        
                    current_iteration[i] += 1
                    if current_iteration[i] > max_iterations:
                        action_sequences.grad = None
                        done[i] = True
                
            to_compute = [objective_all[i] for i in backwards_pass]
            grads = [(torch.empty_like(objective_all[i])*0 + 1).to(self.device) for i in backwards_pass]
            torch.autograd.backward(to_compute, grads)
        
        return [traj.detach() for traj in action_sequences_list]

    def fix_action_sequences(self, 
        action_sequences: torch.Tensor
    ) -> torch.Tensor:

        action_sequences = torch.unsqueeze(action_sequences, dim=0)
        return action_sequences

    def optimize(
        self,
        reward_fun: Callable[[torch.Tensor, bool], torch.Tensor],
        obj_fun: Callable[[torch.Tensor, bool], torch.Tensor], 
        x0: Optional[torch.Tensor] = None,
        default_x0 = None,
        callback: Optional[Callable[[torch.Tensor, torch.Tensor, int], None]] = None,
        is_resampling=True,
        use_opt=True,
        trial_step=None
    ) -> torch.Tensor:
        """Runs the optimization using CEM.

        Args:
            reward_fun: reward function to maximize. Second argument 
                must be a kewword argument "sample", being an
                optional flag for whether the model should be sampled 
                (if False, then deterministic).
            x0 (tensor, optional): initial mean for the population. Must
                be consistent with lower/upper bounds.
            callback (callable(tensor, tensor, int) -> any, optional): if given, this
                function will be called after every iteration, passing it as input the full
                population tensor, its corresponding objective function values, and
                the index of the current iteration. This can be used for logging and plotting
                purposes.

        Returns:
            (torch.Tensor): the best solution found. Shape (horizon x action dim)
        """
        init = (trial_step == 0)

        top_trajectories = self.get_top_trajectories(reward_fun, x0, default_x0, 
            callback, use_opt=use_opt, is_resampling=is_resampling, flag=False, init=init)

        if use_opt:
            top_trajectories = self.optimize_trajectory_batch(obj_fun, top_trajectories)
        
        trajectory_batch = torch.stack(top_trajectories)
        batch_rew = reward_fun(trajectory_batch, sample=False)
        i = torch.argmax(batch_rew)
        best_trajectory = trajectory_batch[i]

        if torch.any(best_trajectory.isnan()) or torch.any(best_trajectory.isinf()):
            best_trajectory = x0
        best_trajectory = torch.squeeze(best_trajectory, dim=0)
        return best_trajectory


class TrajectoryOptimizer:
    """Class for using generic optimizers on trajectory optimization problems.

    This is a convenience class that sets up optimization problem for trajectories, given only
    action bounds and the length of the horizon. Using this class, the concern of handling
    appropriate tensor shapes for the optimization problem is hidden from the users, which only
    need to provide a function that is capable of evaluating trajectories of actions. It also
    takes care of shifting previous solution for the next optimization call, if the user desires.

    The optimization variables for the problem will have shape ``H x A``, where ``H`` and ``A``
    represent planning horizon and action dimension, respectively. The initial solution for the
    optimizer will be computed as (action_ub - action_lb) / 2, for each time step.

    Args:
        optimizer_cfg (omegaconf.DictConfig): the configuration of the optimizer to use.
        action_lb (np.ndarray): the lower bound for actions.
        action_ub (np.ndarray): the upper bound for actions.
        planning_horizon (int): the length of the trajectories that will be optimized.
        replan_freq (int): the frequency of re-planning. This is used for shifting the previous
        solution for the next time step, when ``keep_last_solution == True``. Defaults to 1.
        keep_last_solution (bool): if ``True``, the last solution found by a call to
            :meth:`optimize` is kept as the initial solution for the next step. This solution is
            shifted ``replan_freq`` time steps, and the new entries are filled using th3 initial
            solution. Defaults to ``True``.
    """

    def __init__(
        self,
        optimizer_cfg: omegaconf.DictConfig,
        action_lb: np.ndarray,
        action_ub: np.ndarray,
        planning_horizon: int,
        replan_freq: int = 1,
        keep_last_solution: bool = True,
        resample: bool = True,
    ):
        optimizer_cfg.lower_bound = np.tile(action_lb, (planning_horizon, 1)).tolist()
        optimizer_cfg.upper_bound = np.tile(action_ub, (planning_horizon, 1)).tolist()
        self.optimizer: Optimizer = hydra.utils.instantiate(optimizer_cfg)
        self.initial_solution = (
            ((torch.tensor(action_lb) + torch.tensor(action_ub)) / 2)
            .float()
            .to(optimizer_cfg.device)
        )
        self.initial_solution = self.initial_solution.repeat((planning_horizon, 1))
        self.previous_solution = self.initial_solution.clone()
        self.replan_freq = replan_freq
        self.keep_last_solution = keep_last_solution
        self.horizon = planning_horizon

        self.resample = resample
        self.init = True

    def optimize(
        self,
        reward_fun: Callable[[torch.Tensor, bool], torch.Tensor],
        cost_fun: Callable[[torch.Tensor, bool], torch.Tensor],
        use_opt: bool = True,
        callback: Optional[Callable] = None,
        trial_step=None
    ) -> np.ndarray:
        """Runs the trajectory optimization.

        Args:
            reward_fun: A function that takes an input Tensor (of shape
                batch size x horizon x action dim) of action sequences, and 
                computes their rewards. The function should also take a second
                keyword argument called "sample" that determines whether
                rewards are computed with sampling or not. If sample=False, then
                the underlying model is deterministic.

                The action trajectories sampled with the highest rewards are
                taken as candidates for local optimization.
            cost_fun: A function that takes an input Tensor (of shape
                batch size x horizon x action dim) of action sequences, and
                computes their costs. This is the objective function that
                local optimization will minimize.

                The second argument of this function is a boolean that serves
                as a flag for whether gradients should be propagated through
                this function or not. If True, gradients are propagated.

                Unlike reward_fun, this function must always be deterministic.
                This function is often, but not necessarily, the negative of
                reward_fun. 
            callback (callable, optional): a callback function
                to pass to the optimizer.

        Returns:
            (tuple of np.ndarray and float): the best action sequence.
        """
        if self.resample or self.init or not use_opt:
            best_solution = self.optimizer.optimize(
                reward_fun,
                cost_fun,
                x0=self.previous_solution,
                default_x0=self.initial_solution.clone(),
                callback=callback,
                is_resampling=True,
                use_opt=use_opt,
                trial_step=trial_step
            )
            self.init = False
        else:
            best_solution = self.optimizer.optimize(
                reward_fun,
                cost_fun,
                x0=self.previous_solution,
                default_x0=self.initial_solution.clone(),
                callback=callback,
                is_resampling=False,
                use_opt=use_opt,
                trial_step=trial_step
            )

        if self.keep_last_solution:
            self.previous_solution = best_solution.roll(-self.replan_freq, dims=0)
            self.previous_solution[-self.replan_freq :] = self.initial_solution[0]

        return best_solution.cpu().numpy()

    def reset(self):
        """Resets the previous solution cache to the initial solution."""
        self.previous_solution = self.initial_solution.clone()
        self.init = True

class TrajectoryOptimizerAgent(Agent):
    """Agent that performs trajectory optimization on a given objective function for each action.

    This class uses an internal :class:`TrajectoryOptimizer` object to generate
    sequence of actions, given a user-defined trajectory optimization function.

    Args:
        optimizer_cfg (omegaconf.DictConfig): the configuration of the base optimizer to pass to
            the trajectory optimizer.
        action_lb (sequence of floats): the lower bound of the action space.
        action_ub (sequence of floats): the upper bound of the action space.
        planning_horizon (int): the length of action sequences to evaluate. Defaults to 1.
        replan_freq (int): the frequency of re-planning. The agent will keep a cache of the
            generated sequences an use it for ``replan_freq`` number of :meth:`act` calls.
            Defaults to 1.
        verbose (bool): if ``True``, prints the planning time on the console.

    Note:
        After constructing an agent of this type, the user must call
        :meth:`set_trajectory_eval_fn`. This is not passed to the constructor so that the agent can
        be automatically instantiated with Hydra (which in turn makes it easy to replace this
        agent with an agent of another type via config-only changes).
    """

    def __init__(
        self,
        optimizer_cfg: omegaconf.DictConfig,
        action_lb: Sequence[float],
        action_ub: Sequence[float],
        planning_horizon: int = 1,
        replan_freq: int = 1,
        verbose: bool = False,
        resample = True
    ):
        self.optimizer = TrajectoryOptimizer(
            optimizer_cfg,
            np.array(action_lb),
            np.array(action_ub),
            planning_horizon=planning_horizon,
            replan_freq=replan_freq,
            resample=resample
        )
        self.optimizer_args = {
            "optimizer_cfg": optimizer_cfg,
            "action_lb": np.array(action_lb),
            "action_ub": np.array(action_ub),
        }
        self.trajectory_eval_fn: mbrl.types.TrajectoryEvalFnType = None
        self.actions_to_use: List[np.ndarray] = []
        self.replan_freq = replan_freq
        self.verbose = verbose
        self.resample = resample

    def set_optimize_fun(self, optimize_fun):
        self.local_optimize_fun = optimize_fun

    def set_reward_fun(self, reward_fun):
        self.reward_fun = reward_fun

    def set_cost_fun(self, cost_fun):
        self.cost_fun = cost_fun

    def reset(self, planning_horizon: Optional[int] = None):
        """Resets the underlying trajectory optimizer."""
        if planning_horizon:
            self.optimizer = TrajectoryOptimizer(
                cast(omegaconf.DictConfig, self.optimizer_args["optimizer_cfg"]),
                cast(np.ndarray, self.optimizer_args["action_lb"]),
                cast(np.ndarray, self.optimizer_args["action_ub"]),
                planning_horizon=planning_horizon,
                replan_freq=self.replan_freq,
            )

        self.optimizer.reset()

    def act(self, obs: np.ndarray, use_opt=True, **_kwargs) -> np.ndarray:
        """Issues an action given an observation.

        This method optimizes a full sequence of length ``self.planning_horizon`` and returns
        the first action in the sequence. If ``self.replan_freq > 1``, future calls will use
        subsequent actions in the sequence, for ``self.replan_freq`` number of steps.
        After that, the method will plan again, and repeat this process.

        Args:
            obs (np.ndarray): the observation for which the action is needed.

        Returns:
            (np.ndarray): the action.
        """
        if self.reward_fun is None:
            raise RuntimeError(
                "Please call `set_reward_fun()` before using TrajectoryOptimizerAgent"
            )

        def reward_fun(action_sequences, sample=True):
            return self.reward_fun(obs, action_sequences, sample=sample)
        
        def cost_fun(action_sequences, use_grad=False):
            return self.cost_fun(obs, action_sequences, use_grad=use_grad)

        plan = self.optimizer.optimize(reward_fun, cost_fun, use_opt=use_opt)

        action = plan[0]

        return action, plan

    def plan(self, obs: np.ndarray, **_kwargs) -> np.ndarray:
        """Issues a sequence of actions given an observation.

        Returns s sequence of length self.planning_horizon.

        Args:
            obs (np.ndarray): the observation for which the sequence is needed.

        Returns:
            (np.ndarray): a sequence of actions.
        """
        if self.trajectory_eval_fn is None:
            raise RuntimeError(
                "Please call `set_trajectory_eval_fn()` before using TrajectoryOptimizerAgent"
            )

        def reward_fun(action_sequences, sample=True):
            return self.reward_fun(action_sequences, obs, sample=sample)
        
        def cost_fun(action_sequences, use_grad=False):
            return self.cost_fun(action_sequences, obs, use_grad=use_grad)

        plan = self.optimizer.optimize(reward_fun, cost_fun)
        return plan

def create_trajectory_optim_agent_for_model(
    model_env: mbrl.models.ModelEnv,
    agent_cfg: omegaconf.DictConfig,
    num_particles: int = 1,
) -> TrajectoryOptimizerAgent:
    """Utility function for creating a trajectory optimizer agent for a model environment.

    This is a convenience function for creating a :class:`TrajectoryOptimizerAgent`,
    using :meth:`mbrl.models.ModelEnv.evaluate_action_sequences` as its objective function.


    Args:
        model_env (mbrl.models.ModelEnv): the model environment.
        agent_cfg (omegaconf.DictConfig): the agent's configuration.
        num_particles (int): the number of particles for taking averages of action sequences'
            total rewards.

    Returns:
        (:class:`TrajectoryOptimizerAgent`): the agent.

    """
    complete_agent_cfg(model_env, agent_cfg)
    agent = hydra.utils.instantiate(agent_cfg)

    def reward_fun(initial_state, action_sequences, sample=True):
        return model_env.evaluate_action_sequences(
            action_sequences, initial_state=initial_state, num_particles=num_particles,
            sample=sample
        )

    def cost_fun(initial_state, action_sequences, use_grad=False):
        return model_env.compute_objective(initial_state, action_sequences, 
        num_particles=5, use_grad=use_grad
        )

    agent.set_reward_fun(reward_fun)
    agent.set_cost_fun(cost_fun)
    return agent
