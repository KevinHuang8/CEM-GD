# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import sys
import math

from . import termination_fns


def cartpole(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2

    return (~termination_fns.cartpole(act, next_obs)).float().view(-1, 1)

def cartpole_down(act, next_obs):
    assert len(next_obs.shape) == len(act.shape) == 2

    theta = next_obs[:, 2].view(-1, 1) 
    x = next_obs[:, 0].view(-1, 1)

    theta_reward = 0.5*(torch.cos(theta) + 1)
    position_reward = (1/(1 + torch.exp(-20*(x + 2.2)))) * (1/(1 + torch.exp(-20*(2.2 - x))))

    return ((theta_reward*position_reward)
        * ~termination_fns.cartpole_down(act, next_obs))

def cartpole_travel(act, next_obs):
    assert len(next_obs.shape) == len(act.shape) == 2

    goal = 1.2
    x = next_obs[:, 0].view(-1, 1)
    theta = next_obs[:, 2].view(-1, 1)

    done = termination_fns.cartpole_travel(act, next_obs)
    mostly_done = (done.sum() / done.shape[0]) > 0.4
    position_reward = 4*torch.exp(-0.35*(x - 1.6)**2)*(1 + torch.erf(-3*(x - 1.6))) #8 -(x - goal)**2
    theta_reward = torch.exp(-0.6*math.pi*theta**2) #torch.cos(theta) 
    failure_reward = 0#torch.empty_like(x) - 1e5

    return ~done*(position_reward*theta_reward) + done*failure_reward

def obstacles(act, next_obs):
    assert len(next_obs.shape) == len(act.shape) == 2

    gx = 1
    gy = 1

    x = next_obs[:, 0].view(-1, 1)
    y = next_obs[:, 1].view(-1, 1)

    return -(x - gx)**2 + -(y - gy)**2

def obstacles2(act, next_obs):
    assert len(next_obs.shape) == len(act.shape) == 2

    gx = 1
    gy = 0

    x = next_obs[:, 0].view(-1, 1)
    y = next_obs[:, 1].view(-1, 1)

    return -(x - gx)**2 + -5*(y - gy)**2

def drone_linear(act, next_obs):
    assert len(next_obs.shape) == len(act.shape) == 2

    goal = 1.2
    x = next_obs[:, 0].view(-1, 1)

    done = termination_fns.drone_linear(act, next_obs)
    position_reward = torch.exp(-(x - goal)**2)

    return ~done*position_reward

def inverted_pendulum(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2

    return (~termination_fns.inverted_pendulum(act, next_obs)).float().view(-1, 1)


def halfcheetah(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2

    reward_ctrl = -0.1 * act.square().sum(dim=1)
    reward_run = next_obs[:, 0] - 0.0 * next_obs[:, 2].square()

    return (reward_run + reward_ctrl).view(-1, 1)

def humanoid_standup(act, next_obs):
    assert len(next_obs.shape) == len(act.shape) == 2

    pos_after = next_obs[:, 0].view(-1, 1)
    uph_cost = (pos_after - 0) / 0.003

    quad_ctrl_cost = 0.1 * torch.square(act).sum(dim=1).view(-1, 1)

    reward = uph_cost - quad_ctrl_cost + 1

    return reward

def pusher(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    goal_pos = torch.tensor([0.45, -0.05, -0.323]).to(next_obs.device)

    to_w, og_w = 0.5, 1.25
    tip_pos, obj_pos = next_obs[:, 14:17], next_obs[:, 17:20]

    tip_obj_dist = (tip_pos - obj_pos).abs().sum(axis=1)
    obj_goal_dist = (goal_pos - obj_pos).abs().sum(axis=1)
    obs_cost = to_w * tip_obj_dist + og_w * obj_goal_dist

    act_cost = 0.1 * (act ** 2).sum(axis=1)
    
    return -(obs_cost + act_cost).view(-1, 1)

def reacher(act, next_obs):
    theta1, theta2, theta3, theta4, theta5, theta6, _ = (
        next_obs[:, :1].to(next_obs.device),
        next_obs[:, 1:2].to(next_obs.device),
        next_obs[:, 2:3].to(next_obs.device),
        next_obs[:, 3:4].to(next_obs.device),
        next_obs[:, 4:5].to(next_obs.device),
        next_obs[:, 5:6].to(next_obs.device),
        next_obs[:, 6:].to(next_obs.device),
    )

    rot_axis = torch.cat(
        [
            torch.cos(theta2) * torch.cos(theta1),
            torch.cos(theta2) * torch.sin(theta1),
            -torch.sin(theta2),
        ],
        dim=1,
    ).to(next_obs.device)

    rot_perp_axis = torch.cat(
        [-torch.sin(theta1), torch.cos(theta1), torch.zeros(theta1.shape).to(next_obs.device)], dim=1
    ).to(next_obs.device)

    cur_end = torch.cat(
        [
            0.1 * torch.cos(theta1) + 0.4 * torch.cos(theta1) * torch.cos(theta2),
            0.1 * torch.sin(theta1) + 0.4 * torch.sin(theta1) * torch.cos(theta2) - 0.188,
            -0.4 * torch.sin(theta2),
        ],
        dim=1,
    ).to(next_obs.device)

    for length, hinge, roll in [(0.321, theta4, theta3), (0.16828, theta6, theta5)]:
        perp_all_axis = torch.cross(rot_axis, rot_perp_axis)
        x = torch.cos(hinge) * rot_axis
        y = torch.sin(hinge) * torch.sin(roll) * rot_perp_axis
        z = -torch.sin(hinge) * torch.cos(roll) * perp_all_axis
        new_rot_axis = x + y + z
        new_rot_perp_axis = torch.cross(new_rot_axis, rot_axis)
        mask = (torch.norm(new_rot_perp_axis, dim=1) < 1e-30).view(-1, 1)
        new_rot_perp_axis = new_rot_perp_axis - new_rot_perp_axis * mask + rot_perp_axis * mask
        new_rot_perp_axis = new_rot_perp_axis / torch.norm(
            new_rot_perp_axis, dim=1, keepdim=True
        )
        rot_axis, rot_perp_axis, cur_end = (
            new_rot_axis,
            new_rot_perp_axis,
            cur_end + length * new_rot_axis,
        )

    goal = torch.zeros(3).to(next_obs.device)
    reward = -torch.sum(torch.square(cur_end - goal), dim=1)

    return reward.view(-1, 1)