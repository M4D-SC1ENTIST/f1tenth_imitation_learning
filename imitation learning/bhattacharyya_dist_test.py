import torch
import numpy as np
import yaml
import argparse
import gym

from dictances import bhattacharyya, bhattacharyya_coefficient
from policies.agents.agent_mlp import AgentPolicyMLP
from policies.experts.expert_waypoint_follower import ExpertWaypointFollower
import utils.env_utils as env_utils


# max_step_num = 5000

# Load agent model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bc_agent = AgentPolicyMLP(1080, 256, 2, 0.001, device)
bc_agent.load_state_dict(torch.load('logs/Sim Models for Bhattacharyya Dist Test/BehavioralCloning/BehavioralCloning_model.pkl'))

dagger_agent = AgentPolicyMLP(1080, 256, 2, 0.001, device)
dagger_agent.load_state_dict(torch.load('logs/Sim Models for Bhattacharyya Dist Test/DAgger/DAgger_svidx_0_dist_249_expsamp_823.pkl'))

hgdagger_agent = AgentPolicyMLP(1080, 256, 2, 0.001, device)
hgdagger_agent.load_state_dict(torch.load('logs/Sim Models for Bhattacharyya Dist Test/HGDAgger/HGDAgger_svidx_253_dist_123_expsamp_4548.pkl'))

eil_agent = AgentPolicyMLP(1080, 256, 2, 0.001, device)
eil_agent.load_state_dict(torch.load('logs/Sim Models for Bhattacharyya Dist Test/EIL/EIL_svidx_247_dist_125_expsamp_8376.pkl'))

# Initialize dictionaries
# state_dict = {'idx': [],
#               'poses_x': [],
#               'poses_y': [],
#               'poses_theta': [],
#               'scans': []}

# expert_speed_dict = {}

expert_steer_dict = {}




# bc_agent_speed_dict = {}

bc_agent_steer_dict = {}

# dagger_agent_speed_dict = {}

dagger_agent_steer_dict = {}

# hgdagger_agent_speed_dict = {}

hgdagger_agent_steer_dict = {}

# eil_agent_speed_dict = {}

eil_agent_steer_dict = {}

with open('map/gene_eval_map/config_gene_map.yaml') as file:
    map_conf_dict = yaml.load(file, Loader=yaml.FullLoader)

map_conf = argparse.Namespace(**map_conf_dict)
env = gym.make('f110_gym:f110-v0', map=map_conf.map_path, map_ext=map_conf.map_ext, num_agents=1)
env.add_render_callback(env_utils.render_callback)
start_pose = np.array([[map_conf.sx, map_conf.sy, map_conf.stheta]])
expert = ExpertWaypointFollower(map_conf)

tlad = 0.82461887897713965
vgain = 0.90338203837889

observ, step_reward, done, info = env.reset(start_pose)

curr_idx = 0
while not done:
    # state_dict['idx'].append(i)

    poses_x = observ["poses_x"][0]
    poses_y = observ["poses_y"][0]
    poses_theta = observ["poses_theta"][0]
    scan = observ["scans"][0]

    # Get expert action
    curr_expert_speed, curr_expert_steer = expert.plan(poses_x, poses_y, poses_theta, tlad, vgain)

    # Log expert action
    # expert_speed_dict['idx'].append(curr_idx)
    # expert_speed_dict['speed'].append(curr_expert_speed)
    # expert_steer_dict['idx'].append(curr_idx)
    # expert_steer_dict['steer'].append(curr_expert_steer)
    expert_steer_dict[str(curr_idx)] = curr_expert_steer

    # Concat expert action
    expert_action = np.array([[curr_expert_steer, curr_expert_speed]])

    # Get agent action
    bc_agent_speed, bc_agent_steer = bc_agent.get_action(scan)
    # bc_agent_speed_dict['idx'].append(curr_idx)
    # bc_agent_speed_dict['speed'].append(bc_agent_speed)
    # bc_agent_steer_dict['idx'].append(curr_idx)
    # bc_agent_steer_dict['steer'].append(bc_agent_steer)
    bc_agent_steer_dict[str(curr_idx)] = bc_agent_steer

    dagger_agent_speed, dagger_agent_steer = dagger_agent.get_action(scan)
    # dagger_agent_speed_dict['idx'].append(curr_idx)
    # dagger_agent_speed_dict['speed'].append(dagger_agent_speed)
    # dagger_agent_steer_dict['idx'].append(curr_idx)
    # dagger_agent_steer_dict['steer'].append(dagger_agent_steer)
    dagger_agent_steer_dict[str(curr_idx)] = dagger_agent_steer

    hgdagger_agent_speed, hgdagger_agent_steer = hgdagger_agent.get_action(scan)
    # hgdagger_agent_speed_dict['idx'].append(curr_idx)
    # hgdagger_agent_speed_dict['speed'].append(hgdagger_agent_speed)
    # hgdagger_agent_steer_dict['idx'].append(curr_idx)
    # hgdagger_agent_steer_dict['steer'].append(hgdagger_agent_steer)
    hgdagger_agent_steer_dict[str(curr_idx)] = hgdagger_agent_steer

    eil_agent_speed, eil_agent_steer = eil_agent.get_action(scan)
    # eil_agent_speed_dict['idx'].append(curr_idx)
    # eil_agent_speed_dict['speed'].append(eil_agent_speed)
    # eil_agent_steer_dict['idx'].append(curr_idx)
    # eil_agent_steer_dict['steer'].append(eil_agent_steer)
    eil_agent_steer_dict[str(curr_idx)] = eil_agent_steer





    # Step environment
    observ, step_reward, done, info = env.step(expert_action)
    env.render(mode='human_fast')

    if env.lap_counts[0] > 0:
        break

# Calculate bhattacharyya distance

bc_bhattacharyya = bhattacharyya(expert_steer_dict, bc_agent_steer_dict)
dagger_bhattaryya = bhattacharyya(expert_steer_dict, dagger_agent_steer_dict)
hgdagger_bhattacharyya = bhattacharyya(expert_steer_dict, hgdagger_agent_steer_dict)
eil_bhattacharyya = bhattacharyya(expert_steer_dict, eil_agent_steer_dict)


print('BC Bhattacharyya Metric: ', bc_bhattacharyya)
print('Dagger Bhattacharyya Metric: ', dagger_bhattaryya)
print('HG-Dagger Bhattacharyya Metric: ', hgdagger_bhattacharyya)
print('EIL Bhattacharyya Metric: ', eil_bhattacharyya)
                