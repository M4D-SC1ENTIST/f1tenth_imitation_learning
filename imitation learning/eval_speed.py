import gym
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import os, json
import argparse
import yaml 
import time

import utils.downsampling as downsampling
import utils.env_utils as env_utils

from policies.agents.agent_mlp import AgentPolicyMLP


random_seed = 0

np.random.seed(random_seed)
torch.manual_seed(random_seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

slow_bc_agent = AgentPolicyMLP(108, 128, 2, 0.001, device)
slow_dagger_agent = AgentPolicyMLP(108, 128, 2, 0.001, device)
slow_hg_dagger_agent = AgentPolicyMLP(108, 128, 2, 0.001, device)
slow_eil_agent = AgentPolicyMLP(108, 128, 2, 0.001, device)

normal_bc_agent = AgentPolicyMLP(108, 128, 2, 0.001, device)
normal_dagger_agent = AgentPolicyMLP(108, 128, 2, 0.001, device)
normal_hg_dagger_agent = AgentPolicyMLP(108, 128, 2, 0.001, device)
normal_eil_agent = AgentPolicyMLP(108, 128, 2, 0.001, device)

fast_bc_agent = AgentPolicyMLP(108, 128, 2, 0.001, device)
fast_dagger_agent = AgentPolicyMLP(108, 128, 2, 0.001, device)
fast_hg_dagger_agent = AgentPolicyMLP(108, 128, 2, 0.001, device)
fast_eil_agent = AgentPolicyMLP(108, 128, 2, 0.001, device)



slow_bc_agent.load_state_dict(torch.load('logs/Sim Model/Slow/BehavioralCloning_model.pkl'))
slow_dagger_agent.load_state_dict(torch.load('logs/Sim Model/Slow/DAgger_model.pkl'))
slow_hg_dagger_agent.load_state_dict(torch.load('logs/Sim Model/Slow/HGDAgger_model.pkl'))
slow_eil_agent.load_state_dict(torch.load('logs/Sim Model/Slow/EIL_model.pkl'))

normal_bc_agent.load_state_dict(torch.load('logs/Sim Model/Normal/BehavioralCloning_model.pkl'))
normal_dagger_agent.load_state_dict(torch.load('logs/Sim Model/Normal/DAgger_model.pkl'))
normal_hg_dagger_agent.load_state_dict(torch.load('logs/Sim Model/Normal/HGDAgger_model.pkl'))
normal_eil_agent.load_state_dict(torch.load('logs/Sim Model/Normal/EIL_model.pkl'))

fast_bc_agent.load_state_dict(torch.load('logs/Sim Model/Fast/BehavioralCloning_model.pkl'))
fast_dagger_agent.load_state_dict(torch.load('logs/Sim Model/Fast/DAgger_model.pkl'))
fast_hg_dagger_agent.load_state_dict(torch.load('logs/Sim Model/Fast/HGDAgger_model.pkl'))
fast_eil_agent.load_state_dict(torch.load('logs/Sim Model/Fast/EIL_model.pkl'))

model_dict = {"Slow BC": slow_bc_agent, "Slow DAgger": slow_dagger_agent, "Slow HGDAgger": slow_hg_dagger_agent, "Slow EIL": slow_eil_agent,\
              "Normal BC": normal_bc_agent, "Normal DAgger": normal_dagger_agent, "Normal HGDAgger": normal_hg_dagger_agent, "Normal EIL": normal_eil_agent,\
                "Fast BC": fast_bc_agent, "Fast DAgger": fast_dagger_agent, "Fast HGDAgger": fast_hg_dagger_agent, "Fast EIL": fast_eil_agent}


with open('map/example_map/config_example_map.yaml') as file:
    map_conf_dict = yaml.load(file, Loader=yaml.FullLoader)

map_conf = argparse.Namespace(**map_conf_dict)
env = gym.make('f110_gym:f110-v0', map=map_conf.map_path, map_ext=map_conf.map_ext, num_agents=1)
env.add_render_callback(env_utils.render_callback)

start_pose = np.array([[map_conf.sx, map_conf.sy, map_conf.stheta]])

for key, agent in model_dict.items():
    obs, step_reward, done, info = env.reset(start_pose)

    if env.renderer is None:
        env.render()
    
    while not done:
        raw_lidar_scan = obs["scans"][0]
        processed_lidar_scan = downsampling.downsample(raw_lidar_scan, 108, 'simple')

        action = agent.get_action(processed_lidar_scan)
        action_expand = np.expand_dims(action, axis=0)
        obs, reward, done, _ = env.step(action_expand)

        env.render(mode='human_fast')

    curr_lap_counts = env.lap_counts[0]
    curr_lap_times = env.lap_times[0]

    print("-"*30)
    print("Evaluated Model: ", key)
    print("Total Lap Counts: ", curr_lap_counts)
    print("Total Lap Times: ", curr_lap_times)
    print("Elapsed Time for 1 Round: ", curr_lap_times/curr_lap_counts)
    
