from mimetypes import init
import gym
import torch
import numpy as np

import utils.agent_utils as agent_utils
import utils.expert_utils as expert_utils
import utils.env_utils as env_utils

from dataset import Dataset

def hg_dagger(seed, agent, expert, env, start_pose, observation_shape, downsampling_method, render, render_mode):
    algo_name = "HG-DAgger"
    best_model = agent

    num_of_expert_queries = 0

    if render:
        eval_batch_size = 1
    else:
        eval_batch_size = 10

    init_traj_len = 50
    max_traj_len = 10000
    n_batch_updates_per_iter = 1000
    n_iter = 1

    train_batch_size = 64

    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = Dataset()

    log = {'Number of Samples': [], 
           'Number of Expert Queries': [], 
           'Mean Distance Travelled': [],
           'STDEV Distance Travelled': [],
           'Mean Reward': [],
           'STDEV Reward': []}

    rewards = []

    # Perform BC
    for _ in range(init_traj_len):
        pass

    # Perform HG-DAgger
    while True:
        done = False
        observ, step_reward, done, info = env.reset(start_pose)
        if render:
            if env.renderer is None:
                env.render()

        for i in range(max_traj_len):
            pass