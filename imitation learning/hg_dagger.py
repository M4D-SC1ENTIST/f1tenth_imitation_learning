from mimetypes import init
import gym
import torch
import numpy as np

import utils.agent_utils as agent_utils
import utils.expert_utils as expert_utils
import utils.env_utils as env_utils

from dataset import Dataset

from bc import bc

def hg_dagger(seed, agent, expert, env, start_pose, observation_shape, downsampling_method, render, render_mode):
    best_model_saving_threshold = 500000

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

    # Bootstrap using BC
    agent, log, dataset = bc(seed, agent, expert, env, start_pose, observation_shape, downsampling_method, render, render_mode, purpose='bootstrap')

    # Perform HG-DAgger
    max_num_of_epoch = 500
    max_num_of_rollout = 5

    for i in range(max_num_of_epoch):
        print("-"*30 + ("\ninitial:" if iter == 0 else "\niter {}:".format(iter)))
        for j in range(max_num_of_rollout):
            # TODO: Evaluation

            # Reset environment
            done = False
            observ, step_reward, done, info = env.reset(start_pose)
            # Start rendering
            if render:
                if env.renderer is None:
                    env.render()
            for k in range(max_traj_len):
                pass
            