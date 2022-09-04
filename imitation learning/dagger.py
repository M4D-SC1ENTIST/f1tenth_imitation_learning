import gym
import torch
import numpy as np

import utils.agent_utils as agent_utils
import utils.expert_utils as expert_utils
import utils.env_utils as env_utils

from dataset import Dataset

def dagger(seed, agent, expert, env, start_pose, observation_shape, downsampling_method, render, render_mode):
    algo_name = "DAgger"

    eval_batch_size = 1

    max_traj_len = 10000
    n_batch_updates_per_iter = 1000
    n_iter = 300

    train_batch_size = 256

    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = Dataset()

    # TODO: modify the log
    log = {"expert": {}, "agent": {}}

    # Perform num_iter iterations of DAgger
    for iter in range(n_iter + 1):
        print("-"*30 + ("\ninitial:" if iter == 0 else "\niter {}:".format(iter)))

        # Evaluate the agent's performance
        print("Evaluating agent...")
        mean, stdev = agent_utils.eval(env, agent, start_pose, max_traj_len, eval_batch_size, observation_shape, downsampling_method, render, render_mode)
        log["agent"][iter] = {"mean reward": mean, "stdev reward": stdev}
        print("agent reward: {} (+/- {})".format(mean, stdev))

        if iter == n_iter:
            break

        # Sample a trajectory with the agent and re-lable actions with the expert
        print("Sampling trajectory...")
        data = agent_utils.sample_traj(env, agent, start_pose, max_traj_len, observation_shape, downsampling_method, render, render_mode)
        
        # tlad and vgain are fixed value for the vehicle dynamics model
        tlad = 0.82461887897713965
        vgain = 0.90338203837889

        # Extract necessary input information from observation in the sampled trajectory
        poses_x = data['poses_x']
        poses_y = data['poses_y']
        poses_theta = data['poses_theta']

        # Get expert speed and steer and concat into expert action
        print("Expert labeling...")
        for idx in range(data['actions'].shape[0]):
            curr_poses_x = poses_x[idx][0]
            curr_poses_y = poses_y[idx][0]
            curr_poses_theta = poses_theta[idx][0]

            curr_expert_speed, curr_expert_steer = expert.plan(curr_poses_x, curr_poses_y, curr_poses_theta, tlad, vgain)
            curr_expert_action = np.array([[curr_expert_steer, curr_expert_speed]])
            # Replace original action with expert labeled action
            data["actions"][idx] = curr_expert_action


        # Aggregate the datasets
        print("Aggregating dataset...")
        dataset.add(data)

        # Train the agent
        print("Training agent...")
        for _ in range(n_batch_updates_per_iter):
            train_batch = dataset.sample(train_batch_size)
            agent.train(train_batch["scans"], train_batch["actions"])

    agent_utils.make_log(log, "logs/{}.json".format(algo_name))