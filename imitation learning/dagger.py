import gym
import torch
import numpy as np

import utils.agent_utils as agent_utils

from dataset import Dataset

def dagger(seed, agent, expert, env, start_pose, observation_shape, downsampling_method):
    env_name = "F1TENTH"

    eval_batch_size = 10

    max_traj_len = 1000
    n_batch_updates_per_iter = 1000
    n_iter = 20

    train_batch_size = 64

    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Dataset(downsampling_method)
    log = {"expert": {}, "agent": {}}

    # TODO: evaluate the expert performance

    # Perform num_iter iterations of DAgger
    for iter in range(n_iter + 1):
        print("-"*30 + ("\ninitial:" if iter == 0 else "\niter {}:".format(iter)))

        # Evaluate the agent's performance
        mean, stdev = agent_utils.eval(env, agent, start_pose, max_traj_len, eval_batch_size, observation_shape, downsampling_method)
        log["agent"][iter] = {"mean reward": mean, "stdev reward": stdev}
        print("agent reward: {} (+/- {})".format(mean, stdev))

        # TODO: change rendering for adapting F1TENTH environment
        # if iter % make_gif_every == 0:
            # utils.make_gif(env, agent, "videos/{}/iter_{}.gif".format(env_name, iter), 150)
        if iter == n_iter:
            break

        # Sample a trajectory with the agent and re-lable actions with the expert
        data = agent_utils.sample_traj(env, agent, max_traj_len)
        
        # Extract necessary input information from observation in the sampled trajectory
        curr_obs = data["observs"]
        curr_pose_x = curr_obs['poses_x'][0]
        curr_pose_y = curr_obs['poses_y'][0]
        curr_pose_theta = curr_obs['poses_theta'][0]

        # tlad and vgain are fixed value for the vehicle dynamics model
        tlad = 0.82461887897713965
        vgain = 0.90338203837889

        # Get expert speed and steer and concat into expert action
        expert_speed, expert_steer = expert.plan(curr_pose_x, curr_pose_y, curr_pose_theta, tlad, vgain)
        expert_action = np.array([[expert_steer, expert_speed]])

        # Replace original action with expert labeled action
        data["actions"] = expert_action

        # Aggregate the datasets
        dataset.add(data)

        # Train the agent
        for _ in range(n_batch_updates_per_iter):
            train_batch = dataset.sample(train_batch_size)

            

            agent.train(train_batch["observs"], train_batch["actions"])

    agent_utils.make_log(log, "logs/{}.json".format(env_name))