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
    n_iter = 500

    train_batch_size = 64

    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = Dataset()

    log = {'Iteration': [],
           'Number of Samples': [], 
           'Number of Expert Queries': [], 
           'Mean Distance Travelled': [],
           'STDEV Distance Travelled': [],
           'Mean Reward': [],
           'STDEV Reward': []}

    # Perform num_iter iterations of DAgger
    for iter in range(n_iter + 1):
        print("-"*30 + ("\ninitial:" if iter == 0 else "\niter {}:".format(iter)))


        # Evaluate the agent's performance
        # No evaluation at the initial iteration
        if iter > 0:
            print("Evaluating agent...")
            print("- "*15)
            log["Iteration"].append(iter)
            mean_travelled_distances, stdev_travelled_distances, mean_reward, stdev_reward = agent_utils.eval(env, agent, start_pose, max_traj_len, eval_batch_size, observation_shape, downsampling_method, render, render_mode)
            
            log['Mean Distance Travelled'].append(mean_travelled_distances)
            log['STDEV Distance Travelled'].append(stdev_travelled_distances)
            log['Mean Reward'].append(mean_reward)
            log['STDEV Reward'].append(stdev_reward)
            
            # Replace the best model if the current model is better
            if  len(log['Mean Distance Travelled']) >= 2:
                if log['Mean Distance Travelled'][-1] > log['Mean Distance Travelled'][-2]:
                    best_model = agent

            print("Number of Samples: {}".format(log['Number of Samples'][-1]))
            print("Number of Expert Queries: {}".format(log['Number of Expert Queries'][-1]))
            print("Distance Travelled: {} (+/- {})".format(log['Mean Distance Travelled'][-1], log['STDEV Distance Travelled'][-1]))
            print("Reward: {} (+/- {})".format(log['Mean Reward'][-1], log['STDEV Reward'][-1]))

            print("- "*15)
        if iter == n_iter:
            break


        # Sample a trajectory with the agent and re-lable actions with the expert
        print("Sampling trajectory...")

        # Disable render for the initial iteration as it takes too much time
        # The max trajectory length is also different in the initial iteration
        if iter == 0:
            data = agent_utils.sample_traj(env, agent, start_pose, init_traj_len, observation_shape, downsampling_method, render=False, render_mode=None)
        else:
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

            num_of_expert_queries += 1


        # Aggregate the datasets
        print("Aggregating dataset...")
        dataset.add(data)

        log['Number of Samples'].append(dataset.get_num_of_total_samples())
        log['Number of Expert Queries'].append(num_of_expert_queries)


        # Train the agent
        print("Training agent...")
        for _ in range(n_batch_updates_per_iter):
            train_batch = dataset.sample(train_batch_size)
            agent.train(train_batch["scans"], train_batch["actions"])

    # Save log and the best model
    agent_utils.save_log_and_model(log, best_model, algo_name)