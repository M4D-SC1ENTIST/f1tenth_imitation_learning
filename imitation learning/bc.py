from mimetypes import init
import gym
import torch
import numpy as np

import utils.agent_utils as agent_utils
import utils.expert_utils as expert_utils
import utils.env_utils as env_utils

from dataset import Dataset

def bc(seed, agent, expert, env, start_pose, observation_shape, downsampling_method, render, render_mode, purpose):
    algo_name = "Behavioral Cloning"
    best_model = agent
    longest_distance_travelled = 0

    resume_pose = start_pose

    if render:
        eval_batch_size = 1
    else:
        eval_batch_size = 10

    max_traj_len = 10000

    if purpose == "train":
        n_iter = 2000
    elif purpose == "bootstrap":
        n_iter = 1
    else:
        raise ValueError("purpose must be either 'train' or 'bootstrap'")

    num_of_samples_increment = 500
    
    n_batch_updates_per_iter = 1000

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

    # Perform BC
    for iter in range(n_iter + 1):
        if purpose == "train":
            print("-"*30 + ("\ninitial:" if iter == 0 else "\niter {}:".format(iter)))
        else:
            print("-"*30 + "\nbootstrap using BC:")
        

        # Evaluate the agent's performance
        # No evaluation at the initial iteration
        if iter > 0:
            print("Evaluating agent...")
            print("- "*15)
            mean_travelled_distances, stdev_travelled_distances, mean_reward, stdev_reward = agent_utils.eval(env, agent, start_pose, max_traj_len, eval_batch_size, observation_shape, downsampling_method, render, render_mode)
            
            log['Mean Distance Travelled'].append(mean_travelled_distances)
            log['STDEV Distance Travelled'].append(stdev_travelled_distances)
            log['Mean Reward'].append(mean_reward)
            log['STDEV Reward'].append(stdev_reward)
            
            # Replace the best model if the current model is better
            if log['Mean Distance Travelled'][-1] > longest_distance_travelled:
                best_model = agent

            print("Number of Samples: {}".format(log['Number of Samples'][-1]))
            print("Number of Expert Queries: {}".format(log['Number of Expert Queries'][-1]))
            print("Distance Travelled: {} (+/- {})".format(log['Mean Distance Travelled'][-1], log['STDEV Distance Travelled'][-1]))
            print("Reward: {} (+/- {})".format(log['Mean Reward'][-1], log['STDEV Reward'][-1]))

            print("- "*15)
        
        if iter == n_iter:
            break

        tlad = 0.82461887897713965
        vgain = 0.90338203837889


        # Collect data from the expert
        print("Collecting data from the expert...")
        traj = {"observs": [], "poses_x": [], "poses_y": [], "poses_theta": [], "scans": [], "actions": [], "reward": 0}
        done = False
        obs, step_reward, done, info = env.reset(resume_pose)

        # Start rendering
        if render:
            if env.renderer is None:
                env.render()
        
        if purpose == "train":
            step_num = num_of_samples_increment
        else:
            step_num = 100

        for j in range(step_num):
            traj["observs"].append(obs)
            scan = agent_utils.downsample_and_extract_lidar(obs, observation_shape, downsampling_method)
            traj["scans"].append(scan)
            traj["poses_x"].append(obs["poses_x"][0])
            traj["poses_y"].append(obs["poses_y"][0])
            traj["poses_theta"].append(obs["poses_theta"][0])

            speed, steer = expert.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], tlad, vgain)
            action = np.array([[steer, speed]])

            obs, step_reward, done, info = env.step(action)

            # Update rendering
            if render:
                env.render(mode=render_mode)

            traj["actions"].append(action)
            traj["reward"] += step_reward

            if done:
                break
        
        # To evenly sampling using expert by resuming at the last pose in the next iteration
        resume_pose = np.array([[obs["poses_x"][0], obs["poses_y"][0], obs["poses_theta"][0]]])
        
        traj["observs"] = np.vstack(traj["observs"])
        traj["poses_x"] = np.vstack(traj["poses_x"])
        traj["poses_y"] = np.vstack(traj["poses_y"])
        traj["poses_theta"] = np.vstack(traj["poses_theta"])
        traj["scans"] = np.vstack(traj["scans"])
        traj["actions"] = np.vstack(traj["actions"])


        # Adding to datasets
        print("Adding to dataset...")
        dataset.add(traj)

        log['Number of Samples'].append(dataset.get_num_of_total_samples())
        log['Number of Expert Queries'].append(dataset.get_num_of_total_samples())


        # Train the agent
        print("Training agent...")
        for _ in range(n_batch_updates_per_iter):
            train_batch = dataset.sample(train_batch_size)
            agent.train(train_batch["scans"], train_batch["actions"])
        
        if purpose == "bootstrap":
            return agent, log, dataset

    # Save log and the best model
    agent_utils.save_log_and_model(log, best_model, algo_name)