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
    longest_distance_travelled = 0

    num_of_expert_queries = 0

    if render:
        eval_batch_size = 1
    else:
        eval_batch_size = 10

    init_traj_len = 50
    max_traj_len = 5000
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

    # Perform HG-DAgger
    n_iter = 100    # Number of Epochs

    n_rollout = 5

    tlad = 0.82461887897713965
    vgain = 0.90338203837889

    # Epochs
    for iter in range(n_iter + 1):
        print("-"*30 + ("\ninitial:" if iter == 0 else "\niter {}:".format(iter)))
        # Evaluation
        if iter > 0:
            print("Evaluating agent...")
            print("- "*15)
            # log["Iteration"].append(iter)
            mean_travelled_distances, stdev_travelled_distances, mean_reward, stdev_reward = agent_utils.eval(env, agent, start_pose, max_traj_len, eval_batch_size, observation_shape, downsampling_method, render, render_mode)
            
            log['Mean Distance Travelled'].append(mean_travelled_distances)
            log['STDEV Distance Travelled'].append(stdev_travelled_distances)
            log['Mean Reward'].append(mean_reward)
            log['STDEV Reward'].append(stdev_reward)
            
            # Replace the best model if the current model is better
            if (log['Mean Distance Travelled'][-1] > longest_distance_travelled) and (log['Number of Samples'][-1] < best_model_saving_threshold):
                longest_distance_travelled = log['Mean Distance Travelled'][-1]
                best_model = agent

            print("Number of Samples: {}".format(log['Number of Samples'][-1]))
            print("Number of Expert Queries: {}".format(log['Number of Expert Queries'][-1]))
            print("Distance Travelled: {} (+/- {})".format(log['Mean Distance Travelled'][-1], log['STDEV Distance Travelled'][-1]))
            print("Reward: {} (+/- {})".format(log['Mean Reward'][-1], log['STDEV Reward'][-1]))

            print("- "*15)

        
        if iter == n_iter:
            break

        if iter == 0:
            # Bootstrap using BC
            agent, log, dataset = bc(seed, agent, expert, env, start_pose, observation_shape, downsampling_method, render, render_mode, purpose='bootstrap')
        else:
            # Rollout
            for j in range(n_rollout):
                print("\nrollout {}:".format(j))
                # Reset environment
                done = False
                observ, step_reward, done, info = env.reset(start_pose)
                # Start rendering
                if render:
                    if env.renderer is None:
                        env.render()
                # Timestep of rollout
                for k in range(max_traj_len):
                    # Extract useful observations
                    raw_lidar_scan = observ["scans"][0]
                    downsampled_scan = agent_utils.downsample_and_extract_lidar(observ, observation_shape, downsampling_method)

                    linear_vels_x = observ["linear_vels_x"][0]

                    # Check TTC
                    will_collide, abs_ittc = agent_utils.check_ittc(raw_lidar_scan, linear_vels_x, ittc_threshold = 0.15)

                    # Decide if agent or expert has control
                    # Expert take control if agent is going to collide
                    if will_collide:
                        print("Expert has control")
                        poses_x = observ["poses_x"][0]
                        poses_y = observ["poses_y"][0]
                        poses_theta = observ["poses_theta"][0]

                        curr_expert_speed, curr_expert_steer = expert.plan(poses_x, poses_y, poses_theta, tlad, vgain)
                        curr_action = np.array([[curr_expert_steer, curr_expert_speed]])
                    else:
                        print("Agent has control")
                        curr_action_raw = agent.get_action(downsampled_scan)
                        curr_action = np.expand_dims(curr_action_raw, axis=0)
                    
                    observ, reward, done, _ = env.step(curr_action)

                     # Update rendering
                    if render:
                        env.render(mode=render_mode)
                    
                    if done:
                        break