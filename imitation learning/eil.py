from mimetypes import init
import gym
import torch
import numpy as np

import utils.agent_utils as agent_utils
import utils.expert_utils as expert_utils
import utils.env_utils as env_utils

from dataset import Dataset

from bc import bc

from policies.agents.agent_mlp_eil import AgentPolicyMLPEIL

def eil(seed, agent, expert, env, start_pose, observation_shape, downsampling_method, render, render_mode):
    algo_name = "EIL"
    best_model = agent
    longest_distance_travelled = 0

    alpha_l = 5
    alpha_e = 5

    alpha_e_count = 0

    prev_controlled_by_expert = False

    num_of_expert_queries = 500

    if render:
        eval_batch_size = 1
    else:
        eval_batch_size = 10

    init_traj_len = 50
    max_traj_len = 3000
    n_batch_updates_per_iter = 1000
    

    train_batch_size = 64

    np.random.seed(seed)
    torch.manual_seed(seed)


    good_dataset = Dataset()
    bad_dataset = Dataset()
    interv_dataset = Dataset()


    log = {'Number of Samples': [], 
           'Number of Expert Queries': [], 
           'Mean Distance Travelled': [],
           'STDEV Distance Travelled': [],
           'Mean Reward': [],
           'STDEV Reward': []}

    # Perform HG-DAgger
    n_iter = 267    # Number of Epochs

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
            if (log['Mean Distance Travelled'][-1] > longest_distance_travelled):
                longest_distance_travelled = log['Mean Distance Travelled'][-1]
                best_model = agent

            print("Number of Samples: {}".format(log['Number of Samples'][-1]))
            print("Number of Expert Queries: {}".format(log['Number of Expert Queries'][-1]))
            print("Distance Travelled: {} (+/- {})".format(log['Mean Distance Travelled'][-1], log['STDEV Distance Travelled'][-1]))
            print("Reward: {} (+/- {})".format(log['Mean Reward'][-1], log['STDEV Reward'][-1]))

            print("- "*15)

            # DELETE IT WHEN DOING SIM2REAL
            if log['Number of Expert Queries'][-1] > 30000:
                break

        
        if iter == n_iter:
            break

        

        if iter == 0:
            # Bootstrap using BC
            bc_agent, log, interv_dataset = bc(seed, agent, expert, env, start_pose, observation_shape, downsampling_method, render, render_mode, purpose='bootstrap')
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            weight_temp = bc_agent.state_dict()
            agent = AgentPolicyMLPEIL(observation_shape, 128, 2, 0.001, device)
            agent.load_state_dict(weight_temp)
            # agent.mlp = bc_agent.mlp
        else:
            # Reset environment
            done = False
            observ, step_reward, done, info = env.reset(start_pose)
            # Start rendering
            if render:
                if env.renderer is None:
                    env.render()
            # Timestep of rollout
            good_traj = {"observs": [], "poses_x": [], "poses_y": [], "poses_theta": [], "scans": [], "actions": [], "reward": 0}
            bad_traj = {"observs": [], "poses_x": [], "poses_y": [], "poses_theta": [], "scans": [], "actions": [], "reward": 0}
            interv_traj = {"observs": [], "poses_x": [], "poses_y": [], "poses_theta": [], "scans": [], "actions": [], "reward": 0}
            for _ in range(max_traj_len):    
                # Extract useful observations
                raw_lidar_scan = observ["scans"][0]
                downsampled_scan = agent_utils.downsample_and_extract_lidar(observ, observation_shape, downsampling_method)

                linear_vels_x = observ["linear_vels_x"][0]

               
                poses_x = observ["poses_x"][0]
                poses_y = observ["poses_y"][0]
                poses_theta = observ["poses_theta"][0]
                curr_expert_speed, curr_expert_steer = expert.plan(poses_x, poses_y, poses_theta, tlad, vgain)
                
                expert_action = np.array([[curr_expert_steer, curr_expert_speed]])
                agent_action_raw = agent.get_action(downsampled_scan)
                agent_action = np.expand_dims(agent_action_raw, axis=0)

                curr_agent_steer = agent_action_raw[0]
                curr_agent_speed = agent_action_raw[1]


                # Decide if agent or expert has control
                if (np.abs(curr_agent_steer - curr_expert_steer) > 0.1) or (np.abs(curr_agent_speed - curr_expert_speed) > 1):
                    """
                    poses_x = observ["poses_x"][0]
                    poses_y = observ["poses_y"][0]
                    poses_theta = observ["poses_theta"][0]

                    curr_expert_speed, curr_expert_steer = expert.plan(poses_x, poses_y, poses_theta, tlad, vgain)
                    curr_action = np.array([[curr_expert_steer, curr_expert_speed]])
                    """
                    if not prev_controlled_by_expert:
                        for i in range(alpha_l):
                            if len(good_traj["observs"]) > 0:
                                temp_observ = good_traj["observs"].pop()
                                temp_scan = good_traj["scans"].pop()
                                temp_poses_x = good_traj["poses_x"].pop()
                                temp_poses_y = good_traj["poses_y"].pop()
                                temp_poses_theta = good_traj["poses_theta"].pop()
                                temp_action = good_traj["actions"].pop()

                                bad_traj["observs"].append(temp_observ)
                                bad_traj["scans"].append(temp_scan)
                                bad_traj["poses_x"].append(temp_poses_x)
                                bad_traj["poses_y"].append(temp_poses_y)
                                bad_traj["poses_theta"].append(temp_poses_theta)
                                bad_traj["actions"].append(temp_action)
                    
                    curr_action = expert_action

                    if alpha_e_count < alpha_e:
                        bad_traj["observs"].append(observ)
                        bad_traj["scans"].append(downsampled_scan)
                        bad_traj["poses_x"].append(observ["poses_x"][0])
                        bad_traj["poses_y"].append(observ["poses_y"][0])
                        bad_traj["poses_theta"].append(observ["poses_theta"][0])
                        bad_traj["actions"].append(curr_action)
                        bad_traj["reward"] += step_reward
                    else:
                        interv_traj["observs"].append(observ)
                        interv_traj["scans"].append(downsampled_scan)
                        interv_traj["poses_x"].append(observ["poses_x"][0])
                        interv_traj["poses_y"].append(observ["poses_y"][0])
                        interv_traj["poses_theta"].append(observ["poses_theta"][0])
                        interv_traj["actions"].append(curr_action)
                        interv_traj["reward"] += step_reward

                    num_of_expert_queries += 1

                    alpha_e_count += 1
                    prev_controlled_by_expert = True
                else:
                    """
                    curr_action_raw = agent.get_action(downsampled_scan)
                    curr_action = np.expand_dims(curr_action_raw, axis=0)
                    """
                    if prev_controlled_by_expert:
                        alpha_e_count = 0

                    curr_action = agent_action

                    good_traj["observs"].append(observ)
                    good_traj["scans"].append(downsampled_scan)
                    good_traj["poses_x"].append(observ["poses_x"][0])
                    good_traj["poses_y"].append(observ["poses_y"][0])
                    good_traj["poses_theta"].append(observ["poses_theta"][0])
                    good_traj["actions"].append(curr_action)
                    good_traj["reward"] += step_reward

                    prev_controlled_by_expert = False
                
                observ, reward, done, _ = env.step(curr_action)

                    # Update rendering
                if render:
                    env.render(mode=render_mode)
                
                if done:
                    break
            
            print("Adding to dataset...")
            if len(good_traj["observs"]) > 0:
                good_traj["observs"] = np.vstack(good_traj["observs"])
                good_traj["poses_x"] = np.vstack(good_traj["poses_x"])
                good_traj["poses_y"] = np.vstack(good_traj["poses_y"])
                good_traj["poses_theta"] = np.vstack(good_traj["poses_theta"])
                good_traj["scans"] = np.vstack(good_traj["scans"])
                good_traj["actions"] = np.vstack(good_traj["actions"])
                good_dataset.add(good_traj)

            if len(bad_traj["observs"]) > 0:
                bad_traj["observs"] = np.vstack(bad_traj["observs"])
                bad_traj["poses_x"] = np.vstack(bad_traj["poses_x"])
                bad_traj["poses_y"] = np.vstack(bad_traj["poses_y"])
                bad_traj["poses_theta"] = np.vstack(bad_traj["poses_theta"])
                bad_traj["scans"] = np.vstack(bad_traj["scans"])
                bad_traj["actions"] = np.vstack(bad_traj["actions"])
                bad_dataset.add(bad_traj)

            if len(interv_traj["observs"]) > 0:
                interv_traj["observs"] = np.vstack(interv_traj["observs"])
                interv_traj["poses_x"] = np.vstack(interv_traj["poses_x"])
                interv_traj["poses_y"] = np.vstack(interv_traj["poses_y"])
                interv_traj["poses_theta"] = np.vstack(interv_traj["poses_theta"])
                interv_traj["scans"] = np.vstack(interv_traj["scans"])
                interv_traj["actions"] = np.vstack(interv_traj["actions"])
                interv_dataset.add(interv_traj)

            log['Number of Samples'].append(good_dataset.get_num_of_total_samples() + bad_dataset.get_num_of_total_samples() + interv_dataset.get_num_of_total_samples())
            log['Number of Expert Queries'].append(num_of_expert_queries)

            print("Training agent...")
            for _ in range(n_batch_updates_per_iter):
                good_train_batch = good_dataset.sample(train_batch_size)
                bad_train_batch = bad_dataset.sample(train_batch_size)
                interv_train_batch = interv_dataset.sample(train_batch_size)
                agent.train(good_train_batch["scans"], good_train_batch["actions"], bad_train_batch["scans"], bad_train_batch["actions"], interv_train_batch["scans"], interv_train_batch["actions"])
    
    agent_utils.save_log_and_model(log, best_model, algo_name)