import gym
import numpy as np
from PIL import Image
import os, json 

from . import downsampling

def downsample_and_extract_lidar(observ, observation_shape, downsampling_method):
    """
    Downsamples the lidar data and extracts the relevant features.
    """
    # print("observ: ", observ)
    lidar_scan = observ["scans"][0]
    processed_lidar_scan = downsampling.downsample(lidar_scan, observation_shape, downsampling_method)
    return processed_lidar_scan

def sample_traj(env, policy, start_pose, max_traj_len, observation_shape=108, downsampling_method="simple", render=True, render_mode="human_fast"):
    """
    Samples a trajectory of at most `max_traj_len` timesteps by executing a policy.
    """

    traj = {"observs": [], "poses_x": [], "poses_y": [], "poses_theta": [], "scans": [], "actions": [], "reward": 0}


    done = False
    observ, step_reward, done, info = env.reset(start_pose)

    # Start rendering
    if render:
        if env.renderer is None:
            env.render()

    for _ in range(max_traj_len):
        traj["observs"].append(observ)
        
        scan = downsample_and_extract_lidar(observ, observation_shape, downsampling_method)
        traj["scans"].append(scan)

        traj["poses_x"].append(observ["poses_x"][0])
        traj["poses_y"].append(observ["poses_y"][0])
        traj["poses_theta"].append(observ["poses_theta"][0])

        action = policy.get_action(scan)

        # TODO: for multi-agent the dimension expansion need to be changed
        action_expand = np.expand_dims(action, axis=0)
        # print("action_expand shape: ", action_expand.shape)
        observ, reward, done, _ = env.step(action_expand)

        # Update rendering
        if render:
            env.render(mode=render_mode)

        traj["actions"].append(action)
        traj["reward"] += reward
        if done:
            break
    traj["observs"] = np.vstack(traj["observs"])
    traj["poses_x"] = np.vstack(traj["poses_x"])
    traj["poses_y"] = np.vstack(traj["poses_y"])
    traj["poses_theta"] = np.vstack(traj["poses_theta"])
    traj["scans"] = np.vstack(traj["scans"])
    traj["actions"] = np.vstack(traj["actions"])
    
    return traj

def sample_trajs(env, policy, start_pose, max_traj_len, n_trajs, observation_shape, downsampling_method, render, render_mode):
    """
    Samples `n_trajs` trajectories by repeatedly calling sample_traj().
    """
    data = {"observs":[], "poses_x": [], "poses_y": [], "poses_theta": [], "scans": [], "actions":[], "rewards":[]}
    for _ in range(n_trajs):
        traj = sample_traj(env, policy, start_pose, max_traj_len, observation_shape, downsampling_method, render, render_mode)
        data["observs"].append(traj["observs"])
        data["poses_x"].append(traj["poses_x"])
        data["poses_y"].append(traj["poses_y"])
        data["poses_theta"].append(traj["poses_theta"])
        data["scans"].append(traj["scans"])
        data["actions"].append(traj["actions"])
        data["rewards"].append(traj["reward"])
    data["observs"] = np.concatenate(data["observs"])
    data["poses_x"] = np.concatenate(data["poses_x"])
    data["poses_y"] = np.concatenate(data["poses_y"])
    data["poses_theta"] = np.concatenate(data["poses_theta"])
    data["scans"] = np.concatenate(data["scans"])
    data["actions"] = np.concatenate(data["actions"])
    data["rewards"] = np.array(data["rewards"])
    return data

def eval(env, policy, start_pose, max_traj_len, eval_batch_size, observation_shape, downsampling_method, render, render_mode):
    """
    Evaluates the performance of a policy over `eval_batch_size` trajectories.
    """
    rewards = sample_trajs(env, policy, start_pose, max_traj_len, eval_batch_size, observation_shape, downsampling_method, render, render_mode)["rewards"]
    return np.mean(rewards), np.std(rewards)


def make_log(log, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump(log, f)

