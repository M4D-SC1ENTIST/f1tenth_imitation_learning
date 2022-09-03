import gym
import numpy as np
from PIL import Image
import os, json 

from . import downsampling

def downsample_and_extract_lidar(observ, observation_shape, downsampling_method):
    """Downsamples the lidar data and extracts the relevant features.
    
    Args:
    - observ: the observation dictionary.
    - downsampling_method: the downsampling method to use.
    
    Returns:
    A (T, N) numpy array containing the downsampled lidar data.
    """
    # print("observ: ", observ)
    lidar_scan = observ["scans"][0]
    processed_lidar_scan = downsampling.downsample(lidar_scan, observation_shape, downsampling_method)
    return processed_lidar_scan

def sample_traj(env, policy, start_pose, max_traj_len, observation_shape=108, downsampling_method="simple", render=False):
    """Samples a trajectory of at most `max_traj_len` timesteps by executing a policy.

    Args:
    - env: the environment.
    - policy: the policy.
    - max_traj_len: the maximum number of timesteps in the trajectory.
    - render (optional): whether to render the trajectory as a list of frames.

    Returns:
    A dict of:
    - observs: a (T, O) numpy array containing the observation at each of the T timesteps
        in the trajectory.
    - actions: a (T, A) numpy array containing the action taken at each timestep.
    - reward: the total reward obtained throughout the trajectory.
    - frames (included only if render is set to true): a length-T list of (3, H, W) numpy
        arrays containing the RGB pixel values of the rendered image at each timestep.
    """

    traj = {"observs": [], "poses_x": [], "poses_y": [], "poses_theta": [], "scans": [], "actions": [], "reward": 0}
    if render:
        traj["frames"] = []
    done = False
    observ, step_reward, done, info = env.reset(start_pose)

    # Start rendering
    if env.renderer is None:
        env.render()

    for _ in range(max_traj_len):

        if render:
            traj["frames"].append(env.render(mode='human'))
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
        env.render(mode='human')

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

def sample_trajs(env, policy, start_pose, max_traj_len, n_trajs, observation_shape, downsampling_method):
    """Samples `n_trajs` trajectories by repeatedly calling sample_traj().
    
    Returns:
    A dict of:
    - observs: a (T', O) numpy array containing the observation at each of the T' timesteps
        across all `n_trajs` trajectories.
    - actions: a (T', A) numpy array containing the action taken at each timestep.
    - rewards: a (`n_trajs`, ) numpy array containing the total reward obtained throughout
        each trajectory.
    """
    data = {"observs":[], "poses_x": [], "poses_y": [], "poses_theta": [], "scans": [], "actions":[], "rewards":[]}
    for _ in range(n_trajs):
        traj = sample_traj(env, policy, start_pose, max_traj_len, observation_shape, downsampling_method)
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

def eval(env, policy, start_pose, max_traj_len, eval_batch_size, observation_shape, downsampling_method):
    """Evaluates the performance of a policy over `eval_batch_size` trajectories.
    
    Returns:
    A tuple whose two entries, respectively, are the average and standard deviation of
    the total reward per trajectory.
    """
    rewards = sample_trajs(env, policy, start_pose, max_traj_len, eval_batch_size, observation_shape, downsampling_method)["rewards"]
    return np.mean(rewards), np.std(rewards)

def make_gif(env, policy, filename, max_traj_len):
    """Executes a policy and saves the frames as a GIF."""
    # TODO: change rendering for adapting F1TENTH environment
    frames = sample_traj(env, policy, max_traj_len, render=True)["frames"]
    imgs = [Image.fromarray(frame) for frame in frames]
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    imgs[0].save(filename, save_all=True, append_images=imgs[1:], loop=0)

def make_log(log, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump(log, f)

