

from concurrent.futures import process


def downsample(data, observation_shape, downsampling_method):
    if downsampling_method == "simple":
        obs_gap = int(1080/observation_shape)
        processed_data = data[::obs_gap]
    else:
        processed_data = data
    return processed_data
