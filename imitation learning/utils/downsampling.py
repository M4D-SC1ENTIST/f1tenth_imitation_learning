

def simple_downsampling(data, observation_gap):
    """
    Simple downsampling of data.
    """
    return data[::observation_gap]