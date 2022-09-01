import numpy as np
import utils.downsampling as downsampling

class Dataset(object):
    def __init__(self, downsampling_method):
        self.downsampling_method = downsampling_method
        self.observs, self.actions = None, None

    def add(self, data):
        # TODO: Extract LiDAR scan and downsample it
        assert data["observs"].shape[0] == data["actions"].shape[0]
        if self.observs is None:
            self.observs = data["observs"]
            self.actions = data["actions"]
        else:
            self.observs = np.concatenate([self.observs, data["observs"]])
            self.actions = np.concatenate([self.actions, data["actions"]])

    def sample(self, batch_size):
        idx = np.random.permutation(self.observs.shape[0])[:batch_size]
        return {"observs":self.observs[idx], "actions":self.actions[idx]}