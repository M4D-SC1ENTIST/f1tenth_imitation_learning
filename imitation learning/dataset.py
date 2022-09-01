import numpy as np
import utils.downsampling as downsampling

class Dataset(object):
    def __init__(self):
        self.observs, self.scans, self.actions = None, None, None

    def add(self, data):
        assert data["observs"].shape[0] == data["actions"].shape[0]
        assert data["observs"].shape[0] == data["scans"].shape[0]
        if self.observs is None:
            self.observs = data["observs"]
            self.scans = data["scans"]
            self.actions = data["actions"]
        else:
            self.observs = np.concatenate([self.observs, data["observs"]])
            self.scans = np.concatenate([self.observs, data["scans"]])
            self.actions = np.concatenate([self.actions, data["actions"]])

    def sample(self, batch_size):
        idx = np.random.permutation(self.scans.shape[0])[:batch_size]
        return {"scans":self.scans[idx], "actions":self.actions[idx]}