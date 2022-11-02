import numpy as np


class MotionPredictionError:
    def __init__(self, motion_dim):
        self.motion_dim = motion_dim
        self.position_error = 0
        self.orientation_error = 0
        self.n_data = 0

    def update(self, labels_true, labels_pred):
        if self.motion_dim == '2D':
            self.position_error += np.sum(np.sqrt(np.sum((labels_true[:, :2] - labels_pred[:, :2]) ** 2, axis=1)))
            self.orientation_error += np.sum(np.abs((labels_true[:, 2] - labels_pred[:, 2]) * 180 / np.pi))
        elif self.motion_dim == '3D':
            self.position_error += np.sum(np.sqrt(np.sum((labels_true[:, :3] - labels_pred[:, :3]) ** 2, axis=1)))
            self.orientation_error += np.sum(np.arccos(np.clip(2 * np.sum(labels_true[:, 3:7] * labels_pred[:, 3:7], axis=1) ** 2 - 1, -1, 1)) * 180 / np.pi)
        self.n_data += len(labels_true)

    def get_scores(self):
        return {
            "Position Error": self.position_error / self.n_data,
            "Orientation Error": self.orientation_error / self.n_data,
        }

    def reset(self):
        self.position_error = 0
        self.orientation_error = 0
        self.n_data = 0
