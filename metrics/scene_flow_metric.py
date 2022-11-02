import numpy as np


class FlowError:
    def __init__(self):
        self.full_error = 0
        self.visible_error = 0
        
        self.n_full_data = 0
        self.n_visible_data = 0

    def update(self, flow_gt, flow_pred, surface_mask):
        self.full_error += np.sum((flow_gt - flow_pred) ** 2)
        self.visible_error += np.sum((flow_gt - flow_pred) ** 2 * surface_mask)
        
        self.n_full_data += flow_pred.size
        self.n_visible_data += np.sum(surface_mask)

    def get_score(self):
        return {
            'Full Flow Error' : self.full_error / self.n_full_data,
            'Visible Flow Error' : self.visible_error / self.n_visible_data
        }
        
    def reset(self):
        self.full_error = 0
        self.visible_error = 0
        
        self.n_full_data = 0
        self.n_visible_data = 0
