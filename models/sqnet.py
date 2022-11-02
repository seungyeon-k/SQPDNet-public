import torch
import torch.nn as nn
import torch.nn.functional as F

class SuperquadricNetwork(nn.Module):

    def __init__(self, backbone, **args):
        super(SuperquadricNetwork, self).__init__()
        self.backbone = backbone
        backbone_out_channel = self.backbone.global_feature_dim
        dict_input_dim = {"input_dim": backbone_out_channel}

        self.net_position = MLP(**args["position"], **dict_input_dim)
        self.net_orientation = MLP(**args["orientation"], **dict_input_dim)
        self.net_size = MLP(**args["size"], **dict_input_dim)
        self.net_shape = MLP(**args["shape"], **dict_input_dim)
        self.output_dim_total = sum(args[key]['output_dim'] for key in args.keys() if hasattr(args[key], 'output_dim'))

    def forward(self, x):
        x = self.backbone.global_feature_map(x)

        # activations
        sigmoid = nn.Sigmoid()

        # position
        x_pos = self.net_position(x)
        
        # orientation
        x_ori = self.net_orientation(x)
        x_ori = F.normalize(x_ori, p=2, dim=1)

        # size
        x_size = self.net_size(x)
        x_size = 0.5 * sigmoid(x_size) + 0.03

        # shape
        x_shape = self.net_shape(x)
        x_shape = 1.5 * sigmoid(x_shape) + 0.2

        # concatenate
        x_cat = torch.cat([x_pos, x_ori, x_size, x_shape], dim=1)

        return x_cat

    def train_step(self, x, y, object_info, position, orientation, mean_xyz, diag_len, optimizer, loss_function, clip_grad=None, **kwargs):
        optimizer.zero_grad()

        output = self(x)

        loss = loss_function(output, y)

        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_grad)
        loss.backward(retain_graph=True)
        
        optimizer.step()

        # input point cloud
        pc = x.detach().cpu().permute([0, 2, 1]).numpy()

        # input ground truth points
        pc_gt = y[:,:3,:].cpu().permute([0, 2, 1]).numpy()
        
        # output (dsq parameters)
        output = output.detach().cpu().numpy()

        # object position and orientation
        position = position.cpu().numpy()
        orientation = orientation.cpu().numpy()

        return {"loss": loss.item(), 
                "pc": pc,
                "pc_gt": pc_gt,
                "output": output,
                "object_info": object_info,
                "position": position,
                "orientation": orientation,
                "mean_xyz": mean_xyz,
                "diag_len": diag_len
        }

    def validation_step(self, x, y, object_info, position, orientation, mean_xyz, diag_len, loss_function, **kwargs):
        output = self(x)

        loss = loss_function(output, y)

        # input point cloud
        pc = x.detach().cpu().permute([0, 2, 1]).numpy()

        # input ground truth points
        pc_gt = y[:,:3,:].cpu().permute([0, 2, 1]).numpy()
        
        # output (dsq parameters)
        output = output.detach().cpu().numpy()

        # object position and orientation
        position = position.cpu().numpy()
        orientation = orientation.cpu().numpy()

        return {"loss": loss.item(), 
                "pc": pc,
                "pc_gt": pc_gt,
                "output": output,
                "object_info": object_info,
                "position": position,
                "orientation": orientation,
                "mean_xyz": mean_xyz,
                "diag_len": diag_len
        }

class MLP(nn.Module):
    def __init__(self, **args):
        super(MLP, self).__init__()
        self.l_hidden = args['l_hidden']
        self.output_dim = args['output_dim']
        self.input_dim = args['input_dim']
        l_neurons = self.l_hidden + [self.output_dim]
        
        l_layer = []
        prev_dim = self.input_dim
        for i, n_hidden in enumerate(l_neurons):
            l_layer.append(nn.Linear(prev_dim, n_hidden))
            if i < len(l_neurons) - 1:
                l_layer.append(nn.LeakyReLU(0.2))
            prev_dim = n_hidden

        self.net = nn.Sequential(*l_layer)

    def forward(self, x):
        x = self.net(x)
        return x
