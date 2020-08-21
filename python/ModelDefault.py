import torch
import torch.nn as nn
import torch_geometric.nn as PyG
import numpy as np

def MLP(channels, batch_norm=True):
    return nn.Sequential(*[
        nn.Sequential(nn.Linear(channels[i - 1], channels[i]), nn.ReLU(), nn.BatchNorm1d(channels[i]))
        for i in range(1, len(channels))
    ])

class PointConvNet(nn.Module):
    def __init__(self, net, **kwargs):
        super(PointConvNet, self).__init__()
        
        self.conv = PyG.PointConv(net)
        
    def forward(self, data, batch=None):
        x, pos, batch, edge_index = data.x, data.pos, data.batch, data.edge_index
        pos_z = pos[:,0].view(-1,1)
        pos_x, pos_y = torch.cos(pos[:,1]).view(-1,1), torch.sin(pos[:,1]).view(-1,1)
        pos = torch.cat([pos_x, pos_y, pos_z], dim=-1)
        x = self.conv(x, pos, edge_index)
        return x, pos, batch

class PoolingNet(nn.Module):
    def __init__(self, net):
        super(PoolingNet, self).__init__()
        self.net = net

    def forward(self, x, pos, batch):
        x = self.net(torch.cat([x, pos], dim=1))
        x = PyG.global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.nChannel = 3
        
        self.conv1 = PointConvNet(MLP([self.nChannel+3, 64, 128]))
        self.pool = PoolingNet(MLP([128+3, 128]))

        self.fc = nn.Sequential(
            nn.Linear( 128, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.5),
            nn.Linear( 64,   1),
        )
        
    def forward(self, data):
        x, pos, batch = self.conv1(data)
        x, pos, batch = self.pool(x, pos, batch)
        out = self.fc(x)
        return out

