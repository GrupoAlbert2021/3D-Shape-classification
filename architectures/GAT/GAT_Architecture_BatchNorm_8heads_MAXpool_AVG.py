import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool
import torch.nn.functional as F

class GAT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = GATConv(3, 16, heads=8, concat=False)
        self.conv2 = GATConv(16, 32, heads=8, concat=False)
        self.conv3 = GATConv(32, 64, heads=8, concat=False)
        self.fc = nn.Linear(64, 10)

        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)


        
    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.relu(self.bn3(self.conv3(x, edge_index)))

        # 2. Readout layer: Aggregate node embeddings into a unified graph embedding 
        x = global_max_pool(x, batch)  # [batch_size=32, hidden_channels=64]

        # 3. Apply a final classifier
        x = self.fc(x)
        
        return x
