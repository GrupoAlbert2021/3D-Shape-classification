import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool
import torch.nn.functional as F

class GAT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = GATConv(3, 16, heads=8)
        self.conv2 = GATConv(128, 32, heads=8)
        self.conv3 = GATConv(256, 64, heads=8)
        self.fc = nn.Linear(512, 10)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)


        
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
