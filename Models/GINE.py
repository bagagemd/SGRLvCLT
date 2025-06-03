import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from torch_geometric.nn import GINEConv, global_add_pool


class GINE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, edge_dim, num_classes):
        super(GINE, self).__init__()

        self.conv1 = GINEConv(
            Sequential(
                Linear(input_dim, hidden_dim),
                BatchNorm1d(hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
                ReLU()
            ),
            edge_dim=edge_dim
        )

        self.conv2 = GINEConv(
            Sequential(
                Linear(hidden_dim, hidden_dim),
                BatchNorm1d(hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
                ReLU()
            ),
            edge_dim=edge_dim
        )

        self.conv3 = GINEConv(
            Sequential(
                Linear(hidden_dim, hidden_dim),
                BatchNorm1d(hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
                ReLU()
            ),
            edge_dim=edge_dim
        )

        self.lin1 = Linear(hidden_dim * 3, hidden_dim * 3)
        self.lin2 = Linear(hidden_dim * 3, num_classes)

    def forward(self, x, edge_index, edge_attr, batch):
        h1 = self.conv1(x, edge_index, edge_attr)
        h2 = self.conv2(h1, edge_index, edge_attr)
        h3 = self.conv3(h2, edge_index, edge_attr)

        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)

        h = torch.cat((h1, h2, h3), dim=1)
        h = F.relu(self.lin1(h))
        h = F.dropout(h, p=0.5, training=self.training)
        out = self.lin2(h)

        return out, F.log_softmax(out, dim=1)
