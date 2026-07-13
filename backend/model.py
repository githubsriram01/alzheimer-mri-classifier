# model.py
# This defines the exact same architecture used during training in the notebook
# (Cells 6-7). The layer names and shapes here MUST match the saved checkpoint,
# otherwise load_state_dict() will fail.

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from torch_geometric.nn import GATConv, SAGEConv, global_mean_pool, global_max_pool

N_QUBITS = 8
N_LAYERS = 4

# Quantum simulator device (runs on CPU internally regardless of PyTorch device)
qdev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(qdev, interface="torch")
def circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


weight_shapes = {"weights": (N_LAYERS, N_QUBITS, 3)}


class QuantumFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        x = torch.tanh(x) * torch.pi
        return self.qlayer(x)


class QuantumGraphSAGE(nn.Module):
    """
    Hybrid Quantum-GNN:
    QFE (PennyLane 8-qubit) -> Linear -> GATConv(4-head) -> SAGEConv x2
    -> Mean+Max pooling -> MLP classifier
    """

    def __init__(self, in_channels=8, hidden_channels=128,
                 out_channels=4, n_qubits=8, n_layers=4):
        super().__init__()
        self.qfe = QuantumFeatureExtractor()
        self.input_proj = nn.Linear(n_qubits, hidden_channels)
        self.conv1 = GATConv(hidden_channels, hidden_channels // 2, heads=4, concat=True)
        self.bn1 = nn.BatchNorm1d(hidden_channels * 2)
        self.conv2 = SAGEConv(hidden_channels * 2, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels // 2, out_channels),
        )

    def forward(self, data):
        x, ei, b = data.x, data.edge_index, data.batch
        x = self.qfe(x)
        x = F.relu(self.input_proj(x))
        x = F.elu(self.bn1(self.conv1(x, ei)))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, ei)))
        x = F.relu(self.bn3(self.conv3(x, ei)))
        x = torch.cat([global_mean_pool(x, b), global_max_pool(x, b)], dim=1)
        return self.classifier(x)

    def get_node_embeddings(self, data):
        x, ei = data.x, data.edge_index
        x = self.qfe(x)
        x = F.relu(self.input_proj(x))
        x = F.elu(self.bn1(self.conv1(x, ei)))
        x = F.relu(self.bn2(self.conv2(x, ei)))
        x = F.relu(self.bn3(self.conv3(x, ei)))
        return x
