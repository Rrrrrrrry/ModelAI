import os

from torch_geometric.nn import GCNConv
import torch
import torch.nn.functional as F


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_channels, output_dim):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(input_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x



def train_model(data, epochs=10, lr=0.01, save_path=None):
    model = GCN(input_dim=data.num_features, hidden_channels=16, output_dim=data.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(data.x, data.edge_index)
        loss = criterion(outputs[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
    if save_path is not None:
        torch.save(model, save_path)


def predict(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    return pred, test_acc