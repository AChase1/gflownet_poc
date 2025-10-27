import torch
import torch.nn as nn

class NeuralNetworkModel(nn.Module):
    def __init__(self, num_hidden_layers=512):
        super().__init__()
        self.neural_network = nn.Sequential(
            nn.Linear(6, num_hidden_layers),
            nn.ReLU(),
            nn.Linear(num_hidden_layers, 6),
        )
        
    def forward(self, state):
        return self.neural_network(state).exp() * (1-state)