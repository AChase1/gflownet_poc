"""
Neural Network Model for GFlowNet edge flow prediction.

This module implements a simple feedforward neural network that predicts
the flow values for all possible actions given a current face state.
"""

import torch
import torch.nn as nn

class NeuralNetworkModel(nn.Module):
    def __init__(self, num_hidden_layers=512):
        super().__init__()
        self.neural_network = nn.Sequential(
            # input layer with 6 potential face properties
            nn.Linear(6, num_hidden_layers),
            nn.LeakyReLU(),
            # output layer with 6 flow predictions for each possible action
            nn.Linear(num_hidden_layers, 6),
        )
        
    def forward(self, state):
        """
        Forward propagation of the neural network
        
        Returns the flow predictions for each possible action.
        """
        # .exp() ensures that the flow predictions are positive
        # (1-state) prevents adding features that are already present
        return self.neural_network(state).exp() * (1-state)