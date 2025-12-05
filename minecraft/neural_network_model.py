import torch
import torch.nn as nn

class NeuralNetworkModel(nn.Module):
    def __init__(self, num_inputs, num_hidden):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(self.num_inputs, self.num_hidden), 
            nn.ReLU(),
            nn.Linear(self.num_hidden, self.num_inputs*2)) # outputs num_inputs*2 for forwards and backwards policies
        
        self.logZ = nn.Parameter(torch.ones(1))

    def forward(self, state):
        logits = self.mlp(state)
        
        # calculate the forward policy for each action
        P_F = logits[..., :self.num_inputs] * (1 - state) + state * -100
        P_F = torch.clamp(P_F, min=-50, max=50)
        
        # calculate the backward policy for each action
        P_B = logits[..., self.num_inputs:] * state + (1 - state) * -100
        P_B = torch.clamp(P_B, min=-50, max=50)
        
        return P_F, P_B