import matplotlib.pyplot as pp
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import tqdm

from face import Face

class GFlowNet: 
    def __init__(self):
        self.actions = sorted(Face().face_actions.keys())
        
    def generate_faces(self):
        # Instantiate model and optimizer
        F_sa = FlowModel(512)
        opt = torch.optim.Adam(F_sa.parameters(), 3e-4)

        # Let's keep track of the losses and the faces we sample
        losses = []
        sampled_faces = []
        # To not complicate the code, I'll just accumulate losses here and take a
        # gradient step every `update_freq` episode.
        minibatch_loss = 0
        update_freq = 4
        for episode in tqdm.tqdm(range(50000), ncols=40):
            # Each episode starts with an "empty state"
            state = []
            # Predict F(s, a)
            edge_flow_prediction = F_sa(self.face_to_tensor(state))
            
            for t in range(3):
                # The policy is just normalizing, and gives us the probability of each action
                policy = edge_flow_prediction / edge_flow_prediction.sum()
                # Sample the action
                action = Categorical(probs=policy).sample()
                # "Go" to the next state
                new_state = state + [self.actions[action]]
                print(new_state)

                # Now we want to compute the loss, we'll first enumerate the parents
                parent_states, parent_actions = self.face_parents(new_state)
                # And compute the edge flows F(s, a) of each parent
                px = torch.stack([self.face_to_tensor(p) for p in parent_states])
                pa = torch.tensor(parent_actions).long()
                parent_edge_flow_preds = F_sa(px)[torch.arange(len(parent_states)), pa]
                # Now we need to compute the reward and F(s, a) of the current state,
                # which is currently `new_state`
                if t == 2:
                    # If we've built a complete face, we're done, so the reward is > 0
                    # (unless the face is invalid)
                    reward = self.reward(new_state)
                    # and since there are no children to this state F(s,a) = 0 \forall a
                    edge_flow_prediction = torch.zeros(6)
                else:
                    # Otherwise we keep going, and compute F(s, a)
                    reward = 0
                    edge_flow_prediction = F_sa(self.face_to_tensor(new_state))

                # The loss as per the equation above
                flow_mismatch = (parent_edge_flow_preds.sum() - edge_flow_prediction.sum() - reward).pow(2)
                minibatch_loss += flow_mismatch  # Accumulate
                # Continue iterating
                state = new_state

            # We're done with the episode, add the face to the list, and if we are at an
            # update episode, take a gradient step.
            sampled_faces.append(state)
            if episode % update_freq == 0:
                losses.append(minibatch_loss.item())
                minibatch_loss.backward()
                opt.step()
                opt.zero_grad()
                minibatch_loss = 0
                
            pp.figure(figsize=(10,3))
            pp.plot(losses)
            pp.yscale('log')
            pp.xlabel('Episode')
            pp.ylabel('Loss')
            pp.title('Training Loss Over Time')
            pp.show()

    def reward(self, face: Face):
        if face.has_overlap() or not face.has_two_eyebrows() or not face.has_mouth():
            return 0
        
        if face.is_happy():
            return 3
        
        if face.is_sad(): 
            return 3
        
        if face.is_mad():
            return 2
        
        if face.is_evil():
            return 1
        
        return 1
    
    def face_to_tensor(self, face):
        return torch.tensor([i in face for i in self.actions]).float()
    
    def face_parents(self, state):
        """
        Given a state (list of action names present), return a list of parent
        states and the corresponding parent action indices.

        A parent state is the current state with one action removed. For each
        action present in `state` we produce (parent_state, parent_action).
        This matches the expectation in generate_faces where parent_states is a
        list of state-like lists that can be converted with face_to_tensor and
        stacked into a batch.
        """
        parent_states = []
        parent_actions = []

        # If state is empty there are no parents
        if not state:
            return parent_states, parent_actions

        # For each action in the state, create the parent state formed by
        # removing one occurrence of that action.
        for action in state:
            # create a shallow copy and remove one occurrence of `action`
            parent = list(state)
            try:
                parent.remove(action)
            except ValueError:
                # action not found — skip (shouldn't happen)
                continue
            parent_states.append(parent)
            parent_actions.append(self.actions.index(action))

        return parent_states, parent_actions
    
    

class FlowModel(nn.Module):
    def __init__(self, num_hidden_layers):
        super().__init__()
        self.neural_network = nn.Sequential(
            nn.Linear(6, num_hidden_layers),
            nn.LeakyReLU(),
            nn.Linear(num_hidden_layers, 6),
        )
        
    def forward(self, state):
        return self.neural_network(state).exp() * (1 - state)
    
    
        