import matplotlib.pyplot as pp
from matplotlib.patches import Arc, Circle
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import sys

from face import Face
from neural_network_model import NeuralNetworkModel

class GFlowNet: 
    def __init__(self):
        # used to ensure that any action is sampled 
        # in a consistent order
        self.actions = Face().sorted_actions

    def face_reward(self, face: Face):
        """
        Returns the reward for a given face configuration.
        
        The policy learns to sample faces proportional to the
        reward distribution of the face configurations.
        """
        # invalid faces should not be sampled
        if face.has_overlap() or not face.has_mouth() or not face.has_two_eyebrows():
            return 0
        
        # modify the values below to change the reward distribution
        if face.is_happy():
            return 4   
        elif face.is_sad():
            return 2  
        elif face.is_mad():
            return 2 
        elif face.is_evil():
            return 1  
        else:
            return 0
        
        
    def generate_faces(self, num_faces=50000):
        """
        Generates `num_faces` number of faces. Main training loop for the GFlowNet.
        
        This method implements the core GFlowNet training algorithm:
        1. Initialize a neural network to predict edge flows
        2. For each training step:
           - Start with an empty face
           - Sequentially add 3 face properties to build a complete face
               - At each step, sample an action based on the current policy
           - Calculate the flow matching loss to train the network
        3. Update the network parameters periodically using minibatches
        """

        
        forward_action_policy = NeuralNetworkModel(512)
        
        # optimizer used to train the neural network
        opt = torch.optim.Adam(forward_action_policy.parameters(), 3e-4)  # Adam optimizer

        losses = [] 
        sampled_faces = []
        
        # updating/training the neural network every 4 samples is less computationally intensive
        update_frequency = 4  
        minibatch_loss = 0  

        # progress bar to track the training
        for sample in range(num_faces):
            print(f"Training: {sample}/{num_faces} ({sample/num_faces*100:.2f}%)", flush=True)
            # each sample starts with the base face 
            face = Face()
            
            # the neural network estimates the flow for each action given the current face
            current_edge_flow_prediction = forward_action_policy(face.to_tensor())
            
            # build face by adding 3 face properties (smile/frown + 2 eyebrows)
            num_layers = 3
            for layer in range(num_layers):
                
                # get the probability of each action
                policy = current_edge_flow_prediction / current_edge_flow_prediction.sum()
                
                # choose an action based on the policy
                action = Categorical(probs=policy).sample()
                new_face = face.copy()
                new_face.add_property(self.actions[action.item()])

                # to ensure actions are proportional to their reward, we must maintain flow conservation.
                # 
                # the "flow" leaving a current state to its children (next face property sampling)
                # must equal the "flow" entering the current state from its parents (previous face property sampling).
                #
                # therefore we must calculate the flow entering the current state from its parents
                # and use that in the loss function to verify its accuracy.
                parent_states, parent_actions = new_face.get_parents()
                
                parent_tensors = []
                for parent in parent_states:
                    parent_tensors.append(parent.to_tensor()) 
                px = torch.stack(parent_tensors)
                pa = torch.tensor(parent_actions).long()
                
                # the neural network estimates the flow from each parent state to the current state
                parent_edge_flow_predictions = forward_action_policy(px)[torch.arange(len(parent_states)), pa]

                if layer  == 2:
                    # if its the terminal state (completed face), calculate the reward
                    reward = self.face_reward(new_face)
                    
                    # the terminal state has no outgoing flow
                    current_edge_flow_prediction = torch.zeros(6) 
                else:
                    # intermediate layers have no reward, just flow through
                    reward = 0
                    current_edge_flow_prediction = forward_action_policy(new_face.to_tensor())
                
                # loss function as a means squared error
                # sum of incoming flows should equal sum of outgoing flows + reward
                flow_loss = (parent_edge_flow_predictions.sum() - current_edge_flow_prediction.sum() - reward).pow(2)
                
                minibatch_loss += flow_loss 
                face = new_face

            sampled_faces.append(face)

            # update the policy (neural network) every 4 samplesw
            # with the accumulated flow loss
            if sample % update_frequency == 0:
                losses.append(minibatch_loss.item())
                
                # backward propagation to compute gradients 
                minibatch_loss.backward()
                # update the weights of the policy using the optimizer
                opt.step()
                
                # reset the gradients and accumulated loss
                opt.zero_grad()  
                minibatch_loss = 0  

        # store for displaying results
        self.losses = losses
        self.sampled_faces = sampled_faces
                
    def show_results(self, sample_size=128):
        """
        Display results from the training of the GFlowNet.
        
        Shows the following:
        - the loss differential over time
        - the last 64 generated faces
        - the ratio of generated faces for each face type
        """
        self.plot_losses()
        self.show_sample_faces()
        
        print(f"|\nRESULTS\n|----------------------------------|\n")
        self.show_face_types(sample_size)
        

    def plot_losses(self):
        """
        Plots the loss differential over time.
        
        Ideally, the loss should decrease as the policy learns to sample faces
        proportional to their rewards.
        """
        
        fig = pp.figure(figsize=(10,3))
        pp.plot(self.losses)
        pp.yscale('log')  
        pp.savefig("loss_curve.png")
        pp.close(fig)  # Free memory

    def show_face_types(self, sample_size):
        """
        Displays the ratio of generated faces for each face type generated
        in the last `sample_size` number of samples.
        """
        valid_faces, happy_faces, sad_faces, mad_faces, evil_faces = [], [], [], [], []

        # sorts the last `sample_size` number of generated faces by their face type
        for face in self.sampled_faces[-sample_size:]:
            if self.face_reward(face) > 0:
                valid_faces.append(face)
            if face.is_happy():
                happy_faces.append(face)
            if face.is_sad():
                sad_faces.append(face)
            if face.is_mad():
                mad_faces.append(face)
            if face.is_evil():
                evil_faces.append(face)

        self.show_face_percentage("valid", valid_faces, sample_size)
        self.show_face_percentage("happy", happy_faces, sample_size)
        self.show_face_percentage("sad", sad_faces, sample_size)
        self.show_face_percentage("mad", mad_faces, sample_size)
        self.show_face_percentage("evil", evil_faces, sample_size)

    def show_sample_faces(self):
        """
        Display a 8x8 grid of the last 64 generated faces.
        """
        f, ax = pp.subplots(8,8,figsize=(4,4))
        for i, face in enumerate(self.sampled_faces[-64:]):
            pp.sca(ax[i//8,i%8])
            face.show()
        pp.savefig("generated_faces.png")
        pp.close(f)

    def show_face_percentage(self, face_type, face_list, sample_size):
        print(f" {face_type} faces: {self.get_face_percentage(face_list, sample_size)}%")
    
    def get_face_percentage(self, face_list, sample_size):
        return (len(face_list)/sample_size) * 100
    


    
    
        