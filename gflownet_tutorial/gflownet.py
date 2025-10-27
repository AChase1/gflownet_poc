import matplotlib.pyplot as pp
from matplotlib.patches import Arc, Circle
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import tqdm
import math

from face import Face
from neural_network_model import NeuralNetworkModel

class GFlowNet: 
    def __init__(self):
        
        self.actions = sorted(Face().face_actions.keys())

    def face_reward(self, face: Face):
        if face.has_overlap() or not face.has_mouth() or not face.has_two_eyebrows():
            return 0
        
        if face.is_happy():
            return 4   
        elif face.is_sad():
            return 2  
        elif face.is_mad():
            return 2 
        elif face.is_evil():
            return 1  
        else:
            return 1
        
        
    def generate_faces(self, num_faces=50000):

        forward_action_policy = NeuralNetworkModel(512)
        opt = torch.optim.Adam(forward_action_policy.parameters(), 3e-4) 

        losses = []
        sampled_faces = [] 
        update_frequency = 4

        for sample in tqdm.tqdm(range(num_faces), ncols=40):
            
            face = Face()
            current_edge_flow_prediction = forward_action_policy(face.to_tensor())
            num_layers = 3
            minibatch_loss = torch.tensor(0.0)
            
            for layer in range(num_layers):
                policy = current_edge_flow_prediction / current_edge_flow_prediction.sum()
                action = Categorical(probs=policy).sample()
                face.add_property(self.actions[action.item()])

                
                parent_states, parent_actions = face.get_parents()
                
                parent_tensors = []
                for parent in parent_states:
                    parent_tensors.append(parent.to_tensor()) 

                px = torch.stack(parent_tensors)
                pa = torch.tensor(parent_actions).long()
                parent_edge_flow_predictions = forward_action_policy(px)[torch.arange(len(parent_states)), pa]


                if layer  == 2:
                    reward = self.face_reward(face)
                    current_edge_flow_prediction = torch.zeros(6)
                else:
                    reward = 0
                    current_edge_flow_prediction = forward_action_policy(face.to_tensor())
                
                flow_mismatch = (parent_edge_flow_predictions.sum() - current_edge_flow_prediction.sum() - reward).pow(2)
                minibatch_loss += flow_mismatch  
                
            sampled_faces.append(face)

            if sample % update_frequency == 0:
                losses.append(minibatch_loss.item())
                minibatch_loss.backward()
                opt.step()
                opt.zero_grad()

        self.losses = losses
        self.sampled_faces = sampled_faces
                
    def show_results(self, sample_size=128):
        self.plot_losses()
        print(f"|\nRESULTS\n|----------------------------------|\n")
        self.show_face_types(sample_size)
        self.show_sample_faces(sample_size)

    def plot_losses(self):
        pp.figure(figsize=(10,3))
        pp.plot(self.losses)
        pp.yscale('log')
        pp.show(block=False)

    def show_face_types(self, sample_size):
        valid_faces, happy_faces, sad_faces, mad_faces, evil_faces = [], [], [], [], []

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

    def show_sample_faces(self, sample_size):
        grid_size = int(math.sqrt(sample_size))
        f, ax = pp.subplots(8,8,figsize=(4,4))
        for i, face in enumerate(self.sampled_faces[-64:]):
            pp.sca(ax[i//8,i%8])
            face.show()
        pp.show(block=False)

    def show_face_percentage(self, face_type, face_list, sample_size):
        print(f" {face_type} faces: {self.get_face_percentage(face_list, sample_size)}%")
    
    def get_face_percentage(self, face_list, sample_size):
        return (len(face_list)/sample_size) * 100
    


    
    
        