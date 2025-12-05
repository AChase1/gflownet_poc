
from gdpc.editor import Editor
import matplotlib.pyplot as pp
import torch
from torch.distributions.categorical import Categorical
import numpy as np

from minecraft_utils import MinecraftUtils
from house import House
from neural_network_model import NeuralNetworkModel

# WARNING: the current implementation does not work as expected
# 
# Although the house class supports rewarding full house styles (e.g., house.is_farmhouse()),
# the model does not learn to sample houses proportional to their rewards. When testing, even with 
# the rewards for one feature type below, the model tends to overfit to one particular feature (especially if
# the reward for mixed house styles are >= 1). 
#
# Based on research, this appears to be "mode collapse", where the model fails to explore the full search space.
# Since all possible actions are 4^6 = 4096, the model fails to sufficiently sample the 1/4096 path for a single house style to learn its reward.
#
# A potential solution can be found here: https://arxiv.org/html/2511.09677v1#S4
# called Boosted GFlowNets, which trains multiple GFlowNets as boosters which learn to explore the full search space
# based on outputs from other boosters. 
#
# Suggested next steps: Implement Boosted GFlowNets to address mode collapse.

class MinecraftGFlowNet: 
    def __init__(self):
        # used to ensure that any action is sampled 
        # in a consistent order
        self.actions = House().sorted_actions

    def house_reward(self, house):
        """
        Reward function that helps the model learn to sample houses proportional to their rewards.
        """
        # no reward for invalid houses
        if house.has_overlap() or not house.has_roof() or not house.has_wall() or not house.has_door() or not house.has_window() or not house.has_deco1() or not house.has_deco2():
            return 0.0 
        if len(house.props) != 6:
            return 0.0
         
        
        if House.FARM_PORCH in house.props:
            return 4.0
        if House.MEDIEVAL_CHIMNEY in house.props:
            return 3.0   
        if House.HAUNTED_ENTRANCE in house.props:
            return 2.0
        if House.MODERN_GATE in house.props:
            return 1.0
        return 0.0
        
    
    def get_valid_actions(self, house):
        """
        Get valid actions that don't create duplicate components.
        Returns list of action indices.
        """
        if len(house.props) >= 6:
            return []
        
        # get existing component types
        existing_types = set()
        component_types = ['roof', 'wall', 'door', 'window', 'deco1', 'deco2']
        
        for prop in house.props:
            for comp_type in component_types:
                if comp_type in prop:
                    existing_types.add(comp_type)
        
        # filter actions to prevent duplicates
        valid_indices = []
        for idx, action in enumerate(self.actions):
            action_type = None
            for comp_type in component_types:
                if comp_type in action:
                    action_type = comp_type
                    break
            
            # add action if its component type is NOT already present
            if action_type is not None and action_type not in existing_types:
                valid_indices.append(idx)
        
        return valid_indices
        
        
    # uses the Trajectory Balance implementation, based on the following example: https://colab.research.google.com/drive/1fUMwgu2OhYpQagpzU5mhe9_Esib3Q2VR#scrollTo=rdxf1CEfkt8n
    def generate_houses(self, num_houses=50000):
        forward_action_policy = NeuralNetworkModel(num_inputs=len(self.actions), num_hidden=4092)
        
        # optimizer used to train the neural network
        opt = torch.optim.Adam(forward_action_policy.parameters(), 5e-4)
        # scheduler used to decay the learning rate over time, attempt to help with mode collapse
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9999)

        losses = [] 
        sampled_houses = []
        
        # updating/training the neural network every 4 samples is less computationally intensive
        update_frequency = 2  
        minibatch_loss = 0 
        logZs = []
        
        # temperature is used to force exploration (attempt to solve mode collapse, does not work)
        temperature = 10.0
        temperature_decay = 0.9999

        # progress bar to track the training
        for sample in range(num_houses):
            print(f"Training Builders... {sample}/{num_houses} ({sample/num_houses*100:.2f}%)", flush=True)
            
            # each sample starts with the base face 
            house = House()
            
            # the neural network estimates the forward and backward policies for each action given the current house state
            P_F_s, P_B_s = forward_action_policy(house.to_tensor())
            
            total_P_F = 0
            total_P_B = 0
            
            # choose 6 features for the house (roof, wall, door, window, deco1, deco2)
            num_layers = 6
            
            for layer in range(num_layers):
                
                P_F_s_temp = P_F_s / temperature
                
                # mask invalid actions (only pick actions that don't create duplicate components)
                valid_indices = self.get_valid_actions(house)
                if valid_indices:
                    mask = torch.full((len(self.actions),), -1e9)
                    for idx in valid_indices:
                        mask[idx] = 0
                    P_F_s_masked = P_F_s_temp + mask
                else:
                    P_F_s_masked = P_F_s_temp
                
                # sample an action based on the forward policy
                cat = Categorical(logits=P_F_s_masked)
                action = cat.sample()
                
                # ensure action is valid
                if valid_indices and action.item() not in valid_indices:
                    # fallback to first valid action if somehow invalid
                    action = torch.tensor(valid_indices[0])
                
                new_house = house.copy()
                new_house.add_property(self.actions[action.item()])
                
                total_P_F += cat.log_prob(action)
                
                if layer  == 5:
                    # if its the terminal state (completed house), calculate the reward
                    reward = torch.tensor(self.house_reward(new_house)).float()
                    reward = torch.clamp(reward, min=1e-6) 
                    
                else: 
                    # no reward for intermediate states
                    reward = torch.tensor(0.0)
                    
                
                P_F_s, P_B_s = forward_action_policy(new_house.to_tensor())
                
                P_B_s_temp = P_B_s / temperature
                
                total_P_B += Categorical(logits=P_B_s_temp).log_prob(action)
                
                house = new_house
            
            # decay the temperature over time, attempt to help with mode collapse
            temperature = max(0.8, temperature * temperature_decay)
            
            # calculate the loss function using the Trajectory Balance Loss function (https://colab.research.google.com/drive/1fUMwgu2OhYpQagpzU5mhe9_Esib3Q2VR#scrollTo=jHdyswM1uwbo)
            loss = (forward_action_policy.logZ + total_P_F - torch.log(reward + 1e-8).clip(-20) - total_P_B).pow(2)
            minibatch_loss += loss

            sampled_houses.append(house)

            # update the policy (neural network) every 4 samples with the accumulated flow loss
            if sample % update_frequency == 0:
                losses.append(minibatch_loss.item())
                
                # backward propagation to compute gradients 
                minibatch_loss.backward()
                torch.nn.utils.clip_grad_norm_(forward_action_policy.parameters(), 1.0)  # Clip gradients
                # update the weights of the policy using the optimizer
                opt.step()
                scheduler.step()
                
                # reset the gradients and accumulated loss
                opt.zero_grad()  
                minibatch_loss = 0  
                logZs.append(forward_action_policy.logZ.item())

        # store for displaying results
        self.losses = losses
        self.sampled_houses = sampled_houses
        self.policy = forward_action_policy
        self.logZs = logZs

                
    def show_results(self, editor, origin, sample_size=10):
        """
        Display results from the training of the GFlowNet.
        """
        self.plot_losses()
        self.show_sample_houses(editor, origin)
        self.show_house_types(sample_size)

        

    def plot_losses(self):
        """
        Plots the loss differential over time.
        
        Ideally, the loss should decrease as the policy learns to sample faces
        proportional to their rewards.
        """
        
        fig = pp.figure(figsize=(10,6))
        ax = fig.subplots(2, 1, sharex=True)
        pp.sca(ax[0])
        pp.plot(self.losses)
        pp.yscale('log')
        pp.ylabel('loss')
        pp.sca(ax[1])
        pp.plot(np.exp(self.logZs))
        pp.ylabel('estimated Z') 
        pp.savefig("loss_and_logZ_curve.png")
        pp.close(fig)  

    def show_house_types(self, sample_size):
        """
        Displays the ratio of generated houses by their reward categories.
        """
        # categories based on the house_reward() function
        farm_porches = []   
        medieval_chimneys = []   
        modern_gates = []   
        haunted_entrances = []   
        mixed_houses = [] 
        zero_reward_houses = [] 
        
        for house in self.sampled_houses[-sample_size:]:
            reward = self.house_reward(house)
            
            if reward == 0.0:
                zero_reward_houses.append(house)
                continue
            
            if House.FARM_PORCH in house.props:
                farm_porches.append(house)
            if House.MEDIEVAL_CHIMNEY in house.props:
                medieval_chimneys.append(house)
            if House.MODERN_GATE in house.props:
                modern_gates.append(house)
            if House.HAUNTED_ENTRANCE in house.props:
                haunted_entrances.append(house)
            if House.FARM_PORCH not in house.props and House.MEDIEVAL_CHIMNEY not in house.props and House.MODERN_GATE not in house.props and House.HAUNTED_ENTRANCE not in house.props:
                mixed_houses.append(house)
        
        print("\n=== HOUSE TYPE DISTRIBUTION (by reward) ===")
        print(f"Analyzing last {sample_size} houses:")
        print("----------------------------------------")
        
        self._print_reward_category("Farm Porch:", farm_porches, sample_size)
        self._print_reward_category("Medieval Chimney:", medieval_chimneys, sample_size)
        self._print_reward_category("Modern Gate:", modern_gates, sample_size)
        self._print_reward_category("Haunted Entrance:", haunted_entrances, sample_size)
        self._print_reward_category("Invalid:", zero_reward_houses, sample_size)
        
    def _print_reward_category(self, label, house_list, sample_size):
        """Helper to print a category with percentage."""
        percentage = (len(house_list) / sample_size) * 100
        print(f"{label}: {len(house_list)} houses ({percentage:.1f}%)")

    def show_sample_houses(self, editor, start_origin):
        """
        Display the last 10 generated houses in Minecraft with proper spacing.
        """
        if not self.sampled_houses:  
            print("No houses to display")
            return
        
        # show the last 10 houses
        last_houses = self.sampled_houses[-10:]  
        
       
        houses_per_row = 3 
        spacing = 5  
        
        # calculate maximum house dimensions for spacing
        max_width = max(house.width for house in last_houses)
        max_depth = max(house.depth for house in last_houses)
        
        margin = 10
        x_spacing = max_width + spacing + margin
        z_spacing = max_depth + spacing + margin
        
        # display each house with offset (no overlapping houses)
        for i, house in enumerate(last_houses):
            row = i // houses_per_row
            col = i % houses_per_row
            
            offset_x = col * x_spacing
            offset_z = row * z_spacing
            
            house_origin = (
                start_origin[0] + offset_x,
                start_origin[1],
                start_origin[2] + offset_z
            )
            
            print(f"\nHouse {i}--------------------------------")
            print(f"House properties: {house.props}")
            print(f"House reward: {self.house_reward(house)}")
            house.show(editor, house_origin)
        
        editor.flushBuffer()
        

if __name__ == "__main__":
    
    editor = Editor(buffering=True) 
        
    MinecraftUtils.setup_world(editor)

    build_area = editor.getBuildArea()
    
    # start from empty area
    MinecraftUtils.clear_build_area(editor, build_area)
        
    y = MinecraftUtils.get_ground_height(editor)
    x = build_area.offset.x + 5
    z = build_area.offset.z + 5
    
    # helper to see the build area in the world
    #MinecraftUtils.set_build_area_outline(editor, build_area, y)
       
    gflownet = MinecraftGFlowNet()
    gflownet.generate_houses(num_houses=10000)
    gflownet.show_results(editor, (x, y, z), sample_size=100)