"""
Face class for representing and visualizing facial expressions.

This module implements a simple cartoon face representation where faces are composed
of a base face (circle, eyes) plus optional features that modify the expression:
- Mouth: smile (upward curve) or frown (downward curve)
- Eyebrows: left/right can be raised or lowered independently

The class provides methods to:
1. Build faces by adding properties
2. Check for valid facial expressions
3. Identify specific emotion types (happy, sad, mad, evil)
4. Visualize faces using matplotlib
5. Convert faces to tensor representation for neural networks
"""

import matplotlib.pyplot as pp
import torch as torch
from matplotlib.patches import Arc, Circle



class Face:
    def __init__(self, create_figure=True):
        # used to store the properties that were added to the face
        self.face_properties = []
        self.face_actions = {
            'smile': self.create_smile,
            'frown': self.create_frown,
            'left_eyebrow_down': self.create_left_eyebrow_down,
            'right_eyebrow_down': self.create_right_eyebrow_down,
            'left_eyebrow_up': self.create_left_eyebrow_up,
            'right_eyebrow_up': self.create_right_eyebrow_up,
        }
        
        # used to ensure that any action is sampled in a consistent order
        self.sorted_actions = sorted(self.face_actions.keys()) 
        
    def to_tensor(self):
        """
        Convert face to a binary tensor representation for neural networks.
        
        Returns a 6-element tensor where each element is 1 if that property
        exists in the face, 0 if not.
        
        Example: [1, 0, 0, 1, 0, 0] might represent that only the smile and right eyebrow down exist in the face.
        """
        
        property_flag = []
        for i in self.sorted_actions: # use sorted actions to ensure consistency
            if i in self.face_properties:
                property_flag.append(1)
            else: 
                property_flag.append(0)
        return torch.tensor(property_flag).float()

    def get_parents(self):
        """
        Get all parent states that could have led to this state.
        
        A parent state is created by removing one property from the current state.
        This is used in GFlowNet training to compute the flow conservation loss.
        
        Returns:
            - parent_states: List of Face objects, each missing one property
            - parent_actions: List of indices indicating which property was added to reach this state
        """
        
        parent_states = []
        parent_actions = []
        
        # an empty face has no parents
        if not self.face_properties:
            return parent_states, parent_actions
        
        # create a parent state for each property in the face 
        # by adding all other properties except the current one
        for face_property in self.face_properties:
            parent_face = Face()
            for i in self.face_properties:
                if i != face_property:
                    parent_face.add_property(i)
            
            parent_states.append(parent_face)
            
            # store the action that was used to reach the current state from the parent state
            parent_actions.append(self.sorted_actions.index(face_property))

        return parent_states, parent_actions
    
    def show(self, filename=None):
        """
        Display the face using matplotlib.
        
        Draws the base face first, then adds any additional properties
        (mouth, eyebrows) on top.
        """
        
        self.axis = pp.gca()
        self.__create_base_face()
        for i in self.face_properties:
            self.face_actions[i]()
        pp.axis('equal')
        pp.axis('off')
        
        if filename:
            pp.savefig(filename)
        else: 
            pp.show(block=False) 

    def copy(self):
        new_face = Face()
        new_face.face_properties = self.face_properties.copy()
        return new_face
        
    def has_overlap(self):
        """
        Returns True if the face has conflicting/overlapping properties.
        - both smile and frown
        - both up and down for the same eyebrow
        """
        
        if 'smile' in self.face_properties and 'frown' in self.face_properties:
            return True
        if 'left_eyebrow_up' in self.face_properties and 'left_eyebrow_down' in self.face_properties:
            return True
        if 'right_eyebrow_up' in self.face_properties and 'right_eyebrow_down' in self.face_properties:
            return True
        return False
    
    def has_two_eyebrows(self):
        left = ('left_eyebrow_up' in self.face_properties) or ('left_eyebrow_down' in self.face_properties)
        right = ('right_eyebrow_up' in self.face_properties) or ('right_eyebrow_down' in self.face_properties)
        return left and right
    
    def has_mouth(self):
        return ('smile' in self.face_properties) or ('frown' in self.face_properties)
    
    def is_sad(self):
        return 'frown' in self.face_properties and 'left_eyebrow_up' in self.face_properties and 'right_eyebrow_up' in self.face_properties
    
    def is_mad(self):
        return 'frown' in self.face_properties and 'left_eyebrow_down' in self.face_properties and 'right_eyebrow_down' in self.face_properties
    
    def is_happy(self):
        return 'smile' in self.face_properties and 'left_eyebrow_up' in self.face_properties and 'right_eyebrow_up' in self.face_properties
    
    def is_evil(self):
        return 'smile' in self.face_properties and 'left_eyebrow_down' in self.face_properties and 'right_eyebrow_down' in self.face_properties

    

    def add_property(self, action):
        return self.face_properties.append(action)

    def set_happy(self):
        self.add_property('smile')
        self.add_property('left_eyebrow_up')
        self.add_property('right_eyebrow_up')
    
    def set_sad(self):
        self.add_property('frown')
        self.add_property('left_eyebrow_up')
        self.add_property('right_eyebrow_up')
    
    def set_mad(self):
        self.add_property('frown')
        self.add_property('left_eyebrow_down')
        self.add_property('right_eyebrow_down')
    
    def set_evil(self):
        self.add_property('smile')
        self.add_property('left_eyebrow_down')
        self.add_property('right_eyebrow_down')
        
    def create_smile(self):
        self.axis.add_patch(self.__add_curve(x=0.5, y=0.4, width=0.3, height=0.2, angle=0, theta1=200, theta2=340, color='black'))
        
    def create_frown(self):
        self.axis.add_patch(self.__add_curve(x=0.5, y=0.3, width=0.3, height=0.2, angle=0, theta1=20, theta2=160, color='black'))
        
    def create_left_eyebrow_down(self):
        self.axis.add_line(self.__add_line(x1=0.30, y1=0.70, x2=0.40, y2=0.65, color='black'))

    def create_right_eyebrow_down(self):
        self.axis.add_line(self.__add_line(x1=0.60, y1=0.65, x2=0.70, y2=0.70, color='black'))

    def create_left_eyebrow_up(self):
        self.axis.add_line(self.__add_line(x1=0.30, y1=0.65, x2=0.40, y2=0.70, color='black'))

    def create_right_eyebrow_up(self):
        self.axis.add_line(self.__add_line(x1=0.60, y1=0.70, x2=0.70, y2=0.65, color='black'))

    def __create_base_face(self):
        self.axis.add_patch(self.__add_circle(x=0.5, y=0.5, radius=0.4, color='yellow'))
        self.axis.add_patch(self.__add_circle(x=0.35, y=0.60, radius=0.05, color='black'))
        self.axis.add_patch(self.__add_circle(x=0.65, y=0.60, radius=0.05, color='black'))

    def __add_circle(self, x, y, radius, color):
        return Circle((x, y), radius=radius, color=color)
    
    def __add_curve(self, x, y, width, height, angle, theta1, theta2, color):
        return Arc((x, y), width, height, angle=angle, theta1=theta1, theta2=theta2, color=color)
    
    def __add_line(self, x1, y1, x2, y2, color):
        return pp.Line2D([x1, x2], [y1, y2], color=color)
    
    

        