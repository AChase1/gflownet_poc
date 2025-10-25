import matplotlib.pyplot as pp
from matplotlib.patches import Arc, Circle



class Face:
    def __init__(self):
        self.axis = pp.gca()
        self.face_properties = []
        self.__create_base_face()
        self.face_actions = {
            'smile': self.create_smile,
            'frown': self.create_frown,
            'left_eyebrow_down': self.create_left_eyebrow_down,
            'right_eyebrow_down': self.create_right_eyebrow_down,
            'left_eyebrow_up': self.create_left_eyebrow_up,
            'right_eyebrow_up': self.create_right_eyebrow_up,
        }
        
        
    def has_overlap(self):
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
        
    def create_smile(self):
        self.face_properties.append(self.__add_curve(x=0.5, y=0.4, width=0.3, height=0.2, angle=0, theta1=200, theta2=340, color='black'))
        self.face_properties.append('smile')
        
    def create_frown(self):
        self.axis.add_patch(self.__add_curve(x=0.5, y=0.3, width=0.3, height=0.2, angle=0, theta1=20, theta2=160, color='black'))
        self.face_properties.append('frown')
        
    def create_left_eyebrow_down(self):
        self.axis.add_line(self.__add_line(x1=0.30, y1=0.70, x2=0.40, y2=0.65, color='black'))
        self.face_properties.append('left_eyebrow_down')

    def create_right_eyebrow_down(self):
        self.axis.add_line(self.__add_line(x1=0.60, y1=0.65, x2=0.70, y2=0.70, color='black'))
        self.face_properties.append('right_eyebrow_down')

    def create_left_eyebrow_up(self):
        self.axis.add_line(self.__add_line(x1=0.30, y1=0.65, x2=0.40, y2=0.70, color='black'))
        self.face_properties.append('left_eyebrow_up')

    def create_right_eyebrow_up(self):
        self.axis.add_line(self.__add_line(x1=0.60, y1=0.70, x2=0.70, y2=0.65, color='black'))
        self.face_properties.append('right_eyebrow_up')

    def __create_base_face(self):
        self.axis.add_patch(self.__add_circle(x=0.5, y=0.5, radius=0.4, color='yellow'))
        self.axis.add_patch(self.__add_circle(x=0.35, y=0.60, radius=0.05, color='black'))
        self.axis.add_patch(self.__add_circle(x=0.65, y=0.60, radius=0.05, color='black'))

    def __add_circle(self, x, y, radius, color):
        return Circle((x, y), radius=radius, color=color)
    
    def __create_curve(self, x, y, width, height, angle, theta1, theta2, color):
        return Arc((x, y), width, height, angle=angle, theta1=theta1, theta2=theta2, color=color)
    
    def __add_line(self, x1, y1, x2, y2, color):
        return pp.Line2D([x1, x2], [y1, y2], color=color)
    
    def show(self):
        pp.axis('equal')
        #pp.axis('off')
        pp.show()
        