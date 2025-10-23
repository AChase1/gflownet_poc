import matplotlib.pyplot as pp
from matplotlib.patches import Arc, Circle

class Face:
    def __init__(self):
        self.axis = pp.gca()
        self.__create_base_face()
        self.create_frown()
        self.create_right_eyebrow_up()
        self.create_left_eyebrow_up()
        
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
    
    def show(self):
        pp.axis('equal')
        #pp.axis('off')
        pp.show()