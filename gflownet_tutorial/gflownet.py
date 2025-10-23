from face import Face

class GFlowNet: 
    def __init__(self):
        pass
    
    def reward(self, face: Face):
         
        if face.has_overlap() or not face.has_two_eyebrows() or not face.has_mouth():
            return 0
        
        if face.is_happy():
            return 3
        
        if face.is_sad(): 
            return 2
        
        if face.is_mad():
            return 2
        
        if face.is_evil():
            return 1
        
        return 1
        
        