import matplotlib.pyplot as pp
from face import Face

def show_face_examples():
    """
    Helper function to display example faces.
    
    Shows the base face and the four main emotion types:
    - Happy: smile + raised eyebrows
    - Sad: frown + raised eyebrows
    - Mad: frown + lowered eyebrows
    - Evil: smile + lowered eyebrows
    """
    
    face_types = {
        "base": None,
        "happy": "set_happy",
        "sad": "set_sad",
        "mad": "set_mad",
        "evil": "set_evil"
    }
        
    for face_name, method in face_types.items():
        face = Face()
        if method:
            getattr(face, method)()
        
        pp.figure(figsize=(4, 4))
        filename = f"{face_name}.png"
        face.show(filename=filename)