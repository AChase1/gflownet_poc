from gflownet import GFlowNet
from face import Face
import matplotlib.pyplot as pp

def show_face_examples():

    base_face = Face()
    pp.figure(figsize=(4, 4))
    base_face.show()

    happy_face = Face()
    happy_face.set_happy()
    pp.figure(figsize=(4, 4))
    happy_face.show()

    sad_face = Face()
    sad_face.set_sad()
    pp.figure(figsize=(4, 4))
    sad_face.show()

    mad_face = Face()
    mad_face.set_mad()
    pp.figure(figsize=(4, 4))
    mad_face.show()

    evil_face = Face()
    evil_face.set_evil()
    pp.figure(figsize=(4, 4))
    evil_face.show()
    
    

if __name__ == "__main__":
    # show_face_examples()
    gFlowNet = GFlowNet()
    gFlowNet.generate_faces(num_faces=50000)  
    gFlowNet.show_results(sample_size=500)

    input("\n\nPress ENTER to close all windows")
    pp.close('all')


