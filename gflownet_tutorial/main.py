from gflownet import GFlowNet
from face import Face
import matplotlib.pyplot as pp

def show_face_examples():
    happy_face = Face()
    happy_face.set_happy()
    happy_face.show()

    sad_face = Face()
    sad_face.set_sad()
    sad_face.show()

    mad_face = Face()
    mad_face.set_mad()
    mad_face.show()

    evil_face = Face()
    evil_face.set_evil()
    evil_face.show()
    
    

if __name__ == "__main__":
    # show_face_examples()
    gFlowNet = GFlowNet()
    gFlowNet.generate_faces(num_faces=150000)  # Train for more iterations
    gFlowNet.show_results(sample_size=500)

    input("\n\nPress ENTER to close all windows")
    pp.close('all')


