"""
Main function for a tutorial on the training of a basic GFlowNet 
to generate different "smiley" faces according to a reward function. 

The purpose is to demonstrate a policy that samples faces proportional to their rewards,
enforcing diverse sampling.


The tutorial is based on the GFlowNet tutorial from the following source:
https://colab.research.google.com/drive/1fUMwgu2OhYpQagpzU5mhe9_Esib3Q2VR

The tutorial is part of the course work for a Directed Studies (BIT4000) under Dr. David Thue and the
RISE Research Group at Carleton University, November 2025.
"""

from gflownet import GFlowNet
import matplotlib.pyplot as pp

if __name__ == "__main__":
    # show_face_examples()
    
    # trains a gflownet through constructing faces
    gFlowNet = GFlowNet()
    gFlowNet.generate_faces(num_faces=50000)
    
    # shows loss differential, the last 64 generated faces, and face type percentages
    gFlowNet.show_results(sample_size=500)

    input("\n\nPress ENTER to close all windows")
    pp.close('all')


