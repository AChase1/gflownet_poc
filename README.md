### Table of Contents
> [GFlowNet Proof-Of-Concept](#gflownet-proof-of-concept)<br>
> [GFlowNet Tutorial](#gflownet-tutorial)<br>
>     > [Description](#description)<br>
>     > [How To Use](#how-to-use)<br>
> [Authors](#authors)

</br>

# GFlowNet Proof-Of-Concept
#### A new machine learning framework that sequentially constructs composite samples proportional to a reward function

This repository implements a proof-of-concept for the relatively new machine learning framework Generative Flow Networks, created by Dr. Yoshua Bengio and others out of the University of Montreal. 
Generative Flow Networks (GFlowNet) is an alternative version of a Markov Decision Process (MDP) that produces **diverse solutions** from enforcing policies to choose actions **proportional** to a reward function. 

Related Resources: 
- [The GFlowNet Tutorial (Notion)](https://milayb.notion.site/The-GFlowNet-Tutorial-95434ef0e2d94c24aab90e69b30be9b3)
- [Google Collab Code Tutorial](https://colab.research.google.com/drive/1fUMwgu2OhYpQagpzU5mhe9_Esib3Q2VR)
- [GFlowNet Foundations](https://arxiv.org/abs/2111.09266)

</br>
</br>

## GFlowNet Tutorial

</br>

### Description
This tutorial attempts to demonstrate a basic working GFlowNet by generating different "smiley" faces proportional to its specified reward.

Its purpose it to show a very simplified version of how a GFlowNet enforces a policy to sample actions proportional to a reward distribution, affording the opportunity to generate diverse outputs by not overfitting the results towards a maximized reward. As a result, running the tutorial program should print a ratio for each "smiley" face type according to the distribution in the reward function, and modifying the reward distribution affects the sampling ratios accordingly. 

For Example: 
```python
if face.is_happy():
    return 4 # should generate happy faces ~40% of the time
elif face.is_sad(): 
    return 2 # should generate evil faces ~20% of the time
elif face.is_mad():
    return 2 # should generate mad faces ~20% of the time
elif face.is_evil():
    return 1  # should generate evil faces ~10% of the time
```
 
The tutorial is based from and expanded on the tutorial written by Emmanuel Bengio, which can be found [here](https://colab.research.google.com/drive/1fUMwgu2OhYpQagpzU5mhe9_Esib3Q2VR). The work for this tutorial is part of an assignment for a Directed Studies (BIT4000) under the supervision of Dr. David Thue and the RISE Research Group at Carleton University, November 2025.

</br>

### How To Use

**Prerequisites:** 
- Clone this repository to your local machine
- Download Docker Desktop, if you don't have it already (you can download the application [here](https://docs.docker.com/desktop/))

**Steps to Run:**
1. Make sure the Docker Desktop application is open and runnning
2. Navigate to the cloned repository's folder in your terminal
3. Enter the command `docker compose up` 

</br>

**Trouble running docker? No problem, follow the steps below:**
1. Navigate to the cloned repository's folder in your terminal
2. Download environment dependencies with the following command: 
```bash
pip3 install -r requirements.txt

```
3. Then run the tutorial by entering the following command: 
```bash
python3 run gflownet_tutorial/main.py

```

*try `pip` and `python` if `pip3` and `python3` don't work, python environments love to make things difficult*

</br>

**What to look for?:**

Depending on the number of samples generating, the program should take ~30s to complete (~2min when using Docker). Once completed, look for the following outputs: 
- The ratios for each of the generated face types are printed in the terminal
- A generated PNG image named `generated_faces.png` in the project's parent folder, showing the last 64 "smiley" faces that were generated
- A generated PNG image named `loss_curve.png` in the project's parent folder, which plots the loss differential over all iterations. The graph should depict an exponentially decreasing loss

</br>

> If you wish to expirement with the framework to produce alternative results, simply open the file `gflownet_tutorial/gflownet.py` and modify the reward values in the function `face_reward`. This will produce results reflective of the modified distribution.
> 
> Similarly you can modify how many faces it can generate by changing the `num_faces` parameter in the function `gflownet.generate_faces(..)` in the file `gflownet_tutorial/main.py`

</br>
</br>

# Authors
- **Aaron Chase** (4th Year IMD student at Carleton University)
