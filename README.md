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

Its purpose it to show a very simplified version of how a GFlowNet enforces a policy to sample actions proportional to a reward distribution, affording the opportunity to generate diverse outputs by not overfitting the results towards a maximized reward. 

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

**Trouble running docker? No problem, follow the steps below:**
1. Navigate to the cloned repository's folder in your terminal
2. Download environment dependencies with the following command: 
```bash
pip3 install requirements.txt -r

```
3. Then run the tutorial by entering the following command: 
```bash
python3 run gflownet_tutorial/main.py

```

*try `pip` and `python` if `pip3` and `python3` don't work, python environments love to make things difficult*

</br>

> If you wish to expirement with the framework to produce alternative results, simply open the file `gflownet_tutorial/gflownet.py` and modify the reward values in the function `face_reward`. This will produce results reflective of the modified distribution.
> 
> Similarly you can modify how many faces it can generate by changing the `num_faces` parameter in the function `gflownet.generate_faces(..)` in the file `gflownet_tutorial/main.py`

</br>
</br>

# Authors
- **Aaron Chase** (4th Year IMD student at Carleton University)
