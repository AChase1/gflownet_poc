### Table of Contents
> [GFlowNet Proof-Of-Concept](#gflownet-proof-of-concept)<br>
> [Minecraft GFlowNet](#gflownet-minecraft)<br>
>     [Description](#description-1)
>     [Important: Not Working As Expected (Suggested Fixes)](#important-not-working-as-expected-suggested-fixes)
>     [How To Use](#how-to-use-1)
> [GFlowNet Tutorial](#gflownet-tutorial)<br>
>     [Description](#description)<br>
>     [How To Use](#how-to-use)<br>
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

## Minecraft GFlowNet

</br>

### Description
Found under the `minecraft/` directory, this explores using GFlowNets to sample building composite houses in Minecraft using the [GDPC](https://github.com/avdstaaij/gdpc) and [GDMC](https://github.com/Niels-NTG/gdmc_http_interface). Its purpose is to experiment using GFlowNets for increased control over the generation of diverse composite objects in the procedural generation of environments. GFlowNets affords the ability to accessibly manage the output of the machine learning framework through the manipulation of the reward function to produce desired composite objects/environments and/or force diverse experimentation (preventing overfitting). 

### Important: Not Working As Expected (Suggested Fixes)
Please note that the current implementation for does not work as expected. Although designed to efficiently explore large search spaces and avoid overfitting, when constructing composite objects of >2^3 actions (in this case, 4^6), it appears to do just that. Based on the test outputs and subsequent research, it appears to be "mode collapse", a common pitfall in machine learning. The model fails to fully explore the search space in order to learn the paths for all the terminal states. When specifying granular rewards for specific paths (e.g., house.is_farmhouse()), the model does not sufficiently sample that path enough to learn its corresponding reward. Therefore more common paths with rewards > 0 (e.g., mixed house styles), are learned early and thus overfits the model to continue to train down those paths, failing to fully explore the search space. Several modifications attempted to fix this, including using the Trajectory Balance logic (from the [Google Collab](https://colab.research.google.com/drive/1fUMwgu2OhYpQagpzU5mhe9_Esib3Q2VR#scrollTo=3_XcEbVHB1rs) tutorial) with logits (better for larger search spaces) for sampling houses, as well as small fixes like including a scheduler for stable learning decay and temperature for forced early exploration. Despite this seemingly being the issue, there are probably a number of different factors contributing towards its inaccuracy (e.g., number instability).

One source provides a potential solution for this, called [Boosted GFlowNets](https://arxiv.org/html/2511.09677v1#S4). This paper outlines a solution to improve the exploration process of GFlowNets in order to fully explore large search spaces. The general idea, as I understand it, is to use several GFlowNet models called boosters are trained, exploring alternative paths based on the outputs from other boosters (forcing full explorations of large search spaces). An attempt was made to implement this concept, however unfortunately, given time constraints and mathemical complexity, the implementation failed to work. 

Next Steps: Implement a working Boosted GFlowNet in order to address the mode collapse

### How To Use 

**Minecraft**

1. Must have a Minecraft account and downloaded [Minecraft Java Edition](https://www.minecraft.net/en-us/store/minecraft-java-bedrock-edition-pc?tabs=%7B%22details%22%3A0%7D) (latest version) on your local machine

**GDMC & GDPC**

In order to use GDPC (python library for building in Minecraft using scripts), you must have the GDMC HTTP Interface (used for connecting to the Minecraft). If needed, reference their documentations for troubleshooting the setup process: [GDPC Docs](https://gdpc.readthedocs.io/en/stable/index.html), [GDMC HTTP Interface Docs](https://github.com/Niels-NTG/gdmc_http_interface/blob/master/README.md)

1. Download the [Modrinth App](https://modrinth.com/app)
2. Once downloaded and opened, sign into your Minecraft account on the Modrinth App
    > If you run into errors when signing in, you must have the Minecraft Java Edition game open and running on your local machine
3. Open the "Discover content" menu tab, and search for GDMC HTTP Interface (make sure the "Mods" filter tab is selected)
4. Download the mod
5. Click on the "+" menu tab to create a new instance (ensure the GDMC HTTP Interface mod is enabled for that instance)
6. Click the "Play" button, which will launch a separate instance of the Minecraft Java Edition (notable by its greyed out background)
7. In your editor, after cloning this repository onto your local machine, ensure to install all related dependencies: 
```bash
pip install -r requirements.txt
```

**Create Minecraft World**

You may create any world you like, but for the purposes of the experiment its recommended to use the following Minecraft world configurations: 

- Singleplayer
- (Game) Game Mode: Creative
- (Game) Difficulty: Peaceful
- (Game) Allow Commands: ON
- (World) World Type: Superflat
- (World) Generate Structures: OFF

#### Running the GDPC Tutorial

If you want to simply test the GDPC library, run the script `minecraft/gdpc_house_tutorial.py`: 
```bash
python minecraft/gdpc_house_tutorial.py
```

The terminal should output the associated executed commands, and the house should be created within your already opened minecraft world. 

#### Running the GFlowNet House tutorial

Run the `minecraft/gflownet_house.py` script: 
```bash
python minecraft/gflownet_house.py 
```

The terminal will show its training process, and output the ratio of constructed house types after finishing training. In Minecraft, you'll see the last 10 houses created by the GFlowNet model. To manipulate the output of the model, modify the rewards and/or the house properties in the `house_reward` function.

> Just a reminder that this implementation does not work as expected, please read the Important: Not Working As Expected (Suggested Fixes) section.

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
python3 gflownet_tutorial/main.py

```

*try `pip` and `python` if `pip3` and `python3` don't work, python environments love to make things difficult*

</br>

**What to look for?:**

Depending on the number of samples generating, the program should take ~30s to complete (~2min when using Docker). Once completed, look for the following outputs: 
- The ratios for each of the generated face types are printed in the terminal. The ratios should reflect the distributions specified in the reward function with +/-5%
- A generated PNG image named `generated_faces.png` in the project's folder, showing the last 64 "smiley" faces that were generated
- A generated PNG image named `loss_curve.png` in the project's folder, which plots the loss differential over all iterations. The graph should depict an exponentially decreasing loss

</br>

> If you wish to expirement with the framework to produce alternative results, simply open the file `gflownet_tutorial/gflownet.py` and modify the reward values in the function `face_reward`. This will produce results reflective of the modified distribution.
> 
> Similarly you can modify how many faces it can generate by changing the `num_faces` parameter in the function `gflownet.generate_faces(..)` in the file `gflownet_tutorial/main.py`

</br>
</br>

# Authors
- **Aaron Chase** (4th Year IMD student at Carleton University)
