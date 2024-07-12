# Project 4: Reinforcement Learning (Course project MLDL 2024 - POLITO)

# Policy Adaptivity in Reinforcement Learning: the issue of the Sim-to-Real gap

This repository contains the code for the project Policy Adaptivity in Reinforcement Learning: the issue of the Sim-to-Real gap, for the Machine Learning and Deep Learning 2023/2024 Course at Politecnico di Torino. Official assignment at [Google Doc](https://docs.google.com/document/d/1lcTs-2a9MoKaTJ5Ii4cnxo7J8t-xmsghG_18ai2qx_o/edit?usp=sharing).


## Abstract 

One of the main challenges in the Reinforcement Learning (RL) paradigm today is its application to robotics and the issue of Sim-to-Real transfer: the gap between the simulated and real worlds degrades the performance of the policies once the models are transferred into real robots.
The aim of this report is to compare different approaches to the Sim-to-Real problem and to analyze them in a Sim-to-Sim scenario. First, we investigate and use basic RL algorithms to train a simple control policy for the Hopper environment. Moreover, we implement Uniform Domain Randomization (UDR), a popular approach for overcoming the reality gap in RL, and compare the performances of SAC agents, trained respectively in the original and in the randomized domain.
Finally, we study and investigate policy adaptivity in Domain Randomization. We start with a comparison between Markovian and memory-based policies and incorporating additional insights that can be used to improve real-world application of these techniques.


