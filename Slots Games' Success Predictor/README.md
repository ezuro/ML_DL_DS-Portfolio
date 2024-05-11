# Slots Games' Success Predictor
This project was built to satisfy the need of a real-life company which is dedicated to developing games for online casinos. This was also my thesis project for my master's in Artificial Intelligence and Deep Learning of the University of Alcala de Henares (Spain). The problem of this project was approached using different models, including:

* Feed Forward Neural Network
* Sequential Model

## Problem Overview

The company required a model that predicted if a newly designed Slots Game is going to be successful or not. The model would make its predictions based on the way that the game rewards their players after certain time of playing it. In other words, the model should tell if the game will influence their players to continue playing or not, by the way it provides rewards to them in a certain number of spins.

The objective is to verify if there is any pattern, in the reward systems of Slots Games, that would make them successful or un-successful, and to design a model that could indentify this pattern and predict it.

<img src="images/slotsGamesPic.PNG" alt="drawing" width="250"/>

To build the model we used the results of thousands of spins during different game sessions of old Slots Games, both, successful and unsuccessful. Be aware, that the level of success or failure of an entertainment product such as a video game, is determined by the preference of its audience, and how much time they spent playing it.

I built two models to tackle this issue, one where it was merely analyzed the 'net wins' or 'pay outs' that a player had in a n-spins game session. And another, where it was analyzed different aspects relating to the rewards of the game, such as the number of free spins won, the length of continuous winning and lossing streaks, the frequency of 'near-miss' and 'losses disguised as wins' events and so on.


----
The amt of the payout, the length of a losing streaks, or winning streaks.

## Solution Approach
This problem was tackled from 2 different approaches.

Trying to find if there is a pattern in the payout sequence of the games that influences the player to continue plyaing, using a sequential model.

Feeding the summarized distribution of the game's rewards into a Feed Forward neural network.

