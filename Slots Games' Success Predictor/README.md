# Slots Games' Success Predictor
This project was built to satisfy the need of a real-life company which is dedicated to developing games for online casinos. This was also my thesis project for my master's in Artificial Intelligence and Deep Learning of the University of Alcala de Henares (Spain).

* Feed Forward Neural Network
* Sequential Model

## Problem Description

Online casinos are constantly trying to improve their games so that players can be more entertained and wanting to continue playing.

This is why the company needed a model that helped them predict if a Slots Game was going to be a market success or a market failure, by analyzing the way the game rewarded the players.
In other words, a model that could tell if the way that a game rewards a player, will influence him/her to continue playing.

<img src="images/slotsGamesPic.PNG" alt="drawing" width="250"/>

The amt of the payout, the length of a losing streaks, or winning streaks.

## The Training Data
To train our model we simulated millions of spins of different Slots Games that the company has developed. 

**Describir que los juegos generaron varios samples **

Considering that we are dealing with an entertainment product, we need to understand that the success of a game will be determined by the level of preference of the consumers, in this case, the players, and how much time do they spend playing them.

## Solution Approach
### Sequential Model
This problem was tackled from 2 different approaches, first we tried to analyze the payouts that the players received in the first 100 spins they played to confirm if there was a pattern in the sequence of payouts that could be influecing the players to like or dislike a game.

**Describir los resultsados de la primera iteracion, los resultados despues de solo analizar las sequencias cercanas a la media**

### Feed Forward Neural Network
In the second approach we used a feed forward neural network. In this case, we feed into the model the distribution of relevant reward events such as near-miss, the lenght of winning streaks and lossing streaks, losses disguised as winnings, and so on.

Trying to find if there is a pattern in the payout sequence of the games that influences the player to continue plyaing, using a sequential model.

Feeding the summarized distribution of the game's rewards into a Feed Forward neural network.

