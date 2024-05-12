# Slots Games' Success Predictor
This project was built to satisfy the need of a real-life company which is dedicated to developing games for online casinos. This was also my thesis project for my master's in Artificial Intelligence and Deep Learning of the University of Alcala de Henares (Spain). The problem of this project was approached using different models, including:

* Feed Forward Neural Network
* Sequential Model

# Problem Overview

The company required a model that predicted if a newly designed Slots Game is going to be successful or not (binary output). The model would make its predictions based on the way that the game rewards their players after certain time of playing it. In other words, the model should tell if the game will influence their players to continue playing or not, by the way it provides rewards to them in a certain number of spins.

The objective is to verify if there is any pattern, in the reward systems of Slots Games, that would make them successful or un-successful, and to design a model that could indentify this pattern and predict it.

<img src="images/slotsGamesPic.PNG" alt="drawing" width="250"/>

To train the model we used the results of thousands of spins during different game sessions of both, successful and unsuccessful,  Slots Games. Be aware, that the level of success or failure of an entertainment product such as a video game, is determined by the preference of its audience, and how much time they spent playing it. This was the measure we used to rank the Slots Games whose data was used to train the model.

I built two models to tackle this issue, one where it was merely analyzed the 'net wins' or 'pay outs' that a player had in a n-spins game session. And another, where it was analyzed different aspects relating to the rewards of the game, such as the payout amounts, the number of free spins won, the length of continuous winning and lossing streaks, the frequency of 'near-miss' and 'losses disguised as wins' events, and so on, also within a n-spins game session.

# Solution Approach
## Sequential Model
As mentioned, the issue was tackled from different angles. One of them being the analyzing of the game pay outs during a game session of n-spins. A pay out is simply the amount of money a player won in a single spin, e.g., a player bet $1 and won $0.5 in one spin. 

From the pay outs of each spin in a player's game session, we could form a sequence of pay outs, that can be fed into a sequential neural network. However, we can also use a sequence of the player's balance values after each spin, as the balance is directly affected by the pay outs the player received and it better draws the trajectory of the results obtained by the player. We can easily see how the player's balance increases or decreases according to the pay outs received from the start to the end of the game session.




The above image shows the trajectory of a player's balance in a game session from a Slots Games that is eiether successful or un-successful. each of these games could be good or bad (successful or not-successful).

The idea is to feed the sequences into a neural network and tell the model which of those sequences belong to a successful or unsuccessful game, so that the model can learn to classify them correctly.

## Structure used
## Results Obtained





Trying to find if there is a pattern in the payout sequence of the games that influences the player to continue plyaing, using a sequential model.

Feeding the summarized distribution of the game's rewards into a Feed Forward neural network.

