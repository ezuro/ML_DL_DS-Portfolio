# Slots Games' Success Predictor
This project was built to satisfy the need of a real-life company which is dedicated to developing games for online casinos. This was also my thesis project for my master's in Artificial Intelligence and Deep Learning of the University of Alcala de Henares (Spain). The problem of this project was approached using different models, including:

* Feed Forward Neural Network
* Sequential Models:
  * Recurrent Neural Network (LSTM)
  * Convolutional Nerual Network (1 Dimensional Convolutions)

.... content page

# 1 - Problem Overview

The company required a model that predicted if a newly designed Slots Game is going to be successful or not (binary output). The model would make its predictions based on the way that the game rewards their players after certain time of playing it. In other words, the model should tell if the game will influence their players to continue playing or not, by the way it provides rewards to them in a certain number of spins.

The objective is to verify if there is any pattern, in the reward systems of Slots Games, that would make them successful or un-successful, and to design a model that could indentify this pattern and predict it.

<img src="images/slotsGamesPic.PNG" alt="drawing" width="250"/>

To train the model we used the results of thousands of spins during different game sessions of both, successful and unsuccessful,  Slots Games. Be aware, that the level of success or failure of an entertainment product such as a video game, is determined by the preference of its audience, and how much time they spent playing it. This was the measure we used to rank the Slots Games whose data was used to train the model.

I built two models to tackle this issue, one where it was merely analyzed the 'net wins' or 'pay outs' that a player had in a n-spins game session. And another, where it was analyzed different aspects relating to the rewards of the game, such as the payout amounts, the number of free spins won, the length of continuous winning and lossing streaks, the frequency of 'near-miss' and 'losses disguised as wins' events, and so on, also within a n-spins game session.


# 2 - Solution Approach

## 2.1 - Sequential Models (Recurrent and Convolutional)
As mentioned, the issue was tackled from different angles. One of them being the analyzing of the game pay outs during a game session of n-spins. A pay out is simply the amount of money a player won in a single spin, e.g., a player bet $1 and won $0.5 in one spin. 

From the pay outs of each spin in a player's game session, we could form a sequence of pay outs, that can be fed into a sequential neural network. However, we can also use a sequence of the value of the player's balance after each spin, as the balance is directly affected by the pay outs the player received and it better draws the trajectory of the results obtained by the player. We can easily see how the player's balance increases or decreases according to the pay outs received from the start to the end of the game session.

<img src="images/seqBalance.png" alt="drawing" width="550"/>

The above image shows a graph of the balance's trajectory of different players in different game sessions for a corresponding Slots Game (precisely 1000 game sessions per Slots Game). We can see that all of the players' balance start at $100 and they follow certain path until they reach $0 after a certain number of spins.

The idea is to feed the sequences into a neural network and tell the model which of those sequences belong to a successful or unsuccessful game, so that the model can learn to classify them correctly.

### 2.1.1 - Model Structure
As mentioned, for this sequence data I built 2 different models a Recurrent Neural Network, using Long-Short-Term Memory layers, and a Convolutional Neural Network, using convolutional layers of 1 Dimension.

For both models we tested different activation, optimizer and loss functions, however the best results were achieved using:
* Rectified Linear Unit (ReLU) as activation function for the hidden layers
* Sigmoid as the activation function in the output layer
* Stochastic Gradient Descent as the optimizer
* Binary Cross-Entropy as the loss function.

The models were kept simple (just a few layers), to avoid overfitting problems due to a high complexity.

Recurrent NN structure:
```python 
def recurrent_model(inputShape,learningRate):
    model=Sequential()
    model.add(Bidirectional(LSTM(64,return_sequences=True,activation='relu'), input_shape=(inputShape,1)))     
    model.add(Bidirectional(LSTM(24,activation='relu',return_sequences=True)))
    model.add(Bidirectional(LSTM(5,activation='relu')))

    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    opt = keras.optimizers.SGD(learning_rate=learningRate)
    model.compile(loss='binary_crossentropy', optimizer=opt,metrics=['accuracy'])
    
    return model
 
```
Convolutional NN structure:
```python 
def cnn_model (inputShape,learningRate):
    model = Sequential()

    model.add(Conv1D(filters=254, kernel_size=128,  activation='relu',input_shape=(inputShape,1)))
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(Dropout(0.4))

    model.add(Conv1D(filters=128, kernel_size=35, activation='relu'))
    model.add(MaxPooling1D(2))


    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    opt = keras.optimizers.SGD(learning_rate=learningRate)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model
```

### 2.1.2 - Results Obtained



We first used all the game sessions and obtained these results.

However there was too much outliers, so we decided to use only the sequences that fell in the third quantile and that were much closer to the median.


## 2.2 - Feed Forward Neural Network
Given the results of the sequential model were not so satisfactory due to the possible noise that can be found in the pay out sequences we decided to include other aspects relating to the rewards of the games and summarize them into their distributions and feed them into a Feed Forward neural network. 

Feeding the summarized distribution of the game's rewards into a Feed Forward neural network.



We decided to include the following variables.

For each variable we decided to include the following statistics.

### Structure used
### Results Obtained

