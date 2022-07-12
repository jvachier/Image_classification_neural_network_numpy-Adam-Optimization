# Image classification neural network numpy - Adam Optimization

Author: jvachier <br>
Creation date: July 2022 <br>
Publication date: July 2022 <br>

My goal is to classify 10 digits from 0 to 9 with Neural Networks using Numpy only. The training set (train.csv) contains 42000 labelled pictures and the testing set (test.csv) contains 28000 non-labelled pictures. Each picture has 784 pixels ( 28×28 ) and is the input size of the network. (data from https://www.kaggle.com/c/digit-recognizer)  <br>

The Network is built using two hidden layers using Adam optimization. The first layer contains 128 neurons, the second layer 40 neurons and the output layer 10 neurons. <br>

In this Network, the activation function for the intermediate layers is ReLu and the accuracy is computed. Moreover, the cost function used is the mean squared error (MSE). <br>

This Network is optimized using Adam and compared with another Network without optimization. The learning rate is given by $\alpha$. <br>

The Networks are first trained independently on the training set and then tested using the test set. The accuracy reaches 100% for the Network optimized with Adam. <br>

Reference: <br>

   - 'ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION' D.P Kingma and J. Lei Ba https://arxiv.org/pdf/1412.6980.pdf <br>
    
(Additionally, this code is also available on my Kaggle profile at https://www.kaggle.com/code/jvachier/classification-with-neural-network-adam-numpy)
