# Machine Learning Specialization : Advanced Learning Algorithms
#### 1. [Supervised Machine Learning: Regression and Classification](https://github.com/rutvikjoshi63/Student-admission-prediction)
## 2. Advanced Learning Algorithms
#### 3. [Unsupervised Learning, Recommenders, Reinforcement Learning](https://github.com/rutvikjoshi63/Land-Lunar-Lander-on-landing-pad/tree/main)
## Table Of Contents

  * [1. What is Neural Networks](#1-what-is-neural-networks)
  * [2. Decision Making](#2-decision-making)
  * [3. Procedure](#3-procedure)
  * [4. Learnings](#4-learnings)
  * [5. Additional Notes](#5-additional-notes)
  * [6. Resources](#6-resources)
  * [7. Projects](#7-projects)
    + [7.1 Coffee Roasting at Home](#71-coffee-roasting-at-home)
      - [7.1.1 Code Base - Tensorflow](#711-code-base---tensorflow)
        - [7.1.1.2 DataSet](#7112-dataSet)
        - [7.1.1.3 Tensorflow Model](#7113-tensorflow-model)
        - [7.1.1.4 Layer Functions](#7114-layer-functions)
      - [7.1.2  Code Base - Numpy](#712-code-base---numpy)
        - [7.1.2.2 DataSet](#7122-dataSet)
        - [7.1.2.3 Numpy Model](#7123-numpy-model)
        - [7.1.2.4 Predictions](#7124-predictions)
        - [7.1.2.5 Network Function](#7125-network-function)
      
    + [3.2 Anomaly Detection](#32-anomaly-detection)
      - [3.2.1 Code Base](#321-code-base)
      - [3.2.2 Key Points](#312-key-points)
      - [3.2.3 Decision Making](#313-decision-making)
      - [3.2.4 Procedure](#324-procedure)
      - [3.2.5 Learnings](#325-learnings)
      - [3.2.6 Additional Tips](#326-additional-tips)
    + [3.3 Collaborative Filtering](#33-collaborative-filtering)
      - [3.3.1 Code Base ](#331-code-base)
      - [3.3.2 Key Points](#332-key-points)
      - [3.3.3 Decision Making](#333-decision-making)
      - [3.3.4 Procedure](#334-procedure)
      - [3.3.5 Learnings](#335-learnings)
      - [3.3.6 Additional Tips](#336-additional-tips)
    + [3.4 Content-based filtering](#34-content-based-filtering)
      - [3.4.1 Code Base ](#341-code-base)
      - [3.4.2 Key Points](#342-key-points)
      - [3.4.3 Decision Making](#343-decision-making)
      - [3.4.4 Procedure](#344-procedure)
      - [3.4.5 Learnings](#345-learnings)
      - [3.4.6 Additional Tips](#346-additional-tips)
    + [3.5 Land Lunar Lander on landing pad](#35-land-lunar-lander-on-landing-pad)
      - [3.5.1 Code Base ](#351-code-base)
      - [3.5.2 Procedure](#352-procedure)
      - [3.5.3 Learnings](#353-learnings)

  * [Resources](#resources)

 
# 1. What is Neural Networks
* A neural network is built from layers of neurons.
* Each layer takes a vector of inputs and applies a logistic regression unit to each input, producing a vector of outputs.
* The output of one layer becomes the input for the next layer.
* A neural network can have multiple hidden layers and one output layer.
* Superscripts in square brackets denote the layer associated with a quantity (e.g., w^[2] is a parameter in layer 2).
* Activation functions like sigmoid are used to introduce non-linearity.
* Forward propagation is an algorithm to compute the output of a neural network for a given input.

# 2. Decision Making
* Choose the number of layers and neurons based on the complexity of the problem.
* Define the activation function (e.g., sigmoid, tanh).
* Initialize the weights and biases randomly.

# 3. Procedure
* For each layer (except the first):
    + Multiply the input vector by the weight matrix of the current layer.
    + Add the bias vector for the current layer.
    + Apply the activation function to each element of the resulting vector.
* The output of the final layer is the prediction of the neural network.

# 4. Learnings
* Neural networks are powerful models for complex tasks.
* Understanding the forward propagation algorithm is crucial for using neural networks.
* Notation can be complex but helps with understanding the computations.

# 5. Additional Notes
* Backward propagation is used to train the neural network by adjusting the weights and biases.
* Hyperparameter tuning involves adjusting parameters like the learning rate.
* Regularization techniques can help prevent overfitting.
  
# 6. Resources
DeepLearning.AI: https://www.deeplearning.ai/
Stanford Online: https://online.stanford.edu/
Andrew Ng: https://www.youtube.com/watch?v=779kvo2dxb4

# 7. Projects
## 7.1 Coffee Roasting at Home
### 7.1.1 [Code Base - Tensorflow](https://github.com/rutvikjoshi63/Land-Lunar-Lander-on-landing-pad/tree/main/K-means_ImageCompression)
This lab demonstrates how to build a small neural network using Tensorflow to classify coffee roasting data based on temperature and duration features1.
### **7.1.1.2 DataSet**
The lab uses a dataset of 200 examples of coffee roasting with labels indicating good or bad roasts. The data is normalized and tiled to increase the training set size and reduce the number of training epochs2.
### **7.1.1.3 Tensorflow Model**
The lab shows how to create a sequential model with two dense layers and sigmoid activations. The model is compiled with binary crossentropy loss and Adam optimizer. The model is fitted to the data using 10 epochs and the weights are updated.
### **7.1.1.4 Layer Functions**
The lab visualizes the output of each layer and unit in the network and explains their role in the decision making process. The lab also shows how to make predictions using the trained model and apply a threshold to obtain binary decisions.

### 7.1.2 [Code Base - Numpy](https://github.com/rutvikjoshi63/Land-Lunar-Lander-on-landing-pad/tree/main/K-means_ImageCompression)
This lab teaches how to build a small neural network using Numpy. The network has two layers with sigmoid activations and is trained to classify coffee roasting data.
### **7.1.2.2 DataSet**
The data set contains two features: temperature and duration of roasting. The label is whether the roast is good or not. The data is normalized before feeding to the network.
### **7.1.2.3 Numpy Model**
The lab shows how to implement a dense layer and a sequential model using Numpy. The model takes the input, applies the weights and biases, and computes the activations for each layer.
### **7.1.2.4 Predictions**
The lab shows how to use the trained model to make predictions on new examples. The predictions are probabilities of a good roast. A threshold of 0.5 is used to make binary decisions.
### **7.1.2.4 Network Function**
The lab plots the output of the network as a function of the input features. The plot shows the regions where the network predicts a good or a bad roast. The plot is identical to the one obtained using Tensorflow in the previous lab.


### **3.1.5 Learnings**
  * K-means offers a simple yet effective approach to image compression.
  * Choosing k, handling lossy nature, and minimizing artifacts require careful consideration.
  * Advanced algorithms and understanding human perception can unlock further optimization.
### **3.1.6 Additional Tips**
  * Advanced Clustering Algorithms: Exploring techniques like fuzzy c-means, which allow pixels to belong to multiple clusters with varying degrees of membership, can lead to more nuanced and potentially higher-quality compression.
  * Human Perception and Quality Metrics: Understanding how the human eye perceives color variations and artifacts is crucial for optimizing K-means performance and developing effective quality assessment metrics.
    
