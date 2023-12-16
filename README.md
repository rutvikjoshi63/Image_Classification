# Machine Learning Specialization : Advanced Learning Algorithms
#### 1. [Supervised Machine Learning: Regression and Classification](https://github.com/rutvikjoshi63/Student-admission-prediction)
## 2. [Advanced Learning Algorithms](https://github.com/rutvikjoshi63/Image_Classification)
#### 3. [Unsupervised Learning, Recommenders, Reinforcement Learning](https://github.com/rutvikjoshi63/Land-Lunar-Lander-on-landing-pad/tree/main)
## Table Of Contents

  * [Introduction](#1-introduction)
  * [Skills Developed](#2-skills-developed)
  * [Projects](#3-projects)
    + [3.1 K-means Image Compression](#31-K-means-image-compression)
      - [3.1.1 Code Base](#311-code-base)
      - [3.1.2 Key Points](#312-key-points)
      - [3.1.3 Decision Making](#313-decision-making)
      - [3.1.4 Procedure](#314-procedure)
      - [3.1.5 Learnings](#315-learnings)
      - [3.1.6 Additional Tips](#316-additional-tips)
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

 
# 1. Introduction
Having completed DeepLearning.AI's Machine Learning Specialization, I feel empowered with a robust foundation in this transformative field. Professor Ng's masterful guidance expertly navigated the complexities of supervised and unsupervised learning, neural networks, and reinforcement learning.

The program surpassed mere technical training, delving into industry best practices and ethical considerations, equipping me with a Silicon Valley-caliber approach to AI development. This practical knowledge translates directly into tackling real-world challenges with confidence.

The Specialization provided me with:
# 2. Skills Developed
* Solid understanding of foundational machine learning concepts: From linear regression to deep reinforcement learning, I gained a comprehensive grasp of key algorithms and their applications.
* Hands-on Python skills: Building models using NumPy and scikit-learn solidified my ability to implement these concepts in real-world scenarios.
* Critical thinking and problem-solving: I learned to approach challenges with a data-driven mindset, employing best practices for model evaluation and performance optimization.
* Exposure to cutting-edge advancements: The curriculum covered the latest trends in AI, including recommender systems and deep learning, preparing me for the evolving landscape of the field.

Overall, the Machine Learning Specialization proved to be an invaluable investment, equipping me with the knowledge and skills to embark on a successful career in AI. I am grateful to Professor Ng and the DeepLearning.AI team for crafting such a comprehensive and impactful learning experience.

Looking forward to: Utilizing my newfound expertise to contribute to meaningful AI solutions that address real-world problems and shape the future.

# Resources:

DeepLearning.AI: https://www.deeplearning.ai/
Stanford Online: https://online.stanford.edu/
Andrew Ng: https://www.youtube.com/watch?v=779kvo2dxb4

# 3. Projects
## 3.1 K-means Image Compression
This project was a hands-on introduction to compressing images using K-means clustering to reduce file size while maintaining visual quality.

### 3.1.1 [Code Base](https://github.com/rutvikjoshi63/Land-Lunar-Lander-on-landing-pad/tree/main/K-means_ImageCompression)
### **3.1.2 Key Points**
  * Objective: Reduce image file size while preserving visual quality using K-means clustering.
  * Approach: Group similar pixels into clusters, replacing each pixel's value with its cluster centroid.
### **3.1.3 Decision Making**
  * Number of clusters (k): Balancing compression ratio and visual quality.
  * Lossy vs. Lossless compression: Trade-off between file size and perfect reconstruction.
### **3.1.4 Procedure**
  * Preprocessing: Read the image and convert it to a format suitable for K-means (e.g., pixel values).
  * K-means clustering:
      + Define the desired number of clusters (k).
      + Randomly initialize k cluster centroids.
      + Assign each pixel to the closest cluster centroid.
      + Update the cluster centroids based on the assigned pixels.
      + Repeat steps 3-5 until convergence (no further changes in cluster assignments).
  * Quantization: Replace each pixel's original value with the value of its assigned cluster centroid.
  * Encoding: Encode the quantized image data and cluster centroids for storage or transmission.
### **3.1.5 Learnings**
  * K-means offers a simple yet effective approach to image compression.
  * Choosing k, handling lossy nature, and minimizing artifacts require careful consideration.
  * Advanced algorithms and understanding human perception can unlock further optimization.
### **3.1.6 Additional Tips**
  * Advanced Clustering Algorithms: Exploring techniques like fuzzy c-means, which allow pixels to belong to multiple clusters with varying degrees of membership, can lead to more nuanced and potentially higher-quality compression.
  * Human Perception and Quality Metrics: Understanding how the human eye perceives color variations and artifacts is crucial for optimizing K-means performance and developing effective quality assessment metrics.
    
## 3.2 Anomaly Detection
This project was a hands-on introduction to uncovering hidden patterns and identifying outliers. It challenged me to step beyond the realm of standard prediction and delve into the fascinating world of unsupervised learning.

### 3.2.1 [Code Base](https://github.com/rutvikjoshi63/Land-Lunar-Lander-on-landing-pad/tree/main/AnamolyDetection)
### **3.2.2 Key Points**
  * The algorithm models the probability of each feature in the data using Gaussian distributions.
  * Anomalies are identified as data points with low probability values (determined by a threshold, epsilon).
  * Choosing the right features is crucial for effective anomaly detection.
  * A small number of labeled anomalies can help evaluate and tune the algorithm.
### **3.2.3 Decision Making**
  * Anomaly detection is preferred when you have few positive examples (anomalies) and many negative examples (normal data).
  * Supervised learning is better suited when you have a larger number of positive examples and you can expect future anomalies to be similar to those seen in the training set.
### **3.2.4 Procedure**
  * Data Preparation:
      - Define features relevant to potential anomalies.
      - Gather training data, typically unlabeled (normal examples).
      - Optionally, acquire a small set of labeled anomalies for evaluation.
  * Model Training:
      - Choose an anomaly detection algorithm (e.g., Gaussian Mixture Models, One-Class SVMs).
      - Estimate the model parameters (e.g., mean and variance of Gaussian distributions) based on the training data.
  * Anomaly Scoring:
      - For each new data point, calculate its probability under the model (e.g., using Gaussian density function).
  * Thresholding and Detection:
      - Set a threshold (epsilon) for anomaly score.
      - Data points with score below epsilon are flagged as anomalies.
### **3.2.5 Learnings**
  * Identifying unusual patterns: Anomaly detection excels at identifying data points that deviate significantly from the expected behavior, potentially revealing hidden patterns or anomalies.
  * Unsupervised learning: This technique doesn't require labeled data for training, making it valuable for situations where labeling anomalies is difficult or expensive.
  * Feature engineering: Choosing the right features is crucial for effective anomaly detection. Relevant features directly impact the model's ability to capture meaningful deviations.
  * Parameter tuning: The algorithm's performance depends on parameters like the epsilon threshold. Cross-validation and evaluation sets help optimize these parameters.
  * Trade-offs: Anomaly detection algorithms balance sensitivity (detecting anomalies) and specificity (avoiding false positives). Tuning parameters and choosing features can help strike this balance.
  * Limitations: Anomaly detection may struggle with novel anomalies unseen in the training data and can be sensitive to outliers and noise.
  * Different anomaly detection algorithms are suited for different types of data and problems.
  * Anomaly detection is often used in conjunction with other techniques for comprehensive anomaly identification and analysis.
  * Domain knowledge plays a crucial role in interpreting and validating anomalies flagged by the algorithm.
### **3.2.6 Additional Tips**
  * Cross-validation and test sets are crucial for evaluating and tuning the algorithm.
  * Tuning features and the epsilon threshold can significantly impact performance.
  * Anomaly detection is often used in security applications where attackers might use new tactics.
## 3.3 Collaborative Filtering
Imagine you walk into a bookstore, overwhelmed by the endless rows of books. Suddenly, a friendly bookseller appears, recommending titles based on your favorite authors and genres. That's the magic of collaborative filtering, a powerful technique used by recommendation systems to suggest you'll love!

### 3.3.1 [Code Base](https://github.com/rutvikjoshi63/Land-Lunar-Lander-on-landing-pad/tree/main/CollabarativeFiletring)
### **3.3.2 Key Points**
  * Collaborative filtering is a technique used to recommend items to users based on the preferences of other similar users.
  * It works by analyzing the ratings or interactions users have with different items and identifying patterns of similarity between users.
  * This information is then used to predict how likely a specific user is to like an item they haven't yet rated or interacted with.
  * Collaborative filtering can be used for various applications, including recommending movies, products, music, news articles, and even friends.
### **3.3.3 Decision Making**
  * The decision of whether or not to recommend an item to a user is based on the predicted rating or probability of the user liking the item.
  * This prediction is made by comparing the user's profile to the profiles of similar users and seeing what items those users have liked.
### **3.3.4 Procedure**
There are two main approaches to collaborative filtering:
  * Matrix factorization: This approach decomposes the user-item rating matrix into two lower-dimensional matrices, representing latent factors that capture the underlying patterns of user preferences and item characteristics. These factors are then used to predict missing ratings or recommend new items.
  * Neighborhood-based methods: These methods identify a set of similar users (the neighborhood) for a target user and then recommend items that the users in the neighborhood have liked.
### **3.3.5 Learnings**
  * Collaborative filtering can be a powerful tool for recommending items to users, but it is important to consider its limitations.
  * For example, it can be susceptible to cold start problems (when there is not enough data about a user or item) and can also be biased towards items that are already popular.
  * Despite these limitations, collaborative filtering remains a valuable technique for many recommender systems.
### **3.3.6 Additional Tips**
  * The video lectures you provided discuss two specific algorithms for collaborative filtering: matrix factorization with features and matrix factorization without features.
  * These are just two examples, and there are many other algorithms that can be used for collaborative filtering.
  * The choice of algorithm will depend on the specific data and application.
## 3.4 Content-based filtering
Imagine you walk into a bookstore, overwhelmed by the endless rows of books. Suddenly, a friendly bookseller appears, recommending titles based on your favorite authors and genres. That's the magic of collaborative filtering, a powerful technique used by recommendation systems to suggest you'll love!
### 3.4.1 [Code Base](https://github.com/rutvikjoshi63/Land-Lunar-Lander-on-landing-pad/tree/main/content_base_Filtering/)
### **3.4.2 Key Points**
  * Content-based filtering recommends items based on the features of users and items, unlike collaborative filtering which relies on user ratings.
  * User features can include age, gender, country, past movie ratings, and genre preferences.
  * Item features can include year of release, genre, stars, and critic reviews.
  * The goal is to learn vectors (v_u for users and v_m for movies) that capture user preferences and item characteristics.
  * The recommendation score is calculated as the dot product of v_u and v_m.
### **3.4.3 Decision Making**
  * Choose features that effectively represent users and items.
  * Design a neural network architecture to learn v_u and v_m vectors.
  * Use a cost function based on squared error between predictions and actual ratings to train the model.
### **3.4.4 Procedure**
  * Preprocess data to extract features for users and items.
  * Train the neural network model on user-item pairs with ratings.
  * For a new user, compute the v_u vector based on their features.
  * For a new item, compute the v_m vector based on its features.
  * Recommend items with the highest dot product scores between v_u and v_m for the new user.
### **3.4.5 Learnings**
  * Deep learning is a powerful approach for content-based filtering.
  * Pre-computing item similarities can improve efficiency.
  * Ethical considerations are crucial when designing recommender systems.
### **3.4.6 Additional Tips**
  * Content-based filtering can be combined with collaborative filtering for improved performance.
  * Large-scale recommender systems often use a two-step approach: retrieval and ranking.
  * Retrieval finds a shortlist of plausible items based on precomputed similarities.
  * Ranking uses a learned model to refine the shortlist and recommend the top items.
## 3.5 Land Lunar Lander on landing pad
This project was a hands-on introduction to the world of reinforcement learning, where an agent learns through trial and error, maximizing its reward by making optimal decisions in a dynamic environment.
### 3.5.1 [Code Base](https://github.com/rutvikjoshi63/Land-Lunar-Lander-on-landing-pad/tree/main/LunarLander/Files)

### **3.5.2 Procedure**
  * Modeling the moon environment: I built a physics simulation of the moon's gravity, thrusters, and lander dynamics. This provided the foundation for the agent's decision-making.
  * Crafting a reward function: I defined a reward system that prioritized landing within the designated area at low velocity, penalizing crashes and out-of-bounds landings. This guided the agent's learning towards the desired outcome.
  * Implementing Q-Learning: I used the Q-Learning algorithm, which learns the value of taking different actions in different states. The agent constantly updated its "Q-table," estimating the future rewards associated with each action in a specific situation.
  * Training through exploration and exploitation: The agent initially explored the environment randomly, learning the consequences of each action. As it gained experience, it exploited its knowledge, favoring actions with higher predicted rewards, leading to smoother and safer landings. 
### **3.5.3 Learnings**
  * Understanding the principles of reinforcement learning: I grasped the core concepts of reward maximization, Q-tables, and the trade-off between exploration and exploitation.
  * Applying theoretical knowledge to a real-world scenario: The project bridged the gap between theory and practice, demonstrating how reinforcement learning can be used to solve complex problems.
  * Developing problem-solving skills: I learned to break down a challenging task into smaller components, define objectives, and iterate on my approach to achieve optimal results.

# Image_Classification
image multiclassification model using TensorFlow and Keras that achieved an accuracy of 99.64% on 5000 grayscale images with a 20x20 pixel resolution
