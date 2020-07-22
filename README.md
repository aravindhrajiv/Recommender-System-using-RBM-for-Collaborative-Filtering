# Problem Statement
To make automatic but personalised recommendations when customers are presented to thousands of possibilities and do not know what to look for and to make a framework in which the problem of making recommendations can be formulated. The inner workings of the most popular collaborative
filtering algorithms are to be used. Collaborative Filtering tries to identify similarity among users base on their past behavior, and then recommend items to the user which are liked, bought, or rated highly by similar users.

**Major Use Case** - Personalised product recommendations for an existing user as well as a new user.


# Process at a Glance
![2](https://user-images.githubusercontent.com/67309253/88150525-21b84b80-cc1f-11ea-9df7-1a148fd74e98.JPG)

# Requirements
* Python 3.6.5
* Python Packages
    * Cv2        : Open Source Computer Vision Library
    * Pandas     : for dataframe
    * Numpy      : for numerical computations
    * Flask      : web application framework
    * Os         : for interacting with the operating system
    * Tensorflow : Open Source Dataflow library
    * Pytorch    : Open Source Machine Learning library based on the Torch library
    * Sklearn    : Python Library for many unsupervised and supervised learning algorithms
    * Pickle     : for serializing and de-serializing a Python object structure
    * Json       : for storing and exchanging data with JavaScript object notation
    
# Python Scripts
**Jupyter Notebook** - Final training.ipynb , Prediction for Product Recommendation.ipynb

# Dataset
Download Link - https://www.kaggle.com/c/santander-product-recommendation/data
* **train_ver2.csv** - Dataset used for training the models. It conatins around 13 million rows of data including user profile, user history (products that they bought) etc.
* **reference_sample_001** - Sampled 10% of the above mentioned data as my system couldn't load the entire data.
* **Input Variable** - First 24 columns of the data relates to the User's profile details
* **Target Variables** - Last 24 columns of the data relates to the User's history (products that user has bought)
    
# Code Description
#### Final training.ipynb ####

1. Reading the input excel file from the dataset directory and storing in train list.
   ![1](https://user-images.githubusercontent.com/67309253/88150320-def67380-cc1e-11ea-850d-b549fdf7e477.JPG)
2. ##### Data Pre-Processing steps #####
   * Converting the data types of continuous variables to numeric values.
   * Imputing the missing values of continuous variables with mean of reamining values.
   * Imputing the missing values of categorical variables with max. occured string.
   ![3](https://user-images.githubusercontent.com/67309253/88151890-e0c13680-cc20-11ea-97d7-99e144002b82.JPG)
   * Converting the data types of target variables to integral values.
   * Dropping the date columns for version 1 of the model.
   * Converting the variable of string type to category type
   * One Hot Encoding of the category data typed variabled. 
   ![4](https://user-images.githubusercontent.com/67309253/88154427-7d390800-cc24-11ea-9e4f-aa40280568a3.JPG)
   
3. ##### Random Forest Training #####
   * This is an algorithm deployed to overcome the cold start problem for recommending products for a new user.    
   * The set of algorithms followed in the training are:
       1. **Train-Test Split**: (80%-Training & 20%-Testing)
       ![5](https://user-images.githubusercontent.com/67309253/88156939-c0e14100-cc27-11ea-85fe-db92c2f0f484.JPG)
       2. **Fitting a Random Forest classifier for each of the target variables**
       3. **Saving the RF classifiers into a pickle file** 
       4. **Predicting the output with training data**
       5. **Calculating the training precision,recall and f1 scores**
       ![image](https://user-images.githubusercontent.com/67309253/88157642-a52a6a80-cc28-11ea-8ed7-b3ce3dbf65fc.png)
       6. **Predicting the output with testing data**
       7. **Calculating the testing precision,recall and f1 scores**
       ![image](https://user-images.githubusercontent.com/67309253/88158214-634df400-cc29-11ea-9ad8-c4ca2d7ebe59.png)
       8. **Copying our results into a dataframe and exporting it as .csv file**
       ![image](https://user-images.githubusercontent.com/67309253/88158465-b45de800-cc29-11ea-8fb6-e3f66b1bf005.png)
       
 4. #### Restricted Boltzmann Machine (RBM) Training ####
    * RBMs are a two-layered artificial neural network with generative capabilities. They have the ability to learn a probability distribution over its set of user preferences.
    * RBM uses the learned probabiltiy distribution to predict recommendations on never-before-seen items.
    * The set of algorithms followed in the training are:
         1. **Reshaping the array of target varibles**
         2. **Train-Test Split**: (80%-Training & 20%-Testing)
         3. ** Converting the training and testing data into Pytorch tensors
         ![image](https://user-images.githubusercontent.com/67309253/88160604-7ca46f80-cc2c-11ea-9423-6ccc97bbce02.png)
         4. **Class RBM**
               * Intialising the weight and bias tensors
               ![image](https://user-images.githubusercontent.com/67309253/88160674-9a71d480-cc2c-11ea-8177-937f79137f9a.png)
               * probability of visible vector given hidden vector
               * probability of visible vector given hidden vector
               ![image](https://user-images.githubusercontent.com/67309253/88160721-af4e6800-cc2c-11ea-9435-0668d9715a73.png)
               * Paramater learning funtion
               ![image](https://user-images.githubusercontent.com/67309253/88160882-e7ee4180-cc2c-11ea-9166-3c2e9f78b85a.png)
               * Free energy function - It calculates the overall energy of the system. It inference the stability of the trained model.
               ![image](https://user-images.githubusercontent.com/67309253/88160988-081e0080-cc2d-11ea-998b-3f7e56858a8d.png)
       
         5. **Setting the hyperparameters for the training**
         ![image](https://user-images.githubusercontent.com/67309253/88161213-529f7d00-cc2d-11ea-9c83-f2e41d07cfef.png)
         6. **Training the RBM model with Gibbs Sampling and Contrastive Divergence Learning**
         ![image](https://user-images.githubusercontent.com/67309253/88161392-8f6b7400-cc2d-11ea-9e8b-5f7a4d3061a6.png)
         7. **Plotting the Error vs Epochs graph**
         ![image](https://user-images.githubusercontent.com/67309253/88161545-bcb82200-cc2d-11ea-91d0-894e045eaea6.png)
         8. **Plotting the Free Energy vs Epochs graph**
         ![image](https://user-images.githubusercontent.com/67309253/88161949-4b2ca380-cc2e-11ea-9d43-68f6b4a39b57.png)
         9. ** Saving the RBM model 
         ![image](https://user-images.githubusercontent.com/67309253/88162196-a1014b80-cc2e-11ea-94fe-3a21cf0037f9.png)
         
#### Prediction for Product Recommendation.ipynb ####

1. Loading all the trained models of Random Forest and RBM 
![image](https://user-images.githubusercontent.com/67309253/88162781-6fd54b00-cc2f-11ea-8e1b-caca780fb0ec.png)
2. Storing the trained data as reference to check whether an user is already existing or new one 
![image](https://user-images.githubusercontent.com/67309253/88164734-220e1200-cc32-11ea-9c40-e808d572ecb7.png)
3. Pre-Processing function for the input data 
![image](https://user-images.githubusercontent.com/67309253/88164859-5c77af00-cc32-11ea-9377-098237346a6e.png)
4. Functions for Old User and New User
![image](https://user-images.githubusercontent.com/67309253/88164983-96e14c00-cc32-11ea-917b-f0724fc091ba.png)
5. Sending the input user's product vector to RBM (either from existing data or from random forests' predictions)
![image](https://user-images.githubusercontent.com/67309253/88165259-13742a80-cc33-11ea-8a85-f6a7a39a39b4.png)
6. Predicting k no. of product recommendations and converting it into a dataframe for output
![image](https://user-images.githubusercontent.com/67309253/88165166-e3c52280-cc32-11ea-9e2a-9d6632efb8dc.png)

# Results

   * Random Forest:
   ![image](https://user-images.githubusercontent.com/67309253/88166362-c5602680-cc34-11ea-9047-019969cd222f.png)   
   * RBM:
   **Accuracy - 81 % (RMSE = 0.1939)**
   ![image](https://user-images.githubusercontent.com/67309253/88165565-87aece00-cc33-11ea-86b8-dcd774fade16.png)
   * Product Recommendations:
   ![image](https://user-images.githubusercontent.com/67309253/88166461-e9236c80-cc34-11ea-8619-ceaeefd68655.png)
   * Kaggle Submission for Version 1 model:
   ![image](https://user-images.githubusercontent.com/67309253/88166595-1ff98280-cc35-11ea-88cc-b3d1c07e59d4.png)
   
   
   
