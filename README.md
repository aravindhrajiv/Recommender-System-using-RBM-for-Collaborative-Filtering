# Problem Statement
Printed circuit board (PCB) is the fundamental carrier in electronic devices on which a great number of elements are placed and the quality of the PCB will directly impact the performance of electronic devices.
To avoid shortcomings of manual detection, an automated optical inspection (AOI) based on machine vision need to be proposed.

**Major Use Case** - Detection and Classification of defects in PCBs

# Process at a Glance
![Product_flow_1](https://user-images.githubusercontent.com/67309253/85369942-9da17400-b54b-11ea-8578-6bc97086061a.PNG)

# Requirements
* Python 3.6.5
* Python Packages
    * Cv2        : Open Source Computer Vision Library
    * Pandas     : for dataframe
    * Numpy      : for numerical computations
    * Flask      : web application framework
    * Os         : for interacting with the operating system
    * Tensorflow : Open Source Dataflow library
    * Keras      : Open Source Neural Network library
    * Glob       : for reading/writing files to specific directories
    * Shutil     : for high-level operations of files
    * Sklearn    : Open Source Machine Learning library
    
# Python Scripts
**Jupyter Notebook** - Final Product.ipynb  

# Dataset
* There are 693 images in total of 10 different templates of PCBs 
* Types of defects :
    1. Missing Hole
    2. Mouse Bite
    3. Open Circuit
    4. Short Circuit
    5. Spur
    6. Spurious copper
* **For Classification** - A total of 3200 defects (80%-Training & 20%-Testing)
    * Training data - 2560 defects
    * Testing data - 640 defects 
    
# Code Description
1. Reading the input images from the dataset directory and storing in data and template lists
2. ##### Data Pre-Processing steps #####
   * Converting the images to grayscale and applying a Median Filter for denoising.
   ![ss_1](https://user-images.githubusercontent.com/67309253/85375646-3b00a600-b554-11ea-9476-5fe5b8f94644.PNG)
           
   *  Image Registration: 
   This is an algorithm deployed to overcome 2 main issues of the dataset provided.
       * Rotational Variance - The imput image may not be inclided as per the template image
       * Background Subtraction - The input image may contain other info apart from the template image.
       
       The set of algorithms followed in Image Registration:
       1. **SIFT** (Keypoints Descriptor):
       To find the keypoints which decsribes the input and template images.
       2. **Brute Force Matcher** (Keypoints Matching):
       To find the similar keypoints across the input and template images.
       3. **RANSAC** :
       To decide on the percentage of points that needed to matched from template to input image. We set it as 100% in our example.
       4. **Homography** :
       The input image is brought into the template's shape,allignment, etc.
       ![ss_2](https://user-images.githubusercontent.com/67309253/85377511-d561e900-b556-11ea-99fd-611e84c00baf.PNG)
       
 3. #### Detection Algorithm ####
    * IP Algorithm 1 : This algorithm was effective to find out 4 types of defects out of the 6 mentioned. 
         * Adaptive Thresholding
         * Image Subtraction
         * Mathematical Morphology:
         
               1. Opening with 2x2
               2. Closing with 15x15
               3. Median filtering with 5x5
               4. Closing with 29x29 ellipse
               5. Opening with 3x3
               6. Opening with 1x1
               
         ![ss_3](https://user-images.githubusercontent.com/67309253/85380133-317a3c80-b55a-11ea-9eae-4cf0c58c7b91.PNG)
               
    * IP Algorithm 2: This algorithm was effective to find out the remaining 2 types of defects. 
         * Image Subtraction
         * Binary Thresholding 
         
      ![ss_4](https://user-images.githubusercontent.com/67309253/85380218-4c4cb100-b55a-11ea-8ecd-4f2109101e89.PNG)
         
    * Combining both the algorithms and taking out the best results out of them
    
      ![ss_5](https://user-images.githubusercontent.com/67309253/85380247-553d8280-b55a-11ea-8f6d-ea80ad8c2a8a.PNG)
    * Creating a bounding box around the identified defects and snipping them out of the input image and storing those defects in the respective defect directory (tagging). These would be our input for the classification algorithm.
    
      ![ss_6](https://user-images.githubusercontent.com/67309253/85380244-540c5580-b55a-11ea-9a26-7e6f8af7da9f.PNG)
    * Creating a CSV file regarding the input file, type of defect, location coordinates of defect and other information.
    
      ![ss_7](https://user-images.githubusercontent.com/67309253/85380235-52db2880-b55a-11ea-9de8-f628d5014745.PNG)
  
4. #### Classification Algorithm ####
We make a simple Convolutional Neural Network (CNN) inorder to perform this clssification of defects task.
   
   * Data Pre-processing for CNN :
       1. One-hot encoding of defect classes - target 
       2. Image Resizing to (64x64x3)
       3. Data standarization - Dividing the pixels values by 255
       4. Random Splitting of dataset into 80% Training and 20% Testing
       
      Final shapes of the arrays:
      
        ![ss_8](https://user-images.githubusercontent.com/67309253/85381841-00027080-b55c-11ea-863b-71fea4f7340b.PNG)
    
   * Model Architecture : (Best one)
   
   X_input --> Conv Layer 1 --> Pool Layer 1 --> Conv Layer 2 --> Pool Layer 2 --> Conv Layer 3 --> Pool Layer 3 --> Conv Layer 4 --> Pool Layer 4 --> Flatten --> FC Layer 1 --> FC Layer 2
   
       * X_input : Shape - (?,64,64,3)
       * Conv Layer 1 : Shape - (?,64,64,8)
           * fitler_size = 8, conv_size = 16, stride = 1, padding = same, l2 regularizer = 1e-5
           * BatchNormalization
           * RELU activation
       * Pool Layer 1 : Shape - (?,32,32,8)
           * max pooling - pool_size = 2
       * Conv Layer 2 : Shape - (?,32,32,16)
           * fitler_size = 16, conv_size = 8, stride = 1, padding = same, l2 regularizer = 1e-5
           * BatchNormalization
           * RELU activation
       * Pool Layer 2 : Shape - (?,16,16,16)
           * max pooling - pool_size = 2
       * Conv Layer 3 : Shape - (?,16,16,32)
           * fitler_size = 32, conv_size = 4, stride = 1, padding = same, l2 regularizer = 1e-5
           * BatchNormalization
           * RELU activation
       * Pool Layer 3 : Shape - (?,8,8,32)
           * max pooling - pool_size = 2
       * Conv Layer 4 : Shape - (?,8,8,64)
           * fitler_size = 64, conv_size = 2, stride = 1, padding = same, l2 regularizer = 1e-5
           * BatchNormalization
           * RELU activation
       * Pool Layer 4 : Shape - (?,2,2,64)
           * max pooling - pool_size = 4
       * Flatten : Shape - (?,256)
       * FC Layer 1 : Shape - (?,64)
           * Dense - RELU activation, l2 regularizer = 1e-5
       * FC Layer 2 : Shape - (?,6)
           * Dense - Softmax activation, l2 regularizer = 1e-5
   
   * Model Creation:
       * Optimizer - ADAM
       * Loss - Categorical CrossEntropy
       * Metrics - Accuracy
       
   * Hyperparameters Tuned:
        1. No. of epochs = 40
        2. Mini-batch size = 16
        3. l2 regularizers = 1e-5
        4. Spatial Dropouts2D - only for experiments (not in main model)
        5. Dropouts - only for experiments (not in main model)
        
 5. Saving the model. Sending the entire defect dataset for the model.
 
  ![ss_10](https://user-images.githubusercontent.com/67309253/85394805-c0905000-b56c-11ea-9cb2-744bc28bb4f6.PNG)
   ![ss_11](https://user-images.githubusercontent.com/67309253/85394801-bff7b980-b56c-11ea-812e-d9603516d27b.PNG)
   
 6. Marking the defect using detection algorithm and predicting the class of defect using the classification algorithm in the input images. Later, storing the output images in a different directory.
 

# Results

   * Detection: Detected most of the major defects
   * Classification:
        1. Training Accuracy - 93.8%
        2. Testing Accuracy - 93.2%
   * Confusion Matrix for the entire dataset:
   
     ![ss_9](https://user-images.githubusercontent.com/67309253/85394133-b883e080-b56b-11ea-8cbb-27e139e6f26c.PNG)
   
