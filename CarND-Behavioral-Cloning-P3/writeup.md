#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/left_center_right.png "Recovery Image"
[image2]: ./examples/flip.png "Flip"

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the definition of the nueral network model.
* train.py containing the script that loads, pre-processes data and trains the network
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md or summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 8). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 15). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (train.py line 92-96). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 24).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. The steering angles for left and right camera images are adjusted accordingly by add/substract a angle value. This tells the center camera what to do during autonomous driving if it sees something like from the left/right camera: steering more.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture is given image input and output the steering angle based on the features from the image (e.g. position)

My first step was to use a convolution neural network model similar to the one shows in NVIDIA's paper. I thought this model might be appropriate because it involves multiple convolutional layers that sufficiently grasp the feature on the road.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model to include dropout layers.

Then I append three fully connected layers to get the one-dimensional output on steering anlge.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:

|Layer (type)            |   Filter |  Activation |    Output Shape  | 
|----------------|---|---|---|
|Cropping   | |  |    (None, 70, 320, 3) |      
|Normalization    |   |    |       (None, 70, 320, 3)  |         
|Convolution| 5x5 |Relu |   (None, 33, 158, 24) |
|Convolution| 5x5 |Relu |   (None, 15, 77, 36) |
|Convolution| 5x5 |Relu |  (None, 6, 37, 48)   |
|Convolution| 3x3 |Relu |  (None, 4, 35, 64)   |
|Convolution| 3x3 |Relu |  (None, 2, 33, 64) |
|MaxPooling2D || | (None, 1, 16, 64)    |
|Dropout | | |          (None, 1, 16, 64)   |
|Flatten |  |   |         (None, 1024)      |  
|Fully Connected | |Relu |    (None, 100)   |
|Fully Connected | |Relu |    (None, 50)   |
|Fully Connected | | Relu |    (None, 1)   |

####3. Creation of the Training Set & Training Process

I download the data collected by Udacity, which include 24108 images from left/center/right cameras. The left/right cameras images helps training the car to recover from side of the road.
![alt text][image1]

To augment the data set during training, I flipped images and angles thinking that this would utilize existing data and prevent the overfit on single side turning because of the imbalance in the data. For example, here is an image that has then been flipped:

![alt text][image2]

After the collection process, I had 48216 number of data points. When the model takes the image, it first crops the image by leaving only the road part, and then normalizes the image pixel into value between [-0.5, 0.5]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by the validation lost reaches the minimum. I used an adam optimizer so that manually training the learning rate wasn't necessary.
