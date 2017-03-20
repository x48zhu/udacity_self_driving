#**Traffic Sign Recognition** 
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/grayscale.png "Grayscaling"
[image3]: ./examples/random_noise.png "Random Noise"
[image4]: ./examples/german_traffic_sign1.jpg "Traffic Sign 1"
[image5]: ./examples/german_traffic_sign2.jpg "Traffic Sign 2"
[image6]: ./examples/german_traffic_sign3.jpg "Traffic Sign 3"
[image7]: ./examples/german_traffic_sign4.jpg "Traffic Sign 4"
[image8]: ./examples/german_traffic_sign5.jpg "Traffic Sign 5"
[image9]: ./examples/distribution.png "Data Distribution"

###Writeup / README

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the first code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799;
* The size of validation set is 4410;
* The size of test set is 12630;
* The shape of a traffic sign image is (32, 32, 3) in rgb format;
* The number of unique classes/labels in the data set is 43.

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. 

![Visualization of data set][image1]

Here is a bar chart showing how the data distributed.

![Distribution of data set][image9]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because the traffic signs are identified by the shape/pattern of the symbol, where colour doesn't help too much. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because the value each pixel can vary between 0 to 255 and such large range can make it difficult to converge during training. Therefore the value of each pixel of a grayscale image is normalized to a value between -0.5 to 0.5.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets.

The code for splitting the data into training and validation sets is contained in the fifth code cell of the IPython notebook. 

My final training set had X number of images. My validation set and test set had Y and Z number of images.

The sixth code cell of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data because the number of training images are not evenly distributed. Some classes have much more training data than the others. This may cause the final classifier performs good on those classes with more training data.

To add more data to the the data set, images randomly selected from classes with fewer training data are rotated, transformed by a random scale. This process stops until all classes have the same number of training data.

Here is an example of an original image and an augmented image:

![alt text][image3]

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the eighth cell of the ipython notebook. 

The final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   	     			| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6  	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU                  |                                               |
| Max pooling           | 2x2 stride,  outputs 5x5x16                   |
| Fully connected		| 400 input, 120 output        					|
| RELU  				|             									|
| Fully connected       | 120 input, 84 output                          |
| RELU					|												|
| Fully connected       | 84 input, 43 output                           |
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for defining the trainig pipeline is located in the ninth cell and code for training the model is located in the eleventh cell of the ipython notebook. 

To train the model, I used an Adam optimizer. I run the training for 15 epochs and during each epoch the model is trained by batch with batch size of 128. The learning rate is 0.001

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 93.3%
* validation set accuracy of 87.7%
* test set accuracy of 86.5%

If a well known architecture was chosen:
I adapt LeNet for training my Traffic Sign Classifier, since they aim at similar problem (e.g. image classification) and LeNet is complex enough for our problem.
The validation set accuracy keeps going up and gradually stops growing, which is a sign the training should be stopped otherwise it is likely to overfit. The test set accuracy is reasonable at 87.4%.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] 

![alt text][image5] 

![alt text][image6] 

![alt text][image7] 

![alt text][image8]

Some of these images are hard to predict (possibly be mis-classified) because: first, the backgrounds are different from the training data; second, some pictures are transformed (e.g. picture 3 is rotated by some angle).

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Children crossing    	| Children crossing								|
| No entry				| No entry										|
| Pedestrians	      	| Pedestrians      				 				|
| Right-of-way at the next intersection| Right-of-way at the next intersection|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 87.4%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 96%), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.96         			| Stop sign   									| 
| 0.01    				| Keep right 									|
| 0.008					| Speed limit (30km/h)							|
| 0.007	      			| No vehicles					 				|
| 0.003				    | Speed limit (120km/h)      					|


For the second image, the model is relatively sure that this is a Children crossing (probability of 99%), and the image does contain a stop sign. The top five soft max probabilities were

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| 0.99                  | Children crossing                             | 
| 0.004                 | Dangerous curve to the right                  |
| 8.97e-05              | Beware of ice/snow                            |
| 7.88e-05              | Right-of-way at the next intersection         |
| 1.46-06               | Slippery Road                                 |

For the third image, the model is relatively sure that this is a No entry (probability of 96%), and the image does contain a stop sign. The top five soft max probabilities were

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| 0.96                  | No entry                                      | 
| 0.03                  | Keep left                                     |
| 1.73e-05              | Turn right ahead                              |
| 6.76e-07              | End of all speed and passing limits           |
| 3.49e-07              | Go straight or left                           |

For the forth image, the model is relatively sure that this is a Pedestrians sign (probability of 99%), and the image does contain a stop sign. The top five soft max probabilities were

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| 0.99                  | Pedestrians                                   | 
| 0.002                 | Road narrows on the right                     |
| 0.0009                | Traffic signals                               |
| 4.44e-06              | General caution                               |
| 1.97e-07              | Right-of-way at the next intersection         |

For the fifth image, the model is relatively sure that this is a Right-of-way at the next intersection sign (probability of 99%), and the image does contain a stop sign. The top five soft max probabilities were

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| 0.99                  | Right-of-way at the next intersection         | 
| 0.005                 | Beware of ice/snow                            |
| 9.74e-07              | Slippery road                                 |
| 9.74e-07              | Double curve                                  |
| 3.00e-08              | Dangerous curve to the right                  |
