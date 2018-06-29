# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes, 2x2 subsampling, and depths between 24 and 64 (model-gen.py lines `63-76`) 

The model includes RELU layers to introduce nonlinearity (code line `71`, `73`, and `75`), and the data is normalized in the model using a Keras lambda layer (code line `63`). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model-gen.py lines `72` and `74`). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line `78`).

#### 4. Appropriate training data

Training data will be covered in the section `Creation of the Training Set and Training Process`

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the network in the `Even More Powerful Network` section. I thought this model might be appropriate because its initial results out the gate were strong and only needed more training data as opposed to more network layers.

As I began to have trouble with the validation accuracy and testing using `drive.py` I introduced more training data for tricky areas, particular edges for curves. After I determined my training data to be sufficient, I found the model to be overfitting around epoch 3. In response, I added dropout to the bottom 2 fully connected layers and found the model was better avoiding overfitting.

In order to better improve the the model taken from `Even More Powerful Network` I increased the fully connected layers number of parameters to `200`, `100`, `50`, `1`. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I added more short clips of moving from the outside of a curve to the inside quickly and precisely.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines `65-76`) consisted of a convolution neural network with the following layers and layer sizes.

```python
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
```

#### 3. Creation of the Training Set & Training Process


Choosing the training data was certainly the most important aspect of this project. My training data consisted of the following:

* 2 optimal laps around the track maintaining centrality in the lane as much as possible
* Small clips starting at the edges of the track and moving to the middle. Emphasis was placed on the edges of curves as those were particularly problematic during inital tests.
