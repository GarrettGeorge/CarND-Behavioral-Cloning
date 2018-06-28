# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

### Have the model parameters been tuned appropriately?

Used `adam` optimizer for ease of tuning other parameters and the quickness of prototyping newer and more effcient models.

### Is the training data chosen appropriately?

Choosing the training data was certainly the most important aspect of this project. My training data consisted of the following:

* 2 optimal laps around the track maintaining centrality in the lane as much as possible
* Small clips starting at the edges of the track and moving to the middle. Emphasis was placed on the edges of curves as those were particularly problematic during inital tests.

### Architecture and Training Documentation

When picking a solution I began with the model introduced in the `Even More Powerful Network`. As I began to have trouble with the validation accuracy and testing using `drive.py` I introduced more training data for tricky areas, particular edges for curves. After I determined my training data to be sufficient, I found the model to be overfitting around epoch 3. In response, I added dropout to the bottom 2 fully connected layers and found the model was better avoiding overfitting.

In order to better improve the the model taken from `Even More Powerful Network` I increased the fully connected layers number of parameters to `200`, `100`, `50`, `1`. 
