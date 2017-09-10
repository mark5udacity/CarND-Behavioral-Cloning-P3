#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report
** This here markdown readme is the written report!


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

    ## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists is a direct copy of the NVideo Neural Network architecture.  
This was due to suggestions from the 'what I wish I knew' document and from the lecture notes itself.
The first three layers consist of a convolution neural network with 5x5 filter sizes and depths of 24, 36, and 48. 
Then there are two layers of 3x3 that both have a depth 64.  
Following that are fully-connected layers of 1164, 100, 50, 10 and 1 (the output) layers each.

The model.py defines this style beginning at line 82.
 
The model includes RELU layers to introduce nonlinearity (as defined in the first 5 convulational layers,
 and the data is uses dropout on line 89 after the first fully connected lyaer to avoid overfitting the training data. 

The summary() printout from the keras model is as followed (along with notes of how much training data I collect):

```
Added 3590 samples from data
Added 8036 samples from udacity_data
Added 2240 samples from reverse_data
Added 1311 samples from recover_data
Added 3632 samples from jungle_data
Added 1112 samples from curve_data
Total of 19921 samples to be used for training.
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 80, 320, 3)    0           lambda_1[0][0]                   
____________________________________________________________________________________________________
batchnormalization_1 (BatchNorma (None, 80, 320, 3)    320         cropping2d_1[0][0]               
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 38, 158, 24)   1824        batchnormalization_1[0][0]       
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 17, 77, 36)    21636       convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 7, 37, 48)     43248       convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 5, 35, 64)     27712       convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 3, 33, 64)     36928       convolution2d_4[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 6336)          0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 1164)          7376268     flatten_1[0][0]                  
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 1164)          0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 100)           116500      dropout_1[0][0]                  
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 50)            5050        dense_2[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 10)            510         dense_3[0][0]                    
____________________________________________________________________________________________________
dense_5 (Dense)                  (None, 1)             11          dense_4[0][0]                    
====================================================================================================
Total params: 7,630,007
Trainable params: 7,629,847
Non-trainable params: 160
____________________________
```

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 89).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. 
The training data I used can be broken down into 6 separate training sessions.  The first three I ran my model against
proved promising, with the final three brining my car over the finish line
I used a combination of:
# center lane driving for one lap
# driving in the opposite direction for one lap
# the Udacity-supplied training data
# recovering from the left and right sides of the road 
# training exclusively recorded on curve data, driving back and forth
# the second, jungle, track data, one loop forward and back

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to follow the predominate advice to use the NVidia architecture
This architecture, as the paper goes into, already nicely captures exactly the problem of taking input data and output a single
variable of the steering wheel angle to do.  The NVidia paper is more comprehensive as it is an actual car driving on a
a real road, whereas in this project, we are using simulated data using images taken from a Unity game world model.  
The NVidia paper describes using 80 hours of training data to train its models, but since we are talking about a much
simpler driving course with simpler camera-input, we did not need anywhere near as much.  

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

The main way I 'validated' my model was to see how it did on the track, since that is the most important goal after-all.

I did not find it too useful to look at training and validation set loss as the model rapidly, even within the first epoch,
obtain a low (less than 2%) loss on the training data.  

As mentioned earlier, the first time I tried my model with the full NVidia architecture, it fell off the track at the
very first curve.  Adding the appropriate training data on the second go-round was sufficient to get the car to complete a full loop
successfully, as found in the run1.mp4 video in this same report.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

When I tried the vehicle on the jungle-track, which personally I found hard to drive myself at first, the car almost
was able to get around the whole track in one loop.  However, I had to manually intervene 3 times to get the car unstuck
from goal posts, but I am confident if I give more training data on the jungle track, and consider more normalization,
that I may be able to get the car to successfully go all the way around as well.


####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving.

I then drove the vehicle around the track in the opposite direction.

I thought maybe throttle would be an input, turns out I luckily chose to drive between 10-20 MPH, which is only slight
 more than what I see the drive.py aim for (with 9MPH).  At first when I drove through the track manually, I was 
 going at the max 30MPH and found it hard to complete. 

After supplying the above two data sets, along with the Udacity sets, my car seemed to drive pretty steady at first,
only falling a little wide after the first curve.  So the next three data sets I add were to add recover, 
a different track (the jungle) and curve-focused data.

I did not find it necessary to augment the training data in order to pass, thankfully.  Though if I had to, I imagine
flipping the data and also taking advantage of the left and right cameras would be of use.

After the collection process, I had X19921number of data points. I then preprocessed this data by in two ways,
by using a Keras Lambda layer to normalize the data (as suggested by the Udacity notes) and to crop the data image as
another Keras step that is supªplied with Cropping2D.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 

The ideal number of epochs was 3 as evidenced by the car successfully completing the course.  I am tempted to say that 
even one or two would have been sufficient.  This is because after my first epoch of a run and I am doing now, while typing
up, had a training loss of 0.0371 and a validation loss of 0.270 after the first epoch.
ª
I used an adam optimizer so that manually training the learning rate wasn't necessary.


# Review notes

Upon receiving a review of my project, it turns out I was reading with cv2.imread, which read in images as BGR,
but the Autonomous drive module reads in images as RGB.  I fixed this by using mpimg.

In addition, I augmented the data by adding an extra image that was flipped (along with it's measurement) and
adding the left and right camera views.  In addition, I added two more epochs of training to bring down the validation loss
 by a few percentage points.  The car now definitely stays well within the lines and also does not veer like it did
 in my first run. I went with .16 as the correction value to use for left/right camera measurement views after trying
 with .08, .16, .32 and .64 and observing performance on the track afterwards.
 
 I got very close with the jungle data as well, but not quite well-enough that I would want to be a passenger
 while the car is driving.  I believe with more training data, the car may be able to navigate the jungle course.

## Here is an example of an image from my recorded data set

[//]: # (Image References)
[center_view]: ./write_up_images/center_view.jpg "Center View"

[//]: # (Image References)
[flipped]: ./write_up_images/center_view_flipped.jpg "Flipped Center View"

[//]: # (Image References)
[right]: ./write_up_images/right_view.jpg "Right View"

[//]: # (Left View)
[left]: ./write_up_images/left_view.jpg "Left View"

 ![alt text][center_view]
 ![alt text][flipped]
 ![alt text][right]
 ![alt text][left]
