import csv

import matplotlib.image as mpimg

import cv2
import numpy as np
from keras.layers import BatchNormalization, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Dropout, Flatten, Lambda
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

samples = []
# This is a way I found to keep training data organized by certain categories
def pipeAllLinesToSamples(data_folder):
    with open('./{}/driving_log.csv'.format(data_folder)) as csvfile:
        before_len = len(samples)
        reader = csv.reader(csvfile)
        for line in reader:
            line.append(data_folder)
            samples.append(line)
        print('Added {} samples from {}'.format(len(samples) - before_len, data_folder))

pipeAllLinesToSamples('data') # just me driving through the course in one lap
pipeAllLinesToSamples('udacity_data') # The udacity-supplied data set
pipeAllLinesToSamples('reverse_data') # driving through the course in one reversed lap
# After running with the above three, the car was able to drive up to the first curve
# Because it didn't turn hard enough on the first curve, I figured to add 'recover' data, both as suggested
#   and to capture more training data related turning harder.
pipeAllLinesToSamples('recover_data') # mostly captured of recovering from lane-edges
pipeAllLinesToSamples('jungle_data') # one lap, forward and reversed, of the second, jungle, track
pipeAllLinesToSamples('curve_data') # recorded data of just curves to help balance out the training set more over straight data

print('Total of {} samples to be used for training.'.format(len(samples)))

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

file_name_idx = -1
def stripFolderPaths(image_path):
    return image_path.split('/')[file_name_idx]

def get_image(folder, file_name):
    path = './{}/IMG/{}'.format(folder, stripFolderPaths(file_name))
    return mpimg.imread(path)
    # maybe consider something like this? -> process_image(np.asarray(Image.open(path + row[0])))

# useful constants
center_path_idx = 0
left_path_idx = 1
right_path_idx = 2
measure_idx = 3
fldr_idx = -1

# Thanks to 'generator' section of Behavioral Cloning Project
def generator(samps, batch_size=256):
    num_samples = len(samps)
    while 1: # Loop forever so the generator never terminates
        shuffle(samps)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samps[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                cur_folder = batch_sample[fldr_idx]

                # Add center
                img_center = get_image(cur_folder, batch_sample[center_path_idx])
                images.append(img_center)

                steering_center = float(batch_sample[measure_idx])
                angles.append(steering_center)

                # emphasize jungle-driving to try to pass that track
                if 'jungle_data' == cur_folder:
                    # flip the center image
                    # Thanks to 'Data Augmentation' lecture
                    image_flipped = np.fliplr(img_center)
                    measurement_flipped = -steering_center

                    # create adjusted steering measurements for the side camera images
                    # Thanks to 'Using Multiple Cameras' lecture
                    correction = 0.16
                    steering_left = steering_center + correction
                    steering_right = steering_center - correction

                    img_left = get_image(cur_folder, batch_sample[center_path_idx])
                    img_right = get_image(cur_folder, batch_sample[center_path_idx])

                    images.extend([image_flipped, img_left, img_right])
                    angles.extend([measurement_flipped, steering_left, steering_right])

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

ch, row, col = 3, 80, 320  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x/225 - 0.5,
        input_shape=(160, col, ch)
                 #, output_shape=(ch, 160, col)
                 ))

model.add(Cropping2D(cropping=((55,25), (0,0)), input_shape=(160, 320, 3)))

# https://github.com/0bserver07/Nvidia-Autopilot-Keras/blob/master/model.py
model.add(BatchNormalization(epsilon=0.001,mode=2, axis=1,input_shape=(row,col,ch)))

model.add(Convolution2D(24, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
model.add(Convolution2D(36, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
model.add(Convolution2D(48, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))
model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))
model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.35))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(1, activation='tanh'))

model.summary()



model.compile(loss='mse', optimizer='adam')

model.fit_generator(
    generator=train_generator,
    samples_per_epoch=len(train_samples),
    validation_data=validation_generator,
    nb_val_samples=len(validation_samples),
    nb_epoch=3
)

model.save('model.h5')
