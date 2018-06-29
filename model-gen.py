import csv
import cv2
import numpy as np
import os.path
import sklearn

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

from sklearn.model_selection import train_test_split

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for line in batch_samples:
                for i in range(3):
                    source_path = line[i]
                    filename = source_path.split('/')[-1]
                    current_path = './data/IMG/' + filename
                    if os.path.isfile(current_path):
                        image = cv2.cvtColor(cv2.imread(current_path), cv2.COLOR_BGR2RGB)
                        images.append(image)
                        if i == 1:
                            measurements.append(float(line[3]) + 0.15)
                        elif i == 2:
                            measurements.append(float(line[3]) - 0.15)
                        else:
                            measurements.append(float(line[3]))

            images_with_aug = []
            measurements_with_aug = []
            for image, measurement in zip(images, measurements):
                images_with_aug.append(image)
                measurements_with_aug.append(measurement)
                images_with_aug.append(cv2.flip(image,1))
                measurements_with_aug.append(measurement*-1.0)

            X_train = np.array(images_with_aug)
            y_train = np.array(measurements_with_aug)
            yield sklearn.utils.shuffle(X_train, y_train)

lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Used example network from the 'Even more powerful network' section video
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
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

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= 
            len(train_samples), validation_data=validation_generator, 
            nb_val_samples=len(validation_samples), nb_epoch=5)
model.save('model-gen.h5')
