# nvidia arch, cropping, flipping images and measurements
# back to 5 epochs
# adding L and R camera images and angles if angles not 0

import csv
import cv2
import numpy as np

lines = []
with open('../carnd/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []
for line in lines:
    #find and add center images and steering angles
    source_C = line[0]
    filename_C = source_C[-34:]
    awspath = '../carnd/data/IMG/' + filename_C
    image_C = cv2.imread(awspath)
    images.append(image_C)
    steering_C = float(line[3])
    measurements.append(steering_C)
    opp_image_C = np.fliplr(image_C)
    opp_strg_C = -steering_C
    images.append(opp_image_C)
    measurements.append(opp_strg_C)
    # find and add left images and alter steering angles if angle is not 0
    if steering_C>0:
        source_L = line[1]
        filename_L = source_L[-32:]
        awspath_L = '../carnd/data/IMG/' + filename_L
        image_L = cv2.imread(awspath_L)
        images.append(image_L)
        steering_L = steering_C + .25
        measurements.append(steering_L)
        opp_image_L = np.fliplr(image_L)
        opp_strg_L = -steering_L
        images.append(opp_image_L)
        measurements.append(opp_strg_L)
    # find and add right images and alter steering angles if angle is not 0
    if steering_C<0:
        source_R = line[2]
        filename_R = source_R[-33:]
        awspath_R = '../carnd/data/IMG/' + filename_R
        image_R = cv2.imread(awspath_R)
        images.append(image_R)
        steering_R = steering_C - .25
        measurements.append(steering_R)
        opp_image_R = np.fliplr(image_R)
        opp_strg_R = -steering_R
        images.append(opp_image_R)
        measurements.append(opp_strg_R)
        
X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
 
model = Sequential()
model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
#model.add(MaxPooling2D())
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
#model.add(MaxPooling2D())
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=.2, shuffle=True, nb_epoch=5)

model.save('model.h5')

