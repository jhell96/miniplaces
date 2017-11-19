import keras
from keras.preprocessing import image
from keras import backend as K
import keras_resnet.models
import numpy as np
from DataLoader import *
import os
from scipy import misc
print("Imported modules")

def get_VGG(input_shape):

    model = Sequential([
        Conv2D(64, (3, 3), input_shape=input_shape, padding='same',
               activation='relu'),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same',),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(256, (3, 3), activation='relu', padding='same',),
        Conv2D(256, (3, 3), activation='relu', padding='same',),
        Conv2D(256, (3, 3), activation='relu', padding='same',),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(512, (3, 3), activation='relu', padding='same',),
        Conv2D(512, (3, 3), activation='relu', padding='same',),
        Conv2D(512, (3, 3), activation='relu', padding='same',),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(512, (3, 3), activation='relu', padding='same',),
        Conv2D(512, (3, 3), activation='relu', padding='same',),
        Conv2D(512, (3, 3), activation='relu', padding='same',),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Flatten(),
        Dense(4096, activation='relu'),
        Dense(4096, activation='relu'),
        Dense(1000, activation='softmax')
    ])

    return model


img_width, img_height = 224, 224
num_classes = 100

epochs = 2
batch_size = 32
nb_train_samples = 100000
nb_validation_samples = 10000

train_path = '../data/images/train/'
val_path = '../data/images/val/'
test_path = '../data/images/test/'

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


model = get_VGG(input_shape)
model.compile("adam", "categorical_crossentropy", ["accuracy"])
model.load_weights('weights/trained_vgg_centered.h5')
print("Compiled model")

datagen = image.ImageDataGenerator(
        featurewise_center=True, # set input mean to 0 over the dataset
        samplewise_center=False, # set each sample mean to 0
        featurewise_std_normalization=True, # divide inputs by std of the dataset
        samplewise_std_normalization=False, # divide each input by its std
        zca_whitening=False, # apply ZCA whitening
        rotation_range=20, # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2, # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2, # randomly shift images vertically (fraction of total height)
        horizontal_flip=True, # randomly flip images
        vertical_flip=False) # randomly flip images

files = []
for (d, dn, f) in os.walk(test_path):
    files.extend(f)

imgs = []
for file in files:
    imgs.append(misc.imread(test_path + file))

imgs = np.array(imgs)

datagen.fit(imgs)

train_generator = datagen.flow_from_directory(
        train_path,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        shuffle = True)

validation_generator = datagen.flow_from_directory(
        val_path,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        shuffle = False)

model.fit_generator(
        train_generator,
        steps_per_epoch = nb_train_samples // batch_size, 
        epochs = epochs,
        validation_data = validation_generator,
        validation_steps = nb_validation_samples // batch_size)

model.save_weights('trained_vgg_centered_1.h5')

print("Optimization Finished!")