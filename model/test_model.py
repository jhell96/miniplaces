import keras
from keras.preprocessing import image
from keras.applications import Xception
from keras.metrics import top_k_categorical_accuracy
from keras import backend as K
import keras_resnet.models
import numpy as np
from DataLoader import *
import os
from scipy import misc
print("Imported modules")

img_width, img_height = 128, 128
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

model = Xception(include_top=True, weights=None, input_shape=input_shape, pooling=None, classes=num_classes)
model.load_weights('weights/trained_xception_centered_2.h5')
model.compile("adam", "categorical_crossentropy", ["accuracy", "top_k_categorical_accuracy"])
print("Compiled model")

test_datagen = image.ImageDataGenerator(
        featurewise_center=True, # set input mean to 0 over the dataset
        samplewise_center=False, # set each sample mean to 0
        featurewise_std_normalization=True, # divide inputs by std of the dataset
        samplewise_std_normalization=False, # divide each input by its std
        zca_whitening=False, # apply ZCA whitening
        rotation_range=0, # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0, # randomly shift images horizontally (fraction of total width)
        height_shift_range=0, # randomly shift images vertically (fraction of total height)
        horizontal_flip=False, # randomly flip images
        vertical_flip=False) # randomly flip images

files = []
for (d, dn, f) in os.walk(test_path + 'test/'):
	files.extend(f)

files = sorted(files)

imgs = []
for file in files:
	imgs.append(misc.imread(test_path + 'test/' + file))

imgs = np.array(imgs)

test_datagen.fit(imgs)

test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        shuffle = False,
        class_mode=None)

# results = model.predict(imgs, batch_size=batch_size)
results = model.predict_generator(test_generator, steps = nb_validation_samples//batch_size)
top_5 = np.flip(np.argsort(results, axis=1)[:,-5:], 1)

# mapping = [0] * 100
# f = open("../data/categories.txt", "r")
# for line in f:
#     result = [x.strip() for x in line.split(' ')]
#     category, i = result[0], int(result[1])
#     mapping[i] = category

# mapping = np.array(mapping)
# for i in range(len(top_5)):
#     print(files[i])
#     print(mapping[top_5[i]])

out = open("output.txt", "a") # output file
for i, line in enumerate(top_5):
    t = top_5[i]
    out.write('test/' + files[i] + ' ' + t[0] + ' ' + t[1] + ' ' + t[2] + ' ' + t[3] + ' ' + t[4] + '\n')
    