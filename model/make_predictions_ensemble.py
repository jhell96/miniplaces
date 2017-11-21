import keras
from keras.preprocessing import image
from keras.applications import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.resnet50 import ResNet50
from keras.metrics import top_k_categorical_accuracy
from keras import backend as K
import keras_resnet.models
import numpy as np
import os
from scipy import misc
print("Imported modules")

img_width_x, img_height_x = 128, 128
img_width_inc, img_height_inc = 139, 139
img_width_res, img_height_res = 197, 197
num_classes = 100

epochs = 2
batch_size = 32
nb_train_samples = 100000
nb_validation_samples = 10000

train_path = '../data/images/train/'
val_path = '../data/images/val/'
test_path = '../data/images/test/'

if K.image_data_format() == 'channels_first':
    input_shape_x = (3, img_width_x, img_height_x)
    input_shape_inc = (3, img_width_inc, img_height_inc)
    input_shape_res = (3, img_width_res, img_height_res)
else:
    input_shape_x = (img_width_x, img_height_x, 3)
    input_shape_inc = (img_width_inc, img_height_inc, 3)
    input_shape_res = (img_width_res, img_height_res, 3)

model_x = Xception(include_top=True, weights=None, input_shape=input_shape_x, pooling=None, classes=num_classes)
model_inc = InceptionResNetV2(include_top=True, weights=None, input_shape=input_shape_inc, pooling=None, classes=num_classes)
model_res = ResNet50(include_top=True, weights=None, input_shape=input_shape_res, pooling=None, classes=num_classes)

model_x.load_weights('weights/trained_xception_centered_2.h5')
model_inc.load_weights('weights/trained_inception_resnet_v2_centered_3.h5')
model_res.load_weights('weights/trained_resnet50_centered_1.h5')

model_x.compile("adam", "categorical_crossentropy", ["accuracy", "top_k_categorical_accuracy"])
model_inc.compile("adam", "categorical_crossentropy", ["accuracy", "top_k_categorical_accuracy"])
model_res.compile("adam", "categorical_crossentropy", ["accuracy", "top_k_categorical_accuracy"])
print("Compiled models")

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

test_generator_x = test_datagen.flow_from_directory(
        test_path,
        target_size = (img_width_x, img_height_x),
        batch_size = batch_size,
        shuffle = False,
        class_mode=None)

test_generator_inc = test_datagen.flow_from_directory(
        test_path,
        target_size = (img_width_inc, img_height_inc),
        batch_size = batch_size,
        shuffle = False,
        class_mode=None)

test_generator_res = test_datagen.flow_from_directory(
        test_path,
        target_size = (img_width_res, img_height_res),
        batch_size = batch_size,
        shuffle = False,
        class_mode=None)

results_x = model_x.predict_generator(test_generator_x, steps = nb_validation_samples//batch_size)
results_inc = model_inc.predict_generator(test_generator_inc, steps = nb_validation_samples//batch_size)
results_res = model_res.predict_generator(test_generator_res, steps = nb_validation_samples//batch_size)

# xception - validation -  loss: 2.994, acc: 0.468, top5 acc: 0.754
# incep_res - validation - loss: 2.884, acc: 0.444, top5 acc: 0.734
# resnet50 - validation - loss:, acc:, top5 acc:
acc_sum = (0.754 + 0.734)
results_ensembled = np.average([results_x, results_inc], weights=[0.754/acc_sum, 0.734/(acc_sum)], axis=0)

top_5 = np.flip(np.argsort(results_ensembled, axis=1)[:,-5:], 1)

out = open("output.txt", "a") # output file
for i in range(len(top_5)):
    t = top_5[i]
    out.write('test/' + files[i] + ' ' + str(t[0]) + ' ' + str(t[1]) + ' ' + str(t[2]) + ' ' + str(t[3]) + ' ' + str(t[4]) + '\n')
    
