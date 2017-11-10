# Test program to train ResNet on 
# CIFAR10 dataset (included with keras)

import keras
import keras_resnet.models
print("Imported modules")

shape, classes = (32, 32, 3), 10
x = keras.layers.Input(shape)
model = keras_resnet.models.ResNet50(x, classes=classes)
model.compile("adam", "categorical_crossentropy", ["accuracy"])
print("Compiled model")

(training_x, training_y), (_, _) = keras.datasets.cifar10.load_data()
training_y = keras.utils.np_utils.to_categorical(training_y)
print("Loaded data")

print("Fitting model...")
model.fit(training_x, training_y)
