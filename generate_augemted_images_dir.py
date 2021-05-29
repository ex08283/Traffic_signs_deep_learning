import os, shutil
from keras.preprocessing import image_dataset_from_directory
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
# MNIST handwritten numbers classification.

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import matplotlib.pylab as plot
import numpy as np
from sklearn.metrics import confusion_matrix

# ################### DATA #############################
# Hier generered we augmented images, die we saven naar een directory om late als training images!
epoch = 10

base_dir = 'data_generator_flow'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
validation_dir = os.path.join(base_dir, 'validation')

base_dir = os.path.join('../Borden', 'data2')
os.mkdir(base_dir)
namen_borden = os.listdir('data_generator_flow/train')
namen_subfolder = ['train', 'validation', 'test']
for folder in namen_borden:
    os.mkdir(os.path.join(base_dir, folder))


train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True
                                       ) # recaling images


for path in os.listdir('data_generator_flow/train'):
    print(path)
    for image in os.listdir(os.path.join('data_generator_flow/train', path)):
        print(image)
        single_img = load_img(os.path.join('data_generator_flow/train', path, image))
        image_array = img_to_array(single_img)
        image_array = image_array.reshape((1,) + image_array.shape)
        i = 0
        for batch in train_datagen.flow(image_array, save_to_dir=os.path.join(base_dir, path), save_format='jpg'):
            i += 1
            if i > 20:
                break
