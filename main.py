# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os, shutil
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

from keras import models
from keras import layers
from keras import optimizers
from numpy import random


def print_hi(name):

       def create_dirs():
              print("welke borden zijn er: \n")
              namen_borden = os.listdir('../Borden/borden')
              print(namen_borden)

              print('\n---------------------------------------------------\n')
              print("Aantal borden per category")
              aantal_per_categorie = []
              for bord in namen_borden:
                     loc = os.path.join('../Borden/borden', bord)
                     aantal_per_categorie.append(len(os.listdir(loc)))
                     print(bord, ': {}'.format(len(os.listdir(loc))))

              print(aantal_per_categorie)
              naam_main_dir = 'data'
              os.chdir('../Borden')
              base_dir = os.path.join('../Borden', naam_main_dir)
              os.mkdir(base_dir)

              namen_subfolder = ['train', 'validation', 'test']
              for folder in namen_subfolder:
                     os.mkdir(os.path.join(base_dir, folder))
                     for naam in namen_borden:
                            os.mkdir(os.path.join(base_dir, folder, naam))

              for bord_type in namen_borden:
                     loc = os.path.join('../Borden/borden', bord_type)
                     borden = os.listdir(loc)
                     begin = len(borden)
                     while len(borden) > 0:
                            index = random.choice(len(borden))
                            if len(borden) > 0.8 * begin:
                                   loc1 = os.path.join(base_dir,'validation', bord_type)
                                   shutil.copy(os.path.join(loc, borden[index]), os.path.join(loc1, borden[index]))
                            elif len(borden) > 0.6 * begin:
                                   loc1 = os.path.join(base_dir,'test', bord_type)
                                   shutil.copy(os.path.join(loc, borden[index]), os.path.join(loc1, borden[index]))
                            else:
                                   loc1 = os.path.join(base_dir,'train', bord_type)
                                   shutil.copy(os.path.join(loc, borden[index]), os.path.join(loc1, borden[index]))
                            borden.pop(index)

       create_dirs()




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
