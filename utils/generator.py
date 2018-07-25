import os

import keras
import numpy as np
from PIL import Image
from keras.applications.inception_v3 import preprocess_input
from skimage.transform import resize

from .augment import image_augment_array, to_rgb, image_light_augment_array
from src.image_enhance import image_enhance

class DataGenerator(keras.utils.Sequence):
    # Generates data for Keras
    def __init__(self, list_ids, path, look_up, mapping, inception_pre=True, finger_feature=False,
                 predict=False,
                 batch_size=8, dim=(512, 512), n_channels=1, shuffle=True, mode='L',
                 prop_image=.25, prop_array=.5):
        # Initialization
        self.list_ids = list_ids
        if path:
            self.path = path
        else:
            self.path = ''
        self.look_up = look_up
        self.num_classes = len(mapping)
        self.mapping = mapping
        self.inception_pre = inception_pre
        self.finger_feature = finger_feature
        self.predict = predict
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.mode = mode
        self.prop_image = prop_image
        self.prop_array = prop_array
        self.on_epoch_end()
        self.indexes = np.arange(len(self.list_ids))

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.list_ids) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_ids_temp = [self.list_ids[k] for k in indexes]

        # Generate data
        if not self.predict:
            x, y = self.__data_generation(list_ids_temp)

            return x, y

        else:
            x = self.__data_generation(list_ids_temp)

            return x

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids_temp):
        # Generates data containing batch_size samples X : (n_samples, *dim, n_channels)
        x_image_raw = np.zeros((self.batch_size, *self.dim, self.n_channels))
        x_images = {'raw': x_image_raw}

        if self.inception_pre:
            x_image_pre = np.zeros((self.batch_size, 299, 299, 3))
            x_images['pre'] = x_image_pre

        if self.finger_feature:
            finger = []

        augment = self.prop_array == 0 and self.prop_image == 0

        if not self.predict:
            y = np.zeros((self.batch_size), dtype=int)
            left = self.mapping.get('L')
            right = self.mapping.get('R')


        # Generate data
        for i, sample in enumerate(list_ids_temp):

            flipped = False

            with Image.open(os.path.join(self.path, sample + '.png')) as x_img:
                x_img = x_img.convert(mode=self.mode)

                if augment:
                    x_arr = np.array(x_img)
                else:
                    x_arr, flipped = image_light_augment_array(x_img, self.prop_image, self.prop_array)

                x_arr = image_enhance(x_arr)
                x_arr = np.array(x_arr, dtype=int)

                x_arr_raw = resize(x_arr, output_shape=(self.dim[0], self.dim[1]))

                # normaliz here because preprocess for inception V3 should be clean
                _array_x_raw = x_arr_raw
                # _array_x_raw *= 255.0 / _array_x_raw.max()

                x_images['raw'][i, ] = np.expand_dims(_array_x_raw, 2)

                if self.inception_pre:
                    x_arr_pre = resize(x_arr, output_shape=(299, 299))
                    if self.mode == 'L':
                        x_arr_pre = to_rgb(x_arr_pre)

                    x_images['pre'][i, ] = preprocess_input(x_arr_pre)

                if self.finger_feature:
                    index = int(sample[-2:])
                    if index > 5:
                        index -= 5
                    finger.append(index - 1)

            if not self.predict:
                label = self.look_up.get(sample)
                if not flipped:
                    y[i] = label
                elif label == left:
                    y[i] = right
                elif label == right:
                    y[i] = left
                else:
                    y[i] = label

        if self.finger_feature:
            x_images['finger'] = keras.utils.to_categorical(finger, num_classes=5)

        if self.predict:
            return x_images
        else:
            return x_images, keras.utils.to_categorical(y, num_classes=self.num_classes)

