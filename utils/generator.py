import os

import keras
import numpy as np
from PIL import Image
from keras.applications.inception_v3 import preprocess_input
from skimage import img_as_bool
from skimage.transform import resize

from .augment import image_augment_array, to_rgb, image_light_augment_array, image_binary_augment_array, flip_image, \
    zoom_image

class DataGenerator(keras.utils.Sequence):
    # Generates data for Keras
    def __init__(self, list_ids, path, look_up, mapping, finger_feature=False,
                 binary_mode=True, predict=False,
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
        self.finger_feature = finger_feature
        self.binary_mode = binary_mode
        self.predict = predict
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.mode = mode
        self.prop_image = prop_image
        self.prop_array = prop_array
        self.augment = self.prop_array > 0 or self.prop_image > 0
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

        if self.finger_feature:
            finger = []

        if not self.predict:
            y = np.zeros((self.batch_size), dtype=int)
            left = self.mapping.get('L')
            right = self.mapping.get('R')

        # Generate data
        for i, sample in enumerate(list_ids_temp):

            flipped = False

            with Image.open(os.path.join(self.path, sample.split('/')[-1] + '.png')) as x_img:
                x_img = x_img.convert(mode=self.mode)

                if self.binary_mode:
                    flipped, x_arr = self.read_binary_img(flipped, x_img)
                else:
                    flipped, x_arr = self.read_gray_mode(flipped, x_img)

                x_images['raw'][i,] = np.array(np.expand_dims(x_arr, 2), dtype=int)

            if self.finger_feature:
                index = int(sample[-2:])
                finger.append(index - 1)

            if not self.predict:
                _label = self.extract_and_correct_label(flipped, left, right, sample)
                y[i] = _label

        if self.finger_feature:
            x_images['finger'] = keras.utils.to_categorical(finger, num_classes=10)

        if self.predict:
            return x_images
        else:
            return x_images, keras.utils.to_categorical(y, num_classes=self.num_classes)

    def extract_and_correct_label(self, flipped, left, right, sample):
        label = self.look_up.get(sample)
        if not flipped:
            _label = label
        elif label == left:
            _label = right
        elif label == right:
            _label = left
        else:
            _label = label
        return _label

    def read_gray_mode(self, flipped, x_img):
        if self.augment:
            x_arr, flipped = image_light_augment_array(x_img, self.prop_image, self.prop_array)
        else:
            x_arr = np.array(x_img)
        x_arr *= 255.0 / x_arr.max()
        return flipped, x_arr

    def read_binary_img(self, flipped, x_img):
        if self.augment:
            x_arr, flipped = image_binary_augment_array(x_img, self.prop_image, self.prop_array)
            x_arr = img_as_bool(x_arr)
        else:
            x_arr = np.array(x_img)
            x_arr = img_as_bool(x_arr)
        x_arr = np.array(x_arr, dtype=int)
        return flipped, x_arr


class DataGeneratorPred(keras.utils.Sequence):
    # Generates data for Keras
    def __init__(self, list_ids, path, mapping, finger_feature=False, binary_mode=True,
                 dim=(512, 512), n_channels=1, mode='L'):
        # Initialization
        self.list_ids = list_ids
        if path:
            self.path = path
        else:
            self.path = ''
        self.num_classes = len(mapping)
        self.mapping = mapping
        self.finger_feature = finger_feature
        self.binary_mode = binary_mode
        self.batch_size = 4
        self.dim = dim
        self.n_channels = n_channels
        self.mode = mode
        self.on_epoch_end()
        self.indexes = np.arange(len(self.list_ids))

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.list_ids)))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index:(index+1)]

        # Find list of IDs
        list_ids_temp = [self.list_ids[k] for k in indexes]

        # Generate data
        x = self.__data_generation(list_ids_temp)

        return x

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.list_ids))

    def __data_generation(self, list_ids_temp):
        # Generates data containing batch_size samples X : (n_samples, *dim, n_channels)
        x_image_raw = np.zeros((4, *self.dim, self.n_channels))
        x_images = {'raw': x_image_raw}

        if self.finger_feature:
            finger = []

        # Generate data
        sample = list_ids_temp[0]

        with Image.open(os.path.join(self.path, sample.split('/')[-1] + '.png')) as x_img:
            x_img = x_img.convert(mode=self.mode)

            x_arr_org = np.array(x_img)
            x_arr_org_zoom = zoom_image(x_arr_org, False, True)
            x_arr_lr, _ = flip_image(x_arr_org, False)
            x_arr_lr_zoom = zoom_image(x_arr_lr, False, True)

            if self.binary_mode:
                x_images['raw'][0] = np.array(np.expand_dims(img_as_bool(x_arr_org), 2), dtype=int)
                x_images['raw'][1] = np.array(np.expand_dims(img_as_bool(x_arr_org_zoom), 2), dtype=int)
                x_images['raw'][2] = np.array(np.expand_dims(img_as_bool(x_arr_lr), 2), dtype=int)
                x_images['raw'][3] = np.array(np.expand_dims(img_as_bool(x_arr_lr_zoom), 2), dtype=int)
            else:
                x_images['raw'][0] = np.expand_dims(x_arr_org, 2)
                x_images['raw'][1] = np.expand_dims(x_arr_org_zoom, 2)
                x_images['raw'][2] = np.expand_dims(x_arr_lr, 2)
                x_images['raw'][3] = np.expand_dims(x_arr_lr_zoom, 2)

            if self.finger_feature:
                index = int(sample[-2:])
                for _ in range(4):
                    finger.append(index - 1)

        if self.finger_feature:
            x_images['finger'] = keras.utils.to_categorical(finger, num_classes=10)

        return x_images


def agg_ensemble(pred_array, mapping):
    left = mapping.get('L')
    right = mapping.get('R')

    _shape = pred_array.shape
    samples = _shape[0]//4
    out_array = np.zeros((samples,_shape[1]))

    for sample in range(samples):
        pred_array_sample = pred_array[(sample*4):(sample*4)+4]
        _array = np.zeros((4, _shape[1]))
        _array[:2,] = pred_array_sample[:2,]
        _array[2:, [left,right]] = _array[2:, [right,left]]
        out_array[sample, :] = np.sum(_array, axis=0)

    return out_array
