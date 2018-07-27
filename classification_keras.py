import time

import keras
import keras.backend as K
import numpy as np
from keras import models
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Input, MaxPooling2D, Conv2D, Activation, BatchNormalization, AveragePooling2D, concatenate, \
    GlobalAveragePooling2D, Dense
from sklearn.metrics import confusion_matrix

from utils.analysis import image_confusion_matrix, plot_confusion_matrix
from utils.set_up_db import read_csv_to_dict, concat_ids_and_predictions
from utils.generator import DataGenerator, DataGeneratorPred, agg_ensemble


class Neural_Net():
    def __init__(self, img_size, num_classes, finger_feature=False,):
        """
        class that creates a neural net to classify fingerprints

        :param img_size: [tuple] of image size (w x h) as integer
        :param num_classes: [integer] providing the number of classes
        :param finger_feature: [boolean] whether to include the finger feature in the model (default: False).
        """
        self.img_size = img_size
        self.num_classes = num_classes

        self.finger_feature = finger_feature

        # Sets self to none for pep8
        self.neural_net = None

    def set_net(self, relative_size):
        """
        creates neural net and stores the custom and inception_v3 layers (the latter only if self.inception_pre
            is true). The neural net uses a selection of the blocks from the inception_v3 model. It size can be
            (relativly) adjusted using the relative size.

        :param relative_size: [float] Magnification of each layer
        :return: none
        """
        #  set channel axis, necessary for the concatenation layer (retrieved from the Keras backend)
        if K.image_data_format() == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 3

        # input (name matches that as provided by the generator) in grayscale, therefore only one channel
        raw = Input(shape=(*self.img_size, 1), name='raw')
        inputs = [raw]
        x = conv2d_bn_alt(raw, relative_size * 32, 3, 3, strides=(2, 2), padding='valid')
        x = conv2d_bn_alt(x, relative_size * 32, 3, 3, padding='valid')
        x = conv2d_bn_alt(x, relative_size * 64, 3, 3)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = conv2d_bn_alt(x, relative_size * 80, 1, 1, padding='valid')
        x = conv2d_bn_alt(x, relative_size * 100, 3, 3, padding='valid')
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        # mixed 0:
        branch1x1 = conv2d_bn_alt(x, relative_size * 64, 1, 1)

        branch5x5 = conv2d_bn_alt(x, relative_size * 48, 1, 1)
        branch5x5 = conv2d_bn_alt(branch5x5, relative_size * 64, 5, 5)

        branch3x3dbl = conv2d_bn_alt(x, relative_size * 64, 1, 1)
        branch3x3dbl = conv2d_bn_alt(branch3x3dbl, relative_size * 96, 3, 3)
        branch3x3dbl = conv2d_bn_alt(branch3x3dbl, relative_size * 96, 3, 3)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn_alt(branch_pool, 32, 1, 1)
        x = concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=channel_axis)

        # mixed 1:
        branch1x1 = conv2d_bn_alt(x, relative_size * 64, 1, 1)

        branch5x5 = conv2d_bn_alt(x, relative_size * 48, 1, 1)
        branch5x5 = conv2d_bn_alt(branch5x5, relative_size * 64, 5, 5)

        branch3x3dbl = conv2d_bn_alt(x, relative_size * 64, 1, 1)
        branch3x3dbl = conv2d_bn_alt(branch3x3dbl, relative_size * 96, 3, 3)
        branch3x3dbl = conv2d_bn_alt(branch3x3dbl, relative_size * 96, 3, 3)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn_alt(branch_pool, relative_size * 64, 1, 1)
        x = concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=channel_axis)

        # mixed 2:
        branch1x1 = conv2d_bn_alt(x, relative_size * 64, 1, 1)

        # mirrored compared to mixed 1 (in order to past ci-test)
        branch3x3dbl = conv2d_bn_alt(x, relative_size * 64, 1, 1)
        branch3x3dbl = conv2d_bn_alt(branch3x3dbl, relative_size * 96, 3, 3)
        branch3x3dbl = conv2d_bn_alt(branch3x3dbl, relative_size * 96, 3, 3)

        branch5x5 = conv2d_bn_alt(x, relative_size * 48, 1, 1)
        branch5x5 = conv2d_bn_alt(branch5x5, relative_size * 64, 5, 5)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn_alt(branch_pool, relative_size * 64, 1, 1)

        x = concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=channel_axis)

        # mixed 3:
        branch3x3 = conv2d_bn_alt(x, relative_size * 384, 3, 3, strides=(2, 2), padding='valid')

        branch3x3dbl = conv2d_bn_alt(x, relative_size * 64, 1, 1)
        branch3x3dbl = conv2d_bn_alt(branch3x3dbl, relative_size * 96, 3, 3)
        branch3x3dbl = conv2d_bn_alt(branch3x3dbl, relative_size * 96, 3, 3, strides=(2, 2), padding='valid')

        branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = concatenate([branch3x3, branch3x3dbl, branch_pool], axis=channel_axis)

        # mixed 4:
        branch1x1 = conv2d_bn_alt(x, relative_size * 192, 1, 1)

        branch7x7 = conv2d_bn_alt(x, relative_size * 128, 1, 1)
        branch7x7 = conv2d_bn_alt(branch7x7, relative_size * 128, 1, 7)
        branch7x7 = conv2d_bn_alt(branch7x7, relative_size * 192, 7, 1)

        branch7x7dbl = conv2d_bn_alt(x, relative_size * 128, 1, 1)
        branch7x7dbl = conv2d_bn_alt(branch7x7dbl, relative_size * 128, 7, 1)
        branch7x7dbl = conv2d_bn_alt(branch7x7dbl, relative_size * 128, 1, 7)
        branch7x7dbl = conv2d_bn_alt(branch7x7dbl, relative_size * 128, 7, 1)
        branch7x7dbl = conv2d_bn_alt(branch7x7dbl, relative_size * 192, 1, 7)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn_alt(branch_pool, relative_size * 192, 1, 1)
        x = concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=channel_axis)

        # mixed 5:
        branch1x1 = conv2d_bn_alt(x, relative_size * 192, 1, 1)

        branch7x7 = conv2d_bn_alt(x, relative_size * 192, 1, 1)
        branch7x7 = conv2d_bn_alt(branch7x7, relative_size * 192, 1, 7)
        branch7x7 = conv2d_bn_alt(branch7x7, relative_size * 192, 7, 1)

        branch7x7dbl = conv2d_bn_alt(x, relative_size * 192, 1, 1)
        branch7x7dbl = conv2d_bn_alt(branch7x7dbl, relative_size * 192, 7, 1)
        branch7x7dbl = conv2d_bn_alt(branch7x7dbl, relative_size * 192, 1, 7)
        branch7x7dbl = conv2d_bn_alt(branch7x7dbl, relative_size * 192, 7, 1)
        branch7x7dbl = conv2d_bn_alt(branch7x7dbl, relative_size * 192, 1, 7)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn_alt(branch_pool, relative_size * 192, 1, 1)

        x = concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=channel_axis)

        # mixed 6:
        branch3x3 = conv2d_bn_alt(x, relative_size * 192, 1, 1)
        branch3x3 = conv2d_bn_alt(branch3x3, relative_size * 320, 3, 3, strides=(2, 2), padding='valid')

        branch7x7x3 = conv2d_bn_alt(x, relative_size * 192, 1, 1)
        branch7x7x3 = conv2d_bn_alt(branch7x7x3, relative_size * 192, 1, 7)
        branch7x7x3 = conv2d_bn_alt(branch7x7x3, relative_size * 192, 7, 1)
        branch7x7x3 = conv2d_bn_alt(branch7x7x3, relative_size * 192, 3, 3, strides=(2, 2), padding='valid')

        branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = concatenate([branch3x3, branch7x7x3, branch_pool], axis=channel_axis)

        # end custom inception architecture
        x = GlobalAveragePooling2D()(x)

        # set dense layer (especially needed if pretrained is true)
        x = Dense(int(round(256 * relative_size)), activation='relu')(x)

        # if finger feature is  true include in the model
        if self.finger_feature:
            # name matches generator input, 5 as right/left hand is not taken into account
            finger = Input(shape=(10,), name='finger')
            inputs.append(finger)
            # combine with the output of the dense layer
            x_end = concatenate([x, finger])

        else:
            x_end = x

        # final softmax/dense layer for prediction
        y = Dense(self.num_classes, activation='softmax')(x_end)

        neural_net_model = models.Model(inputs=[*inputs], outputs=y)

        # set neural net
        self.neural_net = neural_net_model

    def compile(self, loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(), metrics=None):
        self.neural_net.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def fit(self, training_generator, validation_generator=None, epochs=None, callbacks=None, **kwargs):
        self.neural_net.fit_generator(generator=training_generator, validation_data=validation_generator,
                                      epochs=epochs, callbacks=callbacks, **kwargs)

    def predict(self, pred_generator):
        return self.neural_net.predict_generator(generator=pred_generator)

    def store_model(self, path):
        self.neural_net.save(path)


def conv2d_bn_alt(x, filters, num_row, num_col, padding='same', strides=(1, 1), name=None):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    filters = int(round(filters))

    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


def train_neural_net(ids_cat, mapping, use_finger_feature=True):
    ids = sorted(list(ids_cat.keys()))
    training_gen = DataGenerator(list_ids=ids[:-500], path='enhanced', look_up=ids_cat,
                                 mapping=mapping, finger_feature=use_finger_feature, binary_mode=True,
                                 batch_size=16,prop_image=0, prop_array=.5)
    valid_gen = DataGenerator(list_ids=ids[-500:], path='enhanced', look_up=ids_cat,
                              mapping=mapping, finger_feature=use_finger_feature, binary_mode=True,
                              batch_size=16,prop_image=0, prop_array=0)

    model = Neural_Net(img_size=(512,512), num_classes=len(mapping), finger_feature=use_finger_feature)
    model.set_net(relative_size=.2)
    model.compile(metrics=['acc'])

    lower_lear = ReduceLROnPlateau(monitor='loss', factor=.33, patience=10, verbose=0, mode='auto', cooldown=10)
    callback_tb = keras.callbacks.TensorBoard()

    model.fit(training_generator=training_gen, validation_generator=valid_gen,
              epochs=56, callbacks=[callback_tb, lower_lear])

    model.store_model('logs/model_{}.h5'.format(int(time.time())))

    return model


def predict_neural_net(model, ids_cat, mapping, use_finger_feature=False):
    ids = sorted(list(ids_cat.keys()))
    pred_ids = ids[-500:]
    pred_gen = DataGeneratorPred(list_ids=pred_ids, path='enhanced', finger_feature=True, mapping=mapping, binary_mode=True)
    preds = model.predict(pred_gen)
    pred_agg = agg_ensemble(pred_array=preds, mapping=mapping)
    _df_pred = concat_ids_and_predictions(pred_ids, pred_agg, ids_cat, mapping)
    confusion_mat = confusion_matrix(_df_pred['pattern'], _df_pred['pred_pattern'], labels=sorted(mapping.keys()))
    image_confusion_matrix(_df_pred, mapping)
    plot_confusion_matrix(confusion_mat, sorted(mapping.keys()))
    _df_pred.to_csv(path_or_buf='logs/db_pred_{}.csv'.format(int(time.time())))


if __name__ == '__main__':
    pass
