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
    def __init__(self, img_size, num_classes, inception_pre=True, finger_feature=False,):
        """
        class that creates a neural net to classify fingerprints

        :param img_size: [tuple] of image size (w x h) as integer
        :param num_classes: [integer] providing the number of classes
        :param inception_pre: [boolean] whether to include the pretrained inception part in the model (default: True).
        :param finger_feature: [boolean] whether to include the finger feature in the model (default: False).
        """
        self.img_size = img_size
        self.num_classes = num_classes

        self.inception_pre = inception_pre
        self.finger_feature = finger_feature

        # Sets self to none for pep8
        self.neural_net = None
        self.inception_v3_layers = None
        self.custom_layers = None

        self.freezed_weight_checks = None
        self.trained_weight_checks = None

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

        # if inception pretrained is true include in model
        if self.inception_pre:
            # name matches generator input and three channels as pretrained inception is  build upon rgb-images
            pre = Input(shape=(299, 299, 3), name='pre')
            inputs.append(pre)

            # insert image into the inception_v3 model
            x_pre = InceptionV3(include_top=False, input_tensor=pre, pooling='avg')
            # x_pre.output to get output from the keras application
            x_combined = concatenate([x, x_pre.output])

            # store layers
            self.inception_v3_layers = x_pre.layers

        # else set x to combined as this is the only part of the architecture
        else:
            x_combined = x

        # set dense layer (especially needed if pretrained is true)
        x_combined = Dense(int(round(256 * relative_size)), activation='relu')(x_combined)

        # if finger feature is  true include in the model
        if self.finger_feature:
            # name matches generator input, 5 as right/left hand is not taken into account
            finger = Input(shape=(5,), name='finger')
            inputs.append(finger)
            # combine with the output of the dense layer
            x_end = concatenate([x_combined, finger])

        else:
            x_end = x_combined

        # final softmax/dense layer for prediction
        y = Dense(self.num_classes, activation='softmax')(x_end)

        neural_net_model = models.Model(inputs=[*inputs], outputs=y)

        # set neural net
        self.neural_net = neural_net_model

        if self.inception_pre:
            self.custom_layers = [x for x in neural_net_model.layers if x not in self.inception_v3_layers]
        else:
            self.custom_layers = [x for x in neural_net_model.layers]

    def pre_compile(self, check=True):
        """
        this freezes the  inception_v3 blocks and stores some layers weights to check for succesfull training/
            freezing

        :param check: [boolean] to indicate if layer weights should be stored
        :return: none
        """
        # if pretrained inception_v3 is used freeze all layers associated with this mode/application
        if self.inception_pre:
            for layer2freeze in self.inception_v3_layers:
                self.neural_net.get_layer(layer2freeze.name).trainable = False

        # if check store (selection of) weights from the custom layers and (if applicable) the inception_v3 layers
        if check:
            if self.inception_pre:
                freezed_batchnorm_conv_weights = self.get_batchnorm_conv_weights(self.inception_v3_layers)
                self.freezed_weight_checks = freezed_batchnorm_conv_weights

            trained_batchnorm_conv_weights = self.get_batchnorm_conv_weights(self.custom_layers)
            self.trained_weight_checks = trained_batchnorm_conv_weights

    @property
    def freezed_weights(self):
        """
        property that indicates if training is (technically) successful.

        :return: [list] with boolean(s) to indicate if training is successful. First boolean is for the custom layers
            the second (if available) is for the inception_v3 layers
        """
        if self.trained_weight_checks is None:
            print('No weights are available to use as reference')
            return None

        checks = self.check_layers_weights_with_reference(layers=self.custom_layers,
                                                          ref_weights=self.trained_weight_checks)
        # all checks should be false for successful training of the custom inception layers
        if not any(checks):
            print('Training of custom Inception successful')
            output = [True]
        else:
            print('Training of custom Inception unsuccessful')
            output = [False]

        if self.freezed_weight_checks:
            checks = self.check_layers_weights_with_reference(layers=self.inception_v3_layers,
                                                              ref_weights=self.freezed_weight_checks)
            # all checks should be true for successful 'training' of the inception_v3 layers
            if all(checks):
                print('Freezing of InceptionV3 successful')
                output.append(True)
            else:
                print('Freezing of InceptionV3 unsuccessful')
                output.append(False)

        # return list
        return output

    def check_layers_weights_with_reference(self, layers, ref_weights):
        """
        Checks if layer weights are equal or different.

        :param layers: layers to be used for comparison
        :param ref_weights: numpy array with the reference weights
        :return: [list] of booleans where True indicates equal weights
        """
        weights = self.get_batchnorm_conv_weights(layers)
        checks = []
        for idx in range(len(weights)):
            checks.append(np.array_equal(ref_weights[idx], weights[idx]))
        return checks

    def get_batchnorm_conv_weights(self, layers):
        batch_conv_weights = []

        for layer in layers:
            if 'batch_normalization' in layer.name:
                batch_conv_weights.append(self.neural_net.get_layer(layer.name).get_weights())
                break

        for layer in layers:
            if 'conv2d' in layer.name:
                batch_conv_weights.append(self.neural_net.get_layer(layer.name).get_weights())
                break

        return batch_conv_weights

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


def train_neural_net(ids_cat, mapping, use_pretrained_inception=False, use_finger_feature=True):
    ids = sorted(list(ids_cat.keys()))
    training_gen = DataGenerator(list_ids=ids[:-500], path='enhanced', look_up=ids_cat, mapping=mapping,
                                 inception_pre=use_pretrained_inception, finger_feature=use_finger_feature,
                                 batch_size=16,prop_image=0.25, prop_array=0.5)
    valid_gen = DataGenerator(list_ids=ids[-500:], path='enhanced', look_up=ids_cat, mapping=mapping,
                              inception_pre=use_pretrained_inception, finger_feature=use_finger_feature,
                              batch_size=16,prop_image=0, prop_array=0)

    model = Neural_Net(img_size=(512,512), num_classes=len(mapping),
                       inception_pre=use_pretrained_inception, finger_feature=use_finger_feature)
    model.set_net(relative_size=.5)
    model.pre_compile()
    model.compile(metrics=['acc'])
    model.freezed_weights

    # return model

    lower_lear = ReduceLROnPlateau(monitor='loss', factor=.33, patience=10, verbose=0, mode='auto', cooldown=10)
    callback_tb = keras.callbacks.TensorBoard()

    model.fit(training_generator=training_gen, validation_generator=valid_gen,
              epochs=16, callbacks=[callback_tb, lower_lear])

    model.freezed_weights
    model.store_model('logs/model_{}.h5'.format(int(time.time())))

    return model


def predict_neural_net(model, ids_cat, mapping, use_pretrained_inception=False, use_finger_feature=True):
    ids = sorted(list(ids_cat.keys()))
    pred_ids = ids[-500:]
    pred_gen = DataGeneratorPred(list_ids=pred_ids, path='enhanced', look_up=ids_cat, mapping=mapping,
                             inception_pre=use_pretrained_inception, finger_feature=use_finger_feature,
                             batch_size=4, prop_image=0, prop_array=0, shuffle=False, predict=True)
    preds = model.predict(pred_gen)
    pred_agg = agg_ensemble(pred_array=preds, mapping=mapping)
    _df_pred = concat_ids_and_predictions(pred_ids, pred_agg, ids_cat, mapping)
    confusion_mat = confusion_matrix(_df_pred['pattern'], _df_pred['pred_pattern'], labels=sorted(mapping.keys()))
    image_confusion_matrix(_df_pred, mapping)
    plot_confusion_matrix(confusion_mat, sorted(mapping.keys()))
    _df_pred.to_csv(path_or_buf='logs/db_pred_{}.csv'.format(int(time.time())))


if __name__ == '__main__':
    pass
