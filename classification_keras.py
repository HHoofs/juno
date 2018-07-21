import time

import keras
import keras.backend as K
import numpy as np
from keras import models
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input, MaxPooling2D, Conv2D, Activation, BatchNormalization, AveragePooling2D, concatenate, \
    GlobalAveragePooling2D, Dense
from sklearn.metrics import confusion_matrix

from utils.analysis import image_confusion_matrix
from utils.set_up_db import read_csv_to_dict, concat_ids_and_predictions
from utils.generator import DataGenerator


class Neural_Net():
    def __init__(self, img_size, num_classes):
        self.img_size = img_size
        self.num_classes = num_classes

        self.neural_net = None
        self.inception_v3_layers = None
        self.custom_inception_layers = None

        self.freezed_weight_checks = None
        self.trained_weight_checks = None

    def set_net(self, relative_size):
        if K.image_data_format() == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 3

        raw = Input(shape=(*self.img_size, 1), name='raw')
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

        branch5x5 = conv2d_bn_alt(x, relative_size * 48, 1, 1)
        branch5x5 = conv2d_bn_alt(branch5x5, relative_size * 64, 5, 5)

        branch3x3dbl = conv2d_bn_alt(x, relative_size * 64, 1, 1)
        branch3x3dbl = conv2d_bn_alt(branch3x3dbl, relative_size * 96, 3, 3)
        branch3x3dbl = conv2d_bn_alt(branch3x3dbl, relative_size * 96, 3, 3)

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

        # end:
        x = GlobalAveragePooling2D()(x)

        pre = Input(shape=(299, 299, 3), name='pre')
        x_pre = InceptionV3(include_top=False, input_tensor=pre, pooling='avg')

        x_combined = concatenate([x, x_pre.output])

        x_combined = Dense(64, activation='relu')(x_combined)

        y = Dense(self.num_classes, activation='softmax')(x_combined)

        neural_net_model = models.Model(inputs=[raw, pre], outputs=y)

        self.neural_net = neural_net_model
        self.inception_v3_layers = x_pre.layers
        self.custom_inception_layers = [x for x in neural_net_model.layers if x not in self.inception_v3_layers]

    def freeze_inception_layers(self, check=True):
        for layer2freeze in self.inception_v3_layers:
            self.neural_net.get_layer(layer2freeze.name).trainable = False

        if check:
            freezed_batchnorm_conv_weights = self.get_batchnorm_conv_weights(self.inception_v3_layers)
            self.freezed_weight_checks = freezed_batchnorm_conv_weights

            trained_batchnorm_conv_weights = self.get_batchnorm_conv_weights(self.custom_inception_layers)
            self.trained_weight_checks = trained_batchnorm_conv_weights

    def check_freezed_trained_weights(self):
        if self.freezed_weight_checks is None:
            print('No weights are available to use as reference')
            return None

        checks = self.check_layers_weights_with_reference(layers=self.inception_v3_layers,
                                                          ref_weights=self.freezed_weight_checks)
        if all(checks):
            print('Freezing of InceptionV3 successful')
            output = [True]
        else:
            print('Freezing of InceptionV3 unsuccessful')
            output = [False]

        checks = self.check_layers_weights_with_reference(layers=self.custom_inception_layers,
                                                          ref_weights=self.trained_weight_checks)
        if not any(checks):
            print('Training of custom Inception successful')
            output.append(True)
        else:
            print('Training of custom Inception unsuccessful')
            output.append(False)

        return output

    def check_layers_weights_with_reference(self, layers, ref_weights):
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
                optimizer=keras.optimizers.Adadelta(), metrics=['acc']):
        self.neural_net.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def fit(self, training_generator, validation_generator=None, epochs=None, steps_per_epoch=32, callbacks=None, **kwargs):
        self.neural_net.fit_generator(generator=training_generator, validation_data=validation_generator,
                                      epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks, **kwargs)

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


def train_neural_net(ids_cat, mapping):
    ids = list(ids_cat.keys())
    training_gen = DataGenerator(list_ids=ids[:-500], path=None, look_up=ids_cat, mapping=mapping,
                                 prop_image=.25, prop_array=.50)
    valid_gen = DataGenerator(list_ids=ids[-500:], path=None, look_up=ids_cat, mapping=mapping,
                              prop_image=0, prop_array=0)

    model = Neural_Net(img_size=(512,512), num_classes=len(mapping))
    model.set_net(relative_size=1)
    model.freeze_inception_layers()
    model.compile()
    model.check_freezed_trained_weights()

    callback_tb = keras.callbacks.TensorBoard()

    model.fit(training_generator=training_gen, validation_generator=valid_gen,
              epochs=16, callbacks=[callback_tb])

    model.check_freezed_trained_weights()
    model.store_model('logs/model_{}.h5'.format(int(time.time())))

    return model

def predict_neural_net(model, ids_cat, mapping):
    ids = list(ids_cat.keys())
    pred_ids = ids[-500:]
    pred_gen = DataGenerator(list_ids=pred_ids, path=None, look_up=ids_cat, mapping=mapping,
                              prop_image=0, prop_array=0, shuffle=False, predict=True)
    preds = model.predict(pred_gen)
    _df_pred = concat_ids_and_predictions(pred_ids, preds, ids_cat, mapping)
    print(confusion_matrix(_df_pred['pattern'], _df_pred['pred_pattern'], labels=sorted(mapping.keys())))
    image_confusion_matrix(_df_pred, mapping)
    _df_pred.to_csv(path_or_buf='logs/db_pred_{}.csv'.format(int(time.time())))

if __name__ == '__main__':
    pass
