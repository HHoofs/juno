import itertools

from PIL import Image
import os
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def image_to_array(path, id, mode='L'):
    if path is None:
        path = ""
    with Image.open(os.path.join(path, id + '.png')) as x_img:
        x_img = x_img.convert(mode=mode)
        return x_img


def image_confusion_matrix(df, mapping):
    plt.figure(figsize=(6.45, 6.45), dpi=300)
    count = 1
    classes = sorted(list(mapping.keys()))
    for i, pattern_x in enumerate(classes):
        for j, pattern_y in enumerate(classes):
            _id = df[df.pattern == pattern_x][pattern_y].idxmax()
            plt.subplot(len(mapping), len(mapping), count)
            plt.imshow(image_to_array(path=None, id=_id), cmap=plt.cm.gray, interpolation='none')
            if i == 0:
                plt.title(pattern_y)
            if j == 0:
                plt.ylabel(pattern_x)
            plt.xticks([])
            plt.yticks([])
            plt.box(on=None)
            count += 1
    plt.savefig('figs/plot_confusion_matrix_images.png')


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure(figsize=(6.45, 4.50), dpi=300)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('figs/plot_confusion_matrix_numbers.png')
