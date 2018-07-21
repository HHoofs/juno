from PIL import Image
import os
import matplotlib

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
    for i, pattern_x in enumerate(mapping.keys()):
        for j, pattern_y in enumerate(mapping.keys()):
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
    plt.savefig('figs/plot_confusion_matrix.png')