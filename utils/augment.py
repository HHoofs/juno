import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
from skimage.exposure import equalize_adapthist
from skimage.transform import PiecewiseAffineTransform, warp, swirl, resize
from skimage.util import random_noise


def affine_transform_image(image, prop=.5):
    if prop:
        if coin_flip(prop):
            factor1 = np.random.uniform(1., 1.5)
            factor2 = np.random.uniform(1., 3.)
        else:
            return image
    else:
        factor1 = 1.5
        factor2 = 3.

    rows, cols = image.shape[0], image.shape[1]

    src_cols = np.linspace(0, cols, 10)
    src_rows = np.linspace(0, rows, 10)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]

    # add sinusoidal oscillation to row coordinates
    dst_rows = src[:, 1] - np.sin(np.linspace(0, factor2 * np.pi, src.shape[0])) * 50
    dst_cols = src[:, 0]
    dst_rows *= factor1
    dst_rows -= factor1 * 50
    dst = np.vstack([dst_cols, dst_rows]).T

    tform = PiecewiseAffineTransform()
    tform.estimate(src, dst)

    out_rows = int(image.shape[0] - factor1 * 50)
    out_cols = cols
    _image = warp(image, tform, output_shape=(out_rows, out_cols), mode='reflect')
    _image = resize(_image, output_shape=(rows, cols))
    return zoom_image(_image, False, True)


def salt_pepper_image(image, prop=.5):
    if prop:
        if coin_flip(prop):
            s_vs_p = np.random.uniform(.2, .8)
            amount = np.random.uniform(.0, .05)
        else:
            return image
    else:
        s_vs_p = .5
        amount = .05

    return random_noise(image, mode='s&p', amount=amount, salt_vs_pepper=s_vs_p)


def swirl_image(image, prop=.5):
    if prop:
        if coin_flip(prop):
            strength = np.random.uniform(0, 3)
            radius = np.random.uniform(0, 250)
        else:
            return image
    else:
        strength = 3
        radius = 250
    return swirl(image, strength=strength, radius=radius)


def flip_image(image, prop=.5):
    if prop:
        if coin_flip(prop):
            return np.fliplr(image), True
        else:
            return image, False
    else:
        return np.fliplr(image), True


def equalize_adapthist_image(image, prop=.5):
    if prop:
        if coin_flip(prop):
            clip_limit = np.random.uniform(0, .05)
        else:
            return image
    else:
        clip_limit = .05

    return equalize_adapthist(image, clip_limit=clip_limit)


def random_noise_image(image, prop=.5):
    if prop:
        if coin_flip(prop):
            variance = np.random.uniform(0, .025)
        else:
            return image
    else:
        variance = .025

    return random_noise(image, mode='gaussian', var=variance)


def speckle_image(image, prop=.5):
    if prop:
        if coin_flip(prop):
            variance = np.random.uniform(0, .05)
        else:
            return image
    else:
        variance = .05

    return random_noise(image, mode='speckle', var=variance)


def zoom_image(image, prop=.5, center=False):
    _size = image.shape
    if prop:
        if coin_flip(prop):
            zoom_factor = np.random.uniform(1, .8)
        else:
            return image
    else:
        zoom_factor = .8

    zoom_size = np.floor(_size[0] * zoom_factor)
    if center:
        x_co = y_co = (_size[0] - zoom_size) / 2
    else:
        x_co, y_co = np.random.randint(0, _size[0] - zoom_size, 2)
    zoom_coord = (int(x_co), int(y_co), int(x_co + zoom_size), int(y_co + zoom_size))

    array = image[zoom_coord[0]:zoom_coord[2], zoom_coord[1]:zoom_coord[3]]
    return resize(array, output_shape=_size)


def image_perspective_transform(image, prop=.5):
    if prop:
        if coin_flip(prop):
            m = np.random.uniform(-.5, .5)
        else:
            return image
    else:
        m = -.5

    _img_size = image.size
    width, height = _img_size
    xshift = abs(m) * width
    new_width = width + int(round(xshift))
    _img = image.transform((new_width, height), Image.AFFINE,
                           (1, m, -xshift if m > 0 else 0, 0, 1, 0), Image.BICUBIC)
    _image = _img.resize((int(width*1.8), int(height*1.8)))
    return image_crop_center(_image, _img_size)


def image_emboss(image, prop=.5):
    if prop:
        if coin_flip(prop):
            return image.filter(ImageFilter.EMBOSS)
        else:
            return image
    else:
        return image.filter(ImageFilter.EMBOSS)


def image_sharpen(image, prop=.5):
    if prop:
        if coin_flip(prop):
            return image.filter(ImageFilter.SHARPEN)
        else:
            return image
    else:
        return image.filter(ImageFilter.SHARPEN)


def image_blur(image, prop=.5):
    if prop:
        if coin_flip(prop):
            return image.filter(ImageFilter.BLUR)
        else:
            return image
    else:
        return image.filter(ImageFilter.BLUR)


def image_edge_enhance(image, prop):
    if prop:
        if coin_flip(prop):
            return image.filter(ImageFilter.EDGE_ENHANCE_MORE)
        else:
            return image
    else:
        return image.filter(ImageFilter.EDGE_ENHANCE_MORE)


def coin_flip(true_prop=.5):
    return np.random.choice([True, False], p=[true_prop, 1-true_prop])


def normalize_image(image):
    max = image.max()
    image *= 255.0 / max
    return image


def crop_center_image(img, height, width):
    h, w, c = img.shape
    dy = (h-height)//2
    dx = (w-width)//2
    y1 = dy
    y2 = y1 + height
    x1 = dx
    x2 = x1 + width
    img = img[y1:y2, x1:x2, :]
    return img


def image_crop_center(image, shape):
    width, height = image.size
    target_width, target_height = shape

    left = (width - target_width)/2
    top = (height - target_height)/2
    right = (width + target_width)/2
    bottom = (height + target_height)/2

    return image.crop((left, top, right, bottom))


def to_rgb(im):
    # I think this will be slow
    w, h = im.shape
    ret = np.zeros((w, h, 3))
    ret[:, :, 0] = im
    ret[:, :, 1] = im
    ret[:, :, 2] = im
    return ret


def image_augment_array(image, prop_image=.5, prop_array=.5):
    """

    :param image:
    :param prop_image:
    :return:
    """
    # Select emboss or edge and apply in prop of the cases
    filter_source = random.choice([image_emboss, image_edge_enhance, image_perspective_transform])
    image = filter_source(image, prop_image)

    filter_sharpen_blur = random.choice([image_sharpen, image_blur])
    image = filter_sharpen_blur(image, prop_image)

    image = np.array(image)

    image, flipped = flip_image(image, prop_array)

    if filter_source is not image_perspective_transform:
        image = zoom_image(image, prop_array)

    image = equalize_adapthist_image(image, prop_array)

    filter_noise = random.choice([salt_pepper_image, speckle_image, random_noise_image])
    image = filter_noise(image, prop_array)

    return image, flipped


def image_light_augment_array(image, prop_image=.5, prop_array=.5, flipping=True):
    """

    :param image:
    :param prop_image:
    :return:
    """
    # Select emboss or edge and apply in prop of the cases
    filter_source = random.choice([image_perspective_transform])
    image = filter_source(image, prop_image)

    filter_sharpen_blur = random.choice([image_sharpen, image_blur])
    image = filter_sharpen_blur(image, prop_image)

    image = np.array(image)

    if flipping:
        image, flipped = flip_image(image, prop_array)
    else:
        flipped = False

    image = zoom_image(image, prop_array)

    image = equalize_adapthist_image(image, prop_array)

    # filter_noise = random.choice([salt_pepper_image])
    # image = filter_noise(image, prop_array)

    return image, flipped


if __name__ == '__main__':
    with Image.open('sd04/png_txt/figs_0/f0001_01.png') as x_img:
        x_img_array = np.array(x_img)

        plt.figure(figsize=(6.45,4.50), dpi=300)
        plt.subplot(3,5,1)
        plt.imshow(x_img, cmap=plt.cm.gray, interpolation='none')
        plt.axis('off')
        plt.title('Orginal')

        plt.subplot(3,5,2)
        plt.imshow(affine_transform_image(x_img_array, False), cmap=plt.cm.gray, interpolation='none')
        plt.axis('off')
        plt.title('Affine')

        plt.subplot(3,5,3)
        plt.imshow(salt_pepper_image(x_img_array, False), cmap=plt.cm.gray, interpolation='none')
        plt.axis('off')
        plt.title('Salt and pepper')

        plt.subplot(3,5,4)
        plt.imshow(swirl_image(x_img_array, False), cmap=plt.cm.gray, interpolation='none')
        plt.axis('off')
        plt.title('Swirl')

        plt.subplot(3,5,5)
        _flip_img, _ = flip_image(x_img_array, False)
        plt.imshow(_flip_img, cmap=plt.cm.gray, interpolation='none')
        plt.axis('off')
        plt.title('Flip')

        plt.subplot(3,5,6)
        plt.imshow(equalize_adapthist_image(x_img_array, False), cmap=plt.cm.gray, interpolation='none')
        plt.axis('off')
        plt.title('Equalize')

        plt.subplot(3,5,7)
        plt.imshow(random_noise_image(x_img_array, False), cmap=plt.cm.gray, interpolation='none')
        plt.axis('off')
        plt.title('Random noise')

        plt.subplot(3,5,8)
        plt.imshow(speckle_image(x_img_array, False), cmap=plt.cm.gray, interpolation='none')
        plt.axis('off')
        plt.title('Speckle')

        plt.subplot(3,5,9)
        plt.imshow(zoom_image(x_img_array, False), cmap=plt.cm.gray, interpolation='none')
        plt.axis('off')
        plt.title('Zoom')

        plt.subplot(3,5,10)
        plt.imshow(image_perspective_transform(x_img, False), cmap=plt.cm.gray, interpolation='none')
        plt.axis('off')
        plt.title('Perspective')

        plt.subplot(3,5,11)
        plt.imshow(image_emboss(x_img, False), cmap=plt.cm.gray, interpolation='none')
        plt.axis('off')
        plt.title('Emboss')

        plt.subplot(3,5,12)
        plt.imshow(image_sharpen(x_img, False), cmap=plt.cm.gray, interpolation='none')
        plt.axis('off')
        plt.title('Sharpen')

        plt.subplot(3,5,13)
        plt.imshow(image_blur(x_img, False), cmap=plt.cm.gray, interpolation='none')
        plt.axis('off')
        plt.title('Blur')

        plt.subplot(3,5,14)
        plt.imshow(image_edge_enhance(x_img, False), cmap=plt.cm.gray, interpolation='none')
        plt.axis('off')
        plt.title('Edge')

        plt.subplot(3,5,15)
        _aug_img, _ = image_augment_array(x_img, 1, 1)
        plt.imshow(_aug_img, cmap=plt.cm.gray, interpolation='none')
        plt.axis('off')
        plt.title('Sample')

        plt.savefig('figs/plot_augmentations.png')
