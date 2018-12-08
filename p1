import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

dir_home = os.path.realpath(__file__)


def get_images(dir_win, dir_lin):
    filenames = os.listdir(dir_win)
    a = []
    for filename in filenames:
        path = os.path.dirname(dir_home) + dir_lin + filename
        pic = Image.open(path)
        pic = pic.resize((24, 24), Image.ANTIALIAS)
        pix = np.array(pic)
        a.append(pix)
    return a


def get_asterisk_images():
    return get_images("Train/asterisk", "\\Train\\asterisk\\")


def get_blade_images():
    return get_images("Train/blade", "\\Train\\blade\\")


def get_gun_images():
    return get_images("Train/gun", "\\Train\\gun\\")


def get_knife_images():
    return get_images("Train/knife", "\\Train\\knife\\")


def get_all_images():
    a = get_asterisk_images()
    b = get_blade_images()
    c = get_gun_images()
    d = get_knife_images()
    return a, b, c, d


def main():
    asterisk_data, blade_data, gun_data, knife_data = get_all_images()
    train_labels = [0] * 115 + [1] * 96 + [2] * 160 + [3] * 461

    asterisk_data = np.array(asterisk_data)
    blade_data = np.array(blade_data)
    gun_data = np.array(gun_data)
    knife_data = np.array(knife_data)

    train_images = np.concatenate((asterisk_data, blade_data, gun_data, knife_data), axis=0)

    class_names = ['asterisk', 'blade', 'gun', 'knife']

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(24, 24)),
        keras.layers.Dense(256, activation=tf.nn.relu),
        keras.layers.Dense(4, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=5)


if __name__ == '__main__':
    main()
