import cv2
import os
import tensorflow as tf
import numpy as np


def rgb2gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    return gray


def resizeimage(image, height, width):
    image = image[34:34+160, 0:160, :]
    image = cv2.resize(image, (height, width))
    gray = rgb2gray(image)

    return gray


def save_image(image, config, index):
    if not os.path.exists(config.imgpath):
        os.makedirs(config.imgpath)

    path = os.path.join(config.imgpath, str(index) + '.png')
    image = cv2.resize(image, (84 * 4, 84 * 4))

    cv2.imwrite(path, image)


def _variable_on_device(name, shape, initializer, trainable=True):
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=trainable)

    return var


def con2d(input,
          output_dim,
          kernel_size,
          stride,
          initilizer = tf.contrib.layers.xavier_initializer(),
          bias = True,
          activation_fn=tf.nn.relu,
          padding='VALID',
          name='conv2d'):

    with tf.variable_scope(name):
        input_dim = input.get_shape()[-1].value
        stride = [1, 1, stride[0], stride[1]]
        kernal_shape = [kernel_size[0], kernel_size[1], input_dim, output_dim]
        w = _variable_on_device('w', kernal_shape, initilizer)
        b = _variable_on_device('b', [output_dim], initializer=tf.constant_initializer(0.0))

        # conv operation
        conv = tf.nn.conv2d(x, w, stride, padding)

        if bias:
            output = conv + b

        if activation_fn is not None:
            out = activation_fn(output)

        return out, w, b


def squared_loss(x):

    return x * x


def huber_loss(x):
    # Huber loss
    try:
        return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
    except:
        return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)


def visualization_tool(states_batch, display=False):
    # visualization sample
    if len(states_batch.shape) == 4:
        v_sample = states_batch[0, ...]
        v_image = np.zeros((84*4, 84, 1))
        for i in range(4):
            tmp = v_sample[:, :, i:i + 1]
            v_image[84 * i:84 * (i + 1), :, :] = tmp
        v_image = v_image.astype(np.uint8)
        v_image = cv2.resize(v_image, (84 * 4, 84 * 4 * 4))
    else:
        v_sample = states_batch
        v_image = np.zeros((84 * 4, 84, 1))
        for i in range(4):
            tmp = v_sample[:, :, i:i + 1]
            v_image[84 * i:84 * (i + 1), :, :] = tmp
        v_image = v_image.astype(np.uint8)
        v_image = cv2.resize(v_image, (84 * 4, 84 * 4 * 4))

    if display:
        cv2.imshow("sample", v_image)
        cv2.waitKey(1000)

    return v_image