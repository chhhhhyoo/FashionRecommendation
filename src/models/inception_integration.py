import tensorflow as tf
from src.models.simple_resnet import ResNetBlock


def inception_module(x, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj):
    # 1x1 convolution
    conv_1x1 = tf.keras.layers.Conv2D(
        filters_1x1, (1, 1), padding='same', activation='relu')(x)

    # 3x3 convolution
    conv_3x3 = tf.keras.layers.Conv2D(
        filters_3x3_reduce, (1, 1), padding='same', activation='relu')(x)
    conv_3x3 = tf.keras.layers.Conv2D(
        filters_3x3, (3, 3), padding='same', activation='relu')(conv_3x3)

    # 5x5 convolution
    conv_5x5 = tf.keras.layers.Conv2D(
        filters_5x5_reduce, (1, 1), padding='same', activation='relu')(x)
    conv_5x5 = tf.keras.layers.Conv2D(
        filters_5x5, (5, 5), padding='same', activation='relu')(conv_5x5)

    # Max pooling
    max_pool = tf.keras.layers.MaxPooling2D(
        (3, 3), strides=(1, 1), padding='same')(x)
    max_pool = tf.keras.layers.Conv2D(
        filters_pool_proj, (1, 1), padding='same', activation='relu')(max_pool)

    # Concatenate the outputs along the depth (channel) axis
    output = tf.keras.layers.concatenate(
        [conv_1x1, conv_3x3, conv_5x5, max_pool], axis=-1)

    return output


def ResNetWithInception(input_shape=(64, 64, 3), classes=6):
    inputs = tf.keras.Input(shape=input_shape)

    # Initial Conv + MaxPool (from original ResNet)
    x = tf.keras.layers.Conv2D(64, (7, 7), strides=(
        2, 2), padding='same', activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # GoogLeNet-style Inception module
    x = inception_module(
        x,
        filters_1x1=64,
        filters_3x3_reduce=96, filters_3x3=128,
        filters_5x5_reduce=16, filters_5x5=32,
        filters_pool_proj=32
    )

    # Residual Blocks with Dropout Regularization (WideResNet-style)
    x = ResNetBlock(x, 128, stride=2, name='conv3_block1', dropout_rate=0.3)
    x = ResNetBlock(x, 128, conv_shortcut=False, name='conv3_block2', dropout_rate=0.3)

    # Global average pooling + Fully-connected layers
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, x)
    return model
