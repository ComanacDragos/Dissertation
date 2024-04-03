from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.layers import ReLU, BatchNormalization, Dropout, Conv2D
from tensorflow.keras.regularizers import l1_l2

L1 = 2e-6
L2 = 2e-5

def activation_generator():
    # return LeakyReLU(alpha=0.1)
    return ReLU()


def dropout_generator():
    return Dropout(rate=0.2)


def batch_norm_generator():
    return BatchNormalization()


def conv_generator(kernel_size, filters, strides=1, conv_type=Conv2D):
    return lambda: conv_type(
        kernel_size=kernel_size,
        filters=filters,
        padding='same',
        strides=strides,
        kernel_initializer=HeNormal(),
        kernel_regularizer=l1_l2(l1=L1, l2=L2)
    )


def depthwise_conv_generator(kernel_size, filters, strides=1, conv_type=Conv2D):
    return lambda: conv_type(
        kernel_size=kernel_size,
        filters=filters,
        padding='same',
        strides=strides,
        depthwise_initializer=HeNormal(),
        depthwise_regularizer=l1_l2(l1=L1, l2=L2)
    )
