from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.layers import LeakyReLU, BatchNormalization, Dropout, Conv2D
from tensorflow.keras.regularizers import l1_l2


def activation_generator():
    return LeakyReLU(alpha=0.1)


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
        kernel_regularizer=l1_l2(l1=1e-6, l2=1e-5)
    )
