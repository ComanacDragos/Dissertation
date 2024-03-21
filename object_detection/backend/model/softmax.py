from tensorflow.keras import backend as K


def softmax(x, t=1):
    """"
    Applies the softmax temperature on the input x, using the temperature t
    """
    x_stabilized = x - K.max(x, axis=-1, keepdims=True)

    return K.exp(x_stabilized / t) / K.sum(K.exp(x_stabilized / t), axis=-1, keepdims=True)
