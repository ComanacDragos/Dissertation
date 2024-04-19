from tensorflow.keras.layers import Concatenate


class FPNNeck:
    def __init__(self, upsample_conv_generator):
        self.upsample_block_generator = upsample_conv_generator

    def __call__(self, inputs):
        x = inputs[-1]
        outputs = [x]
        for layer in reversed(inputs[:-1]):
            x = self.upsample_block_generator(filters=layer.shape[-1], strides=int(layer.shape[1] / x.shape[1]))(x)
            x = Concatenate()([x, layer])
            outputs.append(x)
        return outputs
