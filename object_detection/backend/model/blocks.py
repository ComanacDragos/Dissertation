from tensorflow.keras.layers import Add


class ConvBlock:
    def __init__(
            self,
            conv_generator,
            skip_connection_size=0,
            activation_generator=None,
            dropout_generator=None,
            batch_norm_generator=None,
            pooling_generator=None,
    ):
        self.conv_generator = conv_generator
        self.skip_connection_size = skip_connection_size
        self.activation_generator = activation_generator
        self.dropout_generator = dropout_generator
        self.batch_norm_generator = batch_norm_generator
        self.pooling_generator = pooling_generator

    def __call__(self, inputs):
        x = self.conv_generator()(inputs)

        if self.skip_connection_size:
            for _ in range(self.skip_connection_size):
                x = self.conv_generator()(x)
            x = Add()([inputs, x])

        if self.pooling_generator:
            x = self.pooling_generator()(x)

        if self.batch_norm_generator:
            x = self.batch_norm_generator()(x)

        if self.activation_generator:
            x = self.activation_generator()(x)

        if self.dropout_generator:
            x = self.dropout_generator()(x)
        return x


class InvertedResidualBlock:
    def __init__(
            self,
            expand_conv_generator,
            squeeze_conv_generator,
            depthwise_conv_generator,
            activation_generator,
            batch_norm_generator,
            add_skip_connection=True
    ):
        self.expand_conv_generator = expand_conv_generator
        self.squeeze_conv_generator = squeeze_conv_generator
        self.depthwise_conv_generator = depthwise_conv_generator
        self.activation_generator = activation_generator
        self.batch_norm_generator = batch_norm_generator
        self.add_skip_connection = add_skip_connection

    def __call__(self, inputs):
        x = self.expand_conv_generator()(inputs)
        x = self.batch_norm_generator()(x)
        x = self.activation_generator()(x)

        x = self.depthwise_conv_generator()(x)
        x = self.batch_norm_generator()(x)
        x = self.activation_generator()(x)

        x = self.squeeze_conv_generator()(x)
        x = self.batch_norm_generator()(x)
        x = self.activation_generator()(x)

        if self.add_skip_connection:
            x = Add()([x, inputs])
        return x
