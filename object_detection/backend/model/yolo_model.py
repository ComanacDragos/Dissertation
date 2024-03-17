from tensorflow.keras.layers import Reshape



# add here preprocessing
class YOLOPreprocessing:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass


class YOLOPostprocessing:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass


class YOLOHead:
    def __init__(self, grid_size, no_anchors, no_classes, conv_generator):
        self.grid_size = grid_size
        self.no_anchors = no_anchors
        self.no_classes = no_classes
        self.conv_1x1 = conv_generator(3, no_anchors * (4 + 1 + no_classes))

    def __call__(self, inputs):
        x = self.conv_1x1(inputs)
        return Reshape(*self.grid_size, self.no_anchors, 4 + 1 + self.no_classes)(x)
