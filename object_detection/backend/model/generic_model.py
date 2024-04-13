from overrides import overrides


class GenericModel:
    def __init__(self, backbone, head, neck=None):
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def __call__(self, inputs):
        x = self.backbone(inputs)
        if self.neck:
            x = self.neck(x)
        x = self.head(x)
        return x


class MultiScaleModel(GenericModel):
    @overrides
    def __call__(self, inputs):
        x = self.backbone(inputs)
        if self.neck:
            x = self.neck(x)
        outputs = {}
        for scale in x:
            scale_output = self.head(scale)
            stride = inputs.shape[1] // scale_output[0].shape[1]
            outputs[stride] = scale_output
        return outputs


class Sequential:
    def __init__(self, blocks):
        self.blocks = blocks

    def __call__(self, inputs):
        x = inputs
        for block in self.blocks:
            x = block(x)
        return x
