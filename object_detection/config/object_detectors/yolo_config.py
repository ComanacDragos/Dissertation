from config.common.resnet50_backbone_config import Resnet50BackboneConfig
from backend.model.generic_model import GenericModel
from backend.utils import logger

class YOLOConfig:

    @staticmethod
    def build(input_shape):
        inputs = [Input(input_shape), Input(input_shape)]
        x = ContrastiveModel(
            similarity_function=cosine_similarity,
            backbone=CustomResnetBackboneConfig.build(Conv1D),
            head=ContrastiveHead()
        )(inputs)
        model = Model(inputs=inputs, outputs=x, name='DuplicateCodeClassifier')
        model.summary(print_fn=logger)
        return model