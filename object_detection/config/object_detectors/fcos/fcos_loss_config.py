from backend.loss.fcos_loss import FCOSLoss
from backend.loss.focal_loss import categorical_focal_crossentropy
from backend.loss.iou_loss import iou_loss


class FCOSLossConfig:
    REG_WEIGHT = 1.
    CLASS_WEIGHT = 1.
    CENTERNESS_WEIGHT = 1.
    IOU_LOSS_TYPE = 'iou'

    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.

    @staticmethod
    def build(image_size, strides_weights):
        def iou_loss_wrapper(target, pred):
            return iou_loss(target, pred, FCOSLossConfig.IOU_LOSS_TYPE)

        def focal_loss_wrapper(target, pred):
            return categorical_focal_crossentropy(
                target, pred,
                alpha=FCOSLossConfig.FOCAL_ALPHA,
                gamma=FCOSLossConfig.FOCAL_GAMMA
            )

        return FCOSLoss(
            image_size=image_size,
            strides_weights=strides_weights,
            reg_weight=FCOSLossConfig.REG_WEIGHT,
            class_weight=FCOSLossConfig.CLASS_WEIGHT,
            centerness_weight=FCOSLossConfig.CENTERNESS_WEIGHT,
            class_loss=focal_loss_wrapper,
            reg_loss=iou_loss_wrapper,
        )
