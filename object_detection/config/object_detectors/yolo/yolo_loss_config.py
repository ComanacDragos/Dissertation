from backend.loss.yolo_loss import YOLOLoss


class YOLOLossConfig:
    L_COORD = 5.
    L_NOOBJ = 0.5
    L_CLASS = 3.
    L_OBJ = 2.
    IOU_THRESHOLD = 0.6
    ENABLE_LOGS = False

    @staticmethod
    def build(anchors, no_classes, grid_size):
        return YOLOLoss(
            anchors, no_classes, grid_size,
            l_coord=YOLOLossConfig.L_COORD,
            l_noobj=YOLOLossConfig.L_NOOBJ,
            l_class=YOLOLossConfig.L_CLASS,
            l_obj=YOLOLossConfig.L_OBJ,
            iou_threshold=YOLOLossConfig.IOU_THRESHOLD,
            enable_logs=YOLOLossConfig.ENABLE_LOGS
        )
