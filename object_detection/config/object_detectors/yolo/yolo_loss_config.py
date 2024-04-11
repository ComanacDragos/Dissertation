from backend.loss.yolo_loss import YOLOLoss


class YOLOLossConfig:
    L_COORD = 100.
    L_NOOBJ = 1.
    L_CLASS = 1.
    L_OBJ = 1.
    IOU_THRESHOLD = 0.6
    ENABLE_LOGS = False

    @staticmethod
    def build(anchors, no_classes, grid_size, max_width, max_height):
        return YOLOLoss(
            anchors, no_classes, grid_size,
            l_coord=YOLOLossConfig.L_COORD,
            l_noobj=YOLOLossConfig.L_NOOBJ,
            l_class=YOLOLossConfig.L_CLASS,
            l_obj=YOLOLossConfig.L_OBJ,
            iou_threshold=YOLOLossConfig.IOU_THRESHOLD,
            enable_logs=YOLOLossConfig.ENABLE_LOGS,
            max_width=max_width,
            max_height=max_height
        )
