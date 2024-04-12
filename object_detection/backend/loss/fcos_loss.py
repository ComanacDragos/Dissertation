from collections import OrderedDict

import tensorflow as tf

from backend.loss.custom_loss import CustomLoss


class FCOSLoss(CustomLoss):
    def __init__(self, image_size, strides_weights, reg_weight, class_weight, centerness_weight, class_loss, reg_loss):
        """
        :param image_size: h w
        :param strides_weights: {stride:int -> weight: float}
        :param reg_weight: float
        :param class_weight: float
        :param centerness_weight: float
        :param class_loss: function which returns classification loss
        :param reg_loss: function which returns regression loss
        """
        super(FCOSLoss, self).__init__()
        self.image_size = image_size
        self.strides_weights = strides_weights
        self.reg_weight = reg_weight
        self.class_weight = class_weight
        self.centerness_weight = centerness_weight

        self.class_loss = class_loss
        self.reg_loss = reg_loss
        self.centerness_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

    def call(self, y_true, y_pred: tf.Tensor):
        loss_dict = OrderedDict()
        total_loss = 0.
        for stride, stride_weight in self.strides_weights.items():
            class_gt, centerness_gt, reg_gt = y_true[stride]
            class_pred, centerness_pred, reg_pred = y_pred[stride]

            gt_indicator = tf.reduce_max(class_gt, axis=-1) # TODO: use this
            num_pos = tf.reduce_sum(gt_indicator)

            # compute classification loss
            class_loss = tf.reduce_sum(self.class_loss(class_gt, class_pred))
            class_loss = (class_loss * self.class_weight) / num_pos
            loss_dict[f"class_loss_{stride}"] = class_loss

            # compute regression loss
            reg_loss = tf.reduce_sum(self.reg_loss(reg_gt, reg_pred) * gt_indicator)
            reg_loss = (reg_loss * self.reg_weight) / num_pos
            loss_dict[f"reg_loss_{stride}"] = reg_loss

            # computer centerness loss
            centerness_loss = tf.reduce_sum(
                self.centerness_loss(centerness_gt[..., None], centerness_pred[..., None]) * gt_indicator
            )

            centerness_loss = (centerness_loss * self.centerness_weight) / num_pos
            loss_dict[f"centerness_loss_{stride}"] = centerness_loss

            total_loss += (class_loss + reg_loss + centerness_loss) * stride_weight

        return total_loss, loss_dict


if __name__ == '__main__':
    fake_gt = tf.zeros((10, 20))
    fake_pred = tf.zeros((10, 20))

    print(tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(fake_gt[..., None], fake_pred[..., None]))
