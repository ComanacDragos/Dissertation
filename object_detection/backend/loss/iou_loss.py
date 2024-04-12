import tensorflow as tf


def iou_loss(target, pred, loss_type):
    pred_left = pred[..., 0]
    pred_top = pred[..., 1]
    pred_right = pred[..., 2]
    pred_bottom = pred[..., 3]

    target_left = target[..., 0]
    target_top = target[..., 1]
    target_right = target[..., 2]
    target_bottom = target[..., 3]

    target_area = (target_left + target_right) * \
                  (target_top + target_bottom)
    pred_area = (pred_left + pred_right) * \
                (pred_top + pred_bottom)

    w_intersect = tf.minimum(pred_left, target_left) + tf.minimum(pred_right, target_right)
    g_w_intersect = tf.maximum(pred_left, target_left) + tf.maximum(
        pred_right, target_right)
    h_intersect = tf.minimum(pred_bottom, target_bottom) + tf.minimum(pred_top, target_top)
    g_h_intersect = tf.maximum(pred_bottom, target_bottom) + tf.maximum(pred_top, target_top)
    ac_uion = g_w_intersect * g_h_intersect + 1e-7
    area_intersect = w_intersect * h_intersect
    area_union = target_area + pred_area - area_intersect
    ious = (area_intersect + 1.0) / (area_union + 1.0)
    gious = ious - (ac_uion - area_union) / ac_uion
    if loss_type == 'iou':
        losses = -tf.math.log(ious)
    elif loss_type == 'linear_iou':
        losses = 1 - ious
    elif loss_type == 'giou':
        losses = 1 - gious
    else:
        raise NotImplementedError

    return losses


if __name__ == '__main__':
    a = tf.ones((5, 10, 4))
    b = tf.ones((5, 10, 4))

    print(iou_loss(a, b, loss_type='giou'))
