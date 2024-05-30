import tensorflow as tf

path = "outputs/kitti/fcos/arch/v5_1_scale_4_layers_128_filters_v2/train/checkpoints/model_15_0.4910851716995239_mAP.h5"

tf.keras.models.load_model(path).summary()
