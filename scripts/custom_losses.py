import tensorflow as tf
from tensorflow import keras

def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)

@keras.utils.register_keras_serializable(package='Custom', name='custom_loss')
def custom_loss(y_true, y_pred):
    return mse(y_true, y_pred)
