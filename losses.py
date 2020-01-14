import tensorflow as tf
import numpy as np
from model import real2complex


def mse(y_true, y_pred):
    y_pred_complex = real2complex(y_pred)
    y_true_complex = real2complex(y_true)
    diff = tf.square(tf.abs(y_pred_complex - y_true_complex))
    loss = tf.reduce_mean(tf.reduce_mean(diff,axis=[1,2,3]))

    return loss

def mae(y_true, y_pred):
    y_pred_complex = real2complex(y_pred)
    y_true_complex = real2complex(y_true)
    diff = tf.abs(y_pred_complex - y_true_complex)
    loss = tf.reduce_mean(tf.reduce_mean(diff,axis=[1,2,3]))

    return loss

