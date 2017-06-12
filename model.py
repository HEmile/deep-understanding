import tensorflow as tf
import tensorflow.contrib.layers as layers

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

def model(features, labels, mode):
    input_layer = tf.reshape(features, [-1, 23, 1])
    layer_1 = layers.fully_connected(inputs=input_layer,
                                     num_outputs=100)
    output_layer = layers.fully_connected(inputs=layer_1,
                                          num_outputs=1,
                                          activation_fn=None)
    if mode != learn.ModeKeys.INFER:
        cross_entropy = tf.losses.sigmoid_cross_entropy(
            multi_class_labels=tf.reshape(labels, [-1]),
            logits=output_layer
        )
    if mode == learn.ModeKeys.TRAIN:
        optimizer = tf.contrib.layers.optimize_loss(
            loss=cross_entropy,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=0.001,
            optimizer="SGD")
    activated = tf.sigmoid(output_layer)
    predictions = {
         "classes": tf.round(activated),
         "probabilities": activated
    }
    return model_fn_lib.ModelFnOps(
      mode=mode, predictions=predictions, loss=cross_entropy, train_op=optimizer)