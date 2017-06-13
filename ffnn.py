import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import dataRetriever

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

def get_data():
  data, target = dataRetriever.get_data()
  data = np.array(data, dtype="float64")
  data = np.concatenate((data, np.ones((data.shape[0],1))),axis=1) # add bias term
  data = normalize(data, axis=0, norm='max')
  target = np.array(target, dtype="int64")
  num_labels = len(np.unique(target))
  all_Y = np.eye(num_labels)[target]  # convert to one hot encoding

  return train_test_split(data, all_Y, test_size=0.33, random_state=RANDOM_SEED)

def init_weights(shape):
  """ Weight initialization """
  weights = tf.random_normal(shape, stddev=0.1)
  return tf.Variable(weights)

def forwardprop(X, w_1, w_2):
  """
  Forward-propagation.
  IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
  """
  h = tf.nn.relu(tf.matmul(X, w_1))
  yhat = tf.matmul(h, w_2)
  return yhat


def main():
  train_X, test_X, train_y, test_y = get_data()

  # Layer's sizes
  x_size = train_X.shape[1]  # Number of input nodes: many features and 1 bias
  h_size = 256  # Number of hidden nodes
  y_size = train_y.shape[1]

  # Learning rate
  lr = 0.001

  # Symbols
  X = tf.placeholder("float", shape=[None, x_size])
  y = tf.placeholder("float", shape=[None, y_size]) # since this is just 0-1 classification task

  # Weight initializations
  w_1 = init_weights((x_size, h_size))
  w_2 = init_weights((h_size, y_size))

  # Forward propagation
  yhat = forwardprop(X, w_1, w_2)
  predict = tf.argmax(yhat, dimension=1)

  # Backward propagation
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
  #updates = tf.train.GradientDescentOptimizer(lr).minimize(cost)
  updates = tf.train.AdamOptimizer(lr).minimize(cost)

  # Run SGD
  sess = tf.Session()
  init = tf.global_variables_initializer()
  sess.run(init)

  for epoch in range(10):
    # Train with each example
    for i in range(len(train_X)):
      sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})

    train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                             sess.run(predict, feed_dict={X: train_X, y: train_y}))
    test_accuracy = np.mean(np.argmax(test_y, axis=1) ==
                            sess.run(predict, feed_dict={X: test_X, y: test_y}))

    print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
          % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))

  sess.close()

if __name__ == '__main__':
  main()