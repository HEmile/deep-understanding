import tensorflow as tf
from tensorflow.contrib import learn
import model, dataRetriever
import numpy as np

classifier = learn.Estimator(
    model_fn=model.model, model_dir="/tmp/ann_model2")
tf.initialize_all_variables()
tensors_to_log = {"probabilities": "sigmoid_tensor"}
logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=50)

x, y = dataRetriever.retrieveData()
x = np.array(x, dtype=np.float32)
y = np.array(y, dtype=np.float32)
xtrain, ytrain = x[:25000], y[:25000]
xtest, ytest = x[25000:], y[25000:]

metrics = {"accuracy":
              learn.MetricSpec(
                  metric_fn=tf.metrics.accuracy, prediction_key="classes"),}
eval_results = classifier.evaluate(
      x=xtest, y=ytest, metrics=metrics)
print(eval_results)
for i in range(10):
    classifier.fit(
        x=xtrain,
        y=ytrain,
        batch_size=100,
        steps=25000,
        monitors=[logging_hook]
    )
    eval_results = classifier.evaluate(
          x=xtest, y=ytest, metrics=metrics)
    print(eval_results)