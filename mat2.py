import tensorflow as tf
import numpy as np
from tensorflow.python.client import device_lib
import time

row = 1 
col = 8000
size = 8000
np.random.seed(0)
x_train = np.random.rand(row, col)
y_train = x_train * 2 

X = tf.placeholder("float",[None,col])
with tf.name_scope("fc1"):
    w1 = tf.Variable(tf.random_normal([col,size]), name="jun_1_weight", dtype="float")
    layer1 = tf.matmul(X, w1, name="jun_1_matmul")

with tf.name_scope("fc6"):
    w6 = tf.Variable(tf.random_normal([size,col]), name="jun_6_weight", dtype="float")
    layer6 = tf.matmul(X, w6, name="jun_6_matmul")

session_config = tf.ConfigProto(
    inter_op_parallelism_threads=10,
    intra_op_parallelism_threads=1,
    allow_soft_placement=True)

sess = tf.Session( config=session_config )
v = sess.run(tf.global_variables_initializer())
print("###################### START #####################")

sess.run( (layer1, layer6), feed_dict={X:x_train} )


