import tensorflow as tf
import time

print("Testing GPU training...")
with tf.device('/GPU:0'):
    a = tf.random.normal([1000, 1000])
    b = tf.random.normal([1000, 1000])
    start = time.time()
    for _ in range(1000):
        _ = tf.matmul(a, b)
    end = time.time()

print("Done. Time taken on GPU: {:.2f} sec".format(end - start))
