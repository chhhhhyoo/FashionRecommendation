import tensorflow as tf
import numpy as np


def grad_cam(model, image_path, layer_name='conv5_block3_out', class_index=None):
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(64, 64))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess the input image
    img_array = img_array / 255.0

    # Generate class activation heatmap
    heatmap_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        last_conv_output, preds = heatmap_model(img_array)
        if class_index is None:
            class_index = tf.argmax(preds[0])
        class_output = preds[:, class_index]

    grads = tape.gradient(class_output, last_conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_mean(last_conv_output, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    # Resize heatmap to the original image size
    heatmap = tf.image.resize(
        heatmap, (img_array.shape[1], img_array.shape[2]))
    heatmap = tf.squeeze(heatmap)

    # Apply colormap
    heatmap = tf.keras.utils.normalize(heatmap, axis=-1, order=1)
    heatmap = tf.constant(heatmap)
    heatmap = tf.expand_dims(heatmap, axis=-1)
    heatmap = tf.image.grayscale_to_rgb(heatmap)

    # Apply heatmap overlay
    heatmap = tf.image.convert_image_dtype(heatmap, dtype=tf.uint8)
    heatmap = tf.image.resize(heatmap, (64, 64))

    # Apply heatmap overlay on the original image
    img_heatmap = tf.image.convert_image_dtype(img_array[0], dtype=tf.uint8)
    img_heatmap = tf.image.resize(img_heatmap, (64, 64))
    superimposed_img = tf.addWeighted(img_heatmap, 0.6, heatmap, 0.4, 0)

    return superimposed_img
