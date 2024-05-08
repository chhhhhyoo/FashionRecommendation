import tensorflow as tf
import numpy as np


def grad_cam(model, image_path, layer_name='conv5_block3_out'):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(64, 64))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess the image
    img_array = img_array / 255.0

    # Compute gradient of the predicted category with respect to the given layer's output
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = model(img_array)
        predicted_class = tf.argmax(preds[0])
        class_output = preds[:, predicted_class]
    grads = tape.gradient(class_output, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Compute the heatmap
    heatmap = tf.reduce_mean(last_conv_layer_output, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap)

    heatmap = heatmap[0]

    # Resize heatmap to the size of the original image
    heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))

    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay heatmap on the original image
    superimposed_img = cv2.addWeighted(img_array[0], 0.5, heatmap, 0.5, 0)

    return superimposed_img