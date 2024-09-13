import tensorflow as tf
import numpy as np
import cv2


def grad_cam(model, image_path, layer_name='conv3_block2_out'):
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(64, 64))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image

    # Convert numpy array to tensor
    img_tensor = tf.convert_to_tensor(img_array)

    # Get the specified convolutional layer
    last_conv_layer = model.get_layer(layer_name)
    last_conv_layer_model = tf.keras.models.Model(
        inputs=model.inputs, outputs=last_conv_layer.output)

    # Create a model that maps the input image to the activations of the last conv layer and predictions
    with tf.GradientTape() as tape:
        # Watch the input tensor
        tape.watch(img_tensor)

        # Get the activations of the last convolutional layer
        last_conv_layer_output = last_conv_layer_model(img_tensor)

        # Make predictions on the image
        preds = model(img_tensor)

        # Get the predicted class index
        predicted_class = tf.argmax(preds[0])

        # Compute the gradients of the predicted class with respect to the activations of the last conv layer
        with tf.GradientTape() as tape2:
            tape2.watch(last_conv_layer_output)
            preds = model(img_tensor)
            # Get the output for the predicted class
            predicted_class_output = preds[:, predicted_class]

        # Compute the gradients
        grads = tape2.gradient(predicted_class_output, last_conv_layer_output)

    # Check if gradients are computed
    if grads is None:
        raise ValueError(
            "No gradients were computed. Check the connection between the inputs and outputs.")

    # Pooling and heatmap processing steps
    # Global average pooling over gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(tf.multiply(
        pooled_grads, last_conv_layer_output), axis=-1)  # Weighted sum of activations
    heatmap = tf.maximum(heatmap, 0) / \
        tf.math.reduce_max(heatmap)  # Normalize heatmap
    heatmap = heatmap.numpy()

    # Resize heatmap to the size of the original image
    # Resize to match original image size
    heatmap = cv2.resize(heatmap[0], (img_array.shape[2], img_array.shape[1]))
    heatmap = np.uint8(255 * heatmap)  # Scale heatmap to [0, 255]
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose the heatmap on the original image
    superimposed_img = cv2.addWeighted(img_array[0], 0.6, heatmap, 0.4, 0)

    return superimposed_img

# import tensorflow as tf
# import numpy as np
# import cv2


# def grad_cam(model, image_path, layer_name='conv3_block2_out'):
#     img = tf.keras.preprocessing.image.load_img(
#         image_path, target_size=(64, 64))
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)

#     # Preprocess the image
#     img_array = img_array / 255.0

#     # Convert numpy array back to tensor
#     img_tensor = tf.convert_to_tensor(img_array)

#     # Get the specified layer
#     last_conv_layer = model.get_layer(layer_name)
#     last_conv_layer_model = tf.keras.models.Model(
#         inputs=model.inputs, outputs=last_conv_layer.output)

#     # Create a model that maps the input image to the activations of the last conv layer as well as the output predictions
#     with tf.GradientTape() as tape:
#         tape.watch(img_tensor)
#         last_conv_layer_output = last_conv_layer_model(img_tensor)
#         preds = model(img_tensor)
#         predicted_class = tf.argmax(preds[0])

#         # Ensure the predicted class tensor is watched
#         tape.watch(preds)

#     # Compute gradients of the predicted class with respect to the outputs of the last conv layer
#     grads = tape.gradient(preds, last_conv_layer_output)

#     if grads is None:
#         raise ValueError(
#             "No gradients were computed. Check the connection between the inputs and outputs.")

#     # Pooling and heatmap processing steps
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
#     heatmap = tf.reduce_mean(tf.multiply(
#         pooled_grads, last_conv_layer_output), axis=-1)
#     heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
#     heatmap = heatmap.numpy()

#     # Resize heatmap to the size of the original image
#     heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))
#     heatmap = np.uint8(255 * heatmap)
#     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

#     # Superimpose the heatmap on original image
#     superimposed_img = cv2.addWeighted(img_array[0], 0.6, heatmap, 0.4, 0)

#     return superimposed_img
