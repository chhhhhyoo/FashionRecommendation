import tensorflow as tf

def transfer_learning_resnet50v2(input_shape=(64, 64, 3), classes=6, trainable=False):
    """Builds a transfer learning model using ResNet50V2 with custom top layers."""
    # Load a pre-trained ResNet50V2 model
    base_model = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )

    # Freeze the base model if not trainable
    base_model.trainable = trainable

    # Create new model on top
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)  # Ensure base_model is running in inference mode here
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)  # New FC layer, you can change sizes
    outputs = tf.keras.layers.Dense(classes, activation='softmax')(x)

    # Form the new model
    model = tf.keras.Model(inputs, outputs)
    return model