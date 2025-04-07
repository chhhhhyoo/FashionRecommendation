import tensorflow as tf

def transfer_learning_resnet50v2(input_shape=(64, 64, 3), classes=6, fine_tune_at=100):
    """
    Builds transfer learning model using ResNet50V2 with:
    - Pretrained ImageNet weights
    - Partial fine-tuning after a specified layer
    - Modern classification head with dropout for regularization
    
    Args:
        input_shape (tuple)
        classes (int): number of output classes
        fine_tune_at (int): layer index at which to start fine-tuning

    Returns:
        tf.keras.Model: compiled Keras model ready for training
    """
    # Pre-trained ResNet50V2 model without top classifier
    base_model = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )

    # Freeze all layers before the fine-tune point
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    for layer in base_model.layers[fine_tune_at:]:
        layer.trainable = True

    # Input layer
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.resnet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)  # Inference mode for frozen layers

    # New classification head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    return model