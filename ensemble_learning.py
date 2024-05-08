import tensorflow as tf
from simple_resnet import ResNet50V2
import numpy as np
from hyper_parameters import get_arguments

args = get_arguments()


def create_ensemble(train_dataset, num_models, input_shape=(64, 64, 3), num_classes=6):
    models = []
    dataset_size = tf.data.experimental.cardinality(train_dataset).numpy()
    seed = 42  

    for _ in range(num_models):
        # Reshuffling the dataset
        bootstrap_sample = train_dataset.shuffle(
            dataset_size, seed=seed, reshuffle_each_iteration=True).take(dataset_size)

        # Create and compile the model
        model = ResNet50V2(input_shape=input_shape, classes=num_classes)
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Train the model on the bootstrap sample
        model.fit(bootstrap_sample, epochs=args.epochs, verbose=0)

        models.append(model)
        seed += 1  # Update seed to change shuffle order for next model
    return models
