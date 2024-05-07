import tensorflow as tf
from simple_resnet import ResNet50V2
import numpy as np
from hyper_parameters import get_arguments

args = get_arguments()


def create_ensemble(num_models, input_shape=(64, 64, 3), num_classes=6, train_dataset):
    models = []
    for _ in range(num_models):
        # Creating a bootstrap sample of the training dataset
        bootstrap_sample_indices = np.random.choice(
            len(train_dataset), size=len(train_dataset), replace=True)
        bootstrap_sample = train_dataset.take(bootstrap_sample_indices)

        # Create and compile the model
        model = ResNet50V2(input_shape=input_shape, classes=num_classes)
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Train the model on the bootstrap sample
        model.fit(bootstrap_sample, epochs=args.epochs, verbose=0)

        models.append(model)
    return models
