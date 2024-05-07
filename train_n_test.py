import tensorflow as tf
from fashion_input import prepare_df, load_data_numpy
from simple_resnet import ResNet50V2
import numpy as np
from hyper_parameters import get_arguments
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau, ModelCheckpoint
import os
import datetime
# from transfer_learning import transfer_learning_model
from ensemble_learning import create_ensemble
from grad_cam import grad_cam
import pickle


args = get_arguments()

TRAIN_DIR = 'logs_' + args.version + '/'
TRAIN_LOG_PATH = args.version + '_error.csv'

# Assuming you have a function to generate your dataset


def get_dataset(df, batch_size):
    """Assumes `load_data_numpy` returns suitable numpy arrays for x and y."""
    images, labels, _ = load_data_numpy(df)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
    return dataset


def train():
    train_df = prepare_df(args.train_path, usecols=[
        'image_path', 'category', 'x1', 'y1', 'x2', 'y2'])
    vali_df = prepare_df(args.vali_path, usecols=[
        'image_path', 'category', 'x1', 'y1', 'x2', 'y2'])

    train_dataset = get_dataset(train_df, args.batch_size)
    val_dataset = get_dataset(vali_df, args.batch_size)

    # model = ResNet50V2(input_shape=(64, 64, 3), classes=6)
    # model.compile(optimizer='adam',
    #               loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    num_models = 3  # Adjust as needed
    ensemble_models = create_ensemble(num_models, input_shape=(
        64, 64, 3), num_classes=6, train_dataset=train_dataset)

    # model.fit(train_dataset, epochs=args.epochs, validation_data=val_dataset)
    log_dir = os.path.join(
        "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # # Early Stopping callback based on validation loss
    # early_stopping_loss = EarlyStopping(
    #     monitor='val_loss', min_delta=0.01, patience=3, verbose=1, mode='min')

    early_stopping_loss = EarlyStopping(
        monitor='val_loss',  # Monitor validation loss
        min_delta=0.0001,         # Adjust minimum change threshold
        patience=10,              # Increase patience
        verbose=1,
        mode='min'               # Monitor for loss improvement
    )

    early_stopping_acc = EarlyStopping(
        monitor='val_accuracy',  # Monitor validation accuracy
        min_delta=0.0001,         # Adjust minimum change threshold
        patience=10,              # Increase patience
        verbose=1,
        mode='max'               # Monitor for accuracy improvement
    )

    # # Early Stopping callback based on validation accuracy
    # early_stopping_acc = EarlyStopping(
    #     monitor='val_accuracy', min_delta=0.01, patience=3, verbose=1, mode='max')

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=5, min_lr=0.0001
    )

    model_checkpoint = ModelCheckpoint(
        filepath='best_model.keras', monitor='val_loss', save_best_only=True)

    callbacks_list = [
        tensorboard_callback,
        early_stopping_loss,
        early_stopping_acc,
        reduce_lr,
        model_checkpoint
    ]

    # model.fit(train_dataset, epochs=args.epochs,
    #           validation_data=val_dataset, verbose=1, callbacks=callbacks_list)

    for model in ensemble_models:
        model.fit(train_dataset, epochs=args.epochs,
                  validation_data=val_dataset, verbose=1, callbacks=callbacks_list)

    # Evaluate the ensemble on the validation set
    ensemble_accuracies = []
    for model in ensemble_models:
        _, accuracy = model.evaluate(val_dataset, verbose=0)
        ensemble_accuracies.append(accuracy)

    # Calculate the ensemble accuracy as the mean of individual model accuracies
    ensemble_accuracy = sum(ensemble_accuracies) / len(ensemble_accuracies)
    print("Ensemble Accuracy:", ensemble_accuracy)

    with open('ensemble_models.pkl', 'wb') as f:
        pickle.dump(ensemble_models, f)

    # Save the model
    TRAIN_DIR = 'logs_' + args.version + '/'
    if not os.path.exists(TRAIN_DIR):
        os.makedirs(TRAIN_DIR)
    model.save(os.path.join(TRAIN_DIR, 'final_model.keras'))


if __name__ == "__main__":
    train()

# if __name__ == "__main__":
#     # Load and preprocess the dataset
#     train_dataset = get_dataset(train_df, args.batch_size)
#     val_dataset = get_dataset(vali_df, args.batch_size)

#     # Create an ensemble of models
#     num_models = 3  # Adjust as needed
#     ensemble_models = create_ensemble(train_dataset, num_models, input_shape=(64, 64, 3), num_classes=6)

#     # Train the ensemble models
#     for model in ensemble_models:
#         model.fit(train_dataset, epochs=args.epochs,
#                   validation_data=val_dataset, verbose=1, callbacks=callbacks_list)

#     # Evaluate the ensemble on the validation set
#     ensemble_accuracies = []
#     for model in ensemble_models:
#         _, accuracy = model.evaluate(val_dataset, verbose=0)
#         ensemble_accuracies.append(accuracy)

#     # Calculate the ensemble accuracy as the mean of individual model accuracies
#     ensemble_accuracy = sum(ensemble_accuracies) / len(ensemble_accuracies)
#     print("Ensemble Accuracy:", ensemble_accuracy)
