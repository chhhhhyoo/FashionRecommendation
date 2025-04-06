import tensorflow as tf
from src.data_processing.fashion_input import prepare_df, load_data_numpy
from src.models.simple_resnet import ResNet50V2
from src.models.inception_integration import ResNetWithInception
import numpy as np
from hyper_parameters import get_arguments
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau, ModelCheckpoint
import os
import datetime
from src.models.transfer_learning import transfer_learning_resnet50v2
from src.models.ensemble_learning import create_ensemble
from src.recommendation.grad_cam import grad_cam
import pickle


args = get_arguments()
TRAIN_DIR = 'logs_' + args.version + '/'
TRAIN_LOG_PATH = args.version + '_error.csv'


def get_dataset(df, batch_size):
    """Assumption : `load_data_numpy` returns suitable numpy arrays for x and y."""
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
    # model = transfer_learning_resnet50v2(input_shape=(64, 64, 3), classes=6)  # Use the transfer learning model
    # Use inception + resnet model
    model = ResNetWithInception(input_shape=(64, 64, 3), classes=6)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # num_models = 3  # Adjust as needed
    # ensemble_models = create_ensemble(train_dataset, num_models, input_shape=(
    #     64, 64, 3), num_classes=6)

    log_dir = os.path.join(
        "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # early_stopping_loss = EarlyStopping(
    #     monitor='val_loss',  # Monitor validation loss
    #     min_delta=0.0001,         
    #     patience=10,          
    #     verbose=1,
    #     mode='min'             
    # )

    early_stopping_acc = EarlyStopping(
        monitor='val_accuracy',  # Monitor validation accuracy
        min_delta=0.0001,         # minimum change threshold
        patience=10,             
        verbose=1,
        mode='max',               # Monitor for accuracy improvement
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=5, min_lr=0.0001
    )

    model_checkpoint = ModelCheckpoint(
        filepath='best_model.keras', monitor='val_loss', save_best_only=True)

    callbacks_list = [
        tensorboard_callback,
        early_stopping_acc,
        reduce_lr,
        model_checkpoint
    ]

    model.fit(train_dataset, epochs=args.epochs,
              validation_data=val_dataset, verbose=1, callbacks=callbacks_list)
    
    final_loss, final_accuracy = model.evaluate(val_dataset, verbose=1)
    print(f"\n Final Validation Loss: {final_loss:.4f}")
    print(f" Final Validation Accuracy: {final_accuracy:.4f}")

    # Save the model
    TRAIN_DIR = 'logs_' + args.version + '/'
    if not os.path.exists(TRAIN_DIR):
        os.makedirs(TRAIN_DIR)
    model.save(os.path.join(TRAIN_DIR, 'final_model.keras'))


if __name__ == "__main__":
    train()