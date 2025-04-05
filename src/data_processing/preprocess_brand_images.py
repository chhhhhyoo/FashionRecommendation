import tensorflow as tf
import os
import pickle
from PIL import Image
import numpy as np

# Preprocess and extract features for a single image


def preprocess_and_extract_features(model, image_path):
    image = Image.open(image_path)
    image = image.resize((64, 64))  # Resize to match model input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    feature_vector = model.predict(image)  # Extract feature embeddings
    return feature_vector

# Preprocess and extract features for all brand images and store embeddings


def preprocess_image_pool(model, image_directory):
    image_embeddings = {}
    for filename in os.listdir(image_directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(image_directory, filename)
            feature_vector = preprocess_and_extract_features(model, image_path)
            image_embeddings[image_path] = feature_vector
    return image_embeddings

# Save embeddings to a file


def save_embeddings(image_embeddings, filepath="image_embeddings.pkl"):
    with open(filepath, "wb") as f:
        pickle.dump(image_embeddings, f)


if __name__ == "__main__":
    # Load your trained model (ResNet, ensemble, etc.)
    model = tf.keras.models.load_model(
        'best_model.keras')  # Use your existing model

    # Preprocess the brand images directory
    # Specify your directory with brand images
    image_directory = "path_to_your_brand_images"
    image_embeddings = preprocess_image_pool(model, image_directory)

    # Save the image embeddings to a file
    save_embeddings(image_embeddings, "image_embeddings.pkl")
