import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Preprocess and extract features for a single image


def preprocess_and_extract_features(model, image_path):
    image = Image.open(image_path)
    image = image.resize((64, 64))  # Resize to match model input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    feature_vector = model.predict(image)  # Extract feature embeddings
    return feature_vector

# Find the most similar images based on cosine similarity


def find_similar_images(user_feature_vector, image_embeddings, top_k=5):
    similarity_scores = []
    for image_path, feature_vector in image_embeddings.items():
        similarity = cosine_similarity(user_feature_vector, feature_vector)
        # Store the image path and similarity score
        similarity_scores.append((image_path, similarity[0][0]))

    # Sort the similarity scores in descending order and return the top K results
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    top_k_similar_images = similarity_scores[:top_k]
    return top_k_similar_images

# Display the recommended images


def display_large_image(window_name, image_path):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    img = cv2.imread(image_path)
    cv2.resizeWindow(window_name, 400, 400)
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Load precomputed embeddings from file


def load_embeddings(filepath="image_embeddings.pkl"):
    with open(filepath, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    # Load your trained model
    model = tf.keras.models.load_model('best_model.keras')

    # Load precomputed embeddings for brand images
    image_embeddings = load_embeddings(
        "image_embeddings.pkl")  # <<< This line is added here

    # User input image path
    downloaded_image_path = './test_input.jpg'

    # Extract features from the user input image
    user_feature_vector = preprocess_and_extract_features(
        model, downloaded_image_path)

    # Find the most similar images from the brand image pool
    top_k_similar_images = find_similar_images(
        user_feature_vector, image_embeddings, top_k=5)

    # Display the top-K recommended fashion items based on similarity
    print("Top recommended fashion items based on content similarity:")
    for image_path, similarity in top_k_similar_images:
        print(f"Image: {image_path}, Similarity: {similarity:.4f}")
        display_large_image('Recommended Clothing (Similarity)', image_path)
