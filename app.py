from flask import Flask, render_template, request, redirect, url_for
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load your trained model and precomputed embeddings (once when the server starts)
model = tf.keras.models.load_model('best_model.keras')
image_embeddings = pickle.load(open("image_embeddings.pkl", "rb"))

# Preprocess and extract features for a single image


def preprocess_and_extract_features(image_path):
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

# Route to handle the homepage with image upload


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if the user uploaded a file
        if "image" not in request.files:
            return redirect(request.url)
        file = request.files["image"]
        if file.filename == "":
            return redirect(request.url)

        # Save the uploaded image
        if file:
            image_path = os.path.join("uploads", file.filename)
            file.save(image_path)

            # Process the uploaded image and find similar images
            user_feature_vector = preprocess_and_extract_features(image_path)
            top_k_similar_images = find_similar_images(
                user_feature_vector, image_embeddings, top_k=5)

            # Pass the results to the result page
            similar_image_paths = [img[0] for img in top_k_similar_images]
            return render_template("results.html", similar_images=similar_image_paths)

    return render_template("index.html")


# To run the app
if __name__ == "__main__":
    # Create the uploads folder if it doesn't exist
    if not os.path.exists("uploads"):
        os.makedirs("uploads")

    app.run(debug=True)
