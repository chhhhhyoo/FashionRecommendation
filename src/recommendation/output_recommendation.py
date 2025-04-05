import tensorflow as tf
import numpy as np
from PIL import Image
import os
import cv2
import pandas as pd
from grad_cam import grad_cam
# import pickle


def preprocess_image(image_path):
    image = Image.open(image_path)
    # Resize the image to match the input size of your model
    image = image.resize((64, 64))
    image = np.array(image) / 255.0  # Normalize pixel values
    return image

# Predict category using the trained model


def predict_category(model, image_path):
    preprocessed_image = preprocess_image(image_path)
    preprocessed_image = np.expand_dims(
        preprocessed_image, axis=0)  # Add batch dimension

    # Predict category probabilities
    prediction = model.predict(preprocessed_image)

    # Get the category with the highest probability
    predicted_category_code = np.argmax(prediction)

    # Get the class activation heatmap using Grad-CAM
    grad_cam_heatmap = grad_cam(model, image_path)

    return predicted_category_code, grad_cam_heatmap

# Retrieve clothing recommendations based on predicted category


def get_recommendations(predicted_category_code, category_names, df):
    category_name = category_names.get(predicted_category_code, "Unknown")
    if category_name != "Unknown":
        # Filter DataFrame by category name
        recommended_df = df[df['category_old'] == category_name]
        if not recommended_df.empty:
            recommendations = recommended_df.sample(n=min(5, len(recommended_df)))[
                'image_path'].tolist()
            return recommendations
        else:
            print(
                f"No images found for the recommended category: {category_name}")
    else:
        print("Unknown category predicted.")

    return []


def display_large_image(window_name, image_path):
    # Create a resizable window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    path = os.path.join("..", "img", image_path)
    img = cv2.imread(image_path)
    # Resize the window to 400x400 pixels
    cv2.resizeWindow(window_name, 400, 400)
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Load the trained model
    model = tf.keras.models.load_model(
        'best_model.keras')  # Path to best model

    # Load the CSV file
    df = pd.read_csv("data/processed/vali_modified2.csv")

    # Extract category names from the 'category_old' column
    category_names = {code: name for code, name in zip(
        df['category_old'], df['category_old'])}

    downloaded_image_path = './test_input.jpg'
    # Predict category of the downloaded image
    predicted_category_code, grad_cam_heatmap = predict_category(
        model, downloaded_image_path)

    print("Predicted category code:", predicted_category_code)
    print("Category names:", category_names)

    # # Display the original image
    # print("Original Image:")
    # display_large_image('Original Image', downloaded_image_path)

    # # Display the Grad-CAM heatmap overlaid on the original image
    # print("Grad-CAM Heatmap:")
    # display_large_image('Grad-CAM Heatmap', grad_cam_heatmap)

    # Get recommendations based on the predicted category
    recommendations = get_recommendations(
        predicted_category_code, category_names, df)

    # Display the recommended clothing images
    print("Recommendations based on predicted category:")
    for recommendation in recommendations:
        display_large_image('Recommended Clothing', recommendation)

    # with open('ensemble_models.pkl', 'rb') as f:
    #     ensemble_models = pickle.load(f)

    # df = pd.read_csv('data/test_modified2.csv',
    #                  usecols=['image_path', 'category_old'])

    # category_names = {code: name for code, name in zip(
    #     df['category_old'], df['category_old'])}

    # downloaded_image_path = './test_output.jpg'

    # preprocessed_image = preprocess_image(downloaded_image_path)

    # ensemble_predictions = []
    # for model in ensemble_models:
    #     prediction = model.predict(np.expand_dims(preprocessed_image, axis=0))
    #     ensemble_predictions.append(prediction)

    # averaged_prediction = np.mean(ensemble_predictions, axis=0)

    # predicted_category_code = np.argmax(averaged_prediction)

    # recommendations = get_recommendations(
    #     predicted_category_code, category_names, df)

    # print("Recommendations based on predicted category:")
    # for recommendation in recommendations:
    #     img = cv2.imread(recommendation)
    #     cv2.imshow('Recommended Clothing', img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
