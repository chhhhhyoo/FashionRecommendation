# Fashion Recommendation System

## Overview

This project leverages deep learning techniques to recommend clothing items based on user preferences. By processing images through a modified ResNet model, the system predicts categories and provides relevant fashion recommendations. The goal is to enhance the shopping experience by bridging the gap between fashion preferences and available inventory, improving user engagement in digital fashion platforms.

## Features

- **Image Preprocessing**: Standardizes input images to ensure consistency in model inputs.
- **Fashion Categorization**: Fine-tuned ResNet50V2 model classifies clothing items into predefined categories.
- **Recommendation Generation**: Provides fashion recommendations based on the predicted category from user inputs.
- **Localization Support**: Utilizes bounding boxes to enhance the model's focus on specific image regions.
- **Ensemble Learning**: Implements an ensemble of ResNet models to improve prediction robustness.
- **Content-Based Recommendation**: Uses cosine similarity on feature embeddings to find similar images.
- **Grad-CAM Visualization**: Explains model predictions using activation heatmaps.

## Installation

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- OpenCV
- Pandas
- NumPy

### Setup

Clone the repository:

```bash
git clone https://github.com/chhhhhyoo/FashionRecommendation.git
cd FashionRecommendation
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

[Prepare the Dataset]
To train the model, ensure you have the appropriate dataset structured according to the expectations in the data/ directory.
You can download the csv file from :

<https://github.com/wenxinxu/deep-shopping/blob/master/vali_modified.csv>

You could either use the first block of **split_dataset.ipynb** or **split_dataset.py** to split the dataset as intended.

You will also need to download the img file where you can find it from :

<https://drive.google.com/drive/folders/0B7EVK8r0v71pekpRNUlMS3Z5cUk?resourcekey=0-GHiFnJuDTvzzGuTj6lE6og>

Each code for simple resnet implementation / emsemble learning / transfer learning can be commented or not in order to apply it in training.

[Train the Model]
Then you are ready to initiate training by running:

```bash
python train_n_test.py --batch_size [number] --epochs [number]
```

### Make Recommendation

Save the query image in the working directory and update its path in **output_recommendation.py**.

To predict categories and receive recommendations, use the following command:

```bash
python output_recommendation.py
```

## Model Components

- **ensemble_learning.py**: Manages the creation and training of ensemble models. This script is responsible for bootstrapping the training dataset to train multiple instances of a model, each on slightly different data. This improves the robustness and accuracy of predictions by combining the strengths of multiple models.

- **fashion_input.py**: Handles image loading, preprocessing, and dataset preparation. It processes images by resizing and normalizing them to ensure they are in a suitable format for model input. It also includes functionality to handle localization by cropping images based on specified bounding box coordinates.

- **grad_cam.py**: Provides functionality to generate Grad-CAM visualizations for understanding model decisions. This is useful for debugging and improving model interpretability, allowing users to see which parts of an image are most influential in model predictions.

- **hyper_parameters.py**: Manages command-line arguments for hyperparameter tuning. This script uses argparse to handle various command-line options to configure model training, such as learning rate, number of epochs, and paths to training data.

- **output_recommendation.py**: Integrates all components to predict categories and generate fashion recommendations. It utilizes the trained model to classify input images and then suggests fashion items based on these predictions. Additionally, it uses Grad-CAM to provide visual feedback on what the model is focusing on in the images.

- **train_n_test.py**: Manages the overall training and testing processes. It orchestrates the training sessions, handles model evaluation, and applies transfer learning techniques to improve model performance. It also manages various callbacks to enhance training efficiency, such as early stopping and learning rate reductions.

- **transfer_learning.py**: Implements transfer learning techniques using pretrained models. This script enhances the capability of the fashion recommendation system by leveraging knowledge from large, previously trained models to improve prediction accuracy and reduce training time.
