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

- **`fashion_input.py`**: Handles image preprocessing, including resizing, normalization, and bounding box localization.
- **`preprocessing.py`**: Processes raw datasets for model training and evaluation, including feature extraction, normalization, and train-test split.
- **`simple_resnet.py`**: Defines the ResNet50V2 architecture, which is used as the base model for feature extraction and classification.
- **`inception_integration.py`**: Implements an Inception module integrated with ResNet to enhance feature representation and improve classification performance.
- **`train_n_test.py`**: Handles model training and validation. It loads datasets, applies augmentations, and configures training callbacks such as early stopping and learning rate adjustments.
- **`ensemble_learning.py`**: Implements an ensemble learning strategy by training multiple models on bootstrapped datasets, improving robustness and reducing overfitting.
- **`transfer_learning.py`**: Leverages pre-trained ResNet models and fine-tunes them on the dataset, reducing training time and improving accuracy.
- **`hyper_parameters.py`**: Manages command-line arguments for tuning hyperparameters such as learning rate, batch size, and the number of training epochs.
- **`grad_cam.py`**: Generates Grad-CAM visualizations to interpret model decisions by highlighting important image regions that influence classification.
- **`output_recommendation.py`**: Loads a trained model, takes an input image, predicts the clothing category, and retrieves the most relevant fashion recommendations based on the model's output.

## Future Improvements

- **Real-time Data Integration**: Connect to fashion e-commerce APIs for dynamic recommendations.

- **Scalability Enhancements**: Optimize model for large-scale datasets.

- **Cloud Deployment**: Deploy via FastAPI with a front-end web interface.

- **Personalization**: Improve recommendations using user browsing history and feedback mechanisms.
