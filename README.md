# Fashion Recommendation System

## Overview

This project utilizes a deep learning approach to recommend clothing items based on user preferences. By processing images through a modified ResNet model, the system predicts categories and provides relevant fashion recommendations. This solution will bridge the gap between fashion preferences and available inventory, enhancing user experience in digital fashion platforms.

## Features

- **Image Preprocessing**: Standardizes input images to ensure consistency in model inputs.
- **Fashion Categorization**: Utilizes a fine-tuned ResNet50V2 model to classify clothing items into predefined categories.
- **Recommendation Generation**: Offers fashion recommendations based on the predicted category from user inputs.
- **Localization Support**: Includes functionality(bounding box) to handle image localization tasks, enhancing the model's focus on specific image regions.
- **Ensemble Learning**: Implements an ensemble of ResNet models to improve prediction robustness.

## Installation

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- OpenCV
- Pandas
- NumPy

## Usage

### Training the Model

To train the model, ensure you have the appropriate dataset structured according to the expectations in the data/ directory.
You can downloaad the csv file from :

<https://github.com/wenxinxu/deep-shopping/blob/master/vali_modified.csv>

You could either use the first block of split_dataset.ipynb or split_dataset.py to split the dataset as intended.

You will also need to download the img file where you can find it from :

<https://drive.google.com/drive/folders/0B7EVK8r0v71pekpRNUlMS3Z5cUk?resourcekey=0-GHiFnJuDTvzzGuTj6lE6og>

Then you are ready to initiate training by running:

```bash
python train_n_test.py --batch_size [number] --epochs [number]

### Make Recommendation

To predict categories and receive recommendations, use the following command:

```bash
python output_recommendation.py
