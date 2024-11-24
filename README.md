# Skin Lesion Identifier Using DenseNet-121

This project is a machine learning-based skin lesion identifier built using **DenseNet-121**, a deep convolutional neural network model. The model classifies skin lesions into 8 categories based on medical images. The dataset used for training is from **Kaggle**.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [License](#license)

## Overview

Skin cancer is one of the most common forms of cancer, and early detection is crucial for effective treatment. This project uses a pre-trained **DenseNet-121** model, fine-tuned on a Kaggle skin lesion dataset, to classify skin lesions into 8 different categories.

## Installation

Follow the steps below to set up and run the project:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/skin-lesion-identifier.git
    cd skin-lesion-identifier
    ```

2. **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # For Linux/Mac
    .\venv\Scripts\activate   # For Windows
    ```

3. **Install the required dependencies manually:**
   Since there is no `requirements.txt`, you need to install the necessary dependencies yourself. The key dependencies include:
    ```bash
    pip install torch torchvision matplotlib numpy flask
    ```

4. **Download the dataset from Kaggle** and place it in the `data/` folder.

## Dataset

The dataset used in this project is the **Skin Lesion Classification dataset** from **Kaggle**. It contains images of skin lesions labeled with 8 different lesion classes:

1. **BA-cellulitis**
2. **BA-impetigo**
3. **FU-athlete-foot**
4. **FU-nail-fungus**
5. **FU-ringworm**
6. **PA-cutaneous-larva-migrans**
7. **VI-chickenpox**
8. **VI-shingles**

You can download the dataset from [Kaggle's Skin Disease Dataset](https://www.kaggle.com/datasets/subirbiswas19/skin-disease-dataset).

## Model Architecture

The model uses **DenseNet-121**, a deep convolutional neural network known for its efficient use of parameters and feature reuse. It has been pre-trained on **ImageNet** and then fine-tuned on the skin lesion dataset.

### Key Details:
- **Model**: DenseNet-121
- **Dataset**: Kaggle's Skin-Disease-Dataset
- **Training**: Fine-tuning using pre-trained weights from ImageNet

## Usage

### Training the Model

To train the model, run the `train.py` script:
```bash
python train.py
```
This will start the training process using the dataset stored in the `data/` folder:

### Running the Website

The project also includes a website where you can interact with the model. Follow these steps to launch the website:

1. **Start the website:**

    ```bash
    python app.py
    ```

You should be able to access the website at `http://127.0.0.1:5000/` in your browser, where you can upload an image of a skin lesion for classification.


## License

This project is licensed under the MIT License. You are free to use, modify, and distribute the code for educational and research purposes. For more details, please see the LICENSE file in the repository.

