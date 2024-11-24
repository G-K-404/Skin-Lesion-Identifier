Skin Lesion Identifier Using DenseNet-121
This project is a machine learning-based skin lesion identifier built using DenseNet-121, a deep convolutional neural network model. The model classifies skin lesions into 8 categories based on medical images. The dataset used for training is from Kaggle.

Table of Contents
Overview
Installation
Dataset
Model Architecture
Usage
License
Overview
Skin cancer is one of the most common forms of cancer, and early detection is crucial for effective treatment. This project uses a pre-trained DenseNet-121 model, fine-tuned on a Kaggle skin lesion dataset, to classify skin lesions into 8 different categories.

Installation
Follow the steps below to set up and run the project:

Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/skin-lesion-identifier.git
cd skin-lesion-identifier
Create a virtual environment (optional but recommended):

bash
Copy code
python -m venv venv
source venv/bin/activate  # For Linux/Mac
.\venv\Scripts\activate   # For Windows
Install the required dependencies manually: Since there is no requirements.txt, you need to install the necessary dependencies yourself. The key dependencies include:

bash
Copy code
pip install torch torchvision matplotlib numpy flask
Download the dataset from Kaggle and place it in the data/ folder.

Dataset
The dataset used in this project is the Skin Lesion Classification dataset from Kaggle. It contains images of skin lesions labeled with 8 different lesion classes:

BA-cellulitis
BA-impetigo
FU-athlete-foot
FU-nail-fungus
FU-ringworm
PA-cutaneous-larva-migrans
VI-chickenpox
VI-shingles
You can download the dataset from https://www.kaggle.com/datasets/subirbiswas19/skin-disease-dataset.

Model Architecture
The model uses DenseNet-121, a deep convolutional neural network known for its efficient use of parameters and feature reuse. It has been pre-trained on ImageNet and then fine-tuned on the skin lesion dataset.

Key Details:
Model: DenseNet-121
Dataset: Kaggle's Skin-Disease-Dataset
Training: Fine-tuning using pre-trained weights from ImageNet
Usage
Training the Model
To train the model, run the train.py script:

bash
Copy code
python train.py 
This will start the training process using the dataset stored in the models folder.

Running the Website
The project also includes a website where you can interact with the model. Follow these steps to launch the website:

Start the website:

bash
Copy code
python app.py
You should be able to access the website at http://127.0.0.1:5000/ in your browser, where you can upload an image of a skin lesion for classification.

License
This project is publicly available for educational and research purposes. Feel free to use the code and model.
