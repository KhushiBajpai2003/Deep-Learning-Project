# Kidney Lesion Classifier #

# Overview
This project aims to classify kidney lesions into four classes: cyst, normal, tumor, and stone. It utilizes a Convolutional Neural Network (CNN) model trained on a dataset containing images of kidney lesions. The CNN model is capable of predicting the class of a given kidney image with high accuracy.

# Dataset
The dataset used for training the CNN model can be obtained from Kaggle at the following link: CT Kidney Dataset: Normal, Cyst, Tumor, and Stone.

To use the dataset:

Download the dataset from the provided Kaggle link in the references.
Extract the dataset files to a directory on your local machine.
Ensure that the dataset directory contains subdirectories corresponding to each class (cyst, normal, tumor, and stone), with images stored inside each subdirectory.

# Model Architecture
The CNN model architecture used for this classification task is designed to effectively learn features from the input images and make accurate predictions. It comprises multiple convolutional layers followed by pooling layers to extract hierarchical features from the input images. The extracted features are then flattened and passed through fully connected layers to make predictions.

# Usage
To use the Kidney Lesion Classifier, follow these steps:

Clone this repository to your local machine.
Install the required dependencies listed in the requirements.txt file.
Download and preprocess the kidney images from the provided dataset.
Load the trained CNN model using the provided script.
Use the loaded model to classify kidney images into one of the four classes: cyst, normal, tumor, or stone.

# Evaluation
The performance of the Kidney Lesion Classifier can be evaluated using metrics such as accuracy, precision, recall, and F1-score. These metrics provide insights into the model's ability to correctly classify kidney lesions into their respective classes.

# Contribution
Contributions to the Kidney Lesion Classifier project are welcome! If you find any bugs or have suggestions for improvements, feel free to open an issue or submit a pull request.

# License
This project is licensed under the MIT License.

# References
[1] Nazmul Islam, "CT Kidney Dataset: Normal, Cyst, Tumor, and Stone," Kaggle, Available at: https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone.
