# 🃏 Card Classification with Convolutional Neural Network (CNN)

This project implements a **Convolutional Neural Network (CNN)** using TensorFlow/Keras to classify playing cards into 53 unique classes based on image data. The dataset consists of labeled card images categorized into training, validation, and test sets. The model is trained, evaluated, and visualized to assess its performance.

## 📁 Dataset

The dataset is provided through a `cards.csv` file containing:

- **filepaths**: path to the image file  
- **labels**: human-readable class label (e.g., "King of Hearts")  
- **class index**: integer index representing the label (0-52)  
- **data set**: one of `"train"`, `"valid"`, or `"test"`

Each image is assumed to be a 224x224 RGB image.

## 🔍 Project Structure

- `cards.csv` — metadata for the dataset  
- `your_model.h5` — saved Keras model after training  
- Python script — handles loading, preprocessing, training, and evaluation  

## 🚀 Workflow Overview

1. **Load and preprocess** image data
2. **Shuffle and normalize** datasets
3. **Build CNN architecture** with 5 convolutional layers and fully connected layers
4. **Train model** using training and validation sets
5. **Evaluate** model on the test set
6. **Visualize** loss, accuracy, and predictions

## 🧠 Model Architecture

```text
Input: (224, 224, 3)

[Conv2D + ReLU] -> [MaxPooling2D]
[Conv2D + ReLU] -> [MaxPooling2D]
[Conv2D + ReLU] -> [MaxPooling2D]
[Conv2D + ReLU] -> [MaxPooling2D]
[Conv2D + ReLU] -> [MaxPooling2D]

[Flatten]
[Dense + ReLU]
[Dense + Softmax] → Output: 53 classes

