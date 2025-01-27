# Heart Attack Risk Prediction Using Artificial Neural Networks (ANN)

## Overview

This project aims to predict heart attack risk using an Artificial Neural Network (ANN) based on various health and lifestyle-related features. The dataset undergoes preprocessing, exploratory data analysis (EDA), and transformation techniques such as feature scaling and one-hot encoding to prepare it for modeling.

## Dataset

The dataset includes the following features:
- **Age**
- **Sex**
- **Diet**
- **Blood Pressure** (split into Systolic and Diastolic values)
- **Other lifestyle and health metrics**
- **Target variable**: `Heart Attack Risk`

## Methodology

1. **Data Loading**: The dataset is loaded into a Pandas DataFrame for preprocessing.
2. **Exploratory Data Analysis (EDA)**:
   - Inspecting data for missing values, duplicates, and outliers.
   - Analyzing distributions and relationships between features.
3. **Data Cleaning**:
   - Unnecessary columns like `Patient ID`, `Income`, and `Country` are dropped.
   - `Blood Pressure` is split into `Systolic` and `Diastolic`.
4. **One-Hot Encoding**: Encoding categorical variables (`Sex`, `Diet`) for compatibility with the model.
5. **Feature Scaling**: Scaling numerical features using `MinMaxScaler` for normalization.
6. **Model Building**:
   - A deep learning ANN model is created using Keras with layers optimized for feature extraction and classification.
   - Regularization techniques like BatchNormalization are applied.
7. **Early Stopping**: Used to prevent overfitting during training by monitoring validation loss.
8. **Model Training**: Trained using the Adam optimizer and binary cross-entropy loss function.
9. **Model Evaluation**: Performance metrics such as accuracy are calculated on the test set.

## Model Architecture

- **Input Layer**: 256 neurons (ReLU activation)
- **Hidden Layers**: Multiple layers with 128, 64, 32, 16, and 8 neurons, all with ReLU activation and BatchNormalization.
- **Output Layer**: 1 neuron (Sigmoid activation)

## Results

- **Training Accuracy**: Achieved high accuracy during model training.
- **Validation Accuracy**: Demonstrated consistent performance on the test set.

## Visualization

The training process includes visualizations for:
- **Accuracy vs. Epochs**: Displays training and validation accuracy trends.
- **Loss vs. Epochs**: Shows training and validation loss convergence.

## Requirements

- Python 3.x
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/heart-attack-prediction-ANN.git
