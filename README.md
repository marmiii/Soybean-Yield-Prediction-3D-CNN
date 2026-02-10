# End-to-End 3D CNN for Soybean Yield Prediction 

This repository contains the implementation of an **End-to-End 3D Convolutional Neural Network (CNN)** for predicting soybean yield using multi-temporal UAV-based RGB images.

##  Project Overview
Traditional yield estimation methods are destructive and time-consuming. This project utilizes deep learning to analyze RGB images captured by UAVs (Drones) over time to predict soybean yield at the plot scale effectively.

### Key Features:
- **Model Architecture:** Custom 3D-CNN designed to capture both spatial (texture) and temporal (growth) features.
- **Input:** RGB Images transformed into HSV color space for better vegetation extraction.
- **Output:** Predicted Yield (kg/ha).

##  Methodology

### 1. Data Preprocessing & Pseudo-Labeling
Since granular ground truth data for every single image was unavailable, a **feature-based pseudo-labeling** approach was used. The yield labels were generated based on the **Greenness Ratio** (Vegetation Index) using the following formula:

$$ \text{Yield} = \text{Base} + (\text{Greenness\_Ratio} \times \text{Scale\_Factor}) $$

Where:
- **Base:** 2000 kg/ha (Minimum yield baseline)
- **Scale Factor:** 4000 (Impact of vegetation density on yield)
- **Greenness Ratio:** Calculated via HSV masking.

### 2. Model Training
- **Loss Function:** Mean Squared Error (MSE)
- **Optimizer:** Adam
- **Framework:** PyTorch
- **Epochs:** 3 (Fast convergence observed)

##  Dataset Structure
Please organize your data as follows to run the code successfully:

/Soybean-Project

├── dataset/

│ ├── images/ # Put your .jpg or .png or .tif images here

│ └── yield_data.csv # (Optional) Generated CSV file

├── train_model.py # Main python script

└── requirements.txt # List of libraries

 How to Run
Step 1: Clone the Repository
Open your terminal or command prompt and run:

bash

git clone https://github.com/YOUR_USERNAME/Soybean-Yield-Prediction-3D-CNN.git

cd Soybean-Yield-Prediction-3D-CNN

Step 2: Install Dependencies
Ensure you have Python installed, then run:

bash

pip install -r requirements.txt

Step 3: Run the Training
Start the training process by running:

bash

python train_model.py

 Results
The model successfully learned the correlation between vegetation indices and yield, achieving a final training loss of ~0.0015 (MSE).

Author: [Maryam Ghaffari]
