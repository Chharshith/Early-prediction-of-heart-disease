# Algorithm development for Research Early-prediction-of-heart-disease

# Heart Disease Prediction using Decision Tree Classifier

## Overview
This project implements a Decision Tree Classifier to predict the presence of heart disease using a dataset derived from a web API. The workflow includes data fetching, preprocessing, exploratory data analysis (EDA), and model training. The model is evaluated using metrics such as accuracy, classification report, confusion matrix, and visualizations of the decision tree structure.

## Table of Contents
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Training](#model-training)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Technologies Used
- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- Requests

## Dataset
The dataset used for this project is fetched from a web API and includes the following features:
- `age`: Age of the patient
- `sex`: Gender of the patient (1 = male; 0 = female)
- `cp`: Chest pain type (0-3)
- `trestbps`: Resting blood pressure (in mm Hg)
- `chol`: Serum cholesterol in mg/dl
- `fbs`: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
- `restecg`: Resting electrocardiographic results (0-2)
- `thalach`: Maximum heart rate achieved
- `exang`: Exercise induced angina (1 = yes; 0 = no)
- `oldpeak`: ST depression induced by exercise relative to rest
- `slope`: Slope of the peak exercise ST segment (0-2)
- `ca`: Number of major vessels (0-3) colored by fluoroscopy
- `thal`: Thalassemia (1 = normal; 2 = fixed defect; 3 = reversable defect)
- `target`: Diagnosis of heart disease (1 = presence; 0 = absence)

## Installation
To set up this project, clone the repository and install the required packages using pip:

```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
pip install -r requirements.txt
