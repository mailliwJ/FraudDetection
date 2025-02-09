# Fraud Detection with Imbalanced Learning

## Overview
This project focuses on **fraud detection in credit card transactions** using various **imbalanced learning techniques**. Due to the severe class imbalance in fraud datasets, standard machine learning models tend to perform poorly on the minority (fraudulent) class. This project explores different **sampling strategies** to mitigate this issue and evaluates multiple classification models.

## Project Structure
- **1_exploration.ipynb** → Initial data exploration and visualization
- **2_sampling_trials.ipynb** → Sampling techniques and model evaluation
- **data/** → Contains the dataset and processed versions after sampling
- **models/** → Stores trained models after optimization

## Dataset
The dataset is sourced from **credit card transactions**, where the goal is to **detect fraudulent transactions**. It contains numerical features derived from PCA transformations and a highly imbalanced target variable (`Class` where 1 = Fraud, 0 = Non-Fraud).

## Methods & Techniques
### 1. Exploratory Data Analysis (EDA)
- Visualized class distributions and feature distributions using **histograms, KDE plots and boxplots**.
- Identified potential feature separability between fraudulent and non-fraudulent transactions.
- Trialed a *"from scratch"* threshold based feature selection function.

### 2. Handling Class Imbalance
To improve classification performance, several **sampling techniques** were applied:
- **Under-sampling:**
  - Random Under-Sampler (RUS)
  - NearMiss
- **Over-sampling:**
  - Synthetic Minority Over-Sampling Technique (**SMOTE**)
- **Hybrid Sampling Methods:**
  - **SMOTEENN** (SMOTE + Edited Nearest Neighbors)
  - **SMOTETomek** (SMOTE + Tomek Links)

### 3. Model Training & Evaluation
- **Algorithms Used:**
  - **Logistic Regression**
  - **Random Forest**
  - **XGBoost**
- **Cross-Validation Metrics:**
  - Recall, Precision, F1-Score, PR-AUC, ROC-AUC
- **Confusion Matrices** were analyzed for each sampling technique to assess performance on fraud detection.

### 4. Hyperparameter Tuning
- Used **RandomizedSearchCV** to optimize model parameters.
- Saved the best models for potential deployment.

## Results
- **Hybrid methods like SMOTEENN and SMOTETomek** significantly improved recall without sacrificing precision.
- **XGBoost performed best**, achieving a good balance between precision and recall after tuning.

## References & Sources
- [Combining Over-Sampling and Under-Sampling](https://machinelearningmastery.com/combine-oversampling-and-undersampling-for-imbalanced-classification/)
- [Using Statistics to Identify Outliers](https://machinelearningmastery.com/how-to-use-statistics-to-identify-outliers-in-data/)
- [Under-Sampling Algorithms for Imbalanced Classification](https://machinelearningmastery.com/undersampling-algorithms-for-imbalanced-classification/)

## Future Improvements
- **Feature Engineering:** Explore additional derived features.
- **Deep Learning Models:** Test LSTM/Autoencoders for anomaly detection.
- **Deployment:** Convert the model into an API for real-time fraud detection.

## License
This project is licensed under the MIT License.

