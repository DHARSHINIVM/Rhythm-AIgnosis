# Rhythm-AIgnosis
ECG Signal Classification using Machine Learning

Overview:
This project develops a machine learning model to classify ECG signals into different types (e.g., Atrial Fibrillation (AFF), Arrhythmia (ARR), Congestive Heart Failure (CHF), Normal Sinus Rhythm (NSR)). The project covers data loading, preprocessing, exploratory data analysis (EDA), model training, evaluation, and deployment through Streamlit and Flask web applications.

Features:
Data Loading and Initial Exploration: Loading ECG data and performing initial checks (head, info, shape, nulls).
Data Preprocessing: Handling missing values by imputation, dropping irrelevant columns, and selecting relevant features.
Outlier Detection and Removal: Using IQR method to identify and remove outliers from key numerical features.
Class Imbalance Handling: Applying SMOTE to balance the target variable classes.
Data Scaling: Standardizing numerical features using StandardScaler.
Model Training: Training a Random Forest Classifier.
Model Optimization: Hyperparameter tuning using GridSearchCV to find the best model.
Model Evaluation: Assessing model performance using accuracy, classification reports, and confusion matrices.
Model Persistence: Saving the trained model, scaler, and label encoder using pickle.
Streamlit Web Application: An interactive web interface to predict ECG signal types.
Flask Web Application: A web interface built with Flask for the same prediction task.
Dataset
The project uses the ECGCvdata.csv dataset, which contains various ECG features and a target column ECG_signal indicating the signal type.

Project Structure:
ECGCvdata.csv: The raw dataset.
random_forest_reduced_complexity_model.pkl: The trained Random Forest model (after hyperparameter tuning).
scaler.pkl: The fitted StandardScaler object.
label_encoder.pkl: The fitted LabelEncoder object for ECG_signal.
app.py (Streamlit version): Python script for the Streamlit web application.


Setup and Installation:
To run this project, you'll need Python and the following libraries:

pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn streamlit Flask
Usage
Data Preprocessing and Model Training
The notebook details the steps for data preprocessing, including handling missing values, feature selection, outlier removal, SMOTE for class balancing, and feature scaling. A Random Forest Classifier is trained and optimized using GridSearchCV.

Streamlit Web Application
To run the Streamlit application:

Ensure you have app.py (the Streamlit version), random_forest_reduced_complexity_model.pkl, scaler.pkl, and label_encoder.pkl in the same directory.
Open your terminal, navigate to that directory, and run: streamlit run app.py
Your web browser will open to the Streamlit application, where you can input ECG features and get predictions.

Results:

After training and optimization, the Random Forest model achieved a high accuracy:

Model Accuracy (after GridSearchCV): Approximately 92%
