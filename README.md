🛳️ Titanic Data Analysis & Preprocessing

This project focuses on exploring and preparing the Titanic dataset for machine learning. The goal is to clean the data, understand patterns, handle missing values, encode categorical features, standardize numerical data, detect and remove outliers, and perform exploratory data analysis (EDA).

📌 Task 1: Data Preprocessing

Objective: Prepare the dataset for machine learning models.

Steps Covered:

1)Import Dataset & Explore Basic Info
Loaded dataset using Pandas.
Explored dataset structure with head(), info(), and describe().
Checked for missing values and data types.

2)Handle Missing Values
Age → filled with median.
Embarked → filled with mode (S).
Cabin → dropped or transformed into a binary feature (CabinKnown).
Ticket → dropped (not directly useful).

3)Convert Categorical Features to Numerical
Sex → mapped (male=0, female=1).
Embarked → one-hot encoded (Embarked_Q, Embarked_S).

4)Normalize / Standardize Numerical Features
Standardized Age and Fare using StandardScaler.

5)Outlier Detection & Removal
Visualized outliers in Age and Fare using boxplots.
Removed outliers using the IQR (Interquartile Range) method.

📌 Task 2: Exploratory Data Analysis (EDA)

Objective: Understand the Titanic dataset using statistics and visualizations.

Steps Covered:

1)Summary Statistics
Calculated mean, median, standard deviation, etc.

2)Visualizations
Created histograms and boxplots for numeric features (Age, Fare).
Generated pairplots to explore relationships between features.
Created correlation heatmaps to identify feature relationships.

3)Pattern & Anomaly Detection
Identified trends, patterns, and anomalies in the dataset.
Made feature-level inferences from visuals to support model building.

📊 Tools & Libraries Used

Python 3.x
Pandas – data manipulation
NumPy – numerical operations
Matplotlib / Seaborn – visualization
Scikit-learn – preprocessing, encoding, and scaling

📌 Task 3: Linear Regression

Objective: Implement and understand simple & multiple linear regression.

Steps Covered:

1)Import & Preprocess Dataset
Loaded dataset using Pandas.
Selected independent variables (X) and target variable (y).
Handled missing values and scaled features if needed.

2)Train-Test Split
Split data into training and testing sets using train_test_split.
Used 80% for training and 20% for testing.

3)Fit Linear Regression Model
Used LinearRegression from scikit-learn.
Trained model on training data.
Predicted target values for test set.

4)Model Evaluation
Evaluated model using:
Mean Absolute Error (MAE)
Mean Squared Error (MSE)
R² Score

5)Plot & Interpret Results
For simple regression: plotted regression line against actual data.
For multiple regression: interpreted coefficients to understand feature impact.

📊 Tools & Libraries Used
Python 3.x
Pandas – data manipulation
NumPy – numerical operations
Scikit-learn – model training & evaluation
Matplotlib / Seaborn – visualization

📌 Task 4: Classification with Logistic Regression

Objective: Build a binary classifier using logistic regression.

Steps Covered:

1)Import & Preprocess Dataset
Loaded a binary classification dataset using Pandas.
Handled missing values, encoded categorical variables, and standardized features.

2)Train-Test Split
Split dataset into training and testing sets using train_test_split.
Used 80% for training and 20% for testing.

3)Fit Logistic Regression Model
Used LogisticRegression from scikit-learn.
Trained model on training data.
Predicted class labels and probabilities on the test set.

4)Model Evaluation
Evaluated model using:
Confusion Matrix
Precision, Recall, and F1-Score
ROC Curve and AUC Score

5)Threshold Tuning & Sigmoid Function
Explored different classification thresholds.
Explained how the sigmoid function maps input features to probabilities between 0 and 1.

📊 Tools & Libraries Used
Python 3.x
Pandas – data manipulation
NumPy – numerical operations
Scikit-learn – model training & evaluation
Matplotlib / Seaborn – visualization


- 
