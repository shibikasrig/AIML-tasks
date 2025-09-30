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

🌳 Decision Trees & Random Forests

This project focuses on applying tree-based models for classification using the Heart Disease dataset. The goal is to build and evaluate Decision Trees and Random Forests, understand overfitting, interpret feature importance, and perform model validation.

📌 Task 5: Decision Trees & Random Forests

Objective: Learn and compare Decision Trees and Random Forests for classification, analyze overfitting, and interpret results.

Steps Covered:

1)Import Dataset & Split Data
Loaded the Heart Disease dataset using Pandas.
Split into training and testing sets using train_test_split.

2)Train Decision Tree & Visualize
Built a DecisionTreeClassifier.
Visualized tree structure with plot_tree (limited depth for readability).

3)Overfitting Analysis
Trained trees with increasing depth.
Plotted training vs testing accuracy.
Controlled complexity using max_depth and min_samples_split.

4)Train Random Forest & Compare
Built a RandomForestClassifier with 100 trees.
Compared accuracy with Decision Tree model.
Observed reduced variance and better generalization.

5)Feature Importances
Extracted feature_importances_ from Random Forest.
Plotted bar chart to identify most influential features.

6)Cross-Validation Evaluation
Used 5-fold cross-validation with cross_val_score.
Compared mean accuracy for Decision Tree and Random Forest.

Expected Outcome:
Decision Tree provides interpretability but tends to overfit.
Random Forest improves stability and accuracy using ensemble learning.
Feature importance reveals which attributes strongly influence predictions.

🌸 K-Nearest Neighbors (KNN) Classification

This project focuses on applying the K-Nearest Neighbors algorithm for classification using the Iris dataset. The goal is to understand how KNN works, experiment with different values of K, evaluate performance, and visualize decision boundaries.

📌 Task 6: K-Nearest Neighbors (KNN) Classification

Objective: Implement and analyze KNN for classification problems, tune hyperparameters, and visualize results.

Steps Covered:
1)Load Dataset & Normalize Features
Used the built-in Iris dataset from sklearn.datasets.
Extracted features and target classes.
Standardized numerical features using StandardScaler for fair distance calculations.

2)Build KNN Classifier
Used KNeighborsClassifier from sklearn.neighbors.
Trained initial model with default K (k=5).

3)Experiment with Different K Values
Tested multiple values of K (1–20).
Plotted accuracy vs K to find the optimal value.

4)Evaluate Model Performance
Calculated accuracy score on the test set.
Generated confusion matrix to analyze misclassifications.
Reported precision, recall, and F1-score.

5)Visualize Decision Boundaries
Plotted decision boundaries for two selected features (Petal Length, Petal Width).
Showed how different K values affect decision regions.

Expected Outcome:
Smaller K values (e.g., K=1) may lead to overfitting.
Larger K values smooth out decision boundaries but may underfit.
Optimal K balances bias and variance for better classification accuracy.
Visualization demonstrates how KNN classifies points based on nearest neighbors.


- 
