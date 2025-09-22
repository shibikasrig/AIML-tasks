# Titanic Dataset Preprocessing

This repo contains my work on cleaning and preprocessing the Titanic dataset.  
The aim is to prepare the data so that it can be used for machine learning models.  
I mainly focus on fixing missing values, encoding categorical data, scaling numeric features, and removing outliers.

---

## Steps I Followed

1. **Import and explore the dataset**
   - Used pandas to load the CSV file.
   - Looked at basic info (`head`, `info`, `describe`) and checked for null values.

2. **Handle missing values**
   - Filled missing `Age` with the median.
   - Filled missing `Embarked` with the most common value (`S`).
   - Dropped `Cabin` (too many nulls) but also created a new binary column `CabinKnown`.
   - Dropped `Ticket` (not useful for now).

3. **Encode categorical features**
   - Converted `Sex` into numeric (male=0, female=1).
   - Applied one-hot encoding for `Embarked`.

4. **Scale numeric columns**
   - Standardized `Age` and `Fare` using scikit-learn’s `StandardScaler`.

5. **Handle outliers**
   - Plotted boxplots for `Age` and `Fare` to spot outliers.
   - Removed them using the IQR method.

---

## Tools & Libraries
- Python 3
- Pandas
- NumPy
- Matplotlib / Seaborn
- Scikit-learn

---
