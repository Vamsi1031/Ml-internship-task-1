# Ml-internship-task-1
Machine learning internship daily task usage purpose
# Titanic Data Cleaning & Preprocessing

## Objective
To clean and prepare the Titanic dataset for machine learning.

## Steps Performed

1. **Missing Values Handled**
   - Age: Filled using median
   - Embarked: Filled using mode
   - Cabin: Dropped due to many missing values

2. **Categorical Encoding**
   - Sex: Label encoded (male = 0, female = 1)
   - Embarked: One-hot encoded (kept two columns: Embarked_Q, Embarked_S)

3. **Feature Scaling**
   - Used StandardScaler to scale `Age` and `Fare` columns

4. **Outlier Removal**
   - Removed outliers in `Fare` using IQR method

5. **Result**
   - Final cleaned dataset saved as `titanic_cleaned.csv`
