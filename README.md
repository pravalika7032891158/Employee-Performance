📘 Employee Attrition Prediction – Flask Web App

This project is a Machine Learning–based Employee Attrition Prediction System built using Flask.

The web application allows users to input employee-related features and predicts whether the employee is likely to stay or leave based on trained ML models (Linear Regression, Random Forest, and XGBoost).

🚀 Features

🔹 Flask-based web interface

🔹 Three ML models integrated:

Linear Regression (model_lr.pkl)

Random Forest (model_rf.pkl)

XGBoost (model_xgb.pkl)

🔹 Data preprocessing & feature engineering performed in Jupyter Notebook

🔹 Interactive input form for predictions

🔹 Clean model loading with pickle

🔹 Easy deployable structure

🧠 Models Used:

| Model                      | Purpose                                                     |
| -------------------------- | ----------------------------------------------------------- |
| **Linear Regression (LR)** | Baseline simple regression model for performance prediction |
| **Random Forest (RF)**     | Handles non-linear patterns in employee data                |
| **XGBoost (XGB)**          | High-accuracy boosting model, best performance              |

🖥️ How Prediction Works

User enters employee information (e.g., age, salary, experience, satisfaction score, etc.)

App sends input to selected ML model.

Model predicts "Attrition" or "No Attrition".

Result is displayed on the results page.

🛠️ Important Notes

Use forward slashes in Flask paths:

IBM Files/model_lr.pkl

Avoid backslash escape errors.

Ensure model files are created before running Flask.

Restart Flask after any file changes.

📌 Technologies Used

Python

Flask

HTML / CSS

Machine Learning models (LR, RF, XGBoost)

Pandas, NumPy, Scikit-Learn
