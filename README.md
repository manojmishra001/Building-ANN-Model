
# 🚗 Car Sales Price Prediction Using Machine Learning (ANN)

## 📌 Project Overview

This project focuses on building a **Car Sales Price Prediction model** using **Machine Learning and Artificial Neural Networks (ANN)**.
The objective is to accurately predict the **car purchase price** based on customer demographics, financial attributes, and vehicle-related features.

The project demonstrates a **complete ML lifecycle**, from data preprocessing to model evaluation and prediction.

---

## 🎯 Problem Statement

Car dealerships need accurate pricing predictions to:

* Improve sales strategies
* Understand customer purchasing power
* Offer competitive and personalized pricing

This model predicts the **car sales price** using historical data and customer attributes.

---

## 🚀 Key Features

* Exploratory Data Analysis (EDA)
* Data cleaning & preprocessing
* Handling missing, duplicate & irrelevant columns
* Feature scaling using StandardScaler
* ANN model implementation using TensorFlow/Keras
* Model tuning with Dropout & EarlyStopping
* Performance evaluation using MAE, RMSE & R²
* Visualization using Matplotlib & Seaborn
* Prediction on new customer data

---

## 🧠 Technologies Used

* **Python 3.9+**
* **Pandas & NumPy** – Data handling
* **Scikit-learn** – Preprocessing & evaluation
* **TensorFlow / Keras** – ANN modeling
* **Matplotlib & Seaborn** – Data visualization

---

## 📂 Project Structure

```
├── data/
│   └── car_purchasing.csv
├── notebooks/
│   └── car_price_prediction_ann.ipynb
├── models/
│   └── ann_car_price_model.h5
├── README.md
├── requirements.txt
```

---

## 🛠️ Data Preprocessing

* Removed duplicate rows and columns
* Dropped empty and irrelevant features
* Encoded categorical variables (if any)
* Separated features (X) and target variable (y)
* Train-test split
* Feature scaling using `StandardScaler`
* Target scaling with inverse transformation for predictions

---

## 🏗️ ANN Model Architecture

```text
Input Layer
↓
Dense (128 neurons, ReLU)
↓
Dropout (0.2)
↓
Dense (64 neurons, ReLU)
↓
Output Layer (1 neuron, Linear)
```

* **Optimizer:** Adam
* **Loss Function:** Mean Squared Error (MSE)
* **Evaluation Metrics:** MAE

---

## 📊 Model Evaluation

The model is evaluated using:

* **MAE (Mean Absolute Error)**
* **RMSE (Root Mean Squared Error)**
* **R² Score**

### Visual Analysis:

* Actual vs Predicted car prices
* Error distribution
* Training vs Validation loss curves

---

## 🔮 Prediction

The trained ANN model can predict **car purchase price for new customers**, provided the same preprocessing and scaling steps are applied.

Example use case:

```text
Input: Age, Annual Salary, Credit Score, Net Worth
Output: Predicted Car Purchase Price
```


## 📈 Results Summary

* ANN successfully captures non-linear relationships in customer data
* Predictions closely match actual car prices
* Overfitting controlled using EarlyStopping and Dropout
* Model performs better than traditional linear regression

---

## 📌 Future Enhancements

* Hyperparameter tuning using KerasTuner
* Feature importance analysis
* Model comparison with Random Forest / XGBoost
* Deployment using Flask or FastAPI
* Real-time prediction dashboard

---

## 👤 Author

**Manoj Kumar Mishra**
Finance & Analytics Professional | Machine Learning Enthusiast



### ⭐ If you find this project helpful, please star the repository!

