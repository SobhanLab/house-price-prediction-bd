# 🏠 House Price Prediction (Bangladesh)

A Machine Learning project that predicts house prices in Bangladesh using XGBoost.

---

## 📊 Dataset

* Source: https://www.kaggle.com/datasets/durjoychandrapaul/house-price-bangladesh
* Real estate data including:

  * Bedrooms
  * Bathrooms
  * Floor Area
  * City
  * Location

---

## ⚙️ Problem Statement

Estimate the price of a house based on its features using supervised machine learning.

---

## 🧠 Model

* XGBoost Regressor (tuned)

---

## 📈 Performance

* R² Score: **0.849**

---

## 🔧 Techniques Used

* Data Cleaning (currency formatting)
* Missing value handling
* Outlier Removal (IQR)
* Log Transformation
* One-Hot Encoding
* Model tuning

---

## 💾 Saved Model

* `model.pkl` → trained model for reuse

---

## 🚀 Project Structure

```
house-price-prediction-bd/
│
├── notebook.ipynb
├── model.pkl
├── app.py   (coming soon)
└── README.md
```

---

## ▶️ How to Run

1. Clone this repo:

```
git clone https://github.com/SobhanLab/Learning_Python.git
```

2. Navigate to project folder:

```
cd Project/house-price-prediction-bd
```

3. Open notebook:

```
notebook.ipynb
```

---

## 🌐 Future Improvements

* Build Streamlit web app
* Deploy model online
* Improve feature engineering

---

## 👨‍💻 Author

* Abdus Sobhan
