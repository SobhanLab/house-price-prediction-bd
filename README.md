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
* Handling missing values
* Outlier Removal (IQR method)
* Log Transformation
* One-Hot Encoding
* Model tuning

---

## 💾 Saved Model

* `model.pkl` → trained model for reuse without retraining

---

## 🚀 Project Structure

```
house-price-prediction-bd/
│
├── notebook.ipynb
├── model.pkl
├── app.py
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run Locally

1. Clone the repository:

```
git clone https://github.com/SobhanLab/house-price-prediction-bd.git
```

2. Navigate to the project folder:

```
cd house-price-prediction-bd
```

3. Install dependencies:

```
pip install -r requirements.txt
```

4. Run the app:

```
streamlit run app.py
```

---

## 🌐 Live Demo

(Coming soon after deployment)

---

## 🌱 Future Improvements

* Add full feature input (City, Location)
* Improve prediction accuracy with more data
* Enhance UI/UX of the web app
* Deploy publicly for real users

---

## 👨‍💻 Author

* Abdus Sobhan
