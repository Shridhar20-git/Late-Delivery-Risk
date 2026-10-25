# Late Delivery Risk Prediction

## 📌 Project Overview
Late Delivery Risk Prediction is a data analytics and machine learning project designed to analyze order delivery data and identify potential delivery delays.

The project uses a trained machine learning model and data analysis to help users understand delivery risks and the factors contributing to late deliveries.

## 🎯 Objectives
- Analyze order delivery data.
- Identify important factors associated with late deliveries.
- Predict delivery risk using a trained machine learning model.
- Present risk metrics and insights through a dashboard.

## 🛠️ Technologies Used
- Python
- Streamlit
- Pandas
- Machine Learning
- XGBoost
- JSON and CSV datasets

## 📂 Project Files
- `app.py` — Main application.
- `requirements.txt` — Required Python libraries.
- `xgb_model.json` — Trained XGBoost model.
- `preprocess.json` — Preprocessing configuration.
- `metrics.json` — Model metrics or analysis results.
- `risk_drivers.csv` — Delivery risk factors.
- `scored_orders.csv.gz` — Processed order data.

## 🚀 Run Locally

### 1. Clone the repository
```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the project folder
```bash
cd late-delivery-risk
```

### 3. Install dependencies
```bash
python -m pip install -r requirements.txt
```

### 4. Run the application
```bash
python -m streamlit run app.py
```

The application should open in your browser.

## ☁️ Deployment
The application can be deployed using Streamlit Community Cloud by connecting the GitHub repository and selecting `app.py` as the main application file.

## 📈 Expected Outcome
A dashboard for exploring delivery risk, relevant metrics, and factors associated with late deliveries.

## 👨‍💻 Author
Shridhar Jadhav
