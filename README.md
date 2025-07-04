# 🎬 Netflix Ads: Click-Through Rate (CTR) Prediction System

This project is an end-to-end machine learning system that predicts the probability of a user clicking on an online ad — inspired by ad systems used at Netflix, Google, and Meta. Built with `XGBoost`, `Streamlit`, and a user-friendly Netflix-style UI.

## 🔮 Features

- 📈 Predict CTR (click-through rate) for ads
- 🧠 Trained using real-world features like hour, device type, site ID, and banner position
- 🧰 Uses Label Encoders for categorical input
- 🌐 Fully interactive UI built with Streamlit
- 🚀 One-click deployable on Streamlit Cloud

## 🧠 Tech Stack

- Python
- XGBoost
- Scikit-learn
- Streamlit
- Pandas
- Joblib

## 📁 Folder Structure

```
ctr_netflix_app/
├── app.py                  # Streamlit frontend app
├── requirements.txt        # Packages for deployment
├── model/
│   ├── xgb_model.pkl       # Trained XGBoost model
│   └── label_encoders.pkl  # Label encoders for categorical features
```

## 🚀 How to Run Locally

```bash
git clone <repo-url>
cd ctr_netflix_app
pip install -r requirements.txt
streamlit run app.py
```

## 📦 Deployment

Easily deploy on [Streamlit Cloud](https://streamlit.io/cloud):

1. Upload this folder to GitHub
2. Go to Streamlit Cloud
3. Click "New App", connect your GitHub repo
4. Set `app.py` as the entry point and deploy!

## 📌 Input Features

| Feature             | Type        | Description                      |
|---------------------|-------------|----------------------------------|
| `site_id`           | Categorical | Encoded site identifier          |
| `device_model`      | Categorical | Encoded device model             |
| `hour`              | Numeric     | Hour of the ad impression        |
| `banner_pos`        | Numeric     | Position of the ad banner        |
| `device_type`       | Numeric     | Type of device (mobile, tablet)  |
| `device_conn_type`  | Numeric     | Type of internet connection      |
| `C1`                | Numeric     | Ad campaign category code        |

## 🙋‍♂️ Author

**Kaushik Dasari**  
Beginner ML Engineer | Passionate about building FAANG-ready AI systems
