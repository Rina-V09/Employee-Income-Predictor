# 🧑‍💼 Employee Income Prediction using Machine Learning

This project aims to predict whether an employee's income exceeds $50K per year based on demographic and work-related features. It includes a Streamlit web interface that allows users to input employee data and receive instant predictions.

## 📌 Project Summary

The model is trained on the UCI Adult dataset and classifies whether an individual's income is `<=50K` or `>50K`. It leverages preprocessing, encoding, and a trained machine learning classifier to deliver accurate results.

## 🚀 Features

- Streamlit-based interactive web app
- Input form for features like age, workclass, education, marital status, etc.
- Real-time prediction of income class
- Displays prediction result clearly to user
- Clean, simple UI for non-technical users

## 📁 Project Structure

- `app/` — Contains the Streamlit app (`app.py`)
- `model/` — Pickled model and column list
- `notebooks/` - Jupyter notebooks for data analysis and modeling
- `data/` — Example/test input files
- `assets/` — Images or logos for documentation

## ▶️ Running the App

```bash
pip install -r requirements.txt
streamlit run app/app.py


