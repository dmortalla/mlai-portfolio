# ALS Recommender – Matrix Factorization on Implicit Feedback

This hero app demonstrates a simple recommender system using
Alternating Least Squares (ALS) on implicit feedback data.

Users can:
- Upload a CSV containing `user_id`, `item_id`, and `interaction` columns, or
  use a built-in sample dataset.
- Train an ALS model with configurable hyperparameters.
- Get top-N recommendations for a selected user.

## 🚀 Features

- Implicit feedback modeling (e.g., views, clicks, watch time).
- Sparse user–item interaction matrix construction.
- ALS training with configurable factors, regularization, iterations, and alpha.
- Interactive Streamlit UI for exploring recommendations.

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🌐 Deployment

- Runs well on Streamlit Cloud (CPU).
- Can be deployed on Hugging Face Spaces (Streamlit SDK).
