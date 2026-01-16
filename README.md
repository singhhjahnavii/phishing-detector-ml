Phishing Website Detection using Supervised Machine Learning

Overview
This project implements a supervised machine learning model to detect phishing websites.
A logistic regression classifier is trained on labeled website features to classify
websites as phishing or legitimate.

Dataset

Source: UCI Machine Learning Repository
Instances: 11,055
Features: 30 website-based attributes
Target:
    1 → Phishing
    0 → Legitimate

Model

Algorithm: Logistic Regression
Learning Type: Supervised Learning
Evaluation Metrics:
    Accuracy
    Precision
    Recall
    F1-score

Results

Accuracy: ~92%
Strong balance between precision and recall for phishing detection

Technologies Used

Python
Pandas
NumPy
Scikit-learn
SciPy

How to Run

```bash
pip install -r requirements.txt
python phishing_model.py
```
