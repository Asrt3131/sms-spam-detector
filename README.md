# 📧 SMS Spam Detection

A machine learning project that classifies SMS messages as **Spam** or **Ham** (not spam) with over **95% accuracy**.

![Demo Screenshot](screenshot.png)

---

## 🚀 Live Demo

Try the app here: [YOUR_APP_LINK](YOUR_APP_LINK)

Enter any SMS text and the model will instantly tell you whether it is spam or not.

---

## 📌 Project Overview

Spam messages are annoying and sometimes dangerous (phishing, scams, etc.). This project builds a text classification model that automatically detects spam SMS messages.

The pipeline:
1. Load and clean the SMS dataset
2. Convert text to numerical features using **TF-IDF**
3. Train a **Multinomial Naive Bayes** classifier
4. Evaluate the model (accuracy > 95%)
5. Deploy as an interactive **Streamlit** web app

---

## 🧠 Tech Stack

- **Python 3**
- **Pandas** – data manipulation
- **Scikit-learn** – TF-IDF vectorization, Naive Bayes model, evaluation
- **Joblib** – model serialization
- **Streamlit** – interactive web app

---

## 📂 Dataset

- **Name:** SMS Spam Collection Dataset
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip)
- **Size:** 5,574 SMS messages
- **Labels:** `ham` (non-spam) and `spam`

Distribution:
- Ham: ~4,825 messages (~86%)
- Spam: ~747 messages (~14%)

---

## 📊 Model Performance

| Metric     | Score   |
|------------|---------|
| Accuracy   | ~97%    |
| Precision  | ~99%    |
| Recall     | ~90%    |
| F1-Score   | ~94%    |

> Note: Because the dataset is imbalanced, **F1-Score** and **Recall** are more meaningful than accuracy alone.

---

## 🖼️ Screenshot

![App Screenshot](screenshot.png)

*The Streamlit app detecting a spam message in real time.*

---

## ⚙️ How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/a-srt343/sms-spam-detection.git
cd sms-spam-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py
