# Disaster Message Classifier 🚨

A lightweight Natural Language Processing (NLP) web application built with **Flask** and **Scikit-Learn** that classifies whether a given text message refers to a disaster or not.

---

## 📊 Architecture Diagram

The architecture follows a standard client-server pattern where user input is processed through a Flask backend equipped with a trained machine learning pipeline:

![System Architecture](architecture.png)

---

##  Features

- **Real-time Classification**: Instantly predicts whether a message describes a disaster via a clean HTML frontend.
- **NLP Text Vectorization**: Uses `TfidfVectorizer` with up to 5,000 features.
- **Machine Learning**: Powered by a tuned `LogisticRegression` model with balanced class weights.
- **Dockerized**: Fully containerized using Gunicorn for production-ready deployments.

---

##  Project Structure

```text
nlpapp/
├── static/
│   ├── script.js        # Frontend AJAX prediction handler
│   └── styles.css       # Application styling
├── templates/
│   └── index.html       # User interface template
├── app.py               # Flask application & ML pipeline
├── train.csv            # Training dataset
├── requirements.txt     # Python dependencies
└── Dockerfile           # Docker container configuration
