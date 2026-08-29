# Disaster Message Classifier 

A lightweight Natural Language Processing (NLP) web application built with **Flask** and **Scikit-Learn** that classifies whether a given text message refers to a disaster or not.

---

##  Architecture Diagram

The architecture follows a standard client-server pattern where user input is processed through a Flask backend equipped with a trained machine learning pipeline:

![System Architecture](https://www.plantuml.com/plantuml/png/ZLHPRo8v47xthwZjI1j0WHmIMeM7LJX0kr4E59ZZOQIHuox0mjWzTdKI4kM_Z-tkOHn6aQ8yHFWxlYgNgoySSKlbHZ7s5wrmWr0eBZLpQwaBRla6h2brFZHMeuNJRW8CpDEKvxdKIpWzOOmaAOJPrX5ke6_5IX8AAYtsO2GTT-Jrr-WSNo8C5NTEBYHQnhWWOo7xvD1cm1sKpf_w19925bmJP5_m7WRMF7hWcsxyDx--EeBBMJEoXPAeYTsR9yX6HgpHmj1eub87kuF5UVXZ06yC9pv-3VsYK59maaPxMruKdM9RcIu2xaKv9txF7K8sjy4nXskJe_BicP1hIUqhv5PNcar06EeSV2riJndkQEIFl9B2xuvm3vKi41axXF-r9CcLV8xrm4IPHsWCk5Wlu_qtcJQ4O1xGLjKTLKg0LkjMfwt0B0wqwBLQ7cr3rQioF7Umouv8hCp1bLbAHrB07IujEkSzacP2B6DX9j3-jxvjwC5O4rhVT3tFlmCYrWuEwybk3yDSvGCdh4MirdgNs80tcTxEvd1SMCob87YfpNf0-4IlEtr5h_gtShaYC8lQimTzjzLYPOqsfVCD_2ZH4JJwb_slJSQYC0qK2hd-7kmRpHrQDUjHMiXyqI7BjLiOk_6KNGdLVLJ4TcBu6SFZbS-OmzoxiXpt97-cnF-TkkcFP4mZDOwsSGB6mcdp_PnqHfUpsnjuiUXAHJt8VYrX1iUGtHY2tS5hwXUDac75qSSLBXJVkjtgrg_jBjgd1fKqlEaPRPNKo-IJqRT-ZOT66Gi7tMutHS953epDqTRusTbPWaz_x-y7MRlTgPdZmQGx6RpzVgLMUmMDwWr-DosXx4TDnfFpyJ-CNQ3Eoutw2G00)

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
