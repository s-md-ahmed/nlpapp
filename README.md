# Disaster Message Classifier 🚨

A lightweight Natural Language Processing (NLP) web application built with **Flask** and **Scikit-Learn** that classifies whether a given text message refers to a disaster or not.

---

##  Architecture Diagram

The architecture follows a standard client-server pattern where user input is processed through a Flask backend equipped with a trained machine learning pipeline:

![System Architecture](https://www.plantuml.com/plantuml/png/ZLJRRXen47ttLqmtKWCea8k4gl1GXGkeZN91WLuUAbNExW0MYxqT3qb8b7-llTvIXpPInCCAdpbdvlZOFhCiYTUhN8XtlC0LGf5BfOLTAbr8aYiWizPPtv16WkDs18mTKQMga9h7CikKdiFnKOJtp4EqBbXnZZ3PMCOLT2bTACQKruGT62WhtJB19LehvmZzN5ghPWf929co8KY-MAG4f8MrTQl5tzwGVCDRw96vTy0FNViqlRmuWFD9lQnESuMQnQrvW6HWqYKIz8rcPz7BxSrE_Ky0l10SkVPBw1P5hbB9ocWdAukYLMo2wCpZ0Y2JB6-bHKYcv1Kpk3GPvgvwaggbukO5IjA1i_A8G9s1cqNiCFqU3ToIOxB_tqhjNQ3vGY7suRDMh6Ik7ijvO9IRUwZrPBgSb_dKXJQCOEwGmdG7WGdGQ5phs0fCoi2BJgFXq2O4hxcHcOLjxn8f9xDmOURAiahX1kU4rZgDo8nlAuJF19eVgzs63Wmr8pdJLPxlFP8kBUnNoMxsVQxgJZ9M914-JYKOS2BZwyaK3Wl2JAKCJvLO1nWV-7dB3-N1FwdvWi7CAiqETEr6fmiosgojC_1hZPQXrZtlVgyBKHBZXccEKl_qyhNw5Wrc7SepbSrQJ5BRcQ6LAzcE4FOZ53OZmQ_e3wzwn0ocJbLak4FvjqlvRLMctz9Z74fzBBKZC0J7zTVxn1cTJwwlu8dGhdFkGFBd4YPm2CcLOTWkFCTwfL0KLXbzUOMBN6xizkfMf-scb8y50jMVwGblSgNdqPFH9JBtVPCRWhrsknqZVYDxXZAa2ZyvEVd_O_J6gktTgIg7lL5xr7lvViLIEmCDgXlyMhUuPBVLQ3Ww7NuGuWnrvfxUtm00)

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
