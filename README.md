# Disaster Message Classifier

A full-stack machine learning web application that classifies emergency messages in real time using Natural Language Processing (NLP), wrapped in a clean Flask backend and a responsive dark-mode interface.

---

## Evolution & Context (Foundational Experimentation)
This project represents an important foundational phase in my engineering journey. Built during my early exploration with machine learning and full-stack integration, it served as a practical sandbox to master end-to-end data processing, Scikit-Learn pipelines, Flask routing, and frontend-backend synchronization before scaling up to advanced multi-agent and multimodal systems.

---

## System Architecture & Workflow
The application ingests raw user text through an asynchronous frontend fetch loop, normalizes the text string, applies TF-IDF feature extraction, and feeds the vectorized data into a trained classification model to categorize alerts in real time.
![Uploading diagram (3).png…]()


### Why This Architecture? (Design Decisions & Trade-offs)
* **Why TF-IDF & Logistic Regression?**
  * *The Approach:* Combining `TfidfVectorizer` with a balanced `LogisticRegression` model provides microsecond-level response times, zero GPU dependencies, and high efficiency for lightweight, keyword-driven text classification tasks.
  * *The Fit:* As a foundational project developed and tested locally on standard hardware (such as a 16GB RAM laptop), this setup prioritized fast local iteration cycles, low latency, and straightforward deployment mechanics.

* **Why Class Weight Balancing (`class_weight='balanced'`)?**
  * *The Problem:* Disaster datasets are frequently skewed, containing an overwhelming majority of non-disaster chatter.
  * *The Solution:* Enforcing class balancing ensures the model maintains high recall for critical emergency alerts, preventing missed classifications on minority disaster classes.

---

## Core Features
* **Real-Time Classification:** Instantly evaluates user text strings via an asynchronous JavaScript interface.
* **Tuned Scikit-Learn Pipeline:** Features text normalization, L2 regularization (`C=0.9`), and an optimized `lbfgs` solver.
* **Clean Dark-Mode UI:** Custom dark-themed layout built with vanilla HTML5, CSS3, and JavaScript.
* **Deployment Ready:** Packaged with a monolithic Flask structure and containerization support.

---

## Tech Stack
* **Backend:** Flask, Pandas, Scikit-Learn (`LogisticRegression`, `TfidfVectorizer`)
* **Frontend:** HTML5, CSS3, Vanilla JavaScript (Fetch API)
* **Infrastructure:** Docker, Python Runtime Environment

---

## Project Structure
```text
├── static/
│   ├── styles.css          # Custom dark-mode styling and UI layout
│   └── script.js           # Asynchronous prediction handler and input validation
├── templates/
│   └── index.html          # Frontend UI shell for message entry and result rendering
├── app.py                  # Flask application entrypoint, model training pipeline, and predict routes
├── requirements.txt        # Python package dependencies
└── Dockerfile              # Container configuration file
