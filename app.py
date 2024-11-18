from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

app = Flask(__name__, static_folder="static", template_folder="templates")

# Load and preprocess data
train_df = pd.read_csv("train.csv")


def clean_text(text):
    return text.lower()


train_df["cleaned_text"] = train_df["text"].apply(clean_text)

# Vectorize text
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(train_df["cleaned_text"])
y = train_df["target"]

# Split data and train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(C=0.9, penalty="l2", solver="lbfgs", class_weight="balanced")
model.fit(X_train, y_train)

# Precompute classification reports
train_report_text = classification_report(y_train, model.predict(X_train))
test_report_text = classification_report(y_test, model.predict(X_test))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    user_text = data["message"]
    cleaned_text = clean_text(user_text)
    vectorized_text = tfidf.transform([cleaned_text])
    prediction = model.predict(vectorized_text)[0]
    result = "Disaster" if prediction == 1 else "Not Disaster"

    # Return prediction and classification reports
    return jsonify(
        {
            "result": result,
            "train_report": train_report_text,
            "test_report": test_report_text,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
