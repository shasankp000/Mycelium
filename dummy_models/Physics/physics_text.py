import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

csv_path = os.path.join(os.path.dirname(__file__), "physics_data.csv")
df = pd.read_csv(csv_path)

X = df['Comment']
y = df['binary_label']


# --- Training and saving code commented out to avoid retraining on import ---
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )
#
# vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
# X_train_tfidf = vectorizer.fit_transform(X_train)
# X_test_tfidf = vectorizer.transform(X_test)
#
# svm_clf = LinearSVC(random_state=42)
# svm_clf.fit(X_train_tfidf, y_train)
#
# y_pred = svm_clf.predict(X_test_tfidf)
#
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))
#
# with open("svm_model_physics.pkl", "wb") as f:
#     pickle.dump(svm_clf, f)
# with open("vectorizer_physics.pkl", "wb") as f:
#     pickle.dump(vectorizer, f)

with open("svm_model_physics.pkl", "rb") as f:
    svm_clf = pickle.load(f)
with open("vectorizer_physics.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def predict_text(text):
    text_tfidf = vectorizer.transform([text])
    prediction = svm_clf.predict(text_tfidf)[0]
    return prediction