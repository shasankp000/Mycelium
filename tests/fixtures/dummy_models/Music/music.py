import pandas as pd
import os
import pickle

csv_path = os.path.join(os.path.dirname(__file__), "music_classification_dataset.csv")
music_data = pd.read_csv(csv_path)

music_data.dropna(subset=['sentence'], inplace=True)

print(music_data.head())

print(f'Shape of Dataframe after cleaning: {music_data.shape}')
print((music_data.info()))

X = music_data['sentence']
y = music_data['label']


# --- Training and saving code commented out to avoid retraining on import ---
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )
#
# vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
# X_train_tfidf = vectorizer.fit_transform(X_train)
# X_test_tfidf = vectorizer.transform(X_test)
#
# svm_clf = SVC(kernel='linear', C=1, random_state=42)
# svm_clf.fit(X_train_tfidf, y_train)
#
# y_pred = svm_clf.predict(X_test_tfidf)
#
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))
#
# with open("svm_model_music.pkl", "wb") as f:
#     pickle.dump(svm_clf, f)
# with open("vectorizer_music.pkl", "wb") as f:
#     pickle.dump(vectorizer, f)

with open("svm_model_music.pkl", "rb") as f:
    svm_clf = pickle.load(f)
with open("vectorizer_music.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def predict_text(text):
    text_tfidf = vectorizer.transform([text])
    prediction = svm_clf.predict(text_tfidf)[0]
    return prediction