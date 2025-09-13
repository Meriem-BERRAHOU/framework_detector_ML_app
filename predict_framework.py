import joblib
from utils.features import extract_features
from pathlib import Path
import git
import uuid
import os

def predict_framework(model_path, vectorizer_path, repo_url):

    # Load trained model + vectorizer
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    # Clone repo into cloned_repo/
    repo_path = os.path.join("cloned_repo", f"temp_repo_{uuid.uuid4().hex[:6]}")
    git.Repo.clone_from(repo_url, repo_path)

    # Extract features
    features, code_text = extract_features(repo_path)

    # Vectorize
    text_features = vectorizer.transform([code_text])
    X = list(features.values()) + text_features.toarray().flatten().tolist()

    # Predict framework
    framework = model.predict([X])[0]


    print(f"✅ Framework détecté : {framework}")

    return framework, repo_path


if __name__ == "__main__":
    repo_url = input("📂 Donne l'URL du projet à analyser: ")
    predict_framework(
    'saved_models/logisticRegression_model.joblib',
    'saved_models/logisticRegression_vectorizer.joblib',
        repo_url
    )
