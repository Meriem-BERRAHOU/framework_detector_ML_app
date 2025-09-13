import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
import pandas as pd 
from utils.features import extract_features

def train_svm(save_prefix="saved_models/svm" ,results_prefix="results/svm"):
    
    df = pd.read_csv("data/formations.csv")
    df["framework"] = df["framework"].str.strip().str.lower()

    # Nombre de classes
    n_classes = df["framework"].nunique()
    print("Nombre de classes :", n_classes)

    # Liste des classes uniques
    classes = df["framework"].unique()
    print("Classes :", classes)

    data, labels = [], []
    i=0
    for _, row in df.iterrows():
                repo_path = row['repo_path']
                i = i + 1
                print(f"🔍 Traitement du repo {i}/{len(df)}: {repo_path}")
                try:
                    features, code_text = extract_features(repo_path)
                    if not code_text.strip():
                        print(f"⚠️ Repo vide ignoré: {repo_path}")
                        continue

                    data_item = list(features.values())
                    data_item.append(code_text)
                    data.append(data_item)
                    labels.append(row['framework'])

                except Exception as e:
                    print(f"❌ Erreur pour {repo_path}: {e}")
                    continue

    print("🚀 Training svm...")
    # TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=2000, analyzer="char_wb", ngram_range=(3, 5))
    text_features = vectorizer.fit_transform([x[-1] for x in data])   # ransformer du texte brut en vecteurs numériques
        
    # Combine toutes les caractéristiques
    X = []
    for i, item in enumerate(data):
        combined = list(item[:-1]) + text_features[i].toarray().flatten().tolist()
        X.append(combined)


    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42
    )

    # SVM model (linear kernel)
    model = LinearSVC(max_iter=5000)
    model.fit(X_train, y_train)

    # Évaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Sauvegarde modèle et vectorizer
    joblib.dump(model, f"{save_prefix}_model.joblib")
    joblib.dump(vectorizer, f"{save_prefix}_vectorizer.joblib")

    # Rapport brut (dict)
    report_dict = classification_report(y_test, y_pred, output_dict=True)

    # ✅ Transformer en DataFrame
    df_report = pd.DataFrame(report_dict).transpose()
    print("\n📊 Rapport de classification :")
    print(df_report.to_string(float_format="%.2f")
          )
    
    # ✅ Matrice de confusion
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    df_cm = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)

    print("\n📉 Matrice de confusion :")
    print(df_cm.to_string())

    # Sauvegarde résultats
    results = {
        "model": "LinearSVC",
        "accuracy": acc,
        "report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }
    with open(f"{results_prefix}_results.json", "w") as f:
        json.dump(results, f, indent=4)

    return results
