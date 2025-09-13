import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import json
import pandas as pd
from utils.features import extract_features
from sklearn.preprocessing import LabelEncoder

def train_xgboost(save_prefix="saved_models/xgb",results_prefix='results/xgb'):
    
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

    print("🚀 Training xgboot...")
    # TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=2000, analyzer="char_wb", ngram_range=(3, 5))
    text_features = vectorizer.fit_transform([x[-1] for x in data])   # ransformer du texte brut en vecteurs numériques
        
    # Combine toutes les caractéristiques
    X = []
    for i, item in enumerate(data):
        combined = list(item[:-1]) + text_features[i].toarray().flatten().tolist()
        X.append(combined)

    # Encode les labels (string -> int)
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels_encoded, test_size=0.2, random_state=42
    )

    # XGBoost model
    model = xgb.XGBClassifier(
        objective="multi:softmax",  # for multi-class classification
        eval_metric="mlogloss",
        use_label_encoder=False,
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Évaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Sauvegarde modèle et vectorizer
    joblib.dump(model, f"{save_prefix}_model.joblib")
    joblib.dump(vectorizer, f"{save_prefix}_vectorizer.joblib")

    # Décoder les prédictions
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    y_test_labels = label_encoder.inverse_transform(y_test)

    # Rapport brut (dict)
    report_dict = classification_report(y_test_labels, y_pred_labels, output_dict=True)

    # ✅ Transformer en DataFrame
    df_report = pd.DataFrame(report_dict).transpose()
    print("\n📊 Rapport de classification :")
    print(df_report.to_string(float_format="%.2f")
          )
    
    # ✅ Matrice de confusion
    cm = confusion_matrix(y_test_labels, y_pred_labels, labels=label_encoder.classes_)
    df_cm = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)

    print("\n📉 Matrice de confusion :")
    print(df_cm.to_string())

    # Sauvegarde résultats
    results = {
        "model": "XGBoost",
        "accuracy": acc,
        "report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }
    with open(f"{results_prefix}_results.json", "w") as f:
        json.dump(results, f, indent=4)

    return results
