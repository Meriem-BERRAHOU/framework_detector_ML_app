import joblib
import json
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import RobertaTokenizer, RobertaModel
import pandas as pd
from utils.features import extract_features
import warnings

warnings.filterwarnings("ignore", message="`clean_up_tokenization_spaces` was not set")


def train_codebert_mlp(save_prefix="saved_models/codebert_mlp",results_prefix='results/codebert_mlp'):
    
    df = pd.read_csv("data/formations.csv")

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

    print("🚀 Training Codebert+mlp...")
    # Load CodeBERT tokenizer + model
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    model = RobertaModel.from_pretrained("microsoft/codebert-base")

    texts = [item[-1] for item in data]

    #Toknize the texts
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**encodings)
        # Mean pooling of token embeddings
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42
    )

    # MLP classifier
    clf = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluation
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Save classifier + tokenizer (embedding generator)
    joblib.dump(clf, f"{save_prefix}_clf.joblib")
    joblib.dump(tokenizer, f"{save_prefix}_tokenizer.joblib")

    # Rapport brut (dict)
    report_dict = classification_report(y_test, y_pred, output_dict=True)

    # ✅ Transformer en DataFrame
    df_report = pd.DataFrame(report_dict).transpose()
    print("\n📊 Rapport de classification :")
    print(df_report.to_string(float_format="%.2f")
          )
    
    # ✅ Matrice de confusion
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    df_cm = pd.DataFrame(cm, index=clf.classes_, columns=clf.classes_)

    print("\n📉 Matrice de confusion :")
    print(df_cm.to_string())

    # Save results
    results = {
        "model": "CodeBERT + MLPClassifier",
        "accuracy": acc,
        "report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }
    with open(f"{results_prefix}_results.json", "w") as f:
        json.dump(results, f, indent=4)

    return results
