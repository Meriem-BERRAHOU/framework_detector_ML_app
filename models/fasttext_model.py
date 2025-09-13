import fasttext
import pandas as pd
import joblib
import json
import os
import tempfile
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
from utils.features import extract_features

def train_fasttext(save_prefix="saved_models/fasttext",results_prefix='results/fasttext'):
    
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

    print("🚀 Training FastText...")
    # S'assurer que le dossier existe
    os.makedirs("saved_models", exist_ok=True)

    # Convertir les données en DataFrame pour faciliter le split
    df = pd.DataFrame({
        "text": [x[-1].replace("\n", " ").replace("\r", " ").strip() for x in data],
        "label": labels
    })

    # Split dataset
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

    # Créer fichiers temporaires au format attendu par FastText (__label__ + texte)
    def to_fasttext_format(df, path):
        with open(path, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                f.write(f"__label__{row['label']} {row['text']}\n")

    train_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt").name
    test_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt").name
    to_fasttext_format(train_df, train_file)
    to_fasttext_format(test_df, test_file)

    # Entraînement FastText
    model = fasttext.train_supervised(
        input=train_file,
        lr=0.1,
        epoch=25,
        wordNgrams=2,
        dim=100,
        loss="softmax"
    )

    # Prédiction sur le test set
    y_true = test_df["label"].tolist()
    y_pred = []
    for text in test_df["text"]:
        # Nettoyer le texte : supprimer les \n et \r
        clean_text = text.replace("\n", " ").replace("\r", " ").strip()
        pred = model.predict(clean_text)[0][0].replace("__label__", "")
        y_pred.append(pred)


    acc = accuracy_score(y_true, y_pred)

    # Sauvegarde modèle
    model.save_model(f"{save_prefix}_model.bin")

    # Rapport brut (dict)
    report_dict = classification_report(y_true, y_pred, output_dict=True)

    # ✅ Transformer en DataFrame
    df_report = pd.DataFrame(report_dict).transpose()
    print("\n📊 Rapport de classification :")
    print(df_report.to_string(float_format="%.2f")
          )
    
    labels = model.get_labels()

    # Récupérer les classes (sans le préfixe __label__)
    labels_ft = [lbl.replace("__label__", "") for lbl in model.get_labels()]
    
    # ✅ Matrice de confusion
    cm = confusion_matrix(y_true, y_pred, labels=labels_ft)
    df_cm = pd.DataFrame(cm, index=labels_ft, columns=labels_ft)

    print("\n📉 Matrice de confusion :")
    print(df_cm.to_string())

    # Sauvegarde résultats
    results = {
        "model": "FastText",
        "accuracy": acc,
        "report": classification_report(y_true, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }
    with open(f"{results_prefix}_results.json", "w") as f:
        json.dump(results, f, indent=4)

    return results
