import os
from pathlib import Path

def extract_features(repo_path):
    features = {}

    # Caractéristiques fixes (présence de fichiers)
    features['has_pom.xml'] = int((Path(repo_path) / 'pom.xml').exists())
    features['has_composer.json'] = int((Path(repo_path) / 'composer.json').exists())
    features['has_requirements.txt'] = int((Path(repo_path) / 'requirements.txt').exists())

    # Texte du code
    code_text = ""
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in ['node_modules', 'vendor', 'target', 'build', 'dist', '__pycache__']]
        for file in files:
            if file.endswith(('.java', '.php', '.py', '.xml', '.json')):
                try:
                    with open(Path(root) / file, "r", encoding="utf-8", errors="ignore") as f:
                        code_text += f.read(20000) + " "
                except Exception:
                    continue
    
    print(f"📂 {repo_path} → {len(code_text)} caractères de code extraits")
    return features, code_text
