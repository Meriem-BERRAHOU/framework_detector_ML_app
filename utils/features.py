import os
from pathlib import Path

# def extract_features(repo_path):
#     features = {}

#     # Caractéristiques fixes (présence de fichiers)
#     features['has_pom.xml'] = int((Path(repo_path) / 'pom.xml').exists())
#     #todo add gradle.... 
#     features['has_composer.json'] = int((Path(repo_path) / 'composer.json').exists())
#     features['has_requirements.txt'] = int((Path(repo_path) / 'requirements.txt').exists())

#     # Texte du code
#     code_text = ""
#     for root, dirs, files in os.walk(repo_path):
#         dirs[:] = [d for d in dirs if d not in ['node_modules', 'vendor', 'target', 'build', 'dist', '__pycache__']]
#         for file in files:
#             if file.endswith(('.java', '.php', '.py', '.xml', '.json')):
#                 try:
#                     with open(Path(root) / file, "r", encoding="utf-8", errors="ignore") as f:
#                         code_text += f.read(20000) + " "
#                 except Exception:
#                     continue 
    
#     print(f"📂 {repo_path} → {len(code_text)} caractères de code extraits")

#     return features, code_text 



def extract_features(repo_path):
    features = {}

    # Files that usually define dependencies
    dependency_files = [
        "requirements.txt",   # Python
        "pyproject.toml",     # Python (Poetry, modern setup)
        "setup.py",           # Python
        "package.json",       # Node.js
        "composer.json",      # PHP
        "pom.xml",            # Java (Maven)
        "build.gradle",       # Java (Gradle)
        "build.gradle.kts",   # Kotlin Gradle
        "Gemfile",            # Ruby
        "Pipfile",            # Python (Pipenv)
        "environment.yml",    # Conda
    ]

    code_text = ""
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in ['node_modules', 'vendor', 'target', 'build', 'dist', '__pycache__']]
        for file in files:
            if file in dependency_files:
                try:
                    with open(Path(root) / file, "r", encoding="utf-8", errors="ignore") as f:
                        code_text += f.read(20000) + " "
                except Exception:
                    continue  

    # Features: presence of key dependency files
    features['has_pom.xml'] = int((Path(repo_path) / 'pom.xml').exists())
    features['has_build.gradle'] = int((Path(repo_path) / 'build.gradle').exists() or (Path(repo_path) / 'build.gradle.kts').exists())
    features['has_composer.json'] = int((Path(repo_path) / 'composer.json').exists())
    features['has_requirements.txt'] = int((Path(repo_path) / 'requirements.txt').exists())
    features['has_package.json'] = int((Path(repo_path) / 'package.json').exists())

    print(f"📂 {repo_path} → {len(code_text)} caractères extraits depuis fichiers de dépendances")

    return features, code_text


if __name__ == "__main__":
    features,code_text = extract_features('/workspaces/frame-detector/projets/BookStack-development')
    print('features: ',features)
    print('code text extracted from special files: ',code_text)
