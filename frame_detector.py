import os
import git
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import docker
import joblib

class FrameworkDetector:#la maison de robots..tout le material qu'il exige existe la
    def __init__(self):
        if os.path.exists("framework_model.joblib") and os.path.exists("vectorizer.joblib"):
            self.model = joblib.load("framework_model.joblib")
            self.vectorizer = joblib.load("vectorizer.joblib")
            print("✅ Modèle chargé depuis le disque.")
        else:
            self.model = None   #le cerveau de robot
            self.vectorizer = None   #la machine qui transforme les mots en nombres

        self.docker_templates = {   #les plans de construction des maisons(dockerfiles) des tresors trouvees(framworks)
            'springboot': """FROM openjdk:17
WORKDIR /app
COPY target/*.jar app.jar
CMD ["java", "-jar", "app.jar"]""",
            
            'laravel': """FROM php:8.2-apache
RUN docker-php-ext-install pdo pdo_mysql
COPY . /var/www/html
CMD ["apache2-foreground"]""",
            
            'flask': """FROM python:3.9-slim
WORKDIR /app
# copier tout le projet
COPY . .

# Installer selon le fichier trouvé
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
RUN if [ -f Pipfile ]; then pip install pipenv && pipenv install --system --deploy; fi
RUN if [ -f pyproject.toml ]; then pip install poetry && poetry install --no-dev --no-interaction; fi


CMD ["python", "app.py"]"""


        }
    
    def extract_features(self, repo_path): #le robot examine un projet(extract_feautres)
        features = {}
        
        # 1. chercher les Fichiers caractéristiques(speciaux)
        features['has_pom.xml'] = int((Path(repo_path) / 'pom.xml').exists())
        features['has_composer.json'] = int((Path(repo_path) / 'composer.json').exists())
        features['has_requirements.txt'] = int((Path(repo_path) / 'requirements.txt').exists())
        
        # 2. lire le TOUS Contenu des fichiers code
        code_text = ""
        for root, dirs, files in os.walk(repo_path):
         # Ignorer les dossiers trop gros/inutiles
         dirs[:] = [d for d in dirs if d not in ['node_modules', 'vendor', 'target', 'build', 'dist', '__pycache__']]
    
         for file in files:
          if file.endswith(('.java', '.php', '.py', '.xml', '.json')):
            try:
                file_path = Path(root) / file

                # Ignorer les fichiers trop gros (>1 Mo)
                # if file_path.stat().st_size > 1_000_000:
                #         print(f"⚠️ Fichier trop gros, ignoré: {file_path}")
                #         continue

                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    code_text += f.read(50000) + " "  # max 50KB
            except Exception as e:
                print(f"⚠️ Erreur lecture {file_path}: {e}")
                continue

        
        return features, code_text
    
    def train(self, dataset_path):#le robot apprend
        data = []
        labels = []
        
        # Charge notre collection de projets exemples=lire son cahier d'exemples
        df = pd.read_csv(dataset_path)
        i=0
        for _, row in df.iterrows():
            repo_path = row['repo_path']
            i = i + 1
            print(f"🔍 Traitement du repo {i}/{len(df)}: {repo_path}")
            features, code_text = self.extract_features(repo_path)
            
            # Ajoute les caractéristiques fixes
            data_item = list(features.values())
            # Ajoute le texte du code comme caractéristique
            data_item.append(code_text)
            data.append(data_item)
            labels.append(row['framework'])
        
        # Prépare les données pour l'IA
        self.vectorizer = TfidfVectorizer(max_features=1000)
        text_features = self.vectorizer.fit_transform([x[-1] for x in data])
        
        # Combine toutes les caractéristiques
        X = []
        for i, item in enumerate(data):
            combined = list(item[:-1]) + text_features[i].toarray().flatten().tolist()
            X.append(combined)
        
        # Entraîne notre modèle magique avec un foret decisionnels
        self.model = RandomForestClassifier()
        self.model.fit(X, labels)
        
        # Sauvegarde le modèle entraîné
        joblib.dump((self.model), 'framework_model.joblib')
        joblib.dump((self.vectorizer),'vectorizer.joblib')
    
    def predict(self, repo_url): #le robot devine
        # Clone le dépôt Git=telecherger le projet
        repo_path = "temp_repo"
        if os.path.exists(repo_path):
            os.system(f"rm -rf {repo_path}")
        
        git.Repo.clone_from(repo_url, repo_path)
        
        # Extrait les caractéristiques=examiner le projet
        features, code_text = self.extract_features(repo_path)
        
        # Prépare les données pour la prédiction=transformer les donnees en nombres
        if not hasattr(self, 'vectorizer'):
            raise Exception("Modèle non entraîné !")
        
        text_features = self.vectorizer.transform([code_text])
        X = list(features.values()) + text_features.toarray().flatten().tolist()
        # Devine le framework
        framework = self.model.predict([X])[0]
        
        # Crée le Dockerfile=la maison de framwork deviner
        dockerfile_content = self.docker_templates.get(framework.lower(), "# Dockerfile non disponible")
        
        # Écrit le Dockerfile
        with open(Path(repo_path) / 'Dockerfile', 'w') as f:
            f.write(dockerfile_content)
        
        print(f"✅ Framework détecté : {framework}")
        print(f"📦 Dockerfile créé dans {repo_path}/Dockerfile")
        
        return framework, repo_path
    
    def test_dockerfile(self, repo_path):#tester le dockerfile
        client = docker.from_env()
        try:
            print("🏗 Construction de l'image Docker...")
            image, _ = client.images.build(path=repo_path)
            print("🐳 Image construite avec succès !")
            return True
        except Exception as e:
            print(f"❌ Erreur lors de la construction : {e}")
            return False

# Exemple d'utilisation
# Comment jouer avec le robot
if __name__ == "__main__":
    detective = FrameworkDetector()#allume le robot
    
    # Phase d'entraînement (à faire une seule fois)
    print("🎓 Entraînement du modèle...")
    detective.train("C:/Users/Zbook/Desktop/ai/frame-detector/formations.csv")  # cahier d'exemples contient le repo_path,framwork
    
    # Phase de détection
    repo_url = input("Entrez l'URL du dépôt Git à analyser : ")
    framework, repo_path = detective.predict(repo_url)
    
    # Test du Dockerfile
    if input("Tester le Dockerfile ? (o/n) ").lower() == 'o':
        detective.test_dockerfile(repo_path)


