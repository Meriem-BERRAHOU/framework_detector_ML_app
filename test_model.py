from frame_detector import FrameworkDetector

if __name__ == "__main__":
    detective = FrameworkDetector()

    if detective.vectorizer is None or detective.model is None:
        print("⚠️ Le modèle n'est pas encore entraîné. Lance d'abord train() dans frame-detector.py.")
    else:
        repo_url = input("Entrez l'URL du dépôt Git à analyser : ")
        framework, repo_path = detective.predict(repo_url)

        print(f"✅ Framework détecté : {framework}")
        print(f"📂 Dépôt cloné dans : {repo_path}")

        # Optionnel : test du Dockerfile
        if input("Tester le Dockerfile ? (o/n) ").lower() == "o":
            detective.test_dockerfile(repo_path)
