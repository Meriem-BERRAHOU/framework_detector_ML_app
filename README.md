# 🧠 Git Repository Framework Detection (Machine Learning + API)

Ce projet permet de **détecter automatiquement le framework principal** (Django, Flask, Laravel, Spring Boot, etc.) utilisé dans un dépôt GitHub.  
Il combine du **Machine Learning** pour l’analyse du code et une **API FastAPI** pour exposer le modèle.  

---

## 🚀 Démo en ligne

👉 [Essayer l’API sur Render]([https://git-repo-detector.onrender.com](https://git-repo-framework-detection-machine.onrender.com))  

- **Swagger UI (documentation interactive)** :  
  [`/docs`]([https://git-repo-detector.onrender.com](https://git-repo-framework-detection-machine.onrender.com)/docs)  
- **ReDoc (autre documentation)** :  
  [`/redoc`]([https://git-repo-detector.onrender.com](https://git-repo-framework-detection-machine.onrender.com)/redoc)  

---

## ✨ Fonctionnalités

- 🔍 Analyse un dépôt GitHub donné en entrée  
- 🤖 Détection du framework via un modèle de Machine Learning  
- ⚡ API REST avec FastAPI  
- 📦 Génération automatique d’un `Dockerfile` si le framework est reconnu  

---

## 🛠️ Exemple d’utilisation
Requête POST :
```bash
curl -X POST "https://git-repo-detector.onrender.com/detect" \
     -H "Content-Type: application/json" \
     -d '{"repo_url": "https://github.com/laravel/laravel"}'
```
Réponse JSON :
```bash
{
  "status": "success",
  "framework": "laravel",
  "repo_path": "temp_repo",
  "dockerfile_path": "temp_repo/Dockerfile"
}
```
##📌 Auteur

👩‍💻 Réalisé par Meriem BERRAHOU

