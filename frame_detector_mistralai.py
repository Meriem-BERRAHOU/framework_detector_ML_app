import os
import json
from pathlib import Path
from collections import Counter
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch
from huggingface_hub import login
from huggingface_hub import InferenceClient
import subprocess
import uuid
from openai import OpenAI

mistralai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# Clone GitHub repo 
def clone_repo(github_url: str, base_dest="cloned_repos") -> str:
    # Ensure base folder exists
    os.makedirs(base_dest, exist_ok=True)

    # Create a unique folder for this repo
    repo_id = str(uuid.uuid4())[:8]   # short unique ID
    dest = os.path.join(base_dest, repo_id)

    # Clone into that folder
    subprocess.run(["git", "clone", github_url, dest], check=True)

    return dest

#  Fonction de génération de Dockerfile
def generate_dockerfile_with_mistralai(framework: str) -> str:
    """
    Génère un Dockerfile via Hugging Face API (modèle chat comme Mistral/mistralai-2-Chat)
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant that generates production-ready Dockerfiles."},
        {"role": "user", "content": f"""
Task: Generate a clean, optimized Dockerfile for a {framework} web application.
Make sure it:
- Uses a slim base image
- Installs dependencies efficiently
- Exposes the correct port
- Uses CMD to run the app in production

Output only the Dockerfile content without explanations or extra text.
"""}
    ]

    response = mistralai_client.chat.completions.create(
        model="mistralai/mistral-7b-instruct",
        messages=messages,
        max_tokens=500,
        temperature=0.3,
        top_p=0.9
    )

    dockerfile_raw = response.choices[0].message.content.strip()

    # Post-process: Keep only valid Dockerfile lines 
    dockerfile_clean = []
    for line in dockerfile_raw.splitlines():
        line_stripped = line.strip()
        if line_stripped.startswith(("FROM", "WORKDIR", "COPY", "RUN", "ENV", "EXPOSE", "CMD", "#")):
            dockerfile_clean.append(line)
    return "\n".join(dockerfile_clean)
                  

#  Helpers pour scanner repo 
TEXT_FILE_EXTS = {
    ".py", ".ts", ".java", ".go", ".rs", ".php", ".rb", ".jsx", ".tsx", ".cs", ".cpp", ".c", ".xml", ".json"
}

FRAME_KEYWORDS = {
    "django": ["manage.py", "django", "settings.py", "wsgi.py"],
    "flask": ["flask", "app.py", "wsgi.py", "requirements.txt"],
    "fastapi": ["fastapi", "uvicorn", "main.py", "app.py"],
    "springboot": ["pom.xml", "spring-boot", "Application.java"],
    "laravel": ["artisan", "composer.json", "laravel"],
}


def collect_files(repo_path: str, max_files=200):
    p = Path(repo_path)
    files = []
    for f in p.rglob("*"):
        if f.is_file() and (f.suffix in TEXT_FILE_EXTS or f.name in {"package.json", "requirements.txt", "pyproject.toml", "pom.xml", "composer.json"}):
            files.append(f)
            if len(files) >= max_files:
                break
    return files


def read_snippets(files, max_chars_per_file=4000, sample_lines=200):
    snippets = {}
    for f in files:
        try:
            txt = f.read_text(errors="ignore")
            # shrink to a representative sample: head + tail
            if len(txt) > max_chars_per_file:
                txt = txt[:max_chars_per_file//2] + "\n\n/*...*/\n\n" + txt[-max_chars_per_file//2:]
            snippets[str(f)] = txt
        except Exception as e:
            snippets[str(f)] = ""
    return snippets

#  Fonction de detection de langage de programmation backend
# def detect_language_with_mistralai(snippets: dict) -> str:
#     content = "\n".join(list(snippets.values())[:5])[:2000]

#     messages = [
#         {"role": "system", "content": "You are a tool that identifies the main programming language used in a project."},
#         {"role": "user", "content": f"Here are some file contents:\n{content}\n\nAnswer only with the main backend programming language (e.g., Python, Java, PHP, JavaScript)."}
#     ]

#     response = mistralai_client.chat_completions(messages=messages, max_tokens=50, temperature=0.0)
#     return response.choices[0].message["content"].strip()


def detect_framework_with_mistralai(snippets: dict) -> str:
    content = "\n".join(list(snippets.values())[:5])[:3000]

    messages = [
        {"role": "system", "content": "You are an expert software analyzer."},
        {"role": "user", "content": f"Given the following code snippets:\n{content}\n\nWhich backend framework is most likely used? Choose one from: Flask, Django, Laravel, SpringBoot, Express.js, or Unknown. Answer only with the framework name."}
    ]

    response = mistralai_client.chat.completions.create(model="mistralai/mistral-7b-instruct",messages=messages, max_tokens=50, temperature=0.0)
    return response.choices[0].message.content.strip()


#  Orchestration 
def run_pipeline_from_github(github_url: str, out_path="Generated.Dockerfile"):
    
    print(f"[+] Cloning repo: {github_url}")
    repo_path = clone_repo(github_url)

    print(f"[+] Collecting files from {repo_path}")
    files = collect_files(repo_path)
    print(f"[+] Collected {len(files)} files")

    snippets = read_snippets(files)

    # print("[+] Detecting language...")
    # language = detect_language_with_mistralai(snippets)
    # print(f"    -> {language}")

    print("[+] Detecting framework...")
    framework = detect_framework_with_mistralai(snippets)
    print(f"    -> {framework}")

    print("[+] Generating Dockerfile...")
    dockerfile = generate_dockerfile_with_mistralai(framework)

    out_path = os.path.join(repo_path, "Dockerfile")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(dockerfile)

    print(f"[+] Dockerfile written to {out_path}")

    return { "framework": framework, "dockerfile": out_path}


#  Main 
if __name__ == "__main__":
    github_url = input("Enter the GitHub repo URL: ").strip()
    out_path = "Generated.Dockerfile"

    res = run_pipeline_from_github(github_url, out_path)

    print("\nRESULT SUMMARY:")
    print(json.dumps(res, indent=2))
