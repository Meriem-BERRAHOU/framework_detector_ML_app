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
import re

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

# Files that usually list installed modules / dependencies
DEPENDENCY_FILES = {
    # --- Python ---
    "requirements.txt",    # Python
    "pyproject.toml",      # Python (Poetry)
    "Pipfile",             # Python (Pipenv)

    # --- Node.js / Frontend ---
    "package.json",        # Node.js / React / Vue / Angular
    "package-lock.json",   # Node.js lockfile
    "yarn.lock",           # Yarn lockfile
    "pnpm-lock.yaml",      # pnpm lockfile
    "angular.json",        # Angular
    "next.config.js",      # Next.js
    "nuxt.config.js",      # Nuxt.js (Vue)
    "vite.config.js",      # Vite
    "vue.config.js",       # Vue CLI

    # --- PHP ---
    "composer.json",       # PHP

    # --- Java ---
    "pom.xml",             # Java Maven
    "build.gradle",        # Java Gradle
    "build.gradle.kts",    # Kotlin Gradle

    # --- Ruby ---
    "Gemfile",             # Ruby

    # --- Conda ---
    "environment.yml",     # Conda
}


def collect_dependency_files(repo_path: str, max_files=50):
    """Collect only dependency/manifest files from the repo."""
    p = Path(repo_path)
    files = []
    for f in p.rglob("*"):
        if f.is_file() and f.name in DEPENDENCY_FILES:
            files.append(f)
            if len(files) >= max_files:
                break
    return files


def read_dependency_snippets(files, max_chars_per_file=4000):
    """Read content of dependency files, trimmed if too long."""
    snippets = {}
    for f in files:
        try:
            txt = f.read_text(errors="ignore")
            # Trim large files: head + tail
            if len(txt) > max_chars_per_file:
                txt = txt[:max_chars_per_file//2] + "\n\n/*...*/\n\n" + txt[-max_chars_per_file//2:]
            snippets[str(f)] = txt
        except Exception:
            snippets[str(f)] = ""
    
    return snippets


def analyze_and_detect(snippets: dict) -> dict:
    # Build content with filenames for clarity
    combined_snippets = []
    for path, txt in snippets.items():  # no limit
        combined_snippets.append(f"FILE: {os.path.basename(path)}\n{txt}")

    content = "\n\n".join(combined_snippets)  # no slicing

    # Step 1: Detect frameworks
    detection_messages = [
        {"role": "system", "content": "You are an expert software analyzer."},
        {"role": "user", "content": f"""
    Analyze the following dependency files and output only JSON (no explanations, no comments):

    {content}

    Return only a JSON object like this:

    ```json
    {{
    "backend": "detected_framework",
    "frontend": "detected_framework",
    "database": "detected_framework"
    }}
    ```

    Framework options:

    backend: Flask, Django, FastAPI, Laravel, SpringBoot, Express.js, Unknown
    frontend: React, Angular, Vue, Svelte, null
    database: MySQL, PostgreSQL, MongoDB, SQLite, null
    """}
    ]

    detection_response = mistralai_client.chat.completions.create(
        model="mistralai/mistral-7b-instruct",
        messages=detection_messages,
        max_tokens=300,
        temperature=0.0
    )

    detection_raw = detection_response.choices[0].message.content.strip()

    # Try to extract JSON block
    match = re.search(r"{[\s\S]*}", detection_raw)
    if match:
        try:
            detection_json = json.loads(match.group(0))
        except:
            detection_json = {"backend": "Unknown", "frontend": None, "database": None}
    else:
        detection_json = {"backend": "Unknown", "frontend": None, "database": None}

    # --- FIX: normalize nested format ---
    if "framework" in detection_json:
        detection_json = {
            "backend": detection_json["framework"].get("backend", "Unknown"),
            "frontend": detection_json["framework"].get("frontend", None),
            "database": detection_json.get("database", None)
        }

    return detection_json

def genarate_docker_files(detection_json:dict,snippets:dict) ->dict:

    # Build content with filenames for clarity
    combined_snippets = []
    for path, txt in snippets.items():  # no limit
        combined_snippets.append(f"FILE: {os.path.basename(path)}\n{txt}")

    content = "\n\n".join(combined_snippets)  # no slicing

    backend_docker_content = 'null'
    frontend_docker_content = 'null'
    compose_docker_content = 'null'

    #generating docker file for backend
    if detection_json.get("backend") and detection_json.get("backend") not in ["Unknown", "null", None]:
        back_gen_messages = [
        {"role": "system", "content": "You are a helpful DevOps assistant."},
        {"role": "user", "content": f"""
        Given this detection result: 

        {json.dumps(detection_json)} and this code extracted from the depandacy files : {content}

        Generate:

        -A clean and valid Dockerfile for backend .

        Rules:

        -Use slim base images

        -Expose correct ports

        """}
        ]

        backe_gen_response = mistralai_client.chat.completions.create(
        model="mistralai/mistral-7b-instruct",
        messages=back_gen_messages,
        max_tokens=1000,   
        temperature=0.2    
        )

        backend_docker_content = backe_gen_response.choices[0].message.content.strip()
        backend_docker_content = extract_code_block(backend_docker_content)

    #generating docker file for frontend 
    if detection_json.get("frontend") and detection_json.get("frontend") not in ["Unknown", "null", None]:
        front_gen_messages = [
        {"role": "system", "content": "You are a helpful DevOps assistant."},
        {"role": "user", "content": f"""
        Given this detection result: 

        {json.dumps(detection_json)} and this code extracted from the depandacy files : {content}

        Generate:

        -A clean and valid Dockerfile for frontend .

        Rules:

        -Use slim base images

        -Expose correct ports

        """}
        ]

        front_gen_response = mistralai_client.chat.completions.create(
        model="mistralai/mistral-7b-instruct",
        messages=front_gen_messages,
        max_tokens=1000,
        temperature=0.2
        )

        frontend_docker_content = front_gen_response.choices[0].message.content.strip()
        frontend_docker_content = extract_code_block(frontend_docker_content)

    #generate docker compose file to orchestrate the frontend and backend docker files generated earlier
    if detection_json.get("backend") and detection_json.get("backend") not in ["Unknown", "null", None] and detection_json.get("frontend") and detection_json.get("frontend") not in ["Unknown", "null", None]:    
        compose_gen_messages = [
        {"role": "system", "content": "You are a helpful DevOps assistant."},
        {"role": "user", "content": f"""
        Given this detection result: 

        {json.dumps(detection_json)} and this code extracted from the depandacy files : {content}

        Generate a docker-compose.yml file that includes:
        - the backend service
        - a database service if it exists
        - a frontend service 

        Rules:

        -Use slim base images

        -Expose correct ports

        """}
        ]

        compose_gen_response = mistralai_client.chat.completions.create(
        model="mistralai/mistral-7b-instruct",
        messages=compose_gen_messages,
        max_tokens=1000,  
        temperature=0.2   
        )

        compose_docker_content = compose_gen_response.choices[0].message.content.strip()
        compose_docker_content = extract_code_block(compose_docker_content)

    return {"docker-backend": backend_docker_content,"docker-frontend":frontend_docker_content,"docker-compose":compose_docker_content}

    

def extract_code_block(text: str) -> str:
    # Try to extract fenced code block first
    match = re.search(r"```(?:dockerfile)?\s*([\s\S]*?)```", text, re.IGNORECASE)   
    if match:
        return match.group(1).strip()
    # If no fenced block, just return the raw text
    return text.strip()

# Orchestration

def run_pipeline_from_github(repo_url: str):

    print(f"[+] Cloning repo: {repo_url}")
    repo_path = clone_repo(repo_url)

    print(f"[+] Collecting files from {repo_path}")
    files = collect_dependency_files(repo_path)
    snippets = read_dependency_snippets(files)

    print("[+] Detecting frameworks...")
    detection = analyze_and_detect(snippets)

    print("[+] Generating Dockerfile/Compose...")
    docker_files = genarate_docker_files(detection,snippets)

    outputs = {}

    # Backend Dockerfile
    if docker_files["docker-backend"] and docker_files["docker-backend"] != "null":
        backend_path = os.path.join(repo_path, "Dockerfile.backend")
        with open(backend_path, "w", encoding="utf-8") as f:
            f.write(docker_files["docker-backend"])
        outputs["backend"] = backend_path
        print(f"[+] Backend Dockerfile written to {backend_path}")

    # Frontend Dockerfile
    if docker_files["docker-frontend"] and docker_files["docker-frontend"] != "null":
        frontend_path = os.path.join(repo_path, "Dockerfile.frontend")
        with open(frontend_path, "w", encoding="utf-8") as f:
            f.write(docker_files["docker-frontend"])
        outputs["frontend"] = frontend_path
        print(f"[+] Frontend Dockerfile written to {frontend_path}")

    # Docker Compose
    if docker_files["docker-compose"] and docker_files["docker-compose"] != "null":
        compose_path = os.path.join(repo_path, "docker-compose.yml")
        with open(compose_path, "w", encoding="utf-8") as f:
            f.write(docker_files["docker-compose"])
        outputs["compose"] = compose_path
        print(f"[+] docker-compose.yml written to {compose_path}")

    summary = {
        "frameworks": detection,
        "outputs": outputs
    }

    return summary



# Main entry

if __name__ == "__main__":
    github_url = input("Enter the GitHub repo URL: ").strip()
    result = run_pipeline_from_github(github_url)

    print("\nRESULT SUMMARY:")
    print(json.dumps(result, indent=2))