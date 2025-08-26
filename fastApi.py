from fastapi import FastAPI, Form
from pydantic import BaseModel
from frame_detector import FrameworkDetector

# Create FastAPI app
app = FastAPI(
    title="Framework Detector API",
    description="Detects the framework of a repo and generates Dockerfile",
    version="1.0.0"
)

# Initialize the detector (loads trained model if available)
detector = FrameworkDetector()


# 📌 Schema for training
class TrainRequest(BaseModel):
    dataset_path: str


# 📌 Schema for prediction
class PredictRequest(BaseModel):
    repo_url: str


@app.post("/train")
def train_model(req: TrainRequest):
    """
    Train the model using a CSV dataset.
    The CSV must contain two columns: repo_path, framework.
    """
    detector.train(req.dataset_path)
    return {"status": "success", "message": "Model trained and saved."}


@app.post("/predict")
def predict_framework(req: PredictRequest):
    """
    Predict the framework for a Git repository URL.
    Returns the framework name and the repo path.
    """
    framework, repo_path = detector.predict(req.repo_url)
    return {
        "status": "success",
        "framework": framework,
        "repo_path": repo_path,
        "dockerfile_path": f"{repo_path}/Dockerfile"
    }


@app.post("/test-dockerfile")
def test_dockerfile(repo_path: str = Form(...)):
    """
    Test the Dockerfile generated in the predict step.
    repo_path should be the local path to the cloned repo.
    """
    success = detector.test_dockerfile(repo_path)
    return {
        "status": "success" if success else "failed",
        "repo_path": repo_path
    }


@app.get("/")
def root():
    return {"message": "Welcome to Framework Detector API 🚀"}
