import sys
import subprocess

# Install facenet_pytorch from the specified .whl file
subprocess.run([sys.executable, "-m", "pip", "install", "input/facenet_pytorch-2.2.7-py3-none-any.whl"], check=True)

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pathlib import Path
import torch
from torchvision import transforms
from facenet_pytorch import MTCNN  # Import MTCNN after installation
import cv2
from transformers import ViTForImageClassification, ViTImageProcessor
import os
from PIL import Image
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import random
from fastapi.middleware.cors import CORSMiddleware


DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)

Base.metadata.create_all(bind=engine)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI with SQLite!"}

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
MODEL_PATH = "model/vit_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.get("/testformodel")
def test_model():
    if os.path.exists(MODEL_PATH) and os.path.isfile(MODEL_PATH):
        return {"message": "Model path exists and is a valid file"}
    else:
        return {"error": "Model path does not exist or is not a valid file"}

# Load model
model = ViTForImageClassification.from_pretrained(
    "google/vit-large-patch16-224",
    num_labels=2,
    ignore_mismatched_sizes=True  # Allows replacement of the classifier head
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
model.to(DEVICE)
model.eval()

# Image processor
processor = ViTImageProcessor.from_pretrained("google/vit-large-patch16-224")

# Initialize MTCNN
mtcnn = MTCNN(image_size=224, margin=20, keep_all=False, device=DEVICE)

# Global path for last uploaded video
last_uploaded_path = None


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    global last_uploaded_path

    # Clear the uploads directory
    for existing_file in UPLOAD_DIR.iterdir():
        if existing_file.is_file():
            try:
                existing_file.unlink()  # Attempt to delete the file
            except PermissionError:
                return {"error": f"File {existing_file} is being used by another process"}

    # Save the new file
    filepath = UPLOAD_DIR / file.filename
    with open(filepath, "wb") as f:
        f.write(await file.read())
    last_uploaded_path = filepath

    return {"message": "Upload successful", "filename": file.filename}


@app.get("/predict")
def predict_video():
    if not last_uploaded_path or not last_uploaded_path.exists():
        return JSONResponse(status_code=404, content={"error": "No video uploaded"})

    # Extract faces from random frames
    faces = extract_faces(last_uploaded_path, max_faces=30, max_attempts=90)
    if not faces:
        return {"error": "No faces detected in video"}

    total_probs = torch.zeros(2, device=DEVICE)  # To accumulate probabilities
    num_faces = 0

    # Evaluate each face individually
    for face in faces:
        inputs = processor(images=[face], return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # Run the model and compute probabilities
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            total_probs += probs[0]  # Accumulate probabilities
            num_faces += 1

    if num_faces == 0:
        return {"error": "No confident prediction could be made"}

    # Compute average probabilities
    avg_probs = total_probs / num_faces
    predicted_class = torch.argmax(avg_probs).item()
    confidence = avg_probs[predicted_class].item()

    # Determine the label
    label = "Real" if predicted_class == 0 else "Fake"

    return {
        "prediction": label,
        "confidence": f"{confidence * 100:.2f}%"
    }


def extract_faces(video_path, max_faces=30, max_attempts=90):
    cap = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    extracted_faces = []
    attempts = 0

    try:
        while len(extracted_faces) < max_faces and attempts < max_attempts:
            frame_idx = random.randint(0, n_frames - 1)  # Select a random frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                attempts += 1
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face = mtcnn(Image.fromarray(frame_rgb))  # Detect face using MTCNN
            if face is not None:
                if isinstance(face, torch.Tensor):
                    extracted_faces.append(transforms.ToPILImage()(face))
                else:
                    extracted_faces.append(face)

            attempts += 1
    finally:
        cap.release()  # Ensure the video file is released

    return extracted_faces