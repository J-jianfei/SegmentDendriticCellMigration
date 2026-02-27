from pathlib import Path
from ultralytics.models.sam import SAM3SemanticPredictor

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent

# Initialize predictor with configuration
overrides = dict(
    conf=0.5,
    task="segment",
    mode="predict",
    model=str(ROOT / "sam3.pt"),
    half=True,  # Use FP16 for faster inference
    device="cpu",
    save=True,
    imgsz=644,
)
predictor = SAM3SemanticPredictor(overrides=overrides)

# Set image once for multiple queries
predictor.set_image(str(
        DATA_DIR
        /"WT1-1.png"))

# Query with multiple text prompts
results = predictor(text=["cell"])
