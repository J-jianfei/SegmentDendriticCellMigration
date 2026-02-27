from pathlib import Path

from ultralytics.models.sam import SAM3VideoSemanticPredictor

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent

# Create video semantic predictor (supports text prompts on video)
overrides = dict(
    conf=0.5,
    task="segment",
    mode="predict",
    model=str(ROOT / "sam3.pt"),
    half=False,
    device="cpu",
    imgsz=644,
)
predictor = SAM3VideoSemanticPredictor(overrides=overrides)

results = predictor(
    source=str(
        DATA_DIR
        / "WT1-1.avi"
    ),
    text=["migrating dendritic cells"],
    stream=True,
)



# Process results
for r in results:
    r.show()  # Display frame with tracked objects
