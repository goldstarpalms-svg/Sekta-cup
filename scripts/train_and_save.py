import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.setka_core import load_raw_data
from src.ml_pipeline import train_model_bundle, save_model_bundle

print("Loading data...")
matches, _ = load_raw_data()
print(f"Loaded {len(matches):,} matches")

print("Training model (this may take 10-20 minutes)...")
bundle = train_model_bundle(
    matches,
    algorithm="xgboost",
    max_training_rows=None,
)

print("Saving model...")
save_path = Path("models/setka_ml_bundle.joblib")
save_path.parent.mkdir(parents=True, exist_ok=True)
save_model_bundle(bundle, save_path)

print(f"Model saved to {save_path}")
print(f"Winner accuracy: {[m for m in bundle['metrics'] if m['model'] == 'winner'][0]['accuracy']:.4f}")
