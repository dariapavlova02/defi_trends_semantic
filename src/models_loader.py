import joblib
import json
import os
from tensorflow.keras.models import load_model


def load_predictor_state(predictor, path_prefix):
    """Loads the predictor's state from files."""

    print(f"Loading state from directory: {path_prefix}")

    # 1. Load scalers
    predictor.scalers = joblib.load(os.path.join(path_prefix, "scalers.joblib"))
    print("  - Scalers loaded.")

    # 2. Load parameters
    with open(os.path.join(path_prefix, "best_params.json"), 'r') as f:
        predictor.best_params = json.load(f)
    print("  - Best parameters loaded.")

    # 3. Load models
    for model_name in predictor.best_params.keys():
        if model_name == 'rf':
            predictor.models[model_name] = joblib.load(os.path.join(path_prefix, f"{model_name}_model.joblib"))
        elif model_name in ['lstm', 'gru']:
            predictor.models[model_name] = load_model(os.path.join(path_prefix, f"{model_name}_model.keras"))
    print("  - Models loaded.")
    return predictor

# --- Example of how to use it ---
# # First, you need to create new, empty predictor objects
# loaded_baseline_predictor = DeFiVolumePredictor(use_semantic_features=False)
# loaded_semantic_predictor = DeFiVolumePredictor(use_semantic_features=True)

# # Then, populate them with the saved data
# load_predictor_state(loaded_baseline_predictor, "saved_models/baseline")
# load_predictor_state(loaded_semantic_predictor, "saved_models/semantic")

# print("\nPredictor states have been successfully restored!")
