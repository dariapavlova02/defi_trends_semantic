import joblib
import json
import os
from tensorflow.keras.models import load_model


def save_predictor_state(predictor, path_prefix):
    """Saves the predictor's state (models, scalers, parameters)."""

    # 1. Create the directory if it doesn't exist
    os.makedirs(path_prefix, exist_ok=True)
    print(f"Saving state to directory: {path_prefix}")

    # 2. Save the models
    for model_name, model in predictor.models.items():
        if model_name == 'rf':
            # Scikit-learn models are saved using joblib
            joblib.dump(model, os.path.join(path_prefix, f"{model_name}_model.joblib"))
        elif model_name in ['lstm', 'gru']:
            # Keras models are saved using their own method
            model.save(os.path.join(path_prefix, f"{model_name}_model.keras"))
    print("  - Models saved.")

    # 3. Save the scalers
    joblib.dump(predictor.scalers, os.path.join(path_prefix, "scalers.joblib"))
    print("  - Scalers saved.")

    # 4. Save the best parameters
    with open(os.path.join(path_prefix, "best_params.json"), 'w') as f:
        json.dump(predictor.best_params, f, indent=4)
    print("  - Best parameters saved.")