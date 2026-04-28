import os

# BASE_PATH = os.getcwd()+"/../../../"
BASE_DIR = os.getcwd()+"/../../../"

WAVLM_FEATURES_FILE = os.path.join(BASE_DIR,"data/quick_brown_fox/wavlm_fox_features_non_manifest_control_updated.csv")

MODEL_TAG = "best_auroc_baal"
MODEL_BASE_PATH = os.path.join(BASE_DIR, f"models/fox_model_{MODEL_TAG}")
MODEL_PATH = os.path.join(MODEL_BASE_PATH,"predictive_model/model.pth")
SCALER_PATH = os.path.join(MODEL_BASE_PATH,"scaler/scaler.pth")
MODEL_CONFIG_PATH = os.path.join(MODEL_BASE_PATH,"predictive_model/model_config.json")