import os

# BASE_PATH = os.getcwd()+"/../../../"
BASE_DIR = os.getcwd()+"/../../../"

FACIAL_EXPRESSIONS = {
    'smile': True,
    'surprise': False,
    'disgust': False
}

MODEL_TAG = "best_auroc_baal"

FEATURES_FILE = os.path.join(BASE_DIR,"data/facial_expression_smile/facial_dataset_non_manifest_control_updated.csv")

MODEL_BASE_PATH = os.path.join(BASE_DIR,f"models/facial_expression_smile_{MODEL_TAG}")
MODEL_PATH = os.path.join(MODEL_BASE_PATH,"predictive_model/model.pth")
MODEL_CONFIG_PATH = os.path.join(MODEL_BASE_PATH,"predictive_model/model_config.json")
SCALER_PATH = os.path.join(MODEL_BASE_PATH,"scaler/scaler.pth")