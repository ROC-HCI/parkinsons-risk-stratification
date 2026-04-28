import os

BASE_DIR = os.getcwd()+"/../../../"

FINGER_FEATURES_FILE = os.path.join(BASE_DIR,"data/finger_tapping/finger_tapping_dataset_non_manifest_control_updated.csv")
AUDIO_FEATURES_FILE = os.path.join(BASE_DIR,"data/quick_brown_fox/wavlm_fox_features_non_manifest_control_updated.csv")
FACIAL_FEATURES_FILE = os.path.join(BASE_DIR, "data/facial_expression_smile/facial_dataset_non_manifest_control_updated.csv")

MODEL_BASE_PATH = os.path.join(BASE_DIR,"models")
MODEL_CONFIG_PATH = os.path.join(MODEL_BASE_PATH, "uncertainty_aware_fusion/model_config.json")
MODEL_PATH = os.path.join(MODEL_BASE_PATH, "uncertainty_aware_fusion/model.pth")

FACIAL_EXPRESSIONS = {
    'smile': True,
    'surprise': False,
    'disgust': False
}

MODEL_SUBSETS = {
    0: ['finger_model_both_hand_fusion_baal', 'fox_model_best_auroc_baal', 'facial_expression_smile_best_auroc_baal'],
    1: ['finger_model_both_hand_fusion_baal', 'fox_model_best_auroc_baal'],
    2: ['finger_model_both_hand_fusion_baal', 'facial_expression_smile_best_auroc_baal'],
    3: ['fox_model_best_auroc_baal', 'facial_expression_smile_best_auroc_baal']
}