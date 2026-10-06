# Shared defaults only. Drive paths belong in the Colab notebooks.
SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2
CLASS_NAMES = [
    "adenosis", "fibroadenoma", "phyllodes_tumor", "tubular_adenoma",
    "ductal_carcinoma", "lobular_carcinoma", "mucinous_carcinoma",
    "papillary_carcinoma",
]
NUM_CLASSES = len(CLASS_NAMES)
