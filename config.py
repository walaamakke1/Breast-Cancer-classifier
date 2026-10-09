import os

SUBTYPES = ["A", "F", "PT", "TA", "DC", "LC", "MC", "PC"]
SUBTYPE_NAMES = {
    "A": "adenosis", "F": "fibroadenoma", "PT": "phyllodes_tumor", "TA": "tubular_adenoma",
    "DC": "ductal_carcinoma", "LC": "lobular_carcinoma", "MC": "mucinous_carcinoma",
    "PC": "papillary_carcinoma",
}
BENIGN_SUBTYPES = {"A", "F", "PT", "TA"}
BINARY_CLASSES = ["benign", "malignant"]
MAGNIFICATIONS = [40, 100, 200, 400]

# ImageNet statistics: used for pre-trained backbones 
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SEED = 42


def get_task():
    #Classification task: subtype or binary
    task = os.environ.get("BREAKHIS_TASK", "subtype")
    if task not in ("subtype", "binary"):
        raise ValueError(f"BREAKHIS_TASK must be 'subtype' or 'binary', got '{task}'")
    return task


def get_class_names(task=None):
    task = task or get_task()
    return list(SUBTYPES) if task == "subtype" else list(BINARY_CLASSES)


def get_label_column(task=None):
    #Column of metadata.csv that holds the label 
    task = task or get_task()
    return "label_subtype" if task == "subtype" else "label_binary"


def data_dir():
    # Folder that holds the prepared dataset (images+ metadata.csv + stats.json)
    return os.environ.get("BREAKHIS_DATA_DIR", os.path.join(os.getcwd(), "data", "breakhis"))


def out_dir():
    # Folder where we write checkpoints, logs and plots
    return os.environ.get("BREAKHIS_OUT_DIR", os.path.join(os.getcwd(), "outputs"))


NUM_CLASSES = len(get_class_names())
