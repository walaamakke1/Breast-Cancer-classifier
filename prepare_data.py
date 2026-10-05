from pathlib import Path
import tarfile
import pandas as pd

CLASS_NAMES = [
    'adenosis', 'fibroadenoma', 'phyllodes_tumor', 'tubular_adenoma',
    'ductal_carcinoma', 'lobular_carcinoma', 'mucinous_carcinoma', 'papillary_carcinoma',
]
CODES = ['A', 'F', 'PT', 'TA', 'DC', 'LC', 'MC', 'PC']

#restore the images
def extract_archive(archive_path, raw_dir):
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    if len(list(raw_dir.rglob('SOB_*.png'))) != 7909:
        with tarfile.open(archive_path, 'r:gz') as archive:
            archive.extractall(raw_dir, filter='data')
    return raw_dir

#read metadata
def read_metadata(csv_path, raw_dir):
    df = pd.read_csv(
        csv_path,
        dtype={'sample_id': str, 'patient_id': str}
    )
    return df.sort_values('image').reset_index(drop=True)

#Within each class, shuffle patient groups and assign approximately 70% training, 15% validation, 15% testing
# All images from one patient group stay together

def make_splits(df, seed=42):
    patients = df[['patient_id', 'label']].drop_duplicates()
    assert patients['patient_id'].is_unique, 'A patient group has multiple labels.'
    assignments = {}
    for label in range(8):
        ids = patients.loc[patients.label == label, 'patient_id']
        ids = ids.sample(frac=1, random_state=seed).tolist()
        assert len(ids) >= 3, f'Class {label} needs at least 3 patient groups.'
        n = max(1, round(0.15 * len(ids)))
        for name, selected in [('train', ids[2*n:]), ('val', ids[:n]), ('test', ids[n:2*n])]:
            assignments.update({patient: name for patient in selected})
    result = df.copy()
    result['split'] = result['patient_id'].map(assignments)
    parts = {name: result[result.split == name].reset_index(drop=True)
             for name in ('train', 'val', 'test')}
    validate_splits(parts)
    return parts


def validate_splits(parts):
    for name, frame in parts.items():
        assert set(frame.label) == set(range(8)), f'{name} must contain all 8 classes.'
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        assert set(parts[a].patient_id).isdisjoint(parts[b].patient_id), 'Patient overlap.'


def fixed_splits(df, split_dir, seed=42):
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    saved = split_dir / 'patient_split.csv'
    if saved.exists():
        assignments = pd.read_csv(saved, dtype={'patient_id': str})
    else:
        parts = make_splits(df, seed)
        assignments = pd.concat(parts.values())[['patient_id', 'split']].drop_duplicates()
        assignments.to_csv(saved, index=False)
    result = df.merge(assignments, on='patient_id', validate='many_to_one')
    assert len(result) == len(df), 'Saved split does not cover the current dataset.'
    parts = {name: result[result.split == name].reset_index(drop=True)
             for name in ('train', 'val', 'test')}
    validate_splits(parts)
    for name, frame in parts.items():
        frame.to_csv(split_dir / f'{name}.csv', index=False)
    return parts
