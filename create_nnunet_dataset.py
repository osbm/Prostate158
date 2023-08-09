import os
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
from tqdm import tqdm
import json

RAW_DIR = Path('nnUNet_raw')
OUTPUT_DIR = RAW_DIR / 'Dataset001_Prostate158'

TRAIN_IMAGEs_DIR = OUTPUT_DIR / 'imagesTr'
TRAIN_LABELs_DIR = OUTPUT_DIR / 'labelsTr'
TEST_IMAGEs_DIR = OUTPUT_DIR / 'imagesTs'
TEST_LABELs_DIR = OUTPUT_DIR / 'labelsTs'

SOURCE_DIR = Path('data')

def check_same_shape(folder: Path) -> bool:
    '''
    Checks if every file in the folder has the same shape. This is required for nnUNet.
    True is good, False is bad.
    '''

    files = os.listdir(folder)
    shapes = []
    for file in files:
        shape= nib.load(folder / file).shape
        shapes.append(shape)
    shapes = np.array(shapes)
    return np.all(shapes == shapes[0])

def merge_tumor_anatomy(tumor_mask: np.ndarray, anatomy_mask: np.ndarray) -> np.ndarray:
    '''
    Merges the tumor mask and the anatomy mask into one mask.
    '''
    tumor_mask[tumor_mask == 1] = 3
    anatomy_mask[tumor_mask > 0] = tumor_mask[tumor_mask > 0]
    return anatomy_mask

def create_mask(case_row, target_folder=TRAIN_LABELs_DIR, add_tumor_mask=True):
    anatomy_mask = nib.load(SOURCE_DIR / case_row["t2_anatomy_reader1"])
    affine = anatomy_mask.affine
    result_mask = anatomy_mask.get_fdata()

    if add_tumor_mask and not case_row["adc_tumor_reader1"].endswith("empty.nii.gz"): # if there is a tumor
        tumor_mask = nib.load(SOURCE_DIR / case_row["adc_tumor_reader1"]).get_fdata()
        result_mask = merge_tumor_anatomy(tumor_mask, result_mask)

    result_image = nib.Nifti1Image(result_mask, affine=affine)
    nib.save(result_image, target_folder / f'PROSTATE_{case_row["ID"]}.nii.gz')

def get_tumor_mask(case_row, target_folder):
    if not case_row["adc_tumor_reader1"].endswith("empty.nii.gz"): # if there is a tumor
        tumor_mask = nib.load(SOURCE_DIR / case_row["adc_tumor_reader1"]).get_fdata()

    # read t2_anatomy_reader1 and take its shape


def create_channels(case_row, target_folder=TRAIN_IMAGEs_DIR, add_t2=True, add_adc=False, add_dwi=False):
    '''
    Just copies and renames the files t2, adc, dwi to the target folder with correct names.
    '''
    image_index = 0
    if add_t2:
        shutil.copy(SOURCE_DIR / case_row["t2"], target_folder / f'PROSTATE_{case_row["ID"]}_{image_index:04d}.nii.gz')
        image_index += 1
    if add_adc:
        shutil.copy(SOURCE_DIR / case_row["adc"], target_folder / f'PROSTATE_{case_row["ID"]}_{image_index:04d}.nii.gz')
        image_index += 1
    if add_dwi:
        shutil.copy(SOURCE_DIR / case_row["dwi"], target_folder / f'PROSTATE_{case_row["ID"]}_{image_index:04d}.nii.gz')
        image_index += 1

    if image_index == 0:
        raise ValueError("No image selected")

def sanity_check():
    for folder in [SOURCE_DIR / 'train', SOURCE_DIR / 'test', SOURCE_DIR / 'valid']:
        for case in os.listdir(folder):
            if not check_same_shape(folder / case):
                print(f'Folder {folder} has different shapes for case {case}') # no output, so everything is fine

def create_folders():
    for folder in [RAW_DIR, OUTPUT_DIR, TRAIN_IMAGEs_DIR, TRAIN_LABELs_DIR, TEST_IMAGEs_DIR, TEST_LABELs_DIR]:
        if not folder.exists():
            folder.mkdir()

if __name__ == "__main__":
    create_folders()
    train_df = pd.read_csv(SOURCE_DIR / 'train.csv')
    test_df = pd.read_csv(SOURCE_DIR / 'test.csv')
    valid_df = pd.read_csv(SOURCE_DIR / 'valid.csv')
    sanity_check()
    
    train_df = pd.concat([train_df, valid_df], ignore_index=True)
    # iterate the dataframe rows
    for case_row in tqdm(train_df.to_dict(orient="records")):
        create_mask(case_row, target_folder=TRAIN_LABELs_DIR, add_tumor_mask=True)
        create_channels(case_row, target_folder=TRAIN_IMAGEs_DIR, add_adc=True, add_dwi=True)

    for case_row in tqdm(test_df.to_dict(orient="records")):
        create_mask(case_row, target_folder=TEST_LABELs_DIR, add_tumor_mask=True)
        create_channels(case_row, target_folder=TEST_IMAGEs_DIR, add_adc=True, add_dwi=True)

    dataset_json = {
        "name": "Prostate158",
        "description": "Prostate cancer segmentation dataset",
        "channel_names": {
            "0": "T2",
            "1": "ADC",
            "2": "DWI"
        },
        "labels": {
            "background": 0,
            # "tumor": 1,
            "prostate_inner": 1,
            "prostate_outer": 2,
            "tumor": 3,
        },
        "numTraining": train_df.shape[0],
        "numTest": test_df.shape[0],
        "file_ending": ".nii.gz",
    }


    with open(OUTPUT_DIR / 'dataset.json', 'w') as f:
        json.dump(dataset_json, f, indent=4)
