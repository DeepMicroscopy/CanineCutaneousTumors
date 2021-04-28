import random
import csv
import os
from fnmatch import fnmatch
from pathlib import Path
from slide.slide_container import SlideContainer
from tqdm import tqdm


def create_patches(files, patches_per_slide):
    patches = []
    for i in files:
        patches += patches_per_slide * [i]
    random.shuffle(patches)
    return patches

def load_slides(patch_size=256, label_dict=None, level = None, target_folder=None,annotation_file=None, dataset_type=None):
    os.makedirs(str(target_folder), exist_ok=True)
    pattern = "*.svs"
    container = []

    for path, subdirs, files in os.walk(target_folder):
        for name in tqdm(files[:5]):
            if fnmatch(name, pattern):
                container.append(SlideContainer(Path(os.path.join(path, name)),annotation_file,level, patch_size, patch_size, dataset_type=dataset_type, label_dict = label_dict))
    return container

def train_val_test_split(container):
    slide_names = [c.file.parts[-1] for c in container]

    train_files = []
    valid_files = []
    test_files = []

    with open('../datasets.csv', newline='') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            if row["Dataset"] == "train" and slide_names.__contains__(row['Slide']):
                train_files.append(container[slide_names.index(row['Slide'])])
            elif row["Dataset"] == "val" and slide_names.__contains__(row['Slide']):
                valid_files.append(container[slide_names.index(row['Slide'])])
            elif row["Dataset"] == "test" and slide_names.__contains__(row['Slide']):
                test_files.append(container[slide_names.index(row['Slide'])])
            else:
                pass

    return train_files, valid_files, test_files
