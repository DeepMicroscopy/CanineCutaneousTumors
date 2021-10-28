import random
import csv
import glob
from pathlib import Path
from slide.slide_container import SlideContainer
from tqdm import tqdm


def create_patches(files, patches_per_slide):
    patches = []
    for i in files:
        patches += patches_per_slide * [i]
    random.shuffle(patches)
    return patches

def load_slides(set, patch_size=256, label_dict=None, level = None, target_folder=None,annotation_file=None, dataset_type=None):
    train_files = []
    valid_files = []
    test_files = []

    with open('../datasets.csv', newline='') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in tqdm(list(reader)):
            file = glob.glob("{}/**/{}".format(str(target_folder),row["Slide"]), recursive = True)[0]
            if row["Dataset"] == "train" and set.__contains__("train"):
                train_files.append(SlideContainer(Path(file),annotation_file,level, patch_size, patch_size, dataset_type=dataset_type, label_dict = label_dict))
            elif row["Dataset"] == "val" and set.__contains__("valid"):
                valid_files.append(SlideContainer(Path(file),annotation_file,level, patch_size, patch_size, dataset_type=dataset_type, label_dict = label_dict))
            elif row["Dataset"] == "test" and set.__contains__("test"):
                test_files.append(SlideContainer(Path(file),annotation_file,level, patch_size, patch_size, dataset_type=dataset_type, label_dict = label_dict))
            else:
                pass

    return train_files, valid_files, test_files