import random
import csv


def create_patches(files, patches_per_slide):
    patches = []
    for i in files:
        patches += patches_per_slide * [i]
    random.shuffle(patches)
    return patches

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
