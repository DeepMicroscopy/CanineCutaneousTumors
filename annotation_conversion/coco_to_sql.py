import sys
sys.path.append("../../SlideRunner/")
from tqdm import tqdm
from SlideRunner.dataAccess.database import *
from pathlib import Path
import pandas as pd
import json
import numpy as np
import openslide

def convert(slide_path, annotation_path):
    files = {slide.name: slide for slide in slide_path.rglob("*.svs")}
    rows = []
    with open(annotation_path) as f:
        data = json.load(f)
        categories = {cat["id"]: cat["name"] for cat in data["categories"]}
        for row in data["images"]:
            file_name = row["file_name"]
            image_id = row["id"]
            for annotation in [anno for anno in data['annotations'] if anno["image_id"] == image_id]:
                polygon = annotation["segmentation"]
                cat = categories[annotation["category_id"]]
                rows.append([file_name,polygon, cat])

    df = pd.DataFrame(rows, columns=["file_name", "polygon", "cat"])
    database = Database()
    database.create("canine_cutaneous_tumors.sqlite")
    database.insertAnnotator('Coco')
    database.insertClass('Bone')
    database.insertClass('Cartilage')
    database.insertClass('Dermis')
    database.insertClass('Epidermis')
    database.insertClass('Subcutis')
    database.insertClass('Inflamm/Necrosis')
    database.insertClass('Melanoma')
    database.insertClass('Plasmacytoma')
    database.insertClass('Mast Cell Tumor')
    database.insertClass('PNST')
    database.insertClass('SCC')
    database.insertClass('Trichoblastoma')
    database.insertClass('Histiocytoma')
    classes = database.getAllClasses()
    classes = {name:id for name, id, colour in  classes}
    t = 0
    for slide in tqdm(df["file_name"].unique()):

        imageDf = df[df["file_name"] == slide]

        image_path = str(files[slide])

        image_id = database.insertNewSlide(slide, image_path)
        image_id = database.findSlideWithFilename(slide, image_path)
        image = openslide.open_slide(image_path)
        database.execute('UPDATE Slides set width=%d, height=%d WHERE uid=%d' % (image.dimensions[0], image.dimensions[1], image_id))
        database.db.commit()

        for label, vector in zip(imageDf['cat'], imageDf['polygon']):
            try:
                label_id = classes[str(label)]

                database.insertNewPolygonAnnotation(np.array(vector).reshape((-1,2)),
                                           slideUID=image_id,
                                           classID=label_id,
                                           annotator=1)
            except:
                t += 1

if __name__ == '__main__':
    # Define slide directory
    target_folder = Path("E:/Slides/Canine Skin Tumors")

    # Define annotation path
    annotation_file = "canine_cutaneous_tumors_before.json"

    # Conversion
    convert(target_folder, annotation_file)




