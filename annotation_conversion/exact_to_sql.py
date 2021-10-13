import sys
sys.path.append("../../SlideRunner/")
from tqdm import tqdm
from SlideRunner.dataAccess.database import *
from pathlib import Path
import pandas as pd
import numpy as np
import openslide
import urllib3

from exact_sync.v1.api.annotations_api import AnnotationsApi
from exact_sync.v1.api.images_api import ImagesApi
from exact_sync.v1.api.image_sets_api import ImageSetsApi
from exact_sync.v1.api.annotation_types_api import AnnotationTypesApi
from exact_sync.v1.configuration import Configuration
from exact_sync.v1.api_client import ApiClient

def convert(slide_path, configuration):
    files = {slide.name: slide for slide in slide_path.rglob("*.svs")}
    client = ApiClient(configuration)
    image_sets_api = ImageSetsApi(client)
    annotations_api = AnnotationsApi(client)
    annotation_types_api = AnnotationTypesApi(client)
    images_api = ImagesApi(client)

    image_sets = image_sets_api.list_image_sets()
    annotations = {}

    # These ids will be automatically increased as we go
    rows = []

    for image_set in image_sets.results:
        for product in image_set.product_set:
            for anno_type in annotation_types_api.list_annotation_types(product=product).results:
                annotations[anno_type.id] = anno_type.name

        for image in tqdm(images_api.list_images(image_set=image_set.id, pagination=False).results):
            for annotation in annotations_api.list_annotations(image=image.id, pagination=False, deleted=False).results:
                vector = []
                for i in range(1, (len(annotation.vector) // 2) + 1):
                    vector.append((annotation.vector['x' + str(i)], annotation.vector['y' + str(i)]))
                rows.append([image.filename, vector, annotations[annotation.annotation_type]])

    df = pd.DataFrame(rows, columns=["file_name", "polygon", "cat"])
    database = Database()
    database.create("CATCH.sqlite")
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
    target_folder = Path("D:/Slides/Canine Skin Tumors")

    # EXACT configuration
    configuration = Configuration()
    configuration.verify_ssl = False
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    configuration.username = 'exact'
    configuration.password = 'exact'
    configuration.host = "http://localhost:8000"

    # Conversion
    convert(target_folder, configuration)
