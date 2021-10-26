import sys
sys.path.append("../../SlideRunner/")
import numpy as np
import json
from SlideRunner.dataAccess.database import Database
from shapely.geometry import Polygon
import datetime
from tqdm import tqdm

def create_header():
    json_dict = {}
    json_dict["info"] = {
        "description": 'Pan-Tumor Canine CuTaneous Cancer Histology (CATCH) Dataset',
        "url": "",
        "version": '1.0',
        "year": 2021,
        "contributor": 'Frauke Wilm, Marco Fragoso, Christian Marzahl, Jingna Qiu, Christof Bertram, Robert Klopfleisch, Andreas Maier, Katharina Breininger, Marc Aubreville',
        "date_created": datetime.datetime.utcnow().isoformat(' ')
    }
    json_dict["licenses"] = [{
        "id": 1,
        "name": 'Attribution-NonCommercial-NoDerivs License',
        "url": 'http://creativecommons.org/licenses/by-nc-nd/2.0/'
    }]
    json_dict["categories"] = [
    {
        'id': 1,
        'name': 'Bone',
        "supercategory": 'Tissue',
    },
    {
        'id': 2,
        'name': 'Cartilage',
        'supercategory': 'Tissue',
    },
    {
        'id': 3,
        'name': 'Dermis',
        'supercategory': 'Tissue',
    },
    {
        'id': 4,
        'name': 'Epidermis',
        'supercategory': 'Tissue',
    },
    {
        'id': 5,
        'name': 'Subcutis',
        'supercategory': 'Tissue',
    },
    {
        'id': 6,
        'name': 'Inflamm/Necrosis',
        'supercategory': 'Tissue',
    },
    {
        'id': 7,
        'name': 'Melanoma',
        'supercategory': 'Tumor',
    },
    {
        'id': 8,
        'name': 'Plasmacytoma',
        'supercategory': 'Tumor',
    },
    {
        'id': 9,
        'name': 'Mast Cell Tumor',
        'supercategory': 'Tumor',
    },
    {
        'id': 10,
        'name': 'PNST',
        'supercategory': 'Tumor',
    },
    {
        'id': 11,
        'name': 'SCC',
        'supercategory': 'Tumor',
    },
    {
        'id': 12,
        'name': 'Trichoblastoma',
        'supercategory': 'Tumor',
    },
    {
        'id': 13,
        'name': 'Histiocytoma',
        'supercategory': 'Tumor',
    }
    ]
    return json_dict



def create_annotation(polygon, image_id, annotation_id, is_crowd):

    segmentation = np.array(polygon["Coords"].exterior.coords).ravel().tolist()

    x, y, max_x, max_y = polygon["Coords"].bounds
    width = max_x - x
    height = max_y - y
    bbox = (x, y, width, height)
    area = polygon["Area"]

    annotation = {
        'segmentation': segmentation,
        'iscrowd': is_crowd,
        'image_id': image_id,
        'category_id': polygon["Label"],
        'id': annotation_id,
        'bbox': bbox,
        'area': area
    }

    return annotation

def get_polygon_hierarchy(poly_list):
    for id_outer, outer_poly in poly_list.items():
        for id_inner, inner_poly in poly_list.items():
            if id_outer != id_inner and outer_poly["Coords"].contains(inner_poly["Coords"]):
                outer_poly["Enclosed"].append(id_inner)
                inner_poly["Hierarchy"] += 1



def get_polygon_area(polygon_list, polygon):
    enclosed = 0
    for poly_within in polygon["Enclosed"]:
        enclosed += get_polygon_area(polygon_list, polygon_list[poly_within])
    return polygon["Coords"].area - enclosed



def polys_from_sql(database):
    anno_list = []
    image_list = []

    is_crowd = 0

    # These ids will be automatically increased as we go
    annotation_id = 1
    image_id = 1

    getslides = """SELECT uid, filename, width, height FROM Slides"""
    for currslide, filename, width, height in tqdm(database.execute(getslides).fetchall()):
        database.loadIntoMemory(currslide)
        image_list.append({'license': 1, 'file_name': filename, 'id': image_id, 'width': width, 'height': height})
        poly_list = {}
        for id, annotation in database.annotations.items():
            if len(annotation.labels) != 0 and annotation.deleted != 1:
                if annotation.annotationType == 3:
                    vector = []
                    poly = {}
                    for x, y in annotation.coordinates:
                        vector.append((int(x), int(y)))

                    poly["Coords"] = Polygon(vector)
                    poly["Label"] = annotation.labels[0].classId
                    poly["Hierarchy"] = 0
                    poly["Enclosed"] = []
                    poly_list[id] = poly

        get_polygon_hierarchy(poly_list)
        poly_list = dict(sorted(poly_list.items(), key=lambda x: x[1]['Hierarchy']))

        for id, poly in poly_list.items():
            area = get_polygon_area(poly_list, poly)
            poly["Area"] = area
            anno_list.append(create_annotation(poly, image_id, annotation_id, is_crowd))
            annotation_id += 1
        image_id += 1

    return image_list, anno_list

def convert(annotation_path):
    database = Database()
    database.open(annotation_path)
    json_dict = create_header()
    image_list, anno_list = polys_from_sql(database)
    json_dict["images"] = image_list
    json_dict["annotations"] = anno_list
    with open('CATCH.json', 'w') as f:
        json.dump(json_dict, f)


if __name__ == '__main__':
    # Define annotation path
    annotation_file = "CATCH.sqlite"

    # Conversion
    convert(annotation_file)