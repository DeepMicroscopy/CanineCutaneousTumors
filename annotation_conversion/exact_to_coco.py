from tqdm import tqdm
import numpy as np
from shapely.geometry import Polygon
import json
import datetime
import urllib3

from exact_sync.v1.api.annotations_api import AnnotationsApi
from exact_sync.v1.api.images_api import ImagesApi
from exact_sync.v1.api.image_sets_api import ImageSetsApi
from exact_sync.v1.api.annotation_types_api import AnnotationTypesApi
from exact_sync.v1.configuration import Configuration
from exact_sync.v1.api_client import ApiClient



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



def polys_from_exact(configuration):
    client = ApiClient(configuration)
    image_sets_api = ImageSetsApi(client)
    annotations_api = AnnotationsApi(client)
    annotation_types_api = AnnotationTypesApi(client)
    images_api = ImagesApi(client)

    anno_list = []
    image_list = []

    image_sets = image_sets_api.list_image_sets()
    annotations = {}

    label_dict = {'Bone': 1, 'Cartilage': 2, 'Dermis': 3, 'Epidermis': 4, 'Subcutis': 5, 'Inflamm/Necrosis': 6,
                  'Melanoma': 7, 'Plasmacytoma': 8,'Mast Cell Tumor': 9, 'PNST': 10, 'SCC': 11, 'Trichoblastoma': 12, 'Histiocytoma': 13}
    is_crowd = 0

    # These ids will be automatically increased as we go
    annotation_id = 1
    image_id = 1

    for image_set in image_sets.results:
        for product in image_set.product_set:
            for anno_type in annotation_types_api.list_annotation_types(product=product).results:
                annotations[anno_type.id] = label_dict[anno_type.name]

        for image in tqdm(images_api.list_images(image_set=image_set.id, pagination=False).results):
            image_list.append({'license': 1, 'file_name': image.filename, 'id': image_id, 'width': image.width, 'height': image.height})
            poly_list = {}
            for annotation in annotations_api.list_annotations(image=image.id, pagination=False, deleted=False).results:
                vector = []
                poly = {}
                for i in range(1, (len(annotation.vector) // 2) + 1):
                    vector.append((annotation.vector['x' + str(i)], annotation.vector['y' + str(i)]))

                poly["Coords"] = Polygon(vector)
                poly["Label"] = annotations[annotation.annotation_type]
                poly["Hierarchy"] = 0
                poly["Enclosed"] = []
                poly_list[annotation.id] = poly

            get_polygon_hierarchy(poly_list)
            poly_list = dict(sorted(poly_list.items(), key=lambda x: x[1]['Hierarchy']))

            for id, poly in poly_list.items():
                area = get_polygon_area(poly_list, poly)
                poly["Area"] = area
                anno_list.append(create_annotation(poly, image_id, annotation_id, is_crowd))
                annotation_id += 1
            image_id += 1

    return image_list, anno_list

def convert(configuration):
    json_dict = create_header()
    image_list, anno_list = polys_from_exact(configuration)
    json_dict["images"] = image_list
    json_dict["annotations"] = anno_list
    with open('CATCH.json', 'w') as f:
        json.dump(json_dict, f)


if __name__ == '__main__':
    # EXACT configuration
    configuration = Configuration()
    configuration.verify_ssl = False
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    configuration.username = 'exact'
    configuration.password = 'exact'
    configuration.host = "http://localhost:8000"

    # Conversion
    convert(configuration)





